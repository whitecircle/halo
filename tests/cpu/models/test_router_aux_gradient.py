#!/usr/bin/env python
"""The router aux loss trains routers under reentrant checkpointing as it does with it off. That forward runs under
``no_grad``, so without ``install_router_aux_gradient`` each layer's aux term reaches the loss with no graph."""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
from transformers import GptOssConfig, GptOssForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextForConditionalGeneration
from transformers.models.laguna.configuration_laguna import LagunaConfig
from transformers.models.laguna.modeling_laguna import LagunaForCausalLM

from halo.callbacks.balancing import apply_balancing_mode
from halo.distributed.context_parallel.config import CPConfig
from halo.distributed.context_parallel.model_rules import SUPPORTED_ATTN_IMPLEMENTATIONS
from halo.distributed.context_parallel.ulysses_model import UlyssesModel
from halo.distributed.expert_parallel.families.roster import MOE_LAYER_MAP
from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from halo.distributed.expert_parallel.parallelize import apply_ep_layers, ep_claimed_blocks
from halo.models.config_access import config_export_ready
from halo.models.gradient_checkpointing import enable_scoped_gradient_checkpointing
from halo.models.moe import router_aux_gradient
from halo.models.moe.balancing import resolve_balancing_mode
from halo.models.moe.config_fields import resolve_expert_count
from halo.models.moe.router_logits import declared_routers
from tests.common.models import (
    TINY_GLM5_CONFIG,
    TINY_GLM5_VISION_CONFIG,
    TINY_GPTOSS_CONFIG,
    TINY_LAGUNA_CONFIG,
    TINY_QWEN3_MOE_CONFIG,
)
from tests.common.parallelism import single_process_ep_config
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model

COEF = 0.7
LAYERS = 4
BATCH, SEQ = 2, 24


def _qwen3_moe():
    config = {**TINY_QWEN3_MOE_CONFIG, "num_hidden_layers": LAYERS, "router_aux_loss_coef": COEF}
    return Qwen3MoeForCausalLM(Qwen3MoeConfig(**config))


def _gpt_oss():
    config = {**TINY_GPTOSS_CONFIG, "num_hidden_layers": LAYERS, "router_aux_loss_coef": COEF}
    config["layer_types"] = ["sliding_attention", "full_attention"] * (LAYERS // 2)
    return GptOssForCausalLM(GptOssConfig(**config))


def _laguna():
    return LagunaForCausalLM(LagunaConfig(**{**TINY_LAGUNA_CONFIG, "router_aux_loss_coef": COEF}))


def _glm5_next():
    text = {**TINY_GLM5_CONFIG, "router_aux_loss_coef": COEF}
    return Glm5NextForConditionalGeneration(Glm5NextConfig(text_config=text, vision_config=TINY_GLM5_VISION_CONFIG))


def _tiny(family: str):
    overrides = {"num_hidden_layers": LAYERS, "router_aux_loss_coef": COEF}
    return lambda: tiny_family_model(TINY_MOE_FAMILIES[family], overrides=overrides)


FAMILIES = {
    "qwen3_moe": _qwen3_moe,
    "gpt_oss": _gpt_oss,
    "laguna": _laguna,
    "glm5_next": _glm5_next,
    "qwen3_next": _tiny("qwen3_next"),
    "qwen3_5_moe_text": _tiny("qwen3_5_moe_text"),
}


def _build(family: str, *, balanced: bool, ep: bool = False):
    """``balanced`` applies ``moe_balancing: aux_loss``. Routers get a wide init so the aux term moves them by far more
    than any recompute rounding."""
    torch.manual_seed(0)
    model = FAMILIES[family]().float()
    for level in (model.config, model.config.get_text_config()):
        level._attn_implementation = "eager"
    for router in declared_routers(model):
        torch.nn.init.normal_(router.module.weight, std=0.5)
    if ep:
        apply_ep_layers(model, single_process_ep_config(resolve_expert_count(model.config)))
    if balanced:
        apply_balancing_mode(model, "aux_loss", is_moe=True)
    else:
        model.config.get_text_config().output_router_logits = True
    model.train()
    return model


def _checkpoint(model, *, ep: bool = False, every_n_layers: int = 1) -> None:
    kwargs = {"use_reentrant": True}
    if ep:
        enable_scoped_gradient_checkpointing(model, gradient_checkpointing_kwargs=kwargs)
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs, every_n_layers=every_n_layers)


def _batches(model, count: int = 1, padded: bool = False) -> list[dict]:
    generator = torch.Generator().manual_seed(1)
    vocab = model.config.get_text_config().vocab_size
    batches = []
    for _ in range(count):
        input_ids = torch.randint(1, vocab, (BATCH, SEQ), generator=generator)
        attention_mask = torch.ones_like(input_ids)
        if padded:
            attention_mask[1, SEQ * 2 // 3 :] = 0
        batches.append({"input_ids": input_ids, "attention_mask": attention_mask})
    return batches


def _step(model, batches, *, labels: bool = True):
    """The aux term is added by the model (``labels``) or outside it (the KTO / CP shape)."""
    for batch in batches:
        if labels:
            outputs = model(**batch, labels=batch["input_ids"].masked_fill(batch["attention_mask"] == 0, -100))
            loss = outputs.loss
        else:
            outputs = model(**batch)
            loss = outputs.logits.float().logsumexp(-1).mean() + COEF * outputs.aux_loss
        (loss / len(batches)).backward()
    grads = torch.cat([router.module.weight.grad.flatten() for router in declared_routers(model)])
    return grads, float(loss.detach()), float(outputs.aux_loss.detach())


def _assert_matches_native(got: torch.Tensor, native: torch.Tensor, dropped: torch.Tensor) -> None:
    """``dropped`` is far enough off that a missing aux gradient cannot pass for rounding."""
    assert (native - dropped).norm() > 1e-2 * native.norm(), "the aux term barely moves the routers"
    torch.testing.assert_close(got, native, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("padded", [False, True], ids=["unpadded", "padded"])
@pytest.mark.parametrize("family", FAMILIES)
def test_reentrant_checkpoint_trains_routers_like_no_checkpoint(family, padded):
    """The pooled load, the attention mask and the coefficient all arrive through the family's own loss."""
    model = _build(family, balanced=True)
    batches = _batches(model, padded=padded)
    native, native_loss, native_aux = _step(_build(family, balanced=False), batches)

    _checkpoint(model)
    got, loss, aux = _step(model, batches)

    unrouted = _build(family, balanced=False)
    _checkpoint(unrouted)
    dropped = _step(unrouted, batches)[0]
    _assert_matches_native(got, native, dropped)
    # The coefficient is never rewritten, so the logged loss keeps its aux term and the export the configured value.
    assert loss == pytest.approx(native_loss, rel=1e-6) and aux == pytest.approx(native_aux, rel=1e-6)
    with config_export_ready(model.config):
        exported = model.config.get_text_config().to_dict()
    assert exported["router_aux_loss_coef"] == COEF and model.router_aux_loss_coef == COEF


@pytest.mark.parametrize("family", ["qwen3_moe", "gpt_oss"])
def test_uncheckpointed_layers_keep_the_native_path(family):
    """Unchecked layers collect logits with a graph and must not gain a second copy of the term."""
    batches = _batches(_build(family, balanced=False))
    native = _step(_build(family, balanced=False), batches)[0]

    torch.testing.assert_close(_step(_build(family, balanced=True), batches)[0], native, rtol=0, atol=0)

    mixed = _build(family, balanced=True)
    _checkpoint(mixed, every_n_layers=2)
    torch.testing.assert_close(_step(mixed, batches)[0], native, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("family", ["qwen3_moe", "gpt_oss"])
def test_aux_term_added_outside_the_forward(family):
    """TRL's KTO and the CP wrapper add ``coef * outputs.aux_loss`` themselves, from a forward without labels."""
    batches = _batches(_build(family, balanced=False))
    native = _step(_build(family, balanced=False), batches, labels=False)[0]
    model = _build(family, balanced=True)
    _checkpoint(model)
    unrouted = _build(family, balanced=False)
    _checkpoint(unrouted)
    _assert_matches_native(_step(model, batches, labels=False)[0], native, _step(unrouted, batches, labels=False)[0])


@pytest.mark.parametrize("family", ["qwen3_moe", "gpt_oss"])
def test_gradient_accumulation_and_interleaved_no_grad_forwards(family):
    """Each backward consumes only its own forward's gradients; a ``no_grad`` pass between them (reference or KL)
    leaves no trace."""
    batches = _batches(_build(family, balanced=False), count=3)
    native = _step(_build(family, balanced=False), batches)[0]

    model = _build(family, balanced=True)
    _checkpoint(model)
    for batch in batches:
        outputs = model(**batch, labels=batch["input_ids"])
        with torch.no_grad():
            model(**batches[0], labels=batches[0]["input_ids"])
        (outputs.loss / len(batches)).backward()
    got = torch.cat([router.module.weight.grad.flatten() for router in declared_routers(model)])
    torch.testing.assert_close(got, native, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("family", FAMILIES)
def test_ep_layer_checkpoint_replay(family):
    """The EP layer owns the router at ``ep_size=1`` and replays its dispatch inside a checkpoint scope, so the
    gradient must ride the wrapper's output back to the adopted router."""
    batches = _batches(_build(family, balanced=False))
    native = _step(_build(family, balanced=False, ep=True), batches)[0]
    claimed = {path: MOE_LAYER_MAP[type(block).__name__] for path, block in ep_claimed_blocks(FAMILIES[family]())}

    model = _build(family, balanced=True, ep=True)
    wrapped = {path: type(module) for path, module in model.named_modules() if isinstance(module, EPMoELayerBase)}
    assert claimed and wrapped == claimed
    _checkpoint(model, ep=True)
    unrouted = _build(family, balanced=False, ep=True)
    _checkpoint(unrouted, ep=True)
    _assert_matches_native(_step(model, batches)[0], native, _step(unrouted, batches)[0])


def test_cp_wrapper_resolves_and_routes_through_its_inner_model(tmp_path):
    """The CP wrapper forwards ``output_router_logits`` into its causal LM, so ``auto`` reads that model's forward."""
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'pg'}")
    try:
        inner = _build("qwen3_moe", balanced=False)
        # The Ulysses validator reads a flash label; no attention kernel runs here.
        inner.config._attn_implementation = SUPPORTED_ATTN_IMPLEMENTATIONS[0]
        wrapper = UlyssesModel(inner, CPConfig(1, stage_world_size=1, stage_base_rank=0, nvlink_domain_size=1))
        assert resolve_balancing_mode("auto", wrapper, is_moe=True) == "aux_loss"
        apply_balancing_mode(wrapper, "aux_loss", is_moe=True)
        assert getattr(inner.model, router_aux_gradient._INSTALLED_ATTR, None) is not None
    finally:
        dist.destroy_process_group()


def test_a_gradient_no_recompute_takes_fails_the_backward(monkeypatch):
    monkeypatch.setattr(router_aux_gradient._RouterAuxGradient, "_on_owner", lambda self, owner, args, output: None)
    model = _build("qwen3_moe", balanced=True)
    _checkpoint(model)
    batch = _batches(model)[0]
    with pytest.raises(RuntimeError, match="no recompute of the block owning those routers"):
        model(**batch, labels=batch["input_ids"]).loss.backward()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
