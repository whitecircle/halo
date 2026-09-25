#!/usr/bin/env python
"""The router aux loss must train routers under reentrant gradient checkpointing, exactly as without it.

Every MoE run with gradient checkpointing is forced reentrant, whose original forward runs under
``no_grad``. transformers collects ``outputs.router_logits`` in that pass, so without
:func:`~src.models.moe_aux_loss.install_router_aux_gradient` each checkpointed layer's aux term reaches
the loss with no graph and the router gets none of its gradient. The reference throughout is the same
model with checkpointing off, where the family's own ``load_balancing_loss_func`` backpropagates
natively: with ``moe_balancing: aux_loss`` applied, a checkpointed backward must land the same router
gradient, and an unchecked one must stay exactly native (no term counted twice).

Families differ in router: Qwen3-MoE (softmax over bf16-convention logits), GPT-OSS (biased linear, a
``(hidden, scores)`` block output), Laguna (sigmoid routing on fp32 logits behind a dense first layer)
and GLM-5 Next (the composite wrapper, whose text tower is the collecting backbone).

Run: python tests/cpu/models/test_moe_aux_loss_checkpointing.py
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
from transformers import GptOssConfig, GptOssForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextForConditionalGeneration
from transformers.models.laguna.configuration_laguna import LagunaConfig
from transformers.models.laguna.modeling_laguna import LagunaForCausalLM

from src.distributed.context_parallel.config import CPConfig
from src.distributed.context_parallel.validation import SUPPORTED_ATTN_IMPLEMENTATIONS
from src.distributed.context_parallel.wrapper import UlyssesCPModelWrapper
from src.distributed.expert_parallel.balancing_strategy import apply_balancing_strategy
from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.patching import enable_ep_gradient_checkpointing, patch_moe_model_for_ep
from src.models import moe_aux_loss
from src.models.loading.config_levels import config_export_ready
from src.models.moe_balancing import declared_routers, detect_moe_experts_topk, resolve_balancing_mode
from tests.common.models import (
    TINY_GLM5_CONFIG,
    TINY_GLM5_VISION_CONFIG,
    TINY_GPTOSS_CONFIG,
    TINY_LAGUNA_CONFIG,
    TINY_QWEN3_MOE_CONFIG,
)
from tests.common.parallelism import single_process_ep_config

COEF = 0.7
LAYERS = 4
BATCH, SEQ = 2, 24

# The GLM-5 Next text tower without its linear-attention layers, whose conv kernel is CUDA-only.
_GLM5_TEXT = {**TINY_GLM5_CONFIG, "layer_types": ["deepseek_sparse_attention"] * TINY_GLM5_CONFIG["num_hidden_layers"]}


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
    text = {**_GLM5_TEXT, "router_aux_loss_coef": COEF}
    return Glm5NextForConditionalGeneration(Glm5NextConfig(text_config=text, vision_config=TINY_GLM5_VISION_CONFIG))


FAMILIES = {"qwen3_moe": _qwen3_moe, "gpt_oss": _gpt_oss, "laguna": _laguna, "glm5_next": _glm5_next}


def _build(family: str, *, balanced: bool, ep: bool = False):
    """The seeded tiny model in fp32 with eager attention, the aux loss on at ``COEF``.

    ``balanced`` applies ``moe_balancing: aux_loss`` through the real strategy seam; routers get a
    wide init so the aux term moves them by far more than any recompute rounding.
    """
    torch.manual_seed(0)
    model = FAMILIES[family]().float()
    for level in (model.config, model.config.get_text_config()):
        level._attn_implementation = "eager"
    for router in declared_routers(model):
        torch.nn.init.normal_(router.module.weight, std=0.5)
    if ep:
        patch_moe_model_for_ep(model, single_process_ep_config(detect_moe_experts_topk(model)[0]))
    if balanced:
        apply_balancing_strategy(model, "aux_loss", is_moe=True)
    else:
        model.config.get_text_config().output_router_logits = True
    model.train()
    return model


def _checkpoint(model, *, ep: bool = False, every_n_layers: int = 1) -> None:
    kwargs = {"use_reentrant": True}
    if ep:
        enable_ep_gradient_checkpointing(model, gradient_checkpointing_kwargs=kwargs)
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
    """Forward + backward per micro-batch, the aux term added by the model (``labels``) or outside it
    (the KTO / CP shape); returns the router gradients and the last micro-batch's logged values."""
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
    """``got`` equals the unchecked gradient, where plain reentrant checkpointing (``dropped``) is far
    enough off it that a missing aux gradient could not pass for rounding."""
    assert (native - dropped).norm() > 1e-2 * native.norm(), "the aux term barely moves the routers"
    torch.testing.assert_close(got, native, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("padded", [False, True], ids=["unpadded", "padded"])
@pytest.mark.parametrize("family", FAMILIES)
def test_reentrant_checkpoint_trains_routers_like_no_checkpoint(family, padded):
    """The pooled load, the attention mask and the coefficient all arrive through the family's own
    loss; the recomputed routers must receive exactly the unchecked gradient."""
    model = _build(family, balanced=True)
    batches = _batches(model, padded=padded)
    native, native_loss, native_aux = _step(_build(family, balanced=False), batches)

    _checkpoint(model)
    got, loss, aux = _step(model, batches)

    unrouted = _build(family, balanced=False)
    _checkpoint(unrouted)
    dropped = _step(unrouted, batches)[0]
    _assert_matches_native(got, native, dropped)
    # The coefficient is never rewritten, so the logged loss keeps the coefficient-weighted aux term
    # and the export carries the configured value.
    assert loss == pytest.approx(native_loss, rel=1e-6) and aux == pytest.approx(native_aux, rel=1e-6)
    with config_export_ready(model.config):
        exported = model.config.get_text_config().to_dict()
    assert exported["router_aux_loss_coef"] == COEF and model.router_aux_loss_coef == COEF


@pytest.mark.parametrize("family", ["qwen3_moe", "gpt_oss"])
def test_uncheckpointed_layers_keep_the_native_path(family):
    """No checkpoint, and the layers ``every_n_layers`` leaves unchecked, collect logits with a graph:
    those must stay bit-for-bit native, not gain a second copy of the term."""
    batches = _batches(_build(family, balanced=False))
    native = _step(_build(family, balanced=False), batches)[0]

    torch.testing.assert_close(_step(_build(family, balanced=True), batches)[0], native, rtol=0, atol=0)

    mixed = _build(family, balanced=True)
    _checkpoint(mixed, every_n_layers=2)
    torch.testing.assert_close(_step(mixed, batches)[0], native, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("family", ["qwen3_moe", "gpt_oss"])
def test_aux_term_added_outside_the_forward(family):
    """TRL's KTO and the CP wrapper add ``coef * outputs.aux_loss`` themselves, from a forward without
    labels; the gradient must follow that term with whatever weight the trainer gives it."""
    batches = _batches(_build(family, balanced=False))
    native = _step(_build(family, balanced=False), batches, labels=False)[0]
    model = _build(family, balanced=True)
    _checkpoint(model)
    unrouted = _build(family, balanced=False)
    _checkpoint(unrouted)
    _assert_matches_native(_step(model, batches, labels=False)[0], native, _step(unrouted, batches, labels=False)[0])


@pytest.mark.parametrize("family", ["qwen3_moe", "gpt_oss"])
def test_gradient_accumulation_and_interleaved_no_grad_forwards(family):
    """Each micro-batch's backward consumes only its own forward's gradients, and a ``no_grad``
    forward between a forward and its backward (a reference or KL pass) leaves no trace."""
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


@pytest.mark.parametrize("family", ["qwen3_moe", "gpt_oss"])
def test_ep_wrapper_checkpoint_replay(family):
    """The EP wrapper owns the router at ``ep_size=1`` and replays its dispatch inside a checkpoint
    scope; the gradient must ride the wrapper's output back to the adopted router."""
    batches = _batches(_build(family, balanced=False))
    native = _step(_build(family, balanced=False, ep=True), batches)[0]

    model = _build(family, balanced=True, ep=True)
    assert any(isinstance(module, EPMoELayerBase) for module in model.modules())
    _checkpoint(model, ep=True)
    unrouted = _build(family, balanced=False, ep=True)
    _checkpoint(unrouted, ep=True)
    _assert_matches_native(_step(model, batches)[0], native, _step(unrouted, batches)[0])


def test_cp_wrapper_resolves_and_routes_through_its_inner_model(tmp_path):
    """The CP wrapper forwards ``output_router_logits`` into the causal LM it wraps, so ``auto``
    reads that model's forward, and the hooks land on the backbone beneath the wrapper."""
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'pg'}")
    try:
        inner = _build("qwen3_moe", balanced=False)
        # The Ulysses validator reads a flash label; no attention kernel runs here.
        inner.config._attn_implementation = SUPPORTED_ATTN_IMPLEMENTATIONS[0]
        wrapper = UlyssesCPModelWrapper(inner, CPConfig(cp_size=1, world_size=1, gpus_per_node=1))
        assert resolve_balancing_mode("auto", wrapper, is_moe=True) == "aux_loss"
        apply_balancing_strategy(wrapper, "aux_loss", is_moe=True)
        assert getattr(inner.model, moe_aux_loss._INSTALLED_ATTR, None) is not None
    finally:
        dist.destroy_process_group()


def test_a_gradient_no_recompute_takes_fails_the_backward(monkeypatch):
    """A kept gradient that never reaches its router must stop the step, not vanish silently."""
    monkeypatch.setattr(moe_aux_loss._RouterAuxGradient, "_on_owner", lambda self, owner, args, output: None)
    model = _build("qwen3_moe", balanced=True)
    _checkpoint(model)
    batch = _batches(model)[0]
    with pytest.raises(RuntimeError, match="no recompute of the block owning those routers"):
        model(**batch, labels=batch["input_ids"]).loss.backward()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
