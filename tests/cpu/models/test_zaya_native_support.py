#!/usr/bin/env python
"""Upstream Zaya (transformers >= 5.14) + the toolkit patches that make it trainable.

The hub modeling ships the native ``balancing_biases`` buffer but no load recording, allows
gradient checkpointing that faults in cuDNN on the training image, and never hands
``position_ids`` to its attention layers. ``apply_zaya_patches`` closes all three at model load;
these tests pin each mechanism, plus the facts the native path relies on: the registration is
not shadowed, and the CCA convolution still crosses packed document boundaries (the documented
mixer-class sharing).

    python tests/cpu/models/test_zaya_native_support.py
"""

import pytest
import torch
import transformers.models.zaya.modeling_zaya as zaya_modeling
from accelerate import PartialState
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.zaya import ZayaConfig
from transformers.models.zaya.modeling_zaya import ZayaForCausalLM

from src.models.loading import model_preparation
from src.models.moe_balancing import resolve_balancing_mode
from src.models.patches.remote_code_compat import apply_remote_code_compat_shims
from src.models.patches.zaya import (
    patch_zaya_flash_packed_position_ids,
    patch_zaya_gradient_checkpointing_refusal,
    patch_zaya_router_load_recording,
)
from tests.common.models import TINY_ZAYA_CONFIG

PartialState()  # the patches log through accelerate's logger, which needs the state initialized


SEED = 1234


def _tiny_model() -> ZayaForCausalLM:
    torch.manual_seed(SEED)
    return ZayaForCausalLM(ZayaConfig(**TINY_ZAYA_CONFIG)).eval()


def _packed_inputs(doc_lens: tuple[int, ...], seed: int = 7):
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(4, 512, (1, sum(doc_lens)), generator=g)
    position_ids = torch.cat([torch.arange(n) for n in doc_lens]).unsqueeze(0)
    return input_ids, position_ids


def test_native_registration_is_not_shadowed():
    """The compat shims must leave the native zaya registration alone — a vendored copy registering
    over it (exist_ok=True) silently runs 5.11-era code against 5.14 checkpoints."""
    apply_remote_code_compat_shims()
    assert CONFIG_MAPPING["zaya"].__module__.startswith("transformers.")


def test_balancing_auto_resolves_bias_update_only_with_the_patch():
    """Unpatched, the router lacks the ``expert_load_counter`` slot, detection yields nothing, and
    ``auto`` silently trains WITHOUT balancing — the patch is load-bearing, not cosmetic."""
    model = _tiny_model()
    # The patch mutates the hub class process-wide, and any earlier suite may have applied it, so
    # restore upstream first: skipping the unpatched half would leave only the trivial direction.
    router_cls = zaya_modeling.ZayaRouter
    router_cls.forward = getattr(router_cls.forward, "__wrapped__", router_cls.forward)
    for stamped in ("expert_load_counter", "balancing_active", "_has_discard_expert_slot"):
        if stamped in router_cls.__dict__:
            delattr(router_cls, stamped)
    assert resolve_balancing_mode("auto", model, is_moe=True) != "bias_update"

    patch_zaya_router_load_recording()
    assert resolve_balancing_mode("auto", model, is_moe=True) == "bias_update"


def test_load_recording_counts_expert_selections():
    """The recorder creates its own counter (step-0 microbatches must not be lost to the callback's
    lazy init) and every token lands somewhere: top_k=1 → one count per token."""
    patch_zaya_router_load_recording()
    model = _tiny_model().train()
    router = model.model.layers[1].mlp.gate  # layer 0's router has no EDA but records the same
    assert router.expert_load_counter is None

    input_ids, position_ids = _packed_inputs((5, 4))
    model(input_ids=input_ids, position_ids=position_ids, use_cache=False)
    # Zaya's native buffer always exists, so recording keys on the callback's arming stamp —
    # a moe_balancing=none run must not accumulate counts nothing consumes.
    assert router.expert_load_counter is None

    router.balancing_active = True  # RouterBiasBalancingCallback stamps this at train begin
    model(input_ids=input_ids, position_ids=position_ids, use_cache=False)
    assert router.expert_load_counter is not None  # created by the recorder, not the callback
    assert router.expert_load_counter.sum().item() == input_ids.shape[1]

    model.eval()
    before = router.expert_load_counter.clone()
    model(input_ids=input_ids, position_ids=position_ids, use_cache=False)
    assert torch.equal(router.expert_load_counter, before)  # eval forwards must not pollute loads


def test_discarded_tokens_count_toward_the_discard_slot():
    """The router masks discarded tokens to INDEX 0 before returning — counting them as expert-0
    load would drive the balancing callback to starve expert 0. The recorder routes them back to
    the last slot, which the bias update excludes."""
    patch_zaya_router_load_recording()
    model = _tiny_model().train()
    router = model.model.layers[1].mlp.gate
    router.balancing_active = True  # RouterBiasBalancingCallback stamps this at train begin
    with torch.no_grad():  # force every token onto the discard slot
        router.balancing_biases.fill_(-1e9)
        router.balancing_biases[-1] = 1e9

    input_ids, position_ids = _packed_inputs((5, 4))
    model(input_ids=input_ids, position_ids=position_ids, use_cache=False)
    counter = router.expert_load_counter
    assert counter[-1].item() == input_ids.shape[1], "discarded tokens must land on the discard slot"
    assert counter[:-1].sum().item() == 0.0, "discarded tokens leaked into real experts' loads"


def test_gradient_checkpointing_is_refused():
    patch_zaya_gradient_checkpointing_refusal()
    model = _tiny_model()
    with pytest.raises(ValueError, match="gradient checkpointing"):
        model.gradient_checkpointing_enable()


def test_flash_interface_receives_position_ids_only_with_the_patch():
    """Upstream ``ZayaModel.forward`` never forwards position_ids into its layers, so flash varlen
    packing silently degrades to one dense sequence. If the UNPATCHED half fails, transformers
    fixed it upstream — retire ``patch_zaya_flash_packed_position_ids``."""
    model = _tiny_model()
    input_ids, position_ids = _packed_inputs((5, 4, 3))

    calls: list[bool] = []

    def spy(module, query, key, value, attention_mask, **kwargs):
        calls.append(kwargs.get("position_ids") is not None)
        kwargs.pop("position_ids", None)
        return zaya_modeling.eager_attention_forward(module, query, key, value, attention_mask, **kwargs)

    original_forward = zaya_modeling.ZayaModel.forward
    original_registry = zaya_modeling.ALL_ATTENTION_FUNCTIONS
    zaya_modeling.ALL_ATTENTION_FUNCTIONS.register("flash_spy", spy)
    model.config._attn_implementation = "flash_spy"
    try:
        model(input_ids=input_ids, position_ids=position_ids, use_cache=False)
        assert calls and not any(calls), (
            "flash interfaces received position_ids WITHOUT the patch — transformers fixed "
            "ZayaModel upstream; retire patch_zaya_flash_packed_position_ids"
        )

        patch_zaya_flash_packed_position_ids()
        calls.clear()
        model(input_ids=input_ids, position_ids=position_ids, use_cache=False)
        assert calls and all(calls), "the patch did not deliver position_ids to the flash interface"

        patched_forward = zaya_modeling.ZayaModel.forward
        patch_zaya_flash_packed_position_ids()
        assert zaya_modeling.ZayaModel.forward is patched_forward, "patch must be idempotent"
    finally:
        zaya_modeling.ZayaModel.forward = original_forward
        zaya_modeling.ALL_ATTENTION_FUNCTIONS = original_registry


def test_cca_crosses_packed_documents_by_construction():
    """Doc B's logits move when doc A changes even though the packed attention mask is correct —
    the CCA conv + delayed-value recurrence carry state across the boundary (documented in
    agent-docs/models/zaya.md). If this ever reads ~0, upstream made the CCA document-aware — update the
    per-family isolation matrix in agent-docs/data/collators.md."""
    model = _tiny_model()
    model.config._attn_implementation = "sdpa"
    lens = (6, 6)
    input_ids, position_ids = _packed_inputs(lens)
    variant = input_ids.clone()
    variant[0, : lens[0]] = (variant[0, : lens[0]] + 17) % 512

    with torch.no_grad():
        base = model(input_ids=input_ids, position_ids=position_ids, use_cache=False).logits
        swapped = model(input_ids=variant, position_ids=position_ids, use_cache=False).logits

    drift = (base[0, lens[0] :] - swapped[0, lens[0] :]).abs().max().item()
    assert drift > 1e-3


def test_loader_seam_dispatches_the_zaya_patches(monkeypatch):
    """``apply_family_attention_patches`` is the seam both production loaders call; if its zaya
    gate is dropped, training silently loses bias-update balancing, accepts GC into the cuDNN
    fault, and leaks across packed documents on flash — with every direct-call test still green."""
    calls = []
    monkeypatch.setattr(model_preparation, "apply_zaya_patches", calls.append)

    model_preparation.apply_family_attention_patches(ZayaConfig(**TINY_ZAYA_CONFIG), "flash_attention_2")
    assert calls == ["flash_attention_2"]

    calls.clear()
    model_preparation.apply_family_attention_patches(CONFIG_MAPPING["qwen3"](), "flash_attention_2")
    assert calls == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
