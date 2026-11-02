#!/usr/bin/env python
"""Upstream Zaya (transformers >= 5.14) + the toolkit patches that make it trainable.

The hub modeling ships the native ``balancing_biases`` buffer but no load recording, allows
gradient checkpointing that faults in cuDNN on the training image, and never hands
``position_ids`` to its attention layers or CCA convolutions, so a packed row's documents mix.
``apply_zaya_patches`` closes all three at model load; these tests pin each mechanism, the CCA
rewrite's equality with upstream on single-document rows, and that the registration is not shadowed.

    python tests/cpu/models/test_zaya_native_support.py
"""

import pytest
import torch
import transformers.models.zaya.modeling_zaya as zaya_modeling
from transformers import DynamicCache
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.zaya import ZayaConfig
from transformers.models.zaya.modeling_zaya import ZayaForCausalLM

from halo.model_loading import model_preparation
from halo.model_loading.remote_code import apply_remote_code_compat_shims
from halo.models.moe.balancing import resolve_balancing_mode
from halo.models.zaya import (
    _same_document_masks,
    is_zaya_cca_document_isolated,
    patch_zaya_gradient_checkpointing_refusal,
    patch_zaya_packed_document_isolation,
    patch_zaya_router_load_recording,
)
from tests.common.models import TINY_ZAYA_CONFIG

SEED = 1234


def _tiny_model(**overrides) -> ZayaForCausalLM:
    torch.manual_seed(SEED)
    return ZayaForCausalLM(ZayaConfig(**{**TINY_ZAYA_CONFIG, **overrides})).eval()


def _packed_inputs(doc_lens: tuple[int, ...], seed: int = 7):
    g = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(4, 512, (1, sum(doc_lens)), generator=g)
    position_ids = torch.cat([torch.arange(n) for n in doc_lens]).unsqueeze(0)
    return input_ids, position_ids


def _upstream(forward):
    return getattr(forward, "__wrapped__", forward)


@pytest.fixture
def upstream_zaya():
    """Start from the upstream forwards the isolation patch wraps (an earlier test may have applied it,
    process-wide), and restore whatever was installed afterwards."""
    model_cls, cca_cls = zaya_modeling.ZayaModel, zaya_modeling.ZayaCCAProjection
    saved = (model_cls.forward, cca_cls.forward, zaya_modeling.ALL_ATTENTION_FUNCTIONS)
    model_cls.forward, cca_cls.forward = _upstream(model_cls.forward), _upstream(cca_cls.forward)
    zaya_modeling.ALL_ATTENTION_FUNCTIONS = getattr(saved[2], "_inner", saved[2])
    yield
    model_cls.forward, cca_cls.forward, zaya_modeling.ALL_ATTENTION_FUNCTIONS = saved


def _doc_b_drift(model, lens: tuple[int, int]) -> tuple[float, float]:
    """How far doc B's logits move when doc A changes, and how far packed doc B is from doc B alone."""
    input_ids, position_ids = _packed_inputs(lens)
    variant = input_ids.clone()
    variant[0, : lens[0]] = (variant[0, : lens[0]] + 17) % 512
    with torch.no_grad():
        base = model(input_ids=input_ids, position_ids=position_ids, use_cache=False).logits[0, lens[0] :]
        swapped = model(input_ids=variant, position_ids=position_ids, use_cache=False).logits[0, lens[0] :]
        alone = model(input_ids=input_ids[:, lens[0] :], use_cache=False).logits[0]
    return (base - swapped).abs().max().item(), (base - alone).abs().max().item()


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

    router.balancing_active = True  # BalancingBiasCallback stamps this at train begin
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
    router.balancing_active = True  # BalancingBiasCallback stamps this at train begin
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


def test_flash_interface_receives_position_ids_only_with_the_patch(upstream_zaya):
    """Upstream ``ZayaModel.forward`` never forwards position_ids into its layers, so flash varlen
    packing silently degrades to one dense sequence. If the UNPATCHED half fails, transformers
    fixed it upstream — retire the attention half of ``patch_zaya_packed_document_isolation``."""
    model = _tiny_model()
    input_ids, position_ids = _packed_inputs((5, 4, 3))

    calls: list[bool] = []

    def spy(module, query, key, value, attention_mask, **kwargs):
        calls.append(kwargs.get("position_ids") is not None)
        kwargs.pop("position_ids", None)
        return zaya_modeling.eager_attention_forward(module, query, key, value, attention_mask, **kwargs)

    zaya_modeling.ALL_ATTENTION_FUNCTIONS.register("flash_spy", spy)
    model.config._attn_implementation = "flash_spy"
    model(input_ids=input_ids, position_ids=position_ids, use_cache=False)
    assert calls and not any(calls), (
        "flash interfaces received position_ids WITHOUT the patch — transformers fixed "
        "ZayaModel upstream; retire the attention half of patch_zaya_packed_document_isolation"
    )

    patch_zaya_packed_document_isolation()
    calls.clear()
    model(input_ids=input_ids, position_ids=position_ids, use_cache=False)
    assert calls and all(calls), "the patch did not deliver position_ids to the flash interface"

    patched = (zaya_modeling.ZayaModel.forward, zaya_modeling.ZayaCCAProjection.forward)
    patch_zaya_packed_document_isolation()
    assert (zaya_modeling.ZayaModel.forward, zaya_modeling.ZayaCCAProjection.forward) == patched, (
        "patch must be idempotent"
    )


@pytest.mark.parametrize(("lens", "kernels"), [((6, 6), (2, 2)), ((1, 7), (2, 2)), ((2, 5), (2, 2)), ((3, 6), (3, 3))])
def test_cca_isolates_packed_documents_only_with_the_patch(upstream_zaya, lens, kernels):
    """Upstream, doc B's logits move when doc A changes although the packed attention mask is correct:
    the CCA conv taps and the delayed value read doc A's last tokens. Patched, packed doc B reads as
    doc B alone, at every document length the 2-token receptive field and the delay can straddle. If the
    UNPATCHED half reads ~0, upstream made the CCA document-aware: retire the CCA half of the patch."""
    # fp64 (the grouped expert GEMM takes no fp64, so eager experts), so a leak through a single tap reads
    # far above the reduction-order noise.
    model = _tiny_model(cca_time0=kernels[0], cca_time1=kernels[1], experts_implementation="eager").double()
    model.config._attn_implementation = "sdpa"
    # Zaya initializes the key temperature to zero, which flattens attention and hides the q/k convs.
    for module in model.modules():
        if isinstance(module, zaya_modeling.ZayaQKNorm):
            torch.nn.init.normal_(module.temp, std=1.0)
    drift, _ = _doc_b_drift(model, lens)
    assert drift > 1e-3, "upstream CCA no longer crosses documents"

    patch_zaya_packed_document_isolation()
    drift, versus_alone = _doc_b_drift(model, lens)
    # Not 0.0: doc A changes each expert's batch, which reorders the reductions (~1e-16).
    assert drift < 1e-12, f"doc B moved {drift:.2e} when doc A changed"
    assert versus_alone < 1e-12, f"packed doc B differs from doc B alone by {versus_alone:.2e}"


@pytest.mark.parametrize(
    "rows",
    [
        [[0, 1, 2, 0, 1, 0, 1, 2, 3]],
        [[0, 0, 1, 0, 1, 2, 0, 0, 1], [0, 1, 2, 3, 4, 5, 6, 7, 8]],
        [[3, 4, 0, 1, 2, 0, 1, 2, 3]],
    ],
)
def test_cca_same_document_masks_match_a_per_token_reference(rows):
    """Per shift, token ``t`` keeps token ``t - s`` exactly when no position restart lies in ``(t - s, t]``:
    packed rows, one-token documents, a batch and a row that opens mid-document."""
    position_ids = torch.tensor(rows)
    max_shift = 4
    masks = _same_document_masks(position_ids, position_ids.shape[1], max_shift, position_ids.device)
    for shift, mask in enumerate(masks, start=1):
        expected = [
            [t >= shift and all(row[u] != 0 for u in range(t - shift + 1, t + 1)) for t in range(len(row))]
            for row in rows
        ]
        assert mask.tolist() == expected, shift
    one_row = _same_document_masks(None, 5, 2, position_ids.device)
    assert [mask.tolist() for mask in one_row] == [
        [[False, True, True, True, True]],
        [[False, False, True, True, True]],
    ]


@pytest.mark.parametrize("kernels", [(2, 2), (3, 2), (2, 4)])
def test_patched_cca_equals_upstream_on_one_document(upstream_zaya, kernels):
    """On a single-document row the rewrite is upstream's conv pair and delayed value: q/k/v and every
    parameter gradient match in fp64 on the projection alone, at the release's kernel sizes and others."""
    torch.manual_seed(SEED)
    config = ZayaConfig(**{**TINY_ZAYA_CONFIG, "cca_time0": kernels[0], "cca_time1": kernels[1]})
    projection = zaya_modeling.ZayaCCAProjection(config, layer_idx=0).double()
    for param in projection.parameters():
        torch.nn.init.normal_(param, std=0.3)
    hidden_states = torch.randn(2, 13, TINY_ZAYA_CONFIG["hidden_size"], dtype=torch.float64)

    def run():
        projection.zero_grad()
        outputs = projection(hidden_states, None)
        sum(
            (out * torch.linspace(-1, 1, out.numel(), dtype=out.dtype).view_as(out)).sum() for out in outputs
        ).backward()
        grads = {name: None if p.grad is None else p.grad.clone() for name, p in projection.named_parameters()}
        return [out.detach() for out in outputs], grads

    reference_outputs, reference_grads = run()
    patch_zaya_packed_document_isolation()
    outputs, grads = run()
    for reference, out in zip(reference_outputs, outputs, strict=True):
        assert torch.allclose(out, reference, rtol=0, atol=1e-12)
    for name, grad in grads.items():
        assert grad is not None and torch.allclose(grad, reference_grads[name], rtol=0, atol=1e-12), name


@pytest.mark.parametrize("positions", ["generate", "unset"])
def test_patched_cca_equals_upstream_on_a_left_padded_batch(upstream_zaya, positions):
    """A left-padded batch gives the same logits on every real token as upstream, which zeroes the pads
    through its ``conv_mask``. Generation's positions (pads at 1, the first real token at 0) start a
    document at the first real token; unset positions leave one document per row, where the zeroed pads
    alone keep the row start clean."""
    model = _tiny_model(experts_implementation="eager").double()
    model.config._attn_implementation = "sdpa"
    for module in model.modules():
        if isinstance(module, zaya_modeling.ZayaQKNorm):
            torch.nn.init.normal_(module.temp, std=1.0)
    input_ids = torch.cat([_packed_inputs((9,), seed=seed)[0] for seed in (7, 8)])
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, :3] = 0
    position_ids = None
    if positions == "generate":
        position_ids = (attention_mask.cumsum(-1) - 1).masked_fill(attention_mask == 0, 1)

    def run():
        with torch.no_grad():
            return model(
                input_ids=input_ids, attention_mask=attention_mask, position_ids=position_ids, use_cache=False
            ).logits

    reference = run()
    patch_zaya_packed_document_isolation()
    real = attention_mask.bool()
    assert torch.allclose(run()[real], reference[real], rtol=0, atol=1e-10)


def test_patched_cca_prefill_keeps_the_decode_cache(upstream_zaya):
    """A cache prefill through the rewrite leaves upstream's conv and recurrent state: the next decode step
    (upstream's own path) reads the same logits."""
    model = _tiny_model()
    model.config._attn_implementation = "sdpa"
    input_ids, _ = _packed_inputs((11,))

    def prefill_then_step():
        cache = DynamicCache(config=model.config)
        with torch.no_grad():
            model(input_ids=input_ids[:, :-1], past_key_values=cache, use_cache=True)
            return model(input_ids=input_ids[:, -1:], past_key_values=cache, use_cache=True).logits

    reference = prefill_then_step()
    patch_zaya_packed_document_isolation()
    assert torch.allclose(prefill_then_step(), reference, rtol=0, atol=1e-5)


def test_patched_cca_refuses_an_adapted_conv(upstream_zaya):
    """An adapter on a CCA conv would be silently bypassed (the rewrite reads the conv weights)."""
    patch_zaya_packed_document_isolation()
    model = _tiny_model()
    projection = model.model.layers[0].self_attn.qkv_proj
    projection.conv_qk_grouped = torch.nn.Sequential(projection.conv_qk_grouped)
    input_ids, _ = _packed_inputs((5,))
    with pytest.raises(TypeError, match="lora_target_modules"):
        model(input_ids=input_ids, use_cache=False)


def test_loader_seam_dispatches_the_zaya_patches(monkeypatch):
    """``apply_family_patches`` is the seam both production loaders call; if its zaya
    gate is dropped, training silently loses bias-update balancing, accepts GC into the cuDNN
    fault, and leaks across packed documents — with every direct-call test still green."""
    calls = []
    monkeypatch.setattr(model_preparation, "apply_zaya_patches", lambda: calls.append("zaya"))

    model_preparation.apply_family_patches(ZayaConfig(**TINY_ZAYA_CONFIG), "sdpa")
    assert calls == ["zaya"]

    calls.clear()
    model_preparation.apply_family_patches(CONFIG_MAPPING["qwen3"](), "flash_attention_2")
    assert calls == []


def test_loader_seam_installs_the_document_isolation(upstream_zaya):
    """Through the seam the loaders call, the CCA projection is the document-isolated one: a dropped
    ``patch_zaya_packed_document_isolation`` in ``apply_zaya_patches`` leaves every direct-call test green."""
    assert not is_zaya_cca_document_isolated()
    model_preparation.apply_family_patches(ZayaConfig(**TINY_ZAYA_CONFIG), "sdpa")
    assert is_zaya_cca_document_isolated()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
