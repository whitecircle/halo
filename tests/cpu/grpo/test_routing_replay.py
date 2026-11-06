"""MoE routing replay (R2): forced-selection math on the EP layer bases, per-family gate-weight formulas checked
against the real HF routers, single-count expert load under GC recompute, and the injector's assemble/arm cycle."""

import base64

import numpy as np
import pytest
import torch
from torch import nn
from transformers import PretrainedConfig
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTopKRouter
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeTopKRouter
from transformers.models.qwen3_next.modeling_qwen3_next import Qwen3NextTopKRouter

from halo.distributed.expert_parallel.families.bailing import EPBailingMoELayer
from halo.distributed.expert_parallel.families.gpt_oss import EPGptOssMoELayer
from halo.distributed.expert_parallel.families.qwen3 import EPQwen3MoELayer
from halo.distributed.expert_parallel.families.qwen3_5 import EPQwen3_5MoELayer
from halo.distributed.expert_parallel.families.qwen3_next import EPQwen3NextMoELayer
from halo.models.gradient_checkpointing import RecomputeScope
from halo.models.moe.family_specs import MoEFamilySpec
from halo.trainers.grpo.environmental.rollout_routing import build_rollout_routing
from halo.trainers.grpo.environmental.routing_replay import (
    RoutingReplayInjector,
    build_routing_replay_injector,
    decode_rollout_routing,
    decoder_layer_indices,
)
from tests.common.routing import BareEPLayer, npy_routing_payload, raw_routing_payload

T, E, K = 6, 8, 2  # tokens, experts, top_k


class _BareLayer(BareEPLayer):
    def __init__(self, **attrs):  # noqa: D107 — test double
        super().__init__(top_k=K, num_experts=E, **attrs)


class _SoftmaxGatedLayer(_BareLayer):
    """The softmax over the selected logits, as GptOss's router gates and Qwen3.5's biased route does."""

    _gate_weights_at = EPGptOssMoELayer._gate_weights_at


def _injector(layers) -> RoutingReplayInjector:
    return RoutingReplayInjector(layers, engine_layers=len(layers), layer_indices=list(range(len(layers))))


def _shuffled_valid_indices(natural: torch.Tensor) -> torch.Tensor:
    """Differs from ``natural`` in every row."""
    return (natural + 1) % E


def test_unforced_path_returns_input_unchanged():
    layer = _BareLayer()
    idx = torch.randint(0, E, (T, K))
    out = layer._apply_routing_replay(idx)
    assert out is idx  # the default path must be allocation-free


def test_forced_indices_replace_natural_and_count_flips():
    layer = _BareLayer()
    natural = torch.stack([torch.randperm(E)[:K] for _ in range(T)])
    forced = _shuffled_valid_indices(natural)
    layer._forced_topk_indices = forced
    out = layer._apply_routing_replay(natural)
    assert torch.equal(out, forced)
    flipped, total = layer._replay_flip_counts.tolist()
    assert (flipped, total) == (T, T)


def test_sentinel_rows_keep_natural_selection():
    layer = _BareLayer()
    natural = torch.stack([torch.randperm(E)[:K] for _ in range(T)])
    forced = _shuffled_valid_indices(natural)
    forced[0] = -1  # sentinel: no mask for this token
    layer._forced_topk_indices = forced
    out = layer._apply_routing_replay(natural)
    assert torch.equal(out[0], natural[0])
    assert torch.equal(out[1:], forced[1:])
    flipped, total = layer._replay_flip_counts.tolist()
    assert total == T - 1


def test_matching_forced_mask_counts_zero_flips():
    layer = _BareLayer()
    natural = torch.stack([torch.randperm(E)[:K] for _ in range(T)])
    layer._forced_topk_indices = natural.flip(dims=[1])  # same set per token, different order
    out = layer._apply_routing_replay(natural)
    assert torch.equal(out.sort(dim=-1).values, natural.sort(dim=-1).values)
    flipped, total = layer._replay_flip_counts.tolist()
    assert (flipped, total) == (0, T)


def test_mask_exhaustion_and_slice_shape_raise():
    layer = _BareLayer()
    layer._forced_topk_indices = torch.zeros(T - 1, K, dtype=torch.long)
    with pytest.raises(RuntimeError, match="exhausted"):
        layer._apply_routing_replay(torch.randint(0, E, (T, K)))
    layer = _BareLayer()
    layer._forced_topk_indices = torch.zeros(T, K + 1, dtype=torch.long)
    with pytest.raises(RuntimeError, match="does not match"):
        layer._apply_routing_replay(torch.randint(0, E, (T, K)))


def test_cursor_consumes_chunked_forwards_in_order():
    # The loss forward is row-chunked, so the armed mask spans several forwards, and exact exhaustion wraps for the
    # mu>1 reuse pass.
    layer = _BareLayer()
    full = torch.stack([torch.randperm(E)[:K] for _ in range(2 * T)])
    layer._forced_topk_indices = full
    natural = torch.zeros(T, K, dtype=torch.long)
    first = layer._apply_routing_replay(natural)
    second = layer._apply_routing_replay(natural)
    assert torch.equal(first, full[:T]) and torch.equal(second, full[T:])
    assert layer._forced_cursor == 0
    assert torch.equal(layer._apply_routing_replay(natural), full[:T])


def test_concurrent_checkpoint_frames_replay_their_own_slice():
    # A pipeline stage keeps several microbatches in flight, so two checkpoint frames are open at once. A layer-level
    # cache would hand frame 1's recompute the slice frame 2's forward wrote.
    layer = _BareLayer()
    layer.train()
    full = torch.stack([torch.randperm(E)[:K] for _ in range(2 * T)])
    layer._forced_topk_indices = full
    natural = torch.zeros(T, K, dtype=torch.long)
    frame1, frame2 = RecomputeScope(), RecomputeScope()
    with frame1:
        layer._apply_routing_replay(natural)
    with frame2:
        layer._apply_routing_replay(natural)
    with frame1:  # microbatch 1 recompute
        assert torch.equal(layer._apply_routing_replay(natural), full[:T])
    with frame2:
        assert torch.equal(layer._apply_routing_replay(natural), full[T:])


def test_recompute_call_the_forward_never_made_raises():
    # A recompute reaching the router when the original forward did not (mask armed mid-backward) would route on stale
    # indices.
    layer = _BareLayer()
    layer.train()
    scope = RecomputeScope()
    with scope:
        pass
    layer._forced_topk_indices = torch.zeros(T, K, dtype=torch.long)
    with scope, pytest.raises(RuntimeError, match="different path"):
        layer._apply_routing_replay(torch.zeros(T, K, dtype=torch.long))


def test_gc_recompute_skips_flip_metric_but_still_replaces():
    layer = _BareLayer()
    layer.train()
    natural = torch.stack([torch.randperm(E)[:K] for _ in range(T)])
    forced = _shuffled_valid_indices(natural)
    layer._forced_topk_indices = forced
    scope = RecomputeScope()
    with scope:
        layer._apply_routing_replay(natural)  # original forward
    counts_after_original = layer._replay_flip_counts.clone()
    with scope:  # recompute
        out = layer._apply_routing_replay(natural)
    assert torch.equal(out, forced)
    assert torch.equal(layer._replay_flip_counts, counts_after_original)


def test_group_limited_topk_forced_equals_natural():
    layer = _BareLayer()
    scores = torch.rand(T, E)
    base_idx, base_w = layer._group_limited_topk(scores, scores)
    layer._forced_topk_indices = base_idx.clone()
    forced_idx, forced_w = layer._group_limited_topk(scores, scores)
    assert torch.equal(forced_idx, base_idx)
    assert torch.equal(forced_w, base_w)


def test_group_limited_topk_forced_weights_are_live_gathers():
    layer = _BareLayer()
    scores = torch.rand(T, E)
    natural_idx, _ = layer._group_limited_topk(scores, scores)
    forced = _shuffled_valid_indices(natural_idx)
    layer._forced_topk_indices = forced
    idx, w = layer._group_limited_topk(scores, scores)
    assert torch.equal(idx, forced)
    expected = scores.gather(1, forced)
    expected = expected / (expected.sum(dim=-1, keepdim=True) + 1e-20)
    assert torch.allclose(w, expected)


def test_biased_route_forced_equals_natural():
    layer = _SoftmaxGatedLayer()
    layer.balancing_biases = torch.zeros(E)
    layer.expert_load_counter = None
    logits = torch.randn(T, E)
    base_idx, base_scores = layer._biased_route(logits)
    layer._forced_topk_indices = base_idx.clone()
    forced_idx, forced_scores = layer._biased_route(logits)
    assert torch.equal(forced_idx, base_idx)
    assert torch.equal(forced_scores, base_scores)


def test_forced_selection_keeps_router_grad_path():
    # Replaying weights (the RSPO anti-pattern) would show zero grad at the forced indices.
    layer = _SoftmaxGatedLayer()
    layer.balancing_biases = torch.zeros(E)
    layer.expert_load_counter = None
    logits = torch.randn(T, E, requires_grad=True)
    natural_idx, _ = layer._biased_route(logits)
    layer._forced_topk_indices = _shuffled_valid_indices(natural_idx)
    idx, scores = layer._biased_route(logits)
    # A plain scores.sum() is constant (softmax rows sum to 1) and would zero the grad by algebra.
    (scores * torch.randn_like(scores)).sum().backward()
    grad_at_forced = logits.grad.gather(1, idx)
    assert torch.isfinite(logits.grad).all()
    assert (grad_at_forced.abs().sum(dim=-1) > 0).all(), "router gradient died at the forced selection"


def _router_config(cls, **extra):
    class _Cfg:
        num_experts_per_tok = K
        num_experts = E
        hidden_size = 16
        norm_topk_prob = True

    for k, v in extra.items():
        setattr(_Cfg, k, v)
    return _Cfg


def test_qwen3_gate_weights_at_matches_router():
    router = Qwen3MoeTopKRouter(_router_config(None))
    torch.nn.init.normal_(router.weight)
    hidden = torch.randn(T, 16)
    logits, scores, indices = router(hidden)

    layer = EPQwen3MoELayer.__new__(EPQwen3MoELayer)
    nn.Module.__init__(layer)
    layer.gate = router
    replayed = layer._gate_weights_at(logits, indices)
    assert torch.allclose(replayed, scores, atol=0, rtol=0), "Qwen3 weighting formula drifted from the HF router"


def test_qwen3_5_gate_weights_at_matches_router():
    router = Qwen3_5MoeTopKRouter(_router_config(None))
    torch.nn.init.normal_(router.weight)
    hidden = torch.randn(T, 16)
    logits, scores, indices = router(hidden)

    layer = EPQwen3_5MoELayer.__new__(EPQwen3_5MoELayer)
    nn.Module.__init__(layer)
    layer.gate = router
    replayed = layer._gate_weights_at(logits, indices)
    assert torch.allclose(replayed, scores, atol=0, rtol=0), "Qwen3.5 weighting formula drifted from the HF router"


@pytest.mark.parametrize("norm_topk_prob", [True, False])
def test_qwen3_next_gate_weights_at_matches_router(norm_topk_prob):
    """Qwen3.5-MoE's block, but the router renormalizes only under its own flag, and replay must follow it."""
    router = Qwen3NextTopKRouter(_router_config(None, norm_topk_prob=norm_topk_prob))
    torch.nn.init.normal_(router.weight)
    hidden = torch.randn(T, 16)
    logits, scores, indices = router(hidden)

    layer = EPQwen3NextMoELayer.__new__(EPQwen3NextMoELayer)
    nn.Module.__init__(layer)
    layer.gate = router
    replayed = layer._gate_weights_at(logits, indices)
    assert torch.allclose(replayed, scores, atol=0, rtol=0), "Qwen3-Next weighting formula drifted from the HF router"


def test_bailing_gate_weights_at_formula():
    # BailingMoeV2Gate is hub remote code (not importable here); check against its documented formula:
    # sigmoid -> gather -> renorm(+1e-20) -> routed_scaling_factor (gate expert_bias is selection-only).
    class _Gate(nn.Module):
        routed_scaling_factor = 2.5

    layer = EPBailingMoELayer.__new__(EPBailingMoELayer)
    nn.Module.__init__(layer)
    layer.gate = _Gate()
    layer.top_k = K
    logits = torch.randn(T, E)
    indices = torch.stack([torch.randperm(E)[:K] for _ in range(T)])
    replayed = layer._gate_weights_at(logits, indices)
    scores = torch.sigmoid(logits.float()).gather(1, indices)
    expected = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) * 2.5
    assert torch.allclose(replayed, expected)


def test_record_expert_load_skips_gc_recompute():
    layer = _BareLayer()
    layer.train()
    layer.balancing_biases = torch.zeros(E)
    layer.expert_load_counter = None
    idx = torch.randint(0, E, (T, K))
    scope = RecomputeScope()
    with scope:  # original forward
        layer._record_expert_load(idx)
    counts_after_original = layer.expert_load_counter.clone()
    with scope:  # backward recompute
        layer._record_expert_load(idx)
    assert torch.equal(layer.expert_load_counter, counts_after_original), "GC recompute double-counted"
    # A plain (non-checkpointed) training forward still counts.
    layer._record_expert_load(idx)
    assert torch.equal(layer.expert_load_counter, counts_after_original * 2)


def _capture_rows(layers, rows, seq, chunks=2):
    """Simulate a chunked no-grad forward: each layer appends per-chunk flat [b*seq, K] selections."""
    per_layer_full = [torch.randint(0, E, (rows * seq, K), dtype=torch.int16) for _ in layers]
    boundaries = [(i * rows // chunks) * seq for i in range(chunks + 1)]
    for layer, full in zip(layers, per_layer_full, strict=False):
        assert layer._capture_routing
        layer._captured_routing_chunks.extend(full[a:b] for a, b in zip(boundaries, boundaries[1:], strict=False))
    return per_layer_full


def test_injector_capture_assembles_and_arm_slices_per_layer():
    layers = [_BareLayer(), _BareLayer()]
    injector = _injector(layers)
    rows, seq = 4, 3
    with injector.capture(rows, seq) as captured:
        per_layer = _capture_rows(layers, rows, seq)
    masks = captured[0]
    assert masks.shape == (rows, seq, len(layers), K) and masks.dtype == torch.int16
    for i, full in enumerate(per_layer):
        assert torch.equal(masks[:, :, i, :].reshape(-1, K), full)
    assert not any(layer._capture_routing for layer in layers)

    injector.arm(masks)
    for i, layer in enumerate(layers):
        assert layer._forced_topk_indices.dtype == torch.long
        assert torch.equal(layer._forced_topk_indices, per_layer[i].long())
    # disarm asserts full consumption.
    for layer in layers:
        layer._apply_routing_replay(torch.zeros(rows * seq, K, dtype=torch.long))
    injector.disarm()
    assert all(layer._forced_topk_indices is None for layer in layers)


def test_injector_span_capture_scatters_and_arm_trims():
    """Capture scatters each row segment to its absolute span (padding stays -1 = natural); arm trims back to the same
    span stream, so the cursor gets exactly what the per-row forwards consume."""
    layers = [_BareLayer(), _BareLayer()]
    injector = _injector(layers)
    rows, seq = 3, 6
    spans = [(1, 5), (0, 6), (2, 3)]  # left-pad, full row, 1-token span pre-widened by the helper rule
    widths = [hi - lo for lo, hi in spans]
    per_layer_trimmed = [torch.randint(0, E, (sum(widths), K), dtype=torch.int16) for _ in layers]
    with injector.capture(rows, seq, row_spans=spans) as captured:
        for layer, trimmed in zip(layers, per_layer_trimmed, strict=True):
            offset = 0
            for w in widths:
                layer._captured_routing_chunks.append(trimmed[offset : offset + w])
                offset += w
    masks = captured[0]
    assert masks.shape == (rows, seq, len(layers), K) and masks.dtype == torch.int16
    for i, trimmed in enumerate(per_layer_trimmed):
        offset = 0
        for r, (lo, hi) in enumerate(spans):
            assert torch.equal(masks[r, lo:hi, i, :], trimmed[offset : offset + (hi - lo)])
            offset += hi - lo
        outside = masks[:, :, i, :].clone()
        for r, (lo, hi) in enumerate(spans):
            outside[r, lo:hi] = -1
        assert (outside == -1).all()

    injector.arm(masks, row_spans=spans)
    for i, layer in enumerate(layers):
        assert torch.equal(layer._forced_topk_indices, per_layer_trimmed[i].long())
        for w in widths:
            layer._apply_routing_replay(torch.zeros(w, K, dtype=torch.long))
    injector.disarm()  # exact tiling: no partial-consumption raise


def test_injector_span_capture_count_mismatch_raises():
    layers = [_BareLayer()]
    injector = _injector(layers)
    with pytest.raises(RuntimeError, match="row spans"):
        with injector.capture(2, 4, row_spans=[(0, 4), (1, 3)]):  # expects 6 tokens
            layers[0]._captured_routing_chunks.append(torch.zeros(8, K, dtype=torch.int16))  # full-width 8


def test_injector_arm_span_row_count_mismatch_raises():
    injector = _injector([_BareLayer()])
    with pytest.raises(RuntimeError, match="row_spans"):
        injector.arm(torch.zeros(2, 4, 1, K, dtype=torch.int16), row_spans=[(0, 4)])


def test_injector_capture_token_count_mismatch_raises():
    layers = [_BareLayer()]
    injector = _injector(layers)
    with pytest.raises(RuntimeError, match="recorded"):
        with injector.capture(4, 3):
            layers[0]._captured_routing_chunks.append(torch.zeros(5, K, dtype=torch.int16))


def test_injector_drain_flip_rate_empties_the_counters():
    layers = [_BareLayer(), _BareLayer()]
    injector = _injector(layers)
    layers[0]._replay_flip_counts = torch.tensor([2.0, 10.0])
    layers[1]._replay_flip_counts = torch.tensor([3.0, 10.0])
    assert injector.drain_flip_rate() == pytest.approx(0.25)
    assert injector.drain_flip_rate() is None


def test_injector_rejects_unsupported_family_and_empty():
    with pytest.raises(ValueError, match="at least one EP MoE layer"):
        _injector([])
    unsupported = _BareLayer(FAMILY=MoEFamilySpec(model_types=(), supports_routing_replay=False))
    with pytest.raises(NotImplementedError, match="_BareLayer"):
        _injector([unsupported])


def test_injector_arm_shape_validation():
    injector = _injector([_BareLayer()])
    with pytest.raises(RuntimeError, match="does not match"):
        injector.arm(torch.zeros(2, 3, 5, K, dtype=torch.int16))  # 5 layers vs 1


def test_injector_disarm_flags_partial_consumption():
    layer = _BareLayer()
    injector = _injector([layer])
    injector.arm(torch.zeros(2, T, 1, K, dtype=torch.int16))
    layer._apply_routing_replay(torch.zeros(T, K, dtype=torch.long))  # consume only half
    with pytest.raises(RuntimeError, match="mis-consumed at disarm"):
        injector.disarm()
    assert layer._forced_topk_indices is None  # state still cleared despite the raise


def test_injector_capture_body_exception_propagates_and_resets():
    layers = [_BareLayer(), _BareLayer()]
    injector = _injector(layers)
    with pytest.raises(ValueError, match="boom"):  # not the capture-alignment RuntimeError
        with injector.capture(4, 3):
            layers[0]._captured_routing_chunks.append(torch.zeros(3, K, dtype=torch.int16))
            raise ValueError("boom")
    assert not any(layer._capture_routing for layer in layers)
    assert not any(layer._captured_routing_chunks for layer in layers)


def test_decode_rollout_routing_roundtrip():
    src = np.random.randint(0, E, size=(5, 3, K), dtype=np.uint16)
    out = decode_rollout_routing(npy_routing_payload(src), num_layers=3, top_k=K)
    assert out.dtype == torch.int16 and out.shape == (5, 3, K)
    assert torch.equal(out, torch.from_numpy(src.astype("int16")))
    with pytest.raises(ValueError, match="shape"):
        decode_rollout_routing(npy_routing_payload(np.zeros((5, 3), dtype=np.uint8)), num_layers=3, top_k=K)


def test_decode_rollout_routing_raw_int32():
    """SGLang's wire format is base64 raw int32 rows, the shape implied by the trainer's (L, K)."""
    src = np.random.randint(0, E, size=(5, 3, K), dtype=np.int32)
    payload = base64.b64encode(src.tobytes()).decode()
    out = decode_rollout_routing(payload, num_layers=3, top_k=K)
    assert out.dtype == torch.int16 and out.shape == (5, 3, K)
    assert torch.equal(out, torch.from_numpy(src.astype("int16")))
    with pytest.raises(ValueError, match="multiple"):
        decode_rollout_routing(payload, num_layers=4, top_k=K)
    with pytest.raises(ValueError, match="multiple"):
        decode_rollout_routing(base64.b64encode(b"").decode(), num_layers=3, top_k=K)


def test_decoder_layer_indices_read_the_layer_number_off_the_fqn():
    names = ["model.layers.3.mlp", "model.language_model.layers.7.feed_forward", "model.layers.12"]
    assert decoder_layer_indices(names) == [3, 7, 12]
    with pytest.raises(ValueError, match="decoder-layer index"):
        decoder_layer_indices(["model.mlp"])


def test_engine_rows_of_dense_layers_are_dropped_from_the_mask():
    """Both engines' capturers emit one row per decoder layer (``num_hidden_layers``), so a family whose first layers
    are dense ships rows the trainer has no EP layer for."""
    injector = RoutingReplayInjector([_BareLayer(), _BareLayer()], engine_layers=4, layer_indices=[1, 3])
    src = np.random.randint(0, E, size=(5, 4, K), dtype=np.int32)
    expected = torch.from_numpy(src[:, [1, 3], :].astype("int16"))
    assert torch.equal(injector.decode_engine_mask(raw_routing_payload(src)), expected)
    assert torch.equal(injector.decode_engine_mask(npy_routing_payload(src)), expected)
    # The raw wire carries no shape: a row count that does not divide is the only shape error it surfaces.
    with pytest.raises(ValueError, match="multiple"):
        injector.decode_engine_mask(raw_routing_payload(src[:, [1, 3], :]))
    with pytest.raises(ValueError, match="decoder layers"):
        injector.decode_engine_mask(npy_routing_payload(src[:, :3, :]))


def test_an_all_moe_model_keeps_every_engine_row():
    injector = _injector([_BareLayer(), _BareLayer(), _BareLayer()])
    src = np.random.randint(0, E, size=(2, 3, K), dtype=np.int32)
    assert torch.equal(injector.decode_engine_mask(raw_routing_payload(src)), torch.from_numpy(src.astype("int16")))


def test_the_wire_description_must_cover_every_ep_layer():
    with pytest.raises(ValueError, match="indices for 2 EP layers"):
        RoutingReplayInjector([_BareLayer(), _BareLayer()], engine_layers=4, layer_indices=[1])
    with pytest.raises(ValueError, match="distinct decoder layers"):
        RoutingReplayInjector([_BareLayer(), _BareLayer()], engine_layers=4, layer_indices=[1, 4])
    with pytest.raises(ValueError, match="distinct decoder layers"):
        RoutingReplayInjector([_BareLayer(), _BareLayer()], engine_layers=4, layer_indices=[1, 1])


def test_the_builder_reads_the_layer_count_and_indices_off_the_model():
    class _Block(nn.Module):
        def __init__(self, moe: bool):
            super().__init__()
            self.mlp = _BareLayer() if moe else nn.Identity()

    model = nn.Module()
    model.layers = nn.ModuleList([_Block(False), _Block(True), _Block(False), _Block(True)])
    model.config = PretrainedConfig(num_hidden_layers=4)
    injector = build_routing_replay_injector(model)
    assert injector.num_ep_layers == 2
    assert injector._engine_layers == 4 and injector._layer_indices == [1, 3]


def test_build_rollout_routing_conventions():
    layers = 2

    def _m(tokens):
        return torch.arange(tokens * layers * K, dtype=torch.int16).reshape(tokens, layers, K) % E

    # Row 0: full-request mask (P+C). Row 1: engine omits the last (P+C-1). Row 2: completion-only (C). Row 3: no mask.
    # Row 4: the engine's prompt count differs from the trainer's re-render, so the completion anchors on the engine's.
    prompt_lens, completion_lens = [3, 2, 2, 1, 3], [2, 3, 4, 1, 2]
    masks = [
        (_m(5), 3),
        (_m(4), 2),
        (_m(4), None),
        None,
        (_m(4), 2),  # engine says prompt=2, trainer rendered 3 → 2+2=4 tokens, full coverage @ ep=2
    ]
    routing, stats = build_rollout_routing(masks, prompt_lens, completion_lens, 4, 5, layers, K, E)
    out = routing.pad_rows(range(5), torch.device("cpu"))
    assert out.shape == (5, 9, layers, K)
    assert torch.equal(out[0, 1:4], masks[0][0][:3])
    assert torch.equal(out[0, 4:6], masks[0][0][3:5])
    assert (out[0, 0] == -1).all() and (out[0, 6:] == -1).all()
    assert torch.equal(out[1, 2:4], masks[1][0][:2])
    assert torch.equal(out[1, 4:6], masks[1][0][2:4])
    assert (out[1, 6] == -1).all()
    assert (out[2, :4] == -1).all()
    assert torch.equal(out[2, 4:8], masks[2][0])
    assert (out[3] == -1).all()
    # Row 4: the trainer-side prompt span stays unfilled, since a shifted fill would be silently wrong.
    assert torch.equal(out[4, 4:6], masks[4][0][2:4])
    assert (out[4, :4] == -1).all()
    assert stats == {
        "full": 2,
        "engine_omits_last": 1,
        "completion_only": 1,
        "prompt_len_mismatch": 1,
        "unresolved": 0,
    }


def test_build_rollout_routing_rejects_drift_and_counts_unresolved():
    bad_layers = [(torch.zeros(3, 5, K, dtype=torch.int16), None)]  # 5 layers vs model's 2
    with pytest.raises(ValueError, match="MoE shape"):
        build_rollout_routing(bad_layers, [1], [2], 2, 3, 2, K, E)
    bad_ids = [(torch.full((3, 2, K), E + 3, dtype=torch.int16), None)]
    with pytest.raises(ValueError, match="expert layout"):
        build_rollout_routing(bad_ids, [1], [2], 2, 3, 2, K, E)
    # A coverage matching no convention is counted, never raised, since a per-rank raise would desync ranks into a
    # collective hang. The row keeps natural routing.
    odd_len = [(torch.arange(9 * 2 * K, dtype=torch.int16).reshape(9, 2, K) % E, None)]
    routing, stats = build_rollout_routing(odd_len, [1], [2], 2, 3, 2, K, E)
    assert stats["unresolved"] == 1 and (routing.pad_rows([0], torch.device("cpu")) == -1).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
