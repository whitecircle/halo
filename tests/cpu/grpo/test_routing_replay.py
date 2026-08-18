"""CPU tests for MoE routing replay (R2): forced-selection math on the EP layer bases, the per-family
gate-weight re-derivation formulas (checked against the real HF routers, so formula drift fails),
the GC double-count fix in _record_expert_load, and the trainer-side injector's assemble/arm cycle.

    python tests/cpu/grpo/test_routing_replay.py
"""

import sys

import pytest
import torch
from torch import nn
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTopKRouter
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeTopKRouter

from src.distributed.expert_parallel.base_layer import EPGroupLimitedMoELayerBase
from src.distributed.expert_parallel.gc_scope import EPCheckpointScope
from src.trainers.grpo.rollout.routing_replay import RoutingReplayInjector

T, E, K = 6, 8, 2  # tokens, experts, top_k


class _BareLayer(EPGroupLimitedMoELayerBase):
    """Minimal concrete EP layer for exercising the selection helpers without DeepEP/GPU: bypasses
    ``__init__`` (no dispatcher) and sets exactly the attributes the helpers read.

    Rooted at the class that OWNS the group-limited routing, so the helpers exercised here are the
    ones the families inherit rather than a copy the base happens to also carry."""

    def __init__(self, **attrs):  # noqa: D107 — test double
        nn.Module.__init__(self)
        self.top_k = K
        self.num_experts = E
        self.n_routed_experts = E
        self.n_group = 1
        self.topk_group = 1
        self.norm_topk_prob = True
        self.routed_scaling_factor = 1.0
        self._forced_topk_indices = None
        self._forced_cursor = 0
        self._forced_consumed_total = 0
        self._capture_routing = False
        self._captured_routing_chunks = []
        self._replay_flip_counts = None
        for k, v in attrs.items():
            setattr(self, k, v)

    def forward(self, hidden_states, **kwargs):  # pragma: no cover — never called
        raise NotImplementedError


def _shuffled_valid_indices(natural: torch.Tensor) -> torch.Tensor:
    """A forced mask guaranteed different from ``natural`` per row (shift every index by 1 mod E)."""
    return (natural + 1) % E


# _maybe_replace_selection


def test_unforced_path_returns_input_unchanged():
    layer = _BareLayer()
    idx = torch.randint(0, E, (T, K))
    out = layer._maybe_replace_selection(idx)
    assert out is idx  # identical object: the default path must be bit-identical AND allocation-free


def test_forced_indices_replace_natural_and_count_flips():
    layer = _BareLayer()
    natural = torch.stack([torch.randperm(E)[:K] for _ in range(T)])
    forced = _shuffled_valid_indices(natural)
    layer._forced_topk_indices = forced
    out = layer._maybe_replace_selection(natural)
    assert torch.equal(out, forced)
    flipped, total = layer._replay_flip_counts.tolist()
    assert (flipped, total) == (T, T)


def test_sentinel_rows_keep_natural_selection():
    layer = _BareLayer()
    natural = torch.stack([torch.randperm(E)[:K] for _ in range(T)])
    forced = _shuffled_valid_indices(natural)
    forced[0] = -1  # sentinel: no mask for this token
    layer._forced_topk_indices = forced
    out = layer._maybe_replace_selection(natural)
    assert torch.equal(out[0], natural[0])
    assert torch.equal(out[1:], forced[1:])
    flipped, total = layer._replay_flip_counts.tolist()
    assert total == T - 1  # sentinel token not counted as forced


def test_matching_forced_mask_counts_zero_flips():
    layer = _BareLayer()
    natural = torch.stack([torch.randperm(E)[:K] for _ in range(T)])
    layer._forced_topk_indices = natural.flip(dims=[1])  # same SET per token, different order
    out = layer._maybe_replace_selection(natural)
    assert torch.equal(out.sort(dim=-1).values, natural.sort(dim=-1).values)
    flipped, total = layer._replay_flip_counts.tolist()
    assert (flipped, total) == (0, T)


def test_mask_exhaustion_and_slice_shape_raise():
    layer = _BareLayer()
    layer._forced_topk_indices = torch.zeros(T - 1, K, dtype=torch.long)  # under-covers the forward
    with pytest.raises(RuntimeError, match="exhausted"):
        layer._maybe_replace_selection(torch.randint(0, E, (T, K)))
    layer = _BareLayer()
    layer._forced_topk_indices = torch.zeros(T, K + 1, dtype=torch.long)  # wrong top_k
    with pytest.raises(RuntimeError, match="does not match"):
        layer._maybe_replace_selection(torch.randint(0, E, (T, K)))


def test_cursor_consumes_chunked_forwards_in_order():
    # The loss forward is row-chunked, so the armed mask spans SEVERAL sequential forwards: each call
    # must consume the next token slice, and exact exhaustion wraps for the mu>1 reuse pass.
    layer = _BareLayer()
    full = torch.stack([torch.randperm(E)[:K] for _ in range(2 * T)])
    layer._forced_topk_indices = full
    natural = torch.zeros(T, K, dtype=torch.long)
    first = layer._maybe_replace_selection(natural)
    second = layer._maybe_replace_selection(natural)
    assert torch.equal(first, full[:T]) and torch.equal(second, full[T:])
    assert layer._forced_cursor == 0  # wrapped on exact exhaustion
    assert torch.equal(layer._maybe_replace_selection(natural), full[:T])  # reuse pass re-consumes


def test_concurrent_checkpoint_frames_replay_their_own_slice():
    # A pipeline stage keeps several microbatches in flight, so two checkpoint frames are open at
    # once. Each frame's recompute must replay the slice ITS forward consumed — a layer-level cache
    # would hand frame 1's recompute the slice frame 2's forward wrote.
    layer = _BareLayer()
    layer.train()
    full = torch.stack([torch.randperm(E)[:K] for _ in range(2 * T)])
    layer._forced_topk_indices = full
    natural = torch.zeros(T, K, dtype=torch.long)
    frame1, frame2 = EPCheckpointScope(), EPCheckpointScope()
    with frame1:  # microbatch 1 forward
        layer._maybe_replace_selection(natural)
    with frame2:  # microbatch 2 forward
        layer._maybe_replace_selection(natural)
    with frame1:  # microbatch 1 recompute — still its own slice
        assert torch.equal(layer._maybe_replace_selection(natural), full[:T])
    with frame2:
        assert torch.equal(layer._maybe_replace_selection(natural), full[T:])


def test_recompute_call_the_forward_never_made_raises():
    # A recompute that reaches the router when the original forward did not (mask armed mid-backward)
    # must fail loud rather than silently route on stale indices.
    layer = _BareLayer()
    layer.train()
    scope = EPCheckpointScope()
    with scope:
        pass
    layer._forced_topk_indices = torch.zeros(T, K, dtype=torch.long)
    with scope, pytest.raises(RuntimeError, match="different path"):
        layer._maybe_replace_selection(torch.zeros(T, K, dtype=torch.long))


def test_gc_recompute_skips_flip_metric_but_still_replaces():
    layer = _BareLayer()
    layer.train()
    natural = torch.stack([torch.randperm(E)[:K] for _ in range(T)])
    forced = _shuffled_valid_indices(natural)
    layer._forced_topk_indices = forced
    scope = EPCheckpointScope()
    with scope:
        layer._maybe_replace_selection(natural)  # original in-checkpoint forward (counts + saves)
    counts_after_original = layer._replay_flip_counts.clone()
    with scope:  # the recompute: replaces, but must not double-count
        out = layer._maybe_replace_selection(natural)
    assert torch.equal(out, forced)
    assert torch.equal(layer._replay_flip_counts, counts_after_original)


# Selection-helper equivalence: forced == natural top-k => bit-identical outputs


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


def test_deepseek_biased_route_forced_equals_natural():
    layer = _BareLayer()
    layer.balancing_biases = torch.zeros(E)
    layer.expert_load_counter = None
    logits = torch.randn(T, E)
    base_scores, base_idx = layer._deepseek_biased_route(logits)
    layer._forced_topk_indices = base_idx.clone()
    forced_scores, forced_idx = layer._deepseek_biased_route(logits)
    assert torch.equal(forced_idx, base_idx)
    assert torch.equal(forced_scores, base_scores)


def test_forced_selection_keeps_router_grad_path():
    # The gate weights must stay differentiable wrt the router logits at the FORCED indices —
    # replaying weights (the RSPO anti-pattern) would show zero grad here.
    layer = _BareLayer()
    layer.balancing_biases = torch.zeros(E)
    layer.expert_load_counter = None
    logits = torch.randn(T, E, requires_grad=True)
    _, natural_idx = layer._deepseek_biased_route(logits)
    layer._forced_topk_indices = _shuffled_valid_indices(natural_idx)
    scores, idx = layer._deepseek_biased_route(logits)
    # Weighted loss — a plain scores.sum() is constant (softmax rows sum to 1) and would zero the
    # grad by algebra, not by a broken graph.
    (scores * torch.randn_like(scores)).sum().backward()
    grad_at_forced = logits.grad.gather(1, idx)
    assert torch.isfinite(logits.grad).all()
    assert (grad_at_forced.abs().sum(dim=-1) > 0).all(), "router gradient died at the forced selection"


# Per-family _gate_weights_at vs the REAL HF router (formula-drift guard)


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
    from src.distributed.expert_parallel.layers.qwen3 import EPQwen3MoELayer

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
    from src.distributed.expert_parallel.layers.qwen3_5 import EPQwen3_5MoELayer

    router = Qwen3_5MoeTopKRouter(_router_config(None))
    torch.nn.init.normal_(router.weight)
    hidden = torch.randn(T, 16)
    logits, scores, indices = router(hidden)

    layer = EPQwen3_5MoELayer.__new__(EPQwen3_5MoELayer)
    nn.Module.__init__(layer)
    layer.gate = router
    replayed = layer._gate_weights_at(logits, indices)
    assert torch.allclose(replayed, scores, atol=0, rtol=0), "Qwen3.5 weighting formula drifted from the HF router"


def test_bailing_gate_weights_at_formula():
    # BailingMoeV2Gate is hub remote code (not importable here); check against its documented formula:
    # sigmoid -> gather -> renorm(+1e-20) -> routed_scaling_factor (gate expert_bias is selection-only).
    from src.distributed.expert_parallel.layers.bailing import EPBailingMoELayer

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


# _record_expert_load: GC recompute must not double-count


def test_record_expert_load_skips_gc_recompute():
    layer = _BareLayer()
    layer.train()
    layer.balancing_biases = torch.zeros(E)
    layer.expert_load_counter = None
    idx = torch.randint(0, E, (T, K))
    scope = EPCheckpointScope()
    with scope:  # original in-checkpoint forward
        layer._record_expert_load(idx)
    counts_after_original = layer.expert_load_counter.clone()
    with scope:  # backward recompute re-runs the router forward on the SAME tokens
        layer._record_expert_load(idx)
    assert torch.equal(layer.expert_load_counter, counts_after_original), "GC recompute double-counted"
    # A plain (non-checkpointed) training forward still counts.
    layer._record_expert_load(idx)
    assert torch.equal(layer.expert_load_counter, counts_after_original * 2)


# RoutingReplayInjector: capture assembly, arm/disarm, flip drain, rejection


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
    injector = RoutingReplayInjector(layers)
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
    # Consume every armed row before disarm — disarm asserts full consumption.
    for layer in layers:
        layer._maybe_replace_selection(torch.zeros(rows * seq, K, dtype=torch.long))
    injector.disarm()
    assert all(layer._forced_topk_indices is None for layer in layers)


def test_injector_span_capture_scatters_and_arm_trims():
    """FA4 dense path: per-row span-trimmed forwards round-trip through capture -> arm.

    Capture receives Σ(hi-lo) tokens per layer and scatters each row segment to its absolute span
    (padding stays -1 = natural); arm trims the full-width mask back to the same concatenated span
    stream, so a capture->arm round trip feeds the cursor exactly what the per-row forwards consume.
    """
    layers = [_BareLayer(), _BareLayer()]
    injector = RoutingReplayInjector(layers)
    rows, seq = 3, 6
    spans = [(1, 5), (0, 6), (2, 3)]  # left-pad, full row, 1-token span pre-widened by the helper rule
    widths = [hi - lo for lo, hi in spans]
    per_layer_trimmed = [torch.randint(0, E, (sum(widths), K), dtype=torch.int16) for _ in layers]
    with injector.capture(rows, seq, row_spans=spans) as captured:
        for layer, trimmed in zip(layers, per_layer_trimmed, strict=True):
            # one forward per row, in row order — chunk boundaries at span widths
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
        # positions outside the spans are -1 (natural routing, never replayed)
        outside = masks[:, :, i, :].clone()
        for r, (lo, hi) in enumerate(spans):
            outside[r, lo:hi] = -1
        assert (outside == -1).all()

    injector.arm(masks, row_spans=spans)
    for i, layer in enumerate(layers):
        assert torch.equal(layer._forced_topk_indices, per_layer_trimmed[i].long())
        # per-row forwards of the span widths consume the mask exactly
        for w in widths:
            layer._maybe_replace_selection(torch.zeros(w, K, dtype=torch.long))
    injector.disarm()  # exact tiling: no partial-consumption raise


def test_injector_span_capture_count_mismatch_raises():
    layers = [_BareLayer()]
    injector = RoutingReplayInjector(layers)
    with pytest.raises(RuntimeError, match="row spans"):
        with injector.capture(2, 4, row_spans=[(0, 4), (1, 3)]):  # expects 6 tokens
            layers[0]._captured_routing_chunks.append(torch.zeros(8, K, dtype=torch.int16))  # full-width 8


def test_injector_arm_span_row_count_mismatch_raises():
    injector = RoutingReplayInjector([_BareLayer()])
    with pytest.raises(RuntimeError, match="row_spans"):
        injector.arm(torch.zeros(2, 4, 1, K, dtype=torch.int16), row_spans=[(0, 4)])


def test_injector_capture_token_count_mismatch_raises():
    layers = [_BareLayer()]
    injector = RoutingReplayInjector(layers)
    with pytest.raises(RuntimeError, match="recorded"):
        with injector.capture(4, 3):
            layers[0]._captured_routing_chunks.append(torch.zeros(5, K, dtype=torch.int16))


def test_injector_flip_rate_drains_counters():
    layers = [_BareLayer(), _BareLayer()]
    injector = RoutingReplayInjector(layers)
    layers[0]._replay_flip_counts = torch.tensor([2.0, 10.0])
    layers[1]._replay_flip_counts = torch.tensor([3.0, 10.0])
    assert injector.flip_rate() == pytest.approx(0.25)
    assert injector.flip_rate() is None  # drained


def test_injector_rejects_unsupported_family_and_empty():
    with pytest.raises(ValueError, match="at least one EP MoE layer"):
        RoutingReplayInjector([])
    unsupported = _BareLayer()
    unsupported._supports_routing_replay = False
    with pytest.raises(NotImplementedError, match="_BareLayer"):
        RoutingReplayInjector([unsupported])


def test_injector_arm_shape_validation():
    injector = RoutingReplayInjector([_BareLayer()])
    with pytest.raises(RuntimeError, match="does not match"):
        injector.arm(torch.zeros(2, 3, 5, K, dtype=torch.int16))  # 5 layers vs 1


def test_injector_disarm_flags_partial_consumption():
    layer = _BareLayer()
    injector = RoutingReplayInjector([layer])
    injector.arm(torch.zeros(2, T, 1, K, dtype=torch.int16))  # 2*T tokens armed
    layer._maybe_replace_selection(torch.zeros(T, K, dtype=torch.long))  # consume only half
    with pytest.raises(RuntimeError, match="mis-consumed at disarm"):
        injector.disarm()
    assert layer._forced_topk_indices is None  # state still cleared despite the raise


def test_injector_capture_body_exception_propagates_and_resets():
    layers = [_BareLayer(), _BareLayer()]
    injector = RoutingReplayInjector(layers)
    with pytest.raises(ValueError, match="boom"):  # NOT the capture-alignment RuntimeError
        with injector.capture(4, 3):
            layers[0]._captured_routing_chunks.append(torch.zeros(3, K, dtype=torch.int16))
            raise ValueError("boom")
    assert not any(layer._capture_routing for layer in layers)
    assert not any(layer._captured_routing_chunks for layer in layers)


# R3 transport: routed_experts decode + padded-batch assembly


def _npy_b64(array) -> str:
    import base64
    import io

    import numpy as np

    buf = io.BytesIO()
    np.save(buf, np.asarray(array))
    return base64.b64encode(buf.getvalue()).decode()


def test_decode_rollout_routing_roundtrip():
    import numpy as np

    from src.trainers.grpo.rollout.routing_replay import decode_rollout_routing

    src = np.random.randint(0, E, size=(5, 3, K), dtype=np.uint16)
    out = decode_rollout_routing(_npy_b64(src), num_layers=3, top_k=K)
    assert out.dtype == torch.int16 and out.shape == (5, 3, K)
    assert torch.equal(out, torch.from_numpy(src.astype("int16")))
    with pytest.raises(ValueError, match="shape"):
        decode_rollout_routing(_npy_b64(np.zeros((5, 3), dtype=np.uint8)), num_layers=3, top_k=K)


def test_decode_rollout_routing_raw_int32():
    """SGLang's wire format: base64 raw int32 rows, shape implied — the decoder reshapes by the
    trainer's (L, K) and must reject a byte count they do not divide."""
    import base64

    import numpy as np

    from src.trainers.grpo.rollout.routing_replay import decode_rollout_routing

    src = np.random.randint(0, E, size=(5, 3, K), dtype=np.int32)
    payload = base64.b64encode(src.tobytes()).decode()
    out = decode_rollout_routing(payload, num_layers=3, top_k=K)
    assert out.dtype == torch.int16 and out.shape == (5, 3, K)
    assert torch.equal(out, torch.from_numpy(src.astype("int16")))
    with pytest.raises(ValueError, match="multiple"):
        decode_rollout_routing(payload, num_layers=4, top_k=K)
    with pytest.raises(ValueError, match="multiple"):
        decode_rollout_routing(base64.b64encode(b"").decode(), num_layers=3, top_k=K)


def test_assemble_rollout_masks_conventions():
    from src.trainers.grpo.rollout.routing_replay import assemble_rollout_masks

    layers = 2

    def _m(tokens):
        return torch.arange(tokens * layers * K, dtype=torch.int16).reshape(tokens, layers, K) % E

    # Row 0: full-request mask (P+C). Row 1: engine-omits-last (P+C-1). Row 2: completion-only (C).
    # Row 3: no mask. Row 4: engine prompt count DIFFERS from the trainer's re-render (drifted by 1)
    # — the completion span must still be replayed, anchored on the ENGINE count; prompt stays natural.
    prompt_lens, completion_lens = [3, 2, 2, 1, 3], [2, 3, 4, 1, 2]
    masks = [
        (_m(5), 3),
        (_m(4), 2),
        (_m(4), None),
        None,
        (_m(4), 2),  # engine says prompt=2, trainer rendered 3 → 2+2=4 tokens, full coverage @ ep=2
    ]
    out, stats = assemble_rollout_masks(masks, prompt_lens, completion_lens, 4, 5, layers, K, E)
    assert out.shape == (5, 9, layers, K)
    # Row 0: prompt at [4-3, 4), completion at [4, 6) — fully covered.
    assert torch.equal(out[0, 1:4], masks[0][0][:3])
    assert torch.equal(out[0, 4:6], masks[0][0][3:5])
    assert (out[0, 0] == -1).all() and (out[0, 6:] == -1).all()
    # Row 1: P+C-1 = engine omitted the final position — last completion token stays natural (-1).
    assert torch.equal(out[1, 2:4], masks[1][0][:2])
    assert torch.equal(out[1, 4:6], masks[1][0][2:4])
    assert (out[1, 6] == -1).all()
    # Row 2: completion-only coverage — prompt span stays natural.
    assert (out[2, :4] == -1).all()
    assert torch.equal(out[2, 4:8], masks[2][0])
    # Row 3: untouched.
    assert (out[3] == -1).all()
    # Row 4: completion anchored on the ENGINE prompt count (positions 2:4 of the mask); the
    # trainer-side prompt span is NOT filled (count mismatch — a shifted fill would be silent poison).
    assert torch.equal(out[4, 4:6], masks[4][0][2:4])
    assert (out[4, :4] == -1).all()
    assert stats == {
        "full": 2,
        "engine_omits_last": 1,
        "completion_only": 1,
        "prompt_len_mismatch": 1,
        "unresolved": 0,
    }


def test_assemble_rollout_masks_rejects_drift_and_counts_unresolved():
    from src.trainers.grpo.rollout.routing_replay import assemble_rollout_masks

    bad_layers = [(torch.zeros(3, 5, K, dtype=torch.int16), None)]  # 5 layers vs model's 2
    with pytest.raises(ValueError, match="MoE shape"):
        assemble_rollout_masks(bad_layers, [1], [2], 2, 3, 2, K, E)
    bad_ids = [(torch.full((3, 2, K), E + 3, dtype=torch.int16), None)]  # out-of-range expert ids
    with pytest.raises(ValueError, match="expert layout"):
        assemble_rollout_masks(bad_ids, [1], [2], 2, 3, 2, K, E)
    # A coverage that matches no convention is COUNTED, never raised (a per-rank raise would desync
    # ranks into a collective hang) — the row keeps natural routing.
    odd_len = [(torch.zeros(9, 2, K, dtype=torch.int16), None)]
    out, stats = assemble_rollout_masks(odd_len, [1], [2], 2, 3, 2, K, E)
    assert stats["unresolved"] == 1 and (out == -1).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
