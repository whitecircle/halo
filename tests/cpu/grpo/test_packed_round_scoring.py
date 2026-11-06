#!/usr/bin/env python
"""A packed round through the environmental trainer: the batch build plans and lays it out without changing what it
trains, the no-grad passes run one scoring call per bin on every rank, routing capture and replay follow the bins, and a
window's leading no-op slots run nothing."""

import ast
import inspect
import textwrap
import types

import pytest
import torch
import trl.trainer.grpo_trainer as trl_grpo_trainer
from trl import GRPOTrainer

from halo.configs.grpo_environmental import AsyncRolloutConfig
from halo.trainers.grpo.environmental.packed_rounds import (
    SLOT_ACTIVE_KEY,
    SLOT_IS_ACTIVE_KEY,
    SLOT_ROWS_KEY,
    PackedMicrobatchMixin,
)
from halo.trainers.grpo.environmental.routing_replay import RoutingReplayInjector
from halo.trainers.grpo.environmental.row_order import LengthOrderedRowsMixin
from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer
from halo.trainers.grpo.microbatch_plan import MicrobatchPlan
from halo.trainers.grpo.on_policy import GenerationRoundMixin
from tests.common.env_grpo_batch import batch_host, build, episode, row
from tests.common.length_deal import ragged_round, scoring_host, single_process_accelerator, stacked, tiny_llama
from tests.common.routing import BareEPLayer

_VOCAB = 64
_LENGTHS = (5, 1, 7, 2, 6, 3, 4, 8)
_REWARDS = (1.0, 0.0, 0.5, 0.25, 0.0, 1.0, 0.75, 0.5)
_SLOTS = 4


def _rowwise_scores(_model, ids, mask, keep, compute_entropy=False):
    """Log-probs that depend on the row alone, so any grouping of rows scores each the same."""
    weights = torch.arange(1, ids.size(1) + 1, dtype=torch.float32)
    base = (ids.float() * mask * weights).sum(1, keepdim=True) / 1000.0
    return -(base + torch.arange(keep, dtype=torch.float32) * 0.01), None, None


def _built(packed: bool, budget: int = 16, **host_kwargs):
    rows = [row(list(range(3, 3 + n)), sampling=[-0.1 * (i + 1) for i in range(n)]) for n in _LENGTHS]
    host = batch_host(rows, torch.zeros(1), training=True, is_correction=True, **host_kwargs)
    host._get_per_token_logps_and_entropies = _rowwise_scores
    host.args.steps_per_generation = _SLOTS
    host.async_config = AsyncRolloutConfig(train_on_sampled_tokens=False, packed_microbatch_tokens=budget)
    host._packs_microbatches = packed
    host._packed_dispatch_ceiling = None
    host._broadcast_tensors_from_tp_leader = lambda tensors: tensors
    host._get_tp_or_etp_process_group = lambda: None
    result = build(host, [episode(r) for r in _REWARDS])
    return host, result


def test_the_packed_build_keeps_every_round_tensor_and_plans_every_row_once():
    plain_host, plain = _built(packed=False)
    packed_host, slots = _built(packed=True)
    store = packed_host._round_store
    assert set(store) == set(plain) - {"num_items_in_batch"}
    for key, value in store.items():
        if value is None:
            assert plain[key] is None, key
        else:
            assert torch.equal(value, plain[key]), f"{key} changed under packing"
    assert torch.equal(slots["num_items_in_batch"], plain["num_items_in_batch"])
    rows, active = slots[SLOT_ROWS_KEY], slots[SLOT_ACTIVE_KEY]
    assert rows.shape == active.shape and rows.numel() % _SLOTS == 0
    assert sorted(rows[rows >= 0].tolist()) == list(range(len(_LENGTHS)))
    per_slot = rows.numel() // _SLOTS
    assert active.view(_SLOTS, per_slot)[-1].all(), "the window's last slot must be real"
    assert 1 < (active.view(_SLOTS, per_slot)[:, 0] > 0).sum() < _SLOTS, "the budget should leave no-ops and bins"


def test_the_packed_build_scores_one_call_per_bin_not_one_per_row():
    calls = []

    def recording(model, ids, mask, keep, compute_entropy=False):
        calls.append(ids.size(0))
        return _rowwise_scores(model, ids, mask, keep)

    rows = [row(list(range(3, 3 + n)), sampling=[-0.1] * n) for n in _LENGTHS]
    host = batch_host(rows, torch.zeros(1), training=True, is_correction=True)
    host.args.steps_per_generation = _SLOTS
    host.async_config = AsyncRolloutConfig(train_on_sampled_tokens=False, packed_microbatch_tokens=16)
    host._get_per_token_logps_and_entropies = recording
    host._packs_microbatches, host._packed_dispatch_ceiling = True, None
    host._broadcast_tensors_from_tp_leader = lambda tensors: tensors
    host._get_tp_or_etp_process_group = lambda: None
    build(host, [episode(r) for r in _REWARDS])
    assert 1 < len(calls) <= _SLOTS and sum(calls) == len(_LENGTHS)


@pytest.fixture
def llama_scoring_host():
    host = scoring_host(tiny_llama(_VOCAB, 0), None, chunks=_SLOTS, accelerator=single_process_accelerator())
    host.pad_token_id = 0
    return host


def _recording(host):
    calls = []
    scorer = DistributedAsyncEnvironmentalGRPOTrainer._get_per_token_logps_and_entropies

    def recording(*args, **kwargs):
        calls.append(args[1].size(0))
        return scorer(host, *args, **kwargs)

    host._get_per_token_logps_and_entropies = recording
    return calls


def test_a_plan_scores_each_bin_in_one_call_and_an_empty_bin_in_an_inert_one(llama_scoring_host):
    host = llama_scoring_host
    ids, mask, keep = stacked(
        ragged_round([3, 4, 2, 5, 1, 3, 2, 4, 3, 2, 1, 2], [4, 2, 5, 3, 6, 2, 3, 1, 4, 5, 2, 2], vocab=_VOCAB, seed=4)
    )
    plan = MicrobatchPlan(bins=((7, 2, 9), (0, 4), (), (5, 1, 3)), rows_per_bin=3, slots=_SLOTS, round_rows=12)
    order = torch.tensor(plan.order)
    with torch.no_grad():
        plain = host._scored_rows(host.model, ids, mask, keep, None)
        calls = _recording(host)
        packed = host._scored_rows(host.model, ids[order], mask[order], keep, order, plan)
    assert calls == [3, 2, 1, 3], "every rank must run one scoring call per bin, an empty bin an inert row"
    assert torch.equal(packed[order], plain[order]), "a row's score changed with the rows beside it"
    held = set(plan.order)
    assert all(not packed[i].any() for i in range(12) if i not in held), "rows no bin holds score 0"
    assert any(plain[i].any() for i in range(12) if i not in held), "the unscored rows must otherwise score"


def test_a_plan_of_empty_bins_alone_scores_zero_through_inert_rows(llama_scoring_host):
    host = llama_scoring_host
    ids, mask, keep = stacked(ragged_round([2, 3], [2, 2], vocab=_VOCAB, seed=1))
    plan = MicrobatchPlan(bins=((),), rows_per_bin=1, slots=_SLOTS, round_rows=2)
    calls = _recording(host)
    with torch.no_grad():
        out = host._scored_rows(host.model, ids[:0], mask[:0], keep, torch.tensor([], dtype=torch.long), plan)
    assert calls == [1] and out.shape == (2, keep) and not out.any()


_SEQ, _LAYERS, _TOP_K = 6, 2, 2
_SPANS = [(0, 6), (2, 6), (1, 4), (3, 6), (0, 3), (1, 6)]


def _injector() -> RoutingReplayInjector:
    return RoutingReplayInjector(
        [BareEPLayer(top_k=_TOP_K, num_experts=8) for _ in range(_LAYERS)],
        engine_layers=_LAYERS,
        layer_indices=list(range(_LAYERS)),
    )


def _mask(rows: int) -> torch.Tensor:
    r, s, layer, k = torch.meshgrid(
        torch.arange(rows), torch.arange(_SEQ), torch.arange(_LAYERS), torch.arange(_TOP_K), indexing="ij"
    )
    return (r * 1000 + s * 10 + layer * 3 + k).to(torch.int16)


def _forward(injector, mask, rows):
    """What each layer's dispatch does with a forward's tokens: record them when capture is on."""
    for i, layer in enumerate(injector._layers):
        if not layer._capture_routing:
            continue
        for r in rows:
            lo, hi = _SPANS[r]
            layer._captured_routing_chunks.append(mask[r, lo:hi, i])


def test_a_capture_over_a_subset_of_rows_leaves_the_others_natural():
    injector, mask = _injector(), _mask(len(_SPANS))
    forwarded = torch.tensor([4, 1, 3])
    with injector.capture(
        len(_SPANS), _SEQ, row_spans=[_SPANS[r] for r in forwarded.tolist()], row_order=forwarded
    ) as result:
        _forward(injector, mask, forwarded.tolist())
    captured = result[0]
    assert captured.shape[0] == len(_SPANS)
    for r in range(len(_SPANS)):
        lo, hi = _SPANS[r]
        if r in forwarded.tolist():
            assert torch.equal(captured[r, lo:hi], mask[r, lo:hi]), f"row {r} captured elsewhere"
        else:
            assert (captured[r] == -1).all(), f"row {r} was never forwarded"


def test_an_inert_forward_inside_a_capture_miscounts_unless_it_is_paused():
    """The inert forward of an empty bin carries tokens no declared span holds, so it must not be captured."""
    mask, forwarded = _mask(len(_SPANS)), torch.tensor([4, 1])
    spans = [_SPANS[r] for r in forwarded.tolist()]
    injector = _injector()
    with (
        pytest.raises(RuntimeError, match="wrong token count"),
        injector.capture(len(_SPANS), _SEQ, row_spans=spans, row_order=forwarded),
    ):
        _forward(injector, mask, [4])
        _forward(injector, mask, [3])
        _forward(injector, mask, [1])
    with injector.capture(len(_SPANS), _SEQ, row_spans=spans, row_order=forwarded) as result:
        _forward(injector, mask, [4])
        with injector.paused():
            _forward(injector, mask, [3])
        _forward(injector, mask, [1])
    captured = result[0]
    assert (captured[3] == -1).all() and torch.equal(captured[4, :3], mask[4, :3])


def test_pausing_hides_an_armed_mask_from_the_forward_and_restores_it_with_its_cursor():
    injector, mask = _injector(), _mask(len(_SPANS))
    forwarded = torch.tensor([2, 0])
    injector.arm(mask, row_spans=[_SPANS[r] for r in forwarded.tolist()], row_order=forwarded, forwards_subset=True)
    armed = [layer._forced_topk_indices for layer in injector._layers]
    for layer in injector._layers:
        layer._forced_cursor = 2
    with injector.paused():
        assert all(layer._forced_topk_indices is None for layer in injector._layers)
    for layer, before in zip(injector._layers, armed, strict=True):
        assert layer._forced_topk_indices is before and layer._forced_cursor == 2
    expected = sum(hi - lo for lo, hi in (_SPANS[2], _SPANS[0]))
    assert all(layer._forced_topk_indices.size(0) == expected for layer in injector._layers)


def test_arming_checks_the_forwarded_rows_not_the_masks_rows():
    injector = _injector()
    with pytest.raises(RuntimeError, match="row_spans has 2 entries for 3 forwarded rows"):
        injector.arm(_mask(6), row_spans=[_SPANS[0], _SPANS[1]], row_order=torch.tensor([0, 1, 2]))


def test_a_row_order_shorter_than_the_mask_is_refused_unless_the_caller_declares_a_subset():
    injector, mask = _injector(), _mask(len(_SPANS))
    forwarded = torch.tensor([2, 0])
    spans = [_SPANS[r] for r in forwarded.tolist()]
    with pytest.raises(RuntimeError, match="row_order has 2 entries"):
        injector.arm(mask, row_spans=spans, row_order=forwarded)
    injector.arm(mask, row_spans=spans, row_order=forwarded, forwards_subset=True)


def _noop_host(*, injector=None):
    host = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    host._routing_injector = injector
    host.accelerator = types.SimpleNamespace(device=torch.device("cpu"))
    return host


def test_a_no_op_slot_runs_no_forward_arms_no_replay_and_still_backpropagates():
    class Untouchable:
        def __getattr__(self, name):
            raise AssertionError(f"a no-op slot touched the routing injector ({name})")

    host = _noop_host(injector=Untouchable())
    model = types.SimpleNamespace(training=True)
    loss = DistributedAsyncEnvironmentalGRPOTrainer.compute_loss(host, model, {SLOT_IS_ACTIVE_KEY: False})
    assert loss.requires_grad and loss.item() == 0.0
    loss.backward()
    assert loss.grad is not None


def test_a_no_op_flag_is_ignored_outside_training():
    host = _noop_host()
    assert host._is_noop_slot(types.SimpleNamespace(training=True), {SLOT_IS_ACTIVE_KEY: False})
    assert not host._is_noop_slot(types.SimpleNamespace(training=False), {SLOT_IS_ACTIVE_KEY: False})
    assert not host._is_noop_slot(types.SimpleNamespace(training=True), {"advantages": None})


def test_the_construction_warm_up_leaves_no_expert_load_for_the_first_bias_update(monkeypatch):
    class Router(torch.nn.Module):
        balancing_biases = torch.zeros(2)
        expert_load_counter = None

    router = Router()
    model = torch.nn.Sequential(router)
    host = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    host.model = model
    host.accelerator = types.SimpleNamespace(device=torch.device("cpu"))
    leaf = torch.zeros(1, requires_grad=True)

    def score(_model, ids, *_args, **_kwargs):
        router.expert_load_counter = torch.ones(2)
        return leaf * ids.sum(), None, None

    host._get_per_token_logps_and_entropies = score
    host._finish_fsdp2_backward = lambda _model: None
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr("halo.trainers.grpo.environmental.packed_rounds.barrier", lambda: None)
    host._warm_packed_row_kernels()
    assert router.expert_load_counter is None


def _refill_host(batch_slots: dict, *, packed=True, training=True):
    host = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    host.model = types.SimpleNamespace(training=training)
    host.args = types.SimpleNamespace(steps_per_generation=2, gradient_accumulation_steps=2)
    host.num_iterations = 1
    host._step, host._buffered_inputs = 0, None
    host._packs_microbatches = packed
    host._orders_rows_by_length = False
    host._generate_and_score_completions = lambda _inputs: dict(batch_slots)
    host._broadcast_tensors_from_tp_leader = lambda inputs: inputs
    host._build_slot_view = lambda slot: slot
    return host


def _slots() -> dict:
    return {
        SLOT_ROWS_KEY: torch.tensor([-1, -1, 0, 1]),
        SLOT_ACTIVE_KEY: torch.tensor([0, 0, 1, 1]),
        "num_items_in_batch": torch.tensor(3),
    }


def test_a_packed_refill_splits_the_slot_indices_in_order_and_restores_trls_shuffle():
    shuffle = trl_grpo_trainer.shuffle_sequence_dict
    host = _refill_host(_slots())
    first = host._prepare_inputs({})
    host._step += 1
    second = host._prepare_inputs({})
    assert trl_grpo_trainer.shuffle_sequence_dict is shuffle
    assert first[SLOT_ROWS_KEY].tolist() == [-1, -1] and first[SLOT_ACTIVE_KEY].tolist() == [0, 0]
    assert second[SLOT_ROWS_KEY].tolist() == [0, 1] and second[SLOT_ACTIVE_KEY].tolist() == [1, 1]


def test_a_trl_refill_that_no_longer_shuffles_is_refused_for_packed_rounds(monkeypatch):
    def moved(self, generation_batch):
        self._buffered_inputs = [self._generate_and_score_completions(generation_batch)]
        return self._buffered_inputs[0]

    monkeypatch.setattr(GRPOTrainer, "_prepare_inputs", moved)
    with pytest.raises(RuntimeError, match="moved its buffer shuffle"):
        _refill_host(_slots())._prepare_inputs({})


def test_evaluation_rounds_never_take_the_packed_path():
    host = _refill_host({"advantages": torch.zeros(2)}, training=False)
    host._build_slot_view = lambda slot: pytest.fail("an eval round was gathered as a slot")
    assert host._prepare_inputs({})["advantages"].shape == (2,)


def test_the_mixin_sits_ahead_of_the_deal_and_the_tp_resync_and_is_configured_at_construction():
    mro = DistributedAsyncEnvironmentalGRPOTrainer.__mro__
    assert DistributedAsyncEnvironmentalGRPOTrainer._prepare_inputs is PackedMicrobatchMixin._prepare_inputs
    assert mro.index(PackedMicrobatchMixin) < mro.index(LengthOrderedRowsMixin) < mro.index(GenerationRoundMixin)
    init = ast.parse(textwrap.dedent(inspect.getsource(DistributedAsyncEnvironmentalGRPOTrainer.__init__)))
    called = [n.func.attr for n in ast.walk(init) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert called.index("_configure_packed_microbatches") > called.index("_resolve_length_ordered_rows")
    assert DistributedAsyncEnvironmentalGRPOTrainer._packs_microbatches is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
