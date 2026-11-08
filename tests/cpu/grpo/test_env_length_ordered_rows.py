#!/usr/bin/env python
"""The environmental trainer's length-ordered deal of a round's rows (``length_ordered_microbatches``). It attains the
lock-step minimum of Σ_position max_rank cost, which TRL's shuffle does not, and changes only the row order."""

import ast
import inspect
import textwrap
import types

import pytest
import torch
import trl.trainer.grpo_trainer as trl_grpo_trainer
from trl import GRPOTrainer
from trl.trainer.utils import shuffle_sequence_dict

from halo.configs.grpo_environmental import AsyncRolloutConfig
from halo.trainers.grpo.environmental.routing_replay import RoutingReplayInjector
from halo.trainers.grpo.environmental.row_order import LengthOrderedRowsMixin
from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer
from halo.trainers.grpo.logprob_scoring import dense_row_spans
from halo.trainers.grpo.on_policy import GenerationRoundMixin
from halo.trainers.grpo.rounds.row_order import MICROBATCH_INVARIANT_LOSS_TYPES, length_deal_order, permute_rows
from tests.common.env_grpo_batch import batch_host, build, episode, row
from tests.common.length_deal import (
    loss_host,
    ragged_round,
    scoring_host,
    single_process_accelerator,
    stacked,
    tiny_llama,
    train_round,
)
from tests.common.routing import bare_replay_injector, dense_rows, ragged_routing

_VOCAB = 64
# Losses TRL's real loss below shows to be the same in any deal of a round's rows; bnpo differs and vespo is
# refused at construction.
_ORDER_FREE_LOSSES = ("grpo", "sapo", "dr_grpo", "dapo", "cispo", "luspo")


def _length_ranks(lengths: torch.Tensor) -> torch.Tensor:
    ranked = torch.sort(lengths, descending=True, stable=True).indices
    ranks = torch.empty_like(ranked)
    ranks[ranked] = torch.arange(lengths.numel())
    return ranks


@pytest.mark.parametrize(("rows", "chunks"), [(1, 1), (6, 1), (6, 6), (24, 4), (35, 7), (192, 24)])
def test_the_deal_is_a_permutation_into_micro_batches_of_one_row_per_stratum(rows, chunks):
    lengths = torch.randint(1, 5000, (rows,), generator=torch.Generator().manual_seed(rows))
    order = length_deal_order(lengths, chunks)
    assert sorted(order.tolist()) == list(range(rows))
    ranks = _length_ranks(lengths)[order].view(chunks, rows // chunks)
    want = torch.arange(rows).view(rows // chunks, chunks).t()
    assert torch.equal(ranks, want), "micro-batch k must hold the k-th, (k+chunks)-th, ... longest rows"


def test_every_rank_runs_its_ith_longest_row_at_position_i_without_any_randomness():
    gen = torch.Generator().manual_seed(0)
    per_rank = [torch.randint(1, 40_000, (48,), generator=gen) for _ in range(6)]
    positions = [_length_ranks(lengths)[length_deal_order(lengths, 8)] for lengths in per_rank]
    assert all(torch.equal(p, positions[0]) for p in positions), "ranks paired rows of different length ranks"
    torch.manual_seed(1)
    first = length_deal_order(per_rank[0], 8)
    torch.manual_seed(2)
    assert torch.equal(first, length_deal_order(per_rank[0], 8)), "the deal read an RNG"


def test_ties_keep_their_batch_order():
    order = length_deal_order(torch.tensor([3, 5, 3, 5, 3, 5]), 2)
    assert order.tolist() == [1, 5, 2, 3, 0, 4]


def test_the_deal_attains_the_lockstep_minimum_and_the_shuffle_does_not():
    """Σ_position max_rank length is minimal when every rank lines its rows up by length rank (rearrangement)."""
    gen = torch.Generator().manual_seed(3)
    per_rank = [torch.randint(1, 30_000, (64,), generator=gen) for _ in range(4)]

    def lockstep(orders):
        return torch.stack([lengths[o] for lengths, o in zip(per_rank, orders, strict=True)]).amax(0).sum().item()

    dealt = lockstep([length_deal_order(lengths, 8) for lengths in per_rank])
    floor = torch.stack([torch.sort(lengths, descending=True).values for lengths in per_rank]).amax(0).sum().item()
    shuffled = [lockstep([torch.randperm(64, generator=gen) for _ in per_rank]) for _ in range(64)]
    assert dealt == floor
    assert dealt < min(shuffled), (dealt, min(shuffled))


def test_a_round_that_does_not_split_evenly_is_refused():
    with pytest.raises(ValueError, match="7 rows into 2"):
        length_deal_order(torch.ones(7), 2)


def test_rows_permute_exactly_as_trls_shuffle_permutes_them(monkeypatch):
    order = torch.tensor([2, 0, 3, 1])
    batch = {
        "prompt_ids": torch.arange(8).view(4, 2),
        "advantages": torch.tensor([0.1, 0.2, 0.3, 0.4]),
        "num_items_in_batch": torch.tensor(9.0),
        "old_per_token_logps": None,
        "images": ["a", "b", "c", "d"],
    }
    monkeypatch.setattr(torch, "randperm", lambda n: order)
    trl = shuffle_sequence_dict(batch)
    ours = permute_rows(batch, order)
    assert ours.keys() == trl.keys()
    for key, value in trl.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(ours[key], value), key
        else:
            assert ours[key] == value, key


def _refill_host(*, deal: bool, chunks: int, batch: dict, training: bool = True, accumulation: int | None = None):
    """A trainer built without ``__init__`` over TRL's real ``_prepare_inputs``."""
    host = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    host.model = types.SimpleNamespace(training=training)
    host.args = types.SimpleNamespace(steps_per_generation=chunks, gradient_accumulation_steps=accumulation or chunks)
    host.num_iterations = 1
    host._step = 0
    host._buffered_inputs = None
    host._orders_rows_by_length = deal
    host._generate_and_score_completions = lambda _inputs: dict(batch)
    host._broadcast_tensors_from_tp_leader = lambda inputs: inputs
    return host


def _round(rows: int, seed: int = 0) -> dict:
    gen = torch.Generator().manual_seed(seed)
    prompt_lens = torch.randint(1, 6, (rows,), generator=gen).tolist()
    completion_lens = torch.randint(1, 9, (rows,), generator=gen).tolist()
    return ragged_round(prompt_lens, completion_lens, vocab=_VOCAB, seed=seed)


def _micro_batches(host, chunks: int) -> list[dict]:
    batches = []
    for _ in range(chunks):
        batches.append(host._prepare_inputs({}))
        host._step += 1
    return batches


@pytest.mark.parametrize("rounds_per_step", [1, 2])
def test_a_train_refill_deals_every_key_in_step_and_restores_trls_shuffle(rounds_per_step):
    """Dealt into ``steps_per_generation`` micro-batches, also where an optimizer step spans several rounds."""
    chunks, rows = 4, 16
    batch = _round(rows)
    host = _refill_host(deal=True, chunks=chunks, batch=batch, accumulation=chunks * rounds_per_step)
    shuffle = trl_grpo_trainer.shuffle_sequence_dict
    micro = _micro_batches(host, chunks)

    assert trl_grpo_trainer.shuffle_sequence_dict is shuffle, "TRL's shuffle was left swapped out"
    lengths = torch.cat([batch["prompt_mask"], batch["completion_mask"]], 1).sum(1)
    ranks = _length_ranks(lengths)
    for k, mb in enumerate(micro):
        ids = mb["row_id"]
        assert ranks[ids].tolist() == list(range(k, rows, chunks)), f"micro-batch {k} is not stratum-dealt"
        for key in ("prompt_ids", "completion_ids", "completion_mask", "tool_mask"):
            assert torch.equal(mb[key], batch[key][ids]), f"{key} left its row"
        assert torch.equal(mb["num_items_in_batch"], batch["num_items_in_batch"])


def test_with_the_deal_off_or_in_eval_trls_own_shuffle_path_runs(monkeypatch):
    calls = []
    monkeypatch.setattr(trl_grpo_trainer, "shuffle_sequence_dict", lambda b: calls.append(1) or b)
    _micro_batches(_refill_host(deal=False, chunks=2, batch=_round(4)), 2)
    assert calls == [1], "the deal-off refill skipped TRL's shuffle"
    batch = _round(4)
    eval_inputs = _refill_host(deal=True, chunks=2, batch=batch, training=False)._prepare_inputs({})
    assert calls == [1] and torch.equal(eval_inputs["row_id"], batch["row_id"])


def test_a_trl_refill_that_no_longer_calls_its_shuffle_is_refused(monkeypatch):
    def moved_shuffle(self, generation_batch):
        self._buffered_inputs = [self._generate_and_score_completions(generation_batch)]
        return self._buffered_inputs[0]

    monkeypatch.setattr(GRPOTrainer, "_prepare_inputs", moved_shuffle)
    host = _refill_host(deal=True, chunks=1, batch=_round(2))
    with pytest.raises(RuntimeError, match="moved its buffer shuffle"):
        host._prepare_inputs({})


def test_the_trainer_deals_ahead_of_the_tp_resync_and_resolves_the_knob_at_construction():
    mro = DistributedAsyncEnvironmentalGRPOTrainer.__mro__
    owners = [c for c in mro if "_prepare_inputs" in vars(c)]
    assert owners.index(LengthOrderedRowsMixin) < owners.index(GenerationRoundMixin)
    assert mro.index(LengthOrderedRowsMixin) < mro.index(GenerationRoundMixin)
    init = ast.parse(textwrap.dedent(inspect.getsource(DistributedAsyncEnvironmentalGRPOTrainer.__init__)))
    called = {n.func.attr for n in ast.walk(init) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert "_resolve_length_ordered_rows" in called


def _gate_host(requested=None, *, attn="sdpa", batch_size=1, chunked=True, entropy_bonus=False, **args):
    host = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    host.async_config = AsyncRolloutConfig(length_ordered_microbatches=requested)
    host.args = types.SimpleNamespace(
        **{
            "loss_type": "dapo",
            "top_entropy_quantile": 1.0,
            "gradient_accumulation_steps": 24,
            "steps_per_generation": 24,
            "per_device_train_batch_size": batch_size,
        }
        | args
    )
    host.num_iterations = 1
    host._entropy_bonus_enabled = entropy_bonus
    host._use_chunked_grpo_logprobs = chunked
    host.model = types.SimpleNamespace(config=types.SimpleNamespace(_attn_implementation=attn))
    host.accelerator = types.SimpleNamespace(unwrap_model=lambda model: model)
    return host


_BLOCKED = {
    "misaligned accumulation": ({"gradient_accumulation_steps": 36}, "straddles a generation round"),
    "bnpo": ({"loss_type": "bnpo"}, "normalizes each micro-batch"),
    "entropy quantile": ({"top_entropy_quantile": 0.2}, "thresholds entropy"),
    "entropy bonus": ({"entropy_bonus": True}, "entropy bonus"),
    "padded batch": ({"batch_size": 2}, "padded width"),
    "full logits": ({"chunked": False}, "padded width"),
}


@pytest.mark.parametrize("case", _BLOCKED, ids=list(_BLOCKED))
def test_unset_turns_off_and_true_refuses_where_the_order_would_change_more(case):
    overrides, reason = _BLOCKED[case]
    host = _gate_host(**overrides)
    host._resolve_length_ordered_rows()
    assert host._orders_rows_by_length is False
    with pytest.raises(ValueError, match=reason):
        _gate_host(True, **overrides)._resolve_length_ordered_rows()


def test_the_gate_admits_exactly_the_loss_types_shown_order_free():
    assert sorted(MICROBATCH_INVARIANT_LOSS_TYPES) == sorted(_ORDER_FREE_LOSSES)


@pytest.mark.parametrize("loss_type", _ORDER_FREE_LOSSES)
@pytest.mark.parametrize(("attn", "batch_size"), [("sdpa", 1), ("flash_attention_4", 4)])
def test_unset_and_true_deal_on_every_order_only_config_and_false_never(loss_type, attn, batch_size):
    for requested in (None, True):
        host = _gate_host(requested, loss_type=loss_type, attn=attn, batch_size=batch_size)
        host._resolve_length_ordered_rows()
        assert host._orders_rows_by_length is True, requested
    host = _gate_host(False, loss_type=loss_type, attn=attn, batch_size=batch_size)
    host._resolve_length_ordered_rows()
    assert host._orders_rows_by_length is False


def test_num_iterations_over_one_stays_dealt_when_aligned():
    host = _gate_host(gradient_accumulation_steps=48)
    host.num_iterations = 2
    host._resolve_length_ordered_rows()
    assert host._orders_rows_by_length is True


_TOP_K, _SEQ, _LAYERS = 2, 6, 2
# Real spans over a padded width of _SEQ, and a deal that is not its own inverse, so taking the order where its
# inverse belongs misplaces rows.
_SPANS = [(0, 6), (2, 6), (1, 4), (3, 6)]
_DEALT = torch.tensor([2, 0, 3, 1])


def _injector() -> RoutingReplayInjector:
    return bare_replay_injector(layers=_LAYERS, top_k=_TOP_K)


def _routed_mask(rows: int = len(_SPANS), seq: int = _SEQ) -> torch.Tensor:
    """A ``[rows, seq, layers, top_k]`` mask whose every entry names its row, position and layer."""
    r, s, layer, k = torch.meshgrid(
        torch.arange(rows), torch.arange(seq), torch.arange(_LAYERS), torch.arange(_TOP_K), indexing="ij"
    )
    return (r * 1000 + s * 10 + layer * 3 + k).to(torch.int16)


def _captured(order: torch.Tensor | None) -> torch.Tensor:
    injector, mask = _injector(), _routed_mask()
    forwarded = list(range(len(_SPANS))) if order is None else order.tolist()
    with injector.capture(len(_SPANS), _SEQ, row_spans=[_SPANS[r] for r in forwarded], row_order=order) as result:
        for i, layer in enumerate(injector._layers):
            for r in forwarded:
                lo, hi = _SPANS[r]
                layer._captured_routing_chunks.append(mask[r, lo:hi, i])
    return dense_rows(result[0])


def test_the_replay_deal_is_not_its_own_inverse():
    assert not torch.equal(_DEALT, torch.argsort(_DEALT))


def test_a_capture_over_dealt_forwards_comes_back_in_batch_order():
    assert torch.equal(_captured(_DEALT), _captured(None))


def test_an_armed_mask_feeds_each_dealt_forward_its_own_rows_routing():
    injector, mask = _injector(), _routed_mask()
    forwarded = _DEALT.tolist()
    attended = torch.stack([torch.arange(_SEQ) >= _SPANS[r][0] for r in forwarded])
    injector.arm_rows(ragged_routing(mask), forwarded, attended, [_SPANS[r] for r in forwarded])
    for i, layer in enumerate(injector._layers):
        want = torch.cat([mask[r, _SPANS[r][0] : _SPANS[r][1], i] for r in forwarded])
        assert torch.equal(layer._forced_topk_indices, want.long()), f"layer {i}"


def test_a_row_order_without_per_row_spans_is_refused():
    injector = _injector()
    with pytest.raises(RuntimeError, match="row order without row spans"), injector.capture(4, _SEQ, row_order=_DEALT):
        pass


def _replaying_scorer(injector: RoutingReplayInjector, replayed: dict):
    """A per-row scorer: each forwarded row routes its real span to experts named by its tokens; an armed layer replays
    its mask over that span, kept in ``replayed`` by the row's tokens. The log-probs depend on the row alone."""

    def score(_model, ids, mask, keep, compute_entropy=False):
        for p, (lo, hi) in enumerate(dense_row_spans(mask)[0]):
            tokens = ids[p, lo:hi]
            for i, layer in enumerate(injector._layers):
                natural = (tokens[:, None] * 10 + i * 3 + torch.arange(_TOP_K)).long()
                if layer._capture_routing:
                    layer._captured_routing_chunks.append(natural.to(torch.int16))
                if layer._forced_topk_indices is not None:
                    replayed[(tuple(ids[p].tolist()), i)] = layer._apply_routing_replay(natural)
        weights = torch.arange(1, ids.size(1) + 1, dtype=torch.float32)
        return -(ids.float() * mask * weights).sum(1, keepdim=True).expand(-1, keep) / 1000.0, None, None

    return score


def _step_recompute(rollout_replay: bool, dealt: bool) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """The step batch's real no-grad recompute, routing captured (``recompute``) or replayed (``rollout``)."""
    ids, mask, keep = stacked(_round(12, seed=5))
    injector, replayed = _injector(), {}
    host = scoring_host(None, None, chunks=4, accelerator=single_process_accelerator())
    host._routing_injector, host._rollout_routing_replay = injector, rollout_replay
    host._get_per_token_logps_and_entropies = _replaying_scorer(injector, replayed)
    order = host._no_grad_row_order(mask, "train") if dealt else None
    assert order is None or not torch.equal(order, torch.argsort(order)), "the deal should not be self-inverse"
    engine = ragged_routing(_routed_mask(ids.size(0), ids.size(1))) if rollout_replay else None
    logps, routing = host._recompute_logps_and_routing(ids, mask, keep, engine, "train", order)
    return logps, None if routing is None else dense_rows(routing), replayed


@pytest.mark.parametrize("mode", ["recompute", "rollout"])
def test_the_step_batch_routes_every_dealt_row_as_its_batch_order_run(mode):
    """R2 captures each dealt row's routing into its own batch row, R3 replays each its own engine routing."""
    rollout = mode == "rollout"
    plain_logps, plain_routing, plain_replayed = _step_recompute(rollout, dealt=False)
    dealt_logps, dealt_routing, dealt_replayed = _step_recompute(rollout, dealt=True)
    assert torch.equal(dealt_logps, plain_logps)
    if rollout:
        assert dealt_routing is None and plain_routing is None, "R3 captures nothing: its routing is the engine's"
    else:
        assert torch.equal(dealt_routing, plain_routing), "a dealt row's routing landed in another batch row"
    assert dealt_replayed.keys() == plain_replayed.keys() and len(plain_replayed) == (12 * _LAYERS if rollout else 0)
    for key, value in plain_replayed.items():
        assert torch.equal(dealt_replayed[key], value), f"row/layer {key[1]} replayed another row's routing"
    if rollout:
        ids, mask, _ = stacked(_round(12, seed=5))
        engine, spans = _routed_mask(12, ids.size(1)), dense_row_spans(mask)[0]
        for (tokens, layer), value in dealt_replayed.items():
            row = next(r for r in range(12) if tuple(ids[r].tolist()) == tokens)
            lo, hi = spans[row]
            assert torch.equal(value, engine[row, lo:hi, layer].long()), (
                f"row {row} layer {layer} replayed foreign ids"
            )


def test_the_dealt_recompute_and_reference_forwards_are_bitwise_the_batch_orders():
    host = scoring_host(
        tiny_llama(_VOCAB, 0), tiny_llama(_VOCAB, 1).eval(), chunks=4, accelerator=single_process_accelerator()
    )
    host.beta = 0.04
    ids, mask, keep = stacked(_round(12, seed=5))
    order = host._no_grad_row_order(mask, "train")
    assert order is not None and not torch.equal(order, torch.arange(12)), "the round should not deal to itself"

    with torch.no_grad():
        plain, _ = host._recompute_logps_and_routing(ids, mask, keep, None, "train")
        dealt, _ = host._recompute_logps_and_routing(ids, mask, keep, None, "train", order)
        ref_plain = host._compute_ref_logps(ids, mask, keep)
        ref_dealt = host._compute_ref_logps(ids, mask, keep, order)
    assert torch.equal(dealt, plain)
    assert torch.equal(ref_dealt, ref_plain)
    assert host._no_grad_row_order(mask, "eval") is None


def test_the_batch_build_reads_bitwise_the_same_ratio_masks_and_advantages_dealt():
    """With a row-wise scorer both no-grad passes see each row's own score. The dealt pass forwards rows longest first,
    the batch-order one as built, and the reference forward takes the recompute's order."""
    forwarded, reference_orders = [], []

    def rowwise_scores(_model, ids, mask, keep, compute_entropy=False):
        forwarded.append(mask.sum(1).tolist())
        weights = torch.arange(1, ids.size(1) + 1, dtype=torch.float32)
        base = (ids.float() * mask * weights).sum(1, keepdim=True) / 1000.0
        return -(base + torch.arange(keep, dtype=torch.float32) * 0.01), None, None

    rows = [row(list(range(3, 3 + n)), sampling=[-0.1 * (i + 1) for i in range(n)]) for n in (5, 1, 7, 2, 6, 3, 4, 8)]
    episodes = [episode(r) for r in (1.0, 0.0, 0.5, 0.25, 0.0, 1.0, 0.75, 0.5)]
    results = {}
    for deal in (False, True):
        host = batch_host(rows, torch.zeros(1), training=True, is_correction=True)
        host._get_per_token_logps_and_entropies = rowwise_scores
        host._compute_ref_logps = lambda ids, mask, keep, order, plan=None: reference_orders.append(order)
        host._orders_rows_by_length = deal
        results[deal] = build(host, episodes)
    built, dealt = forwarded
    assert built == [n + 1 for n in (5, 1, 7, 2, 6, 3, 4, 8)] and dealt == sorted(built, reverse=True)
    batch_order, dealt_order = reference_orders
    assert batch_order is None and [built[i] for i in dealt_order.tolist()] == dealt, "reference left the deal"
    assert results[True].keys() == results[False].keys()
    for key, value in results[False].items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(results[True][key], value), key
        else:
            assert results[True][key] is value is None, key


def _loss_runs(loss_type: str) -> dict[bool, tuple[float, dict, list[torch.Tensor]]]:
    """TRL's real ``loss_type`` loss over a ragged round, shuffled and dealt: ``deal -> (summed loss, row id ->
    loss-forward log-probs, gradients)``."""
    chunks, rows = 4, 16
    batch = _round(rows, seed=9)
    runs = {}
    for deal in (False, True):
        host = loss_host(
            tiny_llama(_VOCAB, 0), chunks=chunks, accelerator=single_process_accelerator(), loss_type=loss_type
        )
        total, row_logps = train_round(host, batch, deal=deal)
        runs[deal] = total, row_logps, [p.grad.clone() for p in host.model.parameters()]
    assert runs[False][1].keys() == runs[True][1].keys() == set(range(rows))
    return runs


@pytest.mark.parametrize("loss_type", _ORDER_FREE_LOSSES)
def test_trls_loss_over_the_dealt_micro_batches_matches_the_shuffled_run(loss_type):
    runs = _loss_runs(loss_type)
    (shuffled_loss, shuffled_logps, shuffled_grads), (dealt_loss, dealt_logps, dealt_grads) = runs[False], runs[True]
    for row_id, logps in shuffled_logps.items():
        assert torch.equal(dealt_logps[row_id], logps), f"row {row_id} log-probs moved"
    assert dealt_loss == pytest.approx(shuffled_loss, rel=1e-6, abs=1e-6)
    for got, want in zip(dealt_grads, shuffled_grads, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


def test_bnpos_loss_reads_which_rows_share_a_micro_batch():
    """The loss the gate keeps the deal off for changes with the deal, so the order-free cases are not vacuous."""
    runs = _loss_runs("bnpo")
    shuffled_loss, dealt_loss = runs[False][0], runs[True][0]
    assert abs(dealt_loss - shuffled_loss) > 1e-3 * abs(shuffled_loss), (dealt_loss, shuffled_loss)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
