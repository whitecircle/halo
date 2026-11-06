#!/usr/bin/env python
"""Rank balance deals an online round's rows across ranks by attended tokens: the deal is the same on every rank and
loses no row, the ranks' totals come close on a skewed round, the objective is the unbalanced one, and the setups where
a deal would change it or has nothing to move are refused."""

import ast
import dataclasses
import inspect
import textwrap
import types

import pytest
import torch
import torch.distributed as dist
from trl import GRPOTrainer

import halo.trainers.grpo.rank_balance as rank_balance
from halo.configs.grpo import RankBalanceArguments
from halo.configs.grpo_environmental import AsyncRolloutConfig
from halo.configs.grpo_online import OnlineGRPOScriptArguments
from halo.trainers.distillation.sdpg import DistributedSDPGTrainer
from halo.trainers.grpo.online import DistributedOnlineGRPOTrainer
from halo.trainers.grpo.rank_balance import (
    RANK_BALANCE_KNOB,
    SPREAD_AFTER_KEY,
    SPREAD_BEFORE_KEY,
    RankSpread,
    deal_order,
    deal_rows_across_ranks,
    max_over_mean,
    reject_unbalanceable_rank_tokens,
)
from halo.trainers.grpo.row_order import reject_trainer_refusal
from tests.common.gloo import run_gloo_ranks
from tests.common.length_deal import single_process_accelerator, tiny_llama, with_world_normalizer, world_accelerator
from tests.common.online_rounds import online_host
from tests.common.rank_rounds import ROWS_PER_RANK, VOCAB, row_contents, row_tokens, skewed_round
from tests.common.repo import REPO_ROOT


def _gathered(value):
    out = [None] * dist.get_world_size()
    dist.all_gather_object(out, value)
    return out


def _deal_worker(rank: int, nprocs: int) -> None:
    batch = skewed_round(rank, nprocs)
    dealt, spread = deal_rows_across_ranks(batch, pad_token_id=0)

    before = _gathered(row_tokens(batch).sum().item())
    after = _gathered(row_tokens(dealt).sum().item())
    assert spread.before == pytest.approx(max(before) / (sum(before) / nprocs))
    assert spread.after == pytest.approx(max(after) / (sum(after) / nprocs))
    assert spread.before > 1.5 and spread.after < 1.1 and sum(after) == sum(before)
    assert len(set(_gathered((spread.before, spread.after)))) == 1, "ranks computed different deals"

    original = {k: v for ranks_rows in _gathered(row_contents(batch)) for k, v in ranks_rows.items()}
    received = [row_contents(rows) for rows in _gathered(dealt)]
    assert all(len(rows) == ROWS_PER_RANK for rows in received), "a rank's row count changed"
    flat = {k: v for rows in received for k, v in rows.items()}
    assert sum(len(rows) for rows in received) == len(flat) == nprocs * ROWS_PER_RANK, "a row was lost or duplicated"
    assert flat == original, "a row's tokens or per-token values changed in the move"
    assert any(set(rows) != set(row_contents(batch)) for rows in received), "no row left its rank"

    assert dealt["num_items_in_batch"] is batch["num_items_in_batch"]
    assert dealt["prompt_mask"][:, 0].any() and dealt["completion_mask"][:, -1].any(), "padding was not trimmed"
    assert dealt["prompt_ids"].shape == dealt["prompt_mask"].shape
    assert dealt["completion_ids"].shape == dealt["old_per_token_logps"].shape == dealt["completion_mask"].shape
    assert dealt["completion_ids"].shape == dealt["importance_sampling_ratio"].shape == dealt["tool_mask"].shape


@pytest.mark.parametrize("nprocs", [2, 4])
def test_every_rank_computes_the_same_deal_that_loses_no_row_and_evens_the_ranks_tokens(nprocs):
    run_gloo_ranks(_deal_worker, nprocs, nprocs)


def _balanced_worker(rank: int) -> None:
    batch = skewed_round(1, 2)
    dealt, spread = deal_rows_across_ranks(dict(batch), pad_token_id=0)
    assert spread == RankSpread(1.0, 1.0)
    assert all(torch.equal(dealt[k], batch[k]) for k in ("prompt_ids", "completion_ids", "advantages"))


def test_a_round_the_ranks_already_share_evenly_is_left_as_it_is():
    run_gloo_ranks(_balanced_worker, 2)


def _foreign_worker(rank: int) -> None:
    batch = skewed_round(rank, 2)
    if rank == 1:
        batch["pixel_values"] = torch.zeros(3, 4)
    with pytest.raises(ValueError, match="pixel_values"):
        deal_rows_across_ranks(batch, pad_token_id=0)


def test_a_round_with_image_tensors_on_any_rank_raises_on_every_rank():
    run_gloo_ranks(_foreign_worker, 2)


def _list_worker(rank: int) -> None:
    batch = skewed_round(rank, 2)
    if rank == 0:
        batch["images"] = [None] * ROWS_PER_RANK
    with pytest.raises(ValueError, match="images"):
        deal_rows_across_ranks(batch, pad_token_id=0)


def test_a_round_with_a_per_row_list_on_any_rank_raises_on_every_rank():
    run_gloo_ranks(_list_worker, 2)


def _fill_worker(rank: int, nprocs: int) -> None:
    pad = 7
    batch = skewed_round(rank, nprocs)
    widths = _gathered(batch["completion_ids"].size(1))
    dealt, _ = deal_rows_across_ranks(batch, pad_token_id=pad)
    width = dealt["completion_ids"].size(1)
    checked = 0
    for row, advantage in enumerate(dealt["advantages"].tolist()):
        source = widths[int(advantage // 10)]
        if source < width:
            checked += 1
            assert (dealt["completion_ids"][row, source:] == pad).all()
            assert (dealt["importance_sampling_ratio"][row, source:] == 1).all()
            assert (dealt["completion_mask"][row, source:] == 0).all()
    _gathered(checked)
    assert sum(_gathered(checked)) > 0, "no row came from a narrower rank, so no fill was compared"


def test_columns_a_row_gains_from_a_wider_rank_hold_the_pad_token_and_a_unit_ratio():
    run_gloo_ranks(_fill_worker, 2, 2)


def _window(batch: dict, loss_type: str) -> tuple[float, list[torch.Tensor]]:
    """One optimizer step on this rank's rows through the online trainer's real loss, summed over the world."""
    host = online_host(
        tiny_llama(VOCAB, 0),
        slots=ROWS_PER_RANK,
        budget=0,
        accelerator=world_accelerator(),
        packed=False,
        loss_type=loss_type,
    )
    host._step, host._buffered_inputs = 0, None
    host._generate_and_score_completions = lambda _inputs: dict(batch)
    torch.manual_seed(7)
    total = torch.zeros(())
    for _ in range(host.args.gradient_accumulation_steps):
        inputs = host._prepare_inputs({})
        host._step += 1
        loss = DistributedOnlineGRPOTrainer.compute_loss(host, host.model, inputs)
        loss.backward()
        total += loss.detach()
    grads = torch.cat([p.grad.reshape(-1) for p in host.model.parameters()])
    packed = torch.cat([total.reshape(1), grads])
    dist.all_reduce(packed)
    return packed[0].item(), packed[1:]


def _objective_worker(rank: int, nprocs: int, loss_type: str) -> None:
    batch = with_world_normalizer(skewed_round(rank, nprocs))
    dealt, spread = deal_rows_across_ranks(batch, pad_token_id=0)
    assert spread.after < spread.before, "the control round must move rows"
    plain_loss, plain_grads = _window(batch, loss_type)
    dealt_loss, dealt_grads = _window(dealt, loss_type)
    assert abs(plain_loss) > 1e-4, "a vanishing loss compares nothing"
    assert dealt_loss == pytest.approx(plain_loss, rel=1e-4, abs=1e-6)
    assert torch.allclose(dealt_grads, plain_grads, rtol=1e-3, atol=1e-6)


@pytest.mark.parametrize("loss_type", ["grpo", "dapo"])
def test_the_dealt_round_has_the_unbalanced_rounds_loss_and_gradient(loss_type):
    run_gloo_ranks(_objective_worker, 2, 2, loss_type)


def test_the_deal_gives_each_rank_its_share_heaviest_rows_first_onto_the_lightest_rank():
    tokens = [9, 1, 8, 2, 7, 3, 6, 4]
    order = deal_order(tokens, 2)
    shares = [order[:4], order[4:]]
    assert sorted(order) == list(range(8))
    assert [sum(tokens[r] for r in rows) for rows in shares] == [20, 20]
    assert all(rows == sorted(rows) for rows in shares)
    assert deal_order(tokens, 2) == order, "the deal is a pure function of the counts"


def test_a_deal_ties_to_the_lowest_rank_and_row_and_rejects_unequal_shares():
    assert deal_order([5, 5, 5, 5], 2) == [0, 2, 1, 3]
    with pytest.raises(ValueError, match="equal shares"):
        deal_order([1, 2, 3], 2)


def test_the_spread_is_the_heaviest_ranks_tokens_over_the_mean():
    tokens = [9, 9, 1, 1]
    assert max_over_mean(tokens, range(4), 2) == pytest.approx(18 / 10)
    assert max_over_mean(tokens, deal_order(tokens, 2), 2) == pytest.approx(1.0)
    assert max_over_mean([0, 0], range(2), 2) == 1.0


def _args(**overrides):
    values = {
        "loss_type": "dapo",
        "top_entropy_quantile": 1.0,
        "entropy_coef": 0.0,
        "gradient_accumulation_steps": 4,
    }
    return types.SimpleNamespace(**(values | overrides))


_PARALLEL = types.SimpleNamespace(is_tp_mode=False, is_expert_tp_mode=False)


def _admit(args=None, parallel=_PARALLEL, **overrides):
    reject_unbalanceable_rank_tokens(
        args or _args(), parallel, **({"ranks": 8, "misaligned_accumulation": False} | overrides)
    )


def test_a_setup_where_the_deal_is_sound_is_admitted():
    _admit()
    for loss_type in ("grpo", "sapo", "dr_grpo", "luspo", "dapo", "cispo"):
        _admit(_args(loss_type=loss_type))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"args": _args(loss_type="bnpo")}, "loss_type='bnpo'"),
        ({"args": _args(top_entropy_quantile=0.5)}, "top_entropy_quantile"),
        ({"args": _args(entropy_coef=0.01)}, "entropy_coef=0.01"),
        ({"misaligned_accumulation": True}, "straddles a generation round"),
        ({"parallel": types.SimpleNamespace(is_tp_mode=True, is_expert_tp_mode=False)}, "TP/ETP"),
        ({"parallel": types.SimpleNamespace(is_tp_mode=False, is_expert_tp_mode=True)}, "TP/ETP"),
        ({"ranks": 1}, "balances rows across ranks"),
    ],
)
def test_a_setup_the_deal_would_change_or_could_not_use_is_refused(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _admit(**kwargs)


def test_sdpg_declares_the_refusal_and_the_trainer_reads_it_before_trls_constructor():
    assert DistributedSDPGTrainer.RANK_BALANCE_REFUSAL and DistributedOnlineGRPOTrainer.RANK_BALANCE_REFUSAL is None
    with pytest.raises(ValueError, match="DistributedSDPGTrainer"):
        reject_trainer_refusal(DistributedSDPGTrainer, DistributedSDPGTrainer.RANK_BALANCE_REFUSAL, RANK_BALANCE_KNOB)
    reject_trainer_refusal(DistributedOnlineGRPOTrainer, None, RANK_BALANCE_KNOB)
    init = ast.parse(textwrap.dedent(inspect.getsource(DistributedOnlineGRPOTrainer.__init__)))
    line = {
        (n.func.attr if isinstance(n.func, ast.Attribute) else n.func.id): n.lineno
        for n in ast.walk(init)
        if isinstance(n, ast.Call) and isinstance(n.func, (ast.Attribute, ast.Name))
    }
    assert line["reject_trainer_refusal"] < line["__init__"], "the refusal must precede TRL's constructor"
    assert line["_configure_rank_balance"] > line["_finish_on_policy_init"]


def _configured(**overrides):
    host = object.__new__(DistributedOnlineGRPOTrainer)
    host.args = _args(steps_per_generation=4)
    host.num_iterations = 1
    host.accelerator = types.SimpleNamespace(num_processes=8)
    host.parallelism_config = _PARALLEL
    for name, value in overrides.items():
        setattr(host, name, value)
    return host


def test_the_trainer_refuses_at_construction_what_the_deal_cannot_hold_and_arms_nothing_when_off():
    _configured(
        _balances_rank_tokens=False, args=_args(loss_type="bnpo", steps_per_generation=4)
    )._configure_rank_balance()
    with pytest.raises(ValueError, match="loss_type='bnpo'"):
        _configured(
            _balances_rank_tokens=True, args=_args(loss_type="bnpo", steps_per_generation=4)
        )._configure_rank_balance()
    _configured(_balances_rank_tokens=True)._configure_rank_balance()


def _build_with_deal(monkeypatch, *, packed: bool):
    batch = skewed_round(0, 2)
    batch["old_per_token_logps"] = None
    dealt_calls = []

    def reversing(round_tensors, *, pad_token_id):
        dealt_calls.append(round_tensors)
        flipped = {k: v.flip(0) if isinstance(v, torch.Tensor) and v.ndim else v for k, v in round_tensors.items()}
        return flipped, RankSpread(2.0, 1.25)

    monkeypatch.setattr(rank_balance, "deal_rows_across_ranks", reversing)
    host = online_host(
        tiny_llama(VOCAB, 0), slots=ROWS_PER_RANK, budget=64, accelerator=single_process_accelerator(), packed=packed
    )
    host._balances_rank_tokens = True
    host._metrics = {"train": __import__("collections").defaultdict(list)}
    host._rlrr_config, host._drop_degenerate_groups = None, False
    host._scale_rewards_std_floor, host._balance_token_mass = 0.0, False
    host._last_rewards_per_func = None
    host.beta = 0.0
    scored = []

    def trl_build(self, _inputs):
        out = dict(batch)
        if self._bins_round_scoring:
            scored.append("scored")
            self._round_plan = "stale"
        return out

    monkeypatch.setattr(GRPOTrainer, "_generate_and_score_completions", trl_build)
    return host, batch, dealt_calls


def test_the_round_build_deals_the_final_round_and_records_the_spread(monkeypatch):
    host, batch, dealt_calls = _build_with_deal(monkeypatch, packed=False)
    built = DistributedOnlineGRPOTrainer._generate_and_score_completions(host, [])
    (seen,) = dealt_calls
    assert torch.equal(seen["advantages"], batch["advantages"]), "the deal runs over the round as TRL built it"
    assert torch.equal(built["advantages"], batch["advantages"].flip(0))
    assert host._metrics["train"][SPREAD_BEFORE_KEY] == [2.0] and host._metrics["train"][SPREAD_AFTER_KEY] == [1.25]


def test_an_evaluation_round_is_not_dealt(monkeypatch):
    host, batch, dealt_calls = _build_with_deal(monkeypatch, packed=False)
    host.model.eval()
    built = DistributedOnlineGRPOTrainer._generate_and_score_completions(host, [])
    assert not dealt_calls and torch.equal(built["advantages"], batch["advantages"])


def test_a_plan_the_scoring_passes_made_over_the_undealt_rows_is_not_the_layouts(monkeypatch):
    host, batch, _ = _build_with_deal(monkeypatch, packed=True)
    seen = []
    layout = DistributedOnlineGRPOTrainer._lay_out_packed_round
    monkeypatch.setattr(
        DistributedOnlineGRPOTrainer,
        "_take_round_plan",
        lambda self, round_batch: seen.append(self._round_plan)
        or host._plan_round(torch.cat([round_batch["prompt_mask"], round_batch["completion_mask"]], dim=1)),
    )
    built = DistributedOnlineGRPOTrainer._generate_and_score_completions(host, [])
    assert seen == [None], "the stale plan reached the layout"
    assert torch.equal(host._round_store["advantages"], batch["advantages"].flip(0))
    assert sorted(built["slot_rows"][built["slot_rows"] >= 0].tolist()) == list(range(ROWS_PER_RANK))
    assert layout is not None


def test_the_deal_precedes_the_layout_the_row_spans_and_the_tp_broadcast_in_the_round_build():
    body = ast.parse(textwrap.dedent(inspect.getsource(DistributedOnlineGRPOTrainer._generate_and_score_completions)))
    line = {
        n.func.attr: n.lineno for n in ast.walk(body) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    assert line["_clamp_kl_reference"] < line["_deal_round_across_ranks"] < line["_lay_out_packed_round"]
    assert line["_lay_out_packed_round"] < line["_broadcast_tensors_from_tp_leader"] == line["_with_row_spans"]


def test_only_the_online_config_takes_the_knob_from_its_bundle_and_the_script_hands_it_on():
    assert issubclass(OnlineGRPOScriptArguments, RankBalanceArguments)
    assert not issubclass(AsyncRolloutConfig, RankBalanceArguments)
    (field,) = [f for f in dataclasses.fields(OnlineGRPOScriptArguments) if f.name == "balance_rank_tokens"]
    assert field.default is False
    tree = ast.parse((REPO_ROOT / "scripts" / "training" / "grpo" / "online.py").read_text())
    forwarded = {
        keyword.arg: ast.unparse(keyword.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "trainer_cls"
        for keyword in node.keywords
    }
    assert forwarded["balance_rank_tokens"] == "args.balance_rank_tokens"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
