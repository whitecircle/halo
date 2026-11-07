#!/usr/bin/env python
"""Packed micro-batches on the online trainer: TRL's whole-round old and reference passes plan the round once and score
it bin by bin in batch order, the loss forward then trains the same bins as slots and sums to the unpacked window, and
the setups the packed path cannot hold are refused at construction."""

import ast
import inspect
import textwrap
import types

import pytest
import torch
import trl.trainer.grpo_trainer as trl_grpo_trainer
from trl import GRPOTrainer

import halo.trainers.grpo.environmental.row_order as env_row_order
import halo.trainers.grpo.online as online_module
import halo.trainers.grpo.packed_rounds as packed_rounds
import halo.trainers.grpo.row_order as shared_row_order
from halo.diagnostics.phase_timers import PhaseTimers
from halo.trainers.distillation.sdpg import DistributedSDPGTrainer
from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer
from halo.trainers.grpo.logprob_scoring import LogprobScoringMixin
from halo.trainers.grpo.online import DistributedOnlineGRPOTrainer
from halo.trainers.grpo.packed_rounds import PackedRoundMixin
from tests.common.length_deal import ragged_round, single_process_accelerator, stacked, tiny_llama
from tests.common.online_rounds import old_policy_pass, online_host, train_online_window

_VOCAB = 64
_ROWS, _SLOTS, _BUDGET = 16, 4, 40


def _round(seed: int = 9) -> dict:
    gen = torch.Generator().manual_seed(seed)
    return ragged_round(
        torch.randint(1, 6, (_ROWS,), generator=gen).tolist(),
        torch.randint(1, 9, (_ROWS,), generator=gen).tolist(),
        vocab=_VOCAB,
        seed=seed,
    )


def _online_host(*, packed: bool, loss_type: str = "dapo"):
    return online_host(
        tiny_llama(_VOCAB, 0),
        slots=_SLOTS,
        budget=_BUDGET,
        accelerator=single_process_accelerator(),
        packed=packed,
        loss_type=loss_type,
    )


def _counting_impl(monkeypatch) -> list[int]:
    calls = []
    impl = LogprobScoringMixin._chunked_logps_impl

    def counting(self, unwrapped_model, input_ids, *args):
        calls.append(input_ids.size(0))
        return impl(self, unwrapped_model, input_ids, *args)

    monkeypatch.setattr(LogprobScoringMixin, "_chunked_logps_impl", counting)
    return calls


def test_a_whole_round_pass_scores_bin_by_bin_and_returns_the_per_row_log_probs_in_batch_order(monkeypatch):
    batch = _round()
    legacy = old_policy_pass(_online_host(packed=False), batch)

    calls = _counting_impl(monkeypatch)
    host = _online_host(packed=True)
    with host._scoring_round_in_bins():
        packed = old_policy_pass(host, batch)
    plan = host._round_plan

    assert plan is not None and 1 < plan.num_bins < _ROWS
    assert sorted(calls) == sorted(plan.bin_sizes), "one packed backbone call per bin, none per row"
    assert packed.shape == legacy.shape
    assert torch.allclose(packed, legacy, rtol=1e-4, atol=1e-5), "log-probs left batch order or drifted"


def test_the_unarmed_pass_stays_a_single_call_over_the_rows_it_is_handed(monkeypatch):
    calls = _counting_impl(monkeypatch)
    host = _online_host(packed=True)
    batch = _round()
    old_policy_pass(host, batch)
    assert calls == [_ROWS] and host._round_plan is None


def test_the_old_and_reference_passes_share_one_plan_and_the_layout_takes_it(monkeypatch):
    planned = []
    plan_round = packed_rounds.plan_round_microbatches

    def recording(*args, **kwargs):
        planned.append(1)
        return plan_round(*args, **kwargs)

    monkeypatch.setattr(packed_rounds, "plan_round_microbatches", recording)
    host = _online_host(packed=True)
    batch = _round()
    with host._scoring_round_in_bins():
        old_policy_pass(host, batch)
        first = host._round_plan
        old_policy_pass(host, batch)
        assert host._round_plan is first
    assert len(planned) == 1
    assert host._take_round_plan(batch) is first and host._round_plan is None
    host._take_round_plan(batch)
    assert len(planned) == 2, "a round no pass planned plans itself at the layout, and a plan serves one round"


def test_a_pass_that_asks_for_entropies_or_modality_inputs_is_refused_inside_a_packed_round():
    host = _online_host(packed=True)
    ids, mask, keep = stacked(_round())
    with host._scoring_round_in_bins():
        with pytest.raises(RuntimeError, match="log-probs of text rows only"):
            host._get_per_token_logps_and_entropies(host.model, ids, mask, keep, compute_entropy=True)
        with pytest.raises(RuntimeError, match="log-probs of text rows only"):
            host._get_per_token_logps_and_entropies(host.model, ids, mask, keep, pixel_values=torch.zeros(1))


def test_an_evaluation_round_or_an_unpacked_trainer_arms_nothing():
    for host in (_online_host(packed=False), _online_host(packed=True)):
        host.model.train(host._packs_microbatches is False)
        with host._scoring_round_in_bins():
            assert host._bins_round_scoring is False


@pytest.mark.parametrize("loss_type", ["grpo", "dapo"])
def test_the_packed_online_window_sums_to_the_unpacked_one_with_the_old_pass_on_the_same_bins(loss_type):
    batch = _round()
    legacy_loss, legacy_grads = train_online_window(_online_host(packed=False, loss_type=loss_type), batch)
    assert abs(legacy_loss) > 1e-4, "a vanishing loss compares nothing"
    packed_loss, packed_grads = train_online_window(_online_host(packed=True, loss_type=loss_type), batch)
    assert packed_loss == pytest.approx(legacy_loss, rel=1e-4, abs=1e-6)
    assert all(torch.allclose(g, w, rtol=1e-3, atol=1e-6) for g, w in zip(packed_grads, legacy_grads, strict=True))


def test_the_loss_forward_runs_the_slots_the_old_pass_binned(monkeypatch):
    plans = []
    plan_round = packed_rounds.plan_round_microbatches

    def recording(*args, **kwargs):
        plans.append(plan_round(*args, **kwargs))
        return plans[-1]

    monkeypatch.setattr(packed_rounds, "plan_round_microbatches", recording)
    calls = _counting_impl(monkeypatch)
    train_online_window(_online_host(packed=True), _round())
    (plan,) = plans
    old_pass, loss_forward = calls[: plan.num_bins], calls[plan.num_bins :]
    assert sorted(old_pass) == sorted(plan.bin_sizes)
    assert loss_forward == [plan.rows_per_bin] * plan.num_bins, "one packed call per real slot, no-op slots run none"
    assert plan.num_bins < _SLOTS


def _built_by_the_trainer(host, batch: dict, monkeypatch) -> dict:
    """The trainer's own ``_generate_and_score_completions`` around a TRL build that runs the old-policy pass."""

    def trl_build(self, _inputs):
        return dict(batch) | {"old_per_token_logps": old_policy_pass(self, batch)}

    monkeypatch.setattr(GRPOTrainer, "_generate_and_score_completions", trl_build)
    host._rlrr_config, host._drop_degenerate_groups = None, False
    host._scale_rewards_std_floor, host._balance_token_mass = 0.0, False
    host._last_rewards_per_func = None
    host.beta = 0.0
    return DistributedOnlineGRPOTrainer._generate_and_score_completions(host, [])


def test_the_trainers_round_build_hands_trl_the_slot_layout_of_the_plan_its_old_pass_ran(monkeypatch):
    calls = _counting_impl(monkeypatch)
    host = _online_host(packed=True)
    batch = _round()
    built = _built_by_the_trainer(host, batch, monkeypatch)
    assert set(built) == {"slot_rows", "slot_active", "num_items_in_batch"}
    assert sorted(built["slot_rows"][built["slot_rows"] >= 0].tolist()) == list(range(_ROWS))
    assert 1 < len(calls) < _ROWS and sum(calls) == _ROWS, "the old pass ran one call per bin"
    store = host._round_store
    assert "old_per_token_logps" in store and torch.equal(store["advantages"], batch["advantages"])
    assert host._round_plan is None, "the layout consumed the round's plan"


def test_the_trainers_round_build_is_untouched_without_the_budget_and_in_evaluation(monkeypatch):
    batch = _round()
    for host in (_online_host(packed=False), _online_host(packed=True)):
        host.model.train(host._packs_microbatches is False)
        built = _built_by_the_trainer(host, batch, monkeypatch)
        assert set(built) >= {"advantages", "old_per_token_logps"} and host._round_store is None


def _configured(trainer_cls, **overrides):
    host = object.__new__(trainer_cls)
    host.args = types.SimpleNamespace(
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        loss_type="dapo",
        top_entropy_quantile=1.0,
        entropy_coef=0.0,
        steps_per_generation=4,
        num_iterations=1,
        gradient_accumulation_steps=4,
        eval_strategy="no",
        eval_on_start=False,
    )
    host.num_iterations = 1
    host._use_chunked_grpo_logprobs = True
    host._packed_microbatch_tokens = 4096
    host.model = tiny_llama(_VOCAB, 0)
    host.model.config._attn_implementation = "flash_attention_2"
    host.accelerator = single_process_accelerator()
    host.parallelism_config = types.SimpleNamespace(ep_size=1, requires_rdma=False)
    for name, value in overrides.items():
        setattr(host, name, value)
    return host


def test_the_online_trainer_packs_when_its_setup_is_admitted_and_sets_the_flags():
    host = _configured(DistributedOnlineGRPOTrainer)
    host._configure_packed_microbatches()
    assert host._packs_microbatches is True and host._stashes_row_spans is False


def test_an_unset_budget_changes_nothing():
    host = _configured(DistributedOnlineGRPOTrainer, _packed_microbatch_tokens=None)
    host._configure_packed_microbatches()
    assert host._packs_microbatches is False


def test_sdpg_declares_the_refusal_and_the_online_trainer_configures_after_placing_the_model():
    assert DistributedSDPGTrainer.PACKED_ROWS_REFUSAL and DistributedOnlineGRPOTrainer.PACKED_ROWS_REFUSAL is None
    init = ast.parse(textwrap.dedent(inspect.getsource(DistributedOnlineGRPOTrainer.__init__)))
    called = [n.func.attr for n in ast.walk(init) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    assert called.index("_configure_packed_microbatches") > called.index("_finish_on_policy_init")


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ({"loss_type": "bnpo"}, "loss_type='bnpo'"),
        ({"per_device_train_batch_size": 2}, "per_device_train_batch_size: 1"),
        ({"top_entropy_quantile": 0.5}, "top_entropy_quantile"),
        ({"gradient_accumulation_steps": 6}, "Align them"),
    ],
)
def test_the_online_trainer_refuses_what_the_environmental_one_refuses(args, match):
    host = _configured(DistributedOnlineGRPOTrainer)
    for name, value in args.items():
        setattr(host.args, name, value)
    with pytest.raises(ValueError, match=match):
        host._configure_packed_microbatches()


def test_the_online_trainer_refuses_full_logits_and_a_non_flash_backend():
    host = _configured(DistributedOnlineGRPOTrainer, _use_chunked_grpo_logprobs=False)
    with pytest.raises(ValueError, match="use_chunked_grpo_logprobs: true"):
        host._configure_packed_microbatches()
    host = _configured(DistributedOnlineGRPOTrainer)
    host.model.config._attn_implementation = "sdpa"
    with pytest.raises(ValueError, match="varlen flash attention backend"):
        host._configure_packed_microbatches()


def _refill_host(slot_batch: dict):
    host = object.__new__(DistributedOnlineGRPOTrainer)
    host.model = types.SimpleNamespace(training=True)
    host.args = types.SimpleNamespace(steps_per_generation=2, gradient_accumulation_steps=2)
    host.num_iterations = 1
    host._step, host._buffered_inputs = 0, None
    host._packs_microbatches = True
    host._generate_and_score_completions = lambda _inputs: dict(slot_batch)
    host._broadcast_tensors_from_tp_leader = lambda inputs: inputs
    host._build_slot_view = lambda slot: slot
    return host


_SLOT_BATCH = {
    "slot_rows": torch.tensor([-1, -1, 0, 1]),
    "slot_active": torch.tensor([0, 0, 1, 1]),
    "num_items_in_batch": torch.tensor(3),
}


def test_an_online_refill_splits_the_slot_indices_in_order_and_restores_trls_shuffle():
    shuffle = trl_grpo_trainer.shuffle_sequence_dict
    host = _refill_host(_SLOT_BATCH)
    first = host._prepare_inputs({})
    host._step += 1
    second = host._prepare_inputs({})
    assert trl_grpo_trainer.shuffle_sequence_dict is shuffle
    assert first["slot_rows"].tolist() == [-1, -1] and second["slot_rows"].tolist() == [0, 1]


def test_an_online_refill_after_a_trl_that_no_longer_shuffles_is_refused(monkeypatch):
    def moved(self, generation_batch):
        self._buffered_inputs = [self._generate_and_score_completions(generation_batch)]
        return self._buffered_inputs[0]

    monkeypatch.setattr(GRPOTrainer, "_prepare_inputs", moved)
    with pytest.raises(RuntimeError, match="moved its buffer shuffle"):
        _refill_host(_SLOT_BATCH)._prepare_inputs({})


def test_both_on_policy_trainers_share_one_row_order_and_one_round_layout():
    assert env_row_order.length_deal_order is shared_row_order.length_deal_order
    assert env_row_order.permute_rows is shared_row_order.permute_rows
    assert env_row_order.MICROBATCH_INVARIANT_LOSS_TYPES is shared_row_order.MICROBATCH_INVARIANT_LOSS_TYPES
    for name in ("_prepare_inputs", "_build_slot_view", "_lay_out_packed_round", "_score_bins", "_score_inert_row"):
        assert getattr(DistributedAsyncEnvironmentalGRPOTrainer, name) is getattr(DistributedOnlineGRPOTrainer, name)


def test_the_round_layout_leads_the_tp_resync_and_the_scorer_in_both_on_policy_trainers():
    mro = DistributedOnlineGRPOTrainer.__mro__
    assert DistributedOnlineGRPOTrainer._prepare_inputs is PackedRoundMixin._prepare_inputs
    assert mro.index(PackedRoundMixin) < mro.index(GRPOTrainer)
    assert DistributedOnlineGRPOTrainer._packs_microbatches is False
    assert DistributedOnlineGRPOTrainer.PACKED_ROWS_REFUSAL is None


def test_the_online_weight_push_runs_outside_the_round_being_binned_and_restores_it(monkeypatch):
    seen = []
    monkeypatch.setattr(
        online_module,
        "sync_trainer_weights",
        lambda trainer, client, **_: seen.append((trainer._bins_round_scoring, client)),
    )
    trainer = types.SimpleNamespace(
        _bins_round_scoring=True,
        _rollout_weight_precision="bf16",
        first_push_verified=True,
        _phase_timers=PhaseTimers(enabled=False),
        accelerator=types.SimpleNamespace(is_main_process=True),
        vllm_generation=types.SimpleNamespace(vllm_client="client"),
        unbinned_scoring=lambda: PackedRoundMixin.unbinned_scoring(trainer),
    )

    DistributedOnlineGRPOTrainer._distributed_sync_weights(trainer)

    assert seen == [(False, "client")], "the push's probe forward ran inside the binned round"
    assert trainer._bins_round_scoring, "the round's own scoring passes must stay binned after the push"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
