#!/usr/bin/env python
"""The online trainer's advantage hooks: which combinations are refused, when, and how a rank slices.

* ``use_rlrr`` + ``drop_degenerate_groups`` cancels RLRR's point — the drop keys on raw-reward
  equality, so every all-correct group RLRR just gave length-ranked advantages is masked out of the
  loss — and must be refused at construction like the shaping and std-floor pairings.
* ``multi_objective_aggregation`` other than ``sum_then_normalize`` makes every hook's recompute
  diverge from TRL's; it is refused at construction, not at the first train step.
* A hook computes on the FULL gathered set and slices this rank's rows the way TRL does, so a group
  spanning two ranks (TRL-valid: only ``generation_batch_size % num_generations == 0`` is required)
  must slice correctly rather than be refused.
* An armed hook that finds no stashed rewards in train mode raises instead of returning, and every
  generation batch clears the stash once its hooks have read it.

    python tests/cpu/grpo/test_online_advantage_hook_guards.py
"""

import types
from collections import defaultdict, deque

import numpy as np
import pytest
import torch
from trl import GRPOTrainer

from src.args.mixins import AdvantageShaping, RLRRConfig
from src.trainers.grpo.objective.advantages import degenerate_group_mask
from src.trainers.grpo.objective.relative_rewards import relative_advantages_grouped
from src.trainers.grpo.online import DistributedGRPOTrainer

SUM_THEN_NORMALIZE = types.SimpleNamespace(multi_objective_aggregation="sum_then_normalize")
NORMALIZE_THEN_SUM = types.SimpleNamespace(multi_objective_aggregation="normalize_then_sum")


def _resolve(kwargs: dict, grpo_args=SUM_THEN_NORMALIZE) -> DistributedGRPOTrainer:
    """Run only the pre-super hook resolution on a bare instance (TRL's ctor needs a live server)."""
    trainer = object.__new__(DistributedGRPOTrainer)
    trainer._resolve_advantage_hooks(kwargs, grpo_args)
    return trainer


@pytest.mark.parametrize(
    ("extra", "match"),
    [
        ({"drop_degenerate_groups": True}, "drop_degenerate_groups and rlrr_config"),
        ({"scale_rewards_std_floor": 0.05}, "scale_rewards_std_floor and rlrr_config"),
        ({"advantage_shaping": AdvantageShaping(mode="qae")}, "both replace the advantages"),
    ],
)
def test_rlrr_refuses_every_hook_that_would_cancel_it(extra, match):
    with pytest.raises(ValueError, match=match):
        _resolve({"rlrr_config": RLRRConfig(), **extra})


def test_rlrr_alone_resolves_and_consumes_its_kwargs():
    kwargs = {"rlrr_config": RLRRConfig(), "drop_degenerate_groups": False, "scale_rewards_std_floor": 0.0}
    trainer = _resolve(kwargs)
    assert kwargs == {}, "the hook kwargs must be popped, or TRL's explicit signature rejects them"
    assert trainer._rlrr_config is not None and trainer._recomputes_from_gathered_rewards


def test_drop_without_rlrr_still_masks_degenerate_groups():
    """The refusal is the pairing, not the drop: on its own the drop keeps masking all-alike groups."""
    trainer = _resolve({"drop_degenerate_groups": True})
    assert trainer._drop_degenerate_groups is True
    rewards = torch.tensor([1.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    assert degenerate_group_mask(rewards, 3).tolist() == [True, True, True, False, False, False]


@pytest.mark.parametrize(
    "hook",
    [
        {"drop_degenerate_groups": True},
        {"scale_rewards_std_floor": 0.05},
        {"advantage_shaping": AdvantageShaping(mode="qae")},
        {"rlrr_config": RLRRConfig()},
    ],
)
def test_foreign_aggregation_is_refused_at_construction_once_any_hook_is_armed(hook):
    with pytest.raises(ValueError, match="multi_objective_aggregation='normalize_then_sum'"):
        _resolve(hook, NORMALIZE_THEN_SUM)


def test_foreign_aggregation_is_fine_with_no_hook_armed():
    """No hook, no recompute: TRL's own aggregation runs untouched."""
    trainer = _resolve({}, NORMALIZE_THEN_SUM)
    assert trainer._recomputes_from_gathered_rewards is False


# --- A group spanning two ranks slices the way TRL slices ---

G = 6  # one group of 6 over 2 ranks x 3 rows: TRL-valid (generation_batch_size 6 % 6 == 0)
FULL_REWARDS = torch.tensor([[1.0], [1.0], [1.0], [0.0], [1.0], [0.0]])
FULL_LENGTHS = torch.tensor([2, 30, 10, 5, 20, 5])


def _rank(process_index: int, num_processes: int, *, rlrr=None, drop=False):
    """One rank's view: the stashed rewards are the gathered set, ``gather`` returns world order."""
    n_local = FULL_REWARDS.shape[0] // num_processes
    me = types.SimpleNamespace(
        _rlrr_config=rlrr,
        _advantage_shaping=None,
        _drop_degenerate_groups=drop,
        _scale_rewards_std_floor=0.0,
        _last_rewards_per_func=FULL_REWARDS,
        reward_weights=torch.ones(1),
        num_generations=G,
        model=types.SimpleNamespace(training=True),
        args=types.SimpleNamespace(multi_objective_aggregation="sum_then_normalize"),
        accelerator=types.SimpleNamespace(process_index=process_index, gather=lambda local: FULL_LENGTHS),
        _logs={"advantages": deque([0.0] * G, maxlen=G)},
        _metrics={"train": defaultdict(list)},
    )
    for name in (
        "_gathered_rewards",
        "_local_slice",
        "_install_advantages",
        "_apply_rlrr_advantages",
        "_apply_degenerate_group_drop",
    ):
        setattr(me, name, types.MethodType(getattr(DistributedGRPOTrainer, name), me))
    start = process_index * n_local
    mask = torch.zeros(n_local, int(FULL_LENGTHS.max()), dtype=torch.long)
    for i, length in enumerate(FULL_LENGTHS[start : start + n_local].tolist()):
        mask[i, :length] = 1
    return me, {"advantages": torch.zeros(n_local), "completion_mask": mask}


def test_local_slice_takes_this_ranks_rows_of_a_group_that_spans_ranks():
    full = torch.arange(G, dtype=torch.float32)
    rank0, _ = _rank(0, 2)
    rank1, _ = _rank(1, 2)
    assert rank0._local_slice(full, 3).tolist() == [0.0, 1.0, 2.0]
    assert rank1._local_slice(full, 3).tolist() == [3.0, 4.0, 5.0]


def test_rlrr_hook_on_a_spanning_group_trains_each_rank_on_its_half_of_the_group_result():
    config = RLRRConfig(mode="hrr", lam=4.0)
    expected = relative_advantages_grouped(
        FULL_REWARDS.flatten().numpy(), group_size=G, config=config, lengths=FULL_LENGTHS.float().numpy()
    )
    assert np.std(expected[[0, 1, 2, 4]]) > 0, "fixture: the correct members must be length-ranked apart"

    halves = []
    for process_index in (0, 1):
        me, result = _rank(process_index, 2, rlrr=config)
        me._apply_rlrr_advantages(result)
        halves.append(result["advantages"].tolist())
    assert halves[0] + halves[1] == pytest.approx(expected.tolist(), rel=1e-6), (
        f"the ranks' slices do not reassemble the whole-group RLRR result: {halves} vs {expected.tolist()}"
    )


def test_degenerate_drop_on_a_spanning_group_masks_the_rows_of_each_rank():
    """Groups of 2 over 2 ranks x 3 rows: group 0 (rows 0-1, all-correct) is dropped, group 1 (rows
    2-3) spans the rank boundary and carries signal. The gathered verdict is sliced per rank, so rank
    0 loses exactly its first two rows and rank 1 keeps every row."""
    kept = []
    for process_index in (0, 1):
        me, result = _rank(process_index, 2, drop=True)
        me.num_generations = 2
        me._apply_degenerate_group_drop(result)
        kept.append([n > 0 for n in result["completion_mask"].sum(dim=1).tolist()])
    assert kept == [[False, False, True], [True, True, True]], kept


# --- The gathered-rewards stash: present when a hook needs it, consumed once per batch ---


@pytest.mark.parametrize(
    ("hook", "armed"),
    [
        ("_apply_rlrr_advantages", {"rlrr": RLRRConfig()}),
        ("_apply_degenerate_group_drop", {"drop": True}),
    ],
)
def test_an_armed_hook_without_a_stash_raises_in_train(hook, armed):
    """Returning early would leave TRL's advantages in place while the config says otherwise."""
    me, result = _rank(0, 1, **armed)
    me._last_rewards_per_func = None
    with pytest.raises(RuntimeError, match="no gathered rewards were stashed"):
        getattr(me, hook)(result)


def test_a_missing_stash_is_not_needed_in_eval():
    """The hooks are train-only; eval batches keep TRL's advantages by design."""
    me, result = _rank(0, 1, rlrr=RLRRConfig())
    me._last_rewards_per_func = None
    me.model.training = False
    before = result["advantages"].clone()
    me._apply_rlrr_advantages(result)
    assert torch.equal(result["advantages"], before)


def test_the_generation_batch_consumes_the_stash(monkeypatch):
    """A stash left in place would be read by the next batch's hooks if that batch's scoring skipped
    ``_calculate_rewards``, applying one batch's rewards to another batch's rows."""
    monkeypatch.setattr(GRPOTrainer, "_generate_and_score_completions", lambda self, inputs: {})
    me = object.__new__(DistributedGRPOTrainer)
    me._rlrr_config, me._advantage_shaping, me._drop_degenerate_groups = None, None, False
    me._scale_rewards_std_floor = 0.0
    me.parallelism_config = types.SimpleNamespace(is_tp_mode=False, is_expert_tp_mode=False)
    me._last_rewards_per_func = FULL_REWARDS
    DistributedGRPOTrainer._generate_and_score_completions(me, [])
    assert me._last_rewards_per_func is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
