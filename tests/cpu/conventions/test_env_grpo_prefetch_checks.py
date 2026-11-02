"""The env-GRPO e2e grades a prefetching run by the round each step acquired, not by its hit count.

A hit is a round delivered before its step asks for it, which depends on how long the round runs against the
update: a run that prefetches correctly can finish with no hit, and one that pairs a step with the wrong round can
still hit. The recorder wraps the trainer's acquire, so the check reads what each step took: its first
``prefetch_depth`` steps collect their own prompts and every later one pops a prefetched round, whatever the
server count.
"""

import types

import pytest

from halo.trainers.grpo.objective.logratio import LOGRATIO_MEAN_KEY
from tests.common.env_grpo_e2e import PREFETCH_POPPED_STEPS, record_prefetch, record_prefetch_acquisitions

_ORDER_CHECK = "prefetch_opened_with_depth_cold_rounds_then_popped_every_step"


def _graded(acquisitions: list, depth: int, hits: int) -> tuple[dict[str, bool], list[bool]]:
    """Step a trainer whose acquire returns ``acquisitions`` in turn (``None``: the step collects its own prompts),
    logging ``hits`` and a finite log-ratio every step, and grade it at ``depth``."""
    outcomes = iter(acquisitions)
    trainer = types.SimpleNamespace(
        _acquire_prefetched_round=lambda: next(outcomes),
        _prefetch_enabled=True,
        async_config=types.SimpleNamespace(prefetch_depth=depth),
        state=types.SimpleNamespace(log_history=[]),
    )
    acquired = record_prefetch_acquisitions(trainer)
    for _ in acquisitions:
        trainer._acquire_prefetched_round()
        trainer.state.log_history.append({LOGRATIO_MEAN_KEY: 0.01, "async/prefetch_hits": hits})
    checks: dict[str, bool] = {}
    record_prefetch(trainer, checks, {}, acquired, depth)
    return checks, acquired


@pytest.mark.parametrize("depth", [1, 2])
def test_depth_cold_rounds_then_a_pop_every_step_passes_with_no_hit(depth):
    rounds = [None] * depth + [["rollout"]] * PREFETCH_POPPED_STEPS
    checks, acquired = _graded(rounds, depth, hits=0)
    assert acquired == [False] * depth + [True] * PREFETCH_POPPED_STEPS
    assert all(checks.values()), checks


@pytest.mark.parametrize(
    "rounds",
    [
        [None, None, None, ["rollout"]],
        [None, ["rollout"], ["rollout"], ["rollout"]],
        [None, None, ["rollout"], None],
    ],
    ids=["one-cold-round-too-many", "one-cold-round-too-few", "a-later-step-collected-its-own"],
)
def test_a_step_that_did_not_take_its_round_fails_however_many_hits(rounds):
    checks, _acquired = _graded(rounds, depth=2, hits=3)
    assert not checks[_ORDER_CHECK]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
