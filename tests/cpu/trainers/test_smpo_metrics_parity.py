#!/usr/bin/env python
"""CPU test: SMPO's two loss paths must report the SAME batch metrics from the same numbers.

The non-PP forward sees one whole batch and hands the shared helpers quantities that are already
means (under CP, CP-global means) with unit divisors; the PP loss sees one microbatch at a time and
hands them raw sums with the full-batch divisors. Both fold through ``_smpo_metric_sums`` +
``_smpo_metrics_from_sums``, so these tests fail if either caller starts dividing by something the
other does not — the class of drift (a re-reduced CP mean, a swapped divisor, a key added on one
side only) that no loss curve would show. The PP side's own accumulator — summing over microbatches,
draining at the step boundary, and reporting the full key set from an empty one — is pinned here too,
since it is what makes the two paths comparable at all.

    python tests/cpu/trainers/test_smpo_metrics_parity.py
"""

from types import SimpleNamespace

import pytest
import torch

from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.preference.smpo import SmoothMarginPOTrainer, _PPStepState

BETA = 0.5
TARGET_MARGIN = 0.75
CHOSEN_LOGPS = [-1.0, -2.0, -0.5, -3.0]
REJECTED_LOGPS = [-2.5, -1.0, -1.5, -4.0]
# The SFT divisor (loss-mask tokens) and the logit divisor (those tokens times the vocab) are
# deliberately different numbers, so swapping the two divisors changes the reported value.
CHOSEN_TOKEN_COUNT, REJECTED_TOKEN_COUNT = 30.0, 24.0
# What one PP microbatch contributes: raw logit sums, their token counts, and the NLL sums.
MICROBATCHES = (
    {
        "chosen_logits_sum": 9.0,
        "rejected_logits_sum": 4.0,
        "chosen_logit_tokens": 6.0,
        "rejected_logit_tokens": 8.0,
        "chosen_sft_sum": 4.0,
        "rejected_sft_sum": 3.0,
    },
    {
        "chosen_logits_sum": 21.0,
        "rejected_logits_sum": 12.0,
        "chosen_logit_tokens": 14.0,
        "rejected_logit_tokens": 12.0,
        "chosen_sft_sum": 11.0,
        "rejected_sft_sum": 6.0,
    },
)
# Hand-computed from the constants above — pins WHICH divisor each quantity carries.
EXPECTED = {
    "rewards/chosen": -0.8125,
    "rewards/rejected": -1.125,
    "rewards/accuracies": 0.75,
    "rewards/margins": 0.3125,
    "logps/chosen": -1.625,
    "logps/rejected": -2.25,
    "logits/chosen": 1.5,  # (9 + 21) / (6 + 14)
    "logits/rejected": 0.8,  # (4 + 12) / (8 + 12)
    "sft_loss/chosen": 0.5,  # (4 + 11) / 30
    "sft_loss/rejected": 0.375,  # (3 + 6) / 24
}


def _total(key: str) -> float:
    return sum(microbatch[key] for microbatch in MICROBATCHES)


def _trainer(margin_schedule: bool = False) -> SmoothMarginPOTrainer:
    """A construction-free SMPO carrying only what the loss paths and the metric fold read."""
    trainer = object.__new__(SmoothMarginPOTrainer)
    trainer.use_margin_schedule = margin_schedule
    trainer.target_margin = TARGET_MARGIN
    trainer.model = SimpleNamespace(target_margin=TARGET_MARGIN)  # where the scheduler writes it
    trainer.beta = BETA
    trainer.loss_type = "sigmoid"
    trainer.chosen_sft_ratio = 0.8
    trainer.parallelism_config = ParallelismConfig()  # cp_size reads this (all-ones default)
    trainer._pp_step_state = _PPStepState(
        pair_count=len(CHOSEN_LOGPS),
        chosen_token_count=torch.tensor(CHOSEN_TOKEN_COUNT),
        rejected_token_count=torch.tensor(REJECTED_TOKEN_COUNT),
        target_margin=TARGET_MARGIN,
    )
    trainer.concatenated_forward = lambda model, batch: {
        "chosen_logps": torch.tensor(CHOSEN_LOGPS),
        "rejected_logps": torch.tensor(REJECTED_LOGPS),
        "chosen_sft_loss": torch.tensor(EXPECTED["sft_loss/chosen"]),
        "rejected_sft_loss": torch.tensor(EXPECTED["sft_loss/rejected"]),
        "mean_chosen_logits": torch.tensor(EXPECTED["logits/chosen"]),
        "mean_rejected_logits": torch.tensor(EXPECTED["logits/rejected"]),
    }
    return trainer


def _whole_batch_sums(trainer: SmoothMarginPOTrainer, pair_valid: torch.Tensor | None = None) -> dict:
    """The non-PP call: the whole batch at once, already-mean quantities on unit divisors."""
    chosen_logps = torch.tensor(CHOSEN_LOGPS)
    rejected_logps = torch.tensor(REJECTED_LOGPS)
    unit = torch.ones(())
    return trainer._smpo_metric_sums(
        chosen_rewards=BETA * chosen_logps,
        rejected_rewards=BETA * rejected_logps,
        chosen_logps=chosen_logps,
        rejected_logps=rejected_logps,
        pair_valid=torch.ones_like(chosen_logps) if pair_valid is None else pair_valid,
        chosen_logits_sum=torch.tensor(_total("chosen_logits_sum") / _total("chosen_logit_tokens")),
        rejected_logits_sum=torch.tensor(_total("rejected_logits_sum") / _total("rejected_logit_tokens")),
        chosen_logit_tokens=unit,
        rejected_logit_tokens=unit,
        chosen_sft_sum=torch.tensor(_total("chosen_sft_sum") / CHOSEN_TOKEN_COUNT),
        rejected_sft_sum=torch.tensor(_total("rejected_sft_sum") / REJECTED_TOKEN_COUNT),
    )


def _microbatch_contributions(trainer: SmoothMarginPOTrainer, index: int) -> dict:
    """One PP microbatch's raw sums, as ``_pp_token_loss`` hands them to the shared helper."""
    rows = slice(index * 2, index * 2 + 2)
    chosen_logps = torch.tensor(CHOSEN_LOGPS[rows])
    rejected_logps = torch.tensor(REJECTED_LOGPS[rows])
    return trainer._smpo_metric_sums(
        chosen_rewards=BETA * chosen_logps,
        rejected_rewards=BETA * rejected_logps,
        chosen_logps=chosen_logps,
        rejected_logps=rejected_logps,
        pair_valid=torch.ones_like(chosen_logps),
        **{key: torch.tensor(value) for key, value in MICROBATCHES[index].items()},
    )


def _microbatch_sums(trainer: SmoothMarginPOTrainer) -> dict:
    """The PP call: two microbatches of raw sums, folded the way ``_pp_accumulate_metrics`` folds."""
    accumulated: dict[str, torch.Tensor] = {}
    for index in range(len(MICROBATCHES)):
        for key, value in _microbatch_contributions(trainer, index).items():
            accumulated[key] = value if key not in accumulated else accumulated[key] + value
    return accumulated


def _fold(trainer: SmoothMarginPOTrainer, sums: dict, *, whole_batch: bool) -> dict:
    return trainer._smpo_metrics_from_sums(
        sums,
        pair_count=len(CHOSEN_LOGPS),
        chosen_token_count=1.0 if whole_batch else torch.tensor(CHOSEN_TOKEN_COUNT),
        rejected_token_count=1.0 if whole_batch else torch.tensor(REJECTED_TOKEN_COUNT),
        target_margin=TARGET_MARGIN,
    )


@pytest.mark.parametrize("margin_schedule", [False, True])
def test_the_two_metric_paths_report_the_same_batch(margin_schedule):
    """Whole-batch means on unit divisors must equal microbatch sums on full-batch divisors."""
    trainer = _trainer(margin_schedule)
    non_pp = _fold(trainer, _whole_batch_sums(trainer), whole_batch=True)
    pipelined = _fold(trainer, _microbatch_sums(trainer), whole_batch=False)

    assert non_pp.keys() == pipelined.keys()
    for key in non_pp:
        torch.testing.assert_close(non_pp[key], pipelined[key], msg=f"paths disagree on {key}")
    assert ("target_margin" in non_pp) is margin_schedule
    if margin_schedule:
        # The value, not just the key: a reported constant is the only thing tying the logged series
        # to the margin the loss actually used this step.
        assert non_pp["target_margin"].item() == pytest.approx(TARGET_MARGIN)


def test_each_metric_carries_its_own_divisor():
    """Both paths must land on the hand-computed values — a swapped or extra divisor fails here."""
    trainer = _trainer()
    for label, metrics in (
        ("non-PP", _fold(trainer, _whole_batch_sums(trainer), whole_batch=True)),
        ("PP", _fold(trainer, _microbatch_sums(trainer), whole_batch=False)),
    ):
        assert set(metrics) == set(EXPECTED), label
        for key, expected in EXPECTED.items():
            torch.testing.assert_close(metrics[key], torch.tensor(expected), msg=f"{label} {key}")


def test_already_averaged_quantities_are_reported_untouched():
    """The CP contract: log-probs, logit means and SFT losses arrive as CP-GLOBAL means off PP, so
    the unit-divisor path must hand them back as-is. Re-dividing them by the pair count (the divisor
    the per-pair metrics carry) would silently scale every CP run's metrics by 1/pairs."""
    trainer = _trainer()
    metrics = _fold(trainer, _whole_batch_sums(trainer), whole_batch=True)

    for key in ("logits/chosen", "logits/rejected", "sft_loss/chosen", "sft_loss/rejected"):
        assert metrics[key].item() == pytest.approx(EXPECTED[key]), key
        assert metrics[key].item() != pytest.approx(EXPECTED[key] / len(CHOSEN_LOGPS)), f"{key} re-divided"


def test_invalid_pairs_contribute_nothing():
    """PP eval padding rides in as all-ignore pairs: pair_valid=0 must drop them from every per-pair
    metric, while the divisor stays the real pair count."""
    trainer = _trainer()
    valid = torch.tensor([1.0, 1.0, 0.0, 0.0])
    masked = _fold(trainer, _whole_batch_sums(trainer, pair_valid=valid), whole_batch=True)

    kept = [CHOSEN_LOGPS[0], CHOSEN_LOGPS[1]]
    assert masked["logps/chosen"].item() == pytest.approx(sum(kept) / len(CHOSEN_LOGPS))
    # The untouched, already-averaged quantities do not depend on the mask at all.
    assert masked["logits/chosen"].item() == pytest.approx(EXPECTED["logits/chosen"])


@pytest.mark.parametrize("margin_schedule", [False, True])
def test_the_real_callers_stay_pinned_to_the_shared_helpers(margin_schedule):
    """End to end: ``get_batch_loss_metrics`` and ``_pp_step_metrics`` must agree on names AND
    values for the same batch, or a metric added to one path alone would desync the PP chain
    broadcast (whose key set is pinned) and split the logged series in two."""
    trainer = _trainer(margin_schedule)
    _, non_pp = trainer.get_batch_loss_metrics(model=None, batch={}, train_eval="train")

    pipelined_trainer = _trainer(margin_schedule)
    pipelined_trainer._pp_accumulate_metrics(_microbatch_sums(pipelined_trainer))
    pipelined = pipelined_trainer._pp_step_metrics()

    assert non_pp.keys() == pipelined.keys()
    for key in non_pp:
        torch.testing.assert_close(non_pp[key], pipelined[key], msg=f"callers disagree on {key}")
        assert not non_pp[key].requires_grad, f"{key} pins the autograd graph"


def test_the_accumulator_sums_microbatches_and_drains_at_the_step_boundary():
    """The PP loss hands the accumulator ONE microbatch per call, so the step's metrics are the sum
    over all of them: a fold that overwrote would report only the last microbatch, and one that did
    not drain would make every later step the running total of the run."""
    trainer = _trainer()
    for index in range(len(MICROBATCHES)):
        trainer._pp_accumulate_metrics(_microbatch_contributions(trainer, index))
    stepped = trainer._pp_step_metrics()

    for key, expected in EXPECTED.items():
        torch.testing.assert_close(stepped[key], torch.tensor(expected), msg=f"summed over microbatches: {key}")

    trainer._pp_accumulate_metrics(_microbatch_contributions(trainer, 0))
    fresh = _trainer()
    fresh._pp_accumulate_metrics(_microbatch_contributions(fresh, 0))
    second_step, first_step_only = trainer._pp_step_metrics(), fresh._pp_step_metrics()
    for key, value in second_step.items():
        torch.testing.assert_close(value, first_step_only[key], msg=f"step 2 kept step 1's sums: {key}")


def test_a_stage_that_accumulated_nothing_reports_the_same_names():
    """The PP chain broadcasts values against a key set pinned at setup, so the stages that ran no
    microbatch — every one but the last — must report every name, at zero, or the pin desyncs."""
    empty = _trainer()._pp_step_metrics()
    populated = _trainer()
    populated._pp_accumulate_metrics(_microbatch_sums(populated))

    assert empty.keys() == populated._pp_step_metrics().keys()
    assert all(value.item() == 0.0 for value in empty.values()), empty


def test_eval_prefixes_only_the_caller_side():
    """The helpers return unprefixed names; ``get_batch_loss_metrics`` owns the ``eval_`` prefix."""
    trainer = _trainer()
    _, evaluated = trainer.get_batch_loss_metrics(model=None, batch={}, train_eval="eval")

    assert {name.removeprefix("eval_") for name in evaluated} == set(EXPECTED)
    assert all(name.startswith("eval_") for name in evaluated)
    assert not any(name.startswith("eval_") for name in _fold(trainer, _whole_batch_sums(trainer), whole_batch=True))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
