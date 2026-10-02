#!/usr/bin/env python
"""CPU tests: the GRPO early stop, shared by the online and environmental trainers.

A condition ends training once it breaches on ``early_stop_patience`` readings in a row: entropy out of
its band (either side), the trainer-vs-sampler log-prob gap above its limit (each trainer names the
metric it logs, signed or absolute), or, in env GRPO, every update skipped by the trust-region breaker.
A healthy reading resets the streak, a log without the reading (a step reusing its generation round)
leaves it, eval logs never count, and nothing is armed unless a condition is set.

    python tests/cpu/grpo/test_grpo_early_stop.py
"""

import ast
import dataclasses
import types

import pytest
from accelerate import PartialState
from transformers import TrainerControl, TrainerState

from src.args.mixins import EarlyStopConfig
from src.args.rlvr_online_grpo_args import RLVROnlineGRPOScriptArguments
from src.configs.async_training_config import AsyncTrainingConfig
from src.trainers.grpo.early_stop import (
    LOGRATIO_MEAN_KEY,
    SAMPLING_LOGP_GAP_KEY,
    UPDATE_SKIPPED_KEY,
    GRPOEarlyStopCallback,
    build_early_stop_callback,
)
from src.training.script_runner import run_trainer
from tests.common.utils import REPO_ROOT

PartialState()  # the stop logs through accelerate's logger, which refuses to log without it


def _run(config: EarlyStopConfig, steps: list[dict], gap_key: str = LOGRATIO_MEAN_KEY) -> int | None:
    """Feed logged steps to the callback; the 1-based step at which it ended training, or None."""
    callback, control, state = GRPOEarlyStopCallback(config, gap_key), TrainerControl(), TrainerState()
    for i, logs in enumerate(steps, 1):
        state.global_step = i
        callback.on_log(None, state, control, logs=logs)
        if control.should_training_stop:
            return i
    return None


def _step(entropy=0.24, logratio=-0.002, skipped=0.0):
    return {"entropy": entropy, LOGRATIO_MEAN_KEY: logratio, UPDATE_SKIPPED_KEY: skipped}


BAND = EarlyStopConfig(entropy_band=(0.15, 0.35), patience=3)


def test_entropy_above_the_band_stops_on_the_patience_th_step_in_a_row():
    assert _run(BAND, [_step(), _step(0.4), _step(0.5), _step(0.6), _step(0.7)]) == 4


def test_the_stop_step_is_neither_saved_nor_evaluated():
    """HF evaluates and saves after the log callbacks: a checkpoint of the stop step would come from inside
    the drift."""
    callback, control, state = GRPOEarlyStopCallback(BAND, LOGRATIO_MEAN_KEY), TrainerControl(), TrainerState()
    control.should_save = control.should_evaluate = True
    for _ in range(3):
        callback.on_log(None, state, control, logs=_step(0.5))
    assert control.should_training_stop and not control.should_save and not control.should_evaluate


def test_steps_that_reuse_a_round_neither_break_nor_extend_a_streak():
    """The gap and the breaker's verdict are logged once per generation round; under ``num_iterations: 2``
    every other logged step carries only the entropy. Read as healthy, those steps would reset the streak
    and the stop could never fire."""
    config = EarlyStopConfig(logratio_gap=0.005, on_skipped_updates=True, patience=3)
    round_step, reuse_step = _step(logratio=-0.05, skipped=1.0), {"entropy": 0.24}
    assert _run(config, [round_step, reuse_step] * 4) == 5


def test_an_epoch_end_after_the_stop_saves_and_evaluates_nothing():
    """The loop breaks into one more epoch-end pass, where an epoch save or eval strategy arms both flags."""
    callback, control, state = GRPOEarlyStopCallback(BAND, LOGRATIO_MEAN_KEY), TrainerControl(), TrainerState()
    for _ in range(3):
        callback.on_log(None, state, control, logs=_step(0.5))
    control.should_save = control.should_evaluate = True
    callback.on_epoch_end(None, state, control)
    assert not control.should_save and not control.should_evaluate
    healthy, control = GRPOEarlyStopCallback(BAND, LOGRATIO_MEAN_KEY), TrainerControl()
    control.should_save = True
    healthy.on_epoch_end(None, state, control)
    assert control.should_save, "a run that did not stop keeps its epoch save"


class _StoppableTrainer:
    """Trains one logged step per entry of ``logs`` under the stop, the way the HF loop feeds it, toward a plan
    of ``max_steps``."""

    def __init__(self, logs: list[dict], max_steps: int | None = None):
        self.callback_handler = types.SimpleNamespace(callbacks=[GRPOEarlyStopCallback(BAND, LOGRATIO_MEAN_KEY)])
        self.state = TrainerState(max_steps=len(logs) if max_steps is None else max_steps)
        self.events: list[str] = []
        self._logs = logs

    def train(self, resume_from_checkpoint=None):
        (stop,), control = self.callback_handler.callbacks, TrainerControl()
        for logs in self._logs:
            self.state.global_step += 1
            stop.on_log(None, self.state, control, logs=logs)
            if control.should_training_stop:
                break
        self.events.append("train")

    def cleanup_ep(self):
        self.events.append("cleanup")


_RUNTIME = types.SimpleNamespace(
    parallelism_config=types.SimpleNamespace(
        is_ep_mode=False, is_cp_mode=False, is_tp_mode=False, is_expert_tp_mode=False, data_parallel_size=1
    ),
    mode_suffix="ddp",
    resume_checkpoint=None,
)


def test_a_stopped_run_exits_non_zero_after_its_cleanup_and_any_other_returns():
    """Exiting 0, a stopped run reads as finished to torchrun and a scheduler, and a chained stage would start
    from its output; the EP buffers are released first either way. A run that ends short of its plan without
    a stop (a ``no_duplicates`` sampler yields fewer batches than its length) trained all its data."""
    stopped = _StoppableTrainer([_step(0.5)] * 5)
    with pytest.raises(SystemExit) as exited:
        run_trainer(stopped, _RUNTIME, method_name="GRPO")
    assert exited.value.code == "GRPO training stopped early at step 3 of 5."
    assert stopped.events == ["train", "cleanup"]

    for healthy in (_StoppableTrainer([_step()] * 5), _StoppableTrainer([_step()] * 4, max_steps=5)):
        run_trainer(healthy, _RUNTIME, method_name="GRPO")
        assert healthy.events == ["train", "cleanup"]


def test_entropy_below_the_band_stops_too():
    assert _run(BAND, [_step(0.1)] * 3) == 3


def test_a_step_back_inside_the_band_resets_the_streak():
    assert _run(BAND, [_step(0.4), _step(0.4), _step(0.3), _step(0.4), _step(0.4)]) is None


def test_the_env_gap_reads_the_magnitude_of_a_signed_mean():
    config = EarlyStopConfig(logratio_gap=0.005, patience=2)
    assert _run(config, [_step(logratio=-0.004), _step(logratio=-0.006), _step(logratio=-0.01)]) == 3


def test_the_online_gap_reads_trls_mean_absolute_difference():
    config = EarlyStopConfig(logratio_gap=0.005, patience=2)
    steps = [{"entropy": 0.3, SAMPLING_LOGP_GAP_KEY: gap} for gap in (0.004, 0.007, 0.009)]
    assert _run(config, steps, gap_key=SAMPLING_LOGP_GAP_KEY) == 3


def test_a_run_whose_every_update_is_skipped_stops():
    config = EarlyStopConfig(on_skipped_updates=True, patience=3)
    assert _run(config, [_step(skipped=1.0), _step(skipped=0.5), _step(skipped=1.0), _step(skipped=1.0)]) is None
    assert _run(config, [_step(skipped=1.0)] * 3) == 3


def test_eval_logs_never_count():
    evals = [{"eval_entropy": 0.9, "eval_outcome/solve_rate": 0.1}] * 5
    assert _run(BAND, evals) is None


def test_conditions_keep_separate_streaks():
    """Entropy and the gap alternate out of range: neither holds two steps in a row."""
    config = EarlyStopConfig(entropy_band=(0.15, 0.35), logratio_gap=0.005, patience=2)
    assert _run(config, [_step(0.4), _step(logratio=-0.01), _step(0.4), _step(logratio=-0.01)]) is None


def test_the_env_config_adds_the_breaker_condition_to_the_shared_ones():
    assert not AsyncTrainingConfig().build_early_stop().active
    config = AsyncTrainingConfig(
        early_stop_entropy_band=[0.15, 0.35],
        early_stop_on_skipped_updates=True,
        skip_update_masked_frac=0.4,
        early_stop_patience=4,
    ).build_early_stop()
    assert config == EarlyStopConfig(entropy_band=(0.15, 0.35), on_skipped_updates=True, patience=4)


def test_online_args_carry_the_shared_conditions_and_no_breaker_one():
    """Every shared field reaches the config at a non-default value, so a dropped one cannot pass by
    matching its default; the breaker condition is env-only, since only that trainer skips updates."""
    assert "early_stop_on_skipped_updates" not in {f.name for f in dataclasses.fields(RLVROnlineGRPOScriptArguments)}
    args = RLVROnlineGRPOScriptArguments(
        early_stop_entropy_band=[0.2, 0.6], early_stop_logratio_gap=0.01, early_stop_patience=5
    )
    assert args.build_early_stop() == EarlyStopConfig(entropy_band=(0.2, 0.6), logratio_gap=0.01, patience=5)


def test_the_online_script_hands_both_knobs_to_the_trainer():
    """The shared knobs parse on the online args; a script that drops them from the trainer call leaves
    a run that looks configured and trains without them."""
    tree = ast.parse((REPO_ROOT / "scripts/training/online_grpo/rlvr.py").read_text())
    call = next(
        node for node in ast.walk(tree) if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "trainer_cls"
    )
    passed = {kw.arg: ast.unparse(kw.value) for kw in call.keywords if kw.arg}
    assert passed["balance_token_mass"] == "args.balance_token_mass"
    assert passed["early_stop"] == "args.build_early_stop()"


def test_a_gap_the_run_never_logs_is_refused_and_no_condition_attaches_nothing():
    with pytest.raises(ValueError, match="early_stop_logratio_gap"):
        build_early_stop_callback(EarlyStopConfig(logratio_gap=0.01), gap_key=SAMPLING_LOGP_GAP_KEY, gap_logged=False)
    assert build_early_stop_callback(EarlyStopConfig(), gap_key=SAMPLING_LOGP_GAP_KEY, gap_logged=False) is None
    callback = build_early_stop_callback(
        EarlyStopConfig(logratio_gap=0.01), gap_key=SAMPLING_LOGP_GAP_KEY, gap_logged=True
    )
    assert callback is not None and callback.gap_key == SAMPLING_LOGP_GAP_KEY


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"early_stop_entropy_band": [0.35, 0.15]}, "early_stop_entropy_band"),
        ({"early_stop_entropy_band": [0.15]}, "early_stop_entropy_band"),
        ({"early_stop_logratio_gap": -0.01}, "early_stop_logratio_gap"),
        ({"early_stop_patience": 0}, "early_stop_patience"),
        ({"early_stop_on_skipped_updates": True}, "skip_update_masked_frac"),
        ({"early_stop_patience": 5}, "no early-stop condition"),
    ],
)
def test_a_condition_that_could_never_or_always_fire_is_refused(config, message):
    with pytest.raises(ValueError, match=message):
        AsyncTrainingConfig(**config)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
