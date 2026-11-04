#!/usr/bin/env python
"""The ``head_only`` gate: the baseline a resume keeps, either metric tripping, the checkpoint the verdict names, one
verdict on every rank, the last step's evaluation it arms, and the exit a failed fit takes."""

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from transformers import TrainerControl

from halo.callbacks import mtp as mtp_module
from halo.callbacks.mtp import MtpAcceptanceGate
from halo.entrypoints import script
from halo.entrypoints.script import EarlyStopExit, run_trainer
from tests.common.gloo import run_gloo_ranks

_OK_TV, _OK_GREEDY = 0.8, 0.7
_WORLD = 2


def _reading(step: int, tv: float = _OK_TV, greedy: float = _OK_GREEDY, positions: int = 40) -> dict:
    return {
        "step": step,
        "eval_mtp_acceptance_tv": tv,
        "eval_mtp_acceptance_greedy": greedy,
        "eval_mtp_scored_positions": positions,
    }


def _args(**fields) -> SimpleNamespace:
    return SimpleNamespace(**{"output_dir": "/runs/fit", "save_strategy": "steps", **fields})


def _state(*history: dict, global_step: int = 150) -> SimpleNamespace:
    return SimpleNamespace(log_history=list(history), global_step=global_step, max_steps=global_step)


def _verdict(*history: dict, gain: float = 0.0, args=None) -> MtpAcceptanceGate:
    gate = MtpAcceptanceGate(gain)
    gate.on_train_end(args or _args(), _state(*history), TrainerControl())
    return gate


def test_a_fit_that_gained_on_both_metrics_passes():
    gate = _verdict(_reading(0), _reading(150, tv=0.82, greedy=0.72))
    assert not gate.stopped and gate.stop_reason is None


def test_a_fit_matching_its_baseline_passes_at_no_required_gain():
    assert not _verdict(_reading(0), _reading(150)).stopped


@pytest.mark.parametrize(
    ("final", "named"),
    [
        ({"tv": 0.79}, "eval_mtp_acceptance_tv"),
        ({"greedy": 0.69}, "eval_mtp_acceptance_greedy"),
        ({"tv": float("nan")}, "eval_mtp_acceptance_tv"),
    ],
    ids=["tv", "greedy", "nan"],
)
def test_either_metric_ending_below_its_baseline_fails_the_fit_and_names_the_checkpoint(final, named):
    gate = _verdict(_reading(0), _reading(150, **final))
    assert gate.stopped
    assert named in gate.stop_reason
    assert "/runs/fit/checkpoint-150 holds the rejected head" in gate.stop_reason


def test_the_required_gain_raises_the_bar():
    assert _verdict(_reading(0), _reading(150, tv=0.82, greedy=0.72), gain=0.05).stopped
    assert not _verdict(_reading(0), _reading(150, tv=0.86, greedy=0.76), gain=0.05).stopped


def test_a_resume_keeps_the_baseline_of_the_head_the_fit_started_from():
    """``eval_on_start`` scores the resumed, already fitted head again; judged against that, a fit that kept its gain
    but sagged a little would fail, and one that collapsed from its peak to below the start would pass."""
    history = (_reading(0), _reading(100, tv=0.85, greedy=0.75), _reading(100, tv=0.85, greedy=0.75))
    assert not _verdict(*history, _reading(150, tv=0.84, greedy=0.74)).stopped
    assert _verdict(*history, _reading(150, tv=0.79, greedy=0.74)).stopped


def test_a_run_that_saved_nothing_says_so():
    gate = _verdict(_reading(0), _reading(150, tv=0.1), args=_args(save_strategy="no"))
    assert gate.stopped and "saved no checkpoint" in gate.stop_reason


def test_no_reading_cannot_pass():
    gate = _verdict({"step": 150, "loss": 1.0})
    assert gate.stopped and "found no eval_mtp_acceptance_tv" in gate.stop_reason


def test_a_held_out_split_that_scored_no_position_cannot_pass():
    gate = _verdict(_reading(0, positions=0), _reading(150, positions=0))
    assert gate.stopped and "scored no position" in gate.stop_reason


def test_the_gate_scores_the_last_step_whatever_eval_steps_says():
    gate = MtpAcceptanceGate(0.0)
    control = TrainerControl()
    gate.on_step_end(_args(), _state(), control)
    assert not control.should_evaluate
    control.should_training_stop = True
    gate.on_step_end(_args(), _state(), control)
    assert control.should_evaluate


@pytest.mark.parametrize(
    ("learning_rate", "warmup", "warned"),
    [(1e-5, 20, False), (1e-4, 20, True), (1e-5, 0, True)],
    ids=["measured", "learning_rate", "warmup"],
)
def test_a_schedule_off_the_measured_one_warns_with_the_measurement(learning_rate, warmup, warned, caplog):
    args = _args(learning_rate=learning_rate, get_warmup_steps=lambda _steps: warmup)
    with caplog.at_level(logging.WARNING, logger=mtp_module.__name__):
        MtpAcceptanceGate(0.0).on_train_begin(args, _state(), TrainerControl())
    assert bool(caplog.records) == warned
    if warned:
        assert "0.8188 to 0.7361" in caplog.text


def _judge_on_rank(rank: int, root: str) -> None:
    """Rank 1's own log would pass; rank 0's verdict is the one every rank exits on."""
    final = 0.1 if rank == 0 else 0.9
    gate = _verdict(_reading(0), _reading(150, tv=final))
    (Path(root) / f"rank_{rank}.json").write_text(json.dumps({"stopped": gate.stopped, "reason": gate.stop_reason}))


def test_every_rank_takes_one_verdict(tmp_path):
    run_gloo_ranks(_judge_on_rank, _WORLD, str(tmp_path))
    first, second = (json.loads((tmp_path / f"rank_{rank}.json").read_text()) for rank in range(_WORLD))
    assert first == second and first["stopped"]


class _Trainer:
    def __init__(self, callback):
        self.callback_handler = SimpleNamespace(callbacks=[callback])
        self.state = _state()
        self.model = torch.nn.Module()
        self.cleaned_up = False

    def train(self, resume_from_checkpoint=None):
        pass

    def cleanup_ep(self):
        self.cleaned_up = True


def _context() -> SimpleNamespace:
    config = SimpleNamespace(
        mode_string="",
        has_ep_group=False,
        is_cp_mode=False,
        is_tp_mode=False,
        is_expert_tp_mode=False,
        is_pp_mode=False,
        data_parallel_size=1,
    )
    return SimpleNamespace(parallelism_config=config, mode_suffix="standard", resume_checkpoint=None)


def test_a_failed_fit_exits_after_the_ep_cleanup_saying_why(monkeypatch):
    monkeypatch.setattr(script, "reorder_integration_callbacks_last", lambda trainer: None)
    gate = _verdict(_reading(0), _reading(150, tv=0.1))
    trainer = _Trainer(gate)
    with pytest.raises(EarlyStopExit, match="failed its acceptance gate") as raised:
        run_trainer(trainer, _context(), method_name="SFT")
    assert trainer.cleaned_up
    assert raised.value.code == gate.stop_reason


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
