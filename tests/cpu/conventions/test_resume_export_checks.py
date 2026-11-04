#!/usr/bin/env python
"""The verdicts the save, exact-resume and export phase grades with fail on the artifact they exist to catch.
``tests.common.resume_export`` runs only on the GPU tiers, so each verdict runs here with passing inputs, then
with its one defect."""

import json
import logging
import math
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    Qwen3Config,
    Qwen3ForSequenceClassification,
    TrainingArguments,
)

import halo.checkpoint.parallel.optimizer as optimizer_store
from halo.checkpoint.format import REFERENCE_LOGPS_FILE, SAFETENSORS_WEIGHTS_FILE
from tests.common.checkpoint_io import save_schedule
from tests.common.pp_trainers import build_pp_args
from tests.common.rank_log import logged_messages
from tests.common.resume_export import (
    EXPORT_CHECKS,
    checkpoint_file_checks,
    export_checks,
    final_weight_checks,
    replay_checks,
    restore_log_checks,
    restore_point_checks,
    run_state_checks,
)
from tests.common.tolerances import TOL

SAVE_STEP = 3
TOTAL_STEPS = 6
CHECKPOINT = "/runs/unbroken/checkpoint-3"
UNBROKEN = [3.0, 2.8, 2.6, 2.5, 2.4, 2.3]
TINY = {
    "vocab_size": 64,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 16,
    "num_labels": 3,
    "pad_token_id": 0,
}
# A trained step's displacement, far above any rounding.
STEP = 1e-2


def _model() -> Qwen3ForSequenceClassification:
    torch.manual_seed(0)
    return Qwen3ForSequenceClassification(Qwen3Config(**TINY))


def _trained(*, head: bool = True, backbone: bool = True) -> Qwen3ForSequenceClassification:
    model = _model()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if (head and name.startswith("score.")) or (backbone and name.startswith("model.")):
                param.add_(STEP)
    return model


def _save(model, path: Path) -> str:
    model.save_pretrained(path)
    return str(path)


@pytest.fixture
def base(tmp_path) -> str:
    return _save(_model(), tmp_path / "base")


def _export(checkpoint: str, base: str, export_class=AutoModelForSequenceClassification) -> dict[str, bool]:
    return export_checks(export_class, checkpoint, base, torch.float32)


def test_save_schedule_saves_only_when_asked():
    assert save_schedule(None) == {"save_strategy": "no"}
    assert save_schedule(SAVE_STEP) == {"save_strategy": "steps", "save_steps": SAVE_STEP}


def test_pp_args_save_nothing_unless_asked(tmp_path):
    assert build_pp_args(TrainingArguments, str(tmp_path)).save_strategy == "no"
    saving = build_pp_args(TrainingArguments, str(tmp_path), save_every=SAVE_STEP)
    assert (saving.save_strategy, saving.save_steps) == ("steps", SAVE_STEP)


def test_export_of_a_trained_checkpoint_passes(tmp_path, base):
    checks = _export(_save(_trained(), tmp_path / "trained"), base)
    assert checks == dict.fromkeys(EXPORT_CHECKS, True)


def test_export_through_the_wrong_head_class_fails(tmp_path, base):
    checks = _export(_save(_trained(), tmp_path / "trained"), base, AutoModelForCausalLM)
    assert not checks["export_loads_every_weight"]
    assert not checks["export_is_the_trained_class"]


def test_export_with_an_untrained_head_fails(tmp_path, base):
    checks = _export(_save(_trained(head=False), tmp_path / "trained"), base)
    assert not checks["export_head_trained"]
    assert checks["export_backbone_trained"]


def test_export_with_an_untrained_backbone_fails(tmp_path, base):
    checks = _export(_save(_trained(backbone=False), tmp_path / "trained"), base)
    assert checks["export_head_trained"]
    assert not checks["export_backbone_trained"]


def test_export_missing_a_weight_fails(tmp_path, base):
    checkpoint = Path(_save(_trained(), tmp_path / "trained"))
    weights = checkpoint / SAFETENSORS_WEIGHTS_FILE
    tensors = load_file(weights)
    del tensors["score.weight"]
    save_file(tensors, weights, metadata={"format": "pt"})
    checks = _export(str(checkpoint), base)
    assert not checks["export_loads_every_weight"]
    assert not checks["export_head_trained"]


def test_export_declaring_another_class_fails(tmp_path, base):
    checkpoint = Path(_save(_trained(), tmp_path / "trained"))
    config = json.loads((checkpoint / "config.json").read_text())
    config["architectures"] = ["Qwen3ForCausalLM"]
    (checkpoint / "config.json").write_text(json.dumps(config))
    assert not _export(str(checkpoint), base)["export_is_the_trained_class"]


def test_export_without_weights_fails_every_check(tmp_path, base):
    (tmp_path / "empty").mkdir()
    assert _export(str(tmp_path / "empty"), base) == dict.fromkeys(EXPORT_CHECKS, False)


def test_checkpoint_file_checks_require_each_sidecar(tmp_path):
    checks = checkpoint_file_checks(str(tmp_path), 2, (REFERENCE_LOGPS_FILE,))
    assert "trainer_state.json" in checks and not checks[REFERENCE_LOGPS_FILE]
    (tmp_path / REFERENCE_LOGPS_FILE).write_bytes(b"")
    assert checkpoint_file_checks(str(tmp_path), 2, (REFERENCE_LOGPS_FILE,))[REFERENCE_LOGPS_FILE]


def test_logged_messages_reads_the_optimizer_logger(caplog):
    caplog.set_level(logging.INFO)
    with logged_messages(optimizer_store.logger) as messages:
        optimizer_store.logger.info(f"✓ Restored per-rank optimizer state from {CHECKPOINT}")
    optimizer_store.logger.info("after the block")
    assert messages == [f"✓ Restored per-rank optimizer state from {CHECKPOINT}"]


def test_restore_log_checks_need_the_restore_and_no_warm_restart():
    restored = f"✓ Restored per-rank optimizer state from {CHECKPOINT}"
    warm = f"No optimizer shards at {CHECKPOINT} — ... Warm restart: optimizer reinitialized from scratch."
    assert restore_log_checks([restored], CHECKPOINT) == {
        "optimizer_restore_logged": True,
        "no_warm_restart_logged": True,
    }
    assert not restore_log_checks([restored.replace("checkpoint-3", "checkpoint-6")], CHECKPOINT)[
        "optimizer_restore_logged"
    ]
    assert not restore_log_checks([restored, warm], CHECKPOINT)["no_warm_restart_logged"]


def _capture(step: int = SAVE_STEP, moment: float = 1.0, sched: int = SAVE_STEP) -> dict:
    """A :class:`RestorePointSnapshot` capture: the step, the scheduler and one parameter's Adam state."""
    state = {"w": {"exp_avg": torch.full((2,), moment), "exp_avg_sq": torch.ones(2), "step": torch.tensor(5.0)}}
    return {"global_step": step, "sched_last_epoch": sched, "optimizer": {"state": state}}


def test_restore_point_checks_pass_on_an_exact_restore():
    assert restore_point_checks(_capture(), _capture(), SAVE_STEP) == {
        "optimizer_restored_bit_exact": True,
        "step_and_schedule_restored": True,
    }


@pytest.mark.parametrize(
    ("saved", "restored", "failing"),
    [
        (_capture(), _capture(moment=0.0), "optimizer_restored_bit_exact"),  # a reinitialized moment
        (_capture(), _capture(sched=0), "step_and_schedule_restored"),  # the LR schedule back at warmup
        (_capture(step=TOTAL_STEPS), _capture(), "step_and_schedule_restored"),  # snapshot of the wrong save
        (_capture(), None, "optimizer_restored_bit_exact"),  # the train-begin hook never fired
    ],
)
def test_restore_point_checks_fail_on_what_was_not_restored(saved, restored, failing):
    assert not restore_point_checks(saved, restored, SAVE_STEP)[failing]


def test_replay_checks_pass_on_an_exact_replay():
    checks, metrics = replay_checks(UNBROKEN, list(UNBROKEN), save_step=SAVE_STEP, total_steps=TOTAL_STEPS)
    assert checks == {"history_restored": True, "losses_replay_unbroken": True}
    assert metrics["resume_loss_max_delta"] == 0.0


@pytest.mark.parametrize(
    ("resumed", "failing"),
    [
        ([*UNBROKEN[:4], UNBROKEN[4] + 2 * TOL.replayed_resume_loss_abs, UNBROKEN[5]], "losses_replay_unbroken"),
        ([*UNBROKEN[:5]], "losses_replay_unbroken"),  # stopped a step early, so its tail is off by one
        ([*UNBROKEN[3:5]], "losses_replay_unbroken"),  # fewer resumed steps logged than the run owes
        ([UNBROKEN[1], *UNBROKEN[1:]], "history_restored"),  # the checkpoint's history not carried over
    ],
)
def test_replay_checks_fail_off_the_unbroken_run(resumed, failing):
    checks, _ = replay_checks(UNBROKEN, resumed, save_step=SAVE_STEP, total_steps=TOTAL_STEPS)
    assert not checks[failing]


def test_run_state_checks_need_every_tensor_bit_for_bit():
    saved = {"training": torch.tensor([-1.5, -2.25]), "evaluation": torch.tensor([-0.5])}
    assert run_state_checks(saved, {key: value.clone() for key, value in saved.items()}) == {
        "run_state_restored": True
    }
    assert not run_state_checks(saved, {**saved, "training": torch.tensor([-1.5, -2.0])})["run_state_restored"]
    assert not run_state_checks(saved, {"training": saved["training"]})["run_state_restored"]
    assert not run_state_checks({}, {})["run_state_restored"]


def test_final_weight_checks_hold_the_resumed_checkpoint_to_the_unbroken_one(tmp_path):
    unbroken = _save(_trained(), tmp_path / "unbroken")
    checks, metrics = final_weight_checks(unbroken, _save(_trained(), tmp_path / "same"))
    assert checks == {"final_weights_match_unbroken": True} and metrics["resume_final_weight_rel_l2"] == 0.0
    assert not final_weight_checks(unbroken, _save(_trained(head=False), tmp_path / "drifted"))[0][
        "final_weights_match_unbroken"
    ]
    (tmp_path / "missing").mkdir()
    checks, metrics = final_weight_checks(unbroken, str(tmp_path / "missing"))
    assert not checks["final_weights_match_unbroken"] and metrics["resume_final_weight_rel_l2"] == math.inf


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
