#!/usr/bin/env python
"""The real SFT entry script on CPU with a head: its knobs reach the load, the term reaches the loss and the save carries
the head, on both placements. The one patch is the loader's device map, which names a GPU."""

import json
import subprocess
from pathlib import Path

import pytest

from tests.common.mtp_heads import moved_keys, run_entry, saved_tensors, write_entry_corpus, write_entry_run

_STEPS = 2
_HEAD_PREFIXES = {
    "qwen3_5_moe_text": "mtp.",
    "glm4_moe_lite": "model.layers.2.",
    "nemotron_h": "mtp.",
    "nemotron_h_no_latent": "mtp.",
    "nemotron_h_omni": "language_model.mtp.",
}


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    return write_entry_corpus(tmp_path_factory.mktemp("corpus") / "rows")


def _run(name: str, tmp_path, corpus: str, **fields) -> tuple[subprocess.CompletedProcess, str]:
    done = run_entry(write_entry_run(name, tmp_path, corpus, steps=_STEPS, **fields))
    return done, f"{tmp_path}/run/checkpoint-{_STEPS}"


@pytest.mark.parametrize("name", sorted(_HEAD_PREFIXES))
def test_a_joint_run_trains_the_head_and_the_trunk_and_saves_the_head(name, tmp_path, corpus):
    done, checkpoint = _run(name, tmp_path, corpus, mtp_loss_weight=0.3, mtp_loss_type="soft")
    assert done.returncode == 0, done.stderr[-2000:]
    assert "MTP head attached" in done.stderr + done.stdout
    saved = saved_tensors(checkpoint)
    stored = saved_tensors(str(tmp_path / "tail"))
    head = {key: value for key, value in saved.items() if key.startswith(_HEAD_PREFIXES[name])}
    assert head and moved_keys(head, stored), "the head came back unchanged"
    base = saved_tensors(str(tmp_path / "base"))
    trunk = {key: value for key, value in saved.items() if not key.startswith(_HEAD_PREFIXES[name])}
    assert moved_keys(trunk, base), "the trunk took no step"


@pytest.mark.parametrize("name", sorted(_HEAD_PREFIXES))
def test_a_head_only_run_moves_the_head_alone(name, tmp_path, corpus):
    done, checkpoint = _run(
        name,
        tmp_path,
        corpus,
        mtp_loss_weight=1.0,
        mtp_loss_type="soft",
        mtp_train_scope="head_only",
        mtp_min_acceptance_gain=None,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    saved = saved_tensors(checkpoint)
    head = {key: value for key, value in saved.items() if key.startswith(_HEAD_PREFIXES[name])}
    assert moved_keys(head, saved_tensors(str(tmp_path / "tail"))), "the head came back unchanged"
    trunk = {key: value for key, value in saved.items() if not key.startswith(_HEAD_PREFIXES[name])}
    assert not moved_keys(trunk, saved_tensors(str(tmp_path / "base")))


def test_a_refused_combination_fails_before_the_load(tmp_path, corpus):
    done, _ = _run(
        "qwen3_5_moe_text",
        tmp_path,
        corpus,
        mtp_loss_weight=1.0,
        mtp_train_scope="head_only",
        gradient_checkpointing=True,
    )
    assert done.returncode != 0
    assert "gradient_checkpointing off" in done.stderr + done.stdout
    assert "MTP head attached" not in done.stderr + done.stdout


def test_a_head_the_freeze_patterns_froze_whole_refuses_at_the_trainer(tmp_path, corpus):
    done, _ = _run("qwen3_5_moe_text", tmp_path, corpus, mtp_loss_weight=0.3, freeze_layers_patterns=["mtp.*"])
    assert done.returncode != 0
    assert "froze every parameter" in done.stderr + done.stdout, (done.stderr + done.stdout)[-2000:]


_GATED = {
    "mtp_loss_weight": 0.3,
    "mtp_loss_type": "soft",
    "mtp_train_scope": "head_only",
    "eval_strategy": "steps",
    "eval_steps": 100,
    "per_device_eval_batch_size": 2,
}


def _eval_readings(checkpoint: str) -> list[dict]:
    history = json.loads((Path(checkpoint) / "trainer_state.json").read_text())["log_history"]
    return [entry for entry in history if "eval_mtp_acceptance_tv" in entry]


@pytest.mark.parametrize("name", sorted(_HEAD_PREFIXES))
def test_a_fit_within_its_gate_passes_with_a_reading_before_the_first_step_and_at_the_last(name, tmp_path, corpus):
    done, checkpoint = _run(name, tmp_path, corpus, **_GATED, mtp_min_acceptance_gain=-1.0)
    assert done.returncode == 0, done.stderr[-2000:]
    readings = _eval_readings(checkpoint)
    assert [entry["step"] for entry in readings] == [0, _STEPS]
    for entry in readings:
        assert entry["eval_mtp_scored_positions"] > 0
        assert 0 < entry["eval_mtp_acceptance_tv"] <= 1


@pytest.mark.parametrize("name", sorted(_HEAD_PREFIXES))
def test_a_fit_under_its_gate_exits_non_zero_naming_the_checkpoint_that_holds_the_head(name, tmp_path, corpus):
    done, checkpoint = _run(name, tmp_path, corpus, **_GATED, mtp_min_acceptance_gain=2.0)
    assert done.returncode != 0
    assert "failed its acceptance gate" in done.stderr + done.stdout
    assert f"{checkpoint} holds the rejected head" in done.stderr + done.stdout
    assert (Path(checkpoint) / "trainer_state.json").is_file(), "the last save the verdict names never completed"


def test_a_resumed_fit_judges_against_the_reading_before_its_first_step(tmp_path, corpus):
    done, checkpoint = _run("qwen3_5_moe_text", tmp_path, corpus, **_GATED, mtp_min_acceptance_gain=-1.0)
    assert done.returncode == 0, done.stderr[-2000:]
    config = write_entry_run(
        "qwen3_5_moe_text",
        tmp_path,
        corpus,
        steps=_STEPS + 1,
        resume_from_checkpoint=True,
        **_GATED,
        mtp_min_acceptance_gain=-1.0,
    )
    done = run_entry(config)
    assert done.returncode == 0, done.stderr[-2000:]
    readings = _eval_readings(f"{tmp_path}/run/checkpoint-{_STEPS + 1}")
    assert [entry["step"] for entry in readings] == [0, _STEPS, _STEPS, _STEPS + 1]


@pytest.mark.parametrize("name", sorted(_HEAD_PREFIXES))
def test_a_soft_fit_raises_its_acceptance_and_an_oversized_learning_rate_trips_the_gate(name, tmp_path, corpus):
    """The quantity the gate reads, moved by a real fit either way. The gate is off for the first, as a tiny head's greedy
    acceptance sits at chance, where a step moves it either way."""
    (tmp_path / "fit").mkdir()
    (tmp_path / "oversized").mkdir()
    fields = {**_GATED, "eval_steps": _STEPS}
    done, checkpoint = _run(name, tmp_path / "fit", corpus, **fields, mtp_min_acceptance_gain=None, eval_on_start=True)
    assert done.returncode == 0, done.stderr[-2000:]
    first, last = _eval_readings(checkpoint)
    assert last["eval_mtp_acceptance_tv"] > first["eval_mtp_acceptance_tv"]

    done, checkpoint = _run(name, tmp_path / "oversized", corpus, **_GATED, learning_rate=100.0)
    assert done.returncode != 0
    assert "eval_mtp_acceptance_tv" in done.stderr + done.stdout
    assert f"{checkpoint} holds the rejected head" in done.stderr + done.stdout


def test_a_joint_run_logs_acceptance_at_its_evaluations_with_no_gate(tmp_path, corpus):
    done, checkpoint = _run(
        "qwen3_5_moe_text",
        tmp_path,
        corpus,
        mtp_loss_weight=0.3,
        eval_strategy="steps",
        eval_steps=_STEPS,
        per_device_eval_batch_size=2,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert [entry["step"] for entry in _eval_readings(checkpoint)] == [_STEPS]


def test_a_run_with_no_weight_builds_no_head(tmp_path, corpus):
    done, checkpoint = _run("qwen3_5_moe_text", tmp_path, corpus, mtp_head_source=None)
    assert done.returncode == 0, done.stderr[-2000:]
    assert not any(key.startswith("mtp.") for key in saved_tensors(checkpoint))
    state = json.loads((tmp_path / "run" / f"checkpoint-{_STEPS}" / "trainer_state.json").read_text())
    assert state["global_step"] == _STEPS


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
