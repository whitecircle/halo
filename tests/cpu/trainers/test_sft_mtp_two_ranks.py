#!/usr/bin/env python
"""The real SFT entry script on two gloo ranks, through the trainer's own FSDP2 wrap with a head. With the clip off a
``detached`` run's trunk comes out byte-equal to a run with no head and its ``ntp_ce`` follows that run's loss, while a
``joint`` run's does not."""

import json

import pytest

from tests.common.mtp_heads import moved_keys, run_entry, saved_tensors, write_entry_corpus, write_entry_run

_RANKS = 2
_STEPS = 3
_NO_CLIP = {"max_grad_norm": 0, "full_determinism": True, "seed": 42}
_HEAD_PREFIXES = {
    "qwen3_5_moe_text": "mtp.",
    "glm4_moe_lite": "model.layers.2.",
    "qwen3_5": "mtp.",
    "nemotron_h": "mtp.",
    "nemotron_h_omni": "language_model.mtp.",
}
# A detached run's ntp_ce against a run with no head: the float sum over a rank's targets. A joint run's sits past it
# by the third step.
_SAME_CURVE_BAR = 1e-6
_DIFFERENT_CURVE_BAR = 1e-4


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    return write_entry_corpus(tmp_path_factory.mktemp("corpus") / "rows")


def _run(name: str, target, corpus: str, **fields) -> tuple[list[dict], dict]:
    done = run_entry(write_entry_run(name, target, corpus, steps=_STEPS, **fields), ranks=_RANKS)
    assert done.returncode == 0, (done.stderr + done.stdout)[-3000:]
    assert "FSDP2 (SHARD_GRAD_OP)" in done.stdout + done.stderr or "FSDP2 (FULL_SHARD)" in done.stdout + done.stderr
    checkpoint = target / "run" / f"checkpoint-{_STEPS}"
    history = json.loads((checkpoint / "trainer_state.json").read_text())["log_history"]
    return [entry for entry in history if "loss" in entry], saved_tensors(checkpoint)


def _trunk(saved: dict, name: str) -> dict:
    return {key: value for key, value in saved.items() if not key.startswith(_HEAD_PREFIXES[name])}


def _head(saved: dict, name: str) -> dict:
    return {key: value for key, value in saved.items() if key.startswith(_HEAD_PREFIXES[name])}


def _curve(history: list[dict], key: str) -> list[float]:
    return [float(entry[key]) for entry in history]


def _largest_gap(first: list[float], second: list[float]) -> float:
    return max(abs(a - b) / abs(b) for a, b in zip(first, second, strict=True))


@pytest.mark.parametrize("loss_type", ["hard", "soft"])
@pytest.mark.parametrize("name", ["qwen3_5_moe_text", "glm4_moe_lite", "nemotron_h"])
def test_detached_leaves_the_trunk_byte_equal_to_a_run_with_no_head_and_joint_does_not(
    name, loss_type, tmp_path, corpus
):
    (tmp_path / "off").mkdir()
    (tmp_path / "detached").mkdir()
    (tmp_path / "joint").mkdir()
    off_history, off = _run(name, tmp_path / "off", corpus, mtp_head_source=None, **_NO_CLIP)
    assert not {"ntp_ce", "mtp_ce"} & set(off_history[0])
    assert not _head(off, name), "a run with no head saved one"

    detached_history, detached = _run(
        name,
        tmp_path / "detached",
        corpus,
        mtp_loss_weight=0.3,
        mtp_loss_type=loss_type,
        mtp_train_scope="detached",
        **_NO_CLIP,
    )
    assert _largest_gap(_curve(detached_history, "ntp_ce"), _curve(off_history, "loss")) < _SAME_CURVE_BAR
    assert not moved_keys(_trunk(detached, name), off), "the detached term reached the trunk"
    tail = saved_tensors(tmp_path / "detached" / "tail")
    assert moved_keys(_head(detached, name), tail), "the head took no step"
    assert all(float(value) > 0 for value in _curve(detached_history, "mtp_ce"))

    joint_history, joint = _run(
        name, tmp_path / "joint", corpus, mtp_loss_weight=0.3, mtp_loss_type=loss_type, **_NO_CLIP
    )
    assert moved_keys(_trunk(joint, name), off), "the joint term never reached the trunk"
    assert _largest_gap(_curve(joint_history, "ntp_ce"), _curve(off_history, "loss")) > _DIFFERENT_CURVE_BAR
    # What HF logs for this rank's loss is both terms, each over one normalizer.
    first = joint_history[0]
    assert float(first["loss"]) > float(first["ntp_ce"]) + 0.2 * float(first["mtp_ce"])


_GATE = {
    "mtp_loss_weight": 0.3,
    "mtp_loss_type": "soft",
    "mtp_train_scope": "head_only",
    "eval_strategy": "steps",
    "eval_steps": 100,
    "per_device_eval_batch_size": 1,
}


def _gated_run(name: str, target, corpus: str, *, gain: float, **fields):
    return run_entry(
        write_entry_run(name, target, corpus, steps=_STEPS, **_GATE, mtp_min_acceptance_gain=gain, **fields),
        ranks=_RANKS,
    )


@pytest.mark.parametrize("name", ["qwen3_5_moe_text", "glm4_moe_lite", "nemotron_h"])
def test_the_gate_reads_a_world_of_held_out_rows_through_the_heads_group(name, tmp_path, corpus):
    done = _gated_run(name, tmp_path, corpus, gain=-1.0)
    assert done.returncode == 0, (done.stderr + done.stdout)[-3000:]
    checkpoint = tmp_path / "run" / f"checkpoint-{_STEPS}"
    history = json.loads((checkpoint / "trainer_state.json").read_text())["log_history"]
    readings = [entry for entry in history if "eval_mtp_acceptance_tv" in entry]
    assert [entry["step"] for entry in readings] == [0, _STEPS]
    # Each held-out row once over the world: what one process scores over the whole split.
    (tmp_path / "one").mkdir()
    done = run_entry(
        write_entry_run(name, tmp_path / "one", corpus, steps=_STEPS, **_GATE, mtp_min_acceptance_gain=-1.0)
    )
    assert done.returncode == 0, (done.stderr + done.stdout)[-3000:]
    one_process = json.loads((tmp_path / "one" / "run" / f"checkpoint-{_STEPS}" / "trainer_state.json").read_text())
    expected = next(e for e in one_process["log_history"] if "eval_mtp_scored_positions" in e)
    assert expected["eval_mtp_scored_positions"] > 0
    assert all(entry["eval_mtp_scored_positions"] == expected["eval_mtp_scored_positions"] for entry in readings)


@pytest.mark.parametrize("name", ["qwen3_5_moe_text", "glm4_moe_lite", "nemotron_h"])
def test_a_failed_fit_exits_every_rank_through_the_teardown_naming_the_checkpoint(name, tmp_path, corpus):
    done = _gated_run(name, tmp_path, corpus, gain=2.0, fsdp_reshard_after_forward=True)
    assert done.returncode != 0
    text = done.stderr + done.stdout
    assert f"checkpoint-{_STEPS} holds the rejected head" in text, text[-3000:]
    assert (tmp_path / "run" / f"checkpoint-{_STEPS}" / "trainer_state.json").is_file()


@pytest.mark.parametrize("name", ["qwen3_5_moe_text", "glm4_moe_lite", "nemotron_h"])
def test_stochastic_rounding_leaves_a_detached_trunk_byte_equal_at_bf16(name, tmp_path, corpus):
    """Its seeds key on step and name, so the head's parameters shift no trunk parameter's draw; and the split survives
    the forward's fp32 output wrap, which rebuilds a model output from its keys."""
    bf16 = {**_NO_CLIP, "bf16": True}
    (tmp_path / "off").mkdir()
    (tmp_path / "detached").mkdir()
    _, off = _run(name, tmp_path / "off", corpus, mtp_head_source=None, **bf16)
    history, detached = _run(
        name,
        tmp_path / "detached",
        corpus,
        mtp_loss_weight=0.3,
        mtp_loss_type="soft",
        mtp_train_scope="detached",
        **bf16,
    )
    assert moved_keys(_trunk(off, name), saved_tensors(tmp_path / "off" / "base")), "the trunk took no step"
    assert not moved_keys(_trunk(detached, name), off), "the head changed a trunk parameter's update"
    assert moved_keys(_head(detached, name), saved_tensors(tmp_path / "detached" / "tail")), "the head took no step"
    assert all(value > 0 for value in _curve(history, "mtp_ce"))


@pytest.mark.parametrize("name", sorted(_HEAD_PREFIXES))
def test_head_only_moves_the_head_alone(name, tmp_path, corpus):
    history, saved = _run(
        name,
        tmp_path,
        corpus,
        mtp_loss_weight=1.0,
        mtp_loss_type="soft",
        mtp_train_scope="head_only",
        mtp_min_acceptance_gain=None,
    )
    assert not moved_keys(_trunk(saved, name), saved_tensors(tmp_path / "base")), "a frozen trunk moved"
    assert moved_keys(_head(saved, name), saved_tensors(tmp_path / "tail")), "the head took no step"
    assert all(value > 0 for value in _curve(history, "mtp_ce"))


@pytest.mark.parametrize("zero3", [False, True], ids=["zero2", "zero3"])
@pytest.mark.parametrize("name", ["qwen3_5_moe_text", "glm4_moe_lite", "nemotron_h"])
def test_an_evaluation_between_steps_scores_the_head_and_leaves_training_as_it_was(name, zero3, tmp_path, corpus):
    """Each shard group gathers for the eval forward, the head's too, and what it leaves registered must not reach the
    next step's forward or the optimizer."""
    fields = {
        "mtp_loss_weight": 0.3,
        "mtp_loss_type": "soft",
        "fsdp_reshard_after_forward": zero3,
        "gradient_checkpointing": True,
        **_NO_CLIP,
    }
    (tmp_path / "plain").mkdir()
    (tmp_path / "evaluated").mkdir()
    _, plain = _run(name, tmp_path / "plain", corpus, **fields)
    _, evaluated = _run(
        name,
        tmp_path / "evaluated",
        corpus,
        eval_strategy="steps",
        eval_steps=1,
        per_device_eval_batch_size=1,
        **fields,
    )
    assert set(evaluated) == set(plain)
    assert not moved_keys(evaluated, plain), "an evaluation changed what training wrote"
    assert moved_keys(_head(plain, name), saved_tensors(tmp_path / "plain" / "tail")), "the head took no step"
    state = json.loads((tmp_path / "evaluated" / "run" / f"checkpoint-{_STEPS}" / "trainer_state.json").read_text())
    readings = [entry for entry in state["log_history"] if "eval_mtp_acceptance_tv" in entry]
    assert [entry["step"] for entry in readings] == list(range(1, _STEPS + 1))
    assert all(entry["eval_mtp_scored_positions"] > 0 for entry in readings)


@pytest.mark.parametrize("name", sorted(_HEAD_PREFIXES))
def test_zero3_with_checkpointing_trains_the_head_and_the_trunk(name, tmp_path, corpus):
    history, saved = _run(
        name,
        tmp_path,
        corpus,
        mtp_loss_weight=0.3,
        fsdp_reshard_after_forward=True,
        gradient_checkpointing=True,
    )
    assert moved_keys(_head(saved, name), saved_tensors(tmp_path / "tail")), "the head took no step"
    assert moved_keys(_trunk(saved, name), saved_tensors(tmp_path / "base")), "the trunk took no step"
    assert all(value > 0 for value in _curve(history, "mtp_ce"))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
