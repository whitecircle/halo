#!/usr/bin/env python
"""A resume with an MTP head, through the real entry script: a run stopped at its middle save and resumed writes the same
weights as one that never stopped, on one process and on two gloo ranks, every family's head included. A resume that
turns the head on, or widens what trains, raises the optimizer gate's own refusal."""

import json
import shutil

import pytest
import torch

from tests.common.mtp_heads import run_entry, saved_tensors, write_entry_corpus, write_entry_run

_MIDDLE, _LAST = 2, 4
_DETERMINISTIC = {"full_determinism": True, "seed": 42, "max_grad_norm": 0, "save_steps": _MIDDLE}
_HEAD = {"mtp_loss_weight": 0.3}
_ARMS = {
    "joint_soft": {"mtp_loss_type": "soft"},
    "detached_hard": {"mtp_train_scope": "detached"},
    "head_only_soft": {"mtp_loss_type": "soft", "mtp_train_scope": "head_only", "mtp_min_acceptance_gain": None},
    "head_only_hard": {"mtp_train_scope": "head_only", "mtp_min_acceptance_gain": None},
}


# The joint recipe's masters at one rank per group, where its experts cannot share a group with fp32 params. fp32 experts
# take the GPU rows, as a CPU row's per-expert loop reads them at the forward's dtype, head or no head.
_FP32_MASTERS = {"bf16": True, "fp32_non_ep_params": True, "fp32_router": True, "fsdp_shard_ep1_experts": False}
_FP32_MASTER_KEYS = {"qwen3_5_moe_text": "mtp.fc.weight", "glm4_moe_lite": "model.layers.2.eh_proj.weight"}


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    return write_entry_corpus(tmp_path_factory.mktemp("corpus") / "rows")


def _trained(name: str, target, corpus: str, *, ranks: int, **fields) -> None:
    """A run to the last step, then its last save removed and the run resumed from the middle one."""
    done = run_entry(write_entry_run(name, target, corpus, steps=_LAST, **_DETERMINISTIC, **fields), ranks=ranks)
    assert done.returncode == 0, (done.stderr + done.stdout)[-3000:]


def _losses(checkpoint) -> list[float]:
    state = json.loads((checkpoint / "trainer_state.json").read_text())
    return [entry["loss"] for entry in state["log_history"] if "loss" in entry]


def _assert_resume_replays(name: str, target, corpus: str, *, ranks: int, **fields) -> None:
    _trained(name, target, corpus, ranks=ranks, **fields)
    run = target / "run"
    straight = saved_tensors(run / f"checkpoint-{_LAST}")
    straight_losses = _losses(run / f"checkpoint-{_LAST}")
    shutil.rmtree(run / f"checkpoint-{_LAST}")
    _trained(name, target, corpus, ranks=ranks, resume_from_checkpoint=True, **fields)
    resumed = saved_tensors(run / f"checkpoint-{_LAST}")
    assert set(resumed) == set(straight)
    differing = [key for key in straight if not torch.equal(straight[key], resumed[key])]
    assert not differing, f"{len(differing)} tensor(s) left the uninterrupted run's bytes: {differing[:4]}"
    assert _losses(run / f"checkpoint-{_LAST}") == straight_losses


@pytest.mark.parametrize("arm", ["joint_soft", "detached_hard", "head_only_soft"])
@pytest.mark.parametrize("name", ["qwen3_5_moe_text", "glm4_moe_lite"])
def test_one_process_replays_every_scope_on_either_placement(name, arm, tmp_path, corpus):
    _assert_resume_replays(name, tmp_path, corpus, ranks=1, **_HEAD, **_ARMS[arm])


@pytest.mark.parametrize("name", ["qwen3_5", "qwen3_next", "qwen3_5_moe"])
def test_one_process_replays_the_other_families(name, tmp_path, corpus):
    _assert_resume_replays(name, tmp_path, corpus, ranks=1, **_HEAD, **_ARMS["joint_soft"])


@pytest.mark.parametrize("arm", ["joint_soft", "head_only_hard"])
@pytest.mark.parametrize("name", ["qwen3_5_moe_text", "glm4_moe_lite", "qwen3_5"])
def test_two_ranks_replay_through_the_shard_groups(name, arm, tmp_path, corpus):
    """FSDP2's restore over the head's own group, and under ``head_only`` over a trunk the wrap left out."""
    _assert_resume_replays(name, tmp_path, corpus, ranks=2, **_HEAD, **_ARMS[arm])


@pytest.mark.parametrize("arm", ["joint_soft", "head_only_hard"])
@pytest.mark.parametrize("name", ["qwen3_5_moe_text", "glm4_moe_lite", "qwen3_5"])
def test_two_ranks_replay_over_a_base_load_the_restore_overwrites(name, arm, tmp_path, corpus):
    """No EP layer in the run, so a resume builds from the base and FSDP2's restore writes the head and the trunk."""
    _assert_resume_replays(name, tmp_path, corpus, ranks=2, use_grouped_gemm=False, **_HEAD, **_ARMS[arm])


@pytest.mark.parametrize("name", ["qwen3_5_moe_text", "glm4_moe_lite"])
def test_the_fp32_masters_a_joint_recipe_keeps_replay(name, tmp_path, corpus):
    """Stored at fp32 and computed at bf16 by the shard groups' cast, the head's masters come back from the checkpoint as
    the trunk's do."""
    _assert_resume_replays(name, tmp_path, corpus, ranks=2, **_FP32_MASTERS, **_HEAD, **_ARMS["joint_soft"])
    saved = saved_tensors(tmp_path / "run" / f"checkpoint-{_LAST}")
    assert saved[_FP32_MASTER_KEYS[name]].dtype == torch.float32


def _resumed_under(name: str, target, corpus: str, *, first: dict, second: dict):
    """Two ranks, so the optimizer state is the per-rank shards the gate reads."""
    done = run_entry(write_entry_run(name, target, corpus, steps=_MIDDLE, **_DETERMINISTIC, **first), ranks=2)
    assert done.returncode == 0, (done.stderr + done.stdout)[-3000:]
    return run_entry(
        write_entry_run(name, target, corpus, steps=_LAST, resume_from_checkpoint=True, **_DETERMINISTIC, **second),
        ranks=2,
    )


@pytest.mark.parametrize(
    ("name", "first", "second"),
    [
        (
            "qwen3_5",
            {"mtp_head_source": None, "use_grouped_gemm": False},
            {**_HEAD, **_ARMS["joint_soft"], "use_grouped_gemm": False},
        ),
        ("qwen3_5_moe_text", {**_HEAD, **_ARMS["head_only_soft"]}, {**_HEAD, **_ARMS["joint_soft"]}),
    ],
    ids=["off_to_on_over_a_base_load", "head_only_to_joint"],
)
def test_a_resume_widening_what_trains_raises_the_optimizer_gate(name, first, second, tmp_path, corpus):
    done = _resumed_under(name, tmp_path, corpus, first=first, second=second)
    assert done.returncode != 0
    assert "have no saved optimizer state" in done.stderr + done.stdout, (done.stderr + done.stdout)[-3000:]


def test_a_resume_building_from_a_checkpoint_with_no_head_refuses_before_any_step(tmp_path, corpus):
    """Its model is built from the checkpoint, so the load finds no head first. The remedy is a new run, since a resume
    ignores ``mtp_head_source``."""
    done = _resumed_under(
        "qwen3_5_moe_text", tmp_path, corpus, first={"mtp_head_source": None}, second={**_HEAD, **_ARMS["joint_soft"]}
    )
    assert done.returncode != 0
    assert "a head cannot join a run midway" in done.stderr + done.stdout, (done.stderr + done.stdout)[-3000:]


def test_a_resume_narrowing_to_the_head_restores_its_moments(tmp_path, corpus):
    """``joint`` to ``head_only`` passes that gate: the head's moments are in the shards, and the trunk's go unread."""
    done = _resumed_under(
        "qwen3_5_moe_text",
        tmp_path,
        corpus,
        first={**_HEAD, **_ARMS["joint_soft"]},
        second={**_HEAD, **_ARMS["head_only_soft"]},
    )
    assert done.returncode == 0, (done.stderr + done.stdout)[-3000:]
    run = tmp_path / "run"
    resumed, middle = saved_tensors(run / f"checkpoint-{_LAST}"), saved_tensors(run / f"checkpoint-{_MIDDLE}")
    moved = [key for key in middle if not torch.equal(middle[key], resumed[key])]
    assert moved and all(key.startswith("mtp.") for key in moved), moved[:4]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
