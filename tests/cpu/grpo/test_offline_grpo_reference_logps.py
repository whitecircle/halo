#!/usr/bin/env python
"""Offline GRPO's raw token references survive checkpoints and cannot drift on resume."""

import datetime
import os
import shutil

import pytest
import torch

import src.trainers.mixins.reference_logps as reference_mod
from src.checkpoint.format import REFERENCE_CACHE_DIR_NAME, REFERENCE_LOGPS_FILE
from src.data.collators.offline_grpo import REF_PER_TOKEN_LOGPS_COLUMN
from tests.common.gloo import run_gloo_ranks
from tests.common.offline_grpo_reference import (
    SETTINGS,
    ReferenceStorageTrainer,
    attach_reference,
    reference_dataset,
    reference_rows,
    restore_reference,
)


def _save_first_split(tmp_path):
    trainer = ReferenceStorageTrainer(tmp_path)
    dataset = reference_dataset()
    assert restore_reference(trainer, dataset, "train") is None
    attached = attach_reference(trainer, dataset, "train", reference_rows(), settings=SETTINGS)
    trainer.save_checkpoint()
    return attached, tmp_path / "checkpoint-1"


def test_raw_ragged_reference_is_saved_and_restored_without_a_second_sweep(tmp_path):
    attached, checkpoint = _save_first_split(tmp_path)
    assert checkpoint.joinpath(REFERENCE_LOGPS_FILE).is_file()
    assert attached[REF_PER_TOKEN_LOGPS_COLUMN] == [[-0.25, -1.5], [-0.75], []]

    saved = torch.load(checkpoint / REFERENCE_LOGPS_FILE, map_location="cpu", weights_only=True)["train"]
    assert saved["num_rows"] == 3
    assert set(saved["token_digests"]) == {"prompt_input_ids", "completion_input_ids"}
    assert saved["settings"] == SETTINGS
    assert torch.equal(saved["lengths"], torch.tensor([2, 1, 0]))
    assert torch.equal(saved["values"], torch.tensor([-0.25, -1.5, -0.75]))

    resumed = ReferenceStorageTrainer(tmp_path, checkpoint=str(checkpoint), step=2)
    unchanged = reference_dataset()
    unchanged._fingerprint = "a-new-process-local-fingerprint"
    restored = restore_reference(resumed, unchanged, "train")
    assert restored[REF_PER_TOKEN_LOGPS_COLUMN] == attached[REF_PER_TOKEN_LOGPS_COLUMN]
    resumed.save_checkpoint()
    next_saved = torch.load(tmp_path / "checkpoint-2" / REFERENCE_LOGPS_FILE, weights_only=True)["train"]
    assert torch.equal(next_saved["values"], saved["values"])


def test_live_reference_state_keeps_one_arrow_column_not_an_extra_flat_token_table(tmp_path):
    trainer = ReferenceStorageTrainer(tmp_path)
    attached = attach_reference(trainer, reference_dataset(), "train", reference_rows(), settings=SETTINGS)
    assert set(trainer._reference_logps_by_split["train"]) == {"num_rows", "token_digests", "settings"}
    assert not trainer._resumed_reference_logps
    assert attached[REF_PER_TOKEN_LOGPS_COLUMN] == [[-0.25, -1.5], [-0.75], []]


def test_train_and_eval_splits_survive_when_resume_uses_only_train(tmp_path):
    trainer = ReferenceStorageTrainer(tmp_path)
    train = reference_dataset()
    eval_set = train.select([2, 0])
    attach_reference(trainer, train, "train", reference_rows(), settings=SETTINGS)
    attach_reference(trainer, eval_set, "eval", [reference_rows()[2], reference_rows()[0]], settings=SETTINGS)
    trainer.save_checkpoint()

    resumed = ReferenceStorageTrainer(tmp_path, checkpoint=str(tmp_path / "checkpoint-1"), step=2)
    assert restore_reference(resumed, train, "train") is not None
    resumed.save_checkpoint()
    carried = torch.load(tmp_path / "checkpoint-2" / REFERENCE_LOGPS_FILE, weights_only=True)
    assert set(carried) == {"train", "eval"}
    assert torch.equal(carried["eval"]["lengths"], torch.tensor([0, 2]))


def test_trained_policy_cannot_rescore_when_sidecar_is_missing(tmp_path):
    checkpoint = tmp_path / "checkpoint-1"
    checkpoint.mkdir()
    resumed = ReferenceStorageTrainer(tmp_path, checkpoint=str(checkpoint))

    with pytest.raises(RuntimeError, match="TRAINED checkpoint policy") as raised:
        restore_reference(resumed, reference_dataset(), "train")
    message = str(raised.value)
    assert f"--output_dir={tmp_path}-reference-recovery" in message, "the refusal must name a runnable recovery"
    assert "previous checkpoint's copy" in message, "the cheapest recovery must stay on offer"
    assert "supply" not in message, "offline GRPO refuses dataset-supplied reference columns"

    fresh_weights = ReferenceStorageTrainer(tmp_path)
    assert restore_reference(fresh_weights, reference_dataset(), "train") is None


@pytest.mark.parametrize("change", ["row_order", "prompt", "completion", "settings"])
def test_trained_policy_refuses_a_sidecar_for_other_tokens_or_settings(tmp_path, change):
    _, checkpoint = _save_first_split(tmp_path)
    dataset = reference_dataset()
    settings = dict(SETTINGS)
    if change == "row_order":
        dataset = dataset.select([2, 1, 0])
    elif change == "prompt":
        dataset = dataset.remove_columns("prompt_input_ids").add_column(
            "prompt_input_ids", [[11, 99], [21, 22, 23], [31]]
        )
    elif change == "completion":
        dataset = dataset.remove_columns("completion_input_ids").add_column(
            "completion_input_ids", [[13, 99], [24], []]
        )
    else:
        settings["max_completion_length"] += 1
    resumed = ReferenceStorageTrainer(tmp_path, checkpoint=str(checkpoint))

    with pytest.raises(ValueError, match="does not belong") as raised:
        restore_reference(resumed, dataset, "train", settings=settings)
    if change in ("prompt", "completion"):
        # Same row lengths, so the token digests are what refuse: the causes must be this trainer's.
        assert "max_prompt_length, max_completion_length or drop_degenerate_groups" in str(raised.value)
        assert "dataset_num_proc" not in str(raised.value)


@pytest.mark.parametrize("damage", ["missing_split", "bad_lengths", "bad_values"])
def test_trained_policy_refuses_missing_or_malformed_scores(tmp_path, damage):
    _, checkpoint = _save_first_split(tmp_path)
    path = checkpoint / REFERENCE_LOGPS_FILE
    saved = torch.load(path, weights_only=True)
    if damage == "missing_split":
        saved.pop("train")
    elif damage == "bad_lengths":
        saved["train"]["lengths"] = torch.tensor([2, 2, 0])
    else:
        saved["train"]["values"] = torch.tensor([-0.25, float("nan"), -0.75])
    torch.save(saved, path)
    resumed = ReferenceStorageTrainer(tmp_path, checkpoint=str(checkpoint))

    with pytest.raises((RuntimeError, ValueError), match="lacks 'train'|does not belong"):
        restore_reference(resumed, reference_dataset(), "train")


def test_a_failed_sidecar_write_keeps_the_previous_checkpoint(tmp_path, monkeypatch):
    previous = tmp_path / "checkpoint-0"
    previous.mkdir()
    (previous / "sentinel").write_text("complete")
    trainer = ReferenceStorageTrainer(tmp_path)
    attach_reference(trainer, reference_dataset(), "train", reference_rows(), settings=SETTINGS)

    def fail_save(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(reference_mod.torch, "save", fail_save)
    with pytest.raises(RuntimeError, match="disk full"):
        trainer.save_checkpoint()
    assert not trainer.rotated
    assert (previous / "sentinel").read_text() == "complete"


def _ranked_restore(rank: int, root: str) -> None:
    # Each rank is a node with its own output directory, holding its own copy of the checkpoint.
    os.environ["LOCAL_RANK"] = "0"
    os.environ["LOCAL_WORLD_SIZE"] = "1"
    output = os.path.join(root, f"rank-{rank}")
    trainer = ReferenceStorageTrainer(output, checkpoint=os.path.join(output, "checkpoint-1"))
    try:
        restore_reference(trainer, reference_dataset(), "train")
        outcome = "NO RAISE"
    except Exception as exc:
        outcome = f"{type(exc).__name__}: {exc}"
    with open(os.path.join(root, f"outcome-{rank}.txt"), "w") as result:
        result.write(outcome)


@pytest.mark.parametrize("finite_damage", [False, True])
def test_one_nodes_corrupt_reference_is_rejected_on_every_rank(tmp_path, finite_damage):
    _, source = _save_first_split(tmp_path / "source")
    for rank in range(2):
        checkpoint = tmp_path / f"rank-{rank}" / "checkpoint-1"
        checkpoint.mkdir(parents=True)
        shutil.copy2(source / REFERENCE_LOGPS_FILE, checkpoint / REFERENCE_LOGPS_FILE)
    bad_file = tmp_path / "rank-1" / "checkpoint-1" / REFERENCE_LOGPS_FILE
    saved = torch.load(bad_file, weights_only=True)
    saved["train"]["values"][0] = -9.0 if finite_damage else float("nan")
    torch.save(saved, bad_file)

    run_gloo_ranks(
        _ranked_restore,
        2,
        str(tmp_path),
        env={"DIST_OUTPUT_SHARED_FILESYSTEM": "0"},
        pg_timeout=datetime.timedelta(seconds=15),
    )
    for rank in range(2):
        outcome = (tmp_path / f"outcome-{rank}.txt").read_text()
        assert "NO RAISE" not in outcome
        assert "reference values differs across ranks" in outcome if finite_damage else "non-finite" in outcome
        if not finite_damage:
            assert "rank 1" in outcome
        assert "timed out" not in outcome


def _ranked_interrupted_save(rank: int, root: str) -> None:
    trainer = ReferenceStorageTrainer(root)
    attach_reference(trainer, reference_dataset(), "train", reference_rows(), settings=SETTINGS)
    patch = pytest.MonkeyPatch()

    def interrupted_save(payload, destination, **save_options):
        with open(destination, "wb") as incomplete:
            incomplete.write(b"partial sidecar")
        raise OSError("reference disk full")

    if rank == 0:
        patch.setattr(reference_mod.torch, "save", interrupted_save)
    try:
        trainer.save_checkpoint()
        outcome = "NO RAISE"
    except Exception as exc:
        outcome = f"{type(exc).__name__}: {exc}"
    finally:
        patch.undo()
    with open(os.path.join(root, f"save-outcome-{rank}.txt"), "w") as result:
        result.write(f"rotated={trainer.rotated} {outcome}")


def test_interrupted_writer_rejects_every_rank_before_checkpoint_rotation(tmp_path):
    run_gloo_ranks(_ranked_interrupted_save, 2, str(tmp_path), pg_timeout=datetime.timedelta(seconds=15))
    for rank in range(2):
        outcome = (tmp_path / f"save-outcome-{rank}.txt").read_text()
        assert "reference disk full" in outcome
        assert "rotated=False" in outcome
        assert "NO RAISE" not in outcome
        assert "timed out" not in outcome
    checkpoint = tmp_path / "checkpoint-1"
    assert not (checkpoint / REFERENCE_LOGPS_FILE).exists()
    assert not list(checkpoint.glob(f".{REFERENCE_LOGPS_FILE}.*"))


def _mapped_files() -> list[str]:
    with open("/proc/self/maps") as maps:
        return [line.split(maxsplit=5)[5].strip() for line in maps if len(line.split(maxsplit=5)) == 6]


def test_a_resume_maps_a_staged_copy_so_rotation_can_remove_the_checkpoint(tmp_path):
    """A resume maps its saved scores for the run. Mapped through the checkpoint's own entry, rotation's
    removal of that checkpoint on NFS leaves a fresh ``.nfs*`` file keeping the directory, which HF's
    newest-mtime rotation then keeps in place of a complete checkpoint."""
    attached, checkpoint = _save_first_split(tmp_path)
    resumed = ReferenceStorageTrainer(tmp_path, checkpoint=str(checkpoint), step=2)
    restored = restore_reference(resumed, reference_dataset(), "train")
    mapped = _mapped_files()
    sidecar = os.path.realpath(checkpoint / REFERENCE_LOGPS_FILE)
    assert not any(path.startswith(sidecar) for path in mapped), "the resume maps the checkpoint's own file"
    assert any(f"/{REFERENCE_CACHE_DIR_NAME}/" in path and path.endswith(REFERENCE_LOGPS_FILE) for path in mapped)
    shutil.rmtree(checkpoint)
    assert restored[REF_PER_TOKEN_LOGPS_COLUMN] == attached[REF_PER_TOKEN_LOGPS_COLUMN]
    resumed.save_checkpoint()
    carried = torch.load(tmp_path / "checkpoint-2" / REFERENCE_LOGPS_FILE, weights_only=True)["train"]
    assert torch.equal(carried["values"], torch.tensor([-0.25, -1.5, -0.75]))


def _ranked_partial_restore(rank: int, root: str) -> None:
    os.environ["LOCAL_RANK"] = "0"
    os.environ["LOCAL_WORLD_SIZE"] = "1"
    output = os.path.join(root, f"rank-{rank}")
    trainer = ReferenceStorageTrainer(output, checkpoint=os.path.join(output, "checkpoint-1"))
    try:
        restore_reference(trainer, reference_dataset(), "train")
        outcome = "NO RAISE"
    except Exception as exc:
        outcome = f"{type(exc).__name__}: {exc}"
    with open(os.path.join(root, f"partial-outcome-{rank}.txt"), "w") as result:
        result.write(outcome)


def test_a_node_without_its_checkpoint_copy_fails_the_resume_on_every_rank(tmp_path):
    """Node-local output: each node's writer stages its own copy, so a node missing it is the torn verdict
    on every rank, not a silent resume from the other node's scores or a hang."""
    _, source = _save_first_split(tmp_path / "source")
    checkpoint = tmp_path / "rank-0" / "checkpoint-1"
    checkpoint.mkdir(parents=True)
    shutil.copy2(source / REFERENCE_LOGPS_FILE, checkpoint / REFERENCE_LOGPS_FILE)
    (tmp_path / "rank-1" / "checkpoint-1").mkdir(parents=True)
    run_gloo_ranks(
        _ranked_partial_restore,
        2,
        str(tmp_path),
        env={"DIST_OUTPUT_SHARED_FILESYSTEM": "0"},
        pg_timeout=datetime.timedelta(seconds=15),
    )
    for rank in range(2):
        outcome = (tmp_path / f"partial-outcome-{rank}.txt").read_text()
        assert "present on some ranks but missing on others" in outcome, outcome


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
