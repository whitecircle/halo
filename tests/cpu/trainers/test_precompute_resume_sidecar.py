#!/usr/bin/env python
"""Precomputed reference log-probs ride every checkpoint and replace the sweep on resume.

TRL runs ``precompute_ref_log_probs`` inside ``__init__`` over ``self.ref_model or self.model``. A
Path-B resume (EP, CP, TP and the default ``use_grouped_gemm``) builds the policy from the
checkpoint before the trainer exists, so with no separate reference that sweep scores the TRAINED
weights: the reference equals the policy and every log-ratio is zero. The mixin persists the swept
columns into each checkpoint (``reference_logps.pt``) and, handed the resume checkpoint at
construction, attaches them in place of the sweep once the saved split's row count, token digests
and reference settings match — or refuses where a sweep would be wrong.

The trainers are the real ``DistributedDPOTrainer`` / ``DistributedKTOTrainer`` over a stub reference
forward whose values say which weights produced them: ``BASE`` on the saving run, ``TRAINED`` on the
resumed one (``tests/common/preference_precompute.py``).

    python tests/cpu/trainers/test_precompute_resume_sidecar.py
"""

import contextlib

import pyarrow as pa
import pytest
import torch
from accelerate import PartialState
from datasets import Dataset, Features, List, Value, concatenate_datasets

import src.trainers.preference.precompute as precompute_mod
from src.checkpoint.format import REFERENCE_LOGPS_FILE
from src.trainers.mixins.checkpointing import CheckpointingMixin
from src.trainers.preference.precompute import PrecomputeRefLogpsRankConsistentMixin, _token_digest
from src.training.script_runner import ScriptRuntime
from tests.common.preference_precompute import (
    BASE,
    N_ROWS,
    REFERENCE_COLUMNS,
    TOKEN_COLUMNS,
    TRAINED,
    TRAINERS,
    column,
    precompute_trainer,
    token_rows,
)

PartialState()


@pytest.fixture(params=sorted(TRAINERS))
def kind(request):
    return request.param


def _save_base_run(kind, checkpoint_dir, splits, **trainer_kwargs) -> dict[str, Dataset]:
    """A fresh run's precompute over ``splits`` (name → dataset), persisted into ``checkpoint_dir``."""
    trainer = precompute_trainer(kind, weights=BASE, **trainer_kwargs)
    prepared = {name: trainer._precompute_ref_logps(dataset, name, 2) for name, dataset in splits.items()}
    trainer._persist_trainer_sidecars(str(checkpoint_dir))
    return prepared


def _resumed(kind, checkpoint_dir, **trainer_kwargs):
    """A trainer resuming the production way, whose own sweep would score the trained weights."""
    return precompute_trainer(
        kind, weights=TRAINED, resume_checkpoint=str(checkpoint_dir), policy_from_checkpoint=True, **trainer_kwargs
    )


def test_the_swept_columns_are_persisted_as_training_reads_them(kind, tmp_path):
    prepared = _save_base_run(kind, tmp_path, {"train": token_rows(kind)})["train"]

    saved = torch.load(tmp_path / REFERENCE_LOGPS_FILE, weights_only=True)
    assert set(saved) == {"train"}
    entry = saved["train"]
    assert entry["num_rows"] == N_ROWS
    assert set(entry["columns"]) == set(REFERENCE_COLUMNS[kind])
    assert set(entry["token_digests"]) == set(TOKEN_COLUMNS[kind])
    for name in REFERENCE_COLUMNS[kind]:
        assert prepared.features[name].dtype == "float32"
        assert torch.equal(entry["columns"][name], column(prepared, name))


def test_a_resume_attaches_the_saved_columns_instead_of_sweeping_the_trained_policy(kind, tmp_path):
    base = _save_base_run(kind, tmp_path, {"train": token_rows(kind)})["train"]
    resumed = _resumed(kind, tmp_path)

    prepared = resumed._precompute_ref_logps(token_rows(kind), "train", 2)

    assert resumed.compute_ref_log_probs.batches == 0, "the resume swept the policy, which holds the trained weights"
    for name in REFERENCE_COLUMNS[kind]:
        assert torch.equal(column(prepared, name), column(base, name))
    # The resumed run's own checkpoints carry the same values on, so a second resume restores them.
    resumed._persist_trainer_sidecars(str(tmp_path / "next"))
    again = torch.load(tmp_path / "next" / REFERENCE_LOGPS_FILE, weights_only=True)["train"]
    first = torch.load(tmp_path / REFERENCE_LOGPS_FILE, weights_only=True)["train"]
    assert again["token_digests"] == first["token_digests"]
    assert again["settings"] == first["settings"]
    for name in REFERENCE_COLUMNS[kind]:
        assert torch.equal(again["columns"][name], first["columns"][name])


def test_a_resumed_dataset_with_a_new_fingerprint_still_restores(kind, tmp_path):
    """The identity is the token content: a fingerprint re-drawn by an unhashable map must not refuse."""
    _save_base_run(kind, tmp_path, {"train": token_rows(kind)})
    dataset = token_rows(kind)
    dataset._fingerprint = "redrawn-per-process"
    resumed = _resumed(kind, tmp_path)

    resumed._precompute_ref_logps(dataset, "train", 2)

    assert resumed.compute_ref_log_probs.batches == 0


def test_a_policy_built_from_the_checkpoint_without_saved_columns_refuses(kind, tmp_path):
    trainer = _resumed(kind, tmp_path)

    with pytest.raises(RuntimeError, match="TRAINED weights as the reference") as raised:
        trainer._precompute_ref_logps(token_rows(kind), "train", 2)
    assert "--max_steps=1" in str(raised.value), "the refusal must name a way to recover"
    assert trainer.compute_ref_log_probs.batches == 0


@pytest.mark.parametrize(
    ("policy_from_checkpoint", "ref_model"),
    [(False, None), (True, torch.nn.Linear(1, 1))],
    ids=["policy_from_base", "separate_reference"],
)
def test_a_resume_sweeps_where_the_reference_weights_are_untrained(kind, tmp_path, policy_from_checkpoint, ref_model):
    """An adapter checkpoint (the policy resumes the base) or a separate reference model: the sweep
    scores untrained weights, so a checkpoint without the sidecar still resumes."""
    trainer = precompute_trainer(
        kind, resume_checkpoint=str(tmp_path), policy_from_checkpoint=policy_from_checkpoint, ref_model=ref_model
    )

    trainer._precompute_ref_logps(token_rows(kind), "train", 2)

    assert trainer.compute_ref_log_probs.batches > 0


@pytest.mark.parametrize(
    ("resumed_rows", "reason"),
    [
        (lambda kind: token_rows(kind, N_ROWS - 1), "saved for 4 rows"),
        (lambda kind: token_rows(kind).select(list(reversed(range(N_ROWS)))), "differ from the saved run's"),
    ],
    ids=["fewer_rows", "reordered_rows"],
)
def test_saved_columns_for_other_rows_refuse(kind, tmp_path, resumed_rows, reason):
    _save_base_run(kind, tmp_path, {"train": token_rows(kind)})
    trainer = _resumed(kind, tmp_path)

    with pytest.raises(ValueError, match=reason):
        trainer._precompute_ref_logps(resumed_rows(kind), "train", 2)
    assert trainer.compute_ref_log_probs.batches == 0


@pytest.mark.parametrize(
    ("kind", "bumped"),
    [("dpo", name) for name in TOKEN_COLUMNS["dpo"]] + [("kto", name) for name in TOKEN_COLUMNS["kto"]],
)
def test_every_column_the_reference_reads_is_in_the_digest(tmp_path, kind, bumped):
    """One changed value in any token-id or label column must refuse the saved split."""
    _save_base_run(kind, tmp_path, {"train": token_rows(kind)})
    trainer = _resumed(kind, tmp_path)

    with pytest.raises(ValueError, match=f"'{bumped}'"):
        trainer._precompute_ref_logps(token_rows(kind, bump=bumped), "train", 2)


@pytest.mark.parametrize(
    ("kind", "changed"),
    [
        ("dpo", {"max_length": 8}),
        ("dpo", {"truncation_mode": "keep_end"}),
        ("dpo", {"ld_alpha": 0.5}),
        ("kto", {"max_length": 8}),
    ],
    ids=["dpo_max_length", "dpo_truncation_mode", "dpo_ld_alpha", "kto_max_length"],
)
def test_changed_reference_settings_refuse(tmp_path, kind, changed):
    """The collator truncates at batch time and ``ld_alpha`` reshapes DPO's sums, so a resume that
    changes either reads other values off the same tokens."""
    _save_base_run(kind, tmp_path, {"train": token_rows(kind)})
    trainer = _resumed(kind, tmp_path)
    for name, value in changed.items():
        if name == "ld_alpha":
            trainer.ld_alpha = value
        else:
            setattr(trainer.args, name, value)

    with pytest.raises(ValueError, match="computed under"):
        trainer._precompute_ref_logps(token_rows(kind), "train", 2)


def test_a_kto_resume_that_needs_the_kl_column_the_save_lacks_refuses(tmp_path):
    """The needed set is derived per run (``calculate_KL``), so a resume under a KL loss cannot be
    served by a save written under a KL-free one."""
    kl_free = token_rows("kto").remove_columns(["KL_completion_ids"])
    _save_base_run("kto", tmp_path, {"train": kl_free}, calculate_kl=False)
    trainer = _resumed("kto", tmp_path, calculate_kl=True)

    with pytest.raises(ValueError, match="ref_KL_logps"):
        trainer._precompute_ref_logps(token_rows("kto"), "train", 2)


def test_a_kto_resume_dropping_the_kl_term_restores_ref_logps(tmp_path):
    """The reverse direction is served: the saved ``ref_logps`` still belongs to these rows, and the
    KL completions the save digested are no longer read."""
    base = _save_base_run("kto", tmp_path, {"train": token_rows("kto")}, calculate_kl=True)["train"]
    trainer = _resumed("kto", tmp_path, calculate_kl=False)

    prepared = trainer._precompute_ref_logps(token_rows("kto").remove_columns(["KL_completion_ids"]), "train", 2)

    assert trainer.compute_ref_log_probs.batches == 0
    assert torch.equal(column(prepared, "ref_logps"), column(base, "ref_logps"))
    assert "ref_KL_logps" not in prepared.column_names


def test_every_split_is_restored_under_its_own_name(kind, tmp_path):
    """TRL precomputes the train split and each eval dataset by name; a split the save never held
    refuses under a policy built from the checkpoint."""
    splits = {"train": token_rows(kind), "eval_a": token_rows(kind, 2), "eval_b": token_rows(kind, 3)}
    base = _save_base_run(kind, tmp_path, splits)
    resumed = _resumed(kind, tmp_path)

    for name, dataset in splits.items():
        prepared = resumed._precompute_ref_logps(dataset, name, 2)
        for reference in REFERENCE_COLUMNS[kind]:
            assert torch.equal(column(prepared, reference), column(base[name], reference)), name
    assert resumed.compute_ref_log_probs.batches == 0
    with pytest.raises(RuntimeError, match="lacks that split"):
        resumed._precompute_ref_logps(token_rows(kind, 2), "eval_c", 2)


def test_a_split_saved_on_some_ranks_only_refuses_without_sweeping(kind, tmp_path, monkeypatch):
    """A node whose copy differs must stop the whole world before any rank enters the sweep's
    collectives; the partial verdict comes from the presence consensus."""
    _save_base_run(kind, tmp_path, {"train": token_rows(kind)})
    monkeypatch.setattr(precompute_mod, "rank_consensus", lambda local_ok: (False, True))
    trainer = _resumed(kind, tmp_path)

    with pytest.raises(RuntimeError, match="on some ranks only"):
        trainer._precompute_ref_logps(token_rows(kind), "train", 2)
    assert trainer.compute_ref_log_probs.batches == 0


def test_the_sidecar_is_written_by_the_save_rank_inside_the_fence(kind, tmp_path, monkeypatch):
    trainer = precompute_trainer(kind)
    trainer._precompute_ref_logps(token_rows(kind), "train", 2)
    path = tmp_path / REFERENCE_LOGPS_FILE
    fence: list[tuple[str, bool]] = []

    @contextlib.contextmanager
    def recording_fence():
        fence.append(("enter", path.exists()))
        try:
            yield
        finally:
            fence.append(("exit", path.exists()))

    monkeypatch.setattr(precompute_mod, "barrier_on_exit", recording_fence)
    monkeypatch.setattr(precompute_mod, "fs_aware_save_rank", lambda: False)
    trainer._persist_trainer_sidecars(str(tmp_path))
    assert not path.exists(), "a rank that is not its node's save rank wrote the sidecar"
    assert fence == [("enter", False), ("exit", False)], "every rank must enter the fence"

    fence.clear()
    monkeypatch.setattr(precompute_mod, "fs_aware_save_rank", lambda: True)
    trainer._persist_trainer_sidecars(str(tmp_path))
    assert fence == [("enter", False), ("exit", True)], "the save rank must write inside the fence"


def test_dataset_supplied_columns_are_not_persisted(kind, tmp_path):
    """Columns the dataset already carries (the PP contract) ship with it again on resume."""
    trainer = precompute_trainer(kind)
    supplied = {name: [-1.0] * N_ROWS for name in REFERENCE_COLUMNS[kind]}
    dataset = concatenate_datasets([token_rows(kind), Dataset.from_dict(supplied)], axis=1)

    assert trainer._precompute_ref_logps(dataset, "train", 2) is dataset
    trainer._persist_trainer_sidecars(str(tmp_path))

    assert not (tmp_path / REFERENCE_LOGPS_FILE).exists()


def test_two_splits_sharing_a_name_are_refused(kind):
    trainer = precompute_trainer(kind)
    trainer._precompute_ref_logps(token_rows(kind), "train", 2)

    with pytest.raises(ValueError, match="share the name 'train'"):
        trainer._precompute_ref_logps(token_rows(kind, 2), "train", 2)


def test_a_resume_request_without_the_resume_context_refuses(kind):
    """A hand-built trainer resumed through ``args`` would sweep before ``train()`` sees the
    checkpoint; the scripts always pass the context, ``None`` included."""
    trainer = precompute_trainer(kind, resume_from_checkpoint="out/checkpoint-7")
    with pytest.raises(ValueError, match="built without resume_checkpoint"):
        trainer._precompute_ref_logps(token_rows(kind), "train", 2)
    assert trainer.compute_ref_log_probs.batches == 0

    fresh_script_run = precompute_trainer(kind, resume_from_checkpoint=True, resume_checkpoint=None)
    fresh_script_run._precompute_ref_logps(token_rows(kind), "train", 2)
    assert fresh_script_run.compute_ref_log_probs.batches > 0


def test_the_token_digest_reads_content_in_row_order():
    """Layout-blind (int width, an indices mapping, batch boundaries) but order- and boundary-sensitive."""
    ids = [[5, 6], [7], [8, 9, 10]]
    digest = _token_digest(Dataset.from_dict({"ids": ids}), "ids")

    narrowed = Dataset.from_dict({"ids": ids}, features=Features({"ids": List(Value("int32"))}))
    assert narrowed.features.arrow_schema.field("ids").type == pa.list_(pa.int32())
    assert _token_digest(narrowed, "ids") == digest
    remapped = Dataset.from_dict({"ids": list(reversed(ids))}).select([2, 1, 0])
    assert _token_digest(remapped, "ids") == digest
    original_batch_rows = precompute_mod._DIGEST_BATCH_ROWS
    try:
        precompute_mod._DIGEST_BATCH_ROWS = 2
        assert _token_digest(Dataset.from_dict({"ids": ids}), "ids") == digest
    finally:
        precompute_mod._DIGEST_BATCH_ROWS = original_batch_rows

    assert _token_digest(Dataset.from_dict({"ids": list(reversed(ids))}), "ids") != digest
    assert _token_digest(Dataset.from_dict({"ids": [[5], [6, 7], [8, 9, 10]]}), "ids") != digest


def test_the_trainers_checkpoint_hook_is_the_precompute_mixins(kind):
    """The mixin must precede ``DistributedTrainerMixin`` in the bases, or ``CheckpointingMixin``'s
    empty default shadows the write and no checkpoint carries the columns."""
    trainer_cls = TRAINERS[kind]
    assert trainer_cls._persist_trainer_sidecars is PrecomputeRefLogpsRankConsistentMixin._persist_trainer_sidecars
    mro = trainer_cls.__mro__
    assert mro.index(PrecomputeRefLogpsRankConsistentMixin) < mro.index(CheckpointingMixin)


@pytest.mark.parametrize(
    ("resume_checkpoint", "model_source", "expected"),
    [
        (None, "base/model", False),
        ("out/checkpoint-7", "base/model", False),
        ("out/checkpoint-7", "out/checkpoint-7", True),
    ],
    ids=["fresh", "path_a", "path_b"],
)
def test_the_runtime_reports_a_policy_built_from_the_checkpoint(resume_checkpoint, model_source, expected):
    runtime = ScriptRuntime(None, "ep2", 0, resume_checkpoint, model_source)
    assert runtime.policy_from_checkpoint is expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
