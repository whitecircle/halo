#!/usr/bin/env python
"""Precomputed reference log-probs ride every checkpoint and replace the sweep on resume.

TRL runs ``precompute_ref_log_probs`` inside ``__init__`` over ``self.ref_model or self.model``. A
Path-B resume (EP, CP, TP and the default ``use_grouped_gemm``) builds the policy from the
checkpoint before the trainer exists, so with no separate reference that sweep scores the TRAINED
weights: the reference equals the policy and every log-ratio is zero. The mixin persists the swept
columns into each checkpoint (``reference_logps.pt``) and, handed the resume checkpoint at
construction, attaches them in place of the sweep once the saved split's row count and token digest
match — or refuses where a sweep would be wrong.

The trainers are the real ``DistributedDPOTrainer`` / ``DistributedKTOTrainer`` (their MRO, column
sets and TRL signature columns), built with ``__new__`` over a stubbed TRL sweep whose values say
which weights produced them: ``BASE`` on the saving run, ``TRAINED`` on the resumed one.

    python tests/cpu/trainers/test_precompute_resume_sidecar.py
"""

import contextlib

import numpy as np
import pytest
import torch
from accelerate import PartialState
from datasets import Dataset, concatenate_datasets
from trl.experimental.kto.kto_trainer import KTOTrainer as TrlKTOTrainer
from trl.trainer.dpo_trainer import DPOTrainer as TrlDPOTrainer

from src.checkpoint.format import REFERENCE_LOGPS_FILE
from src.trainers.mixins.checkpointing import CheckpointingMixin
from src.trainers.preference.dpo import DistributedDPOTrainer
from src.trainers.preference.kto import DistributedKTOTrainer
from src.trainers.preference.precompute import PrecomputeRefLogpsRankConsistentMixin, _token_digest
from src.training.script_runner import ScriptRuntime

PartialState()

TRAINERS = {"dpo": DistributedDPOTrainer, "kto": DistributedKTOTrainer}
# The class in each trainer's MRO that owns TRL's sweep, which the stub replaces.
TRL_SWEEPS = {"dpo": TrlDPOTrainer, "kto": TrlKTOTrainer}
BASE = 0.0
TRAINED = 1000.0
N_ROWS = 4


def _rows(kind: str, n: int = N_ROWS, *, bump: int = 0) -> Dataset:
    """Tokenized rows as TRL's ``_prepare_dataset`` leaves them; ``bump`` changes one token id."""
    if kind == "dpo":
        rows = {
            "prompt": [f"q{i}" for i in range(n)],
            "prompt_ids": [[1, i] for i in range(n)],
            "chosen_ids": [[2, i, i] for i in range(n)],
            "rejected_ids": [[3] * (i + 1) for i in range(n)],
        }
        rows["chosen_ids"][-1][-1] += bump
    else:
        rows = {
            "prompt_ids": [[1, i] for i in range(n)],
            "completion_ids": [[2, i] for i in range(n)],
            "KL_completion_ids": [[2, n - 1 - i] for i in range(n)],
            "label": [i % 2 == 0 for i in range(n)],
        }
        rows["completion_ids"][-1][-1] += bump
    return Dataset.from_dict(rows)


def _install_sweep(monkeypatch, kind: str, weights: float) -> list[str]:
    """Replace TRL's sweep with one whose values encode ``weights``; returns the splits it ran on."""
    calls: list[str] = []

    def sweep(self, dataset, name, batch_size):
        calls.append(name)
        values = {
            column: np.asarray([-(weights + 100 * index + row) for row in range(len(dataset))], dtype=np.float32)
            for index, column in enumerate(self._required_ref_logps_columns())
        }
        return concatenate_datasets([dataset, Dataset.from_dict(values)], axis=1)

    monkeypatch.setattr(TRL_SWEEPS[kind], "_precompute_ref_logps", sweep)
    return calls


def _trainer(kind, *, resume_checkpoint=None, policy_from_checkpoint=False, ref_model=None, calculate_kl=True):
    """The real trainer class with the attributes TRL's ``__init__`` holds when it runs the sweep."""
    trainer = TRAINERS[kind].__new__(TRAINERS[kind])
    trainer._init_reference_resume(
        {"resume_checkpoint": resume_checkpoint, "policy_from_checkpoint": policy_from_checkpoint}
    )
    trainer._dataset_presharded = False
    trainer._signature_columns = None
    trainer._is_vision_dataset = False
    trainer.ref_model = ref_model
    trainer.calculate_KL = calculate_kl
    trainer.data_parallel_sweep = contextlib.nullcontext
    return trainer


def _column(dataset: Dataset, column: str) -> torch.Tensor:
    return torch.tensor(list(dataset[column]), dtype=torch.float32)


def _save_base_run(monkeypatch, kind, checkpoint_dir, splits, **trainer_kwargs) -> dict[str, Dataset]:
    """A fresh run's precompute over ``splits`` (name → dataset), persisted into ``checkpoint_dir``."""
    _install_sweep(monkeypatch, kind, BASE)
    trainer = _trainer(kind, **trainer_kwargs)
    prepared = {name: trainer._precompute_ref_logps(dataset, name, 2) for name, dataset in splits.items()}
    trainer._persist_trainer_sidecars(str(checkpoint_dir))
    return prepared


@pytest.fixture(params=sorted(TRAINERS))
def kind(request):
    return request.param


def test_the_swept_columns_are_persisted_as_training_reads_them(kind, monkeypatch, tmp_path):
    prepared = _save_base_run(monkeypatch, kind, tmp_path, {"train": _rows(kind)})["train"]

    saved = torch.load(tmp_path / REFERENCE_LOGPS_FILE, weights_only=True)
    needed = _trainer(kind)._required_ref_logps_columns()
    assert set(saved) == {"train"}
    assert saved["train"]["num_rows"] == N_ROWS
    assert set(saved["train"]["columns"]) == set(needed)
    for column in needed:
        assert torch.equal(saved["train"]["columns"][column], _column(prepared, column))


def test_a_resume_attaches_the_saved_columns_instead_of_sweeping_the_trained_policy(kind, monkeypatch, tmp_path):
    base = _save_base_run(monkeypatch, kind, tmp_path, {"train": _rows(kind)})["train"]
    calls = _install_sweep(monkeypatch, kind, TRAINED)
    resumed = _trainer(kind, resume_checkpoint=str(tmp_path), policy_from_checkpoint=True)

    prepared = resumed._precompute_ref_logps(_rows(kind), "train", 2)

    assert calls == [], "the resume swept the policy, which holds the trained weights"
    needed = resumed._required_ref_logps_columns()
    for column in needed:
        assert torch.equal(_column(prepared, column), _column(base, column))
    # The resumed run's own checkpoints carry the same values on, so a second resume restores them.
    resumed._persist_trainer_sidecars(str(tmp_path / "next"))
    again = torch.load(tmp_path / "next" / REFERENCE_LOGPS_FILE, weights_only=True)
    first = torch.load(tmp_path / REFERENCE_LOGPS_FILE, weights_only=True)
    assert again["train"]["token_digest"] == first["train"]["token_digest"]
    for column in needed:
        assert torch.equal(again["train"]["columns"][column], first["train"]["columns"][column])


def test_a_policy_built_from_the_checkpoint_without_saved_columns_refuses(kind, monkeypatch, tmp_path):
    calls = _install_sweep(monkeypatch, kind, TRAINED)
    trainer = _trainer(kind, resume_checkpoint=str(tmp_path), policy_from_checkpoint=True)

    with pytest.raises(RuntimeError, match="TRAINED weights as the reference"):
        trainer._precompute_ref_logps(_rows(kind), "train", 2)
    assert calls == []


@pytest.mark.parametrize(
    ("policy_from_checkpoint", "ref_model"),
    [(False, None), (True, torch.nn.Linear(1, 1))],
    ids=["policy_from_base", "separate_reference"],
)
def test_a_resume_sweeps_where_the_reference_weights_are_untrained(
    kind, monkeypatch, tmp_path, policy_from_checkpoint, ref_model
):
    """Path A (the Trainer restores the weights only in ``train()``) or a separate reference model:
    the sweep scores untrained weights, so a checkpoint without the sidecar still resumes."""
    calls = _install_sweep(monkeypatch, kind, BASE)
    trainer = _trainer(
        kind, resume_checkpoint=str(tmp_path), policy_from_checkpoint=policy_from_checkpoint, ref_model=ref_model
    )

    trainer._precompute_ref_logps(_rows(kind), "train", 2)

    assert calls == ["train"]


@pytest.mark.parametrize(
    ("resumed_rows", "reason"),
    [
        (lambda kind: _rows(kind, N_ROWS - 1), "saved for 4 rows"),
        (lambda kind: _rows(kind, bump=7), "token ids differ"),
        (lambda kind: _rows(kind).select(list(reversed(range(N_ROWS)))), "token ids differ"),
    ],
    ids=["fewer_rows", "changed_token", "reordered_rows"],
)
def test_saved_columns_for_other_rows_refuse(kind, monkeypatch, tmp_path, resumed_rows, reason):
    _save_base_run(monkeypatch, kind, tmp_path, {"train": _rows(kind)})
    calls = _install_sweep(monkeypatch, kind, TRAINED)
    trainer = _trainer(kind, resume_checkpoint=str(tmp_path), policy_from_checkpoint=True)

    with pytest.raises(ValueError, match=reason):
        trainer._precompute_ref_logps(resumed_rows(kind), "train", 2)
    assert calls == []


def test_a_kto_resume_that_needs_the_kl_column_the_save_lacks_refuses(monkeypatch, tmp_path):
    """The needed set is derived per run (``calculate_KL``), so a resume under a KL loss cannot be
    served by a save written under a KL-free one."""
    _save_base_run(monkeypatch, "kto", tmp_path, {"train": _rows("kto")}, calculate_kl=False)
    trainer = _trainer("kto", resume_checkpoint=str(tmp_path), policy_from_checkpoint=True, calculate_kl=True)

    with pytest.raises(ValueError, match="ref_KL_logps"):
        trainer._precompute_ref_logps(_rows("kto"), "train", 2)


def test_every_split_is_restored_under_its_own_name(kind, monkeypatch, tmp_path):
    """TRL precomputes the train split and each eval dataset by name; a split the save never held
    refuses under a policy built from the checkpoint."""
    splits = {"train": _rows(kind), "eval_a": _rows(kind, 2), "eval_b": _rows(kind, 3, bump=5)}
    base = _save_base_run(monkeypatch, kind, tmp_path, splits)
    calls = _install_sweep(monkeypatch, kind, TRAINED)
    resumed = _trainer(kind, resume_checkpoint=str(tmp_path), policy_from_checkpoint=True)

    for name, dataset in splits.items():
        prepared = resumed._precompute_ref_logps(dataset, name, 2)
        for column in resumed._required_ref_logps_columns():
            assert torch.equal(_column(prepared, column), _column(base[name], column)), name
    assert calls == []
    with pytest.raises(RuntimeError, match="'eval_c'"):
        resumed._precompute_ref_logps(_rows(kind, 2), "eval_c", 2)


def test_dataset_supplied_columns_are_not_persisted(kind, monkeypatch, tmp_path):
    """Columns the dataset already carries (the PP contract) ship with it again on resume."""
    calls = _install_sweep(monkeypatch, kind, BASE)
    trainer = _trainer(kind)
    supplied = {column: [-1.0] * N_ROWS for column in trainer._required_ref_logps_columns()}
    dataset = concatenate_datasets([_rows(kind), Dataset.from_dict(supplied)], axis=1)

    assert trainer._precompute_ref_logps(dataset, "train", 2) is dataset
    trainer._persist_trainer_sidecars(str(tmp_path))

    assert calls == []
    assert not (tmp_path / REFERENCE_LOGPS_FILE).exists()


def test_two_splits_sharing_a_name_are_refused(kind, monkeypatch):
    _install_sweep(monkeypatch, kind, BASE)
    trainer = _trainer(kind)
    trainer._precompute_ref_logps(_rows(kind), "train", 2)

    with pytest.raises(ValueError, match="share the name 'train'"):
        trainer._precompute_ref_logps(_rows(kind, 2), "train", 2)


def test_the_token_digest_reads_content_in_row_order():
    """Layout-blind (int width, an indices mapping) but order- and boundary-sensitive."""
    columns = ["ids"]
    ids = [[5, 6], [7], [8, 9, 10]]
    digest = _token_digest(Dataset.from_dict({"ids": ids}), columns)

    narrowed = Dataset.from_dict({"ids": [np.asarray(row, dtype=np.int32) for row in ids]})
    assert _token_digest(narrowed, columns) == digest
    remapped = Dataset.from_dict({"ids": list(reversed(ids))}).select([2, 1, 0])
    assert _token_digest(remapped, columns) == digest

    assert _token_digest(Dataset.from_dict({"ids": list(reversed(ids))}), columns) != digest
    assert _token_digest(Dataset.from_dict({"ids": [[5], [6, 7], [8, 9, 10]]}), columns) != digest


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
