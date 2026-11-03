#!/usr/bin/env python
"""Every rank attaches the gathered reference log-probs in memory; no rank reads another's file.
On per-node storage TRL's cache file is on one node, so the others raise while the main process hangs."""

import os

import pytest
from accelerate import PartialState
from datasets import Dataset, concatenate_datasets, load_from_disk
from trl.trainer.dpo_trainer import DPOTrainer as TrlDPOTrainer
from trl.trainer.kto_trainer import KTOTrainer as TrlKTOTrainer

from halo.trainers.preference.trl_preference import TRLPreferenceMixin
from tests.common.preference_precompute import (
    BASE,
    N_ROWS,
    REFERENCE_COLUMNS,
    SWEEP_BATCH_SIZE,
    TRAINERS,
    column,
    precompute_trainer,
    token_rows,
)

PartialState()

TRL_BASES = {"dpo": TrlDPOTrainer, "kto": TrlKTOTrainer}


@pytest.fixture(params=sorted(TRAINERS))
def kind(request):
    return request.param


@pytest.mark.parametrize("main_process", [False, True], ids=["other_rank", "main_process"])
def test_every_rank_attaches_the_swept_columns_without_a_file(kind, tmp_path, main_process):
    dataset_dir = tmp_path / "dataset"
    token_rows(kind).save_to_disk(str(dataset_dir))
    dataset = load_from_disk(str(dataset_dir))
    files_before = sorted(os.listdir(dataset_dir))
    trainer = precompute_trainer(kind, main_process=main_process)

    prepared = trainer._precompute_ref_logps(dataset, "train", SWEEP_BATCH_SIZE)

    assert sorted(os.listdir(dataset_dir)) == files_before, "the sweep wrote a file beside the dataset"
    token_sums = trainer.data_collator(list(dataset))
    for index, name in enumerate(REFERENCE_COLUMNS[kind]):
        assert name in prepared.column_names
        assert column(prepared, name).tolist() == (-(BASE + 100 * index + token_sums)).tolist()
    assert trainer.compute_ref_log_probs.batches == N_ROWS // SWEEP_BATCH_SIZE


def test_a_kto_loss_without_kl_attaches_only_ref_logps():
    """``compute_ref_log_probs`` returns the KL term as ``None`` there; it must not shift the columns."""
    trainer = precompute_trainer("kto", calculate_kl=False)
    prepared = trainer._precompute_ref_logps(
        token_rows("kto").remove_columns(["KL_completion_ids"]), "train", SWEEP_BATCH_SIZE
    )

    assert "ref_logps" in prepared.column_names
    assert "ref_KL_logps" not in prepared.column_names


def test_dataset_supplied_columns_skip_the_sweep(kind):
    """The PP contract: columns the dataset carries are trusted, and a dataset missing one is swept."""
    trainer = precompute_trainer(kind)
    supplied = {name: [-1.0] * N_ROWS for name in REFERENCE_COLUMNS[kind]}
    dataset = concatenate_datasets([token_rows(kind), Dataset.from_dict(supplied)], axis=1)

    assert trainer._precompute_ref_logps(dataset, "train", SWEEP_BATCH_SIZE) is dataset
    assert trainer.compute_ref_log_probs.batches == 0

    trainer._precompute_ref_logps(token_rows(kind), "eval", SWEEP_BATCH_SIZE)
    assert trainer.compute_ref_log_probs.batches > 0


def test_the_trainers_route_the_precompute_through_the_mixin(kind):
    """TRL's own sweep is the cache-file hand-off that fails across nodes."""
    trainer_cls, _ = TRAINERS[kind]
    assert trainer_cls._precompute_ref_logps is TRLPreferenceMixin._precompute_ref_logps
    mro = trainer_cls.__mro__
    assert mro.index(TRLPreferenceMixin) < mro.index(TRL_BASES[kind])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
