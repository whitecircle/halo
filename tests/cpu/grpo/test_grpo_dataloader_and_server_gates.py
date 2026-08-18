#!/usr/bin/env python
"""GRPO train-dataloader column pruning + the vLLM server-mode construction gates.

- the GRPO train dataloader builds its dataset/collator pair through the shared
  ``DataParallelDataLoaderMixin._loader_params`` seam (sized by GRPO's ``_train_loader_batch_size``
  hook), so a train dataset that is NOT a ``datasets.Dataset`` still honours
  ``remove_unused_columns`` — via the collator, the base ``Trainer`` contract. A diverged copy that
  only prunes when the dataset exposes ``column_names`` silently trains such a dataset on every
  column and fails here.
- ``_require_vllm_server_mode`` raises instead of no-opping when no training config reaches the
  ctor, and it agrees with ``_is_vllm_server_mode`` on what "server mode" means — a disagreement
  would accept a config the TRL vLLM-client patch then declines to install.

    python tests/cpu/grpo/test_grpo_dataloader_and_server_gates.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from datasets import Dataset
from torch.utils.data import SequentialSampler

from src.trainers.grpo.online import DistributedGRPOTrainer

_ROWS = [{"prompt": "p", "unused": i} for i in range(6)]


def _raw_collator(batch):
    return batch


def _pruned_collator(batch):
    return batch


def _train_loader(dataset, *, remove_unused_columns: bool):
    """Run the real GRPO train-dataloader path over ``dataset``; returns (loader, prune_calls).

    Only the HF/accelerate seams are stubbed (column pruning, sampler, accelerate prepare); the
    dataset/collator decision under test is the production one.
    """
    trainer = object.__new__(DistributedGRPOTrainer)
    trainer.parallelism_config = SimpleNamespace(
        is_tp_mode=True, is_cp_mode=False, is_expert_tp_mode=False, is_pp_mode=False
    )
    trainer._dataset_presharded = False
    trainer.train_dataset = dataset
    trainer._train_batch_size = 2
    trainer.data_collator = _raw_collator
    trainer.args = SimpleNamespace(
        remove_unused_columns=remove_unused_columns,
        steps_per_generation=3,
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=False,
        dataloader_drop_last=False,
        dataloader_prefetch_factor=None,
    )

    calls = {"collator": [], "dataset": []}

    def _prune_collator(collator, description):
        calls["collator"].append(description)
        return _pruned_collator

    def _prune_dataset(ds, description=None):
        calls["dataset"].append(description)
        return ds.remove_columns(["unused"])

    trainer._get_collator_with_removed_columns = _prune_collator
    trainer._remove_unused_columns = _prune_dataset
    trainer._get_train_sampler = lambda: SequentialSampler(dataset)
    trainer._prepare_dataloader = lambda loader: loader
    trainer.get_data_parallel_rank = lambda: 0

    return DistributedGRPOTrainer.get_train_dataloader(trainer), calls


def test_non_datasets_train_dataset_prunes_through_the_collator():
    """A plain-list train dataset has no ``column_names``, so pruning must reach the COLLATOR.

    The diverged copy this replaces passed ``self.data_collator`` through untouched, so
    ``remove_unused_columns`` was silently dropped for every non-``datasets.Dataset`` train dataset.
    """
    loader, calls = _train_loader(list(_ROWS), remove_unused_columns=True)

    assert loader.collate_fn is _pruned_collator, (
        "remove_unused_columns was lost: the raw collator reached the DataLoader for a "
        "non-datasets.Dataset train dataset"
    )
    assert calls["collator"] == ["training"], f"expected one 'training' collator prune, got {calls['collator']}"
    assert calls["dataset"] == [], "a plain list has no columns to prune off the dataset"


def test_datasets_dataset_prunes_the_dataset_not_the_collator():
    """Anti-vacuity for the branch above: a real ``datasets.Dataset`` keeps the base contract —
    columns come off the dataset and the collator is passed through untouched."""
    loader, calls = _train_loader(Dataset.from_list(_ROWS), remove_unused_columns=True)

    assert loader.collate_fn is _raw_collator, "a datasets.Dataset must not have its collator rewrapped"
    assert calls["dataset"] == ["training"], f"expected one 'training' dataset prune, got {calls['dataset']}"
    assert "unused" not in loader.dataset.column_names, "the unused column survived dataset pruning"


def test_train_batch_covers_a_whole_generation_round():
    """The GRPO batch is ``per_device_train_batch_size * steps_per_generation`` — the whole round's
    prompts are fetched at once, not one micro-batch."""
    loader, _ = _train_loader(list(_ROWS), remove_unused_columns=False)
    assert loader.batch_size == 6, f"expected 2 * 3 rows per fetch, got {loader.batch_size}"


def test_missing_training_config_raises():
    """``ctor_config`` resolved no config: silently returning would leave in-process HF generation
    running under FSDP2/EP (slow and rank-divergent) with nothing in the log to say the gate never ran."""
    with pytest.raises(ValueError, match="GRPOConfig"):
        DistributedGRPOTrainer._require_vllm_server_mode(None)


@pytest.mark.parametrize(
    ("use_vllm", "vllm_mode", "match"),
    [
        (False, "server", "use_vllm=True"),
        (True, "colocate", "vllm_mode='server'"),
    ],
)
def test_non_server_configs_are_refused(use_vllm, vllm_mode, match):
    args = SimpleNamespace(use_vllm=use_vllm, vllm_mode=vllm_mode)
    with pytest.raises(ValueError, match=match):
        DistributedGRPOTrainer._require_vllm_server_mode(args)


@pytest.mark.parametrize(
    ("use_vllm", "vllm_mode", "is_server"),
    [(True, "server", True), (True, "colocate", False), (False, "server", False)],
)
def test_server_mode_detection_agrees_with_the_requirement_gate(use_vllm, vllm_mode, is_server):
    """The detector decides whether TRL's vLLM client is swapped for the vendored NCCL one; the
    requirement gate decides whether construction proceeds. A config the gate accepts but the
    detector calls non-server would run unpatched and import a vLLM the training image lacks."""
    args = SimpleNamespace(use_vllm=use_vllm, vllm_mode=vllm_mode)
    assert DistributedGRPOTrainer._is_vllm_server_mode(args) is is_server

    accepted = True
    try:
        DistributedGRPOTrainer._require_vllm_server_mode(args)
    except ValueError:
        accepted = False
    assert accepted is is_server, "the server-mode detector and the requirement gate disagree"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
