"""TRL's ``_prepare_dataset`` runs once per filesystem scope, outside any collective, and is reused.
TRL holds the peers in an NCCL barrier while it tokenizes, and every rank writes its own tokenized copy per run."""

import datetime
import inspect
import json
import os
import re
import time

import pytest
import torch.distributed as dist
from accelerate import PartialState
from datasets import Dataset, DatasetDict, load_from_disk
from trl import DPOConfig, DPOTrainer
from trl.trainer import utils as trl_trainer_utils

from halo.data.sources.loading import load_datasets
from halo.trainers.preference.dpo import DistributedDPOTrainer
from halo.trainers.preference.kto import DistributedKTOTrainer
from halo.trainers.reward.bradley_terry import DistributedRewardTrainer
from halo.trainers.trl_dataset_prep import _MAIN_FIRST_METHODS, TRLDatasetPrepMixin
from tests.common.gloo import run_gloo_ranks
from tests.common.models import QWEN3_0_6B
from tests.common.tokenizers import load_cached_tokenizer, try_cached_tokenizer

WORLD_SIZE = 2

# The load rank's preparation outlasts this: a peer held in a collective dies, one on the store does not.
PG_TIMEOUT_SEC = 8
SLOW_PREPARE_SEC = PG_TIMEOUT_SEC + 6

ROWS = 12


class _CountingDPO(DPOTrainer):
    """TRL's real DPO preparation, counted per rank and slowed on the load rank."""

    def _prepare_dataset(self, dataset, processing_class, args, dataset_name):
        with open(os.path.join(self.tmp_dir, f"prepared_{self.rank}.txt"), "a") as handle:
            handle.write(f"{args.output_dir}\n")
        if self.rank == 0:
            time.sleep(SLOW_PREPARE_SEC)
        return super()._prepare_dataset(dataset, processing_class, args, dataset_name)


class _CoordinatedDPO(TRLDatasetPrepMixin, _CountingDPO):
    def __init__(self, tokenizer, rank: int, tmp_dir: str):
        self._tokenizer = tokenizer
        self._is_vlm = False
        self.rank = rank
        self.tmp_dir = tmp_dir
        # Every real trainer holds process groups; they are what TRL's map closure cannot hash.
        self.group = dist.group.WORLD


# (launch, rows stamped by the toolkit's loader): two launches over each kind of input.
RUNS = (("run1", True), ("run2", True), ("raw1", False), ("raw2", False))


def _worker(rank: int, tmp_dir: str, source: str) -> None:
    PartialState()
    tokenizer = try_cached_tokenizer(QWEN3_0_6B)
    outcome = {}
    for run, stamped in RUNS:
        args = DPOConfig(output_dir=os.path.join(tmp_dir, run), use_cpu=True, bf16=False, report_to=[])
        rows = (
            load_datasets(source, test_size=None, dataset_ratio=1, conversation_field=None)["train"]
            if stamped
            else load_from_disk(source)["train"]
        )
        try:
            prepared = _CoordinatedDPO(tokenizer, rank, tmp_dir)._prepare_dataset(rows, tokenizer, args, "train")
            outcome[run] = prepared.select_columns(["prompt_ids", "chosen_ids", "rejected_ids"]).to_dict()
        except BaseException as exc:
            outcome[run] = {"error": f"{type(exc).__name__}: {exc}"}
    with open(os.path.join(tmp_dir, f"result_{rank}.json"), "w") as handle:
        json.dump(outcome, handle)


def _prepared_runs(tmp_path, rank: int) -> list[str]:
    path = tmp_path / f"prepared_{rank}.txt"
    return path.read_text().split() if path.exists() else []


def test_trl_preparation_runs_once_off_the_watchdog_and_is_reused(tmp_path):
    load_cached_tokenizer(QWEN3_0_6B)
    source = str(tmp_path / "source")
    pairs = Dataset.from_dict(
        {
            "prompt": [f"Question {i}?" for i in range(ROWS)],
            "chosen": [f" Answer {i}." for i in range(ROWS)],
            "rejected": [f" Wrong {i}." for i in range(ROWS)],
        }
    )
    DatasetDict({"train": pairs, "test": pairs}).save_to_disk(source)

    run_gloo_ranks(
        _worker,
        WORLD_SIZE,
        str(tmp_path),
        source,
        pg_timeout=datetime.timedelta(seconds=PG_TIMEOUT_SEC),
        env={"HF_DATASETS_CACHE": str(tmp_path / "hf_datasets")},
    )
    outcomes = [json.loads((tmp_path / f"result_{rank}.json").read_text()) for rank in range(WORLD_SIZE)]

    for rank, outcome in enumerate(outcomes):
        for run, _ in RUNS:
            assert "error" not in outcome[run], f"rank {rank}, {run}: {outcome[run]}"
            assert outcome[run] == outcomes[0]["run1"], f"rank {rank}, {run} prepared other rows"
    assert len(outcomes[0]["run1"]["chosen_ids"]) == ROWS
    assert _prepared_runs(tmp_path, 0) == [str(tmp_path / run) for run in ("run1", "raw1", "raw2")], (
        "the load rank must prepare stamped rows once and unstamped rows on every launch"
    )
    assert _prepared_runs(tmp_path, 1) == [], "the waiting rank repeated TRL's preparation"
    leaked = sorted(name for name in os.listdir(os.path.join(source, "train")) if name.startswith("cache-"))
    assert leaked == [], f"TRL's maps wrote unreusable cache copies beside the source: {leaked}"


def _trl_prepare_owner(cls) -> type | None:
    return next(
        (base for base in cls.__mro__ if "_prepare_dataset" in base.__dict__ and base.__module__.startswith("trl.")),
        None,
    )


def _main_first_scopes(prepare) -> set[str]:
    """The ``PartialState`` main-first scopes ``prepare`` enters, directly or through TRL's helper."""
    source = inspect.getsource(prepare)
    if "global_then_local_main_first" in source:
        source += inspect.getsource(trl_trainer_utils.global_then_local_main_first)
    return set(re.findall(r"\.(\w*main\w*_first)\(", source))


@pytest.mark.parametrize("trainer_cls", [DistributedDPOTrainer, DistributedKTOTrainer, DistributedRewardTrainer])
def test_every_trainer_over_a_trl_preparation_routes_through_the_seam(trainer_cls):
    """One left out would hold its peers in the NCCL barrier and write one tokenized copy per rank per run.
    The SFT-derived trainers skip TRL's preparation (their scripts call ``disable_trl_dataset_prep``)."""
    owner = _trl_prepare_owner(trainer_cls)
    assert owner is not None, f"{trainer_cls.__name__}: TRL no longer prepares its dataset — revisit the seam"
    scopes = _main_first_scopes(owner._prepare_dataset)
    assert scopes, f"{trainer_cls.__name__}: TRL no longer prepares under a main-first scope — revisit the seam"
    assert scopes <= set(_MAIN_FIRST_METHODS), f"{trainer_cls.__name__}: the seam leaves {scopes} live"
    mro = trainer_cls.__mro__
    assert TRLDatasetPrepMixin in mro, f"{trainer_cls.__name__} prepares through TRL uncoordinated"
    assert mro.index(TRLDatasetPrepMixin) < mro.index(owner)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
