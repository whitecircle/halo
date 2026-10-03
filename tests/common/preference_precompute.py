"""The DPO / KTO precompute tests' shared pieces, and the real trainers over a stubbed reference
forward for the CPU tests.

``TRAINERS`` and :func:`column` serve the GPU precompute suites as well. ``precompute_trainer``
builds ``DistributedDPOTrainer`` / ``DistributedKTOTrainer`` with ``__new__`` and the attributes
TRL's ``__init__`` holds when it runs the sweep, so the tests drive the real MRO, column sets and
TRL signature columns. ``compute_ref_log_probs`` is replaced by :class:`StubReference`, whose values
encode which weights "ran" and each row's tokens.
"""

from __future__ import annotations

import contextlib
from types import SimpleNamespace

import torch
from datasets import Dataset
from trl import DPOConfig, KTOConfig

from src.trainers.preference.dpo import DistributedDPOTrainer
from src.trainers.preference.kto import DistributedKTOTrainer

# Each kind's trainer and the TRL config it takes.
TRAINERS = {"dpo": (DistributedDPOTrainer, DPOConfig), "kto": (DistributedKTOTrainer, KTOConfig)}
# The columns each TRL base writes, spelled out rather than read off the trainer under test.
REFERENCE_COLUMNS = {"dpo": ("ref_chosen_logps", "ref_rejected_logps"), "kto": ("ref_logps", "ref_KL_logps")}
# The columns the reference forward reads, per TRL's signature columns.
TOKEN_COLUMNS = {
    "dpo": ("prompt_ids", "chosen_ids", "rejected_ids"),
    "kto": ("prompt_ids", "completion_ids", "KL_completion_ids", "label"),
}
BASE = 0.0
TRAINED = 1000.0
N_ROWS = 4
# The batch size the CPU tests hand the reference sweep.
SWEEP_BATCH_SIZE = 2
MAX_LENGTH = 64


def token_rows(kind: str, n: int = N_ROWS, *, bump: str | None = None) -> Dataset:
    """Tokenized rows as TRL's ``_prepare_dataset`` leaves them; ``bump`` changes one value of that column."""
    if kind == "dpo":
        rows = {
            "prompt": [f"q{i}" for i in range(n)],
            "prompt_ids": [[1, i] for i in range(n)],
            "chosen_ids": [[2, i, i] for i in range(n)],
            "rejected_ids": [[3] * (i + 1) for i in range(n)],
        }
    else:
        rows = {
            "prompt_ids": [[1, i] for i in range(n)],
            "completion_ids": [[2, i] for i in range(n)],
            "KL_completion_ids": [[2, n - 1 - i] for i in range(n)],
            "label": [i % 2 == 0 for i in range(n)],
        }
    if bump == "label":
        rows["label"][-1] = not rows["label"][-1]
    elif bump is not None:
        rows[bump][-1][-1] += 7
    return Dataset.from_dict(rows)


class StubReference:
    """Stands in for TRL's ``compute_ref_log_probs``: one value per output column per row, offset by
    the ``weights`` that "ran" and by the row's token sum, with KTO's KL term ``None`` off the KL loss."""

    def __init__(self, kind: str, weights: float, calculate_kl: bool):
        self.outputs = 2 if kind == "dpo" or calculate_kl else 1
        self.weights = weights
        self.batches = 0

    def __call__(self, batch: torch.Tensor):
        self.batches += 1
        values = [-(self.weights + 100 * index + batch) for index in range(self.outputs)]
        return (values[0], values[1] if self.outputs == 2 else None)


def collate_token_sums(rows: list[dict]) -> torch.Tensor:
    """Each row's sum over its token-id and label columns: the stub reference's per-row input."""
    return torch.tensor(
        [
            sum(sum(value) if isinstance(value, list) else int(value) for key, value in row.items() if key != "prompt")
            for row in rows
        ],
        dtype=torch.float32,
    )


def precompute_trainer(
    kind: str,
    *,
    weights: float = BASE,
    main_process: bool = True,
    ref_model=None,
    calculate_kl: bool = True,
    resume_from_checkpoint=None,
    max_length: int = MAX_LENGTH,
    **resume_context,
):
    """The real trainer class with the attributes TRL's ``__init__`` holds when it runs the sweep.

    ``resume_context`` is what the entry scripts pass (``resume_checkpoint``, ``policy_from_checkpoint``);
    leave it empty for a trainer built without it.
    """
    trainer_cls, _ = TRAINERS[kind]
    trainer = trainer_cls.__new__(trainer_cls)
    trainer._init_reference_resume(dict(resume_context))
    trainer._dataset_presharded = False
    trainer._signature_columns = None
    trainer._is_vision_dataset = False
    trainer.model = torch.nn.Linear(1, 1)
    trainer.ref_model = ref_model
    trainer.calculate_KL = calculate_kl
    trainer.ld_alpha = None
    trainer.args = SimpleNamespace(
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        resume_from_checkpoint=resume_from_checkpoint,
        max_length=max_length,
        truncation_mode="keep_start",
        gradient_checkpointing_kwargs=None,
    )
    trainer.accelerator = SimpleNamespace(
        prepare=lambda loader: loader,
        gather_for_metrics=lambda data: data,
        is_main_process=main_process,
        wait_for_everyone=lambda: None,
        device=torch.device("cpu"),
    )
    trainer.data_collator = collate_token_sums
    trainer.compute_ref_log_probs = StubReference(kind, weights, calculate_kl)
    trainer.data_parallel_sweep = contextlib.nullcontext
    return trainer


def column(dataset: Dataset, name: str) -> torch.Tensor:
    return torch.tensor(list(dataset[name]), dtype=torch.float32)
