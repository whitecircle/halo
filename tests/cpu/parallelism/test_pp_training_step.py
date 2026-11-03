#!/usr/bin/env python
"""One pipeline step through each trainer's real loss adapter, on four gloo ranks, against its objective on the
untransformed batch. Wrong row pairs, normalizers or stages move the loss."""

import contextlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from transformers import PretrainedConfig

from halo.configs.classification import ClassificationConfig
from halo.distributed.parallelism_config import ParallelismConfig
from halo.distributed.pipeline_parallel.groups import build_pipeline_group, build_stage_group
from halo.distributed.pipeline_parallel.stage_losses import TokenLoss
from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from halo.trainers.preference.dpo import DistributedDPOTrainer
from halo.trainers.preference.kto import DistributedKTOTrainer
from halo.trainers.reward.bradley_terry import DistributedRewardTrainer
from halo.trainers.reward.classification import ClassificationTrainer
from halo.trainers.reward.classification_loss import build_classification_loss
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.gloo import run_gloo_ranks
from tests.common.pp_fakes import FakeSchedule, fake_runner

WORLD, PP_SIZE, STAGE_WIDTH = 4, 2, 2
VOCAB, SEQ, PAD = 16, 8, 0
# Four pairs for the paired objectives, split over two microbatches.
ROWS, MICROBATCHES = 8, 2
NUM_LABELS = 3
BETA = 0.5
DESIRABLE_WEIGHT, UNDESIRABLE_WEIGHT = 1.3, 0.7
TOLERANCE = 1e-5
ADAPTERS = ("sft", "dpo", "kto", "reward", "classification")


def _head_width(kind: str) -> int:
    return {"reward": 1, "classification": NUM_LABELS}.get(kind, VOCAB)


def _logits(input_ids: torch.Tensor, kind: str) -> torch.Tensor:
    """Identical on every rank."""
    table = torch.randn(VOCAB, _head_width(kind), generator=torch.Generator().manual_seed(11))
    return table[input_ids]


def _token_logprobs(input_ids: torch.Tensor, kind: str) -> torch.Tensor:
    """0 at position 0."""
    logps = torch.log_softmax(_logits(input_ids, kind)[:, :-1], dim=-1)
    return F.pad(logps.gather(-1, input_ids[:, 1:, None]).squeeze(-1), (1, 0))


def _ragged_rows(generator: torch.Generator, rows: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Real tokens are never the pad id; a prompt of two tokens, a completion to the row's end."""
    lengths = torch.randint(4, SEQ + 1, (rows,), generator=generator)
    real = torch.arange(SEQ)[None, :] < lengths[:, None]
    input_ids = torch.where(real, torch.randint(1, VOCAB, (rows, SEQ), generator=generator), PAD)
    completion = real & (torch.arange(SEQ)[None, :] >= 2)
    return input_ids, real.long(), completion.long()


def _batch(kind: str, dp_rank: int) -> dict:
    generator = torch.Generator().manual_seed(100 + dp_rank)
    input_ids, attention_mask, completion_mask = _ragged_rows(generator, ROWS)
    if kind == "sft":
        labels = torch.where(completion_mask.bool(), input_ids, LABEL_IGNORE_INDEX)
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    if kind == "dpo":
        pairs = ROWS // 2
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
            "ref_chosen_logps": -torch.rand(pairs, generator=generator) * 10,
            "ref_rejected_logps": -torch.rand(pairs, generator=generator) * 10,
        }
    if kind == "kto":
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "completion_mask": completion_mask,
            "ref_logps": -torch.rand(ROWS, generator=generator) * 10,
            "label": [bool(i % 3) for i in range(ROWS)],
        }
    if kind == "reward":
        return {"input_ids": input_ids, "attention_mask": attention_mask}
    labels = torch.randint(0, NUM_LABELS, (ROWS,), generator=generator)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def _last_real_positions(input_ids: torch.Tensor) -> torch.Tensor:
    return (input_ids != PAD).long().cumsum(dim=1).argmax(dim=1)


def _expected(kind: str, dp_rank: int) -> tuple[float, float]:
    """TRL's DPO sigmoid and KTO's apo_zero_unpaired, the Bradley-Terry and class loss at the last real token, and
    the causal-LM token mean."""
    batch = _batch(kind, dp_rank)
    input_ids = batch["input_ids"]
    if kind == "sft":
        counts = [(_batch(kind, rank)["labels"][:, 1:] != LABEL_IGNORE_INDEX).sum() for rank in range(STAGE_WIDTH)]
        normalizer = float(sum(counts)) / STAGE_WIDTH
        scored = batch["labels"][:, 1:] != LABEL_IGNORE_INDEX
        nll = -_token_logprobs(input_ids, kind)[:, 1:]
        return float((nll * scored).sum()) / normalizer, normalizer
    if kind in ("dpo", "kto"):
        sequence_logps = (_token_logprobs(input_ids, kind) * batch["completion_mask"]).sum(dim=1)
        if kind == "dpo":
            chosen, rejected = sequence_logps.chunk(2)
            margin = (chosen - batch["ref_chosen_logps"]) - (rejected - batch["ref_rejected_logps"])
            return float(-F.logsigmoid(BETA * margin).mean()), ROWS / 2
        logratio = sequence_logps - batch["ref_logps"]
        desirable = torch.tensor(batch["label"])
        per_row = torch.where(
            desirable,
            DESIRABLE_WEIGHT * (1 - torch.sigmoid(BETA * logratio)),
            UNDESIRABLE_WEIGHT * torch.sigmoid(BETA * logratio),
        )
        return float(per_row.mean()), float(ROWS)
    pooled = _logits(input_ids, kind)[torch.arange(ROWS), _last_real_positions(input_ids)]
    if kind == "reward":
        chosen, rejected = pooled.squeeze(-1).chunk(2)
        return float(-F.logsigmoid(chosen - rejected).mean()), ROWS / 2
    return float(F.cross_entropy(pooled, batch["labels"])), float(ROWS)


class _ScoringSchedule(FakeSchedule):
    """With no target (a stage holding no loss) it scores nothing."""

    def __init__(self, kind: str):
        super().__init__(n_microbatches=MICROBATCHES)
        self._kind = kind

    def _step_microbatches(self, arg_mbs, kwarg_mbs, target_mbs, losses, return_outputs=False):
        super()._step_microbatches(arg_mbs, kwarg_mbs, target_mbs, [], return_outputs)
        if target_mbs is None:
            return
        for args, target in zip(arg_mbs, target_mbs, strict=True):
            losses.append(self._loss_fn(_logits(args[0], self._kind), target))

    def step(self, *args, target=None, losses=None, return_outputs=False, **kwargs):
        arg_mbs, kwarg_mbs = self._split_inputs(args, kwargs)
        target_mbs = None if target is None else list(torch.tensor_split(target, self._n_microbatches))
        self._step_microbatches(arg_mbs, kwarg_mbs, target_mbs, losses)


def _trainer(kind: str):
    """Built without ``__init__``."""
    cls = {
        "sft": DistributedSFTTrainer,
        "dpo": DistributedDPOTrainer,
        "kto": DistributedKTOTrainer,
        "reward": DistributedRewardTrainer,
        "classification": ClassificationTrainer,
    }[kind]
    trainer = object.__new__(cls)
    trainer.args = SimpleNamespace(device=torch.device("cpu"), max_length=SEQ, center_rewards_coefficient=None)
    trainer.is_deepspeed_enabled = False
    trainer.compute_loss_context_manager = contextlib.ExitStack
    trainer.processing_class = SimpleNamespace(pad_token_id=PAD)
    trainer._tokenizer = trainer.processing_class
    trainer.beta = BETA
    if kind == "dpo":
        trainer.loss_types, trainer.loss_weights = ["sigmoid"], [1.0]
    if kind == "kto":
        trainer.desirable_weight, trainer.undesirable_weight = DESIRABLE_WEIGHT, UNDESIRABLE_WEIGHT
    trainer.model = torch.nn.Module()
    trainer.model.config = PretrainedConfig(pad_token_id=PAD, num_labels=_head_width(kind))
    trainer.train_dataset = SimpleNamespace(column_names=["input_ids", "attention_mask"])
    if kind == "classification":
        trainer.is_multi_label = False
        args = ClassificationConfig(output_dir="unused", bf16=False, use_cpu=True)
        trainer._classification_loss = build_classification_loss(args, trainer.model, None, multi_label=False)
    return trainer


def _step_worker(rank: int, out_dir: str) -> None:
    config = ParallelismConfig(
        world_size=WORLD, gpus_per_node=STAGE_WIDTH, nvlink_domain_size=STAGE_WIDTH, pp_size=PP_SIZE
    )
    stage_group, chain_group = build_stage_group(config), build_pipeline_group(config)
    is_last = config.pp_rank == PP_SIZE - 1
    record = {}
    for kind in ADAPTERS:
        trainer = _trainer(kind)
        trainer.parallelism_config = config
        trainer._pp_stage_group, trainer._pp_chain_group = stage_group, chain_group
        trainer._pp_adapter = adapter = trainer._pp_loss_adapter()
        runner = fake_runner(_ScoringSchedule(kind))
        runner.is_last = is_last
        runner.loss = TokenLoss(adapter.token_loss_fn)
        runner.rows_per_example = type(trainer).FORWARD_ROWS_PER_EXAMPLE
        runner.schedule._loss_fn = runner._loss_fn
        trainer._pp_runner = runner

        loss = trainer._pp_training_step(_batch(kind, config.stage_local_rank))

        expected_loss, expected_normalizer = _expected(kind, config.stage_local_rank)
        record[kind] = {
            "loss": float(loss),
            "expected_loss": expected_loss,
            "normalizer": float(runner._num_items_in_batch),
            "expected_normalizer": expected_normalizer,
        }
    Path(out_dir, f"rank{rank}.json").write_text(json.dumps(record))


def test_a_pp_step_returns_each_objectives_last_stage_loss_on_every_rank(tmp_path):
    """The normalizer: paired objectives' pair count, KTO's and classification's rows, the causal-LM stage tokens
    over DP size."""
    run_gloo_ranks(_step_worker, WORLD, str(tmp_path))
    records = [json.loads(Path(tmp_path, f"rank{rank}.json").read_text()) for rank in range(WORLD)]

    mismatches = [
        f"rank {rank} {kind}: {field} {got[field]} vs {got['expected_' + field]}"
        for rank, record in enumerate(records)
        for kind, got in record.items()
        for field in ("loss", "normalizer")
        if got[field] != pytest.approx(got["expected_" + field], abs=TOLERANCE)
    ]
    assert not mismatches, "\n".join(mismatches)
    assert sorted(records[0]) == sorted(ADAPTERS)
    # Stage replicas hold different batches, so a loss read off the other cannot match.
    for kind in ADAPTERS:
        assert records[0][kind]["expected_loss"] != pytest.approx(records[1][kind]["expected_loss"]), kind


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
