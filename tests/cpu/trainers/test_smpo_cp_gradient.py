#!/usr/bin/env python
"""SMPO under context parallelism trains the unsplit objective: same loss, same gradient.

Each CP rank holds one sequence chunk and all-reduces its partial per-sequence log-prob and NLL sums,
so every rank computes the same loss. Those sum reduces must be autograd-aware: their backward sums
the gradient over the CP group, which FSDP2's world-wide gradient average then divides back by
``cp_size``. An in-place ``dist.all_reduce`` on a grad-carrying tensor instead reaches autograd
through PyTorch's deprecated c10d fallback — identity backward plus an "autograd kernel was not
registered" warning — and trains on ``1/cp_size`` of the gradient unless the loss is rescaled. The
percentile clip's all-gather must likewise see only detached values.

This drives the real ``get_batch_loss_metrics`` over a 2-rank gloo CP group in float64 and compares
the per-sequence log-probs, the loss, the FSDP-averaged gradient and every logged metric with a
``cp_size=1`` run of the same batch. The ``logits/*`` means are logging-only reduces: each rank's
chunk mean would differ from the sequence's.

    python tests/cpu/trainers/test_smpo_cp_gradient.py
"""

import copy
import os
import warnings
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from src.distributed.context_parallel.config import split_sequence_for_cp
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.preference.smpo import SmoothMarginPOTrainer
from tests.common.ports import free_port

CP_WORLD_SIZE = 2
VOCAB = 16
PAD_TOKEN_ID = 0
# The trainer sums log-probs and NLL in fp32 whatever the input dtype, so a chunked sum differs from
# the unsplit one by fp32 summation order (~1 ulp); a cp_size-fold gradient error is 0.5 or more.
REL_TOL = 1e-6
AUTOGRAD_FALLBACK_WARNING = "autograd kernel was not registered"
PER_SEQUENCE_KEYS = ("chosen_logps", "rejected_logps", "chosen_sft_loss", "rejected_sft_loss")


class _TokenTableLM(nn.Module):
    """Logits looked up per token; under CP it forwards only this rank's chunk, like the CP wrapper."""

    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.table = nn.Parameter(torch.randn(VOCAB, VOCAB, generator=generator, dtype=torch.float64))
        self.cp_config = None

    def forward(self, input_ids, attention_mask=None, use_cache=None):
        if self.cp_config is not None:
            input_ids = split_sequence_for_cp(input_ids, self.cp_config)
        return SimpleNamespace(logits=self.table[input_ids])


def _trainer(parallelism_config: ParallelismConfig, cp_config) -> SmoothMarginPOTrainer:
    """A construction-free SMPO carrying what ``get_batch_loss_metrics`` reads."""
    trainer = object.__new__(SmoothMarginPOTrainer)
    trainer.parallelism_config = parallelism_config
    trainer.cp_config = cp_config
    trainer.pad_token_id = PAD_TOKEN_ID
    trainer.label_pad_token_id = -100
    trainer.padding_free = False
    # Clipping on, so the CP quantile's all-gather runs inside the backward-carrying forward.
    trainer.lower_clip_percentile = 0.25
    trainer.upper_clip_percentile = 0.9
    trainer.min_log_prob = -3.0
    trainer.beta = 2.0
    trainer.loss_type = "sigmoid"
    trainer.target_margin = 0.3
    trainer.use_margin_schedule = False
    trainer.chosen_sft_ratio = 0.7
    return trainer


def _batch() -> dict[str, torch.Tensor]:
    """Two pairs with ragged completions, so the loss tokens split unevenly across the two chunks."""
    generator = torch.Generator().manual_seed(1)

    def ids(rows, length):
        return torch.randint(1, VOCAB, (rows, length), generator=generator)

    return {
        "prompt_input_ids": ids(2, 5),
        "prompt_attention_mask": torch.ones(2, 5, dtype=torch.long),
        "chosen_input_ids": ids(2, 6),
        "chosen_attention_mask": torch.tensor([[1] * 6, [1] * 4 + [0] * 2]),
        "rejected_input_ids": ids(2, 7),
        "rejected_attention_mask": torch.tensor([[1] * 7, [1] * 3 + [0] * 4]),
    }


def _loss_and_grad(trainer: SmoothMarginPOTrainer, model: _TokenTableLM, batch: dict) -> tuple:
    """Per-sequence outputs, the loss, the table gradient, the logged metrics and the warnings the
    backward raised."""
    outputs = trainer.concatenated_forward(model, batch)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        loss, metrics = trainer.get_batch_loss_metrics(model, batch)
        loss.backward()
    return outputs, loss.detach(), model.table.grad.clone(), metrics, [str(w.message) for w in caught]


def _worker(rank: int, out_path: str, port: int) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port), RANK=str(rank), WORLD_SIZE=str(CP_WORLD_SIZE))
    dist.init_process_group("gloo", rank=rank, world_size=CP_WORLD_SIZE)
    try:
        batch = _batch()
        reference_model = _TokenTableLM()
        cp_model = copy.deepcopy(reference_model)

        reference = _trainer(ParallelismConfig(world_size=CP_WORLD_SIZE, gpus_per_node=CP_WORLD_SIZE), None)
        ref_outputs, ref_loss, ref_grad, ref_metrics, _ = _loss_and_grad(reference, reference_model, batch)

        parallelism_config = ParallelismConfig(
            cp_size=CP_WORLD_SIZE, world_size=CP_WORLD_SIZE, gpus_per_node=CP_WORLD_SIZE
        )
        cp_config = parallelism_config.create_cp_config()
        cp_model.cp_config = cp_config
        cp_outputs, cp_loss, cp_grad, cp_metrics, cp_warnings = _loss_and_grad(
            _trainer(parallelism_config, cp_config), cp_model, batch
        )
        # FSDP2 averages every gradient over the whole world, CP ranks included (DP=1 here).
        dist.all_reduce(cp_grad, op=dist.ReduceOp.SUM)
        cp_grad /= CP_WORLD_SIZE

        compared = {key: (cp_outputs[key], ref_outputs[key]) for key in PER_SEQUENCE_KEYS}
        compared["loss"] = (cp_loss, ref_loss)
        compared["FSDP-averaged gradient"] = (cp_grad, ref_grad)
        assert cp_metrics.keys() == ref_metrics.keys()
        compared.update({f"metric {key}": (cp_metrics[key], ref_metrics[key]) for key in ref_metrics})
        failures = []
        for name, (cp_value, ref_value) in compared.items():
            # Absolute where the reference is exactly zero (this batch's reward accuracy).
            scale = ref_value.double().norm().item() or 1.0
            rel_err = (cp_value.double() - ref_value.double()).norm().item() / scale
            if not rel_err < REL_TOL:
                ratio = cp_value.double().norm().item() / scale
                failures.append(f"{name}: rel_err={rel_err:.3e} vs the unsplit sequence (||cp||/||ref||={ratio:.4f})")
        fallback = [message for message in cp_warnings if AUTOGRAD_FALLBACK_WARNING in message]
        if fallback:
            failures.append(f"backward went through the c10d autograd fallback: {fallback[0]}")

        # Every rank writes its own verdict: each backpropagates a different chunk.
        with open(f"{out_path}.{rank}", "w") as fh:
            fh.write("PASS" if not failures else "FAIL: " + "; ".join(failures))
    finally:
        dist.destroy_process_group()


def test_cp_loss_and_gradient_match_the_unsplit_sequence(tmp_path):
    out = str(tmp_path / "result.txt")
    mp.start_processes(_worker, args=(out, free_port()), nprocs=CP_WORLD_SIZE, join=True, start_method="spawn")
    results = {}
    for rank in range(CP_WORLD_SIZE):
        with open(f"{out}.{rank}") as fh:
            results[rank] = fh.read()
    assert set(results.values()) == {"PASS"}, results


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
