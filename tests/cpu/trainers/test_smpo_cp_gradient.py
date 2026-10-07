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
chunk mean would differ from the sequence's. A batch whose prompts the collator left-padded must
reach the CP forward with trailing padding only — the stand-in model refuses a left-padded batch as
the CP wrapper does — and still match the unsplit run.

    python tests/cpu/trainers/test_smpo_cp_gradient.py
"""

import copy
import warnings
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from src.distributed.context_parallel.config import split_sequence_for_cp
from src.distributed.context_parallel.wrapper import _reject_left_padding
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.preference.smpo import SmoothMarginPOTrainer
from tests.common.first_step import AUTOGRAD_FALLBACK_WARNING
from tests.common.gloo import run_gloo_ranks

CP_WORLD_SIZE = 2
VOCAB = 16
PAD_TOKEN_ID = 0
# The trainer sums log-probs and NLL in fp32 whatever the input dtype, so a chunked sum differs from
# the unsplit one by fp32 summation order (~1 ulp); a cp_size-fold gradient error is 0.5 or more.
REL_TOL = 1e-6
PER_SEQUENCE_KEYS = ("chosen_logps", "rejected_logps", "chosen_sft_loss", "rejected_sft_loss")


class _TokenTableLM(nn.Module):
    """Logits looked up per token; under CP it refuses a left-padded batch and forwards only this
    rank's chunk, like the CP wrapper."""

    def __init__(self):
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.table = nn.Parameter(torch.randn(VOCAB, VOCAB, generator=generator, dtype=torch.float64))
        self.cp_config = None

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=None):
        if self.cp_config is not None:
            _reject_left_padding(attention_mask)
            input_ids = split_sequence_for_cp(input_ids, self.cp_config)
        return SimpleNamespace(logits=self.table[input_ids])


def _trainer(parallelism_config: ParallelismConfig, cp_config) -> SmoothMarginPOTrainer:
    """A construction-free SMPO carrying what ``get_batch_loss_metrics`` reads."""
    trainer = object.__new__(SmoothMarginPOTrainer)
    trainer.parallelism_config = parallelism_config
    trainer.cp_config = cp_config
    trainer.pad_token_id = PAD_TOKEN_ID
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


def _batch(ragged_prompts: bool) -> dict[str, torch.Tensor]:
    """Two pairs with ragged completions, so the loss tokens split unevenly across the two chunks.

    ``ragged_prompts`` left-pads the second prompt the way ``DataCollatorForSMPO`` does, which the CP
    forward must turn into trailing padding.
    """
    generator = torch.Generator().manual_seed(1)

    def ids(rows, length):
        return torch.randint(1, VOCAB, (rows, length), generator=generator)

    prompt_mask = torch.tensor([[1] * 5, [0] * 2 + [1] * 3]) if ragged_prompts else torch.ones(2, 5, dtype=torch.long)
    return {
        "prompt_input_ids": ids(2, 5).masked_fill(prompt_mask == 0, PAD_TOKEN_ID),
        "prompt_attention_mask": prompt_mask,
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
        loss, metrics, _ = trainer.get_batch_loss_metrics(model, batch)
        loss.backward()
    return outputs, loss.detach(), model.table.grad.clone(), metrics, [str(w.message) for w in caught]


def _worker(rank: int, ragged_prompts: bool) -> None:
    batch = _batch(ragged_prompts)
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
    # Past the last collective, so a failing rank cannot strand its peer; each rank backpropagates a
    # different chunk and so judges its own.
    assert not failures, f"rank {rank}: " + "; ".join(failures)


@pytest.mark.parametrize("ragged_prompts", [False, True], ids=["equal-prompts", "left-padded-prompts"])
def test_cp_loss_and_gradient_match_the_unsplit_sequence(ragged_prompts):
    run_gloo_ranks(_worker, CP_WORLD_SIZE, ragged_prompts)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
