#!/usr/bin/env python
"""SMPO's PP and non-PP losses score a bf16 model's tokens identically, and both match an fp64 oracle.
Agreement alone would also hold if both went back to a bf16 ``log_softmax``, which rounds log-probs to 8 mantissa bits.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from halo.trainers.preference.smpo import SMPOTrainer
from halo.trainers.preference.smpo_pipeline_loss import SMPOPipelineLoss
from tests.common.smpo_trainer import smpo_trainer

VOCAB = 64
PAD_TOKEN_ID = 0
MAX_LENGTH = 24
# fp32 summation order across microbatches; a bf16 log_softmax is ~1e-3 off on this batch.
REL_TOL = 1e-5


class _TokenTableLM(nn.Module):
    """bf16 logits looked up per token, so both paths see bit-identical logits.
    The table is fp32, so its gradient accumulates in fp32; ``oracle`` returns the same bf16-rounded logits in fp64."""

    def __init__(self, oracle: bool = False):
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.table = nn.Parameter(4 * torch.randn(VOCAB, VOCAB, generator=generator))
        self.logits_dtype = torch.float64 if oracle else torch.bfloat16

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=None):
        return SimpleNamespace(logits=self.table[input_ids].to(torch.bfloat16).to(self.logits_dtype))


def _trainer() -> SMPOTrainer:
    return smpo_trainer(pad_token_id=PAD_TOKEN_ID, min_log_prob=-4.0, max_length=MAX_LENGTH)


def _pipeline(trainer: SMPOTrainer) -> SMPOPipelineLoss:
    """The pipeline loss the trainer's adapter hook builds."""
    trainer._pp_loss_adapter()
    return trainer._pp_loss


def _batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)

    def ids(length):
        return torch.randint(1, VOCAB, (4, length), generator=generator)

    def lengths(width, *kept):
        return torch.tensor([[1] * k + [0] * (width - k) for k in kept])

    return {
        "prompt_input_ids": ids(6),
        "prompt_attention_mask": lengths(6, 6, 4, 5, 6).flip(1),
        "chosen_input_ids": ids(8),
        "chosen_attention_mask": lengths(8, 8, 5, 7, 3),
        "rejected_input_ids": ids(7),
        "rejected_attention_mask": lengths(7, 4, 7, 6, 2),
    }


def _non_pp_loss_and_grad(batch, oracle: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    model = _TokenTableLM(oracle)
    loss, _, _ = _trainer().get_batch_loss_metrics(model, batch)
    loss.backward()
    return loss.detach().double(), model.table.grad.double()


def _pp_loss_and_grad(batch, rows_per_microbatch: int) -> tuple[torch.Tensor, torch.Tensor]:
    """The pipeline's contract: one transform and state refresh over the batch, then a loss per microbatch, summed."""
    model = _TokenTableLM()
    pipeline = _pipeline(_trainer())
    inputs = pipeline.batch_transform(batch)
    pipeline.refresh_constants(inputs)
    loss = sum(
        pipeline.token_loss(model(input_ids).logits, labels)
        for input_ids, labels in zip(
            inputs["input_ids"].split(rows_per_microbatch), inputs["labels"].split(rows_per_microbatch), strict=True
        )
    )
    loss.backward()
    return loss.detach().double(), model.table.grad.double()


def _rel_err(actual: torch.Tensor, expected: torch.Tensor) -> float:
    return ((actual - expected).norm() / expected.norm()).item()


@pytest.mark.parametrize("rows_per_microbatch", [8, 2])
def test_pp_and_non_pp_losses_agree_on_a_bf16_model(rows_per_microbatch):
    batch = _batch()
    non_pp_loss, non_pp_grad = _non_pp_loss_and_grad(batch)
    pp_loss, pp_grad = _pp_loss_and_grad(batch, rows_per_microbatch)

    assert _rel_err(pp_loss, non_pp_loss) < REL_TOL, f"loss: PP {pp_loss.item()} vs non-PP {non_pp_loss.item()}"
    assert _rel_err(pp_grad, non_pp_grad) < REL_TOL, f"gradient rel_err {_rel_err(pp_grad, non_pp_grad):.3e}"

    oracle_loss, oracle_grad = _non_pp_loss_and_grad(batch, oracle=True)
    assert _rel_err(non_pp_loss, oracle_loss) < REL_TOL, f"loss: {non_pp_loss.item()} vs fp64 {oracle_loss.item()}"
    assert _rel_err(non_pp_grad, oracle_grad) < REL_TOL, f"gradient rel_err {_rel_err(non_pp_grad, oracle_grad):.3e}"


@torch.no_grad()
def test_the_pp_logit_means_accumulate_in_fp32():
    """Summed in the bf16 logits' own dtype, the masked logit sums would round each row's total to 8 mantissa bits,
    which this vocab's logits resolve."""
    model = _TokenTableLM()
    pipeline = _pipeline(_trainer())
    inputs = pipeline.batch_transform(_batch())
    pipeline.refresh_constants(inputs)
    for input_ids, labels in zip(inputs["input_ids"].split(2), inputs["labels"].split(2), strict=True):
        pipeline.token_loss(model(input_ids).logits, labels)
    metrics = pipeline.step_metrics()

    logits = model(inputs["input_ids"]).logits.double()[:, :-1]
    tokens = inputs["labels"][:, 1:] != -100
    for side, rows in (("chosen", slice(0, None, 2)), ("rejected", slice(1, None, 2))):
        expected = (logits[rows].sum(-1) * tokens[rows]).sum() / (tokens[rows].sum() * VOCAB)
        assert _rel_err(metrics[f"logits/{side}"].double(), expected) < REL_TOL, side


@torch.no_grad()
def test_the_pipeline_predicts_the_non_pp_rewards():
    model, batch = _TokenTableLM(), _batch()
    _, _, rewards = _trainer().get_batch_loss_metrics(model, batch)
    trainer = _trainer()
    adapter = trainer._pp_loss_adapter()
    inputs = adapter.batch_transform(batch)
    pipelined = adapter.predictions_fn(model(inputs["input_ids"]).logits, inputs)

    assert rewards.shape == pipelined.shape == (4, 2)
    torch.testing.assert_close(pipelined, rewards, rtol=REL_TOL, atol=REL_TOL)
    assert adapter.eval_labels_fn(inputs).shape == (4,)


@torch.no_grad()
def test_a_pipeline_step_of_padding_alone_scores_zero():
    """The step state counts no pair, and the margin loss divides by 1 rather than by that 0."""
    pipeline, model = _pipeline(_trainer()), _TokenTableLM()
    inputs = pipeline.batch_transform(_batch())
    pipeline.refresh_constants({key: value[:0] for key, value in inputs.items()})
    filler = torch.full_like(inputs["labels"], -100)

    assert pipeline.constants.pair_count == 1
    assert pipeline.token_loss(model(inputs["input_ids"]).logits, filler).item() == 0.0


def _eval_trainer(real_pairs: int) -> SMPOTrainer:
    """An eval-mode SMPO whose batch's first ``real_pairs`` pairs are the split's own."""
    trainer = _trainer()
    trainer.eval_split_rows = lambda num_rows: real_pairs
    trainer._pp_runner = None
    trainer._peft_has_been_casted_to_bf16 = False
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
    return trainer


@torch.no_grad()
def test_an_eval_batch_scores_its_real_pairs_alone():
    """Pairs 2 and 3 pad the final round by repeating the first pairs, which would count twice if read as real."""
    model, batch = _TokenTableLM(), _batch()
    alone = {key: value[:2] for key, value in batch.items()}
    loss, metrics, rewards = _eval_trainer(2).get_batch_loss_metrics(model, batch, train_eval="eval")
    alone_loss, alone_metrics, alone_rewards = _eval_trainer(2).get_batch_loss_metrics(model, alone, train_eval="eval")

    torch.testing.assert_close(loss, alone_loss, rtol=REL_TOL, atol=REL_TOL)
    assert metrics.keys() == alone_metrics.keys()
    for key, value in metrics.items():
        torch.testing.assert_close(value, alone_metrics[key], rtol=REL_TOL, atol=REL_TOL, msg=key)
    assert rewards.shape == (4, 2), "the rewards keep every pair, the shape the evaluation gather cuts"
    torch.testing.assert_close(rewards[:2], alone_rewards, rtol=REL_TOL, atol=REL_TOL)


@torch.no_grad()
def test_a_rank_of_eval_padding_alone_scores_zero():
    """The margin mean and the SFT terms mean over nothing, which is 0."""
    loss, metrics, rewards = _eval_trainer(0).get_batch_loss_metrics(_TokenTableLM(), _batch(), train_eval="eval")

    assert loss.item() == 0.0
    assert all(torch.isfinite(value) for value in metrics.values()), metrics
    assert rewards.shape == (4, 2)


@torch.no_grad()
def test_the_eval_step_predicts_per_pair_and_weighs_its_real_pairs():
    trainer = _eval_trainer(3)
    stored_rows = set()
    trainer.store_metrics = lambda metrics, train_eval="train", rows=1: stored_rows.add(rows)
    loss, predictions, labels = trainer.prediction_step(_TokenTableLM(), _batch(), prediction_loss_only=False)

    assert predictions.shape == (4, 2) and labels.shape == (4,)
    assert stored_rows == {3}, f"eval metrics must weigh the batch's real pairs, got {stored_rows}"
    assert torch.isfinite(loss)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
