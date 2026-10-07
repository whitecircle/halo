#!/usr/bin/env python
"""CPU test: SMPO's PP and non-PP losses score a bf16 model's tokens identically.

Both paths take their per-token log-probs from one fp32 primitive
(:func:`~src.distributed.pipeline_parallel.losses.next_token_logprobs`), so the same batch must give
the same loss and the same gradient whether it runs as one padded forward or as interleaved pipeline
microbatches, up to fp32 summation order. Agreement alone would also hold if both paths went back to
a bf16 ``log_softmax`` (TRL's ``selective_log_softmax`` on bf16 logits), which rounds every token's
log-prob to bf16's 8-bit mantissa, so both are also held to an fp64 oracle: the same pipeline over
the same bf16-rounded logits held in fp64.

The prediction both paths report is the per-pair ``[pairs, 2]`` (chosen, rejected) rewards, one row
per pair so the evaluation gather can cut a final round's padding pairs; the pipeline's must be the
non-PP step's. In eval the non-PP loss and metrics read the split's own pairs: a batch whose last
pairs pad the split's final round scores as its real pairs alone, while its rewards keep every row.

    python tests/cpu/trainers/test_smpo_logprob_parity.py
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.preference.smpo import SmoothMarginPOTrainer, _PPStepState

VOCAB = 64
PAD_TOKEN_ID = 0
MAX_LENGTH = 24
# fp32 summation order across microbatches; a bf16 log_softmax is ~1e-3 off on this batch.
REL_TOL = 1e-5


class _TokenTableLM(nn.Module):
    """bf16 logits looked up per token, so both paths see bit-identical logits for every position.

    The table itself is fp32, so its gradient accumulates in fp32 whatever row order a path uses.
    ``oracle`` hands back the same bf16-rounded logits in fp64.
    """

    def __init__(self, oracle: bool = False):
        super().__init__()
        generator = torch.Generator().manual_seed(0)
        self.table = nn.Parameter(4 * torch.randn(VOCAB, VOCAB, generator=generator))
        self.logits_dtype = torch.float64 if oracle else torch.bfloat16

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=None):
        return SimpleNamespace(logits=self.table[input_ids].to(torch.bfloat16).to(self.logits_dtype))


def _trainer() -> SmoothMarginPOTrainer:
    """A construction-free SMPO carrying what both loss paths read."""
    trainer = object.__new__(SmoothMarginPOTrainer)
    trainer.parallelism_config = ParallelismConfig()
    trainer.cp_config = None
    trainer.pad_token_id = PAD_TOKEN_ID
    trainer.padding_free = False
    trainer.lower_clip_percentile = trainer.upper_clip_percentile = None
    trainer.min_log_prob = -4.0
    trainer.beta = 2.0
    trainer.loss_type = "sigmoid"
    trainer.target_margin = 0.3
    trainer.use_margin_schedule = False
    trainer.chosen_sft_ratio = 0.7
    trainer.args = SimpleNamespace(max_length=MAX_LENGTH)
    trainer._pp_step_state = _PPStepState()
    return trainer


def _batch() -> dict[str, torch.Tensor]:
    """Four pairs with ragged prompts and completions."""
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
    """The pipeline's contract: one transform and step-state refresh over the batch, then a loss per
    microbatch, summed."""
    model = _TokenTableLM()
    trainer = _trainer()
    inputs = trainer._pp_batch_transform(batch)
    trainer._pp_refresh_step_state(inputs)
    loss = sum(
        trainer._pp_token_loss(model(input_ids).logits, labels)
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
    """Each microbatch hands the step its masked logit sums over the vocab; summed in the bf16 logits'
    own dtype they round each row's total to 8 mantissa bits, which this vocab's logits resolve."""
    model = _TokenTableLM()
    trainer = _trainer()
    inputs = trainer._pp_batch_transform(_batch())
    trainer._pp_refresh_step_state(inputs)
    for input_ids, labels in zip(inputs["input_ids"].split(2), inputs["labels"].split(2), strict=True):
        trainer._pp_token_loss(model(input_ids).logits, labels)
    metrics = trainer._pp_step_metrics()

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
    inputs = trainer._pp_batch_transform(batch)
    pipelined = trainer._pp_smpo_predictions(model(inputs["input_ids"]).logits, inputs)

    assert rewards.shape == pipelined.shape == (4, 2)
    torch.testing.assert_close(pipelined, rewards, rtol=REL_TOL, atol=REL_TOL)
    assert trainer._pp_smpo_eval_labels(inputs).shape == (4,)


@torch.no_grad()
def test_a_pipeline_step_of_padding_alone_scores_zero():
    """An eval step whose every pair pads the final round reaches the schedule as filler rows alone:
    the step state counts no pair, and the margin loss divides by 1 rather than by that 0."""
    trainer, model = _trainer(), _TokenTableLM()
    inputs = trainer._pp_batch_transform(_batch())
    trainer._pp_refresh_step_state({key: value[:0] for key, value in inputs.items()})
    filler = torch.full_like(inputs["labels"], -100)

    assert trainer._pp_step_state.pair_count == 1
    assert trainer._pp_token_loss(model(inputs["input_ids"]).logits, filler).item() == 0.0


def _eval_trainer(real_pairs: int) -> SmoothMarginPOTrainer:
    """An eval-mode SMPO whose batch's first ``real_pairs`` pairs are the split's own."""
    trainer = _trainer()
    trainer.eval_split_rows = lambda num_rows: real_pairs
    trainer._pp_runtime = None
    trainer._peft_has_been_casted_to_bf16 = False
    trainer.accelerator = SimpleNamespace(device=torch.device("cpu"))
    return trainer


@torch.no_grad()
def test_an_eval_batch_scores_its_real_pairs_alone():
    """Pairs 2 and 3 pad the eval split's final round; they repeat the split's first pairs, so read as
    real they would count those pairs twice. The loss and every metric are the two real pairs' own."""
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
    """Every pair pads the final round: the margin mean and the SFT terms mean over nothing, which is 0."""
    loss, metrics, rewards = _eval_trainer(0).get_batch_loss_metrics(_TokenTableLM(), _batch(), train_eval="eval")

    assert loss.item() == 0.0
    assert all(torch.isfinite(value) for value in metrics.values()), metrics
    assert rewards.shape == (4, 2)


@torch.no_grad()
def test_the_eval_step_predicts_per_pair_and_weighs_its_real_pairs():
    trainer = _eval_trainer(3)
    loss, predictions, labels = trainer.prediction_step(_TokenTableLM(), _batch(), prediction_loss_only=False)

    assert predictions.shape == (4, 2) and labels.shape == (4,)
    stored_rows = {rows for entries in trainer._stored_metrics["eval"].values() for _, rows in entries}
    assert stored_rows == {3}, f"eval metrics must weigh the batch's real pairs, got {stored_rows}"
    assert torch.isfinite(loss)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
