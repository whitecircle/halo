#!/usr/bin/env python
"""DPO and KTO sum their sequence log-probs in fp32, on TRL's loss path and in the reference sweep.
TRL sums in bf16, whose grid at ``|logp|`` in [8192, 16384) is 64 nats, missing float64 by tens of nats.
"""

from collections import defaultdict
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
import trl.trainer.dpo_trainer as trl_dpo
from datasets import Dataset
from trl.trainer.dpo_trainer import DataCollatorForPreference
from trl.trainer.kto_trainer import DataCollatorForUnpairedPreference
from trl.trainer.utils import selective_log_softmax, selective_log_softmax_and_entropy

from tests.common.preference_precompute import column, precompute_trainer

VOCAB = 256
PROMPT_TOKENS = 6
# At ~6 nats per token (random targets under unit-scale logits over 256 tokens) both sums sit in the 64-nat band.
LONG_COMPLETIONS = (1700, 1500)
TOLERANCE_NATS = 1e-2
BETA = 0.1


class TableLM(torch.nn.Module):
    """Logits at a position are a fixed row picked by the token there, exact in bf16, so the float64 recomputation sees
    the logits the trainer saw."""

    is_gradient_checkpointing = False

    def __init__(self, seed: int, scale: float = 1.0):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        table = scale * torch.randn(VOCAB, VOCAB, generator=generator)
        self.table = torch.nn.Parameter(table.bfloat16().float())

    def forward(self, input_ids, attention_mask=None, **kwargs):
        return SimpleNamespace(logits=self.table[input_ids].to(torch.bfloat16))


def _token_ids(seed: int, n: int) -> list[int]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(1, VOCAB, (n,), generator=generator).tolist()


def _float64_token_logps(table: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """``[B, T-1]`` log-probs of each next token."""
    logits = table[input_ids[:, :-1]]
    return torch.log_softmax(logits, dim=-1).gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)


def _float64_sequence_logps(model: TableLM, input_ids, completion_mask, weights=None) -> torch.Tensor:
    """Completion-masked float64 sums; ``weights`` reweights each scored position (``ld_alpha``)."""
    scored = completion_mask[:, 1:].double() if weights is None else weights
    return (_float64_token_logps(model.table.detach().double(), input_ids) * scored).sum(dim=-1)


def _ld_alpha_weights(completion_mask: torch.Tensor, ld_alpha: float) -> torch.Tensor:
    """LD-DPO's per-position weight: 1 over the pair's shared completion length, ``ld_alpha`` past it."""
    mask = completion_mask[:, 1:].double()
    position = mask.cumsum(dim=1)
    chosen_len, rejected_len = mask.sum(dim=1).chunk(2)
    shared = torch.minimum(chosen_len, rejected_len).repeat(2).unsqueeze(1)
    return mask * torch.where(position <= shared, 1.0, ld_alpha)


def _trainer(kind: str, **kwargs):
    """The shared precompute trainer, running its own reference pass instead of the helper's stub."""
    trainer = precompute_trainer(kind, **kwargs)
    del trainer.compute_ref_log_probs
    return trainer


def _loss_trainer(kind: str, policy: TableLM, reference: TableLM, **attrs):
    trainer = _trainer(kind, ref_model=reference)
    trainer.model = policy
    trainer.accelerator.gather = trainer.accelerator.gather_for_metrics
    trainer.beta = BETA
    trainer.precompute_ref_logps = False
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer.aux_loss_enabled = False
    trainer._tp_size = 1
    trainer._total_train_tokens = 0
    if kind == "dpo":
        trainer.use_weighting = False
        trainer.f_divergence_type = "reverse_kl"
        trainer.loss_types = ["sigmoid"]
        trainer.loss_weights = [1.0]
    else:
        trainer.loss_type = "kto"
        trainer.desirable_weight = 1.0
        trainer.undesirable_weight = 1.0
    for name, value in attrs.items():
        setattr(trainer, name, value)
    return trainer


def _dpo_rows(completions=LONG_COMPLETIONS) -> list[dict]:
    chosen_len, rejected_len = completions
    return [
        {
            "prompt_ids": _token_ids(0, PROMPT_TOKENS),
            "chosen_ids": _token_ids(1, chosen_len),
            "rejected_ids": _token_ids(2, rejected_len),
        }
    ]


def _kto_rows() -> list[dict]:
    """A desirable and an undesirable row; each one's KL completion is the other's completion."""
    completions = [_token_ids(3 + i, n) for i, n in enumerate(LONG_COMPLETIONS)]
    prompts = [_token_ids(10 + i, PROMPT_TOKENS) for i in range(len(completions))]
    return [
        {
            "prompt_ids": prompts[i],
            "completion_ids": completions[i],
            "KL_completion_ids": completions[1 - i],
            "label": i == 0,
        }
        for i in range(len(completions))
    ]


def _assert_nats_close(actual, expected, what: str) -> None:
    actual, expected = torch.as_tensor(actual, dtype=torch.float64), torch.as_tensor(expected, dtype=torch.float64)
    error = float((actual - expected).abs().max())
    assert error < TOLERANCE_NATS, f"{what}: off the float64 recomputation by {error:.3f} nats"


@pytest.mark.parametrize(
    "variant", [{}, {"ld_alpha": 0.5}, {"use_weighting": True}], ids=["plain", "ld_alpha", "use_weighting"]
)
def test_dpo_loss_sequence_logps_match_float64(variant):
    policy, reference = TableLM(seed=0), TableLM(seed=1)
    trainer = _loss_trainer("dpo", policy, reference, **variant)
    batch = DataCollatorForPreference(pad_token_id=0)(_dpo_rows())
    ids, completion_mask = batch["input_ids"], batch["completion_mask"]
    weights = _ld_alpha_weights(completion_mask, variant["ld_alpha"]) if "ld_alpha" in variant else None

    loss = trainer._compute_loss(policy, batch, return_outputs=False)

    assert torch.isfinite(loss)
    chosen, rejected = _float64_sequence_logps(policy, ids, completion_mask, weights).chunk(2)
    ref_chosen, ref_rejected = _float64_sequence_logps(reference, ids, completion_mask, weights).chunk(2)
    assert 8192 <= -float(chosen) < 16384 and 8192 <= -float(rejected) < 16384, "premise: the 64-nat bf16 grid"
    metrics = trainer._metrics["train"]
    _assert_nats_close(metrics["logps/chosen"], chosen, "policy chosen log-prob")
    _assert_nats_close(metrics["logps/rejected"], rejected, "policy rejected log-prob")
    _assert_nats_close([m / BETA for m in metrics["rewards/chosen"]], chosen - ref_chosen, "chosen log-ratio")
    _assert_nats_close([m / BETA for m in metrics["rewards/rejected"]], rejected - ref_rejected, "rejected log-ratio")


def test_dpo_loss_gradient_matches_float64():
    """The policy gradient through the fp32 log-probs, TRL's in-place completion masking included."""
    policy, reference = TableLM(seed=0), TableLM(seed=1)
    trainer = _loss_trainer("dpo", policy, reference)
    batch = DataCollatorForPreference(pad_token_id=0)(_dpo_rows(completions=(40, 31)))
    ids, completion_mask = batch["input_ids"], batch["completion_mask"]

    trainer._compute_loss(policy, batch, return_outputs=False).backward()

    table = policy.table.detach().double().requires_grad_(True)
    chosen, rejected = (_float64_token_logps(table, ids) * completion_mask[:, 1:]).sum(dim=-1).chunk(2)
    ref_chosen, ref_rejected = _float64_sequence_logps(reference, ids, completion_mask).chunk(2)
    (-F.logsigmoid(BETA * ((chosen - ref_chosen) - (rejected - ref_rejected)))).mean().backward()
    torch.testing.assert_close(policy.table.grad.double(), table.grad, rtol=1e-2, atol=1e-5)


def test_kto_loss_sequence_logps_match_float64():
    policy, reference = TableLM(seed=0), TableLM(seed=1, scale=3.0)
    trainer = _loss_trainer("kto", policy, reference)
    batch = DataCollatorForUnpairedPreference(pad_token_id=0)(_kto_rows())

    loss = trainer._compute_loss(policy, batch, return_outputs=False)

    assert torch.isfinite(loss)
    ids, mask = batch["input_ids"], batch["completion_mask"]
    desirable, undesirable = _float64_sequence_logps(policy, ids, mask)
    ref_desirable, ref_undesirable = _float64_sequence_logps(reference, ids, mask)
    kl_ids, kl_mask = batch["KL_input_ids"], batch["KL_completion_mask"]
    policy_kl, ref_kl = (_float64_sequence_logps(model, kl_ids, kl_mask) for model in (policy, reference))
    kl = (policy_kl - ref_kl).mean()
    assert kl > 0, "premise: an unclamped KL term, so the metric carries its value"
    metrics = trainer._metrics["train"]
    _assert_nats_close(metrics["logps/chosen"], desirable, "policy desirable log-prob")
    _assert_nats_close(metrics["logps/rejected"], undesirable, "policy undesirable log-prob")
    _assert_nats_close([m / BETA for m in metrics["rewards/chosen"]], desirable - ref_desirable, "desirable log-ratio")
    _assert_nats_close(
        [m / BETA for m in metrics["rewards/rejected"]], undesirable - ref_undesirable, "undesirable log-ratio"
    )
    _assert_nats_close(metrics["kl"], kl, "KL term")


@pytest.mark.parametrize("name", ["selective_log_softmax", "selective_log_softmax_and_entropy"])
def test_a_trl_that_no_longer_reads_its_log_prob_functions_is_refused(monkeypatch, name):
    """Were TRL to bind another function under the rebound names, the loss would keep its bf16 sums unnoticed."""
    monkeypatch.setattr(trl_dpo, name, lambda logits, index, **kwargs: logits[..., 0])
    policy = TableLM(seed=0)
    trainer = _loss_trainer("dpo", policy, TableLM(seed=1))
    batch = DataCollatorForPreference(pad_token_id=0)(_dpo_rows(completions=(4, 3)))

    with pytest.raises(RuntimeError, match=name):
        trainer._compute_loss(policy, batch, return_outputs=False)


def test_trl_log_softmax_is_restored_when_the_loss_raises():
    class Raising(TableLM):
        def forward(self, input_ids, attention_mask=None, **kwargs):
            raise RuntimeError("forward failed")

    policy = Raising(seed=0)
    trainer = _loss_trainer("dpo", policy, TableLM(seed=1))
    batch = DataCollatorForPreference(pad_token_id=0)(_dpo_rows(completions=(4, 3)))

    with pytest.raises(RuntimeError, match="forward failed"):
        trainer._compute_loss(policy, batch, return_outputs=False)
    assert trl_dpo.selective_log_softmax is selective_log_softmax
    assert trl_dpo.selective_log_softmax_and_entropy is selective_log_softmax_and_entropy


def test_a_reference_pass_nested_in_the_loss_stays_fp32():
    reference = TableLM(seed=1)
    ref_batch = DataCollatorForPreference(pad_token_id=0)(_dpo_rows())
    nested = []

    class Nesting(TableLM):
        def forward(self, input_ids, attention_mask=None, **kwargs):
            if not nested:
                nested.append(trainer.compute_ref_log_probs(reference, ref_batch))
            return super().forward(input_ids, attention_mask, **kwargs)

    policy = Nesting(seed=0)
    trainer = _loss_trainer("dpo", policy, reference)
    trainer._compute_loss(policy, DataCollatorForPreference(pad_token_id=0)(_dpo_rows(completions=(4, 3))), False)

    want = _float64_sequence_logps(reference, ref_batch["input_ids"], ref_batch["completion_mask"]).chunk(2)
    for got, expected, what in zip(nested[0], want, ("chosen", "rejected"), strict=True):
        _assert_nats_close(got, expected, f"nested reference {what} log-prob")
    assert trl_dpo.selective_log_softmax is selective_log_softmax
    assert trl_dpo.selective_log_softmax_and_entropy is selective_log_softmax_and_entropy


def _swept(kind: str, rows: list[dict], collator) -> tuple[Dataset, TableLM, object]:
    trainer = _trainer(kind)
    trainer.model = TableLM(seed=0)
    trainer.data_collator = collator
    swept = trainer._precompute_ref_logps(Dataset.from_list(rows), "train", len(rows))
    return swept, trainer.model, collator(rows)


def test_dpo_reference_sweep_matches_float64():
    rows = _dpo_rows() + _dpo_rows(completions=tuple(reversed(LONG_COMPLETIONS)))
    swept, model, batch = _swept("dpo", rows, DataCollatorForPreference(pad_token_id=0))

    want_chosen, want_rejected = _float64_sequence_logps(model, batch["input_ids"], batch["completion_mask"]).chunk(2)
    _assert_nats_close(column(swept, "ref_chosen_logps"), want_chosen, "swept ref_chosen_logps")
    _assert_nats_close(column(swept, "ref_rejected_logps"), want_rejected, "swept ref_rejected_logps")


def test_kto_reference_sweep_matches_float64():
    rows = _kto_rows()
    swept, model, batch = _swept("kto", rows, DataCollatorForUnpairedPreference(pad_token_id=0))

    want = _float64_sequence_logps(model, batch["input_ids"], batch["completion_mask"])
    want_kl = _float64_sequence_logps(model, batch["KL_input_ids"], batch["KL_completion_mask"])
    _assert_nats_close(column(swept, "ref_logps"), want, "swept ref_logps")
    _assert_nats_close(column(swept, "ref_KL_logps"), want_kl, "swept ref_KL_logps")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
