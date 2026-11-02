#!/usr/bin/env python
"""The CP forward reads no value of its inputs back to the host, so a CP step pays no device-to-host sync for its
input checks: the left-padding refusal runs on the collated batch instead (``test_cp_left_padding_guard.py``).

The probe runs the production ``UlyssesModel`` entries on meta-device inputs over a meta-device model. A meta
tensor holds no data, so any host read (``.cpu()``, ``.item()``, ``.tolist()``, ``bool()``) raises: on CPU it
stands in for the ``cudaStreamSynchronize`` such a read costs on a GPU. Both entries are probed, the labelled
training forward (SFT, SMPO) under the logits loss and the fused head loss, and ``forward_hidden_states`` (offline
GRPO's scorer), at the first, a middle and the final rank of cp2 and cp4 splits, since the boundary-aware loss
differs on the final rank. Liger has no meta kernel, so the fused run scores through a stand-in of its contract.

    python tests/cpu/parallelism/test_cp_forward_reads_no_input_value.py
"""

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from halo.distributed.context_parallel.sequence_chunks import reject_left_padding
from tests.common.cp_model import unpatched_cp_wrapper

VOCAB, HIDDEN, BATCH, SEQ = 16, 4, 2, 8
CP_POSITIONS = [(1, 0), (2, 0), (2, 1), (4, 2), (4, 3)]


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, HIDDEN, device="meta")

    def forward(self, input_ids, attention_mask=None, position_ids=None, use_cache=None):
        return SimpleNamespace(last_hidden_state=self.embed(input_ids))


class _CausalLM(nn.Module):
    """A meta-device causal LM: a forward through it computes shapes only."""

    def __init__(self):
        super().__init__()
        self.base_model = _Backbone()
        self.lm_head = nn.Linear(HIDDEN, VOCAB, device="meta")

    def forward(self, input_ids, attention_mask=None, labels=None, position_ids=None, **_kwargs):
        return CausalLMOutputWithPast(logits=self.lm_head(self.base_model(input_ids).last_hidden_state))


def _meta_tokens() -> torch.Tensor:
    return torch.empty(BATCH, SEQ, dtype=torch.long, device="meta")


def _fused_head_loss_stand_in(hidden: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The fused head loss's ``(summed loss, predicted tokens)`` contract, computed from shapes only."""
    return hidden.float().sum(), targets.clone()


@pytest.mark.parametrize("fused", [False, True], ids=["logits-loss", "fused-head-loss"])
@pytest.mark.parametrize(("cp_size", "cp_rank"), CP_POSITIONS)
def test_the_labelled_forward_reads_no_input_value(cp_size, cp_rank, fused):
    """SFT's and SMPO's entry: split, forward and the boundary-aware loss, with no host read of the mask,
    the ids or the labels."""
    wrapper = unpatched_cp_wrapper(_CausalLM(), cp_size=cp_size, cp_rank=cp_rank)
    if fused:
        # The cached head loss a run under fused_linear_cross_entropy resolves at its first training forward.
        wrapper.__dict__["_fused_head_loss"] = _fused_head_loss_stand_in
    outputs = wrapper(input_ids=_meta_tokens(), attention_mask=_meta_tokens(), labels=_meta_tokens())
    assert outputs["loss"].is_meta and outputs["loss"].dim() == 0
    if fused:
        assert outputs.logits is None and outputs["token_accuracy"].is_meta
    else:
        assert outputs.logits.shape == (BATCH, SEQ // cp_size, VOCAB)


@pytest.mark.parametrize(("cp_size", "cp_rank"), CP_POSITIONS)
def test_the_hidden_state_forward_reads_no_input_value(cp_size, cp_rank):
    """Offline GRPO's entry: this rank's chunk of backbone hidden states, with no host read of the mask."""
    wrapper = unpatched_cp_wrapper(_CausalLM(), cp_size=cp_size, cp_rank=cp_rank)
    hidden = wrapper.forward_hidden_states(input_ids=_meta_tokens(), attention_mask=_meta_tokens())
    assert hidden.is_meta and hidden.shape == (BATCH, SEQ // cp_size, HIDDEN)


def test_the_probe_catches_a_host_read():
    """Anti-vacuity: the left-padding check itself reads the mask, which a meta tensor refuses, so a forward
    that ran it would fail the probes above."""
    with pytest.raises((NotImplementedError, RuntimeError)):
        reject_left_padding(_meta_tokens())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
