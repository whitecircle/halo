#!/usr/bin/env python
"""SFT and SMPO's actual CP forwards retain boundary supervision and gradient normalization, and SMPO's
holds on a batch whose prompts the collator left-padded."""

import copy
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from transformers.modeling_outputs import CausalLMOutputWithPast

from halo.distributed.context_parallel.config import CPConfig
from halo.distributed.context_parallel.sequence_chunks import cp_local_targets, reject_left_padding
from halo.kernels.logprobs import LABEL_IGNORE_INDEX, next_token_logprobs
from tests.common.cp_model import unpatched_cp_wrapper
from tests.common.gloo import run_gloo_ranks
from tests.common.smpo_trainer import smpo_trainer

VOCAB, SEQ = 17, 12


class _LogitsModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(VOCAB, VOCAB)

    def forward(self, input_ids, **kwargs):
        return CausalLMOutputWithPast(logits=self.embedding(input_ids))


def _sync_gradient(parameter, cp_size):
    assert parameter.grad is not None
    dist.all_reduce(parameter.grad)
    parameter.grad /= cp_size
    return parameter.grad


def _drop_boundary(labels, cp_rank, cp_size):
    """This chunk's next-token targets with its last one, scored against the next chunk, ignored."""
    targets = cp_local_targets(labels, cp_rank, cp_size).clone()
    targets[:, -1] = LABEL_IGNORE_INDEX
    return targets


def _drop_boundary_logprobs(logits, labels, start=0):
    """SMPO's log-probs with each chunk's last position, the one scored against the next chunk, ignored."""
    logps, mask = next_token_logprobs(logits, labels, start)
    return logps, torch.cat([mask[:, :-1], torch.zeros_like(mask[:, -1:])], dim=1)


def _sft_worker(rank, cp_size):
    torch.manual_seed(67)
    baseline = _LogitsModel()
    ids = torch.randint(0, VOCAB, (2, SEQ))
    labels = torch.full_like(ids, LABEL_IGNORE_INDEX)
    # Only chunk-edge targets are supervised; their omission cannot hide inside a long-row mean.
    labels[:, SEQ // cp_size :: SEQ // cp_size] = ids[:, SEQ // cp_size :: SEQ // cp_size]
    mask = torch.ones_like(ids)
    expected = F.cross_entropy(
        baseline(ids).logits[:, :-1].reshape(-1, VOCAB),
        labels[:, 1:].reshape(-1),
        ignore_index=LABEL_IGNORE_INDEX,
    )
    expected.backward()
    config = CPConfig(cp_size, stage_world_size=cp_size, stage_base_rank=0, nvlink_domain_size=cp_size)
    for mutated in (False, True):
        model = copy.deepcopy(baseline)
        model.zero_grad()
        wrapper = unpatched_cp_wrapper(model, cp_config=config)
        if mutated:
            with patch("halo.distributed.context_parallel.ulysses_model.cp_local_targets", _drop_boundary):
                loss = wrapper(ids, attention_mask=mask, labels=labels).loss
        else:
            loss = wrapper(ids, attention_mask=mask, labels=labels).loss
        loss.backward()
        gradient = _sync_gradient(model.embedding.weight, cp_size)
        mean_loss = loss.detach().clone()
        dist.all_reduce(mean_loss)
        mean_loss /= cp_size
        if mutated:
            assert mean_loss == 0
            assert not torch.allclose(gradient, baseline.embedding.weight.grad)
        else:
            torch.testing.assert_close(mean_loss, expected)
            torch.testing.assert_close(gradient, baseline.embedding.weight.grad)
            with torch.no_grad():
                torch.testing.assert_close(wrapper(ids, attention_mask=mask, labels=labels).loss, expected)


def _smpo_host(config):
    return smpo_trainer(
        parallelism_config=SimpleNamespace(cp_size=config.cp_size if config is not None else 1), cp_config=config
    )


def _smpo_worker(rank, cp_size):
    torch.manual_seed(89)
    baseline = _LogitsModel()
    ids = torch.randint(1, VOCAB, (2, SEQ))
    prompt_width = SEQ // cp_size
    batch = {
        "prompt_input_ids": ids[:1, :prompt_width],
        "prompt_attention_mask": torch.ones_like(ids[:1, :prompt_width]),
        "chosen_input_ids": ids[:1, prompt_width:],
        "chosen_attention_mask": torch.ones_like(ids[:1, prompt_width:]),
        "rejected_input_ids": ids[1:, prompt_width:],
        "rejected_attention_mask": torch.ones_like(ids[1:, prompt_width:]),
    }
    reference = _smpo_host(None).concatenated_forward(baseline, batch)
    keys = ("chosen_logps", "rejected_logps", "chosen_sft_loss", "rejected_sft_loss")
    objective = reference["chosen_logps"].sum() - reference["rejected_logps"].sum()
    objective += 0.3 * reference["chosen_sft_loss"]
    objective.backward()
    config = CPConfig(cp_size, stage_world_size=cp_size, stage_base_rank=0, nvlink_domain_size=cp_size)
    for mutated in (False, True):
        model = copy.deepcopy(baseline)
        model.zero_grad()
        wrapper = unpatched_cp_wrapper(model, cp_config=config)
        host = _smpo_host(config)
        if mutated:
            with patch("halo.trainers.preference.smpo.next_token_logprobs", _drop_boundary_logprobs):
                actual = host.concatenated_forward(wrapper, batch)
        else:
            actual = host.concatenated_forward(wrapper, batch)
        loss = actual["chosen_logps"].sum() - actual["rejected_logps"].sum()
        loss += 0.3 * actual["chosen_sft_loss"]
        loss.backward()
        gradient = _sync_gradient(model.embedding.weight, cp_size)
        if mutated:
            assert not torch.allclose(actual["chosen_logps"], reference["chosen_logps"])
            assert not torch.allclose(gradient, baseline.embedding.weight.grad)
        else:
            for key in keys:
                torch.testing.assert_close(actual[key], reference[key])
            torch.testing.assert_close(gradient, baseline.embedding.weight.grad)


def _smpo_ragged_worker(rank, cp_size):
    """Two pairs whose prompts the collator left-pads to unequal lengths, through the real CP wrapper."""
    torch.manual_seed(97)
    baseline = _LogitsModel()
    prompt_mask = torch.tensor([[0, 0, 0, 1, 1], [1, 1, 1, 1, 1]])
    chosen_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]])
    rejected_mask = torch.tensor([[1, 1, 0], [1, 1, 1]])
    batch = {
        "prompt_input_ids": torch.randint(1, VOCAB, prompt_mask.shape) * prompt_mask,
        "prompt_attention_mask": prompt_mask,
        "chosen_input_ids": torch.randint(1, VOCAB, chosen_mask.shape) * chosen_mask,
        "chosen_attention_mask": chosen_mask,
        "rejected_input_ids": torch.randint(1, VOCAB, rejected_mask.shape) * rejected_mask,
        "rejected_attention_mask": rejected_mask,
    }
    reference = _smpo_host(None).concatenated_forward(baseline, batch)
    objective = reference["chosen_logps"].sum() - reference["rejected_logps"].sum()
    (objective + 0.3 * reference["chosen_sft_loss"]).backward()

    model = copy.deepcopy(baseline)
    model.zero_grad()
    config = CPConfig(cp_size, stage_world_size=cp_size, stage_base_rank=0, nvlink_domain_size=cp_size)
    wrapper = unpatched_cp_wrapper(model, cp_config=config)
    wrapper.register_forward_pre_hook(
        lambda _module, _args, kwargs: reject_left_padding(kwargs["attention_mask"]), with_kwargs=True
    )
    actual = _smpo_host(config).concatenated_forward(wrapper, batch)
    loss = actual["chosen_logps"].sum() - actual["rejected_logps"].sum()
    (loss + 0.3 * actual["chosen_sft_loss"]).backward()

    for key in reference:
        torch.testing.assert_close(actual[key], reference[key], msg=key)
    torch.testing.assert_close(_sync_gradient(model.embedding.weight, cp_size), baseline.embedding.weight.grad)


@pytest.mark.parametrize("cp_size", [2, 4])
def test_sft_forward_loss_and_gradients_need_boundary_targets(cp_size):
    run_gloo_ranks(_sft_worker, cp_size, cp_size)


@pytest.mark.parametrize("cp_size", [2, 4])
def test_smpo_concatenated_forward_loss_and_gradients_need_boundary_targets(cp_size):
    run_gloo_ranks(_smpo_worker, cp_size, cp_size)


@pytest.mark.parametrize("cp_size", [2, 4])
def test_smpo_left_padded_prompts_train_the_unsplit_objective(cp_size):
    """The CP attention ignores the mask, so SMPO must hand the wrapper every pad trailing, with labels
    flushed in step: a ragged batch then scores, and backpropagates, exactly as without CP."""
    run_gloo_ranks(_smpo_ragged_worker, cp_size, cp_size)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
