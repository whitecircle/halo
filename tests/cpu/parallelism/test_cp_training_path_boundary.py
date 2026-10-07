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

from src.data.spans import LABEL_IGNORE_INDEX
from src.distributed.context_parallel.config import CPConfig
from src.distributed.pipeline_parallel.losses import next_token_logprobs
from src.trainers.preference.smpo import SmoothMarginPOTrainer
from tests.common.cp_wrapper import unpatched_cp_wrapper
from tests.common.gloo import run_gloo_ranks

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


def _drop_boundary(logits, labels, cp_rank, cp_size):
    chunk = labels.size(1) // cp_size
    return logits[:, :-1], labels[:, cp_rank * chunk + 1 : (cp_rank + 1) * chunk]


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
    config = CPConfig(cp_size=cp_size, world_size=cp_size, gpus_per_node=cp_size)
    for mutated in (False, True):
        model = copy.deepcopy(baseline)
        model.zero_grad()
        wrapper = unpatched_cp_wrapper(model, cp_config=config)
        if mutated:
            with patch("src.distributed.context_parallel.wrapper.cp_shift_against_full_labels", _drop_boundary):
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
    host = object.__new__(SmoothMarginPOTrainer)
    host.cp_config = config
    host.parallelism_config = SimpleNamespace(cp_size=config.cp_size if config is not None else 1)
    host.pad_token_id = 0
    host.padding_free = False
    host.lower_clip_percentile = host.upper_clip_percentile = host.min_log_prob = None
    return host


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
    config = CPConfig(cp_size=cp_size, world_size=cp_size, gpus_per_node=cp_size)
    for mutated in (False, True):
        model = copy.deepcopy(baseline)
        model.zero_grad()
        wrapper = unpatched_cp_wrapper(model, cp_config=config)
        host = _smpo_host(config)
        if mutated:
            with patch("src.trainers.preference.smpo.next_token_logprobs", _drop_boundary_logprobs):
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
    config = CPConfig(cp_size=cp_size, world_size=cp_size, gpus_per_node=cp_size)
    actual = _smpo_host(config).concatenated_forward(unpatched_cp_wrapper(model, cp_config=config), batch)
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
    """The wrapper refuses a left-padded batch, so SMPO must hand it every pad trailing, with labels
    flushed in step: a ragged batch then scores, and backpropagates, exactly as without CP."""
    run_gloo_ranks(_smpo_ragged_worker, cp_size, cp_size)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
