#!/usr/bin/env python
"""A composite's scoring forwards run the cross-rank image consensus only in a trainer whose batches can carry
images: the online trainers keep it, the environmental one's text-only batches skip it."""

from types import SimpleNamespace

import pytest
import torch
from transformers import PretrainedConfig

from halo.runtime.consensus import RankConsensus
from halo.trainers.distillation.sdpg import DistributedSDPGTrainer
from halo.trainers.grpo import logprob_scoring
from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer
from halo.trainers.grpo.logprob_scoring import ChunkedGRPOLogprobsMixin
from halo.trainers.grpo.online import DistributedOnlineGRPOTrainer
from tests.common.env_grpo_batch import batch_host, build, episode, row


class _Scorer(ChunkedGRPOLogprobsMixin):
    """A composite whose chunked scorer is a stub; ``carries_images`` is the trainer's class attribute."""

    def __init__(self, carries_images: bool):
        self._SCORED_BATCHES_CARRY_IMAGES = carries_images
        self.model = torch.nn.Module()
        self.model.config = PretrainedConfig()
        self.model.train()
        self.args = SimpleNamespace(per_device_train_batch_size=1, per_device_eval_batch_size=1)
        self._use_chunked_grpo_logprobs = True
        self._model_accepts_multimodal_cache = True
        self.accelerator = SimpleNamespace(unwrap_model=lambda module: module, is_main_process=False)

    def _full_logits_logps(self, model, input_ids, attention_mask, logits_to_keep, **kwargs):
        return torch.zeros(input_ids.size(0), logits_to_keep), None, None

    def _chunked_completion_logps(self, model, input_ids, attention_mask, logits_to_keep, batch_size, compute_entropy):
        return torch.zeros(input_ids.size(0), logits_to_keep), None


@pytest.fixture
def consensus_calls(monkeypatch):
    calls = []
    monkeypatch.setattr(
        logprob_scoring, "rank_consensus", lambda local_ok: calls.append(local_ok) or RankConsensus(local_ok, local_ok)
    )
    return calls


def _score(scorer: _Scorer, **kwargs):
    return scorer._get_per_token_logps_and_entropies(
        scorer.model, torch.zeros(1, 6, dtype=torch.long), torch.ones(1, 6, dtype=torch.long), 3, **kwargs
    )


def test_the_online_scorer_runs_one_consensus_per_scoring_forward(consensus_calls):
    scorer = _Scorer(carries_images=True)
    _score(scorer)
    _score(scorer, pixel_values=torch.zeros(1, 3))
    assert consensus_calls == [False, True]


def test_the_environmental_scorer_runs_none(consensus_calls):
    scorer = _Scorer(carries_images=False)
    for _ in range(3):
        _score(scorer)
    assert consensus_calls == []


def test_a_modality_input_reaching_a_text_only_scorer_raises_where_it_is_scored(consensus_calls):
    with pytest.raises(RuntimeError, match="batches are built text-only"):
        _score(_Scorer(carries_images=False), pixel_values=torch.zeros(1, 3))
    assert consensus_calls == []


def test_a_model_that_takes_no_images_never_asked_for_consensus(consensus_calls):
    scorer = _Scorer(carries_images=True)
    scorer._model_accepts_multimodal_cache = False
    _score(scorer)
    assert consensus_calls == []


@pytest.mark.parametrize(
    ("trainer_cls", "carries_images"),
    [
        (DistributedOnlineGRPOTrainer, True),
        (DistributedSDPGTrainer, True),
        (DistributedAsyncEnvironmentalGRPOTrainer, False),
    ],
)
def test_each_trainer_declares_whether_its_batches_carry_images(trainer_cls, carries_images):
    assert trainer_cls._SCORED_BATCHES_CARRY_IMAGES is carries_images


def test_the_environmental_batch_holds_no_modality_key():
    """Its dict is assembled from text tensors alone, which is what lets its scoring skip the consensus."""
    host = batch_host([row([7, 8]), row([9])], torch.zeros(2, 2), training=True)
    batch = build(host, [episode(1.0), episode(0.0)])
    assert not set(batch) & ChunkedGRPOLogprobsMixin._MODALITY_KEYS


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
