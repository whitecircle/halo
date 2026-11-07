#!/usr/bin/env python
"""The chunked scorer scores the sampler's candidates beside each sampled token on every row path, aligned with the
completion positions; a packed window trains the surrogate as the per-row window does; the candidate head form is
warmed at construction under the knob only."""

import ast
import inspect
import textwrap
from functools import partial

import pytest
import torch

import halo.trainers.grpo.logprob_scoring as logprob_scoring
from halo.configs.grpo import ScoreCenteringConfig
from halo.configs.grpo_environmental import AsyncRolloutConfig
from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer
from halo.trainers.grpo.objective.loss_inputs import IS_SAMPLING_LOGPS_KEY
from halo.trainers.grpo.objective.score_centering import TOPK_IDS_KEY, TOPK_LOGPS_KEY
from halo.trainers.world_metrics import WorldMetrics
from tests.common.chunked_logprobs import EmbeddingBackboneHarness
from tests.common.length_deal import (
    packed_host,
    plan_for,
    random_ragged_round,
    scoring_host,
    single_process_accelerator,
    tiny_llama,
    train_window,
    windows_close,
)

VOCAB, HIDDEN, CANDIDATES = 40, 8, 4


def _batch():
    # Row 0 is full; row 1 has a left pad and a right pad, so its real completion is 2 of 4 positions.
    input_ids = torch.tensor([[11, 12, 13, 21, 22, 23, 24], [0, 14, 15, 25, 26, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 0, 0]])
    generator = torch.Generator().manual_seed(1)
    candidates = torch.randint(0, VOCAB, (2, 4, CANDIDATES), generator=generator, dtype=torch.int32)
    candidates[0, 1, 2:] = -1
    return input_ids, attention_mask, candidates, 4


@pytest.mark.parametrize(("attn_impl", "batch_size"), [("sdpa", 2), ("sdpa", 1), ("flash_attention_4", 2)])
def test_candidates_are_scored_beside_the_sampled_token_on_every_row_path(attn_impl, batch_size):
    input_ids, attention_mask, candidates, ltk = _batch()
    harness = EmbeddingBackboneHarness(attn_impl)
    plain, plain_entropy = harness._chunked_logps_impl(harness.model, input_ids, attention_mask, ltk, batch_size, True)
    with harness._scoring_with_candidates(candidates):
        logps, entropy = harness._chunked_logps_impl(harness.model, input_ids, attention_mask, ltk, batch_size, True)
    assert logps.shape == (2, ltk, 1 + CANDIDATES) and entropy.shape == (2, ltk)

    real = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    torch.testing.assert_close(logps[..., 0][real], plain[real])
    torch.testing.assert_close(entropy[real], plain_entropy[real])

    hidden = harness.embed(input_ids)[:, :-1, :][:, -ltk:, :]
    expected = torch.log_softmax(harness.model.lm_head(hidden), -1)
    for row, position in real.nonzero().tolist():
        for slot in range(CANDIDATES):
            token = int(candidates[row, position, slot])
            if token < 0:
                assert logps[row, position, 1 + slot] == -torch.inf
            else:
                torch.testing.assert_close(logps[row, position, 1 + slot], expected[row, position, token])


def test_the_gradient_reaches_the_head_and_the_backbone_through_the_candidates():
    input_ids, attention_mask, candidates, ltk = _batch()
    harness = EmbeddingBackboneHarness("sdpa")
    with harness._scoring_with_candidates(candidates):
        logps, _ = harness._chunked_logps_impl(harness.model, input_ids, attention_mask, ltk, 2, False)
    logps[..., 1:][candidates >= 0].sum().backward()
    assert harness.model.lm_head.weight.grad.abs().sum() > 0 and harness.embed.weight.grad.abs().sum() > 0


def test_misaligned_candidates_are_refused_naming_both_shapes():
    input_ids, attention_mask, candidates, ltk = _batch()
    harness = EmbeddingBackboneHarness("sdpa")
    with harness._scoring_with_candidates(candidates[:, :3]):
        with pytest.raises(ValueError, match=r"\(2, 3, 4\).*\(2, 4\)"):
            harness._chunked_logps_impl(harness.model, input_ids, attention_mask, ltk, 2, False)


def test_scoring_without_candidates_is_the_sampled_token_alone():
    input_ids, attention_mask, _, ltk = _batch()
    harness = EmbeddingBackboneHarness("sdpa")
    logps, entropy = harness._chunked_logps_impl(harness.model, input_ids, attention_mask, ltk, 2, True)
    assert logps.shape == entropy.shape == (2, ltk)


# --- a packed window against the per-row window -----------------------------------------------------------------

_LLAMA_VOCAB, _ROWS, _SLOTS, _BUDGET, _K = 64, 16, 4, 40, 8


def _sampler_round(seed: int = 9) -> dict:
    """A ragged round carrying a sampler's top-K candidates, its log-prob of each sampled token and a unit ratio."""
    batch, gen = random_ragged_round(_ROWS, _LLAMA_VOCAB, seed)
    real = batch["completion_mask"].bool()
    sampler = torch.log_softmax(torch.randn(*real.shape, _LLAMA_VOCAB, generator=gen) * 2, dim=-1)
    top = torch.topk(sampler, _K, dim=-1)
    batch[TOPK_IDS_KEY] = torch.where(real[..., None], top.indices, -1).int()
    batch[TOPK_LOGPS_KEY] = torch.where(real[..., None], top.values, -torch.inf)
    sampled = sampler.gather(-1, batch["completion_ids"].unsqueeze(-1)).squeeze(-1)
    batch[IS_SAMPLING_LOGPS_KEY] = torch.where(real, sampled, 0.0)
    batch["importance_sampling_ratio"] = torch.ones_like(batch["importance_sampling_ratio"])
    return batch


def _window(batch: dict, *, packed: bool, centering: ScoreCenteringConfig | None) -> tuple[float, list[torch.Tensor]]:
    host = packed_host(tiny_llama(_LLAMA_VOCAB, 0), slots=_SLOTS, accelerator=single_process_accelerator())
    host._score_centering, host._centering_metrics = centering, WorldMetrics()
    host.vllm_importance_sampling_clip_max = 2.0
    if not packed:
        host._packs_microbatches = False
        return train_window(host, batch)
    plan = plan_for(batch, _BUDGET, _SLOTS, _ROWS)
    assert plan.num_bins < _SLOTS, "a window without a no-op slot leaves the filler rows untested"
    return train_window(host, batch, plan)


_close = partial(windows_close, rel=1e-5, grad_rtol=1e-4)


@pytest.mark.parametrize("weights", ["none", "tis", "mis"])
def test_a_packed_window_trains_the_surrogate_as_the_per_row_window_does(weights):
    batch = _sampler_round()
    centering = ScoreCenteringConfig(top_k=_K, weights=weights)
    per_row = _window(batch, packed=False, centering=centering)
    assert _close(_window(batch, packed=True, centering=centering), per_row)
    plain = _window(batch, packed=False, centering=None)
    assert per_row[0] == pytest.approx(plain[0], rel=1e-6), "the surrogate's value is the policy's log-prob"
    assert not _close(per_row, plain), "centering must change the gradient, or the comparison checks nothing"


def test_negative_control_candidates_moved_to_another_row_change_the_packed_gradient():
    batch = _sampler_round()
    centering = ScoreCenteringConfig(top_k=_K)
    per_row = _window(batch, packed=False, centering=centering)
    moved = batch | {key: batch[key].roll(1, dims=0) for key in (TOPK_IDS_KEY, TOPK_LOGPS_KEY)}
    assert not _close(_window(moved, packed=True, centering=centering), per_row)


# --- the candidate head form is warmed at construction ----------------------------------------------------------


def _warm_host(top_k: int | None, monkeypatch):
    host = scoring_host(tiny_llama(_LLAMA_VOCAB, 0), None, chunks=1, accelerator=single_process_accelerator())
    host.async_config = AsyncRolloutConfig(score_centering_top_k=top_k)
    host.args.num_iterations, host.args.beta, host.args.importance_sampling_level = 1, 0.0, "token"
    host.model.eval()
    calls = []
    head = logprob_scoring.chunked_head_logprobs

    def recording(hidden, weight, targets, *args, compute_entropy=False, **kwargs):
        calls.append((tuple(targets.shape), compute_entropy, torch.is_grad_enabled()))
        return head(hidden, weight, targets, *args, compute_entropy=compute_entropy, **kwargs)

    monkeypatch.setattr(logprob_scoring, "chunked_head_logprobs", recording)
    # The warm-up runs where kernels compile; on the host it would return before scoring anything.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    return host, calls


def test_the_warm_up_scores_the_candidate_form_with_entropy_and_drops_its_gradients(monkeypatch):
    host, calls = _warm_host(5, monkeypatch)
    host._configure_score_centering()
    assert host._score_centering == ScoreCenteringConfig(top_k=5)
    assert calls and all(shape[-1] == 1 + 5 and entropy and grad for shape, entropy, grad in calls), calls
    assert all(p.grad is None for p in host.model.parameters()), "the warm-up's gradients must not reach a step"
    assert not host.model.training, "the model's mode must come back"


def test_without_the_knob_nothing_is_warmed(monkeypatch):
    host, calls = _warm_host(None, monkeypatch)
    host._configure_score_centering()
    assert host._score_centering is None and calls == []


def test_a_loss_the_surrogate_cannot_carry_is_refused_before_anything_is_warmed(monkeypatch):
    host, calls = _warm_host(5, monkeypatch)
    host.args.beta = 0.04
    with pytest.raises(ValueError, match="beta=0.04"):
        host._configure_score_centering()
    assert calls == []


def test_the_trainer_configures_score_centering_once_the_model_is_placed():
    """A dropped call leaves the knob off whatever the config says, and one ahead of the placement warms a model that
    is not the one trained."""
    init = ast.parse(textwrap.dedent(inspect.getsource(DistributedAsyncEnvironmentalGRPOTrainer.__init__))).body[0]
    lines = {
        node.func.attr: node.lineno
        for node in ast.walk(init)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert lines["_configure_packed_microbatches"] < lines["_configure_score_centering"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
