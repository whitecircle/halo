"""CPU tests for the GRPO reward/advantage hardening that holds an env-GRPO chat template together.

Each test pins one run-degrading failure mode:

* structural (special) tokens must always receive policy gradient under ``top_entropy_quantile`` —
  they are the lowest-entropy tokens, so an unprotected quantile filter excludes them entirely and
  the chat template rots until every tool call stops parsing;
* a GRPO group whose completions all score alike carries no signal and must be droppable;
* identical rewards must never be turned into a full-scale advantage — offline GRPO must not
  fabricate a ±1 ranking out of a zero-variance group.

    python tests/cpu/grpo/test_grpo_reward_hardening.py
"""

import logging
import sys

import numpy as np
import pytest
import torch

from src.trainers.grpo.mixins.entropy_mask import ProtectedTokenEntropyMixin
from src.trainers.grpo.objective.advantages import degenerate_group_mask, group_relative_advantages
from src.trainers.grpo.offline import compute_group_advantages

# --- top_entropy_quantile must never exclude structural tokens ---

SPECIAL_IDS = [100, 101]  # <|channel|>, <|call|>: near-deterministic, so the lowest-entropy tokens
NORMAL_IDS = [7, 8, 9]


class _FakeTokenizer:
    all_special_ids = [100]

    def get_added_vocab(self):  # harmony/ChatML scaffolding lands here, not in all_special_ids
        return {"<|call|>": 101}


class _BaseTrainer:
    """Stands in for TRL's GRPOTrainer: the plain top-quantile entropy mask, no token awareness."""

    def get_high_entropy_mask(self, entropies, mask, threshold):
        masked = entropies[mask.bool()]
        cutoff = torch.quantile(masked.float(), threshold)
        return (entropies >= cutoff) & mask.bool()

    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep, *a, **kw):
        return None, None


class _Harness(ProtectedTokenEntropyMixin, _BaseTrainer):
    def __init__(self):
        self.processing_class = _FakeTokenizer()


def _entropy_setup():
    # Specials hold the lowest entropies, as they do under a real chat template.
    completion_ids = torch.tensor([[100, 101, 7, 8, 9]])
    entropies = torch.tensor([[0.01, 0.02, 5.0, 6.0, 7.0]])
    mask = torch.ones_like(completion_ids, dtype=torch.long)
    return completion_ids, entropies, mask


def test_special_tokens_are_excluded_by_the_plain_quantile_mask():
    # Precondition for the test below: TRL's plain mask alone drops every structural token.
    completion_ids, entropies, mask = _entropy_setup()
    plain = _BaseTrainer().get_high_entropy_mask(entropies, mask, 0.8)
    is_special = torch.isin(completion_ids, torch.tensor(SPECIAL_IDS))
    assert not bool((plain & is_special).any()), "precondition: low-entropy special tokens fall outside the quantile"


def test_protected_mixin_always_trains_special_tokens():
    harness = _Harness()
    completion_ids, entropies, mask = _entropy_setup()
    harness._get_per_token_logps_and_entropies(None, completion_ids, mask, completion_ids.size(1))

    got = harness.get_high_entropy_mask(entropies, mask, 0.8)
    is_special = torch.isin(completion_ids, torch.tensor(SPECIAL_IDS))
    assert bool(got[is_special].all()), (
        "structural tokens are outside the trained set — nothing reinforces the chat template and it will rot"
    )
    plain = _BaseTrainer().get_high_entropy_mask(entropies, mask, 0.8)
    assert bool((got | plain).equal(got)), "the mixin must be a superset of TRL's mask, never a replacement"


def test_protected_mask_still_respects_padding():
    harness = _Harness()
    completion_ids = torch.tensor([[100, 7, 101]])
    entropies = torch.tensor([[0.01, 5.0, 0.02]])
    mask = torch.tensor([[1, 1, 0]])  # the padded position carries a special id — protection must not override it
    harness._get_per_token_logps_and_entropies(None, completion_ids, mask, completion_ids.size(1))
    got = harness.get_high_entropy_mask(entropies, mask, 0.8)
    assert not bool(got[0, 2]), "padding must never enter the loss, special id or not"


# --- Degenerate (zero-spread) groups ---


def test_degenerate_group_mask_flags_only_zero_spread_groups():
    rewards = torch.tensor([0.3, 0.3, 0.3, 0.0, 0.5, 1.0])
    got = degenerate_group_mask(rewards, num_generations=3)
    assert got.tolist() == [True, True, True, False, False, False], (
        "a zero-spread group must be dropped (advantage ≡ 0, but its tokens would still dilute the normalizer)"
    )


def test_degenerate_group_mask_noop_without_grouping():
    rewards = torch.tensor([0.3, 0.3])
    assert not degenerate_group_mask(rewards, num_generations=1).any(), "num_generations=1 has no group to judge"


# --- scale_rewards semantics ---


def test_advantages_subtract_the_group_mean():
    rewards = torch.tensor([0.0, 1.0, 0.0, 1.0])
    got = group_relative_advantages(rewards, num_generations=2, scale_rewards="none")
    assert torch.allclose(got, torch.tensor([-0.5, 0.5, -0.5, 0.5]))
    # A group that does not sum to zero leaves a net "reinforce/suppress everything" push in the gradient.
    assert torch.allclose(got.view(-1, 2).sum(dim=1), torch.zeros(2), atol=1e-6)


def test_none_is_not_treated_as_truthy():
    # "none" is a truthy STRING: a bare `if scale_rewards` would scale anyway and silently discard Dr.GRPO.
    rewards = torch.tensor([0.0, 1.0])
    unscaled = group_relative_advantages(rewards, 2, "none")
    assert torch.allclose(unscaled, torch.tensor([-0.5, 0.5])), "scale_rewards='none' must not rescale"


def test_group_scaling_amplifies_a_near_degenerate_group_but_batch_does_not():
    # Why "batch": per-group std division rescales each group to unit variance, so group 0's 0.01 of
    # shaping jitter becomes as influential as group 1's real signal.
    rewards = torch.tensor([0.00, 0.01, 0.0, 1.0])

    by_group = group_relative_advantages(rewards, 2, "group")
    noise_vs_real = by_group[:2].abs().max() / by_group[2:].abs().max()
    assert noise_vs_real > 0.5, "precondition: per-group scaling inflates shaping noise to rival real signal"

    by_batch = group_relative_advantages(rewards, 2, "batch")
    noise_vs_real_batch = by_batch[:2].abs().max() / by_batch[2:].abs().max()
    assert noise_vs_real_batch < 0.05, (
        "'batch' must leave a degenerate group near zero: it divides by the WHOLE batch's std, never by the "
        f"group's own vanishing std (got noise/real = {noise_vs_real_batch:.3f})"
    )


def test_singleton_groups_do_not_produce_nan():
    # std() over one element is NaN under Bessel's correction, and that NaN reaches the optimizer.
    rewards = torch.tensor([0.4, 0.9])
    for scale in ("group", "batch", "none"):
        got = group_relative_advantages(rewards, num_generations=1, scale_rewards=scale)
        assert not torch.isnan(got).any(), f"NaN advantage with scale_rewards={scale!r}"
        assert torch.allclose(got, torch.zeros(2)), "a singleton group has no baseline → zero advantage"


# --- Offline GRPO must not fabricate advantages from a zero-variance group ---


_NO_EMPHASIS = 0.0  # emphasis off, so the advantage is pure ranking


@pytest.mark.parametrize("rewards", [[0.5, 0.5], [0.4, 0.4, 0.4, 0.4]])
def test_quantile_uniform_gives_zero_advantage_on_identical_rewards(rewards):
    # Spreading tied rewards over [-1, 1] would rank completions by array position — signal from nothing.
    got = compute_group_advantages(rewards, "quantile_uniform", _NO_EMPHASIS)
    assert np.allclose(got, 0.0), f"zero-variance group must yield zero advantage, got {got}"


def test_quantile_uniform_still_ranks_a_real_spread():
    got = compute_group_advantages([0.0, 0.5, 1.0], "quantile_uniform", _NO_EMPHASIS)
    assert got[0] < got[1] < got[2], "a genuine reward spread must still produce an ordered advantage"


class _NoSpecialsTokenizer:
    """A tokenizer exposing nothing to protect — the branch that warns."""

    all_special_ids: list[int] = []

    def get_added_vocab(self):
        return {}


class _NoSpecialsHarness(ProtectedTokenEntropyMixin, _BaseTrainer):
    def __init__(self):
        self.processing_class = _NoSpecialsTokenizer()


def test_unprotected_warning_is_emitted_once_not_per_microbatch(caplog):
    """The 'nothing to protect' answer must be cached like any other.

    Returning early without caching re-resolves the tokenizer on every call, so the warning repeats
    once per micro-batch for the whole run — in the one configuration (top_entropy_quantile < 1)
    where a user is watching that log.
    """
    harness = _NoSpecialsHarness()
    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            assert harness._protected_token_ids() is None, "no specials must still resolve to None"
    warnings = [r for r in caplog.records if "entropy mask left unprotected" in r.getMessage()]
    assert len(warnings) == 1, f"expected exactly one warning, got {len(warnings)}"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
