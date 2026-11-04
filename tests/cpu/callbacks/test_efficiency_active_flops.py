"""Per-rank active-FLOPs (S-MFU) in :func:`estimate_rank_flops` and the :class:`EfficiencyCallback` warmup window.
EP moves experts between ranks and creates no arithmetic, so active FLOPs/token is independent of ``ep_size``."""

import pytest
import torch
from torch import nn

from halo.callbacks.efficiency import EfficiencyCallback
from halo.callbacks.model_flops import estimate_rank_flops
from tests.common.parallelism import make_parallelism_config

_NO_PARALLELISM = make_parallelism_config(world_size=1, gpus_per_node=1)

HIDDEN = 64
NUM_EXPERTS = 32
TOP_K = 4
SEQ_LEN = 128


class _Config:
    def __init__(self, num_experts: int, top_k: int):
        self.num_experts = num_experts
        self.num_experts_per_tok = top_k
        self.num_hidden_layers = 2
        self.hidden_size = HIDDEN

    def get_text_config(self):
        return self


class _MoEModel(nn.Module):
    def __init__(self, local_experts: int, expert_tp_size: int = 1):
        super().__init__()
        self.shared = nn.Linear(HIDDEN, HIDDEN, bias=False)
        # The callback recognises a routed expert weight by ".experts." in the parameter path.
        self.mlp = nn.Module()
        self.mlp.experts = nn.ParameterDict(
            {"w": nn.Parameter(torch.zeros(local_experts, HIDDEN, HIDDEN // expert_tp_size))}
        )
        self.config = _Config(NUM_EXPERTS, TOP_K)


def _active_flops_per_token(ep_size: int, expert_tp_size: int = 1) -> float:
    """One EP group filling one NVLink domain, the smallest world these axis sizes are valid on."""
    width = ep_size * expert_tp_size
    parallelism = make_parallelism_config(
        ep_size=ep_size, expert_tp_size=expert_tp_size, world_size=width, gpus_per_node=width
    )
    model = _MoEModel(local_experts=NUM_EXPERTS // ep_size, expert_tp_size=expert_tp_size)
    flops = estimate_rank_flops(model, seq_len=SEQ_LEN, parallelism_config=parallelism, full_model_params=None)
    return flops.active_model_flops_per_token


@pytest.mark.parametrize("ep_size", [2, 4, 8])
def test_active_flops_per_token_is_ep_invariant(ep_size):
    """Without the ``ep_size`` factor on the expert term, S-MFU under-reports by up to 8x on an ep8 run."""
    baseline = _active_flops_per_token(ep_size=1)
    actual = _active_flops_per_token(ep_size=ep_size)
    assert actual == pytest.approx(baseline, rel=1e-9), (
        f"active FLOPs/token at ep_size={ep_size} is {actual:.6e}, expected the ep1 value "
        f"{baseline:.6e}. A rank holds num_experts/ep_size experts but serves the whole EP group's "
        "tokens, so it computes top_k expert-token pairs per local token regardless of ep_size."
    )


def test_active_flops_matches_hand_derivation():
    """Pin the absolute value, so an ep-invariant but wrongly-scaled term cannot pass."""
    shared = HIDDEN * HIDDEN
    one_expert = HIDDEN * HIDDEN
    attention = 12.0 * 2 * SEQ_LEN * HIDDEN  # 12 * L * S * H
    expected = 6.0 * (shared + TOP_K * one_expert) + attention

    for ep_size in (1, 4):
        actual = _active_flops_per_token(ep_size=ep_size)
        assert actual == pytest.approx(expected, rel=1e-9), (
            f"ep_size={ep_size}: active FLOPs/token {actual:.6e} != hand-derived {expected:.6e} "
            f"(6*(shared {shared} + top_k {TOP_K} * expert {one_expert}) + attention {attention})"
        )


@pytest.mark.parametrize("expert_tp_size", [1, 2, 4])
def test_expert_tp_divides_active_flops(expert_tp_size):
    """``expert_tp_size`` must not be multiplied in, because it already divides the local expert params."""
    baseline_expert_term = 6.0 * TOP_K * HIDDEN * HIDDEN
    shared_and_attention = 6.0 * HIDDEN * HIDDEN + 12.0 * 2 * SEQ_LEN * HIDDEN

    actual = _active_flops_per_token(ep_size=8, expert_tp_size=expert_tp_size)
    expected = shared_and_attention + baseline_expert_term / expert_tp_size
    assert actual == pytest.approx(expected, rel=1e-9), (
        f"expert_tp_size={expert_tp_size}: active FLOPs/token {actual:.6e} != {expected:.6e}; the "
        "expert term must scale as 1/expert_tp_size."
    )


class _State:
    def __init__(self, global_step: int):
        self.global_step = global_step
        self.num_input_tokens_seen = None


def _warmup_steps(cb: EfficiencyCallback, start_step: int, steps) -> list[int]:
    cb.on_train_begin(None, _State(start_step), None)
    skipped = []
    for step in steps:
        logs: dict = {}
        cb.on_log(None, _State(step), None, logs)
        if not logs:
            skipped.append(step)
    return skipped


def test_warmup_excludes_exactly_n_steps():
    """HF increments global_step before on_step_end, so the bound must be inclusive, or first-step JIT and allocator
    skew stay in the averages."""
    skipped = _warmup_steps(EfficiencyCallback(_NO_PARALLELISM, n_warmup_steps=2), 0, (1, 2, 3, 4))
    assert skipped == [1, 2], f"expected steps [1, 2] to be warmup, got {skipped}"


def test_zero_warmup_measures_every_step():
    cb = EfficiencyCallback(_NO_PARALLELISM, n_warmup_steps=0)
    assert _warmup_steps(cb, 0, (1,)) == [], "n_warmup_steps=0 must measure the very first step"


def test_warmup_is_relative_to_the_resume_step():
    skipped = _warmup_steps(EfficiencyCallback(_NO_PARALLELISM, n_warmup_steps=2), 500, (501, 502, 503))
    assert skipped == [501, 502], f"expected [501, 502] after resuming at 500, got {skipped}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
