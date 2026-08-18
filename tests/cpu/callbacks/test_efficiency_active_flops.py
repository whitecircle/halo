"""Per-rank active-FLOPs (S-MFU) accounting in :class:`EfficiencyCallback`.

S-MFU is the metric the debugging docs point MoE users at, so its FLOPs/token must describe the work a
rank actually does. The invariant these tests pin: expert parallelism moves experts between ranks but
creates no arithmetic, so per-rank active FLOPs/token is INDEPENDENT of ``ep_size``.
"""

import pytest
import torch
from torch import nn

from src.callbacks.efficiency import EfficiencyCallback
from tests.common.parallelism import make_parallelism_config

# Warmup accounting reads no axis size; a single-process config keeps those tests topology-free.
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
    """Shared trunk plus a routed expert bank holding ``local_experts`` experts on this rank."""

    def __init__(self, local_experts: int, expert_tp_size: int = 1):
        super().__init__()
        self.shared = nn.Linear(HIDDEN, HIDDEN, bias=False)
        # Nested under `mlp` so the parameter path contains ".experts.", which is how the callback
        # recognises a routed expert weight.
        self.mlp = nn.Module()
        # One [local_experts, HIDDEN, HIDDEN/etp] bank — the per-rank slice under EP (and ETP).
        self.mlp.experts = nn.ParameterDict(
            {"w": nn.Parameter(torch.zeros(local_experts, HIDDEN, HIDDEN // expert_tp_size))}
        )
        self.config = _Config(NUM_EXPERTS, TOP_K)


def _active_flops_per_token(ep_size: int, expert_tp_size: int = 1) -> float:
    """Build the rank-local model implied by (ep_size, expert_tp_size) and return its active FLOPs/token.

    The simulated topology is one EP group filling a single NVLink domain — the smallest world these
    axis sizes are valid on, and one the FLOPs arithmetic is indifferent to.
    """
    width = ep_size * expert_tp_size
    parallelism = make_parallelism_config(
        ep_size=ep_size, expert_tp_size=expert_tp_size, world_size=width, gpus_per_node=width
    )
    cb = EfficiencyCallback(parallelism)
    model = _MoEModel(local_experts=NUM_EXPERTS // ep_size, expert_tp_size=expert_tp_size)
    cb._initialize_model_flops(model, SEQ_LEN)
    return cb.state.active_model_flops_per_token


@pytest.mark.parametrize("ep_size", [2, 4, 8])
def test_active_flops_per_token_is_ep_invariant(ep_size):
    """EP redistributes experts; it does not change the arithmetic a rank performs per local token.

    Without the ``ep_size`` factor on the expert term the reported value falls by exactly ``ep_size``,
    so S-MFU under-reports by up to 8x on an ep8 run.
    """
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
    """ETP shards each expert's rows, so a rank really does 1/etp of the expert work.

    Guards the other direction of the same fix: ``expert_tp_size`` must NOT be multiplied in, because
    it already divides the local expert params.
    """
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


def test_warmup_excludes_exactly_n_steps():
    """HF increments global_step before on_step_end, so the bound must be inclusive.

    With an exclusive bound ``n_warmup_steps=2`` skipped only one step, leaving first-step JIT and
    allocator skew inside the reported averages.
    """
    cb = EfficiencyCallback(_NO_PARALLELISM, n_warmup_steps=2)
    cb.state.global_start_step = 0

    skipped = [step for step in (1, 2, 3, 4) if cb._is_warmup_step(_State(step))]
    assert skipped == [1, 2], f"expected steps [1, 2] to be warmup, got {skipped}"


def test_zero_warmup_measures_every_step():
    cb = EfficiencyCallback(_NO_PARALLELISM, n_warmup_steps=0)
    cb.state.global_start_step = 0
    assert not cb._is_warmup_step(_State(1)), "n_warmup_steps=0 must measure the very first step"


def test_warmup_is_relative_to_the_resume_step():
    cb = EfficiencyCallback(_NO_PARALLELISM, n_warmup_steps=2)
    cb.state.global_start_step = 500
    skipped = [step for step in (501, 502, 503) if cb._is_warmup_step(_State(step))]
    assert skipped == [501, 502], f"expected [501, 502] after resuming at 500, got {skipped}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
