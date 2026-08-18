#!/usr/bin/env python
"""Test DistributedTrainerMixin's num_unmasked_output_tokens_seen machinery in isolation.

Exercises ``_extract_output_token_count`` (which batches carry a token-level loss mask) and the
two-phase counter — ``_accumulate_unmasked_output_tokens`` folds each micro-batch's count into an
ON-DEVICE running tensor with no collective, and ``_drain_unmasked_output_tokens`` does the cross-rank
gather + TP/ETP/CP de-duplication once per log — against a fake trainer ``self`` (no GPU / dist init).

Run: python tests/cpu/trainers/test_unmasked_output_tokens_metric.py
"""

import types
from collections import UserDict

import pytest
import torch

from src.trainers.mixins.base import DistributedTrainerMixin

_extract = DistributedTrainerMixin._extract_output_token_count
_accumulate = DistributedTrainerMixin._accumulate_unmasked_output_tokens
_drain = DistributedTrainerMixin._drain_unmasked_output_tokens
_add_logs = DistributedTrainerMixin._add_parallelism_to_logs


def _fake_self(*, training=True, num_processes=1, data_parallel_size=1, gather=None):
    return types.SimpleNamespace(
        model=types.SimpleNamespace(training=training),
        accelerator=types.SimpleNamespace(
            num_processes=num_processes,
            gather=gather or (lambda x: x),
            device=torch.device("cpu"),
        ),
        parallelism_config=types.SimpleNamespace(
            data_parallel_size=data_parallel_size,
            non_dp_replication_factor=num_processes // data_parallel_size,
        ),
    )


def _fake_log_self(*, training=True, global_step=5, cumulative=None):
    """Minimal ``self`` for ``_add_parallelism_to_logs``: all parallelism modes off (single-GPU)."""
    config = types.SimpleNamespace(
        data_parallel_size=1,
        non_dp_replication_factor=1,
        is_ep_tp_mode=False,
        is_ep_cp_mode=False,
        is_tp_mode=False,
        is_cp_mode=False,
        is_expert_tp_mode=False,
    )
    me = types.SimpleNamespace(
        state=types.SimpleNamespace(global_step=global_step),
        parallelism_config=config,
        _has_ep_layers=False,
        model=types.SimpleNamespace(training=training),
        accelerator=types.SimpleNamespace(num_processes=1, gather=lambda x: x, device=torch.device("cpu")),
    )
    # _add_parallelism_to_logs drains first; the real drain gathers a zero here, isolating the emit.
    me._drain_unmasked_output_tokens = types.MethodType(_drain, me)
    if cumulative is not None:
        me._cumulative_unmasked_output_tokens = cumulative
    return me


def test_extract_causal_lm_labels():
    inputs = {
        "input_ids": torch.zeros(2, 5, dtype=torch.long),
        "labels": torch.tensor([[-100, -100, 1, 2, 3], [-100, 4, 5, -100, -100]]),
    }
    assert int(_extract(None, inputs)) == 5


def test_extract_accepts_batchencoding_userdict():
    # BatchEncoding subclasses UserDict, not dict.
    inputs = UserDict(
        {
            "input_ids": torch.zeros(1, 4, dtype=torch.long),
            "labels": torch.tensor([[-100, 1, 2, -100]]),
        }
    )
    assert int(_extract(None, inputs)) == 2


def test_extract_grpo_completion_mask():
    inputs = {"completion_mask": torch.tensor([[1, 1, 0], [1, 0, 0]])}
    assert int(_extract(None, inputs)) == 3


def test_extract_grpo_intersects_tool_mask():
    # completion_mask is attention-valid (tool output included); the loss mask is completion ∧ tool.
    inputs = {
        "completion_mask": torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]]),
        "tool_mask": torch.tensor([[1, 0, 1, 1], [1, 1, 1, 1]]),
    }
    assert int(_extract(None, inputs)) == 4  # 5 completion tokens minus the tool-output one


def test_extract_grpo_tool_mask_bool_dtype():
    # The env trainer carries bool masks; the intersection must count identically.
    inputs = {
        "completion_mask": torch.tensor([[True, True, True], [True, False, False]]),
        "tool_mask": torch.tensor([[True, False, True], [True, True, True]]),
    }
    assert int(_extract(None, inputs)) == 3


def test_extract_offline_grpo_completion_attention_mask():
    # Offline GRPO emits prompt_*/completion_* tensors; the loss reduces over completion_attention_mask.
    inputs = {
        "prompt_input_ids": torch.zeros(2, 4, dtype=torch.long),
        "prompt_attention_mask": torch.ones(2, 4, dtype=torch.long),
        "completion_input_ids": torch.zeros(2, 3, dtype=torch.long),
        "completion_attention_mask": torch.tensor([[1, 1, 0], [1, 1, 1]]),
    }
    assert int(_extract(None, inputs)) == 5


def test_extract_skips_sequence_level_labels():
    # Classification: 1-D per-sequence labels are not token-aligned.
    inputs = {"input_ids": torch.zeros(2, 5, dtype=torch.long), "labels": torch.tensor([0, 1])}
    assert _extract(None, inputs) is None


def test_extract_skips_multilabel_shape_mismatch():
    # Multi-label [B, C] never aligns with input_ids [B, S].
    inputs = {"input_ids": torch.zeros(2, 5, dtype=torch.long), "labels": torch.zeros(2, 3)}
    assert _extract(None, inputs) is None


def test_extract_skips_when_no_mask():
    assert _extract(None, {"input_ids": torch.zeros(2, 5, dtype=torch.long)}) is None
    assert _extract(None, None) is None


def test_accumulate_is_local_no_gather_until_drain():
    # Gathering here would be a per-micro-batch host sync + collective; only the drain may do it.
    me = _fake_self(gather=lambda _: (_ for _ in ()).throw(AssertionError("gather ran during accumulate")))
    _accumulate(me, torch.tensor(7))
    _accumulate(me, torch.tensor(3))
    assert int(me._local_unmasked_token_accum) == 10
    assert not hasattr(me, "_cumulative_unmasked_output_tokens")


def test_drain_single_process():
    me = _fake_self()
    _accumulate(me, torch.tensor(7))
    _accumulate(me, torch.tensor(3))
    _drain(me)
    assert me._cumulative_unmasked_output_tokens == 10
    assert me._local_unmasked_token_accum is None


def test_drain_dedups_replicated_ranks():
    # TP/ETP/CP: both ranks hold identical data, so non_dp_factor=2 divides the duplicate out.
    me = _fake_self(num_processes=2, data_parallel_size=1, gather=lambda x: torch.tensor([int(x), int(x)]))
    _accumulate(me, torch.tensor(5))
    _drain(me)
    assert me._cumulative_unmasked_output_tokens == 5


def test_drain_sums_distinct_dp_ranks():
    # Pure DP: distinct batches, non_dp_factor=1 → the counts sum.
    me = _fake_self(num_processes=2, data_parallel_size=2, gather=lambda _: torch.tensor([5, 3]))
    _accumulate(me, torch.tensor(5))
    _drain(me)
    assert me._cumulative_unmasked_output_tokens == 8


def test_drain_accumulates_across_windows():
    me = _fake_self()
    _accumulate(me, torch.tensor(4))
    _drain(me)
    _accumulate(me, torch.tensor(6))
    _drain(me)
    assert me._cumulative_unmasked_output_tokens == 10


def test_accumulate_counts_when_training_flag_false():
    # FSDP / grad-checkpointing leave model.training transiently False on the first step.
    me = _fake_self(training=False)
    _accumulate(me, torch.tensor(5))
    _drain(me)
    assert me._cumulative_unmasked_output_tokens == 5


def test_accumulate_noop_when_count_none():
    me = _fake_self()
    _accumulate(me, None)
    assert not hasattr(me, "_local_unmasked_token_accum")


def test_drain_gathers_zero_when_nothing_accumulated():
    # The gather is world-wide: skipping it on a rank-local emptiness would desync the collective.
    gathered = []
    me = _fake_self(gather=lambda x: gathered.append(x) or x)
    _drain(me)
    assert len(gathered) == 1 and int(gathered[0]) == 0
    assert not hasattr(me, "_cumulative_unmasked_output_tokens")


def test_drain_keeps_counting_after_an_empty_window():
    me = _fake_self()
    _accumulate(me, torch.tensor(4))
    _drain(me)
    _drain(me)  # empty window
    assert me._cumulative_unmasked_output_tokens == 4


def test_accumulate_casts_to_int64():
    # The drain zero-fills absent ranks with int64; a float mask sum must not change that dtype.
    me = _fake_self()
    _accumulate(me, torch.tensor([1.0, 1.0, 0.0]).sum())
    assert me._local_unmasked_token_accum.dtype == torch.int64


def test_emit_logs_output_tokens_when_training_flag_false():
    # Gating the emit on model.training would hide the metric from the first logged step.
    me = _fake_log_self(training=False, cumulative=42)
    logs = {}
    _add_logs(me, logs)
    assert logs["num_unmasked_output_tokens_seen"] == 42


def test_emit_skips_output_tokens_when_never_counted():
    # Trainers with no token-level loss mask must omit the key entirely, not log a 0.
    me = _fake_log_self(cumulative=None)
    logs = {}
    _add_logs(me, logs)
    assert "num_unmasked_output_tokens_seen" not in logs


def test_emit_skips_before_first_step():
    # global_step == 0 returns before touching the log dict at all.
    me = _fake_log_self(global_step=0, cumulative=42)
    logs = {}
    _add_logs(me, logs)
    assert logs == {}


def test_non_dp_replication_factor_matches_the_ranks_sharing_a_batch():
    """The de-duplication divisor the token counters use, derived once on the config.

    Both call sites (``_drain_unmasked_output_tokens`` and SFT's ``compute_loss``) read the property
    instead of recomputing ``num_processes // data_parallel_size`` by hand. It must equal
    ``pp_size * max(tp, cp, expert_tp)`` — the ranks that see the SAME batch — on every axis
    combination, with EP orthogonal to DP.
    """
    from src.distributed.parallelism_config import ParallelismConfig

    def _config(**kwargs):
        return ParallelismConfig(world_size=8, gpus_per_node=8, nvlink_domain_size=8, **kwargs)

    assert _config().non_dp_replication_factor == 1  # plain FSDP2 DP
    assert _config(tp_size=2).non_dp_replication_factor == 2
    assert _config(cp_size=4).non_dp_replication_factor == 4
    assert _config(ep_size=8).non_dp_replication_factor == 1  # EP is orthogonal to DP
    assert _config(ep_size=2, expert_tp_size=2).non_dp_replication_factor == 2

    # PP stages must land on NVLink-domain boundaries, hence the narrower domain here.
    def _pp_config(**kwargs):
        return ParallelismConfig(world_size=8, gpus_per_node=4, nvlink_domain_size=4, pp_size=2, **kwargs)

    assert _pp_config().non_dp_replication_factor == 2
    assert _pp_config(ep_size=1, expert_tp_size=2).non_dp_replication_factor == 4  # PP + pure ETP

    for config in (_config(), _config(tp_size=2), _config(cp_size=4), _pp_config(expert_tp_size=2)):
        assert config.non_dp_replication_factor == config.world_size // config.data_parallel_size


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
