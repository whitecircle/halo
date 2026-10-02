#!/usr/bin/env python
"""Under autocast the flex-sliding path hands its compiled FlexAttention the autocast dtype, as autocast casts
SDPA's inputs.

Under bf16 autocast a model with fp32 parameters reaches its sliding layers with fp32 queries and keys (the
fp32 rotary embedding promotes the bf16 projections) beside bf16 values. Run at those dtypes, the kernel is a
graph the load-time warm-up never compiled, so each rank compiles it mid-run, and it misses the tuned tiles,
which are keyed on 16-bit inputs. The compiled call is replaced by a spy, so this runs on CPU.

    python tests/cpu/models/test_flex_sliding_autocast.py
"""

from unittest.mock import patch

import pytest
import torch

from src.models.patches import flex_sliding_attention

HEADS, KV_HEADS, SEQ = 4, 2, 200


def _inputs(dim: int, dtypes: tuple[torch.dtype, torch.dtype, torch.dtype], trainable: str = "qkv"):
    q_dtype, k_dtype, v_dtype = dtypes
    return (
        torch.randn(1, HEADS, SEQ, dim, dtype=q_dtype).requires_grad_("q" in trainable),
        torch.randn(1, KV_HEADS, SEQ, dim, dtype=k_dtype).requires_grad_("k" in trainable),
        torch.randn(1, KV_HEADS, SEQ, dim, dtype=v_dtype).requires_grad_("v" in trainable),
    )


def _sliding_call(query, key, value, autocast: bool) -> list[dict]:
    """Run the sliding path on one row and return what each compiled call received."""
    seen = []

    def compiled(q, k, v, *, kernel_options, **kwargs):
        seen.append(
            {
                "dtypes": (q.dtype, k.dtype, v.dtype),
                "requires_grad": (q.requires_grad, k.requires_grad, v.requires_grad),
                "kernel_options": kernel_options,
            }
        )
        return q.clone()

    with (
        patch.object(flex_sliding_attention, "_compiled_flex", compiled),
        torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast),
    ):
        flex_sliding_attention._sliding_flex_attention(query, key, value, [None], scaling=1.0)
    return seen


@pytest.mark.parametrize("trainable", ["qkv", "q"])
def test_mixed_inputs_reach_the_kernel_in_the_autocast_dtype(trainable):
    """The dtypes an fp32-parameter model hands a sliding layer under bf16 autocast must reach the kernel as
    bf16, still in the training graph (every input requiring grad, a frozen one included)."""
    query, key, value = _inputs(64, (torch.float32, torch.float32, torch.bfloat16), trainable)
    (call,) = _sliding_call(query, key, value, autocast=True)
    assert call["dtypes"] == (torch.bfloat16,) * 3, call["dtypes"]
    assert call["requires_grad"] == (True, True, True), call["requires_grad"]


def test_inputs_keep_their_dtype_outside_autocast():
    """Without autocast nothing is cast: an fp32 run compiles and runs fp32."""
    (call,) = _sliding_call(*_inputs(64, (torch.float32,) * 3), autocast=False)
    assert call["dtypes"] == (torch.float32,) * 3, call["dtypes"]


def test_cast_inputs_take_the_tuned_tiles():
    """The tiles are chosen off the inputs the kernel receives: fp32 inputs cast to bf16 under autocast at head
    dim 256 on SM100+ run the tuned tiles, as bf16 inputs do."""
    with patch.object(flex_sliding_attention, "is_blackwell_gpu", lambda: True):
        (call,) = _sliding_call(*_inputs(256, (torch.float32,) * 3), autocast=True)
    assert call["kernel_options"] is flex_sliding_attention._SM100_SLIDING_KERNEL_OPTIONS


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
