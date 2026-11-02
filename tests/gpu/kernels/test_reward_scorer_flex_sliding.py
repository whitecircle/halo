#!/usr/bin/env python
"""A Gemma 4 reward model scores on its CUDA default, ``sdpa_flex_sliding``, as it does on plain SDPA.

``--rm_attn_implementation`` defaults to ``sdpa``, which ``load_reward_model`` hands to ``apply_family_patches``, so
on CUDA a wide-head model builds ``sdpa_flex_sliding``. The scoring scripts run the forward under
``torch.inference_mode`` on their one executor thread, with no warm-up, so the sliding layers' FlexAttention first
compiles there. A tiny random-init Gemma 4 sequence classifier (one sliding layer, one 512-wide global layer) scores
two right-padded batches of conversations longer than its window and the 128-token tile, through
``load_reward_model`` and ``score_conversations_offloaded`` as the scripts call them, on the default and under
``HALO_FLEX_SLIDING=0``, against an fp32 SDPA load. The default must build the variant and run every sliding call on
FlexAttention; in fp32 its scores must match the reference to the kernel's TF32 round-off (a sliding layer that
ignored its window moves them by about their own size), and in bf16 stay within twice SDPA's own bf16 error.

Run: torchrun --nproc_per_node=1 tests/gpu/kernels/test_reward_scorer_flex_sliding.py
"""

import asyncio
import math
import os
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import patch

import numpy as np

from halo.entrypoints.reward_scoring import load_reward_model, score_conversations_offloaded
from halo.model_loading.dtype import configure_float32_matmul_precision
from halo.models.attention import sdpa_flex_sliding
from halo.models.attention.sdpa_flex_sliding import SDPA_FLEX_SLIDING
from tests.common.harness import gpu_test_main, record_check
from tests.common.rank_log import log
from tests.common.tiny_models import build_wide_head_gemma4_reward_checkpoint

SEED = 0
WORDS = tuple(f"w{i}" for i in range(200))
# Words per conversation, all past the window and the tile; two batches of three, each padded to its longest.
CONVERSATION_WORDS = ((300, 260, 210), (170, 290, 150))
BATCH_SIZE = 3
# The compiled kernel's fp32 dots run in TF32.
FP32_ERROR_BOUND = 1e-2
# The two bf16 scorers accumulate differently; the flex one may trail SDPA's bf16 error by this factor.
SDPA_ERROR_RATIO = 2.0
ERROR_FLOOR = 1e-3


def _conversations() -> list[list[dict]]:
    generator = np.random.default_rng(SEED)
    conversations = []
    for lengths in CONVERSATION_WORDS:
        for length in lengths:
            words = generator.choice(WORDS, size=length)
            prompt, answer = " ".join(words[: length // 2]), " ".join(words[length // 2 :])
            conversations.append([{"role": "user", "content": prompt}, {"role": "assistant", "content": answer}])
    return conversations


def _score(executor, checkpoint: str, device: str, dtype: str, *, flex: bool) -> tuple[np.ndarray, str, int]:
    """The scores, the built attention implementation and the number of attention modules FlexAttention serves a
    call of, of a ``load_reward_model`` load on the scripts' path."""
    env = {} if flex else {"HALO_FLEX_SLIDING": "0"}
    with patch.dict(os.environ, env):
        tokenizer, model, rm_device = load_reward_model(
            checkpoint, "sdpa", 4096, device, dtype, trust_remote_code=False
        )
    scores = asyncio.run(
        score_conversations_offloaded(executor, tokenizer, model, rm_device, _conversations(), None, BATCH_SIZE)
    )
    flex_modules = sum(sdpa_flex_sliding._flex_call(module) is not None for module in model.modules())
    return scores, model.config._attn_implementation, flex_modules


def _relative_error(scores: np.ndarray, reference: np.ndarray) -> float:
    return float(np.abs(scores - reference).max() / np.abs(reference).max())


def test_flex_sliding_scores_match_sdpa(ctx) -> None:
    configure_float32_matmul_precision()
    checkpoint = os.path.join(ctx.output_dir, "gemma4_reward_model")
    build_wide_head_gemma4_reward_checkpoint(checkpoint, WORDS, seed=SEED)
    device = str(ctx.device)
    flex_calls = []
    block_sparse = sdpa_flex_sliding._block_sparse_attention

    def counting(*args, **kwargs):
        flex_calls.append(1)
        return block_sparse(*args, **kwargs)

    with (
        ThreadPoolExecutor(max_workers=1) as executor,
        patch.object(sdpa_flex_sliding, "_block_sparse_attention", counting),
    ):
        reference, reference_impl, _ = _score(executor, checkpoint, device, "float32", flex=False)
        sdpa, sdpa_impl, _ = _score(executor, checkpoint, device, "bfloat16", flex=False)
        assert flex_calls == [], f"{len(flex_calls)} FlexAttention calls with HALO_FLEX_SLIDING=0"
        flex_fp32, flex_fp32_impl, flex_fp32_modules = _score(executor, checkpoint, device, "float32", flex=True)
        flex, flex_impl, flex_modules = _score(executor, checkpoint, device, "bfloat16", flex=True)
    assert (reference_impl, sdpa_impl) == ("sdpa", "sdpa")
    assert flex_fp32_impl == flex_impl == SDPA_FLEX_SLIDING
    # FlexAttention serves the sliding layer; the 512-wide global layer runs matmul attention.
    assert (flex_fp32_modules, flex_modules) == (1, 1), "premise: one module FlexAttention serves per load"
    # Every batch is right-padded past the window and the tile, so each served module runs FlexAttention once.
    batches = math.ceil(len(_conversations()) / BATCH_SIZE)
    assert len(flex_calls) == batches * (flex_fp32_modules + flex_modules), "every batch's sliding layer must run it"
    assert all(np.isfinite(scores).all() for scores in (flex_fp32, flex, sdpa))
    fp32_error = _relative_error(flex_fp32, reference)
    flex_error, sdpa_error = _relative_error(flex, reference), _relative_error(sdpa, reference)
    log(
        f"  relative score error vs fp32 SDPA: flex fp32 {fp32_error:.2e}  bf16 flex {flex_error:.2e} sdpa {sdpa_error:.2e}"
    )
    assert fp32_error <= FP32_ERROR_BOUND, fp32_error
    assert flex_error <= SDPA_ERROR_RATIO * sdpa_error + ERROR_FLOOR, (flex_error, sdpa_error)


@gpu_test_main(exact_world_size=1, prefix="reward_flex_sliding")
def run(ctx) -> dict:
    checks: dict[str, bool] = {}
    record_check(checks, "flex_sliding_scores_match_sdpa", lambda: test_flex_sliding_scores_match_sdpa(ctx))
    return {"checks": checks}


if __name__ == "__main__":
    run()
