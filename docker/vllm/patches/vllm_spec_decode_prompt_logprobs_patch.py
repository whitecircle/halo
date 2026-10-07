"""Keep vLLM's prompt log-probs off the hidden-state buffer the speculative drafter overwrites.

``GPUModelRunner.sample_tokens`` runs a padded-batch drafter (MTP, EAGLE, a draft model) before
``_bookkeeping_sync`` reads the target's ``hidden_states`` for prompt log-probs. When the target
forward ran as a CUDA graph (a padded batch inside the capture range, 512 tokens by default), those
hidden states are the graph's output in the global graph pool, which the drafter's graphs share, so
the drafter's replay overwrites them first: every prompt position of the request comes back garbage
(NLL 12-22), deterministically, while the sampled tokens stay correct (their logits are taken before
the drafter runs). Upstream: vllm-project/vllm#53488; the unmerged fix #53520 clones the same tensor.

Fix: on a step that schedules a prompt-log-prob request while speculation is on, the pending
``ExecuteModelState`` carries a clone of ``hidden_states``, which the drafter and the bookkeeping
then both read; the drafter's graphs never own that storage. Every other step is untouched. Only the
V1 runner is patched: the V2 runner takes prompt log-probs before it proposes drafts. Drop this file
once the pinned vLLM carries #53520.
"""

from __future__ import annotations

import logging

import torch

logger = logging.getLogger(__name__)

# Stamped on the replacement so ``Dockerfile.vllm`` can assert the patch took.
PATCH_MARKER = "_halo_spec_decode_prompt_logprobs"

_APPLIED = False


def _schedules_prompt_logprobs(runner, scheduler_output) -> bool:
    """Whether this step prefills a request that asked for prompt log-probs."""
    scheduled = scheduler_output.num_scheduled_tokens
    return any(req_id in scheduled for req_id in runner.num_prompt_logprobs)


def apply() -> None:
    """Idempotently patch ``GPUModelRunner.sample_tokens`` to read prompt log-probs from a private copy."""
    global _APPLIED
    if _APPLIED:
        return

    from vllm.v1.worker import gpu_model_runner  # noqa: PLC0415 — vLLM-only lazy import

    runner_cls = gpu_model_runner.GPUModelRunner
    original_sample_tokens = runner_cls.sample_tokens

    @torch.inference_mode()
    def sample_tokens(self, grammar_output):
        state = self.execute_model_state
        if (
            state is not None
            and self.speculative_config is not None
            and _schedules_prompt_logprobs(self, state.scheduler_output)
        ):
            self.execute_model_state = state._replace(hidden_states=state.hidden_states.clone())
        return original_sample_tokens(self, grammar_output)

    setattr(sample_tokens, PATCH_MARKER, True)
    runner_cls.sample_tokens = sample_tokens
    _APPLIED = True
    logger.info("vllm_spec_decode_prompt_logprobs_patch applied to %s.sample_tokens", runner_cls.__name__)
