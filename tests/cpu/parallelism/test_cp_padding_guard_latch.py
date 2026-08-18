#!/usr/bin/env python
"""The CP left-padding guard must run on EVERY forward — no batch certifies the ones after it.

``_reject_left_padding`` must not report "this batch settled the collator's padding side", because
``UlyssesCPModelWrapper.forward`` would cache that verdict and stop checking. The premise of such a
latch — that trailing pads prove a right-padding collator — is false for ``DataCollatorForSMPO``,
the collator this guard exists for and the one its own error message names, which left-pads PROMPTS
and right-pads COMPLETIONS. A batch whose prompts happen to be equal-length carries only trailing
pads, so it "settles" the question and disarms the guard; every later batch with unequal prompt
lengths then carries leading pads, is never inspected, and runs dense causal attention over them —
every real token attending the pads, with a loss silently different from the same batch without CP.

Driven end-to-end through the REAL wrapper (a real Qwen3 patched by the real
``patch_attention_for_ulysses`` on a real single-rank gloo group), because the defect lives in the
CALLER's caching, not in the predicate: a test that only calls ``_reject_left_padding`` directly
passes either way.

    python tests/cpu/parallelism/test_cp_padding_guard_latch.py
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast

from src.distributed.context_parallel.config import CPConfig
from src.distributed.context_parallel.validation import SUPPORTED_ATTN_IMPLEMENTATIONS
from src.distributed.context_parallel.wrapper import UlyssesCPModelWrapper, _reject_left_padding

VOCAB = 64

UNPADDED = torch.ones(2, 8, dtype=torch.long)
# SMPO-shaped batches: prompts left-padded to a common length, completions right-padded.
# Equal-length prompts -> trailing pads only. This is the batch that would arm a latch.
ARMING = torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]])
# Unequal prompt lengths -> leading pads. Must be rejected however many batches preceded it.
LEFT_PADDED = torch.tensor([[0, 0, 1, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1, 1]])


class _LogitsOnly(torch.nn.Module):
    """Stands in for the wrapped transformer, which cannot run in the CPU tier.

    CP's patched attention calls ``flash_attn_func`` directly and flash-attn has no CPU kernel.
    Only the CP wrapper's own ``forward`` — where the guard sits, above the model call — is under
    test; the real validation and patching still run in :func:`_cp_wrapped_model`.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.proj = torch.nn.Embedding(VOCAB, VOCAB)

    def forward(self, input_ids=None, **_kwargs):
        return CausalLMOutputWithPast(logits=self.proj(input_ids).float())


def _cp_wrapped_model():
    """A real ``UlyssesCPModelWrapper`` over a tiny Qwen3, at cp_size=1 (no all-to-all needed)."""
    config = AutoConfig.for_model(
        "qwen3",
        vocab_size=VOCAB,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=False,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    # `validate_model_for_ulysses` requires a flash implementation, which transformers refuses to
    # instantiate on a CPU host (and would fetch from the hub). Build the tree eager and declare
    # the implementation the validator reads: no attention kernel is ever called.
    model.config._attn_implementation = SUPPORTED_ATTN_IMPLEMENTATIONS[0]
    wrapper = UlyssesCPModelWrapper(model, CPConfig(cp_size=1, world_size=1, gpus_per_node=1))
    wrapper.model = _LogitsOnly(config)
    return wrapper


def _forward(wrapper, mask):
    """One wrapper forward over an all-real-token batch described by ``mask`` (``None`` = no mask)."""
    ids = torch.ones_like(UNPADDED)
    return wrapper(input_ids=ids, attention_mask=mask, labels=ids)


@pytest.mark.parametrize("first", [ARMING, UNPADDED, None], ids=["trailing-pads", "unpadded", "no-mask"])
def test_left_padded_batch_is_rejected_whatever_preceded_it(tmp_path, first):
    """No earlier batch — least of all a right-padded one — may disarm the guard."""
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'pg'}")
    try:
        wrapper = _cp_wrapped_model()
        outputs = _forward(wrapper, first)  # legitimate — must not raise
        # Anti-vacuity: the first forward really went through the wrapper, so the guard was live.
        assert torch.isfinite(outputs["loss"])
        with pytest.raises(ValueError, match="LEFT-padded"):
            _forward(wrapper, LEFT_PADDED)
    finally:
        dist.destroy_process_group()


def test_predicate_reports_no_verdict_a_caller_could_cache():
    """Root cause, pinned directly: the predicate offers no padding-side answer to latch on."""
    assert _reject_left_padding(ARMING) is None
    assert _reject_left_padding(UNPADDED) is None
    assert _reject_left_padding(None) is None


def test_left_padding_still_raises():
    """Anti-vacuity: the guard itself still fires, so the tests above are not passing on a no-op."""
    with pytest.raises(ValueError, match="LEFT-padded"):
        _reject_left_padding(LEFT_PADDED)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
