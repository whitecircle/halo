#!/usr/bin/env python
"""A reward model scores on ``--rm_device cpu`` whatever its head width. For heads wider than flash runs (Gemma 4)
the seam pins ``sdpa`` process-wide to the mem-efficient kernel, which also turns off the backends CPU SDPA runs
on, so a CPU scorer under the pin raises at its first forward."""

import numpy as np
import pytest
import torch
from transformers.integrations import sdpa_attention

from halo.entrypoints import reward_scoring
from tests.common.tiny_models import build_wide_head_gemma4_reward_checkpoint

WORDS = ("a", "b", "c")
# The compute capability the mocked GPU host reports (a Hopper).
_HOST_CAPABILITY = (9, 0)


@pytest.fixture
def sdpa_backends_restored(monkeypatch):
    """The pin is process-wide: one applied here must not take CPU SDPA from the tests after this one."""
    monkeypatch.setattr(sdpa_attention, "use_gqa_in_sdpa", sdpa_attention.use_gqa_in_sdpa)
    backends = {
        torch.backends.cuda.enable_flash_sdp: torch.backends.cuda.flash_sdp_enabled(),
        torch.backends.cuda.enable_mem_efficient_sdp: torch.backends.cuda.mem_efficient_sdp_enabled(),
        torch.backends.cuda.enable_math_sdp: torch.backends.cuda.math_sdp_enabled(),
        torch.backends.cuda.enable_cudnn_sdp: torch.backends.cuda.cudnn_sdp_enabled(),
    }
    yield
    for enable, enabled in backends.items():
        enable(enabled)


@pytest.mark.parametrize("cuda_host", [False, True], ids=["cpu_host", "cuda_host"])
def test_a_wide_head_reward_model_scores_on_cpu(monkeypatch, tmp_path, sdpa_backends_restored, cuda_host):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_host)
    if cuda_host:
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *args, **kwargs: _HOST_CAPABILITY)
    build_wide_head_gemma4_reward_checkpoint(str(tmp_path), WORDS)
    tokenizer, model, device = reward_scoring.load_reward_model(
        str(tmp_path), "sdpa", 512, "cpu", "float32", trust_remote_code=False
    )
    conversation = [{"role": "user", "content": "a b c " * 20}, {"role": "assistant", "content": "c b a " * 20}]
    scores = reward_scoring.score_conversations(tokenizer, model, device, [conversation], None, 1)
    assert model.config._attn_implementation == "sdpa"
    assert scores.shape == (1,) and np.isfinite(scores).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
