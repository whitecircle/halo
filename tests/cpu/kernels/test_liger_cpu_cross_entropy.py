"""CPU test for the ``liger_cross_entropy`` CPU fallback (``src/kernels/liger/cross_entropy.py``).

Every toolkit applier routes ``transformers.loss.loss_utils``' cross-entropy through
``liger_cross_entropy``; Liger's kernel is Triton/CUDA-only, so a CPU-side CE call on that path must
fall back to torch instead of crashing inside the Triton launcher.

    python tests/cpu/kernels/test_liger_cpu_cross_entropy.py
"""

import pytest
import torch
import torch.nn.functional as F

from src.kernels.liger.cross_entropy import liger_cross_entropy


def test_cpu_tensors_fall_back_to_torch():
    torch.manual_seed(0)
    logits = torch.randn(4, 10, requires_grad=True)
    target = torch.randint(0, 10, (4,))
    loss = liger_cross_entropy(logits, target)
    assert torch.allclose(loss, F.cross_entropy(logits.detach(), target))
    loss.backward()  # the fallback keeps the autograd path intact
    assert logits.grad is not None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
