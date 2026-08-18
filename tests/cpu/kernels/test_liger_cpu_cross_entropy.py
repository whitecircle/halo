"""CPU test for the ``liger_cross_entropy`` CPU fallback (``src/kernels/liger/cross_entropy.py``).

The mistral4/zaya appliers patch ``F.cross_entropy`` process-globally; Liger's kernel is
Triton/CUDA-only, so a CPU-side CE call in the same process must fall back to torch instead of
crashing inside the Triton launcher.

    python tests/cpu/kernels/test_liger_cpu_cross_entropy.py
"""

import sys

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
    sys.exit(pytest.main([__file__, "-v"]))
