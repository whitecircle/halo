"""The ``liger_cross_entropy`` CPU fallback (``halo/model_loading/liger/cross_entropy.py``): Liger's kernel is
Triton/CUDA-only, so CPU falls back to torch."""

import pytest
import torch
import torch.nn.functional as F

from halo.model_loading.liger.cross_entropy import liger_cross_entropy


def test_cpu_tensors_fall_back_to_torch():
    torch.manual_seed(0)
    logits = torch.randn(4, 10, requires_grad=True)
    target = torch.randint(0, 10, (4,))
    loss = liger_cross_entropy(logits, target)
    assert torch.allclose(loss, F.cross_entropy(logits.detach(), target))
    loss.backward()  # The fallback keeps the autograd path intact.
    assert logits.grad is not None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
