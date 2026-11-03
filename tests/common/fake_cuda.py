"""A CPU tensor reporting ``is_cuda``, for tests of code branching to a CUDA-only path on that attribute alone."""

import torch


class CudaLike(torch.Tensor):
    """``tensor.as_subclass(CudaLike)``; ops return the subclass, so a kernel stand-in takes
    ``as_subclass(torch.Tensor)`` back first."""

    is_cuda = True
