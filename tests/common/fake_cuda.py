"""A CPU tensor that reports ``is_cuda``, for CPU tests of code that branches to a CUDA-only path on that
attribute alone."""

import torch


class CudaLike(torch.Tensor):
    """``tensor.as_subclass(CudaLike)`` shadows ``is_cuda`` on a CPU tensor. Ops on it return the subclass, so a
    stand-in for the CUDA kernel takes ``as_subclass(torch.Tensor)`` back before computing."""

    is_cuda = True
