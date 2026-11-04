"""A derivation from a tensor, kept for every later reader of the same contents: the layers of one forward, and a
reentrant checkpoint's recompute, whose detached input copies share the tensor's storage and version counter."""

from __future__ import annotations

from collections.abc import Callable

import torch


class TensorMemo[T]:
    """``derive(tensor)`` for the last tensor asked about. The memo keeps that tensor alive, as freed storage could
    come back to a later tensor of other contents at the same address, shape and version."""

    def __init__(self, derive: Callable[[torch.Tensor], T]):
        self._derive = derive
        self._source: torch.Tensor | None = None
        self._key: tuple | None = None
        self._value: T | None = None

    @torch.compiler.disable
    def __call__(self, tensor: torch.Tensor) -> T:
        if tensor.is_inference():
            return self._derive(tensor)
        key = (
            tensor.device,
            tensor.untyped_storage().data_ptr(),
            tensor.storage_offset(),
            tuple(tensor.shape),
            tensor.stride(),
            tensor._version,
        )
        if key != self._key:
            self._value = self._derive(tensor)
            self._source, self._key = tensor, key
        return self._value
