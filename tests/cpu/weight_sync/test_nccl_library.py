#!/usr/bin/env python
"""The ctypes libnccl binding rank 0 of a vLLM weight-transfer group calls through: each library path is
loaded and typed once, a failed call raises with NCCL's own error string, and a dtype NCCL cannot carry
is refused before it reaches the wire.

    python tests/cpu/weight_sync/test_nccl_library.py
"""

import pytest
import torch

from src.distributed.nccl import _nccl_wrapper
from src.distributed.nccl._nccl_wrapper import NCCLLibrary, nccl_data_type

_NCCL_INVALID_ARGUMENT = 4


class _FakeFunction:
    """A libnccl symbol: records its calls and answers with ``result``."""

    def __init__(self, result=0):
        self.result = result
        self.calls: list[tuple] = []
        self.restype = None
        self.argtypes = None

    def __call__(self, *args):
        self.calls.append(args)
        return self.result


class _FakeLibrary:
    def __init__(self):
        self.ncclGetErrorString = _FakeFunction(b"invalid argument")
        self.ncclGetVersion = _FakeFunction()
        self.ncclGetUniqueId = _FakeFunction()
        self.ncclCommInitRank = _FakeFunction()
        self.ncclAllReduce = _FakeFunction()
        self.ncclBroadcast = _FakeFunction(_NCCL_INVALID_ARGUMENT)
        self.ncclCommAbort = _FakeFunction()


@pytest.fixture
def loaded(monkeypatch) -> list[tuple[str, _FakeLibrary]]:
    """Every ``ctypes.CDLL`` load as ``(path, library)``, with the per-path cache emptied around the test."""
    loads: list[tuple[str, _FakeLibrary]] = []

    def cdll(path):
        library = _FakeLibrary()
        loads.append((path, library))
        return library

    monkeypatch.setattr(_nccl_wrapper.ctypes, "CDLL", cdll)
    monkeypatch.delenv("VLLM_NCCL_SO_PATH", raising=False)
    _nccl_wrapper._bind_functions.cache_clear()
    yield loads
    _nccl_wrapper._bind_functions.cache_clear()


def test_each_library_path_is_loaded_and_typed_once(loaded, monkeypatch):
    NCCLLibrary()
    NCCLLibrary()
    monkeypatch.setenv("VLLM_NCCL_SO_PATH", "/opt/nccl/libnccl.so.2")
    NCCLLibrary()

    assert [path for path, _ in loaded] == ["libnccl.so.2", "/opt/nccl/libnccl.so.2"], (
        "the library must load once per path, and VLLM_NCCL_SO_PATH must pick the path"
    )
    _, library = loaded[0]
    for name, restype, argtypes in _nccl_wrapper._FUNCTIONS:
        function = getattr(library, name)
        assert (function.restype, function.argtypes) == (restype, argtypes), f"{name} was bound untyped"


def test_a_failed_call_raises_with_the_nccl_error_string(loaded):
    library = NCCLLibrary()

    with pytest.raises(RuntimeError, match="NCCL error: invalid argument"):
        library.ncclBroadcast(None, None, 1, nccl_data_type(torch.uint8), 0, None, None)

    _, fake = loaded[0]
    assert fake.ncclGetErrorString.calls == [(_NCCL_INVALID_ARGUMENT,)], "the error string was not read for the result"


def test_init_rank_passes_the_arguments_in_nccl_order(loaded):
    """``ncclCommInitRank(comm*, nranks, id, rank)``: a swapped count and rank still type-checks in ctypes."""
    library = NCCLLibrary()
    unique_id = _nccl_wrapper.ncclUniqueId()

    library.ncclCommInitRank(2, unique_id, 0)

    _, fake = loaded[0]
    [(_, world_size, passed_id, rank)] = fake.ncclCommInitRank.calls
    assert (world_size, passed_id, rank) == (2, unique_id, 0)


@pytest.mark.parametrize(
    ("dtype", "nccl_type"),
    [(torch.uint8, 1), (torch.float16, 6), (torch.float32, 7), (torch.bfloat16, 9), (torch.float8_e4m3fn, 10)],
)
def test_weight_dtypes_map_to_the_nccl_numbering(dtype, nccl_type):
    assert nccl_data_type(dtype) == nccl_type


def test_a_dtype_nccl_cannot_carry_is_refused():
    with pytest.raises(ValueError, match="Unsupported dtype"):
        nccl_data_type(torch.complex64)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
