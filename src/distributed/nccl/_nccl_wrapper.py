"""ctypes binding of the libnccl calls rank 0 of a vLLM weight-transfer group makes. Vendored from vLLM v0.18.0
(``vllm.distributed.device_communicators.pynccl_wrapper``), Apache-2.0.

Wire contract: the trainer publishes ``ncclUniqueId`` pickled from this module path, and the vLLM image
unpickles it through a shim at the same path that re-exports vLLM's own class (``Dockerfile.vllm``). The
path, the class name and its ``_fields_`` are pinned by ``tests/cpu/weight_sync/test_wire_pins.py``.
"""

import ctypes
import functools
import logging
import platform
from collections.abc import Callable

import torch

from src.env import env_str

logger = logging.getLogger(__name__)

ncclResult_t = ctypes.c_int
ncclComm_t = ctypes.c_void_p
cudaStream_t = ctypes.c_void_p
buffer_type = ctypes.c_void_p
ncclDataType_t = ctypes.c_int
ncclRedOp_t = ctypes.c_int


class ncclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


# ncclDataType_t of each dtype a weight can carry, numbered as nccl.h numbers them.
_NCCL_DATA_TYPES = {
    torch.int8: 0,
    torch.uint8: 1,
    torch.int32: 2,
    torch.int64: 4,
    torch.float16: 6,
    torch.float32: 7,
    torch.float64: 8,
    torch.bfloat16: 9,
    torch.float8_e4m3fn: 10,
}

# (name, restype, argtypes) of every libnccl function bound.
_FUNCTIONS = (
    ("ncclGetErrorString", ctypes.c_char_p, [ncclResult_t]),
    ("ncclGetVersion", ncclResult_t, [ctypes.POINTER(ctypes.c_int)]),
    ("ncclGetUniqueId", ncclResult_t, [ctypes.POINTER(ncclUniqueId)]),
    ("ncclCommInitRank", ncclResult_t, [ctypes.POINTER(ncclComm_t), ctypes.c_int, ncclUniqueId, ctypes.c_int]),
    (
        "ncclAllReduce",
        ncclResult_t,
        [buffer_type, buffer_type, ctypes.c_size_t, ncclDataType_t, ncclRedOp_t, ncclComm_t, cudaStream_t],
    ),
    (
        "ncclBroadcast",
        ncclResult_t,
        [buffer_type, buffer_type, ctypes.c_size_t, ncclDataType_t, ctypes.c_int, ncclComm_t, cudaStream_t],
    ),
    ("ncclCommAbort", ncclResult_t, [ncclComm_t]),
)


def nccl_data_type(dtype: torch.dtype) -> int:
    """The ``ncclDataType_t`` that carries ``dtype``."""
    try:
        return _NCCL_DATA_TYPES[dtype]
    except KeyError:
        raise ValueError(f"Unsupported dtype: {dtype}") from None


@functools.cache
def _bind_functions(so_file: str) -> dict[str, Callable]:
    """The typed functions of the libnccl at ``so_file``, loaded once per path."""
    try:
        library = ctypes.CDLL(so_file)
    except Exception:
        logger.error(
            "Failed to load NCCL library from %s (%s). Set VLLM_NCCL_SO_PATH to the correct path.",
            so_file,
            platform.platform(),
        )
        raise
    functions = {}
    for name, restype, argtypes in _FUNCTIONS:
        function = getattr(library, name)
        function.restype, function.argtypes = restype, argtypes
        functions[name] = function
    return functions


class NCCLLibrary:
    """libnccl at ``VLLM_NCCL_SO_PATH``, else ``libnccl.so.2``; every call raises on a failed ``ncclResult_t``."""

    def __init__(self):
        so_file = env_str("VLLM_NCCL_SO_PATH")
        if so_file:
            logger.info("Found nccl from VLLM_NCCL_SO_PATH=%s", so_file)
        self._functions = _bind_functions(so_file or "libnccl.so.2")

    def _call(self, name: str, *args) -> None:
        result = self._functions[name](*args)
        if result != 0:
            raise RuntimeError(f"NCCL error: {self._functions['ncclGetErrorString'](result).decode('utf-8')}")

    def ncclGetVersion(self) -> str:
        version = ctypes.c_int()
        self._call("ncclGetVersion", ctypes.byref(version))
        v = str(version.value)
        return f"{v[0]}.{v[1:3].lstrip('0')}.{v[3:].lstrip('0')}"

    def ncclGetUniqueId(self) -> ncclUniqueId:
        unique_id = ncclUniqueId()
        self._call("ncclGetUniqueId", ctypes.byref(unique_id))
        return unique_id

    def ncclCommInitRank(self, world_size: int, unique_id: ncclUniqueId, rank: int) -> ncclComm_t:
        comm = ncclComm_t()
        self._call("ncclCommInitRank", ctypes.byref(comm), world_size, unique_id, rank)
        return comm

    def ncclAllReduce(self, sendbuff, recvbuff, count, datatype, op, comm, stream) -> None:
        self._call("ncclAllReduce", sendbuff, recvbuff, count, datatype, op, comm, stream)

    def ncclBroadcast(self, sendbuff, recvbuff, count, datatype, root, comm, stream) -> None:
        self._call("ncclBroadcast", sendbuff, recvbuff, count, datatype, root, comm, stream)

    def ncclCommAbort(self, comm: ncclComm_t) -> None:
        self._call("ncclCommAbort", comm)
