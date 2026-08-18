"""Pure-Python ctypes wrapper for libnccl.so.2, exposing only the functions weight sync needs. Vendored from vLLM v0.18.0, Apache-2.0."""

import ctypes
import logging
import platform
from dataclasses import dataclass
from typing import Any

import torch

from src.env import env_str

logger = logging.getLogger(__name__)

ncclResult_t = ctypes.c_int
ncclComm_t = ctypes.c_void_p


class ncclUniqueId(ctypes.Structure):
    _fields_ = [("internal", ctypes.c_byte * 128)]


cudaStream_t = ctypes.c_void_p
buffer_type = ctypes.c_void_p
ncclDataType_t = ctypes.c_int
ncclRedOp_t = ctypes.c_int


class ncclDataTypeEnum:
    ncclInt8 = 0
    ncclUint8 = 1
    ncclInt32 = 2
    ncclInt64 = 4
    ncclFloat16 = 6
    ncclFloat32 = 7
    ncclFloat64 = 8
    ncclBfloat16 = 9
    ncclFloat8e4m3 = 10

    @classmethod
    def from_torch(cls, dtype: torch.dtype) -> int:
        mapping = {
            torch.int8: cls.ncclInt8,
            torch.uint8: cls.ncclUint8,
            torch.int32: cls.ncclInt32,
            torch.int64: cls.ncclInt64,
            torch.float16: cls.ncclFloat16,
            torch.float32: cls.ncclFloat32,
            torch.float64: cls.ncclFloat64,
            torch.bfloat16: cls.ncclBfloat16,
            torch.float8_e4m3fn: cls.ncclFloat8e4m3,
        }
        if dtype not in mapping:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return mapping[dtype]


def _find_nccl_library() -> str:
    so_file = env_str("VLLM_NCCL_SO_PATH")
    if so_file:
        logger.info("Found nccl from VLLM_NCCL_SO_PATH=%s", so_file)
        return so_file
    return "libnccl.so.2"


@dataclass
class Function:
    name: str
    restype: Any
    argtypes: list[Any]


class NCCLLibrary:
    """Thin ctypes wrapper around libnccl.so.2 (only the weight-sync functions)."""

    exported_functions = [
        Function("ncclGetErrorString", ctypes.c_char_p, [ncclResult_t]),
        Function("ncclGetVersion", ncclResult_t, [ctypes.POINTER(ctypes.c_int)]),
        Function("ncclGetUniqueId", ncclResult_t, [ctypes.POINTER(ncclUniqueId)]),
        Function(
            "ncclCommInitRank",
            ncclResult_t,
            [
                ctypes.POINTER(ncclComm_t),
                ctypes.c_int,
                ncclUniqueId,
                ctypes.c_int,
            ],
        ),
        Function(
            "ncclAllReduce",
            ncclResult_t,
            [
                buffer_type,
                buffer_type,
                ctypes.c_size_t,
                ncclDataType_t,
                ncclRedOp_t,
                ncclComm_t,
                cudaStream_t,
            ],
        ),
        Function(
            "ncclBroadcast",
            ncclResult_t,
            [
                buffer_type,
                buffer_type,
                ctypes.c_size_t,
                ncclDataType_t,
                ctypes.c_int,
                ncclComm_t,
                cudaStream_t,
            ],
        ),
        Function("ncclCommAbort", ncclResult_t, [ncclComm_t]),
    ]

    path_to_library_cache: dict[str, Any] = {}
    path_to_dict_mapping: dict[str, dict[str, Any]] = {}

    def __init__(self):
        so_file = _find_nccl_library()
        try:
            if so_file not in NCCLLibrary.path_to_dict_mapping:
                lib = ctypes.CDLL(so_file)
                NCCLLibrary.path_to_library_cache[so_file] = lib
            self.lib = NCCLLibrary.path_to_library_cache[so_file]
        except Exception as e:
            logger.error(
                "Failed to load NCCL library from %s (%s). Set VLLM_NCCL_SO_PATH to the correct path.",
                so_file,
                platform.platform(),
            )
            raise e

        if so_file not in NCCLLibrary.path_to_dict_mapping:
            _funcs: dict[str, Any] = {}
            for func in NCCLLibrary.exported_functions:
                f = getattr(self.lib, func.name)
                f.restype = func.restype
                f.argtypes = func.argtypes
                _funcs[func.name] = f
            NCCLLibrary.path_to_dict_mapping[so_file] = _funcs
        self._funcs = NCCLLibrary.path_to_dict_mapping[so_file]

    def _check(self, result: ncclResult_t) -> None:
        if result != 0:
            error_str = self._funcs["ncclGetErrorString"](result).decode("utf-8")
            raise RuntimeError(f"NCCL error: {error_str}")

    def ncclGetVersion(self) -> str:
        version = ctypes.c_int()
        self._check(self._funcs["ncclGetVersion"](ctypes.byref(version)))
        v = str(version.value)
        return f"{v[0]}.{v[1:3].lstrip('0')}.{v[3:].lstrip('0')}"

    def ncclGetUniqueId(self) -> ncclUniqueId:
        unique_id = ncclUniqueId()
        self._check(self._funcs["ncclGetUniqueId"](ctypes.byref(unique_id)))
        return unique_id

    def ncclCommInitRank(self, world_size: int, unique_id: ncclUniqueId, rank: int) -> ncclComm_t:
        comm = ncclComm_t()
        self._check(self._funcs["ncclCommInitRank"](ctypes.byref(comm), world_size, unique_id, rank))
        return comm

    def ncclAllReduce(self, sendbuff, recvbuff, count, datatype, op, comm, stream):
        self._check(self._funcs["ncclAllReduce"](sendbuff, recvbuff, count, datatype, op, comm, stream))

    def ncclBroadcast(self, sendbuff, recvbuff, count, datatype, root, comm, stream):
        self._check(self._funcs["ncclBroadcast"](sendbuff, recvbuff, count, datatype, root, comm, stream))

    def ncclCommAbort(self, comm: ncclComm_t) -> None:
        self._check(self._funcs["ncclCommAbort"](comm))


__all__ = ["NCCLLibrary", "ncclDataTypeEnum", "ncclUniqueId", "ncclComm_t", "cudaStream_t", "buffer_type"]
