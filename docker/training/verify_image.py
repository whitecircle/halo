"""Build-time checks of the training image: the framework stack, the attention kernels of the target GPU,
the cutlass-dsl the locked install re-pins, the optional extras and the installed toolkit all import.

GPU-dependent packages (DeepEP) verify at runtime with ``--gpus``. Run last, on the image that ships:

    python verify_image.py blackwell|hopper
"""

import argparse
import ctypes
import importlib.metadata

import accelerate
import cutlass
import cutlass._mlir
import cutlass.cute.nvgpu
import flash_attn
import gigatoken  # noqa: F401
import nvidia.nvshmem  # noqa: F401
import torch
import transformers
import trl
from flash_attn.flash_attn_interface import flash_attn_func as fa2_flash_attn_func  # noqa: F401
from flashoptim import FlashAdamW  # noqa: F401
from transformers.utils import is_flash_attn_3_available

from halo.weight_sync.clients.vllm import VLLMWeightSyncClient  # noqa: F401


def check_flash_attention_4() -> None:
    from flash_attn.cute import flash_attn_func as fa4_flash_attn_func  # noqa: F401, PLC0415 — Blackwell image only

    print("FA4 flash_attn.cute import OK")


def check_flash_attention_3() -> None:
    """Both import spellings transformers tries, and the version metadata the build writes for them."""
    from flash_attn_3 import flash_attn_func as fa3_flash_attn_func  # noqa: F401, PLC0415 — Hopper image only
    from flash_attn_interface import flash_attn_func as fa3_interface_flash_attn_func  # noqa: F401, PLC0415

    print("FA3 via flash_attn_3: OK")
    print("FA3 via flash_attn_interface: OK")
    print(f"FA3 version via metadata: {importlib.metadata.version('flash_attn_3')}")
    print(f"  - transformers FA3 detection: {is_flash_attn_3_available()}")


def check_cublas_matches_lock() -> None:
    """The cuBLAS torch loaded is the locked ``nvidia-cublas`` wheel, not the system copy."""
    torch.zeros(1)
    library = ctypes.CDLL("libcublas.so.13")
    loaded = []
    for property_type in (0, 1, 2):  # MAJOR_VERSION, MINOR_VERSION, PATCH_LEVEL
        value = ctypes.c_int()
        library.cublasGetProperty(property_type, ctypes.byref(value))
        loaded.append(value.value)
    wheel = importlib.metadata.version("nvidia-cublas")
    if loaded != [int(part) for part in wheel.split(".")[:3]]:
        raise SystemExit(f"loaded cuBLAS {loaded} differs from the nvidia-cublas wheel {wheel}")
    print(f"cuBLAS {wheel}: OK")


ATTENTION_CHECKS = {"blackwell": check_flash_attention_4, "hopper": check_flash_attention_3}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("target_gpu", choices=sorted(ATTENTION_CHECKS))
    target_gpu = parser.parse_args().target_gpu
    print(f"PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}")
    print(f"Transformers: {transformers.__version__}")
    print(f"TRL: {trl.__version__}")
    print(f"Accelerate: {accelerate.__version__}")
    print(f"Flash Attention 2: {flash_attn.__version__}")
    print("  - FA2 flash_attn_func: OK")
    print("cutlass-dsl import OK:", cutlass.__file__)
    check_cublas_matches_lock()
    ATTENTION_CHECKS[target_gpu]()
    print("flashoptim OK")
    print("gigatoken OK")
    print("NVSHMEM: installed")
    print("Weight-sync client: OK")


if __name__ == "__main__":
    main()
