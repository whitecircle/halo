#!/usr/bin/env python
"""What the first call of a kernel costs against the second: a cuBLAS GEMM (lazy library load), a Triton kernel (Liger's
SwiGLU: compile or cache load), ``torch.compile`` of a pointwise chain, and a FlashAttention forward (FA4's CuTe DSL
compile on Blackwell), in that order in one fresh process, so a later row does not pay again a one-time cost an earlier
row paid. Run it once with empty ``TRITON_CACHE_DIR``, ``TORCHINDUCTOR_CACHE_DIR`` and
``FLASH_ATTENTION_CUTE_DSL_CACHE_DIR`` and once again with them filled.

    python tests/gpu/profiling/benchmark_first_call.py --output first-call.json
"""

import argparse
import os
import time
from collections.abc import Callable

import torch
import torch.nn.functional as F
from liger_kernel.ops.swiglu import LigerSiLUMulFunction

from halo.models.attention.selection import flash_kernel, preferred_flash_impl
from tests.gpu.profiling.measurement import run_record, write_record

CACHE_VARS = ("TRITON_CACHE_DIR", "TORCHINDUCTOR_CACHE_DIR", "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR")


def _wall(fn: Callable[[], object]) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - start


def _kernels() -> dict[str, Callable[[], object]]:
    a, b = (torch.randn(8192, 8192, device="cuda", dtype=torch.bfloat16) for _ in range(2))
    q, k, v = (torch.randn(1, 4096, 16, 128, device="cuda", dtype=torch.bfloat16) for _ in range(3))
    flash = flash_kernel(preferred_flash_impl(), varlen=False)
    compiled = torch.compile(lambda x, y: F.silu(x) * y)
    return {
        "cublas_gemm_8192": lambda: a @ b,
        "liger_swiglu_triton": lambda: LigerSiLUMulFunction.apply(a, b),
        "torch_compile_silu_mul": lambda: compiled(a, b),
        f"{preferred_flash_impl()}_forward_4096": lambda: flash(q, k, v, causal=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch.cuda.set_device(0)
    record = run_record()
    # Read after the build: the FA4 kernel lookup sets FLASH_ATTENTION_CUTE_DSL_CACHE_DIR when it is unset.
    kernels = _kernels()
    record["cache_dirs"] = {var: os.environ.get(var) for var in CACHE_VARS}
    record["cache_files_before"] = {
        var: sum(len(files) for _, _, files in os.walk(os.environ[var])) if os.environ.get(var) else None
        for var in CACHE_VARS
    }
    record["rows"] = []
    for name, fn in kernels.items():
        first, second, third = _wall(fn), _wall(fn), _wall(fn)
        record["rows"].append({"kernel": name, "first_s": first, "second_s": second, "third_s": third})
        print(
            f"  {name:>32}  first {first * 1e3:10.2f} ms  second {second * 1e3:8.3f} ms  third {third * 1e3:8.3f} ms"
        )
    write_record(args.output, record)


if __name__ == "__main__":
    main()
