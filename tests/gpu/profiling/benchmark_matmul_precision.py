#!/usr/bin/env python
"""Numerical error of reduced-precision arithmetic on one GPU against an fp64 reference computed from the same rounded
inputs: a long bf16 dot product through a matmul (fp32 and bf16 output) and through reductions (``torch.sum``, and a
running bf16 accumulator), and square GEMMs at each matmul precision.

    python tests/gpu/profiling/benchmark_matmul_precision.py --output precision.json
"""

import argparse
import os

import torch

from tests.gpu.profiling.measurement import GEMM_PRECISIONS, TF32_OVERRIDE_VAR, run_record, write_record

SEED = 0
# Elements summed into each step of the running bf16 accumulator.
RUNNING_CHUNK = 256


def _relative(value: torch.Tensor, reference: torch.Tensor) -> float:
    return ((value.double() - reference).abs() / reference.abs()).item()


def _dot(length: int, generator: torch.Generator) -> dict:
    """Uniform [0, 1) operands, so terms never cancel and every lost bit shows."""
    a = torch.rand(length, device="cuda", generator=generator).bfloat16()
    b = torch.rand(length, device="cuda", generator=generator).bfloat16()
    products = a.double() * b.double()
    reference = products.sum()
    # Each product rounded to bf16 once, as a bf16 elementwise op leaves it.
    terms = (a.float() * b.float()).bfloat16()
    running = torch.zeros((), device="cuda", dtype=torch.bfloat16)
    for chunk in terms.view(-1, RUNNING_CHUNK):
        running += chunk.sum()
    return {
        "length": length,
        "matmul_fp32_out": _relative(torch.mm(a.view(1, -1), b.view(-1, 1), out_dtype=torch.float32), reference),
        "matmul_bf16_out": _relative(a.view(1, -1) @ b.view(-1, 1), reference),
        "sum_of_bf16_products_bf16_result": _relative(terms.sum(), terms.double().sum()),
        "sum_of_bf16_products_fp32_result": _relative(terms.sum(dtype=torch.float32), terms.double().sum()),
        f"running_bf16_accumulator_chunks_of_{RUNNING_CHUNK}": _relative(running, terms.double().sum()),
    }


def _gemm(dim: int, generator: torch.Generator) -> list[dict]:
    a64 = torch.randn(dim, dim, device="cuda", generator=generator, dtype=torch.float64)
    b64 = torch.randn(dim, dim, device="cuda", generator=generator, dtype=torch.float64)
    rows = []
    for name, (dtype, precision) in GEMM_PRECISIONS.items():
        torch.set_float32_matmul_precision(precision)
        a, b = a64.to(dtype), b64.to(dtype)
        reference = a.double() @ b.double()
        error = (a @ b).double() - reference
        rows.append(
            {
                "precision": name,
                "dim": dim,
                "frobenius_relative_error": (error.norm() / reference.norm()).item(),
                "max_abs_error_over_max_abs_reference": (error.abs().max() / reference.abs().max()).item(),
            }
        )
    torch.set_float32_matmul_precision("highest")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dot_lengths", type=int, nargs="+", default=tuple(2**e for e in range(10, 25, 2)))
    parser.add_argument("--gemm_dims", type=int, nargs="+", default=(1024, 4096))
    args = parser.parse_args()

    if os.environ.get(TF32_OVERRIDE_VAR, "0") != "0":
        raise SystemExit(
            f"{TF32_OVERRIDE_VAR}={os.environ[TF32_OVERRIDE_VAR]} lets cuBLAS run fp32 GEMMs on TF32 at precision "
            f"'highest', so the fp32 rows would measure TF32; unset it or set it to 0."
        )
    torch.cuda.set_device(0)
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    record = {**run_record(), "dot": [], "gemm": []}
    for length in args.dot_lengths:
        record["dot"].append(_dot(length, generator))
        print(record["dot"][-1], flush=True)
    for dim in args.gemm_dims:
        record["gemm"] += _gemm(dim, generator)
        print(record["gemm"][-len(GEMM_PRECISIONS) :], flush=True)
    write_record(args.output, record)


if __name__ == "__main__":
    main()
