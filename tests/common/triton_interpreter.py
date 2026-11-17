"""The interpreted kernel tests' preflight: one kernel with a runtime loop bound, the pattern the route-sort and fused
AdamWBF16 kernels take, run under ``TRITON_INTERPRET=1`` set before Triton is imported."""

import torch
import triton
import triton.language as tl

INTERPRETER_MESSAGE = (
    "the CPU tier needs the image's Triton build (make test-cpu): this Triton's interpreter cannot run a kernel with a "
    "runtime loop bound, so the interpreted tests of such kernels fail"
)
_LENGTH = 10
_CHUNK = 4


@triton.jit
def _count_chunks(out_ptr, length, CHUNK: tl.constexpr):
    chunks = 0
    for _ in range(0, length, CHUNK):
        chunks += 1
    tl.store(out_ptr, chunks)


def interpreter_findings() -> list[str]:
    """Empty when the interpreter counts the ``ceil(10 / 4) = 3`` chunks of a runtime bound; else what it did."""
    out = torch.zeros(1, dtype=torch.int32)
    try:
        _count_chunks[(1,)](out, _LENGTH, CHUNK=_CHUNK)
    except Exception as exc:
        return [f"{INTERPRETER_MESSAGE} ({type(exc).__name__}: {exc})"]
    expected = -(-_LENGTH // _CHUNK)
    if out.item() != expected:
        return [f"{INTERPRETER_MESSAGE} (counted {out.item()} chunks, expected {expected})"]
    return []
