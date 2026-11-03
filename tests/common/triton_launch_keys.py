"""The Triton specialization of each launch under ``TRITON_INTERPRET=1``, so a CPU test compares the binaries two
launches would run on a GPU."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import ExitStack, contextmanager
from unittest import mock

from triton.backends.compiler import GPUTarget
from triton.compiler import make_backend
from triton.runtime.interpreter import InterpretedFunction
from triton.runtime.jit import JITFunction, create_function_from_signature

# Specialization is the backend's rule, the same on every CUDA arch.
_KEY_TARGET = GPUTarget("cuda", 100, 32)


@contextmanager
def recorded_launch_keys(kernels: Sequence[InterpretedFunction]) -> Iterator[list[tuple[str, list[str], tuple]]]:
    """Collects ``(kernel name, parameter names, specialization key)`` per launch while active. Raises unless under
    ``TRITON_INTERPRET=1``."""
    if not all(isinstance(kernel, InterpretedFunction) for kernel in kernels):
        raise RuntimeError("launch keys are read from kernels run under TRITON_INTERPRET=1")
    backend = make_backend(_KEY_TARGET)
    launches: list[tuple[str, list[str], tuple]] = []
    with ExitStack() as stack:
        for kernel in kernels:
            stack.enter_context(mock.patch.object(kernel, "run", _recording_run(kernel, backend, launches)))
        yield launches


def moved_arguments(names: list[str], key: tuple, reference: tuple) -> list[str]:
    return [arg for arg, a, b in zip(names, key, reference, strict=True) if a != b]


def _recording_run(kernel: InterpretedFunction, backend, launches: list):
    compiled = JITFunction(kernel.fn, **kernel.kwargs)
    bind = create_function_from_signature(compiled.signature, compiled.params, backend)
    names = [param.name for param in compiled.params]
    run = kernel.run

    def record(*args, grid, warmup, **kwargs):
        launches.append((kernel.__name__, names, bind(*args, **kwargs)[1]))
        return run(*args, grid=grid, warmup=warmup, **kwargs)

    return record
