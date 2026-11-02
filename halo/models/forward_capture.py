"""Scoped capture of a module's forward outputs: a hook for the length of one block, removed on exit."""

from collections.abc import Callable, Iterator
from contextlib import contextmanager
from typing import Any

from torch import nn


@contextmanager
def capturing_forward_outputs(module: nn.Module, project: Callable[[Any], Any]) -> Iterator[list]:
    """Yield the list ``project(output)`` of every forward of ``module`` inside the block is appended to, in call
    order; the hook is removed when the block exits, raised or not."""
    captured: list = []
    handle = module.register_forward_hook(lambda _module, _inputs, output: captured.append(project(output)))
    try:
        yield captured
    finally:
        handle.remove()
