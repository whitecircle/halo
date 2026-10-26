"""Repair of transformers' ``use_kernel_func_from_hub_with_fallback``: route each call by its first
tensor's device (upstream calls the CUDA-only package on CPU too), and import the mapped submodule chain
ahead of upstream, which turns any failure of that import into a silent capture of the ~10x slower torch
body: a broken kernel package raises instead. Installed by the ``src`` package init, before any modeling
module binds it.
"""

import functools
import importlib
import importlib.util
import inspect
import logging

import torch
import transformers.integrations
from transformers.integrations import hub_kernels

from src.log import warn_once

logger = logging.getLogger(__name__)

_PATCHED_FLAG = "_halo_device_aware_kernel_dispatch"
_warned_torch_captures: set[tuple[str, str]] = set()


def _segmented_conv1d(torch_function, hidden_states, *args, **kwargs):
    """Run the torch conv fallback per packed document: it swallows ``seq_idx`` and would carry left
    context across documents. The conv is causal and depthwise, so this matches the CUDA kernel exactly."""
    seq_idx = kwargs.pop("seq_idx", None)
    if seq_idx is None:
        return torch_function(hidden_states, *args, **kwargs)
    out = torch.empty_like(hidden_states)
    for row in range(hidden_states.shape[0]):
        boundaries = (seq_idx[row][1:] != seq_idx[row][:-1]).nonzero().flatten() + 1
        edges = [0, *boundaries.tolist(), hidden_states.shape[-1]]
        for start, end in zip(edges[:-1], edges[1:], strict=False):
            out[row : row + 1, :, start:end] = torch_function(
                hidden_states[row : row + 1, :, start:end], *args, **kwargs
            )
    return out


def _segmented_chunk_gated_delta_rule(torch_function, query, key, value, **kwargs):
    """Run the torch chunked delta rule per packed document: it swallows ``cu_seqlens`` and would carry
    recurrent state across documents. Each document starts from a fresh state, or from its own row of
    ``initial_state``, which then holds one state per document as the CUDA kernel takes it."""
    cu_seqlens = kwargs.pop("cu_seqlens", None)
    if cu_seqlens is None:
        return torch_function(query, key, value, **kwargs)
    g, beta, initial_state = kwargs.pop("g"), kwargs.pop("beta"), kwargs.pop("initial_state", None)
    outputs, states = [], []
    edges = cu_seqlens.tolist()
    if initial_state is not None and initial_state.shape[0] != len(edges) - 1:
        raise ValueError(
            f"initial_state holds {initial_state.shape[0]} states for {len(edges) - 1} segments: with "
            f"cu_seqlens the delta rule takes one initial state per segment."
        )
    for segment, (start, end) in enumerate(zip(edges[:-1], edges[1:], strict=False)):
        out, state = torch_function(
            query[:, start:end],
            key[:, start:end],
            value[:, start:end],
            g=g[:, start:end],
            beta=beta[:, start:end],
            initial_state=None if initial_state is None else initial_state[segment : segment + 1],
            **kwargs,
        )
        outputs.append(out)
        states.append(state)
    state = torch.cat(states, dim=0) if states and states[0] is not None else None
    return torch.cat(outputs, dim=1), state


# Torch fallbacks not faithful to their CUDA package on packed rows, by the kernel name the factory takes.
_CPU_FALLBACK_FIXUPS = {
    "causal_conv1d_fn": _segmented_conv1d,
    "chunk_gated_delta_rule": _segmented_chunk_gated_delta_rule,
}


def _import_mapped_chain(func_name: str, package: str, internal_path: str | None) -> None:
    """Import the mapped submodule chain ahead of upstream, which swallows any failure of that import: a missing
    module (package absent, chain renamed) is left to upstream's fallback and the capture check, any other error
    is a broken kernel package and propagates."""
    chain = getattr(hub_kernels, "_KERNELS_INTERNAL_PATH_MAPPINGS", {}).get(func_name, internal_path)
    if not chain:
        return
    try:
        importlib.import_module(f"{package}.{chain}")
    except ModuleNotFoundError:
        return  # absent package or renamed chain: the capture check reports the fallback


def _capture_fell_back(dispatched, torch_function) -> bool | None:
    """Whether upstream's decoration captured the torch body, best effort; None when unreadable."""
    fn = dispatched
    for _ in range(4):
        if fn is None:
            return None
        try:
            nonlocals = inspect.getclosurevars(fn).nonlocals
        except TypeError:
            return None
        if "implementation" in nonlocals:
            return nonlocals["implementation"] is torch_function
        fn = getattr(fn, "__wrapped__", None)
    return None


def _warn_if_torch_captured(dispatched, torch_function, func_name: str, package: str) -> None:
    try:
        installed = importlib.util.find_spec(package) is not None
    except (ImportError, ValueError):
        installed = False
    if installed and _capture_fell_back(dispatched, torch_function):
        warn_once(
            logger,
            _warned_torch_captures,
            (func_name, package),
            f"kernel dispatch for '{func_name}' captured the torch fallback although '{package}' is "
            f"installed — CUDA forwards will run the slow torch path; the '{package}' kernel failed to "
            f"resolve (import layout changed?)",
        )


def ensure_device_aware_kernel_dispatch() -> None:
    """Install the repaired factory in ``hub_kernels`` and any materialized re-export, once."""
    if getattr(hub_kernels, _PATCHED_FLAG, False):
        return
    original_factory = hub_kernels.use_kernel_func_from_hub_with_fallback

    def device_aware_factory(func_name, package, internal_path=None):
        _import_mapped_chain(func_name, package, internal_path)
        original_decorator = original_factory(func_name, package, internal_path)

        def decorator(torch_function):
            dispatched = original_decorator(torch_function)
            _warn_if_torch_captured(dispatched, torch_function, func_name, package)

            fixup = _CPU_FALLBACK_FIXUPS.get(func_name)

            @functools.wraps(torch_function)
            def device_aware(*args, **kwargs):
                tensor = next((a for a in args if isinstance(a, torch.Tensor)), None)
                if tensor is not None and tensor.device.type != "cuda":
                    if fixup is not None:
                        return fixup(torch_function, *args, **kwargs)
                    return torch_function(*args, **kwargs)
                return dispatched(*args, **kwargs)

            return device_aware

        return decorator

    hub_kernels.use_kernel_func_from_hub_with_fallback = device_aware_factory
    # The lazy ``transformers.integrations`` package caches resolved attributes in its __dict__.
    if "use_kernel_func_from_hub_with_fallback" in vars(transformers.integrations):
        transformers.integrations.use_kernel_func_from_hub_with_fallback = device_aware_factory
    setattr(hub_kernels, _PATCHED_FLAG, True)


ensure_device_aware_kernel_dispatch()
