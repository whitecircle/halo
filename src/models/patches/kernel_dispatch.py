"""Device-aware repair for transformers' kernel-fallback dispatch.

``use_kernel_func_from_hub_with_fallback`` captures the CUDA-only package implementation at
decoration time and calls it unconditionally, so any CPU forward of a conv/linear-attention family
dies with ``Expected x.is_cuda()`` instead of taking the decorated torch path. The wrapped factory
routes each call by the device of its first tensor argument. Installed at import by the ``src``
package init, ahead of every toolkit import of a modeling module (they bind the factory at import).
"""

import functools

import torch
import transformers.integrations
from transformers.integrations import hub_kernels

_PATCHED_FLAG = "_halo_device_aware_kernel_dispatch"


def _segmented_conv1d(torch_function, hidden_states, *args, **kwargs):
    """Run the torch conv fallback per packed document.

    The torch-path ``causal_conv1d_fn`` swallows ``seq_idx`` in ``**kwargs`` and convolves the whole
    row, carrying left context across document boundaries; only the CUDA kernel isolates. The conv is
    causal and depthwise, so per-document application matches the kernel exactly.
    """
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
    """Run the torch chunked delta rule per packed document.

    The torch-path ``chunk_gated_delta_rule`` swallows ``cu_seqlens`` and runs the flattened row as
    one recurrence, carrying recurrent state across document boundaries; only the fla kernel
    isolates. Running each document from a fresh initial state restores the semantics exactly; a
    cached call (``initial_state``) falls through unsegmented rather than inferring per-sequence state.
    """
    cu_seqlens = kwargs.pop("cu_seqlens", None)
    if cu_seqlens is None or kwargs.get("initial_state") is not None:
        return torch_function(query, key, value, **kwargs)
    g, beta = kwargs.pop("g"), kwargs.pop("beta")
    outputs, states = [], []
    edges = cu_seqlens.tolist()
    for start, end in zip(edges[:-1], edges[1:], strict=False):
        out, state = torch_function(
            query[:, start:end],
            key[:, start:end],
            value[:, start:end],
            g=g[:, start:end],
            beta=beta[:, start:end],
            **kwargs,
        )
        outputs.append(out)
        states.append(state)
    state = torch.cat(states, dim=0) if states and states[0] is not None else None
    return torch.cat(outputs, dim=1), state


# CPU-route fixups: the original packages are CUDA-only, and 5.16's torch fallbacks are not always
# faithful to them. Keyed by the decorated function name.
_CPU_FALLBACK_FIXUPS = {
    "causal_conv1d_fn": _segmented_conv1d,
    "chunk_gated_delta_rule": _segmented_chunk_gated_delta_rule,
}


def ensure_device_aware_kernel_dispatch() -> None:
    """Idempotent; patches the factory in ``hub_kernels`` and any materialized re-export."""
    if getattr(hub_kernels, _PATCHED_FLAG, False):
        return
    original_factory = hub_kernels.use_kernel_func_from_hub_with_fallback

    def device_aware_factory(func_name, package, internal_path=None):
        original_decorator = original_factory(func_name, package, internal_path)

        def decorator(torch_function):
            dispatched = original_decorator(torch_function)

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
