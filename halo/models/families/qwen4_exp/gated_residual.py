"""Flash-Next's gated residual on the Triton kernels of ``halo/kernels/gated_residual.py`` and ``inject_streams``, behind
``HALO_GATED_RESIDUAL_KERNELS``. The saved upstream forward is the reference and the fallback where a kernel cannot run
(see ``is_gated_residual_fusable``), so the knob off, or any such module, runs upstream's arithmetic."""

from __future__ import annotations

import functools
import logging

import torch
import torch.nn.functional as F
import transformers.models.qwen4_exp.modeling_qwen4_exp as qwen4_exp_modeling
from torch import nn

from halo.env import env_flag
from halo.kernels.gated_residual import gate_mix, grouped_rms_norm, warm_gated_residual_kernels
from halo.kernels.hyper_connection import inject_streams
from halo.log import warn_once
from halo.models.upstream.upstream_guard import verify_upstream_parameters
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)

_SUPPORTED_DTYPES = (torch.bfloat16, torch.float16, torch.float32)
_FORWARD_ATTR = "_halo_gated_residual_kernels"
# The reason of a call off CUDA, empty or of a shape upstream rejects: expected, so never logged.
_EXPECTED_FALLBACK = "off CUDA, empty or misshapen input"
_warned_fallbacks: set[str] = set()


def uses_gated_residual_kernels() -> bool:
    """Read per call, so a test or an A/B toggles without reinstalling."""
    return env_flag("HALO_GATED_RESIDUAL_KERNELS", True)


def _is_fusable_dtype(x: torch.Tensor) -> bool:
    if x.dtype not in _SUPPORTED_DTYPES:
        return False
    device_type = x.device.type
    return not torch.is_autocast_enabled(device_type) or x.dtype == torch.get_autocast_dtype(device_type)


def _unfusable_reason(module: nn.Module, hyper_input: torch.Tensor) -> str | None:
    """Why the kernels cannot run this call, or ``None``: the module's structure first, then the call's tensor."""
    norm = module.hc_norm
    present = [
        linear
        for linear in (module.input_mix_weight_down, module.input_mix_weight_up, module.block_inject_weight)
        if linear is not None
    ]
    if norm.group_size != module.hidden_size:
        return f"hc_norm.group_size={norm.group_size} off hidden_size={module.hidden_size}"
    if any(type(linear) is not nn.Linear or linear.bias is not None for linear in present):
        return "a mix projection that is not a bias-free nn.Linear"
    if any(
        m._forward_hooks or m._forward_pre_hooks or m._backward_hooks or m._backward_pre_hooks
        for m in (norm, *present)
    ):
        return "a forward or backward hook on the hyper-connection modules"
    # A DTensor parameter is a subclass of Tensor, which this layer cannot import to test for.
    if any(type(p) is not nn.Parameter for m in (norm, *present) for p in m.parameters()):
        return "hyper-connection parameters that are not plain tensors, as sharded DTensors are"
    if not hyper_input.is_cuda or hyper_input.numel() == 0:
        return _EXPECTED_FALLBACK
    if hyper_input.shape[-1] != module.hc_count * module.hidden_size:
        return _EXPECTED_FALLBACK  # upstream's forward raises the ValueError
    if not _is_fusable_dtype(hyper_input):
        return f"stream dtype {hyper_input.dtype} (bf16, fp16 or fp32 matching the active autocast)"
    if any(linear.weight.dtype != hyper_input.dtype for linear in present):
        return f"mix weights in {present[0].weight.dtype} where the streams are {hyper_input.dtype}"
    return None


def _log_fallback(reason: str) -> None:
    if reason != _EXPECTED_FALLBACK and is_global_main_process():
        warn_once(
            logger,
            _warned_fallbacks,
            reason,
            "Qwen3.8-Flash-Next's gated residual runs upstream's forward where its modules have %s.",
            reason,
        )


def is_gated_residual_fusable(module: nn.Module, hyper_input: torch.Tensor) -> bool:
    """Whether ``module`` runs on the kernels for ``hyper_input``: the knob is on, the streams are CUDA bf16, fp16 or
    fp32 (the active autocast's dtype under autocast), and the norm and mix projections are plain, free of forward and
    backward hooks, bias-free and in the streams' dtype. Frozen and trained weights both run on the kernels. A module
    that the knob would take but that cannot run logs its reason once on rank 0."""
    if not uses_gated_residual_kernels():
        return False
    reason = _unfusable_reason(module, hyper_input)
    if reason is None:
        return True
    _log_fallback(reason)
    return False


def _build_forward(original):
    @functools.wraps(original)
    def forward(self, hyper_input):
        if not is_gated_residual_fusable(self, hyper_input):
            return original(self, hyper_input)
        streams, width = self.hc_count, hyper_input.shape[-1]
        normed = grouped_rms_norm(hyper_input.reshape(-1, width), self.hc_norm.weight, self.hc_norm.eps, streams)
        down = F.linear(normed, self.input_mix_weight_down.weight)
        mixed = gate_mix(normed, down, self.input_mix_weight_up.weight, streams)
        mixed = mixed.view(*hyper_input.shape[:-1], -1)
        if self.block_inject_weight is None:
            return mixed
        logits = F.linear(normed, self.block_inject_weight.weight) / streams
        return mixed, hyper_input, (2 * torch.sigmoid(logits)).view(*hyper_input.shape[:-1], streams)

    setattr(forward, _FORWARD_ATTR, True)
    return forward


def patch_qwen4_exp_gated_residual() -> None:
    """Idempotent. Replace ``Qwen4ExpTextGatedResidual.forward`` with the kernel forward that falls back to upstream's;
    a moved upstream signature raises."""
    residual_cls = qwen4_exp_modeling.Qwen4ExpTextGatedResidual
    if getattr(residual_cls.forward, _FORWARD_ATTR, False):
        return
    patch = "patch_qwen4_exp_gated_residual"
    verify_upstream_parameters(residual_cls.__init__, ("self", "config", "use_combine"), patch)
    verify_upstream_parameters(residual_cls.forward, ("self", "hyper_input"), patch)
    residual_cls.forward = _build_forward(residual_cls.forward)


def inject_block(hyper_input: torch.Tensor, out: torch.Tensor, gates: torch.Tensor) -> torch.Tensor:
    """``hyper_input + (out.unsqueeze(-2) * gates.unsqueeze(-1)).flatten(-2)``: the block output injected into every
    stream, on ``inject_streams`` under the knob where the three share a CUDA dtype it supports; any other call logs
    its reason once and runs the eager form."""
    if uses_gated_residual_kernels():
        if not hyper_input.is_cuda or hyper_input.numel() == 0:
            reason = _EXPECTED_FALLBACK
        elif hyper_input.dtype not in _SUPPORTED_DTYPES:
            reason = f"stream dtype {hyper_input.dtype} (bf16, fp16 or fp32)"
        elif not out.dtype == gates.dtype == hyper_input.dtype:
            reason = f"block output {out.dtype} and gates {gates.dtype} where the streams are {hyper_input.dtype}"
        else:
            streams = gates.shape[-1]
            return inject_streams(out, hyper_input.unflatten(-1, (streams, -1)), gates).flatten(-2)
        _log_fallback(reason)
    return hyper_input + (out.unsqueeze(-2) * gates.unsqueeze(-1)).flatten(-2)


def gated_residual_layouts(model: nn.Module) -> set[tuple]:
    """``(device, dtype, norm dtype, streams, hidden size, norm eps, mix rank)`` of each gated residual of ``model``."""
    return {
        (
            module.input_mix_weight_down.weight.device,
            module.input_mix_weight_down.weight.dtype,
            module.hc_norm.weight.dtype,
            module.hc_count,
            module.hidden_size,
            module.hc_norm.eps,
            module.input_mix_weight_down.weight.shape[0],
        )
        for module in model.modules()
        if isinstance(module, qwen4_exp_modeling.Qwen4ExpTextGatedResidual)
    }


def warm_gated_residual(model: nn.Module) -> None:
    """Compile the kernels each CUDA gated residual of ``model`` runs, in its weights' dtypes, so no step compiles one.
    Rank-local, no collective; a no-op with the knob off."""
    if not uses_gated_residual_kernels() or not torch.cuda.is_available():
        return
    for device, dtype, *layout in sorted(gated_residual_layouts(model), key=str):
        if device.type == "cuda" and dtype in _SUPPORTED_DTYPES:
            warm_gated_residual_kernels(device, dtype, *layout)
