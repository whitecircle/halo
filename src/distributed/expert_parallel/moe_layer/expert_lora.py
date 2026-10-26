"""Native grouped LoRA on EP expert weights, and disabling it for a frozen-base reference pass."""

from __future__ import annotations

import contextlib
import logging
import math

import torch
import torch.nn as nn

from src.distributed.expert_parallel.config import EPConfig
from src.kernels.grouped_mm import GROUPED_MM_STRIDE_ALIGNMENT_BYTES
from src.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)

# The adapter GEMMs run at the activation dtype (``_expert_proj`` casts lora_A/B to ``out_dtype``), bf16 at
# the narrowest whatever the experts are stored in, so the rank has to span the stride at bf16.
EXPERT_LORA_RANK_MULTIPLE = GROUPED_MM_STRIDE_ALIGNMENT_BYTES // torch.bfloat16.itemsize


class EPExpertLoraMixin:
    """Native grouped LoRA, mixed into :class:`EPMoELayerBase`, whose construction builds the state annotated
    here: an adapted ``W [E,K,N]`` gains ``_lora_A [E,K,r]`` and ``_lora_B [E,r,N]`` and its base freezes.

    The model-level helpers below select layers by this class, which only the layer base mixes in.
    """

    ep_config: EPConfig
    _expert_lora_attrs: frozenset[str]

    def _init_expert_lora(self) -> None:
        """Register ``{attr}_lora_A``/``_lora_B`` (kaiming A, zero B: initial delta 0) on each 3-D expert
        weight the spec adapts, read off the live layout, and freeze those bases."""
        spec = self.ep_config.expert_lora
        if spec is None:
            return

        self.expert_lora_scaling = spec.scaling
        self.expert_lora_dropout = nn.Dropout(spec.dropout) if spec.dropout > 0 else nn.Identity()

        adapted: list[str] = []
        for name, param in self.expert_named_params():
            if param.dim() != 3 or not spec.adapts(name):
                continue
            if spec.r % EXPERT_LORA_RANK_MULTIPLE:
                raise ValueError(
                    f"lora_r={spec.r} cannot adapt the expert weight {name!r}: the grouped GEMM contracts over the "
                    f"adapter's rank dimension and reads it as a stride spanning a multiple of "
                    f"{GROUPED_MM_STRIDE_ALIGNMENT_BYTES} bytes at the bf16 the adapters run in at the narrowest — "
                    f"use a multiple of {EXPERT_LORA_RANK_MULTIPLE}."
                )
            num_local, k, n = param.shape
            lora_a = nn.Parameter(torch.empty(num_local, k, spec.r, dtype=param.dtype, device=param.device))
            lora_b = nn.Parameter(torch.zeros(num_local, spec.r, n, dtype=param.dtype, device=param.device))
            # 1/sqrt(K) reproduces PEFT's 2-D kaiming_uniform_(a=sqrt(5)); handing torch the 3-D [E, K, r]
            # tensor would fold r into fan_in, warming expert adapters at a different scale than attention.
            bound = 1.0 / math.sqrt(k)
            nn.init.uniform_(lora_a, -bound, bound)
            setattr(self, f"{name}_lora_A", lora_a)
            setattr(self, f"{name}_lora_B", lora_b)
            param.requires_grad_(False)  # freeze base; only A/B train
            adapted.append(name)

        self._expert_lora_attrs = frozenset(adapted)
        if adapted and is_global_main_process():
            logger.info(
                f"{type(self).__name__}: native grouped-LoRA on experts "
                f"(r={spec.r}, alpha={spec.alpha}, dropout={spec.dropout}, rslora={spec.use_rslora}, "
                f"scaling={spec.scaling:.4g}, projections={sorted(adapted)})"
            )

    @property
    def has_expert_lora(self) -> bool:
        """Whether adapters were built: the config can request projections the live layout lacks."""
        return bool(self._expert_lora_attrs)

    def expert_lora_named_params(self) -> list:
        """``(name, param)`` for this layer's adapters, ``lora_A`` then ``lora_B`` per adapted weight."""
        out = []
        for attr in self._expert_lora_attrs:
            out.append((f"{attr}_lora_A", getattr(self, f"{attr}_lora_A")))
            out.append((f"{attr}_lora_B", getattr(self, f"{attr}_lora_B")))
        return out


@contextlib.contextmanager
def disable_expert_adapters(model: nn.Module):
    """Disable the native expert-LoRA deltas of every EP layer in ``model`` for the block, the EP half of
    ``PeftModel.disable_adapter()``: peft does not manage these adapters."""
    layers = [m for m in model.modules() if isinstance(m, EPExpertLoraMixin) and m._expert_lora_attrs]
    previous = [m._expert_adapters_enabled for m in layers]
    for layer in layers:
        layer._expert_adapters_enabled = False
    try:
        yield
    finally:
        for layer, prev in zip(layers, previous, strict=True):
            layer._expert_adapters_enabled = prev


def install_ep_aware_disable_adapter(peft_model) -> None:
    """Make ``peft_model.disable_adapter`` also disable the native expert adapters, so a KL reference pass
    scores the true frozen base. Idempotent."""
    if getattr(peft_model, "_ep_disable_adapter_patched", False):
        return
    original_disable_adapter = peft_model.disable_adapter

    @contextlib.contextmanager
    def _ep_aware_disable_adapter():
        with original_disable_adapter(), disable_expert_adapters(peft_model):
            yield

    peft_model.disable_adapter = _ep_aware_disable_adapter
    peft_model._ep_disable_adapter_patched = True
