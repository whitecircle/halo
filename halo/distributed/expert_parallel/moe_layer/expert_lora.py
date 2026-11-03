"""Native grouped LoRA on EP expert weights: adapters, their checkpoint gather and load, and disabling them."""

from __future__ import annotations

import contextlib
import logging
import math

import torch
import torch.nn as nn

from halo.checkpoint.adapters import adapter_layout_mismatch, expert_lora_checkpoint_keys
from halo.distributed.expert_parallel.config import EPConfig
from halo.kernels.grouped_mm import GROUPED_MM_STRIDE_ALIGNMENT_BYTES
from halo.models.moe.expert_lora_spec import EXPERT_LORA_SUFFIXES
from halo.runtime.collectives import all_gather_cat
from halo.runtime.consensus import reject_across_ranks
from halo.runtime.dtensor import copy_full_tensor, materialize_dtensor
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)

# The adapter GEMMs run at the activation dtype (``_expert_proj`` casts lora_A/B to ``out_dtype``), bf16 at
# the narrowest whatever the experts are stored in, so the rank has to span the stride at bf16.
EXPERT_LORA_RANK_MULTIPLE = GROUPED_MM_STRIDE_ALIGNMENT_BYTES // torch.bfloat16.itemsize


class EPExpertLoraMixin:
    """Native grouped LoRA: ``W [E,K,N]`` gains ``_lora_A [E,K,r]`` and ``_lora_B [E,r,N]`` and its base freezes."""

    ep_config: EPConfig
    expert_start: int
    expert_end: int
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
            param.requires_grad_(False)
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
        out = []
        for attr in self._expert_lora_attrs:
            out.append((f"{attr}_lora_A", getattr(self, f"{attr}_lora_A")))
            out.append((f"{attr}_lora_B", getattr(self, f"{attr}_lora_B")))
        return out

    def gather_expert_lora_state_dict(self, *, retain: bool = True) -> dict:
        """COLLECTIVE over the dispatch-EP group: this layer's adapters at the full expert count, keyed
        ``experts.<attr>.lora_{A,B}``; ``{}`` without adapters or ``retain``. Never ETP-sharded, since
        ``ParallelismConfig`` rejects expert LoRA under expert TP."""
        state: dict[str, torch.Tensor] = {}
        for attr in sorted(self._expert_lora_attrs):
            for suffix, key in zip(EXPERT_LORA_SUFFIXES, expert_lora_checkpoint_keys(attr), strict=True):
                adapter = materialize_dtensor(getattr(self, f"{attr}{suffix}").data)
                adapter = all_gather_cat(adapter, 0, self.ep_config.dispatch_ep_group, self.ep_config.ep_size)
                if retain:
                    state[key] = adapter.contiguous().cpu()
        return state

    def expert_lora_checkpoint_shapes(self) -> dict[str, tuple[int, ...]]:
        """The key and shape :meth:`gather_expert_lora_state_dict` saves each adapter under: the live (global, for an
        FSDP2 DTensor) shape, its expert dim concatenated across the dispatch-EP group."""
        shapes = {}
        for attr in sorted(self._expert_lora_attrs):
            for suffix, key in zip(EXPERT_LORA_SUFFIXES, expert_lora_checkpoint_keys(attr), strict=True):
                experts, *rest = getattr(self, f"{attr}{suffix}").shape
                shapes[key] = (experts * self.ep_config.ep_size, *rest)
        return shapes

    def load_expert_lora_state_dict(self, layer_state: dict) -> None:
        """Load :meth:`gather_expert_lora_state_dict`'s output into the live adapters, slicing this rank's experts.
        COLLECTIVE where the adapters are FSDP2 DTensors (ep1 with ``fsdp_shard_ep1_experts``)."""
        # sorted(): the DTensor copy is a collective, and frozenset order varies per process.
        for attr in sorted(self._expert_lora_attrs):
            for suffix, key in zip(EXPERT_LORA_SUFFIXES, expert_lora_checkpoint_keys(attr), strict=True):
                copy_full_tensor(
                    getattr(self, f"{attr}{suffix}"), layer_state[key][self.expert_start : self.expert_end]
                )


def expert_lora_layers(model: nn.Module) -> list[tuple[str, EPExpertLoraMixin]]:
    return [(name, m) for name, m in model.named_modules() if isinstance(m, EPExpertLoraMixin) and m.has_expert_lora]


def model_has_expert_lora(model: nn.Module) -> bool:
    return bool(expert_lora_layers(model))


def gather_expert_lora_adapters(model: nn.Module, *, retain: bool = True) -> dict[str, torch.Tensor]:
    """COLLECTIVE. Every EP layer's expert adapters on CPU. ``retain=False`` joins every gather and keeps nothing;
    otherwise every rank of a node builds the whole set on the host for one writer (~40 GB/node at ``lora_r: 16`` on a
    397B-class MoE with 8 ranks/node)."""
    adapters: dict[str, torch.Tensor] = {}
    for layer_name, layer in expert_lora_layers(model):
        for key, tensor in layer.gather_expert_lora_state_dict(retain=retain).items():
            adapters[f"{layer_name}.{key}"] = tensor
    return adapters


def expert_lora_checkpoint_shapes(model: nn.Module) -> dict[str, tuple[int, ...]]:
    return {
        f"{layer_name}.{key}": shape
        for layer_name, layer in expert_lora_layers(model)
        for key, shape in layer.expert_lora_checkpoint_shapes().items()
    }


def load_expert_lora_adapters(model: nn.Module, adapter_state: dict[str, torch.Tensor]) -> None:
    """COLLECTIVE. Load :func:`gather_expert_lora_adapters`' output. Every rank first matches ``adapter_state`` to the
    live adapters and the verdicts are joined, so a mismatched tensor raises on every rank before any copy: a short
    read would resume adapters from zero-init, and a saved delta no layer takes would be discarded."""
    mismatch = adapter_layout_mismatch(adapter_state, expert_lora_checkpoint_shapes(model))
    reject_across_ranks(
        mismatch
        and (
            f"The expert-LoRA adapters saved do not match the ones this model builds: {mismatch}. Resume with the "
            f"run's expert configuration (EP / grouped GEMM on, the same expert projections in lora_target_modules, "
            f"the same lora_r, use_grouped_gemm, expert_tensor_parallel_size and GPU architecture) and a complete "
            f"checkpoint."
        ),
        "Expert-LoRA adapter resume",
    )
    for layer_name, layer in expert_lora_layers(model):
        prefix = f"{layer_name}."
        layer.load_expert_lora_state_dict(
            {key.removeprefix(prefix): tensor for key, tensor in adapter_state.items() if key.startswith(prefix)}
        )


@contextlib.contextmanager
def disable_expert_adapters(model: nn.Module):
    """Disable the native expert-LoRA deltas of every EP layer in ``model`` for the block, the EP half of
    ``PeftModel.disable_adapter()``: peft does not manage these adapters."""
    layers = [layer for _, layer in expert_lora_layers(model)]
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
