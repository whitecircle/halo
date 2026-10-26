"""Native grouped LoRA on EP expert weights: the adapters, their checkpoint gather and load, and disabling them for a
frozen-base reference pass."""

from __future__ import annotations

import contextlib
import logging
import math

import torch
import torch.nn as nn

from src.checkpoint.adapters import expert_lora_checkpoint_keys
from src.distributed.expert_parallel.config import EPConfig
from src.kernels.grouped_mm import GROUPED_MM_STRIDE_ALIGNMENT_BYTES
from src.models.moe.expert_lora_spec import EXPERT_LORA_SUFFIXES
from src.runtime.collectives import all_gather_cat, broadcast_from_rank0
from src.runtime.dtensor import copy_full_tensor, materialize_dtensor
from src.runtime.ranks import get_global_rank, is_global_main_process

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

    def gather_expert_lora_state_dict(self, *, retain: bool = True) -> dict:
        """COLLECTIVE over the dispatch-EP group. This layer's adapters gathered to the full expert count, keyed
        ``experts.<attr>.lora_{A,B}`` in matmul convention; ``{}`` without adapters or ``retain``.

        Never ETP-sharded: ``EPConfig`` rejects expert LoRA under expert TP.
        """
        state: dict[str, torch.Tensor] = {}
        for attr in sorted(self._expert_lora_attrs):
            for suffix, key in zip(EXPERT_LORA_SUFFIXES, expert_lora_checkpoint_keys(attr), strict=True):
                adapter = materialize_dtensor(getattr(self, f"{attr}{suffix}").data)
                adapter = all_gather_cat(adapter, 0, self.ep_config.dispatch_ep_group, self.ep_config.ep_size)
                if retain:
                    state[key] = adapter.contiguous().cpu()
        return state

    def load_expert_lora_state_dict(self, layer_state: dict) -> None:
        """Load :meth:`gather_expert_lora_state_dict`'s output into the live adapters, slicing this rank's experts.

        COLLECTIVE where the adapters are FSDP2 DTensors (ep1 with ``fsdp_shard_ep1_experts``): each copy re-shards
        onto the param's mesh. Raises on a key-set mismatch, the trace of a changed expert configuration
        (GptOss renames its attrs by grouped GEMM, ETP and arch) that would otherwise resume from zero-init.
        """
        expected_keys = {key for attr in self._expert_lora_attrs for key in expert_lora_checkpoint_keys(attr)}
        if set(layer_state) != expected_keys:
            raise RuntimeError(
                f"Expert-LoRA resume mismatch in {type(self).__name__}: saved adapter keys "
                f"{sorted(layer_state)} do not match this layer's rebuilt adapters {sorted(expected_keys)}. "
                "Likely a use_grouped_gemm / expert_tensor_parallel_size / GPU-arch change between "
                "checkpoint and resume — resume with the same expert configuration."
            )
        # sorted(): the DTensor copy is a collective, and frozenset order varies per process.
        for attr in sorted(self._expert_lora_attrs):
            for suffix, key in zip(EXPERT_LORA_SUFFIXES, expert_lora_checkpoint_keys(attr), strict=True):
                copy_full_tensor(
                    getattr(self, f"{attr}{suffix}"), layer_state[key][self.expert_start : self.expert_end]
                )


def expert_lora_layers(model: nn.Module) -> list[tuple[str, EPExpertLoraMixin]]:
    """``(module path, layer)`` for every EP layer of ``model`` carrying native expert adapters."""
    return [(name, m) for name, m in model.named_modules() if isinstance(m, EPExpertLoraMixin) and m.has_expert_lora]


def model_has_expert_lora(model: nn.Module) -> bool:
    """Whether any EP layer of ``model`` carries native expert adapters."""
    return bool(expert_lora_layers(model))


def gather_expert_lora_adapters(model: nn.Module, *, retain: bool = True) -> dict[str, torch.Tensor]:
    """COLLECTIVE. Every EP layer's expert adapters on CPU, keyed ``<layer>.experts.<attr>.lora_{A,B}``.

    ``retain=False`` joins every gather and keeps nothing: otherwise every rank of a node builds the whole adapter
    set on the host for one writer (~40 GB/node at ``lora_r: 16`` on a 397B-class MoE with 8 ranks/node).
    """
    adapters: dict[str, torch.Tensor] = {}
    for layer_name, layer in expert_lora_layers(model):
        for key, tensor in layer.gather_expert_lora_state_dict(retain=retain).items():
            adapters[f"{layer_name}.{key}"] = tensor
    return adapters


def load_expert_lora_adapters(model: nn.Module, adapter_state: dict[str, torch.Tensor]) -> None:
    """COLLECTIVE. Load :func:`gather_expert_lora_adapters`' output back into ``model``.

    The "no adapters" decision is rank 0's, so a rank whose adapter file is missing on a non-shared filesystem
    does not skip the DTensor re-shards its peers enter. Every layer is matched before anything loads, and an
    adapter-carrying layer no saved key matches raises: its adapters would resume from zero-init.
    """
    if not broadcast_from_rank0(bool(adapter_state)):
        return
    if not adapter_state:
        raise RuntimeError(
            f"load_expert_lora_adapters: rank {get_global_rank()} received no expert-adapter state while "
            f"global rank 0 did — a torn/partial adapter checkpoint on a non-shared filesystem. This "
            f"rank's adapters would silently resume from zero-init. Resume from a complete checkpoint."
        )
    pending: list[tuple[EPExpertLoraMixin, dict[str, torch.Tensor]]] = []
    missed: list[str] = []
    for layer_name, layer in expert_lora_layers(model):
        prefix = f"{layer_name}.experts."
        layer_state = {
            key[len(f"{layer_name}.") :]: tensor for key, tensor in adapter_state.items() if key.startswith(prefix)
        }
        if layer_state:
            pending.append((layer, layer_state))
        else:
            missed.append(layer_name)
    if missed:
        raise RuntimeError(
            f"load_expert_lora_adapters: {len(missed)} EP layer(s) with expert LoRA matched none of the "
            f"{len(adapter_state)} saved expert-adapter keys: {missed[:5]}"
            f"{' …' if len(missed) > 5 else ''} (first saved key: {next(iter(adapter_state))!r}). "
            f"Their adapters would silently resume from zero-init."
        )
    if not pending:
        # No layer builds expert adapters at all: EP off, use_grouped_gemm off, or the expert projections dropped
        # from lora_target_modules since the save.
        raise RuntimeError(
            f"load_expert_lora_adapters: the checkpoint carries {len(adapter_state)} expert-adapter tensors "
            f"(first: {next(iter(adapter_state))!r}) but this model has no EP layer with expert LoRA to "
            f"receive them, so every saved expert delta would be discarded. Resume with the run's expert "
            f"configuration — EP/grouped-GEMM on and the same expert projections in lora_target_modules."
        )
    for layer, layer_state in pending:
        layer.load_expert_lora_state_dict(layer_state)


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
