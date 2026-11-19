"""Logits of hidden-state rows through a model's output embedding, computed outside the embedding's forward: its weight
read once per step, a LoRA adapter's delta added low-rank from ``lora_A`` and ``lora_B``, the family's head transform
applied. A caller projects a few rows at a time, so the model's ``[tokens, V]`` logits never exist."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from peft.tuners.lora.layer import Linear as LoraLinear
from peft.tuners.tuners_utils import BaseTunerLayer
from torch import nn

from halo.models.head_transform import HeadTransform
from halo.models.introspection.unwrapping import base_transformers_model
from halo.runtime.dtensor import materialize_dtensor


@dataclass(frozen=True)
class _HeadAdapter:
    lora_a: torch.Tensor
    lora_b: torch.Tensor
    lora_b_bias: torch.Tensor | None
    scaling: float
    dropout: nn.Module


@dataclass(frozen=True)
class HeadProjector:
    """What the output embedding computes, held as tensors: the base ``weight`` and ``bias`` cut to the compared
    vocabulary, each active LoRA adapter, the head transform. Built inside the forward that gathers the head, it keeps
    those tensors for every later call, the backward's recompute included."""

    weight: torch.Tensor
    bias: torch.Tensor | None
    adapters: tuple[_HeadAdapter, ...]
    transform: HeadTransform
    casts_adapter_input: bool

    @property
    def width(self) -> int:
        return self.weight.size(0)

    def head_inputs(self, hidden: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """``(scaled, *dropped)``: the head's input and each adapter's dropout of it, over the whole ``hidden``, so the
        dropout draws the mask PEFT's own forward draws on the same input."""
        scaled = self.transform.scale_hidden(hidden)
        x, dropped = scaled, []
        for adapter in self.adapters:
            if self.casts_adapter_input:
                x = x.to(adapter.lora_a.dtype)
            dropped.append(adapter.dropout(x))
        return scaled, *dropped

    def project(self, scaled: torch.Tensor, *dropped: torch.Tensor) -> torch.Tensor:
        """Logits ``[rows, width]`` of rows taken from :meth:`head_inputs`, in PEFT's order of operations."""
        logits = F.linear(scaled, self.weight, self.bias)
        result_dtype = logits.dtype
        for adapter, x in zip(self.adapters, dropped, strict=True):
            delta = F.linear(F.linear(x, adapter.lora_a), adapter.lora_b, adapter.lora_b_bias)
            logits = logits + delta * adapter.scaling
        return self.transform.scale_logits(logits.to(result_dtype))


def _active_adapters(tuner: LoraLinear) -> list[str]:
    """The adapters PEFT's forward adds: none while they are disabled or merged into the base weight."""
    if tuner.disable_adapters or tuner.merged:
        return []
    return [adapter for adapter in tuner.active_adapters if adapter in tuner.lora_A]


@dataclass(frozen=True)
class OutputEmbedding:
    """A model's output embedding as :class:`HeadProjector` reads it: the base linear, the plain LoRA layer wrapping it
    (``None`` without one) and the family's head transform. Holds modules, never tensors, so it outlives a step."""

    linear: nn.Module
    tuner: LoraLinear | None
    transform: HeadTransform

    def projector(self, width: int | None = None) -> HeadProjector:
        """COLLECTIVE where the head's tensors are DTensors (one gather each). ``width`` cuts the vocabulary below the
        head transform's own cut; ``None`` keeps it."""
        kept = [size for size in (self.transform.vocab_size, width) if size is not None]
        rows = slice(min(kept)) if kept else slice(None)

        def gathered(tensor: torch.Tensor | None) -> torch.Tensor | None:
            return None if tensor is None else materialize_dtensor(tensor)[rows]

        tuner = self.tuner
        if tuner is not None and tuner.disable_adapters and tuner.merged:
            raise RuntimeError(
                "The LoRA adapter on the output embedding is merged into its base weight while adapters are disabled. "
                "PEFT's own forward unmerges it first, and the chunked head projection reads the base weight as it is, "
                "so it would score the adapted head. Unmerge the adapter (unmerge_adapter()) before disabling it."
            )
        adapters = ()
        if tuner is not None:
            adapters = tuple(
                _HeadAdapter(
                    lora_a=materialize_dtensor(tuner.lora_A[adapter].weight),
                    lora_b=gathered(tuner.lora_B[adapter].weight),
                    lora_b_bias=gathered(tuner.lora_B[adapter].bias),
                    scaling=tuner.scaling[adapter],
                    dropout=tuner.lora_dropout[adapter],
                )
                for adapter in _active_adapters(tuner)
            )
        return HeadProjector(
            weight=gathered(self.linear.weight),
            bias=gathered(getattr(self.linear, "bias", None)),
            adapters=adapters,
            transform=self.transform,
            casts_adapter_input=getattr(tuner, "cast_input_dtype_enabled", True),
        )


def _head_tuner(model: nn.Module, head: nn.Module) -> BaseTunerLayer | None:
    """The PEFT layer wrapping the output embedding, whichever of the two ``get_output_embeddings()`` resolves to."""
    if isinstance(head, BaseTunerLayer):
        return head
    return next(
        (
            module
            for module in model.modules()
            if isinstance(module, BaseTunerLayer) and module is not head and module.get_base_layer() is head
        ),
        None,
    )


def resolve_output_embedding(model: nn.Module, transform: HeadTransform) -> OutputEmbedding:
    """Raises on an output embedding whose PEFT layer :class:`HeadProjector` does not reproduce. Reads module types
    only, so every rank reaches the same verdict."""
    base = base_transformers_model(model)
    head = base.get_output_embeddings()
    if head is None:
        raise ValueError(f"{type(base).__name__} reports no output embedding (get_output_embeddings() is None).")
    tuner = _head_tuner(base, head)
    if tuner is None:
        return OutputEmbedding(head, None, transform)
    if type(tuner) is not LoraLinear:
        raise ValueError(
            f"The output embedding is wrapped by a PEFT {type(tuner).__name__}, and the chunked head projection "
            f"adds only a plain LoRA Linear's delta. Use a plain LoraConfig on lm_head, or remove lm_head from "
            f"lora_target_modules."
        )
    variants = [adapter for adapter in _active_adapters(tuner) if adapter in tuner.lora_variant]
    if variants:
        raise ValueError(
            f"The LoRA adapter(s) {variants} on the output embedding carry a variant "
            f"({type(tuner.lora_variant[variants[0]]).__name__}, e.g. DoRA), and the chunked head projection adds "
            f"only the plain low-rank delta. Use a plain LoRA on lm_head (use_dora: false), or remove lm_head from "
            f"lora_target_modules."
        )
    return OutputEmbedding(tuner.get_base_layer(), tuner, transform)
