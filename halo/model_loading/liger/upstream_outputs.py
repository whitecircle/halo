"""Output classes upstream Liger's fused forwards import, and the keywords they construct them with. transformers folded
some into the generic outputs and (5.24) rebased others on multimodal outputs without ``rope_deltas`` or
``image_hidden_states``, so the fused forwards fail on import or on construction until those names and fields are put
back."""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass, field, fields, make_dataclass

import torch
from liger_kernel.transformers.model import output_classes
from liger_kernel.transformers.model.output_classes import LigerCausalLMOutputWithPast, LigerMoeCausalLMOutputWithPast
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    MoeModelOutputWithPast,
    MultimodalBaseModelOutputWithPast,
    MultimodalMoeBaseModelOutputWithPast,
)


@dataclass
class _LigerRopeDeltasCausalLMOutput(LigerCausalLMOutputWithPast):
    rope_deltas: torch.LongTensor | None = None


@dataclass
class _LigerRopeDeltasMoeCausalLMOutput(LigerMoeCausalLMOutputWithPast):
    rope_deltas: torch.LongTensor | None = None


# Each name, by the generic output its family's head now returns.
_FOLDED_OUTPUT_CLASSES = {
    "LigerQwen2VLCausalLMOutputWithPast": _LigerRopeDeltasCausalLMOutput,
    "LigerQwen2_5_VLCausalLMOutputWithPast": _LigerRopeDeltasCausalLMOutput,
    "LigerQwen3VLCausalLMOutputWithPast": _LigerRopeDeltasCausalLMOutput,
    "LigerQwen3_5CausalLMOutputWithPast": _LigerRopeDeltasCausalLMOutput,
    "LigerQwen3VLMoeCausalLMOutputWithPast": _LigerRopeDeltasMoeCausalLMOutput,
    "LigerQwen3_5MoeCausalLMOutputWithPast": _LigerRopeDeltasMoeCausalLMOutput,
    "LigerGlm4vMoeCausalLMOutputWithPast": _LigerRopeDeltasMoeCausalLMOutput,
}
_IMAGE = ("image_hidden_states",)
# The keywords each fused forward passes to its output class, beyond what transformers 5.24's class carries. A
# ``None``-default field takes each; the multimodal ones' values are the forward's own and nothing in the toolkit reads
# them off the output.
_LEGACY_FIELDS: dict[str, tuple[str, ...]] = {
    **dict.fromkeys(_FOLDED_OUTPUT_CLASSES, ("rope_deltas",)),
    "LigerGemma3CausalLMOutputWithPast": _IMAGE,
    "LigerGemma4CausalLMOutputWithPast": (*_IMAGE, "audio_hidden_states"),
    "LigerInternVLCausalLMOutputWithPast": _IMAGE,
    "LigerLlavaCausalLMOutputWithPast": _IMAGE,
    "LigerMuseGlimmerCausalLMOutputWithPast": _IMAGE,
    "LigerPaliGemmaCausalLMOutputWithPast": _IMAGE,
}
# A fused forward that constructs transformers' own class, bound in its module rather than in ``output_classes``.
_MODULE_BOUND_LEGACY_FIELDS = {
    "liger_kernel.transformers.model.smolvlm": {"SmolVLMCausalLMOutputWithPast": _IMAGE},
}
_FUSED_FORWARD_PACKAGE = "liger_kernel.transformers.model."
# What those families' decoders return: from transformers 5.24 a composite's is a multimodal output (Qwen3.5 MoE's
# ``Qwen3_5MoeBaseModelOutputWithPast`` derives the MoE one), which carries no ``rope_deltas`` either.
_FOLDED_MODEL_OUTPUTS = (
    BaseModelOutputWithPast,
    MoeModelOutputWithPast,
    MultimodalBaseModelOutputWithPast,
    MultimodalMoeBaseModelOutputWithPast,
)


def _missing(defined: type, names: tuple[str, ...]) -> tuple[str, ...]:
    present = {item.name for item in fields(defined)}
    return tuple(name for name in names if name not in present)


def _with_fields(defined: type, names: tuple[str, ...]) -> type:
    """``defined`` plus a ``None``-default field per name, under its own name and module."""
    return make_dataclass(
        defined.__name__,
        [(name, torch.Tensor | None, field(default=None)) for name in names],
        bases=(defined,),
        namespace={"__module__": defined.__module__, "__qualname__": defined.__qualname__},
    )


def _rebind_in_fused_forwards(name: str, defined: type, replacement: type) -> None:
    for module_name, module in list(sys.modules.items()):
        if module_name.startswith(_FUSED_FORWARD_PACKAGE) and getattr(module, name, None) is defined:
            setattr(module, name, replacement)


def ensure_folded_output_classes() -> None:
    """Bind each missing name to its generic output, give each defined class the fields its forward passes and it
    lacks, and bind a ``None`` ``rope_deltas`` to each model output without one. Idempotent; run before an upstream
    applier, which imports its fused forward when called. A fused-forward module already imported is rebound too."""
    for name, generic in _FOLDED_OUTPUT_CLASSES.items():
        if not hasattr(output_classes, name):
            setattr(output_classes, name, generic)
    for name, names in _LEGACY_FIELDS.items():
        defined = getattr(output_classes, name, None)
        if defined is None or not (missing := _missing(defined, names)):
            continue
        replacement = _with_fields(defined, missing)
        setattr(output_classes, name, replacement)
        _rebind_in_fused_forwards(name, defined, replacement)
    for module_name, classes in _MODULE_BOUND_LEGACY_FIELDS.items():
        module = importlib.import_module(module_name)
        for name, names in classes.items():
            defined = getattr(module, name)
            if missing := _missing(defined, names):
                setattr(module, name, _with_fields(defined, missing))
    for model_output in _FOLDED_MODEL_OUTPUTS:
        if not hasattr(model_output, "rope_deltas"):
            model_output.rope_deltas = None
