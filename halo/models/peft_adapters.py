"""PEFT adapter parameters a tuner layer holds beside the module it wraps, the adapters that split a tied embedding
pair, and the LoRA merge reproduced out of place. Rank-local; a sharded fold lives in ``halo.distributed.lora_fold``."""

from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager, contextmanager, nullcontext

import torch
from peft.tuners.lora.layer import Conv1d, Conv2d, Conv3d, Embedding, Linear, LoraLayer, ParamWrapper
from peft.tuners.lora.variants import (
    DoraConv1dVariant,
    DoraConv2dVariant,
    DoraConv3dVariant,
    DoraEmbeddingVariant,
    DoraLinearVariant,
)
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import ModulesToSaveWrapper

from halo.models.introspection.module_tree import embedding_module, find_module_path

# Exact types, as a subclass (torchao, Intel Neural Compressor) overrides the merge.
_FOLDABLE_LORA_LAYERS = (Linear, Embedding, Conv1d, Conv2d, Conv3d, ParamWrapper)

_CONV_LORA_LAYERS = (Conv1d, Conv2d, Conv3d)

# ``id`` of each base tensor a LoRA merge rewrites → the layers folding into it (:func:`lora_fold_targets`).
LoraFolds = dict[int, list[LoraLayer]]


def has_adapter_layers(model: torch.nn.Module) -> bool:
    """Detect injected PEFT layers without mistaking a backbone's own ``lora_`` weights for adapters."""
    return any(isinstance(module, BaseTunerLayer) for module in model.modules())


def tuner_adapter_param_ids(model: torch.nn.Module) -> set[int]:
    """Structural, so a backbone's own ``lora_A`` stays base."""
    adapters: set[int] = set()
    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            base = {id(param) for param in module.get_base_layer().parameters()}
            adapters.update(id(param) for param in module.parameters() if id(param) not in base)
    return adapters


def _lora_weight(layer: LoraLayer) -> torch.nn.Parameter:
    return layer.get_param() if isinstance(layer, ParamWrapper) else layer.get_base_layer().weight


def _adapters_to_fold(layer: LoraLayer) -> list[str]:
    """Active ones not merged already (PEFT's ``check_adapters_to_merge``), that this layer carries."""
    merged = set(layer.merged_adapters)
    return [
        adapter
        for adapter in layer.active_adapters
        if adapter not in merged and (adapter in layer.lora_A or adapter in layer.lora_embedding_A)
    ]


def _unfoldable(layer: BaseTunerLayer) -> str | None:
    if type(layer) not in _FOLDABLE_LORA_LAYERS:
        return f"tuner layer {type(layer).__name__}"
    if isinstance(layer, _CONV_LORA_LAYERS) and layer.get_base_layer().groups > 1:
        return f"grouped {type(layer).__name__}"
    variants = [layer.lora_variant[adapter] for adapter in _adapters_to_fold(layer) if adapter in layer.lora_variant]
    foldable = (DoraLinearVariant, DoraEmbeddingVariant, DoraConv1dVariant, DoraConv2dVariant, DoraConv3dVariant)
    unknown = [variant for variant in variants if not isinstance(variant, foldable)]
    return f"{type(unknown[0]).__name__} adapter" if unknown else None


def adapter_base_module(module: torch.nn.Module) -> torch.nn.Module:
    """The module holding the weight ``module`` computes from with its adapters off: a tuner layer's base layer, a
    ``modules_to_save`` wrapper's original, else ``module`` itself."""
    if isinstance(module, BaseTunerLayer):
        return module.get_base_layer()
    if isinstance(module, ModulesToSaveWrapper):
        return module.original_module
    return module


def _computed_weight(module: torch.nn.Module) -> object:
    """What ``module`` computes with: the LoRA layer itself where it folds a delta into its base weight, the active
    ``modules_to_save`` copy's weight, else its base weight."""
    if isinstance(module, LoraLayer) and _adapters_to_fold(module):
        return module
    if isinstance(module, ModulesToSaveWrapper):
        copies = [module.modules_to_save[name] for name in module.active_adapters if name in module.modules_to_save]
        if copies:
            return copies[0].weight
    return adapter_base_module(module).weight


def adapted_tied_embeddings(model: torch.nn.Module) -> dict[str, str]:
    """Module path → role (``input embedding``, ``output head``) of each half of ``model``'s tied embedding pair that an
    adapter makes compute with a weight the other half does not: a LoRA layer adds its delta to one half's weight, a
    ``modules_to_save`` copy replaces it. Empty when the pair is untied, no adapter touches it, or both halves compute
    with one copy."""
    halves = {"input embedding": embedding_module(model, "input"), "output head": embedding_module(model, "output")}
    if any(module is None for module in halves.values()):
        return {}
    bases = {role: getattr(adapter_base_module(module), "weight", None) for role, module in halves.items()}
    if bases["input embedding"] is None or bases["input embedding"] is not bases["output head"]:
        return {}
    computed = {role: _computed_weight(module) for role, module in halves.items()}
    if computed["input embedding"] is computed["output head"]:
        return {}
    return {find_module_path(model, halves[role]): role for role in halves if computed[role] is not bases[role]}


def _reject_lora_on_tied_embeddings(model: torch.nn.Module) -> None:
    lora = {
        path: role
        for path, role in adapted_tied_embeddings(model).items()
        if isinstance(model.get_submodule(path), LoraLayer)
    }
    if lora:
        leaves = sorted({path.rpartition(".")[2] for path in lora})
        raise ValueError(
            f"LoRA adapts the {' and '.join(lora.values())} ({', '.join(lora)}) of a tied embedding pair "
            f"(tie_word_embeddings): training computes the two halves with different weights, but they share one "
            f"base tensor, and the out-of-place fold writes that tensor once, so the folded weights would carry the "
            f"delta in both halves. A weight sync or a merged save cannot write this adapter; merge_peft_adapters.py "
            f"(in memory) writes it as an untied model. Remove {', '.join(leaves)} from lora_target_modules."
        )


def lora_fold_targets(model: torch.nn.Module) -> LoraFolds:
    """In merge order, keyed by identity. Raises for a layer the fold cannot reproduce, and for a LoRA layer on either
    half of a tied embedding pair, whose delta the shared tensor would give the other half too."""
    _reject_lora_on_tied_embeddings(model)
    targets: LoraFolds = {}
    for name, module in model.named_modules():
        if not isinstance(module, BaseTunerLayer):
            continue
        unfoldable = _unfoldable(module)
        if unfoldable:
            raise ValueError(
                f"{name}: {unfoldable} cannot be folded out of place, which is implemented for plain and DoRA "
                f"adapters on {[cls.__name__ for cls in _FOLDABLE_LORA_LAYERS]} only, so its delta would not reach "
                f"the merged weights. Use a plain LoraConfig (use_dora allowed) on unquantized linear, "
                f"embedding or ungrouped conv modules in lora_target_modules."
            )
        adapters = _adapters_to_fold(module)
        if not adapters:
            continue
        targets.setdefault(id(_lora_weight(module)), []).append(module)
        bias = getattr(module.get_base_layer(), "bias", None)
        if bias is not None and any(module.lora_bias.get(adapter) for adapter in adapters):
            targets.setdefault(id(bias), []).append(module)
    return targets


@contextmanager
def adapter_tensors_replaced(
    layer: LoraLayer, adapter: str, replace: Callable[[torch.Tensor], torch.Tensor]
) -> Iterator[None]:
    """``adapter``'s B bias, A and B weights and DoRA magnitude seen through ``replace`` inside the block, as PEFT's
    delta and merge read them."""
    holders = [(layer.lora_A[adapter], "weight"), (layer.lora_B[adapter], "weight"), (layer.lora_B[adapter], "bias")]
    if adapter in layer.lora_variant:
        holders.append((layer.lora_magnitude_vector[adapter], "weight"))
    originals = [(module, name, getattr(module, name)) for module, name in holders]
    for module, name, original in originals:
        if original is not None:
            setattr(module, name, torch.nn.Parameter(replace(original.detach()), requires_grad=False))
    try:
        yield
    finally:
        for module, name, original in originals:
            setattr(module, name, original)


@torch.no_grad()
def lora_folded(
    param: torch.nn.Parameter,
    layers: list[LoraLayer],
    *,
    base: torch.Tensor | None = None,
    adapter_scope: Callable[[LoraLayer, str], AbstractContextManager[None]] | None = None,
) -> torch.Tensor:
    """A new tensor holding what PEFT's merge would write into ``param``, starting from ``base`` when given. Every
    adapter's delta is computed inside ``adapter_scope(layer, adapter)``."""
    folded = (param if base is None else base).detach().clone()
    for layer in layers:
        folds_weight = param is _lora_weight(layer)
        for adapter in _adapters_to_fold(layer):
            with adapter_scope(layer, adapter) if adapter_scope else nullcontext():
                if not folds_weight:
                    if layer.lora_bias.get(adapter):
                        folded += layer.lora_B[adapter].bias * layer.scaling[adapter]
                elif adapter in layer.lora_variant:
                    # A variant caches what its unmerge needs; there is no unmerge here.
                    cached = set(layer._caches)
                    folded = layer.lora_variant[adapter].merge_safe(layer, adapter, folded)
                    for key in set(layer._caches) - cached:
                        layer._cache_pop(key)
                else:
                    delta = layer.get_delta_weight(adapter)
                    # Linear adds the delta in its own dtype; the other layers cast it first, as their merges do.
                    folded += delta if isinstance(layer, Linear) else delta.to(folded.dtype)
    return folded


def is_conv_lora_layer(layer: LoraLayer) -> bool:
    return isinstance(layer, _CONV_LORA_LAYERS)
