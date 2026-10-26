"""PEFT adapter parameters: the ones a tuner layer holds beside the module it wraps, and the LoRA fold
reproduced out of place. Rank-local, except :func:`lora_folded`, whose LoRA delta is a DTensor
collective every rank must enter.
"""

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

# The PEFT layers whose merge rewrites one base tensor by ``+= get_delta_weight`` (or DoRA's
# ``merge_safe``), plus a ``lora_bias`` fold, which :func:`lora_folded` reproduces out of place. Exact
# types: a subclass (torchao, Intel Neural Compressor) overrides the merge.
_FOLDABLE_LORA_LAYERS = (Linear, Embedding, Conv1d, Conv2d, Conv3d, ParamWrapper)
_CONV_LORA_LAYERS = (Conv1d, Conv2d, Conv3d)
_FOLDABLE_VARIANTS = (DoraLinearVariant, DoraEmbeddingVariant, DoraConv1dVariant, DoraConv2dVariant, DoraConv3dVariant)

# ``id`` of each base tensor a LoRA merge rewrites → the layers folding into it (:func:`lora_fold_targets`).
LoraFolds = dict[int, list[LoraLayer]]


def tuner_adapter_param_ids(model: torch.nn.Module) -> set[int]:
    """``id`` of every parameter a PEFT tuner layer in ``model`` holds beside the module it wraps: LoRA
    factors, embedding adapters, DoRA magnitudes.

    Structural, so a model's own parameters count as base weights whatever they are named (a
    remote-code backbone's native ``lora_A``), and rank-uniform under FSDP2.
    """
    adapters: set[int] = set()
    for module in model.modules():
        if isinstance(module, BaseTunerLayer):
            base = {id(param) for param in module.get_base_layer().parameters()}
            adapters.update(id(param) for param in module.parameters() if id(param) not in base)
    return adapters


def _lora_weight(layer: LoraLayer) -> torch.nn.Parameter:
    """The base tensor ``layer``'s merge adds its delta to."""
    return layer.get_param() if isinstance(layer, ParamWrapper) else layer.get_base_layer().weight


def _adapters_to_fold(layer: LoraLayer) -> list[str]:
    """The adapters a merge of ``layer`` folds: active ones not merged already (PEFT's
    ``check_adapters_to_merge``), that this layer carries."""
    merged = set(layer.merged_adapters)
    return [
        adapter
        for adapter in layer.active_adapters
        if adapter not in merged and (adapter in layer.lora_A or adapter in layer.lora_embedding_A)
    ]


def _unfoldable(layer: BaseTunerLayer) -> str | None:
    """What makes ``layer``'s merge one :func:`lora_folded` cannot reproduce, or None."""
    if type(layer) not in _FOLDABLE_LORA_LAYERS:
        return f"tuner layer {type(layer).__name__}"
    if isinstance(layer, _CONV_LORA_LAYERS) and layer.get_base_layer().groups > 1:
        return f"grouped {type(layer).__name__}"
    variants = [layer.lora_variant[adapter] for adapter in _adapters_to_fold(layer) if adapter in layer.lora_variant]
    unknown = [variant for variant in variants if not isinstance(variant, _FOLDABLE_VARIANTS)]
    return f"{type(unknown[0]).__name__} adapter" if unknown else None


def lora_fold_targets(model: torch.nn.Module) -> LoraFolds:
    """``id`` of every base tensor PEFT's ``merge_adapter`` would rewrite → the LoRA layers folding
    into it, in merge order.

    Keyed by identity, so a tied weight that ``named_parameters()`` lists under its other name (a
    LoRA'd ``lm_head`` sharing ``embed_tokens``) still gets its delta. Raises on a tuner layer whose
    merge :func:`lora_folded` cannot reproduce (``nn.MultiheadAttention`` LoRA, trainable tokens,
    quantized LoRA, a variant other than DoRA, a grouped conv PEFT cannot merge either), rather than
    send it unfolded.
    """
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


@torch.no_grad()
def lora_folded(param: torch.nn.Parameter, layers: list[LoraLayer]) -> torch.Tensor:
    """A new tensor holding what PEFT's merge of ``layers`` would write into ``param``; ``param`` itself
    is not touched.

    The same arithmetic as the in-place merge (``+= get_delta_weight``, a variant's ``merge_safe``,
    ``lora_B.bias * scaling`` into a folded bias), on ``param``'s own placement: a DTensor folds shard
    by shard, the delta's redistribution a collective every rank must enter. Costs one copy of
    ``param``'s local shard plus the delta.
    """
    folded = param.detach().clone()
    for layer in layers:
        folds_weight = param is _lora_weight(layer)
        for adapter in _adapters_to_fold(layer):
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


def lora_folded_data(param: torch.nn.Parameter, folds: LoraFolds | None) -> torch.Tensor:
    """``param``'s data with its LoRA delta folded in out of place (:func:`lora_folded`) when ``folds``
    (:func:`lora_fold_targets`) names it, else the live data. Collective for a folded DTensor."""
    layers = folds.get(id(param)) if folds else None
    return lora_folded(param, layers) if layers else param.data
