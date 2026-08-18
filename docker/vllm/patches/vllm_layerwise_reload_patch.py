"""Make vLLM's layerwise weight reload actually apply GPT-OSS expert weights.

``initialize_layerwise_reload`` moves every layer's tensors to the meta device and wraps their
``weight_loader``s so the reload can count what arrives. ``GptOssModel``'s weight loading writes the
MoE experts (``w13_weight`` / ``w2_weight`` / biases) and ``OAIAttention.sinks`` with a direct
``param.copy_()``, bypassing the wrapped loader. That copy lands on a meta tensor and is a no-op, the
layer's ``load_numel`` stays 0, and ``finalize_layerwise_reload`` re-registers the saved tensors, so
every ``/update_weights`` reverts the experts and still returns 200 OK while attention, embedding and
lm_head (which use real ``weight_loader``s) load normally. In RL that leaves the generator on its
launch experts, with ``logratio_mean`` drifting negative.

The fix is to leave those layer classes off the meta device and un-wrapped. Their direct ``copy_``
then writes real storage, ``can_load()`` stays False, and ``finalize`` skips them (``info.reset()``)
rather than restoring stale weights. Every other layer keeps the stock layerwise path.

The class set is version-dependent: ``RoutedExperts``, the pluggable layer owning the expert params
(``mlp.experts.routed_experts.*``), is not a ``FusedMoE`` subclass, so both names are listed.
``Dockerfile.vllm`` asserts at build time that the class owning the expert weights in the installed
vLLM is covered by this list, so a refactor fails the build rather than freezing the served experts.
Drop this file once vLLM's gpt-oss loader routes through ``weight_loader``.

Classes are matched by name so this module never imports vLLM's model code, keeping it import-order
safe.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Layer classes whose weight loads bypass the reload lifecycle:
#   FusedMoE / RoutedExperts — expert-weight owners; gpt-oss loads them by direct ``copy_``, and
#     excluding RoutedExperts keeps other families' in-place ``weight_loader`` loads live too
#   OAIAttention — holds gpt-oss ``sinks`` (direct copy)
#   Gemma4Router — a partial ``scale`` load takes layerwise's "delayed processing" branch, which
#     materializes the non-checkpoint ``root_size`` buffer as uninitialized memory into the live one
SKIP_LAYER_NAMES = frozenset({"FusedMoE", "RoutedExperts", "OAIAttention", "Gemma4Router"})

_APPLIED = False


def _is_direct_copy_layer(layer) -> bool:
    return any(cls.__name__ in SKIP_LAYER_NAMES for cls in type(layer).__mro__)


def apply() -> None:
    """Idempotently patch ``layerwise`` so direct-``copy_`` layers keep real storage."""
    global _APPLIED
    if _APPLIED:
        return

    from vllm.model_executor.model_loader.reload import layerwise  # noqa: PLC0415 — vLLM-only lazy import

    original_restore = layerwise.restore_layer_on_meta
    original_wrap = layerwise.initialize_online_processing

    def restore_layer_on_meta(layer, info):
        if _is_direct_copy_layer(layer):
            return  # keep real storage so the model's direct copy_ has somewhere to land
        return original_restore(layer, info)

    def initialize_online_processing(layer):
        if _is_direct_copy_layer(layer):
            return  # load_numel_total stays None, so can_load() is False and finalize skips it
        return original_wrap(layer)

    layerwise.restore_layer_on_meta = restore_layer_on_meta
    layerwise.initialize_online_processing = initialize_online_processing
    _APPLIED = True
    logger.info("vllm_layerwise_reload_patch applied (excluded: %s)", ", ".join(sorted(SKIP_LAYER_NAMES)))
