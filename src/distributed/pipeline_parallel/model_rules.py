"""Whether a model can be split into pipeline stages: the structure gate and the run-settings gate.

The structure gate reads classes, config fields and weight identity, never a tensor's values, so it
holds on a meta-device shell and the stage-aware loader runs it before materializing a weight.
"""

from __future__ import annotations

import torch.nn as nn

from src.distributed.pipeline_parallel.families import PP_SPEC_MAP, resolve_pp_spec
from src.models.config_access import get_config_field, text_config, ties_word_embeddings
from src.models.head_transform import resolve_head_transform
from src.models.module_tree import backbone_with_layers, decoder_layers, embedding_module
from src.models.moe.balancing import resolve_balancing_mode
from src.models.moe.config_fields import has_routed_experts

# Every family spelling of the multi-token-prediction tail-layer count (DeepSeek/Bailing,
# Qwen3.5/3.6, Inkling).
MTP_LAYER_COUNT_FIELDS: tuple[str, ...] = ("num_nextn_predict_layers", "mtp_num_hidden_layers", "num_mtp_layers")


def require_backbone(model: nn.Module) -> nn.Module:
    """``model``'s transformer backbone with a decoder-layer list, refused when there is none."""
    backbone = backbone_with_layers(model)
    if backbone is None:
        raise ValueError(
            f"Pipeline parallelism needs a transformer backbone with a decoder-layer list, but none "
            f"was found on {type(model).__name__}. Supported layouts are exactly those "
            f"backbone_with_layers understands."
        )
    return backbone


def _reject_tied_embeddings(model: nn.Module) -> None:
    """Raise if the input embedding and the output head are one weight.

    Stage 0 holds the embedding and the last stage the head, so each copy would receive only its own
    path's gradient. The declared flag alone disqualifies too: every stage's saved config carries it,
    so ``from_pretrained`` would re-tie the head and discard the trained one.
    """
    declared = ties_word_embeddings(getattr(model, "config", None))
    input_weight = getattr(embedding_module(model, "input"), "weight", None)
    output_weight = getattr(embedding_module(model, "output"), "weight", None)

    # Shared storage cannot be the only signal: the stage-aware loader builds its meta shell with the
    # tie suppressed. The flag counts only where an output embedding exists, since a score head
    # inherits it and ties nothing.
    tied_storage = input_weight is not None and input_weight is output_weight
    if not (tied_storage or (declared and output_weight is not None)):
        return
    raise ValueError(
        f"{type(model).__name__} ties its output head to its input embedding (tie_word_embeddings), "
        f"which pipeline parallelism does not support: the embedding lands on stage 0 and the head on "
        f"the last stage, so the two copies diverge from step 1. Use a checkpoint with untied "
        f"embeddings, or train this model without PP."
    )


def _reject_aux_loss_balancing(model: nn.Module, moe_balancing: str) -> None:
    """Raise if this run's MoE load balancing rides the HF aux-loss path a stage severs.

    A stage runs the backbone and applies the head itself, so the ``*ForCausalLM.forward`` that adds
    the router aux loss never runs and the experts collapse with no error.
    """
    config = getattr(model, "config", None)
    if config is None or not has_routed_experts(config):
        return
    if embedding_module(model, "output") is None:
        return
    if resolve_balancing_mode(moe_balancing, model, is_moe=True) != "aux_loss":
        return
    coef = get_config_field(config, "router_aux_loss_coef", 0) or 0
    if coef <= 0:
        return
    raise ValueError(
        f"{type(model).__name__} balances its experts through the HuggingFace router aux loss "
        f"(moe_balancing={moe_balancing!r} resolves to 'aux_loss', router_aux_loss_coef={coef}), "
        f"which pipeline parallelism drops: a stage never runs the *ForCausalLM.forward that adds it. "
        f"Use moe_balancing=bias_update on a family whose EP wrappers support the aux-loss-free bias "
        f"update, moe_balancing=none to accept unbalanced experts, or train this model without PP."
    )


def _reject_live_mtp_layers(model: nn.Module) -> None:
    """Raise if the model built multi-token-prediction tail layers.

    MTP layers re-embed ``input_ids`` inside the backbone forward, a second cross-stage input stream.
    Liveness is the built decoder list outgrowing ``num_hidden_layers`` (the live remote-code Bailing
    v3 construction): the native families declare the field as checkpoint metadata and drop the MTP
    weights at load, so a declaration-only gate would refuse them.
    """
    text = text_config(getattr(model, "config", None))
    declared = {field: n for field in MTP_LAYER_COUNT_FIELDS if (n := getattr(text, field, 0) or 0) > 0}
    if not declared:
        return
    backbone = backbone_with_layers(model)
    num_hidden = getattr(text, "num_hidden_layers", None)
    # A pre-sliced stage holds fewer layers than num_hidden_layers; its unsplit shell was checked.
    if backbone is not None and num_hidden is not None and len(decoder_layers(backbone)) <= num_hidden:
        return
    detail = ", ".join(f"{field}={n}" for field, n in sorted(declared.items()))
    raise ValueError(
        f"{type(model).__name__} declares {detail} and its decoder list holds the extra layers: "
        f"multi-token-prediction tail layers re-embed input_ids inside the backbone forward, a "
        f"second cross-stage input stream pipeline parallelism cannot carry. Train with the MTP "
        f"head disabled (set every field above to 0 so the tail layers are never built) or "
        f"without PP."
    )


def reject_unsupported_pp_structure(model: nn.Module) -> None:
    """Raise if this model's structure cannot be split into pipeline stages.

    Raises:
        ValueError: the model has no decoder-layer list, its family's
            :class:`~src.distributed.pipeline_parallel.families.PPFamilySpec` carries a refusal, it ties
            its embedding to its head, transforms its head path beyond its declared
            :class:`~src.models.head_transform.HeadTransform`, built multi-token-prediction tail layers,
            or its backbone is not registered (``validated``) in ``PP_FAMILY_SPECS``.
    """
    backbone = require_backbone(model)
    backbone_name = type(backbone).__name__
    spec = resolve_pp_spec(backbone)
    if spec.refusal is not None:
        raise ValueError(f"{backbone_name} is refused for pipeline parallelism. {spec.refusal}")
    _reject_tied_embeddings(model)
    resolve_head_transform(model)  # raises on a head path the last stage cannot reproduce
    _reject_live_mtp_layers(model)
    if not spec.validated:
        registered = ", ".join(name for name, claim in PP_SPEC_MAP.items() if claim.validated)
        raise ValueError(
            f"{backbone_name} is not registered for pipeline parallelism (PP_FAMILY_SPECS); registered: {registered}."
        )


def reject_unsupported_pp_model(model: nn.Module, moe_balancing: str = "auto", *, pre_sliced: bool = False) -> None:
    """Raise if this model cannot be split into pipeline stages under this run's ``moe_balancing``.

    A model the stage-aware loader already sliced (``pre_sliced``) passed the structure gate on its
    meta shell, so only the run-settings gate is left.
    """
    if not pre_sliced:
        reject_unsupported_pp_structure(model)
    _reject_aux_loss_balancing(model, moe_balancing)
