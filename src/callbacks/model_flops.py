"""Per-token training-FLOPS estimation behind MFU / S-MFU: PaLM/Megatron ``6·N`` plus the
attention-score term of :mod:`src.models.attention_layout`.

Every count is the RANK's own (``local_numel`` reads the TP/PP shard off the parameter, the layout
comes off this rank's decoder layers), so the estimate describes what this GPU computes.
"""

from transformers.utils import logging

from src.callbacks.parameter_stats import count_model_parameters
from src.distributed.expert_parallel.expert_weights import expert_weight_roots, experts_container_attrs
from src.distributed.runtime import local_numel
from src.models.attention_layout import AttentionLayout, attention_layout
from src.models.loading.config_levels import get_config_field, text_config

logger = logging.get_logger(__name__)

# Fallback when nothing declares a bound. Not a real upper bound, so callers must not clamp to it.
ASSUMED_MAX_SEQ_LEN = 2048


def _is_expert_param(name: str) -> bool:
    """Whether a named parameter belongs to a sparse (routed) MoE expert.

    Only these take the ``top_k / num_experts`` sparsity factor; shared experts and router/gate params
    are always active. Both vocabularies come from the registry rather than a hardcoded name list.
    """
    name_lower = name.lower()

    if "shared_expert" in name_lower:
        return False
    if ".gate." in name_lower or ".router." in name_lower:
        return False

    if name.rsplit(".", 1)[-1] in expert_weight_roots():
        return True
    return "shared" not in name_lower and any(f".{attr}." in name for attr in experts_container_attrs())


def compute_expert_params(model, trainable_only: bool = False) -> float:
    """Sum the local element count of sparse (routed) expert parameters.

    ``trainable_only=True`` restricts to ``requires_grad`` experts; the default counts every expert,
    since a frozen LoRA base still runs the expert forward and its input gradient.
    """
    return float(
        sum(
            local_numel(p)
            for name, p in model.named_parameters()
            if (p.requires_grad or not trainable_only) and _is_expert_param(name)
        )
    )


def estimate_linear_flops_per_token(model) -> float:
    """The projection term: ``6·N_trainable + 4·N_frozen`` (fwd 2N + bwd 4N; a frozen LoRA base or
    frozen layers skip the weight gradient). A parameter-less (meta) model falls back to ``12·d²·L``."""
    if next(model.parameters(), None) is None:
        config = getattr(model, "config", None)
        d_model = getattr(config, "hidden_size", None)
        n_layers = getattr(config, "num_hidden_layers", None)
        if d_model is None or n_layers is None:
            raise ValueError(
                "Cannot estimate model FLOPS/token: the model exposes no parameters and its config "
                "carries neither hidden_size nor num_hidden_layers."
            )
        return 6 * (12 * d_model * d_model * n_layers)

    all_params, trainable_params = count_model_parameters(model)
    if trainable_params == 0:
        return 6 * all_params
    return 6 * trainable_params + 4 * max(all_params - trainable_params, 0)


def estimate_model_flops_per_token(
    model, seq_length: int = ASSUMED_MAX_SEQ_LEN, pp_size: int = 1, tp_size: int = 1
) -> float:
    """Training FLOPS per token with every token in a ``seq_length`` document: the projection term
    plus the attention-score term (:func:`estimate_attention_flops`)."""
    return estimate_linear_flops_per_token(model) + estimate_attention_flops(model, seq_length, pp_size, tp_size)


def resolve_attention_layout(model, pp_size: int = 1) -> tuple[AttentionLayout, float] | None:
    """This rank's attention layout and the share of it the rank computes.

    The share is 1 when the layout was read off the rank's own decoder layers (a pipeline stage holds
    exactly its slice) and ``1 / pp_size`` when the tree exposes no layer list and the config's full
    depth stands in. ``None`` for a model without a config, or one whose config describes no depth
    (the term is then omitted, loudly — an under-estimate rather than a guess).
    """
    config = getattr(model, "config", None)
    if config is None:
        return None
    decoder = text_config(config)
    if get_config_field(decoder, "layer_types") is None and get_config_field(decoder, "num_hidden_layers") is None:
        logger.warning_once(
            f"{type(config).__name__} declares neither layer_types nor num_hidden_layers (text config "
            "included); the attention-score term is omitted from the FLOPS estimate, so reported MFU "
            "is an under-estimate."
        )
        return None
    layout = attention_layout(model)
    return layout, 1.0 if layout.source == "layers" else 1.0 / max(pp_size, 1)


def estimate_attention_flops(model, seq_length: int, pp_size: int = 1, tp_size: int = 1) -> float:
    """Attention-score FLOPS per token for documents of ``seq_length`` tokens, over this rank's
    layers and its ``1 / tp_size`` share of every layer's heads. 0 without a config.

    The per-layer rule (full, sliding, chunked, sparse, compressed, none) and head width come from
    the layout; CP does not divide the term (already per-token).
    """
    resolved = resolve_attention_layout(model, pp_size)
    if resolved is None:
        return 0.0
    layout, share = resolved
    return layout.flops_per_token(seq_length) * share / max(tp_size, 1)
