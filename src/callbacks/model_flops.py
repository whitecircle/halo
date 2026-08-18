"""Per-token training-FLOPS estimation behind MFU / S-MFU: PaLM/Megatron ``6·N + 12·L·S·H``.

Every count is per rank (``local_numel`` reads the TP/PP shard off the parameter, and the layer count
comes from this rank's module tree), so the estimate describes what this GPU computes.
"""

from transformers.utils import logging

from src.callbacks.parameter_stats import count_model_parameters
from src.distributed.expert_parallel.expert_weights import expert_weight_roots, experts_container_attrs
from src.distributed.runtime import local_numel
from src.models.structure import backbone_with_layers, decoder_layers

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


def estimate_model_flops_per_token(
    model, seq_length: int = ASSUMED_MAX_SEQ_LEN, pp_size: int = 1, tp_size: int = 1
) -> float:
    """Estimate training FLOPS per token: PaLM/Megatron ``6·N_local + 12·L·S·H`` (incl. attention scores).

    6N = linear projections (fwd 2N + bwd 4N); 12LSH = attention QK^T+Attn·V (fwd+bwd) per layer.
    Both terms are per-shard and per-stage, so ``pp_size``/``tp_size`` only reach the attention term
    (see :func:`estimate_attention_flops`); the parameter counts already carry the sharding.
    """
    attn_flops = estimate_attention_flops(model, seq_length, pp_size, tp_size)

    if next(model.parameters(), None) is None:
        config = getattr(model, "config", None)
        d_model = getattr(config, "hidden_size", None)
        n_layers = getattr(config, "num_hidden_layers", None)
        if d_model is None or n_layers is None:
            raise ValueError(
                "Cannot estimate model FLOPS/token: the model exposes no parameters and its config "
                "carries neither hidden_size nor num_hidden_layers."
            )
        return 6 * (12 * d_model * d_model * n_layers) + attn_flops

    all_params, trainable_params = count_model_parameters(model)

    # Frozen params (LoRA base) cost 4N rather than 6N; counting only trainable params would report
    # MFU near zero for adapter runs.
    if trainable_params == 0:
        base_flops = 6 * all_params
    else:
        frozen_params = max(all_params - trainable_params, 0)
        base_flops = 6 * trainable_params + 4 * frozen_params

    return base_flops + attn_flops


def estimate_attention_flops(model, seq_length: int, pp_size: int = 1, tp_size: int = 1) -> float:
    """Estimate attention score FLOPS per token: 12 × L × S × H / tp (QK^T + Attn·V, fwd+bwd). 0 if no config.

    The term derives from ``hidden_size``, which stays global on every rank, so the ``tp_size``
    division is explicit here while the ``6·N`` term takes it from the DTensor shard. CP does not
    divide it (the term is already per-token). ``hidden_size`` comes from
    ``config.get_text_config()``, so a VLM contributes its language tower rather than dropping the
    term; its vision attention is not modelled, which makes VLM MFU a slight under-estimate.
    """
    config = getattr(model, "config", None)
    if config is None:
        return 0.0

    text_config = config.get_text_config()

    hidden_size = getattr(text_config, "hidden_size", None)
    # This rank's own layers rather than config.num_hidden_layers / pp_size: the default pipeline
    # partition is head-weighted, not even (gpt-oss-20b at pp4 splits 8/8/6/2 against an assumed 6).
    backbone = backbone_with_layers(model)
    layers = decoder_layers(backbone) if backbone is not None else None
    n_layers = len(layers) if layers is not None else None
    if n_layers is None:
        config_layers = getattr(text_config, "num_hidden_layers", None)
        n_layers = config_layers / pp_size if config_layers is not None else None

    if n_layers is None or hidden_size is None:
        logger.warning_once(
            f"{type(config).__name__} exposes neither a decoder-layer list nor num_hidden_layers/"
            "hidden_size (text config included); the attention-score term is omitted from the FLOPS "
            "estimate, so reported MFU is an under-estimate."
        )
        return 0.0

    return 12.0 * n_layers * seq_length * hidden_size / max(tp_size, 1)
