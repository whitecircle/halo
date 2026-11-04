"""Per-rank training-FLOPs model behind MFU / S-MFU: PaLM/Megatron ``6·N`` plus the attention-score term. Every count
is this rank's own (``local_numel`` reads the TP/PP shard off the parameter)."""

import logging
from dataclasses import dataclass

from halo.callbacks.parameter_stats import count_model_parameters
from halo.distributed.parallelism_config import ParallelismConfig
from halo.models.attention.layout import AttentionLayout, attention_layout
from halo.models.config_access import get_config_field, text_config
from halo.models.moe.config_fields import resolve_expert_count, resolve_router_topk
from halo.models.moe.family_specs import expert_weight_roots, experts_container_attrs
from halo.runtime.dtensor import local_numel

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RankFlopsModel:
    """This rank's training FLOPs per token, dense and active. Only ``top_k`` of ``num_experts`` fire per token, so the
    active figure charges ``sparsity_factor`` of the expert parameters (MoE-CAP, arXiv:2412.07067). Both include the
    attention term at the length bound, which a step that measured its documents replaces with the measurement."""

    model_flops_per_token: float
    active_model_flops_per_token: float
    attention_flops_per_token: float
    attention_layout: AttentionLayout | None
    attention_share: float
    local_params: float
    full_params: float
    params_ratio: float
    num_experts: int
    top_k: int
    sparsity_factor: float
    local_expert_params: float
    local_active_params: float
    full_active_params: float


def _is_expert_param(name: str) -> bool:
    """The only parameters the sparsity factor scales; shared experts and router/gate parameters are always active."""
    name_lower = name.lower()
    if "shared_expert" in name_lower:
        return False
    if ".gate." in name_lower or ".router." in name_lower:
        return False
    if name.rsplit(".", 1)[-1] in expert_weight_roots():
        return True
    return "shared" not in name_lower and any(f".{attr}." in name for attr in experts_container_attrs())


def _training_flops_per_token(trainable_params: float, frozen_params: float) -> float:
    """``6·N_trainable + 4·N_frozen``: forward 2N and input gradient 2N for every parameter, plus the weight
    gradient 2N for a trainable one. A frozen rank still back-propagates the input gradient upstream needs."""
    return 6.0 * trainable_params + 4.0 * frozen_params


def _attention_term(model, seq_len: int, pp_size: int, tp_size: int) -> tuple[AttentionLayout | None, float, float]:
    """``(layout, share, flops_per_token)``. The share is 1 when the layout was read off the rank's own layers and
    ``1 / pp_size`` when the config's full depth stands in; the rank computes ``1 / tp_size`` of each layer's heads.
    CP does not divide the term (already per token). A config describing no depth omits the term, loudly."""
    config = getattr(model, "config", None)
    if config is None:
        return None, 1.0, 0.0
    decoder = text_config(config)
    if get_config_field(decoder, "layer_types") is None and get_config_field(decoder, "num_hidden_layers") is None:
        logger.warning_once(
            f"{type(config).__name__} declares neither layer_types nor num_hidden_layers (text config "
            "included); the attention-score term is omitted from the FLOPS estimate, so reported MFU "
            "is an under-estimate."
        )
        return None, 1.0, 0.0
    layout = attention_layout(model)
    share = 1.0 if layout.source == "layers" else 1.0 / max(pp_size, 1)
    return layout, share, layout.flops_per_token(seq_len) * share / max(tp_size, 1)


def _moe_param_split(model, num_experts: int, top_k: int) -> tuple[float, float, float, float]:
    """``(sparsity_factor, expert, trainable_expert, frozen_expert)`` local counts. A frozen expert still counts: a
    frozen LoRA base runs the expert forward and its input gradient."""
    if num_experts <= 0 or top_k <= 0:
        return 1.0, 0.0, 0.0, 0.0
    experts = [p for name, p in model.named_parameters() if _is_expert_param(name)]
    expert_params = float(sum(local_numel(p) for p in experts))
    trainable_expert_params = float(sum(local_numel(p) for p in experts if p.requires_grad))
    return (
        top_k / num_experts,
        expert_params,
        trainable_expert_params,
        max(expert_params - trainable_expert_params, 0.0),
    )


def _linear_flops_per_token(model, trainable_params: float, frozen_params: float) -> float:
    """The ``6·N`` term; a parameter-less (meta) model stands in 12·d²·L trainable parameters."""
    if next(model.parameters(), None) is not None:
        return _training_flops_per_token(trainable_params, frozen_params)
    config = getattr(model, "config", None)
    d_model = getattr(config, "hidden_size", None)
    n_layers = getattr(config, "num_hidden_layers", None)
    if d_model is None or n_layers is None:
        raise ValueError(
            "Cannot estimate model FLOPS/token: the model exposes no parameters and its config "
            "carries neither hidden_size nor num_hidden_layers."
        )
    return _training_flops_per_token(12 * d_model * d_model * n_layers, 0)


def estimate_rank_flops(
    model, *, seq_len: int, parallelism_config: ParallelismConfig, full_model_params: float | None
) -> RankFlopsModel:
    """``full_model_params`` is the baseline the distributed-efficiency ratios divide by. Raises on a model it cannot
    measure; the caller degrades."""
    layout, share, attention_flops = _attention_term(
        model, seq_len, parallelism_config.pp_size, parallelism_config.tp_size
    )
    local_params, trainable_params = count_model_parameters(model)
    frozen_params = local_params - trainable_params
    linear_flops = _linear_flops_per_token(model, trainable_params, frozen_params)

    has_full_baseline = full_model_params is not None and full_model_params > 0
    full_params = float(full_model_params) if has_full_baseline else float(local_params)
    params_ratio = full_params / float(local_params) if has_full_baseline else 1.0

    # The shared router probe, so the balancing setup and the MoE load metrics classify a family alike.
    config = getattr(model, "config", None)
    num_experts, top_k = resolve_expert_count(config), resolve_router_topk(config)
    sparsity, expert_params, trainable_expert_params, frozen_expert_params = _moe_param_split(
        model, num_experts, top_k
    )
    # Sparsity applies inside the 6N and 4N buckets separately.
    trainable_active = trainable_params - trainable_expert_params * (1.0 - sparsity)
    frozen_active = frozen_params - frozen_expert_params * (1.0 - sparsity)
    local_active_params = float(trainable_active + frozen_active)
    # This rank holds num_experts/ep_size experts but serves the whole EP group's tokens, so its active FLOPs/token is
    # ep-invariant; expert_tp_size already divides the expert counts.
    expert_duty = sparsity * parallelism_config.ep_size
    trainable_active_flops = trainable_params - trainable_expert_params * (1.0 - expert_duty)
    frozen_active_flops = frozen_params - frozen_expert_params * (1.0 - expert_duty)

    if has_full_baseline and sparsity < 1.0:
        # Both axes shard the expert FFN; counting only EP would push expert_tp_size of the bank into shared.
        full_expert_params = expert_params * parallelism_config.ep_size * parallelism_config.expert_tp_size
        full_shared_params = full_model_params - full_expert_params
        full_active_params = full_shared_params + sparsity * full_expert_params
    else:
        full_active_params = local_active_params

    return RankFlopsModel(
        model_flops_per_token=linear_flops + attention_flops,
        active_model_flops_per_token=_training_flops_per_token(trainable_active_flops, frozen_active_flops)
        + attention_flops,
        attention_flops_per_token=attention_flops,
        attention_layout=layout,
        attention_share=share,
        local_params=float(local_params),
        full_params=full_params,
        params_ratio=params_ratio,
        num_experts=num_experts,
        top_k=top_k,
        sparsity_factor=sparsity,
        local_expert_params=expert_params,
        local_active_params=local_active_params,
        full_active_params=full_active_params,
    )
