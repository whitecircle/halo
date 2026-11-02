"""Apply a :class:`~halo.kernels.lowp.spec.LowpSpec` after EP patching, before FSDP2 wrapping: EP expert
compute and the dense MLP projections go low precision; attention, embeddings, ``lm_head``, norms and the
kept end blocks stay bf16."""

from __future__ import annotations

import logging

import torch.nn as nn
from transformers.distributed.tensor_parallel import ALL_PARALLEL_STYLES, _get_parameter_plan

from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from halo.distributed.mesh import model_tp_mesh
from halo.distributed.tensor_parallel.local_linears import LINEAR_TP_STYLES, install_local_forward
from halo.kernels.lowp.linear import LowPrecisionLinear
from halo.kernels.lowp.quantization import FORMAT_BLOCK_SIZE
from halo.kernels.lowp.spec import LOWP_RECIPES, MLP_PROJECTIONS, LowpSpec, QuantFormat, kept_block_indices
from halo.models.module_tree import decoder_layer_index, find_module_path
from halo.runtime.dtensor import to_local

logger = logging.getLogger(__name__)

# Where a family naming its projections differently (w1/w2/w3, fc1/fc2) is detected as a silent no-op.
_MLP_CONTAINERS = ("mlp", "feed_forward", "ffn")


def _text_backbone_prefix(model: nn.Module) -> str | None:
    """Dotted name of the model's text decoder (``get_decoder()``), so block numbering never counts a vision
    tower's ``layers.N``; None when the whole model is the backbone."""
    get_decoder = getattr(model, "get_decoder", None)
    decoder = get_decoder() if get_decoder is not None else None
    if decoder is None or decoder is model:
        return None
    return find_module_path(model, decoder)


def _warn_if_dense_mlp_unmatched(model: nn.Module, precision: str) -> None:
    """Warn when the family names its MLP projections other than :data:`MLP_PROJECTIONS`; quiet for an
    all-MoE model, whose dense-MLP count is legitimately zero."""
    linears = {
        child_name
        for name, module in model.named_modules()
        if not isinstance(module, EPMoELayerBase) and name.rsplit(".", 1)[-1] in _MLP_CONTAINERS
        for child_name, child in module.named_children()
        if isinstance(child, nn.Linear)
    }
    if linears and not linears & set(MLP_PROJECTIONS):
        logger.warning(
            "lowp_precision=%s with lowp_apply_dense_mlp=true converted NO dense MLP linear: this "
            "model's MLP projections are named %s, not %s — dense low precision will NOT apply. "
            "Extend MLP_PROJECTIONS for this family.",
            precision,
            sorted(linears),
            list(MLP_PROJECTIONS),
        )


def _require_block_aligned(
    subject: str, contraction: int, precision: str, quant_format: QuantFormat, *, split_by: str | None, opt_out: str
) -> None:
    """Raise unless ``contraction`` splits into ``quant_format`` blocks: the fake-quant blocks both GEMM operands
    along it, so a misaligned axis would fail the first forward instead. ``split_by`` names the parallel size that
    shards the axis on this run (None where it is whole), ``opt_out`` the flag that keeps the module bf16."""
    block = FORMAT_BLOCK_SIZE[quant_format]
    if contraction % block == 0:
        return
    fitting = sorted(p for p, r in LOWP_RECIPES.items() if contraction % FORMAT_BLOCK_SIZE[r.quant_format] == 0)
    lower_split = (
        f"lower {split_by} (it shards this axis) so each rank's shard is a multiple of {block}, " if split_by else ""
    )
    raise ValueError(
        f"lowp_precision={precision!r} cannot run {subject}: it contracts over {contraction}, which {quant_format} "
        f"does not split into blocks of {block}. To run it, {lower_split}use a lowp_precision whose block divides "
        f"{contraction} ({fitting or 'none'}), or set {opt_out}: false."
    )


def _require_block_aligned_experts(
    name: str, layer: EPMoELayerBase, precision: str, quant_format: QuantFormat
) -> None:
    """Check every grouped-GEMM expert weight of ``layer``. Stored ``[E, K, N]``, its contraction is dim 1: the
    hidden size, or the intermediate, which expert TP shards. LoRA adapters (bf16 under lowp) and biases are
    exempt."""
    adapters = {adapter for adapter, _ in layer.expert_lora_named_params()}
    for weight_name, weight in layer.expert_named_params():
        if weight_name in adapters or weight.ndim != 3:
            continue
        etp_sharded = layer.expert_tp_size > 1 and layer.EXPERT_TP_SHARD_DIMS.get(weight_name) == 1
        _require_block_aligned(
            f"{name} expert weight {weight_name} {tuple(weight.shape)}",
            weight.shape[1],
            precision,
            quant_format,
            split_by="expert_tensor_parallel_size" if etp_sharded else None,
            opt_out="lowp_apply_moe_experts",
        )


def _require_block_aligned_dense(name: str, linear: nn.Linear, precision: str, quant_format: QuantFormat) -> None:
    """Check one dense projection. Its contraction is the weight's last axis as this rank holds it: whole under
    colwise TP, sharded under rowwise, where the fake-quant blocks the local shard."""
    contraction = to_local(linear.weight).shape[-1]
    _require_block_aligned(
        f"{name} {tuple(linear.weight.shape)}",
        contraction,
        precision,
        quant_format,
        split_by="tensor_parallel_size" if contraction != linear.weight.shape[-1] else None,
        opt_out="lowp_apply_dense_mlp",
    )


def _convert_dense_linear(linear: nn.Linear, name: str, quant_format: QuantFormat, tp_plan: dict, tp_mesh) -> None:
    """Retype one dense projection in place, keeping its TP forward transforms: TP installs them as an
    instance-level ``forward`` a class retype cannot reach, so strip it, retype, and re-install the local-shard
    forward (transformers' own for a style it does not cover)."""
    if "forward" not in vars(linear):
        LowPrecisionLinear.convert_(linear, quant_format)
        return
    style_name = _get_parameter_plan(name, tp_plan, is_weight=False) if tp_plan else None
    if style_name is None or tp_mesh is None:
        raise RuntimeError(
            f"{name} carries an instance-level forward this conversion cannot re-install: it is not a "
            f"TP-plan style on a model with a TP mesh (plan style {style_name!r}, TP mesh "
            f"{'present' if tp_mesh is not None else 'absent'}). Retyping under it would leave the module "
            f"computing in bf16 while lowp_precision reports it converted."
        )
    del linear.forward
    LowPrecisionLinear.convert_(linear, quant_format)
    if style_name in LINEAR_TP_STYLES:
        install_local_forward(linear, style_name, name=name)
    else:
        ALL_PARALLEL_STYLES[style_name].install_forward(linear, tp_mesh)


def _convert_dense_mlps(
    model: nn.Module, precision: str, quant_format: QuantFormat, kept: set[int], backbone_prefix: str | None
) -> int:
    """Retype the text backbone's dense MLP projections outside the ``kept`` blocks; returns how many."""
    tp_plan = getattr(model, "_tp_plan", None) or {}
    tp_mesh = model_tp_mesh(model)
    converted = 0
    for name, module in model.named_modules():
        # A vision tower or projector naming its projections like the text MLP stays bf16, as the
        # export's tower fence assumes.
        if (
            isinstance(module, EPMoELayerBase)
            or (backbone_prefix is not None and not name.startswith(f"{backbone_prefix}."))
            or decoder_layer_index(name, backbone_prefix) in kept
        ):
            continue
        for proj in MLP_PROJECTIONS:
            child = getattr(module, proj, None)
            if isinstance(child, nn.Linear) and not isinstance(child, LowPrecisionLinear):
                path = f"{name}.{proj}" if name else proj
                _require_block_aligned_dense(path, child, precision, quant_format)
                _convert_dense_linear(child, path, quant_format, tp_plan, tp_mesh)
                converted += 1
    return converted


def apply_lowp_compute(model: nn.Module, lowp: LowpSpec, *, expert_weight_cacheable: bool) -> dict:
    """Enable ``lowp`` on ``model`` in place and return ``{precision, moe_layers, dense_linears,
    kept_blocks}``. ``expert_weight_cacheable`` is False when FSDP2 manages the experts: it pins the
    unsharded param's version counter, so a per-step quantization cache would serve step-0 weights forever."""
    lowp.validate()
    if not lowp.enabled:
        return {"precision": "bf16", "moe_layers": 0, "dense_linears": 0, "kept_blocks": []}
    precision, quant_format = lowp.precision, LOWP_RECIPES[lowp.precision].quant_format

    backbone_prefix = _text_backbone_prefix(model)
    block_indices = {
        idx for name, _ in model.named_modules() if (idx := decoder_layer_index(name, backbone_prefix)) is not None
    }
    n_blocks = max(block_indices) + 1 if block_indices else 0
    kept = kept_block_indices(n_blocks, lowp.keep_first_blocks, lowp.keep_last_blocks)

    moe_count = loop_path_layers = 0
    if lowp.apply_moe_experts:
        for name, module in model.named_modules():
            if isinstance(module, EPMoELayerBase) and decoder_layer_index(name, backbone_prefix) not in kept:
                moe_count += 1
                if module.set_lowp_compute(quant_format, weight_cacheable=expert_weight_cacheable):
                    _require_block_aligned_experts(name, module, precision, quant_format)
                else:
                    loop_path_layers += 1
        if loop_path_layers:
            logger.warning(
                "lowp_precision=%s requested but %d/%d MoE layers run the per-expert LOOP path "
                "(grouped GEMM disabled) — low precision will NOT apply to those experts. Set "
                "use_grouped_gemm=true to enable low-precision expert compute.",
                precision,
                loop_path_layers,
                moe_count,
            )

    dense_count = 0
    if lowp.apply_dense_mlp:
        dense_count = _convert_dense_mlps(model, precision, quant_format, kept, backbone_prefix)
        if dense_count == 0:
            _warn_if_dense_mlp_unmatched(model, precision)

    # Loop-path layers carry the format but never honor it, so they count as converted nowhere.
    if moe_count == loop_path_layers and dense_count == 0:
        if kept and not block_indices - kept:
            cause = (
                f"lowp_keep_first_blocks/lowp_keep_last_blocks exclude every one of the {n_blocks} blocks — "
                f"nothing is left to quantize"
            )
        elif loop_path_layers:
            cause = (
                f"all {loop_path_layers} matched MoE layer(s) run the per-expert LOOP path, which "
                f"cannot honor low precision, and no dense MLP nn.Linear matched"
            )
        else:
            cause = "no EP MoE layer and no dense MLP nn.Linear matched"
        raise RuntimeError(
            f"lowp_precision={precision!r} converted zero modules — {cause} "
            f"(apply_dense_mlp={lowp.apply_dense_mlp}, apply_moe_experts={lowp.apply_moe_experts}). "
            f"The run would silently train in bf16. For MoE experts, EP layers with the grouped-GEMM "
            f"path are required (use_grouped_gemm: true)."
        )

    logger.info(
        "Mixed-precision compute enabled: precision=%s (fake-quant) | MoE layers=%d, dense MLP linears=%d, "
        "blocks kept in bf16=%s (master weights remain bf16/fp32)",
        precision,
        moe_count,
        dense_count,
        sorted(kept),
    )
    return {"precision": precision, "moe_layers": moe_count, "dense_linears": dense_count, "kept_blocks": sorted(kept)}
