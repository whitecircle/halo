"""Apply low-precision compute to a model before FSDP wrapping.

Sets per-layer compute precision on every EP MoE layer and swaps dense-MLP ``nn.Linear`` projections for
:class:`~src.kernels.lowp.linear.LowPrecisionLinear`, keeping attention, embeddings, ``lm_head``, norms
and the first/last few blocks in bf16. Call after EP-patching, before FSDP2 wrapping; no-op for bf16.
"""

from __future__ import annotations

import logging
import re

import torch.nn as nn
from transformers.distributed.tensor_parallel import ALL_PARALLEL_STYLES, _get_parameter_tp_plan

from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.tensor_parallel.state_dict import get_tp_mesh
from src.kernels.grouped_gemm import GroupedGemmPrecision
from src.kernels.lowp.linear import PRECISION_TO_FORMAT, LinearPrecision, LowPrecisionLinear

logger = logging.getLogger(__name__)

# The dense projections converted to low precision; attention q/k/v/o stay bf16 (small GEMMs, FA4's
# domain). The export tool (``scripts/after_training/quantize_to_lowp.py``) reads this roster so it
# quantizes exactly what the QAT forward did, instead of drifting behind a second list.
MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
# Only used to detect a family naming its projections differently (w1/w2/w3, fc1/fc2) — a silent no-op.
_MLP_CONTAINERS = ("mlp", "feed_forward", "ffn")

# The knob spellings ARE the LinearPrecision values, so the enum is the roster (no parallel table).
_LOWP_PRECISIONS = tuple(precision.value for precision in LinearPrecision)

_LAYER_INDEX = re.compile(r"\blayers\.(\d+)\b")


def _text_backbone_prefix(model: nn.Module) -> str | None:
    """Dotted-name prefix of the model's text decoder, from the model's own structure.

    Resolved via ``get_decoder()`` so block indexing never counts a vision tower's ``layers.N``. ``None``
    means the whole model is the backbone (a plain text model resolving to itself, or no accessor).
    """
    get_decoder = getattr(model, "get_decoder", None)
    if get_decoder is None:
        return None
    decoder = get_decoder()
    if decoder is None or decoder is model:
        return None
    return next((name for name, module in model.named_modules() if module is decoder), None)


def block_index(module_name: str, backbone_prefix: str | None = None) -> int | None:
    """Transformer-block index of ``module_name``, or None outside the text backbone.

    The export tool numbers CHECKPOINT KEYS through this same function, or the blocks the QAT forward
    kept in bf16 would ship quantized.
    """
    if backbone_prefix is not None:
        if not module_name.startswith(f"{backbone_prefix}."):
            return None
        module_name = module_name[len(backbone_prefix) + 1 :]
    match = _LAYER_INDEX.search(module_name)
    return int(match.group(1)) if match else None


def block_numbering_root(module_name: str) -> str | None:
    """The stack ``module_name``'s block index counts within — everything before ``layers.<N>``.

    :func:`block_index` numbers a name in whatever stack it finds, which is unambiguous only while one
    stack is in play. Having no live module to resolve a backbone prefix from, the export tool asserts
    that instead: two roots among its selected weights means a keep-window would keep the wrong blocks.
    """
    match = _LAYER_INDEX.search(module_name)
    return module_name[: match.start()] if match else None


def kept_block_indices(n_blocks: int, keep_first: int, keep_last: int) -> set[int]:
    """Blocks held at high precision at the two ends of an ``n_blocks``-deep backbone.

    The one home for the ``lowp_keep_first_blocks`` / ``lowp_keep_last_blocks`` window, shared by the
    training-time conversion and the export so a checkpoint quantizes what its forward quantized.
    Out-of-range ends are clamped, not rejected: keeping more blocks than exist keeps all of them.
    """
    kept = set(range(keep_first)) | {n_blocks - 1 - i for i in range(keep_last)}
    return {block for block in kept if 0 <= block < n_blocks}


def _warn_if_dense_mlp_unmatched(model: nn.Module, precision: str) -> None:
    """Warn when the dense-MLP swap matched nothing because the family names its projections differently
    (``w1``/``w2``/``w3``, ``fc1``/``fc2``) and the request is a silent no-op. Stays quiet for an all-MoE
    model, whose dense-MLP count is legitimately zero."""
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
            "Extend MLP_PROJECTIONS (src/kernels/lowp/mixed_precision.py) for this family.",
            precision,
            sorted(linears),
            list(MLP_PROJECTIONS),
        )


def _convert_dense_linear(linear: nn.Linear, name: str, precision: LinearPrecision, tp_plan: dict, tp_mesh) -> None:
    """Retype one dense projection in place, keeping its TP forward transforms.

    transformers' TP installs a module's transforms as an instance-level ``forward`` closed over the bound
    ``nn.Linear.forward``, which a class retype cannot reach. Strip it, retype, then let the plan's own
    style re-install the identical transforms — the same ``install_forward`` the load ran.
    """
    if "forward" not in vars(linear):
        LowPrecisionLinear.convert_(linear, precision)
        return
    style_name = _get_parameter_tp_plan(name, tp_plan, is_weight=False) if tp_plan else None
    if style_name is None or tp_mesh is None:
        raise RuntimeError(
            f"{name} carries an instance-level forward this conversion cannot re-install: it is not a "
            f"TP-plan style on a model with a TP mesh (plan style {style_name!r}, TP mesh "
            f"{'present' if tp_mesh is not None else 'absent'}). Retyping under it would leave the module "
            f"computing in bf16 while lowp_precision reports it converted."
        )
    del linear.forward
    LowPrecisionLinear.convert_(linear, precision)
    ALL_PARALLEL_STYLES[style_name].install_forward(linear, tp_mesh)


def apply_mixed_precision_compute(
    model: nn.Module,
    *,
    precision: str = "bf16",
    apply_dense_mlp: bool = True,
    apply_moe_experts: bool = True,
    keep_first_n_blocks: int = 0,
    keep_last_n_blocks: int = 0,
    expert_weight_cacheable: bool = True,
) -> dict:
    """Enable low-precision matmul compute on ``model`` in place (master weights stay bf16/fp32).

    Args:
        model: the (EP-patched) model, before FSDP2 wrapping.
        precision: ``"bf16"`` (no-op), ``"fp8"`` (mxfp8), ``"fp4"`` (nvfp4, most accurate) or ``"mxfp4"``
            (fast fp4) — the block-scale fake-quant format.
        expert_weight_cacheable: False when experts will be FSDP2-managed (``experts_fsdp_managed``):
            FSDP2 pins the unsharded param's version counter, so the per-step quant cache would serve
            the step-0 quantization forever; the layers then quantize every call instead (the same
            policy :class:`LowPrecisionLinear` applies to dense FSDP2 weights).
        apply_dense_mlp: convert dense MLP ``nn.Linear`` projections (set False to keep dense in bf16).
        apply_moe_experts: set the compute precision on EP MoE layers (set False to keep experts in bf16).
        keep_first_n_blocks / keep_last_n_blocks: blocks kept in bf16 at the network ends (NVFP4 recipe;
            first/last blocks are the most precision-sensitive).

    Returns:
        A summary dict (``precision``, ``moe_layers``, ``dense_linears``, ``kept_blocks``).
    """
    if precision == "bf16":
        return {"precision": "bf16", "moe_layers": 0, "dense_linears": 0, "kept_blocks": []}
    if precision not in _LOWP_PRECISIONS:
        accepted = ", ".join(repr(p) for p in ("bf16", *_LOWP_PRECISIONS))
        raise ValueError(f"precision must be one of {accepted}, got {precision!r}")

    backbone_prefix = _text_backbone_prefix(model)
    block_indices = {
        idx for name, _ in model.named_modules() if (idx := block_index(name, backbone_prefix)) is not None
    }
    n_blocks = (max(block_indices) + 1) if block_indices else 0
    kept = kept_block_indices(n_blocks, keep_first_n_blocks, keep_last_n_blocks)

    moe_count = 0
    loop_path_layers = 0
    if apply_moe_experts:
        grouped_precision = GroupedGemmPrecision(PRECISION_TO_FORMAT[LinearPrecision(precision)])
        for name, module in model.named_modules():
            if isinstance(module, EPMoELayerBase):
                if block_index(name, backbone_prefix) in kept:
                    continue
                honored = module.set_lowp_compute(grouped_precision, weight_cacheable=expert_weight_cacheable)
                moe_count += 1
                if not honored:  # low precision is honored only on the grouped-GEMM path
                    loop_path_layers += 1
        if loop_path_layers:
            logger.warning(
                "lowp_precision=%s requested but %d/%d MoE layers run the per-expert LOOP path "
                "(grouped GEMM disabled) — low precision will NOT apply to those experts. Set "
                "use_grouped_gemm=true to enable low-precision expert compute (GptOss under ETP "
                "always loops and cannot honor it).",
                precision,
                loop_path_layers,
                moe_count,
            )

    dense_count = 0
    if apply_dense_mlp:
        lin_precision = LinearPrecision(precision)
        tp_plan = getattr(model, "_tp_plan", None) or {}
        tp_mesh = get_tp_mesh(model)
        for name, module in model.named_modules():
            if isinstance(module, EPMoELayerBase):
                continue  # MoE experts use the grouped path, not nn.Linear
            # Text backbone only: a vision tower or projector naming its projections like the text
            # MLP stays bf16, which is what the export's tower fence reproduces.
            if backbone_prefix is not None and not name.startswith(f"{backbone_prefix}."):
                continue
            if block_index(name, backbone_prefix) in kept:
                continue
            for proj in MLP_PROJECTIONS:
                child = getattr(module, proj, None)
                if isinstance(child, nn.Linear) and not isinstance(child, LowPrecisionLinear):
                    _convert_dense_linear(child, f"{name}.{proj}" if name else proj, lin_precision, tp_plan, tp_mesh)
                    dense_count += 1
        if dense_count == 0:
            _warn_if_dense_mlp_unmatched(model, precision)

    # Layers on the loop path carry the precision but never honor it, so they count as converted
    # nowhere: a model whose every match loops quantizes exactly as little as one that matched none.
    effective_moe = moe_count - loop_path_layers
    if effective_moe == 0 and dense_count == 0:
        # Converting nothing means the run trains pure bf16 with no signal that the request was lost
        # (e.g. a fused-expert MoE at ep1 with use_grouped_gemm=false: no EP wrapper, no nn.Linear).
        if kept and not block_indices - kept:
            cause = (
                f"keep_first_n_blocks/keep_last_n_blocks exclude every one of the {n_blocks} blocks — "
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
            f"(apply_dense_mlp={apply_dense_mlp}, apply_moe_experts={apply_moe_experts}). "
            f"The run would silently train in bf16. For MoE experts, EP wrappers with the grouped-GEMM "
            f"path are required (use_grouped_gemm: true)."
        )

    summary = {
        "precision": precision,
        "moe_layers": moe_count,
        "dense_linears": dense_count,
        "kept_blocks": sorted(kept),
    }
    logger.info(
        "Mixed-precision compute enabled: precision=%s (fake-quant) | MoE layers=%d, dense MLP linears=%d, "
        "blocks kept in bf16=%s (master weights remain bf16/fp32)",
        precision,
        moe_count,
        dense_count,
        sorted(kept),
    )
    return summary
