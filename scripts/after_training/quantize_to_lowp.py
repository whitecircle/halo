"""Quantize a bf16/fp32 checkpoint to a block-scaled low-precision checkpoint (mxfp8 / mxfp4 / nvfp4).

Pairs with the mixed-precision QAT training path (``lowp_precision: fp8|fp4|mxfp4``, i.e.
mxfp8/nvfp4/mxfp4): a model trained that way keeps bf16/fp32 master weights and its forward already
computes in the block-scaled format. Quantizing those master weights to the same format here
reproduces the training forward exactly (``relerr = 0`` for both mxfp8 and nvfp4), so the converted
model runs in low precision with no accuracy loss beyond what training already saw. A model trained
in plain bf16 can also be converted, but that is post-training quantization rather than QAT.

For each quantized weight ``<name>.weight`` the output stores, following the ``compressed-tensors``
block-scaled convention:

  - ``<name>.weight_packed`` — quantized data: ``float8_e4m3fn`` (mxfp8) or packed ``uint8`` with two
    ``e2m1`` nibbles per byte (nvfp4),
  - ``<name>.weight_scale``  — one scale per block along the contraction axis: ``uint8`` e8m0 (mxfp8,
    block 32) or ``float8_e4m3fn`` (nvfp4, block 16),
  - ``<name>.weight_shape``  — the original ``[out, in]`` shape (the packed nvfp4 data halves the last dim),
  - ``<name>.weight_global_scale`` — nvfp4 only: the per-tensor fp32 multiplier of its two-level scaling
    (the ``e4m3`` block scale is relative to it, so dropping it rescales the weight).

Which weights those are is derived from the rosters that decide what QAT quantized (the trainer's
dense ``MLP_PROJECTIONS`` plus the EP layer classes' expert layouts), so the export cannot quantize a
weight the training forward left in bf16. Every other tensor (attention, embeddings, norms, lm_head,
biases, and a VLM's vision tower / projector) is copied unchanged in its original dtype. The four
``lowp_*`` scope knobs the run trained with must be passed back in (``--lowp_apply_dense_mlp`` /
``--lowp_apply_moe_experts`` / ``--lowp_keep_first_blocks`` / ``--lowp_keep_last_blocks``, defaulting
to the training defaults); they are not recoverable from the checkpoint. A
``quantization_config.json`` records the scheme so a loader (an inference engine, or the toolkit's
:mod:`src.kernels.lowp.quantization` dequantizer) can reconstruct the weights. The block layout is OCP
microscaling (mxfp8) / NVFP4 (1x16 e4m3), the formats TensorRT-LLM and vLLM consume via
``compressed-tensors`` / ModelOpt; mapping this manifest to a specific engine's loader remains an
integration step.

Usage:
  python scripts/after_training/quantize_to_lowp.py \
      --input_dir /mnt/models/my-bf16-ckpt --output_dir /mnt/models/my-mxfp8-ckpt \
      --format mxfp8 --verify
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
from itertools import groupby
from operator import itemgetter

import torch
from safetensors.torch import save_file
from transformers.utils import CONFIG_NAME

from src.checkpoint.format import (
    SAFETENSORS_INDEX_FILE,
    SAFETENSORS_WEIGHTS_FILE,
    copy_checkpoint_aux_files,
    remove_stale_checkpoint_files,
)
from src.checkpoint.tool_io import (
    SAFETENSORS_FLOAT_DTYPES,
    checkpoint_shard_files,
    iter_checkpoint_shard_entries,
    reject_in_place_conversion,
)
from src.distributed.expert_parallel.expert_weights import (
    ep_layer_class_by_model_type,
    experts_container_attrs,
    hf_fused_expert_keys,
    per_expert_layouts,
)
from src.kernels.lowp.mixed_precision import MLP_PROJECTIONS, block_index, block_numbering_root, kept_block_indices
from src.kernels.lowp.quantization import (
    FORMAT_BLOCK_SIZE,
    BlockScaledTensor,
    dequantize,
    quantize_mxfp4,
    quantize_mxfp8,
    quantize_nvfp4,
)
from src.log import configure_cli_logging

configure_cli_logging()
logger = logging.getLogger(__name__)

# Fused MoE experts are bare nn.Parameters (3-D [num_experts, in, out], no .weight suffix). The names
# come from the EP layer classes' _HF_FUSED_EXPERT_KEYS, so a new fused family needs no second list.
_FUSED_EXPERT_SUFFIXES = hf_fused_expert_keys()

# What low precision covers, derived from the two rosters that decide it: the dense MLP projections
# the trainer converts (``MLP_PROJECTIONS``, which the QAT forward quantizes and nothing else) and the
# expert-FFN weight names the EP layer classes declare, fused (3-D) and per-expert un-fused alike. A
# hand-written pattern would quantize weights the QAT forward left in bf16 (any `fc1`/`fc2` or bare
# `mlp` match).
_QUANTIZABLE_WEIGHT_NAMES = tuple(
    dict.fromkeys(
        (*MLP_PROJECTIONS, *_FUSED_EXPERT_SUFFIXES, *(key for layout in per_expert_layouts() for key in layout))
    )
)
_DEFAULT_INCLUDE = rf"\b({'|'.join(_QUANTIZABLE_WEIGHT_NAMES)})\b"
# Second fence: a VLM's vision tower and projector name their projections like the text MLP on some
# families (Qwen-VL's ViT blocks carry gate_proj/up_proj/down_proj), but the trainer converts the text
# backbone only (``_text_backbone_prefix``), so quantizing them would not match the QAT forward.
_DEFAULT_EXCLUDE = (
    r"(norm|layernorm|embed|lm_head|\.bias$|rotary|router|gate\.weight$"
    r"|vision_tower|vision_model|visual|multi_modal_projector)"
)
_QUANTIZERS = {"mxfp8": quantize_mxfp8, "mxfp4": quantize_mxfp4, "nvfp4": quantize_nvfp4}
# Round-trip dequant relerr --verify accepts: the format's own block-scaled error (mxfp8 ~0.02-0.05,
# nvfp4 ~0.08-0.15) with headroom. Above it the cause is structural, usually the contraction axis. The
# QAT forward is exact wherever a value lands in this band.
_VERIFY_RELERR_TOL = {"mxfp8": 0.08, "nvfp4": 0.25, "mxfp4": 0.35}
# Storage format -> the ``lowp_precision`` spelling that trains it.
_FMT_TO_LOWP_PRECISION = {"mxfp8": "fp8", "nvfp4": "fp4", "mxfp4": "mxfp4"}
# Per-expert un-fused hub layout (``...experts.3.gate_proj.weight``): expert weights other than the
# 3-D fused tensor, told apart from a dense MLP projection by this segment alone. The container
# spellings are the EP layer classes' own (``routed_experts`` beside ``experts``).
_PER_EXPERT_SEGMENT = re.compile(rf"\b({'|'.join(experts_container_attrs())})\.\d+\.")
# The dense projections the trainer converts (``MLP_PROJECTIONS``), by name: a per-expert roster
# spelling outside an expert segment (LFM-2's dense ``feed_forward.w1``) trains in bf16 and stays so.
_DENSE_PROJECTION = re.compile(rf"\.({'|'.join(MLP_PROJECTIONS)})\.weight$")


def _is_expert_weight(name: str, ndim: int) -> bool:
    """Whether ``name`` is an EP expert FFN weight — fused 3-D, or one expert of an un-fused layout.

    This is the seam ``--lowp_apply_moe_experts`` turns on, so it is the complement of "dense MLP
    projection" under the same include roster.
    """
    if ndim == 3 and any(name.endswith(suffix) for suffix in _FUSED_EXPERT_SUFFIXES):
        return True
    return _PER_EXPERT_SEGMENT.search(name) is not None


def _should_quantize(
    name: str,
    tensor: torch.Tensor,
    include: re.Pattern,
    exclude: re.Pattern,
    *,
    kept_blocks: frozenset[int] = frozenset(),
    apply_dense_mlp: bool = True,
    apply_moe_experts: bool = True,
) -> bool:
    """Quantize iff the tensor matches --include, not --exclude, is at least 2-D, is either a standard
    ``*.weight`` matrix or a fused 3-D expert tensor (``...experts.gate_up_proj`` etc.), and falls
    inside the training run's low-precision scope. The contraction axis (checked by the caller) must
    be divisible by the format block size.

    The scope arguments mirror ``lowp_apply_dense_mlp`` / ``lowp_apply_moe_experts`` /
    ``lowp_keep_first_blocks`` / ``lowp_keep_last_blocks`` and default to their training defaults, so
    the export reproduces the QAT forward instead of quantizing weights it left in bf16.
    """
    if include.search(name) is None or exclude.search(name) is not None or tensor.dim() < 2:
        return False
    if not (name.endswith(".weight") or (tensor.dim() == 3 and any(name.endswith(s) for s in _FUSED_EXPERT_SUFFIXES))):
        return False
    if block_index(name) in kept_blocks:
        return False
    if _is_expert_weight(name, tensor.dim()):
        return apply_moe_experts
    return apply_dense_mlp and _DENSE_PROJECTION.search(name) is not None


def _reject_unexportable_experts(
    input_dir: str,
    include: re.Pattern,
    exclude: re.Pattern,
    *,
    kept_blocks: frozenset[int],
    apply_moe_experts: bool,
    moe_family: bool,
) -> None:
    """Refuse an expert bank the scope would ship in bf16 while claiming QAT parity.

    Two spellings pass ``_should_quantize`` without matching: a 3-D bank under a name no EP layer
    class declares as fused (Step-3.7's hub ``moe.gate_proj``/``moe.up_proj`` ``[E, M, H]``, whose
    ``moe.down_proj`` alone matches), and a MoE checkpoint whose experts match the roster nowhere
    (Inkling's hub ``w13_weight``/``w2_weight``), which copies the whole model through and still
    writes a ``quantization_config``. Either way the served experts would run in a precision training
    never used. Headers only, ahead of the output directory.
    """
    if not apply_moe_experts:
        return
    expert_matches = 0
    for _shard, reader, name in iter_checkpoint_shard_entries(input_dir):
        if include.search(name) is None or exclude.search(name) is not None or block_index(name) in kept_blocks:
            continue
        header = reader.get_slice(name)
        ndim = len(header.get_shape())
        if header.get_dtype() not in SAFETENSORS_FLOAT_DTYPES:
            continue
        if ndim == 3 and not any(name.endswith(suffix) for suffix in _FUSED_EXPERT_SUFFIXES):
            raise ValueError(
                f"{name!r} is a 3-D floating expert tensor in the low-precision scope, but no EP layer "
                f"class declares its name as a fused expert key ({', '.join(_FUSED_EXPERT_SUFFIXES)}), so "
                f"it would be copied through unquantized beside its quantized siblings — an export that "
                f"does not reproduce the QAT forward. Declare the family's hub fused-expert keys on its "
                f"EP layer class (_HF_FUSED_EXPERT_KEYS), or pass --exclude to leave this bank out."
            )
        expert_matches += _is_expert_weight(name, ndim)
    if moe_family and expert_matches == 0:
        raise ValueError(
            f"{input_dir} is a MoE checkpoint (model_type {_read_model_types(input_dir)}), but no expert "
            f"weight is in the low-precision scope: its expert spelling is none the EP layer roster "
            f"declares, so every expert would be copied through in bf16 under a quantization_config "
            f"that claims QAT parity. Declare the family's hub expert keys on its EP layer class, or "
            f"export with --lowp_apply_moe_experts=false if training left the experts in bf16."
        )


def _read_model_types(input_dir: str) -> list[str]:
    """The ``model_type`` spellings this checkpoint declares, most specific first.

    Both the top level and the nested ``text_config`` are read: multimodal wrappers nest the LM config,
    and it is the LM's family that fixes the fused-expert layout.
    """
    cfg_path = os.path.join(input_dir, "config.json")
    if not os.path.isfile(cfg_path):
        return []
    try:
        with open(cfg_path) as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    candidates = [(cfg.get("text_config") or {}).get("model_type"), cfg.get("model_type")]
    return [candidate for candidate in candidates if candidate]


def _fused_expert_axis(input_dir: str) -> int | None:
    """Contraction axis for this checkpoint's fused 3-D expert tensors, or ``None`` if unresolvable.

    Resolved through the EP layer class that owns the family's checkpoint layout
    (:attr:`~src.distributed.expert_parallel.base_layer.EPMoELayerBase.HF_FUSED_EXPERT_CONTRACTION_AXIS`),
    so a new MoE family gets the right axis by declaring it once on its layer class. ``None`` means the
    ``model_type`` matches no registered family; the caller then raises if a fused expert shows up
    rather than guessing an axis, which would corrupt every dequantized expert.
    """
    by_model_type = ep_layer_class_by_model_type()
    for model_type in _read_model_types(input_dir):
        layer_cls = by_model_type.get(model_type)
        if layer_cls is not None:
            return layer_cls.HF_FUSED_EXPERT_CONTRACTION_AXIS
    return None


def _backbone_block_count(input_dir: str, include: re.Pattern, exclude: re.Pattern) -> int:
    """Depth of the backbone the export quantizes, read off the checkpoint's own key names.

    Counted over the keys that survive ``include``/``exclude`` (every block carries an MLP, and the
    exclude fence already drops a VLM's vision tower), so the numbering the keep-last window is
    measured against is exactly the one the quantization decision uses — no config layer-count field,
    which a composite config nests and a pruned export contradicts. Metadata-only: the shared walk
    hands out the shard reader, so a key-only pass reads no tensor.

    Raises when the selected keys number their blocks under more than one root: two interleaved
    stacks (a vision tower the exclude fence did not catch, a draft/MTP head) would share one index
    space, so the keep-window would hold back whichever stack owns that index. There is no live module
    to resolve a text-backbone prefix from at export time, so this refuses rather than guessing.
    """
    indices: set[int] = set()
    roots: set[str] = set()
    for _shard, _reader, name in iter_checkpoint_shard_entries(input_dir):
        if include.search(name) is None or exclude.search(name) is not None:
            continue
        idx = block_index(name)
        if idx is not None:
            indices.add(idx)
            roots.add(block_numbering_root(name))
    if len(roots) > 1:
        raise ValueError(
            f"--lowp_keep_first_blocks / --lowp_keep_last_blocks need one block numbering, but the "
            f"selected weights span {len(roots)}: {sorted(roots)}. Narrow --include/--exclude to the "
            f"text backbone, or export without a keep window."
        )
    return (max(indices) + 1) if indices else 0


def quantize_checkpoint(
    input_dir: str,
    output_dir: str,
    fmt: str,
    *,
    contraction_axis: int = -1,
    include: str = _DEFAULT_INCLUDE,
    exclude: str = _DEFAULT_EXCLUDE,
    apply_dense_mlp: bool = True,
    apply_moe_experts: bool = True,
    keep_first_blocks: int = 0,
    keep_last_blocks: int = 0,
    verify: bool = False,
) -> None:
    quantizer = _QUANTIZERS[fmt]
    block = FORMAT_BLOCK_SIZE[fmt]
    inc, exc = re.compile(include), re.compile(exclude)
    reject_in_place_conversion(input_dir, output_dir)

    quantized, copied, skipped, max_relerr = [], 0, [], 0.0
    weight_axes: dict[str, int] = {}
    weight_map: dict[str, str] = {}
    total_size = 0
    # Fused 3-D experts contract on the axis their family declares; 2-D weights use contraction_axis.
    fused_expert_axis = _fused_expert_axis(input_dir)
    # Index-driven enumeration: a co-located adapter_model.safetensors is never quantized into the
    # rebuilt index, and a per-rank sharded input is rejected before the output directory exists.
    # Resolved here rather than in the lazy walks below, whose refusal lands after the output exists.
    checkpoint_shard_files(input_dir)
    n_blocks = _backbone_block_count(input_dir, inc, exc) if (keep_first_blocks or keep_last_blocks) else 0
    kept_blocks = frozenset(kept_block_indices(n_blocks, keep_first_blocks, keep_last_blocks))
    _reject_unexportable_experts(
        input_dir,
        inc,
        exc,
        kept_blocks=kept_blocks,
        apply_moe_experts=apply_moe_experts,
        moe_family=fused_expert_axis is not None,
    )
    os.makedirs(output_dir, exist_ok=True)
    # The shared walk opens each shard once, in order, so grouping its entries by shard carries the
    # input's shard layout into the output: one written file per input file, under the same name.
    for shard, entries in groupby(iter_checkpoint_shard_entries(input_dir), key=itemgetter(0)):
        shard_name = os.path.basename(shard)
        out_tensors: dict[str, torch.Tensor] = {}
        for _shard, reader, name in entries:
            t = reader.get_tensor(name)
            if not _should_quantize(
                name,
                t,
                inc,
                exc,
                kept_blocks=kept_blocks,
                apply_dense_mlp=apply_dense_mlp,
                apply_moe_experts=apply_moe_experts,
            ):
                out_tensors[name] = t
                copied += 1
                continue
            if t.dim() == 3:
                if fused_expert_axis is None:
                    raise ValueError(
                        f"{name!r} is a fused 3-D expert tensor, but this checkpoint's model_type "
                        f"{_read_model_types(input_dir) or ['<missing>']} matches no EP layer class, so its "
                        f"contraction axis is unknown. Block-scaling the wrong axis corrupts every "
                        f"dequantized expert. Register the family (HF_MODEL_TYPES + "
                        f"HF_FUSED_EXPERT_CONTRACTION_AXIS on its EP layer class in "
                        f"src/distributed/expert_parallel/layers/), or pass --exclude to skip these weights."
                    )
                axis = fused_expert_axis
            else:
                axis = contraction_axis
            if t.shape[axis] % block != 0:
                out_tensors[name] = t  # contraction axis not block-divisible; leave high precision
                skipped.append(name)
                continue
            base = name[: -len(".weight")] if name.endswith(".weight") else name
            q: BlockScaledTensor = quantizer(
                t.cuda().float() if torch.cuda.is_available() else t.float(), axis=axis, block_size=block
            )
            out_tensors[f"{base}.weight_packed"] = q.data.cpu().contiguous()
            out_tensors[f"{base}.weight_scale"] = q.scales.cpu().contiguous()
            out_tensors[f"{base}.weight_shape"] = torch.tensor(list(t.shape), dtype=torch.int64)
            if q.global_scale is not None:
                out_tensors[f"{base}.weight_global_scale"] = q.global_scale.cpu().contiguous()
            quantized.append(name)
            weight_axes[name] = axis
            if verify:
                rel = ((dequantize(q).float().cpu() - t.float()).norm() / t.float().norm().clamp_min(1e-9)).item()
                # max(x, nan) returns x, so a non-finite weight would read as a clean round-trip.
                if not math.isfinite(rel):
                    raise ValueError(
                        f"{name!r} round-trips to a non-finite value (relerr={rel}); the source weight "
                        f"contains inf/NaN (max |w| = {t.float().abs().max().item()}). Fix the "
                        f"checkpoint — quantizing it produces a corrupt export."
                    )
                max_relerr = max(max_relerr, rel)
        save_file(out_tensors, os.path.join(output_dir, shard_name), metadata={"format": "pt"})
        for key, tensor in out_tensors.items():
            weight_map[key] = shard_name
            total_size += tensor.numel() * tensor.element_size()

    # Anything but a single model.safetensors needs an index so the renamed weight_packed/scale keys
    # stay discoverable. Keyed on the emitted filenames: a lone model-00001-of-00001 input would
    # otherwise get neither.
    wrote_index = set(weight_map.values()) != {SAFETENSORS_WEIGHTS_FILE}
    if wrote_index:
        index = {"metadata": {"total_size": total_size}, "weight_map": weight_map}
        with open(os.path.join(output_dir, SAFETENSORS_INDEX_FILE), "w") as fh:
            json.dump(index, fh, indent=2)

    # The index is in the keep-set only when this run wrote one, else a single-file run would leave a
    # stale index behind.
    keep = set(weight_map.values()) | ({SAFETENSORS_INDEX_FILE} if wrote_index else set())
    remove_stale_checkpoint_files(output_dir, keep=keep)

    scope = {
        "apply_dense_mlp": apply_dense_mlp,
        "apply_moe_experts": apply_moe_experts,
        "keep_first_blocks": keep_first_blocks,
        "keep_last_blocks": keep_last_blocks,
        "kept_blocks": sorted(kept_blocks),
    }
    _write_manifest(input_dir, output_dir, fmt, block, contraction_axis, quantized, weight_axes, scope)
    logger.info(
        f"quantized {len(quantized)} weights to {fmt}; copied {copied} tensors unchanged; "
        f"skipped {len(skipped)} (contraction axis not divisible by {block})."
    )
    if verify:
        tol = _VERIFY_RELERR_TOL[fmt]
        status = "OK (inherent format error)" if max_relerr <= tol else "WARN — check --contraction_axis"
        logger.info(f"round-trip dequant max relerr = {max_relerr:.4f} (warn above {tol}) — {status}")
    if skipped:
        logger.info(f"  left high-precision (not block-divisible): {skipped[:8]}{' …' if len(skipped) > 8 else ''}")


def _write_manifest(
    input_dir: str,
    output_dir: str,
    fmt: str,
    block: int,
    axis: int,
    names: list[str],
    weight_axes: dict[str, int],
    scope: dict,
) -> None:
    """Copy every non-weight file (config, tokenizer, chat_template.jinja, remote-code .py) + drop a
    quantization_config.json describing the block-scaled scheme."""
    copy_checkpoint_aux_files(input_dir, output_dir)
    manifest = {
        "quant_method": "block_scaled",
        "format": fmt,
        "weight_dtype": "float8_e4m3fn" if fmt == "mxfp8" else "uint8_packed_e2m1",
        "scale_dtype": "uint8_e8m0" if fmt in ("mxfp8", "mxfp4") else "float8_e4m3fn",
        "block_size": block,
        "contraction_axis": axis,
        # Per-weight contraction axis: 3-D fused experts differ by family, so a loader reads the axis
        # per weight here and falls back to contraction_axis only for names absent from this map.
        "weight_axes": weight_axes,
        "layout": "compressed-tensors block-scaled (weight_packed / weight_scale / weight_shape)",
        # nvfp4 is two-level: the e4m3 block scale is relative to a per-tensor fp32 global scale.
        "global_scale_key_suffix": ".weight_global_scale" if fmt == "nvfp4" else None,
        # The low-precision scope this export reproduced. Recorded because it is not derivable from
        # the weights: a re-export, or a comparison against the training run, needs the same four
        # ``lowp_*`` values.
        "scope": scope,
        "quantized_weights": names,
        "note": (
            f"Trained with mixed-precision QAT (lowp_precision: {_FMT_TO_LOWP_PRECISION[fmt]}, "
            f"i.e. {fmt}); quantizing the bf16/fp32 master to this format reproduces the QAT "
            "forward exactly. Dequantize via src.kernels.lowp.quantization.dequantize."
        ),
    }
    with open(os.path.join(output_dir, "quantization_config.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # Stamp the scheme into config.json, the key readers consult; it also flips the toolkit's
    # native-quantized detection (EP lazy loading routes such checkpoints to from_pretrained). Do not
    # set quant_method to the raw format name: "mxfp8"/"mxfp4"/"nvfp4" are transformers methods with a
    # different on-disk layout, which would turn the refusal into a wrong-layout load.
    config_path = os.path.join(output_dir, CONFIG_NAME)
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        config["quantization_config"] = manifest
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser(description="Quantize a bf16/fp32 checkpoint to block-scaled mxfp8/mxfp4/nvfp4.")
    p.add_argument("--input_dir", required=True, help="Source checkpoint directory (safetensors).")
    p.add_argument("--output_dir", required=True, help="Output directory for the quantized checkpoint.")
    p.add_argument("--format", choices=["mxfp8", "mxfp4", "nvfp4"], required=True, help="Block-scaled target format.")
    p.add_argument(
        "--contraction_axis",
        type=int,
        default=-1,
        help="Reduction axis to block along, for 2-D `*.weight` matrices only (default -1 = in_features "
        "for [out,in] Linear weights). Fused 3-D expert tensors use the axis their EP layer class "
        "declares and ignore this flag.",
    )
    p.add_argument("--include", default=_DEFAULT_INCLUDE, help="Regex of weight names to quantize.")
    p.add_argument("--exclude", default=_DEFAULT_EXCLUDE, help="Regex of weight names to keep high precision.")
    # The training run's lowp scope, which the checkpoint does not carry. Same names and defaults as
    # the ParallelismConfig knobs, so a config's values transfer verbatim.
    p.add_argument(
        "--lowp_apply_dense_mlp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether training quantized dense MLP projections (lowp_apply_dense_mlp).",
    )
    p.add_argument(
        "--lowp_apply_moe_experts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether training quantized MoE expert FFNs (lowp_apply_moe_experts).",
    )
    p.add_argument(
        "--lowp_keep_first_blocks",
        type=int,
        default=0,
        help="Leading transformer blocks training kept in bf16 (lowp_keep_first_blocks).",
    )
    p.add_argument(
        "--lowp_keep_last_blocks",
        type=int,
        default=0,
        help="Trailing transformer blocks training kept in bf16 (lowp_keep_last_blocks).",
    )
    p.add_argument("--verify", action="store_true", help="Report the max round-trip dequant relerr.")
    return p.parse_args()


if __name__ == "__main__":
    a = parse_args()
    quantize_checkpoint(
        a.input_dir,
        a.output_dir,
        a.format,
        contraction_axis=a.contraction_axis,
        include=a.include,
        exclude=a.exclude,
        apply_dense_mlp=a.lowp_apply_dense_mlp,
        apply_moe_experts=a.lowp_apply_moe_experts,
        keep_first_blocks=a.lowp_keep_first_blocks,
        keep_last_blocks=a.lowp_keep_last_blocks,
        verify=a.verify,
    )
