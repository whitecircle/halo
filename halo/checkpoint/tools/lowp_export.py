"""Quantize a bf16/fp32 checkpoint to a block-scaled low-precision checkpoint (mxfp8 / mxfp4 / nvfp4) in the
compressed-tensors layout, scoped to exactly the weights the QAT forward quantized, with its manifest."""

from __future__ import annotations

import json
import logging
import math
import os
import re
from dataclasses import dataclass, field
from itertools import groupby
from operator import itemgetter

import torch

from halo.checkpoint.atomic import save_safetensors_file
from halo.checkpoint.config_export import write_config_json
from halo.checkpoint.format import SAFETENSORS_METADATA, SAFETENSORS_WEIGHTS_FILE
from halo.checkpoint.reading import (
    SAFETENSORS_FLOAT_DTYPES,
    checkpoint_shard_files,
    detect_model_types,
    iter_checkpoint_shard_entries,
    read_checkpoint_config,
)
from halo.checkpoint.tools.offline_export import copy_checkpoint_aux_files, reject_in_place_conversion
from halo.checkpoint.writing import remove_stale_checkpoint_files, write_merged_index
from halo.kernels.lowp.quantization import FORMAT_BLOCK_SIZE, QUANTIZERS, BlockScaledTensor, dequantize
from halo.kernels.lowp.spec import LOWP_RECIPES, MLP_PROJECTIONS, LowpSpec, kept_block_indices
from halo.models.module_tree import decoder_layer_index, decoder_layer_stack_prefix
from halo.models.moe.expert_layout import HF_FUSED_EXPERT_KEYS
from halo.models.moe.family_specs import experts_container_attrs, find_moe_family_spec, per_expert_layouts

logger = logging.getLogger(__name__)

# What low precision covers, derived from the two rosters that decide it: the dense MLP projections
# the trainer converts (``MLP_PROJECTIONS``, which the QAT forward quantizes and nothing else) and the
# expert-FFN weight names the family specs declare, fused (3-D) and per-expert un-fused alike. A
# hand-written pattern would quantize weights the QAT forward left in bf16 (any `fc1`/`fc2` or bare
# `mlp` match).
_QUANTIZABLE_WEIGHT_NAMES = tuple(
    dict.fromkeys(
        (*MLP_PROJECTIONS, *HF_FUSED_EXPERT_KEYS, *(key for layout in per_expert_layouts() for key in layout))
    )
)
DEFAULT_INCLUDE = rf"\b({'|'.join(_QUANTIZABLE_WEIGHT_NAMES)})\b"
# Second fence: a VLM's vision tower and projector name their projections like the text MLP on some
# families (Qwen-VL's ViT blocks carry gate_proj/up_proj/down_proj), but the trainer converts the text
# backbone only (``_text_backbone_prefix``), so quantizing them would not match the QAT forward.
DEFAULT_EXCLUDE = (
    r"(norm|layernorm|embed|lm_head|\.bias$|rotary|router|gate\.weight$"
    r"|vision_tower|vision_model|visual|multi_modal_projector)"
)
# Round-trip dequant relerr --verify accepts: the format's own block-scaled error (mxfp8 ~0.02-0.05,
# nvfp4 ~0.08-0.15) with headroom. Above it the cause is structural, usually the contraction axis. The
# QAT forward is exact wherever a value lands in this band.
_VERIFY_RELERR_TOL = {"mxfp8": 0.08, "nvfp4": 0.25, "mxfp4": 0.35}
# Per-expert un-fused hub layout (``...experts.3.gate_proj.weight``): expert weights other than the
# 3-D fused tensor, told apart from a dense MLP projection by this segment alone. The container
# spellings are the family specs' own (``routed_experts`` beside ``experts``).
_PER_EXPERT_SEGMENT = re.compile(rf"\b({'|'.join(experts_container_attrs())})\.\d+\.")
# The dense projections the trainer converts (``MLP_PROJECTIONS``), by name: a per-expert roster
# spelling outside an expert segment (LFM-2's dense ``feed_forward.w1``) trains in bf16 and stays so.
_DENSE_PROJECTION = re.compile(rf"\.({'|'.join(MLP_PROJECTIONS)})\.weight$")


def _is_expert_weight(name: str, ndim: int) -> bool:
    """Whether ``name`` is an EP expert FFN weight — fused 3-D, or one expert of an un-fused layout.

    This is the seam ``--lowp_apply_moe_experts`` turns on, so it is the complement of "dense MLP
    projection" under the same include roster.
    """
    if ndim == 3 and name.endswith(HF_FUSED_EXPERT_KEYS):
        return True
    return _PER_EXPERT_SEGMENT.search(name) is not None


def _should_quantize(
    name: str,
    tensor: torch.Tensor,
    include: re.Pattern,
    exclude: re.Pattern,
    *,
    kept_blocks: frozenset[int] = frozenset(),
    apply_dense_mlp: bool = LowpSpec.apply_dense_mlp,
    apply_moe_experts: bool = LowpSpec.apply_moe_experts,
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
    if not (name.endswith(".weight") or (tensor.dim() == 3 and name.endswith(HF_FUSED_EXPERT_KEYS))):
        return False
    if decoder_layer_index(name) in kept_blocks:
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
        if (
            include.search(name) is None
            or exclude.search(name) is not None
            or decoder_layer_index(name) in kept_blocks
        ):
            continue
        header = reader.get_slice(name)
        ndim = len(header.get_shape())
        if header.get_dtype() not in SAFETENSORS_FLOAT_DTYPES:
            continue
        if ndim == 3 and not name.endswith(HF_FUSED_EXPERT_KEYS):
            raise ValueError(
                f"{name!r} is a 3-D floating expert tensor in the low-precision scope, but its name is not a "
                f"fused expert key ({', '.join(HF_FUSED_EXPERT_KEYS)}), so "
                f"it would be copied through unquantized beside its quantized siblings — an export that "
                f"does not reproduce the QAT forward. Add the family's hub fused-expert keys to "
                f"HF_FUSED_EXPERT_KEYS, or pass --exclude to leave this bank out."
            )
        expert_matches += _is_expert_weight(name, ndim)
    if moe_family and expert_matches == 0:
        raise ValueError(
            f"{input_dir} is a MoE checkpoint (model_type {detect_model_types(input_dir)}), but no expert "
            f"weight is in the low-precision scope: its expert spelling is none the family specs "
            f"declare, so every expert would be copied through in bf16 under a quantization_config "
            f"that claims QAT parity. Declare the family's hub expert keys on its MoE family spec, or "
            f"export with --no-lowp_apply_moe_experts if training left the experts in bf16."
        )


def _fused_expert_axis(input_dir: str) -> int | None:
    """Contraction axis for this checkpoint's fused 3-D expert tensors, or ``None`` if unresolvable.

    Resolved through the family spec (``MoEFamilySpec.fused_contraction_axis``). ``None`` means the
    ``model_type`` matches no family; the caller then raises if a fused expert shows up rather than
    guessing an axis, which would corrupt every dequantized expert.
    """
    for model_type in detect_model_types(input_dir):
        if spec := find_moe_family_spec(model_type):
            return spec.fused_contraction_axis
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
        idx = decoder_layer_index(name)
        if idx is not None:
            indices.add(idx)
            roots.add(decoder_layer_stack_prefix(name))
    if len(roots) > 1:
        raise ValueError(
            f"--lowp_keep_first_blocks / --lowp_keep_last_blocks need one block numbering, but the "
            f"selected weights span {len(roots)}: {sorted(roots)}. Narrow --include/--exclude to the "
            f"text backbone, or export without a keep window."
        )
    return (max(indices) + 1) if indices else 0


def _stored_global_scale(name: str, weight: torch.Tensor, global_scale: torch.Tensor) -> torch.Tensor:
    """The ``weight_global_scale`` compressed-tensors stores for an nvfp4 weight, shape ``[1]`` as its
    ``generate_gparam`` writes it: the reciprocal of the quantizer's power-of-two multiplier, so exact.

    An all-zero weight's multiplier is far below fp32's reciprocal range; its codes and block scales are
    all zero, so it stores compressed-tensors' own value for that case, 1.0. An ``inf`` would reach
    vLLM's max over a fused layer's partition scales and zero every partition.
    """
    stored = global_scale.reciprocal().reshape(1).cpu()
    if not torch.isposinf(stored).any():
        return stored
    if weight.any():
        raise ValueError(
            f"{name!r} has a nonzero max |w| of {weight.float().abs().max().item():.3e}, below the range an "
            f"nvfp4 global scale can store (its reciprocal overflows fp32). Pass --exclude to keep it in "
            f"high precision."
        )
    return torch.ones(1, dtype=stored.dtype)


def _contraction_axis(
    name: str, tensor: torch.Tensor, fused_expert_axis: int | None, contraction_axis: int, input_dir: str
) -> int:
    """The axis ``tensor`` is block-scaled along: its family's fused-expert axis for a 3-D bank, else
    ``contraction_axis``; a 3-D bank of a family no spec claims raises rather than guessing."""
    if tensor.dim() != 3:
        return contraction_axis
    if fused_expert_axis is None:
        raise ValueError(
            f"{name!r} is a fused 3-D expert tensor, but this checkpoint's model_type "
            f"{detect_model_types(input_dir) or ['<missing>']} matches no MoE family spec, so its "
            f"contraction axis is unknown. Block-scaling the wrong axis corrupts every "
            f"dequantized expert. Declare the family's MoEFamilySpec (model_types and "
            f"fused_contraction_axis), or pass --exclude to skip "
            f"these weights."
        )
    return fused_expert_axis


def _block_scaled_tensors(name: str, tensor: torch.Tensor, quantized: BlockScaledTensor) -> dict[str, torch.Tensor]:
    """The compressed-tensors entries ``name`` is stored as: packed data, block scales, original shape and,
    for nvfp4, the global scale."""
    base = name[: -len(".weight")] if name.endswith(".weight") else name
    entries = {
        f"{base}.weight_packed": quantized.data.cpu().contiguous(),
        f"{base}.weight_scale": quantized.scales.cpu().contiguous(),
        f"{base}.weight_shape": torch.tensor(list(tensor.shape), dtype=torch.int64),
    }
    if quantized.global_scale is not None:
        entries[f"{base}.weight_global_scale"] = _stored_global_scale(name, tensor, quantized.global_scale)
    return entries


def _round_trip_relerr(name: str, tensor: torch.Tensor, quantized: BlockScaledTensor) -> float:
    """The relative error of ``quantized`` dequantized against ``tensor``; a non-finite one raises."""
    rel = (
        (dequantize(quantized).float().cpu() - tensor.float()).norm() / tensor.float().norm().clamp_min(1e-9)
    ).item()
    # max(x, nan) returns x, so a non-finite weight would read as a clean round-trip.
    if not math.isfinite(rel):
        raise ValueError(
            f"{name!r} round-trips to a non-finite value (relerr={rel}); the source weight "
            f"contains inf/NaN (max |w| = {tensor.float().abs().max().item()}). Fix the "
            f"checkpoint — quantizing it produces a corrupt export."
        )
    return rel


@dataclass(frozen=True)
class _ExportPlan:
    """What one export block-scales: the name filters, the kept blocks and the axis each weight contracts on."""

    fmt: str
    include: re.Pattern
    exclude: re.Pattern
    kept_blocks: frozenset[int]
    apply_dense_mlp: bool
    apply_moe_experts: bool
    fused_expert_axis: int | None
    contraction_axis: int
    input_dir: str


@dataclass
class _ExportTally:
    """What the shards written so far hold: the block-scaled weights and their axes, the tensors left as they
    were, the worst round-trip error and the output index."""

    quantized: list[str] = field(default_factory=list)
    weight_axes: dict[str, int] = field(default_factory=dict)
    skipped: list[str] = field(default_factory=list)
    copied: int = 0
    max_relerr: float = 0.0
    worst_relerr_name: str | None = None
    weight_map: dict[str, str] = field(default_factory=dict)
    total_size: int = 0


def _export_shard(entries, plan: _ExportPlan, tally: _ExportTally, *, verify: bool) -> dict[str, torch.Tensor]:
    """One input shard's tensors as written: every weight the plan selects block-scaled, the rest unchanged."""
    block = FORMAT_BLOCK_SIZE[plan.fmt]
    out_tensors: dict[str, torch.Tensor] = {}
    for _shard, reader, name in entries:
        t = reader.get_tensor(name)
        if not _should_quantize(
            name,
            t,
            plan.include,
            plan.exclude,
            kept_blocks=plan.kept_blocks,
            apply_dense_mlp=plan.apply_dense_mlp,
            apply_moe_experts=plan.apply_moe_experts,
        ):
            out_tensors[name] = t
            tally.copied += 1
            continue
        axis = _contraction_axis(name, t, plan.fused_expert_axis, plan.contraction_axis, plan.input_dir)
        if t.shape[axis] % block != 0:
            out_tensors[name] = t  # contraction axis not block-divisible; leave high precision
            tally.skipped.append(name)
            continue
        q = QUANTIZERS[plan.fmt](t.cuda().float() if torch.cuda.is_available() else t.float(), axis=axis)
        out_tensors.update(_block_scaled_tensors(name, t, q))
        tally.quantized.append(name)
        tally.weight_axes[name] = axis
        if verify:
            rel = _round_trip_relerr(name, t, q)
            if rel > tally.max_relerr:
                tally.max_relerr, tally.worst_relerr_name = rel, name
    return out_tensors


def _log_export(tally: _ExportTally, fmt: str, *, verify: bool) -> None:
    block = FORMAT_BLOCK_SIZE[fmt]
    logger.info(
        f"quantized {len(tally.quantized)} weights to {fmt}; copied {tally.copied} tensors unchanged; "
        f"skipped {len(tally.skipped)} (contraction axis not divisible by {block})."
    )
    if tally.skipped:
        logger.info(
            f"  left high-precision (not block-divisible): {tally.skipped[:8]}{' …' if len(tally.skipped) > 8 else ''}"
        )
    if verify:
        logger.info(
            f"round-trip dequant max relerr = {tally.max_relerr:.4f} (tolerance {_VERIFY_RELERR_TOL[fmt]}) — "
            f"OK (inherent format error)"
        )


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
    # The ``lowp_precision`` spelling that trains this storage format.
    lowp_precision = next(precision for precision, recipe in LOWP_RECIPES.items() if recipe.quant_format == fmt)
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
            f"Trained with mixed-precision QAT (lowp_precision: {lowp_precision}, "
            f"i.e. {fmt}); quantizing the bf16/fp32 master to this format reproduces the QAT "
            "forward exactly. Dequantize via halo.kernels.lowp.quantization.dequantize, whose nvfp4 "
            "global scale is the reciprocal of the stored weight_global_scale."
        ),
    }
    with open(os.path.join(output_dir, "quantization_config.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    # Stamp the scheme into config.json, the key readers consult; it also flips the toolkit's
    # native-quantized detection (EP lazy loading routes such checkpoints to from_pretrained, whose
    # coverage gate refuses the missing *.weight keys). vLLM and SGLang refuse the unknown
    # "block_scaled" method; plain transformers only warns, skips quantization and random-initializes
    # every quantized weight. A raw format name ("mxfp4", "nvfp4", "mxfp8") would instead select a
    # transformers or vLLM method that reads a different on-disk layout.
    config = read_checkpoint_config(output_dir, missing_ok=True)
    if config is not None:
        config["quantization_config"] = manifest
        write_config_json(output_dir, config)


def quantize_checkpoint(
    input_dir: str,
    output_dir: str,
    fmt: str,
    *,
    contraction_axis: int = -1,
    include: str = DEFAULT_INCLUDE,
    exclude: str = DEFAULT_EXCLUDE,
    apply_dense_mlp: bool = LowpSpec.apply_dense_mlp,
    apply_moe_experts: bool = LowpSpec.apply_moe_experts,
    keep_first_blocks: int = LowpSpec.keep_first_blocks,
    keep_last_blocks: int = LowpSpec.keep_last_blocks,
    verify: bool = False,
) -> None:
    inc, exc = re.compile(include), re.compile(exclude)
    reject_in_place_conversion(input_dir, output_dir)

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
    plan = _ExportPlan(
        fmt, inc, exc, kept_blocks, apply_dense_mlp, apply_moe_experts, fused_expert_axis, contraction_axis, input_dir
    )
    tally = _ExportTally()
    os.makedirs(output_dir, exist_ok=True)
    # The shared walk opens each shard once, in order, so grouping its entries by shard carries the
    # input's shard layout into the output: one written file per input file, under the same name.
    for shard, entries in groupby(iter_checkpoint_shard_entries(input_dir), key=itemgetter(0)):
        shard_name = os.path.basename(shard)
        out_tensors = _export_shard(entries, plan, tally, verify=verify)
        save_safetensors_file(out_tensors, os.path.join(output_dir, shard_name), metadata=SAFETENSORS_METADATA)
        for key, tensor in out_tensors.items():
            tally.weight_map[key] = shard_name
            tally.total_size += tensor.numel() * tensor.element_size()

    # Refused before the index and the config are written, so the rejected export cannot be loaded.
    if verify and tally.max_relerr > _VERIFY_RELERR_TOL[fmt]:
        raise ValueError(
            f"--verify: {tally.worst_relerr_name!r} round-trips with relerr {tally.max_relerr:.4f}, past the "
            f"{fmt} tolerance {_VERIFY_RELERR_TOL[fmt]}. Beyond the format's own error the cause is structural, "
            f"usually the wrong --contraction_axis; {output_dir} was stopped before its index and config."
        )

    # Anything but a single model.safetensors needs an index so the renamed weight_packed/scale keys
    # stay discoverable. Keyed on the emitted filenames: a lone model-00001-of-00001 input would
    # otherwise get neither. Either way the sweep drops what this run did not write, a stale index
    # included.
    if set(tally.weight_map.values()) != {SAFETENSORS_WEIGHTS_FILE}:
        write_merged_index(output_dir, tally.weight_map, {"total_size": tally.total_size})
    else:
        remove_stale_checkpoint_files(output_dir, keep={SAFETENSORS_WEIGHTS_FILE})

    scope = {
        "apply_dense_mlp": apply_dense_mlp,
        "apply_moe_experts": apply_moe_experts,
        "keep_first_blocks": keep_first_blocks,
        "keep_last_blocks": keep_last_blocks,
        "kept_blocks": sorted(kept_blocks),
    }
    block = FORMAT_BLOCK_SIZE[fmt]
    _write_manifest(input_dir, output_dir, fmt, block, contraction_axis, tally.quantized, tally.weight_axes, scope)
    _log_export(tally, fmt, verify=verify)
