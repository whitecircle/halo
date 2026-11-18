"""The compressed-tensors side of a low-precision export: the ``quantization_config`` compressed-tensors' loaders read
(vLLM's among them) and the ``re:`` targets that name the quantized modules. Nothing here reads a checkpoint or an
export plan: :mod:`halo.checkpoint.tools.lowp_export` passes the module names in."""

from __future__ import annotations

import re
from typing import Literal

from halo.kernels.lowp.quantization import FORMAT_BLOCK_SIZE
from halo.models.introspection.module_tree import decoder_layer_index, decoder_layer_stack_prefix
from halo.models.moe.family_specs import experts_container_attrs

# How the quantized tensors and ``config.json`` are written: Halo's own manifest and key names, or the
# compressed-tensors layout vLLM's loader reads.
ExportFormat = Literal["block_scaled", "compressed-tensors"]
BLOCK_SCALED: ExportFormat = "block_scaled"
COMPRESSED_TENSORS: ExportFormat = "compressed-tensors"
DEFAULT_EXPORT_FORMAT: ExportFormat = BLOCK_SCALED

# compressed-tensors' ``CompressionFormat`` per block-scaled format, and the ``QuantizationArgs`` its MXFP8, MXFP4 and
# NVFP4A16 presets serialize to (compressed-tensors 0.17.0). MX trains with both GEMM operands quantized, so its
# activations are declared dynamic; nvfp4 is weight-only, as an engine's W4A4 needs a calibrated static input scale.
COMPRESSED_TENSORS_FORMAT = {
    "mxfp8": "mxfp8-quantized",
    "mxfp4": "mxfp4-pack-quantized",
    "nvfp4": "nvfp4-pack-quantized",
}
COMPRESSED_TENSORS_ARGS = {
    "mxfp8": {"num_bits": 8, "strategy": "group", "scale_dtype": "torch.uint8"},
    "mxfp4": {"num_bits": 4, "strategy": "group", "scale_dtype": "torch.uint8"},
    "nvfp4": {"num_bits": 4, "strategy": "tensor_group", "scale_dtype": "torch.float8_e4m3fn"},
}
DYNAMIC_ACTIVATION_FORMATS = ("mxfp8", "mxfp4")
_TARGET_REGEX_PREFIX = "re:"


def compressed_tensors_targets(modules: list[str]) -> list[str]:
    """``re:`` targets naming the quantized ``modules``, one per projection spelling: block indices as an alternation
    (a ``lowp_keep_*`` window leaves blocks out) and expert indices as ``\\d+``. The decoder stack's own prefix is left
    open, since an engine roots a composite's text backbone under another name than the checkpoint does."""
    containers = experts_container_attrs()
    blocks_by_spelling: dict[tuple[str | None, ...], set[int]] = {}
    for module in modules:
        block = decoder_layer_index(module)
        tail = module if block is None else module[len(decoder_layer_stack_prefix(module)) :]
        segments = tail.split(".")
        spelling = tuple(
            None
            if block is not None and position == 1
            else r"\d+"
            if segment.isdigit() and position > 0 and segments[position - 1] in containers
            else re.escape(segment)
            for position, segment in enumerate(segments)
        )
        blocks_by_spelling.setdefault(spelling, set())
        if block is not None:
            blocks_by_spelling[spelling].add(block)
    targets = []
    for spelling, blocks in blocks_by_spelling.items():
        block_alternation = f"(?:{'|'.join(map(str, sorted(blocks)))})"
        body = r"\.".join(block_alternation if segment is None else segment for segment in spelling)
        targets.append(rf"{_TARGET_REGEX_PREFIX}(?:.*\.)?{body}$")
    return targets


def _matches_a_target(module: str, targets: list[str]) -> bool:
    """vLLM's rule for a ``re:`` target: ``re.match`` against the layer name."""
    return any(re.match(target[len(_TARGET_REGEX_PREFIX) :], module) for target in targets)


def reject_ambiguous_targets(targets: list[str], quantized: list[str], modules: list[str]) -> None:
    """Refuse targets that reach a module the export leaves in high precision, which an engine would then build
    quantized and fail to load (a vision tower spelling its MLP like the text backbone's)."""
    quantized_set = set(quantized)
    missed = [module for module in quantized if not _matches_a_target(module, targets)]
    if missed:
        raise RuntimeError(f"compressed-tensors targets do not name quantized module(s) {missed[:4]}: {targets}")
    reached = [module for module in modules if module not in quantized_set and _matches_a_target(module, targets)]
    if reached:
        raise ValueError(
            f"The compressed-tensors targets for this export also match {len(reached)} module(s) it leaves in high "
            f"precision (first few: {reached[:4]}), so an engine would build them quantized and fail to load their "
            f"weights. They share a ``layers.<N>...`` spelling with the quantized backbone under another root. "
            f"Export with --export_format {BLOCK_SCALED}."
        )


def compressed_tensors_config(fmt: str, targets: list[str]) -> dict:
    """The ``quantization_config`` compressed-tensors' loaders read: one config group over ``targets``, with the
    fields ``QuantizationConfig.model_dump_json`` writes."""

    def quantization_args(*, dynamic: bool) -> dict:
        return {
            "num_bits": COMPRESSED_TENSORS_ARGS[fmt]["num_bits"],
            "type": "float",
            "symmetric": True,
            "group_size": FORMAT_BLOCK_SIZE[fmt],
            "strategy": COMPRESSED_TENSORS_ARGS[fmt]["strategy"],
            "block_structure": None,
            "dynamic": dynamic,
            "actorder": None,
            "scale_dtype": COMPRESSED_TENSORS_ARGS[fmt]["scale_dtype"],
            "zp_dtype": None,
            "observer": None if dynamic else "memoryless_minmax",
            "observer_kwargs": {},
        }

    compression_format = COMPRESSED_TENSORS_FORMAT[fmt]
    return {
        "config_groups": {
            "group_0": {
                "targets": targets,
                "weights": quantization_args(dynamic=False),
                "input_activations": quantization_args(dynamic=True) if fmt in DYNAMIC_ACTIVATION_FORMATS else None,
                "output_activations": None,
                "format": compression_format,
            }
        },
        "quant_method": COMPRESSED_TENSORS,
        "kv_cache_scheme": None,
        "format": compression_format,
        "quantization_status": "compressed",
        "global_compression_ratio": None,
        "ignore": [],
    }
