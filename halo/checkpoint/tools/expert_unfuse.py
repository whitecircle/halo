"""Un-fuse a checkpoint's fused MoE expert tensors into the per-expert hub layout its family spec declares,
streamed into a new sharded checkpoint; weights are unchanged (no transpose, no re-quantization)."""

from __future__ import annotations

import logging
import os
from functools import lru_cache

import torch
from transformers.utils import CONFIG_NAME

from halo.checkpoint.format import DEFAULT_MAX_SHARD_SIZE
from halo.checkpoint.reading import (
    checkpoint_shard_files,
    detect_model_type,
    header_nbytes,
    iter_checkpoint_shard_entries,
    iter_checkpoint_tensors,
    read_checkpoint_config,
)
from halo.checkpoint.tools.offline_export import (
    copy_checkpoint_aux_files,
    preflight_resource_warning,
    reject_in_place_conversion,
)
from halo.checkpoint.writing import StreamingShardWriter
from halo.models.moe.config_fields import EXPERT_FFN_WIDTH_FIELDS
from halo.models.moe.expert_layout import HF_FUSED_EXPERT_KEYS, iter_per_expert_split
from halo.models.moe.family_specs import (
    experts_container_attrs,
    find_moe_family_spec,
    per_expert_hub_model_types,
)

logger = logging.getLogger(__name__)


def resolve_per_expert_layout(ckpt_dir: str) -> tuple[str, str, str]:
    """The ``(gate, up, down)`` names ``ckpt_dir``'s family stores experts under, via the specs' declared
    ``model_types``. Fused keys are shared across the roster at identical shapes, so a key-spelling guess would write
    names its loader never reads, and transformers would re-initialize the whole expert bank with a warning."""
    if not os.path.isfile(os.path.join(ckpt_dir, CONFIG_NAME)):
        raise ValueError(
            f"{ckpt_dir} has no {CONFIG_NAME}, so the checkpoint's model_type — and with it the per-expert "
            f"names its loader reads — cannot be resolved. Convert a complete checkpoint directory."
        )
    model_type = detect_model_type(ckpt_dir)
    spec = find_moe_family_spec(model_type)
    supported = ", ".join(per_expert_hub_model_types())
    if spec is None:
        raise ValueError(
            f"{ckpt_dir} has model_type {model_type!r}, which no MoE family spec claims, so the "
            f"per-expert names its loader reads are unknown. Convertible model_types: {supported}."
        )
    layout = spec.expert_layout.per_expert_keys
    if layout is None:
        raise ValueError(
            f"{ckpt_dir} has model_type {model_type!r}, whose checkpoint does not "
            f"store one tensor per expert at all: this script can only rewrite experts to "
            f"'experts.{{i}}.<gate|up|down>.weight', and that family keeps them in a layout no such "
            f"triple names (contiguous fused halves, interleaved fused, or a module nested per expert). "
            f"Every key emitted here would be one its loader never reads. "
            f"Convertible model_types: {supported}. The fused layouts are already what "
            f"from_pretrained and the serving engines read — serve the checkpoint as saved."
        )
    return layout


def _moe_dims(ckpt_dir: str) -> tuple[int | None, int | None, str]:
    """``(hidden_size, expert FFN width, model_type)`` from config.json, VLM ``text_config`` fallback."""
    cfg = read_checkpoint_config(ckpt_dir)
    tc = cfg.get("text_config", {})
    hidden = cfg.get("hidden_size") or tc.get("hidden_size")
    inter = next(
        (value for field in EXPERT_FFN_WIDTH_FIELDS for source in (cfg, tc) if (value := source.get(field))), None
    )
    return hidden, inter, detect_model_type(ckpt_dir)


@lru_cache(maxsize=1)
def _fused_key_kinds() -> tuple[tuple[str, str, bool, bool], ...]:
    """``(key suffix, container attr, splits_gate_up, is_bias)`` per fused expert tensor on the roster; cached, as
    ``_split_fused_key`` asks once per key. Only ``gate_up_proj`` holds two projections to split in half."""
    kinds = []
    # Every declared container spelling, or a family nesting under ``routed_experts`` would read as already per-expert.
    for container in experts_container_attrs():
        for name in HF_FUSED_EXPERT_KEYS:
            is_bias = name.endswith("_bias")
            kinds.append((f".{container}.{name}", container, name.removesuffix("_bias") == "gate_up_proj", is_bias))
    return tuple(kinds)


def _split_fused_key(key: str) -> tuple[str, bool, bool] | None:
    """``(layer_prefix, splits_gate_up, is_bias)`` for a fused expert tensor, else None; the prefix ends at the
    container."""
    for suffix, container, splits_gate_up, is_bias in _fused_key_kinds():
        if key.endswith(suffix):
            return key[: -len(suffix)] + f".{container}", splits_gate_up, is_bias
    return None


def _assert_contiguous_gate_up(shape: torch.Size, hidden: int | None, inter: int | None, model_type: str | None):
    """Check a fused ``gate_up_proj`` is the contiguous ``[E, 2I, H]`` layout, on the tensor, since a mislabelled
    checkpoint or a config whose width does not describe the weights would be split at the wrong offset."""
    if hidden is None or inter is None:
        raise ValueError(
            f"config.json lacks hidden_size or any of {EXPERT_FFN_WIDTH_FIELDS} (checked text_config too); "
            "cannot verify the expert layout. Add them or convert with a family-specific script."
        )
    _, a, b = shape
    if a == 2 * inter and b == hidden:
        return
    if b == 2 * inter and a == hidden:
        raise NotImplementedError(
            f"gate_up_proj {tuple(shape)} is the interleaved [E, H, 2M] layout, not the contiguous halves "
            f"model_type={model_type} is registered as storing. Interleaved is already the layout serving "
            f"engines expect, and a contiguous split would corrupt it. Aborting."
        )
    raise ValueError(
        f"Unrecognized fused gate_up_proj layout {tuple(shape)} for hidden={hidden}, "
        f"expert FFN width={inter} (model_type={model_type})."
    )


def _has_fused_expert_keys(ckpt_dir: str) -> bool:
    """Whether any shard carries a fused expert tensor, from the safetensors headers alone."""
    return any(_split_fused_key(key) is not None for _shard, _reader, key in iter_checkpoint_shard_entries(ckpt_dir))


def _validate_fused_layouts(in_dir: str) -> None:
    """Assert every fused ``gate_up`` is contiguous, headers only, ahead of the first write: the split streams, so a
    refusal halfway would leave a partial directory a pipeline could mistake for a finished one."""
    hidden, inter, model_type = _moe_dims(in_dir)
    for _shard, reader, key in iter_checkpoint_shard_entries(in_dir):
        parsed = _split_fused_key(key)
        if parsed is None:
            continue
        _prefix, splits_gate_up, is_bias = parsed
        if splits_gate_up and not is_bias:
            _assert_contiguous_gate_up(torch.Size(reader.get_slice(key).get_shape()), hidden, inter, model_type)


def _iter_unfused(in_dir: str, layout: tuple[str, str, str] | None):
    """Stream ``(key, tensor)``, splitting fused experts under ``layout``; ``None`` passes everything through. Peak
    host memory is one fused bank plus the writer's staged parts."""
    if layout is None:
        yield from iter_checkpoint_tensors(in_dir)
        return

    gate_name, up_name, down_name = layout
    fused_layers = 0
    for key, tensor in iter_checkpoint_tensors(in_dir):
        parsed = _split_fused_key(key)
        if parsed is None:
            yield key, tensor
            continue
        prefix, splits_gate_up, is_bias = parsed
        param = "bias" if is_bias else "weight"
        if splits_gate_up and not is_bias:
            fused_layers += 1
        # gate/up halves of weight [E, 2I, H] or bias [E, 2I]; the down weight [E, H, I] or bias [E, H] whole.
        names = (gate_name, up_name) if splits_gate_up else (down_name,)
        for expert_key, part in iter_per_expert_split([(tensor, names)]):
            yield f"{prefix}.{expert_key}.{param}", part

    logger.info(f"Un-fused {fused_layers} MoE layer(s) into per-expert tensors.")


def unfuse_checkpoint(in_dir: str, out_dir: str, max_shard_size: str = DEFAULT_MAX_SHARD_SIZE) -> None:
    """Split fused expert tensors into per-expert keys under the names in the checkpoint's family spec."""
    reject_in_place_conversion(in_dir, out_dir)

    # The family gate runs last: a per-rank input must be diagnosed as one; a per-expert checkpoint needs no layout.
    layout = None
    if _has_fused_expert_keys(in_dir):
        layout = resolve_per_expert_layout(in_dir)
        _validate_fused_layouts(in_dir)
    else:
        logger.info(f"No fused expert tensors under a rewritable container spelling in {in_dir} — copying through.")

    input_shards = checkpoint_shard_files(in_dir)
    # Peak RAM is about the largest bank plus its split copies, on top of staged parts; on disk about the input's size.
    largest_tensor = max(
        (header_nbytes(reader.get_slice(key)) for _shard, reader, key in iter_checkpoint_shard_entries(in_dir)),
        default=0,
    )
    with StreamingShardWriter(out_dir, max_shard_size, enabled=True) as writer:
        preflight_resource_warning(
            "unfuse_moe_experts",
            out_dir,
            disk_bytes=sum(os.path.getsize(shard) for shard in input_shards),
            ram_bytes=2 * largest_tensor + writer.max_resident_bytes,
        )

        # Created past every gate above, so a refusal leaves no directory behind.
        os.makedirs(out_dir, exist_ok=True)
        for key, tensor in _iter_unfused(in_dir, layout):
            writer.add(key, tensor)
        writer.close_as_hf_checkpoint()

    # The writer rewrote shards and index; non-weight files carry over, the README.md card gaining the Halo Hub tag.
    copy_checkpoint_aux_files(in_dir, out_dir)
    logger.info(f"✓ Wrote per-expert checkpoint to {out_dir}")
