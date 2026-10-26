"""Un-fuse a checkpoint's fused MoE expert tensors into the per-expert hub layout its family spec declares,
streamed into a new sharded checkpoint; weights are unchanged (no transpose, no re-quantization)."""

from __future__ import annotations

import logging
import os
from functools import lru_cache

import torch
from transformers.utils import CONFIG_NAME

from halo.checkpoint.format import DEFAULT_MAX_SHARD_SIZE
from halo.checkpoint.offline_export import (
    copy_checkpoint_aux_files,
    preflight_resource_warning,
    reject_in_place_conversion,
)
from halo.checkpoint.reading import (
    checkpoint_shard_files,
    detect_model_type,
    header_nbytes,
    iter_checkpoint_shard_entries,
    iter_checkpoint_tensors,
    read_checkpoint_config,
)
from halo.checkpoint.writing import StreamingShardWriter
from halo.models.moe.expert_layout import HF_FUSED_EXPERT_KEYS, iter_per_expert_split
from halo.models.moe.family_specs import (
    experts_container_attrs,
    find_moe_family_spec,
    per_expert_hub_model_types,
    per_expert_layouts,
)

logger = logging.getLogger(__name__)


def resolve_per_expert_layout(ckpt_dir: str) -> tuple[str, str, str]:
    """The ``(gate, up, down)`` per-expert projection names ``ckpt_dir``'s family stores experts under.

    The family is resolved the way the sharded merge resolves it, through the specs' declared
    ``model_types`` rather than a key-spelling heuristic: the fused keys this script rewrites
    are shared across the roster at identical shapes, so an unmatched checkpoint would split and be
    written under names its own loader never reads, and transformers would then re-initialize the
    whole expert bank with a warning rather than an error.
    """
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
    """(hidden_size, moe_intermediate_size, model_type) from config.json, VLM ``text_config`` fallback.

    The family comes from :func:`detect_model_type`, which the family gates here also use (text_config
    fallback included), rather than a second parse of the same key.
    """
    cfg = read_checkpoint_config(ckpt_dir)
    tc = cfg.get("text_config", {})
    hidden = cfg.get("hidden_size") or tc.get("hidden_size")
    # moe_intermediate_size names the expert FFN width on the contiguous fused-GLU families; fall back
    # to intermediate_size for a config that declares only the dense width.
    inter = (
        cfg.get("moe_intermediate_size")
        or tc.get("moe_intermediate_size")
        or cfg.get("intermediate_size")
        or tc.get("intermediate_size")
    )
    return hidden, inter, detect_model_type(ckpt_dir)


@lru_cache(maxsize=1)
def _fused_key_kinds() -> tuple[tuple[str, str, bool, bool], ...]:
    """``(key suffix, expert container attr, splits_gate_up, is_bias)`` per fused expert tensor on the roster.

    Cached: the roster is class-declared and fixed for the process, while ``_split_fused_key`` asks
    for it once per tensor key in the checkpoint.

    Names come from ``HF_FUSED_EXPERT_KEYS`` (the same tuple the sharded merge accepts), so a family
    adding a fused tensor needs no second list here. A fused key that also names a per-expert
    projection (a spec's ``per_expert_keys``) is the down matrix, since a fused gate+up name
    never appears per-expert; everything else is split in half. Each suffix is matched at the end of a
    key, so a ``…_bias`` entry and the weight sharing its prefix cannot shadow each other and the
    generation order does not matter.
    """
    per_expert_names = {name for layout in per_expert_layouts() for name in layout}
    kinds = []
    # Every declared container spelling, not just ``experts``: a family nesting them under
    # ``routed_experts`` would match nothing and be reported as already per-expert.
    for container in experts_container_attrs():
        for name in HF_FUSED_EXPERT_KEYS:
            is_bias = name.endswith("_bias")
            stem = name.removesuffix("_bias")
            kinds.append((f".{container}.{name}", container, stem not in per_expert_names, is_bias))
    if not any(not splits for _suffix, _container, splits, _bias in kinds):
        # Everything classified as a gate/up pair means the declarations drifted apart, and splitting
        # the down matrix in half would halve every expert's output projection.
        raise RuntimeError(
            f"No fused expert key {[k for k, _c, _s, _b in kinds]} matches a declared per-expert projection "
            f"{sorted(per_expert_names)}, so none can be identified as the single down matrix. "
            f"HF_FUSED_EXPERT_KEYS and the specs' per_expert_keys have drifted apart."
        )
    return tuple(kinds)


def _split_fused_key(key: str) -> tuple[str, bool, bool] | None:
    """If ``key`` is a fused expert tensor, return (layer_prefix, splits_gate_up, is_bias); else None.

    layer_prefix is everything up to and including the expert container, so per-expert keys become
    ``{prefix}.{i}.<proj>.{weight,bias}``.
    """
    for suffix, container, splits_gate_up, is_bias in _fused_key_kinds():
        if key.endswith(suffix):
            return key[: -len(suffix)] + f".{container}", splits_gate_up, is_bias
    return None


def _assert_contiguous_gate_up(shape: torch.Size, hidden: int | None, inter: int | None, model_type: str | None):
    """Check that a fused ``gate_up_proj`` really is the contiguous-halves ``[E, 2I, H]`` layout.

    Checked on the tensor rather than the config's model_type: a mislabelled checkpoint, or a config
    whose ``moe_intermediate_size`` does not describe the weights, would otherwise be split at the
    wrong offset and corrupted.
    """
    if hidden is None or inter is None:
        raise ValueError(
            "config.json lacks hidden_size / moe_intermediate_size (checked text_config too); "
            "cannot verify the expert layout. Add them or convert with a family-specific script."
        )
    _, a, b = shape
    if a == 2 * inter and b == hidden:
        return  # contiguous halves [E, 2I, H]: supported
    if b == 2 * inter and a == hidden:
        raise NotImplementedError(
            f"gate_up_proj {tuple(shape)} is the interleaved [E, H, 2M] layout, not the contiguous halves "
            f"model_type={model_type} is registered as storing. Interleaved is already the layout serving "
            f"engines expect, and a contiguous split would corrupt it. Aborting."
        )
    raise ValueError(
        f"Unrecognized fused gate_up_proj layout {tuple(shape)} for hidden={hidden}, "
        f"moe_intermediate_size={inter} (model_type={model_type})."
    )


def _has_fused_expert_keys(ckpt_dir: str) -> bool:
    """Whether any shard carries a fused expert tensor, from the safetensors headers alone.

    Distinguishes an already-per-expert checkpoint from one that needs converting, without reading
    tensor data.
    """
    return any(_split_fused_key(key) is not None for _shard, _reader, key in iter_checkpoint_shard_entries(ckpt_dir))


def _validate_fused_layouts(in_dir: str) -> None:
    """Assert every fused ``gate_up`` tensor really is the contiguous-halves layout, headers only.

    Ahead of the first output write, because the split below is streamed: a shape refused halfway
    through would leave a partial directory that a pipeline could mistake for a finished one.
    """
    hidden, inter, model_type = _moe_dims(in_dir)
    for _shard, reader, key in iter_checkpoint_shard_entries(in_dir):
        parsed = _split_fused_key(key)
        if parsed is None:
            continue
        _prefix, splits_gate_up, is_bias = parsed
        if splits_gate_up and not is_bias:
            _assert_contiguous_gate_up(torch.Size(reader.get_slice(key).get_shape()), hidden, inter, model_type)


def _iter_unfused(in_dir: str, layout: tuple[str, str, str] | None):
    """Stream ``(key, tensor)`` for ``in_dir``, splitting fused expert tensors under ``layout``.

    ``layout=None`` is the already-per-expert checkpoint: every tensor passes through verbatim.
    Streamed rather than accumulated, so peak host memory is one fused expert bank plus the writer's
    pending shard instead of the whole model.
    """
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
    """Split fused expert tensors into per-expert keys under the family's own (gate, up, down) names.

    The names come from the checkpoint's own family spec (``expert_layout.per_expert_keys``), so this
    script never spells a projection name itself.
    """
    reject_in_place_conversion(in_dir, out_dir)

    # The family gate runs last: a per-rank sharded input has to be diagnosed as one, and an
    # already-per-expert checkpoint needs no layout, so resolving the family first would refuse a
    # no-op.
    layout = None
    if _has_fused_expert_keys(in_dir):
        layout = resolve_per_expert_layout(in_dir)
        _validate_fused_layouts(in_dir)
    else:
        logger.info(f"No fused expert tensors under a rewritable container spelling in {in_dir} — copying through.")

    input_shards = checkpoint_shard_files(in_dir)
    # Peak RAM is about the largest fused bank plus the per-expert copies split out of it, on top of
    # the writer's pending output shard. On disk the per-expert artifact is about the input's size.
    largest_tensor = max(
        (header_nbytes(reader.get_slice(key)) for _shard, reader, key in iter_checkpoint_shard_entries(in_dir)),
        default=0,
    )
    writer = StreamingShardWriter(out_dir, max_shard_size, enabled=True)
    preflight_resource_warning(
        "unfuse_moe_experts",
        out_dir,
        disk_bytes=sum(os.path.getsize(shard) for shard in input_shards),
        ram_bytes=2 * largest_tensor + writer.max_bytes,
    )

    # Created past every gate above, so a refusal leaves no directory behind.
    os.makedirs(out_dir, exist_ok=True)
    for key, tensor in _iter_unfused(in_dir, layout):
        writer.add(key, tensor)
    writer.close_as_hf_checkpoint()

    # Non-weight files carry over (config, tokenizer, templates, …), the README.md card gaining the
    # Halo Hub tag; the old safetensors shards and index are not copied, since the writer rewrites them.
    copy_checkpoint_aux_files(in_dir, out_dir)
    logger.info(f"✓ Wrote per-expert checkpoint to {out_dir}")
