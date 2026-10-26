"""Model weight files: save-dtype casts, the gathered-state normalization, the stale-file sweeps and the merged
index, the whole-dict and streaming shard writers, and the tied-embedding reconcile."""

from __future__ import annotations

import glob
import json
import logging
import os
from collections.abc import Callable, Mapping
from typing import Any

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from huggingface_hub.serialization._base import parse_size_to_int

from src.checkpoint.atomic import save_safetensors_file
from src.checkpoint.config_export import save_model_config
from src.checkpoint.format import (
    DEFAULT_MAX_SHARD_SIZE,
    LEGACY_WEIGHTS_FILE,
    SAFETENSORS_FAMILY_GLOB,
    SAFETENSORS_INDEX_FILE,
    SAFETENSORS_METADATA,
    SAFETENSORS_SHARD_PATTERN,
    SAFETENSORS_WEIGHTS_FILE,
    STREAM_PART_PREFIX,
    hf_shard_filename,
    index_declares_ep_shards,
    stream_part_filename,
)
from src.checkpoint.hub_layout import revert_to_load_layout
from src.checkpoint.reading import read_checkpoint_index
from src.models.config_access import ties_word_embeddings, untie_word_embeddings
from src.models.moe.balancing import balancing_param_keys, is_balancing_state_key
from src.models.param_roles import fp32_pinned_names, norm_param_keys
from src.models.unwrapping import strip_peft_adapter_segment

logger = logging.getLogger(__name__)

# Exports are saved bf16 even when training uses fp32 masters; training checkpoints keep the live dtype.
_SAVE_DTYPE = torch.bfloat16
# The tied pair, in the order reconcile_tie_word_embeddings reads them.
_TIE_KEY_SUFFIXES = ("lm_head.weight", "embed_tokens.weight")


def cast_to_save_dtype(t: torch.Tensor) -> torch.Tensor:
    """A floating tensor at the save dtype; the blanket cast under :func:`save_dtype_caster`'s keep-sets."""
    if torch.is_tensor(t) and t.is_floating_point() and t.dtype != _SAVE_DTYPE:
        return t.to(dtype=_SAVE_DTYPE)
    return t


def _as_live(_name: str, tensor: torch.Tensor) -> torch.Tensor:
    return tensor


def _caster_keeping(keep: Callable[[str], bool]) -> Callable[[str, torch.Tensor], torch.Tensor]:
    def cast(name: str, t: torch.Tensor) -> torch.Tensor:
        return t if keep(name) else cast_to_save_dtype(t)

    return cast


def save_dtype_caster(model: torch.nn.Module, *, keep_live_dtype: bool = False):
    """``cast(name, tensor)`` for a save that holds the live model: the save dtype except for the norm
    params, the router-balancing tensors and the fp32 pins (a key also matches with its PEFT adapter
    segment stripped). ``keep_live_dtype`` (a training checkpoint) casts nothing, for an exact resume.
    """
    if keep_live_dtype:
        return _as_live
    keep = norm_param_keys(model) | balancing_param_keys(model) | fp32_pinned_names(model, include_buffers=True)
    return _caster_keeping(lambda name: name in keep or strip_peft_adapter_segment(name) in keep)


def adapter_save_dtype_caster() -> Callable[[str, torch.Tensor], torch.Tensor]:
    """``cast(name, tensor)`` for an exported adapter file: a balancing tensor riding a ``modules_to_save``
    clone keeps its trained dtype, since a bf16-rounded bias flips near-tie top-k picks on serve."""
    return _caster_keeping(lambda name: is_balancing_state_key(strip_peft_adapter_segment(name)))


def normalize_gathered_state_dict(model: torch.nn.Module, state_dict: dict, *, keep_live_dtype: bool = False) -> dict:
    """A gathered state dict in its on-disk form: save-dtype cast (its keep-sets use live key spellings),
    then the hub expert layout."""
    cast = save_dtype_caster(model, keep_live_dtype=keep_live_dtype)
    return revert_to_load_layout(model, {key: cast(key, tensor) for key, tensor in state_dict.items()})


def remove_stale_checkpoint_files(output_dir: str, keep: set[str]) -> None:
    """Delete every ``model*.safetensors`` and index in ``output_dir`` outside ``keep``, since
    ``from_pretrained`` would load a leftover single file over the fresh index. A caller that only
    sometimes owns the directory must gate the call."""
    for stale in glob.glob(os.path.join(output_dir, SAFETENSORS_FAMILY_GLOB)) + glob.glob(
        os.path.join(output_dir, SAFETENSORS_INDEX_FILE)
    ):
        if os.path.basename(stale) not in keep:
            os.remove(stale)


def sweep_after_full_save(output_dir: str) -> None:
    """Remove the weight files a completed ``save_pretrained``-style write did not produce, keeping what is
    on disk after it (a consistent index and its shards, else a single file): a crashed save sweeps nothing.
    """
    index_path = os.path.join(output_dir, SAFETENSORS_INDEX_FILE)
    single_path = os.path.join(output_dir, SAFETENSORS_WEIGHTS_FILE)
    keep: set[str] = set()
    if os.path.isfile(index_path):
        try:
            index = read_checkpoint_index(output_dir)
        except (OSError, json.JSONDecodeError):
            index = {}
        # A per-rank EP/TP index is not a full save's product, so its shard list is no keep-set.
        shards = set() if index_declares_ep_shards(index) else set(index.get("weight_map", {}).values())
        # With both layouts present the newer is this write's; without the tie-break a single-file
        # writer (reset_sinks) loses to stale shards.
        newer_than_single = not os.path.isfile(single_path) or os.path.getmtime(index_path) >= os.path.getmtime(
            single_path
        )
        if shards and newer_than_single and all(os.path.isfile(os.path.join(output_dir, s)) for s in shards):
            keep = shards | {SAFETENSORS_INDEX_FILE}
    if not keep and os.path.isfile(single_path):
        keep = {SAFETENSORS_WEIGHTS_FILE}
    if keep:
        remove_stale_checkpoint_files(output_dir, keep=keep)


def write_merged_index(output_dir: str, weight_map: dict[str, str], metadata: Mapping[str, Any]) -> None:
    """Sweep the model files ``weight_map`` does not claim, then write the safetensors index; ``metadata``
    is its block verbatim, and must be a mapping."""
    if not isinstance(metadata, Mapping):
        raise TypeError(
            f"write_merged_index metadata must be a mapping written verbatim into the index's "
            f"metadata block (e.g. {{'total_size': N}}), got {type(metadata).__name__}: {metadata!r}"
        )
    remove_stale_checkpoint_files(output_dir, set(weight_map.values()) | {SAFETENSORS_INDEX_FILE})
    index = {"metadata": dict(metadata), "weight_map": weight_map}
    with open(os.path.join(output_dir, SAFETENSORS_INDEX_FILE), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)


def publish_streamed_parts(
    output_dir: str, final_names: dict[str, str], weight_map: dict[str, str], metadata: Mapping[str, Any]
) -> None:
    """Rename every streamed part (``{part name: final HF name}``) into place once all are on disk, then
    index a sharded result or sweep the leftovers beside a single ``model.safetensors``."""
    for part, final in final_names.items():
        os.replace(os.path.join(output_dir, part), os.path.join(output_dir, final))
    if len(final_names) > 1:
        write_merged_index(output_dir, weight_map, metadata)
    else:
        remove_stale_checkpoint_files(output_dir, set(final_names.values()))


def save_sharded_state_dict(
    state_dict: dict[str, torch.Tensor], output_dir: str, max_shard_size: str = DEFAULT_MAX_SHARD_SIZE
) -> None:
    """Write a gathered state dict as HF-standard sharded safetensors, split by HF's own splitter. Shards
    land as streamed parts renamed only once all are on disk, so a re-save dying midway cannot mix two
    saves under one index."""
    state_dict = {k: (v.contiguous() if torch.is_tensor(v) else v) for k, v in state_dict.items()}
    split = split_torch_state_dict_into_shards(
        state_dict, filename_pattern=SAFETENSORS_SHARD_PATTERN, max_shard_size=max_shard_size
    )
    final_names: dict[str, str] = {}
    for part_index, (filename, keys) in enumerate(split.filename_to_tensors.items(), start=1):
        part = stream_part_filename(STREAM_PART_PREFIX, part_index)
        save_safetensors_file(
            {k: state_dict[k] for k in keys}, os.path.join(output_dir, part), metadata=SAFETENSORS_METADATA
        )
        final_names[part] = filename
    publish_streamed_parts(output_dir, final_names, split.tensor_to_filename, split.metadata)


def _unique_key_by_suffix(state_dict: dict, suffix: str) -> str | None:
    """The one key equal to ``suffix`` or ending in ``.{suffix}``; ``None`` when absent or ambiguous."""
    matches = [k for k in state_dict if k == suffix or k.endswith(f".{suffix}")]
    if len(matches) > 1:
        logger.warning(
            f"reconcile_tie_word_embeddings: multiple state-dict keys match '{suffix}' "
            f"({matches[:4]}); skipping the tie-consistency check for this save."
        )
        return None
    return matches[0] if matches else None


def is_tie_reconcile_key(name: str) -> bool:
    """Whether ``name`` is one of the pair :func:`reconcile_tie_word_embeddings` compares, which a
    streamed save keeps aside as it passes."""
    return any(name == suffix or name.endswith(f".{suffix}") for suffix in _TIE_KEY_SUFFIXES)


def reconcile_tie_word_embeddings(model, state_dict: dict) -> None:
    """Clear ``tie_word_embeddings`` on every config level declaring it when the saved lm_head and embedding
    diverged in training (FSDP2 shards the pair independently and FLCE trains lm_head directly)."""
    config = getattr(model, "config", None)
    if config is None or not ties_word_embeddings(config):
        return
    lm_key = _unique_key_by_suffix(state_dict, _TIE_KEY_SUFFIXES[0])
    emb_key = _unique_key_by_suffix(state_dict, _TIE_KEY_SUFFIXES[1])
    if lm_key is None or emb_key is None:
        return
    lm, emb = state_dict[lm_key], state_dict[emb_key]
    if lm.shape != emb.shape or not torch.equal(lm, emb):
        untie_word_embeddings(config)
        logger.warning(
            f"{lm_key} and {emb_key} diverged during training; "
            f"saving config with tie_word_embeddings=False for checkpoint consistency."
        )


def write_gathered_checkpoint(
    model: torch.nn.Module, state_dict: dict, output_dir: str, max_shard_size: str = DEFAULT_MAX_SHARD_SIZE
) -> None:
    """Write a whole gathered state dict as HF-standard safetensors, with its config. Save rank only; the
    ``.bin`` fallback is this path's alone, since only it still holds the whole dict."""
    reconcile_tie_word_embeddings(model, state_dict)
    save_model_config(model, output_dir)
    # EP-gathered dicts never arrive here: their saver's gather already emits hub-layout expert keys.
    state_dict = normalize_gathered_state_dict(model, state_dict)
    try:
        save_sharded_state_dict(state_dict, output_dir, max_shard_size=max_shard_size)
    except Exception as e:
        logger.warning(f"sharded safetensors save failed: {e}, using pytorch format")
        # The .bin lands before the sweep: resume prefers an index over it, but sweeping first would
        # let a second failure destroy the only checkpoint the directory held.
        torch.save(state_dict, os.path.join(output_dir, LEGACY_WEIGHTS_FILE))
        try:
            remove_stale_checkpoint_files(output_dir, keep=set())
        except OSError as sweep_error:
            logger.warning(
                f"stale safetensors sweep after the .bin fallback failed: {sweep_error} — "
                f"resume prefers an index over the .bin, so this directory may resume the OLD weights"
            )


class StreamingShardWriter:
    """Accumulate tensors and flush them to numbered safetensors parts under one prefix, whenever the next
    one would overflow ``max_shard_size``. ``enabled=False`` makes every method a no-op for non-writer
    ranks, which still enter each collective gather."""

    def __init__(
        self, output_dir: str, max_shard_size: str, *, enabled: bool, filename_prefix: str = STREAM_PART_PREFIX
    ):
        self.output_dir = output_dir
        self.filename_prefix = filename_prefix
        self.max_bytes = parse_size_to_int(max_shard_size)
        self.enabled = enabled
        self._pending: dict[str, torch.Tensor] = {}
        self._pending_bytes = 0
        self._part = 0
        self._weight_map: dict[str, str] = {}
        self._part_names: list[str] = []
        self._total_bytes = 0

    def add(self, key: str, tensor: torch.Tensor) -> None:
        """Stage one tensor, flushing first if it would overflow the pending part (HF's rule). A key stages
        once, since the first may already sit in a flushed part."""
        if not self.enabled:
            return
        if key in self._weight_map or key in self._pending:
            raise RuntimeError(
                f"{key!r} was staged twice for the checkpoint in {self.output_dir}: a streamed part "
                f"already claims it, so the second tensor cannot replace the first. Two chunks of "
                f"this save name the same tensor (a hand-sliced param emitted by the parameter walk "
                f"AND by its own gather is the known shape) — the writer refuses rather than write "
                f"the key twice at two shapes."
            )
        nbytes = tensor.nbytes
        if self._pending and self._pending_bytes + nbytes > self.max_bytes:
            self._flush()
        self._pending[key] = tensor.contiguous()
        self._pending_bytes += nbytes

    def close(self) -> tuple[dict[str, str], int]:
        """Flush the remainder and return ``(weight_map fragment, bytes written)`` for a multi-writer
        save to merge; call it before the index exchange."""
        if not self.enabled:
            return {}, 0
        self._flush()
        return self._weight_map, self._total_bytes

    def close_as_hf_checkpoint(self) -> int:
        """Flush, rename the parts to HF's names and write the index; return the bytes written. Valid only
        when this writer owns every ``model*.safetensors`` in the directory: it deletes the rest."""
        if not self.enabled:
            return 0
        self._flush()
        if not self._part:
            raise RuntimeError(
                f"Refusing to finalize an empty checkpoint in {self.output_dir}: this writer is the "
                f"designated save rank but no tensors were staged, so there is nothing to name and "
                f"the existing files there would be deleted for nothing."
            )
        renamed = {
            part_name: hf_shard_filename(i, self._part) for i, part_name in enumerate(self._part_names, start=1)
        }
        weight_map = {key: renamed[part_name] for key, part_name in self._weight_map.items()}
        publish_streamed_parts(self.output_dir, renamed, weight_map, {"total_size": self._total_bytes})
        return self._total_bytes

    def _flush(self) -> None:
        if not self._pending:
            return
        self._part += 1
        name = stream_part_filename(self.filename_prefix, self._part)
        save_safetensors_file(self._pending, os.path.join(self.output_dir, name), metadata=SAFETENSORS_METADATA)
        self._weight_map.update(dict.fromkeys(self._pending, name))
        self._part_names.append(name)
        self._total_bytes += self._pending_bytes
        self._pending = {}
        self._pending_bytes = 0


def merge_shard_index(gathered: list[tuple[dict[str, str], int] | None]) -> tuple[dict[str, str], int]:
    """Fold the gathered fragments into one ``(weight_map, total_size)``, rejecting a collision.

    The combine :func:`~src.runtime.collectives.combine_across_ranks` runs on rank 0, which makes the
    raise world-uniform. A key claimed by two different files would resolve to whichever fragment
    merged last and count its bytes twice, producing a checkpoint that loads while holding the wrong
    writer's tensor; both callers are key-disjoint by construction, so a second claim is a defect.

    Bytes are counted once per file: on per-node storage each node's writer emits the same filenames.
    """
    weight_map: dict[str, str] = {}
    counted_files: set[str] = set()
    total_size = 0
    for entry in gathered:
        if not entry:
            continue
        fragment, fragment_bytes = entry
        for key, shard in fragment.items():
            if key in weight_map and weight_map[key] != shard:
                raise RuntimeError(
                    f"checkpoint shards disagree on {key!r}: claimed by both {weight_map[key]!r} and "
                    f"{shard!r}. Every key belongs to exactly one writer, so two claims mean the "
                    f"shard split or the global-name map is wrong."
                )
            weight_map[key] = shard
        files = set(fragment.values())
        if not files & counted_files:
            total_size += fragment_bytes
            counted_files |= files
    return weight_map, total_size
