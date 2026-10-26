"""Model weight files: save-dtype casts, the gathered-state normalization, the stale-file sweeps and the merged
index, the whole-dict and streaming shard writers, and the tied-embedding reconcile.

safetensors has no append handle (``save_file`` takes a whole dict), so bounding the streaming writer's host
memory and honouring ``save_max_shard_size`` are the same mechanism: flush whenever the pending batch would
overflow. A multi-writer caller (the PP save, one rank per stage) hands its weight-map fragment back through
:meth:`StreamingShardWriter.close` to be merged, because no writer knows the global part count; a single writer
knows it at the last flush, so :meth:`StreamingShardWriter.close_as_hf_checkpoint` renames its parts to HF's own
names and writes the index itself.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from collections.abc import Mapping
from typing import Any

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from huggingface_hub.serialization._base import parse_size_to_int
from safetensors.torch import save_file

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
    read_checkpoint_index,
)
from src.checkpoint.hub_layout import revert_to_load_layout
from src.models.moe_balancing import balancing_param_keys
from src.models.param_roles import fp32_pinned_state_keys, norm_param_keys
from src.models.unwrapping import strip_peft_adapter_segment

logger = logging.getLogger(__name__)

# Exports are saved bf16 even when training uses fp32 master weights; training checkpoints keep the
# live dtype (:func:`save_dtype_caster`).
_SAVE_DTYPE = torch.bfloat16
# The tied pair, in the order reconcile_tie_word_embeddings reads them.
_TIE_KEY_SUFFIXES = ("lm_head.weight", "embed_tokens.weight")


def cast_to_save_dtype(t: torch.Tensor) -> torch.Tensor:
    """Cast a tensor to the distributed save dtype (BF16) if it's floating point.

    Blanket cast — save paths that hold the live model use :func:`save_dtype_caster` instead, so
    normalization params keep their trained dtype.
    """
    if torch.is_tensor(t) and t.is_floating_point() and t.dtype != _SAVE_DTYPE:
        return t.to(dtype=_SAVE_DTYPE)
    return t


def _as_live(_name: str, tensor: torch.Tensor) -> torch.Tensor:
    return tensor


def save_dtype_caster(model: torch.nn.Module, *, keep_live_dtype: bool = False):
    """``cast(name, tensor)`` for checkpoint saves that hold the live model.

    Floating tensors go to the save dtype except three tree-derived keep-sets that hold their trained
    dtype — the normalization params, the live router-balancing tensors (hub-respelled) and the
    family's fp32 pins, buffers included — so a direct EP/TP save of an fp32-master run matches its
    merged-shards save and the export quantizes neither the balancing state nor a family's declared
    fp32 modules away.

    ``keep_live_dtype`` (a training checkpoint) casts nothing: every tensor is written at the dtype the
    gather produced, which is the live one, so fp32 masters (``fp32_router``, ``fp32_experts``,
    ``fp32_non_ep_params``) reach disk unrounded for a resume to read back exactly. Decided on the
    tensor, not its name, since a gathered expert's hub key need not name any live parameter.

    Keys also match with their PEFT adapter segment stripped: the EP gather feeds this pre-remap
    keys, where a ``modules_to_save`` router spells its bias ``router.modules_to_save.default.bias``.
    """
    if keep_live_dtype:
        return _as_live
    keep = norm_param_keys(model) | balancing_param_keys(model) | fp32_pinned_state_keys(model)

    def cast(name: str, t: torch.Tensor) -> torch.Tensor:
        return t if name in keep or strip_peft_adapter_segment(name) in keep else cast_to_save_dtype(t)

    return cast


def normalize_gathered_state_dict(model: torch.nn.Module, state_dict: dict, *, keep_live_dtype: bool = False) -> dict:
    """Bring a gathered state dict to its on-disk form: save-dtype cast, then hub expert layout.

    The order is load-bearing: the caster's keep-set is derived from the module tree, so it uses the
    live key spelling and must run before the revert respells the expert keys. Shared by the FSDP2/CP
    and TP gathered writers, whose artifacts must be byte-identical for the same model.
    ``keep_live_dtype`` is :func:`save_dtype_caster`'s.
    """
    cast = save_dtype_caster(model, keep_live_dtype=keep_live_dtype)
    return revert_to_load_layout(model, {key: cast(key, tensor) for key, tensor in state_dict.items()})


def remove_stale_checkpoint_files(output_dir: str, keep: set[str]) -> None:
    """Delete every ``model*.safetensors``/index in ``output_dir`` this save did not write.

    ``from_pretrained`` prefers a single ``model.safetensors`` over the index, so a previous save's
    leftover — a single file where this one sharded, or a higher shard count — would be loaded
    instead of what was just written. One sweep for every writer that owns a model directory, since
    it ``os.remove``s checkpoint files and the callers must not drift apart. A caller that only
    sometimes owns the directory (``reset_sinks`` in-place, an unmerged-PEFT save) must gate the
    call, never pass ``keep=set()`` unconditionally.
    """
    for stale in glob.glob(os.path.join(output_dir, SAFETENSORS_FAMILY_GLOB)) + glob.glob(
        os.path.join(output_dir, SAFETENSORS_INDEX_FILE)
    ):
        if os.path.basename(stale) not in keep:
            os.remove(stale)


def sweep_after_full_save(output_dir: str) -> None:
    """Remove weight files a *completed* ``save_pretrained``-style write did not produce.

    Failure-safe by construction: it runs AFTER the save and derives the keep-set from what is on
    disk NOW (a consistent index keeps its shards, else a single ``model.safetensors`` keeps itself),
    so a save that crashed mid-way — torn index included — sweeps nothing and leaves the directory
    intact. Sweeping first would let a failed save destroy the good checkpoint it held.
    """
    index_path = os.path.join(output_dir, SAFETENSORS_INDEX_FILE)
    single_path = os.path.join(output_dir, SAFETENSORS_WEIGHTS_FILE)
    keep: set[str] = set()
    if os.path.isfile(index_path):
        try:
            index = read_checkpoint_index(output_dir)
        except (OSError, json.JSONDecodeError):
            index = {}
        # A per-rank EP/TP index is not a full save's product; blessing its shard list as a keep-set
        # would leave behind a layout these callers must never produce.
        shards = set() if index_declares_ep_shards(index) else set(index.get("weight_map", {}).values())
        # Both layouts present means one is a previous run's leftover, and the newer write is this
        # one; without the tie-break a single-file writer (reset_sinks) loses to the stale shards.
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
    """Sweep the model files this save did not write, then write the safetensors index over the rest.

    The one index writer every sharded save funnels through, so the metadata shape and the sweep
    contract cannot drift between them. The sweep runs after the shards and BEFORE the index: the
    parts are on disk and ``weight_map`` names exactly the ones this save claims, which is the
    keep-set needed to drop a stale leftover before it can outrank the fresh index.

    ``metadata`` is the index's metadata block verbatim — ``total_size`` plus whatever the layout
    declares. A non-mapping is refused: a bare scalar would produce ``{"metadata": 12345}``, an
    index every reader parses without complaint and no reader can use.
    """
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
    """Rename every streamed part (``{part name: final HF name}``) into place, then index a sharded
    result or sweep the leftovers beside a single ``model.safetensors``.

    The finalize half of a renaming writer's protocol, called once every part is on disk: renames are
    metadata operations, so the window in which the directory mixes two saves is this loop rather
    than the whole (I/O-bound) write. ``weight_map`` already names the final files; ``metadata`` is
    the index's block (:func:`write_merged_index`).
    """
    for part, final in final_names.items():
        # Overwrites a same-named leftover, which is intended; the sweep then only has to consider
        # names this save did not claim.
        os.replace(os.path.join(output_dir, part), os.path.join(output_dir, final))
    if len(final_names) > 1:
        write_merged_index(output_dir, weight_map, metadata)
    else:
        # A single part is the unsharded ``model.safetensors``, which carries no index of its own.
        remove_stale_checkpoint_files(output_dir, set(final_names.values()))


def save_sharded_state_dict(
    state_dict: dict[str, torch.Tensor], output_dir: str, max_shard_size: str = DEFAULT_MAX_SHARD_SIZE
) -> None:
    """Write a gathered state dict as HF-standard sharded safetensors.

    Uses HF's own splitter so the output matches ``save_pretrained`` (single ``model.safetensors``,
    or shards + ``model.safetensors.index.json`` loadable one at a time at 100B+).

    Every shard lands as a ``model-streaming-*`` part and is renamed into its final name only once
    they are ALL on disk (:func:`publish_streamed_parts`, which
    :meth:`StreamingShardWriter.close_as_hf_checkpoint` shares). Writing final names directly is unsafe
    on a re-save into a populated directory: the splitter's names are deterministic, so a same-shard-count re-save that dies after shard *k*
    leaves new shards 1..k beside old shards k+1..N under the old index, and ``from_pretrained``
    then loads half of each model silently. Stale files are swept only once every shard is on disk.
    """
    state_dict = {k: (v.contiguous() if torch.is_tensor(v) else v) for k, v in state_dict.items()}
    split = split_torch_state_dict_into_shards(
        state_dict,
        filename_pattern=SAFETENSORS_SHARD_PATTERN,
        max_shard_size=max_shard_size,
    )
    final_names: dict[str, str] = {}
    for part_index, (filename, keys) in enumerate(split.filename_to_tensors.items(), start=1):
        part = f"{STREAM_PART_PREFIX}-{part_index:05d}.safetensors"
        save_file(
            {k: state_dict[k] for k in keys},
            os.path.join(output_dir, part),
            metadata=SAFETENSORS_METADATA,
        )
        final_names[part] = filename
    publish_streamed_parts(output_dir, final_names, split.tensor_to_filename, split.metadata)


def _unique_key_by_suffix(state_dict: dict, suffix: str) -> str | None:
    """The single state-dict key equal to ``suffix`` or ending in ``.{suffix}``; None when absent.
    Warns and returns None on an ambiguous (multi-key) match."""
    matches = [k for k in state_dict if k == suffix or k.endswith(f".{suffix}")]
    if len(matches) > 1:
        logger.warning(
            f"reconcile_tie_word_embeddings: multiple state-dict keys match '{suffix}' "
            f"({matches[:4]}); skipping the tie-consistency check for this save."
        )
        return None
    return matches[0] if matches else None


def is_tie_reconcile_key(name: str) -> bool:
    """Whether ``name`` is one of the two keys :func:`reconcile_tie_word_embeddings` compares.

    A streamed save never holds the whole dict, so it keeps exactly these keys aside as they pass.
    """
    return any(name == suffix or name.endswith(f".{suffix}") for suffix in _TIE_KEY_SUFFIXES)


def reconcile_tie_word_embeddings(model, state_dict: dict) -> None:
    """Make ``config.tie_word_embeddings`` consistent with the saved tensors.

    Tied embeddings can diverge during distributed training (FSDP2 shards lm_head/embed_tokens
    independently, an fp32 upcast splits ``.data``, FLCE trains lm_head directly), and the flag flips
    to False when both diverged tensors are present so the trained lm_head is honoured. Keys resolve
    by suffix, covering wrapped/VLM layouts; a still-tied checkpoint carries one of the pair and skips.
    """
    config = getattr(model, "config", None)
    if config is None or not getattr(config, "tie_word_embeddings", False):
        return
    lm_key = _unique_key_by_suffix(state_dict, _TIE_KEY_SUFFIXES[0])
    emb_key = _unique_key_by_suffix(state_dict, _TIE_KEY_SUFFIXES[1])
    if lm_key is None or emb_key is None:
        return
    lm, emb = state_dict[lm_key], state_dict[emb_key]
    if lm.shape != emb.shape or not torch.equal(lm, emb):
        config.tie_word_embeddings = False
        logger.warning(
            f"{lm_key} and {emb_key} diverged during training; "
            f"saving config with tie_word_embeddings=False for checkpoint consistency."
        )


def write_gathered_checkpoint(
    model: torch.nn.Module,
    state_dict: dict,
    output_dir: str,
    max_shard_size: str = DEFAULT_MAX_SHARD_SIZE,
) -> None:
    """Write a gathered state dict as HF-standard (auto-sharded) safetensors. Save rank only.

    For a caller that already holds the whole dict (an injected-LoRA merge); the parallel gathered
    saves stream through :func:`~src.distributed.checkpoint.write.stream_gathered_checkpoint` instead,
    with the same normalization applied chunk by chunk. The ``.bin`` fallback is this path's alone:
    it needs a dict still whole after the safetensors write failed, which a streamed save no longer
    has, and at that scale a single ``torch.save`` is not a writable artifact anyway.

    Only the config write is gated on the model carrying one (both config steps skip a model
    without) — the weights owe the same normalization and layout either way, and a raw ``.bin`` is an
    artifact no export tool reads.
    """
    # FSDP2 sharding can break tied embeddings; keep the config honest.
    reconcile_tie_word_embeddings(model, state_dict)
    save_model_config(model, output_dir)
    # Save dtype + hub expert layout. EP-gathered dicts never arrive here: ``select_checkpoint_saver``
    # routes every ``has_ep_layers`` context to the EP saver, whose gather emits hub-layout expert keys.
    state_dict = normalize_gathered_state_dict(model, state_dict)
    try:
        save_sharded_state_dict(state_dict, output_dir, max_shard_size=max_shard_size)
    except Exception as e:
        logger.warning(f"sharded safetensors save failed: {e}, using pytorch format")
        # The .bin goes on disk FIRST, then the safetensors leftovers are swept: resume prefers an
        # index over the .bin so they must go, but sweeping first would let a second failure destroy
        # the only checkpoint the directory held. The sweep never matches the .bin itself.
        torch.save(state_dict, os.path.join(output_dir, LEGACY_WEIGHTS_FILE))
        try:
            remove_stale_checkpoint_files(output_dir, keep=set())
        except OSError as sweep_error:
            logger.warning(
                f"stale safetensors sweep after the .bin fallback failed: {sweep_error} — "
                f"resume prefers an index over the .bin, so this directory may resume the OLD weights"
            )


class StreamingShardWriter:
    """Accumulate tensors and flush them to numbered safetensors parts under one prefix.

    ``enabled=False`` makes every method a no-op, so non-writer ranks, which still enter each
    collective gather, share the call sites without a branch at each one.
    """

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
        """Stage one tensor, flushing the pending batch first if this one would overflow it.

        Flushing before appending is HF's own rule: a part then exceeds the limit only when a single
        tensor does, instead of every part overshooting by its last tensor.

        A key may be staged once. Unlike a state-dict build, a streaming writer cannot resolve a
        second claim: the first tensor is already in a flushed part, so the checkpoint would carry
        the key twice at two shapes with ``total_size`` counting both. A re-added key means the
        caller's chunk sequence claims one tensor from two sources, so it is refused rather than
        overwritten.
        """
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
        nbytes = tensor.numel() * tensor.element_size()
        if self._pending and self._pending_bytes + nbytes > self.max_bytes:
            self._flush()
        self._pending[key] = tensor.contiguous()
        self._pending_bytes += nbytes

    def close(self) -> tuple[dict[str, str], int]:
        """Flush the remainder and return ``(weight_map fragment, bytes written)``.

        Must be called before the index exchange: the index must never name a file that is not on
        disk yet.
        """
        if not self.enabled:
            return {}, 0
        self._flush()
        return self._weight_map, self._total_bytes

    def close_as_hf_checkpoint(self) -> int:
        """Flush, rename the parts to HF-standard names, write the index; return bytes written.

        The single-writer finalizer: what ``from_pretrained`` needs, without ever holding the whole
        state dict. Its part boundaries differ from :func:`save_sharded_state_dict` (this flushes the
        open block before an oversized tensor rather than carrying it alongside), so both layouts load
        identically but neither the file count nor the key-to-file map is comparable.

        Only valid when this writer owns every ``model*.safetensors`` in the directory: it deletes
        the ones it did not write, because ``from_pretrained`` prefers a leftover single
        ``model.safetensors`` over the index. The sweep runs after the parts are on disk, so a
        re-save transiently holds both checkpoints (2x peak disk).

        A multi-writer caller must use :meth:`close`: renaming needs the global part count.
        """
        if not self.enabled:
            return 0
        self._flush()
        if not self._part:
            # An enabled writer with no parts means the caller gathered nothing, and the sweep below
            # would delete the previous checkpoint and put nothing in its place.
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
        # Numbered even when the writer produces a single part, so it cannot collide with a stray
        # `model.safetensors` from another writer.
        name = f"{self.filename_prefix}-{self._part:05d}.safetensors"
        # Without the framework marker safetensors readers do not resolve the file as torch tensors.
        save_file(self._pending, os.path.join(self.output_dir, name), metadata=SAFETENSORS_METADATA)
        self._weight_map.update(dict.fromkeys(self._pending, name))
        self._part_names.append(name)
        self._total_bytes += self._pending_bytes
        self._pending = {}
        self._pending_bytes = 0
