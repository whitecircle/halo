"""Merge the per-node directories of one pipeline-parallel save into one ``from_pretrained``-loadable checkpoint.

On a non-shared output filesystem (``DIST_OUTPUT_SHARED_FILESYSTEM=0``) each node's directory holds the index, the
stage parts it wrote (``model-ppNNNNN-of-NNNNN-*``), the ``model-wrapper-*`` parts and its own ranks' resume state.
"""

import logging
import os
import shutil
from collections.abc import Sequence
from fnmatch import fnmatch

import torch

from src.checkpoint.format import (
    OPTIMIZER_META_FILE,
    OPTIMIZER_SHARD_GLOB,
    PP_STAGE_PART_RE,
    RESUME_SIDECAR_PREFIXES,
    optimizer_shard_filename,
)
from src.checkpoint.offline_export import copy_checkpoint_aux_files, reject_in_place_conversion
from src.checkpoint.reading import read_checkpoint_index, reject_partial_checkpoint
from src.checkpoint.writing import write_merged_index

logger = logging.getLogger(__name__)


def _first_holder(name: str, input_dirs: Sequence[str]) -> str | None:
    """The first input holding ``name``, or None; two holding it at different sizes raise (a torn copy)."""
    holders = [d for d in input_dirs if os.path.isfile(os.path.join(d, name))]
    if len({os.path.getsize(os.path.join(d, name)) for d in holders}) > 1:
        raise ValueError(f"{name} differs in size across {holders}: a torn copy. Re-copy that node's directory.")
    return holders[0] if holders else None


def _resume_state_sources(input_dirs: Sequence[str]) -> dict[str, str]:
    """``{filename: input holding it}`` for the per-rank resume files of every input (optimizer shards, RNG and
    prefetch states, one per global rank) and the optimizer meta every node writes alike.

    Raises:
        ValueError: a per-rank file held at different sizes, inputs whose optimizer metas differ, or a rank
            of the meta whose optimizer shard no input holds.
    """
    names = {
        name
        for input_dir in input_dirs
        for name in os.listdir(input_dir)
        if fnmatch(name, OPTIMIZER_SHARD_GLOB) or name.startswith(RESUME_SIDECAR_PREFIXES)
    }
    sources = {name: _first_holder(name, input_dirs) for name in sorted(names)}
    meta_paths = [os.path.join(input_dir, OPTIMIZER_META_FILE) for input_dir in input_dirs]
    metas = [
        torch.load(path, map_location="cpu", weights_only=True) if os.path.isfile(path) else None
        for path in meta_paths
    ]
    for input_dir, meta in zip(input_dirs[1:], metas[1:], strict=True):
        if meta != metas[0]:
            raise ValueError(
                f"{input_dir} and {input_dirs[0]} carry different {OPTIMIZER_META_FILE}: they are not node "
                f"directories of one save, or one node's optimizer save did not complete."
            )
    if metas[0] is None:
        return sources
    sources[OPTIMIZER_META_FILE] = input_dirs[0]
    missing = [rank for rank in range(metas[0]["num_ranks"]) if optimizer_shard_filename(rank) not in sources]
    if missing:
        raise ValueError(
            f"No node directory holds the optimizer shards of ranks {missing}. Pass every node's directory of "
            f"this save; a rank's shard lives only on its own node."
        )
    return sources


def merge_pp_stage_parts(input_dirs: Sequence[str], output_dir: str) -> None:
    """Write ``output_dir`` from the per-node directories ``input_dirs`` of one pipeline-parallel save; the result
    resumes exactly on the save's topology. Every input must carry the same index; each part it names is copied
    from the first input holding it, every input's per-rank resume files with one optimizer meta, the rank-uniform
    files (config, tokenizer, remote code, trainer state, scheduler, export sidecars) from the first input, and
    the index last. Every input check runs before the first write.

    Raises:
        ValueError: an input that is the output, inputs whose indexes differ, a part no input holds or
            held at different sizes, a refused resume state, or an output that still names an absent shard.
    """
    for input_dir in input_dirs:
        reject_in_place_conversion(input_dir, output_dir)
    index = read_checkpoint_index(input_dirs[0])
    for input_dir in input_dirs[1:]:
        if read_checkpoint_index(input_dir) != index:
            raise ValueError(
                f"{input_dir} and {input_dirs[0]} carry different safetensors indexes: they are not "
                f"node directories of one pipeline-parallel save."
            )

    sources: dict[str, str] = {}
    unheld: list[str] = []
    for part in sorted(set(index["weight_map"].values())):
        source = _first_holder(part, input_dirs)
        if source is None:
            stage = PP_STAGE_PART_RE.match(part)
            unheld.append(f"{part} (stage {int(stage[1])})" if stage else part)
        else:
            sources[part] = source
    if unheld:
        raise ValueError(
            f"No node directory holds {', '.join(unheld)}. Pass every node's directory of this save; a "
            f"stage's parts live only on the nodes that ran it."
        )
    resume_sources = _resume_state_sources(input_dirs)

    os.makedirs(output_dir, exist_ok=True)
    copy_checkpoint_aux_files(input_dirs[0], output_dir)
    for name, source in {**sources, **resume_sources}.items():
        shutil.copy2(os.path.join(source, name), os.path.join(output_dir, name))
    write_merged_index(output_dir, index["weight_map"], index.get("metadata", {}))
    reject_partial_checkpoint(output_dir)
    logger.info(f"Merged {len(sources)} parts from {len(input_dirs)} node directories into {output_dir}")
