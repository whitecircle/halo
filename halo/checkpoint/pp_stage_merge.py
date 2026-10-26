"""Merge the per-node directories of one pipeline-parallel save into one ``from_pretrained``-loadable checkpoint.

On a non-shared output filesystem (``DIST_OUTPUT_SHARED_FILESYSTEM=0``) each node's directory holds the index, the
stage parts it wrote (``model-ppNNNNN-of-NNNNN-*``), the ``model-wrapper-*`` parts and its own ranks' resume state.
"""

import logging
import os
import shutil
from collections.abc import Sequence

from halo.checkpoint.format import PP_STAGE_PART_RE
from halo.checkpoint.offline_export import (
    copy_checkpoint_aux_files,
    first_holder,
    reject_in_place_conversion,
    resume_state_sources,
)
from halo.checkpoint.reading import read_checkpoint_index, reject_partial_checkpoint
from halo.checkpoint.writing import write_merged_index

logger = logging.getLogger(__name__)


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
        source = first_holder(part, input_dirs)
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
    resume_sources = resume_state_sources(input_dirs)

    os.makedirs(output_dir, exist_ok=True)
    copy_checkpoint_aux_files(input_dirs[0], output_dir)
    for name, source in {**sources, **resume_sources}.items():
        shutil.copy2(os.path.join(source, name), os.path.join(output_dir, name))
    write_merged_index(output_dir, index["weight_map"], index.get("metadata", {}))
    reject_partial_checkpoint(output_dir)
    logger.info(f"Merged {len(sources)} parts from {len(input_dirs)} node directories into {output_dir}")
