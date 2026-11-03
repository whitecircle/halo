"""Merge the per-node directories of one pipeline-parallel save into one ``from_pretrained``-loadable checkpoint. On a
non-shared output filesystem each node's directory holds the index, its stage parts, the wrapper parts and its ranks'
resume state."""

import logging
import os
import shutil
from collections.abc import Sequence

from halo.checkpoint.atomic import link_or_copy_file
from halo.checkpoint.format import PP_STAGE_PART_RE
from halo.checkpoint.reading import read_checkpoint_index, reject_partial_checkpoint
from halo.checkpoint.tools.offline_export import (
    copy_checkpoint_aux_files,
    first_holder,
    reject_in_place_conversion,
    resume_state_sources,
)
from halo.checkpoint.writing import write_merged_index

logger = logging.getLogger(__name__)


def merge_pp_stage_parts(input_dirs: Sequence[str], output_dir: str) -> None:
    """Write ``output_dir`` from one PP save's per-node directories; the result resumes exactly on the save's topology.
    Every input must carry the same index and every check runs before the first write. Parts copy from their first
    holder, per-rank resume files hard-link where the filesystem allows, and the index lands last."""
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
    copy_checkpoint_aux_files(input_dirs[0], output_dir, carried=resume_sources)
    for part, source in sources.items():
        shutil.copy2(os.path.join(source, part), os.path.join(output_dir, part))
    for name, source in resume_sources.items():
        link_or_copy_file(os.path.join(source, name), os.path.join(output_dir, name))
    write_merged_index(output_dir, index["weight_map"], index.get("metadata", {}))
    reject_partial_checkpoint(output_dir)
    logger.info(f"Merged {len(sources)} parts from {len(input_dirs)} node directories into {output_dir}")
