"""Merge the per-node directories of one pipeline-parallel save into one ``from_pretrained``-loadable checkpoint.

A save on a non-shared output filesystem (``DIST_OUTPUT_SHARED_FILESYSTEM=0``) leaves each node's
directory holding the whole-model index but only the stage parts that node wrote
(``model-ppNNNNN-of-NNNNN-*``) plus the ``model-wrapper-*`` parts every node writes.
"""

import logging
import os
import re
import shutil
from collections.abc import Sequence

from src.checkpoint.offline_export import copy_checkpoint_aux_files, reject_in_place_conversion
from src.checkpoint.reading import read_checkpoint_index, reject_partial_checkpoint
from src.checkpoint.writing import write_merged_index

logger = logging.getLogger(__name__)


def merge_pp_stage_parts(input_dirs: Sequence[str], output_dir: str) -> None:
    """Write ``output_dir`` from the per-node directories ``input_dirs`` of one pipeline-parallel save.

    Every input must carry the same index. Each part the index names is copied from the first input that
    holds it, the non-weight files (config, generation config, tokenizer, remote code, export sidecars)
    from the first input, and the index last.

    Raises:
        ValueError: an input that is the output, inputs whose indexes differ, a part no input holds or
            held at different sizes, or an output that still names an absent shard.
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
        holders = [d for d in input_dirs if os.path.isfile(os.path.join(d, part))]
        if not holders:
            stage = re.match(r"model-pp(\d+)-of-", part)
            unheld.append(f"{part} (stage {int(stage[1])})" if stage else part)
            continue
        sizes = {os.path.getsize(os.path.join(d, part)) for d in holders}
        if len(sizes) > 1:
            raise ValueError(f"{part} differs in size across {holders}: a torn copy. Re-copy that node's directory.")
        sources[part] = holders[0]
    if unheld:
        raise ValueError(
            f"No node directory holds {', '.join(unheld)}. Pass every node's directory of this save; a "
            f"stage's parts live only on the nodes that ran it."
        )

    os.makedirs(output_dir, exist_ok=True)
    copy_checkpoint_aux_files(input_dirs[0], output_dir)
    for part, source in sources.items():
        shutil.copy2(os.path.join(source, part), os.path.join(output_dir, part))
    write_merged_index(output_dir, index["weight_map"], index.get("metadata", {}))
    reject_partial_checkpoint(output_dir)
    logger.info(f"Merged {len(sources)} parts from {len(input_dirs)} node directories into {output_dir}")
