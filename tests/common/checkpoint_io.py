"""Readers for the artifacts the toolkit's checkpoint writers produce.

Key-set assertions go through the production reader (``load_full_state_dict``): it accepts both
layouts the writers pick between by size (index+shards vs a bare ``model.safetensors``) and refuses a
per-rank EP/TP index, which a test-local reader would accept as a whole checkpoint.
"""

import os

from src.checkpoint.format import SAFETENSORS_INDEX_FILE, load_full_state_dict


def written_keys(output_dir: str) -> set[str]:
    """Every tensor key present in the gathered checkpoint at ``output_dir`` (see module docstring)."""
    state = load_full_state_dict(output_dir)
    assert state is not None, f"no checkpoint at {output_dir}"
    return set(state)


def weight_files(output_dir: str, *, include_index: bool = False) -> list[str]:
    """Sorted ``model*.safetensors`` basenames a save left in ``output_dir``.

    ``include_index`` makes the index policy explicit, so the same helper answers the same question
    for every caller.
    """
    names = sorted(
        name for name in os.listdir(output_dir) if name.startswith("model") and name.endswith(".safetensors")
    )
    if include_index:
        index = os.path.join(output_dir, SAFETENSORS_INDEX_FILE)
        if os.path.isfile(index):
            names.append(SAFETENSORS_INDEX_FILE)
    return names
