"""Readers for the artifacts the toolkit's checkpoint writers produce.

Key-set assertions go through the PRODUCTION reader (``load_full_state_dict``): it accepts both
layouts the writers pick between by size (index+shards vs a bare ``model.safetensors``), and it
REFUSES a per-rank EP/TP index — a test-local reader treating one as a whole checkpoint would pass
on exactly the artifact the writers must never leave behind.
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

    One definition — three checkpoint tests each re-spelled this with a different index policy, so
    the same name answered three different questions. ``include_index`` is that policy, explicit.
    """
    names = sorted(
        name for name in os.listdir(output_dir) if name.startswith("model") and name.endswith(".safetensors")
    )
    if include_index:
        index = os.path.join(output_dir, SAFETENSORS_INDEX_FILE)
        if os.path.isfile(index):
            names.append(SAFETENSORS_INDEX_FILE)
    return names
