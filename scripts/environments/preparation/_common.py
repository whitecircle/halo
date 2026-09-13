"""Helpers shared by the dataset-preparation scripts."""

from pathlib import Path

import pyarrow.parquet as pq


def parquet_parts(directory: str | Path, *, non_empty: bool = False) -> list[str]:
    """The parquet files under ``directory`` in sorted order; ``non_empty`` leaves out zero-row parts.
    Raises when none remain, since every caller needs at least one."""
    parts = sorted(
        str(p) for p in Path(directory).glob("*.parquet") if not non_empty or pq.read_metadata(p).num_rows > 0
    )
    if not parts:
        raise SystemExit(f"no{' non-empty' if non_empty else ''} parquet parts under {directory}")
    return parts
