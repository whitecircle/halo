"""External tables in a gathered checkpoint: table-only files under ``<module path>.<part key>``, written once from the
attached views and linked whole afterwards."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable

from halo.checkpoint.atomic import link_or_copy_file, save_safetensors_file, staged_file
from halo.checkpoint.format import SAFETENSORS_METADATA, SAFETENSORS_SHARD_PATTERN
from halo.checkpoint.writing import StreamingShardWriter
from halo.models.external_tables import ExternalTable, TableFiles, named_external_tables

logger = logging.getLogger(__name__)


GATHERED_TABLE_FILE_PREFIX = SAFETENSORS_SHARD_PATTERN.format(suffix="").removesuffix(".safetensors")


def _table_filename(prefix: str, table_index: int, file_index: int, total_files: int) -> str:
    """Inside the shard glob, so a later save sweeps it once no index names it, and apart from streamed part names."""
    return f"{prefix}-table{table_index:02d}-{file_index:05d}-of-{total_files:05d}.safetensors"


def _write_parts(
    table: ExternalTable, keys: tuple[str, ...], table_index: int, writer: StreamingShardWriter, prefix: str
) -> dict[str, str]:
    """Write the parts into table-only files of at most the writer's shard size; returns their weight map."""
    groups: list[list[int]] = [[]]
    size = 0
    for index, part in enumerate(table.parts):
        if groups[-1] and size + part.nbytes > writer.max_bytes:
            groups.append([])
            size = 0
        groups[-1].append(index)
        size += part.nbytes
    weight_map = {}
    for file_index, group in enumerate(groups, start=1):
        filename = _table_filename(prefix, table_index, file_index, len(groups))
        with staged_file(writer.output_dir, filename, durable=False) as staged:
            save_safetensors_file({keys[i]: table.parts[i] for i in group}, staged, metadata=SAFETENSORS_METADATA)
        weight_map.update(dict.fromkeys((keys[i] for i in group), filename))
    return weight_map


def _link_parts(source: TableFiles, table_index: int, output_dir: str, prefix: str) -> dict[str, str]:
    """Hard-link (copy across filesystems) ``source``'s table-only files into ``output_dir``; returns their map."""
    names = list(dict.fromkeys(source.files))
    renamed = {
        name: _table_filename(prefix, table_index, index, len(names)) for index, name in enumerate(names, start=1)
    }
    for name, final in renamed.items():
        origin, destination = os.path.join(source.directory, name), os.path.join(output_dir, final)
        if not (os.path.exists(destination) and os.path.samefile(origin, destination)):
            link_or_copy_file(origin, destination)
    return {key: renamed[name] for key, name in zip(source.keys, source.files, strict=True)}


def _is_linkable(source: TableFiles | None, keys: tuple[str, ...]) -> bool:
    """Whether ``source`` names table-only files, still on disk, holding exactly ``keys``."""
    return (
        source is not None
        and source.table_only
        and source.keys == keys
        and all(os.path.isfile(os.path.join(source.directory, name)) for name in set(source.files))
    )


def write_external_tables(
    model,
    writer: StreamingShardWriter,
    *,
    checkpoint_name: Callable[[str], str] | None = None,
    file_prefix: str = GATHERED_TABLE_FILE_PREFIX,
) -> None:
    """Put every external table into ``writer``'s directory and index, linked from table-only files holding the keys,
    else written from the attached parts. ``checkpoint_name`` maps a module path to its checkpoint name (a PP stage's
    re-based layer index); ``file_prefix`` keeps writers sharing a directory apart. A table never attached raises."""
    if not writer.enabled:
        return
    name_of = checkpoint_name or str
    for table_index, (name, table) in enumerate(named_external_tables(model)):
        if not table.is_attached:
            raise RuntimeError(
                f"{name} ({type(table).__name__}) holds no table, so this save cannot write it: the model was built "
                f"for an external table but never passed through finalize_loaded_model, which attaches it."
            )
        saved_name = name_of(name)
        keys = tuple(f"{saved_name}.{table.part_key(index)}" for index in range(table.part_count))
        if _is_linkable(table.files, keys):
            weight_map, how = _link_parts(table.files, table_index, writer.output_dir, file_prefix), "linked"
        else:
            weight_map, how = _write_parts(table, keys, table_index, writer, file_prefix), "wrote"
        nbytes = sum(part.nbytes for part in table.parts)
        writer.add_written_files(weight_map, nbytes)
        table.files = TableFiles(writer.output_dir, tuple(weight_map[key] for key in keys), keys, table_only=True)
        logger.info(f"{how.capitalize()} external table {saved_name}: {len(keys)} parts, {nbytes / 1e9:.2f}GB")
