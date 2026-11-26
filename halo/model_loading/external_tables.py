"""Attaching a loaded model's external tables (:class:`~halo.models.external_tables.ExternalTable`) to the
checkpoint its config was read from, through the keys transformers would concatenate into each stock module."""

from __future__ import annotations

import itertools
import os
import threading
from collections.abc import Callable

from halo.checkpoint.reading import resolve_checkpoint_weights
from halo.model_loading.checkpoint_keys import build_key_plan, resolve_model_conversions
from halo.models.external_tables import EXTERNAL_TABLES_STAMP, named_external_tables
from halo.models.introspection.config_access import set_config_field_run_scoped
from halo.models.introspection.param_roles import persistent_buffers
from halo.runtime.ranks import is_local_main_process


def _weights_directory(model, weights_dir: str | None) -> str:
    """``weights_dir`` (a Hub id's snapshot at the agreed commit), else the directory the config names. A bare Hub id
    is refused, as transformers records no revision and its cache could hold another commit's table."""
    if weights_dir is not None:
        return weights_dir
    source = getattr(model.config, "_name_or_path", "") or ""
    if os.path.isdir(source):
        return source
    if source:
        raise FileNotFoundError(
            f"{type(model).__name__} holds an external table, read from the directory its weights were read from, "
            f"but its config names {source!r}, which is no local directory, and the load named no directory of its "
            f"own: transformers records no revision for a Hub id, so the table could come from another commit than "
            f"the weights. Load the model through a Halo loader, which reads the snapshot of the commit every rank "
            f"agreed on, or set model_name_or_path to the checkpoint's local directory (its Hub snapshot)."
        )
    raise FileNotFoundError(
        f"{type(model).__name__} holds an external table but its config names no checkpoint (a build from a "
        f"config object alone): the table is frozen and only a checkpoint supplies it. Load the model from a "
        f"checkpoint that holds the table."
    )


def _prefetch_files(paths: list[str]) -> None:
    """Ask the kernel to read ``paths`` into the page cache, so the first lookups do not fault cold pages."""
    for path in paths:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_WILLNEED)
        finally:
            os.close(descriptor)


def attach_external_tables(
    model, weights_dir: str | None = None, checkpoint_name: Callable[[str], str] | None = None
) -> None:
    """Attach every unattached external table from its checkpoint, prefetch the files once per node and stamp
    ``model.config`` run-scoped. ``checkpoint_name`` maps a module path to its checkpoint name (a PP stage's re-based
    layer index)."""
    name_of = checkpoint_name or str
    tables = {
        f"{name_of(name)}.{table.stock_key}": table
        for name, table in named_external_tables(model)
        if not table.is_attached
    }
    if not tables:
        return
    directory = _weights_directory(model, weights_dir)
    weight_map = resolve_checkpoint_weights(directory).weight_map
    live_names = itertools.chain(
        (name_of(name) for name, _ in model.named_parameters()),
        (name_of(name) for name, _ in persistent_buffers(model)),
        tables,
    )
    plan = build_key_plan(
        weight_map,
        live_names,
        resolve_model_conversions(model, add_legacy=True),
        getattr(model, "base_model_prefix", None) or None,
    )
    sources = {target: group.parts for group in plan.groups for target in group.targets if target in tables}
    unsourced = sorted(set(tables) - set(sources))
    if unsourced:
        raise FileNotFoundError(
            f"The checkpoint in {directory} holds no keys that load into {unsourced}: it is not a checkpoint of "
            f"this model with its external tables (a save that dropped them, or another family's)."
        )
    for target, table in tables.items():
        table.attach(directory, weight_map, [key for key, _ in sources[target]])
    set_config_field_run_scoped(model.config, EXTERNAL_TABLES_STAMP, True)
    if is_local_main_process():
        files = sorted({os.path.join(directory, name) for table in tables.values() for name in table.files.files})
        threading.Thread(target=_prefetch_files, args=(files,), name="external-table-prefetch", daemon=True).start()
