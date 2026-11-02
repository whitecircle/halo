"""Frozen tables a module reads but does not hold as state, attached as zero-copy views of checkpoint tensors. An
:class:`ExternalTable` is no parameter, buffer or state-dict entry, so no cast, wrap, optimizer or state-dict save
reaches it: ``finalize_loaded_model`` attaches it, the gathered checkpoint writers write it themselves, and a
state-dict-only save or a random-weight build of a model holding one is refused (``reject_random_init_tables``)."""

from __future__ import annotations

import copy
import os
from collections.abc import Iterator, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass

import torch
from safetensors import safe_open
from torch import nn

# The run-scoped config stamp under which a family's patched constructor builds an external table in place of
# its stock module; every build without it keeps the stock class.
EXTERNAL_TABLES_STAMP = "_halo_external_tables"


@dataclass(frozen=True)
class TableFiles:
    """Where a table's parts sit on disk: ``keys[i]`` in ``directory/files[i]``. ``table_only`` when those files
    hold nothing but this table's parts, so a save can link them whole."""

    directory: str
    files: tuple[str, ...]
    keys: tuple[str, ...]
    table_only: bool


class ExternalTable(nn.Module):
    """A frozen ``[rows, dim]`` table held as ``part_count`` row-contiguous parts outside the module state.

    The parts are the checkpoint tensors transformers concatenates into the stock module's ``stock_key``, saved
    under ``<module path>.<part_key(i)>``. A deep copy shares them.
    """

    stock_key = "weight"
    # ``str.format`` spelling of part ``index``'s key, relative to the module.
    part_key_format: str

    def __init__(self, rows: int, dim: int, part_count: int):
        super().__init__()
        self.rows = rows
        self.dim = dim
        self.part_count = part_count
        self.parts: tuple[torch.Tensor, ...] = ()
        # Each part's exclusive end row.
        self.part_ends: torch.Tensor | None = None
        # Where table-only copies of the parts were last read or written: what the next save links from.
        self.files: TableFiles | None = None

    def __deepcopy__(self, memo):
        clone = type(self).__new__(type(self))
        memo[id(self)] = clone
        memo.update({id(part): part for part in self.parts})
        clone.__dict__.update(copy.deepcopy(self.__dict__, memo))
        return clone

    @property
    def is_attached(self) -> bool:
        return bool(self.parts)

    def part_key(self, index: int) -> str:
        return self.part_key_format.format(index=index)

    def attach(self, directory: str, weight_map: Mapping[str, str], keys: Sequence[str]) -> None:
        """Take ``keys`` (part order) in ``directory`` as zero-copy views, which share the page cache across
        processes.

        Raises:
            ValueError: the keys, part shapes or dtypes are not the table this module was built for.
        """
        if len(keys) != self.part_count:
            raise ValueError(
                f"{type(self).__name__} is built for {self.part_count} parts but the checkpoint in {directory} "
                f"supplies {len(keys)}: its config's part count does not describe its table."
            )
        misordered = [key for index, key in enumerate(keys) if not key.endswith(f".{self.part_key(index)}")]
        if misordered:
            raise ValueError(
                f"The {type(self).__name__} keys in {directory} are not its parts in order ({misordered[:3]}): "
                f"part i is spelled '{self.part_key_format}'."
            )
        files = tuple(weight_map[key] for key in keys)
        with ExitStack() as stack:
            handles = {
                name: stack.enter_context(safe_open(os.path.join(directory, name), framework="pt", device="cpu"))
                for name in dict.fromkeys(files)
            }
            parts = tuple(handles[name].get_tensor(key) for key, name in zip(keys, files, strict=True))
        widths = {tuple(part.shape[1:]) for part in parts}
        dtypes = {part.dtype for part in parts}
        rows = sum(part.shape[0] for part in parts)
        if widths != {(self.dim,)} or len(dtypes) != 1 or rows != self.rows:
            raise ValueError(
                f"The {type(self).__name__} parts in {directory} hold {rows} rows of widths {sorted(widths)} in "
                f"{sorted(map(str, dtypes))}, but the config builds a single-dtype [{self.rows}, {self.dim}] "
                f"table: the checkpoint and its config.json disagree."
            )
        holding, table_keys = set(files), set(keys)
        table_only = all(key in table_keys for key, name in weight_map.items() if name in holding)
        self.parts = parts
        self.part_ends = torch.tensor([part.shape[0] for part in parts]).cumsum(0)
        self.files = TableFiles(directory, files, tuple(keys), table_only)


def named_external_tables(model: nn.Module) -> Iterator[tuple[str, ExternalTable]]:
    """Every :class:`ExternalTable` in ``model`` with its module path."""
    return ((name, module) for name, module in model.named_modules() if isinstance(module, ExternalTable))


def _table_names(model: nn.Module) -> str:
    return ", ".join(name for name, _ in named_external_tables(model))


def reject_state_dict_save(model: nn.Module) -> None:
    """Refuse a save that writes ``model``'s state dict alone (transformers' ``save_pretrained``) while it holds an
    external table, which no state dict carries: the checkpoint would load with a freshly initialized table."""
    names = _table_names(model)
    if names:
        raise RuntimeError(
            f"{type(model).__name__}.save_pretrained writes its state dict, which holds none of its external tables "
            f"({names}), so the checkpoint would reload with a randomly initialized table. Save it through a Halo "
            f"trainer's save_model, whose gathered writer writes the tables beside the weights."
        )
