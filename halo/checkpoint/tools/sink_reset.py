"""Reset a checkpoint's attention sinks to their dtype minimum, rewriting only the shards holding one, each staged and
verified before it replaces its target; a checkpoint with no sinks is copied through."""

import functools
import logging
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from halo.checkpoint.atomic import create_staged_file, link_or_copy_file, publish_staged_file, save_safetensors_file
from halo.checkpoint.format import SAFETENSORS_INDEX_FILE, SAFETENSORS_METADATA
from halo.checkpoint.model_card import tag_exported_model_card
from halo.checkpoint.reading import checkpoint_shard_files, detect_model_types, reject_partial_checkpoint
from halo.checkpoint.tools.offline_export import (
    copy_checkpoint_aux_files,
    is_non_artifact_entry,
    preflight_resource_warning,
    reject_in_place_conversion,
    resolve_checkpoint_source,
)
from halo.checkpoint.writing import sweep_after_full_save
from halo.models.attention.gpt_oss_sinks import is_sink_key, neutralized_sink_value, uses_gpt_oss_sinks

logger = logging.getLogger(__name__)


def _assert_sinks_at_min(tensors: dict[str, torch.Tensor], sink_keys: list[str], where: str) -> None:
    """Refuse unless every sink in ``sink_keys`` was written at dtype min, on a staged write, so failure ships
    nothing."""
    missing = sorted(set(sink_keys) - set(tensors))
    failed = sorted(
        key for key, tensor in tensors.items() if not (tensor == neutralized_sink_value(tensor.dtype)).all()
    )
    if missing or failed:
        raise RuntimeError(
            f"Sink reset failed verification in {where}: "
            + (f"sink tensors missing from the written checkpoint: {missing[:3]}; " if missing else "")
            + (f"not at dtype min: {failed[:3]}; " if failed else "")
            + "the target checkpoint was left untouched."
        )


def _sink_keys_by_shard(shards: list[str]) -> dict[str, list[str]]:
    """``{shard path: its sink keys}`` for every shard holding one, read from the headers alone."""
    found = {}
    for shard in shards:
        with safe_open(shard, framework="pt") as reader:
            keys = sorted(key for key in reader.keys() if is_sink_key(key))  # noqa: SIM118 - not a mapping
        if keys:
            found[shard] = keys
    return found


def _non_artifact_entries(root: str, directory: str, names: list[str]) -> list[str]:
    """``shutil.copytree``'s ignore: the top-level entries of ``root`` the aux copy leaves behind."""
    if directory != root:
        return []
    return [name for name in names if is_non_artifact_entry(name, is_dir=os.path.isdir(os.path.join(root, name)))]


def _passthrough_copy(source: Path, output_dir: Path, dry_run: bool) -> None:
    """Write ``output_dir`` for a checkpoint with no sinks, since returning without writing would point the next stage
    at a missing or stale directory; it leaves out what a rewriting reset does not carry (a vendor dump, the ``_*``
    entries)."""
    if dry_run or output_dir == source:
        return
    logger.info(f"No sinks to reset — copying the checkpoint to {output_dir} unchanged...")
    shutil.copytree(
        str(source),
        str(output_dir),
        dirs_exist_ok=True,
        ignore=functools.partial(_non_artifact_entries, str(source)),
    )
    tag_exported_model_card(str(output_dir), source_dir=str(source))


def _report_sinks(sink_keys: dict[str, list[str]]) -> None:
    for shard, keys in sink_keys.items():
        with safe_open(shard, framework="pt") as reader:
            for key in keys:
                tensor = reader.get_tensor(key)
                logger.info(
                    f"  {key}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}, "
                    f"values (first 4): {tensor.flatten()[:4].tolist()}"
                )


def _stage_reset_shard(shard: str, keys: list[str], output_dir: Path) -> Path:
    """``shard`` with its sinks at dtype min, written to a staged sibling under ``output_dir`` and verified there."""
    tensors = load_file(shard, device="cpu")
    for key in keys:
        tensors[key].fill_(neutralized_sink_value(tensors[key].dtype))
    staged = create_staged_file(output_dir, os.path.basename(shard))
    try:
        save_safetensors_file(tensors, staged, metadata=SAFETENSORS_METADATA)
        written = {key: tensor for key, tensor in load_file(str(staged), device="cpu").items() if is_sink_key(key)}
        _assert_sinks_at_min(written, keys, str(staged))
    except BaseException:
        staged.unlink(missing_ok=True)
        raise
    return staged


def _publish_reset_shards(sink_keys: dict[str, list[str]], output_dir: Path) -> None:
    """Stage and verify every sink shard, then rename each over its name, only once every stage verified."""
    staged: dict[str, Path] = {}
    try:
        for shard, keys in sink_keys.items():
            staged[os.path.basename(shard)] = _stage_reset_shard(shard, keys, output_dir)
    except BaseException:
        for path in staged.values():
            path.unlink(missing_ok=True)
        raise
    published = []
    for name, path in staged.items():
        try:
            publish_staged_file(path, output_dir / name)
        except OSError as exc:
            pending = [str(staged[pending_name]) for pending_name in staged if pending_name not in published]
            raise RuntimeError(
                f"Sink reset could not publish {name} to {output_dir} ({exc}). "
                + (f"{published} already carry the reset; " if published else "")
                + f"the verified, reset shards still to place are {pending} — rename each over its name by hand."
            ) from exc
        published.append(name)


def _resolve_destination(checkpoint_dir: str, output_dir: str | None, *, dry_run: bool, in_place: bool) -> str:
    """The directory the reset writes: ``checkpoint_dir`` only for ``--in_place`` or a dry run, else an
    ``--output_dir`` that is not the input."""
    if in_place:
        if output_dir is not None:
            raise ValueError(
                "--in_place rewrites the --model_id directory, so it cannot be combined with --output_dir."
            )
        if not os.path.isdir(checkpoint_dir):
            raise ValueError(
                f"--in_place cannot rewrite {checkpoint_dir!r}: it is not a local directory (a HuggingFace "
                f"repo ID, or a path that does not exist). Pass --output_dir instead."
            )
        return checkpoint_dir
    if output_dir is None:
        # A dry run only reads, so it needs no destination.
        if not dry_run:
            raise ValueError(
                "--output_dir is required: this tool replaces a checkpoint's sink tensors, and defaulting "
                "to the input would let a mistyped command rewrite the only copy with no undo. Pass "
                "--output_dir <new dir>, or --in_place to rewrite the --model_id directory deliberately."
            )
        return checkpoint_dir
    reject_in_place_conversion(checkpoint_dir, output_dir)
    return output_dir


def reset_sinks(
    checkpoint_dir: str,
    output_dir: str | None = None,
    dry_run: bool = False,
    in_place: bool = False,
    revision: str | None = None,
) -> int:
    """Reset every attention sink tensor to its dtype minimum; returns how many were found. ``in_place`` rewrites
    ``checkpoint_dir`` itself and has no undo; ``dry_run`` only inspects."""
    # Input gate first: a per-rank EP/TP save stores one rank's slice under its ordinary name, which a reset would ship
    # as the whole tensor, whichever destination was named.
    reject_partial_checkpoint(checkpoint_dir)
    output_dir = _resolve_destination(checkpoint_dir, output_dir, dry_run=dry_run, in_place=in_place)

    source = Path(resolve_checkpoint_source(checkpoint_dir, revision))
    reject_partial_checkpoint(str(source))
    output_path = Path(output_dir) if output_dir != checkpoint_dir else source
    shards = checkpoint_shard_files(str(source))
    sink_keys = _sink_keys_by_shard(shards)
    count = sum(len(keys) for keys in sink_keys.values())
    logger.info(f"Found {count} sink parameters in {len(sink_keys)} of {len(shards)} shard(s)")

    if not sink_keys:
        # A sinks-carrying family whose keys miss the sink spelling would otherwise be copied through
        # and reported reset with its sinks still live.
        if any(
            uses_gpt_oss_sinks(SimpleNamespace(model_type=model_type)) for model_type in detect_model_types(source)
        ):
            raise RuntimeError(
                f"{checkpoint_dir} is a sink-carrying family, but none of its tensors is named like a sink "
                f"(is_sink_key): its sink spelling changed, so nothing would be reset. Extend is_sink_key."
            )
        logger.info("Nothing to reset.")
        _passthrough_copy(source, output_path, dry_run)
        return 0

    _report_sinks(sink_keys)
    if dry_run:
        logger.info("Dry run — no changes written.")
        return count

    # One sink-holding shard is resident at a time; a new output gains every shard (linked where the
    # filesystem allows), an in-place one only the staged copies of the rewritten shards.
    rewritten = [os.path.getsize(shard) for shard in sink_keys]
    preflight_resource_warning(
        "reset_sinks",
        str(output_path),
        disk_bytes=sum(rewritten) if output_path == source else sum(os.path.getsize(s) for s in shards),
        ram_bytes=2 * max(rewritten),
    )

    if output_path == source:
        _publish_reset_shards(sink_keys, output_path)
        tag_exported_model_card(str(output_path), source_dir=str(source))
    else:
        os.makedirs(output_path, exist_ok=True)
        copy_checkpoint_aux_files(str(source), str(output_path))
        for shard in shards:
            if shard not in sink_keys:
                link_or_copy_file(shard, output_path / os.path.basename(shard))
        if (source / SAFETENSORS_INDEX_FILE).is_file():
            # A fresh mtime: the sweep below breaks an index-versus-single-file tie by it.
            shutil.copyfile(source / SAFETENSORS_INDEX_FILE, output_path / SAFETENSORS_INDEX_FILE)
        _publish_reset_shards(sink_keys, output_path)
        # The reset defines the directory: drop weight files it did not produce (an earlier run's shards or file).
        sweep_after_full_save(str(output_path))

    logger.info(f"All {count} sink tensors reset to dtype min, verified, and saved.")
    return count
