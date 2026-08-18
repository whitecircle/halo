#!/usr/bin/env python
"""Reset a checkpoint's attention sinks to the dtype's most negative value, disabling the sink mechanism.

This is the after-training tool for an existing checkpoint; a training run does not need it, since
``reset_sinks: true`` (the default) applies the same reset to the freshly loaded model in every
trainer, before the first step.

Writes the reset checkpoint to ``--output_dir``. ``--dry_run`` reports what would change without
writing; ``--in_place`` rewrites ``--input_dir`` itself (no undo, so it must be asked for).

Usage:
    python scripts/after_training/reset_sinks.py \\
        --input_dir checkpoints/my-model --output_dir checkpoints/my-model-nosinks
"""

import argparse
import logging
import os
import shutil
from pathlib import Path

import torch
from accelerate import PartialState
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

import src.distributed.expert_parallel.layers.roster  # noqa: F401 — registers the EP export roster the config finalizer requires
from scripts._common import add_max_shard_size_arg, add_trust_remote_code_arg
from src.checkpoint.format import DEFAULT_MAX_SHARD_SIZE, SAFETENSORS_WEIGHTS_FILE, sweep_after_full_save
from src.checkpoint.tool_io import (
    STAGING_SUFFIX,
    clear_staging_path,
    iter_checkpoint_tensors,
    preflight_model_load_resources,
    reject_in_place_conversion,
    reject_sharded_checkpoint,
    save_full_checkpoint,
)
from src.log import configure_cli_logging
from src.models.loading.model_preparation import (
    auto_load_model,
)
from src.models.patches.gpt_oss_sinks import (
    SinksPolicy,
    apply_sinks_policy,
    is_sink_key,
    neutralized_sink_value,
    stamped_sinks_policy,
)

configure_cli_logging()
logger = logging.getLogger(__name__)


def _assert_sinks_at_min(tensors: dict[str, torch.Tensor], sink_keys: list[str], where: str) -> None:
    """Every key in ``sink_keys`` must be present in the written ``tensors`` and sit at its dtype min.

    Runs on a staged write before it replaces the target, so a failure ships nothing.
    """
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


def parse_args():
    parser = argparse.ArgumentParser(description="Reset attention sinks in a safetensors checkpoint")
    parser.add_argument(
        "--input_dir",
        type=str,
        required=True,
        help="Checkpoint directory containing model.safetensors, or a Hub repo id to snapshot from "
        "(a repo id pairs with --output_dir, never --in_place)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save the modified checkpoint. Required unless --in_place is given.",
    )
    parser.add_argument(
        "--in_place",
        action="store_true",
        help="Rewrite --input_dir itself instead of writing a copy. There is no undo, so it must "
        "be asked for explicitly (local checkpoints only).",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print sink values without modifying the checkpoint",
    )
    add_max_shard_size_arg(parser, note="A single-file checkpoint is rewritten as the one file it came in as.")
    add_trust_remote_code_arg(parser)
    return parser.parse_args()


def _passthrough_copy(checkpoint_dir: Path, output_dir: Path, dry_run: bool) -> None:
    """Materialise ``output_dir`` for a checkpoint that turned out to carry no sinks.

    A no-op reset still writes the output it was asked for: returning without writing would leave the
    next pipeline stage pointing at a missing directory, or at a stale one of the same name.
    """
    if dry_run or output_dir == checkpoint_dir:
        return
    if not checkpoint_dir.is_dir():
        raise ValueError(
            f"{checkpoint_dir} carries no attention sinks, and as a HuggingFace repo id it has no local "
            f"directory to copy into {output_dir}. Nothing to reset — download it, or drop this step."
        )
    logger.info(f"No sinks to reset — copying the checkpoint to {output_dir} unchanged...")
    shutil.copytree(str(checkpoint_dir), str(output_dir), dirs_exist_ok=True)


def _reset_sinks_safetensors(safetensors_path: Path, output_dir: Path, dry_run: bool) -> int:
    """Reset sinks using direct safetensors load/save (single-file checkpoint)."""
    logger.info(f"Loading state dict from {safetensors_path}...")
    state_dict = load_file(str(safetensors_path), device="cpu")

    checkpoint_dir = safetensors_path.parent
    sink_keys = sorted(k for k in state_dict if is_sink_key(k))
    logger.info(f"Found {len(sink_keys)} sink parameters")

    if not sink_keys:
        logger.info("Nothing to reset.")
        _passthrough_copy(checkpoint_dir, output_dir, dry_run)
        return 0

    for key in sink_keys:
        tensor = state_dict[key]
        min_val = neutralized_sink_value(tensor.dtype)
        logger.info(
            f"  {key}: shape={tensor.shape}, dtype={tensor.dtype}, values (first 4): {tensor.flatten()[:4].tolist()}"
        )
        if not dry_run:
            state_dict[key] = tensor.fill_(min_val)

    if dry_run:
        logger.info("Dry run — no changes written.")
        return len(sink_keys)

    output_safetensors = output_dir / SAFETENSORS_WEIGHTS_FILE

    if output_dir != checkpoint_dir:
        logger.info(f"Copying checkpoint files to {output_dir}...")
        shutil.copytree(str(checkpoint_dir), str(output_dir), dirs_exist_ok=True)

    # Write via a sibling temp file and atomic rename: under --in_place this replaces the only copy,
    # which a kill mid-write would otherwise destroy.
    logger.info(f"Saving updated checkpoint to {output_safetensors}...")
    tmp_path = output_safetensors.with_suffix(".safetensors.tmp")
    save_file(state_dict, str(tmp_path))

    # Verify the staged file before the rename: under --in_place the rename replaces the only copy,
    # so a write that kept live sinks must not get that far.
    logger.info("Verifying...")
    try:
        written = {key: tensor for key, tensor in load_file(str(tmp_path), device="cpu").items() if is_sink_key(key)}
        _assert_sinks_at_min(written, sink_keys, str(tmp_path))
    except BaseException:
        tmp_path.unlink(missing_ok=True)  # discard a staged file that failed verification
        raise
    tmp_path.replace(output_safetensors)
    logger.info(f"All {len(sink_keys)} sink tensors verified — reset to dtype min.")

    return len(sink_keys)


def _verify_sinks_reset(directory: Path, sink_keys: list[str]) -> None:
    """``_assert_sinks_at_min`` over a written checkpoint directory (sharded or single-file)."""
    _assert_sinks_at_min(
        dict(iter_checkpoint_tensors(str(directory), predicate=is_sink_key)), sink_keys, str(directory)
    )


def _discard_path(path: Path) -> None:
    """Remove ``path``, file or directory tree, so a publish can land on the name."""
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _swap_staged_checkpoint(staging_dir: Path, output_dir: Path) -> None:
    """Publish a verified staged checkpoint onto ``output_dir``, entry by entry (``os.replace`` is
    atomic per name on one filesystem, and the staging directory is a sibling).

    Every entry is cloned beside its final name first (a hard link, so it costs no I/O) and the clone
    is replaced into place. Moving the staged entries instead would empty the staging directory as
    the swap proceeds, so a failure part-way would leave neither directory holding a whole checkpoint
    (and under ``--in_place`` that is the only copy). Cloning keeps the staged checkpoint intact
    until the last entry lands.
    """
    os.makedirs(output_dir, exist_ok=True)
    published = False
    for name in os.listdir(staging_dir):
        staged, final = staging_dir / name, output_dir / name
        clone = staging_dir / f"{name}{STAGING_SUFFIX}"
        try:
            if staged.is_dir():
                shutil.copytree(staged, clone)
            else:
                os.link(staged, clone)
            if final.is_dir():
                shutil.rmtree(final)
            os.replace(clone, final)
        except OSError as exc:
            _discard_path(clone)
            raise RuntimeError(
                f"Sink reset could not publish {name} to {output_dir} ({exc}). "
                + ("That directory is now part old, part new; " if published else "")
                + f"the complete, verified checkpoint is in {staging_dir} — move it into place by hand."
            ) from exc
        published = True
    shutil.rmtree(staging_dir, ignore_errors=True)


def _reset_sinks_from_pretrained(
    checkpoint_dir: Path,
    output_dir: Path,
    dry_run: bool,
    trust_remote_code: bool = True,
    max_shard_size: str = DEFAULT_MAX_SHARD_SIZE,
) -> int:
    """Reset sinks by loading the full model via from_pretrained (sharded checkpoints)."""
    logger.info(f"model.safetensors not found, loading full model from {checkpoint_dir}...")
    # Sink models (gpt-oss patched checkpoints) are this script's target, so load with remote code
    # allowed and the widest matching Auto* class.
    model = auto_load_model(
        str(checkpoint_dir),
        trust_remote_code=trust_remote_code,
        dtype="auto",
        device_map="cpu",
    )

    sink_keys = sorted(name for name, _ in model.named_parameters() if is_sink_key(name))
    logger.info(f"Found {len(sink_keys)} sink parameters")
    for key in sink_keys:
        param = model.get_parameter(key)
        logger.info(
            f"  {key}: shape={param.shape}, dtype={param.dtype}, values (first 4): {param.data.flatten()[:4].tolist()}"
        )

    # The trainers' own seam rather than a second walk: it neutralizes what every trainer neutralizes
    # and raises when the layer walk finds nothing on a sinks-carrying model, instead of saving a
    # checkpoint whose sinks are still live. The stamp it leaves is the only signal it did anything;
    # a family that carries no sinks is returned untouched.
    apply_sinks_policy(model, model.config, policy=SinksPolicy.NEUTRALIZED)
    if stamped_sinks_policy(model) is None:
        logger.info("Nothing to reset.")
        _passthrough_copy(checkpoint_dir, output_dir, dry_run)
        return 0

    if dry_run:
        logger.info("Dry run — no changes written.")
        return len(sink_keys)

    logger.info(f"Saving updated model to {output_dir}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint_dir), trust_remote_code=trust_remote_code)
    except Exception as exc:
        # Report the underlying error: a network or permission failure is not the same as a
        # checkpoint that ships no tokenizer, which the aux-file copy already covers.
        logger.warning(f"Tokenizer not saved from {checkpoint_dir} ({type(exc).__name__}: {exc}).")
        tokenizer = None

    # Staged write, verify, then swap, for the same reason the single-file branch stages its temp
    # file: under --in_place this writes over the only copy, and save_pretrained writes many files, so
    # a kill mid-write or a save that kept live sinks must not replace the source. The source's aux
    # files ride along (save_full_checkpoint copies them before writing, so the fresh config wins).
    tmp_dir = Path(clear_staging_path(output_dir))
    try:
        save_full_checkpoint(
            model,
            str(tmp_dir),
            processing_class=tokenizer,
            source_dir=str(checkpoint_dir),
            max_shard_size=max_shard_size,
        )
        _verify_sinks_reset(tmp_dir, sink_keys)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)  # discard a staged save that failed verification
        raise
    _swap_staged_checkpoint(tmp_dir, output_dir)
    # The verified save now defines the directory: drop the weight files it did not produce (a stale
    # shard numbering, a stale single file).
    sweep_after_full_save(str(output_dir))

    logger.info(f"All {len(sink_keys)} sink tensors reset to dtype min, verified, and saved.")
    return len(sink_keys)


def _is_hf_repo(checkpoint_dir: str) -> bool:
    """Check if checkpoint_dir looks like a HuggingFace repo ID (e.g. 'org/model')."""
    return not Path(checkpoint_dir).exists() and "/" in checkpoint_dir


def reset_sinks(
    checkpoint_dir: str,
    output_dir: str | None = None,
    dry_run: bool = False,
    in_place: bool = False,
    trust_remote_code: bool = True,
    max_shard_size: str = DEFAULT_MAX_SHARD_SIZE,
) -> int:
    """Reset all attention sink parameters in a checkpoint.

    Tries direct safetensors load first (single-file checkpoint). Falls back
    to loading the full model via from_pretrained for sharded checkpoints.

    Args:
        checkpoint_dir: Path to checkpoint directory or HuggingFace repo ID.
        output_dir: Directory to save the modified checkpoint. Required unless ``in_place``.
        dry_run: If True, only inspect sink values without modifying.
        in_place: Rewrite ``checkpoint_dir`` itself. Explicit because it has no undo; the sibling
            conversion tools refuse an in-place run outright.
        trust_remote_code: Passed to the model and tokenizer loads.
        max_shard_size: Per-file cap of the sharded re-save (the single-file branch rewrites its one file).

    Returns:
        Number of sink tensors found (and reset, if not dry_run).
    """
    # The input gate runs first: a per-rank EP/TP save lands on the from_pretrained branch, where the
    # real expert keys read as missing and are randomly initialized. That diagnosis is more useful
    # than a missing-destination error, and it holds whichever destination was named.
    if not _is_hf_repo(checkpoint_dir):
        reject_sharded_checkpoint(checkpoint_dir)

    if in_place:
        if output_dir is not None:
            raise ValueError("--in_place rewrites --input_dir, so it cannot be combined with --output_dir.")
        if _is_hf_repo(checkpoint_dir):
            raise ValueError(
                f"--in_place cannot rewrite {checkpoint_dir!r}: it is a HuggingFace repo ID, not a local "
                f"directory. Pass --output_dir instead."
            )
        output_dir = checkpoint_dir
    elif output_dir is None:
        # A dry run only reads (every write below is gated on it), so it needs no destination. The
        # refusal guards the writing invocations, where defaulting to the input would let a mistyped
        # command rewrite the only copy with no undo.
        if not dry_run:
            raise ValueError(
                "--output_dir is required: this tool replaces a checkpoint's sink tensors, and defaulting "
                "to the input meant a mistyped command rewrote the only copy with no undo. Pass "
                "--output_dir <new dir>, or --in_place to rewrite --input_dir deliberately."
            )
        output_dir = checkpoint_dir
    else:
        # Same refusal the sibling conversions make: an --output_dir aimed at the input is the
        # in-place run the flag above exists to make explicit.
        reject_in_place_conversion(checkpoint_dir, output_dir)

    # Both branches hold the whole checkpoint in host RAM (a single-file load, or from_pretrained
    # onto the CPU), and a writing run stages a full copy beside its target.
    preflight_model_load_resources(
        checkpoint_dir, output_dir, tool="reset_sinks", device_map="cpu", writes_full_model=not dry_run
    )

    checkpoint_path = Path(checkpoint_dir)
    output_path = Path(output_dir)
    safetensors_path = checkpoint_path / SAFETENSORS_WEIGHTS_FILE

    if safetensors_path.exists():
        count = _reset_sinks_safetensors(safetensors_path, output_path, dry_run)
        # Only this branch sweeps here: it copytrees the source and rewrites one file, so a stale
        # sharded set can ride along. Never in-place; the from_pretrained branch sweeps its own output
        # after the staged swap.
        if not dry_run and output_path.resolve() != checkpoint_path.resolve():
            sweep_after_full_save(str(output_path))
    else:
        count = _reset_sinks_from_pretrained(
            checkpoint_path, output_path, dry_run, trust_remote_code, max_shard_size=max_shard_size
        )
    return count


if __name__ == "__main__":
    args = parse_args()
    # sanitize_generation_config logs through accelerate's rank-aware logger, which raises without
    # this state, and every gpt-oss checkpoint ships do_sample=False, so it fires on this script's
    # main path.
    PartialState()
    reset_sinks(
        args.input_dir,
        output_dir=args.output_dir,
        dry_run=args.dry_run,
        in_place=args.in_place,
        trust_remote_code=args.trust_remote_code,
        max_shard_size=args.max_shard_size,
    )
