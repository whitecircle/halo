#!/usr/bin/env python
"""Merge several fine-tuned checkpoints of the same architecture into one model.

A self-contained replacement for external merge tools, with no mergekit dependency. Supports the
common weight-space methods:

- ``linear``          — weighted average of the model weights (always normalized by the weight sum).
- ``slerp``           — spherical interpolation between exactly two models (per-tensor), param ``--t``.
- ``task_arithmetic`` — base + Σ wᵢ·(modelᵢ − base): sum the task vectors onto a base model.
- ``ties``            — TIES-merge: trim each task vector to its top-``--density`` magnitudes, elect a
                        sign per parameter, average the sign-agreeing values, add ``--lambda`` × that
                        to the base (arXiv:2306.01708).

Merging streams one tensor at a time across the input checkpoints (each key is loaded from every
model, merged, then written), so peak host memory is a single layer plus one pending output shard
rather than the merged model. Output is HF-sharded safetensors (``--max_shard_size`` sets the
per-file cap). Config, tokenizer and any remote-code modules are copied from ``--tokenizer_source``
(default: the base model, or the first model); a Hub id there is downloaded (weights excluded) and
copied the same way.

Examples:
    # Weighted linear average (weights normalized by their sum)
    python scripts/after_training/merge_models.py --method linear \\
        --models /ckpt/run-a:0.3 /ckpt/run-b:0.7 --output_dir /mnt/merged-linear

    # SLERP halfway between two checkpoints
    python scripts/after_training/merge_models.py --method slerp --t 0.5 \\
        --models /ckpt/run-a /ckpt/run-b --output_dir /mnt/merged-slerp

    # Task arithmetic / TIES onto a base
    python scripts/after_training/merge_models.py --method ties --density 0.6 --lambda 1.0 \\
        --base_model /ckpt/base --models /ckpt/run-a:0.5 /ckpt/run-b:0.5 --output_dir /mnt/merged-ties
"""

from __future__ import annotations

import argparse
import keyword
import logging
import os
import sys
from collections.abc import Callable
from typing import NamedTuple

import torch
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoConfig

import src.distributed.expert_parallel.layers.roster  # noqa: F401 — registers the EP export roster the config finalizer requires
from scripts._common import add_max_shard_size_arg, add_trust_remote_code_arg
from src.checkpoint.config_export import finalize_exported_config
from src.checkpoint.format import (
    DEFAULT_MAX_SHARD_SIZE,
    HF_STREAM_PART_PREFIX,
    WEIGHT_FILE_IGNORE_PATTERNS,
    copy_checkpoint_aux_files,
)
from src.checkpoint.shard_writer import StageShardWriter
from src.checkpoint.tool_io import (
    iter_checkpoint_shard_entries,
    preflight_resource_warning,
    reject_in_place_conversion,
    stored_tensor_nbytes,
)
from src.log import configure_cli_logging
from src.models.loading.dtype import DTYPE_BY_NAME
from src.models.moe_balancing import is_balancing_state_key

configure_cli_logging()
logger = logging.getLogger(__name__)

_SLERP_DOT_THRESHOLD = 0.9995  # above this the vectors are ~colinear → lerp (slerp is unstable)


def _parse_model_spec(spec: str) -> tuple[str, float | None]:
    """``path`` or ``path:weight`` → (path, weight). ``None`` when no weight was given, so the
    method/knob gate can tell an explicit ``:1.0`` from the default.

    Splits on the last colon so absolute paths and ``hf/org/model`` ids are safe; the suffix must
    parse as a float to be treated as a weight, otherwise the whole string is the path.
    """
    if ":" in spec:
        head, _, tail = spec.rpartition(":")
        try:
            return head, float(tail)
        except ValueError:
            return spec, None
    return spec, None


def _weight_map(model_dir: str) -> dict[str, str]:
    """Map every parameter name to the safetensors file holding it (single-file or sharded).

    Keys come from the shards themselves rather than the index's ``weight_map``, so a checkpoint
    whose index has drifted from its files still merges the tensors that are actually there.
    """
    return {key: shard for shard, _reader, key in iter_checkpoint_shard_entries(model_dir)}


class _TensorReader:
    """Lazy per-key tensor reader over a checkpoint's safetensors, caching open file handles."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        self.weight_map = _weight_map(model_dir)
        self._handles: dict[str, object] = {}

    def keys(self) -> set[str]:
        return set(self.weight_map)

    def get(self, key: str) -> torch.Tensor:
        return self._handle(key).get_tensor(key)

    def nbytes(self, key: str) -> int:
        """Storage size of ``key`` from the safetensors header alone — no tensor read."""
        return stored_tensor_nbytes(self._handle(key), key)

    def _handle(self, key: str):
        # The RAM preflight sizes the reference key set (the base's, under task_arithmetic/ties)
        # against model[0] ahead of the merge loop's own coverage check, so the refusal is raised
        # here: a bare KeyError there names neither the model lacking the key nor why it was wanted.
        path = self.weight_map.get(key)
        if path is None:
            raise KeyError(f"key {key!r} missing from {self.model_dir} (model mismatch)")
        handle = self._handles.get(path)
        if handle is None:
            handle = safe_open(path, framework="pt", device="cpu")
            self._handles[path] = handle
        return handle


def _reference_keys(method: str, readers: list[_TensorReader], base_reader: _TensorReader | None) -> list[str]:
    """The key set to merge over. ``task_arithmetic``/``ties`` task vectors are relative to the base,
    so iterate the base key set: iterating model[0] would drop any parameter present in the base (and
    other models) but absent from model[0] (an untied ``lm_head.weight``, say), yielding an
    incomplete, unloadable merged checkpoint. ``linear``/``slerp`` have no base, so use model[0]."""
    source = base_reader if (method in ("task_arithmetic", "ties") and base_reader is not None) else readers[0]
    return sorted(source.keys())


# --- per-tensor merge ops (all compute in float32, caller casts the result) ---


def _merge_linear(tensors: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    # Always a weighted average, so divide by the weight sum: a raw sum (default weights 1.0) would
    # scale every tensor by the model count without raising.
    total = sum(weights)
    if total == 0:
        raise ValueError("linear merge weights sum to zero")
    out = torch.zeros_like(tensors[0], dtype=torch.float32)
    for t, w in zip(tensors, weights, strict=True):
        out += (w / total) * t.float()
    return out


def _merge_slerp(t0: torch.Tensor, t1: torch.Tensor, t: float) -> torch.Tensor:
    """Per-tensor spherical interpolation between two tensors (flattened as one vector).

    The direction cosine is taken on magnitude-normalized copies: a neutralized GptOss sink sits at
    ``dtype.min`` (~3e38), whose square overflows fp32, so a raw norm reads ``inf``, the cosine of two
    identical vectors reads 0, and the resulting arc sums to ``-inf`` where the input was finite.
    """
    v0 = t0.float().flatten()
    v1 = t1.float().flatten()
    scale = torch.maximum(v0.abs().max(), v1.abs().max())
    if scale == 0:
        return torch.lerp(t0.float(), t1.float(), t)
    v0, v1 = v0 / scale, v1 / scale
    n0, n1 = v0.norm(), v1.norm()
    if n0 == 0 or n1 == 0:
        return torch.lerp(t0.float(), t1.float(), t)
    dot = torch.dot(v0 / n0, v1 / n1).clamp(-1.0, 1.0)
    if dot.abs() > _SLERP_DOT_THRESHOLD:
        return torch.lerp(t0.float(), t1.float(), t)  # ~colinear → linear is stable + ≈equal
    theta0 = torch.acos(dot)
    sin0 = torch.sin(theta0)
    s0 = torch.sin((1.0 - t) * theta0) / sin0
    s1 = torch.sin(t * theta0) / sin0
    return ((s0 * v0 + s1 * v1) * scale).reshape(t0.shape)


def _merge_task_arithmetic(base: torch.Tensor, tensors: list[torch.Tensor], weights: list[float]) -> torch.Tensor:
    out = base.float().clone()
    for t, w in zip(tensors, weights, strict=True):
        out += w * (t.float() - base.float())
    return out


def _merge_ties(
    base: torch.Tensor,
    tensors: list[torch.Tensor],
    weights: list[float],
    density: float,
    lambda_: float,
) -> torch.Tensor:
    """TIES: trim each task vector to its top-``density`` magnitudes, elect a per-element sign, then
    average the sign-agreeing (weighted) deltas and add ``lambda_`` × that onto the base."""
    base_f = base.float()
    deltas = []
    for t, w in zip(tensors, weights, strict=True):
        delta = (t.float() - base_f) * w
        if 0.0 < density < 1.0 and delta.numel() > 0:
            k = max(1, round(density * delta.numel()))
            thresh = delta.abs().flatten().kthvalue(delta.numel() - k + 1).values
            delta = torch.where(delta.abs() >= thresh, delta, torch.zeros_like(delta))
        deltas.append(delta)
    stacked = torch.stack(deltas, dim=0)
    elected_sign = torch.sign(stacked.sum(dim=0))
    agree = (torch.sign(stacked) == elected_sign) & (elected_sign != 0)
    weighted = stacked * agree
    count = agree.sum(dim=0).clamp(min=1)
    merged_delta = weighted.sum(dim=0) / count
    return base_f + lambda_ * merged_delta


class _MergeMethod(NamedTuple):
    """One merge method's contract: the op, the per-key tensors the loop hands it, and the scalar CLI
    knobs it reads with their defaults. The ops declare no defaults; the gate, the dispatch and
    ``--help`` all read them from this table."""

    op: Callable[..., torch.Tensor]
    tensor_args: tuple[str, ...]
    knobs: dict[str, float]


_METHODS: dict[str, _MergeMethod] = {
    "linear": _MergeMethod(_merge_linear, ("tensors", "weights"), {}),
    "slerp": _MergeMethod(_merge_slerp, ("t0", "t1"), {"t": 0.5}),
    "task_arithmetic": _MergeMethod(_merge_task_arithmetic, ("base", "tensors", "weights"), {}),
    "ties": _MergeMethod(_merge_ties, ("base", "tensors", "weights"), {"density": 0.6, "lambda": 1.0}),
}
# Per-key tensor arguments the caller controls through a CLI knob rather than a value.
_KNOB_BY_TENSOR_ARG = {"weights": "models:weight", "base": "base_model"}
# Effective values for consumed knobs the caller left unset (None = unset, so explicitness shows).
_KNOB_DEFAULTS: dict[str, float] = {knob: default for m in _METHODS.values() for knob, default in m.knobs.items()}


def _knob_dest(knob: str) -> str:
    """The Python identifier a knob is passed under — ``lambda`` → ``lambda_`` (PEP 8's escape)."""
    return f"{knob}_" if keyword.iskeyword(knob) else knob


def _method_knobs(method: str) -> frozenset[str]:
    """Every knob ``method`` actually reads, scalar and tensor-valued alike."""
    spec = _METHODS[method]
    tensor_knobs = {_KNOB_BY_TENSOR_ARG[name] for name in spec.tensor_args if name in _KNOB_BY_TENSOR_ARG}
    return frozenset(tensor_knobs | set(spec.knobs))


def _check_method_knobs(method: str, explicit: set[str]) -> None:
    """Refuse any explicitly-passed knob the chosen method never reads.

    An ignored knob (``--models a:0.1 b:0.9`` under slerp, ``--t`` under linear) would merge
    something other than what was asked.
    """
    knobs = _method_knobs(method)
    ignored = sorted(explicit - knobs)
    if ignored:
        raise ValueError(
            f"--method {method} does not use {', '.join(ignored)} — it reads only "
            f"{', '.join(sorted(knobs))}. Drop the knob(s) or pick the method "
            f"that consumes them; ignoring them would merge something other than what was asked."
        )


def merge_models(
    model_specs: list[str],
    output_dir: str,
    method: str,
    base_model: str | None = None,
    dtype: str = "bfloat16",
    knobs: dict[str, float] | None = None,
    tokenizer_source: str | None = None,
    max_shard_size: str = DEFAULT_MAX_SHARD_SIZE,
    allow_missing_tokenizer: bool = False,
    trust_remote_code: bool = True,
    verbose: bool = True,
) -> None:
    """Merge ``model_specs`` into ``output_dir``. ``knobs`` carries the scalar per-method settings
    (``t`` / ``density`` / ``lambda``) the chosen method's op declares; anything the method does not
    read is refused rather than ignored."""
    out_dtype = DTYPE_BY_NAME[dtype]
    parsed = [_parse_model_spec(s) for s in model_specs]
    paths = [p for p, _ in parsed]
    given_weights = [w for _, w in parsed]

    given_knobs = {name: value for name, value in (knobs or {}).items() if value is not None}
    unknown = sorted(set(given_knobs) - set(_KNOB_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown merge knob(s) {unknown}; the merge ops declare {sorted(_KNOB_DEFAULTS)}")
    # Every knob the caller set has to be one the method consumes; refused before any I/O.
    explicit = set(given_knobs) | ({"base_model"} if base_model is not None else set())
    if any(w is not None for w in given_weights):
        explicit.add("models:weight")
    _check_method_knobs(method, explicit)
    weights = [1.0 if w is None else w for w in given_weights]
    knob_values = {**_KNOB_DEFAULTS, **given_knobs}

    if method == "slerp" and len(paths) != 2:
        raise ValueError(f"slerp merges exactly two models, got {len(paths)}")
    if method in ("task_arithmetic", "ties") and not base_model:
        raise ValueError(f"{method} requires --base_model (the model the task vectors are relative to)")
    if len(paths) < 2 and method != "slerp":
        raise ValueError("need at least two --models to merge")

    # Every directory this merge reads from: writing the result into one of them would delete the
    # source shards it does not overwrite (see reject_in_place_conversion).
    for source in [*paths, base_model, tokenizer_source]:
        if source and os.path.isdir(source):
            reject_in_place_conversion(source, output_dir)
    readers = [_TensorReader(p) for p in paths]
    base_reader = _TensorReader(base_model) if base_model else None

    # Reference key set to merge over; every contributing model must then provide each key/shape.
    ref_keys = _reference_keys(method, readers, base_reader)

    # Symmetric coverage: the per-key loop raises on a reference key a model lacks, but an extra key
    # is never visited. The realistic case is an untied head, since reconcile_tie_word_embeddings flips
    # tie_word_embeddings only for the model whose lm_head diverged, so the merge would drop it.
    reference = set(ref_keys)
    for reader in [*readers, *([base_reader] if base_reader is not None else [])]:
        extra = sorted(reader.keys() - reference)
        if extra:
            raise ValueError(
                f"{reader.model_dir} carries {len(extra)} tensor(s) absent from the merge's reference "
                f"key set, which would be silently dropped from the merged checkpoint: {extra[:5]}"
                f"{' …' if len(extra) > 5 else ''}. The models do not share a parameter layout — a "
                f"common cause is one being saved untied (lm_head.weight present) and another tied. "
                f"Merge checkpoints with identical key sets."
            )

    # Created past the last input gate, so a refused merge leaves no directory behind.
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        logger.info(f"Merging {len(paths)} models via '{method}' over {len(ref_keys)} tensors → {output_dir}")

    # Dispatch by registry, and pass each op exactly the arguments the method table declares for it.
    spec = _METHODS[method]
    op_knobs = {_knob_dest(knob): knob_values[knob] for knob in spec.knobs}

    writer = StageShardWriter(output_dir, HF_STREAM_PART_PREFIX, max_shard_size, enabled=True)

    # Peak RAM is about one fp32 copy of the largest tensor per contributing model (TIES stacks them
    # all) on top of the writer's pending output shard. On disk the artifact is one input's size.
    contributors = len(readers) + (1 if base_reader is not None else 0)
    largest_tensor = max((readers[0].nbytes(key) for key in ref_keys), default=0)
    preflight_resource_warning(
        "merge_models",
        output_dir,
        disk_bytes=sum(os.path.getsize(shard) for shard in set(readers[0].weight_map.values())),
        ram_bytes=2 * contributors * largest_tensor + writer.max_bytes,
    )

    for i, key in enumerate(ref_keys):
        tensors = [r.get(key) for r in readers]
        shapes = {tuple(x.shape) for x in tensors}
        if len(shapes) != 1:
            raise ValueError(f"shape mismatch for {key!r}: {shapes}")

        # Integer/bool entries are structure, not weights (DeepSeek-V4's int64 tid2eid, expert-count
        # buffers), and averaging them into bf16 would corrupt the model. Pass through when they
        # agree, else refuse.
        if not tensors[0].is_floating_point():
            if any(not torch.equal(t, tensors[0]) for t in tensors[1:]):
                raise ValueError(
                    f"{key!r} is a non-floating-point tensor ({tensors[0].dtype}) that differs "
                    f"between the models — it encodes structure (e.g. a routing table), and there is "
                    f"no meaningful average. Merge models that share it."
                )
            writer.add(key, tensors[0])
            continue

        per_key = {"tensors": tensors, "t0": tensors[0], "t1": tensors[-1], "weights": weights}
        if base_reader is not None:
            per_key["base"] = base_reader.get(key)
        out = spec.op(**{name: per_key[name] for name in spec.tensor_args}, **op_knobs)

        # Balancing tensors export at their trained dtype as on every merge/gather path (mirrors
        # merge_ep_shards): casting the fp32 sign-update biases to --dtype quantizes the routing.
        writer.add(key, out.to(tensors[0].dtype if is_balancing_state_key(key) else out_dtype))
        del out, tensors
        if verbose and (i + 1) % 100 == 0:
            logger.info(f"  merged {i + 1}/{len(ref_keys)} tensors")

    writer.close_as_hf_checkpoint()
    if verbose:
        logger.info(f"  wrote merged weights to {output_dir}")
    _copy_aux_files(
        tokenizer_source or base_model or paths[0],
        output_dir,
        dtype,
        verbose,
        allow_missing_tokenizer=allow_missing_tokenizer,
        trust_remote_code=trust_remote_code,
    )
    logger.info(f"✓ Merged model saved to {output_dir}")


def _copy_aux_files(
    source_dir: str,
    output_dir: str,
    dtype: str,
    verbose: bool,
    *,
    allow_missing_tokenizer: bool,
    trust_remote_code: bool,
) -> None:
    """Reproduce ``source_dir``'s non-weight files (config, tokenizer, processor, …) in ``output_dir``.

    The merged model is the same architecture as the source, so all non-weight metadata is copied
    verbatim, including the processor files for multimodal models (``preprocessor_config.json``,
    ``processor_config.json``, ``chat_template.*``), ``generation_config.json`` and a remote-code
    family's ``.py`` modules, none of which a config plus tokenizer re-save emits. A Hub id is
    downloaded (weights excluded) and copied the same way: re-saving its config instead would write
    the source's ``auto_map`` while shipping none of the modules it names, leaving a directory that
    raises ``does not appear to have a file named modeling_<x>.py``. The config's ``dtype`` is then
    set to the merged dtype so ``from_pretrained`` loads it consistently.
    """
    # A hub --tokenizer_source may ship legacy .pth/.gguf/.h5/.tflite/.ot exports and a vendor weight
    # dump (``original/``), tens of GB that config and tokenizer resolution never needs. The same
    # tuple copy_checkpoint_aux_files refuses, so downloaded and local sources drop alike.
    if not os.path.isdir(source_dir):
        source_dir = snapshot_download(source_dir, ignore_patterns=list(WEIGHT_FILE_IGNORE_PATTERNS))
    # No resume sidecars (rng_state*, scheduler.pt, router_balancing_biases.pt): they describe a
    # single training run's state, and an N-way merged artifact has none, so shipping one input's
    # sidecars would seed a resume or serve with state the merged weights never trained under.
    copy_checkpoint_aux_files(source_dir, output_dir, include_resume_sidecars=False)
    if not any(name.startswith(("tokenizer", "vocab", "spiece", "merges")) for name in os.listdir(output_dir)):
        # Raise rather than warn: a merged artifact with no usable tokenizer would fail at the next
        # pipeline stage's load instead.
        if not allow_missing_tokenizer:
            raise RuntimeError(
                f"{source_dir} shipped no tokenizer files, so the merged checkpoint in {output_dir} "
                f"has none and every from_pretrained-based consumer will fail to build a tokenizer. "
                f"The merged weights are already written — re-run with --tokenizer_source pointing at "
                f"a directory or Hub id that carries the tokenizer (it is copied verbatim), or pass "
                f"--allow_missing_tokenizer if a tokenizer-less artifact is genuinely what you want."
            )
        logger.warning(f"{source_dir} shipped no tokenizer files (--allow_missing_tokenizer).")
    if verbose:
        logger.info(f"  copied config/tokenizer/processor files from {source_dir}")

    # Keep the saved config's dtype consistent with the merged weights, and write the canonical torch
    # name: `--dtype bf16` is accepted but `torch.bf16` does not exist, so a raw write would fail
    # every from_pretrained.
    config = AutoConfig.from_pretrained(output_dir, trust_remote_code=trust_remote_code)
    config.dtype = str(DTYPE_BY_NAME[dtype]).removeprefix("torch.")
    config.save_pretrained(output_dir)
    # The merged directory carries the same config contract as any other export: the merge tools and
    # the pinned servers read a family-less vendor config (Bailing/Ling) and a source-schema family
    # (Step-3.7) through those rewrites alone.
    finalize_exported_config(config, output_dir, source=source_dir)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Merge same-architecture checkpoints (linear / slerp / task_arithmetic / ties).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        help="Local model directories, each optionally 'path:weight'. Not Hub ids — weights are read off disk.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--method", default="linear", choices=list(_METHODS))
    parser.add_argument("--base_model", default=None, help="Base for task_arithmetic / ties.")
    parser.add_argument("--dtype", default="bfloat16", choices=list(DTYPE_BY_NAME))
    # Knob defaults stay None here and resolve in merge_models(), so a knob a method ignores raises;
    # the help text renders each op signature's own default rather than a copy of it.
    parser.add_argument(
        "--density",
        type=float,
        default=None,
        help=f"ties: fraction of deltas kept (default {_KNOB_DEFAULTS['density']}).",
    )
    parser.add_argument(
        "--lambda",
        dest=_knob_dest("lambda"),
        type=float,
        default=None,
        help=f"ties: merged-delta scale (default {_KNOB_DEFAULTS['lambda']}).",
    )
    parser.add_argument(
        "--t",
        type=float,
        default=None,
        help=f"slerp: interpolation factor in [0,1] (default {_KNOB_DEFAULTS['t']}).",
    )
    parser.add_argument(
        "--tokenizer_source",
        default=None,
        help="Where to copy config+tokenizer+remote-code modules from (local dir or Hub id).",
    )
    add_max_shard_size_arg(parser)
    parser.add_argument(
        "--allow_missing_tokenizer",
        action="store_true",
        help="Accept a merged checkpoint whose --tokenizer_source ships no tokenizer files (refused by default: "
        "from_pretrained-based consumers cannot load it).",
    )
    add_trust_remote_code_arg(parser)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    merge_models(
        model_specs=args.models,
        output_dir=args.output_dir,
        method=args.method,
        base_model=args.base_model,
        dtype=args.dtype,
        knobs={knob: getattr(args, _knob_dest(knob)) for knob in _KNOB_DEFAULTS},
        tokenizer_source=args.tokenizer_source,
        max_shard_size=args.max_shard_size,
        allow_missing_tokenizer=args.allow_missing_tokenizer,
        trust_remote_code=args.trust_remote_code,
        verbose=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
