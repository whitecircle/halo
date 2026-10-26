"""Weight-space merges of same-architecture checkpoints (linear, slerp, task arithmetic, TIES), streamed one
tensor at a time into a sharded safetensors checkpoint behind the key-coverage, knob and in-place gates."""

from __future__ import annotations

import keyword
import logging
import math
import os
import re
from collections.abc import Callable
from typing import NamedTuple

import torch
from safetensors import safe_open
from transformers import AutoConfig

from halo.checkpoint.config_export import finalize_exported_config
from halo.checkpoint.format import DEFAULT_MAX_SHARD_SIZE
from halo.checkpoint.offline_export import (
    copy_checkpoint_aux_files,
    preflight_resource_warning,
    reject_in_place_conversion,
)
from halo.checkpoint.reading import header_nbytes, header_numel, iter_checkpoint_shard_entries
from halo.checkpoint.writing import StreamingShardWriter, is_tie_reconcile_key
from halo.model_loading.dtype import DTYPE_BY_NAME
from halo.model_loading.model_preparation import resolve_auto_model_class
from halo.models.config_access import TIE_WORD_EMBEDDINGS_FLAG, config_sources
from halo.models.model_classes import mapped_model_class
from halo.models.moe.balancing import is_balancing_state_key

logger = logging.getLogger(__name__)

_SLERP_DOT_THRESHOLD = 0.9995  # above this the vectors are ~colinear → lerp (slerp is unstable)

# The range each scalar knob means something on. Outside it the ops do not raise, they merge something
# else: a density outside (0, 1] skips the trim and keeps every delta, a t outside [0, 1] extrapolates
# past both models, and a non-finite value writes non-finite weights.
KNOB_DOMAINS: dict[str, tuple[Callable[[float], bool], str]] = {
    "t": (lambda value: 0.0 <= value <= 1.0, "in [0, 1]"),
    "density": (lambda value: 0.0 < value <= 1.0, "in (0, 1]"),
    "lambda": (math.isfinite, "finite"),
}


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


class _TensorReader:
    """Lazy per-key tensor reader over a checkpoint's safetensors, caching open file handles."""

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        # Keys come from the shards themselves rather than the index's ``weight_map``, so a checkpoint
        # whose index has drifted from its files still merges the tensors that are actually there.
        self.weight_map = {key: shard for shard, _reader, key in iter_checkpoint_shard_entries(model_dir)}
        self._handles: dict[str, object] = {}

    def keys(self) -> set[str]:
        return set(self.weight_map)

    def get(self, key: str) -> torch.Tensor:
        return self._handle(key).get_tensor(key)

    def nbytes(self, key: str) -> int:
        """Storage size of ``key`` from the safetensors header alone — no tensor read."""
        return header_nbytes(self._handle(key).get_slice(key))

    def numel(self, key: str) -> int:
        """Element count of ``key`` from the safetensors header alone — no tensor read."""
        return header_numel(self._handle(key).get_slice(key))

    def shape(self, key: str) -> tuple[int, ...]:
        """Shape of ``key`` from the safetensors header alone — no tensor read."""
        return tuple(self._handle(key).get_slice(key).get_shape())

    def _handle(self, key: str):
        path = self.weight_map[key]
        handle = self._handles.get(path)
        if handle is None:
            handle = safe_open(path, framework="pt", device="cpu")
            self._handles[path] = handle
        return handle


def _reference_keys(readers: list[_TensorReader], base_reader: _TensorReader | None) -> list[str]:
    """The key set to merge over. ``task_arithmetic``/``ties`` task vectors are relative to the base,
    so iterate the BASE key set — iterating model[0] would silently drop any parameter present in the
    base (and other models) but absent from model[0] (e.g. an untied ``lm_head.weight``), yielding an
    incomplete, unloadable merged checkpoint. ``linear``/``slerp`` take no base (the knob gate
    refuses one) → use model[0]."""
    return sorted((base_reader if base_reader is not None else readers[0]).keys())


def _excused_base_keys(
    reference: list[str], readers: list[_TensorReader], config_source: str, *, trust_remote_code: bool
) -> set[str]:
    """The base keys no merged model stores and the shipped model never reads from a checkpoint.

    Under ``task_arithmetic``/``ties`` the reference is the base's key set, and a Hub base stores
    tensors its fine-tunes lack because their load rebuilt or dropped them. Judged against the config
    the merge ships (``config_source``'s) and the class it loads through, two kinds qualify: the tied
    head (:func:`_tied_head_key`), and keys the class drops at load
    (``_keys_to_ignore_on_load_unexpected``: Qwen3.5/3.6's ``mtp.*``, GLM-4.7-Flash's
    ``model.layers.47.*``), which every export of a fine-tune leaves out. The merge leaves them out too.
    """
    base_only = {key for key in reference if not any(key in reader.weight_map for reader in readers)}
    if not base_only:
        return set()
    config = AutoConfig.from_pretrained(config_source, trust_remote_code=trust_remote_code)
    auto_class = resolve_auto_model_class(config)
    model_class = mapped_model_class(config, auto_class) or auto_class
    patterns = getattr(model_class, "_keys_to_ignore_on_load_unexpected", None) or ()
    # Matched as transformers matches them: searched anywhere in the key.
    load_ignored = {key for key in base_only if any(re.search(pattern, key) for pattern in patterns)}
    return load_ignored | _tied_head_key(reference, base_only, readers, config)


def _tied_head_key(reference: list[str], base_only: set[str], readers: list[_TensorReader], config) -> set[str]:
    """The tied half of the base's head/embedding pair when the merged models rebuild it, else nothing.

    A save under ``tie_word_embeddings`` writes the embedding alone and transformers rebuilds the head
    from it at load, while some Hub checkpoints store both halves. A base of that kind carries a head
    none of its fine-tunes do, and that head is the merged embedding again rather than a missing
    tensor, provided every merged model stores the other half and the shipped config ties the pair at
    both of its levels: a text-only class reads the text sub-config's flag, a multimodal wrapper its
    own. The pair is the one ``reconcile_tie_word_embeddings`` compares.
    """
    pair = [key for key in reference if is_tie_reconcile_key(key)]
    tied_away = [key for key in pair if key in base_only]
    kept = [key for key in pair if all(key in reader.weight_map for reader in readers)]
    if len(pair) != 2 or len(tied_away) != 1 or len(kept) != 1:
        return set()
    if not all(getattr(level, TIE_WORD_EMBEDDINGS_FLAG, False) for level in config_sources(config)):
        return set()
    return set(tied_away)


def _check_key_coverage(reference: set[str], contributors: list[_TensorReader], excused: set[str]) -> None:
    """Refuse a contributor whose tensors do not match the merge's reference, before any output exists.

    A reference key a contributor lacks has nothing to merge from it, and a key it carries beyond the
    reference is never visited, so the merged checkpoint would silently drop it. A base key the
    shipped model rebuilds or drops at load (``excused``) is neither. A shape that differs between
    contributors has no merge either; read from the headers here rather than discovered mid-write.
    """
    for reader in contributors:
        missing = sorted(reference - reader.keys())
        extra = sorted(reader.keys() - reference - excused)
        if missing or extra:
            raise ValueError(
                f"{reader.model_dir} does not carry the merge's reference key set (the base's under "
                f"task_arithmetic/ties, else the first model's): it lacks {len(missing)} tensor(s) "
                f"{missing[:5]}, which have nothing to merge from it, and carries {len(extra)} more "
                f"{extra[:5]}, which the merged checkpoint would silently drop. A head saved untied "
                f"(lm_head.weight present) in one model and tied in another differs this way; a base "
                f"may store more than its fine-tunes only where the shipped config ties that head or "
                f"its model class drops the keys at load (_keys_to_ignore_on_load_unexpected). Merge "
                f"checkpoints of one parameter layout."
            )
    for key in sorted(reference):
        shapes = {reader.model_dir: reader.shape(key) for reader in contributors}
        if len(set(shapes.values())) > 1:
            raise ValueError(f"shape mismatch for {key!r} across the merged checkpoints: {shapes}")


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
    """One merge method's contract: the op, the per-key tensors the loop hands it, the scalar CLI
    knobs it reads with their defaults, and its working set for the RAM preflight — the peak float32
    copies of one tensor the op allocates beside its inputs, by model count. The ops declare no
    defaults; the gate, the dispatch and ``--help`` all read them from this table."""

    op: Callable[..., torch.Tensor]
    tensor_args: tuple[str, ...]
    knobs: dict[str, float]
    fp32_copies: Callable[[int], float]


# fp32_copies: the allocator's peak with bf16 inputs (every ``.float()`` copies), as test_merge_models pins it.
MERGE_METHODS: dict[str, _MergeMethod] = {
    "linear": _MergeMethod(_merge_linear, ("tensors", "weights"), {}, lambda n_models: 3),
    "slerp": _MergeMethod(_merge_slerp, ("t0", "t1"), {"t": 0.5}, lambda n_models: 5),
    "task_arithmetic": _MergeMethod(_merge_task_arithmetic, ("base", "tensors", "weights"), {}, lambda n_models: 4),
    "ties": _MergeMethod(
        _merge_ties,
        ("base", "tensors", "weights"),
        {"density": 0.6, "lambda": 1.0},
        lambda n_models: 5.25 * n_models + 5,
    ),
}
# Per-key tensor arguments the caller controls through a CLI knob rather than a value.
_KNOB_BY_TENSOR_ARG = {"weights": "models:weight", "base": "base_model"}
# Effective values for consumed knobs the caller left unset (None = unset, so explicitness shows).
KNOB_DEFAULTS: dict[str, float] = {knob: default for m in MERGE_METHODS.values() for knob, default in m.knobs.items()}


def knob_dest(knob: str) -> str:
    """The Python identifier a knob is passed under — ``lambda`` → ``lambda_`` (PEP 8's escape)."""
    return f"{knob}_" if keyword.iskeyword(knob) else knob


def _method_knobs(method: str) -> frozenset[str]:
    """Every knob ``method`` actually reads, scalar and tensor-valued alike."""
    spec = MERGE_METHODS[method]
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


def _check_knob_domains(method: str, values: dict[str, float]) -> None:
    """Refuse a knob value outside the range the method's op is meaningful on."""
    for knob in MERGE_METHODS[method].knobs:
        accepts, domain = KNOB_DOMAINS[knob]
        if not accepts(values[knob]):
            raise ValueError(f"--{knob} must be {domain} for --method {method}, got {values[knob]}")


def _validated_request(
    model_specs: list[str],
    method: str,
    base_model: str | None,
    knobs: dict[str, float] | None,
    tokenizer_source: str | None,
) -> tuple[list[str], list[float], dict[str, float]]:
    """``(model paths, their weights, every knob's value)`` for a merge request, refused before any I/O
    when a knob is unknown, unread by ``method`` or out of its domain, or when the sources do not fit it."""
    parsed = [_parse_model_spec(s) for s in model_specs]
    paths = [p for p, _ in parsed]
    given_weights = [w for _, w in parsed]

    given_knobs = {name: value for name, value in (knobs or {}).items() if value is not None}
    unknown = sorted(set(given_knobs) - set(KNOB_DEFAULTS))
    if unknown:
        raise ValueError(f"unknown merge knob(s) {unknown}; the merge ops declare {sorted(KNOB_DEFAULTS)}")
    # Every knob the caller set has to be one the method consumes.
    explicit = set(given_knobs) | ({"base_model"} if base_model is not None else set())
    if any(w is not None for w in given_weights):
        explicit.add("models:weight")
    _check_method_knobs(method, explicit)
    weights = [1.0 if w is None else w for w in given_weights]
    if not all(math.isfinite(w) for w in weights):
        raise ValueError(f"--models weights must be finite, got {weights}")
    knob_values = {**KNOB_DEFAULTS, **given_knobs}
    _check_knob_domains(method, knob_values)

    if method == "slerp" and len(paths) != 2:
        raise ValueError(f"slerp merges exactly two models, got {len(paths)}")
    if method in ("task_arithmetic", "ties") and not base_model:
        raise ValueError(f"{method} requires --base_model (the model the task vectors are relative to)")
    if len(paths) < 2 and method != "slerp":
        raise ValueError("need at least two --models to merge")
    # Every source is a local directory, so --trust_remote_code's local default holds for all of them: a
    # Hub repo here would run a third-party repo's own config code under a default meant for local files.
    if tokenizer_source is not None and not os.path.isdir(tokenizer_source):
        raise ValueError(
            f"--tokenizer_source {tokenizer_source!r} is not a local directory. Download the repo first "
            f"(huggingface-cli download {tokenizer_source} --exclude '*.safetensors' --local-dir <dir>) and "
            f"point --tokenizer_source at that directory."
        )
    return paths, weights, knob_values


def _stream_merged_tensors(
    ref_keys: list[str],
    readers: list[_TensorReader],
    base_reader: _TensorReader | None,
    method: str,
    weights: list[float],
    knob_values: dict[str, float],
    out_dtype: torch.dtype,
    writer: StreamingShardWriter,
    verbose: bool,
) -> None:
    """Merge ``ref_keys`` one at a time through ``method``'s op into ``writer``, so only one key's copies
    are resident."""
    # Dispatch by registry, and pass each op exactly the arguments the method table declares for it.
    spec = MERGE_METHODS[method]
    op_knobs = {knob_dest(knob): knob_values[knob] for knob in spec.knobs}
    for i, key in enumerate(ref_keys):
        tensors = [r.get(key) for r in readers]

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
            # Left bound, the other models' copies would sit beside the next key's reads.
            del tensors
            continue

        per_key = {"tensors": tensors, "t0": tensors[0], "t1": tensors[-1], "weights": weights}
        if base_reader is not None:
            per_key["base"] = base_reader.get(key)
        out = spec.op(**{name: per_key[name] for name in spec.tensor_args}, **op_knobs)

        # Balancing tensors export at their trained dtype as on every merge/gather path (mirrors
        # merge_ep_shards): casting the fp32 sign-update biases to --dtype quantizes the routing.
        writer.add(key, out.to(tensors[0].dtype if is_balancing_state_key(key) else out_dtype))
        # per_key holds the inputs too: left bound, they would sit beside the next key's reads.
        del out, tensors, per_key
        if verbose and (i + 1) % 100 == 0:
            logger.info(f"  merged {i + 1}/{len(ref_keys)} tensors")


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
    paths, weights, knob_values = _validated_request(model_specs, method, base_model, knobs, tokenizer_source)

    # Every directory this merge reads from: writing the result into one of them would delete the
    # source shards it does not overwrite (see reject_in_place_conversion).
    for source in [*paths, base_model, tokenizer_source]:
        if source and os.path.isdir(source):
            reject_in_place_conversion(source, output_dir)
    readers = [_TensorReader(p) for p in paths]
    base_reader = _TensorReader(base_model) if base_model else None
    contributors = [*readers, *([base_reader] if base_reader is not None else [])]

    # Reference key set to merge over, minus the base keys the shipped model rebuilds or drops at load;
    # every contributor must then provide exactly these keys, at one shape.
    aux_source = tokenizer_source or base_model or paths[0]
    ref_keys = _reference_keys(readers, base_reader)
    excused = _excused_base_keys(ref_keys, readers, aux_source, trust_remote_code=trust_remote_code)
    ref_keys = [key for key in ref_keys if key not in excused]
    _check_key_coverage(set(ref_keys), contributors, excused)

    # Created past the last input gate, so a refused merge leaves no directory behind.
    os.makedirs(output_dir, exist_ok=True)

    if verbose:
        logger.info(f"Merging {len(paths)} models via '{method}' over {len(ref_keys)} tensors → {output_dir}")

    writer = StreamingShardWriter(output_dir, max_shard_size, enabled=True)

    # Peak RAM is the costliest key, every contributor's copy as stored plus the method's fp32 working
    # set over it (an integer pass-through key counted as if merged), and the writer's pending output
    # shard. On disk the artifact is one input's size.
    working_bytes_per_element = MERGE_METHODS[method].fp32_copies(len(readers)) * torch.float32.itemsize
    costliest_key_bytes = max(
        (
            sum(reader.nbytes(key) for reader in contributors)
            + math.ceil(working_bytes_per_element * readers[0].numel(key))
            for key in ref_keys
        ),
        default=0,
    )
    preflight_resource_warning(
        "merge_models",
        output_dir,
        disk_bytes=sum(os.path.getsize(shard) for shard in set(readers[0].weight_map.values())),
        ram_bytes=costliest_key_bytes + writer.max_bytes,
    )

    _stream_merged_tensors(
        ref_keys, readers, base_reader, method, weights, knob_values, DTYPE_BY_NAME[dtype], writer, verbose
    )

    writer.close_as_hf_checkpoint()
    if verbose:
        logger.info(f"  wrote merged weights to {output_dir}")
    _copy_aux_files(
        aux_source,
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

    The merged model is the same architecture as the source, so all non-weight metadata is copied,
    including the processor files for multimodal models (``preprocessor_config.json``,
    ``processor_config.json``, ``chat_template.*``), ``generation_config.json`` and a remote-code
    family's ``.py`` modules, none of which a config plus tokenizer re-save emits: re-saving the config
    instead would write the source's ``auto_map`` while shipping none of the modules it names, leaving a
    directory that raises ``does not appear to have a file named modeling_<x>.py``. Two files are
    rewritten after the copy: the config's ``dtype`` is set to the merged dtype so ``from_pretrained``
    loads it consistently, and the ``README.md`` card gains the Halo Hub tag, or is created holding it.
    """
    # No resume state (include_resume_sidecars=False): it describes a single training run, and an N-way
    # merged artifact has none, so shipping one input's would seed a resume or serve with state the
    # merged weights never trained under.
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
