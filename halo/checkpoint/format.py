"""Every on-disk checkpoint name two modules share, and the builders that spell them.

One spelling per artifact: a typo degrades silently (a weightless checkpoint, an LR schedule re-warming
from step 0, an orphaned shard).
"""

from __future__ import annotations

import re
from fnmatch import fnmatch

from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import IMAGE_PROCESSOR_NAME, PROCESSOR_NAME, VIDEO_PROCESSOR_NAME

from halo.checkpoint.atomic import WITHHELD_FILE_SUFFIX

# Per-file cap for gathered safetensors saves, shared by every save path and the arg default.
DEFAULT_MAX_SHARD_SIZE = "5GB"

SAFETENSORS_INDEX_FILE = "model.safetensors.index.json"
# The marker every safetensors writer stamps, without which readers do not resolve torch tensors.
SAFETENSORS_METADATA = {"format": "pt"}
# The index ``metadata.format`` stamp of a per-rank EP save, and the key infix of one rank's expert slice.
EP_SHARDED_FORMAT = "ep_sharded"
EP_SHARD_KEY_INFIX = ".shard_"
EP_SHARD_KEY_RE = re.compile(rf"(.+)\{EP_SHARD_KEY_INFIX}(\d+)$")
# HF's shard filename pattern and the spellings derived from it, so a writer and a sweep cannot disagree.
SAFETENSORS_SHARD_PATTERN = "model{suffix}.safetensors"
SAFETENSORS_WEIGHTS_FILE = SAFETENSORS_SHARD_PATTERN.format(suffix="")
SAFETENSORS_FAMILY_GLOB = SAFETENSORS_SHARD_PATTERN.format(suffix="*")
_EP_SHARD_FILE_RE = re.compile(
    "^" + re.escape(SAFETENSORS_SHARD_PATTERN).replace(re.escape("{suffix}"), r"-\d+-of-\d+") + "$"
)
# A renaming writer's in-flight parts: outside HF's final pattern, inside the sweep's glob.
STREAM_PART_PREFIX = "model-streaming"
# A PP checkpoint's part names. Each stage picks its own prefix without coordinating with the other writers: a
# global "k-of-n" counter is not derivable locally once a stage emits a variable number of parts.
PP_WRAPPER_PART_PREFIX = "model-wrapper"
# A pipeline stage's part-name prefix (:func:`pp_stage_part_prefix`) and its matcher: group 1 is the stage,
# group 2 the stage count.
_PP_STAGE_PART = "model-pp{}-of-{}"
PP_STAGE_PART_RE = re.compile(_PP_STAGE_PART.format(r"(\d+)", r"(\d+)"))
LEGACY_WEIGHTS_FILE = "pytorch_model.bin"
# Every filename that makes a directory a whole-model checkpoint, in the layout cascade's order.
WHOLE_MODEL_WEIGHT_FILES = (SAFETENSORS_INDEX_FILE, SAFETENSORS_WEIGHTS_FILE, LEGACY_WEIGHTS_FILE)

# HF Trainer's own filenames, spelled here because ``transformers.trainer`` is a ~1 s import.
TRAINER_STATE_FILE = "trainer_state.json"
# Where the base save's trainer_state.json waits until every other file of the checkpoint is on disk.
UNCOMMITTED_TRAINER_STATE_FILE = f".{TRAINER_STATE_FILE}{WITHHELD_FILE_SUFFIX}"
SCHEDULER_STATE_FILE = "scheduler.pt"
# HF Trainer's replicated optimizer state, which the sharded modes deliberately replace.
OPTIMIZER_STATE_FILES = ("optimizer.pt", "optimizer.bin")
# Written after every rank's optimizer shard, so its presence vouches for a complete shard set.
OPTIMIZER_META_FILE = "optimizer_meta.pt"
# The warm-restart remedies name the per-rank optimizer shards by this glob, so every filename derives from it.
OPTIMIZER_SHARD_GLOB = "optimizer_shard_*.pt"
ROUTER_BALANCING_BIASES_FILE = "router_balancing_biases.pt"
REFERENCE_LOGPS_FILE = "reference_logps.pt"
# An output_dir entry the run keeps beside its checkpoints and never exports, like the Trainer's Hub push.
RUN_INTERNAL_PREFIX = "_"
REFERENCE_CACHE_DIR_NAME = f"{RUN_INTERNAL_PREFIX}reference_cache"
INCOMPLETE_CHECKPOINTS_DIR_NAME = f"{RUN_INTERNAL_PREFIX}incomplete_checkpoints"
# The step directories HF's own ``get_last_checkpoint`` recognizes, and no others.
CHECKPOINT_STEP_DIR_RE = re.compile(rf"^{PREFIX_CHECKPOINT_DIR}-(\d+)$")
PREFETCH_PENDING_PREFIX = "prefetch_pending"

# PEFT's adapter filenames, spelled here because ``peft`` is a multi-second import; weights in
# load-preference order.
ADAPTER_SAFETENSORS_FILE = "adapter_model.safetensors"
ADAPTER_BIN_FILE = "adapter_model.bin"
ADAPTER_CONFIG_FILE = "adapter_config.json"
ADAPTER_WEIGHT_NAMES = (ADAPTER_SAFETENSORS_FILE, ADAPTER_BIN_FILE)
# Where a merge keeps the adapter's config: in the root it would make a merged model read as an adapter.
MERGED_ADAPTER_CONFIG_DIR = "original_adapter_config"
# A sidecar, never adapter_config.json, so stock PEFT loads the adapter unchanged.
TRAINING_PROVENANCE_FILE = "training_provenance.json"
# The training-state records an exported artifact carries beside its weights.
EXPORT_SIDECAR_FILES = (ROUTER_BALANCING_BIASES_FILE, TRAINING_PROVENANCE_FILE)
PROCESSOR_FILES = (PROCESSOR_NAME, IMAGE_PROCESSOR_NAME, VIDEO_PROCESSOR_NAME)
# A merged checkpoint's unmerged resume adapter lives in a subdirectory, since a root one would load on
# top of weights already holding its delta; the marker's presence alone is the resume verdict.
RESUME_ADAPTER_DIR = "resume_adapter"
RESUME_ADAPTER_MARKER_FILE = "resume_adapter.json"
# The files that vouch for a checkpoint, each written last in its own half: the trainer state (and its
# uncommitted spelling), the meta that vouches for the optimizer shard set, and the resume adapter marker.
CHECKPOINT_COMPLETION_MARKERS = (
    TRAINER_STATE_FILE,
    UNCOMMITTED_TRAINER_STATE_FILE,
    OPTIMIZER_META_FILE,
    RESUME_ADAPTER_MARKER_FILE,
)
# Resume state the aux copy drops by name when it drops the resume sidecars.
UNSUFFIXED_RESUME_ENTRIES = (TRAINER_STATE_FILE, RESUME_ADAPTER_DIR, RESUME_ADAPTER_MARKER_FILE)

# Never carried over by the aux copy: a stray weight file would shadow the fresh safetensors.
WEIGHT_FILE_SUFFIXES = (".safetensors", ".bin", ".pt")
# Foreign-framework exports, which the aux copy skips.
FOREIGN_EXPORT_SUFFIXES = (".pth", ".gguf", ".h5", ".msgpack", ".onnx", ".onnx_data", ".tflite", ".ot", ".mlmodel")
# Resume state exempt from the weight skip, by name and by prefix: losing one restarts the LR schedule,
# zeroes the router biases, re-draws every shuffle or skips a pending prefetch batch.
RESUME_SIDECAR_FILES = (SCHEDULER_STATE_FILE, ROUTER_BALANCING_BIASES_FILE, REFERENCE_LOGPS_FILE)
RESUME_SIDECAR_PREFIXES = ("rng_state", PREFETCH_PENDING_PREFIX)
# Vendor dumps of the same weights (gpt-oss ``original/``, ``metal/``); exact names, since
# ``original_adapter_config/`` must be kept.
WEIGHT_DUMP_DIRS = ("original", "consolidated", "metal")
# The dumps as Hub-download patterns: a checkpoint tool's fetch skips them as the aux copy does.
WEIGHT_DUMP_IGNORE_PATTERNS = tuple(f"{name}/*" for name in WEIGHT_DUMP_DIRS)


def ep_shard_filename(rank: int, world_size: int) -> str:
    """Filename of one rank's slice of a per-rank EP save: 0-based and always suffixed, since an
    unsuffixed ``model.safetensors`` reads as a whole model."""
    return SAFETENSORS_SHARD_PATTERN.format(suffix=f"-{rank:05d}-of-{world_size:05d}")


def optimizer_shard_filename(rank: int) -> str:
    """Filename of one rank's optimizer-state shard."""
    return OPTIMIZER_SHARD_GLOB.replace("*", f"{rank:05d}")


def prefetch_pending_filename(rank: int, world_size: int) -> str:
    """Filename of one rank's pending prefetch rounds; the world size in it keeps a resume at another
    width from reading another layout's prompts."""
    return f"{PREFETCH_PENDING_PREFIX}-{rank:05d}-of-{world_size:05d}.pt"


def is_per_rank_resume_file(name: str) -> bool:
    """Whether ``name`` is one rank's own resume file of a save: its optimizer shard, RNG state or pending
    prefetch rounds."""
    return fnmatch(name, OPTIMIZER_SHARD_GLOB) or name.startswith(RESUME_SIDECAR_PREFIXES)


def is_ep_shard(name: str) -> bool:
    """Whether ``name`` is a per-rank EP save slice (never a sibling adapter or a stale single file)."""
    return _EP_SHARD_FILE_RE.match(name) is not None


def hf_shard_filename(part: int, total_parts: int, *, indexed: bool = False) -> str:
    """HF's own name for part ``part`` of ``total_parts``: the unsharded file when there is only one and no index
    names other files beside it, since ``from_pretrained`` reads a ``model.safetensors`` over an index."""
    suffix = "" if total_parts == 1 and not indexed else f"-{part:05d}-of-{total_parts:05d}"
    return SAFETENSORS_SHARD_PATTERN.format(suffix=suffix)


def stream_part_filename(prefix: str, part_index: int) -> str:
    """Name of a renaming writer's in-flight part ``part_index`` under ``prefix``."""
    return f"{prefix}-{part_index:05d}.safetensors"


def pp_stage_part_prefix(pp_rank: int, pp_size: int) -> str:
    """The part-name prefix of pipeline stage ``pp_rank``'s checkpoint shards."""
    return _PP_STAGE_PART.format(f"{pp_rank:05d}", f"{pp_size:05d}")


def index_declares_ep_shards(index: dict) -> bool:
    """Whether a parsed safetensors index marks a per-rank EP save (a gathered or PP index carries no
    marker)."""
    meta = index.get("metadata", {}) or {}
    return meta.get("format") == EP_SHARDED_FORMAT


def ep_sharded_index_metadata(total_size: int, ep_size: int) -> dict:
    """The index ``metadata`` block a per-rank EP save writes, carrying the :data:`EP_SHARDED_FORMAT` marker."""
    return {"total_size": total_size, "ep_size": ep_size, "format": EP_SHARDED_FORMAT}
