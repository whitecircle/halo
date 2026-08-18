"""Load MoE models for Expert Parallelism.

:func:`load_ep_model` is the entry point: resolves the checkpoint, detects the format, and dispatches
to the lazy safetensors loader (default, :mod:`.lazy_loader`) or the sequential ``from_pretrained`` +
EP-patch fallback (:func:`_load_ep_model_huggingface`). EP is orthogonal to DP (Megatron-LM style).
"""

import json
import os
import time
from collections.abc import Callable
from typing import TypeVar

import torch
from accelerate.logging import get_logger
from huggingface_hub import snapshot_download
from transformers import AutoModelForCausalLM

from src.checkpoint.format import EP_SHARDED_FORMAT, SAFETENSORS_INDEX_FILE
from src.distributed.expert_parallel.config import EPConfig, get_num_experts
from src.distributed.expert_parallel.lazy_loader import (
    lazy_loader_supports_checkpoint,
    load_ep_model_lazy,
)
from src.distributed.expert_parallel.patching import (
    create_ep_buffers,
    patch_moe_model_for_ep,
)
from src.distributed.filesystem import sequential_load_within_node
from src.distributed.runtime import (
    DeferredRankFailure,
    broadcast_from_rank0,
    get_global_rank,
    get_local_rank,
    is_global_main_process,
    log_global_load_duration_seconds,
    move_model_to_local_device,
)
from src.models.loading.checkpoint_coverage import from_pretrained_verified
from src.models.loading.lazy_safetensors.weights import (
    has_safetensors_checkpoint,
    resolve_run_dtype,
)
from src.models.patches.attention import revalidate_attn_kwarg
from src.models.patches.buffer_fixes import finalize_loaded_model

_T = TypeVar("_T")

logger = get_logger(__name__)


def decide_lazy_loadable(local_dir: str | None, layout_supported: Callable[[str], bool]) -> bool:
    """Whether EVERY rank takes the lazy safetensors path for ``local_dir`` — decided by rank 0.

    COLLECTIVE — every rank must call it. The lazy and fallback branches enter different store phases
    and different world collectives, while every input to the decision is a best-effort per-rank
    filesystem probe: a partially populated cache on a non-shared filesystem splits the verdict, and a
    split gate deadlocks with no diagnostic. So rank 0's answer is broadcast and nobody consults their
    own.

    ``layout_supported`` (whether the lazy fuser can materialize this family's expert layout) is a
    parameter, since the EP and PP loaders each answer it for their own layout. Short-circuited, so it
    is only asked about a directory that exists and holds safetensors.
    """
    return broadcast_from_rank0(
        local_dir is not None and has_safetensors_checkpoint(local_dir) and layout_supported(local_dir)
    )


def _detect_checkpoint_format(path: str) -> str:
    """Detect checkpoint format: ``ep_sharded`` (save_ep_model sharded=True) or ``huggingface``."""
    index_path = os.path.join(path, SAFETENSORS_INDEX_FILE)
    if os.path.exists(index_path):
        with open(index_path) as f:
            index = json.load(f)
        if index.get("metadata", {}).get("format") == EP_SHARDED_FORMAT:
            return EP_SHARDED_FORMAT

    return "huggingface"


def reject_ep_sharded_checkpoint(local_dir: str | None, model_name_or_path: str) -> None:
    """Refuse a per-rank EP-sharded checkpoint on EVERY rank, with the merge instructions.

    COLLECTIVE — every rank must call it, and only rank 0 probes. The probe is a filesystem read
    (``json.load`` of the safetensors index) sitting in front of world collectives, so it is fenced:
    a torn or half-fetched index on a non-shared filesystem would otherwise take rank 0 out while its
    peers block in the broadcast, and a split verdict sends ranks down different branches. An
    ep_sharded directory's per-rank keys match no expert pattern, so a loader that got past here
    would plan nothing and surface as "never materialized" instead of "merge the shards".
    """
    guard = DeferredRankFailure(f"EP checkpoint-format probe of {model_name_or_path!r}")
    local_verdict = None
    if is_global_main_process():
        local_verdict = guard.run(
            lambda: _detect_checkpoint_format(local_dir) if local_dir is not None else "huggingface"
        )
    verdict = broadcast_from_rank0(local_verdict)
    # After the broadcast, before anything branches on the verdict: a torn index would otherwise
    # take rank 0 out here while every peer blocks in the broadcast above.
    guard.reject()
    if verdict != EP_SHARDED_FORMAT:
        return
    if local_dir is None:
        raise RuntimeError(
            f"[Rank {get_global_rank()}] Global rank 0 found a per-rank EP-sharded checkpoint at "
            f"{model_name_or_path!r}, but this rank could not resolve it to a local directory — a "
            f"partially populated cache on a non-shared filesystem. Merge the shards with "
            f"scripts/after_training/merge_ep_shards.py and point every node at the result."
        )
    _load_ep_model_sharded(local_dir)  # always raises, with the merge instructions


def resolve_hub_or_local_dir(model_name_or_path: str, revision: str | None = None) -> str | None:
    """Resolve ``model_name_or_path`` to a local directory for disk-based EP loading.

    Local dirs → absolute paths; Hub repo ids → cache snapshot dir at ``revision`` (cache first,
    download if needed). Returns ``None`` if unresolvable — a cache miss is routine (INFO), a failed
    download is a network/auth/gated-repo problem that costs every rank the full checkpoint in CPU
    RAM (WARNING).
    """
    if os.path.isdir(model_name_or_path):
        return os.path.abspath(model_name_or_path)
    try:
        return snapshot_download(model_name_or_path, revision=revision, local_files_only=True)
    except Exception as e:
        logger.info("No cached snapshot for %r (%s); resolving from the Hub.", model_name_or_path, e)
    try:
        return snapshot_download(model_name_or_path, revision=revision)
    except Exception as e:
        logger.warning(
            "Could not resolve %r to a local snapshot (%s); skipping EP lazy loading via disk path — "
            "every rank will materialize the full checkpoint through from_pretrained.",
            model_name_or_path,
            e,
        )
        return None


def _timed_ep_load(method: str, load_fn: Callable[[], _T]) -> _T:
    """Run an EP load path and log global wall span (earliest start → latest end across ranks)."""
    t_start_wall = time.time()
    loaded = load_fn()
    # Deliberately not in a ``finally``: this is a collective, so running it while one rank is
    # unwinding an exception turns that rank's traceback into an NCCL timeout on all the others.
    log_global_load_duration_seconds(
        tag="EP",
        method=method,
        t_start_wall=t_start_wall,
        t_end_wall=time.time(),
    )
    return loaded


def load_ep_model(
    model_name_or_path: str,
    ep_config: EPConfig,
    config,
    dtype=None,
    trust_remote_code: bool = True,
    model_class=None,
    max_concurrent_loading: int | None = None,
    lazy: bool = True,
    revision: str | None = None,
    **model_kwargs,
) -> torch.nn.Module:
    """Load a MoE model for EP training.

    The checkpoint format (``huggingface`` vs ``ep_sharded``) is auto-detected. With
    ``lazy=True`` (default) and a resolvable safetensors checkpoint, each rank
    reads only its expert slice in parallel (:mod:`.lazy_loader`); otherwise falls
    back to per-rank ``from_pretrained`` + EP patch. ``model_name_or_path`` may be a
    Hub id (resolved to the cached snapshot dir), local path, or EP checkpoint dir. ``config`` is the
    caller's already-loaded model config, required so that no loader re-reads it per rank.
    """
    rank = get_global_rank()

    # Local snapshot dir needed for lazy I/O — must be pinned to the requested revision.
    local_dir = resolve_hub_or_local_dir(model_name_or_path, revision=revision)

    reject_ep_sharded_checkpoint(local_dir, model_name_or_path)

    if not lazy:
        logger.info(
            f"[Rank {rank}] EP lazy loading disabled — using HuggingFace from_pretrained + EP patch "
            f"(higher CPU RAM than lazy safetensors path)"
        )
        return _timed_ep_load(
            "huggingface_ep_lazy_disabled",
            lambda: _load_ep_model_huggingface(
                model_name_or_path,
                ep_config,
                config,
                dtype,
                trust_remote_code,
                model_class=model_class,
                max_concurrent_loading=max_concurrent_loading,
                revision=revision,
                **model_kwargs,
            ),
        )

    # Lazy first; HF fallback when there is no snapshot/safetensors or the expert layout is unmappable.
    use_lazy = decide_lazy_loadable(local_dir, lazy_loader_supports_checkpoint)

    if use_lazy:
        if local_dir is None:
            raise RuntimeError(
                f"[Rank {rank}] Global rank 0 resolved a lazy-loadable safetensors checkpoint for "
                f"{model_name_or_path!r} but this rank could not resolve it to a local directory — a "
                f"partially populated cache on a non-shared filesystem. Pre-download the checkpoint on "
                f"every node, or pass ep_lazy_loading=False."
            )
        logger.info(f"[Rank {rank}] Using lazy safetensors loader from {local_dir}")
        return _timed_ep_load(
            "lazy_safetensors",
            lambda: load_ep_model_lazy(
                local_dir,
                ep_config,
                config,
                dtype=dtype,
                trust_remote_code=trust_remote_code,
                model_class=model_class,
                **model_kwargs,
            ),
        )

    # Diagnostic only, and derived here rather than beside the gate so the probes cost nothing on the
    # lazy path. Rank-local: a rank that disagreed with rank 0's verdict reports its own reason. The
    # level separates the two kinds: an unmappable expert layout is a class-declared family property
    # (the HF path IS its supported route), while a missing snapshot or missing safetensors is an
    # environment problem that silently costs every rank the full checkpoint in CPU RAM.
    if local_dir is None:
        fallback_reason, log = "no_local_snapshot", logger.warning
    elif not has_safetensors_checkpoint(local_dir):
        fallback_reason, log = "no_safetensors", logger.warning
    else:
        fallback_reason, log = "lazy_incompatible_expert_layout", logger.info
    log(f"[Rank {rank}] Using HuggingFace from_pretrained + EP patch (reason={fallback_reason})")
    return _timed_ep_load(
        f"huggingface_ep_fallback_{fallback_reason}",
        lambda: _load_ep_model_huggingface(
            model_name_or_path,
            ep_config,
            config,
            dtype,
            trust_remote_code,
            model_class=model_class,
            max_concurrent_loading=max_concurrent_loading,
            revision=revision,
            **model_kwargs,
        ),
    )


def _load_ep_model_sharded(checkpoint_dir: str) -> torch.nn.Module:
    """Always raises: sharded EP checkpoints must be merged or saved gathered before loading."""
    raise NotImplementedError(
        f"Direct loading of sharded EP checkpoints is not supported.\n\n"
        f"Sharded checkpoints must be merged first using merge_ep_shards.py:\n\n"
        f"    python scripts/after_training/merge_ep_shards.py \\\n"
        f"        --input_dir {checkpoint_dir} \\\n"
        f"        --output_dir /path/to/merged_checkpoint\n\n"
        f"Then load the merged checkpoint:\n\n"
        f"    model = load_ep_model('/path/to/merged_checkpoint', ep_config)\n\n"
        f"Alternatively, use save_ep_model(sharded=False) to save in gathered format "
        f"which can be loaded directly without merging."
    )


def _load_ep_model_huggingface(
    model_name_or_path: str,
    ep_config: EPConfig,
    config,
    dtype=None,
    trust_remote_code: bool = True,
    model_class=None,
    max_concurrent_loading: int | None = None,
    revision: str | None = None,
    **model_kwargs,
) -> torch.nn.Module:
    """Load EP model from HuggingFace checkpoint.

    Files must be downloaded/cached first. ``device_map="cpu"`` keeps unconverted bf16 keys as
    safetensors mmap views (page cache, shared across ranks), but every conversion-mapping result —
    a fused ``gate_up_proj`` or a stacked per-expert bank — is an anonymous CPU tensor, so a converted
    MoE family stages its whole expert set per rank here; ``max_concurrent_loading`` bounds how many
    ranks do so at once.
    """
    if model_class is None:
        model_class = AutoModelForCausalLM

    rank = get_global_rank()
    logger.info(f"[Rank {rank}] Loading model for EP: {model_name_or_path}")

    num_experts = get_num_experts(config)
    logger.info(f"[Rank {rank}] Model has {num_experts} experts")

    if ep_config.num_experts is None:
        ep_config.finalize_expert_assignment(num_experts)

    dtype = resolve_run_dtype(dtype, config)

    revalidate_attn_kwarg(model_kwargs, config)

    # Bounded ranks per node at a time — an unbounded fan-in CPU-OOMs on large MoE.
    logger.info(f"[Rank {rank}] Waiting for sequential model loading (local_rank={get_local_rank()})...")
    with sequential_load_within_node(max_concurrent=max_concurrent_loading):
        logger.info(f"[Rank {rank}] Loading model to CPU...")
        model = from_pretrained_verified(
            model_class,
            model_name_or_path,
            config=config,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            device_map="cpu",
            revision=revision,
            **model_kwargs,
        )

        # Match the lazy loader's uniform parameter dtype: from_pretrained honors
        # _keep_in_fp32_modules[_strict] (Inkling pins its short convolutions in fp32) and FSDP2
        # rejects mixed-dtype parameters in one shard group. Parameters only, on both loaders —
        # casting buffers would downcast fp32 state families keep deliberately (Zaya's balancing
        # biases). The isinstance guard skips the "auto" spelling, which nn.Module.to reads as a
        # device.
        if isinstance(dtype, torch.dtype):
            for param in model.parameters():
                if param.is_floating_point() and param.dtype != dtype:
                    param.data = param.data.to(dtype)

        logger.info(f"[Rank {rank}] Applying EP patching...")
        model = patch_moe_model_for_ep(model, ep_config)
        model = move_model_to_local_device(model)

    # Collective — every EP rank must participate, and only once all ranks are loaded and on GPU.
    create_ep_buffers(model)

    # from_pretrained materializes only the keys the checkpoint carries, so non-persistent buffers
    # hold garbage and the tied lm_head shadow stays on meta; the bf16 cast above compounds it.
    # Repair after the device move.
    finalize_loaded_model(model)

    logger.info(f"[Rank {rank}] Model loaded successfully")
    return model
