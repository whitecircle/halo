"""Loading for the frozen teacher / reference models a run holds beside its policy.

These models are never sharded, but their load must match the policy's in everything that changes a
logprob: attention backend, GptOss sink policy, pinned revision. The parallelism gates on holding a
separate dense reference are here too.
"""

import contextlib

import torch
from accelerate.logging import get_logger
from transformers import AutoConfig, AutoModelForImageTextToText, PreTrainedModel
from trl import ModelConfig

from src.distributed.filesystem import fs_aware_main_first
from src.distributed.loading.warmup import warm_attention_kernels
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.tensor_parallel.state_dict import input_embeddings_tp_sharded
from src.models.loading.checkpoint_coverage import from_pretrained_verified
from src.models.loading.dtype import resolve_training_dtype
from src.models.loading.model_preparation import (
    apply_family_attention_patches,
    auto_load_model,
    finalize_run_model,
)
from src.models.loading.tokenizer_setup import setup_model_and_tokenizer
from src.models.patches.attention import resolve_attn_implementation
from src.models.patches.buffer_fixes import finalize_loaded_model
from src.models.patches.gpt_oss_sinks import SinksPolicy
from src.models.patches.remote_code_compat import apply_remote_code_compat_shims

logger = get_logger(__name__)


def load_frozen_auxiliary_model(
    model_name_or_path: str,
    *,
    dtype: torch.dtype,
    revision: str | None = None,
    trust_remote_code: bool = False,
    attn_implementation: str | None = None,
    reset_sinks: bool = True,
    is_vlm: bool = False,
    device_map: dict | str | int | torch.device | None = None,
    download_tag: str | None = None,
) -> PreTrainedModel:
    """Load an unparallelized frozen model that scores the policy: a preference reference or a distillation teacher.

    Each logprob here is one half of the objective (a DPO logratio, a distillation target), so a
    mismatch against the policy biases the loss rather than raising. Hence:

    * ``revision`` pins the config fetch as well as the weight fetch, and is this model's own pin —
      a teacher and its student are normally different repos.
    * the backend is resolved by the same :func:`resolve_attn_implementation` the policy loader runs,
      against this model's config and the run's dtype, so the per-family limits (DeepSeek-V4
      eager-only, Gemma4 head_dim-512, fp32 vs FlashAttention) apply. Auto-detection is the widest
      gap: an unset request pins the reference to SDPA while the policy takes FA4 on Blackwell.
    * the sinks policy is applied here rather than by the caller, since ``reset_sinks=True`` is what
      permits a sink-dropping backend; skipping the reset leaves GptOss running sdpa over live sinks.
    * ``excuse_task_head=False`` keeps the coverage gate on the task head: this model is only scored,
      so an absent head means a randomly initialized one on one side of the objective.

    ``is_vlm`` pins ``AutoModelForImageTextToText`` rather than resolving the class from the config;
    ``download_tag`` coordinates the hub fetch main-rank-first (every rank must reach it equally often);
    ``device_map`` is forwarded to the weight load.
    """
    # Before the config fetch, which can already import a remote modeling file: a teacher preloaded
    # ahead of the policy would otherwise be the process's first remote-code load and miss the shims.
    apply_remote_code_compat_shims()
    # The config fetch is inside the coordinated block: it is this path's first hub contact for the
    # repo, so on a cold cache every rank of every node would otherwise hit the hub at once.
    with fs_aware_main_first(download_tag) if download_tag else contextlib.nullcontext():
        config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code, revision=revision)
        resolved_attn = resolve_attn_implementation(config, attn_implementation, dtype, sinks_reset=reset_sinks)
        apply_family_attention_patches(config, resolved_attn)
        load_kwargs = {
            "revision": revision,
            "dtype": dtype,
            "trust_remote_code": trust_remote_code,
            "attn_implementation": resolved_attn,
            "device_map": device_map,
            "excuse_task_head": False,
        }
        if is_vlm:
            model = from_pretrained_verified(AutoModelForImageTextToText, model_name_or_path, **load_kwargs)
        else:
            model = auto_load_model(model_name_or_path, **load_kwargs)

    # Repairs non-persistent buffers; an uninitialized inv_freq biases every logprob this model scores.
    finalize_loaded_model(model)
    finalize_run_model(
        model,
        config,
        sinks_policy=SinksPolicy.from_flags(reset_sinks=reset_sinks),
        attn_implementation=resolved_attn,
    )
    # Same FA4 warm-up as the policy loader: the first scoring forward would otherwise JIT-compile its
    # kernels mid-step on whichever rank reaches it first, while peers run ahead into the next
    # collective. Outside the coordinated block, since the warm-up itself barriers.
    warm_attention_kernels(model, dtype=dtype)
    return model


def place_and_freeze(auxiliary: PreTrainedModel, policy: torch.nn.Module) -> torch.device:
    """Move ``auxiliary`` onto ``policy``'s device, switch it to eval and freeze it. Returns the device.

    All three steps are needed together: on a different device the forward raises, in train mode
    dropout perturbs every target, and with live gradients the optimizer would step the model.
    """
    device = policy.device
    auxiliary.to(device)
    auxiliary.eval()
    for param in auxiliary.parameters():
        param.requires_grad = False
    return device


def load_reference_model_for_preference(
    args,
    model_config: ModelConfig,
    training_config,
    parallelism_config: ParallelismConfig,
    tokenizer,
    *,
    is_vlm: bool,
    method: str,
    reset_sinks: bool = True,
    attn_default: str | None = None,
):
    """Load the frozen reference model for a preference trainer (DPO/KTO), or ``None`` under PEFT.

    Under PEFT the reference is the adapter-free base → ``None`` (native EP expert-LoRA must set
    ``precompute_ref_log_probs``, since grouped expert adapters cannot be toggled). Full finetune loads
    an unparallelized copy. ``reset_sinks`` and ``attn_default`` must mirror the policy load so the
    reference's logprobs come from the same kernel and the same sink semantics.
    """
    if model_config.use_peft:
        if parallelism_config.expert_lora is not None and not training_config.precompute_ref_log_probs:
            raise ValueError(
                f"{method} with native EP expert-LoRA requires precompute_ref_log_probs=True: TRL "
                f"builds the reference by disabling adapters, but grouped expert adapters cannot be "
                f"toggled. Set precompute_ref_log_probs: true, or drop expert targets from "
                f"lora_target_modules."
            )
        return None

    # PP is included for the same reason as EP/TP: the trainer refuses a live reference under it
    # (``reject_pp_ref_model``), and without this branch a pure-PP run (pp>1, ep_group_size==1,
    # tp_size==1) loads a full dense reference on every rank before reaching that refusal.
    if parallelism_config.is_ep_mode or parallelism_config.is_tp_mode or parallelism_config.is_pp_mode:
        # Precomputed log-probs come from the untrained policy; a resume re-derives them trained.
        if training_config.precompute_ref_log_probs:
            return None
        raise ValueError(
            f"Full-finetune {method} under EP/TP/PP cannot hold a separate dense reference model. "
            f"Set precompute_ref_log_probs: true (reference log-probs are computed from the "
            f"untrained policy before the first step), or use PEFT (--use_peft)."
        )
    model_ref = load_frozen_auxiliary_model(
        model_config.model_name_or_path,
        dtype=resolve_training_dtype(training_config),
        # Unpinned, the reference loads hub main and shifts every logratio.
        revision=getattr(model_config, "model_revision", None),
        trust_remote_code=model_config.trust_remote_code,
        # The policy's own request, resolved against the reference's config: a logratio is a
        # difference of two logprobs, so a kernel differing between them biases the objective. The
        # fallback must be the policy's too, or an unset config auto-detects against a pinned policy.
        attn_implementation=model_config.attn_implementation or attn_default,
        reset_sinks=reset_sinks,
        is_vlm=is_vlm,
        # When the reference loads before the policy's snapshot this is the repo's first hub contact,
        # so every rank would otherwise fetch at once.
        download_tag="reference_model",
    )
    setup_model_and_tokenizer(args, model_ref, tokenizer, embeddings_sharded=input_embeddings_tp_sharded)
    return model_ref
