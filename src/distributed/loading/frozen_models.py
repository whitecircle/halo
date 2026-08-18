"""Frozen auxiliary models: the teacher / reference copies a run loads beside its policy.

They are loaded here, not through :func:`~src.distributed.loading.model_loading.load_distributed_model`,
because they are never sharded — but they must mirror the policy's load in every way that changes a
logprob: the same attention validator against the same request, the same GptOss sink policy, the same
pinned revision. The parallelism gates live here too, since which shapes can hold a separate dense
reference at all is a ParallelismConfig question.
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

    Every step exists to keep that model numerically comparable to the policy it is paired with, because
    each of its logprobs is one half of the objective (a DPO logratio, a distillation target) and a
    mismatch biases the loss rather than crashing:

    * ``revision`` reaches the CONFIG fetch as well as the weight fetch — a pinned checkpoint paired
      with hub-main's config is a second way to load the wrong model. It is this model's OWN pin: a
      teacher and its student are normally different repos, so the policy's commit would 404 here.
    * the backend is resolved by the same :func:`resolve_attn_implementation` the policy loader runs,
      against THIS model's config and the run's dtype, so the per-family limits (DeepSeek-V4
      eager-only, Gemma4's head_dim-512, fp32 vs FlashAttention) are applied to the model being
      loaded. Its trailing validator alone is not enough — auto-detection is the widest gap
      (``or "sdpa"`` pins a reference to SDPA while the policy takes FA4 on Blackwell).
    * the sinks policy is applied HERE and not by the caller, because ``sinks_reset=True`` is what
      APPROVES a sink-dropping backend: resolving under that premise and then skipping the reset
      leaves a GptOss model running sdpa over live sinks, shifting every logprob by nats silently.

    * the checkpoint-coverage gate keeps the task head (``excuse_task_head=False``): this model is
      SCORED, never trained, so an absent head is not the "architecture adds one on top of a base
      checkpoint" case the excuse exists for — it is a randomly initialized head producing plausible
      numbers on one side of the objective.

    ``is_vlm`` pins ``AutoModelForImageTextToText`` rather than resolving the class from the config;
    ``download_tag`` coordinates the hub fetch main-rank-first (every rank must reach it equally often);
    ``device_map`` is forwarded to the weight load.
    """
    # Before the CONFIG fetch, which is already enough to import a remote modeling file: a frozen
    # teacher preloaded ahead of the policy would otherwise be the first remote-code load of the
    # process and reach none of the shims the policy loader applies.
    apply_remote_code_compat_shims()
    # The config fetch sits INSIDE the coordinated block: it is this path's first hub contact for the
    # repo (nothing pre-populates a teacher snapshot), and outside it every rank of every node would
    # hit the hub at once on a cold cache — the exact herd fs_aware_main_first exists to order.
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

    # An uninitialized inv_freq here biases every logprob this model scores, silently.
    finalize_loaded_model(model)
    finalize_run_model(
        model,
        config,
        sinks_policy=SinksPolicy.from_flags(reset_sinks=reset_sinks),
        attn_implementation=resolved_attn,
    )
    # Same FA4 warm-up the policy loader runs, and for the same reason: this model's first scoring
    # forward would otherwise JIT-compile its kernels mid-step (~10s) on whichever rank reaches it
    # first, while its peers run ahead into the next collective. OUTSIDE the coordinated block — the
    # warm-up barriers, and a collective inside a main-first body blocks the main rank alone against
    # peers sitting on a store key.
    warm_attention_kernels(model, dtype=dtype)
    return model


def place_and_freeze(auxiliary: PreTrainedModel, policy: torch.nn.Module) -> torch.device:
    """Move ``auxiliary`` onto ``policy``'s device, switch it to eval and freeze it. Returns the device.

    The scoring half of a paired objective (a distillation teacher, a self-distillation reference) is
    unparallelized, so all three steps belong together: on a different device its forward raises, in
    train mode its dropout perturbs every target, and with live gradients the optimizer would step it.
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

    Under PEFT the reference is the adapter-free base → ``None`` (asserting native EP expert-LoRA has
    ``precompute_ref_log_probs`` set, since grouped expert adapters can't be toggled). Full finetune loads
    an unparallelized copy. ``reset_sinks`` and ``attn_default`` must mirror the policy load: the reference
    resolves its attention implementation through the same validator, from the same request, and gets the
    same GptOss sink reset/freeze, so its logprobs are computed under the same kernel and the same sink
    semantics as the policy's.
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

    # PP belongs here for the same reason as EP/TP, and the trainer already refuses a live reference
    # under it (``reject_pp_ref_model``) — without this branch a pure-PP run (pp>1, ep_group_size==1,
    # tp_size==1) loads a full dense reference on EVERY rank and only then hits that refusal, so the
    # one supported shape (full finetune + precompute) costs a whole model load before it fails.
    if parallelism_config.is_ep_mode or parallelism_config.is_tp_mode or parallelism_config.is_pp_mode:
        # Precomputed log-probs come from the untrained policy — but a resume re-derives them trained.
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
        # An unpinned reference silently loads hub main and shifts every logratio.
        revision=getattr(model_config, "model_revision", None),
        trust_remote_code=model_config.trust_remote_code,
        # The policy's own request, resolved against the reference's config: every logratio is a
        # difference of two logprobs, so a kernel that differs between them biases the objective.
        # The fallback must be the policy's too — an unset config here would auto-detect (FA4 on
        # Blackwell) against a policy pinned to the caller's default.
        attn_implementation=model_config.attn_implementation or attn_default,
        reset_sinks=reset_sinks,
        is_vlm=is_vlm,
        # Coordinated: when the reference loads before the policy's snapshot, this is the repo's
        # first hub contact and every rank of every node would otherwise fetch at once.
        download_tag="reference_model",
    )
    setup_model_and_tokenizer(args, model_ref, tokenizer, embeddings_sharded=input_embeddings_tp_sharded)
    return model_ref
