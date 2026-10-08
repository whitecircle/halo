"""Loading for the frozen teacher / reference models a run holds beside its policy.

These models are never sharded, but their load must match the policy's in everything that changes a
logprob: attention backend, GptOss sink policy, pinned revision. The parallelism rules on holding a
separate dense reference are here too.
"""

import contextlib
import dataclasses
import logging

import torch
from accelerate.utils import is_peft_model
from transformers import AutoConfig, AutoModelForImageTextToText, PreTrainedModel
from trl import ModelConfig

from src.distributed.filesystem import joined_node_load
from src.distributed.loading.model_source import resolve_model_source
from src.distributed.loading.peft_setup import has_attention_lora_targets
from src.distributed.loading.warmup import warm_attention_kernels
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.tensor_parallel.state_dict import input_embeddings_tp_sharded
from src.log import warn_once
from src.models.loading.checkpoint_coverage import from_pretrained_verified
from src.models.loading.dtype import cast_parameters_to_run_dtype, resolve_training_dtype
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

# Stdlib, not accelerate's adapter: warn_once marks a message seen on every rank, so each rank must emit it.
logger = logging.getLogger(__name__)

# ``load_frozen_reference_model``'s default ``revision``: the policy's own pin.
_POLICY_REVISION = object()


@dataclasses.dataclass(frozen=True)
class ReferenceAlternatives:
    """A caller's ways around holding a frozen reference, named by every reference warning and refusal.

    ``peft`` is named only where an adapter can train: tensor parallelism refuses every LoRA shape
    (``_validate_lora_tp_compatibility``), so under TP the text is ``always`` alone.
    """

    always: str
    peft: str | None = None

    def under(self, parallelism_config: ParallelismConfig) -> str:
        """The alternatives this parallelism shape can run."""
        if self.peft is None or parallelism_config.is_tp_mode:
            return self.always
        return f"{self.always} {self.peft}"


# The DPO/KTO way around a dense reference, shared by the script loader and the trainer gate so the
# two report it once.
PREFERENCE_REFERENCE_ALTERNATIVES = ReferenceAlternatives(
    always="precompute_ref_log_probs: true scores the same reference once, from the untrained policy, exactly "
    "and with no second model.",
    peft="PEFT (use_peft: true, ref_model=None) is the other way around it.",
)
# The on-policy GRPO way around it, shared the same way.
ON_POLICY_GRPO_REFERENCE_ALTERNATIVES = ReferenceAlternatives(
    always="beta: 0 drops the KL term and its reference.",
    peft="use_peft: true with an attention target scores the reference as the policy with its adapter disabled.",
)

# Messages already reported: the script loader and the trainer gate report one reference once.
_REPORTED_REFERENCES: set[str] = set()


def warn_unparallelized_reference(parallelism_config: ParallelismConfig, avoid: ReferenceAlternatives) -> None:
    """Warn that a frozen reference beside a policy sharded by EP, ETP or TP is a whole dense replica per rank.

    Its log-probs are still the policy's function on the reference weights. The model's own MoE forward is
    what the EP layers are tested to reproduce (tests/gpu/parallelism/ep/test_ep_correctness.py), so the
    two differ by kernel numerics only, as they do under the grouped-GEMM wrapper at ``ep_size == 1``,
    which nothing refuses. The cost is memory and time: every rank holds the full model the parallelism
    shards, experts included, and runs the per-expert forward. ``avoid`` is the caller's way around it.
    """
    if parallelism_config.is_ep_mode or parallelism_config.is_tp_mode:
        message = (
            f"A frozen, unparallelized reference beside a policy sharded by EP/ETP/TP "
            f"(expert_parallel_size={parallelism_config.ep_size}, "
            f"expert_tensor_parallel_size={parallelism_config.expert_tp_size}, "
            f"tensor_parallel_size={parallelism_config.tp_size}) is a whole dense replica on every rank, "
            f"experts included, run through the model's own MoE forward: its log-probs match the policy's "
            f"up to kernel numerics, but budget for its memory and its slower forward. "
            f"{avoid.under(parallelism_config)}"
        )
        warn_once(logger, _REPORTED_REFERENCES, message, message)


def on_policy_grpo_holds_reference(beta: float, peft_config, model=None) -> bool:
    """TRL ``GRPOTrainer``'s own rule for holding a frozen KL reference: ``beta != 0`` on a policy no PEFT
    adapter wraps, neither through a ``peft_config`` nor as a ``PeftModel`` passed in. Native expert-only LoRA
    builds no ``PeftModel``, so it holds one. ``model`` may be a name or absent, which no adapter wraps."""
    wrapped = isinstance(model, torch.nn.Module) and is_peft_model(model)
    return beta != 0.0 and peft_config is None and not wrapped


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
    """Load an unparallelized frozen model that scores the policy: a reference or a distillation teacher.

    Each logprob here is one half of the objective (a DPO logratio, a distillation target), so a
    mismatch against the policy biases the loss rather than raising. Hence:

    * ``revision`` pins the config fetch as well as the weight fetch, and is this model's own pin —
      a teacher and its student are normally different repos.
    * the backend is resolved by the same :func:`resolve_attn_implementation` the policy loader runs,
      against this model's config and the run's dtype, so the per-family limits (DeepSeek-V4
      eager-only, Gemma4 head_dim-512, fp32 vs FlashAttention) apply, and the model is built with what
      the shared :func:`apply_family_attention_patches` returns for it (Gemma 4's ``sdpa_flex_sliding``).
      Auto-detection is the widest gap: an unset request pins the reference to SDPA while the policy
      takes FA4 on Blackwell.
    * the sinks policy is applied here rather than by the caller, since ``reset_sinks=True`` is what
      permits a sink-dropping backend; skipping the reset leaves GptOss running sdpa over live sinks.
    * every floating parameter takes ``dtype``, as in the policy loaders: a family's fp32-pinned
      modules would otherwise score in a different precision than the policy they anchor.
    * ``excuse_task_head=False`` keeps the coverage gate on the task head: this model is only scored,
      so an absent head means a randomly initialized one on one side of the objective.

    ``is_vlm`` pins ``AutoModelForImageTextToText`` rather than resolving the class from the config;
    ``download_tag`` fetches the source main-rank-first and agrees it across ranks before the first
    read (:func:`~src.distributed.loading.model_source.resolve_model_source`; every rank must reach
    it equally often, and an untagged load is a single-process one); ``device_map`` is forwarded to
    the weight load.
    """
    # Before the config fetch, which can already import a remote modeling file: a teacher preloaded
    # ahead of the policy would otherwise be the process's first remote-code load and miss the shims.
    apply_remote_code_compat_shims()
    # This path's first hub contact for the repo: on a cold cache every rank of every node would
    # otherwise hit the hub at once, and per-node caches could each hold a different commit.
    if download_tag:
        revision = resolve_model_source(model_name_or_path, revision, tag=download_tag)
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code, revision=revision)
    backend = resolve_attn_implementation(config, attn_implementation, dtype, sinks_reset=reset_sinks)
    resolved_attn = apply_family_attention_patches(config, backend)
    load_kwargs = {
        "revision": revision,
        "dtype": dtype,
        "trust_remote_code": trust_remote_code,
        "attn_implementation": resolved_attn,
        "device_map": device_map,
        "excuse_task_head": False,
    }
    # Joined, so a load failing on one node (a torn local copy, a host OOM) raises on every rank.
    # Unthrottled: the text path keeps each rank's copy in host memory until the trainer places it,
    # so admitting the ranks in batches would bound nothing.
    with (
        joined_node_load(f"Frozen model load from {model_name_or_path}", max_concurrent=0)
        if download_tag
        else contextlib.nullcontext()
    ):
        if is_vlm:
            model = from_pretrained_verified(AutoModelForImageTextToText, model_name_or_path, **load_kwargs)
        else:
            model = auto_load_model(model_name_or_path, **load_kwargs)
    cast_parameters_to_run_dtype(model, dtype)

    # Repairs non-persistent buffers; an uninitialized inv_freq biases every logprob this model scores.
    finalize_loaded_model(model)
    finalize_run_model(
        model,
        config,
        sinks_policy=SinksPolicy.from_flags(reset_sinks=reset_sinks),
        attn_implementation=resolved_attn,
    )
    # Same attention warm-up as the policy loader: the first scoring forward would otherwise JIT-compile its
    # kernels mid-step on whichever rank reaches it first, while peers run ahead into the next
    # collective.
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


def load_frozen_reference_model(
    args,
    model_config: ModelConfig,
    training_config,
    tokenizer,
    model_name_or_path: str,
    *,
    is_vlm: bool,
    reset_sinks: bool,
    attn_default: str | None,
    revision: str | None | object = _POLICY_REVISION,
) -> PreTrainedModel:
    """Load the frozen reference a policy is scored against, from ``model_name_or_path``, as the policy loaded.

    The policy's revision pin, remote-code flag, attention request and sinks policy, then the same
    tokenizer-driven vocabulary and special-token ids. ``model_name_or_path`` is the weights the
    reference anchors to: the run's configured model (never a trained resume checkpoint) unless the
    caller names a separate reference repo. A reference from another repo passes that repo's
    own ``revision`` (``None`` for its main): the policy's pin names a commit in the policy's repo only.
    """
    model_ref = load_frozen_auxiliary_model(
        model_name_or_path,
        dtype=resolve_training_dtype(training_config),
        # Unpinned, the reference loads hub main and shifts every logratio.
        revision=getattr(model_config, "model_revision", None) if revision is _POLICY_REVISION else revision,
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


def load_reference_model_for_on_policy_grpo(
    args,
    model_config: ModelConfig,
    training_config,
    parallelism_config: ParallelismConfig,
    tokenizer,
    *,
    peft_config,
    reset_sinks: bool,
    attn_default: str | None,
) -> PreTrainedModel | None:
    """Load the frozen KL reference of an on-policy GRPO run, or ``None`` where TRL's ``GRPOTrainer`` holds none.

    TRL holds one at ``beta != 0`` on a policy no PEFT adapter wraps: a full fine-tune, or native expert-only
    LoRA, which builds no ``PeftModel``. A PEFT policy is its own reference with the adapter disabled. The
    trainer hands this model to TRL in place of the one TRL would build from the policy's name: fp32, the hub's
    default revision, the default attention, no buffer repair. It loads from the run's configured model, never
    a resume checkpoint, as the policy loaded, so ``reset_sinks`` and ``attn_default`` must be the policy's.
    The class resolves from the config as the policy's does (the on-policy scripts refuse
    ``text_only_model``), so a multimodal checkpoint gets its multimodal class on both sides.
    """
    if not on_policy_grpo_holds_reference(training_config.beta, peft_config):
        return None
    warn_unparallelized_reference(parallelism_config, ON_POLICY_GRPO_REFERENCE_ALTERNATIVES)
    return load_frozen_reference_model(
        args,
        model_config,
        training_config,
        tokenizer,
        model_config.model_name_or_path,
        is_vlm=False,
        reset_sinks=reset_sinks,
        attn_default=attn_default,
    )


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
    """Load the frozen reference model for a preference trainer (DPO/KTO), or ``None`` where none is held.

    Under PEFT the reference is the adapter-free base → ``None``: TRL scores it inside the
    ``PeftModel``'s ``disable_adapter()``, which also drops the native EP expert adapters
    (``make_disable_adapter_ep_aware``). Expert-only LoRA has no ``PeftModel``, so it must set
    ``precompute_ref_log_probs``. Full finetune loads an unparallelized copy, under EP/TP with the
    replica's cost reported (:func:`warn_unparallelized_reference`), or precomputes where the
    parallelism shards the policy. ``reset_sinks`` and ``attn_default`` must mirror the policy load so
    the reference's logprobs come from the same kernel and the same sink semantics.
    """
    if model_config.use_peft:
        if (
            parallelism_config.expert_lora is not None
            and not has_attention_lora_targets(model_config)
            and not training_config.precompute_ref_log_probs
        ):
            raise ValueError(
                f"{method} with expert-only native EP LoRA requires precompute_ref_log_probs=True: "
                f"every lora_target_modules entry names an expert projection, so the model is never "
                f"PEFT-wrapped and TRL would build its own reference, an unsharded fp32 dense copy of "
                f"the whole model on every rank. Set precompute_ref_log_probs: true, or add an attention "
                f"target so the adapter-disabled policy is the reference."
            )
        return None

    sharded = parallelism_config.is_ep_mode or parallelism_config.is_tp_mode or parallelism_config.is_pp_mode
    # Precomputed log-probs come from the untrained policy; a resume restores them from the checkpoint.
    if sharded and training_config.precompute_ref_log_probs:
        return None
    # A pipeline stage holds a slice of the model and the schedule owns every forward, so the trainer
    # refuses a live reference (``reject_pp_ref_model``); refused here before a dense copy loads.
    if parallelism_config.is_pp_mode:
        raise ValueError(
            f"Full-finetune {method} under PP needs precompute_ref_log_probs: true: a pipeline stage holds a "
            f"slice of the model and the schedule owns every forward, so no stage can score a live "
            f"reference model, and the trainer refuses one."
        )
    warn_unparallelized_reference(parallelism_config, PREFERENCE_REFERENCE_ALTERNATIVES)
    return load_frozen_reference_model(
        args,
        model_config,
        training_config,
        tokenizer,
        model_config.model_name_or_path,
        is_vlm=is_vlm,
        reset_sinks=reset_sinks,
        attn_default=attn_default,
    )
