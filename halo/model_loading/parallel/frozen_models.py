"""The frozen teacher / reference models a run holds beside its policy: never sharded, but loaded to
match the policy in everything that changes a logprob, and the parallelism rules on holding one."""

from __future__ import annotations

import contextlib
import dataclasses
import logging
from typing import TYPE_CHECKING

import torch
from accelerate.utils import is_peft_model
from transformers import AutoModelForImageTextToText, PreTrainedModel

from halo.distributed.parallelism_config import ParallelismConfig
from halo.distributed.tensor_parallel.hf_plan import is_input_embedding_tp_sharded
from halo.log import warn_once
from halo.model_loading.checkpoint_coverage import from_pretrained_verified
from halo.model_loading.dtype import cast_parameters_to_run_dtype, resolve_training_dtype
from halo.model_loading.finalization import finalize_loaded_model
from halo.model_loading.model_preparation import (
    apply_family_patches,
    apply_run_model_policies,
    auto_load_model,
    resolve_build_config,
)
from halo.model_loading.parallel.model_source import resolve_model_source
from halo.model_loading.parallel.peft_setup import wraps_in_peft_model
from halo.model_loading.parallel.whole_model import warm_attention_kernels
from halo.model_loading.tokenizer_setup import configure_run_tokenizer
from halo.models.attention.gpt_oss_sinks import SinksPolicy
from halo.models.attention.selection import resolve_attn_implementation
from halo.runtime.load_coordination import joined_node_load

if TYPE_CHECKING:
    # Annotation only: the entry scripts' argument layer sits beside this package.
    from halo.configs.script_arguments import ModelArguments

# warn_once marks a message seen on every rank, so each rank must emit it.
logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class ReferenceAlternatives:
    """A caller's ways around a frozen reference; ``peft`` only where an adapter can train, as TP refuses LoRA."""

    always: str
    peft: str | None = None

    def under(self, parallelism_config: ParallelismConfig) -> str:
        """The alternatives this parallelism shape can run."""
        if self.peft is None or parallelism_config.is_tp_mode:
            return self.always
        return f"{self.always} {self.peft}"


# The DPO/KTO way around a dense reference, shared by the script loader and the trainer gate.
PREFERENCE_REFERENCE_ALTERNATIVES = ReferenceAlternatives(
    always="precompute_ref_log_probs: true scores the same reference once, from the untrained policy, exactly "
    "and with no second model.",
    peft="PEFT (use_peft: true, ref_model=None) is the other way around it.",
)
ON_POLICY_GRPO_REFERENCE_ALTERNATIVES = ReferenceAlternatives(
    always="beta: 0 drops the KL term and its reference.",
    peft="use_peft: true with an attention target scores the reference as the policy with its adapter disabled.",
)

# Messages already reported: the script loader and the trainer gate report one reference once.
_reported_references: set[str] = set()


def warn_unparallelized_reference(parallelism_config: ParallelismConfig, avoid: ReferenceAlternatives) -> None:
    """Warn that a frozen reference beside a policy sharded by EP, ETP or TP is a whole dense replica per rank; its
    log-probs differ by kernel numerics only, so the cost is memory and time."""
    if parallelism_config.has_ep_group or parallelism_config.is_tp_mode:
        message = (
            f"A frozen, unparallelized reference beside a policy sharded by EP/ETP/TP "
            f"(expert_parallel_size={parallelism_config.ep_size}, "
            f"expert_tensor_parallel_size={parallelism_config.expert_tp_size}, "
            f"tensor_parallel_size={parallelism_config.tp_size}) is a whole dense replica on every rank, "
            f"experts included, run through the model's own MoE forward: its log-probs match the policy's "
            f"up to kernel numerics, but budget for its memory and its slower forward. "
            f"{avoid.under(parallelism_config)}"
        )
        warn_once(logger, _reported_references, message, message)


def on_policy_grpo_holds_reference(beta: float, peft_config, model=None) -> bool:
    """TRL ``GRPOTrainer``'s rule for holding a frozen KL reference: ``beta != 0`` on a policy no PEFT adapter wraps.
    Native expert-only LoRA builds no ``PeftModel``, so it holds one."""
    wrapped = isinstance(model, torch.nn.Module) and is_peft_model(model)
    return beta != 0.0 and peft_config is None and not wrapped


def load_frozen_auxiliary_model(
    model_name_or_path: str,
    *,
    dtype: torch.dtype,
    revision: str | None = None,
    trust_remote_code: bool,
    attn_implementation: str | None = None,
    reset_sinks: bool = True,
    is_vlm: bool = False,
    device_map: dict | str | int | torch.device | None = None,
    download_tag: str | None,
) -> PreTrainedModel:
    """Load an unparallelized frozen reference or teacher, matching the policy load in attention backend, family
    patches, sinks and dtype, as a mismatch biases the loss rather than raising. ``is_vlm`` pins
    ``AutoModelForImageTextToText``; a ``download_tag`` makes the load COLLECTIVE-EQUIVALENT, ``None``
    single-process."""
    # This path's first hub contact for the repo: on a cold cache every rank of every node would
    # otherwise hit the hub at once, and per-node caches could each hold a different commit.
    source = resolve_model_source(model_name_or_path, revision, tag=download_tag) if download_tag else None
    if source is not None:
        revision = source.revision
    config = resolve_build_config(model_name_or_path, revision=revision, trust_remote_code=trust_remote_code)
    sinks_policy = SinksPolicy.from_flags(reset_sinks=reset_sinks)
    backend = resolve_attn_implementation(config, attn_implementation, dtype, sinks_policy=sinks_policy)
    resolved_attn = apply_family_patches(config, backend)
    load_kwargs = {
        "config": config,
        "revision": revision,
        "dtype": dtype,
        "trust_remote_code": trust_remote_code,
        "attn_implementation": resolved_attn,
        "device_map": device_map,
        "excuse_task_head": False,
    }
    # Joined, so a load failing on one node raises on every rank. Unthrottled: each rank's copy stays in host memory
    # until the trainer places it, so admitting ranks in batches would bound nothing.
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
    finalize_loaded_model(model, weights_dir=None if source is None else source.local_dir)
    apply_run_model_policies(
        model,
        config,
        sinks_policy=sinks_policy,
        attn_implementation=resolved_attn,
    )
    warm_attention_kernels(model, dtype=dtype)
    return model


def place_and_freeze(auxiliary: PreTrainedModel, policy: torch.nn.Module) -> torch.device:
    """Move ``auxiliary`` to ``policy``'s device, eval and freeze it: on another device the forward raises, in train
    mode dropout perturbs every target, and with live gradients the optimizer would step it."""
    device = policy.device
    auxiliary.to(device)
    auxiliary.eval()
    for param in auxiliary.parameters():
        param.requires_grad = False
    return device


def load_frozen_reference_model(
    args,
    model_config: ModelArguments,
    training_config,
    tokenizer,
    model_name_or_path: str,
    *,
    is_vlm: bool,
    reset_sinks: bool,
    attn_default: str | None,
) -> PreTrainedModel:
    """Load the frozen reference as the policy loaded: its remote-code flag, attention request, sinks policy and
    tokenizer state. ``model_name_or_path`` is the run's configured model (never a trained resume checkpoint) at the
    policy's revision pin, or a separate reference repo at its main."""
    model_ref = load_frozen_auxiliary_model(
        model_name_or_path,
        dtype=resolve_training_dtype(training_config),
        # Unpinned, the policy's own weights would load at hub main and shift every logratio.
        revision=(
            getattr(model_config, "model_revision", None)
            if model_name_or_path == model_config.model_name_or_path
            else None
        ),
        trust_remote_code=model_config.trust_remote_code,
        # The policy's own request, as a logratio is a difference of two logprobs and a differing kernel biases the
        # objective; the fallback too, or an unset config auto-detects against a pinned policy.
        attn_implementation=model_config.attn_implementation or attn_default,
        reset_sinks=reset_sinks,
        is_vlm=is_vlm,
        # Possibly the repo's first hub contact, so every rank would otherwise fetch at once.
        download_tag="reference_model",
    )
    configure_run_tokenizer(args, model_ref, tokenizer, embeddings_sharded=is_input_embedding_tp_sharded)
    return model_ref


def load_reference_model_for_on_policy_grpo(
    args,
    model_config: ModelArguments,
    training_config,
    parallelism_config: ParallelismConfig,
    tokenizer,
    *,
    peft_config,
    reset_sinks: bool,
    attn_default: str | None,
) -> PreTrainedModel | None:
    """The frozen KL reference of an on-policy GRPO run, or ``None`` where TRL holds none. It replaces the one TRL
    would build (fp32, hub main, default attention, no buffer repair), so ``reset_sinks`` and ``attn_default`` must be
    the policy's."""
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
    model_config: ModelArguments,
    training_config,
    parallelism_config: ParallelismConfig,
    tokenizer,
    *,
    is_vlm: bool,
    method: str,
    reset_sinks: bool,
    attn_default: str | None,
):
    """The frozen reference for DPO/KTO, or ``None``. Under PEFT it is the adapter-free base, which expert-only LoRA
    lacks, so that must precompute. A full fine-tune loads an unparallelized copy, or precomputes where parallelism
    shards the policy."""
    if model_config.use_peft:
        if (
            parallelism_config.expert_lora is not None
            and not wraps_in_peft_model(model_config)
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

    sharded = parallelism_config.has_ep_group or parallelism_config.is_tp_mode or parallelism_config.is_pp_mode
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
