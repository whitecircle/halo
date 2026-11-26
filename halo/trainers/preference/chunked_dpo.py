"""Bounded DPO head scoring, without full-vocabulary logits in the policy or reference forward."""

from __future__ import annotations

from collections.abc import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.tuners.tuners_utils import BaseTunerLayer
from trl.models.utils import _ForwardRedirection, disable_gradient_checkpointing

from halo.configs.preference import DPOScriptArguments
from halo.distributed.expert_parallel.deepep.capacity import bump_forward_generation
from halo.distributed.expert_parallel.moe_layer.expert_lora import model_has_expert_lora
from halo.env import is_accelerate_launch
from halo.kernels.head_logprobs import chunked_head_logprobs
from halo.models.head_transform import resolve_head_transform
from halo.models.introspection.modality import config_declares_multimodality
from halo.models.introspection.module_tree import hidden_state_backbone
from halo.models.introspection.unwrapping import base_transformers_model, find_peft_model
from halo.models.moe.config_fields import has_routed_experts, resolve_active_router_aux_loss_coef
from halo.models.moe.family_specs import QWEN3, find_moe_family_spec_for_config
from halo.runtime.dtensor import materialize_dtensor
from halo.trainers.reference_logps import is_reference_entry, reference_regeneration_steps

DPO_SCORER_KEY = "logprob_scorer"
CHUNKED_DPO_SCORER = "chunked_head_logprobs"


def reject_chunked_dpo_options(model, args, parallelism, *, peft_config=None, compute_metrics=None) -> None:
    if not isinstance(model, nn.Module) or args is None or parallelism is None:
        raise ValueError("use_chunked_dpo_logprobs needs a loaded model, DPOConfig and ParallelismConfig.")
    if any(getattr(parallelism, axis) > 1 for axis in ("tp_size", "expert_tp_size", "cp_size", "pp_size")):
        raise ValueError("use_chunked_dpo_logprobs does not support TP, ETP, CP or PP; disable it for those axes.")
    if is_accelerate_launch() or getattr(args, "deepspeed", None):
        raise ValueError(
            "use_chunked_dpo_logprobs requires Halo-managed FSDP2; launch with torchrun, without DeepSpeed."
        )
    losses = args.loss_type if isinstance(args.loss_type, list) else [args.loss_type]
    if losses != ["sigmoid"] or args.f_divergence_type != "reverse_kl":
        raise ValueError("use_chunked_dpo_logprobs supports only loss_type: sigmoid with reverse_kl divergence.")
    if args.use_weighting or args.ld_alpha is not None or args.label_smoothing != 0:
        raise ValueError("use_chunked_dpo_logprobs refuses weighting, ld_alpha and nonzero label_smoothing.")
    if args.padding_free or not args.precompute_ref_log_probs or args.sync_ref_model:
        raise ValueError(
            "use_chunked_dpo_logprobs needs padded rows and precompute_ref_log_probs, without sync_ref_model."
        )
    if compute_metrics is not None:
        raise ValueError(
            "use_chunked_dpo_logprobs does not return raw logits for compute_metrics; evaluate loss only."
        )
    base = base_transformers_model(model)
    if peft_config is not None or find_peft_model(model) is not None or model_has_expert_lora(model):
        raise ValueError("use_chunked_dpo_logprobs supports full fine-tuning only; remove the adapters or disable it.")
    if any(isinstance(module, BaseTunerLayer) for module in base.modules()):
        raise ValueError("use_chunked_dpo_logprobs refuses adapters, including an adapted lm_head.")
    if getattr(base, "is_loaded_in_4bit", False) or getattr(base, "is_loaded_in_8bit", False):
        raise ValueError("use_chunked_dpo_logprobs does not support a quantized base; load full-precision weights.")
    if config_declares_multimodality(base.config):
        raise ValueError(
            "use_chunked_dpo_logprobs is text-only; load the text model rather than a multimodal composite."
        )
    if resolve_active_router_aux_loss_coef(base) > 0 or getattr(args, "router_aux_loss_coef", None):
        raise ValueError("use_chunked_dpo_logprobs does not carry router aux_loss; disable active aux-loss balancing.")
    if has_routed_experts(base.config):
        if find_moe_family_spec_for_config(base.config) is not QWEN3 or parallelism.ep_size != 2:
            raise ValueError("use_chunked_dpo_logprobs supports MoE only for Qwen3-MoE at EP2.")
    elif parallelism.ep_size != 1:
        raise ValueError("Dense use_chunked_dpo_logprobs runs without expert parallelism.")
    head = base.get_output_embeddings()
    if not isinstance(head, nn.Linear) or getattr(head.forward, "__func__", None) is not nn.Linear.forward:
        raise ValueError("use_chunked_dpo_logprobs needs an unadapted plain Linear lm_head.")
    if hidden_state_backbone(base) is None:
        raise ValueError("use_chunked_dpo_logprobs needs a backbone separate from its output head.")


class ChunkedDPOScoringMixin:
    def __init__(self, *args, use_chunked_dpo_logprobs: bool = DPOScriptArguments.use_chunked_dpo_logprobs, **kwargs):
        if type(use_chunked_dpo_logprobs) is not bool:
            raise ValueError("use_chunked_dpo_logprobs must be a boolean.")
        self.use_chunked_dpo_logprobs = use_chunked_dpo_logprobs
        self._dpo_head_transforms = {}
        self._dpo_scored_datasets = {}
        self._dpo_forward_redirection = _ForwardRedirection()
        if use_chunked_dpo_logprobs:
            if args:
                raise ValueError("use_chunked_dpo_logprobs requires keyword trainer arguments.")
            reject_chunked_dpo_options(
                kwargs.get("model"),
                kwargs.get("args"),
                kwargs.get("parallelism_config"),
                peft_config=kwargs.get("peft_config"),
                compute_metrics=kwargs.get("compute_metrics"),
            )
            if kwargs.get("ref_model") is not None:
                reject_chunked_dpo_options(kwargs["ref_model"], kwargs["args"], kwargs["parallelism_config"])
            if kwargs.get("moe_balancing") == "aux_loss":
                raise ValueError("use_chunked_dpo_logprobs does not support moe_balancing: aux_loss.")
        super().__init__(*args, **kwargs)
        if use_chunked_dpo_logprobs and self.aux_loss_enabled:
            raise ValueError("use_chunked_dpo_logprobs cannot train an active router aux_loss; disable that term.")

    def compute_ref_log_probs(self, model, inputs):
        if not getattr(self, "use_chunked_dpo_logprobs", False):
            return super().compute_ref_log_probs(model, inputs)
        with torch.no_grad(), disable_gradient_checkpointing(self.model, self.args.gradient_checkpointing_kwargs):
            logps, _ = self._chunked_dpo_logps(model, inputs, compute_entropy=False)
        return logps.sum(dim=1, dtype=torch.float32).chunk(2)

    def _compute_loss(self, model, inputs, return_outputs):
        if not getattr(self, "use_chunked_dpo_logprobs", False):
            return super()._compute_loss(model, inputs, return_outputs)
        if return_outputs:
            raise ValueError("use_chunked_dpo_logprobs does not return raw logits; set prediction_loss_only=True.")
        if self.aux_loss_enabled:
            raise ValueError("use_chunked_dpo_logprobs cannot train an active router aux_loss.")
        logps, entropy = self._chunked_dpo_logps(model, inputs, compute_entropy=True)
        chosen, rejected = logps.sum(dim=1, dtype=torch.float32).chunk(2)
        missing = [name for name in self.REFERENCE_COLUMNS if name not in inputs]
        if missing:
            raise ValueError(f"use_chunked_dpo_logprobs needs precomputed {missing}; run its reference sweep first.")
        chosen_ratio = chosen - inputs["ref_chosen_logps"].float()
        rejected_ratio = rejected - inputs["ref_rejected_logps"].float()
        loss = -F.logsigmoid(self.beta * (chosen_ratio - rejected_ratio)).mean() * self.loss_weights[0]
        self._record_chunked_dpo_metrics(inputs, entropy, chosen, rejected, chosen_ratio, rejected_ratio)
        return loss

    def _precompute_ref_logps(self, dataset, name, batch_size):
        if not getattr(self, "use_chunked_dpo_logprobs", False):
            return super()._precompute_ref_logps(dataset, name, batch_size)
        supplied = any(column in (dataset.column_names or []) for column in self.REFERENCE_COLUMNS)
        if supplied and not any(dataset is scored for scored in self._dpo_scored_datasets.values()):
            raise ValueError(
                "use_chunked_dpo_logprobs refuses supplied reference columns without scorer provenance. "
                "Remove ref_chosen_logps/ref_rejected_logps and let the bounded reference sweep compute them."
            )
        scored = super()._precompute_ref_logps(dataset, name, batch_size)
        self._dpo_scored_datasets[name] = scored
        return scored

    def _reference_mismatch_refusal(self, name, entry: object, identity: Mapping, mismatch: str) -> str:
        expected = identity["settings"].get(DPO_SCORER_KEY)
        if is_reference_entry(entry) and entry["settings"].get(DPO_SCORER_KEY) != expected:
            return (
                f"Regenerate the '{name}' reference log-probs for this scorer: "
                f"{reference_regeneration_steps(self._reference_resume_checkpoint)}. The saved scorer is "
                f"{entry['settings'].get(DPO_SCORER_KEY, 'full_logits')}; this run uses "
                f"{expected or 'full_logits'}. Their scores must not be mixed on resume."
            )
        return super()._reference_mismatch_refusal(name, entry, identity, mismatch)

    def _chunked_dpo_logps(self, model, inputs, *, compute_entropy):
        unexpected = set(inputs) - {"input_ids", "attention_mask", "completion_mask", *self.REFERENCE_COLUMNS}
        if unexpected:
            raise ValueError(
                f"use_chunked_dpo_logprobs accepts text preference tensors only; unsupported: {sorted(unexpected)}."
            )
        ids, mask = inputs["input_ids"], inputs["completion_mask"].bool()
        if ids.ndim != 2 or ids.shape != mask.shape or ids.shape != inputs["attention_mask"].shape or ids.shape[0] % 2:
            raise ValueError("Chunked DPO needs aligned [chosen, rejected] text rows and masks.")
        mask = mask[:, 1:] & inputs["attention_mask"][:, 1:].bool()
        unwrapped = self.accelerator.unwrap_model(model)
        original_forward = unwrapped.forward
        try:
            return self._dpo_forward_redirection(
                model,
                unwrapped,
                self._chunked_dpo_forward,
                unwrapped,
                ids,
                inputs["attention_mask"],
                mask,
                compute_entropy,
            )
        finally:
            unwrapped.forward = original_forward

    def _chunked_dpo_forward(self, model, input_ids, attention_mask, completion_mask, compute_entropy):
        base = base_transformers_model(model)
        transform = self._dpo_head_transforms.get(id(base))
        if transform is None:
            transform = resolve_head_transform(base)
            self._dpo_head_transforms[id(base)] = transform
        backbone = hidden_state_backbone(base)
        bump_forward_generation()
        with self.accelerator.autocast():
            hidden = backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        head = base.get_output_embeddings()
        weight, bias = transform.kept_rows(materialize_dtensor(head.weight), materialize_dtensor(head.bias))
        logps, entropy = chunked_head_logprobs(
            transform.scale_hidden(hidden[:, :-1][completion_mask]),
            weight,
            input_ids[:, 1:][completion_mask],
            bias,
            1.0,
            logit_scale=transform.logit_scale,
            softcap=transform.softcap,
            compute_entropy=compute_entropy,
        )
        token_logps = logps.new_zeros(completion_mask.shape).masked_scatter(completion_mask, logps)
        token_entropy = (
            entropy.new_zeros(completion_mask.shape).masked_scatter(completion_mask, entropy)
            if entropy is not None
            else None
        )
        return token_logps, token_entropy

    def _record_chunked_dpo_metrics(self, inputs, entropy, chosen, rejected, chosen_ratio, rejected_ratio):
        mode = "train" if self.model.training else "eval"
        metrics, accelerator = self._metrics[mode], self.accelerator
        mask = inputs["completion_mask"][:, 1:].bool() & inputs["attention_mask"][:, 1:].bool()
        entropy_sum = accelerator.gather_for_metrics((entropy * mask).sum()).sum()
        count = accelerator.gather_for_metrics(mask.sum()).sum()
        metrics["entropy"].append((entropy_sum / count).item() if count > 0 else 0.0)
        if mode == "train":
            self._total_train_tokens += accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
        metrics["num_tokens"] = [self._total_train_tokens]
        chosen_rewards, rejected_rewards = self.beta * chosen_ratio.detach(), self.beta * rejected_ratio.detach()
        values = {
            "rewards/chosen": chosen_rewards,
            "rewards/rejected": rejected_rewards,
            "rewards/accuracies": (chosen_rewards > rejected_rewards).float(),
            "rewards/margins": chosen_rewards - rejected_rewards,
            "logps/chosen": chosen.detach(),
            "logps/rejected": rejected.detach(),
        }
        for name, value in values.items():
            metrics[name].append(accelerator.gather(value).mean().item())
