"""Parallelism + LoRA compatibility validation for distributed trainers.

Read-only checks run once during setup: inspect ``self.parallelism_config`` / ``self.model`` and
raise with an actionable message, or return. :func:`ctor_positions` / :func:`ctor_value` /
:func:`ctor_config` read the argument a gate validates out of a trainer ``__init__``'s ``*args``,
before the checks below run, and :func:`disable_trl_liger` clears the TRL flag a gate rejects.
"""

from __future__ import annotations

import inspect

import torch.nn as nn
from accelerate.logging import get_logger
from peft import PeftModel

from src.distributed.checkpoint.peft import find_peft_model
from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.expert_weights import has_ep_lora
from src.distributed.parallelism_config import accelerate_launch_rejection
from src.models.loading.config_levels import config_sources, set_config_field_run_scoped
from src.models.moe_balancing import (
    config_has_experts,
    mark_router_logits_forced_off,
)
from src.models.patches.gpt_oss_sinks import has_live_attention_sinks

logger = get_logger(__name__)


def _declared_ctor_parameters(trainer_cls: type) -> list[str]:
    """The named ctor parameters ``trainer_cls`` really takes, ``self`` dropped.

    TRL wraps some trainers in a ``(*args, **kwargs)`` deprecation shim that forwards to the class
    holding the real signature (``KTOTrainer`` → ``trl.experimental.kto``). A shim declares no slots,
    so the positions its callers must fill are the next informative signature up the MRO — which is
    exactly what the shim forwards its ``*args`` to.
    """
    for cls in trainer_cls.__mro__:
        init = cls.__dict__.get("__init__")
        if init is None:
            continue
        named = [
            name
            for name, param in inspect.signature(init).parameters.items()
            if name != "self" and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
        ]
        if named:
            return named
    return []


def ctor_positions(trainer_cls: type, *names: str) -> dict[str, int]:
    """``name → *args slot`` for the named ctor parameters, read off ``trainer_cls``'s signature.

    Derived rather than hand-written so a TRL release that inserts or reorders a parameter moves the
    slot with it — a stale index would silently hand :func:`ctor_value` a neighbouring argument.

    Raises:
        ValueError: a name is absent from the signature — the installed TRL no longer matches the
            trainer's ctor gate, and guessing a slot would gate on the wrong argument.
    """
    params = _declared_ctor_parameters(trainer_cls)
    missing = [name for name in names if name not in params]
    if missing:
        raise ValueError(
            f"{trainer_cls.__name__}.__init__ declares no parameter named {missing}; the installed "
            f"TRL signature no longer matches this trainer's ctor gate."
        )
    return {name: params.index(name) for name in names}


def ctor_value(ctor_args: tuple, kwargs: dict, name: str, positions: dict[str, int]):
    """A trainer ctor parameter by keyword or by the trainer's positional signature slot.

    ``positions`` is the trainer's :func:`ctor_positions` table, so each slot is the installed TRL
    base signature's own rather than a hand-maintained index.
    """
    if name in kwargs:
        return kwargs[name]
    position = positions[name]
    return ctor_args[position] if len(ctor_args) > position else None


def ctor_config(args: tuple, kwargs: dict, position: int = 2):
    """Extract the training config from HF/TRL trainer ctor args — ``kwargs['args']`` or the
    conventional positional slot (2 for ``(model, ref/reward, args, ...)`` signatures, 1 for
    ``RewardTrainer``).

    Truthiness, not membership: an explicit ``args=None`` keyword falls through to the positional
    slot, which is what lets ``_require_vllm_server_mode`` see a positionally-passed config. Every
    caller's slot is pinned against the installed TRL signatures by
    ``tests/cpu/trainers/test_ctor_positions_derived.py``, so a release that reorders the parameters
    fails a test rather than silently reading a neighbouring argument.
    """
    return kwargs.get("args") or (args[position] if len(args) > position else None)


def disable_trl_liger(training_args, reason: str | None = None) -> bool:
    """Force TRL's ``use_liger_kernel`` off (model-level Liger kernels applied by
    ``load_distributed_model`` are unaffected). Returns True when the flag was on and is now
    cleared; ``reason`` (the caller's rationale) is logged as a warning when given."""
    if training_args is None or not getattr(training_args, "use_liger_kernel", False):
        return False
    if reason:
        logger.warning(reason)
    training_args.use_liger_kernel = False
    return True


def disable_trl_liger_grpo_loss(training_args) -> None:
    """Keep TRL's ``use_liger_kernel`` off for the GRPO trainers (shared by online + environmental).

    On GRPO the flag swaps the loss for ``LigerFusedLinearGRPOLoss``, which breaks the global
    ``num_items_in_batch`` normalizer, bypasses chunked-logprobs OOM protection, and drops the
    entropy path. Halo's toolkit default sets it True, so force it off before TRL caches it.
    """
    disable_trl_liger(
        training_args,
        "Disabling TRL's use_liger_kernel for GRPO: it swaps the loss for the fused Liger GRPO "
        "loss (breaks global token normalization and chunked logprobs). Model-level Liger "
        "kernels are still applied by load_distributed_model.",
    )


def has_non_expert_lora(model) -> bool:
    """Whether ``model`` carries LoRA weights outside the EP expert layers.

    Structural, not ``isinstance(model, PeftModel)``: adapters injected in place with
    ``peft.inject_adapter_in_model`` (the embedding path — ``SentenceTransformer`` cannot be
    PEFT-wrapped) leave the model an ordinary ``nn.Module`` while carrying real adapters. EP's native
    grouped expert adapters are excluded — they live on FSDP-ignored expert weights, not on the
    TP-sharded backbone.
    """
    ep_param_ids = {id(p) for m in model.modules() if isinstance(m, EPMoELayerBase) for p in m.parameters()}
    return any("lora_" in name and id(param) not in ep_param_ids for name, param in model.named_parameters())


def active_router_aux_loss_coef(model) -> float:
    """The router aux-loss coefficient this model would actually add to its loss, else ``0``.

    Both the outer and the text config are consulted (:func:`~src.models.loading.config_levels.config_sources`): a multimodal
    MoE declares the routing fields on the text config, while ``output_router_logits`` may be
    toggled on either.
    """
    config = getattr(model, "config", None)
    if config is None:
        return 0.0
    for cfg in config_sources(config):
        if not getattr(cfg, "output_router_logits", False):
            continue
        coef = getattr(cfg, "router_aux_loss_coef", 0) or 0
        if coef > 0:
            return float(coef)
    return 0.0


class ParallelismValidationMixin:
    """Validates requested parallelism modes and LoRA/EP/TP compatibility. Mixed into the trainer."""

    def _validate_parallelism_modes(self):
        """Validate that the requested parallelism modes are supported by this trainer."""
        config = self.parallelism_config

        if config.is_tp_mode and not self._supports_tp:
            raise ValueError(f"{self.__class__.__name__} does not support Tensor Parallelism (TP)")
        if config.needs_ep_wrappers and not self._supports_ep:
            raise ValueError(f"{self.__class__.__name__} does not support Expert Parallelism (EP) / Grouped GEMM")
        if config.is_cp_mode and not self._supports_cp:
            raise ValueError(f"{self.__class__.__name__} does not support Context Parallelism (CP)")
        if config.is_pp_mode and not self._supports_pp:
            reason = self._pp_unsupported_reason or (
                "a pipeline stage holds only a contiguous subset of the layers, so this trainer's "
                "loss cannot be expressed as a last-stage function of logits and labels"
            )
            raise ValueError(
                f"{self.__class__.__name__} does not support Pipeline Parallelism (PP): {reason}. "
                f"See agent-docs/parallelism/pipeline-parallelism.md for the supported trainers."
            )

        if message := accelerate_launch_rejection(config):
            raise ValueError(message)

    def _validate_reference_model(self, ref_model):
        """Reject an explicit reference model under EP/TP — it is never parallelized.

        It stays a plain dense replica running the unpatched MoE path, so its log-probs mismatch the
        policy and the KL term is silently wrong. Use PEFT/LoRA (``ref_model=None``) or
        ``precompute_ref_log_probs=True``.
        """
        config = self.parallelism_config
        if ref_model is not None and (config.is_ep_mode or config.is_tp_mode):
            raise ValueError(
                f"An explicit ref_model is not supported under EP/TP (ep_size={config.ep_size}, "
                f"tp_size={config.tp_size}): the reference is not parallelized, so it would run the "
                f"unpatched dense path and its log-probs would not match the policy's. Use PEFT/LoRA "
                f"(ref_model=None) or precompute_ref_log_probs=True."
            )

    def _validate_implicit_reference_model(self) -> None:
        """Guard the reference model TRL builds for itself when it cannot match the policy.

        :meth:`_validate_reference_model` covers an EXPLICIT ``ref_model``. TRL also builds one
        implicitly when ``beta != 0`` and the model is not PEFT-wrapped, and the scripts clear
        ``model_init_kwargs`` after loading the policy — so it loads fp32, from the hub's default
        revision, with the config-default attention, as a dense replica no parallelism touches.

        Two separate problems, with two separate gates. Live attention sinks make the KL **wrong**:
        the policy is restricted to sink-carrying attention while the reference is not, so the pair
        compute different log-probs for identical tokens. That is a property of the loaded weights,
        not of the sharding — it holds under plain FSDP2 DP, TP and ``ep_size == 1`` exactly as it
        does under EP, so it raises unconditionally. The un-sharded fp32 replica is merely
        **wasteful**, and worst under EP (experts replicated per rank), so that stays a warning.
        """
        # TRL nulls ref_model for the two cases needing no second model (beta == 0, PEFT), so a live one IS implicit.
        if getattr(self, "ref_model", None) is None:
            return
        model = getattr(self, "model", None)
        if model is not None and has_live_attention_sinks(model):
            raise ValueError(
                "beta != 0 without PEFT makes TRL build its own reference model, and this policy "
                "carries LIVE attention sinks (reset_sinks: false). The policy is restricted to "
                "sink-carrying attention while the reference is not, so their log-probs differ for "
                "identical tokens and the KL term is biased on every token. Use beta: 0 (the "
                "bands/clip are the trust region), or use_peft: true (the adapter is disabled to "
                "get the reference), or precompute the reference log-probs."
            )
        if not self.parallelism_config.is_ep_mode:
            return
        logger.warning(
            "beta != 0 without PEFT under EP: TRL built its own reference model from the model path "
            "with no dtype, revision or attn_implementation — an fp32 DENSE replica per rank "
            "(experts included, un-sharded) loaded from the hub's default revision. Budget for it, "
            "and do not expect a pinned model_revision to reach it. beta: 0 or use_peft: true "
            "avoids the second model entirely."
        )

    def _validate_lora_ep_compatibility(self):
        """Reject PEFT-wrapped LoRA inside EP layers; allow the native grouped adapters.

        A PEFT ``LoraLayer`` on an EP module has rank-specific inputs (post-dispatch) and breaks grad
        sync. Only native grouped-LoRA (``<attr>_lora_A``/``_lora_B`` leaves) is allowlisted.
        """
        model = self._top_level_model()
        if not isinstance(model, PeftModel):
            return

        offending = []
        for name, module in model.named_modules():
            if not isinstance(module, EPMoELayerBase):
                continue
            native = {f"{attr}_lora_{w}" for attr in module._expert_lora_attrs for w in ("A", "B")}
            for param_name, _ in module.named_parameters():
                if "lora_" in param_name and param_name not in native:
                    offending.append(f"{name}.{param_name}")

        if offending:
            raise ValueError(
                f"PEFT-wrapped LoRA adapters detected inside EP-wrapped MoE layers. "
                f"Expert layers are distributed across ranks by EP, so a PEFT LoraLayer on them would "
                f"have rank-specific inputs and break gradient sync.\n"
                f"\n"
                f"Offending parameters ({len(offending)}):\n"
                + "\n".join(f"  - {p}" for p in offending[:10])
                + (f"\n  ... and {len(offending) - 10} more" if len(offending) > 10 else "")
                + "\n\n"
                "To LoRA-tune experts under EP, list the expert projections (gate_proj/up_proj/"
                "down_proj/gate_up_proj/experts) in lora_target_modules: split_expert_lora_targets "
                "routes them to native grouped adapters instead of PEFT. LoRA on non-EP layers "
                "(attention, embeddings, lm_head) is unaffected."
            )

    def _validate_expert_lora_peft_config(self):
        """Reject a co-resident attention ``LoraConfig`` the grouped expert adapters cannot match.

        The ``lora_target_modules`` peel vets what it builds the spec from, but a trainer can also be
        handed a ``peft_config`` directly (the documented route for AdaLoRA/(IA)3), which never passes
        through it. Checked here against the LIVE config, so both routes land on the same gate — a
        setting that reached only the attention half would otherwise be two adapters under one recipe.
        """
        spec = self.parallelism_config.expert_lora
        if spec is None or not has_ep_lora(self._top_level_model()):
            return
        peft_model = find_peft_model(self.model)
        if peft_model is None:
            return  # expert-only run: no attention config to disagree with

        conflicts = [
            f"[{name}] {conflict}"
            for name, cfg in peft_model.peft_config.items()
            for conflict in spec.peft_config_conflicts(cfg)
        ]
        if conflicts:
            raise ValueError(
                "The PEFT config carries settings the native grouped expert adapters cannot reproduce, "
                "so they would apply to the attention adapters only:\n"
                + "\n".join(f"  - {c}" for c in conflicts)
                + "\n\nThe grouped adapters express r / lora_alpha / lora_dropout / use_rslora. Drop the "
                "setting, or drop the expert projections from lora_target_modules and LoRA the attention "
                "modules alone."
            )

    def _validate_lora_dropout_live(self):
        """Warn when the configured ``lora_dropout`` was zeroed out from under the user.

        TRL's ``disable_dropout_in_model`` zeroes every ``nn.Dropout`` in the model, and PEFT builds
        ``lora_dropout`` as one. It runs after the adapter wrap on the preference, reward and offline-GRPO
        trainers (``disable_dropout`` defaults to ``True``), so the YAML value is inert there while it is
        honoured on SFT — same config, two behaviours. Deliberate for reference-vs-policy determinism;
        the point here is that it stops being silent.
        """
        peft_model = find_peft_model(self.model)
        spec = self.parallelism_config.expert_lora
        configured = max(
            [float(getattr(cfg, "lora_dropout", 0) or 0) for cfg in getattr(peft_model, "peft_config", {}).values()]
            + [spec.dropout if spec is not None else 0.0]
        )
        if configured <= 0:
            return

        # One substring covers PEFT's ``lora_dropout.<adapter>`` and the EP layers' ``expert_lora_dropout``.
        live = [
            module.p
            for name, module in self._top_level_model().named_modules()
            if isinstance(module, nn.Dropout) and "lora_dropout" in name
        ]
        if live and not any(p > 0 for p in live):
            logger.warning(
                f"lora_dropout={configured} is configured but every LoRA dropout in the live model is 0: "
                f"this trainer disables dropout after the adapter wrap (disable_dropout=True), which "
                f"zeroes PEFT's lora_dropout along with the model's own. Set disable_dropout=False to keep "
                f"it, or drop lora_dropout to stop expecting regularization that is not applied."
            )

    def _validate_expert_lora_realized(self):
        """A peeled ``ExpertLoraSpec`` must materialize as EP grouped adapters, else the expert LoRA
        targets silently vanish (e.g. MoE with ``use_grouped_gemm: false`` and no EP)."""
        spec = self.parallelism_config.expert_lora
        if spec is None:
            return
        model = self._top_level_model()
        if not has_ep_lora(model):
            raise ValueError(
                "lora_target_modules contained expert projections that were routed to native EP "
                "grouped-LoRA (split_expert_lora_targets), but no EP/grouped MoE layer built the "
                "adapters — the expert LoRA would silently not exist. Enable use_grouped_gemm or EP "
                "for MoE expert LoRA, or remove the expert projections from lora_target_modules."
            )

    def _validate_router_aux_loss_consumable(self):
        """Reject MoE load balancing that rides an aux loss this trainer's objective never adds.

        ``*ForCausalLM.forward`` adds ``router_aux_loss_coef * load_balancing_loss_func(...)`` only
        where it builds the loss itself from ``labels``. Most preference, log-prob, distillation and
        pooled-head objectives forward without labels and assemble their own loss, so the aux term
        never enters the graph: the routing collapses over training while the run still pays for
        ``output_router_logits``. A trainer can also re-add the term itself — TRL's ``KTOTrainer``
        does — which is why consumption is DECLARED per trainer class
        (``_consumes_router_aux_loss``) rather than inferred from the forward signature. Same
        contract :mod:`src.distributed.pipeline_parallel.split` enforces for a PP stage.
        """
        if self._consumes_router_aux_loss:
            return
        model = self._top_level_model()
        coef = active_router_aux_loss_coef(model)
        if coef <= 0:
            return
        detail = (
            f"{type(self).__name__} computes its loss from the model's outputs (no ``labels`` "
            f"forward), so the HuggingFace router aux loss is never added: expert load is left "
            f"unbalanced and the run still materializes a [B*S, num_experts] router-logit tensor "
            f"per MoE layer per forward."
        )
        if self._moe_balancing == "aux_loss":
            raise ValueError(
                f"moe_balancing='aux_loss' balances nothing on this trainer. {detail} Use "
                f"moe_balancing='bias_update' (aux-loss-free DeepSeek-V3 bias update, on families "
                f"whose EP wrappers support it) or moe_balancing='none'."
            )
        if self._moe_balancing == "auto":
            # ``auto`` set the flag before this trainer existed; take it back rather than pay for the tensor.
            # The stamp is what makes it stick — MoEMetricsCallback re-enables the flag at train begin.
            set_config_field_run_scoped(model.config, "output_router_logits", False)
            mark_router_logits_forced_off(model.config)
            action = (
                "moe_balancing='auto' resolved to aux_loss before this trainer was built, so "
                "output_router_logits is being turned back OFF for this run (moe/* metrics go with it)."
            )
        else:
            action = (
                f"output_router_logits was set outside the balancing strategy "
                f"(moe_balancing='{self._moe_balancing}'), so it is left ON and keeps costing that "
                f"tensor every forward."
            )
        logger.warning(
            f"output_router_logits is ON with router_aux_loss_coef={coef}, but that aux loss is "
            f"INERT here. {detail} {action} Set moe_balancing='bias_update' for real balancing on "
            f"families whose EP wrappers support it."
        )

    def _validate_load_best_model_reloadable(self):
        """Refuse ``load_best_model_at_end`` shapes whose end-of-run reload is refused or unvalidated.

        Decidable only here: PEFT is applied during ``super().__init__``, and adapter-only runs
        reload in place and are supported. A full fine-tune under EP/CP (base weights only load at
        construction) always saves checkpoints that ship base weights, so the checkpoint loader's
        best-model refusal would fire at the END of training — after the whole run. Fail here.

        Pure TP reloads through ``distribute_tensor`` into the live DTensor placements
        (:meth:`~src.distributed.checkpoint.loader.CheckpointLoader._load_tp`). TP+DP composes
        FSDP2 over TP, whose 2-D placement ``distribute_tensor`` does not invert for packed
        projections, so its loader refuses every reload but the constructed-from-checkpoint skip —
        and a best-model reload can never be that.
        """
        config = self.parallelism_config
        if not self.args.load_best_model_at_end:
            return
        # merge_expert_lora_on_save folds the adapter into a FULL base checkpoint, which the loader refuses.
        adapter_only = find_peft_model(self.model) is not None or config.expert_lora is not None
        if adapter_only and not config.merge_expert_lora_on_save:
            return
        is_moe = config_has_experts(getattr(self.model, "config", None))
        if config.cp_size > 1 or (is_moe and config.needs_ep_wrappers):
            raise ValueError(
                "load_best_model_at_end is unsupported for this shape as a full fine-tune: under "
                "EP/CP base weights only load at construction, so the end-of-run best-model reload "
                "would be refused after the whole run, and the export would otherwise carry the "
                "LAST weights. Drop load_best_model_at_end and export the best checkpoint directly."
            )
        if config.tp_size > 1 and config.data_parallel_size > 1:
            raise ValueError(
                "load_best_model_at_end is unsupported under TP+DP as a full fine-tune: the best-model "
                "reload distributes each checkpoint tensor into the live DTensor placements, and FSDP2 "
                "over TP stacks a strided dp shard on the tp shard — a 2-D placement distribute_tensor "
                "does not invert for packed projections (right shape, wrong rows), so the end-of-run "
                "reload is refused. Drop load_best_model_at_end and export the best checkpoint "
                "directly, or run pure TP (tensor_parallel_size == world size), where the reload is "
                "supported."
            )

    def _validate_lora_tp_compatibility(self):
        """Reject LoRA/PEFT adapters under Tensor Parallelism (``tp_size > 1``), expert LoRA included.

        PEFT's ``lora_A``/``lora_B`` are plain tensors outside the TP graph: the replicated matrix
        diverges (per-rank init, never broadcast) and the sharded one is corrupted by the TP grad
        sync. CP and pure ETP leave attention unsharded, so LoRA there is fine.

        Adapters count whether PEFT wrapped the model or injected them in place — the embedding path
        does the latter, so an ``isinstance`` check alone would let it through. EP's native grouped
        expert adapters are counted separately: they are deliberately excluded from both the
        attention-adapter test and the TP replicated-grad sweep (by param identity), which left
        expert-only LoRA under EP+TP as the one adapter shape no gate reached.
        """
        model = self._top_level_model()
        if has_ep_lora(model):
            raise ValueError(
                "Native EP expert LoRA is not supported with Tensor Parallelism (tp_size > 1).\n"
                "The expert adapters live on the EP-distributed expert weights, so every TP gate "
                "skips them by param identity: neither the attention-adapter check nor the TP "
                "replicated-grad sync sees them, and no gradient-equivalence or save/merge test "
                "covers an EP+TP adapter. Rather than train and export on an unvalidated path, this "
                "combination is refused — the same allowlist discipline the parallelism axes use.\n\n"
                "Use one of:\n"
                "  - expert LoRA under EP without TP (validated: save, resume and merge-on-save)\n"
                "  - full fine-tuning under EP+TP (no adapters)"
            )
        if not isinstance(model, PeftModel) and not has_non_expert_lora(model):
            return
        raise ValueError(
            "LoRA/PEFT adapters are not supported with Tensor Parallelism (tp_size > 1).\n"
            "TP shards the attention/MLP base layers as DTensors, but PEFT adapters are added as "
            "plain tensors outside the TP graph: the replicated adapter matrix diverges across "
            "ranks (per-rank init, never broadcast) and the sharded one is corrupted by the TP "
            "replicated-grad sync. The resulting adapter is rank-inconsistent and will not reload "
            "correctly.\n\n"
            "Use one of:\n"
            "  - LoRA with FSDP2 data parallelism (no TP)\n"
            "  - LoRA with Context Parallelism (CP) and/or pure Expert-TP (ETP) — attention "
            "weights stay unsharded there, so the adapter is replicated and correct\n"
            "  - Full fine-tuning under TP (no adapters)"
        )
