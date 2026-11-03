"""LoRA / PEFT / expert-LoRA refusals under the parallelism modes, read off the live model after its wrap."""

from peft import PeftModel

from halo.checkpoint.parallel.adapters import saved_adapter_tensors
from halo.distributed.expert_parallel.moe_layer.expert_lora import model_has_expert_lora
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.log import KEY_PREVIEW_COUNT
from halo.model_loading.tokenizer_setup import vocabulary_grown
from halo.models.peft_adapters import lora_fold_targets, tuner_adapter_param_ids
from halo.models.unwrapping import find_peft_model


def reject_peft_lora_in_ep_layers(model) -> None:
    """PEFT-wrapped LoRA inside EP layers sees rank-specific (post-dispatch) inputs, which break gradient sync; only
    the native grouped adapters (``<attr>_lora_A`` / ``_lora_B`` leaves) are allowed there."""
    if not isinstance(model, PeftModel):
        return
    offending = []
    for name, module in named_ep_layers(model):
        native = {adapter_name for adapter_name, _ in module.expert_lora_named_params()}
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
            + "\n".join(f"  - {p}" for p in offending[:KEY_PREVIEW_COUNT])
            + (f"\n  ... and {len(offending) - KEY_PREVIEW_COUNT} more" if len(offending) > KEY_PREVIEW_COUNT else "")
            + "\n\n"
            "To LoRA-tune experts under EP, list the expert projections (gate_proj/up_proj/"
            "down_proj/gate_up_proj/experts) in lora_target_modules: split_expert_lora_targets "
            "routes them to native grouped adapters instead of PEFT. LoRA on non-EP layers "
            "(attention, embeddings, lm_head) is unaffected."
        )


def reject_expert_lora_peft_conflicts(model, expert_lora) -> None:
    """Checked against the live config, so a ``peft_config`` handed to the trainer directly lands on the same gate as
    the ``lora_target_modules`` peel."""
    if expert_lora is None or not model_has_expert_lora(model):
        return
    peft_model = find_peft_model(model)
    if peft_model is None:
        return  # expert-only run: no attention config to disagree with
    conflicts = [
        f"[{name}] {conflict}"
        for name, cfg in peft_model.peft_config.items()
        for conflict in expert_lora.peft_config_conflicts(cfg)
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


def reject_unsaved_grown_vocabulary(model, *, accelerate_manages_fsdp: bool) -> None:
    """No base checkpoint holds the added rows. Accelerate-managed FSDP saves through PEFT's embedding check, which
    skips offline, and PEFT's selection, once a LoRA adapts one embedding layer, drops an untied plain partner. An
    expert-only LoRA run has no PEFT adapter: its resume regrows the rows."""
    if not vocabulary_grown(model):
        return
    peft_model = find_peft_model(model)
    if peft_model is None:
        return
    if accelerate_manages_fsdp:
        raise ValueError(
            "added_special_tokens grew the vocabulary of an adapter run under accelerate-managed FSDP, whose adapter "
            "saves the base Trainer writes with PEFT's own check for a resized embedding, which reads the base config "
            "again and skips it offline or without the Hub: the added rows, which no base checkpoint has, would be "
            "left out. Launch with torchrun (toolkit FSDP2), or grow the vocabulary before the run with "
            "scripts/before_training/patch_vocab.py."
        )
    selected = {id(tensor) for tensor in saved_adapter_tensors(peft_model).values()}
    layers = {"input embedding": peft_model.get_input_embeddings(), "output head": peft_model.get_output_embeddings()}
    unsaved = [role for role, layer in layers.items() if layer is not None and id(layer.weight) not in selected]
    if unsaved:
        raise ValueError(
            f"added_special_tokens grew the vocabulary, but PEFT's adapter selection leaves out the {' and '.join(unsaved)}: "
            f"once lora_target_modules adapts one layer of an untied embedding pair, PEFT saves only the adapted "
            f"layers' base weights, so those added rows would reach no checkpoint. Adapt both embedding layers or "
            f"neither, or grow the vocabulary before the run with scripts/before_training/patch_vocab.py."
        )


def reject_unrealized_expert_lora(model, expert_lora) -> None:
    """Refuse a peeled ``ExpertLoraSpec`` no EP/grouped layer built (e.g. MoE without grouped GEMM or EP)."""
    if expert_lora is not None and not model_has_expert_lora(model):
        raise ValueError(
            "lora_target_modules contained expert projections that were routed to native EP "
            "grouped-LoRA (split_expert_lora_targets), but no EP/grouped MoE layer built the "
            "adapters — the expert LoRA would silently not exist. Enable use_grouped_gemm or EP "
            "for MoE expert LoRA, or remove the expert projections from lora_target_modules."
        )


def reject_lora_under_tp(model) -> None:
    """PEFT's adapters are plain tensors outside the TP graph: the replicated matrix diverges and the sharded one is
    corrupted by the TP grad sync. Native expert adapters are no tuner layer, and every TP gate skips them by
    identity."""
    if model_has_expert_lora(model):
        raise ValueError(
            "Native EP expert LoRA is not supported with Tensor Parallelism (tensor_parallel_size > 1).\n"
            "The expert adapters live on the EP-distributed expert weights, so every TP gate "
            "skips them by param identity: neither the attention-adapter check nor the TP "
            "replicated-grad sync sees them, and no gradient-equivalence or save/merge test "
            "covers an EP+TP adapter. Rather than train and export on an unvalidated path, this "
            "combination is refused — the same allowlist discipline the parallelism axes use.\n\n"
            "Use one of:\n"
            "  - expert LoRA under EP without TP (validated: save, resume and merge-on-save)\n"
            "  - full fine-tuning under EP+TP (no adapters)"
        )
    if not isinstance(model, PeftModel) and not tuner_adapter_param_ids(model):
        return
    raise ValueError(
        "LoRA/PEFT adapters are not supported with Tensor Parallelism (tensor_parallel_size > 1).\n"
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


def reject_unmergeable_expert_lora_save(
    model, *, merge_expert_lora_on_save: bool, accelerate_manages_fsdp: bool
) -> None:
    """Refuse ``merge_expert_lora_on_save`` with no native expert adapters to fold, on accelerate-managed FSDP v1's
    flat-param shards, or with a PEFT layer the out-of-place fold cannot reproduce."""
    if not merge_expert_lora_on_save:
        return
    if not model_has_expert_lora(model):
        raise ValueError(
            "merge_expert_lora_on_save=True but this model built no native grouped expert "
            "adapters, so there is nothing to fold and the flag would be silently ignored. It "
            "requires an EP/grouped-GEMM MoE run whose lora_target_modules names expert "
            "projections (gate_proj/up_proj/down_proj/gate_up_proj/experts). Attention-only "
            "LoRA saves adapters instead — merge those with "
            "scripts/after_training/merge_peft_adapters.py."
        )
    if accelerate_manages_fsdp:
        raise ValueError(
            "merge_expert_lora_on_save=True is not supported under accelerate-managed FSDP: "
            "adapters are flat-param shards there and the base Trainer's save owns the layout, "
            "so the merged gather would write shard views. Launch with torchrun (mixin-managed "
            "FSDP2), or drop merge_expert_lora_on_save and keep adapter checkpoints."
        )
    peft_model = find_peft_model(model)
    if peft_model is not None:
        lora_fold_targets(peft_model)
