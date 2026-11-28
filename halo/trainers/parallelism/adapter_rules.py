"""LoRA / PEFT / expert-LoRA refusals under the parallelism modes, read off the live model after its wrap."""

from peft import PeftModel
from peft.tuners.tuners_utils import BaseTunerLayer
from peft.utils import AuxiliaryTrainingWrapper
from torch.distributed.fsdp import FSDPModule

from halo.checkpoint.parallel.adapters import live_adapter_tensors
from halo.distributed.expert_parallel.moe_layer.expert_lora import model_has_expert_lora
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.fsdp import fsdp2_param_groups
from halo.log import KEY_PREVIEW_COUNT
from halo.model_loading.tokenizer_setup import vocabulary_grown
from halo.models.introspection.unwrapping import find_peft_model
from halo.models.peft_adapters import lora_fold_targets, tuner_adapter_param_ids


def reject_peft_lora_in_ep_layers(model) -> None:
    """PEFT-wrapped LoRA inside EP layers sees rank-specific (post-dispatch) inputs, which break gradient sync; only
    the native grouped adapters (``<attr>_lora_A`` / ``_lora_B`` leaves) are allowed there. A tuner layer wrapping an
    EP layer, PEFT's ``target_parameters`` route, holds its adapters outside the EP gradient sync."""
    if find_peft_model(model) is None:
        return
    ep_layers = named_ep_layers(model)
    offending = []
    for name, module in ep_layers:
        native = {adapter_name for adapter_name, _ in module.expert_lora_named_params()}
        for param_name, _ in module.named_parameters():
            if "lora_" in param_name and param_name not in native:
                offending.append(f"{name}.{param_name}")
    ep_ids = {id(module) for _, module in ep_layers}
    offending.extend(
        f"{name} (a {type(module).__name__} around an EP layer)"
        for name, module in model.named_modules()
        if isinstance(module, BaseTunerLayer)
        and any(id(inner) in ep_ids for inner in module.get_base_layer().modules())
    )
    if offending:
        raise ValueError(
            f"PEFT-wrapped LoRA adapters detected inside or around EP-wrapped MoE layers. "
            f"Expert layers are distributed across ranks by EP, so a PEFT LoraLayer on them would "
            f"have rank-specific inputs and break gradient sync.\n"
            f"\n"
            f"Offending parameters ({len(offending)}):\n"
            + "\n".join(f"  - {p}" for p in offending[:KEY_PREVIEW_COUNT])
            + (f"\n  ... and {len(offending) - KEY_PREVIEW_COUNT} more" if len(offending) > KEY_PREVIEW_COUNT else "")
            + "\n\n"
            "To LoRA-tune experts under EP, list the expert projections (gate_proj/up_proj/"
            "down_proj/gate_up_proj/experts) in lora_target_modules: split_expert_lora_targets "
            "routes them to native grouped adapters instead of PEFT. Leave target_parameters "
            "unset: PEFT wraps the module holding a targeted parameter, here the EP layer. LoRA on "
            "non-EP layers (attention, embeddings, lm_head) is unaffected."
        )


def _adapter_param_ids(peft_model: PeftModel) -> set[int]:
    """Every tensor ``disable_adapter()`` freezes and thaws: a tuner layer's beside its base layer's, and a
    ``modules_to_save`` copy's or trainable-token delta's beside the module the wrapper holds."""
    held = tuner_adapter_param_ids(peft_model)
    for module in peft_model.modules():
        if isinstance(module, AuxiliaryTrainingWrapper):
            wrapped = {id(param) for param in module.original_module.parameters()}
            held.update(id(param) for param in module.parameters() if id(param) not in wrapped)
    return held


def _adapters_held_outside(model, peft_model: PeftModel) -> list[str]:
    """The trained adapter tensors an FSDP2 shard group outside ``peft_model`` holds, by parameter name. Read off the
    groups themselves: the wrap keeps a tensor reachable under two names, a token delta a tied pair shares, in the
    root's group, wherever the module tree shows it."""
    adapters = _adapter_param_ids(peft_model)
    inside = {id(group) for group in fsdp2_param_groups(peft_model)}
    names = {id(param): name for name, param in model.named_parameters()}
    return sorted(
        names[id(fsdp_param.sharded_param)]
        for group in fsdp2_param_groups(model)
        if id(group) not in inside
        for fsdp_param in group.fsdp_params
        if fsdp_param.sharded_param.requires_grad and id(fsdp_param.sharded_param) in adapters
    )


def reject_peft_model_below_fsdp2_root(model, *, reshard_after_forward: bool) -> None:
    """``model`` is the FSDP2 root. Under ZeRO-2 a forward leaves each group gathered, and PEFT's ``disable_adapter()``
    reshards only the FSDP2 modules inside the ``PeftModel`` before it enables the adapters again, so the adapters a
    group above it holds would stay frozen after a reference pass. Rank-uniform: every rank builds the same wrap."""
    peft_model = find_peft_model(model)
    if peft_model is None or peft_model is model or reshard_after_forward:
        return
    held = _adapters_held_outside(model, peft_model)
    if held:
        # FSDP2 swaps the root's class for a subclass of it.
        wrapper = next(cls for cls in type(model).__mro__ if not issubclass(cls, FSDPModule)).__name__
        raise ValueError(
            f"The PeftModel sits inside a {wrapper}, which FSDP2 shards as its root, and under ZeRO-2 "
            f"(fsdp_reshard_after_forward: false) a forward leaves the root group gathered. After a disable_adapter() "
            f"reference pass PEFT reshards only the FSDP2 modules inside the PeftModel before it enables the adapters "
            f"again, so the {len(held)} adapter tensors the root group holds would stay frozen "
            f"({held[:KEY_PREVIEW_COUNT]}). Hand the trainer the PeftModel itself (get_peft_model over the "
            f"{wrapper}), set fsdp_reshard_after_forward: true where the run has no EP group, or keep "
            f"lora_target_modules and lora_modules_to_save off the modules those tensors adapt."
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
    selected = {id(tensor) for tensor in live_adapter_tensors(peft_model).values()}
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
