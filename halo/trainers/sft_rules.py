"""The SFT script's data-option and MTP-head refusals, run before the model load."""

from halo.config_validation import reject_non_default_args
from halo.data.collators.collator_rules import reject_padding_free_conflicts
from halo.env import is_accelerate_launch
from halo.model_loading.dtype import resolve_quantization_config
from halo.models.context_window import is_bounded_length
from halo.models.mtp.spec import require_mtp_head_spec


def reject_unsupported_sft_data_options(sft_config, dist_args) -> None:
    reject_padding_free_conflicts(
        padding_free=sft_config.padding_free,
        packing=sft_config.packing,
        use_context_parallel=dist_args.context_parallel_size > 1,
    )
    if dist_args.pipeline_parallel_size > 1 and sft_config.padding_free:
        # The trainer refuses this too, but only after the model load and the dataset prep.
        raise ValueError(
            "Pipeline Parallelism (PP) is NOT compatible with padding_free mode (the flattened "
            "width varies every step while the pipeline's P2P buffers freeze on the first). Use "
            "packing instead."
        )
    # Packing uses max_length as the fixed pack size that bounds memory; null resolves to the context window.
    if sft_config.packing and not is_bounded_length(sft_config.max_length):
        raise ValueError(
            "packing=True requires an explicit max_length (the fixed pack/sequence size that bounds "
            "memory); it cannot default to the model context window. Set max_length in the config."
        )
    if sft_config.eval_packing and not sft_config.packing:
        raise ValueError(
            "eval_packing=True packs nothing without packing=True: it can only turn packing off for the "
            "eval split. Set packing: true to pack both splits, or remove eval_packing."
        )


def reject_unsupported_tree_packing(args, sft_config, dist_args) -> None:
    """A tree row that loses its attention plan runs as one causal sequence, so its branches attend each other at a
    plausible loss. Closes every path that would drop or reshape the plan."""
    row_parallelism = (dist_args.context_parallel_size, dist_args.pipeline_parallel_size)
    requirements = {
        "tree_share_markers (what the rows share is found up to one of them)": bool(args.tree_share_markers),
        "packing: true (trees are packed rows)": sft_config.packing,
        "packing_strategy: bfd (the other strategies cut a row below a tree)": sft_config.packing_strategy == "bfd",
        "eval_packing left on (one collator serves both splits)": sft_config.eval_packing is not False,
        "train_on_completions_only: true (a shared prefix carries no labels)": args.train_on_completions_only,
        "no context or pipeline parallelism (each slices or re-collates the row)": max(row_parallelism) == 1,
        "no tensor parallelism (unverified under tree attention)": dist_args.tensor_parallel_size == 1,
        "torch_compile off (unverified under tree attention)": not sft_config.torch_compile,
        "train_sinks off (the sink-gradient rescale is unverified under tree attention)": not args.train_sinks,
    }
    unmet = [requirement for requirement, met in requirements.items() if not met]
    if unmet:
        raise ValueError("tree_packing needs: " + "; ".join(unmet) + ".")


def reject_unsupported_mtp_options(args, sft_config, model_config, dist_args, checkpoint_config) -> None:
    """Refuse what a run with a head cannot honor, naming each requirement it misses. A checkpoint config the probe
    could not read leaves the family's verdict to the load."""
    if args.mtp_loss_weight == 0:
        return
    spec = require_mtp_head_spec(checkpoint_config) if checkpoint_config is not None else None
    requirements = {
        "context_parallel_size: 1 (the CP wrapper forwards no labels to the head)": dist_args.context_parallel_size
        == 1,
        "tensor_parallel_size: 1 (TP shards the decoder list only, so the head stays replicated)": (
            dist_args.tensor_parallel_size == 1
        ),
        "pipeline_parallel_size: 1 (the head needs the embeddings as a second stream across stages)": (
            dist_args.pipeline_parallel_size == 1
        ),
        "use_hsdp off (unverified with the head's own shard group)": not dist_args.use_hsdp,
        "a torchrun launch (the accelerate launch does not take the head's shard group)": not is_accelerate_launch(),
        "use_peft off (an adapter would freeze the head)": not model_config.use_peft,
        "no quantized load (the head's tail loads as floats)": (
            resolve_quantization_config(model_config, sft_config) is None
        ),
        "lowp_precision: bf16 (unverified on the head's layer)": dist_args.lowp_precision == "bf16",
        "torch_compile off (the head runs in forward hooks)": not sft_config.torch_compile,
        "tree_packing off (a tree row's branches share no next token)": not args.tree_packing,
        "images_field unset (the head trains on text rows)": args.images_field is None,
        "label_smoothing_factor: 0 (HF pops the labels the head's term reads)": not sft_config.label_smoothing_factor,
        "loss_type: nll (the head's term sums one cross-entropy)": sft_config.loss_type
        in (None, "nll", "chunked_nll"),
        "save_sharded_ep off (this family's export writes trunk copies only a gathered save holds)": (
            spec is None or not spec.trunk_copies or not args.save_sharded_ep
        ),
        "mtp_head_source unset under init_from_scratch (that run draws its head)": (
            args.mtp_head_source is None or not dist_args.init_from_scratch
        ),
    }
    if args.mtp_train_scope == "head_only":
        requirements |= {
            "gradient_checkpointing off (the frozen trunk runs with no graph, so only replay caches would be added)": (
                not sft_config.gradient_checkpointing
            ),
            "gradient_checkpointing_plan: off (it measures a trunk that holds no graph)": (
                args.gradient_checkpointing_plan == "off"
            ),
            "freeze_layers_patterns and unfreeze_layers_patterns unset (the scope decides what trains)": (
                args.freeze_layers_patterns is None and args.unfreeze_layers_patterns is None
            ),
            "init_from_scratch off (a frozen trunk of random weights gives the head nothing to fit)": (
                not dist_args.init_from_scratch
            ),
            "moe_balancing: auto or none (a frozen trunk's routers neither train nor count their load)": (
                args.moe_balancing in ("auto", "none")
            ),
            "fp32_non_ep_params off (nothing casts a frozen trunk's fp32 masters to the compute dtype)": (
                not dist_args.fp32_non_ep_params
            ),
            "fp32_experts off (unverified on a frozen trunk)": not dist_args.fp32_experts,
            "an eval split, or mtp_min_acceptance_gain: null (the acceptance gate scores held-out rows)": (
                args.mtp_min_acceptance_gain is None or sft_config.eval_strategy != "no"
            ),
        }
    unmet = [requirement for requirement, met in requirements.items() if not met]
    if unmet:
        raise ValueError(
            f"mtp_loss_weight with mtp_train_scope: {args.mtp_train_scope} needs: " + "; ".join(unmet) + "."
        )


def settle_mtp_balancing(args) -> None:
    """``auto`` on a frozen trunk resolves to ``none``, whose routers neither train nor count their load; the refusal
    above leaves ``none`` the one other value here."""
    if args.mtp_train_scope == "head_only":
        args.moe_balancing = "none"


def reject_unsupported_vlm_sft_options(args, sft_config, *, preprocessed: bool) -> None:
    """Refuse what the VLM path does not implement; a pre-processed dataset trains as baked, so only ``tree_packing``
    and the MTP head are refused on it."""
    reject_non_default_args("Halo SFT on the VLM path", args, "tree_packing", "mtp_loss_weight")
    if preprocessed:
        return
    if sft_config.packing or sft_config.padding_free:
        raise ValueError("packing / padding_free are not supported for VLM training (images cannot be packed).")
    if args.train_on_last_assistant_only:
        raise ValueError(
            "train_on_last_assistant_only is not implemented on the VLM path (all assistant turns train)."
        )
