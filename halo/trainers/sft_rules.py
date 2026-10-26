"""The SFT script's data-option refusals, as pure functions of what they read, run before the model load."""

from halo.config_validation import reject_non_default_args
from halo.data.collators.collator_rules import reject_padding_free_conflicts
from halo.models.context_window import is_bounded_length


def reject_unsupported_sft_data_options(sft_config, dist_args) -> None:
    """Refuse the packing and padding-free combinations no SFT run can serve, before the checkpoint probe."""
    reject_padding_free_conflicts(
        padding_free=sft_config.padding_free,
        packing=sft_config.packing,
        use_context_parallel=dist_args.context_parallel_size > 1,
    )
    if dist_args.pipeline_parallel_size > 1 and sft_config.padding_free:
        # The trainer refuses this too, keyed on the live collator, but only after the model load and the dataset prep.
        raise ValueError(
            "Pipeline Parallelism (PP) is NOT compatible with padding_free mode (the flattened "
            "width varies every step while the pipeline's P2P buffers freeze on the first). Use "
            "packing instead."
        )
    # A null/non-positive max_length resolves to the model's context window, but packing uses it as the
    # fixed pack size that bounds memory, so it must be explicit there.
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
    """Refuse the options tree packing cannot honour.

    A tree row that reaches the model without its attention plan runs as one causal sequence and its
    branches attend each other, so every path that would drop or reshape the plan is closed here
    instead of being left to produce a plausible loss.
    """
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


def reject_unsupported_vlm_sft_options(args, sft_config, *, preprocessed: bool) -> None:
    """Refuse the options the VLM path does not implement, which would otherwise train something other than
    the config states; a pre-processed dataset trains as baked, so only ``tree_packing`` is refused on it."""
    reject_non_default_args("Halo SFT on the VLM path", args, "tree_packing")
    if preprocessed:
        return
    if sft_config.packing or sft_config.padding_free:
        raise ValueError("packing / padding_free are not supported for VLM training (images cannot be packed).")
    if args.train_on_last_assistant_only:
        raise ValueError(
            "train_on_last_assistant_only is not implemented on the VLM path (all assistant turns train)."
        )
