"""The collator ``scripts/training/sft.py`` selects, for GPU suites and benchmarks training on rows TRL tokenizes."""

from halo.data.collators.collator_rules import evaluation_batch_size
from halo.data.collators.factory import select_data_collator


def build_sft_collator(tokenizer, sft_config, model_config, parallelism_config):
    """The SFT script's collator for full-sequence rows; None under ``packing`` or ``padding_free``, whose rows TRL
    packs and collates itself."""
    if sft_config.packing or sft_config.padding_free:
        return None
    cp_mode = parallelism_config.is_cp_mode
    return select_data_collator(
        tokenizer,
        train_on_completions_only=False,
        model_config=model_config,
        pad_to_multiple_of=parallelism_config.cp_size if cp_mode else sft_config.pad_to_multiple_of,
        use_context_parallel=cp_mode,
        per_device_train_batch_size=sft_config.per_device_train_batch_size,
        per_device_eval_batch_size=evaluation_batch_size(sft_config),
        keeps_packed_rows=parallelism_config.pp_size > 1,
        emit_varlen_one_document=True,
    )
