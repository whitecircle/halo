#!/usr/bin/env python
"""Distributed reward-model training with Expert and Tensor Parallelism support.

Bradley-Terry reward modeling on preference pairs (chosen vs rejected), text or vision-language.
The model class follows the checkpoint (``AutoModelForSequenceClassification``); the data path
follows the run (``is_vlm_run``), so a natively-multimodal checkpoint trained on text-only pairs
takes the text path, where TRL chat-templates and tokenizes the raw columns itself. A run that
declares image data loads an ``AutoProcessor`` as the processing class and takes the vision pair:
the rendered-text map in ``src/data/pipeline/preferences.py``, placeholder expansion at collation in
``src/data/collators/vlm_preference.py``.

Multimodal reward modeling needs a multimodal sequence-classification head, which transformers
ships for only a few families (Gemma3, dense Qwen3.5, T5Gemma2, ModernVBert); the toolkit registers
two more itself in src/models/seq_cls_heads.py — Gemma 4 and MoE Qwen3.5/3.6, each in both
spellings a checkpoint can carry (composite and text tower). Any other multimodal checkpoint is refused up front, before the model load.

CP is not supported (the score head pools the complete sequence); use EP and/or TP. PP admits a
multimodal checkpoint only for a run that feeds no images: the split keeps the text tower and score
head, the untouched vision tensors ride every checkpoint under the wrapper layout, and an
image-feeding run is refused by the pipeline gate.

Usage:
    torchrun --nproc_per_node=8 scripts/training/preference/rewards.py \\
        examples/reward/qwen3_5/rm-qwen3.5-9b-skywork-pref80k.yaml --expert_parallel_size=8
"""

from transformers import AutoModelForSequenceClassification
from trl import ModelConfig, RewardConfig

from src.args.distributed_args import DistributedArguments
from src.args.reward_args import RMScriptArguments
from src.data.sources.loading import alias_images_column, alias_tools_column
from src.data.vlm import is_vlm_run
from src.distributed.loading.peft_setup import setup_peft_model
from src.distributed.loading.vlm_setup import (
    load_vlm_processor,
    require_multimodal_sequence_classification_head,
)
from src.distributed.runtime import barrier
from src.models.loading.model_preparation import log_model_info
from src.trainers.reward.bradley_terry import DistributedRewardTrainer
from src.training.environment import run_training
from src.training.parser import H4ArgumentParser
from src.training.script_runner import (
    apply_distributed_trainer_config,
    apply_max_length,
    build_training_callbacks,
    distributed_trainer_kwargs,
    init_training_script,
    install_resolved_tokenizer,
    load_script_datasets,
    load_script_model,
    log_script_dataset_examples,
    padded_workload_attn_implementation,
    reject_unsupported_args,
    run_trainer,
)


def main():
    parser = H4ArgumentParser((RMScriptArguments, RewardConfig, ModelConfig, DistributedArguments))
    args, reward_config, model_config, dist_args = parser.parse()

    # text_only_model: the score head is pinned to AutoModelForSequenceClassification, so the loader
    # would only warn and build the wrapper anyway. log_decoded_samples: the rows reach the trainer
    # untokenized on both arms (TRL / the VLM map tokenize inside it), so the decoded-sample writer
    # would have nothing to decode. Reject both rather than accept a flag that cannot take effect.
    reject_unsupported_args(
        "Reward modeling",
        text_only_model=dist_args.text_only_model,
        log_decoded_samples=args.log_decoded_samples,
    )

    # RewardConfig also declares eos_token/pad_token; the resolve-conflict parser captures the YAML
    # keys there, so sync_tokens mirrors them back onto the script args the tokenizer setup reads.
    runtime = init_training_script(
        args,
        reward_config,
        model_config,
        dist_args,
        script_prefix="reward",
        trainer_cls=DistributedRewardTrainer,
        supports_cp=False,
        sync_tokens=("eos_token", "pad_token"),
    )
    parallelism_config = runtime.parallelism_config

    # Before the model load: a multimodal family with no score head otherwise fails inside Auto*
    # resolution on every rank, naming no alternative. After the distributed init, so the hub read it
    # makes is ordered main-rank-first.
    require_multimodal_sequence_classification_head(model_config)

    model, tokenizer = load_script_model(
        runtime,
        reward_config,
        model_config,
        dist_args,
        model_class=AutoModelForSequenceClassification,
        model_config_overrides={"num_labels": 1},
        attn_implementation=padded_workload_attn_implementation(model_config, sinks_reset=dist_args.reset_sinks),
    )
    tokenizer = apply_max_length(reward_config, args, model, tokenizer)

    peft_config = setup_peft_model(args, model, model_config, "SEQ_CLS")

    log_model_info(model, tokenizer)

    ds, dataset_presharded = load_script_datasets(args, parallelism_config)
    # Ahead of the dispatch: is_vlm_run reads images_field while the vision map reads the aliased
    # column, so the alias keeps the two verdicts identical and a mistyped name detectable.
    ds = alias_images_column(ds, args.images_field, str(args.dataset))
    # TRL's RewardTrainer renders the column literally named ``tools`` into the chat template and no
    # other, so a declared tools column is aliased onto that spelling rather than rendered toolless.
    ds = alias_tools_column(ds, args.tools_field, str(args.dataset))

    # The run's data path, decided now that the dataset is known: a multimodal checkpoint carrying
    # text-only pairs is a text run. The model class is unaffected; it followed the checkpoint above.
    is_vlm = is_vlm_run(args, model_config.model_name_or_path, ds, config=model.config)
    if is_vlm:
        # The VLM pair render (src/data/pipeline/preferences.py) templates the prompt without
        # `tools=`, so the aliased column would render toolless on this arm.
        reject_unsupported_args("Reward modeling (VLM)", tools_field=args.tools_field)
    # The processor only on the VLM branch: TRL's text tokenize map calls tokenizer-only methods
    # (get_vocab, pad_token) a ProcessorMixin does not carry, and the trainer reads the modality of
    # this object as the run's verdict. Its inner tokenizer is the resolved, right-padded one, the
    # side TRL's text collator pads on and the one the pooled head's rightmost-non-pad rule expects.
    processing_class = tokenizer
    if is_vlm:
        processing_class = install_resolved_tokenizer(load_vlm_processor(model_config), tokenizer, True)

    # No pre-tokenization pass here on either branch: TRL's RewardTrainer chat-templates and tokenizes
    # the raw chosen/rejected columns itself (natively supporting implicit-prompt datasets and
    # filtering over-length rows), and the trainer's VLM branch renders the pairs in its own map.
    train_dataset = ds["train"]
    eval_dataset = ds["test"]

    log_script_dataset_examples({"train": train_dataset, "test": eval_dataset}, tokenizer, args, reward_config)

    apply_distributed_trainer_config(reward_config, parallelism_config)

    barrier()

    callbacks = build_training_callbacks(args, reward_config, model, parallelism_config)

    trainer = DistributedRewardTrainer(
        model=model,
        processing_class=processing_class,
        args=reward_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        callbacks=callbacks,
        **distributed_trainer_kwargs(args, dist_args, parallelism_config, dataset_presharded=dataset_presharded),
    )
    run_trainer(
        trainer,
        runtime,
        method_name="Reward",
        extra_start_log=[f"modality: {'vlm' if is_vlm else 'text'}"],
    )


if __name__ == "__main__":
    run_training(main)()
