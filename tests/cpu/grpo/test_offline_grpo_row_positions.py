#!/usr/bin/env python
"""Offline GRPO's real collator, join and scorer on a family whose attention reads row indices: a row's completion
log-probs match the row forwarded alone. DeepSeek-V4's compressors cut KV windows at fixed indices, Mistral 4's query
scale steps at a position, so a left-padded row would score as another row."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from datasets import Dataset
from transformers import Mistral3Config, MistralConfig

from halo.configs.grpo_offline import OfflineGRPOConfig
from halo.data.collators.offline_grpo import OfflineGRPODataCollatorWithPadding
from halo.distributed.parallelism_config import ParallelismConfig
from halo.models.moe.family_specs import attention_reads_row_positions
from halo.trainers.grpo.logprob_scoring import LogprobScoringMixin, rows_forward_densely
from halo.trainers.grpo.offline import trainer as offline_module
from halo.trainers.grpo.offline.trainer import OfflineGRPOTrainer
from tests.common.offline_grpo import (
    MISTRAL4_SCALE_STEP,
    MISTRAL4_STEPPED_ROPE,
    OFFLINE_WORDS,
    make_offline_tokenizer,
    offline_grpo_config,
    offline_grpo_dataset,
)
from tests.common.tiny_models import TINY_DENSE_FAMILIES, TINY_MOE_FAMILIES, TINY_MOE_VLM_FAMILIES, tiny_family_model

_STEPPED = {"rope_parameters": MISTRAL4_STEPPED_ROPE}
# The published Mistral Small 4 checkpoints are the composite ``mistral3``.
_ROWS = {
    "deepseek_v4": (TINY_MOE_FAMILIES["deepseek_v4"], {}, (24, 6), "eager"),
    "mistral4": (TINY_MOE_FAMILIES["mistral4"], _STEPPED, (40, 6), "sdpa"),
    "mistral3": (TINY_MOE_VLM_FAMILIES["mistral3"], _STEPPED, (40, 6), "sdpa"),
}
_COMPLETIONS = (8, 4)


class _Scorer(LogprobScoringMixin):
    temperature = 1.0


def _model(family, overrides, attn):
    torch.manual_seed(0)
    model = tiny_family_model(family, overrides=overrides).float().eval()
    model.config._attn_implementation = attn
    return model


def _documents(model, prompts) -> list[list[int]]:
    vocab = model.config.get_text_config().vocab_size
    generator = torch.Generator().manual_seed(1)
    return [
        torch.randint(3, vocab, (prompt + completion,), generator=generator).tolist()
        for prompt, completion in zip(prompts, _COMPLETIONS, strict=True)
    ]


def _alone(model, document, completion: int) -> torch.Tensor:
    ids = torch.tensor([document])
    logits = model(input_ids=ids, use_cache=False).logits[0, :-1].float()
    return torch.log_softmax(logits, -1).gather(-1, ids[0, 1:, None])[-completion:, 0]


@pytest.mark.parametrize("name", list(_ROWS))
def test_a_left_padded_row_scores_as_it_does_alone(name):
    family, overrides, prompts, attn = _ROWS[name]
    model = _model(family, overrides, attn)
    documents = _documents(model, prompts)
    features = [
        {
            "prompt_input_ids": document[:prompt],
            "completion_input_ids": document[prompt:],
            "group_id": 0,
            "group_size": 2,
            "advantage": 1.0,
        }
        for document, prompt in zip(documents, prompts, strict=True)
    ]
    batch = OfflineGRPODataCollatorWithPadding(pad_token_id=0)(features)
    ids, mask, keep = OfflineGRPOTrainer._joined_rows(batch)
    assert rows_forward_densely(model, batch_size=2), "the premise: batch 2 alone takes the padded path"
    with torch.no_grad():
        logps, _ = _Scorer()._chunked_logps_impl(model, ids, mask, keep, 2, False)
        for row, (document, completion) in enumerate(zip(documents, _COMPLETIONS, strict=True)):
            worst = (logps[row, :completion] - _alone(model, document, completion)).abs().max()
            assert worst < 1e-4, f"row {row} moved by {worst} against itself alone"


def test_a_relative_attention_keeps_the_padded_path_at_batch_two():
    model = tiny_family_model(TINY_DENSE_FAMILIES["qwen3"])
    model.config._attn_implementation = "sdpa"
    assert not rows_forward_densely(model, batch_size=2)
    assert not attention_reads_row_positions(model.config)
    assert not attention_reads_row_positions(Mistral3Config(text_config=MistralConfig())), "a dense tower's wrapper"


def _mixed_length_groups(count: int) -> Dataset:
    """A long prompt beside a short one, so a padded batch would push the short row past the fixed indices."""
    records = []
    for index in range(count):
        word = OFFLINE_WORDS[index % len(OFFLINE_WORDS)]
        repeats = 1 if index % 2 else MISTRAL4_SCALE_STEP
        records.append(
            {
                "prompt": " ".join(["question", *[word] * repeats, "answer"]),
                "completions": [f"yes {word}", f"no {word} maybe"],
                "rewards": [1.0, -1.0],
            }
        )
    return Dataset.from_list(records)


@pytest.mark.parametrize("kl_beta", [0.0, 0.1], ids=["no_reference", "reference_sweep"])
@pytest.mark.parametrize("name", list(_ROWS))
def test_a_full_run_forwards_every_row_alone_with_no_flag_set(tmp_path, name, kl_beta):
    """The trainer as a user runs it, ``use_chunked_grpo_logprobs`` unset at batch 2, its run-start reference sweep and
    its training steps both: no forward of the backbone may hold a pad or a second row."""
    family, overrides, _prompts, attn = _ROWS[name]
    tokenizer = make_offline_tokenizer()
    torch.manual_seed(0)
    model = tiny_family_model(family, tokenizer, overrides=overrides).float()
    model.config._attn_implementation = attn
    forwards: list[tuple[int, bool, bool]] = []
    model.base_model.register_forward_pre_hook(
        lambda _module, _args, kwargs: forwards.append(
            (
                kwargs["input_ids"].size(0),
                kwargs.get("attention_mask") is None,
                bool((kwargs["input_ids"][0] == tokenizer.pad_token_id).any()),
            )
        ),
        with_kwargs=True,
    )
    args = offline_grpo_config(
        str(tmp_path),
        steps=2,
        save_steps=2,
        seed=0,
        kl_beta=kl_beta,
        sequence_length=64,
        evaluate=False,
        save=False,
        per_device_train_batch_size=2,
        gradient_checkpointing=False,
        use_chunked_grpo_logprobs=False,
        use_cpu=True,
        bf16=False,
    )
    trainer = OfflineGRPOTrainer(
        model=model,
        args=args,
        train_dataset=_mixed_length_groups(8),
        processing_class=tokenizer,
        parallelism_config=ParallelismConfig(),
        moe_balancing="none",
        resume_checkpoint=None,
    )
    result = trainer.train()
    assert result.global_step == 2 and torch.isfinite(torch.tensor(result.training_loss))
    assert len(forwards) >= 2 * args.per_device_train_batch_size
    assert forwards == [(1, True, False)] * len(forwards), forwards


class _StopAtTheScorerChoice(Exception):
    pass


@pytest.mark.parametrize(
    ("family", "chunked"),
    [
        (TINY_MOE_FAMILIES["deepseek_v4"], True),
        (TINY_MOE_FAMILIES["mistral4"], True),
        (TINY_MOE_VLM_FAMILIES["mistral3"], True),
        (TINY_DENSE_FAMILIES["qwen3"], False),
    ],
    ids=["deepseek_v4", "mistral4", "mistral3", "qwen3"],
)
def test_the_trainer_takes_the_chunked_scorer_on_such_a_family(tmp_path, family, chunked):
    """The full-logits forward has no dense path, so a left-padded row reaches attention on it."""
    args = OfflineGRPOConfig(
        output_dir=str(tmp_path), use_cpu=True, bf16=False, use_liger_kernel=False, report_to="none"
    )
    trainer = OfflineGRPOTrainer.__new__(OfflineGRPOTrainer)
    with (
        patch.object(offline_module, "prepare_policy_model", side_effect=_StopAtTheScorerChoice),
        pytest.raises(_StopAtTheScorerChoice),
    ):
        trainer.__init__(
            model=tiny_family_model(family),
            args=args,
            parallelism_config=ParallelismConfig(),
            train_dataset=offline_grpo_dataset(1),
            processing_class=make_offline_tokenizer(),
        )
    assert trainer._use_chunked_grpo_logprobs is chunked


def _pipeline_construction(tmp_path, family, **batch_sizes):
    """The constructor up to its scorer choice, under a pipeline."""
    args = OfflineGRPOConfig(
        output_dir=str(tmp_path),
        use_cpu=True,
        bf16=False,
        use_liger_kernel=False,
        report_to="none",
        max_prompt_length=8,
        max_completion_length=8,
        **batch_sizes,
    )
    trainer = OfflineGRPOTrainer.__new__(OfflineGRPOTrainer)
    pipeline = SimpleNamespace(is_pp_mode=True, is_cp_mode=False, expert_lora=None)
    with (
        patch.object(offline_module, "prepare_policy_model", side_effect=_StopAtTheScorerChoice),
        pytest.raises(_StopAtTheScorerChoice),
    ):
        trainer.__init__(
            model=tiny_family_model(family),
            args=args,
            parallelism_config=pipeline,
            train_dataset=offline_grpo_dataset(1),
            processing_class=make_offline_tokenizer(),
        )
    return trainer


def test_a_pipeline_cannot_forward_rows_alone_so_one_row_per_batch_is_all_it_takes(tmp_path):
    """Its stages forward the collated batch whole, and DeepSeek-V4 is a registered pipeline family."""
    family = TINY_MOE_FAMILIES["deepseek_v4"]
    one_row = {"per_device_train_batch_size": 1, "per_device_eval_batch_size": 1}
    assert not _pipeline_construction(tmp_path, family, **one_row)._use_chunked_grpo_logprobs
    with pytest.raises(ValueError, match="row index"):
        _pipeline_construction(tmp_path, family, per_device_train_batch_size=2, per_device_eval_batch_size=1)
    with pytest.raises(ValueError, match="row index"):
        _pipeline_construction(
            tmp_path, family, per_device_train_batch_size=1, per_device_eval_batch_size=2, eval_strategy="steps"
        )


def test_a_pipeline_on_a_relative_attention_keeps_its_batch(tmp_path):
    family = TINY_DENSE_FAMILIES["qwen3"]
    trainer = _pipeline_construction(tmp_path, family, per_device_train_batch_size=2, per_device_eval_batch_size=2)
    assert not trainer._use_chunked_grpo_logprobs


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
