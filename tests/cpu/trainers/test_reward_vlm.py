#!/usr/bin/env python
"""CPU tests for vision-language Bradley-Terry reward modeling.

The DATA path follows the RUN, not the checkpoint: a text preference recipe on a natively-multimodal
checkpoint (Gemma 4, Qwen3.5/3.6) must reach TRL's own tokenize map with a plain TOKENIZER, while a
run that declares image data loads an ``AutoProcessor`` — the object that puts
``processor_config.json`` beside the exported weights — and takes the vision map + collator.

Driven through ``rewards.py:main()`` rather than a restatement of the branch: the seam is only
correct if the script reaches it with the dataset in hand, after the model load.

Also pinned here:
* the trainer builds the vision collator for a processor and none for a tokenizer (collator and
  ``_prepare_dataset`` are one contract and must not be paired by hand);
* the vision columns survive HF's signature-column pruning;
* the checkpoint context hands the PROCESSOR to the save;
* a multimodal family with no sequence-classification head is refused before the distributed init,
  off the same config-modality predicate the run-time probe uses;
* pipeline parallelism still refuses a multimodal checkpoint through the shared gate.

Run: python tests/cpu/trainers/test_reward_vlm.py  (or pytest)
"""

import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch.nn as nn
from datasets import Dataset, DatasetDict
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
from trl import RewardTrainer

from src.data.collators.vlm_preference import DataCollatorForVLMPreference
from src.distributed.loading import vlm_setup
from src.distributed.parallelism_config import ParallelismConfig
from src.models.modality import config_declares_multimodality, is_vlm_model
from src.models.structure import resolve_tokenizer
from src.trainers.mixins.base import DistributedTrainerMixin
from src.trainers.reward.bradley_terry import DistributedRewardTrainer
from tests.common.utils import load_script_module
from tests.cpu.data.test_vlm_preference_collator import IMAGE_TOKEN, StubProcessor, StubTokenizer, make_row

# A model id the name heuristic reads as multimodal, so the checkpoint verdict holds offline.
_VLM_MODEL_ID = "stub/qwen3.5-9b"

_PROMPT = [{"role": "user", "content": "Describe the picture"}]
_CHOSEN = [{"role": "assistant", "content": "A red square"}]
_REJECTED = [{"role": "assistant", "content": "A blue circle"}]


class StubTextTokenizer(StubTokenizer):
    """The stub tokenizer plus the one method TRL's text reward map calls."""

    def apply_chat_template(self, messages, tools=None, return_dict=False, **kwargs):
        rendered = "".join(f"[{message['role']}] {message['content']} [end]\n" for message in messages)
        ids = self._encode(rendered)
        return {"input_ids": ids} if return_dict else rendered


@pytest.fixture(autouse=True)
def _accelerate_state(monkeypatch):
    from accelerate import PartialState

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    PartialState()  # the coordinated dataset ops and the accelerate logger need an initialized state


def _dataset(extra_columns: dict | None = None) -> DatasetDict:
    data = {"prompt": [_PROMPT], "chosen": [_CHOSEN], "rejected": [_REJECTED], **(extra_columns or {})}
    split = Dataset.from_dict(data)
    return DatasetDict({"train": split, "test": split})


def _image_dataset(column: str = "images") -> DatasetDict:
    row = make_row()
    data = {
        "prompt": [row["prompt"]],
        "chosen": [row["chosen"]],
        "rejected": [row["rejected"]],
        column: [row["images"]],
    }
    split = Dataset.from_dict(data)
    return DatasetDict({"train": split, "test": split})


def _run_rewards(tmp_path, dataset: DatasetDict, yaml_body: str = "", processor=None):
    """Run ``rewards.py:main()`` to trainer construction.

    Returns ``(captured trainer kwargs, tokenizer, processor, processor_loader)`` — the loader is a
    mock so a test can assert the text branch never builds a processor at all.
    """
    module = load_script_module("scripts/training/preference/rewards.py", "halo_test_rewards_dispatch")

    config = tmp_path / "config.yaml"
    config.write_text(
        f"model_name_or_path: {_VLM_MODEL_ID}\n"
        f"dataset:\n- dummy/dataset\n"
        f"output_dir: {tmp_path / 'out'}\n"
        f"bf16: false\nuse_cpu: true\nmax_length: 512\n{yaml_body}"
    )

    tokenizer = StubTextTokenizer()
    processor = processor if processor is not None else StubProcessor(StubTokenizer())
    model = SimpleNamespace(config=CONFIG_MAPPING["gemma3"]())
    runtime = SimpleNamespace(
        parallelism_config=SimpleNamespace(
            cp_size=1,
            is_cp_mode=False,
            is_ep_mode=False,
            pp_size=1,
            get_data_parallel_rank=lambda: 0,
            data_parallel_size=1,
        ),
        model_source=_VLM_MODEL_ID,
        mode_suffix="",
        local_rank=0,
        resume_checkpoint=None,
    )
    captured: dict = {}
    processor_loader = mock.Mock(return_value=processor)

    def capture_trainer(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    patches = [
        mock.patch.object(module, "require_multimodal_sequence_classification_head"),
        mock.patch.object(module, "init_training_script", return_value=runtime),
        mock.patch.object(module, "load_script_model", return_value=(model, tokenizer)),
        mock.patch.object(module, "apply_max_length", side_effect=lambda cfg, args, model, tok: tok),
        mock.patch.object(module, "setup_peft_model", return_value=None),
        mock.patch.object(module, "log_model_info"),
        mock.patch.object(module, "load_script_datasets", return_value=(dataset, False)),
        mock.patch.object(module, "load_vlm_processor", processor_loader),
        mock.patch.object(module, "log_script_dataset_examples"),
        mock.patch.object(module, "build_training_callbacks", return_value=[]),
        mock.patch.object(module, "barrier"),
        mock.patch.object(module, "DistributedRewardTrainer", side_effect=capture_trainer),
        mock.patch.object(module, "run_trainer"),
        mock.patch("src.training.parser.install_log_tee"),
        mock.patch.object(sys, "argv", ["prog", str(config)]),
    ]
    with mock.patch.object(module, "run_training", lambda fn: fn):
        for patch in patches:
            patch.start()
        try:
            module.main()
        finally:
            for patch in reversed(patches):
                patch.stop()
    return captured, tokenizer, processor, processor_loader


# --- the run dispatch -----------------------------------------------------------------------------


def test_image_dataset_on_a_multimodal_checkpoint_hands_the_trainer_the_processor(tmp_path):
    captured, tokenizer, processor, _loader = _run_rewards(tmp_path, _image_dataset())

    assert captured["processing_class"] is processor
    # install_resolved_tokenizer is what makes the processor tokenize with the run's resolved,
    # right-padded tokenizer rather than the checkpoint's own.
    assert processor.tokenizer is tokenizer


def test_text_dataset_on_a_multimodal_checkpoint_still_hands_the_trainer_a_tokenizer(tmp_path):
    """Anti-vacuity for the dispatch: the text path must reach TRL with a TOKENIZER, and must not
    build a processor at all — TRL's tokenize map calls get_vocab/pad_token, which a processor
    does not carry."""
    captured, tokenizer, _processor, processor_loader = _run_rewards(tmp_path, _dataset())

    assert captured["processing_class"] is tokenizer
    processor_loader.assert_not_called()


def test_images_field_declares_the_run_and_aliases_the_column(tmp_path):
    """A hub column under another name: the alias is what keeps the run verdict and the map's column
    the same one."""
    captured, _tokenizer, processor, _loader = _run_rewards(
        tmp_path, _image_dataset(column="pictures"), yaml_body="images_field: pictures\n"
    )

    assert captured["processing_class"] is processor
    assert "images" in captured["train_dataset"].column_names


def test_mistyped_images_field_raises_before_training(tmp_path):
    with pytest.raises(ValueError, match="images_field='nope' names a column"):
        _run_rewards(tmp_path, _image_dataset(), yaml_body="images_field: nope\n")


# --- trainer construction -------------------------------------------------------------------------


def _construct(processing_class, max_length=64, data_collator=None):
    """Run the real ``DistributedRewardTrainer.__init__`` body with the bases stubbed out."""
    captured: dict = {}

    def fake_reward_init(self, *args, **kwargs):
        captured.update(kwargs)

    with (
        mock.patch.object(RewardTrainer, "__init__", fake_reward_init),
        mock.patch.object(DistributedTrainerMixin, "_init_distributed_config", lambda self, kwargs: kwargs),
        mock.patch.object(DistributedTrainerMixin, "_setup_distributed_modes", lambda self: None),
    ):
        trainer = DistributedRewardTrainer(
            model=nn.Linear(2, 2),
            processing_class=processing_class,
            args=SimpleNamespace(max_length=max_length),
            data_collator=data_collator,
        )
    return trainer, captured


def test_processor_selects_the_vision_collator_and_a_tokenizer_selects_trls():
    processor = StubProcessor(StubTokenizer())

    vlm_trainer, vlm_kwargs = _construct(processor)
    assert vlm_trainer._is_vlm is True
    assert isinstance(vlm_kwargs["data_collator"], DataCollatorForVLMPreference)
    assert vlm_kwargs["data_collator"].processor is processor
    assert vlm_kwargs["data_collator"].max_length == 64  # the run's budget, not a collator default

    text_trainer, text_kwargs = _construct(StubTextTokenizer())
    assert text_trainer._is_vlm is False
    assert text_kwargs["data_collator"] is None, "the text path must keep TRL's own preference collator"


def test_an_explicit_collator_is_not_overridden():
    sentinel = object()
    _trainer, kwargs = _construct(StubProcessor(StubTokenizer()), data_collator=sentinel)
    assert kwargs["data_collator"] is sentinel


def test_trl_gets_the_tokenizer_while_the_trainer_keeps_the_processor():
    """TRL's reward ctor settles the pad token through ``pad_token``/``get_vocab`` — attributes no
    real ProcessorMixin carries (verified against Qwen3VLProcessor), so handing it the processor
    raises AttributeError before the first batch. It must therefore see the inner tokenizer, while
    ``self.processing_class`` ends up the processor: that object is what ``save_pretrained`` writes,
    and without it the export ships no processor_config.json."""
    processor = StubProcessor(StubTokenizer())

    trainer, kwargs = _construct(processor)

    assert kwargs["processing_class"] is processor.tokenizer
    assert trainer.processing_class is processor
    assert trainer._processor is processor


# --- dataset preparation ---------------------------------------------------------------------------


def _bare_trainer(is_vlm: bool, processor=None):
    trainer = DistributedRewardTrainer.__new__(DistributedRewardTrainer)
    trainer._is_vlm = is_vlm
    trainer._processor = processor
    return trainer


def test_vlm_prepare_renders_both_sides_and_keeps_the_margin():
    processor = StubProcessor(StubTokenizer())
    row = make_row()
    dataset = Dataset.from_dict(
        {
            "prompt": [row["prompt"]],
            "chosen": [row["chosen"]],
            "rejected": [row["rejected"]],
            "images": [row["images"]],
            "margin": [0.25],
        }
    )

    prepared = _bare_trainer(True, processor)._prepare_dataset(
        dataset, resolve_tokenizer(processor), SimpleNamespace(dataset_num_proc=1, max_length=512), "train"
    )

    assert set(prepared.column_names) == {"chosen_text", "rejected_text", "images", "margin"}
    assert IMAGE_TOKEN in prepared[0]["chosen_text"]
    assert prepared[0]["margin"] == pytest.approx(0.25)
    # A second pass over already-rendered rows is a no-op, not a second (failing) map.
    assert (
        _bare_trainer(True, processor)._prepare_dataset(
            prepared, resolve_tokenizer(processor), SimpleNamespace(dataset_num_proc=1, max_length=512), "train"
        )
        is prepared
    )


def test_vlm_prepare_drops_rows_over_the_text_budget():
    processor = StubProcessor(StubTokenizer())
    short = make_row(chosen="Short", rejected="Tiny")
    long_row = make_row(chosen="A very long winded answer indeed going on and on", rejected="Tiny")
    dataset = Dataset.from_list([short, long_row])

    prepared = _bare_trainer(True, processor)._prepare_dataset(
        dataset, resolve_tokenizer(processor), SimpleNamespace(dataset_num_proc=1, max_length=12), "train"
    )

    assert len(prepared) == 1
    assert "Short" in prepared[0]["chosen_text"]


def test_vlm_prepare_raises_when_the_budget_empties_the_split():
    """The verdict is agreed across ranks before the raise: on a presharded corpus the split is a
    per-rank fact, and a lone raise would strand the peers in the trainer's construction collectives."""
    processor = StubProcessor(StubTokenizer())
    dataset = Dataset.from_list([make_row()])
    probes: list[bool] = []

    def _agree(local: bool, subject, probe: str) -> bool:
        probes.append(local)
        return local

    with (
        mock.patch("src.trainers.reward.bradley_terry.agree_probe_across_ranks", _agree),
        pytest.raises(ValueError, match="exceed max_length=2"),
    ):
        _bare_trainer(True, processor)._prepare_dataset(
            dataset, resolve_tokenizer(processor), SimpleNamespace(dataset_num_proc=1, max_length=2), "eval"
        )
    assert probes == [True]


def test_text_prepare_still_runs_trls_own_tokenize_map():
    """Anti-vacuity: the text branch must produce TRL's chosen_ids/rejected_ids, not our columns."""
    tokenizer = StubTextTokenizer()
    dataset = Dataset.from_dict({"prompt": [_PROMPT], "chosen": [_CHOSEN], "rejected": [_REJECTED]})

    prepared = _bare_trainer(False)._prepare_dataset(
        dataset, tokenizer, SimpleNamespace(dataset_num_proc=1, max_length=512), "train"
    )

    assert {"chosen_ids", "rejected_ids"} <= set(prepared.column_names)
    assert "chosen_text" not in prepared.column_names


def test_signature_columns_keep_the_vision_columns():
    """TRL names only chosen_ids/rejected_ids/margin; without the collator's declaration every
    column the vision collator reads is pruned before the first batch."""
    trainer = _bare_trainer(True)
    trainer._signature_columns = None
    trainer.data_collator = DataCollatorForVLMPreference(processor=StubProcessor(StubTokenizer()))

    trainer._set_signature_columns_if_needed()

    assert {"chosen_text", "rejected_text", "images"} <= set(trainer._signature_columns)
    assert {"chosen_ids", "rejected_ids", "margin"} <= set(trainer._signature_columns)


# --- export -----------------------------------------------------------------------------------------


def test_the_checkpoint_context_hands_the_processor_to_the_save():
    """The processor IS the saved processing class — that is what writes processor_config.json
    next to the weights, without which the exported reward model cannot be served on images."""
    processor = StubProcessor(StubTokenizer())
    model = nn.Linear(2, 2)
    trainer = _bare_trainer(True)
    trainer.processing_class = processor
    trainer.parallelism_config = ParallelismConfig()
    trainer.model = model
    trainer._fsdp_wrapped = False
    trainer._accelerate_manages_fsdp = False
    trainer.save_sharded_ep = False
    trainer.args = SimpleNamespace(save_max_shard_size=None)

    with (
        mock.patch.object(DistributedRewardTrainer, "_has_ep_layers", False),
        mock.patch.object(DistributedRewardTrainer, "_top_level_model", lambda self: model),
        mock.patch.object(DistributedRewardTrainer, "_get_tp_rank", lambda self: 0),
        mock.patch.object(DistributedRewardTrainer, "_find_cp_wrapper", lambda self: None),
        mock.patch("src.trainers.mixins.checkpointing.fs_aware_save_rank", return_value=True),
        mock.patch("src.trainers.mixins.checkpointing.has_ep_lora", return_value=False),
    ):
        ctx = trainer._checkpoint_context()

    assert ctx.tokenizer is processor


# --- family and parallelism gates ---------------------------------------------------------------------


def _model_config(name: str):
    return SimpleNamespace(model_name_or_path=name, trust_remote_code=False, model_revision=None)


@pytest.mark.parametrize("model_type", ["qwen3_vl", "llava"])
def test_multimodal_family_without_a_score_head_is_refused(model_type):
    with mock.patch.object(vlm_setup.AutoConfig, "from_pretrained", return_value=CONFIG_MAPPING[model_type]()):
        with pytest.raises(ValueError, match="no sequence-classification head") as excinfo:
            vlm_setup.require_multimodal_sequence_classification_head(_model_config(f"stub/{model_type}"))
    # Actionable: the message names the families that DO work, off the live registry.
    assert "gemma4" in str(excinfo.value)


@pytest.mark.parametrize("model_type", ["gemma3", "qwen3_5", "qwen3"])
def test_supported_and_text_only_families_pass_the_gate(model_type):
    with mock.patch.object(vlm_setup.AutoConfig, "from_pretrained", return_value=CONFIG_MAPPING[model_type]()):
        vlm_setup.require_multimodal_sequence_classification_head(_model_config(f"stub/{model_type}"))


def test_both_spellings_of_a_declared_vision_tower_classify_the_same():
    """The roster reads config CLASSES while the run-time probe reads config INSTANCES, and the two
    spell a vision tower differently: a class declares it in ``sub_configs``, while a remote-code
    config is in neither that nor the ITT mapping and only builds ``self.vision_config``. One
    predicate must see both spellings, or the gate above refuses a family the loader then routes to
    the VLM path — or passes one whose images it drops.
    """

    class _DeclaredOnTheClass:
        model_type = "halo_test_unregistered_vlm"
        sub_configs = {"vision_config": object}

    live_instance = SimpleNamespace(model_type="halo_test_unregistered_vlm", vision_config=SimpleNamespace())
    text_only = SimpleNamespace(model_type="halo_test_unregistered_text")

    assert config_declares_multimodality(_DeclaredOnTheClass) is True
    assert config_declares_multimodality(live_instance) is True
    assert config_declares_multimodality(text_only) is False
    # Unregistered, so the run-time probe honours the config signal instead of vetoing on model_type.
    assert is_vlm_model("stub/checkpoint", config=live_instance) is True


def test_a_text_backbone_registered_under_image_text_to_text_stays_text_only():
    """mistral4's ITT entry resolves to ``Mistral4ForCausalLM`` — a text backbone registered there as
    an upstream quirk. Reading that entry without the ``*ForCausalLM`` exception routes it to
    AutoProcessor/VLM setup, which fails outright on a family that ships no processor."""
    mistral4 = CONFIG_MAPPING["mistral4"]
    assert MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES["mistral4"].endswith("ForCausalLM"), "premise"

    assert config_declares_multimodality(mistral4) is False
    assert config_declares_multimodality(mistral4()) is False
    assert is_vlm_model("stub/mistral4-base", config=mistral4()) is False


def test_the_seq_cls_roster_holds_the_multimodal_families_and_no_text_one():
    """Anti-drift on the shared predicate's blast radius: this roster is what the refusal above
    offers as the way forward, so a family gained or lost here changes a user-facing gate."""
    roster = set(vlm_setup.multimodal_sequence_classification_model_types())

    assert {"gemma4", "qwen3_5", "qwen3_5_moe"} <= roster
    assert roster.isdisjoint({"mistral4", "qwen3", "qwen3_moe", "gpt_oss"})


def test_pipeline_parallelism_refuses_a_multimodal_reward_run_fed_images():
    """The shared PP gate binds through the mixin — the stage split keeps only the text backbone, so
    a multimodal checkpoint fed image data is refused (a text-only run of it is admitted)."""
    trainer = _bare_trainer(True)
    trainer.parallelism_config = SimpleNamespace(is_pp_mode=True)
    trainer.save_sharded_ep = False
    training_args = SimpleNamespace(
        max_length=512,
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs=None,
        eval_strategy="no",
        activation_offloading=False,
        torch_compile=False,
    )
    kwargs = {
        "model": SimpleNamespace(config=CONFIG_MAPPING["gemma3"]()),
        "train_dataset": Dataset.from_dict({"chosen": ["a"], "rejected": ["b"], "images": [[]]}),
    }

    with pytest.raises(ValueError, match="Vision-language training is not supported under pipeline"):
        trainer._maybe_prepare_pipeline_model(kwargs, training_args)


def test_reward_trainer_declares_no_context_parallel_support():
    """CP would split the sequence the score head pools; the declaration IS the gate."""
    assert DistributedRewardTrainer._supports_cp is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
