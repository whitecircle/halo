#!/usr/bin/env python
"""Vision-language Bradley-Terry reward modeling.
A text recipe on a multimodal checkpoint gets a plain tokenizer; a run with image data loads an ``AutoProcessor``."""

import sys
from types import SimpleNamespace
from unittest import mock

import pytest
import torch.nn as nn
from datasets import Dataset, DatasetDict
from transformers import AutoConfig
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.auto.modeling_auto import MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES
from trl import RewardConfig, RewardTrainer

from halo.data.collators.vlm_preference import DataCollatorForVLMPreference
from halo.distributed.parallelism_config import ParallelismConfig
from halo.entrypoints import model_setup
from halo.model_loading.tokenizer_setup import resolve_tokenizer
from halo.models.introspection.modality import config_declares_multimodality, is_vlm_model
from halo.models.upstream.seq_cls_heads import multimodal_sequence_classification_model_types
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from halo.trainers.reward.bradley_terry import DistributedRewardTrainer
from tests.common.repo import load_script_module
from tests.common.tiny_models import config_only_module
from tests.cpu.data.test_vlm_preference_collator import IMAGE_TOKEN, StubProcessor, StubTokenizer, make_row

# A model id the name heuristic reads as multimodal, so the checkpoint verdict holds offline.
_VLM_MODEL_ID = "stub/qwen3.5-9b"

_PROMPT = [{"role": "user", "content": "Describe the picture"}]
_CHOSEN = [{"role": "assistant", "content": "A red square"}]
_REJECTED = [{"role": "assistant", "content": "A blue circle"}]
_SIDES = ("prompt", "chosen", "rejected")


class StubTextTokenizer(StubTokenizer):
    """The stub tokenizer plus the one method TRL's text reward map calls."""

    def apply_chat_template(self, messages, tools=None, return_dict=False, **kwargs):
        rendered = "".join(f"[{message['role']}] {message['content']} [end]\n" for message in messages)
        ids = self._encode(rendered)
        return {"input_ids": ids} if return_dict else rendered


@pytest.fixture(autouse=True)
def _offline_hub(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


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


def _run_rewards(tmp_path, dataset: DatasetDict, yaml_body: str = "", processor=None, model_type: str = "gemma3"):
    """Run ``rewards.py:main()`` to trainer construction.
    The processor loader is a mock, so a test can assert the text branch never builds a processor."""
    module = load_script_module("scripts/training/reward/reward_model.py", "halo_test_rewards_dispatch")

    config = tmp_path / "config.yaml"
    config.write_text(
        f"model_name_or_path: {_VLM_MODEL_ID}\n"
        f"dataset:\n- dummy/dataset\n"
        f"output_dir: {tmp_path / 'out'}\n"
        f"bf16: false\nuse_cpu: true\nmax_length: 512\n{yaml_body}"
    )

    tokenizer = StubTextTokenizer()
    processor = processor if processor is not None else StubProcessor(StubTokenizer())
    model = nn.Module()
    model.config = CONFIG_MAPPING[model_type]()
    runtime = SimpleNamespace(
        parallelism_config=SimpleNamespace(
            cp_size=1,
            is_cp_mode=False,
            has_ep_group=False,
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
        mock.patch.object(module, "reject_multimodal_model_without_seq_cls_head"),
        mock.patch(
            "halo.entrypoints.data_setup.probe_checkpoint",
            return_value=(model.config, is_vlm_model(_VLM_MODEL_ID, model.config)),
        ),
        mock.patch.object(module, "init_training_script", return_value=runtime),
        mock.patch.object(module, "load_model_and_tokenizer", return_value=(model, tokenizer)),
        mock.patch.object(module, "apply_max_length", side_effect=lambda cfg, args, model, tok: tok),
        mock.patch.object(module, "configure_adapter_training", return_value=None),
        mock.patch.object(module, "log_model_info"),
        mock.patch.object(module, "load_script_datasets", return_value=(dataset, False)),
        mock.patch.object(module, "load_vlm_processor", processor_loader),
        mock.patch.object(module, "log_script_dataset_examples"),
        mock.patch.object(module, "configure_balancing_and_build_callbacks", return_value=[]),
        mock.patch.object(module, "barrier"),
        mock.patch.object(module, "DistributedRewardTrainer", side_effect=capture_trainer),
        mock.patch.object(module, "run_trainer"),
        mock.patch("halo.entrypoints.script.install_log_tee"),
        mock.patch.object(sys, "argv", ["prog", str(config)]),
    ]
    with mock.patch.object(module, "with_distributed_teardown", lambda fn: fn):
        for patch in patches:
            patch.start()
        try:
            module.main()
        finally:
            for patch in reversed(patches):
                patch.stop()
    return captured, tokenizer, processor, processor_loader


def test_image_dataset_on_a_multimodal_checkpoint_hands_the_trainer_the_processor(tmp_path):
    captured, tokenizer, processor, _loader = _run_rewards(tmp_path, _image_dataset())

    assert captured["processing_class"] is processor
    # install_resolved_tokenizer makes the processor tokenize with the run's resolved, right-padded tokenizer.
    assert processor.tokenizer is tokenizer


def test_text_dataset_on_a_multimodal_checkpoint_still_hands_the_trainer_a_tokenizer(tmp_path):
    """TRL's tokenize map calls get_vocab/pad_token, which a processor does not carry."""
    captured, tokenizer, _processor, processor_loader = _run_rewards(tmp_path, _dataset())

    assert captured["processing_class"] is tokenizer
    processor_loader.assert_not_called()


def test_images_field_declares_the_run_and_aliases_the_column(tmp_path):
    """A hub column under another name: the alias keeps the run verdict and the map's column the same one."""
    captured, _tokenizer, processor, _loader = _run_rewards(
        tmp_path, _image_dataset(column="pictures"), yaml_body="images_field: pictures\n"
    )

    assert captured["processing_class"] is processor
    assert "images" in captured["train_dataset"].column_names


@pytest.mark.parametrize(
    ("dataset", "yaml_body"),
    [(_image_dataset(), ""), (_image_dataset("pictures"), "images_field: pictures\n")],
    ids=["image_column", "images_field"],
)
def test_a_text_checkpoint_refuses_image_data_it_would_drop(tmp_path, dataset, yaml_body):
    with pytest.raises(ValueError, match="is a text checkpoint, but this run declares image data"):
        _run_rewards(tmp_path, dataset, yaml_body=yaml_body, model_type="qwen3")


def test_a_text_checkpoint_on_text_pairs_stays_a_text_run(tmp_path):
    captured, tokenizer, _processor, processor_loader = _run_rewards(tmp_path, _dataset(), model_type="qwen3")
    assert captured["processing_class"] is tokenizer
    processor_loader.assert_not_called()


@pytest.mark.parametrize("column", ["prompt", "chosen", "rejected"])
@pytest.mark.parametrize(
    ("part", "message"),
    [
        ({"type": "video", "video": "a.mp4"}, "no Halo collator feeds"),
        ({"type": "image", "image": "a.png"}, "on the TEXT pipeline"),
    ],
    ids=["video", "image"],
)
def test_a_text_run_refuses_a_media_part_on_any_row_ahead_of_trls_tokenize_map(tmp_path, column, part, message):
    """TRL templates these rows itself, into placeholders with no pixels. The first row carries none, so only a read of
    every row sees it, and no image column makes this an image run."""
    role = "user" if column == "prompt" else "assistant"
    rows = [
        {
            key: [{"role": role if key == column else "user", "content": [{"type": "text", "text": "hi"}]}]
            for key in _SIDES
        }
        for _ in range(2)
    ]
    rows[1][column] = [{"role": role, "content": [part, {"type": "text", "text": "this?"}]}]
    split = Dataset.from_list(rows)
    with pytest.raises(ValueError, match=message):
        _run_rewards(tmp_path, DatasetDict({"train": split, "test": split}))


def test_mistyped_images_field_raises_before_training(tmp_path):
    with pytest.raises(ValueError, match="images_field='nope' names a column"):
        _run_rewards(tmp_path, _image_dataset(), yaml_body="images_field: nope\n")


def _construct(processing_class, max_length=64, data_collator=None):
    captured: dict = {}

    def fake_reward_init(self, *args, **kwargs):
        captured.update(kwargs)

    with (
        mock.patch.object(RewardTrainer, "__init__", fake_reward_init),
        mock.patch.object(DistributedTrainerMixin, "_init_distributed_config", lambda self, kwargs, **_: kwargs),
        mock.patch.object(DistributedTrainerMixin, "_configure_distributed_modes", lambda self: None),
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
    """TRL's reward ctor settles the pad token through ``pad_token``/``get_vocab``, which no ProcessorMixin carries.
    ``self.processing_class`` stays the processor, since ``save_pretrained`` writes it into processor_config.json."""
    processor = StubProcessor(StubTokenizer())

    trainer, kwargs = _construct(processor)

    assert kwargs["processing_class"] is processor.tokenizer
    assert trainer.processing_class is processor
    assert trainer._processor is processor


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
    """On a presharded corpus the split is a per-rank fact, and a lone raise would strand the peers in the construction
    collectives."""
    processor = StubProcessor(StubTokenizer())
    dataset = Dataset.from_list([make_row()])
    probes: list[bool] = []

    def _agree(local: bool, subject, probe: str) -> bool:
        probes.append(local)
        return local

    with (
        mock.patch("halo.trainers.reward.bradley_terry.agree_probe_across_ranks", _agree),
        pytest.raises(ValueError, match="exceed max_length=2"),
    ):
        _bare_trainer(True, processor)._prepare_dataset(
            dataset, resolve_tokenizer(processor), SimpleNamespace(dataset_num_proc=1, max_length=2), "eval"
        )
    assert probes == [True]


def test_text_prepare_still_runs_trls_own_tokenize_map(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "hf_datasets"))
    tokenizer = StubTextTokenizer()
    dataset = Dataset.from_dict({"prompt": [_PROMPT], "chosen": [_CHOSEN], "rejected": [_REJECTED]})
    args = RewardConfig(
        output_dir=str(tmp_path), use_cpu=True, bf16=False, report_to=[], dataset_num_proc=1, max_length=512
    )

    prepared = _bare_trainer(False)._prepare_dataset(dataset, tokenizer, args, "train")

    assert {"chosen_ids", "rejected_ids"} <= set(prepared.column_names)
    assert "chosen_text" not in prepared.column_names


def test_signature_columns_keep_the_vision_columns():
    """TRL names only chosen_ids/rejected_ids/margin; undeclared vision columns are pruned before the first batch."""
    trainer = _bare_trainer(True)
    trainer._signature_columns = None
    trainer.data_collator = DataCollatorForVLMPreference(processor=StubProcessor(StubTokenizer()))

    trainer._set_signature_columns_if_needed()

    assert {"chosen_text", "rejected_text", "images"} <= set(trainer._signature_columns)
    assert {"chosen_ids", "rejected_ids", "margin"} <= set(trainer._signature_columns)


def test_the_checkpoint_context_hands_the_processor_to_the_save():
    """The processor writes processor_config.json next to the weights; without it the export cannot serve images."""
    processor = StubProcessor(StubTokenizer())
    model = nn.Linear(2, 2)
    trainer = _bare_trainer(True)
    trainer.processing_class = processor
    trainer.parallelism_config = ParallelismConfig()
    trainer.model = model
    trainer._fsdp_wrapped = False
    trainer._accelerate_manages_fsdp = False
    trainer.save_sharded_ep = False
    trainer.save_max_shard_size = None

    with (
        mock.patch.object(DistributedRewardTrainer, "_ep_layers", lambda self: []),
        mock.patch.object(DistributedRewardTrainer, "_top_level_model", lambda self: model),
        mock.patch.object(DistributedRewardTrainer, "_get_tp_rank", lambda self: 0),
        mock.patch.object(DistributedRewardTrainer, "_find_cp_wrapper", lambda self: None),
        mock.patch("halo.trainers.checkpointing.is_fs_save_rank", return_value=True),
        mock.patch("halo.trainers.checkpointing.model_has_expert_lora", return_value=False),
    ):
        ctx = trainer._checkpoint_context()

    assert ctx.tokenizer is processor


def _model_config(name: str):
    return SimpleNamespace(model_name_or_path=name, trust_remote_code=False, model_revision=None)


@pytest.mark.parametrize("model_type", ["qwen3_vl", "llava"])
def test_multimodal_family_without_a_score_head_is_refused(model_type):
    with mock.patch.object(AutoConfig, "from_pretrained", return_value=CONFIG_MAPPING[model_type]()):
        with pytest.raises(ValueError, match="no sequence-classification head") as excinfo:
            model_setup.reject_multimodal_model_without_seq_cls_head(_model_config(f"stub/{model_type}"))
    # The message names the families that work, off the live registry.
    assert "gemma4" in str(excinfo.value)


@pytest.mark.parametrize("model_type", ["gemma3", "qwen3_5", "qwen3"])
def test_supported_and_text_only_families_pass_the_gate(model_type):
    with mock.patch.object(AutoConfig, "from_pretrained", return_value=CONFIG_MAPPING[model_type]()):
        model_setup.reject_multimodal_model_without_seq_cls_head(_model_config(f"stub/{model_type}"))


def test_both_spellings_of_a_declared_vision_tower_classify_the_same():
    """The roster reads config classes (``sub_configs``), the run-time probe reads instances (``self.vision_config``).
    One predicate must see both, or the gate refuses a family the loader routes to the VLM path, or passes one whose
    images it drops."""

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
    """mistral4's ITT entry resolves to ``Mistral4ForCausalLM``, an upstream quirk.
    Routed to AutoProcessor/VLM setup, it fails outright on a family that ships no processor."""
    mistral4 = CONFIG_MAPPING["mistral4"]
    assert MODEL_FOR_IMAGE_TEXT_TO_TEXT_MAPPING_NAMES["mistral4"].endswith("ForCausalLM"), "premise"

    assert config_declares_multimodality(mistral4) is False
    assert config_declares_multimodality(mistral4()) is False
    assert is_vlm_model("stub/mistral4-base", config=mistral4()) is False


def test_the_seq_cls_roster_holds_the_multimodal_families_and_no_text_one():
    """The refusal above offers this roster as the way forward, so a change here changes a user-facing gate."""
    roster = set(multimodal_sequence_classification_model_types())

    assert {"gemma4", "qwen3_5", "qwen3_5_moe"} <= roster
    assert roster.isdisjoint({"mistral4", "qwen3", "qwen3_moe", "gpt_oss"})


def test_pipeline_parallelism_refuses_a_multimodal_reward_run_fed_images():
    """The stage split keeps only the text backbone; a text-only run of the same checkpoint is admitted."""
    trainer = _bare_trainer(True)
    trainer.parallelism_config = SimpleNamespace(is_pp_mode=True)
    trainer.save_sharded_ep = False
    training_args = SimpleNamespace(
        max_length=512,
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs=None,
        eval_strategy="no",
        eval_on_start=False,
        activation_offloading=False,
        torch_compile=False,
    )
    kwargs = {
        "model": config_only_module(CONFIG_MAPPING["gemma3"]()),
        "train_dataset": Dataset.from_dict({"chosen": ["a"], "rejected": ["b"], "images": [[]]}),
    }

    with pytest.raises(ValueError, match="Vision-language training is not supported under pipeline"):
        trainer._split_pipeline_stage(kwargs, training_args)


def test_reward_trainer_declares_no_context_parallel_support():
    """CP would split the sequence the score head pools; the declaration is the gate."""
    assert DistributedRewardTrainer.SUPPORTS_CP is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
