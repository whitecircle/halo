#!/usr/bin/env python
"""Every model card Halo writes carries the ``halo`` Hugging Face Hub tag.

A model uploaded from a Halo output lists under ``halo`` on the Hub, the way TRL- and PEFT-written
cards list under ``trl`` and ``peft``. The tag rides two carriers: the loaded model's ``model_tags``
(which PEFT's own adapter card reads), and the ``README.md`` card the two export finalizers tag — the
config finalizer every full-model writer ends with, and the non-weight copy every tool that builds an
export from a source directory runs. The writers that reach neither tag their own output, and the
embedding script tags the card data sentence-transformers writes its card from. An existing card
(PEFT's, TRL's, the source model's) changes in its ``tags`` entry only; a fresh card holds the tag
alone, and a card whose metadata is not YAML fails naming the file to repair.

    python tests/cpu/checkpoint/test_hub_model_card_tags.py
"""

import json
import os
import re
import stat
from types import SimpleNamespace

import pytest
import torch
from accelerate import PartialState
from huggingface_hub import ModelCard
from huggingface_hub.repocard import metadata_load
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GptOssConfig, GptOssForCausalLM, PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
from trl import ModelConfig
from trl.trainer.utils import generate_model_card

import src.distributed.expert_parallel.layers.roster  # noqa: F401  registers the roster every config writer requires
from scripts.after_training.convert_to_bf16 import convert_to_bf16
from scripts.after_training.reset_sinks import reset_sinks
from scripts.training.embedding import build_sentence_transformer
from src.checkpoint import model_card
from src.checkpoint.config_export import finalize_exported_config, save_model_config
from src.checkpoint.format import copy_checkpoint_aux_files
from src.checkpoint.model_card import tag_model_card
from src.configs.embedding_config import EmbeddingConfig
from src.models.loading.model_preparation import finalize_run_model
from src.models.patches.gpt_oss_sinks import SinksPolicy

PartialState()  # the tools' loads log through accelerate's rank-aware logger

HALO_TAG = "halo"
CARD = "README.md"
_TINY_QWEN3 = {
    "vocab_size": 64,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "max_position_embeddings": 64,
    "tie_word_embeddings": False,
}
# A card's metadata beyond ``tags`` must survive verbatim: two model-index entries, one carrying
# fields ModelCardData's EvalResult has no slot for, and keys it would reorder.
_EVAL_CARD = """---
license: mit
model-index:
- name: model-a
  results:
  - task: {type: text-generation, name: Text Generation}
    dataset: {name: ARC, type: ai2_arc, config: ARC-Challenge, split: test}
    metrics:
    - {type: acc_norm, value: 0.5, name: normalized accuracy}
    source: {url: 'https://a', name: Leaderboard A}
  - task: {type: text-generation, name: Other Name}
    dataset: {name: ARC second name, type: ai2_arc, config: ARC-Challenge, split: test}
    metrics:
    - {type: acc, value: 0.4, name: accuracy, custom_key: 7}
    source: {url: 'https://b', name: Leaderboard B}
- name: model-b
  results:
  - task: {type: text-generation}
    dataset: {name: HS, type: hellaswag}
    metrics:
    - {type: acc, value: 0.9}
language:
- en
---
# body
"""
# A source model's own card: every field and the body must survive the export untouched.
_SOURCE_CARD = """---
library_name: transformers
license: apache-2.0
base_model: Qwen/Qwen3-0.6B
tags:
- text-generation
---

# Source model

Card body written by the model's authors.
"""


def _tiny_qwen3() -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    return Qwen3ForCausalLM(Qwen3Config(**_TINY_QWEN3))


def _tiny_tokenizer() -> PreTrainedTokenizerFast:
    """Built in-process, so the adapter conversion resolves a processing class without the network."""
    backend = Tokenizer(models.WordLevel({"<unk>": 0, "<eos>": 1, "hello": 2}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>", pad_token="<eos>")


def _card(directory) -> ModelCard:
    return ModelCard.load(directory / CARD)


def test_a_fresh_card_holds_the_tag_and_claims_no_library(tmp_path):
    tag_model_card(str(tmp_path))
    assert _card(tmp_path).data.to_dict() == {"tags": [HALO_TAG]}
    assert sorted(os.listdir(tmp_path)) == [CARD], "the staged card was left beside the real one"
    assert stat.S_IMODE((tmp_path / CARD).stat().st_mode) & 0o044 == 0o044, "the card is not readable by others"


def test_an_existing_card_keeps_its_metadata_tags_and_body(tmp_path):
    (tmp_path / CARD).write_text(_SOURCE_CARD)
    body = ModelCard(_SOURCE_CARD).text

    tag_model_card(str(tmp_path))
    card = _card(tmp_path)
    assert card.data.library_name == "transformers"
    assert card.data.license == "apache-2.0"
    assert card.data.base_model == "Qwen/Qwen3-0.6B"
    assert card.data.tags == ["text-generation", HALO_TAG]
    assert card.text == body

    tagged = (tmp_path / CARD).read_bytes()
    tag_model_card(str(tmp_path))
    assert (tmp_path / CARD).read_bytes() == tagged, "a card that already carries the tag is rewritten"


def test_metadata_beyond_the_tags_round_trips_verbatim(tmp_path):
    (tmp_path / CARD).write_text(_EVAL_CARD)
    before = metadata_load(tmp_path / CARD)

    tag_model_card(str(tmp_path))
    after = metadata_load(tmp_path / CARD)
    assert after.pop("tags") == [HALO_TAG]
    assert after == before
    assert list(after) == list(before)


def test_a_scalar_tags_entry_becomes_a_list_beside_the_halo_tag(tmp_path):
    (tmp_path / CARD).write_text("---\ntags: foo\n---\nbody\n")
    tag_model_card(str(tmp_path))
    assert metadata_load(tmp_path / CARD)["tags"] == ["foo", HALO_TAG]


def test_crlf_line_endings_and_the_body_are_kept(tmp_path):
    (tmp_path / CARD).write_bytes(b"---\r\nlicense: mit\r\n---\r\nbody line\r\nsecond line\r\n")
    tag_model_card(str(tmp_path))
    assert (tmp_path / CARD).read_bytes() == (
        b"---\r\nlicense: mit\r\ntags:\r\n- halo\r\n---\r\nbody line\r\nsecond line\r\n"
    )


def test_a_card_without_metadata_keeps_its_body_under_a_new_block(tmp_path):
    (tmp_path / CARD).write_text("# Title\n\nSome body\n")
    tag_model_card(str(tmp_path))
    assert (tmp_path / CARD).read_text() == "---\ntags:\n- halo\n---\n# Title\n\nSome body\n"


def test_a_symlinked_card_is_replaced_not_written_through(tmp_path):
    """A Hub-cache snapshot links README.md into a blob shared by every snapshot of that repo."""
    blob = tmp_path / "blob"
    blob.write_text(_SOURCE_CARD)
    (tmp_path / "export").mkdir()
    (tmp_path / "export" / CARD).symlink_to(blob)

    tag_model_card(str(tmp_path / "export"))
    assert blob.read_text() == _SOURCE_CARD
    assert not (tmp_path / "export" / CARD).is_symlink()
    assert metadata_load(tmp_path / "export" / CARD)["tags"] == ["text-generation", HALO_TAG]


def test_a_failed_write_leaves_the_card_and_no_staged_copy(tmp_path, monkeypatch):
    (tmp_path / CARD).write_text(_SOURCE_CARD)

    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(model_card, "metadata_save", fail)
    with pytest.raises(OSError, match="disk full"):
        tag_model_card(str(tmp_path))
    assert sorted(os.listdir(tmp_path)) == [CARD]
    assert (tmp_path / CARD).read_text() == _SOURCE_CARD


@pytest.mark.parametrize("metadata", ["tags: [a, b\n", "- a\n- b\n"], ids=["bad-yaml", "not-a-mapping"])
def test_a_card_whose_metadata_is_not_a_yaml_mapping_names_the_file(tmp_path, metadata):
    (tmp_path / CARD).write_text(f"---\n{metadata}---\nbody\n")
    with pytest.raises(ValueError, match=rf"(?s){re.escape(str(tmp_path / CARD))}.*Repair or remove it"):
        tag_model_card(str(tmp_path))


def test_the_parallel_writers_config_step_tags_the_checkpoint_card(tmp_path):
    """``save_model_config`` is the config write of every gathered/EP/PP saver: none writes a card itself."""
    save_model_config(_tiny_qwen3(), str(tmp_path))
    assert _card(tmp_path).data.tags == [HALO_TAG]


def test_the_config_finalizer_adds_the_tag_to_a_trainer_written_card(tmp_path):
    """A library card already in the directory — TRL's here — keeps its own tags beside Halo's."""
    model = _tiny_qwen3()
    model.config.save_pretrained(tmp_path)
    generate_model_card(
        base_model="Qwen/Qwen3-0.6B",
        model_name="run",
        hub_model_id=None,
        dataset_name=None,
        tags=["trl", "sft"],
        wandb_url=None,
        trackio_url=None,
        trainer_name="SFT",
    ).save(tmp_path / CARD)
    trl_body = _card(tmp_path).text

    finalize_exported_config(model.config, str(tmp_path), source=None)
    card = _card(tmp_path)
    assert card.data.library_name == "transformers"
    assert card.data.tags == ["generated_from_trainer", "trl", "sft", HALO_TAG]
    assert card.text == trl_body


def test_a_peft_adapter_of_a_loaded_model_carries_the_tag(tmp_path):
    """PEFT's card replaces its tags with the base model's ``model_tags``: the load stamps them there."""
    model = _tiny_qwen3()
    finalize_run_model(model, model.config, sinks_policy=SinksPolicy.NEUTRALIZED, attn_implementation="eager")
    get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj"])).save_pretrained(tmp_path)

    card = _card(tmp_path)
    assert card.data.library_name == "peft"
    assert HALO_TAG in card.data.tags
    assert "lora" in card.data.tags


def test_an_export_carries_the_source_card_tagged_and_leaves_the_source_alone(tmp_path):
    source, output = tmp_path / "source", tmp_path / "export"
    source.mkdir()
    output.mkdir()
    (source / CARD).write_text(_SOURCE_CARD)
    (source / "config.json").write_text(json.dumps({"model_type": "qwen3"}))

    copy_checkpoint_aux_files(str(source), str(output))
    assert (source / CARD).read_text() == _SOURCE_CARD
    card = _card(output)
    assert card.data.tags == ["text-generation", HALO_TAG]
    assert card.data.license == "apache-2.0"
    assert card.text == ModelCard(_SOURCE_CARD).text


def test_an_export_names_the_source_card_it_cannot_tag(tmp_path):
    """The weights are already written by then; the re-run recopies the source, so that is the file to fix."""
    source, output = tmp_path / "source", tmp_path / "export"
    source.mkdir()
    output.mkdir()
    (source / CARD).write_text("---\ntags: [a, b\n---\nbody\n")

    with pytest.raises(ValueError, match=rf"Repair or remove {re.escape(str(source / CARD))}, then re-run"):
        copy_checkpoint_aux_files(str(source), str(output))


def test_an_export_of_a_cardless_source_gets_a_tagged_card(tmp_path):
    source, output = tmp_path / "source", tmp_path / "export"
    source.mkdir()
    output.mkdir()
    (source / "config.json").write_text(json.dumps({"model_type": "qwen3"}))

    copy_checkpoint_aux_files(str(source), str(output))
    assert not (source / CARD).exists()
    assert _card(output).data.tags == [HALO_TAG]


def test_the_single_file_sinks_reset_tags_its_output(tmp_path):
    """That branch copies the source tree and rewrites one file, reaching neither finalizer."""
    source, output = tmp_path / "source", tmp_path / "reset"
    torch.manual_seed(0)
    config = GptOssConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=32,
        num_hidden_layers=2,
        num_local_experts=2,
        num_experts_per_tok=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        sliding_window=32,
        tie_word_embeddings=False,
    )
    GptOssForCausalLM(config).to(torch.bfloat16).save_pretrained(source)
    assert (source / "model.safetensors").is_file(), "premise: the single-file branch is the one exercised"

    assert reset_sinks(str(source), output_dir=str(output)) > 0
    assert _card(output).data.tags == [HALO_TAG]


def test_the_sinks_reset_passthrough_tags_its_output(tmp_path):
    """A checkpoint with no sinks is copied through unchanged, and that copy is still an export."""
    source, output = tmp_path / "source", tmp_path / "copy"
    _tiny_qwen3().save_pretrained(source)
    assert (source / "model.safetensors").is_file(), "premise: the single-file branch is the one exercised"

    assert reset_sinks(str(source), output_dir=str(output)) == 0
    assert _card(output).data.tags == [HALO_TAG]


def test_the_unmerged_adapter_conversion_tags_its_output(tmp_path):
    """The tool loads its base untagged, so PEFT's card for the converted adapter has no Halo tag of its own."""
    base, adapter, output = tmp_path / "base", tmp_path / "adapter", tmp_path / "converted"
    _tiny_qwen3().save_pretrained(base)
    _tiny_tokenizer().save_pretrained(base)
    peft_model = get_peft_model(
        Qwen3ForCausalLM.from_pretrained(base), LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj"])
    )
    peft_model.peft_config["default"].base_model_name_or_path = str(base)
    peft_model.save_pretrained(adapter)

    convert_to_bf16(str(adapter), str(output), "causal_lm", is_peft=True)
    card = _card(output)
    assert card.data.library_name == "peft"
    assert HALO_TAG in card.data.tags


def test_the_embedding_pipeline_card_carries_the_tag(tmp_path):
    """sentence-transformers writes its card from ``model_card_data`` and never reads ``model_tags``."""
    base = tmp_path / "base"
    _tiny_qwen3().save_pretrained(base)
    _tiny_tokenizer().save_pretrained(base)
    runtime = SimpleNamespace(
        parallelism_config=SimpleNamespace(is_ep_mode=False, is_tp_mode=False), model_source=str(base)
    )
    embedding_config = EmbeddingConfig(
        output_dir=str(tmp_path / "run"), bf16=False, pooling_mode="mean", normalize_embeddings=False, max_length=32
    )
    st_model = build_sentence_transformer(
        runtime,
        embedding_config,
        ModelConfig(model_name_or_path=str(base)),
        SimpleNamespace(reset_sinks=True, train_sinks=False),
    )

    st_model.save(str(tmp_path / "embedding"))
    card = _card(tmp_path / "embedding")
    assert card.data.library_name == "sentence-transformers"
    assert HALO_TAG in card.data.tags
    assert "sentence-transformers" in card.data.tags


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
