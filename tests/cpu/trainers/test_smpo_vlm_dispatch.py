#!/usr/bin/env python
"""SMPO text-vs-VLM dispatch: the data path follows the run, not the checkpoint.
A text run on a multimodal checkpoint keeps the processor to save its config, but maps with the resolved tokenizer."""

from functools import partial
from types import SimpleNamespace
from unittest import mock

import pytest
from datasets import Dataset
from transformers import ProcessorMixin

from halo.data.pipeline.map_cache_key import kwargs_fingerprint
from halo.data.pipeline.preferences import tokenize_preference_row, tokenize_vlm_preference_row
from halo.trainers.preference.preference_rules import reject_smpo_pp_options, reject_smpo_vlm_options
from halo.trainers.preference.smpo import resolve_vlm_mode
from tests.cpu.trainers.test_smpo_vlm import EOS_ID, StubProcessor, StubTokenizer, make_features, make_trainer

# Already chat-templated strings, as the script's text path (prepare_preference_datasets) hands the trainer.
TEXT_ROW = {
    "prompt": "[user] Which color [end]\n",
    "chosen": "[assistant] Red [end]\n",
    "rejected": "[assistant] Blue [end]\n",
}


class ProcessorStub(StubProcessor, ProcessorMixin):
    """Also satisfies the ``isinstance(..., ProcessorMixin)`` check the trainer's guard runs."""


def _capture_map(trainer, dataset, processing_class):
    """``_prepare_dataset`` with the dataset map stubbed out; returns the map fn and its fn_kwargs."""
    captured = {}

    def _fake_map(mapped_dataset, map_fn, **kwargs):
        captured["fn"] = map_fn
        captured["fn_kwargs"] = kwargs["fn_kwargs"]
        return mapped_dataset

    with mock.patch("halo.trainers.preference.smpo.coordinated_map", _fake_map):
        trainer._prepare_dataset(dataset, processing_class, trainer.args, "train")
    return captured


def test_text_run_on_multimodal_checkpoint_maps_with_the_tokenizer():
    processor = ProcessorStub(StubTokenizer())
    trainer = make_trainer(is_vlm=False)

    captured = _capture_map(trainer, Dataset.from_dict({k: [v] for k, v in TEXT_ROW.items()}), processor)

    assert captured["fn"] is tokenize_preference_row
    assert captured["fn_kwargs"]["processing_class"] is processor.tokenizer


def test_vlm_run_maps_with_the_processor():
    processor = ProcessorStub(StubTokenizer())
    trainer = make_trainer(is_vlm=True)
    features = make_features()

    captured = _capture_map(trainer, Dataset.from_list([{**features, "images": []}]), processor)

    assert captured["fn"] is tokenize_vlm_preference_row
    assert captured["fn_kwargs"]["processing_class"] is processor


def test_text_run_on_multimodal_checkpoint_tokenizes_end_to_end():
    """``tokenize_preference_row`` calls ``processing_class(text, add_special_tokens=False)``, and a processor reads
    the second slot as images."""
    tokenizer = StubTokenizer()
    processor = ProcessorStub(tokenizer)
    trainer = make_trainer(is_vlm=False)

    prepared = trainer._prepare_dataset(
        Dataset.from_dict({k: [v] for k, v in TEXT_ROW.items()}), processor, trainer.args, "train"
    )

    row = prepared[0]
    assert set(prepared.column_names) == {"prompt_input_ids", "chosen_input_ids", "rejected_input_ids"}
    assert row["prompt_input_ids"] == tokenizer._encode(TEXT_ROW["prompt"])
    assert row["chosen_input_ids"] == tokenizer._encode(TEXT_ROW["chosen"]) + [EOS_ID]
    assert row["rejected_input_ids"] == tokenizer._encode(TEXT_ROW["rejected"]) + [EOS_ID]


@pytest.mark.parametrize(
    ("gate", "attrs", "parallelism_config", "message"),
    [
        (reject_smpo_vlm_options, {"padding_free": True}, None, "padding_free"),
        (reject_smpo_vlm_options, {}, SimpleNamespace(cp_size=2), "Context Parallelism"),
        (
            partial(reject_smpo_pp_options, peft_config=None),
            {},
            SimpleNamespace(is_pp_mode=True),
            "pipeline parallelism",
        ),
    ],
)
def test_vlm_gates_follow_the_run_verdict(gate, attrs, parallelism_config, message):
    """Gating on the checkpoint alone would leave these three unreachable for text data on a multimodal checkpoint."""

    def _run(is_vlm):
        args = {"padding_free": False, "lower_clip_percentile": None, "upper_clip_percentile": None, **attrs}
        gate(SimpleNamespace(**args), parallelism_config, is_vlm=is_vlm)

    with pytest.raises(ValueError, match=message):
        _run(is_vlm=True)
    _run(is_vlm=False)


def test_is_vlm_true_requires_a_processor():
    tokenizer = StubTokenizer()
    processor = ProcessorStub(tokenizer)

    with pytest.raises(ValueError, match="ProcessorMixin"):
        resolve_vlm_mode(tokenizer, True)

    assert resolve_vlm_mode(processor, True) is True
    assert resolve_vlm_mode(processor, False) is False
    # Omitted, it derives from the processing class: the contract for direct trainer users.
    assert resolve_vlm_mode(processor, None) is True
    assert resolve_vlm_mode(tokenizer, None) is False


def test_text_checkpoint_cache_key_unchanged():
    """The fingerprint folds the object's type and identity fields, so a wrapper or copy would re-tokenize old data."""
    tokenizer = StubTokenizer()
    trainer = make_trainer(is_vlm=False)

    captured = _capture_map(trainer, Dataset.from_dict({k: [v] for k, v in TEXT_ROW.items()}), tokenizer)

    assert captured["fn_kwargs"]["processing_class"] is tokenizer
    text_fingerprint = kwargs_fingerprint(captured["fn_kwargs"])
    processor_fingerprint = kwargs_fingerprint({**captured["fn_kwargs"], "processing_class": ProcessorStub(tokenizer)})
    assert text_fingerprint != processor_fingerprint


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
