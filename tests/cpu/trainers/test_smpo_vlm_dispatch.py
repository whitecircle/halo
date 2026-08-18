#!/usr/bin/env python
"""SMPO text-vs-VLM dispatch: the DATA path follows the RUN, not the checkpoint.

A text preference run on a natively-multimodal checkpoint (Gemma 4, Qwen3.5/3.6) is an ordinary
text run — the script decides that with ``is_vlm_run`` and passes the verdict as ``is_vlm``, while
``processing_class`` stays the processor (every saved checkpoint owes a ``processor_config.json``).
That split is what these tests pin:

* the TEXT branch must map with the resolved TOKENIZER — ``tokenize_preference_row`` calls
  ``processing_class(text, add_special_tokens=False)`` (a processor reads the second slot as images)
  and reads ``bos_token_id``/``eos_token_id``, which a ProcessorMixin does not carry;
* the padding_free / CP / PP gates follow the run verdict — untripped for a text run, unchanged for
  a VLM run;
* ``is_vlm=True`` without a processor is refused at construction, not at the first collated batch;
* a plain text run's dataset-map cache key is untouched (a moved fingerprint re-tokenizes every
  existing run's data).
"""

from types import SimpleNamespace
from unittest import mock

import pytest
from datasets import Dataset
from transformers import ProcessorMixin

from src.data.pipeline.processing import _get_kwargs_fingerprint
from src.trainers.preference.smpo import (
    SmoothMarginPOTrainer,
    tokenize_preference_row,
    tokenize_vlm_preference_row,
)
from tests.cpu.trainers.test_smpo_vlm import EOS_ID, StubProcessor, StubTokenizer, make_features, make_trainer

# Already chat-templated strings — the shape the script's text path (prepare_preference_datasets)
# hands the trainer.
TEXT_ROW = {
    "prompt": "[user] Which color [end]\n",
    "chosen": "[assistant] Red [end]\n",
    "rejected": "[assistant] Blue [end]\n",
}


class ProcessorStub(StubProcessor, ProcessorMixin):
    """The VLM stub processor, also satisfying the ``isinstance(..., ProcessorMixin)`` contract the
    trainer's guard checks (a real run always hands it a transformers processor)."""


@pytest.fixture(autouse=True)
def _accelerate_state():
    from accelerate import PartialState

    PartialState()  # the accelerate logger inside _prepare_dataset requires an initialized state


def _capture_map(trainer, dataset, processing_class):
    """``_prepare_dataset`` with the dataset map stubbed out; returns the map fn and its fn_kwargs."""
    captured = {}

    def _fake_map(mapped_dataset, map_fn, **kwargs):
        captured["fn"] = map_fn
        captured["fn_kwargs"] = kwargs["fn_kwargs"]
        # A column the VLM branch's cast_column can re-type; the text branch passes it through.
        return Dataset.from_dict({"images": [[]]})

    with mock.patch("src.trainers.preference.smpo.coordinated_map", _fake_map):
        trainer._prepare_dataset(dataset, processing_class, "train")
    return captured


def test_text_run_on_multimodal_checkpoint_maps_with_the_tokenizer():
    """The row fn gets the processor's inner tokenizer, not the processor."""
    processor = ProcessorStub(StubTokenizer())
    trainer = make_trainer(is_vlm=False)

    captured = _capture_map(trainer, Dataset.from_dict({k: [v] for k, v in TEXT_ROW.items()}), processor)

    assert captured["fn"] is tokenize_preference_row
    assert captured["fn_kwargs"]["processing_class"] is processor.tokenizer


def test_vlm_run_maps_with_the_processor():
    """The VLM branch is unchanged: raw rows, processor, VLM row fn."""
    processor = ProcessorStub(StubTokenizer())
    trainer = make_trainer(is_vlm=True)
    features = make_features()

    captured = _capture_map(trainer, Dataset.from_list([{**features, "images": []}]), processor)

    assert captured["fn"] is tokenize_vlm_preference_row
    assert captured["fn_kwargs"]["processing_class"] is processor


def test_text_run_on_multimodal_checkpoint_tokenizes_end_to_end():
    """The real map produces text-path token ids — the processor would raise (or mis-tokenize)."""
    tokenizer = StubTokenizer()
    processor = ProcessorStub(tokenizer)
    trainer = make_trainer(is_vlm=False)

    prepared = trainer._prepare_dataset(Dataset.from_dict({k: [v] for k, v in TEXT_ROW.items()}), processor, "train")

    row = prepared[0]
    assert set(prepared.column_names) == {"prompt_input_ids", "chosen_input_ids", "rejected_input_ids"}
    assert row["prompt_input_ids"] == tokenizer._encode(TEXT_ROW["prompt"])
    assert row["chosen_input_ids"] == tokenizer._encode(TEXT_ROW["chosen"]) + [EOS_ID]
    assert row["rejected_input_ids"] == tokenizer._encode(TEXT_ROW["rejected"]) + [EOS_ID]


@pytest.mark.parametrize(
    ("gate", "attrs", "parallelism_config", "message"),
    [
        ("_validate_vlm_mode", {"padding_free": True}, None, "padding_free"),
        ("_validate_vlm_mode", {}, SimpleNamespace(cp_size=2), "Context Parallelism"),
        ("_reject_pp_explicit_options", {}, SimpleNamespace(is_pp_mode=True), "pipeline parallelism"),
    ],
)
def test_vlm_gates_follow_the_run_verdict(gate, attrs, parallelism_config, message):
    """A VLM run raises exactly as before; the same options are legal for a text run.

    The text half is the point of the run verdict: gating on the checkpoint alone leaves these three
    unreachable for text preference data on a multimodal checkpoint, which is what CP/padding_free/PP
    recipes need.
    """

    def _run(is_vlm):
        trainer = make_trainer(is_vlm=is_vlm, **attrs)
        args = (parallelism_config,) if gate == "_validate_vlm_mode" else (parallelism_config, None)
        getattr(trainer, gate)(*args)

    with pytest.raises(ValueError, match=message):
        _run(is_vlm=True)
    _run(is_vlm=False)


def test_is_vlm_true_requires_a_processor():
    tokenizer = StubTokenizer()
    processor = ProcessorStub(tokenizer)

    with pytest.raises(ValueError, match="ProcessorMixin"):
        SmoothMarginPOTrainer._resolve_vlm_mode(tokenizer, True)

    assert SmoothMarginPOTrainer._resolve_vlm_mode(processor, True) is True
    assert SmoothMarginPOTrainer._resolve_vlm_mode(processor, False) is False
    # Omitted → the pre-run-verdict derivation, the contract for direct trainer users.
    assert SmoothMarginPOTrainer._resolve_vlm_mode(processor, None) is True
    assert SmoothMarginPOTrainer._resolve_vlm_mode(tokenizer, None) is False


def test_text_checkpoint_cache_key_unchanged():
    """A plain text run must key the same map cache as before — no re-tokenization of existing runs.

    ``processing_class`` is the tokenizer there, and it must reach ``fn_kwargs`` as that same object:
    the fingerprint folds the object's type and identity fields, so any wrapper/copy would move the
    key. The processor comparison shows the fingerprint really does move when the object changes.
    """
    tokenizer = StubTokenizer()
    trainer = make_trainer(is_vlm=False)

    captured = _capture_map(trainer, Dataset.from_dict({k: [v] for k, v in TEXT_ROW.items()}), tokenizer)

    assert captured["fn_kwargs"]["processing_class"] is tokenizer
    text_fingerprint = _get_kwargs_fingerprint(captured["fn_kwargs"])
    processor_fingerprint = _get_kwargs_fingerprint(
        {**captured["fn_kwargs"], "processing_class": ProcessorStub(tokenizer)}
    )
    assert text_fingerprint != processor_fingerprint


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
