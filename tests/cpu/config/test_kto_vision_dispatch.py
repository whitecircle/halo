#!/usr/bin/env python
"""KTO's vision dispatch is column-keyed upstream, so embedded image parts are refused in ``kto.py``. With no image
column TRL takes the text path, where the template expands parts into placeholder tokens with no pixels, and KTO
renders nothing of its own for ``reject_image_content`` to catch."""

import sys
import types
from unittest import mock

import pytest
from datasets import Dataset, DatasetDict
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from halo.entrypoints.model_setup import PreferenceModels
from tests.common.repo import load_script_module

# Ids the name heuristic reads the same offline as against the hub, paired with a config that decides the same.
_VLM_MODEL_ID = "stub/qwen3.5-9b"
_TEXT_MODEL_ID = "stub/qwen3-8b"

_TEXT_PROMPT = [{"role": "user", "content": [{"type": "text", "text": "what is this?"}]}]
_IMAGE_PROMPT = [{"role": "user", "content": [{"type": "image", "image": "b64"}, {"type": "text", "text": "this?"}]}]
_PLACEHOLDER_PROMPT = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "this?"}]}]
_COMPLETION = [{"role": "assistant", "content": [{"type": "text", "text": "a cat"}]}]


class _TrainerHandoffReached(Exception):
    """Raised by the stubbed example logger: main() runs only as far as the trainer hand-off."""


@pytest.fixture(autouse=True)
def _hub_offline(monkeypatch):
    """The modality probe resolves from the name heuristic and the already-loaded config, never the hub."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


def _script_module():
    return load_script_module("scripts/training/preference/kto.py", "halo_test_kto_dispatch")


def _dataset(
    prompt: list | None = None,
    completion: list | None = None,
    completion_column: str = "completion",
    extra_columns: dict | None = None,
) -> DatasetDict:
    data = {
        "prompt": [prompt if prompt is not None else _TEXT_PROMPT],
        completion_column: [completion if completion is not None else _COMPLETION],
        "label": [True],
        **(extra_columns or {}),
    }
    return DatasetDict({"train": Dataset.from_dict(data), "test": Dataset.from_dict(data)})


def _run_kto(
    tmp_path,
    dataset: DatasetDict,
    *,
    yaml_body: str = "",
    vlm_checkpoint: bool = True,
    logged: dict | None = None,
    model_load: Exception | None = None,
) -> None:
    """Run ``preference/kto.py:main()`` up to the trainer hand-off. ``log_dataset_examples`` is the last call that
    sees the splits as the trainer will, so it records them into ``logged`` and stops the run. ``model_load`` is
    raised by the model load in its place."""
    module = _script_module()
    model_id = _VLM_MODEL_ID if vlm_checkpoint else _TEXT_MODEL_ID
    config = tmp_path / "config.yaml"
    config.write_text(
        f"model_name_or_path: {model_id}\n"
        f"dataset:\n- dummy/dataset\n"
        f"output_dir: {tmp_path / 'out'}\n"
        f"bf16: false\nuse_cpu: true\nmax_length: 512\n{yaml_body}"
    )

    model_config = CONFIG_MAPPING["gemma3"]() if vlm_checkpoint else CONFIG_MAPPING["qwen3"]()
    model = types.SimpleNamespace(config=model_config)
    tokenizer = types.SimpleNamespace(padding_side="right")
    runtime = types.SimpleNamespace(
        parallelism_config=types.SimpleNamespace(cp_size=1, is_cp_mode=False, pp_size=1),
        model_source=model_id,
        mode_suffix="",
        local_rank=0,
        resume_checkpoint=None,
        policy_from_checkpoint=False,
    )

    def capture_and_stop(datasets, *_args, **_kwargs):
        if logged is not None:
            logged.update(datasets)
        raise _TrainerHandoffReached

    def load_models(context, *_args, **kwargs):
        assert context is runtime, "the policy load must read the checkpoint identity off the script context"
        return PreferenceModels(model, None, None, tokenizer, types.SimpleNamespace(tokenizer=None))

    patches = [
        mock.patch.object(module, "init_training_script", return_value=runtime),
        mock.patch.object(module, "load_preference_models", side_effect=model_load or load_models),
        mock.patch.object(module, "load_script_datasets", return_value=(dataset, False)),
        mock.patch.object(module, "log_script_dataset_examples", side_effect=capture_and_stop),
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


def test_embedded_image_parts_without_an_image_column_raise(tmp_path):
    """The refusal has to name both ways out, since the shape is not fixable from the message alone otherwise."""
    with pytest.raises(ValueError, match="embed image content parts") as raised:
        _run_kto(tmp_path, _dataset(prompt=_IMAGE_PROMPT))
    message = str(raised.value)
    assert "'images' column" in message
    assert "drop the image parts" in message


def test_unfilled_image_placeholders_without_an_image_column_raise(tmp_path):
    """A part carrying no payload renders to the same placeholder tokens and its Arrow schema declares no image, so
    only the row probe sees it."""
    with pytest.raises(ValueError, match="embed image content parts"):
        _run_kto(tmp_path, _dataset(prompt=_PLACEHOLDER_PROMPT))


def test_embedded_image_parts_in_the_configured_completion_column_raise(tmp_path):
    dataset = _dataset(completion=_IMAGE_PROMPT, completion_column="response")
    with pytest.raises(ValueError, match="embed image content parts"):
        _run_kto(tmp_path, dataset, yaml_body="completion_field: response\n")


def test_embedded_image_parts_are_refused_on_a_text_checkpoint_too(tmp_path):
    """A text checkpoint renders the same placeholder tokens, and TRL's own vision guards never fire without an
    image column."""
    with pytest.raises(ValueError, match="embed image content parts"):
        _run_kto(tmp_path, _dataset(prompt=_IMAGE_PROMPT), vlm_checkpoint=False)


def test_text_rows_on_a_multimodal_checkpoint_reach_the_trainer(tmp_path):
    with pytest.raises(_TrainerHandoffReached):
        _run_kto(tmp_path, _dataset())


def test_image_column_dataset_reaches_the_trainer(tmp_path):
    """The vision path is TRL's to take, embedded placeholders and all: its collator fills them from the column."""
    with pytest.raises(_TrainerHandoffReached):
        _run_kto(tmp_path, _dataset(prompt=_PLACEHOLDER_PROMPT, extra_columns={"images": [[]]}))


@pytest.mark.parametrize(
    ("column", "extra_columns"),
    [("prompt", {}), ("completion", {}), ("prompt", {"images": [[], []]})],
    ids=["text_path_prompt", "text_path_completion", "vision_path_prompt"],
)
@pytest.mark.parametrize(
    "part",
    [{"type": "video", "video": "a.mp4"}, {"type": "image_url", "image_url": {"url": "a.png"}}],
    ids=["video", "image_url"],
)
def test_a_media_part_on_any_row_is_refused_ahead_of_trls_render(tmp_path, column, extra_columns, part):
    """TRL templates these rows itself, into placeholders with no pixels. The first row carries none, so only a read of
    every row sees it."""
    role = "user" if column == "prompt" else "assistant"
    media = [{"role": role, "content": [part, {"type": "text", "text": "this?"}]}]
    text = _TEXT_PROMPT if column == "prompt" else _COMPLETION
    other_column, other = ("completion", _COMPLETION) if column == "prompt" else ("prompt", _TEXT_PROMPT)
    data = {column: [text, media], other_column: [other, other], "label": [True, False], **extra_columns}
    dataset = DatasetDict(
        {"train": Dataset.from_list([dict(zip(data, row, strict=True)) for row in zip(*data.values(), strict=True)])}
    )
    with pytest.raises(ValueError, match="no Halo collator feeds"):
        _run_kto(tmp_path, dataset)


def test_images_field_renames_the_column_to_the_spelling_trl_reads(tmp_path):
    """TRL matches the column by name, so anything else must arrive renamed or the run trains as text with its
    images dropped."""
    logged: dict = {}
    dataset = _dataset(prompt=_PLACEHOLDER_PROMPT, extra_columns={"pictures": [[]]})
    with pytest.raises(_TrainerHandoffReached):
        _run_kto(tmp_path, dataset, yaml_body="images_field: pictures\n", logged=logged)
    train = logged["train"]
    assert "pictures" not in train.column_names
    sample = next(iter(train))
    assert "images" in sample or "image" in sample, "TRL's vision probe would read this row as text"


def test_images_field_already_named_image_is_left_alone(tmp_path):
    """``image`` is TRL's single-image spelling; renaming it would hand the collator a bare image where it expects a
    list."""
    logged: dict = {}
    dataset = _dataset(prompt=_PLACEHOLDER_PROMPT, extra_columns={"image": ["b64"]})
    with pytest.raises(_TrainerHandoffReached):
        _run_kto(tmp_path, dataset, yaml_body="images_field: image\n", logged=logged)
    assert "image" in logged["train"].column_names


def test_images_field_naming_a_missing_column_raises(tmp_path):
    """A mistyped column would otherwise leave the run on the text path with no images at all."""
    with pytest.raises(ValueError, match="images_field='pictures' names a column the dataset"):
        _run_kto(tmp_path, _dataset(), yaml_body="images_field: pictures\n")


@pytest.mark.parametrize(
    ("knob", "message"),
    [("completion_field", "completion_field='response'"), ("label_field", "label_field='response'")],
)
def test_a_mistyped_render_column_is_refused_before_the_model_load(tmp_path, knob, message):
    """The typo would otherwise surface only after the policy and reference load."""
    with pytest.raises(ValueError, match=message):
        _run_kto(
            tmp_path, _dataset(), yaml_body=f"{knob}: response\n", model_load=AssertionError("the model was loaded")
        )


def test_an_empty_eval_split_is_still_the_eval_split(tmp_path):
    """A pre-sharded eval split can leave a rank no rows; read by truthiness, that rank alone would skip the
    evaluation its peers entered."""
    dataset = _dataset()
    dataset["test"] = dataset["test"].select([])
    logged: dict = {}
    with pytest.raises(_TrainerHandoffReached):
        _run_kto(tmp_path, dataset, vlm_checkpoint=False, logged=logged)
    assert logged["test"] is not None and len(logged["test"]) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
