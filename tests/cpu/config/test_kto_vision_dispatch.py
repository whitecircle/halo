#!/usr/bin/env python
"""KTO's vision dispatch is column-keyed upstream, so embedded image parts must be refused here.

TRL's ``KTOTrainer`` decides vision-vs-text from the first row's COLUMNS (``image``/``images``) and
templates the raw ``prompt``/``completion`` columns itself. A dataset that embeds
``{"type": "image"}`` parts in its messages while shipping no image column therefore takes the TEXT
path, where the chat template expands every part into vision placeholder tokens with no pixels
behind them — and KTO renders nothing of its own, so the per-row ``reject_image_content`` backstop
the other preference scripts inherit never runs. ``kto.py`` owns that guard, plus the
``images_field`` rename onto the spelling TRL matches by name.

Driven through ``main()`` rather than a restatement of the branch: the guard is only correct if the
script reaches it with the dataset in hand.

Run: pytest tests/cpu/config/test_kto_vision_dispatch.py
"""

import sys
import types
from unittest import mock

import pytest
from datasets import Dataset, DatasetDict
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from tests.common.utils import load_script_module

# Model ids the name heuristic reads the same way offline as against the hub, paired with a config
# that decides the same — so the checkpoint verdict never depends on a network round-trip.
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
    """No network from this tier: the modality probe resolves from the name heuristic and the
    already-loaded config, never a hub round-trip."""
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
) -> None:
    """Run ``preference/kto.py:main()`` up to the trainer hand-off.

    Everything before the dataset is stubbed (model load, reference load, tokenizer resolution); the
    dispatch, the guard and the column renames are the real code under test. ``log_dataset_examples``
    is the last call that sees the splits exactly as the trainer will, so it records them into
    ``logged`` and stops the run.
    """
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
    )

    def capture_and_stop(datasets, *_args, **_kwargs):
        if logged is not None:
            logged.update(datasets)
        raise _TrainerHandoffReached

    patches = [
        mock.patch.object(module, "init_training_script", return_value=runtime),
        mock.patch.object(
            module,
            "load_model_for_training",
            return_value=(model, types.SimpleNamespace(tokenizer=None), tokenizer, vlm_checkpoint),
        ),
        mock.patch.object(module, "load_reference_model_for_preference", return_value=None),
        mock.patch.object(module, "apply_max_length", side_effect=lambda cfg, args, model, tok: tok),
        mock.patch.object(module, "install_resolved_tokenizer", side_effect=lambda pc, tok, is_vlm: pc),
        mock.patch.object(module, "setup_peft_model", return_value=None),
        mock.patch.object(module, "log_model_info"),
        mock.patch.object(module, "load_script_datasets", return_value=(dataset, False)),
        mock.patch.object(module, "log_script_dataset_examples", side_effect=capture_and_stop),
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


# --- the guard ------------------------------------------------------------------------------------


def test_embedded_image_parts_without_an_image_column_raise(tmp_path):
    """The silent path this closes: TRL sees no image column, templates the row as text, and trains
    the expanded vision placeholders as ordinary tokens. The refusal has to name both ways out —
    the shape is not fixable from the message alone otherwise."""
    with pytest.raises(ValueError, match="embed image content parts") as raised:
        _run_kto(tmp_path, _dataset(prompt=_IMAGE_PROMPT))
    message = str(raised.value)
    assert "'images' column" in message
    assert "drop the image parts" in message


def test_unfilled_image_placeholders_without_an_image_column_raise(tmp_path):
    """A part carrying no payload renders to the same pixel-less placeholder tokens, and its Arrow
    schema declares no image at all — only the row probe sees it."""
    with pytest.raises(ValueError, match="embed image content parts"):
        _run_kto(tmp_path, _dataset(prompt=_PLACEHOLDER_PROMPT))


def test_embedded_image_parts_in_the_configured_completion_column_raise(tmp_path):
    """The completion column is configurable, so the probe must follow ``completion_field`` rather
    than the default spelling."""
    dataset = _dataset(completion=_IMAGE_PROMPT, completion_column="response")
    with pytest.raises(ValueError, match="embed image content parts"):
        _run_kto(tmp_path, dataset, yaml_body="completion_field: response\n")


def test_embedded_image_parts_are_refused_on_a_text_checkpoint_too(tmp_path):
    """The shape is a dataset error, not a modality one: a text checkpoint renders the same
    placeholder tokens, and TRL's own vision guards never fire without an image column."""
    with pytest.raises(ValueError, match="embed image content parts"):
        _run_kto(tmp_path, _dataset(prompt=_IMAGE_PROMPT), vlm_checkpoint=False)


# --- what must still run --------------------------------------------------------------------------


def test_text_rows_on_a_multimodal_checkpoint_reach_the_trainer(tmp_path):
    """Anti-vacuity: KTO on a natively-multimodal checkpoint with text data is a supported run and
    must reach the trainer untouched."""
    with pytest.raises(_TrainerHandoffReached):
        _run_kto(tmp_path, _dataset())


def test_image_column_dataset_reaches_the_trainer(tmp_path):
    """With the column TRL probes for, the vision path is TRL's to take — embedded placeholders and
    all (its collator fills them from the column)."""
    with pytest.raises(_TrainerHandoffReached):
        _run_kto(tmp_path, _dataset(prompt=_PLACEHOLDER_PROMPT, extra_columns={"images": [[]]}))


# --- images_field ---------------------------------------------------------------------------------


def test_images_field_renames_the_column_to_the_spelling_trl_reads(tmp_path):
    """TRL matches the image column by name — ``"image" in sample or "images" in sample`` for the
    vision probe, ``example["images"]`` in the collator — so a hub column named anything else must
    arrive renamed, or the run trains as text with its images dropped."""
    logged: dict = {}
    dataset = _dataset(prompt=_PLACEHOLDER_PROMPT, extra_columns={"pictures": [[]]})
    with pytest.raises(_TrainerHandoffReached):
        _run_kto(tmp_path, dataset, yaml_body="images_field: pictures\n", logged=logged)
    train = logged["train"]
    assert "pictures" not in train.column_names
    sample = next(iter(train))
    assert "images" in sample or "image" in sample, "TRL's vision probe would read this row as text"


def test_images_field_already_named_image_is_left_alone(tmp_path):
    """``image`` is TRL's own single-image spelling; renaming it to the list spelling would hand the
    collator a bare image where it expects a list."""
    logged: dict = {}
    dataset = _dataset(prompt=_PLACEHOLDER_PROMPT, extra_columns={"image": ["b64"]})
    with pytest.raises(_TrainerHandoffReached):
        _run_kto(tmp_path, dataset, yaml_body="images_field: image\n", logged=logged)
    assert "image" in logged["train"].column_names


def test_images_field_naming_a_missing_column_raises(tmp_path):
    """A mistyped column would otherwise leave the run on the text path with no images at all."""
    with pytest.raises(ValueError, match="images_field='pictures' names a column the dataset"):
        _run_kto(tmp_path, _dataset(), yaml_body="images_field: pictures\n")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
