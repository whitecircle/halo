#!/usr/bin/env python
"""Online GRPO, environmental GRPO and reward modeling read their dataset before the model load, as the other scripts
do, so a wrong dataset or column fails before a multi-hundred-GB load."""

import contextlib
import sys
import types
from unittest import mock

import pytest
from datasets import Dataset, DatasetDict

from tests.common.repo import load_script_module

_MODEL_ID = "stub/qwen3-4b"
_TURNS = [{"role": "user", "content": "hi"}]
_ONLINE = ("scripts/training/grpo/online.py", "max_completion_length: 64\n")
_ENVIRONMENTAL = ("scripts/training/grpo/environmental.py", "environment_type: react_math\n")
_REWARD = ("scripts/training/reward/reward_model.py", "")


def _rows(**columns) -> DatasetDict:
    split = Dataset.from_dict({"prompt": [_TURNS], "answer": ["1"], **columns})
    return DatasetDict({"train": split, "test": split})


def _run_with_the_model_load_failing(tmp_path, script: tuple[str, str], config_body: str, rows: DatasetDict) -> None:
    path, script_body = script
    module = load_script_module(path, "halo_test_dataset_refusal_" + path.rpartition("/")[2].removesuffix(".py"))
    config = tmp_path / "config.yaml"
    config.write_text(
        f"model_name_or_path: {_MODEL_ID}\ndataset:\n- dummy/dataset\noutput_dir: {tmp_path / 'out'}\n"
        f"bf16: false\nuse_cpu: true\n{script_body}{config_body}"
    )
    runtime = types.SimpleNamespace(parallelism_config=types.SimpleNamespace(), model_source=_MODEL_ID)
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.object(module, "init_training_script", return_value=runtime))
        stack.enter_context(mock.patch.object(module, "load_script_datasets", return_value=(rows, False)))
        stack.enter_context(
            mock.patch.object(module, "load_model_and_tokenizer", side_effect=AssertionError("the model was loaded"))
        )
        # They read the stub checkpoint's config, which no hub holds.
        if hasattr(module, "reject_multimodal_model_without_seq_cls_head"):
            stack.enter_context(mock.patch.object(module, "reject_multimodal_model_without_seq_cls_head"))
        if hasattr(module, "probe_checkpoint"):
            stack.enter_context(mock.patch.object(module, "probe_checkpoint", return_value=(None, False)))
        stack.enter_context(mock.patch("halo.entrypoints.data_setup.probe_checkpoint", return_value=(None, False)))
        stack.enter_context(mock.patch("halo.entrypoints.script.install_log_tee"))
        stack.enter_context(mock.patch.object(sys, "argv", ["prog", str(config)]))
        module.main()


@pytest.mark.parametrize(
    ("script", "config_body", "columns", "message"),
    [
        (_ONLINE, "answer_field: solution\n", {}, "answer_field='solution'"),
        (_ONLINE, "", {"images": [[]]}, r"RLVR Online GRPO is text-only.*\['images'\]"),
        (_ENVIRONMENTAL, "answer_field: solution\n", {}, "answer_field='solution'"),
        (_ENVIRONMENTAL, "context_fields:\n- hints\n", {}, "context_fields='hints'"),
        (_ENVIRONMENTAL, "", {"images": [[]]}, r"Environmental GRPO is text-only.*\['images'\]"),
        (_REWARD, "images_field: pictures\n", {}, "images_field='pictures'"),
        (_REWARD, "tools_field: tool_list\n", {}, "tools_field='tool_list'"),
        (_REWARD, "", {"images": [[]]}, "is a text checkpoint"),
        (_REWARD, "images_field: images\n", {"images": [[]]}, "is a text checkpoint"),
    ],
    ids=[
        "online-answer_field",
        "online-image_column",
        "environmental-answer_field",
        "environmental-context_fields",
        "environmental-image_column",
        "reward-images_field",
        "reward-tools_field",
        "reward-image_column_on_a_text_checkpoint",
        "reward-images_field_on_a_text_checkpoint",
    ],
)
def test_a_wrong_column_is_refused_before_the_model_load(tmp_path, script, config_body, columns, message):
    with pytest.raises(ValueError, match=message):
        _run_with_the_model_load_failing(tmp_path, script, config_body, _rows(**columns))


@pytest.mark.parametrize("script", [_ONLINE, _ENVIRONMENTAL, _REWARD], ids=["online", "environmental", "reward"])
def test_a_clean_dataset_reaches_the_model_load(tmp_path, script):
    with pytest.raises(AssertionError, match="the model was loaded"):
        _run_with_the_model_load_failing(tmp_path, script, "", _rows())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
