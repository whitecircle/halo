#!/usr/bin/env python
"""Text-only training methods must REFUSE an image-carrying dataset, not quietly drop it.

RLVR online GRPO, embedding training, environmental GRPO, offline GRPO and classification have no
vision path at all, and each prunes its dataset to the columns it names
(``remove_unused_columns``, the trainer's signature filter, an explicit ``remove_columns``,
env-GRPO's ``keep_columns``). A vision dataset handed to one of them therefore trains on the rows'
text alone — no warning, full cost, a checkpoint that never saw a pixel. The guard reads the shared
``VLM_IMAGE_COLUMNS`` spellings, so it refuses exactly what a VLM-capable method routes on.

Driven through each ``main()``: the guard is only worth anything where the script reaches it with
the dataset in hand, before the map that drops the column.

Run: pytest tests/cpu/config/test_text_only_image_column_refusal.py
"""

import sys
import types
from pathlib import Path
from unittest import mock

import pytest
from datasets import Dataset, DatasetDict

from src.data.vlm import VLM_IMAGE_COLUMNS
from tests.common.utils import load_script_module

_MODEL_ID = "stub/qwen3-4b"

_TEXT_TURNS = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]


class _PastTheGuard(Exception):
    """Raised by the first stub past the guard — the dataset was accepted."""


@pytest.fixture(autouse=True)
def _hub_offline(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


def _script_module(relative_path: str):
    name = f"halo_test_{Path(relative_path).stem}_image_refusal"
    return load_script_module(f"scripts/training/{relative_path}", name)


def _runtime():
    return types.SimpleNamespace(
        parallelism_config=types.SimpleNamespace(cp_size=1, is_cp_mode=False, pp_size=1, is_ep_mode=False),
        model_source=_MODEL_ID,
        mode_suffix="",
        local_rank=0,
        resume_checkpoint=None,
    )


def _dataset(columns: dict, image_column: str | None) -> DatasetDict:
    data = {**columns, **({image_column: [[]]} if image_column else {})}
    return DatasetDict({"train": Dataset.from_dict(data), "test": Dataset.from_dict(data)})


def _run(module, tmp_path, config_body: str, patches: list) -> None:
    """Run a training script's ``main()`` under ``patches``, from a minimal YAML."""
    config = tmp_path / "config.yaml"
    config.write_text(
        f"model_name_or_path: {_MODEL_ID}\n"
        f"dataset:\n- dummy/dataset\n"
        f"output_dir: {tmp_path / 'out'}\n"
        f"bf16: false\nuse_cpu: true\n{config_body}"
    )
    patches = [
        *patches,
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


def _run_rlvr(tmp_path, dataset: DatasetDict) -> None:
    """Run ``online_grpo/rlvr.py:main()`` to its dataset stage."""
    module = _script_module("online_grpo/rlvr.py")
    tokenizer = types.SimpleNamespace(padding_side="right")

    patches = [
        mock.patch.object(module, "init_training_script", return_value=_runtime()),
        mock.patch.object(module, "load_script_model", return_value=(types.SimpleNamespace(config=None), tokenizer)),
        mock.patch.object(module, "apply_prompt_completion_window", return_value=(tokenizer, None)),
        mock.patch.object(module, "setup_peft_model", return_value=None),
        mock.patch.object(module, "log_model_info"),
        mock.patch.object(module, "load_script_datasets", return_value=(dataset, False)),
        mock.patch.object(module, "process_dataset_with_map_and_filter", side_effect=_PastTheGuard),
    ]
    _run(module, tmp_path, "max_completion_length: 64\n", patches)


def _run_embedding(tmp_path, dataset: DatasetDict) -> None:
    """Run ``embedding.py:main()`` to its dataset stage."""
    module = _script_module("embedding.py")

    patches = [
        mock.patch.object(module, "init_training_script", return_value=_runtime()),
        mock.patch.object(module, "load_script_datasets", return_value=(dataset, False)),
        mock.patch.object(module, "build_sentence_transformer", side_effect=_PastTheGuard),
    ]
    _run(module, tmp_path, "max_length: 512\n", patches)


def _run_environmental_grpo(tmp_path, dataset: DatasetDict) -> None:
    """Run ``environmental_grpo.py:main()`` to its dataset stage."""
    module = _script_module("environmental_grpo.py")
    tokenizer = types.SimpleNamespace(padding_side="right")

    patches = [
        mock.patch.object(module, "init_training_script", return_value=_runtime()),
        mock.patch.object(module, "load_script_model", return_value=(types.SimpleNamespace(config=None), tokenizer)),
        mock.patch.object(
            module, "setup_model_and_tokenizer", side_effect=lambda args, model, tok, window, **kwargs: tok
        ),
        mock.patch.object(module, "get_model_context_window", return_value=4096),
        mock.patch.object(module, "setup_peft_model", return_value=None),
        mock.patch.object(module, "log_model_info"),
        mock.patch.object(module, "load_script_datasets", return_value=(dataset, False)),
        mock.patch.object(module, "process_dataset", side_effect=_PastTheGuard),
    ]
    _run(module, tmp_path, "environment_type: code_contests\n", patches)


def _run_offline_grpo(tmp_path, dataset: DatasetDict) -> None:
    """Run ``offline_grpo.py:main()`` to its dataset stage."""
    module = _script_module("offline_grpo.py")
    tokenizer = types.SimpleNamespace(padding_side="right")

    patches = [
        mock.patch.object(module, "init_training_script", return_value=_runtime()),
        mock.patch.object(module, "load_script_model", return_value=(types.SimpleNamespace(config=None), tokenizer)),
        mock.patch.object(module, "apply_prompt_completion_window", return_value=(tokenizer, None)),
        mock.patch.object(module, "setup_peft_model", return_value=None),
        mock.patch.object(module, "log_model_info"),
        mock.patch.object(module, "load_script_datasets", return_value=(dataset, False)),
        mock.patch.object(module, "coordinated_map", side_effect=_PastTheGuard),
    ]
    _run(module, tmp_path, "max_completion_length: 64\n", patches)


def _run_classification(tmp_path, dataset: DatasetDict) -> None:
    """Run ``classification.py:main()`` to its dataset stage (it loads the dataset before the model)."""
    module = _script_module("classification.py")

    patches = [
        mock.patch.object(module, "init_training_script", return_value=_runtime()),
        mock.patch.object(module, "load_script_datasets", return_value=(dataset, False)),
        mock.patch.object(module, "require_prompt_or_text_column", side_effect=_PastTheGuard),
    ]
    _run(module, tmp_path, "max_length: 512\n", patches)


# Each text-only method with the dataset shape it actually consumes, and the name its refusal uses.
_CASES = {
    "rlvr": (_run_rlvr, {"prompt": ["2+2?"], "answer": ["4"]}, "RLVR Online GRPO"),
    "embedding": (_run_embedding, {"anchor": ["question"], "positive": ["answer"]}, "Embedding training"),
    "environmental_grpo": (_run_environmental_grpo, {"prompt": ["2+2?"], "answer": ["4"]}, "Environmental GRPO"),
    "offline_grpo": (
        _run_offline_grpo,
        {"prompt": [_TEXT_TURNS], "completions": [[_TEXT_TURNS]], "rewards": [[1.0]]},
        "Offline GRPO",
    ),
    "classification": (_run_classification, {"prompt": [_TEXT_TURNS], "label": [0]}, "Classification"),
}


@pytest.mark.parametrize("case", sorted(_CASES))
def test_image_column_is_refused_before_it_can_be_pruned(case, tmp_path):
    """Every method that cannot serve images refuses a declared image column in its own pipeline.

    One representative spelling per script: the refusal is a single shared set-intersection against
    ``VLM_IMAGE_COLUMNS`` (``reject_image_columns``), so the spelling axis is swept once below
    rather than per script.
    """
    runner, columns, method = _CASES[case]
    with pytest.raises(ValueError, match=f"{method} is text-only"):
        runner(tmp_path, _dataset(columns, VLM_IMAGE_COLUMNS[0]))


@pytest.mark.parametrize("column", VLM_IMAGE_COLUMNS)
def test_every_image_column_spelling_is_refused(column, tmp_path):
    """The spelling set is policed on one representative script (shared intersection, see above)."""
    runner, columns, method = _CASES["classification"]
    with pytest.raises(ValueError, match=f"{method} is text-only"):
        runner(tmp_path, _dataset(columns, column))


@pytest.mark.parametrize("case", sorted(_CASES))
def test_clean_text_dataset_still_reaches_training(case, tmp_path):
    """Anti-vacuity: the guard rejects image columns, not datasets."""
    runner, columns, _ = _CASES[case]
    with pytest.raises(_PastTheGuard):
        runner(tmp_path, _dataset(columns, None))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
