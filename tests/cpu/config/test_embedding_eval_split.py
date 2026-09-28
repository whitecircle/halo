#!/usr/bin/env python
"""``embedding.py`` hands the trainer the dataset's ``test`` split whenever the dataset has one.

A pre-sharded eval split can leave a rank no rows. Chosen by truthiness, that rank alone got no eval
dataset (or fell through to another split) and skipped the evaluation its peers entered.

Driven through ``main()`` up to the trainer construction, with the model build stubbed.

Run: pytest tests/cpu/config/test_embedding_eval_split.py
"""

import sys
import types
from unittest import mock

import pytest
from datasets import Dataset, DatasetDict

from tests.common.utils import load_script_module

_MODEL_ID = "stub/qwen3-4b"


class _TrainerReached(Exception):
    """Raised by the stubbed trainer, carrying the kwargs ``main()`` built it with."""


def _eval_dataset_handed_to_the_trainer(tmp_path, dataset: DatasetDict):
    module = load_script_module("scripts/training/embedding.py", "halo_test_embedding_eval_split")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"model_name_or_path: {_MODEL_ID}\ndataset:\n- dummy/dataset\noutput_dir: {tmp_path / 'out'}\n"
        "bf16: false\nuse_cpu: true\nmax_length: 512\n"
    )
    runtime = types.SimpleNamespace(
        parallelism_config=types.SimpleNamespace(cp_size=1, is_cp_mode=False, pp_size=1, is_ep_mode=False),
        model_source=_MODEL_ID,
        mode_suffix="",
        local_rank=0,
        resume_checkpoint=None,
    )
    model = types.SimpleNamespace(get_sentence_embedding_dimension=lambda: 8)

    def build_trainer(**kwargs):
        raise _TrainerReached(kwargs)

    patches = [
        mock.patch.object(module, "init_training_script", return_value=runtime),
        mock.patch.object(module, "load_script_datasets", return_value=(dataset, False)),
        mock.patch.object(module, "build_sentence_transformer", return_value=model),
        mock.patch.object(module, "build_peft_config", return_value=None),
        mock.patch.object(module, "apply_distributed_trainer_config"),
        mock.patch.object(module, "barrier"),
        mock.patch.object(module, "build_training_callbacks", return_value=[]),
        mock.patch.object(module, "EmbeddingTrainer", side_effect=build_trainer),
        mock.patch("src.training.parser.install_log_tee"),
        mock.patch.object(sys, "argv", ["prog", str(config)]),
    ]
    for patch in patches:
        patch.start()
    try:
        with pytest.raises(_TrainerReached) as reached:
            module.main()
    finally:
        for patch in reversed(patches):
            patch.stop()
    return reached.value.args[0]["eval_dataset"]


def _pairs(rows: int) -> Dataset:
    return Dataset.from_dict({"anchor": ["question"] * rows, "positive": ["answer"] * rows})


def test_an_empty_test_split_is_still_the_eval_split(tmp_path):
    eval_dataset = _eval_dataset_handed_to_the_trainer(tmp_path, DatasetDict({"train": _pairs(2), "test": _pairs(0)}))
    assert eval_dataset is not None and len(eval_dataset) == 0


def test_no_test_split_means_no_eval_dataset(tmp_path):
    assert _eval_dataset_handed_to_the_trainer(tmp_path, DatasetDict({"train": _pairs(2)})) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
