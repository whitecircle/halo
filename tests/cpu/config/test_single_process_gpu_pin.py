#!/usr/bin/env python
"""A single-process run trains on the one GPU it bound, never under ``nn.DataParallel``.

HF's Trainer wraps the model in ``nn.DataParallel`` whenever ``args.n_gpu > 1``, which is what a
plain ``python`` launch reports on a multi-GPU host. The token-normalized losses are not written for
that wrapper (the logged loss and the gradients scale with the GPU count), so
``init_training_script`` pins the training config to one GPU before a trainer is built from it.

Run: pytest tests/cpu/config/test_single_process_gpu_pin.py
"""

import ast
import inspect
import logging

import pytest
from transformers import TrainingArguments

from src.training import script_runner
from src.training.script_runner import init_training_script, pin_single_process_to_bound_gpu


def _training_args(tmp_path, visible_gpus: int) -> TrainingArguments:
    args = TrainingArguments(output_dir=str(tmp_path), report_to=[], per_device_train_batch_size=4)
    _ = args.n_gpu  # device setup has run, so the override below is not written over
    args._n_gpu = visible_gpus
    return args


def _pin_warnings(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == script_runner.__name__]


def test_several_visible_gpus_are_pinned_to_one(tmp_path, caplog):
    args = _training_args(tmp_path, visible_gpus=4)
    assert args.train_batch_size == 16  # the DataParallel batch HF would otherwise train with

    with caplog.at_level(logging.WARNING, logger=script_runner.__name__):
        pin_single_process_to_bound_gpu(args)

    assert args.n_gpu == 1
    assert args.train_batch_size == 4
    assert args.eval_batch_size == args.per_device_eval_batch_size
    (warning,) = _pin_warnings(caplog)
    assert "4 GPUs are visible" in warning
    assert "halo launch" in warning


@pytest.mark.parametrize("visible_gpus", [0, 1])
def test_one_or_no_gpu_is_left_alone(tmp_path, caplog, visible_gpus):
    args = _training_args(tmp_path, visible_gpus=visible_gpus)

    with caplog.at_level(logging.WARNING, logger=script_runner.__name__):
        pin_single_process_to_bound_gpu(args)

    assert args.n_gpu == visible_gpus
    assert _pin_warnings(caplog) == []


def test_init_training_script_applies_the_pin_to_its_training_config():
    """Every entry script builds its trainer from the config this phase returns control with."""
    function = ast.parse(inspect.getsource(init_training_script)).body[0]
    pins = [
        node
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "pin_single_process_to_bound_gpu"
    ]
    assert len(pins) == 1
    (argument,) = pins[0].args
    assert isinstance(argument, ast.Name) and argument.id == "training_config"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
