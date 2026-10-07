#!/usr/bin/env python
"""``keep_all_dataset_columns`` turns HF's column pruning off for a trainer that reads columns its
model's forward never names (env GRPO's rollout context, offline GRPO's collator, the SBERT collator).

    python tests/cpu/trainers/test_keep_all_dataset_columns.py
"""

import types
from unittest import mock

import pytest

from src.trainers.mixins import validation
from src.trainers.mixins.validation import keep_all_dataset_columns


def test_the_flag_is_cleared_and_the_reason_said_once():
    args = types.SimpleNamespace(remove_unused_columns=True)
    with mock.patch.object(validation, "logger") as logger:
        keep_all_dataset_columns(args, "the collator reads every column")
    assert args.remove_unused_columns is False
    logger.warning.assert_called_once_with("the collator reads every column")


def test_a_flag_already_off_is_left_alone_and_nothing_is_said():
    args = types.SimpleNamespace(remove_unused_columns=False)
    with mock.patch.object(validation, "logger") as logger:
        keep_all_dataset_columns(args, "unused")
    assert args.remove_unused_columns is False
    logger.warning.assert_not_called()


def test_without_a_reason_the_flag_is_cleared_silently_and_no_args_is_a_no_op():
    args = types.SimpleNamespace(remove_unused_columns=True)
    with mock.patch.object(validation, "logger") as logger:
        keep_all_dataset_columns(args)
        keep_all_dataset_columns(None)
    assert args.remove_unused_columns is False
    logger.warning.assert_not_called()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
