#!/usr/bin/env python
"""Test that CP setup runs the trainer-args gate and adopts the loaded wrapper's CP config.

``reject_cp_trainer_args`` rejects the settings whose loss/metric path re-derives a result
from ``(logits, labels)`` outside the CP wrapper — label smoothing, ``loss_type='dft'``, and eval
metric hooks — each of which silently pairs this rank's sequence chunk with full-length labels.
The check is only worth anything if the CP setup path actually calls it. The trainer must reuse the
wrapper's config rather than build a second one: that would create a second set of CP process groups.

Run: python tests/cpu/trainers/test_cp_trainer_args_gate.py
"""

import types

import pytest
from accelerate import PartialState

from src.distributed.context_parallel.config import CPConfigError
from src.trainers.mixins.base import DistributedTrainerMixin

PartialState()  # the mixin's accelerate logger requires an initialized state

_configure_cp = DistributedTrainerMixin._configure_cp_wrapper


_LOADED_CP_CONFIG = object()


class _Wrapper:
    """Stands in for the loaded UlyssesModel (found via the injected _find_cp_wrapper)."""

    cp_config = _LOADED_CP_CONFIG


def _second_cp_config():
    raise AssertionError("the trainer built a second CPConfig instead of adopting the wrapper's")


def _fake_self(**args_overrides):
    args = types.SimpleNamespace(
        label_smoothing_factor=0.0,
        loss_type="nll",
        eval_strategy="no",
    )
    for name, value in args_overrides.items():
        setattr(args, name, value)
    wrapper = _Wrapper()
    me = types.SimpleNamespace(
        args=args,
        compute_metrics=None,
        preprocess_logits_for_metrics=None,
        parallelism_config=types.SimpleNamespace(cp_config=_second_cp_config),
    )
    me._find_cp_wrapper = lambda: wrapper
    return me


def test_clean_args_adopt_the_loaded_wrappers_config():
    me = _fake_self()
    _configure_cp(me)
    assert me.cp_config is _LOADED_CP_CONFIG


def test_label_smoothing_is_rejected():
    with pytest.raises(CPConfigError, match="label_smoothing_factor"):
        _configure_cp(_fake_self(label_smoothing_factor=0.1))


def test_dft_loss_type_is_rejected():
    with pytest.raises(CPConfigError, match="dft"):
        _configure_cp(_fake_self(loss_type="dft"))


@pytest.mark.parametrize("eval_strategy", ["steps", "no"])
def test_metric_hooks_are_rejected_whatever_the_eval_strategy(eval_strategy):
    """``eval_strategy='no'`` is not a promise the hooks go unused.

    ``evaluate()`` / ``predict()`` run the metric path on demand, whatever the strategy the trainer
    was built with, and there the hooks pair this rank's sequence-chunk logits with the full-length
    labels. A refusal gated on the strategy accepts exactly that trainer and misaligns at the first
    manual ``evaluate()`` — a silently wrong metric, mid-run.
    """
    with_compute = _fake_self(eval_strategy=eval_strategy)
    with_compute.compute_metrics = lambda _preds: {}
    with pytest.raises(CPConfigError, match="compute_metrics"):
        _configure_cp(with_compute)

    with_preprocess = _fake_self(eval_strategy=eval_strategy)
    with_preprocess.preprocess_logits_for_metrics = lambda logits, labels: logits
    with pytest.raises(CPConfigError, match="preprocess_logits_for_metrics"):
        _configure_cp(with_preprocess)

    # Anti-vacuity: the hooks are what is refused, not evaluation. Loss-only eval still configures.
    _configure_cp(_fake_self(eval_strategy=eval_strategy))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
