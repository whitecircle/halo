#!/usr/bin/env python
"""CP setup refuses settings that pair a rank's sequence chunk with full-length labels and adopts the wrapper's config.
A second config would create a second set of CP process groups."""

import types

import pytest

import halo.trainers.parallelism.mode_setup as mode_setup
from halo.distributed.context_parallel.config import CPConfigError
from halo.trainers.distributed_trainer import DistributedTrainerMixin

_configure_cp = DistributedTrainerMixin._adopt_cp_wrapper


_LOADED_CP_CONFIG = object()


class _Wrapper:
    """Stands in for the loaded UlyssesModel."""

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
    me._top_level_model = lambda: wrapper
    return me


@pytest.fixture(autouse=True)
def _find_the_stand_in(monkeypatch):
    monkeypatch.setattr(mode_setup, "find_ulysses_model", lambda model: model if isinstance(model, _Wrapper) else None)


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
    """``evaluate()`` and ``predict()`` run the metric path whatever the strategy, misaligning logits and labels."""
    with_compute = _fake_self(eval_strategy=eval_strategy)
    with_compute.compute_metrics = lambda _preds: {}
    with pytest.raises(CPConfigError, match="compute_metrics"):
        _configure_cp(with_compute)

    with_preprocess = _fake_self(eval_strategy=eval_strategy)
    with_preprocess.preprocess_logits_for_metrics = lambda logits, labels: logits
    with pytest.raises(CPConfigError, match="preprocess_logits_for_metrics"):
        _configure_cp(with_preprocess)

    # Only the hooks are refused: loss-only eval still configures.
    _configure_cp(_fake_self(eval_strategy=eval_strategy))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
