#!/usr/bin/env python
"""``evaluate``'s unequal-batch escape hatch: what it neutralizes, what it refuses, what it restores.

When ranks get different eval batch counts (or the count is unmeasurable — an iterable dataset), the
default eval loop's per-step ``gather_for_metrics`` deadlocks. The escape hatch makes each rank score
its own shard: ``gather_function`` becomes identity, and so does ``accelerator.pad_across_processes``
— ``evaluation_loop`` pads logits/labels across processes on a path that never routes through
``gather_function``.

Two things about that neutralization must hold, and neither is visible from a passing eval:

* **Refusal.** TRL's GRPO entropy threshold (``top_entropy_quantile < 1.0``) pads-and-gathers per
  prediction step through the SAME accelerator. Identity padding then hands ``all_gather`` ragged
  tensors and the job SIGABRTs; real padding deadlocks on genuinely unequal counts. There is no
  working combination, so the branch raises instead of choosing which way to die.
* **Restoration by deletion.** ``pad_across_processes`` is a CLASS method; the hatch plants an
  instance attribute over it. Restoring by re-assigning the bound method would pin that shadow
  permanently — every later gather in the process would silently keep the identity semantics. The
  attribute has to be *deleted* off ``__dict__``.

Run: ``python tests/cpu/trainers/test_eval_gather_escape_hatch.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

import src.trainers.mixins.base as mixin_module
from src.trainers.mixins.base import DistributedTrainerMixin

# Any quantile < 1.0 turns entropy masking on; the sentinel is what the stubbed super().evaluate()
# returns, so an assert matching it proves the call passed THROUGH the escape hatch unchanged.
_MASKING_QUANTILE = 0.8
_SENTINEL_METRICS = {"eval_loss": 0.5}

_SENTINEL = object()


class _Accelerator:
    """Only the seam under test. ``pad_across_processes`` is defined on the CLASS, which is what
    makes "restore by deletion" vs "restore by re-assignment" observable at all."""

    def pad_across_processes(self, tensor, *args, **kwargs):
        return ("class-padded", tensor)


class _InnerTrainer:
    """Stands in for the TRL/transformers ``Trainer.evaluate`` the mixin delegates to. It records
    what the two seams did AT THE MOMENT OF THE CALL — after the hatch, before the restore."""

    def evaluate(self, *args, **kwargs):
        self.observed.append((self.gather_function(_SENTINEL), self.accelerator.pad_across_processes(_SENTINEL)))
        return dict(_SENTINEL_METRICS)


class _StubTrainer(DistributedTrainerMixin, _InnerTrainer):
    """Real MRO so the mixin's zero-arg ``super().evaluate()`` resolves; no ``Trainer.__init__``."""

    def __init__(self, *, top_entropy_quantile: float = 1.0, unequal: bool = True):
        self.args = SimpleNamespace(top_entropy_quantile=top_entropy_quantile)
        self.accelerator = _Accelerator()
        self.gather_function = lambda tensor: ("real-gather", tensor)
        self.observed: list = []
        self._unequal = unequal

    def _needs_custom_accelerator(self) -> bool:
        return True

    def _eval_ranks_have_unequal_batches(self, eval_args, eval_kwargs) -> bool:
        return self._unequal


@pytest.fixture
def distributed():
    """The branch is gated on a live process group; nothing below needs a real one."""
    with (
        patch.object(mixin_module.dist, "is_initialized", return_value=True),
        patch.object(mixin_module, "barrier", lambda *args, **kwargs: None),
    ):
        yield


def test_entropy_masking_is_refused_on_the_unequal_batch_branch(distributed):
    trainer = _StubTrainer(top_entropy_quantile=_MASKING_QUANTILE)

    with pytest.raises(ValueError, match="top_entropy_quantile") as excinfo:
        trainer.evaluate()

    assert "1.0" in str(excinfo.value), "the message must name the setting that unblocks the run"
    assert trainer.observed == [], "the inner evaluate must not have been reached"


def test_the_refusal_fires_before_anything_is_neutralized(distributed):
    """A raise from inside the ``try`` would leave the swap in place — the ``finally`` only covers
    what it wraps. Nothing may be touched on the way out."""
    trainer = _StubTrainer(top_entropy_quantile=_MASKING_QUANTILE)

    with pytest.raises(ValueError):
        trainer.evaluate()

    assert trainer.gather_function(_SENTINEL) == ("real-gather", _SENTINEL)
    assert "pad_across_processes" not in trainer.accelerator.__dict__


def test_both_seams_are_identity_inside_the_hatch_and_restored_after(distributed):
    trainer = _StubTrainer()

    assert trainer.evaluate() == _SENTINEL_METRICS

    gathered, padded = trainer.observed[0]
    assert gathered is _SENTINEL, "gather_function must be identity while the inner loop runs"
    assert padded is _SENTINEL, "pad_across_processes must be identity too — evaluation_loop pads there"

    assert trainer.gather_function(_SENTINEL) == ("real-gather", _SENTINEL)
    assert trainer.accelerator.pad_across_processes(_SENTINEL) == ("class-padded", _SENTINEL)


def test_the_accelerator_is_restored_by_deleting_the_instance_attribute(distributed):
    """The behavioural difference from re-assigning the bound method: ``__dict__`` must come out
    clean, so the class method is what every later caller resolves to."""
    trainer = _StubTrainer()
    trainer.evaluate()

    assert "pad_across_processes" not in trainer.accelerator.__dict__


def test_the_hatch_is_unwound_when_the_inner_evaluate_raises(distributed):
    trainer = _StubTrainer()
    with patch.object(_InnerTrainer, "evaluate", side_effect=RuntimeError("boom")):
        with pytest.raises(RuntimeError, match="boom"):
            trainer.evaluate()

    assert trainer.gather_function(_SENTINEL) == ("real-gather", _SENTINEL)
    assert "pad_across_processes" not in trainer.accelerator.__dict__


def test_the_equal_batch_path_never_swaps_either_seam(distributed):
    """Identity gathering on equal batches would report rank-0's shard as the global metric, so the
    hatch must stay shut — and with it the refusal, which is only about the hatch."""
    trainer = _StubTrainer(top_entropy_quantile=_MASKING_QUANTILE, unequal=False)

    assert trainer.evaluate() == _SENTINEL_METRICS

    gathered, padded = trainer.observed[0]
    assert gathered == ("real-gather", _SENTINEL)
    assert padded == ("class-padded", _SENTINEL)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
