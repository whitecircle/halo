#!/usr/bin/env python
"""HALO_DATASET_NUM_PROC resolution.

Falsy-override guard: applying the pin as ``env_int(...) or min(cpu//4, 4)`` makes an explicit
``0`` falsy and silently replaces it with the CPU-derived value — on a heterogeneous fleet exactly
the silent divergence the pin exists to prevent. The resolver fails loud on a pin below 1
(``num_proc`` counts worker processes; 1 disables multiprocessing) and honors any valid explicit
value.

Run: pytest tests/cpu/data/test_processing_threads.py
"""

import multiprocessing
import sys

import pytest

from src.data.pipeline.processing import _resolve_dataset_num_proc


@pytest.fixture(autouse=True)
def _clear_pin(monkeypatch):
    """Unset unless a test sets it."""
    monkeypatch.delenv("HALO_DATASET_NUM_PROC", raising=False)


def test_unset_derives_from_cpu_count():
    # max(1, ...): a <=7-core runner floors the quarter-share to 0, which the resolver refuses.
    assert _resolve_dataset_num_proc() == max(1, min(multiprocessing.cpu_count() // 4, 4))


def test_explicit_pin_wins_over_derived_value(monkeypatch):
    """The fleet-wide pin must be honored verbatim — including values the CPU-derived cap would
    never produce (num_proc keys the per-worker cache file set)."""
    monkeypatch.setenv("HALO_DATASET_NUM_PROC", "7")
    assert _resolve_dataset_num_proc() == 7


def test_pin_of_one_disables_multiprocessing_but_is_valid(monkeypatch):
    monkeypatch.setenv("HALO_DATASET_NUM_PROC", "1")
    assert _resolve_dataset_num_proc() == 1


def test_zero_pin_fails_loud_not_silently_ignored(monkeypatch):
    """An explicit 0 is falsy → a truthiness fallback silently swaps in the derived value; it must raise."""
    monkeypatch.setenv("HALO_DATASET_NUM_PROC", "0")
    with pytest.raises(ValueError, match="HALO_DATASET_NUM_PROC"):
        _resolve_dataset_num_proc()


def test_negative_pin_fails_loud(monkeypatch):
    monkeypatch.setenv("HALO_DATASET_NUM_PROC", "-2")
    with pytest.raises(ValueError, match="must be >= 1"):
        _resolve_dataset_num_proc()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
