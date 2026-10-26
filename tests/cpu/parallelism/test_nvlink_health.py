#!/usr/bin/env python
"""NVLink health classification: only errors FEC did not absorb may fail the check.

The check exists to separate a real fabric fault from the correctable churn that produces nonfatal
``Xid 145`` log spam. Both directions matter: a marginal-but-corrected link must NOT fail a preflight
gate (crying wolf makes the gate useless), and a hard error must never be downgraded to a warning.

Run: ``pytest tests/cpu/parallelism/test_nvlink_health.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import sys

import pytest

from tests.common.repo import load_script_module

nvlink_health = load_script_module("scripts/profiling/nvlink_health.py", register=True)


def _counters(*, clean: int = 10**13, corrected: int = 0, deep: dict[int, int] | None = None, **hard: int):
    """Counter dict in the shape ``nvidia-smi nvlink -e`` reports."""
    values = {key: 0 for key in nvlink_health.HARD_ERROR_KEYS}
    values.update(hard)
    values["FEC Errors - 0"] = clean
    values["FEC Errors - 1"] = corrected
    for bin_index, count in (deep or {}).items():
        values[f"FEC Errors - {bin_index}"] = count
    return values


def test_clean_link_is_neither_unhealthy_nor_marginal():
    result = nvlink_health.assess_link(_counters(corrected=100))
    assert not result["unhealthy"]
    assert not result["marginal"]
    assert result["correction_rate"] == pytest.approx(1e-11, rel=1e-6)


def test_hard_error_fails_the_check():
    """A replay/CRC/recovery counter means FEC did not absorb it — data was at risk."""
    for key in ("Rx Errors", "Link recovery successful events", "Symbol Errors"):
        result = nvlink_health.assess_link(_counters(**{key: 1}))
        assert result["unhealthy"], f"{key} must fail the check"
        assert result["hard_errors"] == {key: 1}


def test_high_correction_rate_is_marginal_not_unhealthy():
    """The observed real-world case: ~1e-7 corrected codewords, full bandwidth, no data at risk.

    Failing here would make the preflight gate unusable on hardware that is working correctly.
    """
    result = nvlink_health.assess_link(_counters(clean=10**13, corrected=10**6))
    assert result["marginal"]
    assert not result["unhealthy"]
    assert result["correction_rate"] > nvlink_health.CORRECTION_RATE_WARN


def test_isolated_deep_bin_event_is_marginal_not_unhealthy():
    """One codeword needing 3 symbol corrections out of ~1e13 is link noise, not a fault."""
    result = nvlink_health.assess_link(_counters(corrected=10, deep={3: 1}))
    assert result["marginal"], "a deep-bin event is worth watching"
    assert not result["unhealthy"], "a single deep-bin event must not fail an otherwise healthy link"
    assert result["deep_fec_bins"] == {3: 1}


def test_hard_error_outranks_correctable_churn():
    """A link with both must report unhealthy, never be softened to marginal."""
    result = nvlink_health.assess_link(_counters(corrected=10**6, deep={3: 5}, **{"Rx Errors": 2}))
    assert result["unhealthy"]
    assert not result["marginal"]


def test_an_unrecognized_counter_set_fails_instead_of_reading_clean():
    """The pre-FEC driver spelling (Replay/Recovery/CRC) matches none of the counters this check
    reads; a link reporting only those carries a CRC fault the check would otherwise call clean."""
    with pytest.raises(RuntimeError, match="unrecognized NVLink counter set"):
        nvlink_health.assess_link({"Replay Errors": 0, "Recovery Errors": 0, "CRC Errors": 3})


def test_a_link_reporting_only_fec_bins_is_still_assessed():
    """The guard keys on the counter families, not on every hard-error key being present."""
    result = nvlink_health.assess_link({"FEC Errors - 0": 10**13, "FEC Errors - 1": 10})
    assert not result["unhealthy"]
    assert not result["marginal"]


def _stub_nvidia_smi(monkeypatch, speeds: dict[int, dict[int, str | None]]) -> None:
    """``nvidia-smi`` for the GPUs keyed in ``speeds``: every listed link reports clean counters, and its
    ``nvlink -s`` line the given speed (``None``: no line at all)."""

    def _answer(*args: str) -> str:
        if args == ("--list-gpus",):
            return "".join(f"GPU {gpu}: NVIDIA B200\n" for gpu in speeds)
        _nvlink, flag, _index, gpu = args
        links = speeds[int(gpu)]
        if flag == "-e":
            return "".join(f"\t Link {link}: FEC Errors - 0: 1000\n\t Link {link}: Rx Errors: 0\n" for link in links)
        return "".join(f"\t Link {link}: {speed}\n" for link, speed in links.items() if speed is not None)

    monkeypatch.setattr(nvlink_health, "_nvidia_smi", _answer)
    monkeypatch.setattr(sys, "argv", ["nvlink_health.py"])


_UP = "53.125 GB/s"


def test_a_fully_up_fabric_passes(monkeypatch, capsys):
    """Anti-vacuity for the two checks below: equal, active link sets exit clean."""
    _stub_nvidia_smi(monkeypatch, {0: {0: _UP, 1: _UP}, 1: {0: _UP, 1: _UP}})

    assert nvlink_health.main() == 0
    assert "0 down" in capsys.readouterr().out


@pytest.mark.parametrize("speed", ["<inactive>", None])
def test_an_inactive_link_fails_the_check(monkeypatch, capsys, speed):
    """A down link has clean error counters, yet it takes its bandwidth from DeepEP and NCCL."""
    _stub_nvidia_smi(monkeypatch, {0: {0: _UP, 1: speed}, 1: {0: _UP, 1: _UP}})

    assert nvlink_health.main() == 1
    out = capsys.readouterr().out
    assert "GPU 0 link  1: DOWN" in out and "1 down" in out, out


def test_a_link_down_on_every_gpu_fails_the_check(monkeypatch, capsys):
    """Each link is judged on its own entry, not against its peers: the same link down on every GPU (an
    NVSwitch plane out) leaves the peer counts equal, and an unwired link nvidia-smi lists reads the same."""
    _stub_nvidia_smi(monkeypatch, {0: {0: _UP, 1: "<inactive>"}, 1: {0: _UP, 1: "<inactive>"}})

    assert nvlink_health.main() == 1
    out = capsys.readouterr().out
    assert "2 down" in out and "0 GPU(s) short of links" in out, out


def test_a_gpu_short_of_links_fails_the_check(monkeypatch, capsys):
    """A link absent from both listings is invisible per link; its GPU has fewer active links than its peers."""
    _stub_nvidia_smi(monkeypatch, {0: {0: _UP}, 1: {0: _UP, 1: _UP}})

    assert nvlink_health.main() == 1
    assert "GPU 0: 1 active links, peers have 2" in capsys.readouterr().out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
