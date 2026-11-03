import importlib
import logging
import os

import pytest

import halo.env as env_mod
from halo.env import env_flag, env_float, env_int, env_non_negative_int, env_positive_float, env_positive_int, env_str
from tests.common.repo import probe_findings

_THREADS_MARKER = "TORCH_THREADS:"

_VARS = ("HALO_TEST_A", "HALO_TEST_B")


def _clear():
    for v in _VARS:
        os.environ.pop(v, None)


def setup_function(_):
    _clear()


def teardown_function(_):
    _clear()


def test_env_flag_truthiness():
    for truthy in ("1", "true", "TRUE", "Yes", "on"):
        os.environ["HALO_TEST_A"] = truthy
        assert env_flag("HALO_TEST_A") is True
    for falsy in ("0", "false", "FALSE", "No", "off", " off "):
        os.environ["HALO_TEST_A"] = falsy
        assert env_flag("HALO_TEST_A", default=True) is False


def test_env_flag_typo_keeps_the_default_and_warns_once_per_name(monkeypatch, caplog):
    """A typo must not flip a default-on knob off, and a flag read on every call must not warn on every call."""
    monkeypatch.setattr(env_mod, "_warned_flags", set())
    os.environ["HALO_TEST_A"] = "ture"
    os.environ["HALO_TEST_B"] = "banana"
    with caplog.at_level(logging.WARNING, logger=env_mod.__name__):
        assert env_flag("HALO_TEST_A", default=True) is True
        assert env_flag("HALO_TEST_A", default=True) is True
        assert env_flag("HALO_TEST_B", default=False) is False
    warned = [record.getMessage() for record in caplog.records]
    assert len(warned) == 2, warned
    assert "HALO_TEST_A='ture'" in warned[0] and "using True" in warned[0]
    assert "HALO_TEST_B='banana'" in warned[1] and "using False" in warned[1]


def test_env_flag_default_when_absent():
    assert env_flag("HALO_TEST_A") is False
    assert env_flag("HALO_TEST_A", default=True) is True


def test_env_flag_empty_value_falls_back_to_default():
    """Compose's ``${HALO_FOO:-}`` and a blank ``--env-file`` line export the name empty; reading that as False would
    disable every default-on knob."""
    for empty in ("", "   ", "\t"):
        os.environ["HALO_TEST_A"] = empty
        assert env_flag("HALO_TEST_A", default=True) is True
        assert env_flag("HALO_TEST_A", default=False) is False


def test_env_int_parse_and_fallback():
    os.environ["HALO_TEST_A"] = "42"
    assert env_int("HALO_TEST_A", 0) == 42
    assert env_int("HALO_TEST_B", 7) == 7
    os.environ["HALO_TEST_A"] = "not-an-int"
    assert env_int("HALO_TEST_A", 7) == 7


def test_env_float_parse_and_fallback():
    os.environ["HALO_TEST_A"] = "1.5"
    assert env_float("HALO_TEST_A", 0.0) == 1.5
    assert env_float("HALO_TEST_B", 2.5) == 2.5
    os.environ["HALO_TEST_A"] = "nope"
    assert env_float("HALO_TEST_A", 2.5) == 2.5


def test_env_positive_helpers_reject_non_positive_values():
    """A 0 or negative timeout would expire on entry, silently disarming the watchdog it configures."""
    for value in ("0", "-3"):
        os.environ["HALO_TEST_A"] = value
        assert env_positive_int("HALO_TEST_A", 7) == 7
        assert env_positive_float("HALO_TEST_A", 2.5) == 2.5
    os.environ["HALO_TEST_A"] = "-0.5"
    assert env_positive_float("HALO_TEST_A", 2.5) == 2.5


def test_env_positive_helpers_accept_positive_values():
    os.environ["HALO_TEST_A"] = "42"
    assert env_positive_int("HALO_TEST_A", 7) == 42
    os.environ["HALO_TEST_A"] = "1.5"
    assert env_positive_float("HALO_TEST_A", 2.5) == 1.5
    assert env_positive_int("HALO_TEST_B", 7) == 7
    assert env_positive_float("HALO_TEST_B", 2.5) == 2.5


def test_env_non_negative_int_keeps_zero_and_refuses_a_negative(caplog):
    """``0`` means auto or off; a negative is truthy, so a caller's ``if value:`` would act on it."""
    for value, expected in (("0", 0), ("3", 3)):
        os.environ["HALO_TEST_A"] = value
        assert env_non_negative_int("HALO_TEST_A", 7) == expected
    assert not caplog.records
    os.environ["HALO_TEST_A"] = "-1"
    with caplog.at_level(logging.WARNING, logger=env_mod.__name__):
        assert env_non_negative_int("HALO_TEST_A", 7) == 7
    assert "HALO_TEST_A=-1" in caplog.text and "using 7" in caplog.text


@pytest.mark.parametrize(("raw", "expected"), [("0", 1), ("3", 3)])
def test_the_package_import_applies_a_valid_thread_count_only(monkeypatch, raw, expected):
    """A non-positive count would raise from ``torch.set_num_threads`` at ``import halo``."""
    monkeypatch.setenv("HALO_TORCH_NUM_THREADS", raw)
    script = f"import halo, torch; print({_THREADS_MARKER!r} + str(torch.get_num_threads()))"
    assert probe_findings(script, _THREADS_MARKER) == [str(expected)]


def test_env_number_empty_value_falls_back_silently():
    """Reading an empty value as malformed would warn on every rank at import, for a variable nobody set."""
    for empty in ("", "   ", "\t"):
        os.environ["HALO_TEST_A"] = empty
        assert env_int("HALO_TEST_A", 7) == 7
        assert env_float("HALO_TEST_A", 2.5) == 2.5


def test_env_str_present_absent_and_empty():
    assert env_str("HALO_TEST_A") is None
    assert env_str("HALO_TEST_A", "fallback") == "fallback"
    # A set empty string is returned as-is (os.environ.get semantics); falling through to the default would make a
    # chain skip a real value.
    os.environ["HALO_TEST_A"] = ""
    assert env_str("HALO_TEST_A", "fallback") == ""
    os.environ["HALO_TEST_A"] = "value"
    assert env_str("HALO_TEST_A", "fallback") == "value"


def test_env_str_nested_fallback_chain():
    def resolve():
        return env_str("HALO_TEST_A", env_str("HALO_TEST_B", "home-default"))

    assert resolve() == "home-default"
    os.environ["HALO_TEST_B"] = "legacy"
    assert resolve() == "legacy"
    os.environ["HALO_TEST_A"] = "primary"
    assert resolve() == "primary"


def test_data_path_derives_from_scratch_root():
    os.environ.pop("HALO_DATA_ROOT", None)
    mod = importlib.reload(env_mod)
    try:
        assert os.path.join(os.path.expanduser("~"), ".cache", "halo") == mod.HALO_DATA_ROOT
        assert not mod.HALO_DATA_ROOT.startswith("/mnt")
        assert mod.data_path("s3_datasets") == os.path.join(mod.HALO_DATA_ROOT, "s3_datasets")
        assert mod.data_path("profiling", "torch") == os.path.join(mod.HALO_DATA_ROOT, "profiling", "torch")

        os.environ["HALO_DATA_ROOT"] = "/scratch/big"
        mod = importlib.reload(env_mod)
        assert mod.data_path("s3_datasets") == "/scratch/big/s3_datasets"
        assert mod.data_path("profiling", "memory") == "/scratch/big/profiling/memory"
    finally:
        os.environ.pop("HALO_DATA_ROOT", None)
        importlib.reload(env_mod)  # restore the module-level constant for other tests


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
