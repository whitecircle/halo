"""CPU tests for the torch-free env-var helpers in :mod:`src.env`.

Covers the truthiness/parse-and-fallback contract every HALO_/DIST_ knob relies on, and pins the
:func:`env_str` fallback-chain semantics that ``src/data/sources/dataset_cache.py`` depends on
(``HALO_S3_DATASET_CACHE_DIR`` → ``S3_DATASET_CACHE_DIR`` → home default).
"""

import os

import pytest

import src.env as env_mod
from src.env import env_flag, env_float, env_int, env_positive_float, env_positive_int, env_str

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
    for falsy in ("0", "false", "no", "off", "", "banana"):
        os.environ["HALO_TEST_A"] = falsy
        assert env_flag("HALO_TEST_A") is False


def test_env_flag_default_when_absent():
    assert env_flag("HALO_TEST_A") is False
    assert env_flag("HALO_TEST_A", default=True) is True


def test_env_flag_empty_value_falls_back_to_default():
    """A set-but-empty value means "unset", so a default-on flag stays on.

    Docker/compose pass-through (``- HALO_FOO``) exports the name with an empty value when it is
    absent from the caller's environment; reading that as False would silently disable every
    default-on knob (HALO_EP_CAPACITY_DEDUP, HALO_LOWP_COMPILE, ...).
    """
    for empty in ("", "   ", "\t"):
        os.environ["HALO_TEST_A"] = empty
        assert env_flag("HALO_TEST_A", default=True) is True
        assert env_flag("HALO_TEST_A", default=False) is False


def test_env_int_parse_and_fallback():
    os.environ["HALO_TEST_A"] = "42"
    assert env_int("HALO_TEST_A", 0) == 42
    # Absent → default; malformed → warns and falls back (never raises).
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


def test_env_positive_helpers_accept_positive_and_none_default():
    os.environ["HALO_TEST_A"] = "42"
    assert env_positive_int("HALO_TEST_A", 7) == 42
    os.environ["HALO_TEST_A"] = "1.5"
    assert env_positive_float("HALO_TEST_A", 2.5) == 1.5
    # Absent stays the default — including a None default (the "no override" spelling).
    assert env_positive_int("HALO_TEST_B", 7) == 7
    assert env_positive_float("HALO_TEST_B", None) is None


def test_env_number_empty_value_falls_back_silently():
    """Set-but-empty is "unset" for the numeric helpers too, and must not warn.

    Same compose pass-through as :func:`env_flag`'s case. Reading ``""`` as malformed would leave a
    module-level ``env_int`` warning on every rank at import, for a variable nobody set.
    """
    for empty in ("", "   ", "\t"):
        os.environ["HALO_TEST_A"] = empty
        assert env_int("HALO_TEST_A", 7) == 7
        assert env_float("HALO_TEST_A", 2.5) == 2.5


def test_env_str_present_absent_and_empty():
    # Absent → default (None when omitted).
    assert env_str("HALO_TEST_A") is None
    assert env_str("HALO_TEST_A", "fallback") == "fallback"
    # A set value is returned as-is, INCLUDING the empty string (os.environ.get semantics) — the
    # empty string must NOT fall through to the default, or a fallback chain would skip a real value.
    os.environ["HALO_TEST_A"] = ""
    assert env_str("HALO_TEST_A", "fallback") == ""
    os.environ["HALO_TEST_A"] = "value"
    assert env_str("HALO_TEST_A", "fallback") == "value"


def test_env_str_nested_fallback_chain():
    """Nesting a chain in the default resolves primary → secondary → literal, with a set-but-empty
    primary still winning."""

    def resolve():
        return env_str("HALO_TEST_A", env_str("HALO_TEST_B", "home-default"))

    assert resolve() == "home-default"
    os.environ["HALO_TEST_B"] = "legacy"
    assert resolve() == "legacy"
    os.environ["HALO_TEST_A"] = "primary"
    assert resolve() == "primary"


def test_data_path_derives_from_scratch_root():
    """data_path joins under HALO_DATA_ROOT; the default is a home-dir cache (never the root FS or
    a hardcoded /mnt), and an explicit HALO_DATA_ROOT overrides it."""
    import importlib

    os.environ.pop("HALO_DATA_ROOT", None)
    mod = importlib.reload(env_mod)
    try:
        # Default falls back to ~/.cache/halo — a writable home dir, not /mnt or a full root FS.
        assert os.path.join(os.path.expanduser("~"), ".cache", "halo") == mod.HALO_DATA_ROOT
        assert not mod.HALO_DATA_ROOT.startswith("/mnt")
        assert mod.data_path("s3_datasets") == os.path.join(mod.HALO_DATA_ROOT, "s3_datasets")
        assert mod.data_path("profiling", "torch") == os.path.join(mod.HALO_DATA_ROOT, "profiling", "torch")

        # An explicit root wins and every derived path relocates with it — one knob.
        os.environ["HALO_DATA_ROOT"] = "/scratch/big"
        mod = importlib.reload(env_mod)
        assert mod.data_path("s3_datasets") == "/scratch/big/s3_datasets"
        assert mod.data_path("profiling", "memory") == "/scratch/big/profiling/memory"
    finally:
        os.environ.pop("HALO_DATA_ROOT", None)
        importlib.reload(env_mod)  # restore the module-level constant for other tests


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
