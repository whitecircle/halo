#!/usr/bin/env python
"""``S3Client`` retries follow the AWS configuration: an explicit ``retries`` key would override
``AWS_MAX_ATTEMPTS`` and ``AWS_RETRY_MODE``. The toolkit's defaults (standard mode, 4 attempts) apply only to the
keys left unset."""

import pytest

from halo.data.sources.s3_client import S3Client


@pytest.fixture
def aws_config(tmp_path, monkeypatch):
    """An isolated AWS configuration: no ambient retry settings, an empty profile file the test may fill."""
    config_file = tmp_path / "config"
    config_file.write_text("")
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config_file))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "credentials"))
    for name in ("AWS_MAX_ATTEMPTS", "AWS_RETRY_MODE", "AWS_PROFILE", "AWS_DEFAULT_PROFILE"):
        monkeypatch.delenv(name, raising=False)
    return config_file


def _retries() -> dict:
    return S3Client(bucket="b").boto_client.meta.config.retries


def test_unset_aws_configuration_takes_the_toolkit_defaults(aws_config):
    assert _retries() == {"total_max_attempts": 4, "mode": "standard"}


def test_environment_retry_settings_reach_the_client(aws_config, monkeypatch):
    monkeypatch.setenv("AWS_MAX_ATTEMPTS", "12")
    monkeypatch.setenv("AWS_RETRY_MODE", "adaptive")
    assert _retries() == {"total_max_attempts": 12, "mode": "adaptive"}


def test_profile_retry_settings_reach_the_client(aws_config):
    aws_config.write_text("[default]\nmax_attempts = 7\nretry_mode = adaptive\n")
    assert _retries() == {"total_max_attempts": 7, "mode": "adaptive"}


@pytest.mark.parametrize(
    ("variable", "value", "expected"),
    [
        ("AWS_MAX_ATTEMPTS", "9", {"total_max_attempts": 9, "mode": "standard"}),
        ("AWS_RETRY_MODE", "legacy", {"total_max_attempts": 4, "mode": "legacy"}),
    ],
)
def test_each_unset_key_keeps_its_own_default(aws_config, monkeypatch, variable, value, expected):
    """Setting one knob must not drop the toolkit's default for the other."""
    monkeypatch.setenv(variable, value)
    assert _retries() == expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
