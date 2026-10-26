"""A training run's INFO root: ``configure_training_logging`` sets it, and ``parse_script_args`` and the GPU test
harness call it before the process group starts.

python tests/cpu/test_log.py
"""

import logging

import pytest

from src.entrypoints import script
from src.log import configure_training_logging
from tests.common import harness


@pytest.fixture
def warning_root():
    """The root logger at the WARNING level the package import leaves it at, restored afterwards."""
    root = logging.getLogger()
    level = root.level
    root.setLevel(logging.WARNING)
    yield root
    root.setLevel(level)


class _StubParser:
    def __init__(self, dataclass_types):
        self.dataclass_types = dataclass_types

    def load_values(self) -> dict:
        return {}

    def parse_dict(self, values: dict) -> tuple:
        return ("parsed",)


def test_configure_training_logging_sets_the_root_to_info(warning_root):
    configure_training_logging()

    assert warning_root.level == logging.INFO


def test_parse_script_args_raises_the_root_before_the_process_group_starts(warning_root, monkeypatch):
    levels_at_group_start = []
    monkeypatch.setattr(script, "init_distributed", lambda: levels_at_group_start.append(warning_root.level))
    monkeypatch.setattr(script, "HaloArgumentParser", _StubParser)
    monkeypatch.setattr(script, "install_log_tee", lambda output_dir: None)

    assert script.parse_script_args(object) == "parsed"
    assert levels_at_group_start == [logging.INFO]


def test_gpu_test_main_raises_the_root_before_the_group_starts_and_the_body_runs(warning_root, monkeypatch, tmp_path):
    levels = {}

    def init_distributed():
        levels["group start"] = warning_root.level
        return 0, 1, 0

    monkeypatch.setattr(harness, "init_distributed", init_distributed)
    monkeypatch.setattr(harness, "setup_cache_dirs", lambda prefix, rank: (str(tmp_path), str(tmp_path)))
    for name in ("teardown_distributed", "cleanup_memory", "cleanup_dirs", "emit_result"):
        monkeypatch.setattr(harness, name, lambda *args, **kwargs: None)

    @harness.gpu_test_main(partial_state=False)
    def run(ctx):
        levels["body"] = warning_root.level
        return {"checks": {"ran": True}}

    with pytest.raises(SystemExit) as exited:
        run()

    assert exited.value.code == 0
    assert levels == {"group start": logging.INFO, "body": logging.INFO}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
