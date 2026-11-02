"""CPU tests for the same-device fast path on Liger's launch context (``pin_liger_device_context``).

Liger 0.8.4 enters ``torch.cuda.device`` around every RMSNorm, SwiGLU, GeGLU and fused CE+TVD launch. The pin
rebinds ``device_context`` in every Liger ops module that imported the helper to one that skips the switch for a
tensor already on the current device and enters Liger's own context otherwise. The modules expected here are read
off Liger's source, not off the scan under test. On another Liger release, or one without distribution metadata,
nothing is rebound and a warning says so once; on the verified release a rebound helper and a foreign rebinding must
raise. The kernels and the cross-device launch are checked on GPU (``tests/gpu/kernels/test_liger_launch_context.py``).

    python tests/cpu/model_loading/test_liger_launch_context.py
"""

import ast
import contextlib
import functools
import importlib
import importlib.util
import logging
import sys
from importlib.metadata import PackageNotFoundError
from pathlib import Path

import liger_kernel.ops
import pytest
import torch
from liger_kernel.ops import utils as liger_ops_utils
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from halo.model_loading.liger import launch_context
from halo.model_loading.liger.dispatch import sanitize_liger_config
from halo.model_loading.liger.launch_context import (
    VERIFIED_LIGER_VERSION,
    pin_liger_device_context,
    same_device_context,
)

HELPER = liger_ops_utils.device_context


def _modules_importing_the_helper() -> frozenset[str]:
    """Every ``liger_kernel.ops`` module whose source has ``from liger_kernel.ops.utils import device_context``."""
    root = Path(liger_kernel.ops.__file__).parent
    names = set()
    for path in root.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                isinstance(node, ast.ImportFrom)
                and node.module == liger_ops_utils.__name__
                and any(alias.name == "device_context" and alias.asname is None for alias in node.names)
            ):
                relative = path.relative_to(root).with_suffix("")
                names.add(".".join((liger_kernel.ops.__name__, *relative.parts)))
    return frozenset(names)


IMPORTERS = _modules_importing_the_helper()


@pytest.fixture(autouse=True)
def unpinned(monkeypatch):
    """Every importer back on Liger's helper and no release warned yet, restored at teardown so neither the pin nor
    the warn-once state leaks across tests."""
    for name in IMPORTERS:
        monkeypatch.setattr(importlib.import_module(name), "device_context", HELPER)
    monkeypatch.setattr(launch_context, "_warned_versions", set())


@pytest.fixture
def logged_warnings(caplog):
    """The messages the launch-context module logs at WARNING or above."""
    caplog.set_level(logging.WARNING, logger=launch_context.logger.name)
    return lambda: [record.getMessage() for record in caplog.records if record.name == launch_context.logger.name]


@pytest.fixture(params=["0.8.5", "0.8.4.post1", "0.8.4+local", "0.8.40"])
def another_release(request, monkeypatch):
    """An installed liger-kernel other than the verified release; three of the spellings start with it."""
    monkeypatch.setattr(launch_context, "version", lambda package: request.param)
    return request.param


def _fresh_launch_context():
    """A new copy of the launch-context module, which captures whatever Liger's utils binds now."""
    spec = importlib.util.find_spec(launch_context.__name__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def without_the_helper(monkeypatch):
    """A fresh copy of the launch-context module imported while Liger's utils has no ``device_context``."""
    monkeypatch.delattr(liger_ops_utils, "device_context")
    return _fresh_launch_context()


@pytest.fixture
def current_device():
    return 0


@pytest.fixture
def recorded_switches(monkeypatch, current_device):
    """Each device ``torch.cuda.device`` is entered with, under a stand-in whose current device is ``current_device``."""
    entered = []

    class RecordingDevice:
        def __init__(self, device):
            self.device = device

        def __enter__(self):
            entered.append(self.device)

        def __exit__(self, *exc):
            return False

    monkeypatch.setattr(torch.cuda, "device", RecordingDevice)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: current_device)
    return entered


def test_the_oracle_names_the_measured_kernels():
    expected = {f"liger_kernel.ops.{op}" for op in ("rms_norm", "swiglu", "geglu", "fused_ce_tvd")}
    assert expected <= IMPORTERS


def test_every_importer_is_rebound():
    pinned = pin_liger_device_context()
    assert set(pinned) == IMPORTERS
    for name in IMPORTERS:
        assert sys.modules[name].device_context is same_device_context, name


def test_the_helper_itself_is_left_alone():
    pin_liger_device_context()
    assert liger_ops_utils.device_context is HELPER


def test_the_pin_is_idempotent():
    first = pin_liger_device_context()
    assert pin_liger_device_context() == first
    for name in IMPORTERS:
        assert sys.modules[name].device_context is same_device_context, name


def test_sanitize_pins_the_launch_context(logged_warnings):
    sanitize_liger_config({}, needs_ep_layers=False, model_config=CONFIG_MAPPING["qwen3"]())
    for name in IMPORTERS:
        assert sys.modules[name].device_context is same_device_context, name
    assert logged_warnings() == []


def test_the_verified_version_is_installed():
    assert launch_context.version("liger-kernel") == VERIFIED_LIGER_VERSION


def test_another_liger_release_keeps_liger_context(another_release, logged_warnings):
    assert pin_liger_device_context() == ()
    for name in IMPORTERS:
        assert sys.modules[name].device_context is HELPER, name
    [message] = logged_warnings()
    assert f"liger-kernel {another_release} is installed" in message
    assert "Liger's own launch context stays" in message


def test_another_liger_release_warns_once_across_both_sites(another_release, logged_warnings):
    pin_liger_device_context()
    config = sanitize_liger_config({}, needs_ep_layers=False, model_config=CONFIG_MAPPING["qwen3"]())
    assert config == {}
    pin_liger_device_context()
    assert len(logged_warnings()) == 1
    for name in IMPORTERS:
        assert sys.modules[name].device_context is HELPER, name


def test_another_liger_release_warns_on_the_main_process_only(another_release, logged_warnings, monkeypatch):
    monkeypatch.setattr(launch_context, "is_global_main_process", lambda: False)
    assert pin_liger_device_context() == ()
    assert logged_warnings() == []


def test_a_liger_without_metadata_keeps_liger_context(monkeypatch, logged_warnings):
    def missing(package):
        raise PackageNotFoundError(package)

    monkeypatch.setattr(launch_context, "version", missing)
    assert pin_liger_device_context() == ()
    for name in IMPORTERS:
        assert sys.modules[name].device_context is HELPER, name
    [message] = logged_warnings()
    assert "liger-kernel has no distribution metadata" in message


def test_another_liger_release_leaves_foreign_bindings_alone(another_release, monkeypatch):
    @contextlib.contextmanager
    def device_context(device):
        yield

    monkeypatch.setattr(liger_ops_utils, "device_context", device_context)
    rms_norm = sys.modules["liger_kernel.ops.rms_norm"]
    monkeypatch.setattr(rms_norm, "device_context", device_context)
    assert pin_liger_device_context() == ()
    assert liger_ops_utils.device_context is device_context
    assert rms_norm.device_context is device_context


def test_a_release_without_the_helper_imports_and_keeps_liger_context(
    without_the_helper, monkeypatch, logged_warnings
):
    release = "0.8.5"
    monkeypatch.setattr(without_the_helper, "version", lambda package: release)
    assert without_the_helper.pin_liger_device_context() == ()
    [message] = logged_warnings()
    assert f"liger-kernel {release} is installed" in message
    for name in IMPORTERS:
        assert sys.modules[name].device_context is HELPER, name


def test_the_verified_release_without_the_helper_raises(without_the_helper):
    with pytest.raises(RuntimeError, match="not Liger's own helper"):
        without_the_helper.pin_liger_device_context()
    for name in IMPORTERS:
        assert sys.modules[name].device_context is HELPER, name


def test_a_rebound_helper_raises(monkeypatch):
    @contextlib.contextmanager
    def device_context(device):
        yield

    monkeypatch.setattr(liger_ops_utils, "device_context", device_context)
    with pytest.raises(RuntimeError, match="not Liger's own helper"):
        pin_liger_device_context()


@pytest.mark.parametrize(
    ("module", "qualname"),
    [(__name__, "device_context"), (liger_ops_utils.__name__, "patched.device_context")],
    ids=["foreign-module", "foreign-qualname"],
)
def test_a_helper_rebound_before_import_raises(monkeypatch, module, qualname):
    """Rebound before the toolkit and every importer captured it, the helper passes the identity check, so the
    names alone must refuse it."""

    @contextlib.contextmanager
    def device_context(device):
        yield

    device_context.__module__ = module
    device_context.__qualname__ = qualname
    monkeypatch.setattr(liger_ops_utils, "device_context", device_context)
    for name in IMPORTERS:
        monkeypatch.setattr(sys.modules[name], "device_context", device_context)
    fresh = _fresh_launch_context()
    with pytest.raises(RuntimeError, match="not Liger's own helper"):
        fresh.pin_liger_device_context()
    for name in IMPORTERS:
        assert sys.modules[name].device_context is device_context, name


def test_a_wrapper_with_liger_names_raises(monkeypatch):
    @functools.wraps(HELPER)
    def wrapper(device):
        return HELPER(device)

    assert (wrapper.__module__, wrapper.__qualname__) == (HELPER.__module__, HELPER.__qualname__)
    monkeypatch.setattr(liger_ops_utils, "device_context", wrapper)
    with pytest.raises(RuntimeError, match="not Liger's own helper"):
        pin_liger_device_context()


def test_a_foreign_rebinding_raises(monkeypatch):
    rms_norm = sys.modules["liger_kernel.ops.rms_norm"]
    monkeypatch.setattr(rms_norm, "device_context", lambda device: contextlib.nullcontext())
    with pytest.raises(RuntimeError, match=r"liger_kernel\.ops\.rms_norm\.device_context is"):
        pin_liger_device_context()


def test_no_importer_raises(monkeypatch):
    for name in IMPORTERS:
        monkeypatch.delattr(sys.modules[name], "device_context")
    with pytest.raises(RuntimeError, match="No liger_kernel.ops module imports"):
        pin_liger_device_context()


@pytest.mark.parametrize("current_device", [0, 1])
def test_the_current_device_skips_the_switch(recorded_switches, current_device):
    with same_device_context(torch.device("cuda", current_device)):
        pass
    assert recorded_switches == []


@pytest.mark.parametrize("current_device", [1])
def test_device_zero_switches_off_another_current_device(recorded_switches):
    with same_device_context(torch.device("cuda", 0)):
        pass
    assert recorded_switches == [torch.device("cuda", 0)]


@pytest.mark.parametrize(
    "device",
    [torch.device("cuda", 1), torch.device("cuda"), "cuda:0", 0],
    ids=["other-gpu", "no-index", "string", "int"],
)
def test_any_other_cuda_device_enters_liger_context(recorded_switches, device):
    with same_device_context(device):
        pass
    assert recorded_switches == [torch.device(device)]


def test_a_cpu_tensor_enters_liger_context(recorded_switches):
    context = same_device_context(torch.device("cpu"))
    assert isinstance(context, contextlib._GeneratorContextManager)
    assert context.func is HELPER.__wrapped__
    with context:
        pass
    assert recorded_switches == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
