"""Environments CPU suite support. The image ships no ``pytest-asyncio``, so ``pytest_pyfunc_call`` runs coroutine
tests through ``asyncio.run``. The registry is process-global, so a test registering a stub takes
``isolated_registry``."""

import asyncio
import inspect

import pytest

from halo.environments.registry import _environment_registry


@pytest.fixture
def isolated_registry():
    """Undo whatever env types the test registers, keeping other entries as the test left them."""
    before = dict(_environment_registry)
    yield
    for name in set(_environment_registry) - set(before):
        del _environment_registry[name]
    _environment_registry.update({k: v for k, v in before.items() if _environment_registry.get(k) is not v})


def pytest_pyfunc_call(pyfuncitem):
    test_fn = pyfuncitem.obj
    if inspect.iscoroutinefunction(test_fn):
        kwargs = {name: pyfuncitem.funcargs[name] for name in pyfuncitem._fixtureinfo.argnames}
        asyncio.run(test_fn(**kwargs))
        return True  # signal to pytest that we ran the test
    return None
