"""Pytest support for the environments CPU suite.

Several environment / Ray-actor tests are ``async def`` (they exercise the
async reset/step API and the asyncio-based rollout manager). The image ships
no ``pytest-asyncio`` plugin, so without this hook pytest would collect those
coroutine tests and fail them with "async def functions are not natively
supported". This ``pytest_pyfunc_call`` hook runs any coroutine test through
``asyncio.run`` and reports the real result.

The environment registry is process-global module state. Tests that register a
stub factory take ``isolated_registry`` to drop it again, so a later test that
sweeps the registered types resolves real environments rather than the stubs.
The fixture is opt-in, not autouse: several test modules import the registry
lazily inside a test body, and a blanket teardown would roll back the built-in
registrations that import performs.
"""

import asyncio
import inspect

import pytest


@pytest.fixture
def isolated_registry():
    """Undo whatever env types the test registers, keeping other entries as the test left them.

    The registry is imported here rather than at module scope: a conftest import runs before every
    test in the tree, and pulling the environments package in that early reorders module
    initialisation for unrelated suites.
    """
    from src.environments.registry import _ENVIRONMENT_REGISTRY

    before = dict(_ENVIRONMENT_REGISTRY)
    yield
    for name in set(_ENVIRONMENT_REGISTRY) - set(before):
        del _ENVIRONMENT_REGISTRY[name]
    _ENVIRONMENT_REGISTRY.update({k: v for k, v in before.items() if _ENVIRONMENT_REGISTRY.get(k) is not v})


def pytest_pyfunc_call(pyfuncitem):
    test_fn = pyfuncitem.obj
    if inspect.iscoroutinefunction(test_fn):
        # Only pass through the fixture-resolved args the test actually declares.
        kwargs = {name: pyfuncitem.funcargs[name] for name in pyfuncitem._fixtureinfo.argnames}
        asyncio.run(test_fn(**kwargs))
        return True  # signal to pytest that we ran the test
    return None
