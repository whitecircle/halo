"""First-fit FLA autotuning for tests gating an fla-backed model against a reference. fla's autotuners pick per key on
timing, so a cold-cache pick is noise, and two fp32-correct picks round differently."""

import sys

from triton.runtime.autotuner import Autotuner
from triton.runtime.jit import KernelInterface

from halo.kernels.fla_launch import triton_autotuner


def _loaded_fla_autotuners() -> list[Autotuner]:
    """Every Triton autotuner behind a kernel of the ``fla`` modules this process has imported."""
    found: dict[int, Autotuner] = {}
    for name, module in list(sys.modules.items()):
        if module is None or not name.startswith("fla."):
            continue
        for value in vars(module).values():
            if isinstance(value, KernelInterface) and (tuner := triton_autotuner(value)) is not None:
                found[id(tuner)] = tuner
    return list(found.values())


def _launch_once(kernel_call, quantiles) -> list[float]:
    """An autotune benchmarker that launches a config once and scores every config that launches alike."""
    kernel_call()
    return [0.0] * len(quantiles)


def pin_fla_autotune_first_fit(ctx) -> None:
    """Run every ``fla`` kernel on the first autotune config that launches. Call after the model is built, which is
    what imports fla. A config the device cannot launch still scores ``inf``. The disk cache is off while pinned: it
    would hand back a timed pick."""
    pinned = [(tuner, tuner.cache, tuner.cache_results) for tuner in _loaded_fla_autotuners()]
    for tuner, _, _ in pinned:
        tuner.cache, tuner.cache_results, tuner.do_bench = {}, False, _launch_once

    def restore() -> None:
        for tuner, cache, cache_results in pinned:
            tuner.cache, tuner.cache_results = cache, cache_results
            del tuner.do_bench

    ctx.on_teardown(restore)
