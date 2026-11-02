"""First-fit FLA autotuning for the tests that gate an fla-backed model against a reference.

fla's autotuners pick a config per key on timing, so on a cold cache the pick is noise, and two picks are each
fp32-correct but round differently. Two runs launched on different keys then round apart (a whole batch against its
microbatches: the autotuners key on the batch size, ``chunk_local_cumsum_vector_kernel`` among them), and two runs on
one key whose inputs round differently land as far apart as the launch's pick makes them (the gated delta rule's decay
gradient, which reaches ``A_log``).
"""

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
    """Run every ``fla`` kernel in this process on the first of its autotune configs that launches.

    Each autotuner keeps its config list and still launches every config once per key, so one the device cannot
    launch scores ``inf`` as in stock tuning (``chunk_gated_delta_rule_bwd_kernel_dhu_blockdim64``'s first config
    needs more shared memory than sm_90 and sm_10x have in fp32), and every other scores alike, so the pick is the
    first in list order. The disk cache is off while pinned: it would hand back a timed pick. Call it after the model
    is built, which is what imports fla.
    """
    pinned = [(tuner, tuner.cache, tuner.cache_results) for tuner in _loaded_fla_autotuners()]
    for tuner, _, _ in pinned:
        tuner.cache, tuner.cache_results, tuner.do_bench = {}, False, _launch_once

    def restore() -> None:
        for tuner, cache, cache_results in pinned:
            tuner.cache, tuner.cache_results = cache, cache_results
            del tuner.do_bench

    ctx.on_teardown(restore)
