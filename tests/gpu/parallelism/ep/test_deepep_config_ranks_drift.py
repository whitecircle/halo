#!/usr/bin/env python
"""``DEEPEP_V1_CONFIG_RANKS`` must equal the widths DeepEP V1 itself ships a tuned config for.

The toolkit table is a MIRROR of ``deep_ep.Buffer``'s two ``config_map`` dicts, kept so an
``ep_group_size`` V1 has no tuning for is refused by ``ParallelismConfig._validate_ep_buffer_backend``
at config time. Nothing else notices when a DeepEP bump moves that table:

* a width DROPPED upstream stays in ours, so the run passes the config gate and dies on DeepEP's own
  bare ``assert num_ranks in config_map`` at the first dispatch — after the whole multi-GPU load;
* a width ADDED upstream stays out of ours, so a legal topology is refused with a message listing
  widths that are no longer the real set.

Both directions are caught by DERIVING the accepted set from the installed extension (probe every
width up to :data:`PROBE_MAX_RANKS`) and comparing it to the toolkit's constant. Dispatch and combine
are probed separately: one toolkit table stands for both, so an upstream split between them would
leave that single constant wrong for one of the two paths.

The check is rank-identical table arithmetic; it needs a GPU only because ``deep_ep`` is a CUDA
extension.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/ep/test_deepep_config_ranks_drift.py

Requirements:
    - DeepEP installed (both prebuilt images ship it). Builds no model and no buffer.
"""

from src.distributed.expert_parallel.config import DEEPEP_V1_CONFIG_RANKS
from src.distributed.expert_parallel.extension import deep_ep
from tests.common.harness import gpu_test_main, record_check
from tests.common.utils import log

# Widths to probe. Well past the largest tuned entry (160) and past any EP group this toolkit can
# build, so an upstream addition lands inside the range rather than beyond it.
PROBE_MAX_RANKS = 1024


def accepted_widths(get_config) -> frozenset[int]:
    """Rank widths ``get_config`` returns a tuned ``Config`` for, probed one by one.

    DeepEP guards its lookup with a bare ``assert``, which ``python -O`` strips — a ``KeyError`` then
    takes its place. Both mean "no tuned config for this width".
    """
    accepted = set()
    for num_ranks in range(1, PROBE_MAX_RANKS + 1):
        try:
            config = get_config(num_ranks)
        except (AssertionError, KeyError):
            continue
        assert config is not None, f"DeepEP returned no config for {num_ranks} ranks"
        accepted.add(num_ranks)
    return frozenset(accepted)


def assert_table_matches(name: str, upstream: frozenset[int]) -> None:
    """The toolkit constant must be exactly the widths ``Buffer.<name>`` accepts."""
    missing = sorted(upstream - DEEPEP_V1_CONFIG_RANKS)
    stale = sorted(DEEPEP_V1_CONFIG_RANKS - upstream)
    assert upstream == DEEPEP_V1_CONFIG_RANKS, (
        f"DEEPEP_V1_CONFIG_RANKS has drifted from deep_ep.Buffer.{name}: upstream-only widths "
        f"{missing} are refused at config time though they would run, and toolkit-only widths "
        f"{stale} pass the gate and then trip DeepEP's own assert at the first dispatch. Set "
        f"DEEPEP_V1_CONFIG_RANKS in src/distributed/expert_parallel/config.py to {sorted(upstream)}."
    )


def assert_one_table_answers_for_both(dispatch: frozenset[int], combine: frozenset[int]) -> None:
    """A single toolkit constant is only correct while DeepEP tunes both paths for the same widths."""
    assert dispatch == combine, (
        f"deep_ep now tunes dispatch and combine for different widths (dispatch-only "
        f"{sorted(dispatch - combine)}, combine-only {sorted(combine - dispatch)}); one "
        f"DEEPEP_V1_CONFIG_RANKS can no longer stand for both — split the gate per path."
    )


def run(ctx) -> dict:
    buffer_cls = deep_ep().Buffer
    dispatch = accepted_widths(buffer_cls.get_dispatch_config)
    combine = accepted_widths(buffer_cls.get_combine_config)
    log(f"deep_ep tuned widths: dispatch={sorted(dispatch)}, combine={sorted(combine)}")
    log(f"toolkit DEEPEP_V1_CONFIG_RANKS={sorted(DEEPEP_V1_CONFIG_RANKS)}")

    checks: dict[str, bool] = {}
    record_check(checks, "dispatch_table_matches", lambda: assert_table_matches("get_dispatch_config", dispatch))
    record_check(checks, "combine_table_matches", lambda: assert_table_matches("get_combine_config", combine))
    record_check(checks, "dispatch_and_combine_agree", lambda: assert_one_table_answers_for_both(dispatch, combine))
    return {"checks": checks}


main = gpu_test_main(min_world_size=1, prefix="deepep_config_ranks_drift", partial_state=False)(run)

if __name__ == "__main__":
    main()
