#!/usr/bin/env python
"""``DEEPEP_V1_CONFIG_RANKS`` equals the widths DeepEP V1 ships a tuned config for, derived from the installed
extension.

A width dropped upstream passes the config gate and dies on DeepEP's bare assert at the first dispatch, after the whole
load; one added is refused wrongly. Dispatch and combine are probed apart, since one table stands for both.
"""

from halo.distributed.expert_parallel.deepep.backends import deep_ep
from halo.distributed.expert_parallel.deepep.capacity import DEEPEP_V1_CONFIG_RANKS
from tests.common.harness import gpu_test_main, record_check
from tests.common.rank_log import log

# Past the largest tuned entry (160) and any EP group this toolkit can build, so an upstream addition lands inside the
# range.
PROBE_MAX_RANKS = 1024


def accepted_widths(get_config) -> frozenset[int]:
    """DeepEP guards its lookup with a bare ``assert``, which ``python -O`` strips for a ``KeyError``; both mean no
    tuned config."""
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
    missing = sorted(upstream - DEEPEP_V1_CONFIG_RANKS)
    stale = sorted(DEEPEP_V1_CONFIG_RANKS - upstream)
    assert upstream == DEEPEP_V1_CONFIG_RANKS, (
        f"DEEPEP_V1_CONFIG_RANKS has drifted from deep_ep.Buffer.{name}: upstream-only widths "
        f"{missing} are refused at config time though they would run, and toolkit-only widths "
        f"{stale} pass the gate and then trip DeepEP's own assert at the first dispatch. Set "
        f"DEEPEP_V1_CONFIG_RANKS in halo/distributed/expert_parallel/config.py to {sorted(upstream)}."
    )


def assert_one_table_answers_for_both(dispatch: frozenset[int], combine: frozenset[int]) -> None:
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
