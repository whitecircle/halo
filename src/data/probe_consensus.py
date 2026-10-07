"""Cross-rank agreement for the data probes the loading path branches on.

Each probe reads the local rank's view while the branch it decides runs coordinated work (one NCCL
barrier plus two store phases per dataset operation), so a verdict that differs by rank pairs a
barrier against a store wait and leaves the phase counters out of step.
"""

import logging
from collections.abc import Callable

from src.distributed.filesystem import store_join_recorded_failure
from src.distributed.runtime import fs_aware_load_rank, rank_consensus

# Stdlib logger, not the accelerate one: the disagreement below is reported by the rank whose own
# probe lost, which is rarely rank 0 and would be silenced by a main-process-only logger.
logger = logging.getLogger(__name__)


def agree_probe_across_ranks(local: bool, subject, probe: str) -> bool:
    """All-reduce MAX consensus for a probe whose ``True`` verdict is the authoritative one.

    ``True`` wins for every caller: an S3 probe only errs toward ``False`` (a transient
    credential/throttling fault), an image declaration anywhere in a mixed corpus makes the whole run
    multimodal, and an emptied split on any rank is fatal for all of them. ``subject`` names what was
    probed in the disagreement warning. Returns the local verdict when not distributed.
    """
    agreed = rank_consensus(local)[1]
    if agreed and not local:
        logger.warning(
            f"{probe} disagreed across ranks for {subject}: this rank read False, another read True. "
            f"Using the agreed verdict so every rank takes the same data path and the coordinated "
            f"dataset operations stay in lock-step."
        )
    return agreed


def agree_input_probe_across_ranks(probe: Callable[[], bool], subject, probe_name: str) -> bool:
    """:func:`agree_probe_across_ranks` for a probe of the run's INPUT, run once per filesystem scope.

    Every rank of one input-filesystem scope reads the same answer — a Hub file, a path on the
    scope's volume, the scope's S3 control-file mirror — so only its load rank
    (:func:`~src.distributed.runtime.fs_aware_load_rank`) runs ``probe`` and the rest abstain with
    ``False``, which MAX cannot let win over a probing rank's ``True``. A Hub probe from every rank
    is otherwise one request per rank, a storm the Hub rate-limits at scale. COLLECTIVE: every rank
    joins the probing ranks' outcome over the store, so a probe that raises raises its cause on every
    rank, then runs the one all-reduce, probing or not.
    """
    probing = fs_aware_load_rank()
    local, failure = False, None
    if probing:
        try:
            local = probe()
        except BaseException as exc:  # joined below, so every rank raises with it
            failure = exc
    store_join_recorded_failure(f"input_probe/{probe_name}", failure, f"Probing {subject} ({probe_name})")
    if probing:
        return agree_probe_across_ranks(local, subject, probe_name)
    return rank_consensus(False)[1]
