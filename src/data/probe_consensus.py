"""Cross-rank agreement for the data probes the loading path branches on.

A leaf so both halves of the data package (:mod:`src.data.sources`, :mod:`src.data.pipeline`) can
reach it. Each probe reads THIS rank's view while the branch it decides runs coordinated work — one
NCCL barrier plus two store phases per dataset operation — so a verdict that differs by rank pairs a
barrier against a store wait and leaves the phase counters permanently out of step.
"""

import logging

from src.distributed.runtime import rank_consensus

# Stdlib logger, not the accelerate one: the disagreement below is reported BY the rank whose own
# probe lost, which is rarely rank 0 and would be silenced by a main-process-only logger.
logger = logging.getLogger(__name__)


def agree_probe_across_ranks(local: bool, subject, probe: str) -> bool:
    """All-reduce MAX consensus for a probe whose ``True`` verdict is the authoritative one.

    Every caller's ``True`` is the answer that must win: an S3 probe only errs toward ``False`` (a
    transient credential/throttling fault), an image declaration anywhere in a mixed corpus makes the
    whole run multimodal, and an emptied split on any rank is fatal for all of them. ``subject`` names
    what was probed in the disagreement warning. No-op (the local verdict) when not distributed.
    """
    agreed = rank_consensus(local)[1]
    if agreed and not local:
        logger.warning(
            f"{probe} disagreed across ranks for {subject}: this rank read False, another read True. "
            f"Using the agreed verdict so every rank takes the same data path and the coordinated "
            f"dataset operations stay in lock-step."
        )
    return agreed
