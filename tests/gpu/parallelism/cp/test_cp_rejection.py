#!/usr/bin/env python
"""
Test that trainers correctly declare CP/EP/TP support via class attributes.

Validates the _supports_cp, _supports_ep, and _supports_tp class attributes
on all distributed trainers. Trainers that don't support a parallelism mode
should reject it at initialization time; this test checks the class-level
declarations that drive that rejection (``TRAINER_SUPPORT_MAP``) and, for the
CP-rejecting trainers, that the real validation path still raises.

No model loading is required -- these are pure class attribute inspections.

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/parallelism/cp/test_cp_rejection.py
"""

import traceback

from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.distillation.teacher_distillation import DistributedDistillationTrainer
from src.trainers.embedding.trainer import EmbeddingTrainer
from src.trainers.grpo.offline import OfflineGRPOTrainer
from src.trainers.mixins.validation import ParallelismValidationMixin
from src.trainers.preference.dpo import DistributedDPOTrainer
from src.trainers.preference.smpo import SmoothMarginPOTrainer
from src.trainers.reward.bradley_terry import DistributedRewardTrainer
from src.trainers.reward.classification import ClassificationTrainer
from src.trainers.sft import DistributedSFTTrainer
from tests.common.harness import gpu_test_main
from tests.common.utils import log

# Trainers that declare _supports_cp == False must reject a cp_size>1 config at
# validation time. We invoke the real validation path so the test fails if the
# ValueError stops firing — not merely if the class attribute is edited.
CP_REJECTING_TRAINERS = [DistributedDPOTrainer, OfflineGRPOTrainer]

# Test Definitions

# (TrainerClass, expected_cp, expected_ep, expected_tp)
TRAINER_SUPPORT_MAP = [
    (DistributedSFTTrainer, True, True, True),
    (SmoothMarginPOTrainer, True, True, True),
    (OfflineGRPOTrainer, False, True, True),
    (DistributedDPOTrainer, False, True, True),
    (DistributedRewardTrainer, False, True, True),
    (ClassificationTrainer, False, True, True),
    (DistributedDistillationTrainer, False, True, True),
    (EmbeddingTrainer, False, True, True),
]


def test_supports_cp() -> tuple[bool, list[str]]:
    """
    Verify _supports_cp attribute for each trainer class.

    Returns:
        (all_passed, list of failure messages)
    """
    failures = []

    for trainer_cls, expected_cp, _expected_ep, _expected_tp in TRAINER_SUPPORT_MAP:
        name = trainer_cls.__name__
        actual = getattr(trainer_cls, "_supports_cp", None)

        if actual is None:
            failures.append(f"{name}: _supports_cp attribute not found")
        elif actual != expected_cp:
            failures.append(f"{name}._supports_cp = {actual}, expected {expected_cp}")

    return len(failures) == 0, failures


def test_cp_rejection_fires() -> tuple[bool, list[str]]:
    """
    Verify the validation path actually raises ValueError on cp_size>1.

    This is the load-bearing check: a static class attribute can read False while
    the rejection is silently dead. We build a minimal stub bound to the real
    ``ParallelismValidationMixin._validate_parallelism_modes`` (no model load) with
    a cp_size=2 ParallelismConfig and assert a ValueError mentioning "Context
    Parallelism" fires for each non-CP trainer.

    Returns:
        (all_passed, list of failure messages)
    """
    failures = []

    cp_config = ParallelismConfig(cp_size=2)
    assert cp_config.is_cp_mode, "cp_size=2 should set is_cp_mode (test precondition)"

    for trainer_cls in CP_REJECTING_TRAINERS:
        name = trainer_cls.__name__

        # Sanity: the trainer must actually declare no CP support, else this test
        # would vacuously pass.
        if getattr(trainer_cls, "_supports_cp", None) is not False:
            failures.append(f"{name}: expected _supports_cp == False for a CP-rejecting trainer")
            continue

        # Minimal stub carrying exactly what _validate_parallelism_modes reads.
        stub = type(
            name,
            (ParallelismValidationMixin,),
            {"_supports_cp": False, "_supports_ep": True, "_supports_tp": True},
        )()
        stub.parallelism_config = cp_config

        try:
            stub._validate_parallelism_modes()
            failures.append(f"{name}: cp_size=2 did NOT raise (rejection is dead)")
        except ValueError as e:
            msg = str(e)
            if "Context Parallelism" not in msg:
                failures.append(f"{name}: raised ValueError without 'Context Parallelism': {msg!r}")
        except Exception as e:
            failures.append(f"{name}: raised {type(e).__name__} instead of ValueError: {e}")

    return len(failures) == 0, failures


def test_supports_ep() -> tuple[bool, list[str]]:
    """
    Verify _supports_ep is True for all trainers in our list.

    Returns:
        (all_passed, list of failure messages)
    """
    failures = []

    for trainer_cls, _expected_cp, expected_ep, _expected_tp in TRAINER_SUPPORT_MAP:
        name = trainer_cls.__name__
        actual = getattr(trainer_cls, "_supports_ep", None)

        if actual is None:
            failures.append(f"{name}: _supports_ep attribute not found")
        elif actual != expected_ep:
            failures.append(f"{name}._supports_ep = {actual}, expected {expected_ep}")

    return len(failures) == 0, failures


def test_supports_tp() -> tuple[bool, list[str]]:
    """
    Verify _supports_tp is True for all trainers in our list.

    Returns:
        (all_passed, list of failure messages)
    """
    failures = []

    for trainer_cls, _expected_cp, _expected_ep, expected_tp in TRAINER_SUPPORT_MAP:
        name = trainer_cls.__name__
        actual = getattr(trainer_cls, "_supports_tp", None)

        if actual is None:
            failures.append(f"{name}: _supports_tp attribute not found")
        elif actual != expected_tp:
            failures.append(f"{name}._supports_tp = {actual}, expected {expected_tp}")

    return len(failures) == 0, failures


# Test Runner

ALL_TESTS = [
    ("test_supports_cp", test_supports_cp),
    ("test_cp_rejection_fires", test_cp_rejection_fires),
    ("test_supports_ep", test_supports_ep),
    ("test_supports_tp", test_supports_tp),
]


def run(ctx):
    """Run all CP rejection tests."""
    log(f"\n{'=' * 70}")
    log("  Trainer CP/EP/TP Support Attribute Tests")
    log(f"  World size: {ctx.world_size}")
    log(f"  Trainers under test: {len(TRAINER_SUPPORT_MAP)}")
    log(f"{'=' * 70}\n")

    # Print the support matrix for reference
    log("  Trainer Support Matrix:")
    log(f"  {'Trainer':<38} {'CP':>4} {'EP':>4} {'TP':>4}")
    log(f"  {'-' * 38} {'-' * 4} {'-' * 4} {'-' * 4}")
    for trainer_cls, cp, ep, tp in TRAINER_SUPPORT_MAP:
        name = trainer_cls.__name__
        log(f"  {name:<38} {str(cp):>4} {str(ep):>4} {str(tp):>4}")
    log("")

    checks = {}
    for test_name, test_fn in ALL_TESTS:
        log(f"  [{test_name}] Running...")
        try:
            ok, failures = test_fn()
        except Exception as e:
            log(f"  [{test_name}] UNHANDLED EXCEPTION: {e}")
            if ctx.rank == 0:
                traceback.print_exc()
            ok = False
            failures = [f"Exception: {e}"]

        checks[test_name] = ok
        log(f"  [{test_name}] {'PASS' if ok else 'FAIL'}")
        if not ok:
            for f in failures:
                log(f"    - {f}")

    return {"checks": checks}


main = gpu_test_main(min_world_size=1, prefix="cp_rejection", partial_state=False)(run)

if __name__ == "__main__":
    main()
