"""What a code-contests episode records of its runs and submissions under ``trajectory.info``, and what that
record says: accepted, lost to the backend, inconclusive, or failed on a program never run as submitted."""

from typing import Any

from halo.environments.tasks.coding.program_source import is_output_only_change

# Per graded submission, in order; ``episode/resubmission_improved`` reads it.
SUBMISSION_PASS_FRACS_KEY = "submission_pass_fracs"
IDENTICAL_RESUBMISSIONS_KEY = "identical_resubmissions"
# Normalized; one whose grade the backend lost is not kept. Private: it leaves the record with the grading payload.
GRADED_PROGRAMS_KEY = "_graded_programs"
# Per failed graded submission, whether a scratchpad run was still left when its turn began: a run spent
# in the same turn returned nothing the program could have been fixed from.
FAILED_SUBMISSION_RUNS_LEFT_KEY = "_failed_submission_runs_left"
TURN_RUNS_LEFT_KEY = "_turn_runs_left"
# Normalized, and how many had run as the turn began: a run in the submission's own turn informed nothing.
RUN_PROGRAMS_KEY = "_run_programs"
TURN_RUN_PROGRAMS_KEY = "_turn_run_programs"
# Per failed graded submission, whether its program had run as submitted (:func:`has_run_as_submitted`), and the
# public mark of a failure that had not while a run was left: the private lists leave with the grading payload.
FAILED_SUBMISSION_RAN_KEY = "_failed_submission_ran"
UNRUN_FAILED_SUBMISSION_KEY = "unrun_failed_submission"
# The episode fact a judge check stands exempt under: every failed graded submission came with no
# scratchpad run left, so none of them could have been tested first.
FAILED_WITH_NO_RUNS_LEFT = "failed_with_no_runs_left"


def has_run_as_submitted(program: str, info: dict[str, Any]) -> bool:
    """Comments, whitespace and dropped output statements aside (:func:`is_output_only_change`)."""
    runs = info.get(RUN_PROGRAMS_KEY, [])
    ran = runs[: info.get(TURN_RUN_PROGRAMS_KEY, len(runs))]
    return any(run == program or is_output_only_change(run, program) for run in ran)


def has_unrun_failed_submission(info: dict[str, Any]) -> bool:
    """A failed graded submission never run as submitted while a run was left."""
    runs_left = info.get(FAILED_SUBMISSION_RUNS_LEFT_KEY, [])
    ran = info.get(FAILED_SUBMISSION_RAN_KEY, [])
    return any(left and not run for left, run in zip(runs_left, ran, strict=True))


def has_graded_submission(info: dict[str, Any]) -> bool:
    return "submission_result" in info and info.get("tests_total", 0) > 0


def is_accepted(info: dict[str, Any]) -> bool:
    return has_graded_submission(info) and info["tests_total"] == info.get("tests_passed", 0)


def _judged_failures(info: dict[str, Any]) -> int:
    """Graded minus passed minus those lost to the backend; a grade with no ``tests_graded`` counts as fully graded."""
    graded = info.get("tests_graded", info.get("tests_total", 0))
    return graded - info.get("tests_passed", 0) - info.get("tests_infra_errors", 0)


def is_grading_infra_outage(info: dict[str, Any]) -> bool:
    """Tests lost to the backend, none passed and none failed."""
    return info.get("tests_infra_errors", 0) > 0 and info.get("tests_passed", 0) == 0 and _judged_failures(info) == 0


def is_grade_inconclusive(info: dict[str, Any]) -> bool:
    """Not every test passed, yet none of the judged ones failed: the rest went ungraded or were lost. All-or-nothing,
    such a grade would score a solution that may pass every test 0."""
    return (
        info.get("tests_passed", 0) < info.get("tests_total", 0)
        and _judged_failures(info) == 0
        and not is_grading_infra_outage(info)
    )
