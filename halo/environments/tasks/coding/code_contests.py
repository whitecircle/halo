"""Competitive-programming environment with hidden-test grading (Codeforces, APPS): the model tests programs in a
scratchpad and submits through ``submit_solution``, which grades 1 when the submission passes every hidden test
and 0 otherwise, priced by the reward's environment term. The run fixes one language, or lists several and
lets the model pick one per program."""

import json
import logging
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from halo.config_validation import require_finite, require_int
from halo.environments.episode_keys import (
    ANSWER_KEY,
    EPISODE_INVALID_KEY,
    EPISODE_INVALID_REASON_KEY,
    EPISODE_SLICES_KEY,
    SANDBOX_FAULT_KEY,
    SOLVE_RATE_KEY,
)
from halo.environments.episode_scoring import EpisodeGrade
from halo.environments.messages import TRUNCATION_MARKER, StepOutcome, Trajectory
from halo.environments.protocols.native_tool_use import NativeToolUseEnvironment
from halo.environments.sandbox.base import SANDBOX_DEFAULT_TIMEOUT, SandboxExecutor
from halo.environments.sandbox.factory import build_sandbox, warn_if_unisolated
from halo.environments.sandbox.languages import LanguageSpec, require_language
from halo.environments.sandbox.repl import format_sandbox_repl_output
from halo.environments.tasks.coding.code_contests_prompts import (
    AUDIT_CHECKS,
    IDENTICAL_RESUBMISSION_REPLY,
    MISLABELLED_LANGUAGE_REPLY,
    NO_STDIN_NOTE,
    SCRATCHPAD_TIME_LIMIT_NOTE,
    SUBMISSION_AFTER_ACCEPT_REPLY,
    SUBMIT_TOOL,
    UNSTATED_TIME_LIMIT_NOTE,
    build_contest_tools,
    default_system_prompt,
)
from halo.environments.tasks.coding.grading import DEFAULT_MAX_OUTPUT_SIZE, GradeResult, GradingSpec, grade_solution
from halo.environments.tasks.coding.program_source import find_evident_language, host_build_error, normalized_program
from halo.environments.tasks.coding.submission_record import (
    FAILED_SUBMISSION_RAN_KEY,
    FAILED_SUBMISSION_RUNS_LEFT_KEY,
    FAILED_WITH_NO_RUNS_LEFT,
    GRADED_PROGRAMS_KEY,
    IDENTICAL_RESUBMISSIONS_KEY,
    RUN_PROGRAMS_KEY,
    SUBMISSION_PASS_FRACS_KEY,
    TURN_RUN_PROGRAMS_KEY,
    TURN_RUNS_LEFT_KEY,
    UNRUN_FAILED_SUBMISSION_KEY,
    has_graded_submission,
    has_run_as_submitted,
    has_unrun_failed_submission,
    is_accepted,
    is_grade_inconclusive,
    is_grading_infra_outage,
)
from halo.environments.tasks.coding.verdicts import VERDICT_DETAIL_OUTCOME, find_stated_float_tolerance
from halo.environments.tool_accounting import EPISODE_TOOL_BUDGETS_KEY, MAGNITUDE_WHY
from halo.environments.tools.definitions import (
    NativeToolCall,
    NativeToolRegistry,
    NativeToolResult,
    ToolArgumentError,
    ToolCallRefused,
)

logger = logging.getLogger(__name__)

DEFAULT_REASONING_EFFORT = "medium"

# The knobs an evaluation protocol may pin, at the value a run takes when neither its config nor its
# protocol sets one.
EVAL_PROTOCOL_KNOB_DEFAULTS: dict[str, int] = {"max_submissions": 2, "max_test_calls": 5}
# Evaluation protocol -> the knobs it pins. ``harness`` pins none: the configured budgets stand, the
# agentic loop the environment trains, scored as attempts-until-accept. ``leaderboard`` grades one
# program per sample with no scratchpad, the setting a benchmark's pass@k is measured in.
EVAL_PROTOCOLS: dict[str, dict[str, int]] = {
    "harness": {},
    "leaderboard": {"max_submissions": 1, "max_test_calls": 0},
}
DEFAULT_EVAL_PROTOCOL = "harness"


def _require_eval_protocol_pins(eval_protocol: str) -> dict[str, int]:
    """The knobs ``eval_protocol`` pins; an unknown protocol raises."""
    if eval_protocol not in EVAL_PROTOCOLS:
        raise ValueError(f"eval_protocol must be one of {sorted(EVAL_PROTOCOLS)}, got {eval_protocol!r}")
    return EVAL_PROTOCOLS[eval_protocol]


def without_eval_protocol_pins(config: dict[str, Any], eval_protocol: str, source: str) -> dict[str, Any]:
    """``config`` without the knobs ``eval_protocol`` pins, each dropped one logged. For a contract
    written under another protocol (a training config's env options, an effort profile's budgets):
    its values give way to the pins instead of contradicting them."""
    pins = _require_eval_protocol_pins(eval_protocol)
    dropped = sorted(pins.keys() & config.keys())
    if dropped:
        logger.info("eval_protocol %r pins %s; the values %s sets give way", eval_protocol, dropped, source)
    return {key: value for key, value in config.items() if key not in pins}


class CodeContestsEnvironment(NativeToolUseEnvironment):
    """Competitive-programming environment with hidden-test grading (Codeforces, code_contests, APPS).

    ``language`` (``python``/``cpp``/``c``, or a list) drives the scratchpad and grading through one SandboxExecutor;
    with a list the model names each program's language. Per-problem ``checker`` / ``time_limit`` from the ``answer``
    payload override the ``output_comparison`` default, and ``eval_protocol`` (:data:`EVAL_PROTOCOLS`) pins its knobs.
    """

    # Agentic solvers iterate test→fix→submit, which the protocol's generic budget cuts mid-loop.
    DEFAULT_MAX_TURNS = 15

    SHAPING_COMPONENTS = ("submission", "resubmission", "unrun_submission")

    EPISODE_FACTS = (FAILED_WITH_NO_RUNS_LEFT,)

    # The ``answer`` payload IS the hidden test set: without it every episode grades against zero
    # tests, grading 0 whatever it submitted.
    requires_answer = True

    # Per-tool shaping off so correctness dominates; configs may re-enable small values.
    DEFAULT_TOOL_SUCCESS_REWARD = 0.0
    DEFAULT_TOOL_ERROR_PENALTY = 0.0

    # The protocol's empty-turn nudge offers a final answer, which here ends the episode ungraded: every
    # nudge here names the one graded channel instead. Same rule as the protocol's: the fact and the
    # action, never an ask for shorter reasoning.
    LENGTH_CUTOFF_NUDGE = (
        "Your previous turn was cut off before you made a tool call, so nothing was recorded, and only a "
        "solution sent with submit_solution is graded. Make your tool call now with the best solution you have."
    )
    LENGTH_CUTOFF_IN_CALL_NUDGE = (
        "Your previous turn reached its length limit while writing a tool call, so the call was not run and "
        "nothing was recorded, and only a solution sent with submit_solution is graded. Make the call again, "
        "keeping your reasoning out of the program's comments."
    )
    EMPTY_TURN_NUDGE = (
        "Your previous turn ended without a tool call (one written inside your reasoning is not run), so "
        "nothing was recorded, and only a solution sent with submit_solution is graded. Make your tool call "
        "now with the best solution you have."
    )

    # Effort level -> profile. ``thinking_tokens`` is the level's per-turn CoT budget; a config adds the
    # interaction budgets (``max_submissions``, ``max_test_calls``) so effort buys iteration too.
    REASONING_EFFORT_PROFILES = {
        "low": {"thinking_tokens": 4096},
        "medium": {"thinking_tokens": 8192},
        "high": {"thinking_tokens": 16384},
    }
    # The task's own profile keys over the base's: the per-episode interaction budgets.
    EFFORT_PROFILE_KEY_MINIMA = {"max_submissions": 1, "max_test_calls": 0}

    def __init__(
        self,
        timeout_per_test: float = SANDBOX_DEFAULT_TIMEOUT,
        max_output_size: int = DEFAULT_MAX_OUTPUT_SIZE,
        system_prompt: str | None = None,
        sandbox: SandboxExecutor | None = None,
        sandbox_backend: str | None = None,
        sandbox_url: str | None = None,
        language: str | Sequence[str] = "python",
        output_comparison: str = "exact",
        verdict_detail: str = VERDICT_DETAIL_OUTCOME,
        stop_on_first_failure: bool = False,
        max_time_limit: float = SANDBOX_DEFAULT_TIMEOUT,
        compiled_time_limit_scale: float = 1.0,
        max_grading_seconds: float | None = None,
        max_submissions: int | None = None,
        max_test_calls: int | None = None,
        submission_reward: float = 0.0,
        resubmission_penalty: float = 0.0,
        unrun_submission_penalty: float = 0.0,
        reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
        reasoning_effort_profiles: dict[str, dict[str, int | float | bool]] | None = None,
        eval_protocol: str = DEFAULT_EVAL_PROTOCOL,
        **kwargs,
    ):
        knobs = self._resolve_eval_protocol_knobs(
            eval_protocol, max_submissions=max_submissions, max_test_calls=max_test_calls
        )
        max_submissions, max_test_calls = knobs["max_submissions"], knobs["max_test_calls"]
        self.eval_protocol = eval_protocol
        specs = self._resolve_languages(language)
        for knob, minimum in self.EFFORT_PROFILE_KEY_MINIMA.items():
            require_int(type(self).__name__, ge=minimum, **{knob: knobs[knob]})
        require_finite(
            type(self).__name__,
            ge=0,
            why=MAGNITUDE_WHY,
            submission_reward=submission_reward,
            resubmission_penalty=resubmission_penalty,
            unrun_submission_penalty=unrun_submission_penalty,
        )
        self.languages = tuple(spec.name for spec in specs)
        self.language = self.languages[0]
        self.chooses_language = len(self.languages) > 1
        # Bootstraps a weak base that never submits; self-neutralizes within a GRPO group once all do.
        self.submission_reward = submission_reward
        # Each graded submission after the first probes the judge; free, probing would out-earn testing.
        self.resubmission_penalty = resubmission_penalty
        # Failure-only: the policy tests when it is unsure, and an outcome-only price would charge the testing.
        self.unrun_submission_penalty = unrun_submission_penalty
        self.max_submissions = max_submissions
        self.max_test_calls = max_test_calls
        self.sandbox = sandbox or build_sandbox(backend=sandbox_backend, url=sandbox_url)
        warn_if_unisolated(self.sandbox, type(self).__name__)
        # The one reader of these knobs, so every submission and the offline re-grade (``to_meta``) share it.
        self.grading_spec = GradingSpec(
            sandbox=self.sandbox,
            comparison=output_comparison,
            verdict_detail=verdict_detail,
            language=self.language,
            max_output_size=max_output_size,
            stop_on_first_failure=stop_on_first_failure,
            default_timeout=timeout_per_test,
            max_time_limit=max_time_limit,
            compiled_time_limit_scale=compiled_time_limit_scale,
            max_grading_seconds=max_grading_seconds,
        )
        chooses = self.chooses_language
        test_tool, submit_tool = build_contest_tools(
            specs,
            self.sandbox,
            run_test=self._run_test_in if chooses else self._run_test,
            submit=self._submit_in if chooses else self._submit,
        )
        self.test_tool_name = test_tool.name
        # What a veto judge listing ``checks: environment`` audits with, naming this run's scratchpad.
        self.JUDGE_CHECKS = tuple(
            replace(check, description=check.description.format(test_tool=test_tool.name)) for check in AUDIT_CHECKS
        )
        super().__init__(
            tool_registry=NativeToolRegistry().register(test_tool).register(submit_tool),
            system_prompt=system_prompt or default_system_prompt(specs, self.test_tool_name, self.grading_spec),
            reasoning_effort=reasoning_effort,
            reasoning_effort_profiles=reasoning_effort_profiles,
            **kwargs,
        )
        # After the base validated every key: a protocol pinning the interaction budgets fixes them at every level.
        self.reasoning_effort_profiles = {
            level: without_eval_protocol_pins(entry, eval_protocol, f"reasoning_effort_profiles[{level!r}]")
            for level, entry in self.reasoning_effort_profiles.items()
        }

    @staticmethod
    def _resolve_eval_protocol_knobs(eval_protocol: str, **configured: int | None) -> dict[str, int]:
        """Each :data:`EVAL_PROTOCOL_KNOB_DEFAULTS` knob under ``eval_protocol``: its pin, else the
        configured value, else the default. A configured value contradicting a pin raises — the run
        would carry the protocol's name without following it."""
        pins = _require_eval_protocol_pins(eval_protocol)
        conflicts = {k: v for k, v in configured.items() if v is not None and k in pins and v != pins[k]}
        if conflicts:
            raise ValueError(
                f"eval_protocol {eval_protocol!r} pins {pins}; the config contradicts it with {conflicts}"
            )
        return {
            knob: pins.get(knob, default if configured.get(knob) is None else configured[knob])
            for knob, default in EVAL_PROTOCOL_KNOB_DEFAULTS.items()
        }

    @staticmethod
    def _resolve_languages(language: str | Sequence[str]) -> list[LanguageSpec]:
        """The run's language set as specs, in the order given: one name fixes the language, a list
        lets the model choose per program. Empty, unknown or repeated names raise."""
        names = [language] if isinstance(language, str) else list(language)
        if not names:
            raise ValueError("language must name at least one language")
        specs: list[LanguageSpec] = []
        for name in names:
            spec = require_language(name)
            if any(spec.name == known.name for known in specs):
                raise ValueError(f"language lists {spec.name!r} twice")
            specs.append(spec)
        return specs

    def find_mislabel(self, code: str, language: str | None) -> str | None:
        """The listed language ``code``, sent as ``language``, is evidently written in instead
        (:func:`find_evident_language`), else ``None``, always so where the run fixes its language. Both tools
        refuse such a program unspent, and the offline re-grader skips it the same way."""
        if not self.chooses_language or language is None:
            return None
        return find_evident_language(code, language, self.languages)

    def _reset_single(
        self,
        prompt: str | list[dict[str, str]],
        context: dict[str, Any] | None = None,
    ) -> Trajectory:
        """Build the base trajectory, then store the problem's grading data via the hook and the float
        tolerance its statement states; a payload carrying no time limit gets the one each test runs under
        stated in the task message."""
        traj = super()._reset_single(prompt, context)
        self._store_problem_data(traj, context or {})
        statement = prompt if isinstance(prompt, str) else "\n".join(str(m.get("content") or "") for m in prompt)
        traj.info["_stated_tolerance"] = find_stated_float_tolerance(statement)
        if traj.info["_time_limit"] is None:
            limits = {name: self.grading_spec.time_limit_for(name, None) for name in self.languages}
            if len(set(limits.values())) == 1:
                limit_text = f"{limits[self.language]:g} s"
            else:
                limit_text = ", ".join(f"{limit:g} s in {name}" for name, limit in limits.items())
            traj.append_to_last_user(UNSTATED_TIME_LIMIT_NOTE.format(limits=limit_text))
        return traj

    def _store_problem_data(self, traj: Trajectory, context: dict[str, Any]) -> None:
        """Store the tests, optional checker and time limit the submission is graded against, from a bare list,
        ``{"test_cases": [...]}`` or Codeforces' ``{"tests": [...], "checker": ..., "time_limit": ...}``; one holding
        no tests raises. Kept in ``traj.info``, so concurrent episodes never clobber each other.
        """
        answer = self._parse_answer(context)
        if isinstance(answer, dict):
            test_cases = answer.get("tests") or answer.get("test_cases") or []
            checker = answer.get("checker")
            time_limit = answer.get("time_limit")
        else:
            test_cases, checker, time_limit = answer, None, None
        if not test_cases:
            raise ValueError(
                f"{ANSWER_KEY!r} holds no tests: expected a non-empty list, or a non-empty 'tests'/'test_cases'"
            )

        traj.info["_test_cases"] = test_cases
        traj.info["_checker"] = checker
        traj.info["_time_limit"] = float(time_limit) if time_limit else None
        traj.info["tests_passed"] = 0
        traj.info["tests_total"] = len(test_cases)

    @staticmethod
    def _parse_answer(context: dict[str, Any]) -> dict[str, Any] | list[Any]:
        """``context["answer"]`` as a dict or list, a JSON string decoded; any other payload fails the episode at
        reset, since graded against no tests it would score 0, indistinguishable from a wrong solution.
        """
        answer = context.get(ANSWER_KEY, {})
        if isinstance(answer, str):
            try:
                answer = json.loads(answer)
            except ValueError as exc:
                raise ValueError(f"unparseable {ANSWER_KEY!r} payload ({len(answer)} chars): {exc}") from exc
        if not isinstance(answer, (dict, list)):
            raise ValueError(f"{ANSWER_KEY!r} must be a dict or list of tests, got {type(answer).__name__}")
        return answer

    def _apply_effort_profile(
        self, traj: Trajectory, level: str | None, profile: dict[str, int | float | bool]
    ) -> None:
        """Stamp the episode's interaction budgets — the level's, else the class caps — as the per-tool
        caps the protocol enforces. The task message states none of them: the level the chat template
        states and the engine's caps are what control an episode, and a refused call is told so."""
        max_subs = profile.get("max_submissions", self.max_submissions)
        max_tests = profile.get("max_test_calls", self.max_test_calls)
        traj.info[EPISODE_TOOL_BUDGETS_KEY].update({self.test_tool_name: max_tests, SUBMIT_TOOL: max_subs})

    def _execute_tool_calls(
        self, tool_calls: list[NativeToolCall], trajectory: Trajectory
    ) -> tuple[list[NativeToolResult], float]:
        # Whether a scratchpad run is left as the turn's calls begin, and how many programs had run by then: what
        # a failed submission in the turn is judged by.
        info = trajectory.info
        info[TURN_RUNS_LEFT_KEY] = self._tool_budget_exhausted(trajectory, self.test_tool_name) is None
        info[TURN_RUN_PROGRAMS_KEY] = len(info.get(RUN_PROGRAMS_KEY, []))
        return super()._execute_tool_calls(tool_calls, trajectory)

    def _run_test_in(self, code: str, language: str, stdin: str = "") -> str:
        """The scratchpad handler when the run lets the model choose: ``language`` is required, so a
        call without it fails to bind and is refused unspent."""
        return self._run_test(code, language, stdin)

    def _run_test(self, code: str, language: str | None = None, stdin: str = "") -> str:
        """Run a scratchpad test in ``language`` (the run's, or the call's choice) on ``stdin`` under the per-test
        limit the episode's problem is graded at; a direct call outside an episode runs uncapped at the grading
        default.
        """
        language = self._call_language(language, self.test_tool_name)
        trajectory = self.active_trajectory()
        self._refuse_mislabelled(code, language, self.test_tool_name, trajectory)
        if trajectory is not None:
            self._note_language(trajectory, language)
        stated = trajectory.info.get("_time_limit") if trajectory is not None else None
        timeout = self.grading_spec.time_limit_for(language, stated)
        result = self.sandbox.run(code, stdin=stdin, timeout=timeout, language=language)
        if trajectory is not None:
            trajectory.info.setdefault(RUN_PROGRAMS_KEY, []).append(normalized_program(code, language))
        output = format_sandbox_repl_output(result, timeout)
        if result.timed_out:
            output += f" {SCRATCHPAD_TIME_LIMIT_NOTE}"
        notes = []
        # A run on no input (whitespace is none) says so, since neither the parse error of a program that reads
        # input nor output computed from nothing names the cause; a build failure ran nothing.
        if not stdin.strip() and not result.compile_failed and not host_build_error(code, language, self.sandbox):
            notes.append(NO_STDIN_NOTE)
        return self._fit_observation(output, notes)

    def _call_language(self, language: str | None, tool: str) -> str:
        """The language a program runs and is graded in: the run's one, or the call's ``language`` when the run lists
        several (the schema requires and enumerates it, so this guards only the direct-call path).
        """
        if not self.chooses_language:
            return self.language
        if language is None or language not in self.languages:
            raise ToolArgumentError(f"{tool}: language must be one of {', '.join(self.languages)}, got {language!r}")
        return language

    def _refuse_mislabelled(self, code: str, language: str, tool: str, trajectory: Trajectory | None) -> None:
        """Refuse a program :meth:`find_mislabel` names another language for unrun, as a tool error with the
        call returned to the budget and the turn flagged like any refusal: a free, trainable refusal is a
        wasted turn that every solve reinforces."""
        evident = self.find_mislabel(code, language)
        if evident is None:
            return
        if trajectory is not None:
            self._uncount_tool_call(trajectory, tool)
        verb, spent = ("run", "scratchpad run") if tool == self.test_tool_name else ("graded", "submission")
        raise ToolCallRefused(
            MISLABELLED_LANGUAGE_REPLY.format(verb=verb, evident=evident, language=language, spent=spent)
        )

    def _note_language(self, trajectory: Trajectory, language: str) -> None:
        """Record the language a program was written in: the episode's ``language`` slice (the last
        one used, which is the graded submission's once it submits) and how often it switched."""
        if not self.chooses_language:
            return
        slices = trajectory.info.setdefault(EPISODE_SLICES_KEY, {})
        previous = slices.get("language")
        if previous is not None and previous != language:
            trajectory.info["language_switches"] = trajectory.info.get("language_switches", 0) + 1
        slices["language"] = language

    def _fit_observation(self, output: str, notes: list[str]) -> str:
        """``output`` with ``notes`` on lines after it, the output cut when the whole would pass the
        protocol's observation cap (``max_observation_chars``), which cuts from the end, so the notes stay whole."""
        tail = "".join(f"\n{note}" for note in notes)
        cap = self.max_observation_chars
        if cap and len(output) + len(tail) > cap:
            # The marker counting every character of the output bounds the one a cut adds.
            marker = len(TRUNCATION_MARKER.format(dropped=len(output)))
            output = self._truncate_observation(output, max(1, cap - len(tail) - marker))
        return output + tail

    def _submit_in(self, code: str, language: str) -> str:
        """The submission handler when the run lets the model choose (``language`` required, as above)."""
        return self._submit(code, language)

    def _submit(self, code: str, language: str | None = None) -> str:
        """Grade a submission against the active episode's tests in ``language`` (the run's, or the
        call's choice). The per-episode cap is the protocol's (``max_submissions``); the count it keeps
        is incremented before this runs, so the first admitted call is submission one."""
        language = self._call_language(language, SUBMIT_TOOL)
        trajectory = self.active_trajectory()
        if trajectory is None:
            raise ValueError("submit_solution called outside an active episode.")
        info = trajectory.info
        if is_accepted(info):
            # A later call in the turn that submitted the accept: the episode ends on this turn, and
            # grading another program could only replace the solve.
            self._refund_tool_call(trajectory, SUBMIT_TOOL)
            return SUBMISSION_AFTER_ACCEPT_REPLY
        self._refuse_mislabelled(code, language, SUBMIT_TOOL, trajectory)
        program = normalized_program(code, language)
        self._refuse_identical_resubmission(program, trajectory)
        self._note_language(trajectory, language)

        if self._submissions(trajectory) == 1:
            # Did any scratchpad run precede the first submission (``episode/tested_before_submission``)?
            info["tested_before_submission"] = self._test_calls(trajectory) > 0
        grade = self._grade_submission(code, trajectory, language)
        info["submission_language"] = language
        info["tests_passed"] = grade.passed
        info["tests_total"] = grade.total
        info["tests_ran_ok"] = grade.ran_ok
        info["tests_infra_errors"] = grade.infra_errors
        info["tests_graded"] = grade.graded
        info["grading_budget_hit"] = grade.budget_hit
        info["submission_result"] = grade.details
        info.setdefault(SUBMISSION_PASS_FRACS_KEY, []).append(grade.passed / grade.total if grade.total else 0.0)
        if not (is_accepted(info) or is_grading_infra_outage(info) or is_grade_inconclusive(info)):
            info.setdefault(FAILED_SUBMISSION_RUNS_LEFT_KEY, []).append(info.get(TURN_RUNS_LEFT_KEY, True))
            info.setdefault(FAILED_SUBMISSION_RAN_KEY, []).append(has_run_as_submitted(program, info))
            info[UNRUN_FAILED_SUBMISSION_KEY] = has_unrun_failed_submission(info)
        # The graded artifact an external scorer reads (``_scoring_sample``); private, so it leaves
        # the record with the grading payload.
        info["_submitted_code"] = code
        if not grade.infra_errors:
            # A grade the backend lost says nothing about the program, so the same one may come back.
            info.setdefault(GRADED_PROGRAMS_KEY, []).append(program)
        return grade.details

    def _refuse_identical_resubmission(self, program: str, trajectory: Trajectory) -> None:
        """Refuse a normalized ``program`` identical to one this episode already graded unrun, the submission
        returned to the budget: it would draw the same verdict, so grading it only probes the judge."""
        if program not in trajectory.info.get(GRADED_PROGRAMS_KEY, []):
            return
        self._uncount_tool_call(trajectory, SUBMIT_TOOL)
        trajectory.info[IDENTICAL_RESUBMISSIONS_KEY] = trajectory.info.get(IDENTICAL_RESUBMISSIONS_KEY, 0) + 1
        raise ToolCallRefused(IDENTICAL_RESUBMISSION_REPLY)

    def _submissions(self, trajectory: Trajectory) -> int:
        """Graded-submission calls admitted so far (the protocol counts a call before its handler runs)."""
        return self._tool_calls_made(trajectory, SUBMIT_TOOL)

    def _test_calls(self, trajectory: Trajectory) -> int:
        """Scratchpad calls admitted so far."""
        return self._tool_calls_made(trajectory, self.test_tool_name)

    def _grade_submission(self, code: str, trajectory: Trajectory, language: str | None = None) -> GradeResult:
        """Grade ``code`` against the episode's tests → :class:`GradeResult`, in ``language`` (the run's
        default when unset)."""
        return grade_solution(
            code,
            trajectory.info.get("_test_cases", []),
            self.grading_spec,
            checker=trajectory.info.get("_checker"),
            time_limit=trajectory.info.get("_time_limit"),
            stated_tolerance=trajectory.info.get("_stated_tolerance"),
            language=language,
        )

    def _step_single(self, trajectory: Trajectory, action: str, context: dict[str, Any] | None = None) -> StepOutcome:
        """Native tool step, then end the episode once the submission budget is spent or a submission passed every
        hidden test, after which a resubmission could only lose the solve.
        """
        outcome = super()._step_single(trajectory, action, context)
        info = outcome.trajectory.info
        # A turn that also booked a sandbox fault ends on it, uncompleted, whatever it submitted.
        spent = self._tool_budget_exhausted(outcome.trajectory, SUBMIT_TOOL) is not None
        if not outcome.done and (spent or is_accepted(info)) and SANDBOX_FAULT_KEY not in info:
            info["completed"] = True
            return outcome._replace(done=True)
        return outcome

    def _grade_episode(self, trajectory: Trajectory, context: dict[str, Any] | None = None) -> EpisodeGrade:
        """The objective is 1 when the submitted solution passed every hidden test, else 0 (the judge's accept, what
        pass@1 counts); the shaping rungs bootstrap the tool-use loop and price probing the judge, and none pays on a
        grade lost to the backend or inconclusive (:func:`is_grade_inconclusive`), which leaves the GRPO group
        baseline.
        """
        info = trajectory.info
        tests_total = info.get("tests_total", 0)
        # A never-submitted episode is valid: its 0 is the policy's.
        submitted = has_graded_submission(info)
        infra_outage = submitted and is_grading_infra_outage(info)
        inconclusive = submitted and is_grade_inconclusive(info)
        graded_content = submitted and not (infra_outage or inconclusive)
        # ``setdefault``: a sandbox fault a tool call booked earlier keeps its own reason.
        if infra_outage:
            info[EPISODE_INVALID_KEY] = True
            info.setdefault(
                EPISODE_INVALID_REASON_KEY, "code grade lost to the sandbox backend: no test passed or failed"
            )
        elif inconclusive:
            passed = info.get("tests_passed", 0)
            ungraded = tests_total - info.get("tests_graded", tests_total)
            info[EPISODE_INVALID_KEY] = True
            info.setdefault(
                EPISODE_INVALID_REASON_KEY,
                f"code grade inconclusive: {passed} of {tests_total} tests passed and none failed "
                f"({ungraded} ungraded, {info.get('tests_infra_errors', 0)} lost to the sandbox backend)",
            )

        objective = 1.0 if graded_content and is_accepted(info) else 0.0
        # Gated on graded_content, not submission_count: _submit bumps the count before grading.
        submission = self.submission_reward if graded_content else 0.0
        resubmission = -self.resubmission_penalty * max(0, self._submissions(trajectory) - 1)
        unrun = -self.unrun_submission_penalty if not objective and info.get(UNRUN_FAILED_SUBMISSION_KEY) else 0.0
        return EpisodeGrade(
            objective, {"submission": submission, "resubmission": resubmission, "unrun_submission": unrun}
        )

    def _final_answer(self, trajectory: Trajectory) -> str | None:
        """What the episode delivered: the last submitted program as a fenced block, or nothing — a
        scorer then reads that no submission was made, never the tool-call turn as an answer."""
        code = trajectory.info.get("_submitted_code")
        if code is None:
            return None
        return f"```{trajectory.info['submission_language']}\n{code}\n```"

    def _scoring_reference(self, trajectory: Trajectory) -> None:
        """The hidden tests are the grader's payload, never a reference a judge reads."""
        return None

    def _episode_facts(self, trajectory: Trajectory) -> frozenset[str]:
        """:data:`FAILED_WITH_NO_RUNS_LEFT` when the episode failed graded submissions and every one of them
        came with the scratchpad budget spent."""
        runs_left = trajectory.info.get(FAILED_SUBMISSION_RUNS_LEFT_KEY, [])
        return frozenset({FAILED_WITH_NO_RUNS_LEFT}) if runs_left and not any(runs_left) else frozenset()

    def rollout_metrics(self, trajectory: Trajectory) -> dict[str, float]:
        """CodeContests diagnostics: task outcome, submission behavior, and the reward decomposition."""
        metrics = super().rollout_metrics(trajectory)
        info = trajectory.info
        tests_total = info.get("tests_total", 0)
        graded = has_graded_submission(info)
        if graded:
            metrics["outcome/test_pass_frac"] = info.get("tests_passed", 0) / tests_total
            metrics[SOLVE_RATE_KEY] = 1.0 if is_accepted(info) else 0.0
        else:
            metrics["outcome/test_pass_frac"] = 0.0
            metrics[SOLVE_RATE_KEY] = 0.0
        submissions = self._submissions(trajectory)
        metrics["episode/submission_rate"] = 1.0 if submissions > 0 else 0.0
        metrics["episode/test_calls"] = float(self._test_calls(trajectory))
        metrics["episode/identical_resubmissions"] = float(info.get(IDENTICAL_RESUBMISSIONS_KEY, 0))
        metrics["episode/unrun_failed_submission"] = 1.0 if info.get(UNRUN_FAILED_SUBMISSION_KEY) else 0.0
        if submissions > 0:
            # Mean over submitting episodes: the share that ran the scratchpad before submitting.
            metrics["episode/tested_before_submission"] = 1.0 if info.get("tested_before_submission") else 0.0
        if submissions > 1:
            # Over resubmitting episodes: the share of resubmissions that beat every earlier result.
            fracs = info.get(SUBMISSION_PASS_FRACS_KEY, [])
            improved = sum(1 for i in range(1, len(fracs)) if fracs[i] > max(fracs[:i]))
            metrics["episode/resubmission_improved"] = improved / (submissions - 1)
        if self.chooses_language:
            metrics["episode/language_switches"] = float(info.get("language_switches", 0))
        if graded:
            metrics["episode/grading_infra_outage"] = 1.0 if is_grading_infra_outage(info) else 0.0
            metrics["episode/grade_inconclusive"] = 1.0 if is_grade_inconclusive(info) else 0.0
            # Partial grading is invisible in the pass fraction, which keeps the full pool as its
            # denominator: an ungraded remainder reads exactly like a wrong solution.
            metrics["episode/tests_graded_frac"] = info.get("tests_graded", 0) / tests_total
            metrics["episode/grading_budget_hit"] = 1.0 if info.get("grading_budget_hit") else 0.0
        return metrics
