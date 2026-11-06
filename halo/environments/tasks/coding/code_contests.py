"""Competitive-programming environment with hidden-test grading: the model tests programs in a scratchpad and
submits through ``submit_solution``, which grades 1 when the submission passes every hidden test and 0 otherwise."""

import json
import logging
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from halo.config_validation import require_finite, require_int
from halo.environments.episode_keys import (
    ANSWER_KEY,
    COMPLETED_KEY,
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
    IDENTICAL_RESUBMISSION_NOTE,
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

# Used when neither a run's config nor its protocol sets the knob.
EVAL_PROTOCOL_KNOB_DEFAULTS: dict[str, int] = {"max_submissions": 2, "max_test_calls": 5}
# ``harness`` pins none: the agentic loop the environment trains, scored as attempts-until-accept. ``leaderboard``
# grades one program per sample with no scratchpad, the setting a benchmark's pass@k is measured in.
EVAL_PROTOCOLS: dict[str, dict[str, int]] = {
    "harness": {},
    "leaderboard": {"max_submissions": 1, "max_test_calls": 0},
}
DEFAULT_EVAL_PROTOCOL = "harness"


def _require_eval_protocol_pins(eval_protocol: str) -> dict[str, int]:
    if eval_protocol not in EVAL_PROTOCOLS:
        raise ValueError(f"eval_protocol must be one of {sorted(EVAL_PROTOCOLS)}, got {eval_protocol!r}")
    return EVAL_PROTOCOLS[eval_protocol]


def without_eval_protocol_pins(config: dict[str, Any], eval_protocol: str, source: str) -> dict[str, Any]:
    """For a contract written under another protocol (a training config's env options, an effort profile's budgets):
    its values give way to the pins instead of contradicting them."""
    pins = _require_eval_protocol_pins(eval_protocol)
    dropped = sorted(pins.keys() & config.keys())
    if dropped:
        logger.info("eval_protocol %r pins %s; the values %s sets give way", eval_protocol, dropped, source)
    return {key: value for key, value in config.items() if key not in pins}


class CodeContestsEnvironment(NativeToolUseEnvironment):
    """Competitive-programming environment with hidden-test grading (Codeforces, code_contests, APPS).

    With a ``language`` list the model names each program's language. Per-problem ``checker`` / ``time_limit`` from
    the ``answer`` payload override the ``output_comparison`` default.
    """

    # Agentic solvers iterate test→fix→submit, which the protocol's generic budget cuts mid-loop.
    DEFAULT_MAX_TURNS = 15

    SHAPING_COMPONENTS = ("submission", "resubmission")

    EPISODE_FACTS = (FAILED_WITH_NO_RUNS_LEFT,)

    # The ``answer`` payload is the hidden test set; without it every episode grades 0 whatever it submitted.
    requires_answer = True

    # Per-tool shaping off so correctness dominates; configs may re-enable small values.
    DEFAULT_TOOL_SUCCESS_REWARD = 0.0
    DEFAULT_TOOL_ERROR_PENALTY = 0.0

    # The protocol's empty-turn nudge offers a final answer, which here ends the episode ungraded, so these name the
    # one graded channel. Same rule as the protocol's: the fact and the action, never an ask for shorter reasoning.
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

    # The task's own profile keys over the base's: per-episode interaction budgets, so effort buys iteration as well as
    # the reasoning its budget leaves free.
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
        reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
        reasoning_effort_profiles: dict[str, dict[str, int | float]] | None = None,
        eval_protocol: str = DEFAULT_EVAL_PROTOCOL,
        **kwargs,
    ):
        if kwargs.get("tool_budgets"):
            raise ValueError(
                f"{type(self).__name__} sets its tools' budgets from max_submissions and max_test_calls at every "
                "reset, so tool_budgets would never apply. Set those instead, or per level in "
                "reasoning_effort_profiles."
            )
        if kwargs.get("requires_answer") is False:
            raise ValueError(
                f"{type(self).__name__} grades against the hidden tests in the dataset's answer column, so "
                f"requires_answer cannot be false: without that gate an answerless dataset fails at its first "
                f"submission. Remove it."
            )
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
        )
        self.languages = tuple(spec.name for spec in specs)
        self.language = self.languages[0]
        self.chooses_language = len(self.languages) > 1
        # Bootstraps a weak base that never submits; self-neutralizes within a GRPO group once all do.
        self.submission_reward = submission_reward
        # Each graded submission after the first probes the judge; free, probing would out-earn testing.
        self.resubmission_penalty = resubmission_penalty
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
        test_tool, submit_tool = build_contest_tools(specs, self.sandbox, run_test=self._run_test, submit=self._submit)
        self.test_tool_name = test_tool.name
        # What a veto judge listing ``checks: environment`` audits with.
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
        # After the base validated every key: a protocol pinning the interaction budgets fixes them at every level, so
        # there a level differs only in the line its template states.
        self.reasoning_effort_profiles = {
            level: without_eval_protocol_pins(entry, eval_protocol, f"reasoning_effort_profiles[{level!r}]")
            for level, entry in self.reasoning_effort_profiles.items()
        }

    @staticmethod
    def _resolve_eval_protocol_knobs(eval_protocol: str, **configured: int | None) -> dict[str, int]:
        """Each knob's pin, else the configured value, else the default. A configured value contradicting a pin
        raises: the run would carry the protocol's name without following it."""
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
        """One name fixes the language, a list lets the model choose per program. Empty, unknown or repeated raise."""
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
        """The listed language ``code`` is evidently written in instead of ``language``, else ``None`` (always where
        the run fixes its language). Both tools refuse such a program unspent; the offline re-grader skips it too."""
        if not self.chooses_language or language is None:
            return None
        return find_evident_language(code, language, self.languages)

    def _reset_single(
        self,
        prompt: str | list[dict[str, str]],
        context: dict[str, Any] | None = None,
    ) -> Trajectory:
        """A payload with no time limit gets the one each test runs under stated in the task message."""
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
        """Accepts a bare list, ``{"test_cases": [...]}`` or ``{"tests": [...], "checker": ..., "time_limit": ...}``;
        no tests raises. Kept in ``traj.info``, so concurrent episodes never clobber each other."""
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
        """Any payload but a dict, list or JSON string fails the episode at reset, since graded against no tests it
        would score 0, indistinguishable from a wrong solution."""
        answer = context.get(ANSWER_KEY, {})
        if isinstance(answer, str):
            try:
                answer = json.loads(answer)
            except ValueError as exc:
                raise ValueError(f"unparseable {ANSWER_KEY!r} payload ({len(answer)} chars): {exc}") from exc
        if not isinstance(answer, (dict, list)):
            raise ValueError(f"{ANSWER_KEY!r} must be a dict or list of tests, got {type(answer).__name__}")
        return answer

    def _apply_effort_profile(self, traj: Trajectory, profile: dict[str, int | float]) -> None:
        """The task message states none of these budgets: the level the chat template states and the engine's caps
        control an episode, and a refused call is told so."""
        max_subs = profile.get("max_submissions", self.max_submissions)
        max_tests = profile.get("max_test_calls", self.max_test_calls)
        traj.info[EPISODE_TOOL_BUDGETS_KEY].update({self.test_tool_name: max_tests, SUBMIT_TOOL: max_subs})

    def _execute_tool_calls(
        self, tool_calls: list[NativeToolCall], trajectory: Trajectory
    ) -> tuple[list[NativeToolResult], float]:
        # What a failed submission in the turn is judged by.
        info = trajectory.info
        info[TURN_RUNS_LEFT_KEY] = self._exhausted_tool_cap(trajectory, self.test_tool_name) is None
        info[TURN_RUN_PROGRAMS_KEY] = len(info.get(RUN_PROGRAMS_KEY, []))
        return super()._execute_tool_calls(tool_calls, trajectory)

    def _run_test(self, code: str, language: str | None = None, stdin: str = "") -> str:
        """Run a scratchpad test under the per-test limit the episode's problem is graded at; outside an episode it
        runs uncapped at the grading default."""
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
        # Neither the parse error of a program that reads input nor output computed from nothing names the cause;
        # a build failure ran nothing.
        if not stdin.strip() and not result.compile_failed and not host_build_error(code, language, self.sandbox):
            notes.append(NO_STDIN_NOTE)
        return self._fit_observation(output, notes)

    def _call_language(self, language: str | None, tool: str) -> str:
        """The run's language, or the call's when the run lists several (the schema requires and enumerates it, so
        this guards only the direct-call path)."""
        if not self.chooses_language:
            return self.language
        if language is None or language not in self.languages:
            raise ToolArgumentError(f"{tool}: language must be one of {', '.join(self.languages)}, got {language!r}")
        return language

    def _refuse_mislabelled(self, code: str, language: str, tool: str, trajectory: Trajectory | None) -> None:
        """Refused as a tool error with the call returned to the budget and the turn flagged: a free, trainable
        refusal is a wasted turn that every solve reinforces."""
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
        """The episode's ``language`` slice is the last one used, which is the graded submission's once it submits."""
        if not self.chooses_language:
            return
        slices = trajectory.info.setdefault(EPISODE_SLICES_KEY, {})
        previous = slices.get("language")
        if previous is not None and previous != language:
            trajectory.info["language_switches"] = trajectory.info.get("language_switches", 0) + 1
        slices["language"] = language

    def _fit_observation(self, output: str, notes: list[str]) -> str:
        """``output`` with ``notes`` after it, the output cut to fit ``max_observation_chars``, since the protocol's
        own cap cuts from the end and would eat the notes."""
        tail = "".join(f"\n{note}" for note in notes)
        cap = self.max_observation_chars
        if cap and len(output) + len(tail) > cap:
            # The marker counting every character of the output bounds the one a cut adds.
            marker = len(TRUNCATION_MARKER.format(dropped=len(output)))
            output = self._truncate_observation(output, max(1, cap - len(tail) - marker))
        return output + tail

    def _submit(self, code: str, language: str | None = None) -> str:
        """Grade against the episode's tests. The protocol keeps the ``max_submissions`` count and increments it
        before this runs, so the first admitted call is submission one. A program identical to one already graded
        (:func:`normalized_program`) takes that grade again, unregraded, behind :data:`IDENTICAL_RESUBMISSION_NOTE`:
        the same verdict, spent and priced like any submission."""
        language = self._call_language(language, SUBMIT_TOOL)
        trajectory = self.active_trajectory()
        if trajectory is None:
            raise ValueError("submit_solution called outside an active episode.")
        info = trajectory.info
        if is_accepted(info):
            # A later call in the accepting turn; grading another program could only replace the solve.
            self._refund_tool_call(trajectory, SUBMIT_TOOL)
            return SUBMISSION_AFTER_ACCEPT_REPLY
        self._refuse_mislabelled(code, language, SUBMIT_TOOL, trajectory)
        self._note_language(trajectory, language)

        if self._submissions(trajectory) == 1:
            info["tested_before_submission"] = self._test_calls(trajectory) > 0
        program = normalized_program(code, language)
        earlier = info.get(GRADED_PROGRAMS_KEY, {}).get(program)
        if earlier is None:
            grade = self._grade_submission(code, trajectory, language)
        else:
            ordinal, grade = earlier
            info[IDENTICAL_RESUBMISSIONS_KEY] = info.get(IDENTICAL_RESUBMISSIONS_KEY, 0) + 1
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
        # What an external scorer reads (``_scoring_sample``); private, so it leaves with the grading payload.
        info["_submitted_code"] = code
        if earlier is not None:
            return f"{IDENTICAL_RESUBMISSION_NOTE.format(ordinal=ordinal)}\n{grade.details}"
        if not (grade.infra_errors or is_grade_inconclusive(info)):
            # A grade the backend lost, or one the budget cut short, says nothing about the program: the same one may
            # come back.
            info.setdefault(GRADED_PROGRAMS_KEY, {})[program] = (self._submissions(trajectory), grade)
        return grade.details

    def _submissions(self, trajectory: Trajectory) -> int:
        """The protocol counts a call before its handler runs."""
        return self._tool_calls_made(trajectory, SUBMIT_TOOL)

    def _test_calls(self, trajectory: Trajectory) -> int:
        return self._tool_calls_made(trajectory, self.test_tool_name)

    def _grade_submission(self, code: str, trajectory: Trajectory, language: str | None = None) -> GradeResult:
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
        """Also ends the episode once the submission budget is spent or a submission passed every hidden test, after
        which a resubmission could only lose the solve."""
        outcome = super()._step_single(trajectory, action, context)
        info = outcome.trajectory.info
        # A turn that also booked a sandbox fault ends on it, uncompleted, whatever it submitted.
        spent = self._exhausted_tool_cap(outcome.trajectory, SUBMIT_TOOL) is not None
        if not outcome.done and (spent or is_accepted(info)) and SANDBOX_FAULT_KEY not in info:
            info[COMPLETED_KEY] = True
            return outcome._replace(done=True)
        return outcome

    def _grade_episode(self, trajectory: Trajectory, context: dict[str, Any] | None = None) -> EpisodeGrade:
        """1 when the submitted solution passed every hidden test, else 0 (what pass@1 counts). Shaping bootstraps the
        tool-use loop and prices probing the judge; none pays on a grade lost to the backend or inconclusive
        (:func:`is_grade_inconclusive`), which leaves the GRPO group baseline."""
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
        return EpisodeGrade(objective, {"submission": submission, "resubmission": resubmission})

    def _final_answer(self, trajectory: Trajectory) -> str | None:
        """The last submitted program as a fenced block, or nothing, so a scorer reads that no submission was made,
        never the tool-call turn as an answer."""
        code = trajectory.info.get("_submitted_code")
        if code is None:
            return None
        return f"```{trajectory.info['submission_language']}\n{code}\n```"

    def _scoring_reference(self, trajectory: Trajectory) -> None:
        """The hidden tests are the grader's payload, never a reference a judge reads."""
        return None

    def _episode_facts(self, trajectory: Trajectory) -> frozenset[str]:
        runs_left = trajectory.info.get(FAILED_SUBMISSION_RUNS_LEFT_KEY, [])
        return frozenset({FAILED_WITH_NO_RUNS_LEFT}) if runs_left and not any(runs_left) else frozenset()

    def rollout_metrics(self, trajectory: Trajectory) -> dict[str, float]:
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
            metrics["episode/tested_before_submission"] = 1.0 if info.get("tested_before_submission") else 0.0
        if submissions > 1:
            fracs = info.get(SUBMISSION_PASS_FRACS_KEY, [])
            improved = sum(1 for i in range(1, len(fracs)) if fracs[i] > max(fracs[:i]))
            metrics["episode/resubmission_improved"] = improved / (submissions - 1)
        if self.chooses_language:
            metrics["episode/language_switches"] = float(info.get("language_switches", 0))
        if graded:
            metrics["episode/grading_infra_outage"] = 1.0 if is_grading_infra_outage(info) else 0.0
            metrics["episode/grade_inconclusive"] = 1.0 if is_grade_inconclusive(info) else 0.0
            # The pass fraction's denominator is the full pool, so an ungraded remainder reads like a wrong answer.
            metrics["episode/tests_graded_frac"] = info.get("tests_graded", 0) / tests_total
            metrics["episode/grading_budget_hit"] = 1.0 if info.get("grading_budget_hit") else 0.0
        return metrics
