"""Competitive-programming environment with hidden-test grading (Codeforces, APPS).

Models write a solution, test it with the scratchpad tool, then submit via ``submit_solution`` which
runs it against hidden tests through a SandboxExecutor. The grade is 1 when the submitted solution
passes every hidden test and 0 otherwise, priced by the reward's environment term. The run fixes one
language, or lists several and lets the model pick one per program.
"""

import json
import logging
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

from src.environments.base import (
    EPISODE_INVALID_KEY,
    EPISODE_INVALID_REASON_KEY,
    EPISODE_SLICES_KEY,
    EPISODE_TOOL_BUDGETS_KEY,
    SANDBOX_FAULT_KEY,
    SOLVE_RATE_KEY,
    EpisodeGrade,
    Trajectory,
    require_magnitudes,
)
from src.environments.envs.protocols.native import NativeToolUseEnvironment
from src.environments.envs.tasks.coding.grading import (
    DEFAULT_MAX_OUTPUT_SIZE,
    VERDICT_DETAIL_OUTCOME,
    VERDICT_DETAILS,
    GradeResult,
    GradingSpec,
    grade_solution,
)
from src.environments.sandbox.base import (
    SANDBOX_DEFAULT_TIMEOUT,
    LanguageSpec,
    SandboxExecutor,
    require_language,
)
from src.environments.sandbox.repl import format_sandbox_repl_output
from src.environments.sandbox.resolve import resolve_sandbox, warn_if_unisolated
from src.environments.tools.definitions import NativeTool, NativeToolRegistry, ToolArgumentError, ToolParameter
from src.rewards.samples import ScoringSample

logger = logging.getLogger(__name__)

# Effort level -> profile. ``thinking_tokens`` is the level's CoT budget (per turn, or per episode
# by the run's thinking-budget scope); a config adds the interaction budgets (``max_submissions``,
# ``max_test_calls``) so effort buys iteration too.
REASONING_EFFORT_PROFILES: dict[str, dict[str, int | float]] = {
    "low": {"thinking_tokens": 4096},
    "medium": {"thinking_tokens": 8192},
    "high": {"thinking_tokens": 16384},
}
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

SUBMIT_TOOL = "submit_solution"
# Pass fraction of each graded submission, in order: what ``episode/resubmission_improved`` reads.
SUBMISSION_PASS_FRACS_KEY = "submission_pass_fracs"
NO_STDIN_NOTE = "(No stdin was passed to this run; give the program its input in the `stdin` argument.)"
STARVED_RUN_NOTE = (
    "(No stdin was passed and the program printed nothing, so this run was not counted; give the program its "
    "input in the `stdin` argument.)"
)
# Scratchpad runs returned for getting no input and printing nothing (``episode/starved_test_runs``).
STARVED_TEST_RUNS_KEY = "starved_test_runs"
# Follows a scratchpad timeout, whose limit is the one this problem's graded tests run under.
SCRATCHPAD_TIME_LIMIT_NOTE = "(the per-test time limit this problem is graded at)"
# Closes every scratchpad reply while the episode caps the scratchpad.
SCRATCHPAD_RUNS_LEFT_NOTE = "(Scratchpad runs left: {left} of {cap}.)"


def eval_protocol_pins(eval_protocol: str) -> dict[str, int]:
    """The knobs ``eval_protocol`` pins; an unknown protocol raises."""
    if eval_protocol not in EVAL_PROTOCOLS:
        raise ValueError(f"eval_protocol must be one of {sorted(EVAL_PROTOCOLS)}, got {eval_protocol!r}")
    return EVAL_PROTOCOLS[eval_protocol]


def without_eval_protocol_pins(config: dict[str, Any], eval_protocol: str, source: str) -> dict[str, Any]:
    """``config`` without the knobs ``eval_protocol`` pins, each dropped one logged. For a contract
    written under another protocol (a training config's env options, an effort profile's budgets):
    its values give way to the pins instead of contradicting them."""
    pins = eval_protocol_pins(eval_protocol)
    dropped = sorted(pins.keys() & config.keys())
    if dropped:
        logger.info("eval_protocol %r pins %s; the values %s sets give way", eval_protocol, dropped, source)
    return {key: value for key, value in config.items() if key not in pins}


class CodeContestsEnvironment(NativeToolUseEnvironment):
    """Competitive-programming environment with hidden-test grading (Codeforces, code_contests, APPS).

    ``language`` (``python``/``cpp``/``c``, or a list of them) drives both the test tool and grading
    through the same SandboxExecutor; with a list the model names each program's language in the
    tool call and is graded in it. The grade is 1 when the solution passed to ``submit_solution``, the
    single graded channel, passes every hidden test, else 0; a never-submitted solution grades 0.
    Grading is data-driven: per-problem ``checker`` / ``time_limit`` from the ``answer`` payload (dict
    or JSON string, carrying ``tests``/``test_cases``) override the ``output_comparison`` default, so
    one env covers exact-match and Codeforces sets. ``eval_protocol`` names the evaluation contract
    (:data:`EVAL_PROTOCOLS`), which pins the knobs it fixes.
    """

    # Agentic solvers iterate test→fix→submit, which the protocol's generic budget cuts mid-loop.
    DEFAULT_MAX_TURNS = 15

    SHAPING_COMPONENTS = ("submission", "resubmission")

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
        "Your previous turn ended without a tool call, so nothing was recorded, and only a solution sent "
        "with submit_solution is graded. Make your tool call now with the best solution you have."
    )

    REASONING_EFFORT_PROFILES = REASONING_EFFORT_PROFILES
    # The task's own profile keys over the base's: the per-episode interaction budgets.
    EFFORT_PROFILE_KEY_MINIMA = {"max_submissions": 1, "max_test_calls": 0}

    CODE_SYSTEM_PROMPT = (
        "You are an expert competitive programmer. Write a correct and efficient {language} solution "
        "to the problem below, at the standard expected to pass a rated contest.\n\n"
        "Your solution must be a complete program that reads input from stdin and writes the answer to "
        "stdout, within the stated time and memory limits. Submit it with the submit_solution tool to be "
        "graded against the hidden tests — this is the only graded channel, so a solution you do not "
        "submit scores nothing.\n\n"
        "Put every piece of code in a tool call — the test tool to try it, submit_solution to be graded — "
        "never in your message. Your message to the user must contain no code: write only a brief summary "
        "of what you did this turn (your approach and what you tested)."
    )
    # Appended when the run lists several languages: the choice is the model's, per program.
    LANGUAGE_CHOICE_PROMPT = " Choose each program's language with the tool's language argument ({names})."
    # Appended when the set mixes interpreted and compiled languages: the grading contract each runs under
    # (``multiplier`` is empty at scale 1, else e.g. ``"2x "``).
    TIME_LIMIT_FLOOR_PROMPT = (
        " A {interpreted} solution gets at least {floor:g} s per test; a compiled one runs at {multiplier}the "
        "problem's stated time limit."
    )

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
        max_starved_run_refunds: int = 1,
        submission_reward: float = 0.0,
        resubmission_penalty: float = 0.0,
        reasoning_effort: str | None = DEFAULT_REASONING_EFFORT,
        reasoning_effort_profiles: dict[str, dict[str, int | float]] | None = None,
        eval_protocol: str = DEFAULT_EVAL_PROTOCOL,
        **kwargs,
    ):
        knobs = self._resolve_eval_protocol_knobs(
            eval_protocol, max_submissions=max_submissions, max_test_calls=max_test_calls
        )
        max_submissions, max_test_calls = knobs["max_submissions"], knobs["max_test_calls"]
        self.eval_protocol = eval_protocol
        specs = self._resolve_languages(language)
        if output_comparison not in ("exact", "tokens"):
            raise ValueError(f"output_comparison must be 'exact' or 'tokens', got {output_comparison!r}")
        if verdict_detail not in VERDICT_DETAILS:
            raise ValueError(f"verdict_detail must be one of {VERDICT_DETAILS}, got {verdict_detail!r}")
        if max_submissions < 1:
            raise ValueError(f"max_submissions must be >= 1, got {max_submissions}")
        if max_test_calls < 0:
            raise ValueError(f"max_test_calls must be >= 0, got {max_test_calls}")
        if (
            isinstance(max_starved_run_refunds, bool)
            or not isinstance(max_starved_run_refunds, int)
            or max_starved_run_refunds < 0
        ):
            raise ValueError(f"max_starved_run_refunds must be an int >= 0, got {max_starved_run_refunds!r}")
        if max_grading_seconds is not None and max_grading_seconds <= 0:
            raise ValueError(f"max_grading_seconds must be > 0 or None, got {max_grading_seconds}")
        if max_time_limit < timeout_per_test:
            # The prompt promises an interpreted solution at least timeout_per_test per test; a clamp
            # below it would grade under a contract the model was never told.
            raise ValueError(
                f"max_time_limit ({max_time_limit}) must be >= timeout_per_test ({timeout_per_test}), "
                f"the per-test floor the task contract states"
            )
        require_magnitudes(submission_reward=submission_reward, resubmission_penalty=resubmission_penalty)
        # The canonical names the model may name; ``language`` is the run's default (and the grading
        # contract's), the only one when the run fixes it.
        self.languages = tuple(spec.name for spec in specs)
        self.language = self.languages[0]
        self.chooses_language = len(self.languages) > 1
        # Bootstraps a weak base that never submits; self-neutralizes within a GRPO group once all do.
        self.submission_reward = submission_reward
        # Each graded submission after the first is a probe of the judge. Free, and with only the last
        # one counting, probing out-earns testing in the scratchpad within a GRPO group at every effort;
        # the task message states the price, so stopping is an option.
        self.resubmission_penalty = resubmission_penalty
        # Reaching the cap ends the episode; further calls are rejected as tool errors.
        self.max_submissions = max_submissions
        self.max_test_calls = max_test_calls
        # A first input-less silent run is a slip the note corrects; returning every one makes a run that
        # reads nothing free to repeat, so past this many an episode's silent runs count like any other.
        self.max_starved_run_refunds = max_starved_run_refunds
        # Without the interaction half, the strategy collapses to submit-and-fix at every effort level.
        # Read off the overrides (the class profiles carry only thinking budgets) because the tool
        # descriptions built below defer to the task message whenever a level binds interaction.
        self._profiles_bind_interaction = any(
            set(p) - {"thinking_tokens"} for p in (reasoning_effort_profiles or {}).values()
        )
        self.sandbox = sandbox or resolve_sandbox(backend=sandbox_backend, url=sandbox_url)
        warn_if_unisolated(self.sandbox, type(self).__name__)
        # Built once and the single reader of these knobs: every submission of the run is graded under
        # the same contract, and the offline re-grader takes it (via ``to_meta``) to reproduce the
        # verdicts. ``max_time_limit`` caps a per-problem limit so a mis-scaled solution can't pin a
        # rollout worker; ``max_grading_seconds`` bounds the sequential test run of one submission, so
        # a several-hundred-test problem does not grade for tens of minutes and stall the round.
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

        test_tool = self._build_test_tool(specs)
        self.test_tool_name = test_tool.name

        if self._profiles_bind_interaction:
            submit_budget = (
                "Your graded-submission budget for this task is stated in the task message; reaching it ends the task."
            )
        elif max_submissions == 1:
            submit_budget = (
                "This is your only graded submission and it ends the task, so submit only once you are confident."
            )
        else:
            submit_budget = (
                f"You get up to {max_submissions} graded submissions, "
                "so you can read the verdict and fix once before the budget runs out."
            )
        registry = NativeToolRegistry()
        registry.register(test_tool)
        registry.register(
            NativeTool(
                name=SUBMIT_TOOL,
                description=(
                    f"Submit a complete {self._language_phrase(specs)} program to be graded against the problem's "
                    f"hidden tests — the only way to score this problem. The program must read from stdin and "
                    f"write to stdout. Returns how many tests passed. {submit_budget}"
                ),
                parameters=[
                    ToolParameter(
                        "code",
                        "string",
                        f"Complete {self._language_phrase(specs)} program reading stdin, writing stdout",
                    ),
                    *self._language_parameters(specs),
                ],
                handler=self._submit_in if self.chooses_language else self._submit,
                budget_message="Submission limit reached ({cap}); this submission is not graded.",
            )
        )

        super().__init__(
            tool_registry=registry,
            system_prompt=system_prompt
            or self._default_system_prompt(specs, timeout_per_test, compiled_time_limit_scale),
            reasoning_effort=reasoning_effort,
            reasoning_effort_profiles=reasoning_effort_profiles,
            **kwargs,
        )
        # After the base has validated every profile key, pinned ones included. A profile's interaction
        # budgets make effort buy iteration; a protocol pinning them fixes the budget at every level,
        # so effort buys thinking alone there (a training config's ladder keeps its thinking budgets).
        self.reasoning_effort_profiles = {
            level: without_eval_protocol_pins(entry, eval_protocol, f"reasoning_effort_profiles[{level!r}]")
            for level, entry in self.reasoning_effort_profiles.items()
        }

    @staticmethod
    def _resolve_eval_protocol_knobs(eval_protocol: str, **configured: int | None) -> dict[str, int]:
        """Each :data:`EVAL_PROTOCOL_KNOB_DEFAULTS` knob under ``eval_protocol``: its pin, else the
        configured value, else the default. A configured value contradicting a pin raises — the run
        would carry the protocol's name without following it."""
        pins = eval_protocol_pins(eval_protocol)
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

    @staticmethod
    def _language_phrase(specs: Sequence[LanguageSpec]) -> str:
        """The language set as prose (``python``, ``python or cpp``) — canonical names, the values the
        tool argument takes."""
        return " or ".join(spec.name for spec in specs)

    @staticmethod
    def _language_parameters(specs: Sequence[LanguageSpec]) -> list[ToolParameter]:
        """The ``language`` tool argument when the run lets the model choose; none when it fixes one."""
        if len(specs) < 2:
            return []
        names = [spec.name for spec in specs]
        return [ToolParameter("language", "string", f"Language of the program: {', '.join(names)}", enum=names)]

    def _default_system_prompt(self, specs: Sequence[LanguageSpec], floor: float, compiled_scale: float) -> str:
        """The solver's role and task contract, with the language choice and its grading terms when
        the run lists several languages."""
        prompt = self.CODE_SYSTEM_PROMPT.format(language=self._language_phrase(specs))
        if len(specs) < 2:
            return prompt
        prompt += self.LANGUAGE_CHOICE_PROMPT.format(names=", ".join(spec.name for spec in specs))
        interpreted = [spec.name for spec in specs if not spec.is_compiled]
        if interpreted and len(interpreted) < len(specs):
            multiplier = "" if compiled_scale == 1 else f"{compiled_scale:g}x "
            prompt += self.TIME_LIMIT_FLOOR_PROMPT.format(
                interpreted=" or ".join(interpreted), floor=floor, multiplier=multiplier
            )
        return prompt

    def _build_test_tool(self, specs: Sequence[LanguageSpec]) -> NativeTool:
        """Build the code-testing scratchpad tool for the run's language set.

        Runs through the same SandboxExecutor that grades ``submit_solution`` (isolated subprocess, not the
        in-process restricted REPL which blocks imports) on the stdin the call supplies; never the graded tests.
        """
        python_only = len(specs) == 1 and specs[0].name == "python"
        name = "python_repl" if python_only else "run_code"
        verb = "Runs" if all(not spec.is_compiled for spec in specs) else "Compiles and runs"
        if self._profiles_bind_interaction:
            test_budget = "Your scratchpad budget for this task is stated in the task message."
        elif self.max_test_calls:
            test_budget = (
                f"You may run it up to {self.max_test_calls} times this task, so test deliberately, then submit."
            )
        else:
            test_budget = "This tool is disabled for this task."
        description = (
            f"Optional scratchpad — this does NOT submit your solution. {verb} the complete "
            f"{self._language_phrase(specs)} program you pass on the stdin you give it and returns its output; "
            "the standard library is available. It has no access to the graded tests, so feed it the statement's "
            f"sample input or your own. {test_budget} Use submit_solution to be graded."
        )
        return NativeTool(
            name=name,
            description=description,
            parameters=[
                ToolParameter("code", "string", f"Complete {self._language_phrase(specs)} program"),
                ToolParameter(
                    "stdin", "string", "Input the program reads from standard input (empty by default)", required=False
                ),
                *self._language_parameters(specs),
            ],
            handler=self._run_test_in if self.chooses_language else self._run_test,
            budget_message=(
                "Test limit reached ({cap}); the scratchpad is exhausted. Submit your solution with submit_solution."
            ),
        )

    def _call_language(self, language: str | None, tool: str) -> str:
        """The language a program runs and is graded in: the run's one language, or — when the run
        lists several — the call's ``language`` argument. The tool schema makes that argument required
        and enumerates the set, so the protocol refuses a missing or foreign value before the call is
        admitted; this guards the direct-call path."""
        if not self.chooses_language:
            return self.language
        if language is None or language not in self.languages:
            raise ToolArgumentError(f"{tool}: language must be one of {', '.join(self.languages)}, got {language!r}")
        return language

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

    def _submissions(self, trajectory: Trajectory) -> int:
        """Graded-submission calls admitted so far (the protocol counts a call before its handler runs)."""
        return self._tool_calls_made(trajectory, SUBMIT_TOOL)

    @staticmethod
    def _improved_resubmissions(trajectory: Trajectory) -> int:
        """Graded submissions after the first whose pass fraction beat every earlier one."""
        fracs = trajectory.info.get(SUBMISSION_PASS_FRACS_KEY, [])
        return sum(1 for i in range(1, len(fracs)) if fracs[i] > max(fracs[:i]))

    def _test_calls(self, trajectory: Trajectory) -> int:
        """Scratchpad calls admitted so far."""
        return self._tool_calls_made(trajectory, self.test_tool_name)

    def _refunds_left(self, trajectory: Trajectory | None) -> bool:
        """Whether an input-less silent run is still returned to the budget: up to ``max_starved_run_refunds``
        per episode, always outside one (a direct call keeps no budget)."""
        return trajectory is None or trajectory.info.get(STARVED_TEST_RUNS_KEY, 0) < self.max_starved_run_refunds

    def _run_test_in(self, code: str, language: str, stdin: str = "") -> str:
        """The scratchpad handler when the run lets the model choose: ``language`` is required, so a
        call without it fails to bind and is refused unspent."""
        return self._run_test(code, language, stdin)

    def _submit_in(self, code: str, language: str) -> str:
        """The submission handler when the run lets the model choose (``language`` required, as above)."""
        return self._submit(code, language)

    def _run_test(self, code: str, language: str | None = None, stdin: str = "") -> str:
        """Run a scratchpad test in ``language`` (the run's, or the call's choice) on ``stdin``, under
        the per-test limit the episode's problem is graded at (the grading default outside an episode).
        The per-episode cap is the protocol's (``max_test_calls``); a direct call with no active episode
        runs uncapped."""
        language = self._call_language(language, self.test_tool_name)
        trajectory = self.active_trajectory()
        if trajectory is not None:
            self._note_language(trajectory, language)
        stated = trajectory.info.get("_time_limit") if trajectory is not None else None
        timeout = self.grading_spec.time_limit_for(language, stated)
        stdin = str(stdin or "")
        result = self.sandbox.run(code, stdin=stdin, timeout=timeout, language=language)
        lines = [format_sandbox_repl_output(result, timeout)]
        clean_exit = result.returncode in (0, None)
        if result.timed_out:
            lines[0] += f" {SCRATCHPAD_TIME_LIMIT_NOTE}"
        elif (
            not stdin
            and not result.compile_failed
            and clean_exit
            and not (result.stdout + result.stderr).strip()
            and self._refunds_left(trajectory)
        ):
            # Given no input, the program told the model nothing, so the run is returned to the budget.
            lines.append(STARVED_RUN_NOTE)
            if trajectory is not None:
                self._refund_tool_call(trajectory, self.test_tool_name)
                trajectory.info[STARVED_TEST_RUNS_KEY] = trajectory.info.get(STARVED_TEST_RUNS_KEY, 0) + 1
        elif not stdin and not result.compile_failed and (not clean_exit or not result.stdout.strip()):
            # A program that reads input it was not given ends in a parse error, which does not name the
            # cause; the traceback it returns still spends the run. A build failure is not that.
            lines.append(NO_STDIN_NOTE)
        cap = trajectory.info[EPISODE_TOOL_BUDGETS_KEY].get(self.test_tool_name) if trajectory is not None else None
        if cap is not None:
            # The protocol counted this run before the handler ran.
            lines.append(SCRATCHPAD_RUNS_LEFT_NOTE.format(left=max(0, cap - self._test_calls(trajectory)), cap=cap))
        return "\n".join(lines)

    def _submit(self, code: str, language: str | None = None) -> str:
        """Grade a submission against the active episode's tests in ``language`` (the run's, or the
        call's choice). The per-episode cap is the protocol's (``max_submissions``); the count it keeps
        is incremented before this runs, so the first admitted call is submission one."""
        language = self._call_language(language, SUBMIT_TOOL)
        trajectory = self.active_trajectory()
        if trajectory is None:
            raise ValueError("submit_solution called outside an active episode.")
        self._note_language(trajectory, language)

        if self._submissions(trajectory) == 1:
            # Did any scratchpad run precede the first submission (``episode/tested_before_submission``)?
            trajectory.info["tested_before_submission"] = self._test_calls(trajectory) > 0
        grade = self._grade_submission(code, trajectory, language)
        trajectory.info["submission_language"] = language
        trajectory.info["tests_passed"] = grade.passed
        trajectory.info["tests_total"] = grade.total
        trajectory.info["tests_ran_ok"] = grade.ran_ok
        trajectory.info["tests_infra_errors"] = grade.infra_errors
        trajectory.info["tests_graded"] = grade.graded
        trajectory.info["grading_budget_hit"] = grade.budget_hit
        trajectory.info["submission_result"] = grade.details
        trajectory.info.setdefault(SUBMISSION_PASS_FRACS_KEY, []).append(
            grade.passed / grade.total if grade.total else 0.0
        )
        # The graded artifact an external scorer reads (``_scoring_sample``); private, so it leaves
        # the record with the grading payload.
        trajectory.info["_submitted_code"] = code
        return grade.details

    def _step_single(
        self, trajectory: Trajectory, action: str, context: dict[str, Any] | None = None
    ) -> tuple[Trajectory, float, bool, bool, dict[str, Any]]:
        """Native tool step, then end the episode once the submission budget is spent.

        A submit_solution call is just a tool call, so without this the model keeps going after submitting.
        """
        trajectory, reward, done, truncated, info = super()._step_single(trajectory, action, context)
        # A turn that also booked a sandbox fault ends on it, uncompleted, whatever it submitted.
        spent = self._tool_budget_exhausted(trajectory, SUBMIT_TOOL) is not None
        if not done and spent and SANDBOX_FAULT_KEY not in trajectory.info:
            trajectory.info["completed"] = True
            done = True
        return trajectory, reward, done, truncated, info

    def _grade_submission(self, code: str, trajectory: Trajectory, language: str | None = None) -> GradeResult:
        """Grade ``code`` against the episode's tests → :class:`GradeResult`, in ``language`` (the run's
        default when unset)."""
        return grade_solution(
            code,
            trajectory.info.get("_test_cases", []),
            self.grading_spec,
            checker=trajectory.info.get("_checker"),
            time_limit=trajectory.info.get("_time_limit"),
            language=language or self.language,
        )

    def _reset_single(
        self,
        prompt: str | list[dict[str, str]],
        context: dict[str, Any] | None = None,
    ) -> Trajectory:
        """Build the base trajectory, then store the problem's grading data via the hook."""
        traj = super()._reset_single(prompt, context)
        self._store_problem_data(traj, context or {})
        return traj

    def _apply_effort_profile(self, traj: Trajectory, level: str | None, profile: dict[str, int | float]) -> None:
        """Stamp the episode's interaction budgets — the level's, else the class caps — as the per-tool
        caps the protocol enforces, and state the contract in the task message when any profile binds
        interaction (the tool descriptions then defer to it, so it must exist on every episode, an
        undetermined level included)."""
        max_subs = profile.get("max_submissions", self.max_submissions)
        max_tests = profile.get("max_test_calls", self.max_test_calls)
        traj.info[EPISODE_TOOL_BUDGETS_KEY].update({self.test_tool_name: max_tests, SUBMIT_TOOL: max_subs})
        if not self._profiles_bind_interaction:
            return
        price_rule = "; every resubmission costs part of the score" if self.resubmission_penalty > 0 else ""
        last_wins = f" (the last one is the graded result{price_rule})" if max_subs > 1 else ""
        contract = (
            f"\n\nBudgets for this task: {max_subs} graded submission{'s' if max_subs != 1 else ''}"
            f"{last_wins}, {max_tests} scratchpad run{'s' if max_tests != 1 else ''}."
        )
        traj.append_to_last_user(contract)

    @staticmethod
    def _parse_answer(context: dict[str, Any]) -> dict[str, Any] | list[Any]:
        """Return ``context["answer"]`` as a dict or list, decoding a JSON-string answer from datasets.

        Any other payload raises, failing the episode at reset: graded against no tests, it would
        score 0 inside its GRPO group, indistinguishable from a wrong solution.
        """
        answer = context.get("answer", {})
        if isinstance(answer, str):
            try:
                answer = json.loads(answer)
            except ValueError as exc:
                raise ValueError(f"unparseable 'answer' payload ({len(answer)} chars): {exc}") from exc
        if not isinstance(answer, (dict, list)):
            raise ValueError(f"'answer' must be a dict or list of tests, got {type(answer).__name__}")
        return answer

    def _store_problem_data(self, traj: Trajectory, context: dict[str, Any]) -> None:
        """Store the tests, optional checker, and time limit the submission is graded against.

        Accepts a bare list, ``{"test_cases": [...]}``, or the Codeforces
        ``{"tests": [...], "checker": ..., "time_limit": ...}``; a payload holding no tests raises, as
        in :meth:`_parse_answer`. Written to ``traj.info`` so concurrent Ray-rollout episodes don't
        clobber each other; ``_submit`` reads it via the active-trajectory ContextVar.
        """
        answer = self._parse_answer(context)
        if isinstance(answer, dict):
            test_cases = answer.get("tests") or answer.get("test_cases") or []
            checker = answer.get("checker")
            time_limit = answer.get("time_limit")
        else:
            test_cases, checker, time_limit = answer, None, None
        if not test_cases:
            raise ValueError("'answer' holds no tests: expected a non-empty list, or a non-empty 'tests'/'test_cases'")

        traj.info["_test_cases"] = test_cases
        traj.info["_checker"] = checker
        traj.info["_time_limit"] = float(time_limit) if time_limit else None
        traj.info["tests_passed"] = 0
        traj.info["tests_total"] = len(test_cases)

    @staticmethod
    def _judged_failures(info: dict[str, Any]) -> int:
        """Tests the grade judged and the program failed: graded minus passed minus those lost to the
        backend. A grade with no ``tests_graded`` counts as fully graded."""
        graded = info.get("tests_graded", info.get("tests_total", 0))
        return graded - info.get("tests_passed", 0) - info.get("tests_infra_errors", 0)

    @classmethod
    def _grading_infra_outage(cls, info: dict[str, Any]) -> bool:
        """True when the backend lost the grade: tests lost to it, none passed and none failed."""
        return (
            info.get("tests_infra_errors", 0) > 0
            and info.get("tests_passed", 0) == 0
            and cls._judged_failures(info) == 0
        )

    @classmethod
    def _grade_inconclusive(cls, info: dict[str, Any]) -> bool:
        """True when the grade reached no verdict short of an outage: not every test passed, yet none of
        the judged ones failed — the rest went ungraded (the grading budget ran out) or were lost to the
        backend. All-or-nothing, such a grade would score a solution that may pass every test 0."""
        return (
            info.get("tests_passed", 0) < info.get("tests_total", 0)
            and cls._judged_failures(info) == 0
            and not cls._grading_infra_outage(info)
        )

    def _grade_episode(self, trajectory: Trajectory, context: dict[str, Any] | None = None) -> EpisodeGrade:
        """The objective is 1 when the SUBMITTED solution passed every hidden test, else 0 — the judge's
        accept, which is what pass@1 counts; partial credit pays a brute force that passes the small tests
        and times out on the large ones. ``submit_solution`` is the single graded channel, so an
        unsubmitted solution grades 0. The shaping rungs (the submission bonus, the resubmission price)
        bootstrap the tool-use loop a weak base can't escape and price probing the judge; each is small
        next to the objective. The grade read is the last submission's. No rung pays on an all-infra-error
        grade or an inconclusive one (:meth:`_grade_inconclusive`): neither says anything about the code,
        and both leave the GRPO group baseline. A row holding no tests never reaches this grade: reset
        refuses it (:meth:`_store_problem_data`)."""
        info = trajectory.info
        tests_total = info.get("tests_total", 0)
        # A never-submitted episode is valid: its 0 is the policy's.
        submitted = "submission_result" in info and tests_total > 0
        infra_outage = submitted and self._grading_infra_outage(info)
        inconclusive = submitted and self._grade_inconclusive(info)
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

        objective = 1.0 if graded_content and info.get("tests_passed", 0) == tests_total else 0.0
        # Gated on graded_content, not submission_count: _submit bumps the count before grading.
        submission = self.submission_reward if graded_content else 0.0
        resubmission = -self.resubmission_penalty * max(0, self._submissions(trajectory) - 1)
        return EpisodeGrade(objective, {"submission": submission, "resubmission": resubmission})

    def _scoring_sample(self, trajectory: Trajectory) -> ScoringSample:
        """An external scorer reads the submitted program, not the tool-call turn that carried it."""
        sample = super()._scoring_sample(trajectory)
        code = trajectory.info.get("_submitted_code")
        if code is None:
            return sample
        language = trajectory.info.get("submission_language", self.language)
        # The hidden tests are the grader's payload, not a reference answer a judge should read.
        return replace(
            sample, completion=[{"role": "assistant", "content": f"```{language}\n{code}\n```"}], reference=None
        )

    def rollout_metrics(self, trajectory: Trajectory) -> dict[str, float]:
        """CodeContests diagnostics: task outcome, submission behavior, and the reward decomposition."""
        metrics = super().rollout_metrics(trajectory)
        info = trajectory.info
        tests_total = info.get("tests_total", 0)
        graded = "submission_result" in info
        if graded and tests_total > 0:
            passed = info.get("tests_passed", 0)
            metrics["outcome/test_pass_frac"] = passed / tests_total
            metrics[SOLVE_RATE_KEY] = 1.0 if passed == tests_total else 0.0
        else:
            metrics["outcome/test_pass_frac"] = 0.0
            metrics[SOLVE_RATE_KEY] = 0.0
        submissions = self._submissions(trajectory)
        metrics["episode/submission_rate"] = 1.0 if submissions > 0 else 0.0
        metrics["episode/test_calls"] = float(self._test_calls(trajectory))
        metrics["episode/starved_test_runs"] = float(info.get(STARVED_TEST_RUNS_KEY, 0))
        if submissions > 0:
            # Mean over submitting episodes: the share that ran the scratchpad before submitting.
            metrics["episode/tested_before_submission"] = 1.0 if info.get("tested_before_submission") else 0.0
        if submissions > 1:
            # Over resubmitting episodes: the share of resubmissions that beat every earlier result.
            metrics["episode/resubmission_improved"] = self._improved_resubmissions(trajectory) / (submissions - 1)
        if self.chooses_language:
            metrics["episode/language_switches"] = float(info.get("language_switches", 0))
        if graded and tests_total > 0:
            metrics["episode/grading_infra_outage"] = 1.0 if self._grading_infra_outage(info) else 0.0
            metrics["episode/grade_inconclusive"] = 1.0 if self._grade_inconclusive(info) else 0.0
            # Partial grading is invisible in the pass fraction, which keeps the full pool as its
            # denominator: an ungraded remainder reads exactly like a wrong solution.
            metrics["episode/tests_graded_frac"] = info.get("tests_graded", 0) / tests_total
            metrics["episode/grading_budget_hit"] = 1.0 if info.get("grading_budget_hit") else 0.0
        return metrics
