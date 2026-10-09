"""Competitive-programming environment with hidden-test grading (Codeforces, APPS).

Models write a solution, test it with the scratchpad tool, then submit via ``submit_solution`` which
runs it against hidden tests through a SandboxExecutor. The grade is 1 when the submitted solution
passes every hidden test and 0 otherwise, priced by the reward's environment term. The run fixes one
language, or lists several and lets the model pick one per program.
"""

import json
import logging
import re
from collections.abc import Sequence
from typing import Any

from src.environments.base import (
    ANSWER_KEY,
    EPISODE_INVALID_KEY,
    EPISODE_INVALID_REASON_KEY,
    EPISODE_SLICES_KEY,
    EPISODE_TOOL_BUDGETS_KEY,
    SANDBOX_FAULT_KEY,
    SOLVE_RATE_KEY,
    TRUNCATION_MARKER,
    EpisodeGrade,
    Trajectory,
    require_count,
    require_magnitudes,
)
from src.environments.envs.protocols.native import NativeToolUseEnvironment
from src.environments.envs.tasks.coding.comments import (
    asserts,
    comment_chars,
    deliberation_cues,
    reads_stdin,
    reasoning_in_comments,
    strip_comments,
)
from src.environments.envs.tasks.coding.grading import (
    DEFAULT_MAX_OUTPUT_SIZE,
    VERDICT_DETAIL_OUTCOME,
    GradeResult,
    GradingSpec,
    grade_solution,
    host_build_error,
    python_syntax_error,
)
from src.environments.sandbox.base import (
    SANDBOX_DEFAULT_TIMEOUT,
    LanguageSpec,
    SandboxExecutor,
    SandboxResult,
    require_language,
)
from src.environments.sandbox.repl import format_sandbox_repl_output
from src.environments.sandbox.resolve import resolve_sandbox, warn_if_unisolated
from src.environments.tools.definitions import (
    NativeTool,
    NativeToolRegistry,
    NativeToolResult,
    ToolArgumentError,
    ToolCallRefused,
    ToolParameter,
    UninformativeReply,
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

SUBMIT_TOOL = "submit_solution"
# Pass fraction of each graded submission, in order: what ``episode/resubmission_improved`` reads.
SUBMISSION_PASS_FRACS_KEY = "submission_pass_fracs"
NO_STDIN_NOTE = "(No stdin was passed to this run; if the program reads input, pass it in the `stdin` argument.)"
# The most an input-less run of a program that reads input may print and still have shown the model nothing: one
# token, no whitespace inside it, no wider than the widest 64-bit integer (``-9223372036854775808``), room for what a
# program that parsed nothing prints (a default answer such as ``0``, ``No`` or ``-1``, a ``None``, an uninitialized
# value). Words, a list or a labelled value (``All tests passed``, ``[3, 1, 2]``, ``N=7 count: 5586``) show something.
TRIVIAL_OUTPUT_CHARS = 20
# Scratchpad runs given no input that showed nothing (:func:`_input_less_run_showed_nothing`,
# ``episode/starved_test_runs``), and the turns flagged untrainable for holding one beside nothing but refused or
# unknown calls (``episode/starved_turns``).
STARVED_TEST_RUNS_KEY = "starved_test_runs"
STARVED_TURNS_KEY = "starved_turns"
# Programs refused for carrying the reasoning in their comments (``episode/reasoning_in_comments_calls``),
# and the comment and code characters of every program a call carried (``episode/code_comment_share``).
REASONING_IN_COMMENTS_KEY = "reasoning_in_comments_calls"
COMMENT_CHARS_KEY = "comment_chars"
CODE_CHARS_KEY = "code_chars"
# The refusal of a program whose comments carry its reasoning (:mod:`.comments`): the fact and what to do.
REASONING_IN_COMMENTS_REPLY = (
    "Not {verb}: the program's comments carry your reasoning. Keep the reasoning in your thinking and send the "
    "program again with documentation comments only."
)
# Follows a scratchpad timeout, whose limit is the one this problem's graded tests run under.
SCRATCHPAD_TIME_LIMIT_NOTE = "(the per-test time limit this problem is graded at)"
# The refusals of a call past the episode's budget for the tool: the fact and what to do, never the
# budget's size (the level the chat template states and the engine's caps control what an episode may do).
SCRATCHPAD_BUDGET_SPENT_REPLY = (
    f"Not run: this task's scratchpad budget is spent. Submit your solution with {SUBMIT_TOOL}."
)
SUBMISSION_BUDGET_SPENT_REPLY = "Not graded: this task's submission budget is spent."
# Appended to the task message when the payload carries no time limit (``limits``: ``5 s``, or per
# language); the statement itself may still name one.
UNSTATED_TIME_LIMIT_NOTE = "\n\nEach test of this problem runs under {limits}."
# The reply to a program whose call names a language it is evidently not written in.
MISLABELLED_LANGUAGE_REPLY = (
    'Not {verb}: this looks like {evident} code sent with language "{language}"; send it again with language '
    '"{evident}". No {spent} was spent.'
)
# The reply to a submission after one that passed every hidden test.
SUBMISSION_AFTER_ACCEPT_REPLY = (
    "Not graded: an earlier submission already passed every test, so it stands and the task ends."
)
# A program identical to one this episode already graded, comments aside, would draw the same verdict: refused
# unspent, the turn flagged like any refusal (``episode/identical_resubmissions``). Graded programs are kept
# normalized (``_graded_programs``, private: it leaves the record with the grading payload); one whose grade the
# backend lost is not kept.
IDENTICAL_RESUBMISSION_REPLY = (
    "Not graded: this program is identical to one already graded and would receive the same verdict. Change it "
    "before submitting again."
)
IDENTICAL_RESUBMISSIONS_KEY = "identical_resubmissions"
GRADED_PROGRAMS_KEY = "_graded_programs"

# Marks of C or C++ source for the language-label check: an include directive, a main function, a
# namespace directive or a std-qualified name; else statements ending in ``;`` on this share of the
# non-blank lines.
_C_FAMILY_MARKS = re.compile(r"^\s*#\s*include\s*[<\"]|\bint\s+main\s*\(|\busing\s+namespace\b|\bstd::", re.MULTILINE)
_C_FAMILY_STATEMENT_SHARE = 0.3
# Marks of C++ over C: a namespace directive, a std-qualified name, a C++ standard header (a name with
# no ``.h``, or ``bits/stdc++.h``), the iostream objects, a template. A C header (``<stdio.h>``) marks C.
_CPP_MARKS = re.compile(
    r"\busing\s+namespace\b|\bstd::|^\s*#\s*include\s*<(?:bits/stdc\+\+\.h|\w+)>|\bcin\b|\bcout\b|\btemplate\s*<",
    re.MULTILINE,
)
_C_MARKS = re.compile(r"^\s*#\s*include\s*<[\w/]+\.h>", re.MULTILINE)
# Marks of Python source: a function definition, an import, a print call.
_PYTHON_MARKS = re.compile(r"^\s*(?:def\s+\w+\s*\(|import\s+\w|from\s+[\w.]+\s+import\b)|\bprint\s*\(", re.MULTILINE)


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


def _looks_like_c_family(code: str) -> bool:
    """Whether ``code`` carries a mark of C or C++ source (:data:`_C_FAMILY_MARKS`, or the statement share)."""
    if _C_FAMILY_MARKS.search(code):
        return True
    lines = [line.rstrip() for line in code.splitlines() if line.strip()]
    return bool(lines) and sum(line.endswith(";") for line in lines) >= _C_FAMILY_STATEMENT_SHARE * len(lines)


def _c_family_language(code: str, compiled: Sequence[str]) -> str | None:
    """The language among ``compiled`` that C or C++ source is written in: ``cpp`` on a C++ mark (none
    when the run lists no ``cpp``), ``c`` on a C header without one; source with neither, or C on a run
    listing no ``c``, is ambiguous and takes the first listed."""
    if _CPP_MARKS.search(code):
        return "cpp" if "cpp" in compiled else None
    if "c" in compiled and _C_MARKS.search(code):
        return "c"
    return compiled[0] if compiled else None


def normalized_program(code: str, language: str) -> str:
    """``code`` in ``language`` as the identity check reads it: the language first (the same text is another
    program in another language), then the code without its comments (:func:`strip_comments`), trailing
    whitespace off every line and every blank line dropped, indentation kept (it is the program under Python)."""
    lines = [line.rstrip() for line in strip_comments(code, language).splitlines()]
    return f"{language}\n" + "\n".join(line for line in lines if line)


def evident_language(code: str, language: str, offered: Sequence[str]) -> str | None:
    """The language among ``offered`` that ``code``, sent as ``language``, is evidently written in
    instead, else ``None``.

    C or C++ sent as Python: it does not compile as Python and carries a C-family mark but no Python
    one; C++ is told from C by its own marks (:func:`_c_family_language`). Python sent as a compiled
    language: it compiles as Python and carries a Python mark but no C-family one. Each reading takes
    the parse and the marks together, so a valid program quoting the other language in a string or a
    comment keeps its label, and broken Python with C-like semicolons keeps its syntax error.
    """
    spec = require_language(language)
    if spec.name == "python":
        if not _looks_like_c_family(code) or _PYTHON_MARKS.search(code) or python_syntax_error(code) is None:
            return None
        return _c_family_language(code, [name for name in offered if require_language(name).is_compiled])
    if (
        spec.is_compiled
        and "python" in offered
        and _PYTHON_MARKS.search(code)
        and not _looks_like_c_family(code)
        and python_syntax_error(code) is None
    ):
        return "python"
    return None


def _input_less_run_showed_nothing(code: str, language: str, result: SandboxResult) -> bool:
    """Whether a run of ``code`` given no input showed the model nothing to act on. Only a clean exit (no
    timeout, return code 0) can, its reply being its stdout alone: for a program that reads the run's input
    (:func:`reads_stdin`), stdout empty or one token of at most :data:`TRIVIAL_OUTPUT_CHARS`, output computed
    from nothing; for one that reads none or supplies its own, stdout empty and no assertion in the code
    (:func:`asserts`), a draft whose code never ran. A self-check passes by asserting or printing, so a silent
    one that does neither shows nothing either."""
    if result.timed_out or result.returncode not in (0, None):
        return False
    stdout = result.stdout.strip()
    if reads_stdin(code, language):
        return len(stdout.split()) <= 1 and len(stdout) <= TRIVIAL_OUTPUT_CHARS
    return not stdout and not asserts(code, language)


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

    CODE_SYSTEM_PROMPT = (
        "You are an expert competitive programmer. Write a correct and efficient {language} solution "
        "to the problem below, at the standard expected to pass a rated contest.\n\n"
        "Your solution must be a complete program that reads input from stdin and writes the answer to "
        "stdout, within the stated time and memory limits. Submit it with the submit_solution tool to be "
        "graded against the hidden tests — this is the only graded channel, so a solution you do not "
        "submit scores nothing.\n\n"
        "Put every piece of code in a tool call — {test_tool} to try it, submit_solution to be graded — "
        "never in your message. Your message to the user must contain no code: write only a brief summary "
        "of what you did this turn (your approach and what you tested)."
    )
    # Appended when the run lists several languages: the choice is the model's, per program.
    LANGUAGE_CHOICE_PROMPT = " Choose each program's language with the tool's language argument ({names})."
    # Appended when the set mixes interpreted and compiled languages: the grading contract each runs under
    # (``multiplier`` is empty at scale 1, else e.g. ``"2x "``).
    TIME_LIMIT_FLOOR_PROMPT = (
        " A {interpreted} solution gets at least {floor:g} s per test; a compiled one runs at {multiplier}the "
        "problem's stated time limit, at most {cap:g} s."
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
        require_count("max_submissions", max_submissions, 1)
        require_count("max_test_calls", max_test_calls, 0)
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
        # one counting, probing out-earns testing in the scratchpad within a GRPO group at every effort.
        self.resubmission_penalty = resubmission_penalty
        # Reaching the cap ends the episode; further calls are rejected as tool errors.
        self.max_submissions = max_submissions
        self.max_test_calls = max_test_calls
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

        registry = NativeToolRegistry()
        registry.register(test_tool)
        registry.register(
            NativeTool(
                name=SUBMIT_TOOL,
                description=(
                    f"Submit a complete {self._language_phrase(specs)} program to be graded against the problem's "
                    f"hidden tests — the only way to score this problem. The program must read from stdin and "
                    f"write to stdout.{self._toolchain_clause(specs)} Returns how many tests passed. A submission that "
                    "passes every test ends the task; otherwise the last graded submission is the one that counts."
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
                budget_message=SUBMISSION_BUDGET_SPENT_REPLY,
            )
        )

        super().__init__(
            tool_registry=registry,
            system_prompt=system_prompt or self._default_system_prompt(specs),
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

    def _default_system_prompt(self, specs: Sequence[LanguageSpec]) -> str:
        """The solver's role and task contract, with the language choice and its grading terms (the
        grading contract's) when the run lists several languages."""
        prompt = self.CODE_SYSTEM_PROMPT.format(language=self._language_phrase(specs), test_tool=self.test_tool_name)
        if len(specs) < 2:
            return prompt
        prompt += self.LANGUAGE_CHOICE_PROMPT.format(names=", ".join(spec.name for spec in specs))
        interpreted = [spec.name for spec in specs if not spec.is_compiled]
        if interpreted and len(interpreted) < len(specs):
            grading = self.grading_spec
            scale = grading.compiled_time_limit_scale
            prompt += self.TIME_LIMIT_FLOOR_PROMPT.format(
                interpreted=" or ".join(interpreted),
                floor=grading.default_timeout,
                multiplier="" if scale == 1 else f"{scale:g}x ",
                cap=grading.max_time_limit,
            )
        return prompt

    def _toolchain_clause(self, specs: Sequence[LanguageSpec]) -> str:
        """What the sandbox builds or runs each language with, as one sentence a tool description carries;
        empty where the backend states none."""
        phrases = [f"{spec.name} {phrase}" for spec in specs if (phrase := self.sandbox.toolchain(spec.name))]
        return f" Here {' and '.join(phrases)}." if phrases else ""

    def _build_test_tool(self, specs: Sequence[LanguageSpec]) -> NativeTool:
        """Build the code-testing scratchpad tool for the run's language set.

        Runs through the same SandboxExecutor that grades ``submit_solution`` (isolated subprocess, not the
        in-process restricted REPL which blocks imports) on the stdin the call supplies; never the graded tests.
        """
        python_only = len(specs) == 1 and specs[0].name == "python"
        name = "python_repl" if python_only else "run_code"
        verb = "Runs" if all(not spec.is_compiled for spec in specs) else "Compiles and runs"
        description = (
            f"Optional scratchpad — this does NOT submit your solution. {verb} the complete "
            f"{self._language_phrase(specs)} program you pass on the stdin you give it and returns what it prints "
            f"to stdout (stderr too when it fails); the standard library is available.{self._toolchain_clause(specs)} "
            "It has no access to the graded tests, so feed it the statement's sample input or your own. "
            f"Use {SUBMIT_TOOL} to be graded."
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
            budget_message=SCRATCHPAD_BUDGET_SPENT_REPLY,
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

    def _run_test_in(self, code: str, language: str, stdin: str = "") -> str:
        """The scratchpad handler when the run lets the model choose: ``language`` is required, so a
        call without it fails to bind and is refused unspent."""
        return self._run_test(code, language, stdin)

    def _submit_in(self, code: str, language: str) -> str:
        """The submission handler when the run lets the model choose (``language`` required, as above)."""
        return self._submit(code, language)

    def mislabelled_as(self, code: str, language: str | None) -> str | None:
        """The listed language ``code``, sent as ``language``, is evidently written in instead
        (:func:`evident_language`), else ``None``, always so where the run fixes its language. Both tools
        refuse such a program unspent, and the offline re-grader skips it the same way."""
        if not self.chooses_language or language is None:
            return None
        return evident_language(code, language, self.languages)

    def _refuse_mislabelled(self, code: str, language: str, tool: str, trajectory: Trajectory | None) -> str | None:
        """The reply refusing a program :meth:`mislabelled_as` names another language for, the call
        returned to the budget unpaid (:meth:`_refund_tool_call`); ``None`` when the label fits."""
        evident = self.mislabelled_as(code, language)
        if evident is None:
            return None
        if trajectory is not None:
            self._refund_tool_call(trajectory, tool)
        verb, spent = ("run", "scratchpad run") if tool == self.test_tool_name else ("graded", "submission")
        return MISLABELLED_LANGUAGE_REPLY.format(verb=verb, evident=evident, language=language, spent=spent)

    def _refuse_reasoning_in_comments(
        self, code: str, language: str, tool: str, trajectory: Trajectory | None
    ) -> None:
        """Refuse a program whose comments carry its reasoning (:func:`reasoning_in_comments` on its
        :func:`comment_chars` and :func:`deliberation_cues`) unrun, the call returned to the budget: it costs
        the turn and the protocol's error price, never a run or a submission, and a turn of nothing else is
        flagged untrainable (:class:`ToolCallRefused`). Records every program's comment and code characters
        first, the guard's own signal. The thinking cap bounds the reasoning channel alone and no reasoning
        term counts a call's arguments, so a turn the cap closes could carry its thought on in a program's
        comments; this reads them."""
        comments, rest = comment_chars(code, language)
        if trajectory is not None:
            trajectory.info[COMMENT_CHARS_KEY] = trajectory.info.get(COMMENT_CHARS_KEY, 0) + comments
            trajectory.info[CODE_CHARS_KEY] = trajectory.info.get(CODE_CHARS_KEY, 0) + rest
        if not reasoning_in_comments(comments, rest, lambda: deliberation_cues(code, language)):
            return
        if trajectory is not None:
            self._uncount_tool_call(trajectory, tool)
            trajectory.info[REASONING_IN_COMMENTS_KEY] = trajectory.info.get(REASONING_IN_COMMENTS_KEY, 0) + 1
        raise ToolCallRefused(
            REASONING_IN_COMMENTS_REPLY.format(verb="run" if tool == self.test_tool_name else "graded")
        )

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

    def _run_test(self, code: str, language: str | None = None, stdin: str = "") -> str:
        """Run a scratchpad test in ``language`` (the run's, or the call's choice) on ``stdin``, under
        the per-test limit the episode's problem is graded at (the grading default outside an episode).
        The per-episode cap is the protocol's (``max_test_calls``); a direct call with no active episode
        runs uncapped."""
        language = self._call_language(language, self.test_tool_name)
        trajectory = self.active_trajectory()
        refusal = self._refuse_mislabelled(code, language, self.test_tool_name, trajectory)
        if refusal is not None:
            return refusal
        self._refuse_reasoning_in_comments(code, language, self.test_tool_name, trajectory)
        if trajectory is not None:
            self._note_language(trajectory, language)
        stated = trajectory.info.get("_time_limit") if trajectory is not None else None
        timeout = self.grading_spec.time_limit_for(language, stated)
        result = self.sandbox.run(code, stdin=stdin, timeout=timeout, language=language)
        output = format_sandbox_repl_output(result, timeout)
        if result.timed_out:
            output += f" {SCRATCHPAD_TIME_LIMIT_NOTE}"
        notes = []
        starved = False
        # A run on no input (whitespace is none) says so, since neither the parse error of a program that reads
        # input nor output computed from nothing names the cause; a build failure ran nothing.
        if not stdin.strip() and not result.compile_failed and not host_build_error(code, language, self.sandbox):
            notes.append(NO_STDIN_NOTE)
            starved = _input_less_run_showed_nothing(code, language, result)
            if starved and trajectory is not None:
                trajectory.info[STARVED_TEST_RUNS_KEY] = trajectory.info.get(STARVED_TEST_RUNS_KEY, 0) + 1
        reply = self._fit_observation(output, notes)
        # Spent all the same: a turn of nothing else is flagged, so it cannot buy the next turn a full cap.
        return UninformativeReply(reply) if starved else reply

    def _submit(self, code: str, language: str | None = None) -> str:
        """Grade a submission against the active episode's tests in ``language`` (the run's, or the
        call's choice). The per-episode cap is the protocol's (``max_submissions``); the count it keeps
        is incremented before this runs, so the first admitted call is submission one."""
        language = self._call_language(language, SUBMIT_TOOL)
        trajectory = self.active_trajectory()
        if trajectory is None:
            raise ValueError("submit_solution called outside an active episode.")
        refusal = self._refuse_mislabelled(code, language, SUBMIT_TOOL, trajectory)
        if refusal is not None:
            return refusal
        if self._accepted(trajectory.info):
            # A later call in the turn that submitted the accept: the episode ends on this turn, and
            # grading another program could only replace the solve.
            self._refund_tool_call(trajectory, SUBMIT_TOOL)
            return SUBMISSION_AFTER_ACCEPT_REPLY
        self._refuse_reasoning_in_comments(code, language, SUBMIT_TOOL, trajectory)
        self._refuse_identical_resubmission(code, language, trajectory)
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
        if not grade.infra_errors:
            # A grade the backend lost says nothing about the program, so the same one may come back.
            trajectory.info.setdefault(GRADED_PROGRAMS_KEY, []).append(normalized_program(code, language))
        return grade.details

    def _refuse_identical_resubmission(self, code: str, language: str, trajectory: Trajectory) -> None:
        """Refuse a program identical to one this episode already graded (:func:`normalized_program`) unrun, the
        submission returned to the budget: it would draw the same verdict, so grading it only probes the judge."""
        if normalized_program(code, language) not in trajectory.info.get(GRADED_PROGRAMS_KEY, []):
            return
        self._uncount_tool_call(trajectory, SUBMIT_TOOL)
        trajectory.info[IDENTICAL_RESUBMISSIONS_KEY] = trajectory.info.get(IDENTICAL_RESUBMISSIONS_KEY, 0) + 1
        raise ToolCallRefused(IDENTICAL_RESUBMISSION_REPLY)

    @staticmethod
    def _accepted(info: dict[str, Any]) -> bool:
        """Whether the last graded submission passed every hidden test."""
        return "submission_result" in info and 0 < info.get("tests_total", 0) == info.get("tests_passed", 0)

    def _record_tool_interaction(self, results: list[NativeToolResult], trajectory: Trajectory) -> dict[str, Any]:
        """The protocol's record, counting a turn it flagged that held a starved run (``episode/starved_turns``)."""
        info = super()._record_tool_interaction(results, trajectory)
        if any(r.uninformative for r in results) and self._last_assistant_message(trajectory).calls_rejected:
            trajectory.info[STARVED_TURNS_KEY] = trajectory.info.get(STARVED_TURNS_KEY, 0) + 1
        return info

    def _step_single(
        self, trajectory: Trajectory, action: str, context: dict[str, Any] | None = None
    ) -> tuple[Trajectory, float, bool, bool, dict[str, Any]]:
        """Native tool step, then end the episode once the submission budget is spent or a submission
        passed every hidden test, after which a resubmission could only lose the solve.

        A submit_solution call is just a tool call, so without this the model keeps going after submitting.
        """
        trajectory, reward, done, truncated, info = super()._step_single(trajectory, action, context)
        # A turn that also booked a sandbox fault ends on it, uncompleted, whatever it submitted.
        spent = self._tool_budget_exhausted(trajectory, SUBMIT_TOOL) is not None
        if not done and (spent or self._accepted(trajectory.info)) and SANDBOX_FAULT_KEY not in trajectory.info:
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
            language=language,
        )

    def _reset_single(
        self,
        prompt: str | list[dict[str, str]],
        context: dict[str, Any] | None = None,
    ) -> Trajectory:
        """Build the base trajectory, then store the problem's grading data via the hook; a payload
        carrying no time limit gets the one each test runs under stated in the task message."""
        traj = super()._reset_single(prompt, context)
        self._store_problem_data(traj, context or {})
        if traj.info["_time_limit"] is None:
            limits = {name: self.grading_spec.time_limit_for(name, None) for name in self.languages}
            if len(set(limits.values())) == 1:
                limit_text = f"{limits[self.language]:g} s"
            else:
                limit_text = ", ".join(f"{limit:g} s in {name}" for name, limit in limits.items())
            traj.append_to_last_user(UNSTATED_TIME_LIMIT_NOTE.format(limits=limit_text))
        return traj

    def _apply_effort_profile(self, traj: Trajectory, level: str | None, profile: dict[str, int | float]) -> None:
        """Stamp the episode's interaction budgets — the level's, else the class caps — as the per-tool
        caps the protocol enforces. The task message states none of them: the level the chat template
        states and the engine's caps are what control an episode, and a refused call is told so."""
        max_subs = profile.get("max_submissions", self.max_submissions)
        max_tests = profile.get("max_test_calls", self.max_test_calls)
        traj.info[EPISODE_TOOL_BUDGETS_KEY].update({self.test_tool_name: max_tests, SUBMIT_TOOL: max_subs})

    @staticmethod
    def _parse_answer(context: dict[str, Any]) -> dict[str, Any] | list[Any]:
        """Return ``context["answer"]`` as a dict or list, decoding a JSON-string answer from datasets.

        Any other payload raises, failing the episode at reset: graded against no tests, it would
        score 0 inside its GRPO group, indistinguishable from a wrong solution.
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
            raise ValueError(
                f"{ANSWER_KEY!r} holds no tests: expected a non-empty list, or a non-empty 'tests'/'test_cases'"
            )

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

        objective = 1.0 if graded_content and self._accepted(info) else 0.0
        # Gated on graded_content, not submission_count: _submit bumps the count before grading.
        submission = self.submission_reward if graded_content else 0.0
        resubmission = -self.resubmission_penalty * max(0, self._submissions(trajectory) - 1)
        return EpisodeGrade(objective, {"submission": submission, "resubmission": resubmission})

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

    def rollout_metrics(self, trajectory: Trajectory) -> dict[str, float]:
        """CodeContests diagnostics: task outcome, submission behavior, and the reward decomposition."""
        metrics = super().rollout_metrics(trajectory)
        info = trajectory.info
        tests_total = info.get("tests_total", 0)
        graded = "submission_result" in info
        if graded and tests_total > 0:
            metrics["outcome/test_pass_frac"] = info.get("tests_passed", 0) / tests_total
            metrics[SOLVE_RATE_KEY] = 1.0 if self._accepted(info) else 0.0
        else:
            metrics["outcome/test_pass_frac"] = 0.0
            metrics[SOLVE_RATE_KEY] = 0.0
        submissions = self._submissions(trajectory)
        metrics["episode/submission_rate"] = 1.0 if submissions > 0 else 0.0
        metrics["episode/test_calls"] = float(self._test_calls(trajectory))
        metrics["episode/starved_test_runs"] = float(info.get(STARVED_TEST_RUNS_KEY, 0))
        metrics["episode/starved_turns"] = float(info.get(STARVED_TURNS_KEY, 0))
        metrics["episode/reasoning_in_comments_calls"] = float(info.get(REASONING_IN_COMMENTS_KEY, 0))
        metrics["episode/identical_resubmissions"] = float(info.get(IDENTICAL_RESUBMISSIONS_KEY, 0))
        program_chars = info.get(COMMENT_CHARS_KEY, 0) + info.get(CODE_CHARS_KEY, 0)
        if program_chars:
            metrics["episode/code_comment_share"] = info.get(COMMENT_CHARS_KEY, 0) / program_chars
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
