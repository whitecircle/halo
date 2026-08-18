"""Competitive-programming environment with hidden-test grading (Codeforces, APPS).

Models write a solution, test it with the REPL tool, then submit via ``submit_solution`` which runs it
against hidden tests through a SandboxExecutor. Reward = fraction of tests passed.
"""

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from src.environments.base import (
    EPISODE_INVALID_KEY,
    OBJECTIVE_REWARD_KEY,
    SOLVE_RATE_KEY,
    VALID_REASONING_EFFORTS,
    Trajectory,
    require_magnitudes,
)
from src.environments.envs.protocols.native import NativeToolUseEnvironment
from src.environments.envs.tasks.coding.grading import (
    DEFAULT_MAX_OUTPUT_SIZE,
    GradeResult,
    GradingSpec,
    grade_solution,
)
from src.environments.sandbox.base import SANDBOX_DEFAULT_TIMEOUT, SandboxExecutor, require_language
from src.environments.sandbox.repl import run_code_via_sandbox
from src.environments.sandbox.resolve import resolve_sandbox
from src.environments.tools.definitions import NativeTool, NativeToolRegistry, ToolBudgetExhausted, ToolParameter

logger = logging.getLogger(__name__)

# Per-asyncio-Task, so concurrent episodes reach their own trajectory from the submit handler.
_ACTIVE_TRAJECTORY: ContextVar = ContextVar("codecontests_active_trajectory", default=None)


# Effort level -> profile. ``thinking_tokens`` is the per-turn CoT budget; ``max_submissions``/
# ``max_test_calls`` are per-episode interaction budgets, so effort also buys iteration.
# ``token_cost`` prices total generated tokens (reward units per 1k), covering the thinking and
# visible channels together.
REASONING_EFFORT_PROFILES: dict[str, dict[str, int | float]] = {
    "low": {"thinking_tokens": 4096},
    "medium": {"thinking_tokens": 8192},
    "high": {"thinking_tokens": 16384},
}
DEFAULT_REASONING_EFFORT = "medium"

# Accepted profile keys and the minimum value each admits.
_PROFILE_KEY_MINIMA: dict[str, int] = {
    "thinking_tokens": 1,
    "max_submissions": 1,
    "max_test_calls": 0,
    "tested_submission_reward": 0,
    "token_cost": 0,
}
_PROFILE_KEYS = frozenset(_PROFILE_KEY_MINIMA)


class CodeContestsEnvironment(NativeToolUseEnvironment):
    """Competitive-programming environment with hidden-test grading (Codeforces, code_contests, APPS).

    ``language`` (``python``/``cpp``/``c``) drives both the test tool and grading through the same
    SandboxExecutor. Reward = (tests_passed / tests_total) * success_reward, credited only on
    ``submit_solution``; a never-submitted solution scores ``failure_reward``. Grading is data-driven:
    per-problem ``checker`` / ``time_limit`` from the ``answer`` payload (dict or JSON string, carrying
    ``tests``/``test_cases``) override the ``output_comparison`` default, so one env covers exact-match
    and Codeforces sets.
    """

    # A test/fix/submit loop needs more turns than the protocol's generic budget allows.
    DEFAULT_MAX_TURNS = 15

    # Per-tool shaping off so correctness dominates; configs may re-enable small values.
    DEFAULT_TOOL_SUCCESS_REWARD = 0.0
    DEFAULT_TOOL_ERROR_PENALTY = 0.0

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

    def __init__(
        self,
        timeout_per_test: float = SANDBOX_DEFAULT_TIMEOUT,
        max_output_size: int = DEFAULT_MAX_OUTPUT_SIZE,
        system_prompt: str | None = None,
        repl_timeout: float = SANDBOX_DEFAULT_TIMEOUT,
        sandbox: SandboxExecutor | None = None,
        sandbox_backend: str | None = None,
        sandbox_url: str | None = None,
        language: str = "python",
        output_comparison: str = "exact",
        stop_on_first_failure: bool = False,
        max_time_limit: float = SANDBOX_DEFAULT_TIMEOUT,
        max_grading_seconds: float | None = None,
        max_submissions: int = 2,
        max_test_calls: int = 5,
        submission_reward: float = 0.0,
        execution_progress_reward: float = 0.0,
        reasoning_effort: str = DEFAULT_REASONING_EFFORT,
        reasoning_effort_profiles: dict[str, dict[str, int | float]] | None = None,
        **kwargs,
    ):
        spec = require_language(language)
        if output_comparison not in ("exact", "tokens"):
            raise ValueError(f"output_comparison must be 'exact' or 'tokens', got {output_comparison!r}")
        if max_submissions < 1:
            raise ValueError(f"max_submissions must be >= 1, got {max_submissions}")
        if max_test_calls < 0:
            raise ValueError(f"max_test_calls must be >= 0, got {max_test_calls}")
        if max_grading_seconds is not None and max_grading_seconds <= 0:
            raise ValueError(f"max_grading_seconds must be > 0 or None, got {max_grading_seconds}")
        require_magnitudes(submission_reward=submission_reward, execution_progress_reward=execution_progress_reward)
        self.language = spec.name
        # Bootstraps a base model that never submits; the term cancels within a GRPO group once all do.
        self.submission_reward = submission_reward
        # Fraction of graded tests that ran: the only within-group signal when every completion fails.
        self.execution_progress_reward = execution_progress_reward
        # Reaching the cap ends the episode; further calls are rejected as tool errors.
        self.max_submissions = max_submissions
        self.max_test_calls = max_test_calls
        self.repl_timeout = repl_timeout
        # Effort levels also carry interaction budgets, so they differ by more than the CoT allowance.
        self.reasoning_effort_profiles = self._merge_profiles(reasoning_effort_profiles)
        self._profiles_bind_interaction = any(
            set(p) - {"thinking_tokens"} for p in self.reasoning_effort_profiles.values()
        )
        self.sandbox = sandbox or resolve_sandbox(backend=sandbox_backend, url=sandbox_url)
        # Built once, so every submission of the run is graded under the same settings; the offline
        # re-grader reproduces the verdicts from it via ``to_meta``. ``max_time_limit`` caps a
        # per-problem limit so a mis-scaled solution cannot pin a rollout worker, and
        # ``max_grading_seconds`` bounds the sequential test run of one submission, so a
        # several-hundred-test problem does not stall the round.
        self.grading_spec = GradingSpec(
            sandbox=self.sandbox,
            comparison=output_comparison,
            language=self.language,
            max_output_size=max_output_size,
            stop_on_first_failure=stop_on_first_failure,
            default_timeout=timeout_per_test,
            max_time_limit=max_time_limit,
            max_grading_seconds=max_grading_seconds,
        )

        test_tool = self._build_test_tool()

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
                name="submit_solution",
                description=(
                    f"Submit a complete {self.language} program to be graded against the problem's hidden "
                    f"tests — the only way to score this problem. The program must read from stdin and write "
                    f"to stdout. Returns how many tests passed. {submit_budget}"
                ),
                parameters=[
                    ToolParameter("code", "string", f"Complete {self.language} program reading stdin, writing stdout"),
                ],
                handler=self._submit,
            )
        )

        default_prompt = self.CODE_SYSTEM_PROMPT.format(language=self.language)
        super().__init__(
            tool_registry=registry,
            system_prompt=system_prompt or default_prompt,
            reasoning_effort=reasoning_effort,
            **kwargs,
        )

    def thinking_budget_for_effort(self, effort: str) -> int | None:
        """Bind a resolved effort level to its CoT budget; ``None`` for an unknown level (fall back to global)."""
        return self.reasoning_effort_profiles.get(effort, {}).get("thinking_tokens")

    @staticmethod
    def _merge_profiles(overrides: dict[str, dict[str, int | float]] | None) -> dict[str, dict[str, int | float]]:
        """Validate ``reasoning_effort_profiles`` overrides and merge them per level over the defaults."""
        profiles = {level: dict(entry) for level, entry in REASONING_EFFORT_PROFILES.items()}
        for level, entry in (overrides or {}).items():
            if level not in VALID_REASONING_EFFORTS:
                raise ValueError(
                    f"reasoning_effort_profiles level must be one of {VALID_REASONING_EFFORTS}, got {level!r}"
                )
            unknown = set(entry) - _PROFILE_KEYS
            if unknown:
                raise ValueError(
                    f"reasoning_effort_profiles[{level!r}] has unknown keys {sorted(unknown)}; "
                    f"allowed: {sorted(_PROFILE_KEYS)}"
                )
            for key, minimum in _PROFILE_KEY_MINIMA.items():
                if key in entry and entry[key] < minimum:
                    raise ValueError(f"{key} for effort {level!r} must be >= {minimum}, got {entry[key]}")
            profiles[level].update(entry)
        return profiles

    def _episode_budget(self, trajectory: Trajectory, name: str) -> int:
        """The episode's cap for ``name`` (``max_submissions``/``max_test_calls``): the per-effort
        profile stamp when present, else the class cap."""
        return trajectory.info.get(f"episode_{name}", getattr(self, name))

    def _build_test_tool(self) -> NativeTool:
        """Build the language-appropriate code-testing scratchpad tool.

        Runs through the same SandboxExecutor that grades ``submit_solution`` (isolated subprocess,
        not the in-process restricted REPL, which blocks imports). Sees no stdin and no graded tests.
        """
        name = "python_repl" if self.language == "python" else "run_code"
        verb = "Runs" if self.language == "python" else "Compiles and runs"
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
            f"{self.language} program you pass and returns its output; the standard library is available. "
            "It has no access to the graded tests and is given no stdin, so embed any input you want to try "
            f"directly in the code. {test_budget} Use submit_solution to be graded."
        )
        return NativeTool(
            name=name,
            description=description,
            parameters=[ToolParameter("code", "string", f"Complete {self.language} program")],
            handler=self._run_test,
        )

    def _run_test(self, code: str) -> str:
        """Run a scratchpad test, enforcing ``max_test_calls`` per episode.

        An over-cap call raises :class:`ToolBudgetExhausted`, so it classifies as a tool error
        (charged ``tool_error_penalty``, not paid ``tool_success_reward``) without the traceback a
        genuine tool fault gets. The cap is per-episode state, so a direct call with no active
        trajectory (a unit test, say) runs uncapped.
        """
        trajectory = _ACTIVE_TRAJECTORY.get()
        if trajectory is not None:
            cap = self._episode_budget(trajectory, "max_test_calls")
            if trajectory.info.get("test_call_count", 0) >= cap:
                raise ToolBudgetExhausted(
                    f"Test limit reached ({cap}); the scratchpad is exhausted. "
                    "Submit your solution with submit_solution."
                )
            trajectory.info["test_call_count"] = trajectory.info.get("test_call_count", 0) + 1
        return run_code_via_sandbox(code, sandbox=self.sandbox, timeout=self.repl_timeout, language=self.language)

    @contextmanager
    def _episode_binding(self, trajectory: Trajectory) -> Iterator[None]:
        """Bind the active trajectory so submit_solution grades against that episode's test cases."""
        token = _ACTIVE_TRAJECTORY.set(trajectory)
        try:
            yield
        finally:
            _ACTIVE_TRAJECTORY.reset(token)

    def _submit(self, code: str) -> str:
        """Grade a submission against the active episode's tests, enforcing ``max_submissions``.

        An over-cap call raises (a tool error, as in :meth:`_run_test`) without overwriting the
        graded result.
        """
        trajectory = _ACTIVE_TRAJECTORY.get()
        if trajectory is None:
            raise ValueError("submit_solution called outside an active episode.")

        cap = self._episode_budget(trajectory, "max_submissions")
        if trajectory.info.get("submission_count", 0) >= cap:
            raise ToolBudgetExhausted(f"Submission limit reached ({cap}); this submission is not graded.")

        trajectory.info["submission_count"] = trajectory.info.get("submission_count", 0) + 1
        if trajectory.info["submission_count"] == 1:
            # Ordering flag for the tested-submission bonus: whether a scratchpad run preceded this.
            trajectory.info["tested_before_submission"] = trajectory.info.get("test_call_count", 0) > 0
        grade = self._grade_submission(code, trajectory)
        trajectory.info["tests_passed"] = grade.passed
        trajectory.info["tests_total"] = grade.total
        trajectory.info["tests_ran_ok"] = grade.ran_ok
        trajectory.info["tests_infra_errors"] = grade.infra_errors
        trajectory.info["tests_graded"] = grade.graded
        trajectory.info["grading_budget_hit"] = grade.budget_hit
        trajectory.info["submission_result"] = grade.details
        return grade.details

    def _step_single(
        self, trajectory: Trajectory, action: str, context: dict[str, Any] | None = None
    ) -> tuple[Trajectory, float, bool, bool, dict[str, Any]]:
        """Native tool step, then end the episode once the submission budget is spent.

        A submit_solution call is an ordinary tool call, so without this the episode would continue
        after the budget is spent.
        """
        trajectory, reward, done, truncated, info = super()._step_single(trajectory, action, context)
        if not done and trajectory.info.get("submission_count", 0) >= self._episode_budget(
            trajectory, "max_submissions"
        ):
            trajectory.info["completed"] = True
            done = True
        return trajectory, reward, done, truncated, info

    def _grade_submission(self, code: str, trajectory: Trajectory) -> GradeResult:
        """Grade ``code`` against the episode's tests → :class:`GradeResult`."""
        return grade_solution(
            code,
            trajectory.info.get("_test_cases", []),
            self.grading_spec,
            checker=trajectory.info.get("_checker"),
            time_limit=trajectory.info.get("_time_limit"),
        )

    def _reset_single(
        self,
        prompt: str | list[dict[str, str]],
        context: dict[str, Any] | None = None,
    ) -> Trajectory:
        """Build the base trajectory, then store the problem's grading data via the hook."""
        traj = super()._reset_single(prompt, context)
        self._store_problem_data(traj, context or {})
        self._apply_interaction_budgets(traj, context or {})
        return traj

    def _apply_interaction_budgets(self, traj: Trajectory, context: dict[str, Any]) -> None:
        """Stamp the episode's interaction budgets and state the contract in the task message.

        Runs whenever any profile carries interaction keys, because the tool descriptions then defer
        to the task message, so the budgets must be stated on every episode. A level without
        interaction keys, or one undetermined at reset
        (:meth:`~src.environments.base.BaseEnvironment.reset_effort_level` returns ``None``), gets the
        class caps, stated explicitly rather than implied.
        """
        if not self._profiles_bind_interaction:
            return
        level = self.reset_effort_level(context)
        profile = self.reasoning_effort_profiles.get(level, {}) if level is not None else {}
        max_subs = profile.get("max_submissions", self.max_submissions)
        max_tests = profile.get("max_test_calls", self.max_test_calls)
        traj.info["episode_max_submissions"] = max_subs
        traj.info["episode_max_test_calls"] = max_tests
        # Deliberately absent from the stated budgets: it acts through the reward, not the prompt.
        traj.info["episode_tested_submission_reward"] = float(profile.get("tested_submission_reward", 0.0))
        # Charged by the trainer against total generated tokens; the env does not see token counts.
        traj.info["episode_token_cost"] = float(profile.get("token_cost", 0.0))
        last_wins = " (the last one is the graded result)" if max_subs > 1 else ""
        contract = (
            f"\n\nBudgets for this task: {max_subs} graded submission{'s' if max_subs != 1 else ''}"
            f"{last_wins}, {max_tests} scratchpad run{'s' if max_tests != 1 else ''}."
        )
        for message in reversed(traj.messages):
            if message.role == "user":
                message.content += contract
                break

    @staticmethod
    def _parse_answer(context: dict[str, Any]) -> Any:
        """Return ``context["answer"]`` as a Python object, decoding a JSON-string answer from datasets."""
        answer = context.get("answer", {})
        if isinstance(answer, str):
            try:
                return json.loads(answer)
            except ValueError:
                # An unparseable answer yields zero tests, which grades like a wrong solution; logged
                # so a malformed shard is visible rather than training as signal.
                logger.warning("Unparseable 'answer' payload (%d chars); grading with no tests", len(answer))
                return {}
        return answer

    def _store_problem_data(self, traj: Trajectory, context: dict[str, Any]) -> None:
        """Store the tests, optional checker, and time limit the submission is graded against.

        Accepts a bare list, ``{"test_cases": [...]}``, or the Codeforces
        ``{"tests": [...], "checker": ..., "time_limit": ...}``. Written to ``traj.info`` so
        concurrent Ray-rollout episodes do not overwrite each other; ``_submit`` reads it via the
        active-trajectory ContextVar.
        """
        answer = self._parse_answer(context)
        if isinstance(answer, dict):
            test_cases = answer.get("tests") or answer.get("test_cases") or []
            checker = answer.get("checker")
            time_limit = answer.get("time_limit")
        elif isinstance(answer, list):
            test_cases, checker, time_limit = answer, None, None
        else:
            test_cases, checker, time_limit = [], None, None

        traj.info["_test_cases"] = test_cases
        traj.info["_checker"] = checker
        traj.info["_time_limit"] = float(time_limit) if time_limit else None
        traj.info["test_cases_count"] = len(test_cases)
        traj.info["has_checker"] = checker is not None
        traj.info["tests_passed"] = 0
        traj.info["tests_total"] = len(test_cases)
        traj.info["submission_count"] = 0
        traj.info["test_call_count"] = 0

    @staticmethod
    def _grading_infra_outage(info: dict[str, Any]) -> bool:
        """True when the grade carries no signal: infra errors occurred and nothing ran or passed.

        Robust under ``stop_on_first_failure``, where an early backend error short-circuits grading.
        """
        return (
            info.get("tests_infra_errors", 0) > 0
            and info.get("tests_ran_ok", 0) == 0
            and info.get("tests_passed", 0) == 0
        )

    def _compute_reward(
        self,
        trajectory: Trajectory,
        context: dict[str, Any] | None = None,
    ) -> float:
        """Reward ladder over the objective (fraction of hidden tests passed by the submitted solution).

        ``submit_solution`` is the only graded channel; an unsubmitted solution scores
        ``failure_reward``. The shaping terms (submission_reward, execution_progress, tool-use)
        bootstrap the tool-use loop for a weak base model; each is small relative to
        ``success_reward`` and cancels within a GRPO group.
        """
        info = trajectory.info
        # No term may pay on a zero-test row or an all-infra-error grade: neither reflects the code.
        graded = "submission_result" in info
        tests_total = info.get("tests_total", 0)
        infra_outage = graded and tests_total > 0 and self._grading_infra_outage(info)
        graded_content = graded and tests_total > 0 and not infra_outage
        if infra_outage:
            # The backend failed, so the forced failure is not policy signal: drop the row from the
            # GRPO group baseline. A never-submitted or zero-test episode is not marked.
            info[EPISODE_INVALID_KEY] = True

        if graded_content:
            objective = (info.get("tests_passed", 0) / tests_total) * self.success_reward
        else:
            objective = self.failure_reward

        # Gated on graded_content, not submission_count: _submit bumps the count before grading.
        submission = self.submission_reward if graded_content else 0.0
        execution = (
            self.execution_progress_reward * (info.get("tests_ran_ok", 0) / tests_total) if graded_content else 0.0
        )
        # Paid once for testing before the first submission; a per-call constant could be earned repeatedly.
        tested = (
            info.get("episode_tested_submission_reward", 0.0)
            if graded_content and info.get("tested_before_submission")
            else 0.0
        )
        tool_shaping = self._tool_use_shaping(trajectory)

        # Components must sum exactly to the returned scalar; the trainer checks the composition residue.
        info["reward_components"] = {
            OBJECTIVE_REWARD_KEY: objective,
            "reward/submission": submission,
            "reward/execution": execution,
            "reward/tested_submission": tested,
            "reward/tool_shaping": tool_shaping,
            "reward/turn_shaping": trajectory.total_reward,
        }
        return self._shaped_base_reward(trajectory) + objective + submission + execution + tested

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
        metrics["episode/submission_rate"] = 1.0 if info.get("submission_count", 0) > 0 else 0.0
        metrics["episode/test_calls"] = float(info.get("test_call_count", 0))
        if info.get("submission_count", 0) > 0:
            # Mean over submitting episodes gives the test-first rate the tested-submission bonus targets.
            metrics["episode/tested_before_submission"] = 1.0 if info.get("tested_before_submission") else 0.0
        if graded and tests_total > 0:
            metrics["episode/grading_infra_outage"] = 1.0 if self._grading_infra_outage(info) else 0.0
            # Partial grading is invisible in the pass fraction, which keeps the full pool as its
            # denominator, so an ungraded remainder reads like a wrong solution.
            metrics["episode/tests_graded_frac"] = info.get("tests_graded", 0) / tests_total
            metrics["episode/grading_budget_hit"] = 1.0 if info.get("grading_budget_hit") else 0.0
        metrics.update(info.get("reward_components", {}))
        return metrics

    def _tool_use_engaged(self, trajectory: Trajectory) -> bool:
        """Gate multi_turn_reward on a real submission, so repeated test calls alone earn nothing."""
        return trajectory.info.get("submission_count", 0) > 0
