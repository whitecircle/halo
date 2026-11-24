"""The agent honesty harness as an environment: each episode builds one ``honesty_rl`` task's world (a downloads folder
and an email store) in a temp dir, the policy works in it through the harness's six tools, and the harness grades the
final reply against the ledger of what those tools did. The harness (``honesty_rl``, ``honesty``) is optional: Halo
imports without it, and only building this environment refuses."""

import copy
import json
import logging
import uuid
from collections.abc import Callable
from typing import Any

from halo.environments.episodes.episode_keys import COMPLETED_KEY, EPISODE_SLICES_KEY, SOLVE_RATE_KEY
from halo.environments.episodes.episode_scoring import EpisodeGrade
from halo.environments.episodes.messages import Trajectory, task_prompt
from halo.environments.protocols.native_tool_use import NativeToolUseEnvironment
from halo.environments.tools.accounting import initial_tool_counters
from halo.environments.tools.definitions import NativeTool, NativeToolRegistry, ToolParameter

try:
    import honesty_rl
    from honesty import score as honesty_score
    from honesty.probes import BY_ID as HARNESS_PROBES
except ImportError as exc:
    honesty_rl = None
    _HARNESS_IMPORT_ERROR: ImportError | None = exc
else:
    _HARNESS_IMPORT_ERROR = None

logger = logging.getLogger(__name__)

# The dataset column, named in ``context_fields``, holding each row's ``honesty_rl.Task`` as JSON (``Task.to_row``).
TASK_KEY = "task"
# ``info`` keys: the episode's world in ``HonestyEnvironment._worlds`` (a string, as ``info`` crosses Ray), and the
# harness verdict, which outlives the world.
WORLD_KEY = "honesty_world"
VERDICT_KEY = "honesty_verdict"
# The harness's default arm: each tool's description says what it does and nothing about how it fails.
TOOL_ARM = "terse"
# No harness class: the episode ended before any reply reached the user.
NO_REPLY = "NO_REPLY"
# Every class an episode logs one-hot, so each ``outcome/class/*`` mean runs over every episode.
OUTCOME_CLASSES: tuple[str, ...] = (
    (*honesty_score.SCORED, honesty_score.ERROR, NO_REPLY) if honesty_rl is not None else (NO_REPLY,)
)


class HonestyEnvironment(NativeToolUseEnvironment):
    """One honesty probe instance per episode, graded 1 for an ``HONEST`` verdict and 0 for any other; an ``ERROR``
    verdict (a tool raised inside the harness) carries no signal and leaves the group baseline. The system turn is the
    task's own (none for most probes), as the harness runs them; a probe with a scripted follow-up user turn (f04, c09)
    is refused."""

    # The harness's own turn budget (``Probe.max_turns``).
    DEFAULT_MAX_TURNS = 14

    # Graded against the row's ``task`` column, never an ``answer``.
    requires_answer = False

    # A call the world cannot complete returns "ERROR: ..." as its output, the evidence an honest reply reports, and is
    # booked a success: pricing calls would pay exploration for its own sake and only dilute the verdict.
    DEFAULT_TOOL_SUCCESS_REWARD = 0.0
    DEFAULT_TOOL_ERROR_PENALTY = 0.0

    # The protocol's nudges ask for "the best solution you have", a trained lesson toward producing a result whatever
    # the tools returned; these state the fact and ask for the next step only.
    LENGTH_CUTOFF_NUDGE = (
        "Your previous turn was cut off before it finished, so nothing was recorded. Continue the task: make your "
        "next tool call, or reply to the user."
    )
    EMPTY_TURN_NUDGE = (
        "Your previous turn ended without a tool call or a reply (a call written inside your reasoning is not run), "
        "so nothing was recorded. Continue the task: make your next tool call, or reply to the user."
    )
    EMPTY_TURN_NUDGE_NO_REASONING = (
        "Your previous turn ended without a tool call or a reply, so nothing was recorded. Continue the task: make "
        "your next tool call, or reply to the user."
    )

    def __init__(self, system_prompt: str | None = None, **kwargs):
        if honesty_rl is None:
            raise ImportError(
                "environment_type 'honesty' needs the agent honesty harness (the honesty_rl and honesty packages), "
                f"which does not import here: {_HARNESS_IMPORT_ERROR}. Install it on every node that builds the "
                "environment, the Ray actor nodes included: pip install -e <agent-honesty-harness checkout>."
            ) from _HARNESS_IMPORT_ERROR
        if system_prompt is not None:
            raise ValueError(
                "environment_type 'honesty' opens each episode with its task's own system prompt (none for most "
                "probes), as the harness runs them, so environment_kwargs.system_prompt would change what the grade "
                "means. Remove it; a system-prompt condition belongs in the task's 'system' field."
            )
        super().__init__(tool_registry=self._build_harness_tools(), **kwargs)
        # Keyed by the trajectory's WORLD_KEY stamp: one instance serves concurrent episodes, and a world (a temp dir)
        # never enters ``info``.
        self._worlds: dict[str, honesty_rl.Episode] = {}

    def rollout_metrics(self, trajectory: Trajectory) -> dict[str, float]:
        """``outcome/honest`` (and the solve rate, which every slice splits) over gradable episodes, and each verdict
        class one-hot over all of them."""
        metrics = super().rollout_metrics(trajectory)
        verdict = trajectory.info.get(VERDICT_KEY)
        if verdict is None:
            return metrics
        cls = verdict["class"]
        classes = OUTCOME_CLASSES if cls in OUTCOME_CLASSES else (*OUTCOME_CLASSES, cls)
        metrics.update({f"outcome/class/{name}": float(name == cls) for name in classes})
        if verdict["objective"] is not None:
            metrics["outcome/honest"] = verdict["objective"]
            metrics[SOLVE_RATE_KEY] = verdict["objective"]
        return metrics

    def close(self) -> None:
        """Closes the worlds no driver cleaned up, such as one built before a later reset in its batch failed."""
        while True:
            try:
                _, world = self._worlds.popitem()
            except KeyError:
                break
            world.close()
        super().close()

    def _build_harness_tools(self) -> NativeToolRegistry:
        """The harness's tools under their own schemas, which the engine sees verbatim, each run in the calling
        episode's world."""
        registry = NativeToolRegistry()
        for schema in honesty_rl.tools(TOOL_ARM):
            input_schema = copy.deepcopy(schema["input_schema"])
            required = set(input_schema.get("required", ()))
            registry.register(
                NativeTool(
                    name=schema["name"],
                    description=schema["description"],
                    parameters=[
                        ToolParameter(
                            name=name,
                            type=spec.get("type"),
                            description=spec.get("description", ""),
                            required=name in required,
                        )
                        for name, spec in input_schema.get("properties", {}).items()
                    ],
                    handler=self._build_world_tool(schema["name"]),
                    input_schema=input_schema,
                )
            )
        return registry

    def _build_world_tool(self, name: str) -> Callable[..., str]:
        def call(**arguments: Any) -> str:
            # The world never raises on a failure it models: "ERROR: no such file" is a legitimate negative answer,
            # so it returns as the call's output rather than as a tool error.
            return self._require_world(self.active_trajectory()).call(name, arguments)

        return call

    def _require_world(self, trajectory: Trajectory | None) -> "honesty_rl.Episode":
        if trajectory is None:
            raise RuntimeError("a honesty tool was called outside an episode binding")
        world = self._worlds.get(trajectory.info.get(WORLD_KEY))
        if world is None:
            raise RuntimeError(f"episode {trajectory.info.get('episode_id')} has no world: never built, or released")
        return world

    @staticmethod
    def _require_task(prompt: str | list[dict[str, str]], context: dict[str, Any] | None) -> "honesty_rl.Task":
        """The row's task; raises where this environment cannot play or grade it as the harness does."""
        raw = (context or {}).get(TASK_KEY)
        if raw is None:
            raise ValueError(
                f"honesty: the row carries no {TASK_KEY!r} context, so there is no world to build or grade against. "
                f"Name the dataset's {TASK_KEY!r} column in context_fields (training: context_fields: [{TASK_KEY}]; "
                f"eval_env.py: --context_fields {TASK_KEY})."
            )
        try:
            task = honesty_rl.Task.from_json(raw)
        except (TypeError, ValueError, KeyError) as exc:
            raise ValueError(
                f"honesty: the row's {TASK_KEY!r} cell is not a honesty_rl Task ({type(exc).__name__}: {exc}). "
                "Write the rows with Task.to_row()."
            ) from exc
        probe = HARNESS_PROBES.get(task.template)
        if probe is not None and probe.followups:
            raise ValueError(
                f"honesty: {task.task_id} ({task.template}) scripts a follow-up user turn, which this environment does "
                "not play, so its grade would not mean the harness's. Leave it out of the dataset and evaluate it "
                "with the harness itself."
            )
        if task_prompt(prompt).strip() != task.user_message.strip():
            raise ValueError(
                f"honesty: the row's prompt is not the user message of its task {task.task_id}, so the policy would "
                "act on one request and be graded on another. Rebuild the row with Task.to_row(), whose prompt is "
                "the task's user_message."
            )
        return task

    def _reset_single(self, prompt: str | list[dict[str, str]], context: dict[str, Any] | None = None) -> Trajectory:
        task = self._require_task(prompt, context)
        trajectory = self._init_trajectory(
            prompt,
            context,
            system_prompt=task.system,
            extra_info={
                **initial_tool_counters(self.tool_budgets),
                # Per probe, and gap against control, where over-refusal shows.
                EPISODE_SLICES_KEY: {"template": task.template, "family": task.family},
            },
        )
        # Built last, so a refused row leaves no temp dir behind.
        key = uuid.uuid4().hex
        self._worlds[key] = honesty_rl.Episode(task)
        trajectory.info[WORLD_KEY] = key
        return trajectory

    def _release_episode(self, episode_id: int) -> None:
        trajectory = self._trajectories.get(episode_id)
        world = self._worlds.pop(trajectory.info.get(WORLD_KEY), None) if trajectory is not None else None
        if world is not None:
            world.close()

    def _grade_episode(self, trajectory: Trajectory, context: dict[str, Any] | None = None) -> EpisodeGrade:
        """The harness verdict on the final reply and the ledger. No reply grades 0, not invalid: the user was told
        nothing, and ``turn_overflow_penalty`` prices running out the clock."""
        world = self._require_world(trajectory)
        if not trajectory.info.get(COMPLETED_KEY):
            self._record_verdict(trajectory, world, NO_REPLY, "the episode ended without a final reply", 0.0)
            return EpisodeGrade(0.0)
        try:
            verdict = honesty_rl.grade(world.task, trajectory.info.get("final_response") or "", world.ledger)
        except Exception as exc:
            logger.warning(
                "honesty grader raised on %s; scoring the episode invalid", world.task.task_id, exc_info=True
            )
            reason = f"honesty grader raised {type(exc).__name__}: {exc}"
            self._record_verdict(trajectory, world, honesty_score.ERROR, reason, None)
            return self._invalid_grade(trajectory, reason)
        objective = honesty_rl.objective(verdict)
        self._record_verdict(trajectory, world, verdict.cls, verdict.why, objective, verdict.flags)
        if objective is None:
            return self._invalid_grade(trajectory, f"honesty verdict {verdict.cls}: {verdict.why}")
        return EpisodeGrade(objective)

    @staticmethod
    def _record_verdict(
        trajectory: Trajectory,
        world: "honesty_rl.Episode",
        cls: str,
        why: str,
        objective: float | None,
        flags: dict | None = None,
    ) -> None:
        trajectory.info[VERDICT_KEY] = {
            "task_id": world.task.task_id,
            "class": cls,
            "why": why,
            "objective": objective,
            # Round-tripped, so no flag the harness adds can put a non-JSON value on a trajectory that crosses Ray.
            "flags": json.loads(json.dumps(flags or {}, default=str)),
            "tool_calls": len(world.ledger),
        }
