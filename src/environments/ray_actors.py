"""Ray actors for async environmental-GRPO rollout collection: each :class:`EnvironmentActor` plays episodes
against a rollout server's ``/v1/chat/completions`` (vLLM or SGLang), and :class:`RolloutManager` fans prompts
across actors and server URLs under a bounded semaphore. Weight sync is separate (:mod:`src.weight_sync`)."""

import asyncio
import inspect
import logging
import tempfile
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import cached_property, partial
from typing import Any
from urllib.parse import urlparse

import aiohttp
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from src.environments.engine_wire import build_payload, turn_from_chat_completion, with_cut_call_as_written
from src.environments.environment_base import EnvironmentBase
from src.environments.episode import (
    RolloutResult,
    TurnGeneration,
    describe_exception,
    generate_turn,
    is_terminal_client_status,
    play_episode,
)
from src.environments.reasoning_effort import EpisodeEffort, bind_episode_effort
from src.environments.registry import build_environment
from src.environments.rollout_config import RolloutConfig
from src.inference.endpoints import is_loopback

logger = logging.getLogger(__name__)

# The engine-paused seconds a deadline credits: the count, or an awaitable of it (a read of the actor's Ray copy).
_PausedClock = Callable[[], float | Awaitable[float]]

# Ray's plasma socket lives under the temp dir and AF_UNIX paths cap at ~107 bytes: a deep TMPDIR crashes ray.init.
_MAX_RAY_TEMP_DIR_LEN = 40
_RAY_FALLBACK_TEMP_DIR = "/tmp/ray"

# In-flight rollouts per actor when ``max_concurrent_rollouts`` is unset: actors multiplex episodes on one loop,
# so oversubscribing keeps the servers fed while a turn is graded.
_DEFAULT_ROLLOUTS_PER_WORKER = 4


class RolloutHTTPError(RuntimeError):
    """A non-200 from the rollout server, its status carried as data: the status decides retry versus abandon
    (:func:`_should_giveup`), where re-parsing the server-controlled message could misread a quoted status."""

    def __init__(self, status: int, backend: str, body: str):
        super().__init__(f"{backend} error (status {status}): {body}")
        self.status = status
        self.body = body


class DeadlineExpired(TimeoutError):
    """A deadline that ran out of engine-serving time; ``paused_seconds`` is the engine pause it credited."""

    def __init__(self, message: str, paused_seconds: float):
        super().__init__(message)
        self.paused_seconds = paused_seconds


def _is_shutdown_error(exc: BaseException) -> bool:
    """A benign teardown error (the session or loop closed at shutdown under an in-flight rollout), matched on
    the text aiohttp and Ray give it, since neither types it: retrying is pointless, and it logs quietly."""
    s = str(exc)
    return "Session is closed" in s or "Event loop is closed" in s or "INTENDED_USER_EXIT" in s


def _should_giveup(exc: BaseException) -> bool:
    """Backoff give-up predicate: a benign shutdown error, or a genuine client 4xx read off
    :class:`RolloutHTTPError`'s status, never its server-controlled message (429/408 and an engine fault retry)."""
    return (
        isinstance(exc, RolloutHTTPError) and is_terminal_client_status(exc.status, exc.body)
    ) or _is_shutdown_error(exc)


def _http_url(url: str) -> str:
    """A server URL as the client dials it: ``http://`` where it names no scheme."""
    return url if "://" in url else f"http://{url}"


async def _awaited(awaitable):
    """``asyncio`` task bodies must be coroutines; a Ray ``ObjectRef`` is only awaitable."""
    return await awaitable


async def _clock_reading(reading: float | Awaitable[float]) -> float:
    return await reading if inspect.isawaitable(reading) else reading


async def _await_with_deadline(
    awaitable, timeout: float, paused_clock: _PausedClock | None = None, *, what: str = "episode"
):
    """Await ``awaitable`` under ``timeout`` seconds of engine-serving time, raising :class:`DeadlineExpired`: the
    weight-sync pauses ``paused_clock`` reads are credited back, read only at the start and at expiry."""
    loop = asyncio.get_running_loop()
    started = loop.time()
    at_start = paused_clock() if paused_clock is not None else 0.0
    task = loop.create_task(_awaited(awaitable))
    credit = 0.0
    remaining = timeout
    try:
        while True:
            try:
                return await asyncio.wait_for(asyncio.shield(task), remaining)
            except TimeoutError:
                if task.done():
                    # Finished as the deadline fired: its own result, or its own TimeoutError.
                    return task.result()
            if paused_clock is not None:
                issued = loop.time()
                at_start = await _clock_reading(at_start)
                credit = max(0.0, await _clock_reading(paused_clock()) - at_start)
                remaining = timeout - (issued - started - credit)
            else:
                remaining = 0.0
            if remaining <= 0:
                if task.done():
                    # Finished while the clock was being read: the result stands.
                    return task.result()
                raise DeadlineExpired(
                    f"{what} exceeded its {timeout:.0f}s deadline ({credit:.0f}s of engine pause excluded)", credit
                )
    finally:
        if not task.done():
            task.cancel()


class EnginePauseClock:
    """Seconds the rollout engines have spent paused for weight syncs, an open window included. The trainer
    thread opens and closes windows while the episode loops read them, so the total and the open window's start
    live in one tuple, replaced whole: a read never pairs one state's total with another's window."""

    def __init__(self):
        self._state: tuple[float, float | None] = (0.0, None)

    def begin(self) -> None:
        self._state = (self._state[0], time.monotonic())

    def end(self, seconds: float) -> None:
        """Close the window, crediting ``seconds`` in place of the open window's own measure."""
        self._state = (self._state[0] + seconds, None)

    def paused_seconds(self) -> float:
        total, opened_at = self._state
        return total if opened_at is None else total + time.monotonic() - opened_at


# The copy the actor processes read. A sync actor runs one submitter's calls in order, so a window's
# begin never lands after its end; zero CPUs keeps it off the actor pool's slots.
_RemoteEnginePauseClock = ray.remote(num_cpus=0, max_restarts=-1)(EnginePauseClock)


@dataclass
class _SyncExpiries:
    """Request deadlines one episode saw expire on a request in flight across an engine pause."""

    count: int = 0

    def record(self, exc: BaseException) -> None:
        if isinstance(exc, DeadlineExpired) and exc.paused_seconds > 0:
            self.count += 1


def ray_init_kwargs(**overrides) -> dict:
    """Kwargs for ``ray.init``, with a short-tempdir fallback."""
    # Dashboard off: unused by the RL actors and exposes Ray's jobs-API HTTP surface
    # (CVE-2023-48022 job-submission RCE, CVE-2025-62593 DNS rebinding).
    kwargs: dict = {"ignore_reinit_error": True, "include_dashboard": False, **overrides}
    if len(tempfile.gettempdir()) > _MAX_RAY_TEMP_DIR_LEN:
        kwargs["_temp_dir"] = _RAY_FALLBACK_TEMP_DIR
    return kwargs


# Actors are stateless between episodes, so unlimited restarts are preferable to a permanent gap in
# the round-robin, where every episode routed to a dead actor returns a masked row.
@ray.remote(max_restarts=-1)
class EnvironmentActor:
    """A Ray actor playing environment episodes against the rollout server's ``/v1/chat/completions`` (chat
    template applied server-side); its env, HTTP session and tool schema are built on first use."""

    def __init__(
        self,
        actor_id: int,
        env_type: str | tuple[type, dict[str, Any]],
        env_config: dict[str, Any],
        pause_clock=None,
    ):
        """``pause_clock`` is the manager's Ray copy of its :class:`EnginePauseClock`; each request's deadline
        credits the pause it reads. Without one the deadline is wall-clock."""
        self.actor_id = actor_id
        self.env_type = env_type
        self.env_config = env_config
        self._paused_clock: _PausedClock | None = (
            None if pause_clock is None else lambda: pause_clock.paused_seconds.remote()
        )

    @cached_property
    def _env(self) -> EnvironmentBase:
        return build_environment(self.env_type, self.env_config)

    @cached_property
    def _tools_schema(self) -> list[dict] | None:
        """OpenAI-format tools list, or None if the env has no tools. Static per run."""
        return self._env.get_tools_schema()

    @cached_property
    def _http_client(self) -> aiohttp.ClientSession:
        # force_close: the server drops idle keepalives between batches, so a pooled socket
        # yields ServerDisconnectedError whose retries stall this rank at the next collective.
        connector = aiohttp.TCPConnector(force_close=True, enable_cleanup_closed=True)
        # No session timeout: aiohttp's would charge a weight-sync pause to the request, so each
        # request runs under the pause-aware deadline in _generate instead.
        return aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None), connector=connector)

    async def run_episode(
        self,
        prompt: str,
        context: dict[str, Any] | None,
        server_url: str,
        config: RolloutConfig,
    ) -> RolloutResult:
        """Run a complete multi-turn episode; a fault anywhere in it comes back as an errored row."""
        start = time.time()
        sync_expiries = _SyncExpiries()
        try:
            return await self._play_episode(prompt, context, server_url, config, start, sync_expiries)
        except Exception as e:
            # exc_info: the traceback is all that localizes a fault this catch-all turns into a masked row.
            logger.error(f"Actor {self.actor_id} error: {e}", exc_info=True)
            # The partial trajectory is dropped: ``error`` keeps the row out of the baseline and the step. Typed,
            # never ``str(e)``: a bare ``TimeoutError`` stringifies to "", which the mask reads as no error.
            return RolloutResult.errored(
                prompt,
                describe_exception(e),
                latency=time.time() - start,
                requests_expired_in_sync=sync_expiries.count,
            )

    async def _play_episode(
        self,
        prompt: str,
        context: dict[str, Any] | None,
        server_url: str,
        config: RolloutConfig,
        start: float,
        sync_expiries: _SyncExpiries,
    ) -> RolloutResult:
        """The episode played (:func:`play_episode`) under its bound effort, as the row the trainer reads."""
        env = self._env
        # Bound once per episode: every turn shares the level and its caps.
        effort = bind_episode_effort(
            context,
            env,
            max_tokens=config.max_tokens,
            max_thinking_tokens=config.max_thinking_tokens,
            max_episode_tokens=config.max_episode_tokens,
            max_answer_tokens=config.max_answer_tokens,
        )
        request = partial(self._generate_turn, self._http_client, server_url, sync_expiries)
        played = await play_episode(env, prompt, context, effort, request, config, log_prefix=f"Actor {self.actor_id}")
        traj = played.trajectory
        episode_metrics = env.rollout_metrics(traj) if traj else {}
        sampled = [turn.token_logprobs for turn in played.turns if turn.token_logprobs]
        if sampled:
            episode_metrics["logps/sampling_mean"] = sum(sum(lps) for lps in sampled) / sum(
                len(lps) for lps in sampled
            )
        return RolloutResult(
            prompt=prompt,
            trajectory=traj,
            episode_length=len(played.turns),
            total_reward=traj.total_reward if traj else 0.0,
            metrics=episode_metrics,
            # Natural terminal state, not a max_turns truncation; both set done=True.
            success=bool(traj and traj.done and not traj.truncated),
            latency=time.time() - start,
            generation_tokens=played.generated,
            requests_expired_in_sync=sync_expiries.count,
        )

    async def _generate_turn(
        self,
        client: aiohttp.ClientSession,
        server_url: str,
        sync_expiries: _SyncExpiries,
        messages: list[dict[str, str]],
        config: RolloutConfig,
        effort: EpisodeEffort,
    ) -> TurnGeneration:
        """One turn under the shared retry policy (:func:`generate_turn`): a genuine client 4xx or a shutdown error
        is terminal, any other transport or engine fault retried."""
        return await generate_turn(
            partial(
                self._generate,
                client,
                server_url,
                messages,
                config,
                effort.level,
                effort.thinking_budget,
                effort.budget_soft,
            ),
            config,
            retry_on=(TimeoutError, aiohttp.ClientError, RuntimeError),
            giveup=_should_giveup,
            log_prefix=f"Actor {self.actor_id}",
            on_failure=sync_expiries.record,
        )

    async def _generate(
        self,
        client: aiohttp.ClientSession,
        server_url: str,
        messages: list[dict[str, str]],
        config: RolloutConfig,
        reasoning_effort: str | None = None,
        reasoning_budget: int | None = None,
        budget_soft: bool = False,
    ) -> TurnGeneration:
        """One ``/v1/chat/completions`` request, and a ``/detokenize`` for a call cut at the cap, each under
        ``config.request_timeout`` of engine-serving time; raw aiohttp, since the server URL is chosen per call."""
        url = _http_url(server_url.rstrip("/"))
        payload = self._build_payload(messages, config, reasoning_effort, reasoning_budget, budget_soft)

        async def _exchange() -> dict:
            async with client.post(f"{url}/v1/chat/completions", json=payload) as resp:
                if resp.status != 200:
                    raise RolloutHTTPError(resp.status, config.backend, await resp.text())
                return await resp.json()

        async def _detokenize(ids: list[int]) -> str:
            async with client.post(f"{url}/detokenize", json={"model": payload.get("model"), "tokens": ids}) as resp:
                if resp.status != 200:
                    raise RolloutHTTPError(resp.status, config.backend, await resp.text())
                return (await resp.json())["prompt"]

        data = await _await_with_deadline(_exchange(), config.request_timeout, self._paused_clock, what="request")
        return await with_cut_call_as_written(
            turn_from_chat_completion(data, config),
            config,
            lambda ids: _await_with_deadline(
                _detokenize(ids), config.request_timeout, self._paused_clock, what="decode"
            ),
            f"Actor {self.actor_id}",
        )

    def _build_payload(
        self,
        messages: list[dict],
        config: RolloutConfig,
        reasoning_effort: str | None = None,
        reasoning_budget: int | None = None,
        budget_soft: bool = False,
    ) -> dict[str, Any]:
        """The turn's request payload, with this actor's env tool schema bound onto the wire format."""
        return build_payload(messages, config, reasoning_effort, self._tools_schema, reasoning_budget, budget_soft)

    async def shutdown(self):
        """Close the HTTP session and the environment (idempotent; called by RolloutManager.shutdown)."""
        client = self.__dict__.pop("_http_client", None)
        if client is not None:
            await client.close()
        env = self.__dict__.pop("_env", None)
        if env is not None:
            try:
                env.close()
            except Exception:  # a failed teardown must not block the rest of shutdown
                logger.warning("Actor %s: environment close failed", self.actor_id, exc_info=True)


class RolloutManager:
    """Coordinate parallel rollout collection across Ray actors and rollout servers: bounded
    concurrency (asyncio.Semaphore) + round-robin scheduling over both actors and server URLs."""

    def __init__(
        self,
        num_workers: int,
        env_type: str | tuple[type, dict[str, Any]],
        env_config: dict[str, Any],
        server_urls: list[str],
        rollout_config: RolloutConfig,
        max_concurrent_rollouts: int | None = None,
        pool_replicas: int = 1,
    ):
        """``num_workers`` is the actor budget for the whole Ray cluster, divided among the ``pool_replicas`` peers
        that build a pool against it (the DP size on a shared cluster, else 1)."""
        self.num_workers = max(1, num_workers // max(1, pool_replicas))
        self.env_type = env_type
        self.env_config = env_config
        self.server_urls = server_urls
        self.rollout_config = rollout_config

        requested = max_concurrent_rollouts or self.num_workers * _DEFAULT_ROLLOUTS_PER_WORKER
        if requested < self.num_workers:
            logger.warning(
                f"max_concurrent_rollouts ({requested}) < num_workers ({self.num_workers}), "
                f"clamping to {self.num_workers}"
            )
            requested = self.num_workers
        self.max_concurrent = requested

        self._actors: list = []
        self._started = False
        self._actor_idx = 0
        self._url_idx = 0
        # Credited back to every in-flight episode's deadline; the episodes' loop reads it. The actors
        # run in their own processes, so their requests read a Ray copy this manager feeds alongside.
        self._pause_clock = EnginePauseClock()
        self._actor_pause_clock = None

        logger.info(
            f"RolloutManager: {self.num_workers} workers, "
            f"{len(server_urls)} rollout servers, max_concurrent={self.max_concurrent}"
        )

    def begin_engine_pause(self) -> None:
        """Mark the engines paused for a weight sync: in-flight episode and request deadlines stop counting."""
        self._pause_clock.begin()
        if self._actor_pause_clock is not None:
            self._actor_pause_clock.begin.remote()

    def end_engine_pause(self, seconds: float) -> None:
        """Close the pause window, crediting the ``seconds`` the forwarding rank measured for its push in place of
        this rank's own estimate: the pause is the servers', timed where the push ran."""
        self._pause_clock.end(seconds)
        if self._actor_pause_clock is not None:
            self._actor_pause_clock.end.remote(seconds)

    @property
    def paused_seconds(self) -> float:
        """Engine-paused seconds so far, an open window included."""
        return self._pause_clock.paused_seconds()

    def warn_if_servers_unreachable_from_actors(self, multinode: bool) -> None:
        """Warn when a rollout-server URL is loopback on a multi-node job: placement only prefers this node, so an
        actor elsewhere reaches no engine unless one runs on every node."""
        urls = self.server_urls
        hosts = [urlparse(_http_url(u)).hostname or "" for u in urls]
        if not multinode or not any(is_loopback(host) for host in hosts):
            return
        backend = self.rollout_config.backend
        logger.warning(
            "Multi-node async GRPO but a %s URL is loopback (%s). Ray actors may be scheduled on nodes where "
            "no %s server listens on localhost; their connection-refused rollouts become error rows masked out "
            "of the loss and the group baseline, shrinking the effective batch. Use resolvable host IPs/DNS in "
            "rollout_server_url / rollout_server_configs.",
            backend,
            urls,
            backend,
        )

    async def start(self):
        """Init Ray (if needed) and spawn the actor pool, preferring this node."""
        if self._started:
            return
        if not ray.is_initialized():
            # Local-mode fallback for direct library use: on every entry-script path the trainer's
            # _init_async_components has already initialized Ray with the configured ray_address.
            ray.init(**ray_init_kwargs())
        # Prefer this node, or a shared cluster scatters this rank's actors. `soft` alone pins an actor to a live
        # but saturated node indefinitely; `_spill_on_unavailable` makes the affinity a preference.
        strategy = NodeAffinitySchedulingStrategy(
            node_id=ray.get_runtime_context().get_node_id(), soft=True, _spill_on_unavailable=True
        )
        self._actor_pause_clock = _RemoteEnginePauseClock.options(scheduling_strategy=strategy).remote()
        actor_cls = EnvironmentActor.options(scheduling_strategy=strategy)
        self._actors = [
            actor_cls.remote(i, self.env_type, self.env_config, self._actor_pause_clock)
            for i in range(self.num_workers)
        ]
        self._started = True
        logger.info(f"Started {len(self._actors)} environment actors")

    async def shutdown(self):
        """Shut down every actor (closing envs and HTTP sessions) and drop the pool."""
        if not self._started:
            return
        for actor in self._actors:
            try:
                await actor.shutdown.remote()
            except Exception as e:
                logger.warning(f"Error shutting down actor: {e}", exc_info=True)
        if self._actor_pause_clock is not None:
            ray.kill(self._actor_pause_clock)
            self._actor_pause_clock = None
        self._actors = []
        self._started = False
        logger.info("RolloutManager shutdown complete")

    def _next_actor(self):
        actor = self._actors[self._actor_idx % len(self._actors)]
        self._actor_idx += 1
        return actor

    def _next_url(self) -> str:
        url = self.server_urls[self._url_idx % len(self.server_urls)]
        self._url_idx += 1
        return url

    async def collect_rollouts(
        self,
        prompts: list[str],
        contexts: list[dict[str, Any]] | None = None,
    ) -> list[RolloutResult]:
        """Collect rollouts for a batch of prompts (ordered results)."""
        if not self._started:
            await self.start()
        if contexts is None:
            contexts = [None] * len(prompts)

        sem = asyncio.Semaphore(self.max_concurrent)
        results: list[RolloutResult | None] = [None] * len(prompts)
        errors: list[Exception | None] = [None] * len(prompts)

        async def _run(idx: int, prompt: str, ctx: dict | None):
            async with sem:
                ref = None
                try:
                    ref = self._next_actor().run_episode.remote(
                        prompt=prompt,
                        context=ctx,
                        server_url=self._next_url(),
                        config=self.rollout_config,
                    )
                    results[idx] = await _await_with_deadline(
                        ref, self.rollout_config.episode_timeout, lambda: self.paused_seconds
                    )
                except Exception as e:
                    if isinstance(e, TimeoutError):
                        # Ray forbids force=True on an async-actor task, so it would never cancel.
                        ray.cancel(ref, force=False)
                    self._log_failure(idx, e)
                    errors[idx] = e

        # Paired before the first task starts, so a length mismatch submits no episode.
        episodes = list(zip(prompts, contexts, strict=True))
        tasks = [asyncio.create_task(_run(i, p, c)) for i, (p, c) in enumerate(episodes)]
        await asyncio.gather(*tasks, return_exceptions=True)

        # Typed, never ``str(e)``: the deadline's bare ``TimeoutError`` stringifies to "", and an empty
        # ``error`` is a VALID zero-reward group member to ``rollout_valid_mask``.
        return [
            result
            if result is not None
            else RolloutResult.errored(prompt, describe_exception(error) if error is not None else "Unknown error")
            for prompt, result, error in zip(prompts, results, errors, strict=True)
        ]

    def _log_failure(self, idx: int, error: Exception) -> None:
        """Log a rollout lost on the manager's side (its deadline, a shutdown, anything else) by its cause."""
        if isinstance(error, TimeoutError):
            logger.error(f"Rollout {idx} exceeded the {self.rollout_config.episode_timeout}s deadline")
        elif _is_shutdown_error(error):
            logger.debug(f"Rollout {idx} abandoned at shutdown: {error}")
        else:
            # Same reasoning as the actor-side catch-all: an unclassified failure costs
            # the whole rollout, and the traceback is what localizes it.
            logger.error(f"Rollout {idx} failed: {error}", exc_info=True)
