"""Ray actors for async environmental-GRPO rollouts: :class:`EnvironmentActor` plays episodes against a rollout server,
and :class:`RolloutManager` fans prompts across actors and server URLs under a bounded semaphore."""

import asyncio
import hashlib
import inspect
import logging
import os
import tempfile
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import cached_property, partial
from typing import Any
from urllib.parse import urlparse

import aiohttp
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from halo.environments.engine_wire import build_payload, turn_from_chat_completion, with_cut_call_as_written
from halo.environments.environment_base import EnvironmentBase
from halo.environments.episode import (
    RolloutResult,
    TurnGeneration,
    describe_exception,
    generate_turn,
    is_terminal_client_status,
    play_episode,
)
from halo.environments.reasoning_effort import EpisodeEffort, bind_episode_effort
from halo.environments.registry import build_environment
from halo.environments.rollout_config import RolloutConfig
from halo.inference.endpoints import is_loopback

logger = logging.getLogger(__name__)

# Engine-paused seconds, or an awaitable of them (a read of the actor's Ray copy).
_PausedClock = Callable[[], float | Awaitable[float]]

# Ray's plasma socket lives under the temp dir and AF_UNIX paths cap at ~107 bytes: a deep TMPDIR crashes ray.init.
_MAX_RAY_TEMP_DIR_LEN = 40
_RAY_LINK_PARENT = "/tmp"

# Actors multiplex episodes on one loop, so oversubscribing keeps the servers fed while a turn is graded.
_DEFAULT_ROLLOUTS_PER_WORKER = 4


class RolloutHTTPError(RuntimeError):
    """A non-200, status carried as data: re-parsing the server-controlled message could misread a quoted status."""

    def __init__(self, status: int, backend: str, body: str):
        super().__init__(f"{backend} error (status {status}): {body}")
        self.status = status
        self.body = body


class DeadlineExpired(TimeoutError):
    """``paused_seconds`` is the engine pause the deadline credited."""

    def __init__(self, message: str, paused_seconds: float):
        super().__init__(message)
        self.paused_seconds = paused_seconds


def _is_shutdown_error(exc: BaseException) -> bool:
    """A benign teardown error, matched on text since neither aiohttp nor Ray types it; retrying is pointless."""
    s = str(exc)
    return "Session is closed" in s or "Event loop is closed" in s or "INTENDED_USER_EXIT" in s


def _should_giveup(exc: BaseException) -> bool:
    """Backoff give-up: a shutdown error or a genuine client 4xx (429/408 and an engine fault retry)."""
    return (
        isinstance(exc, RolloutHTTPError) and is_terminal_client_status(exc.status, exc.body)
    ) or _is_shutdown_error(exc)


def _http_url(url: str) -> str:
    return url if "://" in url else f"http://{url}"


async def _awaited(awaitable):
    """``asyncio`` task bodies must be coroutines; a Ray ``ObjectRef`` is only awaitable."""
    return await awaitable


async def _clock_reading(reading: float | Awaitable[float]) -> float:
    return await reading if inspect.isawaitable(reading) else reading


async def _await_with_deadline(
    awaitable, timeout: float, paused_clock: _PausedClock | None = None, *, what: str = "episode"
):
    """Await under ``timeout`` seconds of engine-serving time, raising :class:`DeadlineExpired`. The weight-sync pauses
    ``paused_clock`` reads are credited back, read only at the start and at expiry."""
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
    """Seconds the engines spent paused for weight syncs, an open window included. The trainer thread writes while the
    episode loops read, so total and window start live in one tuple replaced whole: a read never mixes two states."""

    def __init__(self):
        self._state: tuple[float, float | None] = (0.0, None)

    def begin(self) -> None:
        self._state = (self._state[0], time.monotonic())

    def end(self, seconds: float) -> None:
        """Credit ``seconds`` in place of the open window's own measure."""
        self._state = (self._state[0] + seconds, None)

    def paused_seconds(self) -> float:
        total, opened_at = self._state
        return total if opened_at is None else total + time.monotonic() - opened_at


# A sync actor runs one submitter's calls in order, so a window's begin never lands after its end; zero CPUs keeps it
# off the actor pool's slots.
_RemoteEnginePauseClock = ray.remote(num_cpus=0, max_restarts=-1)(EnginePauseClock)


@dataclass
class _SyncExpiries:
    """Request deadlines one episode saw expire across an engine pause."""

    count: int = 0

    def record(self, exc: BaseException) -> None:
        if isinstance(exc, DeadlineExpired) and exc.paused_seconds > 0:
            self.count += 1


def _short_ray_temp_dir() -> str | None:
    """A short ``_temp_dir`` for a temp root too deep for Ray's socket paths: a link under ``/tmp`` to ``<root>/ray``,
    so logs and spilled objects stay on the root's volume. ``None`` when the root is short enough."""
    root = os.environ.get("RAY_TMPDIR") or tempfile.gettempdir()
    if len(root) <= _MAX_RAY_TEMP_DIR_LEN:
        return None
    target = os.path.join(root, "ray")
    os.makedirs(target, exist_ok=True)
    link = os.path.join(_RAY_LINK_PARENT, f"halo-ray-{os.getuid()}-{hashlib.md5(target.encode()).hexdigest()[:8]}")
    staged = f"{link}.{uuid.uuid4().hex}"
    os.symlink(target, staged)
    os.replace(staged, link)  # atomic, so concurrent launches over one root agree on the link
    return link


def ray_init_kwargs(**overrides) -> dict:
    # Dashboard off: unused by the RL actors and exposes Ray's jobs-API HTTP surface
    # (CVE-2023-48022 job-submission RCE, CVE-2025-62593 DNS rebinding).
    kwargs: dict = {"ignore_reinit_error": True, "include_dashboard": False, **overrides}
    temp_dir = _short_ray_temp_dir()
    if temp_dir is not None:
        kwargs["_temp_dir"] = temp_dir
    return kwargs


# Stateless between episodes; a permanent gap in the round-robin would mask every episode routed to the dead actor.
@ray.remote(max_restarts=-1)
class EnvironmentActor:
    """Plays episodes against a rollout server; its env, HTTP session and tool schema are built on first use."""

    def __init__(
        self,
        actor_id: int,
        env_type: str | tuple[type, dict[str, Any]],
        env_config: dict[str, Any],
        pause_clock=None,
    ):
        """``pause_clock``: a Ray copy of the manager's :class:`EnginePauseClock`; none means wall-clock deadlines."""
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
        return self._env.get_tools_schema()

    @cached_property
    def _http_client(self) -> aiohttp.ClientSession:
        # force_close: the server drops idle keepalives between batches, so a pooled socket
        # yields ServerDisconnectedError whose retries stall this rank at the next collective.
        connector = aiohttp.TCPConnector(force_close=True, enable_cleanup_closed=True)
        # No session timeout: aiohttp's would charge a weight-sync pause; _generate's deadline credits it.
        return aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=None), connector=connector)

    async def run_episode(
        self,
        prompt: str,
        context: dict[str, Any] | None,
        server_url: str,
        config: RolloutConfig,
    ) -> RolloutResult:
        """A fault anywhere comes back as an errored row."""
        start = time.time()
        sync_expiries = _SyncExpiries()
        try:
            return await self._play_episode(prompt, context, server_url, config, start, sync_expiries)
        except Exception as e:
            # exc_info: the traceback is all that localizes a fault this catch-all turns into a masked row.
            logger.error(f"Actor {self.actor_id} error: {e}", exc_info=True)
            # ``error`` keeps the row out of the baseline and the step. Typed, never ``str(e)``: a bare
            # ``TimeoutError`` stringifies to "", which the mask reads as no error.
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
            terminated_naturally=bool(traj and traj.done and not traj.truncated),
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
        """A genuine client 4xx or a shutdown error is terminal; any other transport or engine fault retries."""
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
        """One chat-completions request, and a ``/detokenize`` for a call cut at the cap, each under
        ``config.request_timeout`` of engine-serving time. Raw aiohttp, since the server URL is chosen per call."""
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
        return build_payload(messages, config, reasoning_effort, self._tools_schema, reasoning_budget, budget_soft)

    async def shutdown(self):
        """Idempotent; called by RolloutManager.shutdown."""
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
    """Parallel rollout collection: bounded concurrency plus round-robin over both actors and server URLs."""

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
        building a pool against it (the DP size on a shared cluster, else 1)."""
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
        # Credited back to in-flight deadlines. Actors in their own processes read a Ray copy fed alongside.
        self._pause_clock = EnginePauseClock()
        self._actor_pause_clock = None

        logger.info(
            f"RolloutManager: {self.num_workers} workers, "
            f"{len(server_urls)} rollout servers, max_concurrent={self.max_concurrent}"
        )

    def begin_engine_pause(self) -> None:
        """In-flight episode and request deadlines stop counting."""
        self._pause_clock.begin()
        if self._actor_pause_clock is not None:
            self._actor_pause_clock.begin.remote()

    def end_engine_pause(self, seconds: float) -> None:
        """Credit the ``seconds`` the forwarding rank measured, not this rank's estimate: the pause is the servers',
        timed where the push ran."""
        self._pause_clock.end(seconds)
        if self._actor_pause_clock is not None:
            self._actor_pause_clock.end.remote(seconds)

    @property
    def paused_seconds(self) -> float:
        return self._pause_clock.paused_seconds()

    def warn_if_servers_unreachable_from_actors(self, multinode: bool) -> None:
        """Warn on a loopback URL in a multi-node job: placement only prefers this node, so an actor elsewhere reaches
        no engine unless one runs on every node."""
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
        if self._started:
            return
        if not ray.is_initialized():
            # Local-mode fallback for direct library use; entry scripts have already initialized Ray with ray_address.
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
        if isinstance(error, TimeoutError):
            logger.error(f"Rollout {idx} exceeded the {self.rollout_config.episode_timeout}s deadline")
        elif _is_shutdown_error(error):
            logger.debug(f"Rollout {idx} abandoned at shutdown: {error}")
        else:
            # An unclassified failure costs the whole rollout; the traceback is what localizes it.
            logger.error(f"Rollout {idx} failed: {error}", exc_info=True)
