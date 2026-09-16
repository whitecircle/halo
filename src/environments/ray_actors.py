"""Ray actors for async Environmental GRPO rollout collection.

Each actor (own process) drives env interactions and generates via the rollout server's
/v1/chat/completions (vLLM or SGLang).
RolloutManager fans prompts across actors and server URLs round-robin under a bounded Semaphore.
Weight sync is separate, over NCCL (:mod:`src.trainers.grpo.rollout.weight_sync_clients`)."""

import asyncio
import logging
import tempfile
import time
from collections.abc import Callable
from dataclasses import replace
from functools import cached_property
from typing import Any
from urllib.parse import urlparse

import aiohttp
import backoff
import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from src.configs.rollout_config import RolloutConfig
from src.environments.base import EPISODE_ERROR_KEY, Trajectory
from src.environments.engine_wire import build_payload, capture_generation_tokens, capture_routing_mask
from src.environments.episode import (
    EpisodeDispatcher,
    RolloutResult,
    TurnGeneration,
    bind_episode_effort,
    step_context_from_generation,
)
from src.environments.registry import create_environment
from src.inference.response import FINISH_REASON_ABORT, get_finish_reason, get_reasoning_text
from src.log import warn_once

logger = logging.getLogger(__name__)

_RETRYABLE_4XX = {408, 429}
# Backends already warned that their completions carry no ``usage.completion_tokens`` (once per process).
_COMPLETION_TOKENS_MISSING_WARNED: set[str] = set()


class RolloutHTTPError(RuntimeError):
    """A non-200 from the rollout server, carrying its status as data.

    The status decides retry versus abandon (:func:`_is_client_error`). The response body is
    server-controlled text, so recovering the status by re-parsing the message would let a 503 whose
    body quotes a "status 400" upstream error abandon the whole batch.
    """

    def __init__(self, status: int, backend: str, body: str):
        super().__init__(f"{backend} error (status {status}): {body}")
        self.status = status


def _describe_exc(exc: BaseException) -> str:
    """``Type: message``, keeping the type when the message is empty.

    ``asyncio.TimeoutError``, the usual signature of a stalled engine, stringifies to ``""``, so
    interpolating the exception alone carries no information.
    """
    text = str(exc)
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def _is_client_error(exc: BaseException) -> bool:
    """Return True for non-retryable client errors (give up).

    The status is read off :class:`RolloutHTTPError` rather than out of a message. Only genuine 4xx
    (minus retryable 429/408) are terminal; anything that is not a rollout HTTP error is retryable.
    """
    if not isinstance(exc, RolloutHTTPError):
        return False
    return 400 <= exc.status < 500 and exc.status not in _RETRYABLE_4XX


def _is_shutdown_error(exc: BaseException) -> bool:
    """True for benign teardown-induced errors (session/loop closed at shutdown while a prefetched
    rollout is in flight). Retrying is pointless; log quietly, not as ERROR.

    Matched on message text, unlike :func:`_is_client_error`: these strings come from aiohttp and Ray,
    which expose no typed marker for them."""
    s = str(exc)
    return "Session is closed" in s or "Event loop is closed" in s or "INTENDED_USER_EXIT" in s


def _should_giveup(exc: BaseException) -> bool:
    """Backoff give-up predicate: non-retryable client 4xx, or a benign shutdown error."""
    return _is_client_error(exc) or _is_shutdown_error(exc)


async def _resolve_ref(ref):
    """``asyncio`` task bodies must be coroutines; a Ray ``ObjectRef`` is only awaitable."""
    return await ref


async def _await_with_deadline(ref, timeout: float, paused_clock: Callable[[], float] | None = None):
    """Await a Ray ``ObjectRef`` under ``timeout`` seconds of engine-serving time, raising ``TimeoutError``.

    ``paused_clock`` reads the manager's count of seconds the engines have spent paused for weight
    syncs: a paused engine freezes every in-flight generation, so that window is credited back to
    the episode instead of charged to it. The ref is awaited in one task, shielded across the re-arms
    a credit forces; on expiry the task is cancelled.
    """
    loop = asyncio.get_running_loop()
    started = loop.time()
    paused_at_start = paused_clock() if paused_clock is not None else 0.0
    task = loop.create_task(_resolve_ref(ref))
    try:
        while True:
            credit = paused_clock() - paused_at_start if paused_clock is not None else 0.0
            remaining = timeout - (loop.time() - started - credit)
            if remaining <= 0:
                raise TimeoutError(
                    f"episode exceeded its {timeout:.0f}s deadline ({credit:.0f}s of engine pause excluded)"
                )
            try:
                return await asyncio.wait_for(asyncio.shield(task), remaining)
            except TimeoutError:
                if task.done():
                    raise  # the episode's own TimeoutError, not the deadline's
    finally:
        if not task.done():
            task.cancel()


# Ray's plasma socket lives under the temp dir and AF_UNIX paths cap at ~107 bytes: a deep TMPDIR
# overflows it and crashes ``ray.init``.
MAX_RAY_TEMP_DIR_LEN = 40
RAY_FALLBACK_TEMP_DIR = "/tmp/ray"

# In-flight rollouts per actor when ``max_concurrent_rollouts`` is unset. Actors multiplex episodes on
# one event loop, so the pool size is not the concurrency limit; this oversubscribes it enough to keep
# the servers fed while a turn is being graded. Documented as the derived default in
# ``agent-docs/reference/configuration-reference.md``.
DEFAULT_ROLLOUTS_PER_WORKER = 4


def ray_init_kwargs(**overrides) -> dict:
    """Kwargs for ``ray.init``, with a short-tempdir fallback."""
    # Dashboard off: unused by the RL actors and exposes Ray's jobs-API HTTP surface
    # (CVE-2023-48022 job-submission RCE, CVE-2025-62593 DNS rebinding).
    kwargs: dict = {"ignore_reinit_error": True, "include_dashboard": False, **overrides}
    if len(tempfile.gettempdir()) > MAX_RAY_TEMP_DIR_LEN:
        kwargs["_temp_dir"] = RAY_FALLBACK_TEMP_DIR
    return kwargs


# Actors are stateless between episodes, so unlimited restarts are preferable to a permanent gap in
# the round-robin, where every episode routed to a dead actor returns a masked row.
@ray.remote(max_restarts=-1)
class EnvironmentActor:
    """Isolated Ray actor for environment episode execution. Generates via the rollout server's
    /v1/chat/completions (chat template applied server-side). Env, HTTP client, and tool schema are
    lazily initialized."""

    def __init__(
        self,
        actor_id: int,
        env_type: str | tuple[type, dict[str, Any]],
        env_config: dict[str, Any],
    ):
        self.actor_id = actor_id
        self.env_type = env_type
        self.env_config = env_config

        self._env = None
        self._http_client = None

    def _get_env(self):
        if self._env is None:
            self._env = create_environment(self.env_type, self.env_config)
        return self._env

    @cached_property
    def _tools_schema(self) -> list[dict] | None:
        """OpenAI-format tools list, or None if the env has no tools. Static per run."""
        return self._get_env().get_tools_schema()

    async def _get_http_client(self, timeout: float) -> aiohttp.ClientSession:
        if self._http_client is None:
            # force_close: the server drops idle keepalives between batches, so a pooled socket
            # yields ServerDisconnectedError whose retries stall this rank at the next collective.
            connector = aiohttp.TCPConnector(force_close=True, enable_cleanup_closed=True)
            self._http_client = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=timeout), connector=connector
            )
        return self._http_client

    async def run_episode(
        self,
        prompt: str,
        context: dict[str, Any] | None,
        server_url: str,
        config: RolloutConfig,
    ) -> RolloutResult:
        """Run a complete multi-turn episode."""
        start = time.time()
        generation_tokens = 0
        logp_sum, logp_count = 0.0, 0
        eid = None

        try:
            env = self._get_env()
            client = await self._get_http_client(config.request_timeout)
            episode = EpisodeDispatcher(env)
            episode_ids, steps = await episode.reset([prompt], [context])
            eid, step = episode_ids[0], steps[0]
            length = 0

            # Bound once per episode: every turn shares the level and its caps.
            effort = bind_episode_effort(
                context, env, max_tokens=config.max_tokens, max_thinking_tokens=config.max_thinking_tokens
            )
            ep_config = replace(config, max_tokens=effort.max_tokens, max_thinking_tokens=effort.thinking_budget)

            for _ in range(env.max_turns):
                if step.done:
                    break

                gen = await self._generate_turn(client, server_url, step.observation, ep_config, effort.level)
                generation_tokens += gen.tokens
                if gen.token_logprobs:
                    logp_sum += sum(gen.token_logprobs)
                    logp_count += len(gen.token_logprobs)

                steps = await episode.step([eid], [gen.text], [step_context_from_generation(context, gen)])
                step = steps[0]
                length += 1

            traj = env.get_trajectories([eid])[0]
            effort.stamp(traj)

            episode_metrics = env.rollout_metrics(traj) if traj else {}
            if logp_count:
                episode_metrics["logps/sampling_mean"] = logp_sum / logp_count

            return RolloutResult(
                prompt=prompt,
                trajectory=traj,
                episode_length=length,
                total_reward=traj.total_reward if traj else 0.0,
                metrics=episode_metrics,
                # Natural terminal state, not a max_turns truncation; both set done=True.
                success=bool(traj and traj.done and not traj.truncated),
                latency=time.time() - start,
                generation_tokens=generation_tokens,
            )

        except Exception as e:
            # exc_info: this catch-all turns any episode fault into a masked zero-reward row, so the
            # traceback is what localizes it.
            logger.error(f"Actor {self.actor_id} error: {e}", exc_info=True)
            # Reward stays 0 and the partial trajectory is dropped, unlike the eval driver, which
            # finalizes the episode as truncated and keeps what it earned: ``error`` puts this row
            # outside ``rollout_valid_mask``, so the trainer excludes it from the group baseline and
            # drops its tokens from the step. Accrued reward could therefore only reach the logged
            # reward mean, where a half-episode's partial credit is noise. Typed, never ``str(e)``: a
            # bare ``TimeoutError`` stringifies to "", which the mask reads as no error.
            reason = _describe_exc(e)
            return RolloutResult(
                prompt=prompt,
                trajectory=Trajectory(done=True, info={EPISODE_ERROR_KEY: reason}),
                total_reward=0.0,
                success=False,
                latency=time.time() - start,
                error=reason,
            )

        finally:
            # On every path: a stateful env (SweEnvironment) leaks a sandbox working dir per rollout.
            if eid is not None and self._env is not None:
                try:
                    self._env.cleanup([eid])
                except Exception:  # cleanup must never mask the episode result
                    logger.debug("Actor %s: cleanup failed for episode %s", self.actor_id, eid, exc_info=True)

    async def _generate_turn(
        self,
        client: aiohttp.ClientSession,
        server_url: str,
        messages: list[dict[str, str]],
        config: RolloutConfig,
        reasoning_effort: str | None = None,
    ) -> TurnGeneration:
        """One turn's generation, re-issued for the same observation while the engine aborts it.

        An abort is the engine's doing (SGLang's sync pause drops every in-flight request), so the
        fragment never reaches the env: stepping it would spend the episode's length-cutoff recovery
        cap on a cut the policy did not make. Bounded by ``config.max_retries`` re-issues per turn;
        past that the episode errors into a masked row.
        """
        for _ in range(config.max_retries + 1):
            gen = await self._generate(client, server_url, messages, config, reasoning_effort)
            if gen.finish_reason != FINISH_REASON_ABORT:
                return gen
            logger.warning(f"Actor {self.actor_id}: the {config.backend} engine aborted the turn; re-issuing it")
        raise RuntimeError(
            f"the {config.backend} engine aborted the same turn {config.max_retries + 1} times in a row "
            f"(finish_reason={FINISH_REASON_ABORT!r}); the episode is dropped rather than stepped with a fragment"
        )

    async def _generate(
        self,
        client: aiohttp.ClientSession,
        server_url: str,
        messages: list[dict[str, str]],
        config: RolloutConfig,
        reasoning_effort: str | None = None,
    ) -> TurnGeneration:
        """Call /v1/chat/completions with backoff-based retry; returns the turn's :class:`TurnGeneration`
        (capture fields populated per the ``RolloutConfig`` flags).

        Uses raw aiohttp rather than the shared OpenAI client: the training rollout load-balances each
        request across a pool of server URLs chosen per call, and needs the actor-tagged backoff with
        retryable-4xx classification.
        """
        url = server_url.rstrip("/")
        if not url.startswith("http"):
            url = f"http://{url}"

        payload = self._build_payload(messages, config, reasoning_effort)

        @backoff.on_exception(
            backoff.expo,
            (asyncio.TimeoutError, aiohttp.ClientError, RuntimeError),
            # backoff counts total attempts: max_retries=0 would never match and retry forever.
            max_tries=config.max_retries + 1,
            factor=config.retry_base_wait,
            giveup=_should_giveup,
            logger=None,  # handlers below replace backoff's own logging
            on_backoff=lambda d: logger.warning(
                f"Actor {self.actor_id}: rollout retry {d['tries']}/{config.max_retries} "
                f"after {d['wait']:.1f}s — {_describe_exc(d['exception'])}"
            ),
            on_giveup=lambda d: (
                logger.debug(f"Actor {self.actor_id}: request abandoned at shutdown")
                if _is_shutdown_error(d["exception"])
                else logger.warning(
                    f"Actor {self.actor_id}: gave up after {d['tries']} tries — {_describe_exc(d['exception'])}"
                )
            ),
        )
        async def _request():
            async with client.post(f"{url}/v1/chat/completions", json=payload) as resp:
                if resp.status != 200:
                    raise RolloutHTTPError(resp.status, config.backend, await resp.text())
                data = await resp.json()
            choice = data["choices"][0]
            usage = data.get("usage", {})
            msg = choice["message"]
            text = msg.get("content") or ""
            reasoning = get_reasoning_text(msg) or ""
            tool_calls = msg.get("tool_calls") or []
            if config.capture_token_ids:
                token_ids, token_logprobs, prompt_token_ids = capture_generation_tokens(choice, data, config.backend)
            else:
                token_ids = token_logprobs = prompt_token_ids = None
            tokens = usage.get("completion_tokens") or 0
            if not tokens and text:
                # The captured ids are the only honest length; a word count of the text is not one.
                tokens = len(token_ids) if token_ids else 0
                warn_once(
                    logger,
                    _COMPLETION_TOKENS_MISSING_WARNED,
                    config.backend,
                    "%s returned a completion without usage.completion_tokens; generation-token metrics "
                    "count the captured token ids instead (0 when none are captured).",
                    config.backend,
                )
            routing_mask = capture_routing_mask(choice, data) if config.capture_routed_experts else None
            # The engine's prompt length anchors the mask; the trainer's re-render can differ by a token.
            routing_prompt_tokens = usage.get("prompt_tokens") if routing_mask else None
            return TurnGeneration(
                text=text,
                tool_calls=tool_calls,
                reasoning=reasoning,
                tokens=tokens,
                token_ids=token_ids,
                token_logprobs=token_logprobs,
                routing_mask=routing_mask,
                routing_prompt_tokens=routing_prompt_tokens,
                prompt_token_ids=prompt_token_ids,
                finish_reason=get_finish_reason(
                    choice, completion_tokens=usage.get("completion_tokens"), max_tokens=config.max_tokens
                ),
            )

        return await _request()

    def _build_payload(
        self, messages: list[dict], config: RolloutConfig, reasoning_effort: str | None = None
    ) -> dict[str, Any]:
        """The turn's request payload, with this actor's env tool schema bound onto the wire format."""
        return build_payload(messages, config, reasoning_effort, self._tools_schema)

    async def shutdown(self):
        """Close the HTTP session and the environment (idempotent; called by RolloutManager.shutdown)."""
        if self._http_client:
            await self._http_client.close()
            self._http_client = None
        if self._env:
            try:
                self._env.close()
            except Exception:  # a failed teardown must not block the rest of shutdown
                logger.warning("Actor %s: environment close failed", self.actor_id, exc_info=True)
            self._env = None


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
        """``num_workers`` is the actor budget for the whole Ray cluster; ``pool_replicas`` is how many
        peers build their own pool against it (the DP size on a shared cluster, 1 when each rank runs
        its own), so the budget is divided rather than oversubscribed once per rank."""
        self.num_workers = max(1, num_workers // max(1, pool_replicas))
        self.env_type = env_type
        self.env_config = env_config
        self.server_urls = server_urls
        self.rollout_config = rollout_config

        requested = max_concurrent_rollouts or self.num_workers * DEFAULT_ROLLOUTS_PER_WORKER
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
        # Seconds the engines have spent paused for weight syncs, credited back to every in-flight
        # episode's deadline. The trainer thread opens and closes each window; the episodes' loop reads it.
        self._paused_seconds = 0.0
        self._pause_started_at: float | None = None

        logger.info(
            f"RolloutManager: {self.num_workers} workers, "
            f"{len(server_urls)} rollout servers, max_concurrent={self.max_concurrent}"
        )

    def begin_engine_pause(self) -> None:
        """Mark the engines paused for a weight sync: in-flight episode deadlines stop counting."""
        self._pause_started_at = time.monotonic()

    def end_engine_pause(self, seconds: float) -> None:
        """Close the pause window, crediting the ``seconds`` the forwarding rank measured for its push.

        The measured figure replaces this rank's own estimate — the pause is the servers', timed
        where the push ran. Credited before the window closes, so a deadline read between the two
        statements never sees less than either.
        """
        self._paused_seconds += seconds
        self._pause_started_at = None

    @property
    def paused_seconds(self) -> float:
        """Engine-paused seconds so far, an open window included."""
        total = self._paused_seconds
        if self._pause_started_at is not None:
            total += time.monotonic() - self._pause_started_at
        return total

    def warn_if_servers_unreachable_from_actors(self, multinode: bool) -> None:
        """Warn when a rollout-server URL is loopback on a multi-node job. Call on a single rank.

        Placement only prefers this node (see :meth:`start`), so an actor can land anywhere. A loopback
        URL then reaches a node with no engine listening and every episode routed there returns a
        rollout error, which :func:`~src.trainers.grpo.environmental.rollout_valid_mask` excludes from
        the GRPO baseline: the effective batch shrinks until the all-masked guard halts the run. A
        loopback URL is valid only if an engine runs on every node.
        """
        loopback = {"localhost", "127.0.0.1", "0.0.0.0", "::1"}
        urls = self.server_urls
        if not multinode or not any(urlparse(u if "://" in u else f"http://{u}").hostname in loopback for u in urls):
            return
        backend = self.rollout_config.backend
        logger.warning(
            "Multi-node async GRPO but a %s URL is loopback (%s). Ray actors may "
            "be scheduled on nodes where no %s server listens on localhost, causing "
            "connection-refused rollouts silently returned as zero-reward. Use "
            "resolvable host IPs/DNS in rollout_server_url / rollout_server_configs.",
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
            # _init_async_state has already initialized Ray with the configured ray_address.
            ray.init(**ray_init_kwargs())
        # Prefer this node; without it a shared cluster scatters this rank's actors cluster-wide.
        # `soft` alone spills only when the node is dead or infeasible: a live but saturated node pins
        # the actor there and waits indefinitely, which on the CPU-actor-tier topology
        # (agent-docs/infrastructure/ray.md) keeps every actor on the training node instead of
        # spilling. `_spill_on_unavailable` makes the affinity an actual preference.
        try:
            local_node_id = ray.get_runtime_context().get_node_id()
            actor_cls = EnvironmentActor.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=local_node_id, soft=True, _spill_on_unavailable=True
                )
            )
        except Exception:  # affinity is an optimization, not fatal
            actor_cls = EnvironmentActor
        self._actors = [actor_cls.remote(i, self.env_type, self.env_config) for i in range(self.num_workers)]
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
                        logger.error(f"Rollout {idx} exceeded the {self.rollout_config.episode_timeout}s deadline")
                    elif _is_shutdown_error(e):
                        logger.debug(f"Rollout {idx} abandoned at shutdown: {e}")
                    else:
                        # Same reasoning as the actor-side catch-all: an unclassified failure costs
                        # the whole rollout, and the traceback is what localizes it.
                        logger.error(f"Rollout {idx} failed: {e}", exc_info=True)
                    errors[idx] = e

        tasks = [asyncio.create_task(_run(i, p, c)) for i, (p, c) in enumerate(zip(prompts, contexts, strict=False))]
        await asyncio.gather(*tasks, return_exceptions=True)

        final = []
        for i in range(len(prompts)):
            r = results[i]
            if r is None:
                # Typed, never ``str(e)``: the deadline's bare ``TimeoutError`` stringifies to "", and an
                # empty ``error`` is a VALID zero-reward group member to ``rollout_valid_mask``.
                msg = _describe_exc(errors[i]) if errors[i] is not None else "Unknown error"
                r = RolloutResult(
                    prompt=prompts[i],
                    trajectory=Trajectory(done=True, info={EPISODE_ERROR_KEY: msg}),
                    success=False,
                    error=msg,
                )
            final.append(r)

        return final
