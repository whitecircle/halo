"""Pool of vendored NCCL weight-sync clients, one per rollout server, plus the preflights that probe
each server before the trainer is built (context window, sampler and prompt log-prob semantics).

No vllm/sglang package dependency; generation requests go over HTTP separately (see ray_actors.py).
Constraints: weight sync runs from the main process only; the trainer must use a different GPU than
the rollout server (NCCL requires distinct devices); one server URL per weight-sync group.
"""

import concurrent.futures
import logging
import math
import threading
from collections.abc import Callable, Iterable
from functools import partial
from typing import Any

import torch
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from src.configs.rollout_config import SAMPLER_FILTERS_OFF
from src.distributed.nccl.clients.base import (
    WEIGHT_SYNC_CHUNK_BYTES,
    BaseWeightSyncClient,
    SamplerLogprobSemantics,
    payload_bytes,
    resolve_sync_device,
    snapshot_param,
    starts_new_chunk,
    validate_syncable_param,
)
from src.distributed.nccl.registry import resolve_weight_sync_client
from src.distributed.runtime import raise_rank0_failure
from src.models.structure import resolve_tokenizer

logger = logging.getLogger(__name__)

# A preflight request retries a refused connection or a 5xx, never a read past its timeout: every rank
# waits on rank 0's probes, so a stalled server must cost one timeout per request, not six.
_PREFLIGHT_RETRY = Retry(
    total=3,
    connect=3,
    read=0,
    status=3,
    status_forcelist=[500, 502, 503],
    backoff_factor=1,
    allowed_methods=["POST", "GET"],
)

# Prompt-log-prob consistency probe: a counting sequence cut past vLLM's CUDA-graph capture range
# (512 tokens), so the whole prefills eagerly, and prefixes that each prefill as a captured graph.
_PROMPT_LOGPROB_PROBE_TEXT = " ".join(str(n) for n in range(2048))
_PROMPT_LOGPROB_PROBE_TOKENS = 1100
_PROMPT_LOGPROB_PROBE_PREFIXES = (8, 64, 128, 256, 448)
# Batch-shape bf16 noise stays under ~0.5 nats per token; the drafter overwrite moves every one by 8-22.
_PROMPT_LOGPROB_TOLERANCE_NATS = 2.0
# Any servable model all but predicts a counting sequence, while garbage prompt log-probs average 12-22
# nats: the bound catches a whole sequence that itself prefilled as a graph (a raised capture range).
_PROMPT_LOGPROB_PROBE_MAX_MEAN_NLL = 4.0
# Every rank waits on rank 0's probe in the verdict broadcast, so a hung server must not outlast it.
_PROMPT_LOGPROB_PROBE_TIMEOUT_S = 60.0


def _probe_server(
    client_cls: type[BaseWeightSyncClient],
    url: str,
    probe: Callable[[BaseWeightSyncClient], Any],
    default: Any,
    failure: str,
) -> Any:
    """``probe`` over a fresh client for ``url``, or ``default`` when the server cannot answer it (down,
    or a model card without the field): a preflight probe never fails the run by itself. ``failure``
    completes the warning ``Could not <failure> <url>``."""
    client = None
    try:
        client = client_cls(base_url=url)
        for scheme in ("http://", "https://"):
            client.session.mount(scheme, HTTPAdapter(max_retries=_PREFLIGHT_RETRY))
        return probe(client)
    except Exception as e:
        logger.warning(f"Could not {failure} {url}: {e}")
        return default
    finally:
        if client is not None:
            client.session.close()


def verify_context_window(
    client_cls: type[BaseWeightSyncClient],
    urls: list[str],
    single_turn_tokens: int,
    full_trajectory_tokens: int | None,
) -> None:
    """Verify each rollout server's context window fits the configured generation budget.

    ``single_turn_tokens`` (max_prompt_length + per-call generation cap) is a hard requirement, so this
    raises. ``full_trajectory_tokens`` (worst-case multi-turn budget) is advisory, since a rollout
    growing past the context OOMs the trainer's forward before the fail-on-overflow check, so it warns.
    """
    for url in urls:
        mml = _probe_server(client_cls, url, client_cls.served_max_model_len, None, "read max_model_len from")
        if mml is None:
            continue
        logger.info(f"Rollout server {url}: max_model_len={mml}")
        if single_turn_tokens > mml:
            raise ValueError(
                f"Rollout server {url} context window ({mml}) is smaller than one rollout turn "
                f"({single_turn_tokens} = max_prompt_length + per-turn generation). Lower "
                f"rollout_max_tokens/max_completion_length or serve a longer-context model."
            )
        if full_trajectory_tokens and full_trajectory_tokens > mml:
            logger.warning(
                f"Rollout server {url} context window ({mml}) < worst-case trajectory budget "
                f"({full_trajectory_tokens} = max_prompt_length + what an episode may generate: the smaller of "
                f"max_turns × rollout_max_tokens and rollout_max_episode_tokens). A long multi-turn rollout that "
                f"grows past {mml} tokens can OOM the training forward before the fail-on-overflow check — set "
                f"rollout_max_episode_tokens, or lower max_turns or rollout_max_tokens, so the worst case fits."
            )


def verify_sampler_logprob_reference(
    client_cls: type[BaseWeightSyncClient],
    urls: list[str],
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    repetition_penalty: float,
    sequence_ratio_active: bool,
) -> None:
    """Refuse a rollout server whose per-token logprobs are not the reference the IS ratio divides by.

    The trainer scores its log-probs at the sampling temperature and divides by the engine's reported
    sampling log-probs, so those must already carry the temperature: against vLLM's default raw
    (pre-temperature) values every weight becomes π^T / π^1, tilted toward improbable tokens on every
    step — entropy inflates at T > 1, collapses at T < 1 — while the ratio still reads ≈ 1. A reference
    renormalized over the sampler's cut (vLLM ``processed_logprobs`` under a top-p < 1, a top-k or a
    min-p, all applied before its logprobs; SGLang's reference precedes all three) lifts every uncertain
    position by the mass the cut kept; a consumer that sums the per-token log-ratios over a sequence
    (``sequence_ratio_active``: the trajectory geometric band, OPSM's per-trajectory mean, or a
    sequence-level IS mode) reads the sum as drift or a collapsing sequence weight, so that pairing is
    refused. The probe reads the renormalization off a top-p cut. ``top_k`` cuts above 0, the reading
    TRL's off value (0) and the rollout config's (-1) share. A ``repetition_penalty`` other than 1 is
    refused under the same consumers without a probe: both engines' reported logprobs carry it and the
    trainer's never do. An unverifiable server warns: a preflight probe never fails the run by itself.
    """
    if sequence_ratio_active and repetition_penalty != 1.0:
        raise ValueError(
            f"repetition_penalty={repetition_penalty} while the per-token log-ratios are summed over each "
            f"sequence: both engines' sampling logprobs carry the penalty and the trainer's recomputed ones do "
            f"not, so every penalized position reads as drift in the trajectory geometric band and OPSM, and a "
            f"sequence-level vLLM IS ratio collapses. Sample with repetition_penalty: 1.0 "
            f"(rollout_repetition_penalty: 1.0 on the environmental arm), or take the ratio per token: a "
            f"token_* vllm_importance_sampling_mode, or drop isr_geo_band_min/max and isr_opsm_delta."
        )
    cuts = [
        f"{name}={value}"
        for name, value, on in (
            ("top_p", top_p, top_p < 1.0),
            ("top_k", top_k, top_k > 0),
            ("min_p", min_p, min_p > 0),
        )
        if on
    ]
    if temperature == 1.0 and not (cuts and sequence_ratio_active):
        return
    # Each probe request moves one knob from the identity sampler, so every other filter goes out off.
    probe = partial(client_cls.probe_sampler_logprob_semantics, **SAMPLER_FILTERS_OFF)
    for url in urls:
        semantics = _probe_server(
            client_cls, url, probe, SamplerLogprobSemantics(None, None), "probe the sampler-logprob semantics of"
        )
        if temperature != 1.0:
            if semantics.temperature_applied is None:
                logger.warning(
                    f"Rollout server {url}: could not verify that its logprobs carry the sampling temperature "
                    f"({temperature}); the IS ratio is biased if they are raw pre-temperature values (vLLM's default)."
                )
            elif not semantics.temperature_applied:
                raise ValueError(
                    f"Rollout server {url} reports RAW (pre-temperature) logprobs while the sampling temperature is "
                    f"{temperature}: the IS ratio would divide the trainer's temperature-{temperature} log-probs "
                    f"by temperature-1 ones, biasing every token weight. Serve vLLM with "
                    f"`--logprobs-mode processed_logprobs` (the compose recipe sets it), or leave "
                    f"SGLANG_RETURN_ORIGINAL_LOGPROB unset on SGLang, or sample at temperature 1.0."
                )
        if cuts and sequence_ratio_active:
            if semantics.nucleus_renormalized is None:
                logger.warning(
                    f"Rollout server {url}: could not verify whether its logprobs are renormalized over the "
                    f"sampler's cut; with {', '.join(cuts)} a renormalized reference biases every "
                    f"sequence-summed log-ratio (geometric band, OPSM, sequence-level IS)."
                )
            elif semantics.nucleus_renormalized:
                raise ValueError(
                    f"Rollout server {url} reports logprobs renormalized over the sampler's cut while "
                    f"{', '.join(cuts)} and the per-token log-ratios are summed over each sequence: every "
                    f"uncertain position is lifted by the mass the cut kept, so the trajectory geometric band and "
                    f"OPSM's per-trajectory mean log-ratio read the sum as drift and a sequence-level vLLM IS ratio "
                    f"collapses toward 0 (sequence_mask only zeroes ratios ABOVE the cap, so the run stalls "
                    f"silently). Sample with every cut off — top_p: 1.0, top_k: 0, min_p: 0.0 (rollout_top_p: 1.0, "
                    f"rollout_top_k: -1, rollout_min_p: 0.0 on the environmental arm) — or take the ratio per "
                    f"token: a token_* vllm_importance_sampling_mode, or drop isr_geo_band_min/max and isr_opsm_delta."
                )


def _probe_prompt_logprobs(client: BaseWeightSyncClient, token_ids: list[int]) -> tuple[float, float, int]:
    """``(mean NLL, gap, length)``: the whole sequence's mean negative log-prob, and the largest per-position
    gap between it and a prefix scored alone, with that prefix's length. A non-finite log-prob reads as an
    infinite gap."""
    anchor, rest = token_ids[:1], token_ids[1:]
    score = partial(client.score_completion_logprobs, anchor, timeout=_PROMPT_LOGPROB_PROBE_TIMEOUT_S)
    whole = score(rest)
    divergences = []
    for length in _PROMPT_LOGPROB_PROBE_PREFIXES:
        gaps = [abs(a - b) for a, b in zip(score(rest[: length - 1]), whole[: length - 1], strict=True)]
        divergences.append((max(gaps) if all(map(math.isfinite, gaps)) else math.inf, length))
    gap, length = max(divergences)
    return -sum(whole) / len(whole), gap, length


def verify_engine_prompt_logprobs(client_cls: type[BaseWeightSyncClient], urls: list[str], tokenizer) -> None:
    """Refuse a rollout server whose prompt log-probs do not follow from the tokens alone.

    ``isr_engine_reference`` reads every re-scored row off the engine's prompt log-probs. vLLM 0.26.0 under
    speculative decoding corrupts them whenever the target prefill runs as a CUDA graph: the drafter's graph
    overwrites the hidden states they are read from, and every position of the request comes back garbage
    while sampling stays correct (vllm-project/vllm#53488; the toolkit's vLLM image patches it). The probe
    scores a counting sequence long enough to prefill eagerly, which any servable model predicts, then
    prefixes of it alone at captured-graph lengths, and compares the shared positions. An unverifiable
    server warns: a preflight probe never fails the run by itself.
    """
    token_ids = resolve_tokenizer(tokenizer)(_PROMPT_LOGPROB_PROBE_TEXT)["input_ids"][:_PROMPT_LOGPROB_PROBE_TOKENS]
    probe = partial(_probe_prompt_logprobs, token_ids=token_ids)
    for url in urls:
        result = _probe_server(client_cls, url, probe, None, "probe the prompt log-probs of")
        if result is None:
            continue
        mean_nll, gap, length = result
        # Negated so a NaN mean, for which every comparison is False, is refused too.
        if not (mean_nll <= _PROMPT_LOGPROB_PROBE_MAX_MEAN_NLL and gap <= _PROMPT_LOGPROB_TOLERANCE_NATS):
            raise ValueError(
                f"Rollout server {url} returns prompt log-probs that do not follow from the tokens: a "
                f"{len(token_ids)}-token counting probe scored whole averages {mean_nll:.2f} nats per token (bound "
                f"{_PROMPT_LOGPROB_PROBE_MAX_MEAN_NLL}), and its first {length} tokens scored alone differ from the "
                f"same positions by up to {gap:.2f} nats (tolerance {_PROMPT_LOGPROB_TOLERANCE_NATS}), so "
                f"isr_engine_reference would compare the sampling log-probs against garbage. The known cause is "
                f"vLLM 0.26.0 with speculative decoding (MTP) and without the toolkit image's prompt-log-prob patch: "
                f"the drafter's CUDA graph overwrites the hidden states the prompt log-probs are read from whenever "
                f"the prefill runs as a graph (vllm-project/vllm#53488). Serve vLLM from the image Dockerfile.vllm "
                f"builds, turn speculative decoding off, or set isr_engine_reference: false."
            )
        logger.info(
            f"Rollout server {url}: prompt log-probs follow from the tokens (probe mean NLL {mean_nll:.3f}, "
            f"largest prefix gap {gap:.3f} nats)"
        )


def _preflight_failure(error: Exception) -> str:
    """A preflight's own refusal verbatim; any other failure with its type, which ``str`` alone drops."""
    return str(error) if isinstance(error, ValueError) else f"{type(error).__name__}: {error}"


def verify_context_window_synced(
    urls: list[str], single_turn_tokens: int, full_trajectory_tokens: int | None = None, *, backend: str
) -> None:
    """Collective-safe :func:`verify_context_window`; call on every rank.

    The backend lookup runs on every rank: it reads no server, and a rank-0-only raise there would
    leave the peers blocked in the verdict broadcast.
    """
    client_cls = resolve_weight_sync_client(backend)
    raise_rank0_failure(
        partial(verify_context_window, client_cls, urls, single_turn_tokens, full_trajectory_tokens),
        _preflight_failure,
        ValueError,
    )


def verify_sampler_logprob_reference_synced(
    urls: list[str],
    *,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    repetition_penalty: float,
    sequence_ratio_active: bool,
    backend: str,
) -> None:
    """Collective-safe :func:`verify_sampler_logprob_reference`; call on every rank."""
    client_cls = resolve_weight_sync_client(backend)
    raise_rank0_failure(
        partial(
            verify_sampler_logprob_reference,
            client_cls,
            urls,
            temperature,
            top_p,
            top_k,
            min_p,
            repetition_penalty,
            sequence_ratio_active,
        ),
        _preflight_failure,
        ValueError,
    )


def verify_engine_prompt_logprobs_synced(urls: list[str], tokenizer, *, backend: str) -> None:
    """Collective-safe :func:`verify_engine_prompt_logprobs`; call on every rank."""
    client_cls = resolve_weight_sync_client(backend)
    raise_rank0_failure(
        partial(verify_engine_prompt_logprobs, client_cls, urls, tokenizer), _preflight_failure, ValueError
    )


class InferenceClientManager:
    """Manage one weight-sync client per rollout server (each with its own NCCL group and unique
    group_port), keeping multiple servers in weight-sync with the trainer."""

    def __init__(
        self,
        server_configs: list[dict[str, Any]],
        *,
        connection_timeout: float,
        client_cls: type[BaseWeightSyncClient],
        base_group_port: int,
    ):
        """``server_configs`` is a list of ``{"url", "group_port", "group_host"}`` dicts, one client
        per server (only ``url`` is required).

        ``base_group_port`` is the first trainer-side NCCL group port, handed to server N that
        declares no ``group_port`` of its own as ``base_group_port + N``. It is the run's
        ``vllm_group_port``, so the setting means the same thing whichever client shape is built.

        ``client_cls`` selects the engine (``resolve_weight_sync_client``); every server in one
        manager speaks the same one, since they are replicas of a single served policy.
        """
        self.server_configs = server_configs
        self.connection_timeout = connection_timeout
        self.base_group_port = base_group_port
        self._clients = []
        self._initialized = False
        self._device = None
        # The pushed model's module names, applied to every client built (rebuilt ones included).
        # None until a push scopes them, so a client built before then keeps the engine's full groups
        # and refuses an incomplete one rather than sending its halves apart.
        self._co_load_module_names: tuple[str, ...] | None = None
        # Bytes buffered since the last chunk went out. The manager makes the chunk decision because
        # only it can tell when every server is done with the shared snapshots.
        self._buffered_bytes = 0
        # NCCL groups cannot form concurrently: serializes reconnects across parallel flush threads.
        self._reconnect_lock = threading.Lock()
        # Single construction seam, so init and reconnect cannot drift onto different engines.
        self._client_factory = client_cls

        # group_port is bound on the trainer host, so two servers sharing one collide.
        effective_ports = [self._group_port(i) for i in range(len(server_configs))]
        if len(set(effective_ports)) != len(effective_ports):
            raise ValueError(
                f"Duplicate weight-sync group_port across servers: {effective_ports}. Each server "
                f"needs a UNIQUE group_port — it is bound on the trainer host (one listener "
                f"per server), so distinct {client_cls.BACKEND_NAME} hosts do NOT make a shared "
                f"port safe. URLs: {[c['url'] for c in server_configs]}"
            )

        logger.info(
            f"InferenceClientManager created for {len(server_configs)} servers: {[c['url'] for c in server_configs]}"
        )

    def _group_port(self, index: int) -> int:
        """The trainer-side NCCL group port for one server: its configured value, else the base + index."""
        return self.server_configs[index].get("group_port", self.base_group_port + index)

    def _connect_client(self, index: int, device: torch.device) -> BaseWeightSyncClient:
        """Build server ``index``'s client and form its NCCL group on ``device``: the one construction
        path, shared by the first connect and a reconnect so both bind the same port and NIC."""
        config = self.server_configs[index]
        client = self._client_factory(
            base_url=config["url"],
            group_port=self._group_port(index),
            connection_timeout=self.connection_timeout,
            # Optional per-server routable trainer NIC for the NCCL group (multi-homed nodes).
            group_host=config.get("group_host"),
        )
        client.init_communicator(device=device)
        if self._co_load_module_names is not None:
            client.scope_co_load_groups(self._co_load_module_names)
        return client

    def init_communicators(self, device: torch.device | str | int):
        """Create a separate NCCL process group per server (sequentially — groups can't init
        concurrently). ``device`` is the trainer's GPU, which must differ from every server's.

        A server that fails to connect raises with its own error, after the clients already
        connected are closed.
        """
        if self._initialized:
            logger.warning("InferenceClientManager already initialized, skipping")
            return

        # Stored in the form the clients resolve: the shared snapshots are staged on it.
        device = resolve_sync_device(device)

        for i, config in enumerate(self.server_configs):
            url = config["url"]
            port = self._group_port(i)
            logger.info(
                f"Initializing {self._client_factory.BACKEND_NAME} weight-sync client "
                f"{i + 1}/{len(self.server_configs)}: {url} (port {port}, device {device})"
            )
            try:
                client = self._connect_client(i, device)
            except Exception as e:
                # Else the already-initialized clients hold their TCPStore listeners until atexit.
                self.close_communicators()
                raise RuntimeError(
                    f"{self._client_factory.BACKEND_NAME} weight-sync client {i + 1}/{len(self.server_configs)} "
                    f"for {url} (group_port {port}) failed to initialize: {type(e).__name__}: {e}"
                ) from e
            self._clients.append(client)
            logger.info(f"  Connected to {url}")

        self._device = device  # reconnect_client re-forms NCCL groups on the same trainer device
        self._initialized = True
        logger.info(f"InferenceClientManager initialized: {len(self._clients)} clients connected")

    def update_named_param(self, name: str, weights: torch.Tensor):
        """Send one pre-gathered named parameter to all servers.

        One read-only snapshot per param on the sync device is shared by reference across every
        client's buffer, so the staged chunk costs ~1× rather than N_servers×, and a client's
        flush/clear drops only its own references.

        The chunk decision is the manager's, not each client's: a shared snapshot may only be released
        once every server has sent the chunk holding it. The buffer is therefore drained here, on every
        server concurrently.
        """
        if not self._initialized:
            raise RuntimeError("InferenceClientManager not initialized. Call init_communicators() first.")
        if not self._clients:
            return
        # On the source, before the snapshot and before any chunk quiesces a server (this path never
        # reaches the client's own update_named_param, which is where the single-server check lives).
        validate_syncable_param(name, weights)
        # Flushed before the budget is exceeded, on the same rule the clients buffer by, and before
        # the snapshot, so the staged chunk never holds more than the budget plus this param.
        if starts_new_chunk(self._buffered_bytes, payload_bytes(weights), WEIGHT_SYNC_CHUNK_BYTES):
            self._flush_chunk_to_every_server()
        snapshot = snapshot_param(weights, self._device)
        for client in self._clients:
            client.buffer_param(name, snapshot)
        self._buffered_bytes += payload_bytes(snapshot)

    def scope_co_load_groups(self, module_names: Iterable[str]) -> None:
        """Bind every client's co-load groups to the served model (see the client method)."""
        self._co_load_module_names = tuple(module_names)
        for client in self._clients:
            client.scope_co_load_groups(self._co_load_module_names)

    def abort_weight_update(self):
        """Close the open update on every server after a failed sync; never raises (see the client)."""
        for client in self._clients:
            client.abort_weight_update()
        self._buffered_bytes = 0

    def _flush_chunk_to_every_server(self):
        """Send the buffered chunk to every server concurrently.

        The mid-gather half of the streamed sync: the update stays open on each server (the tail and
        the close come from ``reset_prefix_cache``), so this is the point where a chunk stops being
        replayable. A server that fails from here on is reported rather than reconnected. The shared
        snapshots are released as each client drains its buffer, i.e. only once the chunk is on the
        wire everywhere.
        """
        self._run_on_every_client(lambda index: self._clients[index].flush_chunk(), "chunk flush")
        # Every client holds the same snapshots, so what one kept back for a co-load partner they all did.
        self._buffered_bytes = max((client.buffered_bytes for client in self._clients), default=0)

    def reset_prefix_cache(self):
        """Send the tail chunk to all rollout servers and close their updates (no-op if nothing was buffered).

        Servers flush on concurrent threads, each client on its own NCCL communicator and streams,
        but they share the forwarding rank's GPU, its NICs and this process, so the fan-out costs
        the sum of the pushes rather than the slowest one and the stall grows with the server count.
        The async snapshot copies complete before any producer thread reads them. Returns only once
        every flush is done.

        Raises RuntimeError if a server still fails after one reconnect and re-flush attempt; the
        alternative would leave that server serving stale-policy rollouts.
        """
        if not self._initialized or not self._clients:
            return
        self._run_on_every_client(lambda index: self._clients[index].reset_prefix_cache(), "final flush")
        self._buffered_bytes = 0

    def _run_on_every_client(self, operation: Callable[[int], None], what: str) -> None:
        """Run ``operation(index)`` on every client concurrently, raising one aggregated failure.

        Indexed rather than closed over a client object: the recovery path below swaps a client in
        the pool, and the retry has to reach the replacement.
        """
        run = partial(self._with_recovery, operation, what)
        if len(self._clients) == 1:  # no thread-pool overhead for the single-server case
            errors = [run(0)]
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self._clients)) as executor:
                errors = list(executor.map(run, range(len(self._clients))))
        errors = [e for e in errors if e is not None]
        if errors:
            raise RuntimeError(
                f"Weight sync ({what}) failed for {len(errors)}/{len(self._clients)} "
                f"{self._client_factory.BACKEND_NAME} server(s) — "
                f"continuing would train on stale-policy rollouts. Failures: {'; '.join(errors)}"
            )

    def _with_recovery(self, operation: Callable[[int], None], what: str, index: int) -> str | None:
        """Run one client's step; on failure make one reconnect and retry attempt where that can work.

        Returns ``None`` on success (including a recovered step) or the error string for the
        aggregated raise. The failed client's buffer is retained through the reconnect attempt
        (``reconnect_client`` moves the buffered references onto the fresh client) and dropped only
        when the retry also fails; clearing first would lose the params the retry needs.

        A client that has already streamed a chunk cannot be recovered: the trainer keeps no copy of
        what went out, so a fresh engine would receive only the remainder and serve a model that is
        part old and part new. That case is reported with the reason.
        """
        client = self._clients[index]
        url = self.server_configs[index]["url"]
        try:
            operation(index)
            return None
        except Exception as e:  # aggregated and re-raised by the caller
            first_error = e
        if not client.can_replay_sync:
            client.drain_param_buffer()  # dropped rather than re-broadcast: this sync cannot be replayed
            return (
                f"{url}: {first_error} (not retried: this sync was already streaming, so the engine "
                f"holds part of the new weights and the trainer no longer has the rest to re-send — "
                f"restart that server before serving again)"
            )
        logger.error(f"Weight-sync {what} failed for {url}: {first_error}; attempting one reconnect")
        try:
            with self._reconnect_lock:  # NCCL groups cannot form concurrently
                self.reconnect_client(index)  # swaps the pool entry `operation` reaches
            operation(index)
            logger.warning(f"Weight sync to {url} recovered after reconnect")
            return None
        except Exception as retry_error:  # aggregated and re-raised by the caller
            # The pool entry is the only client that can still hold this sync's snapshot: a failed
            # reconnect leaves the original there, and a failed retry leaves the new client, whose
            # predecessor was already drained by ``reconnect_client``.
            self._clients[index].drain_param_buffer()
            return f"{url}: {first_error} (reconnect retry failed: {retry_error})"

    def reconnect_client(self, index: int):
        """Rebuild the weight-sync client for one server after an engine container restart.

        The old NCCL communicator cannot re-form, so this builds a fresh client on the server's
        configured ``group_port``: retire the old client (draining its pending buffer and lifting any
        pause it left on the server, which also releases the port listener), wait for the server, form
        the NCCL group on the trainer device, move the drained buffer onto the new client, and swap it
        into the pool. Returns the new client; raises if the server stays unreachable.

        The replacement starts on base-checkpoint weights, so it is only usable inside
        :meth:`_with_recovery`, whose retained buffer is this sync's full param snapshot and is
        re-flushed immediately. Any other caller must give the new client a full push of its own.
        """
        if not self._initialized or self._device is None:
            raise RuntimeError("InferenceClientManager.reconnect_client requires init_communicators() first.")
        if not 0 <= index < len(self._clients):
            raise IndexError(f"Client index {index} out of range (have {len(self._clients)} clients)")
        url = self.server_configs[index]["url"]
        old = self._clients[index]
        logger.warning(
            f"Reconnecting weight-sync client for {url} on group_port {self._group_port(index)}; "
            f"full re-sync required before rollouts"
        )
        # Retire the old client first: it holds the /resume for the pause its failed sync left behind.
        buffered = old.drain_param_buffer()
        try:
            old.close_communicator()
        except Exception as e:  # the old client is already dead; never mask the rebuild
            logger.warning(f"Error closing stale client for {url}: {e}")
        client = self._connect_client(index, self._device)
        for name, snapshot in buffered:
            client.buffer_param(name, snapshot)
        self._clients[index] = client
        return client

    def close_communicators(self):
        """Close every engine client connection and clean up."""
        for client in self._clients:
            try:
                client.close_communicator()
            except Exception as e:
                logger.warning(f"Error closing client: {e}")

        self._clients = []
        self._initialized = False
        logger.info("InferenceClientManager closed all connections")

    @property
    def num_servers(self) -> int:
        """Number of rollout servers configured (or connected if initialized)."""
        if self._initialized:
            return len(self._clients)
        return len(self.server_configs)
