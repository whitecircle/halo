"""Which NCCL transport the trainer↔rollout-server weight-sync group forms on, read from NCCL's own debug log. Every
failure of that group is a hang at the first collective (different NCCL nets, aws-ofi-nccl builds, cuMem on one
side only), so the check forms the group against a live server, pushes one checkpoint tensor unchanged through the
trainer's streamed path and parses the transport, in a fresh process: NCCL reads its debug settings once."""

import glob
import json
import logging
import os
import re
import tempfile
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import (
    EntryNotFoundError,
    HFValidationError,
    LocalEntryNotFoundError,
    RepositoryNotFoundError,
)
from safetensors import safe_open

from halo.checkpoint.format import SAFETENSORS_INDEX_FILE, SAFETENSORS_WEIGHTS_FILE
from halo.runtime.ranks import has_process_group
from halo.weight_sync.chunking import resolve_sync_device
from halo.weight_sync.clients.base import POST_TIMEOUT_S, WeightSyncClientBase
from halo.weight_sync.registry import resolve_weight_sync_client

logger = logging.getLogger(__name__)

# The parameter pushed when none is named: the input embedding every family carries under its language model.
DEFAULT_PARAM_SUFFIX = "embed_tokens.weight"
# Transport verdict → the prefix NCCL prints after ``via`` on a channel line of that transport:
# ``NET/Libfabric/.../GDRDMA`` is EFA with GPUDirect, ``P2P/CUMEM`` same-host CUDA IPC.
TRANSPORT_PREFIXES = {"efa": "NET/Libfabric", "ib": "NET/IB", "socket": "NET/Socket", "p2p": "P2P/", "shm": "SHM"}
VERDICT_UNKNOWN = "unknown"
_LOG_STEM = "nccl-weight-sync-preflight"
# ``NET/<net>/<device>[(proxy rank)][/GDRDMA]``, ``P2P/<kind>[/<mode>...][ pointer]`` or ``SHM[/<mode>...]``.
_CHANNEL_RE = re.compile(
    r"\bvia (NET/[^/\s]+/\d+(?:\(\d+\))?(?:/GDRDMA)?|P2P/[A-Za-z]+(?:/[A-Za-z]+)*(?: pointer)?|SHM(?:/[A-Za-z]+)*)"
)
_PLUGIN_RE = re.compile(r"NET/OFI Initializing aws-ofi-nccl (\S+)")
_NET_PLUGIN_RE = re.compile(r"NET/Plugin: Loaded net plugin (\S+) \(v(\d+)\)")
_PROVIDER_RE = re.compile(r"NET/OFI Selected provider is (\S+), fabric is (\S+)")
_PLUGIN_MISSING_RE = re.compile(r"NET/Plugin: Could not find: (.+?)\s*$", re.M)
# The plugin's init probe for in-order RDMA writes and sends, answered by rdma-core's EFA provider (so it differs
# between userspace builds); a failed probe forces NCCL_PROTO=simple, on a line exempt platforms skip.
_IN_ORDER_UNSUPPORTED_RE = re.compile(r"FI_OPT_EFA_\w+_IN_ORDER_ALIGNED_128_BYTES not supported")
_FORCED_SIMPLE_PROTOCOL_RE = re.compile(r"NET/OFI Need to force simple protocol")
_NCCL_VERSION_RE = re.compile(r"NCCL version (\S+)")


@dataclass
class TransportReport:
    """What the NCCL debug log says about the group the trainer formed."""

    nccl_version: str | None = None
    plugin: str | None = None
    net_plugin: str | None = None
    net_plugin_api: int | None = None
    provider: str | None = None
    fabric: str | None = None
    plugin_missing: list[str] = field(default_factory=list)
    transports: list[str] = field(default_factory=list)
    # Both None without a plugin: the in-order probe's answer, and whether it forced NCCL_PROTO=simple here.
    in_order_writes: bool | None = None
    forced_simple_protocol: bool | None = None

    @property
    def verdict(self) -> str:
        """One word for the data path: the first channel transport's family."""
        for transport in self.transports:
            for verdict, prefix in TRANSPORT_PREFIXES.items():
                if transport.startswith(prefix):
                    return verdict
        return VERDICT_UNKNOWN

    @property
    def gpudirect(self) -> bool:
        return any(transport.endswith("/GDRDMA") for transport in self.transports)


@dataclass(frozen=True)
class TransportCheckResult:
    """One check: the run, then the :class:`TransportReport` fields; ``asdict`` is the ``--json`` output."""

    server_url: str
    backend: str
    model_id: str
    served_model: str
    param: str
    payload_gb: float
    init_seconds: float
    close_seconds: float
    rates_gbps: list[float]
    # False when the push altered the served model: the checkpoint read here is not the one the server loaded.
    served_unchanged: bool
    verdict: str
    gpudirect: bool
    nccl_log_dir: str
    nccl_version: str | None
    plugin: str | None
    net_plugin: str | None
    net_plugin_api: int | None
    provider: str | None
    fabric: str | None
    plugin_missing: list[str]
    transports: list[str]
    in_order_writes: bool | None
    forced_simple_protocol: bool | None


def parse_nccl_log(text: str) -> TransportReport:
    """Read the transport facts out of an ``NCCL_DEBUG=INFO`` log (``INIT,NET,P2P`` subsystems)."""
    report = TransportReport()
    if match := _NCCL_VERSION_RE.search(text):
        report.nccl_version = match.group(1)
    if match := _PLUGIN_RE.search(text):
        report.plugin = match.group(1)
    if match := _NET_PLUGIN_RE.search(text):
        report.net_plugin, report.net_plugin_api = match.group(1), int(match.group(2))
    if match := _PROVIDER_RE.search(text):
        report.provider, report.fabric = match.group(1), match.group(2)
    if report.plugin is not None:
        report.in_order_writes = _IN_ORDER_UNSUPPORTED_RE.search(text) is None
        report.forced_simple_protocol = _FORCED_SIMPLE_PROTOCOL_RE.search(text) is not None
    report.plugin_missing = sorted(set(_PLUGIN_MISSING_RE.findall(text)))
    seen: dict[str, None] = {}
    for match in _CHANNEL_RE.finditer(text):
        seen.setdefault(match.group(1))
    report.transports = list(seen)
    return report


def arm_nccl_debug(log_dir: str) -> str:
    """Point NCCL's debug output at ``log_dir``, before the process's first NCCL call; returns the glob to read."""
    if has_process_group():
        raise RuntimeError(
            "torch.distributed is already initialized in this process: NCCL has read its debug "
            "settings and the preflight cannot observe the transport. Run it in a fresh process."
        )
    pattern = os.path.join(log_dir, f"{_LOG_STEM}.%p.log")
    os.environ.update({"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,NET,P2P"})
    os.environ["NCCL_DEBUG_FILE"] = pattern
    # Named up front: a hung formation, the case this tool exists for, leaves the log the only thing to read.
    logger.info(f"NCCL debug log: {pattern}")
    return os.path.join(log_dir, f"{_LOG_STEM}.*.log")


def _checkpoint_file(model_id: str, filename: str, revision: str | None) -> str:
    if os.path.isdir(model_id):
        path = os.path.join(model_id, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        return path
    try:
        return hf_hub_download(model_id, filename, revision=revision)
    except LocalEntryNotFoundError:
        # Not the repo lacking the file but an uncached one offline: still a FileNotFoundError, so an uncached
        # index falls through to the single-file layout.
        raise
    except EntryNotFoundError as e:
        raise FileNotFoundError(f"{model_id}/{filename}") from e
    except (HFValidationError, RepositoryNotFoundError) as e:
        # The served id is whatever the server was launched with: a path on its own host, or an alias.
        raise ValueError(
            f"{model_id!r} is neither a directory on this host nor a Hub repo it can read. Pass "
            f"--model-id (and --revision) naming the checkpoint the server loaded."
        ) from e


def _resolve_param(name: str | None, keys: list[str], model_id: str) -> str:
    """``name`` checked against the checkpoint's keys, or the first :data:`DEFAULT_PARAM_SUFFIX` key."""
    if name is None:
        candidates = sorted(key for key in keys if key.endswith(DEFAULT_PARAM_SUFFIX))
        if len(candidates) != 1:
            raise KeyError(
                f"{model_id} has {len(candidates)} *{DEFAULT_PARAM_SUFFIX} parameters ({candidates}); name "
                f"the one to push with --param"
            )
        return candidates[0]
    if name not in keys:
        raise KeyError(
            f"{name!r} is not a parameter of {model_id}; pick one with --param, e.g. "
            f"{sorted(keys)[:8]} ... ({len(keys)} keys)"
        )
    return name


def load_checkpoint_tensor(
    model_id: str, name: str | None, *, revision: str | None = None, device: str = "cpu"
) -> tuple[str, torch.Tensor]:
    """``(name, tensor)`` of one checkpoint tensor (the input embedding when ``name`` is ``None``), fetching only
    the index and the shard holding it."""
    try:
        index_path = _checkpoint_file(model_id, SAFETENSORS_INDEX_FILE, revision)
    except FileNotFoundError:
        shard = _checkpoint_file(model_id, SAFETENSORS_WEIGHTS_FILE, revision)
        with safe_open(shard, framework="pt", device=device) as fh:
            name = _resolve_param(name, fh.keys(), model_id)
            return name, fh.get_tensor(name)
    with open(index_path, encoding="utf-8") as fh:
        weight_map = json.load(fh)["weight_map"]
    name = _resolve_param(name, list(weight_map), model_id)
    shard = _checkpoint_file(model_id, weight_map[name], revision)
    with safe_open(shard, framework="pt", device=device) as fh:
        keys = fh.keys()
        if name not in keys:
            raise KeyError(f"{name!r} is not in {shard}, which the index names for it")
        return name, fh.get_tensor(name)


def _served_model_id(client: WeightSyncClientBase) -> str:
    """The model id the server advertises on ``/v1/models``."""
    cards = client.request("GET", "/v1/models", retry=True, timeout=POST_TIMEOUT_S).json().get("data", [])
    if not cards:
        raise RuntimeError(f"{client.base_url}/v1/models lists no model")
    return cards[0]["id"]


def greedy_completion(client: WeightSyncClientBase, model_id: str) -> str:
    """A deterministic completion, compared before and after the push to prove the model unchanged: coarse
    enough to survive numeric jitter, fine enough to catch a wrong tensor."""
    resp = client.request(
        "POST",
        "/v1/completions",
        retry=True,
        timeout=POST_TIMEOUT_S,
        json={"model": model_id, "prompt": "The capital of France is", "max_tokens": 8, "temperature": 0, "seed": 0},
    )
    return resp.json()["choices"][0]["text"]


def run_transport_check(
    *,
    server_url: str,
    backend: str,
    group_port: int,
    group_host: str | None,
    device: str,
    param: str | None,
    model_id: str | None,
    revision: str | None,
    rounds: int,
    connection_timeout: float,
) -> TransportCheckResult:
    """Form the group, push ``param`` ``rounds`` times, tear down, and report the trainer side's transport and
    rates (the server's own log carries its half)."""
    log_dir = tempfile.mkdtemp(prefix="weight-sync-preflight-")
    log_glob = arm_nccl_debug(log_dir)
    torch_device = resolve_sync_device(device)
    if torch_device.type == "cuda":
        torch.cuda.set_device(torch_device)

    client_cls = resolve_weight_sync_client(backend)
    client = client_cls(
        base_url=server_url,
        group_port=group_port,
        connection_timeout=connection_timeout,
        group_host=group_host,
    )
    # The completion goes under the served name (vLLM rejects any other); --model-id names the checkpoint.
    served = _served_model_id(client)
    checkpoint = model_id or served
    param, tensor = load_checkpoint_tensor(checkpoint, param, revision=revision, device=str(torch_device))
    nbytes = tensor.nbytes
    before = greedy_completion(client, served)

    started = time.perf_counter()
    client.init_communicator(device=torch_device)
    init_seconds = time.perf_counter() - started
    rates_gbps: list[float] = []
    try:
        for _ in range(rounds):
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)
            started = time.perf_counter()
            client.update_named_param(param, tensor)
            client.finish_weight_update()
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)
            rates_gbps.append(nbytes / (time.perf_counter() - started) / 1e9)
    finally:
        started = time.perf_counter()
        client.close_communicator()
        close_seconds = time.perf_counter() - started
    after = greedy_completion(client, served)

    nccl_log = "".join(
        Path(path).read_text(encoding="utf-8", errors="replace") for path in sorted(glob.glob(log_glob))
    )
    report = parse_nccl_log(nccl_log)
    return TransportCheckResult(
        server_url=server_url,
        backend=backend,
        model_id=checkpoint,
        served_model=served,
        param=param,
        payload_gb=nbytes / 1e9,
        init_seconds=init_seconds,
        close_seconds=close_seconds,
        rates_gbps=rates_gbps,
        served_unchanged=before == after,
        verdict=report.verdict,
        gpudirect=report.gpudirect,
        nccl_log_dir=log_dir,
        **asdict(report),
    )


def format_report(result: TransportCheckResult) -> str:
    """Human summary of :func:`run_transport_check`'s result."""
    rates = result.rates_gbps
    lines = [
        f"weight-sync transport preflight: {result.backend} at {result.server_url}",
        f"  transport      {result.verdict}  ({', '.join(result.transports) or 'no channel lines'})"
        + ("  GPUDirect RDMA" if result.gpudirect else ""),
        f"  nccl           {result.nccl_version or '?'}",
        f"  plugin         {result.plugin or 'none loaded'}"
        + (f"  net {result.net_plugin} v{result.net_plugin_api}" if result.net_plugin else ""),
        f"  provider       {result.provider or '-'}" + (f" ({result.fabric})" if result.fabric else ""),
        f"  payload        {result.param} of {result.model_id}, {result.payload_gb:.3f} GB"
        + (f" (served as {result.served_model})" if result.served_model != result.model_id else ""),
        f"  group          formed in {result.init_seconds:.2f}s, closed in {result.close_seconds:.2f}s",
        "  rounds         "
        + (", ".join(f"{rate:.1f} GB/s" for rate in rates) if rates else "none")
        + " (one parameter per push: a latency-bound floor, not the full-model rate)",
        f"  served model   {'unchanged' if result.served_unchanged else 'CHANGED by the push'}",
    ]
    if result.plugin_missing:
        lines.append(f"  missing plugin {', '.join(result.plugin_missing)} (NCCL fell back)")
    if result.forced_simple_protocol or result.in_order_writes is False:
        forced = "forced NCCL_PROTO=simple" if result.forced_simple_protocol else "left NCCL_PROTO as set"
        lines.append(
            f"  protocol       no in-order RDMA from this side's rdma-core; the plugin {forced}. The "
            f"server must match: the same EFA userspace build, or NCCL_PROTO=simple on both ends"
        )
    lines.append(f"  nccl log       {result.nccl_log_dir}")
    return "\n".join(lines)
