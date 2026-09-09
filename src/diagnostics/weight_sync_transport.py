"""Weight-sync transport preflight: which NCCL transport the trainer↔rollout-server group forms on.

Every failure of that group is a hang, not an error: two containers on different NCCL nets, on
different aws-ofi-nccl builds, or with cuMem enabled on one side only all form the group and then
spin at the first collective. This forms the group against a live server with the toolkit's own
client, pushes one real checkpoint parameter (its unchanged value, so the served model is unchanged),
and reads the transport NCCL chose out of its own debug log — ``NET/Libfabric/.../GDRDMA`` is EFA
with GPUDirect, ``NET/IB`` NCCL's own InfiniBand transport, ``NET/Socket`` TCP over the host network,
``P2P/CUMEM`` same-host CUDA IPC and ``SHM`` same-host shared memory.

The pushed value is the checkpoint's own, so the served model is unchanged as long as the
checkpoint named (``--model-id``, else the id the server advertises) is the one the server loaded;
a greedy completion before and after the push is the check, coarse enough to survive numeric
jitter and fine enough to catch a wrong tensor.

The NCCL debug variables are set here, before the first NCCL call, and NCCL reads them once per
process; the preflight therefore runs in a fresh process (``scripts/profiling/weight_sync_transport.py``)
rather than inside a trainer. Only the trainer side of the group is observable here; the server's
own log carries its half.
"""

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

from src.checkpoint.format import SAFETENSORS_INDEX_FILE, SAFETENSORS_WEIGHTS_FILE
from src.distributed.nccl.clients.base import BaseWeightSyncClient, payload_bytes, resolve_sync_device
from src.distributed.nccl.registry import resolve_weight_sync_client

logger = logging.getLogger(__name__)

# The parameter pushed when none is named: the input embedding, which every family carries under its
# language model (``model.embed_tokens.weight``; ``model.language_model.embed_tokens.weight`` on the
# multimodal ones).
DEFAULT_PARAM_SUFFIX = "embed_tokens.weight"
# Transport verdicts, keyed by the prefix NCCL prints after ``via`` on each channel line.
VERDICT_EFA = "efa"
VERDICT_IB = "ib"
VERDICT_SOCKET = "socket"
VERDICT_P2P = "p2p"
VERDICT_SHM = "shm"
VERDICT_UNKNOWN = "unknown"
VERDICTS = (VERDICT_EFA, VERDICT_IB, VERDICT_SOCKET, VERDICT_P2P, VERDICT_SHM)
_VERDICT_PREFIXES = (
    ("NET/Libfabric", VERDICT_EFA),
    ("NET/IB", VERDICT_IB),
    ("NET/Socket", VERDICT_SOCKET),
    ("P2P/", VERDICT_P2P),
    ("SHM", VERDICT_SHM),
)
_NCCL_DEBUG_ENV = {"NCCL_DEBUG": "INFO", "NCCL_DEBUG_SUBSYS": "INIT,NET,P2P"}
_LOG_STEM = "nccl-weight-sync-preflight"
# ``NET/<net>/<device>[(proxy rank)][/GDRDMA]``, ``P2P/<kind>[/<mode>...][ pointer]`` or ``SHM[/<mode>...]``.
_CHANNEL_RE = re.compile(
    r"\bvia (NET/[^/\s]+/\d+(?:\(\d+\))?(?:/GDRDMA)?|P2P/[A-Za-z]+(?:/[A-Za-z]+)*(?: pointer)?|SHM(?:/[A-Za-z]+)*)"
)
_PLUGIN_RE = re.compile(r"NET/OFI Initializing aws-ofi-nccl (\S+)")
_NET_PLUGIN_RE = re.compile(r"NET/Plugin: Loaded net plugin (\S+) \(v(\d+)\)")
_PROVIDER_RE = re.compile(r"NET/OFI Selected provider is (\S+), fabric is (\S+)")
_PLUGIN_MISSING_RE = re.compile(r"NET/Plugin: Could not find: (.+?)\s*$", re.M)
# The plugin probes each endpoint for in-order RDMA writes and sends at init; the answer comes from
# rdma-core's EFA provider, so it differs between userspace builds. Where a probe fails the plugin
# forces NCCL_PROTO=simple itself, on a separate line it skips on the platforms it exempts.
_IN_ORDER_UNSUPPORTED_RE = re.compile(r"FI_OPT_EFA_\w+_IN_ORDER_ALIGNED_128_BYTES not supported")
_FORCED_SIMPLE_PROTOCOL_RE = re.compile(r"NET/OFI Need to force simple protocol")
_NCCL_VERSION_RE = re.compile(r"NCCL version (\S+)")
_COMPLETION_PROMPT = "The capital of France is"
_COMPLETION_TOKENS = 8
_HTTP_TIMEOUT_S = 60.0


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
    # Both None without a plugin. The probe answer, and whether the plugin forced NCCL_PROTO=simple
    # on this side because of it.
    in_order_writes: bool | None = None
    forced_simple_protocol: bool | None = None

    @property
    def verdict(self) -> str:
        """One word for the data path: the first channel transport's family."""
        for transport in self.transports:
            for prefix, verdict in _VERDICT_PREFIXES:
                if transport.startswith(prefix):
                    return verdict
        return VERDICT_UNKNOWN

    @property
    def gpudirect(self) -> bool:
        return any(transport.endswith("/GDRDMA") for transport in self.transports)


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
    """Point NCCL's debug output at ``log_dir`` for this process; returns the file glob to read back.

    Must run before the first NCCL call in the process (NCCL reads these once). A process that already
    formed a torch.distributed group has an initialized NCCL and cannot be re-armed.
    """
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        raise RuntimeError(
            "torch.distributed is already initialized in this process: NCCL has read its debug "
            "settings and the preflight cannot observe the transport. Run it in a fresh process."
        )
    pattern = os.path.join(log_dir, f"{_LOG_STEM}.%p.log")
    os.environ.update(_NCCL_DEBUG_ENV)
    os.environ["NCCL_DEBUG_FILE"] = pattern
    # Named up front: a formation that hangs is the case this tool exists for, and the log is then
    # the only thing to read.
    logger.info(f"NCCL debug log: {pattern}")
    return os.path.join(log_dir, f"{_LOG_STEM}.*.log")


def read_nccl_log(pattern: str) -> str:
    return "".join(Path(path).read_text(encoding="utf-8", errors="replace") for path in sorted(glob.glob(pattern)))


def _checkpoint_file(model_id: str, filename: str, revision: str | None) -> str:
    if os.path.isdir(model_id):
        path = os.path.join(model_id, filename)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        return path
    try:
        return hf_hub_download(model_id, filename, revision=revision)
    except LocalEntryNotFoundError:
        raise  # not in the cache and the Hub unreachable: not the same as the file not existing
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
    """One tensor of the checkpoint ``model_id`` serves, from its safetensors shard: ``name``, or the
    input embedding when ``None``. Returns the resolved name with the tensor.

    Only the index and the shard holding the tensor are fetched; a hub checkpoint lands in the
    Hugging Face cache like any other download.
    """
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


def served_model_id(client: BaseWeightSyncClient) -> str:
    """The model id the server advertises on the OpenAI route both engines serve."""
    cards = client.served_model_cards(timeout=_HTTP_TIMEOUT_S)
    if not cards:
        raise RuntimeError(f"{client.base_url}/v1/models lists no model")
    return cards[0]["id"]


def greedy_completion(client: BaseWeightSyncClient, model_id: str) -> str:
    """A deterministic completion, compared before and after the push to prove the model unchanged."""
    resp = client.session.post(
        f"{client.base_url}/v1/completions",
        json={
            "model": model_id,
            "prompt": _COMPLETION_PROMPT,
            "max_tokens": _COMPLETION_TOKENS,
            "temperature": 0,
            "seed": 0,
        },
        timeout=_HTTP_TIMEOUT_S,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["text"]


def run_preflight(
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
) -> dict:
    """Form the group, push ``param`` ``rounds`` times, tear down, and report transport plus rates.

    Returns a JSON-ready dict; ``verdict`` is one of the transport words above and ``served_unchanged``
    is False when the push altered the served model, i.e. the checkpoint read here is not the one the
    server loaded.
    """
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
    # The completion is requested under the served name (vLLM rejects any other); the checkpoint
    # read here is the one the server loaded, named by --model-id when the served id is not it.
    served = served_model_id(client)
    checkpoint = model_id or served
    param, tensor = load_checkpoint_tensor(checkpoint, param, revision=revision, device=str(torch_device))
    nbytes = payload_bytes(tensor)
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
            client.sync_model_weights([(param, tensor)])
            if torch_device.type == "cuda":
                torch.cuda.synchronize(torch_device)
            rates_gbps.append(nbytes / (time.perf_counter() - started) / 1e9)
    finally:
        started = time.perf_counter()
        client.close_communicator()
        close_seconds = time.perf_counter() - started
    after = greedy_completion(client, served)

    report = parse_nccl_log(read_nccl_log(log_glob))
    return {
        "server_url": server_url,
        "backend": backend,
        "model_id": checkpoint,
        "served_model": served,
        "param": param,
        "payload_gb": nbytes / 1e9,
        "init_seconds": init_seconds,
        "close_seconds": close_seconds,
        "rates_gbps": rates_gbps,
        "served_unchanged": before == after,
        "verdict": report.verdict,
        "gpudirect": report.gpudirect,
        "nccl_log_dir": log_dir,
        **asdict(report),
    }


def format_report(result: dict) -> str:
    """Human summary of :func:`run_preflight`'s result."""
    rates = result["rates_gbps"]
    lines = [
        f"weight-sync transport preflight: {result['backend']} at {result['server_url']}",
        f"  transport      {result['verdict']}  ({', '.join(result['transports']) or 'no channel lines'})"
        + ("  GPUDirect RDMA" if result["gpudirect"] else ""),
        f"  nccl           {result['nccl_version'] or '?'}",
        f"  plugin         {result['plugin'] or 'none loaded'}"
        + (f"  net {result['net_plugin']} v{result['net_plugin_api']}" if result["net_plugin"] else ""),
        f"  provider       {result['provider'] or '-'}" + (f" ({result['fabric']})" if result["fabric"] else ""),
        f"  payload        {result['param']} of {result['model_id']}, {result['payload_gb']:.3f} GB"
        + (f" (served as {result['served_model']})" if result["served_model"] != result["model_id"] else ""),
        f"  group          formed in {result['init_seconds']:.2f}s, closed in {result['close_seconds']:.2f}s",
        "  rounds         "
        + (", ".join(f"{rate:.1f} GB/s" for rate in rates) if rates else "none")
        + " (one parameter per push: a latency-bound floor, not the full-model rate)",
        f"  served model   {'unchanged' if result['served_unchanged'] else 'CHANGED by the push'}",
    ]
    if result["plugin_missing"]:
        lines.append(f"  missing plugin {', '.join(result['plugin_missing'])} (NCCL fell back)")
    if result["forced_simple_protocol"] or result["in_order_writes"] is False:
        forced = "forced NCCL_PROTO=simple" if result["forced_simple_protocol"] else "left NCCL_PROTO as set"
        lines.append(
            f"  protocol       no in-order RDMA from this side's rdma-core; the plugin {forced}. The "
            f"server must match: the same EFA userspace build, or NCCL_PROTO=simple on both ends"
        )
    lines.append(f"  nccl log       {result['nccl_log_dir']}")
    return "\n".join(lines)
