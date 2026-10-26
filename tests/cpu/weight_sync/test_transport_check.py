#!/usr/bin/env python
"""The weight-sync transport preflight reads NCCL's own debug log and a checkpoint shard.

The transport verdict is the whole point of the tool: a parser that misses the ``via NET/...``
channel lines, or reads GPUDirect off the wrong field, turns a socket fallback into a passing
preflight. The excerpts below are verbatim NCCL 2.31 output for the transports the trainer meets:
EFA with GPUDirect, NCCL's own InfiniBand transport, the socket fallback, same-host CUDA IPC and
shared memory. The push is pinned against a recording client, and the CLI's gate (``--expect``,
the unchanged-server check) against a stubbed preflight result.

    python tests/cpu/weight_sync/test_transport_check.py
"""

import json
import os
import tempfile
from dataclasses import asdict

import pytest
import torch
from safetensors.torch import save_file

from halo.weight_sync import transport_check as wst
from halo.weight_sync.chunking import resolve_sync_device
from halo.weight_sync.clients.vllm import VLLMWeightSyncClient
from tests.common.fake_engines import FakeVLLMServer
from tests.common.repo import load_script_module
from tests.common.weight_sync import Wire

preflight_cli = load_script_module("scripts/profiling/weight_sync_transport.py", register=True)

EFA_LOG = """\
host:1:257 [0] NCCL INFO NCCL version 2.31.2+cuda13.3
host:1:257 [0] NCCL INFO NET/Plugin: Loaded net plugin Libfabric (v12)
host:1:257 [0] NCCL INFO NET/OFI Initializing aws-ofi-nccl git-1f0a976
host:1:257 [0] NCCL INFO NET/OFI Selected provider is efa, fabric is efa-direct (found 16 nics)
host:1:257 [0] NCCL INFO Channel 00/0 : 1[0] -> 0[0] [receive] via NET/Libfabric/0/GDRDMA/flush=None
host:1:257 [0] NCCL INFO Channel 01/0 : 1[0] -> 0[0] [receive] via NET/Libfabric/0/GDRDMA/flush=None
host:1:257 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[0] [send] via NET/Libfabric/0/GDRDMA
host:1:257 [0] NCCL INFO Channel 01/0 : 0[0] -> 1[0] [send] via NET/Libfabric/0/GDRDMA
host:1:257 [0] NCCL INFO Connected all rings, use ring PXN 0 GDR 1
"""

IB_LOG = """\
host:1:257 [0] NCCL INFO NCCL version 2.31.2+cuda13.3
host:1:257 [0] NCCL INFO NET/IB : Using [0]mlx5_0:1/IB [RO]; OOB eth0:10.0.0.7<0>
host:1:257 [0] NCCL INFO Channel 00/0 : 1[0] -> 0[0] [receive] via NET/IB/0/GDRDMA
host:1:257 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[0] [send] via NET/IB/0/GDRDMA
"""

SOCKET_LOG = """\
host:1:257 [0] NCCL INFO NCCL version 2.31.2+cuda13.3
host:1:257 [0] NCCL INFO NET/Plugin: Could not find: ofi libnccl-net-ofi.so
host:1:257 [0] NCCL INFO Channel 00/0 : 1[0] -> 0[0] [receive] via NET/Socket/0
host:1:257 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[0] [send] via NET/Socket/0
"""

P2P_LOG = """\
host:1:1 [0] NCCL INFO NCCL version 2.31.2+cuda13.3
host:1:1 [0] NCCL INFO Channel 30/0 : 0[0] -> 1[6] via P2P/CUMEM
host:1:1 [0] NCCL INFO Channel 31/0 : 0[0] -> 1[6] via P2P/CUMEM
"""

SHM_LOG = """\
host:1:1 [0] NCCL INFO NCCL version 2.31.2+cuda13.3
host:1:1 [0] NCCL INFO Channel 00/0 : 0[0] -> 1[6] via SHM/direct/direct
"""

_PROBE_FAILED = "host:1:257 [0] NCCL INFO NET/OFI Setting FI_OPT_EFA_WRITE_IN_ORDER_ALIGNED_128_BYTES not supported.\n"
_SENDRECV_PROBE_FAILED = (
    "host:1:257 [0] NCCL INFO NET/OFI Setting FI_OPT_EFA_SENDRECV_IN_ORDER_ALIGNED_128_BYTES not supported.\n"
)
_FORCED_SIMPLE = (
    "host:1:257 [0] NCCL INFO NET/OFI Need to force simple protocol: "
    "FI_OPT_EFA_WRITE_IN_ORDER_ALIGNED_128_BYTES not supported\n"
)
_PROVIDER_LINE = "host:1:257 [0] NCCL INFO NET/OFI Selected provider"
# The --json keys, in order: what a launch script parsing the check's output reads.
_JSON_KEYS = [
    "server_url",
    "backend",
    "model_id",
    "served_model",
    "param",
    "payload_gb",
    "init_seconds",
    "close_seconds",
    "rates_gbps",
    "served_unchanged",
    "verdict",
    "gpudirect",
    "nccl_log_dir",
    "nccl_version",
    "plugin",
    "net_plugin",
    "net_plugin_api",
    "provider",
    "fabric",
    "plugin_missing",
    "transports",
    "in_order_writes",
    "forced_simple_protocol",
]


def test_efa_log_reads_as_efa_with_gpudirect_and_the_plugin_identity():
    report = wst.parse_nccl_log(EFA_LOG)
    assert report.verdict == "efa"
    assert report.gpudirect is True
    assert report.transports == ["NET/Libfabric/0/GDRDMA"], "channel lines must be deduplicated, flush suffix dropped"
    assert report.plugin == "git-1f0a976"
    assert (report.net_plugin, report.net_plugin_api) == ("Libfabric", 12)
    assert (report.provider, report.fabric) == ("efa", "efa-direct")
    assert report.nccl_version == "2.31.2+cuda13.3"
    assert report.plugin_missing == []
    assert (report.in_order_writes, report.forced_simple_protocol) == (True, False)


def test_a_plugin_that_forced_the_simple_protocol_is_reported():
    """The forced protocol is invisible in a passing run and hangs a peer on a different rdma-core
    build; a report that only names the transport would call that pair healthy. The probe answer
    and the forcing are separate lines: the plugin skips the forcing on platforms it exempts."""
    probe_failed = EFA_LOG.replace(_PROVIDER_LINE, _PROBE_FAILED + _PROVIDER_LINE)
    forced = EFA_LOG.replace(_PROVIDER_LINE, _PROBE_FAILED + _FORCED_SIMPLE + _PROVIDER_LINE)
    assert (
        wst.parse_nccl_log(probe_failed).in_order_writes,
        wst.parse_nccl_log(probe_failed).forced_simple_protocol,
    ) == (
        False,
        False,
    )
    assert (wst.parse_nccl_log(forced).in_order_writes, wst.parse_nccl_log(forced).forced_simple_protocol) == (
        False,
        True,
    )
    sendrecv = EFA_LOG.replace(_PROVIDER_LINE, _SENDRECV_PROBE_FAILED + _FORCED_SIMPLE + _PROVIDER_LINE)
    assert wst.parse_nccl_log(sendrecv).in_order_writes is False, "the send/recv in-order probe is one of the two"
    socket = wst.parse_nccl_log(SOCKET_LOG)
    assert (socket.in_order_writes, socket.forced_simple_protocol) == (None, None), "no plugin, no probe"


def test_infiniband_reads_as_ib_not_efa():
    report = wst.parse_nccl_log(IB_LOG)
    assert report.verdict == "ib"
    assert report.gpudirect is True
    assert report.plugin is None and report.provider is None


def test_socket_fallback_reads_as_socket_and_names_the_missing_plugin():
    report = wst.parse_nccl_log(SOCKET_LOG)
    assert report.verdict == "socket"
    assert report.gpudirect is False
    assert report.plugin is None
    assert report.plugin_missing == ["ofi libnccl-net-ofi.so"]


def test_same_host_cuda_ipc_reads_as_p2p_and_shared_memory_as_shm():
    assert wst.parse_nccl_log(P2P_LOG).verdict == "p2p"
    assert wst.parse_nccl_log(P2P_LOG).transports == ["P2P/CUMEM"]
    assert wst.parse_nccl_log(SHM_LOG).verdict == "shm"


def test_a_proxied_net_channel_keeps_its_gpudirect_suffix():
    """NCCL's PXN send form puts the proxy rank before ``/GDRDMA``; the parser must not drop the
    suffix on it and report a GPUDirect path as host-staged."""
    proxied = EFA_LOG.replace("[send] via NET/Libfabric/0/GDRDMA", "[send] via NET/Libfabric/0(1)/GDRDMA")
    report = wst.parse_nccl_log(proxied)
    assert "NET/Libfabric/0(1)/GDRDMA" in report.transports
    assert report.verdict == "efa" and report.gpudirect is True


def test_a_log_without_channel_lines_is_unknown_not_a_guess():
    assert wst.parse_nccl_log("host:1:1 [0] NCCL INFO NCCL version 2.31.2+cuda13.3\n").verdict == wst.VERDICT_UNKNOWN


@pytest.fixture
def nccl_debug_env(monkeypatch):
    """The NCCL debug variables, restored once the test ends."""
    for key in ("NCCL_DEBUG", "NCCL_DEBUG_SUBSYS", "NCCL_DEBUG_FILE"):
        # Recorded whether or not the key exists, so the arming cannot leak past the test into the
        # torchrun subprocesses the GPU tier spawns with this process's environment.
        monkeypatch.setenv(key, os.environ.get(key, ""))


@pytest.mark.usefixtures("nccl_debug_env")
def test_arming_sets_the_debug_variables_before_nccl_reads_them(tmp_path):
    pattern = wst.arm_nccl_debug(str(tmp_path))
    assert os.environ["NCCL_DEBUG"] == "INFO"
    assert "NET" in os.environ["NCCL_DEBUG_SUBSYS"]
    assert os.environ["NCCL_DEBUG_FILE"].startswith(str(tmp_path))
    assert pattern.startswith(str(tmp_path)) and pattern.endswith(".log")


def test_arming_refuses_a_process_whose_nccl_already_read_its_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="fresh process"):
        wst.arm_nccl_debug(str(tmp_path))


def test_checkpoint_tensor_comes_from_the_shard_the_index_names(tmp_path):
    """The same key in two shards with different values: only the index says which one is the
    checkpoint's, and a loader scanning shards in name order would push the wrong tensor."""
    stale = torch.zeros(3, 2, dtype=torch.bfloat16)
    embed = torch.arange(6, dtype=torch.bfloat16).reshape(3, 2)
    save_file({"model.embed_tokens.weight": stale}, str(tmp_path / "model-00001-of-00002.safetensors"))
    save_file(
        {"model.embed_tokens.weight": embed, "model.norm.weight": torch.ones(2)},
        str(tmp_path / "model-00002-of-00002.safetensors"),
    )
    (tmp_path / wst.SAFETENSORS_INDEX_FILE).write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": "model-00002-of-00002.safetensors",
                    "model.norm.weight": "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    name, tensor = wst.load_checkpoint_tensor(str(tmp_path), "model.embed_tokens.weight")
    assert name == "model.embed_tokens.weight" and torch.equal(tensor, embed)
    assert wst.load_checkpoint_tensor(str(tmp_path), None)[0] == "model.embed_tokens.weight", (
        "the default is the embedding"
    )
    with pytest.raises(KeyError, match="lm_head.weight"):
        wst.load_checkpoint_tensor(str(tmp_path), "lm_head.weight")


def test_the_default_parameter_is_the_embedding_under_any_prefix(tmp_path):
    """Multimodal families keep the embedding under the language model; the default must find it
    there rather than assume the dense spelling, and refuse to guess between two candidates."""
    save_file(
        {
            "model.language_model.embed_tokens.weight": torch.zeros(2, 2),
            "model.visual.patch_embed.weight": torch.ones(2),
        },
        str(tmp_path / wst.SAFETENSORS_WEIGHTS_FILE),
    )
    name, tensor = wst.load_checkpoint_tensor(str(tmp_path), None)
    assert name == "model.language_model.embed_tokens.weight" and tensor.shape == (2, 2)
    save_file({"model.norm.weight": torch.ones(2)}, str(tmp_path / wst.SAFETENSORS_WEIGHTS_FILE))
    with pytest.raises(KeyError, match="0 \\*embed_tokens.weight"):
        wst.load_checkpoint_tensor(str(tmp_path), None)
    save_file(
        {"model.a.embed_tokens.weight": torch.zeros(2), "model.b.embed_tokens.weight": torch.zeros(2)},
        str(tmp_path / wst.SAFETENSORS_WEIGHTS_FILE),
    )
    with pytest.raises(KeyError, match="2 \\*embed_tokens.weight"):
        wst.load_checkpoint_tensor(str(tmp_path), None)


def test_a_hub_checkpoint_is_read_at_the_requested_revision(monkeypatch, tmp_path):
    """``--revision`` must reach every Hub download, and a served id that is neither a directory here
    nor a Hub repo must name ``--model-id`` rather than surface a bare validation error."""
    save_file({"model.embed_tokens.weight": torch.zeros(2, 2)}, str(tmp_path / wst.SAFETENSORS_WEIGHTS_FILE))
    downloads: list[tuple[str, str, str | None]] = []

    def fake_download(repo_id, filename, revision=None):
        downloads.append((repo_id, filename, revision))
        if filename == wst.SAFETENSORS_INDEX_FILE:
            raise wst.EntryNotFoundError("no index")
        return str(tmp_path / filename)

    monkeypatch.setattr(wst, "hf_hub_download", fake_download)
    name, _ = wst.load_checkpoint_tensor("org/model", None, revision="v2")
    assert name == "model.embed_tokens.weight"
    assert downloads == [
        ("org/model", wst.SAFETENSORS_INDEX_FILE, "v2"),
        ("org/model", wst.SAFETENSORS_WEIGHTS_FILE, "v2"),
    ]

    def not_a_repo(repo_id, filename, revision=None):
        raise wst.HFValidationError(f"{repo_id!r} is not a valid repo id")

    monkeypatch.setattr(wst, "hf_hub_download", not_a_repo)
    with pytest.raises(ValueError, match="--model-id"):
        wst.load_checkpoint_tensor("/mnt/models/on-the-server", None)


class _StreamedClient(VLLMWeightSyncClient):
    """The vLLM client over a fake server's HTTP routes: forming the group only records the sync device,
    and a :class:`Wire` stands in for the engine's update phases."""

    def init_communicator(self, device=0):
        self._sync_device = resolve_sync_device(device)


@pytest.mark.usefixtures("nccl_debug_env")
def test_each_round_is_the_trainers_streamed_push(tmp_path, monkeypatch):
    """A round stages the checkpoint tensor on the sync device as a copy and streams it inside one update
    it closes, as every trainer sync does. An engine handed the loaded tensor itself means the round
    skipped that copy, so the rate it reports is not the trainer's."""
    # The preflight's NCCL log directory is a fresh mkdtemp; this keeps it under tmp_path.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    embed = torch.arange(8, dtype=torch.bfloat16).reshape(4, 2)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    save_file({"model.embed_tokens.weight": embed}, str(checkpoint / wst.SAFETENSORS_WEIGHTS_FILE))
    loaded: list[torch.Tensor] = []
    load = wst.load_checkpoint_tensor

    def recording_load(*args, **kwargs):
        name, tensor = load(*args, **kwargs)
        loaded.append(tensor)
        return name, tensor

    wire = Wire()
    monkeypatch.setattr(wst, "load_checkpoint_tensor", recording_load)
    monkeypatch.setattr(
        wst, "resolve_weight_sync_client", lambda backend: lambda **kwargs: wire.attach(_StreamedClient(**kwargs))
    )
    server = FakeVLLMServer()
    try:
        result = wst.run_transport_check(
            server_url=server.url,
            backend="vllm",
            group_port=0,
            group_host=None,
            device="cpu",
            param=None,
            model_id=str(checkpoint),
            revision=None,
            rounds=2,
            connection_timeout=5.0,
        )
    finally:
        server.close()

    assert len(result.rates_gbps) == 2
    assert list(asdict(result)) == _JSON_KEYS, "the --json keys or their order moved"
    assert (wire.opened, wire.closed) == (2, 2), "each round must open one update and close it"
    [source] = loaded
    assert len(wire.chunks) == 2
    for [(name, sent)] in wire.chunks:
        assert name == "model.embed_tokens.weight" and torch.equal(sent, embed)
        assert sent.data_ptr() != source.data_ptr(), "the round pushed the loaded tensor, not a staged copy"


def _result(**overrides) -> wst.TransportCheckResult:
    fields = {
        "server_url": "http://server:8000",
        "backend": "vllm",
        "model_id": "Qwen/Qwen3-8B",
        "served_model": "policy",
        "param": "model.embed_tokens.weight",
        "payload_gb": 1.2,
        "init_seconds": 3.0,
        "close_seconds": 0.1,
        "rates_gbps": [40.0, 41.0],
        "served_unchanged": True,
        "verdict": "efa",
        "gpudirect": True,
        "nccl_log_dir": "/tmp/preflight",
        **asdict(wst.parse_nccl_log(EFA_LOG)),
    }
    return wst.TransportCheckResult(**{**fields, **overrides})


def test_the_cli_gates_on_the_expected_transport_and_an_unchanged_server(monkeypatch, capsys):
    """The exit code is the gate a launch script reads: a transport other than ``--expect`` or a
    push that altered the served model must be 1, and the report must name which."""
    results: list[dict] = []
    monkeypatch.setattr(preflight_cli, "run_transport_check", lambda **kwargs: results.pop(0))

    def run(*argv: str) -> int:
        monkeypatch.setattr("sys.argv", ["weight_sync_transport.py", "--server-url", "http://server:8000", *argv])
        return preflight_cli.main()

    results.append(_result())
    assert run("--expect", "efa") == 0
    out = capsys.readouterr().out
    assert "transport      efa" in out and "(served as policy)" in out

    results.append(_result())
    assert run("--expect", "socket") == 1
    assert "transport is efa, expected socket" in capsys.readouterr().err

    results.append(_result(served_unchanged=False))
    assert run() == 1, "a push that altered the served model must fail the preflight even without --expect"
    assert "changed the served model" in capsys.readouterr().err

    results.append(_result())
    assert run("--json") == 0
    printed = json.loads(capsys.readouterr().out)
    assert list(printed) == _JSON_KEYS, "the --json keys or their order moved"
    assert printed["verdict"] == "efa" and printed["transports"] == ["NET/Libfabric/0/GDRDMA"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
