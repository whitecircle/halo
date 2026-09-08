#!/usr/bin/env python
"""The weight-sync transport preflight reads NCCL's own debug log and a checkpoint shard.

The transport verdict is the whole point of the tool: a parser that misses the ``via NET/...``
channel lines, or reads GPUDirect off the wrong field, turns a socket fallback into a passing
preflight. The excerpts below are verbatim NCCL 2.31 output from the three transports the trainer
meets: EFA with GPUDirect, the socket fallback, and same-host CUDA IPC.

    python tests/cpu/diagnostics/test_weight_sync_transport.py
"""

import json
import os
import sys

import pytest
import torch
from safetensors.torch import save_file

from src.diagnostics import weight_sync_transport as wst

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


def test_efa_log_reads_as_efa_with_gpudirect_and_the_plugin_identity():
    report = wst.parse_nccl_log(EFA_LOG)
    assert report.verdict == wst.VERDICT_EFA
    assert report.gpudirect is True
    assert report.transports == ["NET/Libfabric/0/GDRDMA"], "channel lines must be deduplicated, flush suffix dropped"
    assert report.plugin == "git-1f0a976"
    assert (report.net_plugin, report.net_plugin_api) == ("Libfabric", 12)
    assert (report.provider, report.fabric) == ("efa", "efa-direct")
    assert report.nccl_version == "2.31.2+cuda13.3"
    assert report.plugin_missing == []


def test_a_plugin_that_forced_the_simple_protocol_is_reported():
    """The forced protocol is invisible in a passing run and hangs a peer on a different rdma-core
    build; a report that only names the transport would call that pair healthy."""
    forced = EFA_LOG.replace(
        "host:1:257 [0] NCCL INFO NET/OFI Selected provider",
        "host:1:257 [0] NCCL INFO NET/OFI Setting FI_OPT_EFA_WRITE_IN_ORDER_ALIGNED_128_BYTES not supported.\n"
        "host:1:257 [0] NCCL INFO NET/OFI Selected provider",
    )
    assert wst.parse_nccl_log(forced).in_order_writes is False
    assert wst.parse_nccl_log(EFA_LOG).in_order_writes is True
    assert wst.parse_nccl_log(SOCKET_LOG).in_order_writes is None, "no plugin, no probe"


def test_socket_fallback_reads_as_socket_and_names_the_missing_plugin():
    report = wst.parse_nccl_log(SOCKET_LOG)
    assert report.verdict == wst.VERDICT_SOCKET
    assert report.gpudirect is False
    assert report.plugin is None
    assert report.plugin_missing == ["ofi libnccl-net-ofi.so"]


def test_same_host_cuda_ipc_reads_as_p2p():
    report = wst.parse_nccl_log(P2P_LOG)
    assert report.verdict == wst.VERDICT_P2P
    assert report.transports == ["P2P/CUMEM"]


def test_a_log_without_channel_lines_is_unknown_not_a_guess():
    assert wst.parse_nccl_log("host:1:1 [0] NCCL INFO NCCL version 2.31.2+cuda13.3\n").verdict == wst.VERDICT_UNKNOWN


def test_arming_sets_the_debug_variables_before_nccl_reads_them(tmp_path, monkeypatch):
    for key in ("NCCL_DEBUG", "NCCL_DEBUG_SUBSYS", "NCCL_DEBUG_FILE"):
        monkeypatch.delenv(key, raising=False)
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
    embed = torch.arange(6, dtype=torch.bfloat16).reshape(3, 2)
    save_file({"model.embed_tokens.weight": embed}, str(tmp_path / "model-00001-of-00002.safetensors"))
    save_file({"model.norm.weight": torch.ones(2)}, str(tmp_path / "model-00002-of-00002.safetensors"))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "weight_map": {
                    "model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                    "model.norm.weight": "model-00002-of-00002.safetensors",
                }
            }
        )
    )
    assert torch.equal(wst.load_checkpoint_tensor(str(tmp_path), "model.embed_tokens.weight"), embed)
    with pytest.raises(KeyError, match="model.norm.weight"):
        wst.load_checkpoint_tensor(str(tmp_path), "lm_head.weight")


def test_single_file_checkpoint_needs_no_index(tmp_path):
    save_file({"model.embed_tokens.weight": torch.zeros(2, 2)}, str(tmp_path / "model.safetensors"))
    assert wst.load_checkpoint_tensor(str(tmp_path), "model.embed_tokens.weight").shape == (2, 2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
