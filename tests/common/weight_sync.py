"""Weight-sync client stand-ins shared by the CPU suites."""

from unittest.mock import patch

from src.distributed.nccl.clients.sglang import SGLangWeightSyncClient


def offline_sglang_client(base_url: str = "http://localhost:30000") -> SGLangWeightSyncClient:
    """A SGLang client built by its real ``__init__`` minus the live-server probe (``check_server``)."""
    with patch.object(SGLangWeightSyncClient, "check_server"):
        return SGLangWeightSyncClient(base_url=base_url)
