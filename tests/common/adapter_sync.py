"""The adapter sync as a launch with no recorded base builds it, shared by the CPU and GPU tiers."""

from halo.trainers.grpo.rollout.adapter_sync import AdapterSync
from halo.weight_sync.adapter_routes import AdapterRoutes
from halo.weight_sync.adapter_versions import AdapterVersions
from halo.weight_sync.clients.sglang import SGLangWeightSyncClient


def build_adapter_sync(serving, versions: AdapterVersions) -> AdapterSync:
    routes = AdapterRoutes(SGLangWeightSyncClient.is_adapter_missing)
    return AdapterSync(serving, versions, routes, None)
