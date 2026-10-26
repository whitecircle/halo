"""Build-time checks of the SGLang server image: the routes, request fields, rendezvous convention and
expert loader layouts the weight-sync client relies on.

No test suite can run these: they import SGLang, which the training image does not have, so the image
build is the only gate. Upstream renames these between releases, and each check fails the build on a
change that would otherwise surface mid-run as a 404, a dropped field, a hung group or a sync that
writes no expert:

    python3 verify_image.py
"""

import dataclasses
import inspect
import pathlib

import msgspec
import sglang
from sglang.srt.entrypoints.http_server import app
from sglang.srt.managers.io_struct import InitWeightsUpdateGroupReqInput, UpdateWeightsFromDistributedReqInput
from sglang.srt.model_executor.model_runner_components.weight_updater import WeightUpdater
from sglang.srt.utils.common import init_custom_process_group


def check_weight_sync_routes() -> None:
    """The control endpoints the client drives exist; mounted sub-routers carry no ``.path``, hence the
    ``getattr``."""
    paths = {p for p in (getattr(r, "path", None) for r in app.routes) if p}
    required = {
        "/init_weights_update_group",
        "/update_weights_from_distributed",
        "/begin_weight_update",
        "/end_weight_update",
        "/server_info",
        "/pause_generation",
        "/continue_generation",
        "/flush_cache",
        "/health",
    }
    missing = required - paths
    assert not missing, f"sglang lost weight-sync routes: {sorted(missing)}"
    print(f"weight-sync routes: OK ({len(required)} present)")


def check_request_fields() -> None:
    """The request fields ``SGLangWeightSyncClient`` sends are the wire contract, and a renamed field is
    dropped server-side. Read whichever representation the release uses."""

    def names(cls) -> set[str]:
        fields = dataclasses.fields(cls) if dataclasses.is_dataclass(cls) else msgspec.structs.fields(cls)
        return {f.name for f in fields}

    missing = {"master_address", "master_port", "rank_offset", "world_size", "group_name", "backend"} - names(
        InitWeightsUpdateGroupReqInput
    )
    assert not missing, f"InitWeightsUpdateGroupReqInput lost: {missing}"
    missing = {"names", "dtypes", "shapes", "group_name", "flush_cache", "abort_all_requests"} - names(
        UpdateWeightsFromDistributedReqInput
    )
    assert not missing, f"UpdateWeightsFromDistributedReqInput lost: {missing}"
    print("weight-sync request schemas: OK")


def check_rendezvous_convention() -> None:
    """The trainer vendors SGLang's rendezvous convention (src/weight_sync/transport/c10d_group.py): a
    group_name-prefixed store joined via ``_new_process_group_helper``, trainer at rank 0. A change
    upstream would hang the group with no error."""
    source = inspect.getsource(init_custom_process_group)
    assert "PrefixStore(group_name, store)" in source, "sglang no longer prefixes the store with group_name"
    assert "_new_process_group_helper" in source, "sglang no longer builds the group via _new_process_group_helper"
    assert "src=0" in inspect.getsource(WeightUpdater.receive_weights_from_distributed), (
        "sglang no longer broadcasts from rank 0"
    )
    assert "rank_offset + self.tp_rank" in inspect.getsource(WeightUpdater.init_weights_update_group), (
        "sglang changed the rank_offset convention"
    )
    print("rendezvous + broadcast convention: OK (trainer is rank 0, store prefixed by group_name)")


def check_expert_loader_layouts() -> None:
    """The trainer streams each family's experts in its hub checkpoint layout, and these loaders read
    exactly that: gpt_oss the fused interleaved pair, qwen3_moe the per-expert tensors. A loader that
    moved to the other spelling would drop every expert of a sync without error."""
    models = pathlib.Path(sglang.__file__).parent / "srt" / "models"
    gpt_oss = (models / "gpt_oss.py").read_text()
    qwen3_moe = (models / "qwen3_moe.py").read_text()
    assert "make_expert_params_mapping_fused" in gpt_oss, "gpt_oss does not load the fused hub pair"
    assert "make_expert_params_mapping_fused" not in qwen3_moe and "make_expert_params_mapping(" in qwen3_moe, (
        "qwen3_moe does not map the per-expert hub tensors"
    )
    print("expert loader layouts: OK (gpt_oss fused, qwen3_moe per-expert)")


CHECKS = (
    check_weight_sync_routes,
    check_request_fields,
    check_rendezvous_convention,
    check_expert_loader_layouts,
)

if __name__ == "__main__":
    for check in CHECKS:
        check()
