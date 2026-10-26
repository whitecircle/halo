"""Repeated trainer connect → sync → close cycles against one live rollout server, rebinding one group
port each cycle: the body of both engines' weight-transfer re-init suites.

A cycle that leaves either end's NCCL communicator alive strands its device memory while ``/health``
keeps answering 200, until a later init fails outright; the device-memory checks catch that, and the
served-policy checks prove each re-init still lands its sync. Whole-checkpoint cycles then catch a
trainer-side staging buffer left in a pool only the closed client allocated from.
"""

import subprocess
from collections.abc import Iterable

import torch
from transformers import AutoModelForCausalLM

from src.env import env_str
from src.weight_sync.clients.base import WeightSyncClientBase
from tests.common.models import QWEN3_0_6B
from tests.common.on_policy_e2e import probe_top_logprobs, record_served_baseline
from tests.common.utils import log
from tests.common.weight_sync import push, weight_transfer_port

# The checkpoint the vLLM tier's dense endpoint serves, which the re-init and pool-reconnect suites push.
VLLM_REINIT_MODEL = env_str("HALO_TEST_VLLM_REINIT_MODEL", QWEN3_0_6B)
# The final RMSNorm scales every logit, so a cycle's push shifts the whole top-k without reordering
# it into a degenerate distribution the next cycle could no longer move.
PERTURBED_PARAM = "model.norm.weight"
PERTURBATION = 1.02
# An unreleased communicator strands ~633 MiB per cycle on each end of a vLLM 0.26.0 pair, so a dozen
# leaking cycles overshoot this budget several times over.
MAX_MEMORY_GROWTH_MIB = 1024
# Connect/sync/close cycles that push every parameter (over 1 GiB for the default checkpoint), so each
# client fills full-size staging buffers: one a closed client strands costs about a chunk per cycle,
# and the growth after the first such cycle overshoots the budget.
WHOLE_CHECKPOINT_CYCLES = 4
CONNECTION_TIMEOUT_S = 120.0


def _server_gpu_used_mib(server_gpu: str | None) -> int:
    """Used device memory on the server's GPU(s), read out of process.

    The server holds a GPU outside ``CUDA_VISIBLE_DEVICES`` (it must — a rank cannot broadcast to
    itself), so torch cannot see it here; ``nvidia-smi`` enumerates every GPU regardless.
    ``server_gpu`` is the server's GPU as ``nvidia-smi`` indexes it; ``None`` means every GPU this
    process does not own.
    """
    query = ["nvidia-smi", "--query-gpu=index,uuid,memory.used", "--format=csv,noheader,nounits"]
    if server_gpu:
        query += ["-i", server_gpu]
    rows = subprocess.run(query, capture_output=True, text=True, timeout=30, check=True).stdout
    owned = {str(torch.cuda.get_device_properties(i).uuid) for i in range(torch.cuda.device_count())}
    used = 0
    for row in rows.splitlines():
        _, uuid, memory = (field.strip() for field in row.split(","))
        if server_gpu or uuid.removeprefix("GPU-") not in owned:
            used += int(memory)
    return used


def _trainer_gpu_used_mib(device: torch.device) -> int:
    """Used device memory on this rank's GPU, driver-level — NCCL's allocations sit outside torch's."""
    free, total = torch.cuda.mem_get_info(device)
    return (total - free) // 2**20


def _sync(
    client_class: type[WeightSyncClientBase],
    server_url: str,
    named_params: Iterable[tuple[str, torch.Tensor]],
    device: torch.device,
    group_port: int,
) -> None:
    """One whole trainer lifetime against the server: connect, one quiesced update, disconnect."""
    client = client_class(base_url=server_url, group_port=group_port, connection_timeout=CONNECTION_TIMEOUT_S)
    client.init_communicator(device=device)
    try:
        push(client, named_params)
    finally:
        client.close_communicator()


def _probe_or_none(server_url: str, model_name: str) -> dict[str, float] | None:
    """The served policy's fingerprint, or ``None`` if the server can no longer produce one."""
    try:
        return probe_top_logprobs(server_url, model_name)
    except Exception as e:  # noqa: BLE001 — the verdict is "unusable", whatever the engine failed with
        log(f"served-policy probe failed: {type(e).__name__}: {e}")
        return None


def run_reinit_cycles(
    ctx,
    client_class: type[WeightSyncClientBase],
    *,
    server_url: str,
    model_name: str,
    server_gpu: str | None,
    cycles: int,
    group_port_knob: str,
) -> dict:
    """``cycles`` connect/sync/disconnect cycles of ``client_class`` against the server at ``server_url``
    serving ``model_name``, then a restoring cycle: the ``gpu_test_main`` result of the suite. The port is
    drawn once (``group_port_knob`` pins it) and rebound by every cycle."""
    log(f"{client_class.BACKEND_NAME} weight-transfer re-init: {cycles} cycles against {server_url}")
    checks: dict[str, bool] = {}
    baseline = record_served_baseline(server_url, model_name, checks)

    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(ctx.device)
    param = dict(model.named_parameters())[PERTURBED_PARAM]
    original = param.detach().clone()

    # Itself part of the invariant: close_communicator must release the port on both ends or the next
    # cycle cannot form its group.
    group_port = weight_transfer_port(group_port_knob)
    probes = [baseline]
    errors: list[str] = []
    # Both baselines are taken after the first cycle: the first sync loads the NCCL library, sizes
    # the staging buffers and warms both engines' allocators, and only what a SECOND and later cycle
    # adds on top of that is a leak.
    server_before = trainer_before = 0
    for cycle in range(1, cycles + 1):
        with torch.no_grad():
            param.mul_(PERTURBATION)
        try:
            _sync(client_class, server_url, [(PERTURBED_PARAM, param.detach())], ctx.device, group_port)
        except Exception as e:  # noqa: BLE001 — a refused re-init is the defect under test, not an error
            errors.append(f"cycle {cycle}: {type(e).__name__}: {e}")
            log(errors[-1])
            break
        if cycle == 1:
            server_before, trainer_before = _server_gpu_used_mib(server_gpu), _trainer_gpu_used_mib(ctx.device)
        probes.append(probe_top_logprobs(server_url, model_name))

    server_growth = _server_gpu_used_mib(server_gpu) - server_before
    trainer_growth = _trainer_gpu_used_mib(ctx.device) - trainer_before

    # Only the trainer is measured across these: vLLM 0.26.0's packed consumer receives on fresh
    # streams per update, so its reserved memory grows with whole-model pushes until PyTorch's stream
    # pool wraps, whatever the trainer does. They leave the served model as the light cycles did, and
    # the restore below proves they landed exactly.
    whole_completed = whole_before = 0
    while not errors and whole_completed < WHOLE_CHECKPOINT_CYCLES:
        named_params = ((name, tensor.detach()) for name, tensor in model.named_parameters())
        try:
            _sync(client_class, server_url, named_params, ctx.device, group_port)
        except Exception as e:  # noqa: BLE001 — same verdict as a failed cycle
            errors.append(f"whole-checkpoint cycle {whole_completed + 1}: {type(e).__name__}: {e}")
            log(errors[-1])
            break
        whole_completed += 1
        if whole_completed == 1:
            whole_before = _trainer_gpu_used_mib(ctx.device)
    whole_growth = _trainer_gpu_used_mib(ctx.device) - whole_before

    # The restore is a cycle of its own, and its exact-equality verdict is what proves the engine
    # reloaded the pushed tensor rather than drifting under repeated reloads.
    restored: dict[str, float] | None = None
    if not errors:
        try:
            _sync(client_class, server_url, [(PERTURBED_PARAM, original)], ctx.device, group_port)
        except Exception as e:  # noqa: BLE001 — same verdict as a failed cycle
            errors.append(f"restore cycle: {type(e).__name__}: {e}")
            log(errors[-1])
        else:
            restored = probe_top_logprobs(server_url, model_name)

    completed = len(probes) - 1
    checks["every_cycle_reconnected_and_synced"] = not errors
    checks["every_cycle_moved_the_served_policy"] = completed == cycles and all(
        probes[i] != probes[i - 1] for i in range(1, len(probes))
    )
    checks["served_policy_restored_by_a_final_sync"] = restored == baseline
    checks["server_frees_its_communicator_each_cycle"] = completed == cycles and server_growth <= MAX_MEMORY_GROWTH_MIB
    checks["trainer_frees_its_communicator_each_cycle"] = (
        completed == cycles and trainer_growth <= MAX_MEMORY_GROWTH_MIB
    )
    checks["trainer_frees_its_staging_buffers_each_cycle"] = (
        whole_completed == WHOLE_CHECKPOINT_CYCLES and whole_growth <= MAX_MEMORY_GROWTH_MIB
    )
    checks["server_still_generates"] = _probe_or_none(server_url, model_name) is not None

    log(
        f"{completed}/{cycles} cycles; server +{server_growth} MiB, trainer +{trainer_growth} MiB; "
        f"{whole_completed}/{WHOLE_CHECKPOINT_CYCLES} whole-checkpoint cycles, trainer +{whole_growth} MiB"
    )
    return {
        "checks": ctx.broadcast_checks(checks),
        "metrics": {
            "cycles_completed": completed,
            "server_growth_mib": server_growth,
            "trainer_growth_mib": trainer_growth,
            "whole_checkpoint_trainer_growth_mib": whole_growth,
            "first_error": errors[0] if errors else "",
        },
    }
