"""Repeated connect, sync and close cycles on one group port against one live server, the re-init suite's body on
either engine. A leaked NCCL communicator strands device memory while ``/health`` stays 200; whole-checkpoint cycles
catch a staging buffer left in a pool only the closed client allocated from."""

import subprocess
from collections.abc import Iterable

import torch
from transformers import AutoModelForCausalLM

from halo.env import env_str
from halo.weight_sync.clients.base import WeightSyncClientBase
from tests.common.models import QWEN3_0_6B
from tests.common.on_policy_e2e import probe_top_logprobs, record_served_baseline
from tests.common.rank_log import log
from tests.common.weight_sync import push, weight_transfer_port

# The checkpoint the vLLM tier's dense endpoint serves.
VLLM_REINIT_MODEL = env_str("HALO_TEST_VLLM_REINIT_MODEL", QWEN3_0_6B)
# The final RMSNorm scales every logit, so a push shifts the top-k without reordering it into a distribution the next
# cycle cannot move.
PERTURBED_PARAM = "model.norm.weight"
PERTURBATION = 1.02
# An unreleased communicator strands ~633 MiB per cycle on each end of a vLLM pair (measured on 0.26.0).
MAX_MEMORY_GROWTH_MIB = 1024
# Cycles pushing every parameter (over 1 GiB by default), so each client fills full-size staging buffers. A stranded
# one costs about a chunk per cycle and overshoots the budget.
WHOLE_CHECKPOINT_CYCLES = 4
CONNECTION_TIMEOUT_S = 120.0


def _server_gpu_used_mib(server_gpu: str | None) -> int:
    """Used device memory on the server's GPU(s) via ``nvidia-smi``, since the server sits outside
    ``CUDA_VISIBLE_DEVICES``; ``None`` means every GPU not ours."""
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
    """Used device memory on this rank's GPU, driver-level, since NCCL's allocations sit outside torch's."""
    free, total = torch.cuda.mem_get_info(device)
    return (total - free) // 2**20


def _sync(
    client_class: type[WeightSyncClientBase],
    server_url: str,
    named_params: Iterable[tuple[str, torch.Tensor]],
    device: torch.device,
    group_port: int,
) -> None:
    """One whole trainer lifetime: connect, one quiesced update, disconnect."""
    client = client_class(base_url=server_url, group_port=group_port, connection_timeout=CONNECTION_TIMEOUT_S)
    client.init_communicator(device=device)
    try:
        push(client, named_params)
    finally:
        client.close_communicator()


def serve_the_checkpoint(
    client_class: type[WeightSyncClientBase], server_url: str, model, device: torch.device, group_port: int
) -> None:
    """Push the loaded checkpoint whole: the server outlives a tier's rows and may still serve an earlier row's trained
    weights."""
    _sync(client_class, server_url, ((name, t.detach()) for name, t in model.named_parameters()), device, group_port)


def _probe_or_none(server_url: str, model_name: str) -> dict[str, float] | None:
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
    """``cycles`` connect/sync/disconnect cycles, then a restoring cycle; the ``gpu_test_main`` result. The port is
    drawn once and rebound by every cycle."""
    log(f"{client_class.BACKEND_NAME} weight-transfer re-init: {cycles} cycles against {server_url}")
    checks: dict[str, bool] = {}
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16).to(ctx.device)
    param = dict(model.named_parameters())[PERTURBED_PARAM]
    original = param.detach().clone()

    # Part of the invariant: close_communicator must release the port on both ends or the next cycle cannot form its
    # group.
    group_port = weight_transfer_port(group_port_knob)
    serve_the_checkpoint(client_class, server_url, model, ctx.device, group_port)
    baseline = record_served_baseline(server_url, model_name, checks)
    probes = [baseline]
    errors: list[str] = []
    # After the first cycle: the first sync loads NCCL, sizes staging buffers and warms allocators, so only later
    # growth is a leak.
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

    # Trainer only: vLLM 0.26.0's packed consumer receives on fresh streams per update, so its reserved memory grows
    # until PyTorch's stream pool wraps, whatever the trainer does.
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

    # Exact equality here proves the engine reloaded the pushed tensor rather than drifting under repeated reloads.
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
