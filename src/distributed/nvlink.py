"""NVLink / MNNVL topology probes: the ``NVLINK_DOMAIN_SIZE`` reader, the per-GPU fabric-clique read,
the cross-check of a declared domain against the fabric the GPUs report, and the Multi-Node NVLink
prerequisites. Nothing here builds a process group; ``ParallelismConfig`` consumes the results.
"""

import functools
import logging
import os
import socket
import subprocess

import torch
import torch.distributed as dist

from src.env import env_int
from src.log import warn_once
from src.runtime.consensus import reject_across_ranks
from src.runtime.ranks import has_process_group, is_global_main_process, is_multi_rank_run

logger = logging.getLogger(__name__)

# Definite "this GPU is not on an MNNVL fabric", distinct from ``None`` ("could not tell"): only the
# definite answer lets the domain cross-check reject a too-wide domain. Never a real clique id.
NO_FABRIC = -1
# Kernel-side capability for cross-OS-node NVLink P2P, populated by the nvidia-imex service.
_MNNVL_IMEX_DIR = "/dev/nvidia-caps-imex-channels"

_warned_domain_sizes: set[int] = set()


def requested_nvlink_domain_size() -> int | None:
    """The GPU count ``NVLINK_DOMAIN_SIZE`` declares (up to a whole NVL72 rack), or None when unset.

    A malformed or non-positive value warns and reads as unset (0 would divide by zero in the rank
    math). No NVML auto-detection: counting the domain across ranks needs a collective that would hang
    where NVML succeeds on only some ranks.
    """
    domain = env_int("NVLINK_DOMAIN_SIZE", None)
    if domain is not None and domain < 1:
        warn_once(
            logger,
            _warned_domain_sizes,
            domain,
            "NVLINK_DOMAIN_SIZE=%d is not a positive GPU count — ignoring it and falling back to the "
            "per-node NVLink domain. Set it to the real domain size (e.g. 72 on an NVL72 rack).",
            domain,
        )
        return None
    return domain


def _parse_fabric_clique(nvidia_smi_query: str) -> int | None:
    """``CliqueId`` from an ``nvidia-smi -q`` dump: an id, :data:`NO_FABRIC` (no ``Fabric`` block, or
    clique 0, as every plain HGX/NVL8 part reports), or ``None`` (registration not ``COMPLETED``, or an
    unparseable id). Parsed within the ``Fabric`` block alone, which the top-level ``GPU Fabric GUID``
    line must not open: ``State`` recurs elsewhere in the dump.
    """
    indent, state, clique = None, None, None
    for line in nvidia_smi_query.splitlines():
        stripped = line.strip()
        if indent is None:
            if stripped == "Fabric":
                indent = len(line) - len(line.lstrip())
            continue
        if stripped and (len(line) - len(line.lstrip())) <= indent:
            break
        key, _, value = stripped.partition(":")
        key, value = key.strip(), value.strip()
        if key == "State":
            state = value
        elif key == "CliqueId":
            clique = value
    if indent is None:
        return NO_FABRIC  # no Fabric block at all: definitely not an MNNVL part
    if state is None or state.lower() != "completed":
        return None  # fabric present but registration still in progress — cannot tell
    try:
        return int(clique) or NO_FABRIC  # clique 0 is "not on a fabric", not a real island
    except (TypeError, ValueError):
        return None


@functools.cache
def _fabric_clique_for_device(device_index: int) -> int | None:
    """One device's fabric clique, cached: membership is fixed once the driver is up, and the ~60 ms
    read runs on every ``ParallelismConfig`` construction (tens of thousands in a config sweep)."""
    try:
        # By PCI bus id, not index: nvidia-smi enumerates every host GPU while the local rank indexes
        # the CUDA_VISIBLE_DEVICES-masked view, so under masking the read would hit another GPU.
        props = torch.cuda.get_device_properties(device_index)
        bus_id = f"{props.pci_domain_id:08x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
        query = subprocess.run(
            ["nvidia-smi", "-q", "-i", bus_id],
            capture_output=True,
            text=True,
            timeout=30,  # a driver fault must not stall startup
            check=True,
        )
    except Exception:  # advisory only, never fatal
        return None
    return _parse_fabric_clique(query.stdout)


def nvlink_fabric_clique_id() -> int | None:
    """This GPU's NVLink fabric-clique id, :data:`NO_FABRIC`, or ``None`` when it cannot be read; two
    GPUs share NVLink iff they report the same clique.

    Read out of process: NVML's Python fabric getters take a versioned struct whose layout must match
    the driver's, and a mismatch segfaults uncatchably on every rank.
    """
    try:
        return _fabric_clique_for_device(torch.cuda.current_device())
    except Exception:  # advisory only, never fatal
        return None


def node_placement_contradiction(host_ids: list[str], gpus_per_node: int) -> str | None:
    """Why the ranks are not laid out ``gpus_per_node`` per node, or ``None`` when they are.

    ``host_ids[rank]`` is that rank's machine identity. Every node-local group is a contiguous rank
    block, one machine only while rank ``r`` sits on node ``r // gpus_per_node``; cyclic rank numbering
    (``srun --distribution=cyclic``) breaks that, and DeepEP then takes its intranode CUDA-IPC path to
    peers on another host.
    """
    if gpus_per_node <= 0 or not host_ids:
        return None
    straddling = {
        block: sorted(set(hosts))
        for block in range(len(host_ids) // gpus_per_node)
        if len(set(hosts := host_ids[block * gpus_per_node : (block + 1) * gpus_per_node])) > 1
    }
    if not straddling:
        return None
    first = min(straddling)
    return (
        f"Ranks are not placed {gpus_per_node} per node: rank block(s) {sorted(straddling)} each span "
        f"more than one machine (block {first} holds {straddling[first]}). Every node-local EP/CP/TP/ETP "
        f"group is a contiguous rank block, so those groups would straddle machines over the network "
        f"while requires_rdma reports False. Launch one torchrun per node with --node_rank, or place "
        f"ranks in blocks (srun --distribution=block), so consecutive ranks share a machine."
    )


def fabric_domain_contradiction(
    cliques: list[int], node_widths: list[int], nvlink_domain_size: int, world_size: int
) -> str | None:
    """Why the declared domain contradicts the gathered per-rank cliques and node widths, or ``None``.
    The width verdict comes first: fabric-less ranks share one sentinel, so a domain across two of
    them passes the straddle check."""
    # Domain-major contiguous blocks — the same layout the node-local EP/CP groups are cut from.
    fabricless: dict[int, int] = {}
    for domain, start in enumerate(range(0, world_size, nvlink_domain_size)):
        block = slice(start, start + nvlink_domain_size)
        if all(clique == NO_FABRIC for clique in cliques[block]):
            width = min(node_widths[block])
            if 0 < width < nvlink_domain_size:
                fabricless[domain] = width
    if fabricless:
        return (
            f"NVLINK_DOMAIN_SIZE={nvlink_domain_size} exceeds the GPUs per node of domain(s) "
            f"{sorted(fabricless)}, where every rank reports NO NVLink fabric — there is no multi-node "
            f"NVLink for them to span, so a domain wider than one node cannot exist there. Their "
            f"'node-local' EP/TP/CP groups would straddle nodes over the network while requires_rdma "
            f"reports False. Unset NVLINK_DOMAIN_SIZE (per-node NVLink), or set it to at most "
            f"{min(fabricless.values())}."
        )
    straddling = [
        domain
        for domain in range(world_size // nvlink_domain_size)
        if len(set(cliques[domain * nvlink_domain_size : (domain + 1) * nvlink_domain_size])) > 1
    ]
    if not straddling:
        return None
    ranks_per_clique = {clique: cliques.count(clique) for clique in sorted(set(cliques))}
    return (
        f"NVLINK_DOMAIN_SIZE={nvlink_domain_size} does not match the NVLink fabric: domain(s) "
        f"{straddling} span more than one fabric clique, so their 'node-local' EP/TP/CP groups "
        f"would be part NVLink and part RDMA while requires_rdma still reports False. Set "
        f"NVLINK_DOMAIN_SIZE to a value that divides the real clique size "
        f"(this job's ranks per clique: {ranks_per_clique})."
    )


def validate_nvlink_domain_against_fabric(nvlink_domain_size: int, world_size: int, gpus_per_node: int = 0) -> None:
    """COLLECTIVE: check the declared NVLink domain against the fabric the GPUs report.

    The domain is otherwise an unchecked assertion: one straddling two racks makes every "node-local"
    group half RDMA while ``requires_rdma`` reports False, and one left at ``gpus_per_node`` on NVL72
    caps expert sharding. Every rank gathers (clique, node width, machine) and decides from the same
    list, so it never raises on a subset of ranks; a ``None`` clique anywhere no-ops the fabric legs.
    ``gpus_per_node=0`` skips the fabric-less width leg.
    """
    if not has_process_group():
        return
    # all_gather_object fills exactly get_world_size() slots; an oversized list would keep pre-fill Nones.
    group_world = dist.get_world_size()
    if group_world <= 1 or group_world != world_size:
        return
    slots: list[tuple[int | None, int, str] | None] = [None] * group_world
    dist.all_gather_object(slots, (nvlink_fabric_clique_id(), gpus_per_node, socket.gethostname()))
    gathered = [entry for entry in slots if entry is not None]
    if len(gathered) != group_world:
        return

    reason = node_placement_contradiction([entry[2] for entry in gathered], gpus_per_node)
    if reason:
        raise ValueError(reason)

    cliques = [entry[0] for entry in gathered]
    if any(clique is None for clique in cliques):
        return  # unreadable somewhere — a partial view is not evidence

    node_widths = [entry[1] for entry in gathered]
    reason = fabric_domain_contradiction(cliques, node_widths, nvlink_domain_size, world_size)
    if reason:
        raise ValueError(reason)
    if all(clique == NO_FABRIC for clique in cliques):
        return  # no fabric anywhere, every domain within a node — nothing left to cross-check

    # NO_FABRIC is one sentinel shared by every fabric-less node, not an island of that many ranks,
    # so such nodes cap the advice at their own width instead of widening it past what they can span.
    local_widths = [cliques.count(clique) for clique in set(cliques) - {NO_FABRIC}]
    fabricless_width = min(
        (width for clique, width in zip(cliques, node_widths, strict=True) if clique == NO_FABRIC), default=0
    )
    if fabricless_width:
        local_widths.append(fabricless_width)
    narrowest_local = min(local_widths)
    if narrowest_local > nvlink_domain_size and is_global_main_process():
        logger.warning(
            "NVLINK_DOMAIN_SIZE=%d is smaller than the NVLink width this job actually has (%d GPUs "
            "share one NVLink domain on its narrowest node), so node-local EP/TP/CP are capped below "
            "the fabric already available. Raise it to the largest divisor of %d your model's expert "
            "count can use.",
            nvlink_domain_size,
            narrowest_local,
            narrowest_local,
        )


def imex_channels_present() -> bool:
    """Whether this node exposes IMEX channels, the kernel side of cross-OS-node NVLink P2P. Never
    raises: it runs inside a rank-uniform gate that must reach its collective."""
    try:
        return bool(os.listdir(_MNNVL_IMEX_DIR))
    except OSError:
        return False


def mnnvl_prerequisite_reason(imex_ok: bool, fabric_clique: int | None) -> str | None:
    """Why this rank cannot take part in Multi-Node NVLink, or ``None`` when it can. Called only where
    the topology requires MNNVL, where a ``None`` clique is an incomplete Fabric Manager registration,
    not the benign "could not tell"."""
    if not imex_ok:
        return (
            f"no IMEX channels at {_MNNVL_IMEX_DIR}. Cross-OS-node NVLink P2P needs NVIDIA Fabric "
            f"Manager plus the IMEX service (nvidia-imex) with matching channels on every node, and "
            f"NCCL >= 2.25.2"
        )
    if fabric_clique is None:
        return (
            "the GPU's NVLink fabric registration is not COMPLETED (nvidia-smi 'Fabric State'), or "
            "nvidia-smi could not be read at all — Fabric Manager has not finished bringing this "
            "node into the fabric"
        )
    if fabric_clique == NO_FABRIC:
        return "the GPU reports no NVLink fabric clique at all, so it is not on an MNNVL fabric"
    return None


def verify_mnnvl_prerequisites(nvlink_domain_size: int, gpus_per_node: int) -> None:
    """COLLECTIVE when MNNVL is required: verify the Multi-Node NVLink prerequisites on every rank.

    Required means ``nvlink_domain_size > gpus_per_node`` (NVL72). The per-rank verdicts are joined,
    so one node missing IMEX channels or still registering fails by name instead of as a peer-memory
    fault mid-dispatch. Below that threshold, and outside a live multi-rank job, this only advises.
    """
    if nvlink_domain_size <= gpus_per_node:
        if not is_global_main_process():
            return
        # Only a real clique id is evidence of a fabric; neither NO_FABRIC nor "could not tell" is.
        if requested_nvlink_domain_size() is None and nvlink_fabric_clique_id() not in (None, NO_FABRIC):
            logger.warning(
                "An NVLink fabric (NVL72/MNNVL?) appears present but NVLINK_DOMAIN_SIZE is "
                "unset, so node-local TP/CP/EP are capped at gpus_per_node=%d. Set "
                "NVLINK_DOMAIN_SIZE (e.g. 72) to use NVLink-wide parallelism across the rack.",
                gpus_per_node,
            )
        return

    reason = mnnvl_prerequisite_reason(imex_channels_present(), nvlink_fabric_clique_id())
    what = (
        f"Multi-Node NVLink prerequisite check (NVLINK_DOMAIN_SIZE={nvlink_domain_size} > "
        f"gpus_per_node={gpus_per_node}, so node-local EP/TP/CP groups span OS nodes over NVLink)"
    )
    if not is_multi_rank_run():
        if reason and is_global_main_process():
            logger.warning("%s: %s.", what, reason)
        return
    reject_across_ranks(reason, what, exc_type=ValueError)
    if is_global_main_process():
        logger.info(
            "Multi-Node NVLink enabled: nvlink_domain_size=%d across gpus_per_node=%d (IMEX "
            "channels and a registered fabric on every rank). Ensure NCCL_MNNVL_ENABLE is not disabled.",
            nvlink_domain_size,
            gpus_per_node,
        )
