"""NVLink / MNNVL topology probes: the per-GPU fabric-clique read, the cross-check of a declared
``NVLINK_DOMAIN_SIZE`` against the fabric the GPUs report, and the Multi-Node NVLink prerequisites.

Nothing here builds a process group; the results are consumed by
:class:`~src.distributed.parallelism_config.ParallelismConfig`.
"""

import functools
import logging
import os
import socket
import subprocess

import torch
import torch.distributed as dist

from src.distributed.runtime import is_global_main_process, is_multi_rank_run, reject_across_ranks
from src.env import env_int

logger = logging.getLogger(__name__)

# Bounds the nvidia-smi query so a driver fault cannot stall startup.
_NVIDIA_SMI_TIMEOUT_S = 30

# Definite "this GPU is not on an MNNVL fabric", distinct from ``None`` ("could not tell"): only the
# definite answer lets the domain cross-check reject a too-wide domain. Never a real clique id.
NO_FABRIC = -1

# Kernel-side capability for cross-OS-node NVLink P2P, populated by the nvidia-imex service.
_MNNVL_IMEX_DIR = "/dev/nvidia-caps-imex-channels"


def get_nvlink_domain_size(default: int) -> int:
    """GPUs reachable over NVLink: ``gpus_per_node`` normally, up to the whole rack on NVL72.

    ``NVLINK_DOMAIN_SIZE`` env → ``default``; the caller supplies the fallback because only it knows
    its per-node count (a hand-built ``ParallelismConfig`` sets ``gpus_per_node``, which
    ``get_local_world_size()`` would override). No NVML auto-detection: counting the domain across
    ranks needs a collective that would hang where NVML succeeds on only some ranks.
    """
    domain = env_int("NVLINK_DOMAIN_SIZE", None)  # malformed → warn + fall back, not a crash
    if domain is not None:
        if domain > 0:
            return domain
        # 0 is ParallelismConfig's "unset" sentinel: it passes the validators and then raises a bare
        # ZeroDivisionError in the rank math; a negative one corrupts every group.
        logger.warning(
            "NVLINK_DOMAIN_SIZE=%d is not a positive GPU count — ignoring it and falling back to the "
            "per-node NVLink domain. Set it to the real domain size (e.g. 72 on an NVL72 rack).",
            domain,
        )
    return default


def _parse_fabric_clique(nvidia_smi_query: str) -> int | None:
    """``CliqueId`` from an ``nvidia-smi -q`` dump: an id, :data:`NO_FABRIC`, or ``None``.

    The three cases differ: :data:`NO_FABRIC` (no ``Fabric`` block, or clique 0) is the definite
    "not on an MNNVL fabric" every plain HGX/NVL8 part reports, while ``None`` (registration not
    ``COMPLETED``, or an unparseable id) is "could not tell" over a possibly real fabric. Only the
    definite answer lets the cross-check judge a declared domain on non-MNNVL hardware.

    Parsed within the ``Fabric`` block alone: ``State`` is a common key name elsewhere in the dump,
    and the top-level ``GPU Fabric GUID`` line must not open the block.
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


def nvlink_fabric_clique_id() -> int | None:
    """This GPU's NVLink fabric-clique id, :data:`NO_FABRIC`, or ``None`` when it cannot be read.

    Two GPUs share NVLink iff they report the same clique. Three-way contract in
    :func:`_parse_fabric_clique`; ``None`` here also covers a missing or failing ``nvidia-smi``.

    Read out of process rather than through NVML's Python binding: the binding's fabric getters take
    a versioned out-param struct whose layout must match the driver's, and a mismatch segfaults
    inside the C call, uncatchably, on every rank during ``ParallelismConfig.__post_init__``.
    """
    try:
        return _fabric_clique_for_device(torch.cuda.current_device())
    except Exception:  # advisory only, never fatal
        return None


@functools.cache
def _fabric_clique_for_device(device_index: int) -> int | None:
    """One device's fabric clique. Cached for the life of the process.

    Fabric membership is fixed once the driver is up, and the ~60 ms read runs on every
    ``ParallelismConfig`` construction through the MNNVL advisory — negligible for a training
    process, but not for a config sweep that builds tens of thousands of them.
    """
    try:
        # By PCI bus id, not index: nvidia-smi enumerates every host GPU while the local rank indexes
        # the CUDA_VISIBLE_DEVICES-masked view, so under masking the read would hit another GPU.
        props = torch.cuda.get_device_properties(device_index)
        bus_id = f"{props.pci_domain_id:08x}:{props.pci_bus_id:02x}:{props.pci_device_id:02x}.0"
        query = subprocess.run(
            ["nvidia-smi", "-q", "-i", bus_id],
            capture_output=True,
            text=True,
            timeout=_NVIDIA_SMI_TIMEOUT_S,
            check=True,
        )
    except Exception:  # advisory only, never fatal
        return None
    return _parse_fabric_clique(query.stdout)


def detect_nvlink_fabric_present() -> bool:
    """Best-effort, non-collective check for a multi-node NVLink (MNNVL) fabric.

    Only a real clique id counts; neither the definite negative nor "could not tell" is evidence.
    """
    clique = nvlink_fabric_clique_id()
    return clique is not None and clique != NO_FABRIC


def node_placement_contradiction(host_ids: list[str], gpus_per_node: int) -> str | None:
    """Why the ranks are not laid out ``gpus_per_node`` per node, or ``None`` when they are.

    ``host_ids[rank]`` is that rank's machine identity. Pure, so it is testable without a job, and
    identical on every rank because the gathered list is.

    Every node-local group (EP, CP, TP, ETP) is a contiguous global-rank block
    (:mod:`~src.distributed.group_layout`), which is one machine only while rank ``r`` sits on node
    ``r // gpus_per_node``. One torchrun per node makes that true; round-robin rank numbering
    (``srun --distribution=cyclic`` with an exported rendezvous) does not, and then the groups
    straddle machines while ``requires_rdma`` reports False, so DeepEP takes its intranode CUDA-IPC
    path to peers on another host.
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


def validate_nvlink_domain_against_fabric(nvlink_domain_size: int, world_size: int, gpus_per_node: int = 0) -> None:
    """Check the declared NVLink domain against the fabric the GPUs actually report.

    ``nvlink_domain_size`` is otherwise an unchecked assertion about the hardware: a domain
    straddling two racks makes every "node-local" group half RDMA while ``requires_rdma`` reports
    False, and one left at ``gpus_per_node`` on an NVL72 rack caps expert sharding at 4-8 GPUs.

    Collective and rank-uniform by construction: every rank contributes its clique id and decides
    from the same gathered list, so this cannot raise on a subset of ranks. A ``None`` anywhere
    no-ops the fabric legs, since a partial view is not evidence; :data:`NO_FABRIC` is evidence, and
    ``gpus_per_node`` then rejects a domain wider than the node its ranks sit on (omit it to skip
    that leg). Fabric verdicts are taken per domain, as one job may hold an NVLink island and
    fabric-less nodes at once.

    The machine identity rides along in the same gather and carries its own verdict
    (:func:`node_placement_contradiction`), taken first because the fabric legs and the rank layout
    rest on it and it needs no clique read.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return
    # Size the buffer from the live group, not the declared world: all_gather_object fills exactly
    # get_world_size() slots, so an oversized list keeps pre-fill Nones the straddle math misreads.
    group_world = dist.get_world_size()
    if group_world <= 1 or group_world != world_size:
        return
    # The node width and the machine identity ride along with the clique id: both are per-node, so
    # the verdicts below have to be taken over the gathered values, not this rank's own.
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

    # Fabric-less ranks share one sentinel, so a domain laid across two of them shows one clique to
    # the straddle check below and passes it; the gathered node widths are the only evidence that
    # such a domain cannot exist.
    node_widths = [entry[1] for entry in gathered]
    # Domain-major contiguous blocks — the same layout the node-local EP/CP groups are cut from.
    fabricless: dict[int, int] = {}
    for domain, start in enumerate(range(0, world_size, nvlink_domain_size)):
        block = slice(start, start + nvlink_domain_size)
        if all(clique == NO_FABRIC for clique in cliques[block]):
            width = min(node_widths[block])
            if 0 < width < nvlink_domain_size:
                fabricless[domain] = width
    if fabricless:
        narrowest = min(fabricless.values())
        raise ValueError(
            f"NVLINK_DOMAIN_SIZE={nvlink_domain_size} exceeds the GPUs per node of domain(s) "
            f"{sorted(fabricless)}, where every rank reports NO NVLink fabric — there is no multi-node "
            f"NVLink for them to span, so a domain wider than one node cannot exist there. Their "
            f"'node-local' EP/TP/CP groups would straddle nodes over the network while requires_rdma "
            f"reports False. Unset NVLINK_DOMAIN_SIZE (per-node NVLink), or set it to at most "
            f"{narrowest}."
        )
    if all(clique == NO_FABRIC for clique in cliques):
        return  # no fabric anywhere, every domain within a node — nothing left to cross-check

    straddling = [
        domain
        for domain in range(world_size // nvlink_domain_size)
        if len(set(cliques[domain * nvlink_domain_size : (domain + 1) * nvlink_domain_size])) > 1
    ]
    ranks_per_clique = {clique: cliques.count(clique) for clique in sorted(set(cliques))}
    if straddling:
        raise ValueError(
            f"NVLINK_DOMAIN_SIZE={nvlink_domain_size} does not match the NVLink fabric: domain(s) "
            f"{straddling} span more than one fabric clique, so their 'node-local' EP/TP/CP groups "
            f"would be part NVLink and part RDMA while requires_rdma still reports False. Set "
            f"NVLINK_DOMAIN_SIZE to a value that divides the real clique size "
            f"(this job's ranks per clique: {ranks_per_clique})."
        )
    # NO_FABRIC is one sentinel shared by every fabric-less node, not an island of that many ranks,
    # so such nodes cap the advice at their own width instead of widening it past what they can span.
    local_widths = [count for clique, count in ranks_per_clique.items() if clique != NO_FABRIC]
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
    """Whether this node exposes IMEX channels — the kernel side of cross-OS-node NVLink P2P.

    A filesystem check needing no NVIDIA tooling; an unlistable caps dir returns False rather than
    raising, since this runs inside a rank-uniform gate that must reach its collective.
    """
    try:
        return bool(os.listdir(_MNNVL_IMEX_DIR))
    except OSError:
        return False


def mnnvl_prerequisite_reason(imex_ok: bool, fabric_clique: int | None) -> str | None:
    """Why this rank cannot take part in Multi-Node NVLink, or ``None`` when it can.

    Pure verdict for one rank, called only where the topology requires MNNVL. There a ``None``
    clique is not the benign "could not tell" it is elsewhere: an incomplete Fabric Manager
    registration reports it, which is the failure this check exists to catch.
    """
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


def check_mnnvl_prerequisites(nvlink_domain_size: int, gpus_per_node: int) -> None:
    """Check Multi-Node NVLink prerequisites on every rank. Collective when MNNVL is required.

    Required means ``nvlink_domain_size > gpus_per_node`` (NVL72). Every rank judges its own node
    and the verdicts are joined, so one node missing IMEX channels or still registering with Fabric
    Manager fails by name instead of as a peer-memory fault mid-dispatch; a rank-0-only check would
    miss the failing node.

    Below that threshold, and outside a live multi-rank job, this only advises: a fabric present
    while ``NVLINK_DOMAIN_SIZE`` is unset caps node-local TP/CP/EP at one node for no reason.
    """
    if nvlink_domain_size <= gpus_per_node:
        if not is_global_main_process():
            return
        # Gate on the resolved value: a malformed NVLINK_DOMAIN_SIZE falls back to the per-node
        # domain, exactly when the advisory is needed.
        requested = env_int("NVLINK_DOMAIN_SIZE", None)
        if (requested is None or requested < 1) and detect_nvlink_fabric_present():
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
