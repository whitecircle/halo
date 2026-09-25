"""Tensor-parallel placement for LoRA factors injected into an HF-native TP model."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
from peft import LoraConfig
from peft.tuners.lora.layer import LoraLayer
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard, distribute_tensor
from transformers.distributed.tensor_parallel import ALL_PARALLEL_STYLES, _get_parameter_tp_plan

from src.distributed.mesh import MeshDim, mesh_dim_names

_SUPPORTED_STYLES = {"colwise": 0, "rowwise": 1}
_SUPPORTED_INITIALIZATIONS = {True, False, "gaussian", "orthogonal"}
_BRIDGE_MARKER = "_halo_tp_lora_bridge_factor_ids"


@dataclass(frozen=True)
class _LoraTPTarget:
    name: str
    layer: LoraLayer
    adapter_name: str
    style_name: str
    shard_dim: int
    tp_mesh: DeviceMesh
    sharded_factor: nn.Linear
    replicated_factor: nn.Linear
    global_shape: torch.Size
    local_shape: torch.Size


def apply_tp_to_lora(model: nn.Module) -> int:
    """Place PEFT LoRA factors over an already tensor-parallel base model.

    PEFT 0.18 builds adapter factors from each DTensor base weight's local shape. A plain TP style
    would therefore shard the affected factor twice. This bridge wraps that local factor with the
    base weight's global metadata, then installs only the style's forward transforms. Replicated
    factors remain plain parameters so Halo's TP gradient synchronization counts them once.

    Returns the number of LoRA target modules bridged. A zero return means this bridge already
    converted every target. The function validates the whole model before changing any parameter,
    and rejects DTensor factors it did not create.
    """
    tp_model = _base_model(model)
    adapter_name, config = _validate_config(model)
    targets = _collect_targets(tp_model, adapter_name)
    _reject_embedding_and_head_targets(tp_model, targets)

    markers = [getattr(target.layer, _BRIDGE_MARKER, None) for target in targets]
    if any(marker is not None for marker in markers):
        if not all(marker is not None for marker in markers):
            raise ValueError("LoRA TP targets mix bridge-converted and unconverted layers.")
        for target, marker in zip(targets, markers, strict=True):
            factor_ids = (id(target.sharded_factor.weight), id(target.replicated_factor.weight))
            if marker != factor_ids:
                raise ValueError(f"LoRA TP target {target.name!r} changed factors after bridging.")
            _validate_bridged_target(target)
        return 0
    if any(isinstance(target.sharded_factor.weight, DTensor) for target in targets):
        raise ValueError(
            "LoRA TP found DTensor factors not created by this bridge. Their initialization and "
            "replicated-factor synchronization have not been validated."
        )

    devices = sorted(
        {
            target.sharded_factor.weight.device.index
            for target in targets
            if target.sharded_factor.weight.device.type == "cuda"
        }
    )
    with torch.random.fork_rng(devices=devices):
        for target in targets:
            _bridge_target(target, config.init_lora_weights)
            setattr(
                target.layer,
                _BRIDGE_MARKER,
                (id(target.sharded_factor.weight), id(target.replicated_factor.weight)),
            )
    return len(targets)


def _base_model(model: nn.Module) -> nn.Module:
    getter = getattr(model, "get_base_model", None)
    return getter() if callable(getter) else model


def _validate_config(model: nn.Module) -> tuple[str, LoraConfig]:
    configs = getattr(model, "peft_config", None)
    if not isinstance(configs, dict) or not configs:
        raise ValueError("LoRA TP bridging requires a PEFT model with exactly one configured adapter.")
    if len(configs) != 1:
        raise ValueError(f"LoRA TP bridging supports one adapter, but found {sorted(configs)}.")

    adapter_name, config = next(iter(configs.items()))
    conflicts = []
    for field in ("use_dora", "lora_bias", "modules_to_save", "trainable_token_indices", "target_parameters"):
        if getattr(config, field, None):
            conflicts.append(field)
    if getattr(config, "bias", "none") != "none":
        conflicts.append("bias")
    if float(getattr(config, "lora_dropout", 0.0) or 0.0) > 0:
        conflicts.append("lora_dropout")
    for field in ("alora_invocation_tokens", "arrow_config", "megatron_config"):
        if getattr(config, field, None):
            conflicts.append(field)
    if conflicts:
        raise ValueError(
            "LoRA TP bridging supports standard linear LoRA only; unsupported LoraConfig fields are set: "
            f"{sorted(conflicts)}."
        )

    initialization = getattr(config, "init_lora_weights", True)
    if initialization not in _SUPPORTED_INITIALIZATIONS:
        raise ValueError(
            f"LoRA TP does not support init_lora_weights={initialization!r}. PiSSA, OLoRA, LoftQ, EVA, "
            "and CorDA require a full base weight or full layer input; use True, False, 'gaussian', or "
            "'orthogonal'."
        )
    return adapter_name, config


def _collect_targets(tp_model: nn.Module, adapter_name: str) -> list[_LoraTPTarget]:
    tp_plan = getattr(tp_model, "_tp_plan", None) or {}
    if not tp_plan:
        raise ValueError("LoRA TP bridging requires the applied HF TP plan on model._tp_plan.")

    targets = []
    for name, layer in tp_model.named_modules():
        if not isinstance(layer, LoraLayer):
            continue
        if set(layer.lora_A) != {adapter_name} or set(layer.lora_B) != {adapter_name}:
            raise ValueError(
                f"LoRA target {name!r} does not contain exactly adapter {adapter_name!r} in both A and B factors."
            )
        if getattr(layer, "lora_variant", None):
            raise ValueError(f"LoRA target {name!r} uses a LoRA variant; TP supports vanilla linear LoRA only.")

        base_layer = layer.get_base_layer()
        if not isinstance(base_layer, nn.Linear):
            raise ValueError(
                f"LoRA target {name!r} wraps {type(base_layer).__name__}; TP supports nn.Linear targets only."
            )
        base_weight = base_layer.weight
        if not isinstance(base_weight, DTensor):
            raise ValueError(
                f"LoRA target {name!r} has a plain base weight. Every target must be covered by HF-native "
                "tensor parallelism."
            )

        style_name = _get_parameter_tp_plan(f"{name}.weight", tp_plan)
        if style_name not in _SUPPORTED_STYLES:
            raise ValueError(
                f"LoRA target {name!r} uses TP style {style_name!r}; only literal 'colwise' and 'rowwise' "
                "are supported. Gathered, packed, split, and embedding styles are not compatible."
            )
        shard_dim = _SUPPORTED_STYLES[style_name]
        tp_mesh = _validate_base_placement(name, base_weight, shard_dim)

        a_factor = layer.lora_A[adapter_name]
        b_factor = layer.lora_B[adapter_name]
        if not isinstance(a_factor, nn.Linear) or not isinstance(b_factor, nn.Linear):
            raise ValueError(f"LoRA target {name!r} does not use linear A/B factors.")
        if a_factor.bias is not None or b_factor.bias is not None:
            raise ValueError(f"LoRA target {name!r} has an adapter bias; lora_bias is unsupported under TP.")

        rank = layer.r[adapter_name]
        out_features, in_features = base_weight.shape
        if style_name == "colwise":
            sharded_factor, replicated_factor = b_factor, a_factor
            global_shape = torch.Size((out_features, rank))
            local_shape = torch.Size((base_weight.to_local().shape[0], rank))
            replicated_shape = torch.Size((rank, in_features))
        else:
            sharded_factor, replicated_factor = a_factor, b_factor
            global_shape = torch.Size((rank, in_features))
            local_shape = torch.Size((rank, base_weight.to_local().shape[1]))
            replicated_shape = torch.Size((out_features, rank))

        if replicated_factor.weight.shape != replicated_shape:
            raise ValueError(
                f"LoRA target {name!r} has replicated factor shape {tuple(replicated_factor.weight.shape)}, "
                f"expected {tuple(replicated_shape)}."
            )
        if isinstance(replicated_factor.weight, DTensor):
            raise ValueError(
                f"LoRA target {name!r} made its replicated factor a DTensor. It must remain a plain parameter "
                "for Halo's TP gradient sync and grad-norm accounting."
            )
        if sharded_factor.weight.shape not in (local_shape, global_shape):
            raise ValueError(
                f"LoRA target {name!r} has sharded factor shape {tuple(sharded_factor.weight.shape)}, expected "
                f"local {tuple(local_shape)} or global {tuple(global_shape)}."
            )

        targets.append(
            _LoraTPTarget(
                name=name,
                layer=layer,
                adapter_name=adapter_name,
                style_name=style_name,
                shard_dim=shard_dim,
                tp_mesh=tp_mesh,
                sharded_factor=sharded_factor,
                replicated_factor=replicated_factor,
                global_shape=global_shape,
                local_shape=local_shape,
            )
        )

    if not targets:
        raise ValueError("LoRA TP bridging found no PEFT LoraLayer targets in the base model.")
    return targets


def _validate_base_placement(name: str, weight: DTensor, shard_dim: int) -> DeviceMesh:
    mesh = weight.device_mesh
    names = mesh_dim_names(mesh)
    if mesh.ndim != 1 or names != (MeshDim.TP,):
        raise ValueError(
            f"LoRA target {name!r} uses base-weight mesh dims {names or ('unnamed',)}; the bridge "
            "requires a named one-dimensional TP mesh."
        )
    placements = weight.placements
    actual = placements[0] if len(placements) == 1 else None
    if not isinstance(actual, Shard) or actual.dim % weight.ndim != shard_dim:
        raise ValueError(
            f"LoRA target {name!r} has base-weight placements {placements}, but its TP style requires "
            f"Shard({shard_dim})."
        )
    return mesh


def _reject_embedding_and_head_targets(tp_model: nn.Module, targets: list[_LoraTPTarget]) -> None:
    endpoints = set()
    for getter_name in ("get_input_embeddings", "get_output_embeddings"):
        getter = getattr(tp_model, getter_name, None)
        if not callable(getter):
            continue
        endpoint = getter()
        if endpoint is None:
            continue
        endpoints.add(endpoint)
        if isinstance(endpoint, LoraLayer):
            endpoints.add(endpoint.get_base_layer())

    offenders = [
        target.name for target in targets if target.layer in endpoints or target.layer.get_base_layer() in endpoints
    ]
    if offenders:
        raise ValueError(
            f"LoRA TP does not support input-embedding or lm_head targets; matched {offenders}. "
            "Target attention or MLP projections instead."
        )


def _validate_bridged_target(target: _LoraTPTarget) -> None:
    weight = target.sharded_factor.weight
    if weight.device_mesh != target.tp_mesh or weight.shape != target.global_shape:
        raise ValueError(
            f"Bridged TP factor for {target.name!r} has mesh/shape {weight.device_mesh}/{tuple(weight.shape)}, "
            f"expected {target.tp_mesh}/{tuple(target.global_shape)}."
        )
    placements = weight.placements
    actual = placements[0] if len(placements) == 1 else None
    if not isinstance(actual, Shard) or actual.dim % weight.ndim != target.shard_dim:
        raise ValueError(
            f"Bridged TP factor for {target.name!r} has placements {placements}, expected Shard({target.shard_dim})."
        )


def _bridge_target(target: _LoraTPTarget, initialization: bool | str) -> None:
    style = ALL_PARALLEL_STYLES[target.style_name]
    if initialization == "orthogonal":
        full_a, full_b = _orthogonal_pair(target)
        sharded_full = full_b if target.style_name == "colwise" else full_a
        replicated_full = full_a if target.style_name == "colwise" else full_b
        _copy_replicated(target, replicated_full)
        _distribute_full_factor(target, sharded_full)
    else:
        _broadcast_replicated(target)
        starts_nonzero = target.style_name == "rowwise" or initialization is False
        if starts_nonzero:
            full = _initialized_full_factor(target, initialization)
            _distribute_full_factor(target, full)
        elif target.sharded_factor.weight.shape == target.global_shape:
            full = target.sharded_factor.weight.detach().clone()
            _broadcast(full, target.tp_mesh)
            _distribute_full_factor(target, full)
        else:
            _wrap_local_factor(target)

    style.install_forward(target.sharded_factor, target.tp_mesh)


def _initialized_full_factor(target: _LoraTPTarget, initialization: bool | str) -> torch.Tensor:
    weight = target.sharded_factor.weight
    full = torch.empty(target.global_shape, device=weight.device, dtype=torch.float32)
    if target.tp_mesh.get_local_rank() == 0:
        if initialization == "gaussian":
            nn.init.normal_(full, std=1 / target.layer.r[target.adapter_name])
        else:
            nn.init.kaiming_uniform_(full, a=math.sqrt(5))
    full = full.to(weight.dtype)
    _broadcast(full, target.tp_mesh)
    return full


def _orthogonal_pair(target: _LoraTPTarget) -> tuple[torch.Tensor, torch.Tensor]:
    rank = target.layer.r[target.adapter_name]
    if rank % 2:
        raise ValueError(f"Orthogonal LoRA initialization requires an even rank, got {rank} for {target.name!r}.")
    out_features, in_features = target.layer.get_base_layer().weight.shape
    device = target.sharded_factor.weight.device
    a = torch.empty((rank, in_features), device=device, dtype=torch.float32)
    b = torch.empty((out_features, rank), device=device, dtype=torch.float32)
    if target.tp_mesh.get_local_rank() == 0:
        q, _ = torch.linalg.qr(torch.randn((rank, rank), device=device))
        a.copy_(torch.randn((in_features, rank // 2), device=device).mm(q[0::2]).T / 10.0)
        b.copy_(torch.randn((rank // 2, out_features), device=device).T.mm(q[1::2]) / 10.0)
    a = a.to(target.layer.lora_A[target.adapter_name].weight.dtype)
    b = b.to(target.layer.lora_B[target.adapter_name].weight.dtype)
    _broadcast(a, target.tp_mesh)
    _broadcast(b, target.tp_mesh)
    return a, b


def _copy_replicated(target: _LoraTPTarget, full: torch.Tensor) -> None:
    with torch.no_grad():
        target.replicated_factor.weight.copy_(full)
    _broadcast_replicated(target)


def _broadcast_replicated(target: _LoraTPTarget) -> None:
    weight = target.replicated_factor.weight
    with torch.no_grad():
        _broadcast(weight, target.tp_mesh)
    _assert_bitwise_replicated(weight, target.tp_mesh, target.name)


def _broadcast(tensor: torch.Tensor, mesh: DeviceMesh) -> None:
    group = mesh.get_group()
    source = dist.get_process_group_ranks(group)[0]
    dist.broadcast(tensor, src=source, group=group)


def _assert_bitwise_replicated(weight: torch.Tensor, mesh: DeviceMesh, name: str) -> None:
    bits = weight.detach().contiguous().view(torch.uint8)
    gathered = [torch.empty_like(bits) for _ in range(mesh.size())]
    dist.all_gather(gathered, bits, group=mesh.get_group())
    if any(not torch.equal(peer, gathered[0]) for peer in gathered[1:]):
        raise RuntimeError(f"LoRA replicated factor for {name!r} differs across the TP group after broadcast.")


def _distribute_full_factor(target: _LoraTPTarget, full: torch.Tensor) -> None:
    placement = Shard(target.shard_dim)
    distributed = distribute_tensor(full, target.tp_mesh, [placement], src_data_rank=None)
    target.sharded_factor._parameters["weight"] = nn.Parameter(
        distributed,
        requires_grad=target.sharded_factor.weight.requires_grad,
    )


def _wrap_local_factor(target: _LoraTPTarget) -> None:
    weight = target.sharded_factor.weight
    if weight.shape != target.local_shape:
        raise ValueError(
            f"LoRA target {target.name!r} has local factor shape {tuple(weight.shape)}, expected "
            f"{tuple(target.local_shape)} before DTensor wrapping."
        )
    stride = torch.empty(target.global_shape, device="meta").stride()
    distributed = DTensor.from_local(
        weight,
        target.tp_mesh,
        [Shard(target.shard_dim)],
        run_check=False,
        shape=target.global_shape,
        stride=stride,
    )
    target.sharded_factor._parameters["weight"] = nn.Parameter(distributed, requires_grad=weight.requires_grad)
