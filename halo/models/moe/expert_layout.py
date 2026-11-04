"""One runtime-to-hub expert layout per MoE family: the transform the EP gather and the offline shard merge both
apply to full expert banks (every expert, every expert-TP shard) held in the matmul convention ``[E, K, N]``."""

from __future__ import annotations

import enum
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import torch

# Fused 3-D expert tensor names (dim 0 = expert) as an HF checkpoint spells them, longest first so a suffix
# matcher never reads a ``_bias`` name as its weight. ``up_proj`` is a gateless expert's bank.
HF_FUSED_EXPERT_KEYS: tuple[str, ...] = ("gate_up_proj_bias", "down_proj_bias", "gate_up_proj", "down_proj", "up_proj")
# The three projections of a GLU expert stored apart, as runtime attributes and as a per-expert hub spelling.
SEPARATE_GLU_PROJECTIONS: tuple[str, str, str] = ("gate_proj", "up_proj", "down_proj")
# The two projections of a gateless expert, under one spelling at runtime and on the hub.
SEPARATE_MLP_PROJECTIONS: tuple[str, str] = ("up_proj", "down_proj")


class ExpertLayoutKind(enum.Enum):
    # Contiguous ``[gate; up]`` halves, transposed to ``F.linear`` convention.
    FUSED = "fused"
    # FUSED, then one tensor per expert under ``per_expert_keys``.
    FUSED_SPLIT_PER_EXPERT = "fused_split_per_expert"
    # Gate, up and down stored apart, written one tensor per expert under ``per_expert_keys``.
    SEPARATE_GLU = "separate_glu"
    # Up and down of a gateless expert (Nemotron-H), written one tensor per expert under their own names.
    SEPARATE_MLP = "separate_mlp"
    # GPT-OSS: ``[g0, u0, g1, u1, ...]`` with biases, kept in matmul convention.
    INTERLEAVED_GLU = "interleaved_glu"


# The runtime attributes each kind has a branch for; expert TP stores the GLU halves apart under the plain names,
# GPT-OSS's grouped GEMM under ``*_gmm``.
_RUNTIME_KEYS: dict[ExpertLayoutKind, frozenset[str]] = {
    ExpertLayoutKind.FUSED: frozenset({*HF_FUSED_EXPERT_KEYS, "gate_proj", "up_proj"}),
    ExpertLayoutKind.FUSED_SPLIT_PER_EXPERT: frozenset({"gate_up_proj", "gate_proj", "up_proj", "down_proj"}),
    ExpertLayoutKind.SEPARATE_GLU: frozenset(SEPARATE_GLU_PROJECTIONS),
    ExpertLayoutKind.SEPARATE_MLP: frozenset(SEPARATE_MLP_PROJECTIONS),
    ExpertLayoutKind.INTERLEAVED_GLU: frozenset(
        {
            *HF_FUSED_EXPERT_KEYS,
            *(f"{half}_proj{suffix}" for half in ("gate", "up") for suffix in ("", "_bias", "_gmm", "_gmm_bias")),
        }
    ),
}


@dataclass(frozen=True)
class ExpertLayout:
    """``per_expert_keys`` is the ``(gate, up, down)`` spelling of a one-tensor-per-expert checkpoint. ``key_prefix``
    is empty where the EP layer replaces the expert container itself (Gemma 4)."""

    kind: ExpertLayoutKind = ExpertLayoutKind.FUSED
    per_expert_keys: tuple[str, str, str] | None = None
    key_prefix: str = "experts."

    def __post_init__(self):
        per_expert = (ExpertLayoutKind.FUSED_SPLIT_PER_EXPERT, ExpertLayoutKind.SEPARATE_GLU)
        if self.kind in per_expert and self.per_expert_keys is None:
            raise TypeError(
                f"ExpertLayout({self.kind.name}) writes one tensor per expert but names no per_expert_keys."
            )


def interleave_gate_up(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """``[g0, u0, g1, u1, ...]`` along the last axis, for weights ``[E, H, M]`` and biases ``[E, M]`` alike."""
    interleaved = torch.empty(*gate.shape[:-1], 2 * gate.shape[-1], device=gate.device, dtype=gate.dtype)
    interleaved[..., ::2] = gate
    interleaved[..., 1::2] = up
    return interleaved


def split_interleaved_gate_up(interleaved: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return interleaved[..., ::2], interleaved[..., 1::2]


def iter_per_expert_split(banks: Sequence[tuple[torch.Tensor, tuple[str, ...]]]) -> Iterator[tuple[str, torch.Tensor]]:
    """``("<expert>.<name>", view)`` expert by expert, each bank's expert slice split evenly into one view per name."""
    for expert in range(banks[0][0].shape[0]):
        for bank, names in banks:
            for name, part in zip(names, bank[expert].chunk(len(names)), strict=True):
                yield f"{expert}.{name}", part


def _per_expert_linear_weights(
    banks: Sequence[tuple[torch.Tensor, tuple[str, ...]]], prefix: str, device: torch.device | str
) -> dict[str, torch.Tensor]:
    """Matmul-convention banks stored apart, one ``F.linear``-convention ``<name>.weight`` per expert."""
    return {
        f"{prefix}{key}.weight": part.transpose(0, 1).contiguous().to(device)
        for key, part in iter_per_expert_split(banks)
    }


def to_hub_expert_tensors(
    layout: ExpertLayout, tensors: Mapping[str, torch.Tensor], *, device: torch.device | str
) -> dict[str, torch.Tensor]:
    """Keyed relative to the EP layer before the family's export renames. Raises on a tensor the layout has no branch
    for, as an expert tensor dropped from the export loads and is wrong."""
    accepted = _RUNTIME_KEYS[layout.kind]
    if unexpected := set(tensors) - accepted:
        raise ValueError(
            f"The {layout.kind.name} expert layout found unexpected expert params {sorted(unexpected)}: it covers "
            f"exactly {sorted(accepted)}; refusing to drop them."
        )
    prefix = layout.key_prefix
    if layout.kind is ExpertLayoutKind.SEPARATE_GLU:
        banks = [
            (tensors[attr], (name,))
            for attr, name in zip(SEPARATE_GLU_PROJECTIONS, layout.per_expert_keys, strict=True)
        ]
        return _per_expert_linear_weights(banks, prefix, device)

    if layout.kind is ExpertLayoutKind.SEPARATE_MLP:
        banks = [(tensors[name], (name,)) for name in SEPARATE_MLP_PROJECTIONS]
        return _per_expert_linear_weights(banks, prefix, device)

    if layout.kind is ExpertLayoutKind.INTERLEAVED_GLU:
        gate = next((name for name in ("gate_proj_gmm", "gate_proj") if name in tensors), None)
        if gate is None:
            gate_up, gate_up_bias = tensors["gate_up_proj"], tensors.get("gate_up_proj_bias")
        else:
            up = gate.replace("gate", "up", 1)
            gate_up = interleave_gate_up(tensors[gate], tensors[up])
            gate_up_bias = None
            if f"{gate}_bias" in tensors:
                gate_up_bias = interleave_gate_up(tensors[f"{gate}_bias"], tensors[f"{up}_bias"])
        down = tensors["down_proj"]
    else:
        # Concatenating the transposed halves builds the [E, 2M, H] bank in one allocation.
        gate_up = (
            tensors["gate_up_proj"].transpose(1, 2).contiguous()
            if "gate_up_proj" in tensors
            else torch.cat([tensors["gate_proj"].transpose(1, 2), tensors["up_proj"].transpose(1, 2)], dim=1)
        ).to(device)
        down = tensors["down_proj"].transpose(1, 2).contiguous().to(device)
        gate_up_bias = tensors.get("gate_up_proj_bias")
        if layout.kind is ExpertLayoutKind.FUSED_SPLIT_PER_EXPERT:
            gate_name, up_name, down_name = layout.per_expert_keys
            banks = [(gate_up, (gate_name, up_name)), (down, (down_name,))]
            return {f"{prefix}{key}.weight": part.contiguous() for key, part in iter_per_expert_split(banks)}

    hub = {
        "gate_up_proj": gate_up,
        "down_proj": down,
        "gate_up_proj_bias": gate_up_bias,
        "down_proj_bias": tensors.get("down_proj_bias"),
    }
    return {f"{prefix}{name}": tensor.to(device) for name, tensor in hub.items() if tensor is not None}
