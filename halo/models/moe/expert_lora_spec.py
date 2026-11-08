"""The native expert-LoRA vocabulary: which stored expert weights take an adapter, and the recipe."""

import math
from dataclasses import dataclass, field

import torch

# The ``lora_target_modules`` vocabulary derives from this; a family adding a weight root is CI-guarded into it.
LORA_PROJECTION_COVERAGE: dict[str, frozenset[str]] = {
    "gate_up_proj": frozenset({"gate", "up"}),
    "gate_proj": frozenset({"gate"}),
    "up_proj": frozenset({"up"}),
    "down_proj": frozenset({"down"}),
    "gate_proj_gmm": frozenset({"gate"}),
    "up_proj_gmm": frozenset({"up"}),
}
EXPERT_PROJECTIONS: frozenset[str] = frozenset().union(*LORA_PROJECTION_COVERAGE.values())
# ``lora_target_modules`` aliases that stand for the whole FFN.
EXPERT_CONTAINER_TARGETS: frozenset[str] = frozenset({"experts", "mlp.experts"})
EXPERT_LORA_SUFFIXES: tuple[str, str] = ("_lora_A", "_lora_B")


@dataclass(frozen=True)
class ExpertLoraSpec:
    """The ``LoraConfig`` fields the grouped adapters implement; ``split_expert_lora_targets`` rejects the rest."""

    r: int
    alpha: float
    dropout: float = 0.0
    projections: frozenset[str] = field(default_factory=lambda: EXPERT_PROJECTIONS)
    use_rslora: bool = False

    @property
    def scaling(self) -> float:
        return self.alpha / math.sqrt(self.r) if self.use_rslora else self.alpha / self.r

    def adapts(self, weight_attr: str) -> bool:
        coverage = LORA_PROJECTION_COVERAGE.get(weight_attr)
        return coverage is not None and bool(coverage & self.projections)

    def peft_config_conflicts(self, peft_config) -> list[str]:
        """Settings the grouped expert adapters cannot reproduce; the caller words and times the refusal."""
        conflicts = []
        if getattr(peft_config, "use_rslora", False) != self.use_rslora:
            conflicts.append(
                f"use_rslora={getattr(peft_config, 'use_rslora', False)} on the attention adapters vs "
                f"{self.use_rslora} on the expert adapters — the two halves would scale differently"
            )
        for field_name, honoured in (("use_dora", False), ("lora_bias", False), ("init_lora_weights", True)):
            value = getattr(peft_config, field_name, honoured)
            if value != honoured:
                conflicts.append(f"{field_name}={value!r} has no grouped-expert implementation")
        return conflicts


def expert_lora_delta(lora_a: torch.Tensor, lora_b: torch.Tensor, scaling: float) -> torch.Tensor:
    """The fp32 per-expert delta ``scaling * A @ B`` of one weight's grouped adapters, ``[E, K, r]`` by ``[E, r, N]``,
    which a merge adds to the weight in its matmul convention."""
    return scaling * torch.bmm(lora_a.float(), lora_b.float())


def expert_target_projections(target: str) -> frozenset[str] | None:
    if target in EXPERT_CONTAINER_TARGETS:
        return EXPERT_PROJECTIONS
    return LORA_PROJECTION_COVERAGE.get(target)


def reject_expert_lora_with_expert_tp() -> None:
    """``ParallelismConfig`` checks it before the checkpoint downloads."""
    raise ValueError(
        "Expert LoRA is not supported with expert_tensor_parallel_size > 1: the replicated "
        "adapter half receives partial gradients under expert TP and drifts across ranks. "
        "Use EP without expert TP for expert adapters, or remove expert projections from "
        "lora_target_modules."
    )
