"""The native expert-LoRA vocabulary: which stored expert weights take an adapter, and the recipe."""

import math
from dataclasses import dataclass, field

# Logical projections a stored expert weight carries, keyed on its leading attribute segment. The
# ``lora_target_modules`` vocabulary derives from it, and a family adding a weight root is CI-guarded into it.
LORA_PROJECTION_COVERAGE: dict[str, frozenset[str]] = {
    "gate_up_proj": frozenset({"gate", "up"}),
    "gate_proj": frozenset({"gate"}),
    "up_proj": frozenset({"up"}),
    "down_proj": frozenset({"down"}),
    "gate_proj_gmm": frozenset({"gate"}),
    "up_proj_gmm": frozenset({"up"}),
}
# Every logical projection any family stores; what a container-level target requests.
EXPERT_PROJECTIONS: frozenset[str] = frozenset().union(*LORA_PROJECTION_COVERAGE.values())
# ``lora_target_modules`` aliases naming the module that holds the experts: they stand for the whole FFN.
EXPERT_CONTAINER_TARGETS: frozenset[str] = frozenset({"experts", "mlp.experts"})


@dataclass(frozen=True)
class ExpertLoraSpec:
    """Native grouped-LoRA recipe for EP-distributed experts.

    ``projections`` is the logical set (``gate``/``up``/``down``), resolved to stored weights through
    :data:`LORA_PROJECTION_COVERAGE`. Mirrors the ``LoraConfig`` fields the grouped adapters implement;
    ``split_expert_lora_targets`` rejects the rest where the spec is built.
    """

    r: int
    alpha: float
    dropout: float = 0.0
    projections: frozenset[str] = field(default_factory=lambda: EXPERT_PROJECTIONS)
    use_rslora: bool = False

    @property
    def scaling(self) -> float:
        """PEFT's adapter scaling: ``alpha / sqrt(r)`` under rsLoRA, else ``alpha / r``."""
        return self.alpha / math.sqrt(self.r) if self.use_rslora else self.alpha / self.r

    def adapts(self, weight_attr: str) -> bool:
        """Whether the stored expert weight ``weight_attr`` takes an adapter."""
        coverage = LORA_PROJECTION_COVERAGE.get(weight_attr)
        return coverage is not None and bool(coverage & self.projections)

    def peft_config_conflicts(self, peft_config) -> list[str]:
        """The ``LoraConfig`` settings the grouped expert adapters cannot reproduce, which would apply
        to a co-resident attention adapter alone; the caller words and times the refusal."""
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


def expert_target_projections(target: str) -> frozenset[str] | None:
    """The logical projections a ``lora_target_modules`` entry requests, or None if it names no expert weight."""
    if target in EXPERT_CONTAINER_TARGETS:
        return EXPERT_PROJECTIONS
    return LORA_PROJECTION_COVERAGE.get(target)


def reject_expert_lora_with_expert_tp() -> None:
    """Raise for native expert LoRA under expert TP: adapters are never ETP-sharded, so the replicated
    half would take a partial gradient. ``ParallelismConfig`` checks it before the checkpoint downloads,
    ``EPConfig`` again for a hand-built config."""
    raise ValueError(
        "Expert LoRA is not supported with expert_tensor_parallel_size > 1: the replicated "
        "adapter half receives partial gradients under expert TP and drifts across ranks. "
        "Use EP without expert TP for expert adapters, or remove expert projections from "
        "lora_target_modules."
    )
