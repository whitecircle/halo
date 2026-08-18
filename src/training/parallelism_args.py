"""Build a :class:`ParallelismConfig` from parsed ``DistributedArguments``.

Lives above ``src/distributed`` because it reads the trainer class the entry script is about to
build — the one place the parallelism package would otherwise have to look upward.
"""

from dataclasses import MISSING, fields
from typing import TYPE_CHECKING

from src.distributed.parallelism_config import ParallelismConfig

if TYPE_CHECKING:
    from src.distributed.expert_parallel.config import ExpertLoraSpec

# The lowp_* shape knobs — forwarded only for SFT, so a non-default value anywhere else is a knob the
# user set that nothing reads. Defaults come from the ParallelismConfig fields, never restated here.
_LOWP_SHAPE_KNOBS: tuple[str, ...] = (
    "lowp_apply_dense_mlp",
    "lowp_apply_moe_experts",
    "lowp_keep_first_blocks",
    "lowp_keep_last_blocks",
)

# Knob -> its default, read off the dataclass fields so the SFT-only gate can never drift from the
# class declaration; a knob name that stops matching a field raises here (KeyError) at import.
_PC_FIELD_DEFAULTS = {f.name: f.default for f in fields(ParallelismConfig) if f.default is not MISSING}
_LOWP_SHAPE_DEFAULTS: dict[str, bool | int] = {name: _PC_FIELD_DEFAULTS[name] for name in _LOWP_SHAPE_KNOBS}
# Forwarded only under ``allow_low_precision``, so they are excluded from the by-name forwarding below.
_LOWP_KNOBS: frozenset[str] = frozenset({"lowp_precision", *_LOWP_SHAPE_KNOBS})

# Rows ONE model call carries per dataset example, declared on the trainer class. Absent == 1: a
# trainer that does not concatenate is the ordinary case, and 1 is its correct multiplier.
FORWARD_ROWS_PER_EXAMPLE_ATTR = "_forward_rows_per_example"


def same_name_arg_forwards(dist_args) -> tuple[str, ...]:
    """``ParallelismConfig`` fields ``dist_args`` declares under the same name, in field order.

    The by-name half of :func:`parallelism_config_from_args`' wiring: derived from the two
    declarations rather than listed, so a knob added to both is forwarded by construction. Fields the
    builder renames (``expert_parallel_size`` → ``ep_size``) or derives are absent by definition, and
    the ``lowp_*`` knobs are excluded because only SFT forwards them.
    """
    return tuple(
        f.name
        for f in fields(ParallelismConfig)
        if f.init and f.name not in _LOWP_KNOBS and hasattr(dist_args, f.name)
    )


def forward_rows_per_example(trainer_cls) -> int:
    """Sequence rows one model call carries per dataset example, per ``trainer_cls``.

    The preference and Bradley-Terry reward trainers score chosen and rejected in a SINGLE
    concatenated forward (``torch.cat([chosen, rejected], dim=0)``), so their MoE layers present
    twice the rows ``per_device_train_batch_size`` names and the DeepEP dispatch ceiling must be
    judged against that. Read off the class that owns the forward; anything that declares nothing
    runs one row per example.
    """
    return max(1, int(getattr(trainer_cls, FORWARD_ROWS_PER_EXAMPLE_ATTR, 1) or 1))


def parallelism_config_from_args(
    dist_args,
    *,
    training_config=None,
    trainer_cls=None,
    supports_cp: bool = True,
    supports_pp: bool = True,
    allow_low_precision: bool = False,
    supports_init_from_scratch: bool = False,
    expert_lora: "ExpertLoraSpec | None" = None,
) -> "ParallelismConfig":
    """Build a :class:`ParallelismConfig` from parsed ``DistributedArguments`` — shared EP/CP/TP/ETP
    wiring for every training entry script.

    Args:
        dist_args: parsed ``DistributedArguments``.
        training_config: the parsed TRL/HF training config, read for the run's declared per-rank
            token budget (``rows × per_device_train_batch_size × max_length``) — what
            :meth:`ParallelismConfig.validate_against_model_config` judges against DeepEP's dispatch
            ceilings. ``None``, or a config declaring no ``max_length``, leaves that gate off.
        trainer_cls: the trainer class this script builds, read for
            :data:`FORWARD_ROWS_PER_EXAMPLE_ATTR` — how many sequence rows ONE model call carries per
            dataset example. The preference and Bradley-Terry reward trainers score chosen and
            rejected in a single concatenated forward, so their MoE layers present twice the rows a
            token budget computed from ``per_device_train_batch_size`` alone would predict. Declared
            on the trainer (the object that owns the forward), never as a per-script constant.
        supports_cp: ``False`` rejects a CLI ``context_parallel_size > 1`` and forces CP off (all
            non-CP trainers; SFT leaves it ``True``).
        supports_pp: ``False`` rejects a CLI ``pipeline_parallel_size > 1`` at config time. The
            trainer's ``_supports_pp`` gate would reject it anyway, but only after the model (and
            for distillation/GRPO scripts a second model or a vLLM probe) has already loaded.
        allow_low_precision: forward the ``lowp_*`` knobs (SFT only); elsewhere a non-``bf16``
            ``lowp_precision`` is rejected.
        expert_lora: peeled native EP expert-LoRA spec. Must be passed here, not assigned onto the
            returned config: only construction runs ``__post_init__``, where the PP rejection lives,
            and :meth:`ParallelismConfig.create_ep_config` caches the ``EPConfig`` it builds, so a
            later assignment never reaches the layers.
    """
    if not allow_low_precision:
        if dist_args.lowp_precision != "bf16":
            raise ValueError(
                f"lowp_precision={dist_args.lowp_precision!r} is only validated for SFT training; this "
                "trainer runs in bf16. Remove lowp_precision from the config (or use the SFT script)."
            )
        # The other lowp_* knobs are not forwarded here either, so a config that sets them outside SFT
        # parses cleanly and does nothing. Reject rather than let the user believe they took effect.
        ignored = [name for name, default in _LOWP_SHAPE_DEFAULTS.items() if getattr(dist_args, name) != default]
        if ignored:
            raise ValueError(
                f"{', '.join(ignored)} apply only to SFT training and would be silently dropped here. "
                "Remove them from the config, or run the SFT script."
            )

    # Only SFT threads init_from_scratch into the loader; elsewhere it would silently fine-tune pretrained.
    if not supports_init_from_scratch and dist_args.init_from_scratch:
        raise ValueError(
            "init_from_scratch is only supported by the SFT script (scripts/training/sft.py); "
            "this script would silently ignore it and train pretrained weights."
        )

    if not supports_cp and dist_args.context_parallel_size > 1:
        raise ValueError(
            "This trainer does not support Context Parallelism (CP). Use Expert Parallelism (EP) "
            "and/or Tensor Parallelism (TP) instead, or remove --context_parallel_size."
        )

    if not supports_pp and dist_args.pipeline_parallel_size > 1:
        raise ValueError(
            "This trainer does not support Pipeline Parallelism (PP). Use Expert Parallelism (EP) "
            "and/or Tensor Parallelism (TP) instead, or remove --pipeline_parallel_size."
        )

    # The PP seams (config validation, rank math, stage/loss contracts) ship, but the schedule
    # engine behind PipelineRuntime does not yet — rejected here, at the single production entry
    # point, before any rank math or model loading is touched.
    if dist_args.pipeline_parallel_size > 1:
        raise ValueError(
            "Pipeline parallelism is not yet available in this release. Set "
            "--pipeline_parallel_size=1 (the default) and shard with EP/TP/CP instead; see "
            "agent-docs/parallelism/pipeline-parallelism.md."
        )

    effective_cp_size = dist_args.context_parallel_size if supports_cp else 1

    # Every field the two dataclasses spell the SAME way forwards itself, so a knob added to both is
    # wired by construction rather than by a hand-maintained line that can go missing. Only the
    # renames, the derivations and the SFT-gated lowp block below stay explicit.
    kwargs = {name: getattr(dist_args, name) for name in same_name_arg_forwards(dist_args)}
    kwargs.update(
        ep_size=dist_args.expert_parallel_size,
        cp_size=effective_cp_size,
        tp_size=dist_args.tensor_parallel_size,
        expert_tp_size=dist_args.expert_tensor_parallel_size,
        pp_size=dist_args.pipeline_parallel_size,
        pp_schedule=dist_args.pipeline_schedule,
        pp_microbatches=dist_args.pipeline_microbatches,
        pp_split=dist_args.pipeline_split,
        ep_fp32_router=dist_args.fp32_router,
        ep_fp32_experts=dist_args.fp32_experts,
        expert_lora=expert_lora,
        # Read here rather than in the entry script so the arithmetic sits with the gate that judges
        # it. The two factors stay apart because only one of them is knowable now: a trainer whose
        # config has no max_length at all (generation-shaped budgets) declares no rows and no budget,
        # while ``max_length: null`` declares its rows and leaves the length to the gate's
        # context-window resolution.
        ep_rows_per_device=(
            forward_rows_per_example(trainer_cls)
            * int(getattr(training_config, "per_device_train_batch_size", 0) or 0)
            if hasattr(training_config, "max_length")
            else 0
        ),
        ep_declared_max_length=int(getattr(training_config, "max_length", 0) or 0),
    )
    if allow_low_precision:
        kwargs.update(
            lowp_precision=dist_args.lowp_precision,
            lowp_apply_dense_mlp=dist_args.lowp_apply_dense_mlp,
            lowp_apply_moe_experts=dist_args.lowp_apply_moe_experts,
            lowp_keep_first_blocks=dist_args.lowp_keep_first_blocks,
            lowp_keep_last_blocks=dist_args.lowp_keep_last_blocks,
        )
    return ParallelismConfig(**kwargs)
