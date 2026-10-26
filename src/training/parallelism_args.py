"""Build a :class:`ParallelismConfig` from parsed ``DistributedArguments``."""

from dataclasses import fields
from typing import TYPE_CHECKING

from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.parallelism_rules import reject_cross_domain_determinism
from src.kernels.lowp.spec import LowpSpec
from src.models.context_window import is_bounded_length

if TYPE_CHECKING:
    from src.models.moe.expert_lora_spec import ExpertLoraSpec

# Rows one model call carries per dataset example, declared on the trainer class. Absent means 1,
# the multiplier for a trainer that does not concatenate.
FORWARD_ROWS_PER_EXAMPLE_ATTR = "_forward_rows_per_example"


def same_name_arg_forwards(dist_args) -> tuple[str, ...]:
    """``ParallelismConfig`` fields ``dist_args`` declares under the same name, in field order.

    The by-name half of :func:`parallelism_config_from_args`' wiring: derived from the two
    declarations rather than listed, so a knob added to both is forwarded automatically. Fields the
    builder renames (``expert_parallel_size`` → ``ep_size``) or derives do not appear here.
    """
    return tuple(f.name for f in fields(ParallelismConfig) if f.init and hasattr(dist_args, f.name))


def forward_rows_per_example(trainer_cls) -> int:
    """Sequence rows one model call carries per dataset example, per ``trainer_cls``.

    The preference and Bradley-Terry reward trainers score chosen and rejected in one concatenated
    forward (``torch.cat([chosen, rejected], dim=0)``), so their MoE layers present twice the rows
    ``per_device_train_batch_size`` names, and the DeepEP dispatch ceiling is judged against that.
    A trainer that declares nothing runs one row per example.
    """
    return max(1, int(getattr(trainer_cls, FORWARD_ROWS_PER_EXAMPLE_ATTR, 1) or 1))


def parallelism_config_from_args(
    dist_args,
    *,
    trainer_cls,
    training_config=None,
    allow_low_precision: bool = False,
    supports_init_from_scratch: bool = False,
    expert_lora: "ExpertLoraSpec | None" = None,
) -> ParallelismConfig:
    """Build a :class:`ParallelismConfig` from parsed ``DistributedArguments`` — shared EP/CP/TP/ETP/PP
    wiring for every training entry script.

    Args:
        dist_args: parsed ``DistributedArguments``.
        training_config: the parsed TRL/HF training config, read for the run's declared per-rank
            token budget (``rows × per_device_train_batch_size × max_length``) — what
            :meth:`ParallelismConfig.validate_against_model_config` judges against DeepEP's dispatch
            ceilings. ``None``, or a config declaring no ``max_length``, leaves that gate off. Its
            ``full_determinism`` is refused here on an EP group spanning NVLink domains.
        trainer_cls: the trainer class this script builds. Its ``_supports_cp`` / ``_supports_pp``
            gates reject a ``context_parallel_size`` / ``pipeline_parallel_size`` above 1 here, at
            config time: the trainer's own check would reject it too, but only after the model (and
            for distillation/GRPO scripts a second model or a vLLM probe) has already loaded. Also
            read for :data:`FORWARD_ROWS_PER_EXAMPLE_ATTR`, the number of sequence rows one model
            call carries per dataset example. The preference and Bradley-Terry reward trainers score
            chosen and rejected in a single concatenated forward, so their MoE layers present twice
            the rows a token budget computed from ``per_device_train_batch_size`` alone would predict.
        allow_low_precision: accept the ``lowp_*`` knobs (SFT only); elsewhere a non-default one is
            rejected.
        expert_lora: peeled native EP expert-LoRA spec. Must be passed here, not assigned onto the
            returned config: only construction runs ``__post_init__``, where its PP and expert-TP
            refusals live, and :meth:`ParallelismConfig.ep_config` caches the ``EPConfig`` it
            builds, so a later assignment never reaches the layers.
    """
    lowp = LowpSpec(**{f.name: getattr(dist_args, f"lowp_{f.name}") for f in fields(LowpSpec)})
    if not allow_low_precision:
        if lowp.precision != "bf16":
            raise ValueError(
                f"lowp_precision={lowp.precision!r} is only validated for SFT training; this "
                "trainer runs in bf16. Remove lowp_precision from the config (or use the SFT script)."
            )
        # A config setting the other lowp_* keys outside SFT would parse cleanly and have no effect.
        ignored = lowp.non_default_shape_fields()
        if ignored:
            raise ValueError(
                f"{', '.join(ignored)} apply only to SFT training and would be silently dropped here. "
                "Remove them from the config, or run the SFT script."
            )

    # Only SFT threads init_from_scratch into the loader; elsewhere the run would fine-tune pretrained weights.
    if not supports_init_from_scratch and dist_args.init_from_scratch:
        raise ValueError(
            "init_from_scratch is only supported by the SFT script (scripts/training/sft.py); "
            "this script would silently ignore it and train pretrained weights."
        )

    if not trainer_cls._supports_cp and dist_args.context_parallel_size > 1:
        raise ValueError(
            f"{trainer_cls.__name__} does not support Context Parallelism (CP). Use Expert Parallelism "
            "(EP) and/or Tensor Parallelism (TP) instead, or remove --context_parallel_size."
        )

    if not trainer_cls._supports_pp and dist_args.pipeline_parallel_size > 1:
        raise ValueError(
            f"{trainer_cls.__name__} does not support Pipeline Parallelism (PP). Use Expert Parallelism "
            "(EP) and/or Tensor Parallelism (TP) instead, or remove --pipeline_parallel_size."
        )

    # Fields the two dataclasses spell identically forward themselves; only the renames and the
    # derivations stay explicit.
    kwargs = {name: getattr(dist_args, name) for name in same_name_arg_forwards(dist_args)}
    kwargs.update(
        ep_size=dist_args.expert_parallel_size,
        cp_size=dist_args.context_parallel_size,
        tp_size=dist_args.tensor_parallel_size,
        expert_tp_size=dist_args.expert_tensor_parallel_size,
        pp_size=dist_args.pipeline_parallel_size,
        pp_schedule=dist_args.pipeline_schedule,
        pp_microbatches=dist_args.pipeline_microbatches,
        pp_split=dist_args.pipeline_split,
        ep_fp32_router=dist_args.fp32_router,
        ep_fp32_experts=dist_args.fp32_experts,
        expert_lora=expert_lora,
        lowp=lowp,
        # The two factors stay separate because only one of them is knowable here: a config with no
        # max_length field at all (generation-shaped budgets) declares neither rows nor budget, while
        # ``max_length: null`` declares its rows and leaves the length to the gate's context-window
        # resolution unless a bounded prompt + completion budget caps each row.
        ep_rows_per_device=(
            forward_rows_per_example(trainer_cls)
            * int(getattr(training_config, "per_device_train_batch_size", 0) or 0)
            if hasattr(training_config, "max_length")
            else 0
        ),
        ep_declared_max_length=declared_row_length(training_config),
    )
    config = ParallelismConfig(**kwargs)
    reject_cross_domain_determinism(config, bool(getattr(training_config, "full_determinism", False)))
    return config


def declared_row_length(training_config) -> int:
    """The token cap a config puts on one training row, or 0 when it declares none.

    ``max_length`` where set; otherwise a bounded prompt + completion budget, which the trainers that
    carry both (offline GRPO) truncate each row to. 0 leaves the EP capacity gate to the model's
    context window.
    """
    max_length = getattr(training_config, "max_length", None)
    if is_bounded_length(max_length):
        return int(max_length)
    prompt = getattr(training_config, "max_prompt_length", None)
    completion = getattr(training_config, "max_completion_length", None)
    if is_bounded_length(prompt) and is_bounded_length(completion):
        return int(prompt) + int(completion)
    return 0
