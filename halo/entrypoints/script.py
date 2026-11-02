"""The training script lifecycle: parse the command line, set the run up, assemble its callbacks, train,
and tear the process group down."""

import asyncio
import functools
import inspect
import logging
from collections.abc import Callable, Sequence
from datetime import timedelta
from typing import Any, NamedTuple

import torch
import torch.distributed as dist
from huggingface_hub import constants as hub_constants
from transformers import TrainingArguments

from halo.callbacks.balancing import configure_moe_balancing
from halo.callbacks.early_stop import StopsTrainingEarly
from halo.callbacks.factory import build_run_callbacks, reorder_integration_callbacks_last
from halo.config_validation import differs_from_default
from halo.configs.parallelism_arguments import parallelism_config_from_args
from halo.configs.parser import HaloArgumentParser, expand_strftime_directives
from halo.data.pipeline.coordinated_map import datasets_cache_dir
from halo.distributed.expert_parallel.deepep.dispatcher import destroy_all_dispatchers
from halo.distributed.parallelism_config import ParallelismConfig
from halo.distributed.rank_uniform_env import agree_rank_uniform_env
from halo.entrypoints.run_setup import configure_training_run, install_log_tee, prepare_distributed_resume
from halo.env import nccl_timeout
from halo.log import configure_training_logging
from halo.model_loading.parallel.peft_setup import split_expert_lora_targets
from halo.runtime.collectives import broadcast_from_rank0
from halo.runtime.consensus import raise_rank0_failure
from halo.runtime.filesystem import verify_input_filesystem_sharing, verify_output_filesystem_sharing
from halo.runtime.process_group import init_distributed
from halo.runtime.ranks import get_local_rank, has_process_group, is_global_main_process

logger = logging.getLogger(__name__)


class EarlyStopExit(SystemExit):
    """A run an early stop ended: it exits non-zero after the teardown a finished run gets, since the
    stop's verdict is rank-uniform and every peer leaves training at the same step."""


class ScriptContext(NamedTuple):
    """What :func:`init_training_script` resolves for a training script."""

    parallelism_config: ParallelismConfig
    local_rank: int
    resume_checkpoint: str | None
    model_source: str
    trainer_cls: type

    @property
    def mode_suffix(self) -> str:
        """The parallelism mode the run is named and logged by."""
        return self.parallelism_config.mode_string or "standard"

    @property
    def policy_from_checkpoint(self) -> bool:
        """Whether the policy loads its weights from the resume checkpoint (a Path-B resume)."""
        return self.resume_checkpoint is not None and self.model_source == self.resume_checkpoint


def _reject_ddp_timeout(parsed: Sequence[Any]) -> None:
    """Refuse a ``ddp_timeout`` set away from its default: the process group is up before any config is built,
    its NCCL watchdog set by ``DIST_NCCL_TIMEOUT_MINUTES``, so the field would never apply."""
    for args in parsed:
        if isinstance(args, TrainingArguments) and differs_from_default(args, "ddp_timeout"):
            minutes = nccl_timeout() // timedelta(minutes=1)
            raise ValueError(
                f"ddp_timeout={args.ddp_timeout} would never apply: the process group starts before the config is "
                f"parsed, with the NCCL watchdog DIST_NCCL_TIMEOUT_MINUTES sets ({minutes} min here). Remove "
                f"ddp_timeout and set DIST_NCCL_TIMEOUT_MINUTES in the launch environment, alike on every node."
            )


def parse_script_args(*dataclass_types: type) -> Any:
    """COLLECTIVE: raise the root logger to INFO, probe the HF caches against the declared input sharing, parse
    the command line into ``dataclass_types``, returned in order (the instance alone for one type), and tee the
    console into the run's ``output_dir``. A ``ddp_timeout`` the parsed config sets is refused.

    The process group comes up before any dataclass is built: constructing a TrainingArguments touches
    ``self.device``, which makes accelerate initialize the default process group without ``device_id``,
    costing a ``new_group`` per mesh dim instead of one ``ncclCommSplit`` and leaving every barrier to
    infer its device.
    """
    configure_training_logging()
    init_distributed()
    # Before the first Hub read: a follower reads only the cache its scope's main rank fills.
    verify_input_filesystem_sharing(
        {"HF_HUB_CACHE": hub_constants.HF_HUB_CACHE, "HF_DATASETS_CACHE": datasets_cache_dir()}
    )
    parser = HaloArgumentParser(dataclass_types)
    values = parser.load_values()
    if isinstance(values.get("output_dir"), str):
        # Rank 0's expansion: a per-rank datetime.now() can straddle a second boundary, and on a
        # non-shared output filesystem each node's save rank would then tee run.log elsewhere.
        values["output_dir"] = broadcast_from_rank0(expand_strftime_directives(values["output_dir"]))
    output = parser.parse_dict(values)
    _reject_ddp_timeout(output)
    install_log_tee(
        next((obj.output_dir for obj in output if isinstance(getattr(obj, "output_dir", None), str)), None)
    )
    return output[0] if len(output) == 1 else output


def _pin_single_process_to_bound_gpu(training_config) -> None:
    """Keep a process on the one GPU it bound.

    With several GPUs visible HF wraps the model in ``nn.DataParallel``, which the token-normalized
    losses are not written for: the logged loss and the gradients scale with the GPU count.
    """
    visible = training_config.n_gpu
    if visible <= 1:
        return
    if is_global_main_process():
        logger.warning(
            f"{visible} GPUs are visible to this process; it trains on the one it bound. A multi-GPU run "
            f"takes one process per GPU: `halo launch <method> <config> -n <gpus>` (torchrun)."
        )
    training_config._n_gpu = 1


def init_training_script(
    args,
    training_config,
    model_config,
    dist_args,
    *,
    script_prefix: str,
    trainer_cls: type,
    split_expert_lora: bool = True,
    supports_init_from_scratch: bool = False,
) -> ScriptContext:
    """The pre-model phase of a training script, after :func:`parse_script_args` brought the process group
    up, in the order every script depends on: the device bind, the ``ParallelismConfig``, the run setup
    (named ``{script_prefix}-{mode}``) and the resume resolution.

    ``trainer_cls`` supplies the CP/PP/low-precision gates, so a mode the trainer refuses fails before any
    model, reference or rollout probe loads. ``split_expert_lora`` peels MoE expert targets out of
    ``lora_target_modules`` into native EP grouped LoRA; only SFT passes ``supports_init_from_scratch``.
    """
    local_rank = get_local_rank()
    if torch.cuda.is_available():  # CUDA-less interpreters (config-parse checks, CPU boxes) stay usable
        torch.cuda.set_device(local_rank)
    _pin_single_process_to_bound_gpu(training_config)

    # Before the weight load (a check placed after it would be the first collective a straggler
    # misses, and the watchdog would then time out here) and after the device bind (the gather
    # runs over NCCL).
    agree_rank_uniform_env()

    # Peeled before the config is built: __post_init__ validates expert_lora (PP rejects it) and
    # ep_config caches its EPConfig, so a later assignment would not reach the layers.
    parallelism_config = parallelism_config_from_args(
        dist_args,
        training_config=training_config,
        trainer_cls=trainer_cls,
        expert_lora=split_expert_lora_targets(model_config) if split_expert_lora else None,
        supports_init_from_scratch=supports_init_from_scratch,
    )

    # Before the resume resolves, the run reads its weights from the base.
    context = ScriptContext(parallelism_config, local_rank, None, model_config.model_name_or_path, trainer_cls)
    configure_training_run(
        args,
        training_config,
        f"{script_prefix}-{context.mode_suffix}",
        overwrite_output_dir=args.overwrite_output_dir,
    )
    # output_dir is final and still unwritten: the last point at which a mis-declared
    # DIST_*_SHARED_FILESYSTEM surfaces as a startup error rather than as a resume that restarts
    # every non-zero node at global_step=0.
    verify_output_filesystem_sharing(training_config.output_dir)
    resume_checkpoint, model_source = prepare_distributed_resume(
        training_config, model_config, parallelism_config, overwrite_output_dir=args.overwrite_output_dir
    )
    return context._replace(resume_checkpoint=resume_checkpoint, model_source=model_source)


def distributed_trainer_kwargs(args, parallelism_config: ParallelismConfig, *, dataset_presharded: bool):
    """The constructor kwargs every ``DistributedTrainerMixin`` trainer takes, all together: a call site
    missing one gathers a save the config asked to shard, re-shards a per-rank dataset, or drops the
    checkpoint knobs or the non-finite step limit."""
    return {
        "parallelism_config": parallelism_config,
        "moe_balancing": args.moe_balancing,
        "save_sharded_ep": args.save_sharded_ep,
        "save_max_shard_size": args.save_max_shard_size,
        "allow_optimizer_warm_restart": args.allow_optimizer_warm_restart,
        "max_consecutive_nonfinite_steps": args.max_consecutive_nonfinite_steps,
        "dataset_presharded": dataset_presharded,
    }


def configure_balancing_and_build_callbacks(
    args,
    training_config,
    model,
    context: ScriptContext,
    *,
    generate_callback=None,
    max_seq_len: int | None = None,
) -> list:
    """COLLECTIVE. Configure the run's MoE balancing on ``model`` for ``context.trainer_cls``'s loss and
    weight-sync capabilities, then build its callback roster for that mode."""
    balancing_mode = configure_moe_balancing(
        args,
        model,
        context.parallelism_config,
        policy_gradient_loss=context.trainer_cls.POLICY_GRADIENT_LOSS,
        syncs_to_external_generator=context.trainer_cls.SYNCS_TO_EXTERNAL_GENERATOR,
    )
    return build_run_callbacks(
        args,
        training_config,
        model,
        context.parallelism_config,
        balancing_mode=balancing_mode,
        generate_callback=generate_callback,
        max_seq_len=max_seq_len,
    )


def run_trainer(
    trainer,
    context: ScriptContext,
    *,
    method_name: str,
    extra_start_log: Sequence[str] | None = None,
) -> None:
    """Train, between the start log (mode, axis sizes, ``extra_start_log``) and the EP cleanup; a run
    the early stop ended raises :class:`EarlyStopExit` on every rank."""
    # Integrations add themselves at the HEAD and would consume `logs` before our callbacks' on_log.
    reorder_integration_callbacks_last(trainer)

    parallelism_config = context.parallelism_config
    if is_global_main_process():
        logger.info(f"Starting {method_name} training (mode: {context.mode_suffix})...")
        if parallelism_config.has_ep_group:
            logger.info(f"  EP size: {parallelism_config.ep_size}")
        if parallelism_config.is_cp_mode:
            logger.info(f"  CP size: {parallelism_config.cp_size}")
        if parallelism_config.is_tp_mode:
            logger.info(f"  TP size: {parallelism_config.tp_size}")
        if parallelism_config.is_expert_tp_mode:
            logger.info(f"  ETP size: {parallelism_config.expert_tp_size}")
        logger.info(f"  DP size: {parallelism_config.data_parallel_size}")
        for line in extra_start_log or ():
            logger.info(f"  {line}")

    trainer.train(resume_from_checkpoint=context.resume_checkpoint)

    trainer.cleanup_ep()

    state = trainer.state
    # An early stop fails the run, so a scheduler or a chained stage does not take its output for a
    # finished one.
    if any(
        isinstance(callback, StopsTrainingEarly) and callback.stopped
        for callback in trainer.callback_handler.callbacks
    ):
        raise EarlyStopExit(f"{method_name} training stopped early at step {state.global_step} of {state.max_steps}.")
    if is_global_main_process():
        # A batch sampler that yields fewer batches than its length also ends a run short of the plan:
        # ``no_duplicates`` under ``dataloader_drop_last`` drops the partial batches its duplicates leave.
        if state.global_step < state.max_steps:
            logger.warning(f"{method_name} training ended at step {state.global_step} of {state.max_steps}.")
        else:
            logger.info(f"{method_name} training completed successfully!")


def _tear_down_distributed() -> None:
    """COLLECTIVE: destroy the EP dispatchers, then the process group."""
    destroy_all_dispatchers()
    if has_process_group():
        dist.destroy_process_group()


def with_distributed_teardown(main_fn):
    """``main_fn`` with the distributed teardown run after a success or an :class:`EarlyStopExit`;
    used as ``with_distributed_teardown(main)()``.

    Dispatchers go before the process group: a DeepEP Gin buffer outliving the group communicator faults
    with a sticky ``cudaErrorIllegalAddress`` that hides the original traceback. Any other exception
    propagates with no teardown: both halves are collectives, and a rank failing mid-step has peers still
    inside the step's own collective, so entering another parks it in the NCCL watchdog for the full
    timeout while torchrun sees no exit to reap the peers on.
    """

    @functools.wraps(main_fn)
    def wrapper(*args, **kwargs):
        try:
            result = main_fn(*args, **kwargs)
        except EarlyStopExit:
            _tear_down_distributed()
            raise
        _tear_down_distributed()
        return result

    return wrapper


def verify_backend_on_rank0(probe: Callable[[], object], what: str) -> None:
    """COLLECTIVE: probe an external backend once on global rank 0 and raise its failure on every rank.

    Every rank calls it behind a rank-uniform gate; the backend takes one request rather than one per
    rank, and the ranks raise together instead of the peers waiting in the next collective. A ``probe``
    returning a coroutine is run to completion. The ``RuntimeError`` reads ``"<what> probe failed: <error>"``.
    """

    def run_probe() -> None:
        outcome = probe()
        if inspect.iscoroutine(outcome):
            asyncio.run(outcome)

    raise_rank0_failure(run_probe, lambda e: f"{what} probe failed: {e}")
