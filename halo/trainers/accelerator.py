"""A plain accelerator on the mixin-managed path, the base Trainer's when accelerate owns the wrap."""

import logging
import os

import torch
from accelerate import Accelerator
from accelerate.utils import DataLoaderConfiguration

from halo.config_validation import differs_from_default
from halo.env import is_accelerate_fsdp_launch, is_accelerate_launch
from halo.runtime.ranks import get_global_world_size, is_global_main_process
from halo.trainers.trainer_rules import FSDP_SHAPING_KNOBS

logger = logging.getLogger(__name__)


def accelerate_manages_fsdp(config) -> bool:
    return is_accelerate_fsdp_launch() and not config.has_custom_parallelism


def accelerate_manages_ddp(config) -> bool:
    return is_accelerate_launch() and not is_accelerate_fsdp_launch() and not config.has_custom_parallelism


def build_plain_accelerator(args) -> Accelerator:
    """An Accelerator without DDP: on the mixin-managed path FSDP2 syncs the gradients."""
    accelerator_config = args.accelerator_config.to_dict()
    dataloader_params = ["split_batches", "dispatch_batches", "even_batches", "use_seedable_sampler"]
    dataloader_config = DataLoaderConfiguration(
        **{param: accelerator_config.pop(param) for param in dataloader_params},
        data_seed=args.data_seed,
        non_blocking=accelerator_config.pop("non_blocking", False),
    )
    kwargs = {
        "gradient_accumulation_steps": 1,  # Trainer drives accumulation
        "dataloader_config": dataloader_config,
    }
    if args.bf16:
        kwargs["mixed_precision"] = "bf16"
    elif args.fp16:
        kwargs["mixed_precision"] = "fp16"
    return Accelerator(**kwargs)


def no_wrap_prepare(original_prepare):
    """A ``prepare`` that marks modules, optimizers and schedulers prepared without wrapping them."""

    def prepare(*objects):
        prepared = []
        for obj in objects:
            if isinstance(obj, (torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler)):
                obj._is_accelerate_prepared = True
                prepared.append(obj)
            elif obj is None:
                prepared.append(None)
            else:
                prepared.append(original_prepare(obj))
        return prepared[0] if len(prepared) == 1 else tuple(prepared)

    return prepare


class AcceleratorMixin:
    @property
    def _accelerate_manages_wrap(self) -> bool:
        return self._accelerate_manages_fsdp or self._accelerate_manages_ddp

    @property
    def _mixin_manages_ep_layers(self) -> bool:
        """``needs_ep_layers`` holds by default even for dense models, and an accelerate launch keeps its own DDP/FSDP
        choice."""
        return self.parallelism_config.needs_ep_layers and not self._accelerate_manages_wrap

    def create_accelerator_and_postprocess(self):
        if not self._needs_custom_accelerator():
            super().create_accelerator_and_postprocess()
            self._warn_knobs_unapplied_under_accelerate()
        else:
            # torch.compile is applied manually after FSDP wrapping (_apply_torch_compile).
            compile_keys = ["ACCELERATE_DYNAMO_BACKEND", "ACCELERATE_DYNAMO_MODE"]
            env_backup = {k: os.environ.pop(k) for k in compile_keys if k in os.environ}
            try:
                self.accelerator = build_plain_accelerator(self.args)
            finally:
                os.environ.update(env_backup)
            self.gather_function = self.accelerator.gather_for_metrics
            logging.getLogger("accelerate.accelerator").setLevel(logging.WARNING)
            self.is_deepspeed_enabled = False
            # Runs before any FSDP2 wrap; the checkpoint path relies on _fsdp_wrapped being False here.
            self.is_fsdp_enabled = False
            if is_global_main_process():
                logger.info(f"Created Accelerator with distributed_type: {self.accelerator.state.distributed_type}")

        # In both branches, since the fp32 logits upcast costs the same under an accelerate-managed launch.
        self._disable_native_amp_logits_upcast()

    def _needs_custom_accelerator(self) -> bool:
        config = self.parallelism_config
        has_parallelism = (
            config.has_ep_group or config.is_cp_mode or config.is_tp_mode or self._mixin_manages_ep_layers
        )
        return has_parallelism or (get_global_world_size() > 1 and not self._accelerate_manages_wrap)

    def _warn_knobs_unapplied_under_accelerate(self) -> None:
        """Warn about knobs only the mixin's own FSDP2 wrap implements, since accelerate's plugin replaces it here."""
        config = self.parallelism_config
        requested = [n for n in (*FSDP_SHAPING_KNOBS, "fp32_grad_reduce") if differs_from_default(config, n)]
        if not requested or not is_global_main_process():
            return
        logger.warning(
            f"These parallelism knobs are IGNORED under 'accelerate launch' (accelerate owns the "
            f"FSDP/DDP wrap, so the mixin never applies them): {sorted(requested)}. Launch with "
            f"torchrun to get them, or drop them from the config."
        )

    def _disable_native_amp_logits_upcast(self) -> None:
        """Drop ``native_amp``'s upcast of the full ``[B, S, V]`` logits. Under fp16 it stays, since it also gates
        ``unscale_gradients`` and clearing it would clip and step on GradScaler-scaled gradients."""
        if self.args.fp16:
            return
        self.accelerator.native_amp = False
        if is_global_main_process():
            logger.info("Disabled native_amp to prevent fp32 output conversion (avoids OOM for long sequences)")

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        """Skip DDP wrapping where FSDP2, the QLoRA sweep, the EP layers, TP, ETP or PP sync the gradients."""
        config = self.parallelism_config
        if not (
            self._fsdp_wrapped
            or self._qlora_grad_sync
            or self._mixin_manages_ep_layers
            or config.is_tp_mode
            or config.is_expert_tp_mode
            or config.is_pp_mode
        ):
            return super()._inner_training_loop(batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval)

        original_prepare = self.accelerator.prepare
        self.accelerator.prepare = no_wrap_prepare(original_prepare)
        try:
            return super()._inner_training_loop(batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval)
        finally:
            self.accelerator.prepare = original_prepare
