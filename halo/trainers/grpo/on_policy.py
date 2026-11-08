"""The construction spine and generation-round geometry of the on-policy GRPO trainers. The init mixin's open and close
run in one load-bearing order; :class:`GenerationRoundMixin` fits TRL's loader, sampler and buffer to the DP-sharded
loader."""

import contextlib
import logging
from collections.abc import Iterator

import trl.trainer.grpo_trainer as trl_grpo_trainer
from torch.utils.data import Dataset, Sampler
from trl import GRPOTrainer
from trl.trainer.utils import RepeatSampler, disable_dropout_in_model

from halo.model_loading.parallel.frozen_models import (
    ON_POLICY_GRPO_REFERENCE_ALTERNATIVES,
    on_policy_grpo_holds_reference,
    place_and_freeze,
)
from halo.models.fp8_rollout import RolloutWeightPrecision
from halo.models.introspection.config_access import config_sources
from halo.runtime.ranks import is_global_main_process
from halo.trainers.construction.ctor_slots import get_ctor_arg, get_ctor_config, resolve_ctor_positions
from halo.trainers.construction.training_args_overrides import disable_trl_liger
from halo.trainers.grpo.grpo_rules import reject_unhonored_trl_options, reject_unread_reference
from halo.trainers.grpo.rollout.weight_push import reject_unsyncable_model

logger = logging.getLogger(__name__)

GRPO_CTOR_POSITIONS = resolve_ctor_positions(GRPOTrainer, "model", "args")
# The slot of the adapter config, which with the model decides whether TRL holds a KL reference.
_PEFT_CONFIG_CTOR_POSITIONS = resolve_ctor_positions(GRPOTrainer, "peft_config")


class GenerationRoundMixin:
    """Mixed in ahead of ``DistributedTrainerMixin`` so ``super()`` reaches TRL's."""

    def _train_loader_batch_size(self) -> int:
        """``steps_per_generation`` micro-batches are drawn together."""
        return self._train_batch_size * self.args.steps_per_generation

    def _get_train_sampler(self, dataset: Dataset | None = None) -> Sampler:
        """TRL's ``generation_batch_size`` geometry is world-rate, while this loader consumes ``data_parallel_size``
        batches per fetch, so each round would re-roll only the leading ``dp/world`` fraction of a prompt block."""
        if not self._needs_custom_dataloader():
            return super()._get_train_sampler(dataset)
        if dataset is None:
            dataset = self.train_dataset
        consumers = 1 if self._dataset_presharded else self.parallelism_config.data_parallel_size
        per_rank_rows = self._train_loader_batch_size()
        completions_per_round = per_rank_rows * consumers
        if completions_per_round % self.num_generations != 0:
            raise ValueError(
                f"The DP-rate generation batch ({completions_per_round} = per_device_train_batch_size "
                f"({self._train_batch_size}) * steps_per_generation ({self.args.steps_per_generation}) "
                f"* {'1 (pre-sharded per DP rank)' if self._dataset_presharded else f'data_parallel_size ({consumers})'}) "
                f"must be divisible by num_generations ({self.num_generations}), so every generation "
                f"round covers whole prompt groups."
            )
        if self.accelerator.num_processes != consumers and per_rank_rows % self.num_generations != 0:
            # TP/ETP siblings duplicate their DP slice into TRL's world-order gather; only whole groups regroup.
            raise ValueError(
                f"With TP/ETP active, the per-rank generation rows (per_device_train_batch_size "
                f"({self._train_batch_size}) * steps_per_generation ({self.args.steps_per_generation}) "
                f"= {per_rank_rows}) must be divisible by num_generations ({self.num_generations}): "
                f"a prompt group spanning DP ranks breaks TRL's world-gathered advantage grouping "
                f"once TP siblings inject duplicate blocks."
            )
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=completions_per_round // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=self.args.seed,
        )

    @property
    def _misaligned_accumulation(self) -> bool:
        """``gradient_accumulation_steps`` straddling TRL's buffer refill, so one optimizer step consumes micro-batches
        of two policies and TRL recomputes the old log-probs."""
        generate_every = self.args.steps_per_generation * self.num_iterations
        return self.args.gradient_accumulation_steps % generate_every != 0

    def _prepare_inputs(self, generation_batch):
        """Re-sync the micro-batch across the TP/ETP group: TRL shuffles the buffer with ``torch.randperm`` on the
        per-rank-seeded CPU generator, so TP siblings would all-reduce partial activations of different samples."""
        inputs = super()._prepare_inputs(generation_batch)
        return self._broadcast_tensors_from_tp_leader(inputs)


class OnPolicyGRPOInitMixin:
    POLICY_GRADIENT_LOSS = True
    SYNCS_TO_EXTERNAL_GENERATOR = True
    REFERENCE_ALTERNATIVES = ON_POLICY_GRPO_REFERENCE_ALTERNATIVES
    GENERATION_SHAPED_ROWS = True
    # What the weight push stores on the rollout servers; the trainer's own weights stay bf16.
    _rollout_weight_precision: RolloutWeightPrecision = "bf16"

    def _begin_on_policy_init(self, args: tuple, kwargs: dict) -> tuple[object, dict]:
        """Returns ``(config, kwargs)``. TRL's Liger flag must go off before ``_init_distributed_config``, and
        ``save_completions`` and ``ref_model`` are taken since TRL's ctor would refuse either. Refusals run ahead of
        TRL's ctor, which on online GRPO opens the rollout client."""
        self._save_completions = kwargs.pop("save_completions", True)
        self._supplied_ref_model = kwargs.pop("ref_model", None)
        training_args = get_ctor_config(args, kwargs, GRPO_CTOR_POSITIONS)
        if self._supplied_ref_model is not None and training_args is not None:
            reject_unread_reference(
                self._supplied_ref_model,
                training_args.beta,
                read=on_policy_grpo_holds_reference(
                    training_args.beta,
                    get_ctor_arg(args, kwargs, "peft_config", _PEFT_CONFIG_CTOR_POSITIONS),
                    get_ctor_arg(args, kwargs, "model", GRPO_CTOR_POSITIONS),
                ),
            )
        # The toolkit default turns it on, and on GRPO it routes the log-probs past the family's head transform.
        disable_trl_liger(
            training_args,
            "Disabling TRL's use_liger_kernel for GRPO: it sends the log-probs through TRL's chunked LM "
            "head, which skips the family's head transform. Model-level Liger kernels are still applied "
            "by load_distributed_model.",
        )
        kwargs = self._init_distributed_config(
            kwargs, training_args=training_args, ctor_args=args, ctor_positions=GRPO_CTOR_POSITIONS
        )
        if training_args is not None:
            reject_unhonored_trl_options(
                training_args,
                use_chunked_grpo_logprobs=self._use_chunked_grpo_logprobs,
                needs_custom_dataloader=self._needs_custom_dataloader(),
                args=args,
                kwargs=kwargs,
            )
        return training_args, kwargs

    @contextlib.contextmanager
    def _supplying_kl_reference(self) -> Iterator[None]:
        """Hand TRL's constructor the caller's frozen KL reference in place of its own ``create_model_from_path`` build
        (fp32, default revision and attention, no buffer repair or resize). Refuses one asked for with none passed and
        a passed one never asked for."""
        supplied, self._supplied_ref_model = self._supplied_ref_model, None
        handed = False
        build = trl_grpo_trainer.create_model_from_path

        def hand_over(model_id, *args, **kwargs):
            nonlocal handed
            # TRL builds a policy passed by name with the same call, before the trainer holds a model.
            if getattr(self, "model", None) is None:
                return build(model_id, *args, **kwargs)
            if supplied is None:
                raise ValueError(
                    f"beta={self.args.beta} on a policy no PEFT adapter wraps holds a frozen KL reference, and no "
                    f"ref_model was passed: TRL would build {model_id!r} itself in fp32, from the hub's default "
                    "revision, with the default attention and no repair of its non-persistent buffers, biasing the "
                    "KL against the policy. Pass ref_model=load_reference_model_for_on_policy_grpo(...), as the "
                    "GRPO training scripts do. Or hold none: "
                    f"{self.REFERENCE_ALTERNATIVES.under(self.parallelism_config)}"
                )
            handed = True
            place_and_freeze(supplied, self.model)
            return supplied

        trl_grpo_trainer.create_model_from_path = hand_over
        try:
            yield
        finally:
            trl_grpo_trainer.create_model_from_path = build
        reject_unread_reference(supplied, self.args.beta, read=handed)

    def _finish_on_policy_init(self) -> None:
        """Dropout goes last to reach the EP expert-LoRA dropout; the full-logits plane is weighed once the model is
        placed; the sync gate refuses a model the engine cannot update here rather than at the first push."""
        self._configure_distributed_modes()
        self._validate_held_reference_model()
        self._reject_unreproducible_chunked_head()
        self._verify_full_logits_fit(self._loss_logits_width())
        reject_unsyncable_model(self.model, self._rollout_backend, self._rollout_weight_precision)
        self._install_weight_sync()
        self._disable_dropout_for_onpolicy()

    def _install_weight_sync(self) -> None:
        """No-op by default: a trainer whose rollout side owns its client (the environmental one) pushes through it."""

    def _disable_dropout_for_onpolicy(self):
        """Recomputed log-probs must match vLLM's dropout-free sampling: active dropout lowers every recomputed
        log-prob (Jensen), biasing the importance ratio, the k3 KL estimator and the PPO ratio. Idempotent."""
        if not self.args.disable_dropout:
            if is_global_main_process():
                logger.info("On-policy RL: disabling dropout for importance-sampling / vLLM consistency.")
            self.args.disable_dropout = True
        disable_dropout_in_model(self.model)
        if getattr(self, "ref_model", None) is not None:
            disable_dropout_in_model(self.ref_model)

        # A config-float dropout inside attention/MLP survives disable_dropout_in_model and still perturbs log-probs.
        config = getattr(self.model, "config", None)
        if config is None:
            return
        for cfg in config_sources(config):
            for field in ("attention_dropout", "attn_pdrop", "hidden_dropout", "resid_pdrop", "dropout"):
                val = getattr(cfg, field, 0) or 0
                if isinstance(val, (int, float)) and val > 0 and is_global_main_process():
                    logger.warning(
                        "On-policy RL: model config %s=%s is a non-module dropout that cannot be zeroed by "
                        "disable_dropout_in_model — it perturbs recomputed log-probs vs vLLM's dropout-free "
                        "sampling (biases the importance ratio). Set it to 0 in the model config.",
                        field,
                        val,
                    )
