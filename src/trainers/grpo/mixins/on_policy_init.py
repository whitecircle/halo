"""Construction spine shared by the on-policy GRPO trainers (online RLVR and environmental).

Both constructors open the same way (resolve the config, turn off TRL's Liger flag, extract the distributed
kwargs, refuse the TRL options they cannot honor) and close the same way (parallel modes, reference and head-path
gates, weight-sync wiring, then dropout off); the closing order is load-bearing, so it lives here.
"""

import contextlib
from collections.abc import Iterator

import datasets
import torch
import trl.trainer.grpo_trainer as trl_grpo_trainer
from accelerate.logging import get_logger
from trl import GRPOTrainer
from trl.trainer.utils import disable_dropout_in_model

from src.model_loading.parallel.frozen_models import (
    ON_POLICY_GRPO_REFERENCE_ALTERNATIVES,
    on_policy_grpo_holds_reference,
    place_and_freeze,
)
from src.models.config_access import config_sources
from src.trainers.ctor_slots import get_ctor_arg, get_ctor_config, resolve_ctor_positions
from src.trainers.grpo.mixins.chunked_logprobs import LogitsWidth
from src.trainers.grpo.rollout.weight_push import reject_unsyncable_model
from src.trainers.training_args_overrides import disable_trl_liger

logger = get_logger(__name__, log_level="info")

# TRL GRPOTrainer positional slots, for ctor params arriving via *args — derived from the installed signature.
GRPO_CTOR_POSITIONS = resolve_ctor_positions(GRPOTrainer, "model", "args")
# The slot of the adapter config, which with the model decides whether TRL holds a KL reference.
_PEFT_CONFIG_CTOR_POSITIONS = resolve_ctor_positions(GRPOTrainer, "peft_config")
_DATASET_CTOR_POSITIONS = resolve_ctor_positions(GRPOTrainer, "train_dataset", "eval_dataset")


def disable_trl_liger_grpo_loss(training_args) -> None:
    """Keep TRL's ``use_liger_kernel`` off for the on-policy GRPO trainers.

    On GRPO the flag sends the log-probs through TRL's own chunked LM head, which reads the backbone
    directly and so skips the family's head transform and the toolkit's chunked sweep. Halo's toolkit
    default sets it True, so force it off before TRL caches it.
    """
    disable_trl_liger(
        training_args,
        "Disabling TRL's use_liger_kernel for GRPO: it sends the log-probs through TRL's chunked LM "
        "head, which skips the family's head transform. Model-level Liger kernels are still applied "
        "by load_distributed_model.",
    )


def _unread_reference(beta: float) -> ValueError:
    """The refusal of a ``ref_model`` the run never reads, raised before TRL's constructor and after it."""
    return ValueError(
        f"ref_model was passed, but this run holds no KL reference (beta={beta}, or a policy a PEFT adapter "
        "wraps, scored with the adapter disabled), so it would never be read. Drop ref_model."
    )


class OnPolicyGRPOInitMixin:
    """Open/close halves of an on-policy GRPO constructor. Mixed into the trainer."""

    POLICY_GRADIENT_LOSS = True
    SYNCS_TO_EXTERNAL_GENERATOR = True
    REFERENCE_ALTERNATIVES = ON_POLICY_GRPO_REFERENCE_ALTERNATIVES

    def _begin_on_policy_init(self, args: tuple, kwargs: dict) -> tuple[object, dict]:
        """Resolve the training config and extract the distributed kwargs; ``(config, kwargs)``.

        The config is resolved here, positionally or by keyword, because TRL's Liger flag has to
        be switched off on it before ``_init_distributed_config`` runs; the positionals are forwarded
        so that a positional model reaches the mixin's MoE gates too. ``save_completions`` is taken
        here for :class:`DecoupledCompletionsLogMixin` and ``ref_model`` for
        :meth:`_supplying_kl_reference`, since TRL's ctor would refuse either. A ``ref_model`` the run would
        never read is refused here, ahead of TRL's ctor, which on online GRPO opens the rollout client; the
        hand-off re-checks it against what TRL actually asks for. So are the TRL options the trainer cannot
        honor (:meth:`_refuse_unhonored_trl_options`), which read ``_use_chunked_grpo_logprobs``: each
        constructor sets it first.
        """
        self._save_completions = kwargs.pop("save_completions", True)
        self._supplied_ref_model = kwargs.pop("ref_model", None)
        training_args = get_ctor_config(args, kwargs, GRPO_CTOR_POSITIONS)
        if (
            self._supplied_ref_model is not None
            and training_args is not None
            and not on_policy_grpo_holds_reference(
                training_args.beta,
                get_ctor_arg(args, kwargs, "peft_config", _PEFT_CONFIG_CTOR_POSITIONS),
                get_ctor_arg(args, kwargs, "model", GRPO_CTOR_POSITIONS),
            )
        ):
            raise _unread_reference(training_args.beta)
        disable_trl_liger_grpo_loss(training_args)
        kwargs = self._init_distributed_config(
            kwargs, training_args=training_args, ctor_args=args, ctor_positions=GRPO_CTOR_POSITIONS
        )
        if training_args is not None:
            self._refuse_unhonored_trl_options(training_args, args, kwargs)
        return training_args, kwargs

    def _refuse_unhonored_trl_options(self, training_args, args: tuple, kwargs: dict) -> None:
        """Refuse the TRL GRPO options this trainer cannot honor.

        ``use_adaptive_entropy`` keeps its coefficient in a sidecar TRL restores only through the base
        optimizer load, which a sharded resume replaces. ``entropy_coef`` trains through the entropy's
        gradient, which the chunked sweep detaches. The DP-sharded loader's prompt-group sampler needs
        a dataset with a length, which an ``IterableDataset`` lacks.
        """
        if training_args.use_adaptive_entropy:
            raise ValueError(
                "use_adaptive_entropy is not supported: a sharded resume does not restore TRL's adaptive "
                "entropy coefficient. Set a fixed entropy_coef instead."
            )
        if training_args.entropy_coef != 0.0 and self._use_chunked_grpo_logprobs:
            raise ValueError(
                f"entropy_coef={training_args.entropy_coef} trains nothing with use_chunked_grpo_logprobs: the "
                "chunked sweep returns a detached entropy. Set entropy_coef: 0.0 or turn the chunked sweep off."
            )
        if not self._needs_custom_dataloader():
            return
        for name in ("train_dataset", "eval_dataset"):
            dataset = get_ctor_arg(args, kwargs, name, _DATASET_CTOR_POSITIONS)
            splits = dataset.values() if isinstance(dataset, dict) else [dataset]
            if any(
                isinstance(split, (datasets.IterableDataset, torch.utils.data.IterableDataset)) for split in splits
            ):
                raise ValueError(
                    f"{name} is an IterableDataset, which the DP-sharded loader this run uses cannot group "
                    "into prompt groups (its sampler needs a length). Load it as a map-style Dataset."
                )

    @contextlib.contextmanager
    def _supplying_kl_reference(self) -> Iterator[None]:
        """Run TRL's constructor with the frozen KL reference the caller loaded in place of the one TRL builds.

        TRL builds a reference at ``beta != 0`` on a policy no PEFT adapter wraps, by ``create_model_from_path``
        on the policy config's name: fp32, the hub's default revision, the default attention, no repair of the
        non-persistent buffers, no vocabulary resize. Here that call returns the ``ref_model`` passed
        (:func:`~src.model_loading.parallel.frozen_models.load_reference_model_for_on_policy_grpo`), placed and
        frozen, and TRL's dropout, head cast and ``prepare_model`` then run on it. TRL's own rule decides when
        it asks, so the two refusals follow that rule: a reference it asks for with none passed (raised before
        the rollout client connects), and a passed one it never asks for (the backstop of the up-front check in
        :meth:`_begin_on_policy_init`).
        """
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
        if supplied is not None and not handed:
            raise _unread_reference(self.args.beta)

    def _finish_on_policy_init(self) -> None:
        """Realize the parallel modes, gate the reference model, the chunked head path and the
        full-logits plane, gate and wire weight sync, disable dropout.

        Dropout goes last: it must reach the EP expert-LoRA dropout that ``_configure_distributed_modes``
        realizes, or the recomputed log-probs drift from the engine's dropout-free sampling. The plane
        is checked once the model is placed, so the free memory it is weighed against is real. The
        weight-sync gate refuses a model the trainer's engine (``_rollout_backend``) cannot take an
        update for here, not as an opaque server-side error at the first push.
        """
        self._configure_distributed_modes()
        self._validate_held_reference_model()
        self._resolve_chunked_head_transform()
        self._check_full_logits_fit(self._loss_logits_width())
        reject_unsyncable_model(self.model, self._rollout_backend)
        self._setup_weight_sync()
        self._disable_dropout_for_onpolicy()

    def _setup_weight_sync(self) -> None:
        """Install this trainer's engine weight sync. Nothing by default: a trainer whose rollout side owns
        its client (the environmental one) pushes through it."""

    def _loss_logits_width(self) -> LogitsWidth | None:
        """The completion logits row this trainer's loss forward carries and the setting that bounds it;
        ``None`` when no setting does."""
        raise NotImplementedError

    def _disable_dropout_for_onpolicy(self):
        """Force dropout off, after the parallel modes so it reaches the EP grouped expert-LoRA dropout. Idempotent.

        The recomputed log-probs must match vLLM's dropout-free sampling: active dropout lowers every
        recomputed log-prob (Jensen), biasing the importance ratio, the k3 KL estimator and the PPO ratio.
        """
        if not self.args.disable_dropout:
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
                if isinstance(val, (int, float)) and val > 0:
                    logger.warning(
                        "On-policy RL: model config %s=%s is a non-module dropout that cannot be zeroed by "
                        "disable_dropout_in_model — it perturbs recomputed log-probs vs vLLM's dropout-free "
                        "sampling (biases the importance ratio). Set it to 0 in the model config.",
                        field,
                        val,
                    )
