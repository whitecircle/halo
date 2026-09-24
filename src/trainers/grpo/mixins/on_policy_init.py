"""Construction spine shared by the on-policy GRPO trainers (online RLVR and environmental).

Both constructors open the same way (resolve a config that may have arrived positionally, disable
TRL's Liger GRPO loss, extract the distributed kwargs) and close the same way: realize the parallel
modes, gate the implicit reference model and the chunked sweep's head path, wire weight sync, then
disable dropout. The closing order is load-bearing, so it is defined here rather than in each
constructor.
"""

from trl import GRPOTrainer

from src.trainers.grpo.mixins.chunked_logprobs import LogitsWidth
from src.trainers.mixins.validation import ctor_config, ctor_positions, disable_trl_liger

# TRL GRPOTrainer positional slots, for ctor params arriving via *args — derived from the installed signature.
# Shared with SDPG, whose *args reach GRPOTrainer through the online trainer.
GRPO_CTOR_POSITIONS = ctor_positions(GRPOTrainer, "model", "args")


def disable_trl_liger_grpo_loss(training_args) -> None:
    """Keep TRL's ``use_liger_kernel`` off for the on-policy GRPO trainers.

    On GRPO the flag swaps the loss for ``LigerFusedLinearGRPOLoss``, which breaks the global
    ``num_items_in_batch`` normalizer, bypasses chunked-logprobs OOM protection, and drops the
    entropy path. Halo's toolkit default sets it True, so force it off before TRL caches it.
    """
    disable_trl_liger(
        training_args,
        "Disabling TRL's use_liger_kernel for GRPO: it swaps the loss for the fused Liger GRPO "
        "loss (breaks global token normalization and chunked logprobs). Model-level Liger "
        "kernels are still applied by load_distributed_model.",
    )


class OnPolicyGRPOInitMixin:
    """Open/close halves of an on-policy GRPO constructor. Mixed into the trainer."""

    def _begin_on_policy_init(self, args: tuple, kwargs: dict) -> tuple[object, dict]:
        """Resolve the training config and extract the distributed kwargs; ``(config, kwargs)``.

        The config is resolved here, positionally or by keyword, because TRL's Liger GRPO loss has to
        be switched off on it before ``_init_distributed_config`` runs; the positionals are forwarded
        so that a positional model reaches the mixin's MoE gates too.
        """
        training_args = ctor_config(args, kwargs, GRPO_CTOR_POSITIONS)
        disable_trl_liger_grpo_loss(training_args)
        return training_args, self._init_distributed_config(
            kwargs, training_args=training_args, ctor_args=args, ctor_positions=GRPO_CTOR_POSITIONS
        )

    def _finish_on_policy_init(self) -> None:
        """Realize the parallel modes, gate the reference model, the chunked head path and the
        full-logits plane, wire weight sync, disable dropout.

        Dropout goes last: it must reach the EP expert-LoRA dropout that ``_setup_distributed_modes``
        realizes, or the recomputed log-probs drift from the engine's dropout-free sampling. The plane
        is checked once the model is placed, so the free memory it is weighed against is real.
        """
        self._setup_distributed_modes()
        self._validate_implicit_reference_model()
        self._resolve_chunked_head_transform()
        self._check_full_logits_fit(self._loss_logits_width())
        self._setup_weight_sync()
        self._disable_dropout_for_onpolicy()

    def _setup_weight_sync(self) -> None:
        """Wire and gate this trainer's engine weight sync, before the first rollout can use it."""
        raise NotImplementedError

    def _loss_logits_width(self) -> LogitsWidth | None:
        """The completion logits row this trainer's loss forward carries and the setting that bounds it;
        ``None`` when no setting does."""
        raise NotImplementedError
