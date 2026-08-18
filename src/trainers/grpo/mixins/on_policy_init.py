"""Construction spine shared by the on-policy GRPO trainers (online RLVR and environmental).

Both constructors open the same way (resolve a config that may have arrived positionally, disable
TRL's Liger GRPO loss, extract the distributed kwargs) and close the same way: realize the parallel
modes, gate the implicit reference model, wire weight sync, then disable dropout. The closing order
is load-bearing, so it is defined here rather than in each constructor.
"""

from src.trainers.mixins.validation import ctor_config, disable_trl_liger_grpo_loss


class OnPolicyGRPOInitMixin:
    """Open/close halves of an on-policy GRPO constructor. Mixed into the trainer."""

    def _begin_on_policy_init(self, args: tuple, kwargs: dict) -> tuple[object, dict]:
        """Resolve the training config and extract the distributed kwargs; ``(config, kwargs)``.

        The config may arrive positionally, which ``_init_distributed_config``'s ``kwargs["args"]``
        fallback cannot see; its Liger EP/TP/CP filter, non-shared-FS ``save_on_each_node`` forcing
        and EP/CP reentrant override would then no-op.
        """
        training_args = ctor_config(args, kwargs)
        disable_trl_liger_grpo_loss(training_args)
        return training_args, self._init_distributed_config(kwargs, training_args=training_args)

    def _finish_on_policy_init(self) -> None:
        """Realize the parallel modes, gate the reference model, wire weight sync, disable dropout.

        Dropout goes last: it must reach the EP expert-LoRA dropout that ``_setup_distributed_modes``
        realizes, or the recomputed log-probs drift from the engine's dropout-free sampling.
        """
        self._setup_distributed_modes()
        self._validate_implicit_reference_model()
        self._setup_weight_sync()
        self._disable_dropout_for_onpolicy()

    def _setup_weight_sync(self) -> None:
        """Wire and gate this trainer's engine weight sync, before the first rollout can use it."""
        raise NotImplementedError
