"""Construction spine shared by the on-policy GRPO trainers (online RLVR and environmental).

Both constructors open the same way (resolve a config that may have arrived positionally, disable
TRL's Liger GRPO loss, extract the distributed kwargs), run TRL's constructor with the frozen KL
reference handed in, and close the same way: realize the parallel modes, gate the reference model and
the chunked sweep's head path, gate and wire weight sync, then disable dropout. The closing order is
load-bearing, so it is defined here rather than in each constructor.
"""

import contextlib
from collections.abc import Iterator

import trl.trainer.grpo_trainer as trl_grpo_trainer
from trl import GRPOTrainer

from src.distributed.loading.frozen_models import (
    ON_POLICY_GRPO_REFERENCE_ALTERNATIVES,
    on_policy_grpo_holds_reference,
    place_and_freeze,
)
from src.trainers.grpo.mixins.chunked_logprobs import LogitsWidth
from src.trainers.grpo.rollout.weight_sync import validate_weight_sync_support
from src.trainers.mixins.validation import ctor_config, ctor_positions, ctor_value, disable_trl_liger

# TRL GRPOTrainer positional slots, for ctor params arriving via *args — derived from the installed signature.
GRPO_CTOR_POSITIONS = ctor_positions(GRPOTrainer, "model", "args")
# The slot of the adapter config, which with the model decides whether TRL holds a KL reference.
_PEFT_CONFIG_CTOR_POSITIONS = ctor_positions(GRPOTrainer, "peft_config")


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


def _unread_reference(beta: float) -> ValueError:
    """The refusal of a ``ref_model`` the run never reads, raised before TRL's constructor and after it."""
    return ValueError(
        f"ref_model was passed, but this run holds no KL reference (beta={beta}, or a policy a PEFT adapter "
        "wraps, scored with the adapter disabled), so it would never be read. Drop ref_model."
    )


class OnPolicyGRPOInitMixin:
    """Open/close halves of an on-policy GRPO constructor. Mixed into the trainer."""

    _reference_alternatives = ON_POLICY_GRPO_REFERENCE_ALTERNATIVES

    def _begin_on_policy_init(self, args: tuple, kwargs: dict) -> tuple[object, dict]:
        """Resolve the training config and extract the distributed kwargs; ``(config, kwargs)``.

        The config is resolved here, positionally or by keyword, because TRL's Liger GRPO loss has to
        be switched off on it before ``_init_distributed_config`` runs; the positionals are forwarded
        so that a positional model reaches the mixin's MoE gates too. ``save_completions`` is taken
        here for :class:`DecoupledCompletionsLogMixin` and ``ref_model`` for
        :meth:`_supplying_kl_reference`, since TRL's ctor would refuse either. A ``ref_model`` the run would
        never read is refused here, ahead of TRL's ctor, which on online GRPO opens the rollout client; the
        hand-off re-checks it against what TRL actually asks for.
        """
        self._save_completions = kwargs.pop("save_completions", True)
        self._supplied_ref_model = kwargs.pop("ref_model", None)
        training_args = ctor_config(args, kwargs, GRPO_CTOR_POSITIONS)
        if (
            self._supplied_ref_model is not None
            and training_args is not None
            and not on_policy_grpo_holds_reference(
                training_args.beta,
                ctor_value(args, kwargs, "peft_config", _PEFT_CONFIG_CTOR_POSITIONS),
                ctor_value(args, kwargs, "model", GRPO_CTOR_POSITIONS),
            )
        ):
            raise _unread_reference(training_args.beta)
        disable_trl_liger_grpo_loss(training_args)
        return training_args, self._init_distributed_config(
            kwargs, training_args=training_args, ctor_args=args, ctor_positions=GRPO_CTOR_POSITIONS
        )

    @contextlib.contextmanager
    def _supplying_kl_reference(self) -> Iterator[None]:
        """Run TRL's constructor with the frozen KL reference the caller loaded in place of the one TRL builds.

        TRL builds a reference at ``beta != 0`` on a policy no PEFT adapter wraps, by ``create_model_from_path``
        on the policy config's name: fp32, the hub's default revision, the default attention, no repair of the
        non-persistent buffers, no vocabulary resize. Here that call returns the ``ref_model`` passed
        (:func:`~src.distributed.loading.frozen_models.load_reference_model_for_on_policy_grpo`), placed and
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
                    "KL against the policy. Pass ref_model=load_reference_model_for_on_policy_grpo(...) "
                    "(src/distributed/loading/frozen_models.py), as the GRPO training scripts do. Or hold none: "
                    f"{self._reference_alternatives.under(self.parallelism_config)}"
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

        Dropout goes last: it must reach the EP expert-LoRA dropout that ``_setup_distributed_modes``
        realizes, or the recomputed log-probs drift from the engine's dropout-free sampling. The plane
        is checked once the model is placed, so the free memory it is weighed against is real. The
        weight-sync gate refuses a model the trainer's engine (``_rollout_backend``) cannot take an
        update for here, not as an opaque server-side error at the first push.
        """
        self._setup_distributed_modes()
        self._validate_held_reference_model()
        self._resolve_chunked_head_transform()
        self._check_full_logits_fit(self._loss_logits_width())
        validate_weight_sync_support(self.model, self._rollout_backend)
        self._setup_weight_sync()
        self._disable_dropout_for_onpolicy()

    def _setup_weight_sync(self) -> None:
        """Install this trainer's engine weight sync. Nothing by default: a trainer whose rollout side owns
        its client (the environmental one) pushes through it."""

    def _loss_logits_width(self) -> LogitsWidth | None:
        """The completion logits row this trainer's loss forward carries and the setting that bounds it;
        ``None`` when no setting does."""
        raise NotImplementedError
