"""Construction refusals of the on-policy GRPO trainers and their advantage hooks. The constructors call them in a
fixed order, ahead of TRL's constructor, which on online GRPO opens the NCCL group to the rollout server."""

import datasets
import torch
from trl import GRPOTrainer

from halo.configs.grpo import RLRRConfig
from halo.trainers.construction.ctor_slots import get_ctor_arg, resolve_ctor_positions
from halo.trainers.grpo.objective.advantages import scales_rewards
from halo.trainers.grpo.objective.loss_inputs import TOKEN_SUM_LOSS_TYPES

_GATED_CTOR_POSITIONS = resolve_ctor_positions(
    GRPOTrainer, "train_dataset", "eval_dataset", "tools", "environment_factory", "rollout_func"
)


def reject_unread_reference(ref_model, beta: float, *, read: bool) -> None:
    """Checked before TRL's constructor and after it."""
    if ref_model is not None and not read:
        raise ValueError(
            f"ref_model was passed, but this run holds no KL reference (beta={beta}, or a policy a PEFT adapter "
            "wraps, scored with the adapter disabled), so it would never be read. Drop ref_model."
        )


def reject_unhonored_trl_options(
    training_args, *, use_chunked_grpo_logprobs: bool, needs_custom_dataloader: bool, args: tuple, kwargs: dict
) -> None:
    """``use_adaptive_entropy`` keeps its coefficient in a sidecar TRL restores only through the base optimizer load,
    which a sharded resume replaces. ``entropy_coef`` trains through the entropy's gradient, which the chunked sweep
    detaches. The DP-sharded loader's prompt-group sampler needs a dataset with a length."""
    if training_args.use_adaptive_entropy:
        raise ValueError(
            "use_adaptive_entropy is not supported: a sharded resume does not restore TRL's adaptive "
            "entropy coefficient. Set a fixed entropy_coef instead."
        )
    if training_args.entropy_coef != 0.0 and use_chunked_grpo_logprobs:
        raise ValueError(
            f"entropy_coef={training_args.entropy_coef} trains nothing with use_chunked_grpo_logprobs: the "
            "chunked sweep returns a detached entropy. Set entropy_coef: 0.0 or turn the chunked sweep off."
        )
    if not needs_custom_dataloader:
        return
    for name in ("train_dataset", "eval_dataset"):
        dataset = get_ctor_arg(args, kwargs, name, _GATED_CTOR_POSITIONS)
        splits = dataset.values() if isinstance(dataset, dict) else [dataset]
        if any(isinstance(split, (datasets.IterableDataset, torch.utils.data.IterableDataset)) for split in splits):
            raise ValueError(
                f"{name} is an IterableDataset, which the DP-sharded loader this run uses cannot group "
                "into prompt groups (its sampler needs a length). Load it as a map-style Dataset."
            )


def require_vllm_server_mode(grpo_args):
    """Refuses in-process HF generation (slow and rank-divergent under FSDP2/EP) and colocate (no vLLM in the
    image)."""
    if grpo_args is None:
        raise ValueError(
            "DistributedOnlineGRPOTrainer requires a GRPOConfig: pass it as the `args` keyword or in "
            "the conventional positional slot. Without it the vLLM server-mode requirement, the "
            "parallelism config and the Liger overrides all silently go unapplied."
        )
    if not grpo_args.use_vllm:
        raise ValueError(
            "DistributedOnlineGRPOTrainer requires vLLM server-mode generation: set use_vllm=True "
            "and vllm_mode='server'. In-process HF generation (use_vllm=False) is not "
            "supported — it is slow and desyncs ranks under FSDP2/EP. Run a separate vLLM "
            "container (docker-compose.vllm.yml) and point vllm_server_host/vllm_server_port "
            "at it (weights sync over NCCL)."
        )
    if grpo_args.vllm_mode != "server":
        raise ValueError(
            f"DistributedOnlineGRPOTrainer supports only vllm_mode='server', got "
            f"'{grpo_args.vllm_mode}'. Colocate mode would launch an in-process vLLM on the "
            f"training GPUs, but the training image has no vLLM (ABI-incompatible "
            f"with the training PyTorch/Transformers). Run a separate vLLM "
            f"container (docker-compose.vllm.yml) and set vllm_mode='server' "
            f"with vllm_server_host/vllm_server_port (weights sync over NCCL)."
        )
    return grpo_args


def reject_tool_calling_loop(ctor_args: tuple, kwargs: dict) -> None:
    """TRL's tool-calling loop regenerates while any rank-local completion still calls a tool, and each round is a
    collective generate, so ranks whose completions stop calling tools at different rounds desync."""
    given = [
        name
        for name in ("tools", "environment_factory")
        if get_ctor_arg(ctor_args, kwargs, name, _GATED_CTOR_POSITIONS)
    ]
    if given:
        raise ValueError(
            f"DistributedOnlineGRPOTrainer does not run TRL's tool-calling loop ({', '.join(given)}): it loops as "
            "long as some completion on this rank calls a tool, each round a collective vLLM generate, so ranks "
            "fall out of step. Train tool use with Async GRPO with Environments "
            "(scripts/training/grpo/environmental.py), whose episodes run in the rollout actors."
        )


def reject_unshared_rollout_func(parallelism_config, ctor_args: tuple, kwargs: dict) -> None:
    """``rollout_func`` bypasses the ``_generate_single_turn`` override that hands TP/ETP siblings their leader's
    completions, so each would forward its own rollouts into the in-forward collectives."""
    if not (parallelism_config.is_tp_mode or parallelism_config.is_expert_tp_mode):
        return
    if get_ctor_arg(ctor_args, kwargs, "rollout_func", _GATED_CTOR_POSITIONS) is not None:
        raise ValueError(
            "rollout_func is not supported under tensor or expert-tensor parallelism: it replaces the "
            "generation that broadcasts the group leader's completions, so siblings sharing one sharded "
            "replica would forward different tokens and deadlock in the model's collectives. Drop "
            "rollout_func, or run without tensor_parallel_size / expert_tensor_parallel_size."
        )


def reject_inert_std_floor(scale_rewards: str | bool | None, std_floor: float) -> None:
    if std_floor > 0 and not scales_rewards(scale_rewards):
        raise ValueError(
            f"scale_rewards_std_floor={std_floor} with scale_rewards={scale_rewards!r}: the floor bounds the std "
            "the advantages divide by, and this run divides by none. Set scale_rewards to 'group' or 'batch', "
            "or drop the floor."
        )


def reject_unbalanceable_token_mass(args) -> None:
    """Two masks inside TRL's loss drop tokens after the balance has weighed them: the entropy quantile and the
    off-policy sequence mask, which drops negative sequences alone and so turns the net push positive."""
    if args.loss_type not in TOKEN_SUM_LOSS_TYPES:
        raise ValueError(
            f"balance_token_mass needs a loss whose loss tokens share one normalizer ({', '.join(TOKEN_SUM_LOSS_TYPES)}), "
            f"got loss_type={args.loss_type!r}: there a row does not pull with its token count, so the balance "
            "would add a push instead of cancelling one."
        )
    if args.top_entropy_quantile < 1.0:
        raise ValueError(
            f"balance_token_mass with top_entropy_quantile={args.top_entropy_quantile}: the entropy mask drops tokens "
            "inside the loss, after the balance has weighed them. Set top_entropy_quantile: 1.0."
        )
    if args.off_policy_mask_threshold is not None:
        raise ValueError(
            f"balance_token_mass with off_policy_mask_threshold={args.off_policy_mask_threshold}: the mask drops "
            "negative-advantage sequences inside the loss, after the balance has weighed them, so the round's net "
            "push turns positive. Unset one of them."
        )


def reject_conflicting_advantage_hooks(
    grpo_args,
    rlrr_config: RLRRConfig | None,
    *,
    std_floor: float,
    drop_degenerate_groups: bool,
    balance_token_mass: bool,
    recomputes_from_gathered_rewards: bool,
) -> None:
    """Refuse the online advantage hooks the GRPO config or another hook cancels or diverges from."""
    reject_inert_std_floor(grpo_args.scale_rewards, std_floor)
    if balance_token_mass:
        reject_unbalanceable_token_mass(grpo_args)
    if rlrr_config is not None:
        if std_floor > 0:
            raise ValueError(
                "scale_rewards_std_floor and rlrr_config cannot be combined: RLRR replaces the "
                "group-normalized advantages wholesale with relative-ranking ones, which never divide "
                "by a reward std, so the floor would silently do nothing. Set only one."
            )
        if drop_degenerate_groups:
            raise ValueError(
                "drop_degenerate_groups and rlrr_config cannot be combined: the drop keys on raw-reward "
                "equality, so it masks out of the loss every all-correct group RLRR just gave length-ranked "
                "advantages — the signal RLRR exists to keep. Set only one."
            )
    # Every hook re-derives its values from the gathered rewards summed as TRL's sum_then_normalize branch does.
    if recomputes_from_gathered_rewards and grpo_args.multi_objective_aggregation != "sum_then_normalize":
        raise ValueError(
            "RLRR / drop_degenerate_groups / scale_rewards_std_floor recompute rewards "
            "with TRL's sum_then_normalize aggregation; multi_objective_aggregation="
            f"{grpo_args.multi_objective_aggregation!r} would silently diverge."
        )
