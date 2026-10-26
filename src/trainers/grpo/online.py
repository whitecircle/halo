"""Online GRPO trainer under EP / TP / ETP and their supported combinations (no CP or PP).

Server-mode vLLM generation and weight sync run through ``TrlVLLMClient``; vLLM is not installed in
the training environment (ABI-incompatible extensions, older transformers than the image).
"""

import contextlib
import types
from collections.abc import Iterator
from typing import Any

import torch
import trl.generation.vllm_client as _trl_vllm_client
import trl.generation.vllm_generation as _trl_vllm_generation
from accelerate.logging import get_logger
from trl import GRPOTrainer

from src.configs.grpo import EarlyStopConfig, RLRRConfig
from src.environments.rollout_config import VLLM_BACKEND
from src.trainers.distributed_trainer import DistributedTrainerMixin
from src.trainers.grpo.early_stop import SAMPLING_LOGP_GAP_KEY, build_early_stop_callback
from src.trainers.grpo.grpo_rules import (
    reject_conflicting_advantage_hooks,
    reject_tool_calling_loop,
    reject_unshared_rollout_func,
    require_vllm_server_mode,
)
from src.trainers.grpo.logprob_scoring import ChunkedGRPOLogprobsMixin, LogitsWidth, ProtectedTokenEntropyMixin
from src.trainers.grpo.objective.advantages import degenerate_group_mask, group_relative_advantages
from src.trainers.grpo.objective.logratio import KL_CLAMP_FRAC_KEY, clamp_ref_logps, kl_clamp_counts
from src.trainers.grpo.objective.loss_inputs import (
    DEGENERATE_GROUP_FRAC_KEY,
    gathered_num_items,
    narrow_loss_masks,
    record_token_mass,
)
from src.trainers.grpo.objective.relative_rewards import relative_advantages_grouped
from src.trainers.grpo.on_policy import GenerationRoundMixin, OnPolicyGRPOInitMixin
from src.trainers.grpo.rollout.completions_logging import DecoupledCompletionsLogMixin
from src.trainers.grpo.rollout.trl_vllm_client import TrlVLLMClient
from src.trainers.grpo.rollout.weight_push import sync_trainer_weights
from src.trainers.token_metrics import effective_loss_mask
from src.trainers.world_metrics import world_sums

logger = get_logger(__name__, log_level="info")


class DistributedGRPOTrainer(
    OnPolicyGRPOInitMixin,
    GenerationRoundMixin,
    ProtectedTokenEntropyMixin,
    ChunkedGRPOLogprobsMixin,
    DecoupledCompletionsLogMixin,
    DistributedTrainerMixin,
    GRPOTrainer,
):
    """Online GRPO over server-mode vLLM rollouts. No CP (``SUPPORTS_CP`` stays off) and no PP (below)."""

    PP_REFUSAL = (
        "the vLLM weight sync gathers and sends the model's full named_parameters() from one "
        "rank-set in a fixed collective order, but under PP no rank holds all the layers, so the "
        "sync needs per-stage gathers assembled across stages; the rollout phase also forwards the "
        "TRAINING model outside compute_loss (old / reference per-token log-probs), which would have "
        "to run as a forward-only pass across the whole pipeline. The per-microbatch update itself "
        "is a single forward over precomputed advantages and reference log-probs, so the loss is not "
        "the blocker"
    )
    # Online GRPO is vLLM-only by construction (TRL's server-mode generation), so the engine the
    # weight-sync gate reads is fixed here rather than configured.
    _rollout_backend = VLLM_BACKEND

    def __init__(self, *args, **kwargs):
        self._use_chunked_grpo_logprobs = kwargs.pop("use_chunked_grpo_logprobs", False)
        training_args, kwargs = self._begin_on_policy_init(args, kwargs)
        grpo_args = require_vllm_server_mode(training_args)
        reject_tool_calling_loop(args, kwargs)
        reject_unshared_rollout_func(self.parallelism_config, args, kwargs)
        self._adopt_advantage_hooks(kwargs)
        reject_conflicting_advantage_hooks(
            grpo_args,
            self._rlrr_config,
            std_floor=self._scale_rewards_std_floor,
            drop_degenerate_groups=self._drop_degenerate_groups,
            balance_token_mass=self._balance_token_mass,
            recomputes_from_gathered_rewards=self._recomputes_from_gathered_rewards,
        )
        # Built before TRL's ctor opens the NCCL group to the rollout server, so a refused condition costs nothing.
        early_stop = build_early_stop_callback(
            kwargs.pop("early_stop", EarlyStopConfig()),
            gap_key=SAMPLING_LOGP_GAP_KEY,
            gap_logged=grpo_args.vllm_importance_sampling_correction,
        )

        self._last_rewards_per_func: torch.Tensor | None = None

        with self._installing_trl_vllm_client(), self._supplying_kl_reference():
            super().__init__(*args, **kwargs)
        if early_stop is not None:
            self.add_callback(early_stop)

        if self._rlrr_config is not None:
            logger.info("RLRR relative-reward shaping enabled (mode=%s)", self._rlrr_config.mode)

        # The k3 tail clamp engages only when TRL's recompute gate produces old_per_token_logps.
        recompute_gate = self._misaligned_accumulation or (self.use_vllm and self.vllm_importance_sampling_correction)
        if self.beta != 0.0 and not recompute_gate:
            logger.warning(
                "beta=%s without the old-logps recompute (vllm_importance_sampling_correction off, "
                "aligned accumulation): the k3 KL tail clamp cannot engage — a single suppressed "
                "token can dominate the KL gradient. Enable the vLLM IS correction to restore it.",
                self.beta,
            )

        self._finish_on_policy_init()

    @staticmethod
    def sums_sequence_logratio(grpo_args) -> bool:
        """Whether TRL's vLLM IS correction sums the per-token log-ratios over each sequence — the online arm's
        counterpart of ``ISMaskConfig.sums_sequence_logratio``, the consumer the sampler-logprob preflight gates
        a nucleus-renormalized reference against: one ratio per sequence (``sequence_*`` modes), or ``vespo``,
        whose gamma weight takes the per-token ratios' log-sum in place of the per-token multiply. Off with the
        correction off, whatever the mode or loss says."""
        if not grpo_args.vllm_importance_sampling_correction:
            return False
        return grpo_args.loss_type == "vespo" or grpo_args.vllm_importance_sampling_mode.startswith("sequence")

    def _adopt_advantage_hooks(self, kwargs: dict[str, Any]) -> None:
        """Pop the advantage-hook kwargs TRL's explicit signature would reject."""
        self._rlrr_config: RLRRConfig | None = kwargs.pop("rlrr_config", None)
        self._drop_degenerate_groups: bool = kwargs.pop("drop_degenerate_groups", False)
        self._scale_rewards_std_floor: float = kwargs.pop("scale_rewards_std_floor", 0.0)
        self._balance_token_mass: bool = kwargs.pop("balance_token_mass", False)

    @staticmethod
    @contextlib.contextmanager
    def _installing_trl_vllm_client() -> Iterator[None]:
        """Install ``TrlVLLMClient`` and force TRL's ``is_vllm_available`` True for the duration of TRL's ctor,
        since vLLM is not installed here; server mode, the only shape TRL builds that client for, is required
        first."""
        originals = (
            _trl_vllm_generation.is_vllm_available,
            _trl_vllm_client.is_vllm_available,
            _trl_vllm_generation.VLLMClient,
        )
        _trl_vllm_generation.is_vllm_available = _trl_vllm_client.is_vllm_available = lambda: True
        _trl_vllm_generation.VLLMClient = TrlVLLMClient
        try:
            yield
        finally:
            (
                _trl_vllm_generation.is_vllm_available,
                _trl_vllm_client.is_vllm_available,
                _trl_vllm_generation.VLLMClient,
            ) = originals

    def _loss_logits_width(self) -> LogitsWidth | None:
        """TRL keeps one logit past the completion for the next-token shift."""
        if self.max_completion_length is None:
            return None
        return LogitsWidth(self.max_completion_length + 1, "max_completion_length")

    def _install_weight_sync(self) -> None:
        """Replace ``VLLMGeneration.sync_weights`` with the distributed-aware push: TRL unfolds DTensors only
        under ``is_fsdp_enabled``, which the toolkit's ``fully_shard`` under accelerate MULTI_GPU never sets, so
        it would forward them verbatim and deadlock against the trainer-vLLM NCCL group."""
        if getattr(self, "vllm_generation", None) is None:
            raise RuntimeError(
                "TRL built no vllm_generation for this trainer, so the distributed-aware weight sync "
                "cannot be installed and every rollout would be generated by an engine that never "
                "receives the trained weights. Server-mode vLLM is required (use_vllm=True, "
                "vllm_mode='server') — check vllm_server_host/vllm_server_port reach a running server."
            )

        self.vllm_generation.sync_weights = types.MethodType(type(self)._distributed_sync_weights, self)
        config = self.parallelism_config
        logger.info(
            "Installed distributed-aware vLLM weight sync "
            f"(ep={config.has_ep_group}, tp={config.is_tp_mode}, "
            f"expert_tp={config.is_expert_tp_mode})"
        )

    def _distributed_sync_weights(self) -> None:
        """COLLECTIVE: the EP/TP/FSDP2/PEFT-aware ``VLLMGeneration.sync_weights``; the client TRL holds on
        ``vllm_generation`` forwards from the main process."""
        client = self.vllm_generation.vllm_client if self.accelerator.is_main_process else None
        sync_trainer_weights(self, client)

    def _generate_single_turn(self, *args, **kwargs):
        """Override TRL's single-turn vLLM generation to be TP-consistent.

        TRL slices the result by ``process_index``, so TP ranks land on different completions and
        deadlock at the first all-reduce in ``model.forward``. Broadcast the tuple from the TP leader.
        """
        result = super()._generate_single_turn(*args, **kwargs)
        return self._broadcast_object_from_tp_leader(result)

    def _calculate_rewards(self, *args, **kwargs):
        """Stash TRL's gathered per-function rewards, so the advantage hooks see the full group set."""
        rewards_per_func = super()._calculate_rewards(*args, **kwargs)
        if self._recomputes_from_gathered_rewards:
            self._last_rewards_per_func = rewards_per_func
        return rewards_per_func

    @property
    def _recomputes_from_gathered_rewards(self) -> bool:
        """Whether any hook re-derives advantages or masks from the stashed full reward set; the stash in
        :meth:`_calculate_rewards` and every consumer read this one gate."""
        return self._rlrr_config is not None or self._drop_degenerate_groups or self._scale_rewards_std_floor > 0

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        """Generate, score, run the advantage hooks, and broadcast the scored tensors across the TP/ETP group,
        whose generation ``_generate_single_turn`` already shares: CPU-side post-processing may diverge."""
        result = super()._generate_and_score_completions(inputs)

        # Before the TP/ETP broadcast, so the reshaped tensors are what gets broadcast.
        self._apply_rlrr_advantages(result)
        self._apply_std_floor_advantages(result)
        self._apply_degenerate_group_drop(result)
        self._apply_token_mass_balance(result)
        # Consumed: the next generation batch must stash its own rewards, never reuse these.
        self._last_rewards_per_func = None
        # After the drop, so the clamp share reads the loss mask the KL term is averaged on.
        self._clamp_kl_reference(result)

        return self._broadcast_tensors_from_tp_leader(result)

    def _apply_rlrr_advantages(self, result: dict[str, torch.Tensor | Any]) -> None:
        """Replace the group-normalized advantages with RLRR relative-ranking ones, recomputed on the full
        gathered reward set in TRL's group order. Train mode only."""
        if self._rlrr_config is None:
            return
        gathered = self._gathered_rewards()
        if gathered is None:
            return
        rewards, unscorable = gathered

        # Same process order as the rewards, so the length re-ranking sees each group's members.
        local_lengths = result["completion_mask"].sum(dim=1).to(rewards.device)
        full_lengths = self.accelerator.gather(local_lengths)

        # ``valid`` both keeps the unscorable placeholders out of their group's ranking and returns
        # their own advantage as 0, which is TRL's unscorable handling.
        advantages_full = torch.from_numpy(
            relative_advantages_grouped(
                rewards.float().cpu().numpy(),
                group_size=self.num_generations,
                config=self._rlrr_config,
                lengths=full_lengths.float().cpu().numpy(),
                valid=(~unscorable).cpu().numpy(),
            )
        )

        self._install_advantages(result, advantages_full, "RLRR")

    def _apply_std_floor_advantages(self, result: dict[str, torch.Tensor | Any]) -> None:
        """Replace TRL's group-normalized advantages with ones whose std divisor honours
        ``scale_rewards_std_floor``, recomputed on the full gathered reward set and re-sliced to this
        rank. Train mode only. TRL divides by a fixed ``std + 1e-4`` and cannot take the floor.
        """
        if self._scale_rewards_std_floor <= 0:
            return
        gathered = self._gathered_rewards()
        if gathered is None:
            return
        rewards, unscorable = gathered
        advantages_full = group_relative_advantages(
            rewards.float(),
            self.num_generations,
            self.args.scale_rewards,
            valid_mask=~unscorable,
            already_gathered=True,
            std_floor=self._scale_rewards_std_floor,
        )
        advantages_full = advantages_full.masked_fill(unscorable, 0.0)  # match TRL's unscorable handling
        self._install_advantages(result, advantages_full, "scale_rewards_std_floor")

    def _apply_degenerate_group_drop(self, result: dict[str, torch.Tensor | Any]) -> None:
        """Mask the completions of all-equal-reward groups out of the loss (parity with the
        environmental trainer): their advantage is already 0, but their tokens would still inflate
        the DAPO normalizer and dilute the groups that carry signal. Train mode only."""
        if not self._drop_degenerate_groups:
            return
        gathered = self._gathered_rewards()
        if gathered is None:
            return
        rewards, unscorable = gathered
        # Scorable members only: an unscorable placeholder's reward must not hide an all-alike group.
        drop_full = degenerate_group_mask(rewards, self.num_generations, valid_mask=~unscorable)
        drop = self._local_slice(drop_full, result["completion_mask"].shape[0])
        self._metrics["train"][DEGENERATE_GROUP_FRAC_KEY].append(drop_full.float().mean().item())
        # Narrow EVERY mask TRL composes into its loss mask, else untrained tokens inflate the normalizer.
        present = [name for name in ("completion_mask", "tool_mask") if result.get(name) is not None]
        narrowed = narrow_loss_masks(drop, *(result[name] for name in present))
        result.update(zip(present, narrowed, strict=True))
        # Recompute the global DAPO normalizer post-drop; the gate is rank-uniform, so the gather is safe.
        if "num_items_in_batch" in result:
            loss_mask = effective_loss_mask(result)
            if loss_mask is None:
                raise RuntimeError(
                    "drop_degenerate_groups cannot recompute the global DAPO normalizer: the scored "
                    "batch carries no completion_mask, so the post-drop token count is unknowable and "
                    "the loss would normalize by the pre-drop count."
                )
            result["num_items_in_batch"] = gathered_num_items(loss_mask, self.accelerator.gather)

    def _apply_token_mass_balance(self, result: dict[str, torch.Tensor | Any]) -> None:
        """Log the batch's net token mass and, under ``balance_token_mass``, cancel it
        (:func:`~src.trainers.grpo.objective.loss_inputs.record_token_mass`), as the environmental
        trainer does: a token weighs its place in the loss times TRL's vLLM IS ratio. Train mode only."""
        if not self.model.training:
            return
        loss_mask = effective_loss_mask(result)
        if loss_mask is None:
            raise RuntimeError(
                "The net token mass cannot be weighed: TRL's scored batch carries no completion_mask, so no "
                "row's trained token count is known."
            )
        advantages = result["advantages"]
        balance = record_token_mass(
            advantages,
            loss_mask.to(advantages.dtype),
            result.get("importance_sampling_ratio"),
            self.accelerator.gather,
            self._metrics["train"],
            self._balance_token_mass,
        )
        if balance is not None:
            self._install_advantages(result, balance.apply(self.accelerator.gather(advantages)), "balance_token_mass")

    def _clamp_kl_reference(self, result: dict[str, torch.Tensor | Any]) -> None:
        """The k3 tail clamp on the reference log-probs, and ``kl_clamp_frac`` over the batch's loss
        tokens (:func:`kl_clamp_counts`). Both tensors exist only when ``beta != 0`` and TRL's
        recompute gate fired, config verdicts every rank shares, so the all-reduce is reached by all or none."""
        old_logps = result.get("old_per_token_logps")
        ref_logps = result.get("ref_per_token_logps")
        if old_logps is None or ref_logps is None:
            return
        result["ref_per_token_logps"], clamped = clamp_ref_logps(ref_logps, old_logps)
        mode = "train" if self.model.training else "eval"
        clamped_tokens, loss_tokens = world_sums(kl_clamp_counts(clamped, effective_loss_mask(result)))
        self._metrics[mode][KL_CLAMP_FRAC_KEY].append(clamped_tokens / loss_tokens if loss_tokens else 0.0)

    def _gathered_rewards(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Stashed full (gathered) rewards as ``(rewards, unscorable)``, or ``None`` outside train.

        ``rewards`` aggregates the reward functions as TRL's ``sum_then_normalize`` branch does (weighted
        nansum), in the gathered process order; ``unscorable`` marks rows where every reward fn returned NaN
        (TRL forces their advantage to 0). Only an armed hook calls it, so a missing stash in train mode, TRL
        scoring the batch past :meth:`_calculate_rewards`, raises rather than leave the hook unapplied.
        """
        if not self.model.training:
            return None
        if self._last_rewards_per_func is None:
            raise RuntimeError(
                "An advantage hook (RLRR / drop_degenerate_groups / scale_rewards_std_floor) is armed "
                "but no gathered rewards were stashed for this generation batch: TRL's scoring no "
                "longer goes through _calculate_rewards, so the hook would silently leave TRL's "
                "advantages in place."
            )
        rewards_per_func = self._last_rewards_per_func
        weights = self.reward_weights.to(rewards_per_func.device).unsqueeze(0)
        rewards = (rewards_per_func * weights).nansum(dim=1)  # [total], gathered order
        return rewards, torch.isnan(rewards_per_func).all(dim=1)

    def _local_slice(self, full: torch.Tensor, n_local: int) -> torch.Tensor:
        """This rank's rows of a gathered per-completion tensor, TRL's own world-order process slice. The hooks
        compute on the full gathered set, where groups are contiguous, so a group may span two ranks."""
        start = self.accelerator.process_index * n_local
        return full[start : start + n_local]

    def _install_advantages(
        self, result: dict[str, torch.Tensor | Any], advantages_full: torch.Tensor, what: str
    ) -> None:
        """Slice ``advantages_full`` (a hook's full gathered set) to this rank, install it, and realign TRL's
        advantage log, so the logged record carries the values trained on.

        TRL fills ``_logs["advantages"]`` inside ``_generate_and_score_completions``, before any hook runs, in
        the gathered tensor's world order and length, so overwriting that tail realigns the record row for row.
        """
        local = self._local_slice(advantages_full, result["advantages"].shape[0])
        result["advantages"] = local.to(device=result["advantages"].device, dtype=result["advantages"].dtype)
        logged = self._logs["advantages"]
        replacement = advantages_full.tolist()
        if len(logged) < len(replacement):
            raise RuntimeError(
                f"{what} cannot realign the completions record: TRL logged {len(logged)} advantages "
                f"for this generation batch but the recomputed set has {len(replacement)}. Overwriting "
                f"the tail would attribute each row's advantage to the wrong completion."
            )
        for i, value in enumerate(replacement):
            logged[len(logged) - len(replacement) + i] = value
