"""Online GRPO under EP / TP / ETP (no CP or PP). Server-mode vLLM runs through ``TrlVLLMClient``; vLLM is not
installed in the training environment (ABI-incompatible extensions, older transformers than the image)."""

import contextlib
import logging
import types
from collections.abc import Iterator
from typing import Any

import torch
import trl.generation.vllm_client as _trl_vllm_client
import trl.generation.vllm_generation as _trl_vllm_generation
from trl import GRPOTrainer

from halo.configs.grpo import EarlyStopConfig, RLRRConfig
from halo.diagnostics.phase_timers import Phase
from halo.environments.rollout_config import VLLM_BACKEND
from halo.runtime.ranks import is_global_main_process
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from halo.trainers.grpo.early_stop import SAMPLING_LOGP_GAP_KEY, build_early_stop_callback
from halo.trainers.grpo.grpo_rules import (
    reject_conflicting_advantage_hooks,
    reject_tool_calling_loop,
    reject_unshared_rollout_func,
    require_vllm_server_mode,
)
from halo.trainers.grpo.logprob_scoring import ChunkedGRPOLogprobsMixin, LogitsWidth, ProtectedTokenEntropyMixin
from halo.trainers.grpo.objective.advantages import degenerate_group_mask, group_relative_advantages
from halo.trainers.grpo.objective.logratio import KL_CLAMP_FRAC_KEY, clamp_ref_logps, kl_clamp_counts
from halo.trainers.grpo.objective.loss_inputs import (
    DEGENERATE_GROUP_FRAC_KEY,
    gathered_num_items,
    narrow_loss_masks,
    record_token_mass,
)
from halo.trainers.grpo.objective.relative_rewards import rlrr_advantages_grouped
from halo.trainers.grpo.on_policy import GenerationRoundMixin, OnPolicyGRPOInitMixin
from halo.trainers.grpo.packed_rounds import PackedRoundMixin
from halo.trainers.grpo.packed_rules import reject_unpackable_trainer
from halo.trainers.grpo.phase_timing import PhaseTimedMixin
from halo.trainers.grpo.rollout.completions_logging import DecoupledCompletionsLogMixin
from halo.trainers.grpo.rollout.trl_vllm_client import TrlVLLMClient
from halo.trainers.grpo.rollout.weight_push import sync_trainer_weights
from halo.trainers.grpo.row_span_stash import RowSpanStashMixin
from halo.trainers.token_metrics import effective_loss_mask
from halo.trainers.world_metrics import world_sums

logger = logging.getLogger(__name__)


class DistributedOnlineGRPOTrainer(
    OnPolicyGRPOInitMixin,
    PackedRoundMixin,
    GenerationRoundMixin,
    PhaseTimedMixin,
    ProtectedTokenEntropyMixin,
    ChunkedGRPOLogprobsMixin,
    RowSpanStashMixin,
    DecoupledCompletionsLogMixin,
    DistributedTrainerMixin,
    GRPOTrainer,
):
    PP_REFUSAL = (
        "the vLLM weight sync gathers and sends the model's full named_parameters() from one "
        "rank-set in a fixed collective order, but under PP no rank holds all the layers, so the "
        "sync needs per-stage gathers assembled across stages; the rollout phase also forwards the "
        "TRAINING model outside compute_loss (old / reference per-token log-probs), which would have "
        "to run as a forward-only pass across the whole pipeline. The per-microbatch update itself "
        "is a single forward over precomputed advantages and reference log-probs, so the loss is not "
        "the blocker"
    )
    # TRL's server-mode generation is vLLM-only, so the engine the weight-sync gate reads is fixed here.
    _rollout_backend = VLLM_BACKEND

    def __init__(self, *args, **kwargs):
        self._use_chunked_grpo_logprobs = kwargs.pop("use_chunked_grpo_logprobs", False)
        self._log_phase_times = kwargs.pop("log_phase_times", False)
        self._packed_microbatch_tokens = kwargs.pop("packed_microbatch_tokens", None)
        training_args, kwargs = self._begin_on_policy_init(args, kwargs)
        if self._packed_microbatch_tokens is not None:
            reject_unpackable_trainer(type(self), self.PACKED_ROWS_REFUSAL)
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

        if self._rlrr_config is not None and is_global_main_process():
            logger.info("RLRR relative-reward shaping enabled (mode=%s)", self._rlrr_config.mode)

        # The k3 tail clamp engages only when TRL's recompute gate produces old_per_token_logps.
        recompute_gate = self._misaligned_accumulation or (self.use_vllm and self.vllm_importance_sampling_correction)
        if self.beta != 0.0 and not recompute_gate and is_global_main_process():
            logger.warning(
                "beta=%s without the old-logps recompute (vllm_importance_sampling_correction off, "
                "aligned accumulation): the k3 KL tail clamp cannot engage — a single suppressed "
                "token can dominate the KL gradient. Enable the vLLM IS correction to restore it.",
                self.beta,
            )

        self.pad_token_id = self._tokenizer.pad_token_id
        self._finish_on_policy_init()
        self._configure_phase_timers()
        self._configure_row_span_stash()
        self._configure_packed_microbatches()

    @staticmethod
    def sums_sequence_logratio(grpo_args) -> bool:
        """Whether TRL's vLLM IS correction sums per-token log-ratios over each sequence (``sequence_*`` modes, or
        ``vespo``), the consumer the sampler-logprob preflight gates a nucleus-renormalized reference against."""
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
        """Force TRL's ``is_vllm_available`` True during TRL's ctor, since vLLM is not installed here; server mode, the
        only shape TRL builds that client for, is required first."""
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
        """TRL unfolds DTensors only under ``is_fsdp_enabled``, which the toolkit's ``fully_shard`` under accelerate
        MULTI_GPU never sets, so it would forward them verbatim and deadlock against the trainer-vLLM NCCL group."""
        if getattr(self, "vllm_generation", None) is None:
            raise RuntimeError(
                "TRL built no vllm_generation for this trainer, so the distributed-aware weight sync "
                "cannot be installed and every rollout would be generated by an engine that never "
                "receives the trained weights. Server-mode vLLM is required (use_vllm=True, "
                "vllm_mode='server') — check vllm_server_host/vllm_server_port reach a running server."
            )

        self.vllm_generation.sync_weights = types.MethodType(type(self)._distributed_sync_weights, self)
        config = self.parallelism_config
        if is_global_main_process():
            logger.info(
                "Installed distributed-aware vLLM weight sync "
                f"(ep={config.has_ep_group}, tp={config.is_tp_mode}, "
                f"expert_tp={config.is_expert_tp_mode})"
            )

    def _distributed_sync_weights(self) -> None:
        """COLLECTIVE. The client TRL holds on ``vllm_generation`` forwards from the main process."""
        client = self.vllm_generation.vllm_client if self.accelerator.is_main_process else None
        with self._phase_timers.span(Phase.SYNC, "host"), self.unbinned_scoring():
            sync_trainer_weights(self, client)

    def _generate_single_turn(self, *args, **kwargs):
        """TRL slices the result by ``process_index``, so TP ranks land on different completions and deadlock at the
        first all-reduce in ``model.forward``; broadcast from the TP leader."""
        with self._phase_timers.span(Phase.ROLLOUT, "host"):
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
        return self._rlrr_config is not None or self._drop_degenerate_groups or self._scale_rewards_std_floor > 0

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        """Broadcast the scored tensors across the TP/ETP group, since CPU-side post-processing may diverge."""
        with self._phase_timers.span(Phase.BUILD, "host"), self._scoring_round_in_bins():
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
        if self._packs_microbatches and self.model.training:
            result = self._lay_out_packed_round(result, self._take_round_plan(result))

        return self._with_row_spans(self._broadcast_tensors_from_tp_leader(result))

    def _apply_rlrr_advantages(self, result: dict[str, torch.Tensor | Any]) -> None:
        """Replace advantages with RLRR ones, recomputed on the full gathered reward set in TRL's group order. Train
        mode only."""
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
            rlrr_advantages_grouped(
                rewards.float().cpu().numpy(),
                group_size=self.num_generations,
                config=self._rlrr_config,
                lengths=full_lengths.float().cpu().numpy(),
                valid=(~unscorable).cpu().numpy(),
            )
        )

        self._install_advantages(result, advantages_full, "RLRR")

    def _apply_std_floor_advantages(self, result: dict[str, torch.Tensor | Any]) -> None:
        """Replace advantages with ones whose std divisor honours ``scale_rewards_std_floor``, on the full gathered
        set; TRL divides by a fixed ``std + 1e-4``. Train mode only."""
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
        """Mask all-equal-reward groups out of the loss: their advantage is 0, but their tokens would inflate the DAPO
        normalizer and dilute the groups that carry signal. Train mode only."""
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
        """Log the net token mass and, under ``balance_token_mass``, cancel it; a token weighs its place in the loss
        times TRL's vLLM IS ratio. Train mode only."""
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
        """``old_per_token_logps`` and ``ref_per_token_logps`` exist only when ``beta != 0`` and TRL's recompute gate
        fired, config verdicts every rank shares, so all or none reach the all-reduce."""
        old_logps = result.get("old_per_token_logps")
        ref_logps = result.get("ref_per_token_logps")
        if old_logps is None or ref_logps is None:
            return
        result["ref_per_token_logps"], clamped = clamp_ref_logps(ref_logps, old_logps)
        mode = "train" if self.model.training else "eval"
        clamped_tokens, loss_tokens = world_sums(kl_clamp_counts(clamped, effective_loss_mask(result)))
        self._metrics[mode][KL_CLAMP_FRAC_KEY].append(clamped_tokens / loss_tokens if loss_tokens else 0.0)

    def _gathered_rewards(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """``(rewards, unscorable)``, or ``None`` outside train; ``rewards`` aggregates as TRL's
        ``sum_then_normalize``. A missing stash in train mode raises rather than leave the hook unapplied."""
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
        """TRL's world-order process slice. Hooks compute on the full gathered set, where groups are contiguous, so a
        group may span two ranks."""
        start = self.accelerator.process_index * n_local
        return full[start : start + n_local]

    def _install_advantages(
        self, result: dict[str, torch.Tensor | Any], advantages_full: torch.Tensor, what: str
    ) -> None:
        """Slice to this rank, install, and realign TRL's advantage log. TRL fills ``_logs["advantages"]`` before any
        hook runs, in world order and length, so overwriting that tail realigns the record row for row."""
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
