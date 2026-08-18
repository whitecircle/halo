"""Online GRPO trainer under EP / TP / EP+TP (no CP — use the SFT trainer for EP+CP).

Server-mode vLLM weight sync uses a vendored NCCL client; vLLM is not installed in the
training environment (ABI-incompatible extensions, older transformers than the image).
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

from src.args.mixins import AdvantageShaping, RLRRConfig
from src.distributed.nccl.clients.vllm import VLLMWeightSyncClient
from src.trainers.grpo.mixins.chunked_logprobs import ChunkedGRPOLogprobsMixin
from src.trainers.grpo.mixins.dataloader import GRPOTrainDataLoaderMixin
from src.trainers.grpo.mixins.entropy_mask import ProtectedTokenEntropyMixin
from src.trainers.grpo.mixins.generation_buffer import GRPOGenerationBufferMixin
from src.trainers.grpo.mixins.on_policy_init import OnPolicyGRPOInitMixin
from src.trainers.grpo.objective.advantages import group_relative_advantages
from src.trainers.grpo.objective.application import (
    DEGENERATE_GROUP_FRAC_KEY,
    degenerate_drop_rows,
    gathered_num_items,
    narrow_loss_masks,
)
from src.trainers.grpo.objective.logratio import clamp_ref_logps
from src.trainers.grpo.objective.relative_rewards import relative_advantages_grouped
from src.trainers.grpo.rollout.completions_logging import log_with_decoupled_completions
from src.trainers.grpo.rollout.weight_sync import sync_trainer_weights, validate_weight_sync_support
from src.trainers.mixins.base import DistributedTrainerMixin
from src.trainers.mixins.loss_masks import effective_loss_mask

logger = get_logger(__name__, log_level="info")


def _vllm_available_stub() -> bool:
    """Stand in for TRL's ``is_vllm_available`` while the vendored NCCL-only client is installed."""
    return True


class DistributedGRPOTrainer(
    OnPolicyGRPOInitMixin,
    GRPOTrainDataLoaderMixin,
    GRPOGenerationBufferMixin,
    ProtectedTokenEntropyMixin,
    ChunkedGRPOLogprobsMixin,
    DistributedTrainerMixin,
    GRPOTrainer,
):
    """Online GRPO trainer with EP / TP / EP+TP support.

    CP is not supported; use ``DistributedSFTTrainer`` for EP+CP.
    """

    _supports_pp = False
    _pp_unsupported_reason = (
        "the vLLM weight sync gathers and sends the model's full named_parameters() from one "
        "rank-set in a fixed collective order, but under PP no rank holds all the layers, so the "
        "sync needs per-stage gathers assembled across stages; the rollout phase also forwards the "
        "TRAINING model outside compute_loss (old / reference per-token log-probs), which would have "
        "to run as a forward-only pass across the whole pipeline. The per-microbatch update itself "
        "is a single forward over precomputed advantages and reference log-probs, so the loss is not "
        "the blocker"
    )
    # No rollout backend of its own: online GRPO is vLLM-only by construction, so the weight-sync
    # layout resolver falls through to the base client's default.
    _rollout_backend: str | None = None
    # The objective never passes labels into the forward, so Liger CE/FLCE cannot fire.
    _loss_outside_model_forward = True

    def __init__(self, *args, **kwargs):
        training_args, kwargs = self._begin_on_policy_init(args, kwargs)

        self._rlrr_config: RLRRConfig | None = kwargs.pop("rlrr_config", None)

        # Same recompute-and-reslice hook as RLRR, hence mutually exclusive with it.
        # ``build_advantage_shaping`` already returns None at the default 'mean' mode.
        self._advantage_shaping: AdvantageShaping | None = kwargs.pop("advantage_shaping", None)
        if self._advantage_shaping is not None and self._rlrr_config is not None:
            raise ValueError("advantage_shaping and rlrr_config both replace the advantages — set only one.")
        self._drop_degenerate_groups: bool = kwargs.pop("drop_degenerate_groups", False)

        # Range-validated by RangeValidatedConfig._validate_ranges (finiteness included).
        self._scale_rewards_std_floor: float = kwargs.pop("scale_rewards_std_floor", 0.0)
        if self._scale_rewards_std_floor > 0 and self._rlrr_config is not None:
            raise ValueError(
                "scale_rewards_std_floor and rlrr_config cannot be combined: RLRR replaces the "
                "group-normalized advantages wholesale with relative-ranking ones, which never divide "
                "by a reward std, so the floor would silently do nothing. Set only one."
            )

        self._last_rewards_per_func: torch.Tensor | None = None

        self._use_chunked_grpo_logprobs = kwargs.pop("use_chunked_grpo_logprobs", False)

        self._save_completions = kwargs.pop("save_completions", True)

        self._require_vllm_server_mode(training_args)

        with self._patch_trl_for_vendored_vllm_client(training_args):
            super().__init__(*args, **kwargs)

        if self._rlrr_config is not None:
            logger.info("RLRR relative-reward shaping enabled (mode=%s)", self._rlrr_config.mode)

        # ``reward_funcs`` is populated by TRL's ctor, so this gate can only be read post-super().
        shaping_mode = self._advantage_shaping.mode if self._advantage_shaping is not None else None
        if shaping_mode == "neg_mask_hard" and len(self.reward_funcs) > 1:
            logger.warning(
                "advantage_mode='neg_mask_hard' with multiple reward functions: the hard-group gate uses "
                "the TOTAL weighted reward (no objective decomposition here) — set advantage_hard_group_threshold "
                "for that scale, or keep a single (objective) reward function."
            )

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

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """Emit the completions parquet (``save_completions``) decoupled from the console table."""
        log_with_decoupled_completions(self, logs, start_time, super().log, save_completions=self._save_completions)

    @staticmethod
    def _is_vllm_server_mode(grpo_args) -> bool:
        """Detect whether this trainer was constructed for server-mode vLLM.

        Reads the declared ``GRPOConfig`` fields directly — a getattr default here and in
        :meth:`_require_vllm_server_mode` could disagree and leave TRL unpatched in a shape the
        requirement gate had already accepted.
        """
        return grpo_args.use_vllm and grpo_args.vllm_mode == "server"

    @staticmethod
    def _require_vllm_server_mode(grpo_args) -> None:
        """Require vLLM server-mode; reject in-process HF generation (``use_vllm=False``, slow +
        rank-divergent under FSDP2/EP) and colocate (no vLLM in the training image). The raised
        messages carry the rationale."""
        if grpo_args is None:
            raise ValueError(
                "DistributedGRPOTrainer requires a GRPOConfig: pass it as the `args` keyword or in "
                "the conventional positional slot. Without it the vLLM server-mode requirement, the "
                "parallelism config and the Liger overrides all silently go unapplied."
            )
        if not grpo_args.use_vllm:
            raise ValueError(
                "DistributedGRPOTrainer requires vLLM server-mode generation: set use_vllm=True "
                "and vllm_mode='server'. In-process HF generation (use_vllm=False) is not "
                "supported — it is slow and desyncs ranks under FSDP2/EP. Run a separate vLLM "
                "container (docker-compose.vllm.yml) and point vllm_server_host/vllm_server_port "
                "at it (weights sync over NCCL)."
            )
        if grpo_args.vllm_mode != "server":
            raise ValueError(
                f"DistributedGRPOTrainer supports only vllm_mode='server', got "
                f"'{grpo_args.vllm_mode}'. Colocate mode would launch an in-process vLLM on the "
                f"training GPUs, but the training image has no vLLM (ABI-incompatible "
                f"with the training PyTorch/Transformers). Run a separate vLLM "
                f"container (docker-compose.vllm.yml) and set vllm_mode='server' "
                f"with vllm_server_host/vllm_server_port (weights sync over NCCL)."
            )

    @contextlib.contextmanager
    def _patch_trl_for_vendored_vllm_client(self, grpo_args) -> Iterator[None]:
        """Swap TRL's vLLM client + availability checks for the duration of init.

        vLLM isn't installed, so force ``is_vllm_available`` True and substitute the vendored
        NCCL-only ``VLLMWeightSyncClient``. Reverted after ``super().__init__``.
        """
        if not self._is_vllm_server_mode(grpo_args):
            yield
            return

        originals = (
            _trl_vllm_generation.is_vllm_available,
            _trl_vllm_client.is_vllm_available,
            _trl_vllm_generation.VLLMClient,
        )
        _trl_vllm_generation.is_vllm_available = _vllm_available_stub
        _trl_vllm_client.is_vllm_available = _vllm_available_stub
        _trl_vllm_generation.VLLMClient = VLLMWeightSyncClient
        try:
            yield
        finally:
            (
                _trl_vllm_generation.is_vllm_available,
                _trl_vllm_client.is_vllm_available,
                _trl_vllm_generation.VLLMClient,
            ) = originals

    def _generate_single_turn(self, prompt_ids, images, multimodal_fields):
        """Override TRL's single-turn vLLM generation to be TP-consistent.

        TRL slices the result by ``process_index``, so TP ranks land on different completions and
        deadlock at the first all-reduce in ``model.forward``. Broadcast the tuple from the TP leader.
        """
        result = super()._generate_single_turn(prompt_ids, images, multimodal_fields)
        return self._broadcast_object_from_tp_leader(result)

    def _generate_and_score_completions(
        self, inputs: list[dict[str, torch.Tensor | Any]]
    ) -> dict[str, torch.Tensor | Any]:
        """Generate, score, and broadcast scoring tensors across the TP/ETP group.

        Generation is already broadcast in ``_generate_single_turn``; this post-call broadcast
        guards against non-deterministic CPU-side post-processing diverging a tensor between ranks.
        """
        result = super()._generate_and_score_completions(inputs)

        # Before the TP/ETP broadcast, so the reshaped tensors are what gets broadcast.
        self._apply_rlrr_advantages(result)
        self._apply_advantage_shaping(result)
        self._apply_degenerate_group_drop(result)

        # k3 tail clamp; both tensors exist only when beta != 0 AND TRL's recompute gate fired.
        old_logps = result.get("old_per_token_logps")
        ref_logps = result.get("ref_per_token_logps")
        if old_logps is not None and ref_logps is not None:
            result["ref_per_token_logps"], kl_clamp_frac = clamp_ref_logps(ref_logps, old_logps)
            mode = "train" if self.model.training else "eval"
            self._metrics[mode]["kl_clamp_frac"].append(kl_clamp_frac.item())

        return self._broadcast_tensors_from_tp_leader(result)

    def _calculate_rewards(self, *args, **kwargs):
        """Capture the (gathered) per-function rewards so the advantage hooks see the full group set.

        TRL gathers ``rewards_per_func`` across processes before group-normalizing; stashing it here
        gives the RLRR / advantage-shaping / degenerate-drop hooks the same full reward set.
        """
        rewards_per_func = super()._calculate_rewards(*args, **kwargs)
        if self._recomputes_from_gathered_rewards:
            self._last_rewards_per_func = rewards_per_func
        return rewards_per_func

    @property
    def _recomputes_from_gathered_rewards(self) -> bool:
        """Whether any hook re-derives advantages/masks from the stashed full reward set.

        The stash in :meth:`_calculate_rewards` and every consumer below read this same gate. A knob
        covered by one but not the other leaves the stash ``None``, so the consumer early-returns and
        the setting never reaches the math.
        """
        return (
            self._rlrr_config is not None
            or self._advantage_shaping is not None
            or self._drop_degenerate_groups
            or self._scale_rewards_std_floor > 0
        )

    def _gathered_rewards(self) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Stashed full (gathered) rewards as ``(rewards, unscorable)``, or ``None`` outside train.

        ``rewards`` aggregates the reward functions as TRL's ``sum_then_normalize`` branch does
        (weighted nansum), in the gathered process order; ``unscorable`` marks rows where every
        reward fn returned NaN (TRL forces their advantage to 0).
        """
        if not self.model.training or self._last_rewards_per_func is None:
            return None
        if self.args.multi_objective_aggregation != "sum_then_normalize":
            raise ValueError(
                "advantage_shaping / RLRR / drop_degenerate_groups recompute rewards with TRL's "
                "sum_then_normalize aggregation; multi_objective_aggregation="
                f"{self.args.multi_objective_aggregation!r} would silently diverge."
            )
        rewards_per_func = self._last_rewards_per_func
        weights = self.reward_weights.to(rewards_per_func.device).unsqueeze(0)
        rewards = (rewards_per_func * weights).nansum(dim=1)  # [total], gathered order
        return rewards, torch.isnan(rewards_per_func).all(dim=1)

    def _local_slice(self, full: torch.Tensor, n_local: int, what: str) -> torch.Tensor:
        """This rank's rows of a gathered per-completion tensor; asserts whole groups per rank."""
        if n_local % self.num_generations != 0:
            raise ValueError(
                f"{what} requires each rank's rollout count ({n_local}) to be a multiple of "
                f"num_generations ({self.num_generations}); a group is split across ranks. Use the "
                f"default generation_batch_size derivation or make it divisible by "
                f"num_processes * num_generations."
            )
        start = self.accelerator.process_index * n_local
        return full[start : start + n_local]

    def _apply_advantage_shaping(self, result: dict[str, torch.Tensor | Any]) -> None:
        """Replace TRL's group-normalized advantages with the shaped ones (see :class:`AdvantageShaping`),
        recomputed on the full gathered reward set and re-sliced to this rank. Train mode only.

        Also the only path that applies ``scale_rewards_std_floor``, so it runs for the floor alone
        (``shaping=None`` is a supported input to :func:`group_relative_advantages`) — TRL's own
        advantages are computed with a fixed ``1e-4`` divisor and cannot honour the floor.
        """
        if self._advantage_shaping is None and self._scale_rewards_std_floor <= 0:
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
            shaping=self._advantage_shaping,
            gate_rewards=rewards.float(),
            already_gathered=True,
            std_floor=self._scale_rewards_std_floor,
        )
        advantages_full = advantages_full.masked_fill(unscorable, 0.0)  # match TRL's unscorable handling
        self._install_advantages(result, advantages_full, "advantage_shaping")

    def _install_advantages(
        self, result: dict[str, torch.Tensor | Any], advantages_full: torch.Tensor, what: str
    ) -> None:
        """Slice ``advantages_full`` to this rank, install it, and realign TRL's advantage log.

        Both advantage-replacement hooks recompute on the full gathered set, so both must re-slice
        and both must re-log; doing one without the other trains on values the logged record does
        not carry.

        TRL fills ``_logs["advantages"]`` with its own group-normalized values inside
        ``_generate_and_score_completions``, before either hook runs. The gathered tensor carries the
        same world order and length TRL appended, so overwriting that tail realigns the record row
        for row.
        """
        local = self._local_slice(advantages_full, result["advantages"].shape[0], what)
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
        drop_full, degenerate_frac = degenerate_drop_rows(rewards, self.num_generations, valid_mask=~unscorable)
        drop = self._local_slice(drop_full, result["completion_mask"].shape[0], "drop_degenerate_groups")
        self._metrics["train"][DEGENERATE_GROUP_FRAC_KEY].append(degenerate_frac)
        # Narrow every mask TRL composes into its loss mask, else untrained tokens inflate the normalizer.
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

    def _apply_rlrr_advantages(self, result: dict[str, torch.Tensor | Any]) -> None:
        """Replace group-normalized advantages with RLRR relative-ranking advantages.

        Recomputes on the full gathered reward set (matching TRL's group-normalization order), then
        re-slices to this process. In-place on ``result["advantages"]``. Train mode only.
        """
        if self._rlrr_config is None:
            return
        gathered = self._gathered_rewards()
        if gathered is None:
            return
        rewards, unscorable = gathered

        # Same process order as the rewards, so the length re-ranking sees each group's members.
        local_lengths = result["completion_mask"].sum(dim=1).to(rewards.device)
        full_lengths = self.accelerator.gather(local_lengths)

        advantages_full = torch.from_numpy(
            relative_advantages_grouped(
                rewards.float().cpu().numpy(),
                group_size=self.num_generations,
                config=self._rlrr_config,
                lengths=full_lengths.float().cpu().numpy(),
            )
        ).masked_fill(unscorable.cpu(), 0.0)

        self._install_advantages(result, advantages_full, "RLRR")

    def _setup_weight_sync(self) -> None:
        """Replace ``VLLMGeneration.sync_weights`` with the distributed-aware version.

        TRL only unfolds DTensors when ``is_fsdp_enabled`` is set, but the toolkit applies FSDP2 via
        ``fully_shard`` under accelerate MULTI_GPU, so it forwards DTensors verbatim and deadlocks
        against the trainer↔vLLM NCCL group. The replacement gathers EP then TP shards, then broadcasts.

        Also the construction gate for syncable weights: a quantized (QLoRA) base must fail here, not
        as an opaque server-side error at the first sync.
        """
        validate_weight_sync_support(self.model)
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
            f"(ep={config.is_ep_mode}, tp={config.is_tp_mode}, "
            f"expert_tp={config.is_expert_tp_mode})"
        )

    def _distributed_sync_weights(self) -> None:
        """EP/TP/FSDP2/PEFT-aware replacement for ``VLLMGeneration.sync_weights``.

        All ranks join the gathers; only global-main (and, under TP, TP-rank 0) forwards. The client
        is held on ``vllm_generation`` here, whereas the environmental trainer holds one directly.
        """
        client = self.vllm_generation.vllm_client if self.accelerator.is_main_process else None
        sync_trainer_weights(self, client)
