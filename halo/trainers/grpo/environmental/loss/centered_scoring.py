"""Score centering in the environmental GRPO loss forward: the scorer returns a surrogate log-prob whose value is the
policy's and whose gradient is the centered estimator, so TRL's coefficient, masks, normalizers and the packed loss
scale apply unchanged."""

from collections.abc import Mapping

import torch

from halo.configs.grpo import ScoreCenteringConfig
from halo.trainers.grpo.environmental.environmental_rules import reject_unsupported_score_centering_loss
from halo.trainers.grpo.objective.logratio import SAMPLER_CERTAIN_LOGPROB
from halo.trainers.grpo.objective.loss_inputs import IS_SAMPLING_LOGPS_KEY
from halo.trainers.grpo.objective.score_centering import (
    TOPK_IDS_KEY,
    TOPK_LOGPS_KEY,
    ScoreCenteringMetrics,
    centered_logps,
)
from halo.trainers.metrics.token_metrics import effective_loss_mask
from halo.trainers.metrics.world_metrics import WorldMetrics

_METRIC_PREFIX = "score_centering"


class ScoreCenteringMixin:
    """Mixed in ahead of ``ProtectedTokenEntropyMixin``, beside ``LossForwardImportanceMixin``. The host provides
    ``async_config`` and calls :meth:`_configure_score_centering` once the model is placed. The loss forward of a
    train micro-batch is the only scoring call that centers, since the weights need the live head and a no-grad pass
    over the candidates would double its cost."""

    _score_centering: ScoreCenteringConfig | None = None
    _centered_batch: Mapping | None = None
    _centering_metrics: WorldMetrics | None = None

    def _configure_score_centering(self) -> None:
        """COLLECTIVE when the run's config turns score centering on, as it warms the candidate head form; first refuses
        a loss setup the surrogate cannot carry."""
        self._score_centering = self.async_config.build_score_centering_config()
        if self._score_centering is None:
            return
        reject_unsupported_score_centering_loss(
            self._score_centering.top_k, self.args, misaligned_accumulation=self._misaligned_accumulation
        )
        self._centering_metrics = WorldMetrics()
        self._warm_candidate_head_kernels()

    def _warm_candidate_head_kernels(self) -> None:
        """COLLECTIVE. The loss forward scores ``1 + K`` targets per position, a head form with tile graphs of its own
        (``chunked_head_logprobs``): one entropy-carrying pass with candidates armed compiles them here
        (:meth:`_warm_scoring_pass`)."""
        if not torch.cuda.is_available():
            return
        keep = 4
        ids = torch.zeros(2, 2 * keep, dtype=torch.long, device=self.accelerator.device)
        candidates = torch.zeros(2, keep, self._score_centering.top_k, dtype=torch.int32, device=ids.device)
        with self._scoring_with_candidates(candidates):
            self._warm_scoring_pass(ids, keep, compute_entropy=True)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """A train micro-batch's loss forward reads the sampler's candidates off ``inputs``."""
        if self._score_centering is None or not (model.training and isinstance(inputs, dict)):
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )
        self._centered_batch = inputs
        try:
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch
            )
        finally:
            self._centered_batch = None

    def training_step(self, model, inputs, num_items_in_batch=None):
        """COLLECTIVE on a round's last micro-step under score centering: the round's centering metrics fold across
        the ranks into one log."""
        loss = super().training_step(model, inputs, num_items_in_batch)
        if self._score_centering is not None and self._step % self.args.steps_per_generation == 0:
            self._centering_metrics.flush(self._metrics["train"])
        return loss

    def _get_per_token_logps_and_entropies(self, model, input_ids, attention_mask, logits_to_keep, *args, **kwargs):
        """TRL's one grad-enabled scoring call with entropy is the loss forward, the only one centered; the others
        (no-grad passes, the reference, warm-ups) pass through."""
        batch = self._centered_batch
        if batch is None or not (torch.is_grad_enabled() and kwargs.get("compute_entropy")):
            return super()._get_per_token_logps_and_entropies(
                model, input_ids, attention_mask, logits_to_keep, *args, **kwargs
            )
        for key in (TOPK_IDS_KEY, TOPK_LOGPS_KEY, IS_SAMPLING_LOGPS_KEY):
            if key not in batch:
                raise RuntimeError(
                    f"Score centering found no {key!r} in the loss batch: the rollout did not carry the sampler's "
                    "candidates to the loss forward."
                )
        with self._scoring_with_candidates(batch[TOPK_IDS_KEY]):
            scored = super()._get_per_token_logps_and_entropies(
                model, input_ids, attention_mask, logits_to_keep, *args, **kwargs
            )
        return (self._center(batch, scored[0]), *scored[1:])

    def _center(self, batch: Mapping, logps: torch.Tensor) -> torch.Tensor:
        """The surrogate ``lp_y.detach() + (X - X.detach())`` of the scored ``logps`` ``[rows, T, 1 + K]``: its value
        is the sampled token's log-prob, so TRL's ``exp(s - old)`` is unchanged, and its gradient is ``grad X``."""
        if logps.dim() != 3:
            raise RuntimeError(
                "Score centering needs the candidates' log-probs from the chunked head, but the loss forward scored "
                "the sampled token alone (a multimodal batch takes the full-logits path)."
            )
        config = self._score_centering
        lp_y, lp_head = logps[..., 0], logps[..., 1:]
        sampler_y = batch[IS_SAMPLING_LOGPS_KEY].to(lp_y.dtype)
        head_mask = batch[TOPK_IDS_KEY] >= 0
        # A token the sampler emitted with probability 1 made no sampling choice, and one without candidates has no
        # head to center on; both train on their plain score.
        active = torch.isfinite(sampler_y) & (sampler_y < SAMPLER_CERTAIN_LOGPROB) & head_mask.any(-1)
        centered, metrics = centered_logps(
            lp_y,
            lp_head,
            sampler_y,
            batch[TOPK_LOGPS_KEY].to(lp_y.dtype),
            head_mask,
            active,
            weights=config.weights,
            tis_clip=self.vllm_importance_sampling_clip_max,
            mis_low=config.mis_low,
            mis_high=config.mis_high,
        )
        self._record_centering(batch, metrics, active)
        return lp_y.detach() + (centered - centered.detach())

    def _record_centering(self, batch: Mapping, metrics: ScoreCenteringMetrics, active: torch.Tensor) -> None:
        """This micro-batch's sums over the loss tokens that were centered, and their share of the loss tokens, on the
        device until the round's flush."""
        loss_mask = effective_loss_mask(batch).bool()
        centered = loss_mask & active
        count = centered.sum()
        world = self._centering_metrics
        world.fraction(f"{_METRIC_PREFIX}/active_frac", count, loss_mask.sum())
        world.fraction(f"{_METRIC_PREFIX}/weight_zero_frac", (centered & (metrics.weight == 0)).sum(), count)
        for name, value in metrics.as_log_dict().items():
            world.fraction(f"{_METRIC_PREFIX}/{name}_mean", torch.where(centered, value.double(), 0.0).sum(), count)
