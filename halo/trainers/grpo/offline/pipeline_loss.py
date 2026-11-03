"""Offline GRPO under pipeline parallelism: the last stage's loss contract and the pipeline reference sweep."""

import torch

from halo.data.collators.fixed_shape import pad_to_fixed_length
from halo.data.collators.offline_grpo import REF_PER_TOKEN_LOGPS_COLUMN
from halo.data.completion_spans import completion_labels
from halo.kernels.logprobs import LABEL_IGNORE_INDEX, loss_token_counts_per_row, token_logprobs
from halo.runtime.consensus import DeferredRankFailure
from halo.trainers.grpo.objective.offline_loss import offline_loss_normalizer, offline_loss_numerator
from halo.trainers.parallelism.pipeline_loss_adapter import PipelineLossAdapter, rows_with_labels


class OfflineGRPOPipelineMixin:
    """Over the host's ``_joined_rows``, ``_token_objective``, ``_live_min_log_prob`` and ``_record_sign_metrics``.
    Mixed in ahead of the spine's pipeline mixin, whose ``_configure_pipeline_parallel`` builds the adapter once."""

    # Non-None only while the PP reference sweep drives the schedule: the last stage's loss stashes
    # its per-token log-probs here instead of computing the objective.
    _pp_ref_sweep: list[torch.Tensor] | None = None

    def _pp_loss_adapter(self) -> PipelineLossAdapter:
        """Every denominator reads batch metadata alone, so the microbatch numerators share the runner's one per-step
        normalizer. At ``kl_beta != 0`` the reference log-probs ship as a third side tensor. Seeds the PP-only live
        clamp floor."""
        self._pp_min_log_prob: float | None = self.min_log_prob
        extra_target_keys = ("advantage", "group_size")
        if self.beta != 0.0:
            extra_target_keys += (REF_PER_TOKEN_LOGPS_COLUMN,)
        return PipelineLossAdapter(
            token_loss_fn=self._pp_token_loss,
            batch_transform=self._pp_batch_transform,
            normalizer=self._pp_normalizer,
            extra_target_keys=extra_target_keys,
            # Row filler for partial eval batches: with 0, an inert row's 1/group_size is inf → NaN.
            pad_spec={"group_size": 1},
            eval_normalizer=self._pp_normalizer,
        )

    def _pp_token_loss(self, logits: torch.Tensor, target: dict[str, torch.Tensor]) -> torch.Tensor:
        """The offline-GRPO numerator over one microbatch; the runner divides by ``_pp_normalizer``. While
        :meth:`_pp_score_reference_batch` drives the schedule it stashes the raw log-probs instead."""
        token_logps, mask = token_logprobs(logits, target["labels"])
        if self._pp_ref_sweep is not None:
            self._pp_ref_sweep.append(token_logps.detach())
            return token_logps.new_zeros(())

        advantages = target["advantage"].float()
        ref_logps = target[REF_PER_TOKEN_LOGPS_COLUMN] if self.beta != 0.0 else None
        per_token_loss, sample_values = self._token_objective(
            token_logps, advantages, ref_logps, self._pp_min_log_prob
        )
        self._record_sign_metrics(sample_values, advantages, mask, rows=rows_with_labels(target["labels"]))
        return offline_loss_numerator(per_token_loss, mask, target["group_size"], loss_type=self.loss_type)

    def _pp_batch_transform(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Right-pads to ``args.max_length`` (the P2P shapes freeze on the first step). With a KL term the reference
        log-probs land on the last stage's shifted grid: completion token ``j`` at input position ``P + j`` is scored
        one position earlier. Only the reference sweep calls this before the column exists."""
        input_ids, attention_mask, _ = self._joined_rows(batch)
        prompt_ids = batch["prompt_input_ids"]
        completion_ids, completion_mask = batch["completion_input_ids"], batch["completion_attention_mask"]
        labels = torch.cat(
            [torch.full_like(prompt_ids, LABEL_IGNORE_INDEX), completion_labels(completion_ids, completion_mask)],
            dim=1,
        )
        padded = pad_to_fixed_length(
            {
                "input_ids": (input_ids, self.padding_value),
                "attention_mask": (attention_mask, 0),
                "labels": (labels, LABEL_IGNORE_INDEX),
            },
            self.args.max_length,
            "The fixed shape is max_prompt_length + max_completion_length; raise them to fit the rows.",
        )
        self._pp_min_log_prob = self._live_min_log_prob
        out = {**padded, "advantage": batch["advantage"], "group_size": batch["group_size"]}
        if self.beta != 0.0 and self._pp_ref_sweep is None:
            ref_logps = batch.get(REF_PER_TOKEN_LOGPS_COLUMN)
            if ref_logps is None:
                raise RuntimeError(
                    f"kl_beta={self.beta} under pipeline parallelism needs the '{REF_PER_TOKEN_LOGPS_COLUMN}' "
                    f"column the construction-time reference sweep writes, but this batch carries none — "
                    f"the dataset was replaced after construction without it."
                )
            prompt_width = prompt_ids.size(1)
            shifted = ref_logps.new_zeros((input_ids.size(0), self.args.max_length - 1), dtype=torch.float32)
            shifted[:, prompt_width - 1 : prompt_width - 1 + ref_logps.size(1)] = ref_logps.float()
            out[REF_PER_TOKEN_LOGPS_COLUMN] = shifted
        return out

    def _pp_normalizer(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Shifted completion-token counts equal the unshifted ones ``compute_loss`` reads: the collator guarantees a
        non-empty prompt, so position 0 is never a completion token."""
        return offline_loss_normalizer(
            loss_token_counts_per_row(inputs["labels"]).float(),
            inputs["group_size"],
            loss_type=self.loss_type,
            max_completion_length=self.max_completion_length,
        )

    def _pp_score_reference_batch(self, batch) -> list[torch.Tensor]:
        frozen_rows = self.args.per_device_train_batch_size
        pads = self._pp_frozen_row_pads()
        real_rows = batch["prompt_input_ids"].size(0)
        prompt_width = batch["prompt_input_ids"].size(1)
        completion_width = batch["completion_input_ids"].size(1)
        valid = batch["completion_attention_mask"].bool().cpu()
        # Signals both the transform (no reference column yet) and last-stage loss (stash scores).
        self._pp_ref_sweep = []
        try:
            guard = DeferredRankFailure("Preparing a pipeline KL reference batch")
            inputs = guard.run(lambda: self._pp_pad_rows_to_frozen(self._pp_batch_transform(batch), frozen_rows, pads))
            guard.reject()
            logps = self._pp_reference_sweep_pass(inputs)
            completion_logps = logps[:real_rows, prompt_width - 1 : prompt_width - 1 + completion_width].float().cpu()
            return [values[mask] for values, mask in zip(completion_logps, valid, strict=True)]
        finally:
            self._pp_ref_sweep = None

    def _pp_reference_sweep_pass(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """COLLECTIVE across the chain and the stage's replicas. One forward-only pass whose last-stage loss stashes
        its per-token log-probs, which every rank receives."""
        self._pp_ref_sweep = []
        self._pp_runner.eval_loss(
            inputs["input_ids"],
            inputs["labels"],
            attention_mask=inputs["attention_mask"],
            num_items_in_batch=1.0,
            extra_targets={key: inputs[key] for key in ("advantage", "group_size")},
        )
        stashed = torch.cat(self._pp_ref_sweep) if self.parallelism_config.is_last_pp_stage else None
        self._pp_ref_sweep = None
        return self._pp_broadcast_output_from_last_stage(stashed)
