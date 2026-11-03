"""On-policy SDPG (arXiv:2606.04036): ``L = L_GRPO + beta(k) * L_OPD``. The same policy, no-grad on a hint revealing
the gold ``answer``, is the teacher of a reverse KL gated to positive-advantage rollouts. Text-only."""

import logging

import torch

from halo.data.collators.self_distill import format_field_names, privileged_hint
from halo.log import warn_once
from halo.model_loading.tokenizer_setup import resolve_tokenizer
from halo.trainers.distillation.divergences import global_token_mean, token_divergence
from halo.trainers.distillation.opd_term import OPDTermMixin, privileged_teacher_pass
from halo.trainers.evaluation import split_rows_head
from halo.trainers.grpo.online import DistributedOnlineGRPOTrainer
from halo.trainers.stored_metrics import StoredMetricsMixin
from halo.trainers.token_metrics import effective_loss_mask

logger = logging.getLogger(__name__)


def positive_advantage_gate(
    loss_mask: torch.Tensor,
    advantages: torch.Tensor,
    enabled: bool,
) -> torch.Tensor:
    """``loss_mask``, restricted when ``enabled`` to rows of strictly positive advantage. A zero advantage is a tied or
    unscorable group, where the verifier expressed no preference to pull the student toward."""
    if not enabled:
        return loss_mask
    adv = advantages.unsqueeze(1) if advantages.dim() == 1 else advantages
    return loss_mask * (adv > 0).to(loss_mask.dtype)


class DistributedSDPGTrainer(OPDTermMixin, StoredMetricsMixin, DistributedOnlineGRPOTrainer):
    _tag_names = ["trl", "grpo", "sdpg"]

    PP_REFUSAL = (
        "it inherits online GRPO's cross-stage weight-sync and rollout-phase blockers, and adds a "
        "second no-grad forward of the whole model per batch on the [teacher_prompt ⧺ completion] "
        "sequence — a differently-shaped pass whose last-stage logits the student's loss consumes, "
        "which the pipeline's frozen P2P buffers cannot carry and its runner has no channel to feed "
        "back into the step. The advantage-gated token denominator is knowable before the step from "
        "the rollout batch and is not a blocker"
    )

    def __init__(
        self,
        *args,
        sdpg_answer_field: str,
        opd_positive_advantage_only: bool,
        **kwargs,
    ):
        self._adopt_sdpg_arguments(kwargs)
        self.sdpg_answer_field = sdpg_answer_field
        self.opd_positive_advantage_only = opd_positive_advantage_only
        self._warned_missing_answer: set = set()

        super().__init__(*args, **kwargs)

        # Every rank constructs the trainer, so this raise is world-uniform; a per-rollout raise would
        # leave the peers of the rank that drew the bad row waiting in the next collective.
        columns = getattr(self.train_dataset, "column_names", None)
        hints_the_answer = "answer" in format_field_names(self.sdpg.sdpg_hint_template)
        if (
            self.sdpg.sdpg_beta_base != 0.0
            and hints_the_answer
            and columns is not None
            and self.sdpg_answer_field not in columns
        ):
            raise ValueError(
                f"SDPG's privileged teacher needs the gold answer in column "
                f"{self.sdpg_answer_field!r}, which the train dataset does not carry (has: "
                f"{sorted(columns)}). Every hint would assert an EMPTY answer as fact and the OPD "
                f"term would distil toward a misled teacher. Set sdpg_answer_field, or "
                f"sdpg_beta_base: 0 to drop the term."
            )

    def _generate_and_score_completions(self, inputs):
        # Answers in prompt row order, aligned 1:1 with rollout rows (RepeatSampler already expanded).
        answers = [x.get(self.sdpg_answer_field) for x in inputs]
        output = super()._generate_and_score_completions(inputs)
        if self.sdpg.sdpg_beta_base != 0.0:
            output["teacher_prompt_ids"], output["teacher_prompt_mask"] = self._build_teacher_prompts(
                output["prompt_ids"], output["prompt_mask"], answers
            )
        return output

    def _build_teacher_prompts(self, prompt_ids, prompt_mask, answers):
        """Left-padded ``[prompt + hint]`` token ids per rollout row."""
        tok = resolve_tokenizer(self.processing_class)
        rows = []
        for i in range(prompt_ids.size(0)):
            real = prompt_ids[i][prompt_mask[i].bool()].tolist()
            hint = privileged_hint(self.sdpg.sdpg_hint_template, answer=answers[i])
            if hint is None:
                # Warned, not raised: this runs per rollout row on one rank.
                warn_once(
                    logger,
                    self._warned_missing_answer,
                    "missing_answer",
                    "SDPG: a rollout row carries no %r value, so its privileged teacher runs on the "
                    "plain prompt (no hint). The OPD term degenerates to self-distillation for those "
                    "rows — check the dataset's answer column.",
                    self.sdpg_answer_field,
                )
            rows.append(real + ([] if hint is None else tok.encode(hint, add_special_tokens=False)))
        max_len = max(len(r) for r in rows)
        ids = torch.full((len(rows), max_len), tok.pad_token_id, dtype=prompt_ids.dtype, device=prompt_ids.device)
        mask = torch.zeros((len(rows), max_len), dtype=prompt_mask.dtype, device=prompt_mask.device)
        for i, r in enumerate(rows):  # left-pad (matches GRPO prompt padding side)
            ids[i, max_len - len(r) :] = torch.tensor(r, dtype=prompt_ids.dtype, device=prompt_ids.device)
            mask[i, max_len - len(r) :] = 1
        return ids, mask

    def _compute_loss(self, model, inputs):
        if self.sdpg.sdpg_beta_base == 0.0:
            return super()._compute_loss(model, inputs)
        if "teacher_prompt_ids" not in inputs:
            raise RuntimeError(
                f"sdpg_beta_base={self.sdpg.sdpg_beta_base} but the batch carries no teacher_prompt_ids, so "
                f"the OPD term would be skipped and the step would train plain GRPO. "
                f"_generate_and_score_completions builds them; an override of it, or of the batch "
                f"buffering, must keep the teacher_prompt_* keys, or set sdpg_beta_base: 0 to drop the term."
            )

        # The GRPO loss forward hands its completion logits back, so the student runs forward and backward once.
        with self._capturing_completion_logits() as student_logits:
            loss = super()._compute_loss(model, inputs)
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        rows = [part.size(0) for part in student_logits]
        if sum(rows) != completion_ids.size(0) or any(
            part.size(1) != completion_ids.size(1) for part in student_logits
        ):
            raise RuntimeError(
                f"The OPD term reads the student's completion logits off the GRPO loss forward, which handed "
                f"back {[tuple(part.shape[:2]) for part in student_logits]} for a {tuple(completion_ids.shape)} "
                f"completion batch: TRL's _compute_loss no longer scores the policy through "
                f"_get_per_token_logps_and_entropies, or an override forwards around it. Keep the loss forward "
                f"on that scorer, or set sdpg_beta_base: 0 to drop the term."
            )
        with privileged_teacher_pass(model):
            teacher_logits = self._completion_logits(
                model, inputs["teacher_prompt_ids"], inputs["teacher_prompt_mask"], completion_ids, completion_mask
            )
        opd_per_token = torch.cat(
            [
                token_divergence(self.sdpg_loss_fn, student, teacher, self.sdpg.sdpg_temperature)
                for student, teacher in zip(student_logits, teacher_logits.split(rows), strict=True)
            ]
        )

        # The tokens the GRPO loss trains: a tool-output token is attention-valid but never trained.
        gate = positive_advantage_gate(
            effective_loss_mask(inputs).to(opd_per_token.dtype), inputs["advantages"], self.opd_positive_advantage_only
        )
        # Rows padding an eval split's final round repeat its first rows, so the OPD mean reads the split's own.
        real_rows = self.eval_split_rows(gate.size(0))
        opd = global_token_mean(split_rows_head(opd_per_token, real_rows), split_rows_head(gate, real_rows))

        beta = self._opd_beta()
        self.store_batch_metrics(self._opd_metrics(opd, beta), "train" if model.training else "eval", real_rows)
        # TRL normalizes inside _compute_loss, so an undivided per-microbatch OPD mean inflates beta by grad_accum.
        if model.training:
            opd = opd / self.current_gradient_accumulation_steps
        return loss + beta * opd

    def _completion_logits(self, model, prefix_ids, prefix_mask, completion_ids, completion_mask):
        """``[B, C, V]``. Where the forward takes ``logits_to_keep`` the prompt's rows are never built. The prefixes
        differ in length; aligning on the sequence end pairs these rows with the loss forward's."""
        keep = completion_ids.size(1) + 1
        input_ids = torch.cat([prefix_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prefix_mask, completion_mask], dim=1)
        trim = {"logits_to_keep": keep} if "logits_to_keep" in self.model_kwarg_keys else {}
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, **trim).logits
        return logits[:, -keep:-1, :]
