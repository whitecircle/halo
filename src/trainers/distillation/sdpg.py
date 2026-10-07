"""On-policy SDPG trainer (Self-Distilled Policy Gradient, arXiv:2606.04036).

Online GRPO plus a privileged self-distillation term: the same model is run as a privileged teacher
seeing a hint revealing the gold answer, and the student is distilled toward the teacher's full-vocab
next-token distribution via reverse KL ``D_KL(p ‖ SG[q])``, gated to positive-advantage rollouts::

    L = L_GRPO  +  beta(k) * L_OPD

The teacher is the same policy under ``torch.no_grad`` (no second model). Dataset rows carry the
``prompt`` (the RLVR script's ``process_for_rlvr`` renders it to text) and the gold ``answer`` the
hint reveals. Text-only (the hint is tokenizer-built).
"""

import logging

import torch

from src.args.mixins import format_field_names
from src.data.collators.self_distill import privileged_hint
from src.log import warn_once
from src.models.structure import resolve_tokenizer
from src.trainers.distillation.losses import global_token_mean, privileged_teacher_pass
from src.trainers.distillation.opd_term import OPDTermMixin
from src.trainers.grpo.online import DistributedGRPOTrainer
from src.trainers.mixins.dataloader import split_rows_head
from src.trainers.mixins.loss_masks import effective_loss_mask
from src.trainers.mixins.stored_metrics import StoredMetricsMixin

# Stdlib, not accelerate's adapter: the warning below fires per rollout row on whichever rank drew
# it, and the adapter drops everything off rank 0 (warn_once cannot pass main_process_only).
logger = logging.getLogger(__name__)


def positive_advantage_gate(
    loss_mask: torch.Tensor,
    advantages: torch.Tensor,
    enabled: bool,
) -> torch.Tensor:
    """The OPD token gate: the tokens the GRPO loss trains, optionally restricted to
    strictly-positive-advantage rows.

    Strict ``> 0``: a zero advantage means a tied or unscorable group, where the verifier expressed
    no preference, so those rows must not pull the student toward the teacher.

    Args:
        loss_mask: ``[B, C]`` mask, 1 on the trained completion tokens (``effective_loss_mask``).
        advantages: ``[B]`` or ``[B, 1]`` per-sample advantages.
        enabled: when False the gate is the loss mask alone.
    """
    if not enabled:
        return loss_mask
    adv = advantages.unsqueeze(1) if advantages.dim() == 1 else advantages
    return loss_mask * (adv > 0).to(loss_mask.dtype)


class DistributedSDPGTrainer(OPDTermMixin, StoredMetricsMixin, DistributedGRPOTrainer):
    """On-policy SDPG: online GRPO + privileged-teacher reverse-KL OPD on positive-advantage rollouts."""

    _tag_names = ["trl", "grpo", "sdpg"]

    _supports_pp = False
    _pp_unsupported_reason = (
        "it inherits online GRPO's cross-stage weight-sync and rollout-phase blockers, and adds a "
        "second no-grad forward of the whole model per batch on the [teacher_prompt ⧺ completion] "
        "sequence — a differently-shaped pass whose last-stage logits the student's loss consumes, "
        "which the pipeline's frozen P2P buffers cannot carry and its runtime has no channel to feed "
        "back into the step. The advantage-gated token denominator is knowable before the step from "
        "the rollout batch and is not a blocker"
    )

    def __init__(
        self,
        *args,
        sdpg_answer_field: str = "answer",
        opd_positive_advantage_only: bool = True,
        **kwargs,
    ):
        self._adopt_sdpg_arguments(kwargs)
        self.sdpg_answer_field = sdpg_answer_field
        self.opd_positive_advantage_only = opd_positive_advantage_only
        # Warned once for the whole run: a single answer-less row means the column is unusable.
        self._warned_missing_answer: set = set()

        super().__init__(*args, **kwargs)

        # Every rank constructs the trainer, so this raise is world-uniform; a per-rollout raise would
        # leave the peers of the rank that drew the bad row waiting in the next collective.
        columns = getattr(self.train_dataset, "column_names", None)
        hints_the_answer = "answer" in format_field_names(self.sdpg_hint_template)
        if (
            self.sdpg_beta_base != 0.0
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
        if self.sdpg_beta_base != 0.0:
            output["teacher_prompt_ids"], output["teacher_prompt_mask"] = self._build_teacher_prompts(
                output["prompt_ids"], output["prompt_mask"], answers
            )
        return output

    def _build_teacher_prompts(self, prompt_ids, prompt_mask, answers):
        """Left-padded ``[prompt + hint]`` token ids per rollout row (hint reveals the gold answer)."""
        tok = resolve_tokenizer(self.processing_class)
        rows = []
        for i in range(prompt_ids.size(0)):
            real = prompt_ids[i][prompt_mask[i].bool()].tolist()
            hint = privileged_hint(self.sdpg_hint_template, answer=answers[i])
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
        loss = super()._compute_loss(model, inputs)
        if self.sdpg_beta_base == 0.0:
            return loss
        if "teacher_prompt_ids" not in inputs:
            raise RuntimeError(
                f"sdpg_beta_base={self.sdpg_beta_base} but the batch carries no teacher_prompt_ids, so "
                f"the OPD term would be skipped and the step would train plain GRPO. "
                f"_generate_and_score_completions builds them; an override of it, or of the batch "
                f"buffering, must keep the teacher_prompt_* keys, or set sdpg_beta_base: 0 to drop the term."
            )

        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        student_logits = self._completion_logits(
            model, inputs["prompt_ids"], inputs["prompt_mask"], completion_ids, completion_mask
        )
        with privileged_teacher_pass(model):
            teacher_logits = self._completion_logits(
                model, inputs["teacher_prompt_ids"], inputs["teacher_prompt_mask"], completion_ids, completion_mask
            )

        opd_per_token = self.sdpg_loss_fn(student_logits, teacher_logits, self.sdpg_temperature).sum(-1)

        # The tokens the GRPO loss trains: a tool-output token is attention-valid but never trained.
        gate = positive_advantage_gate(
            effective_loss_mask(inputs).to(opd_per_token.dtype), inputs["advantages"], self.opd_positive_advantage_only
        )
        # Rows padding an eval split's final round repeat its first rows: the OPD mean reads the
        # split's own (in train, the whole batch, unsliced).
        real_rows = self.eval_split_rows(gate.size(0))
        opd = global_token_mean(split_rows_head(opd_per_token, real_rows), split_rows_head(gate, real_rows))

        beta = self._opd_beta()
        self.store_batch_metrics(self._opd_metrics(opd, beta), "train" if model.training else "eval", real_rows)
        # TRL normalizes inside _compute_loss, so an undivided per-microbatch OPD mean inflates beta
        # by grad_accum.
        if model.training:
            opd = opd / self.current_gradient_accumulation_steps
        return loss + beta * opd

    def _completion_logits(self, model, prefix_ids, prefix_mask, completion_ids, completion_mask):
        """Full-vocab logits predicting each completion token of ``[prefix + completion]`` → ``[B, C, V]``.

        Where the forward takes ``logits_to_keep`` (TRL's ``model_kwarg_keys``, as its own GRPO forward
        reads them) the head stops at the last ``C + 1`` positions, so the prompt's ``[B, P, V]`` rows are
        never built. The prefixes differ in length; aligning on the sequence end pairs the two forwards' rows.
        """
        keep = completion_ids.size(1) + 1
        input_ids = torch.cat([prefix_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prefix_mask, completion_mask], dim=1)
        trim = {"logits_to_keep": keep} if "logits_to_keep" in self.model_kwarg_keys else {}
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, **trim).logits
        return logits[:, -keep:-1, :]
