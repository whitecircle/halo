"""On-policy SDPG trainer (Self-Distilled Policy Gradient, arXiv:2606.04036).

Online GRPO plus a privileged self-distillation term: the same model is run as a privileged teacher
seeing a hint revealing the gold answer, and the student is distilled toward the teacher's full-vocab
next-token distribution via reverse KL ``D_KL(p ‖ SG[q])``, gated to positive-advantage rollouts::

    L = L_GRPO  +  beta(k) * L_OPD

The teacher is the same policy under ``torch.no_grad`` (no second model). Dataset is online-GRPO
format ``{"prompt": [...], "answer": "..."}``. Text-only (the hint is tokenizer-built).
"""

import logging
from dataclasses import fields

import torch

from src.args.mixins import SDPGArguments
from src.log import warn_once
from src.models.structure import resolve_tokenizer
from src.trainers.distillation.losses import (
    beta_warmup_decay,
    get_self_distillation_loss_fn,
    positive_advantage_gate,
    privileged_teacher_pass,
)
from src.trainers.grpo.online import DistributedGRPOTrainer
from src.trainers.mixins.stored_metrics import StoredMetricsMixin
from src.trainers.mixins.validation import ctor_config, disable_trl_liger

# Stdlib, not accelerate's adapter: the warning below fires per rollout row on whichever rank drew
# it, and the adapter drops everything off rank 0 (warn_once cannot pass main_process_only).
logger = logging.getLogger(__name__)


class DistributedSDPGTrainer(StoredMetricsMixin, DistributedGRPOTrainer):
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
        # Tunables arrive under the names and defaults SDPGArguments declares: popped by its fields
        # and adopted as attributes.
        sdpg = SDPGArguments(**{f.name: kwargs.pop(f.name) for f in fields(SDPGArguments) if f.name in kwargs})
        vars(self).update(vars(sdpg))
        self.sdpg_loss_fn = get_self_distillation_loss_fn(self.sdpg_loss)
        self.sdpg_answer_field = sdpg_answer_field
        self.opd_positive_advantage_only = opd_positive_advantage_only
        # Warned once for the whole run: a single answer-less row means the column is unusable.
        self._warned_missing_answer: set = set()

        disable_trl_liger(
            ctor_config(args, kwargs),
            "Disabling use_liger_kernel for SDPG: the fused GRPO-Liger loss bypasses the OPD term.",
        )

        super().__init__(*args, **kwargs)

        # Every rank constructs the trainer, so this raise is world-uniform; a per-rollout raise would
        # leave the peers of the rank that drew the bad row waiting in the next collective.
        columns = getattr(self.train_dataset, "column_names", None)
        if self.sdpg_beta_base != 0.0 and columns is not None and self.sdpg_answer_field not in columns:
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
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        rows = []
        for i in range(prompt_ids.size(0)):
            real = prompt_ids[i][prompt_mask[i].bool()].tolist()
            answer = answers[i]
            if answer is None or not str(answer).strip():
                # No hint rather than one stating an empty answer, which would mislead the teacher
                # instead of privileging it. Warned, not raised: this runs per rollout row on one rank.
                warn_once(
                    logger,
                    self._warned_missing_answer,
                    "missing_answer",
                    "SDPG: a rollout row carries no %r value, so its privileged teacher runs on the "
                    "plain prompt (no hint). The OPD term degenerates to self-distillation for those "
                    "rows — check the dataset's answer column.",
                    self.sdpg_answer_field,
                )
                hint_ids = []
            else:
                hint_ids = tok.encode(self.sdpg_hint_template.format(answer=answer), add_special_tokens=False)
            rows.append(real + hint_ids)
        max_len = max(len(r) for r in rows)
        ids = torch.full((len(rows), max_len), pad_id, dtype=prompt_ids.dtype, device=prompt_ids.device)
        mask = torch.zeros((len(rows), max_len), dtype=prompt_mask.dtype, device=prompt_mask.device)
        for i, r in enumerate(rows):  # left-pad (matches GRPO prompt padding side)
            ids[i, max_len - len(r) :] = torch.tensor(r, dtype=prompt_ids.dtype, device=prompt_ids.device)
            mask[i, max_len - len(r) :] = 1
        return ids, mask

    def _compute_loss(self, model, inputs):
        loss = super()._compute_loss(model, inputs)
        if self.sdpg_beta_base == 0.0 or "teacher_prompt_ids" not in inputs:
            return loss

        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        comp_len = completion_ids.size(1)

        # The last comp_len positions of each forward, so the two align despite different prefixes.
        student_logits = self._completion_logits(
            model, inputs["prompt_ids"], inputs["prompt_mask"], completion_ids, completion_mask, comp_len
        )
        with privileged_teacher_pass(model):
            teacher_logits = self._completion_logits(
                model,
                inputs["teacher_prompt_ids"],
                inputs["teacher_prompt_mask"],
                completion_ids,
                completion_mask,
                comp_len,
            )

        opd_per_token = self.sdpg_loss_fn(student_logits, teacher_logits, self.sdpg_temperature).sum(-1)

        gate = positive_advantage_gate(
            completion_mask.to(opd_per_token.dtype), inputs["advantages"], self.opd_positive_advantage_only
        )
        opd = (opd_per_token * gate).sum() / gate.sum().clamp(min=1.0)

        beta = beta_warmup_decay(
            int(self.state.global_step),
            int(self.state.max_steps),
            self.sdpg_beta_base,
            self.sdpg_beta_warmup_steps,
            self.sdpg_beta_decay_steps,
        )
        mode = "train" if model.training else "eval"
        self.store_metrics({"opd_loss": opd.detach(), "opd_beta": beta}, train_eval=mode)
        # TRL normalizes inside _compute_loss, so an undivided per-microbatch OPD mean inflates beta
        # by grad_accum.
        if model.training:
            opd = opd / self.current_gradient_accumulation_steps
        return loss + beta * opd

    def _completion_logits(self, model, prefix_ids, prefix_mask, completion_ids, completion_mask, comp_len):
        """Full-vocab logits at the completion positions of ``[prefix + completion]`` → ``[B, comp_len, V]``."""
        input_ids = torch.cat([prefix_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prefix_mask, completion_mask], dim=1)
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
        return logits[:, -(comp_len + 1) : -1, :]
