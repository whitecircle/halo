"""On-policy SDPG (arXiv:2606.04036): ``L = L_GRPO + beta(k) * L_OPD``. The same policy, no-grad on a hint revealing
the gold ``answer``, is the teacher of a reverse KL gated to positive-advantage rollouts. Text-only."""

import functools
import logging

import torch

from halo.configs.distillation import SDPGArguments
from halo.data.collators.self_distill import format_field_names, hint_literal_bytes, privileged_hint
from halo.log import warn_once
from halo.model_loading.dtype import resolve_training_dtype
from halo.model_loading.tokenizer_setup import resolve_tokenizer
from halo.trainers.distillation.divergences import pooled_token_mean, projected_divergence, token_divergence
from halo.trainers.distillation.opd_term import OPDTermMixin, privileged_teacher_pass
from halo.trainers.grpo.logprob_scoring import FULL_LOGITS_BYTES_PER_LOGIT, CompletionTerm, HeadSweep
from halo.trainers.grpo.online import DistributedOnlineGRPOTrainer
from halo.trainers.grpo.rounds.rank_balance import PROMPT_ROWS
from halo.trainers.loop.evaluation import split_rows_head
from halo.trainers.metrics.stored_metrics import StoredMetricsMixin
from halo.trainers.metrics.token_metrics import effective_loss_mask

logger = logging.getLogger(__name__)

TEACHER_PROMPT_ROWS = ("teacher_prompt_ids", "teacher_prompt_mask")


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

    PACKED_ROWS_REFUSAL = (
        "the OPD term reads the student's completion logits off the loss forward and pairs them row for row with a "
        "teacher pass over [teacher_prompt ⧺ completion] rows sized by the micro-batch, neither of which a packed "
        "slot's flattened scorer and filler rows provide"
    )

    ROUND_PROMPT_ROWS = (*PROMPT_ROWS, TEACHER_PROMPT_ROWS)

    @classmethod
    def declared_row_length(cls, training_config, *sources) -> int | None:
        """The online bound plus the hint template's literal text, since the teacher forwards the prompt, the hint and
        the completion through the same dispatch. An answer the template names has no config bound, so its tokens are
        not counted: the dispatcher's own check on the batch in hand stays the backstop for them."""
        row = super().declared_row_length(training_config, *sources)
        sdpg = next((source for source in sources if isinstance(source, SDPGArguments)), None)
        if row is None:
            return None
        if sdpg is None:
            raise ValueError(
                "SDPG's teacher rows are longer than the online rows by the hint, which sizes the DeepEP dispatch "
                "gate, but no row source carries the SDPG arguments. Pass the script arguments as a row source."
            )
        return row + (hint_literal_bytes(sdpg.sdpg_hint_template) if sdpg.sdpg_beta_base != 0.0 else 0)

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

    @property
    def _full_logits_bytes_per_logit(self) -> int:
        """GRPO's plane and, with the OPD term on, the teacher's in the run's precision."""
        if self.sdpg.sdpg_beta_base == 0.0:
            return FULL_LOGITS_BYTES_PER_LOGIT
        return FULL_LOGITS_BYTES_PER_LOGIT + resolve_training_dtype(self.args).itemsize

    def _build_privileged_rows(self, result, inputs):
        if self.sdpg.sdpg_beta_base == 0.0:
            return {}
        # Answers in prompt row order, aligned 1:1 with rollout rows (RepeatSampler already expanded).
        answers = [x.get(self.sdpg_answer_field) for x in inputs]
        if len(answers) != result["prompt_ids"].size(0):
            raise RuntimeError(
                f"SDPG got {len(answers)} answers for {result['prompt_ids'].size(0)} rollout rows, so a teacher "
                f"prompt would carry another row's answer."
            )
        ids, mask = self._build_teacher_prompts(result["prompt_ids"], result["prompt_mask"], answers)
        return {"teacher_prompt_ids": ids, "teacher_prompt_mask": mask}

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

        # The tokens the GRPO loss trains: a tool-output token is attention-valid but never trained.
        gate = positive_advantage_gate(
            effective_loss_mask(inputs).float(), inputs["advantages"], self.opd_positive_advantage_only
        )
        prefix = (inputs["teacher_prompt_ids"], inputs["teacher_prompt_mask"])
        completion = (inputs["completion_ids"], inputs["completion_mask"])
        teacher_hidden = None
        if self._use_chunked_grpo_logprobs:
            unwrapped = self.accelerator.unwrap_model(model)
            with privileged_teacher_pass(model):
                teacher_hidden = self._chunked_forward_redirection(
                    model, unwrapped, self._teacher_hidden_states, unwrapped, *prefix, *completion
                )
        # The GRPO loss forward hands the student's completion rows to the term, so the student runs forward and
        # backward once.
        term = CompletionTerm(gate, functools.partial(self._opd_scores, teacher_hidden))
        with self._scoring_with_completion_term(term):
            loss = super()._compute_loss(model, inputs)
        opd_per_token = self._opd_token_divergence(model, prefix, completion, term)

        # Rows padding an eval split's final round repeat its first rows, so the OPD mean reads the split's own.
        real_rows = self.eval_split_rows(gate.size(0))
        opd = pooled_token_mean(split_rows_head(opd_per_token, real_rows), split_rows_head(gate, real_rows))

        beta = self._opd_beta()
        self.store_batch_metrics(self._opd_metrics(opd, beta), "train" if model.training else "eval", real_rows)
        # TRL normalizes inside _compute_loss, so an undivided per-microbatch OPD mean inflates beta by grad_accum.
        if model.training:
            opd = opd / self.current_gradient_accumulation_steps
        return loss + beta * opd

    def _opd_token_divergence(self, model, prefix, completion, term: CompletionTerm) -> torch.Tensor:
        """``[B, C]`` divergence of the student from the teacher: the chunked scorer's values, or the full-logits
        scorer's student logits against the teacher's."""
        completion_ids = completion[0]
        parts = term.per_token or term.logits
        if sum(part.size(0) for part in parts) != completion_ids.size(0) or any(
            part.size(1) != completion_ids.size(1) for part in parts
        ):
            raise RuntimeError(
                f"The OPD term reads the student's completion rows off the GRPO loss forward, which handed back "
                f"{[tuple(part.shape[:2]) for part in parts]} for a {tuple(completion_ids.shape)} completion batch: "
                f"TRL's _compute_loss no longer scores the policy through _get_per_token_logps_and_entropies, or an "
                f"override forwards around it. Keep the loss forward on that scorer, or set sdpg_beta_base: 0 to drop "
                f"the term."
            )
        if term.per_token:
            return torch.cat(term.per_token)
        with privileged_teacher_pass(model):
            teacher_logits = self._completion_logits(model, *prefix, *completion)
        rows = [part.size(0) for part in term.logits]
        return torch.cat(
            [
                token_divergence(self.sdpg_loss_fn, student, teacher, self.sdpg.sdpg_temperature)
                for student, teacher in zip(term.logits, teacher_logits.split(rows), strict=True)
            ]
        )

    def _opd_scores(
        self, teacher_hidden: torch.Tensor, head: HeadSweep, student: torch.Tensor, positions: torch.Tensor
    ) -> torch.Tensor:
        """The divergence at ``positions`` of the ``[B, C]`` window, both sides projected through the head the GRPO
        sweep gathered: the student's with a gradient, the teacher's without."""
        teacher = teacher_hidden.flatten(0, 1).index_select(0, positions)
        return projected_divergence(
            self.sdpg_loss_fn,
            head.project,
            student,
            head.project,
            teacher,
            self.sdpg.sdpg_temperature,
            vocab=head.width,
        )

    def _teacher_hidden_states(self, unwrapped_model, prefix_ids, prefix_mask, completion_ids, completion_mask):
        """``[B, C, H]`` at the positions predicting the completion, run inside the redirected forward where FSDP2 has
        gathered the root parameters. The head is applied inside the GRPO loss forward, at the positions the gate
        keeps."""
        return self._backbone_hidden_state(
            unwrapped_model,
            torch.cat([prefix_ids, completion_ids], dim=1),
            torch.cat([prefix_mask, completion_mask], dim=1),
            completion_ids.size(1),
        )

    def _completion_logits(self, model, prefix_ids, prefix_mask, completion_ids, completion_mask):
        """``[B, C, V]``. Where the forward takes ``logits_to_keep`` the prompt's rows are never built. The prefixes
        differ in length; aligning on the sequence end pairs these rows with the loss forward's."""
        keep = completion_ids.size(1) + 1
        input_ids = torch.cat([prefix_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prefix_mask, completion_mask], dim=1)
        trim = {"logits_to_keep": keep} if "logits_to_keep" in self.model_kwarg_keys else {}
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, **trim).logits
        return logits[:, -keep:-1, :]
