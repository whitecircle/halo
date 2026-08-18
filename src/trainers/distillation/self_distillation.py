"""Offline privileged-context self-distillation SFT trainer (offline SDPG approximation).

Offline SFT approximation of SDPG (arXiv:2606.04036): no generation, scores the fixed dataset
response tokens with static confidence weights. The faithful on-policy method is
:class:`DistributedSDPGTrainer`. A single model is student ``p = pi(. | x, y_<t)`` and privileged
teacher ``q = pi(. | c, x, y_<t)`` (``c`` = gold-answer hint). Per optimizer step::

    L = L_sft  +  beta(k) * L_OPD  +  alpha * L_ref

Teacher forward reuses the same model under ``torch.no_grad()`` (no second model). EP and TP; not CP
(the privileged teacher uses a second, differently-lengthed sequence).
"""

from dataclasses import fields

import torch
from accelerate.logging import get_logger

from src.args.mixins import SDPGArguments
from src.args.self_distill_args import SelfDistillationArguments
from src.data.spans import LABEL_IGNORE_INDEX, resolve_eos_token_ids
from src.distributed.loading.frozen_models import place_and_freeze
from src.distributed.runtime import rank_consensus
from src.models.structure import resolve_tokenizer
from src.trainers.distillation.losses import (
    beta_warmup_decay,
    get_self_distillation_loss_fn,
    logits_forward_inputs,
    masked_token_mean,
    privileged_teacher_pass,
    shifted_token_cross_entropy,
)
from src.trainers.mixins.stored_metrics import StoredMetricsMixin
from src.trainers.sft import DistributedSFTTrainer

logger = get_logger(__name__, log_level="info")


# Opt-in per family: replay is only sound when both passes call get_image_features once, identically.
_VISION_REUSE_MODEL_TYPES = {"lfm2_vl"}


class DistributedSelfDistillationTrainer(StoredMetricsMixin, DistributedSFTTrainer):
    """SFT trainer with an SDPG-style privileged-context self-distillation auxiliary loss."""

    _supports_cp = False  # privileged teacher uses a separate, longer sequence
    _supports_pp = False
    # Unconditional, unlike the SFT parent's CP-only property: the SDPG objective is its own mean
    # on every axis, so HF must never rescale it by num_items_in_batch.
    _loss_is_own_mean = True
    # Unlike its SFT base, this trainer forwards WITHOUT labels and builds the CE itself.
    _consumes_router_aux_loss = False
    # Same reason: compute_loss reads outputs.logits, so eval must not take the base's skip_logits path.
    _loss_reads_logits = True
    _pp_unsupported_reason = (
        "the privileged-teacher term needs a second, no-grad forward of the WHOLE model per batch on "
        "the hint-carrying sequence (and a third through the frozen reference when reference_kl_coef "
        "> 0), whose logits the student's loss then consumes; under PP no rank holds the whole model, "
        "the schedule owns every forward, and the runtime's forward-only pass yields logits on the "
        "last stage only, with no channel to feed them into the same step's loss. The objective "
        "itself is per-row and microbatch-decomposable, so this is a pipeline-machinery gap (a "
        "second forward-only pass whose last-stage output becomes an extra target), not a loss one"
    )

    def __init__(
        self,
        *args,
        reference_model: torch.nn.Module | None = None,
        reference_kl_coef: float = 0.0,
        reference_kl_loss: str = "unnormalized_kl",
        confidence_weight_opd: bool = True,
        opd_exclude_eos: bool = True,
        **kwargs,
    ):
        # The tunables arrive under the names SDPGArguments declares and default to what it declares:
        # popped by its fields and adopted as attributes. The dataset-side fields stay with the script
        # (it bakes the hint into the teacher prompts), so one handed here reaches TRL's signature.
        dataset_side = SelfDistillationArguments.DATASET_SIDE_SDPG_FIELDS
        sdpg = SDPGArguments(
            **{
                f.name: kwargs.pop(f.name)
                for f in fields(SDPGArguments)
                if f.name in kwargs and f.name not in dataset_side
            }
        )
        vars(self).update({name: value for name, value in vars(sdpg).items() if name not in dataset_side})
        self.sdpg_loss_fn = get_self_distillation_loss_fn(self.sdpg_loss)
        self.reference_kl_coef = reference_kl_coef
        self.reference_kl_loss_fn = get_self_distillation_loss_fn(reference_kl_loss) if reference_kl_coef > 0 else None
        self.confidence_weight_opd = confidence_weight_opd
        self.opd_exclude_eos = opd_exclude_eos
        self._reference_model = reference_model
        self._warned_response_mismatch = False
        self._vision_reuse_setup = False
        self._vision_reuse_active = False
        self._vision_cache = None
        self._vision_record = False
        self._vision_replay = False

        super().__init__(*args, **kwargs)

        self._resolve_stop_token_ids()

        if self._reference_model is not None and self.reference_kl_coef > 0:
            self._validate_reference_model(self._reference_model)
            self._setup_reference_model()

    def _resolve_stop_token_ids(self):
        """EOS/stop token ids excluded from OPD when ``opd_exclude_eos``, via ``resolve_eos_token_ids``."""
        tok = resolve_tokenizer(self.processing_class)
        model_config = getattr(getattr(self, "model", None), "config", None)
        self._stop_token_ids = set(resolve_eos_token_ids(tok, model_config))
        self._stop_ids_tensor = None

    def _maybe_setup_vision_reuse(self, model) -> bool:
        """Install a caching wrapper around the VLM's ``get_image_features`` (once).

        Returns whether vision-feature reuse is active. The wrapper records the student's image
        features and replays them on the teacher pass, gated by ``self._vision_record`` /
        ``self._vision_replay`` so it is inert outside the bracketed region.
        """
        if self._vision_reuse_setup:
            return self._vision_reuse_active
        self._vision_reuse_setup = True

        inner = model
        while hasattr(inner, "module"):
            inner = inner.module
        mm = getattr(inner, "model", None)
        model_type = getattr(getattr(inner, "config", None), "model_type", None)
        if model_type not in _VISION_REUSE_MODEL_TYPES or mm is None or not hasattr(mm, "get_image_features"):
            return False

        original = mm.get_image_features

        def cached_get_image_features(*args, **kwargs):
            if self._vision_replay and self._vision_cache is not None:
                return self._vision_cache
            out = original(*args, **kwargs)
            if self._vision_record:
                self._vision_cache = out
            return out

        mm.get_image_features = cached_get_image_features
        self._vision_reuse_active = True
        logger.info(f"Vision-feature reuse enabled for OPD teacher forward (model_type={model_type})")
        return True

    def _setup_reference_model(self):
        """Move the frozen reference model to the student device and disable gradients."""
        device = place_and_freeze(self._reference_model, self.model)
        logger.info(f"Reference model moved to {device} and frozen (alpha={self.reference_kl_coef})")

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self._validate_inputs(inputs)

        teacher = {k[len("teacher_") :]: inputs.pop(k) for k in list(inputs) if k.startswith("teacher_")}
        confidence_weights = inputs.pop("confidence_weights", None)

        labels = inputs["labels"]
        # Full-vocab logits (no fused-LCE shortcut); CE computed manually below.
        model_inputs = logits_forward_inputs(inputs)

        reuse = self._maybe_setup_vision_reuse(model)
        self._vision_cache = None
        self._vision_record, self._vision_replay = reuse, False

        student_outputs = model(**model_inputs)
        student_logits = student_outputs.logits
        self._vision_record = False

        sft_loss = self._weighted_cross_entropy(student_logits, labels, confidence_weights)
        loss = sft_loss

        mode = "train" if self.model.training else "eval"
        metrics = {"sft_loss": sft_loss.detach()}

        # The teacher forward issues collectives, so whether it runs must be a world-wide decision.
        run_teacher = teacher.get("input_ids") is not None and self.sdpg_beta_base != 0.0
        # Both halves of the verdict, so EVERY rank raises on a mismatch; on "any" alone the
        # teacher-carrying ranks march into the collective forward and hang.
        all_teacher, any_teacher = rank_consensus(run_teacher)
        if any_teacher and not all_teacher:
            raise RuntimeError(
                "Some ranks have a teacher branch in this batch and others do not — the teacher forward "
                "is a collective, so the batch must carry teacher_* keys on every rank. Check the collator."
            )
        run_teacher = any_teacher
        if run_teacher:
            # Every per-token tensor must come from the teacher branch: its sequence is longer (the hint).
            per_token = (
                "input_ids",
                "attention_mask",
                "mm_token_type_ids",
                "token_type_ids",
                "position_ids",
                "cache_position",
            )
            teacher_inputs = {k: v for k, v in model_inputs.items() if k not in per_token}
            teacher_inputs["input_ids"] = teacher["input_ids"]
            teacher_inputs["attention_mask"] = teacher["attention_mask"]
            for extra in ("mm_token_type_ids", "token_type_ids"):
                if extra in teacher:
                    teacher_inputs[extra] = teacher[extra]
            self._vision_replay = reuse
            with privileged_teacher_pass(model):
                teacher_logits = model(**teacher_inputs).logits
            self._vision_replay = False
            self._vision_cache = None

            opd_loss = self._opd_loss(
                student_logits,
                labels,
                teacher_logits,
                teacher["labels"],
                confidence_weights if self.confidence_weight_opd else None,
            )
            beta = beta_warmup_decay(
                int(self.state.global_step),
                int(self.state.max_steps),
                self.sdpg_beta_base,
                self.sdpg_beta_warmup_steps,
                self.sdpg_beta_decay_steps,
            )
            loss = loss + beta * opd_loss
            metrics["opd_loss"] = opd_loss.detach()
            metrics["beta"] = beta

        if self.reference_kl_coef > 0 and self._reference_model is not None:
            with torch.no_grad():
                ref_logits = self._reference_model(**model_inputs).logits
            ref_loss = self._reference_loss(student_logits, ref_logits, labels)
            loss = loss + self.reference_kl_coef * ref_loss
            metrics["reference_kl"] = ref_loss.detach()

        self._record_metrics(metrics, mode)

        if return_outputs:
            return loss, student_outputs
        return loss

    def _weighted_cross_entropy(self, logits, labels, sample_weights):
        """Token-mean SFT cross-entropy with optional per-sample confidence weighting."""
        shift_labels = labels[:, 1:]
        token_ce = shifted_token_cross_entropy(logits[:, :-1, :], shift_labels)
        return masked_token_mean(token_ce, shift_labels != LABEL_IGNORE_INDEX, sample_weights)

    def _opd_loss(self, student_logits, student_labels, teacher_logits, teacher_labels, sample_weights):
        """Full-vocab OPD loss on the shared response tokens.

        Student and teacher sequences differ in length (the teacher carries the hint), so response
        rows are gathered per sample by their label masks and aligned positionally (identical token
        IDs). ``opd_exclude_eos`` drops the EOS/stop token from OPD (but not from SFT).
        """
        student_shift = student_logits[:, :-1, :]
        student_ids = student_labels[:, 1:]
        student_mask = student_ids != LABEL_IGNORE_INDEX
        teacher_shift = teacher_logits[:, :-1, :]
        teacher_mask = teacher_labels[:, 1:] != LABEL_IGNORE_INDEX

        per_sample = []
        for b in range(student_shift.size(0)):
            student_rows = student_shift[b][student_mask[b]]
            teacher_rows = teacher_shift[b][teacher_mask[b]]
            n = min(student_rows.size(0), teacher_rows.size(0))
            if n == 0:
                per_sample.append(student_logits.new_zeros(()))
                continue
            if student_rows.size(0) != teacher_rows.size(0) and not self._warned_response_mismatch:
                logger.warning(
                    "Self-distillation: student/teacher response length mismatch "
                    f"({student_rows.size(0)} vs {teacher_rows.size(0)}); aligning on the first "
                    f"{n} tokens. Usually caused by max_length truncation — raise max_length."
                )
                self._warned_response_mismatch = True

            resp_ids = student_ids[b][student_mask[b]][:n]
            keep = self._opd_keep_mask(resp_ids)
            sr, tr = student_rows[:n][keep], teacher_rows[:n][keep]
            if sr.size(0) == 0:
                per_sample.append(student_logits.new_zeros(()))
                continue
            token_loss = self.sdpg_loss_fn(sr, tr, self.sdpg_temperature)
            per_sample.append(token_loss.sum(-1).mean())

        loss = torch.stack(per_sample)
        if sample_weights is not None:
            loss = loss * sample_weights.to(loss.dtype)
        return loss.mean()

    def _opd_keep_mask(self, resp_ids):
        """Per-response-token boolean keep-mask: drops EOS/stop tokens when opd_exclude_eos."""
        keep = torch.ones(resp_ids.size(0), dtype=torch.bool, device=resp_ids.device)
        if self.opd_exclude_eos and self._stop_token_ids:
            if self._stop_ids_tensor is None:
                # Memoized: runs per sample per microbatch.
                self._stop_ids_tensor = torch.tensor(sorted(self._stop_token_ids), device=resp_ids.device)
            keep &= ~torch.isin(resp_ids, self._stop_ids_tensor)
        return keep

    def _reference_loss(self, student_logits, reference_logits, labels):
        """Unnormalized-KL (or KL) regularization to the frozen reference on response tokens."""
        student_shift = student_logits[:, :-1, :]
        reference_shift = reference_logits[:, :-1, :]
        response_mask = labels[:, 1:] != LABEL_IGNORE_INDEX
        token_loss = self.reference_kl_loss_fn(student_shift, reference_shift, self.sdpg_temperature)
        return masked_token_mean(token_loss, response_mask)

    def _record_metrics(self, metrics, mode):
        """Store auxiliary-loss metrics as detached on-device values; ``StoredMetricsMixin`` drains
        them once per log window, so recording adds no per-microbatch host sync."""
        self.store_metrics(
            {key: value.detach() if torch.is_tensor(value) else float(value) for key, value in metrics.items()},
            train_eval=mode,
        )
