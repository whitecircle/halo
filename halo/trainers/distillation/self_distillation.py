"""Offline privileged-context self-distillation SFT trainer, the offline approximation of SDPG.

No generation: the fixed dataset response is scored by one model as student ``pi(. | x, y_<t)`` and,
under ``torch.no_grad()``, as privileged teacher ``pi(. | c, x, y_<t)`` (``c`` = gold-answer hint), with
``L = L_sft + beta(k) * L_OPD + alpha * L_ref``. The on-policy variant is :class:`DistributedSDPGTrainer`.
"""

import logging

import torch

from halo.configs.distillation import SelfDistillScriptArguments
from halo.data.turn_markers import resolve_eos_token_ids
from halo.data.vlm_render import SEQUENCE_ALIGNED_VISION_KEYS
from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from halo.model_loading.parallel.frozen_models import ReferenceAlternatives, warn_unparallelized_reference
from halo.model_loading.tokenizer_setup import resolve_tokenizer
from halo.models.config_access import model_type_matches
from halo.models.unwrapping import unwrap_framework_wrappers
from halo.runtime.ranks import is_global_main_process
from halo.trainers.distillation.divergences import (
    logits_forward_inputs,
    masked_token_mean,
    place_frozen_scorer,
    require_divergence,
    shifted_token_cross_entropy,
)
from halo.trainers.distillation.opd_term import OPDTermMixin, privileged_teacher_pass
from halo.trainers.evaluation import split_rows_head
from halo.trainers.sft import DistributedSFTTrainer
from halo.trainers.stored_metrics import StoredMetricsMixin

logger = logging.getLogger(__name__)


# Opt-in per family: replay is only sound when both passes call get_image_features once, identically.
_VISION_REUSE_MODEL_TYPES = ("lfm2_vl",)


def _teacher_inputs(model_inputs: dict, teacher: dict) -> dict:
    """The teacher forward's inputs: the student's, with every per-token tensor taken from the ``teacher_*``
    branch, whose sequence the hint lengthens."""
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
    for extra in SEQUENCE_ALIGNED_VISION_KEYS:
        if extra in teacher:
            teacher_inputs[extra] = teacher[extra]
    return teacher_inputs


class DistributedSelfDistillationTrainer(OPDTermMixin, StoredMetricsMixin, DistributedSFTTrainer):
    """SFT trainer with an SDPG-style privileged-context self-distillation auxiliary loss."""

    SUPPORTS_CP = False  # privileged teacher uses a separate, longer sequence
    SUPPORTS_LOWP_PRECISION = False
    # Unlike the SFT parent's CP-only opt-out, the SDPG objective is its own mean on every axis.
    LOSS_IS_OWN_MEAN = True
    # Unlike its SFT base, this trainer forwards without labels and builds the CE itself.
    ROUTER_AUX_LOSS_ADDED_BY = None
    # Same reason: compute_loss reads outputs.logits, so eval must not take the base's skip_logits path.
    LOSS_READS_LOGITS = True
    PP_REFUSAL = (
        "the privileged-teacher term needs a second, no-grad forward of the WHOLE model per batch on "
        "the hint-carrying sequence (and a third through the frozen reference when reference_kl_coef "
        "> 0), whose logits the student's loss then consumes; under PP no rank holds the whole model, "
        "the schedule owns every forward, and the runner's forward-only pass yields logits on the "
        "last stage only, with no channel to feed them into the same step's loss. The objective "
        "itself is per-row and microbatch-decomposable, so this is a pipeline-machinery gap (a "
        "second forward-only pass whose last-stage output becomes an extra target), not a loss one"
    )

    def __init__(
        self,
        *args,
        reference_kl_coef: float,
        reference_kl_loss: str,
        confidence_weight_opd: bool,
        opd_exclude_eos: bool,
        reference_model: torch.nn.Module | None = None,
        **kwargs,
    ):
        if reference_kl_coef > 0 and reference_model is None:
            raise ValueError(
                f"reference_kl_coef={reference_kl_coef} weights a KL anchor to a frozen reference, but "
                f"no reference_model was passed, so L_ref would be silently dropped. Pass the reference "
                f"(the self_distill script loads it from reference_model_name_or_path), or set "
                f"reference_kl_coef: 0."
            )
        # The dataset-side fields stay with the script, which bakes the hint into the teacher prompts.
        self._adopt_sdpg_arguments(kwargs, exclude=SelfDistillScriptArguments.DATASET_SIDE_SDPG_FIELDS)
        self.reference_kl_coef = reference_kl_coef
        self.reference_kl_loss_fn = require_divergence(reference_kl_loss) if reference_kl_coef > 0 else None
        self.confidence_weight_opd = confidence_weight_opd
        self.opd_exclude_eos = opd_exclude_eos
        self._reference_model = reference_model
        self._vision_reuse_setup = False
        self._vision_reuse_active = False
        self._vision_cache = None
        self._vision_record = False
        self._vision_replay = False

        super().__init__(*args, **kwargs)

        # Every rank builds the same collator, so this refusal is world-uniform and needs no collective.
        if self.sdpg_beta_base != 0.0 and not getattr(self.data_collator, "builds_teacher_branch", False):
            raise ValueError(
                f"sdpg_beta_base={self.sdpg_beta_base} needs the privileged teacher branch (teacher_* keys) in "
                f"every batch, and {type(self.data_collator).__name__} builds none. Use a SelfDistill collator "
                f"built with a hint_template, or set sdpg_beta_base: 0 to drop the term."
            )
        self._resolve_stop_token_ids()

        if self.reference_kl_coef > 0:
            warn_unparallelized_reference(
                self.parallelism_config, ReferenceAlternatives("reference_kl_coef: 0 loads no reference.")
            )
            # Keyed to the policy's own tokenizer, so only the row counts are checked; the script checks the token
            # ids of a reference from another repo before any load.
            tokenizer = resolve_tokenizer(self.processing_class)
            self._reference_vocab_width, device = place_frozen_scorer(
                self._reference_model, self.model, tokenizer, tokenizer
            )
            if is_global_main_process():
                logger.info(f"Reference model moved to {device} and frozen (alpha={self.reference_kl_coef})")

    def _resolve_stop_token_ids(self):
        """EOS/stop token ids excluded from OPD when ``opd_exclude_eos``, via ``resolve_eos_token_ids``."""
        tok = resolve_tokenizer(self.processing_class)
        model_config = getattr(getattr(self, "model", None), "config", None)
        self._stop_token_ids = set(resolve_eos_token_ids(tok, model_config))
        self._stop_ids_tensor = None

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self._warn_once_on_empty_labels(inputs)

        teacher = {k[len("teacher_") :]: inputs.pop(k) for k in list(inputs) if k.startswith("teacher_")}
        confidence_weights = inputs.pop("confidence_weights", None)

        labels = inputs["labels"]
        # Full-vocab logits (no fused-LCE shortcut); CE computed manually below.
        model_inputs = logits_forward_inputs(inputs)

        reuse = self._install_vision_reuse(model)
        self._vision_cache = None
        self._vision_record, self._vision_replay = reuse, False

        student_outputs = model(**model_inputs)
        self._vision_record = False
        # Rows padding an eval split's final round repeat its first rows: every term reads the
        # split's own (in train, the whole batch, unsliced).
        real = self.eval_split_rows(labels.size(0))
        student_logits, labels = split_rows_head(student_outputs.logits, real), split_rows_head(labels, real)
        if confidence_weights is not None:
            confidence_weights = split_rows_head(confidence_weights, real)

        sft_loss = self._weighted_cross_entropy(student_logits, labels, confidence_weights)
        loss = sft_loss

        metrics = {"sft_loss": sft_loss.detach()}

        # The teacher forward is a collective: config-gated, never batch-gated, so every rank enters it.
        if self.sdpg_beta_base != 0.0:
            self._vision_replay = reuse
            with privileged_teacher_pass(model):
                teacher_logits = split_rows_head(model(**_teacher_inputs(model_inputs, teacher)).logits, real)
            self._vision_replay = False
            self._vision_cache = None
            # Each branch's response rows are gathered by its own mask, which the collators align
            # (require_aligned_responses); opd_exclude_eos drops EOS/stop tokens from OPD, not SFT.
            opd_loss = self._response_divergence(
                self.sdpg_loss_fn,
                student_logits,
                self._opd_mask(labels),
                teacher_logits,
                self._opd_mask(split_rows_head(teacher["labels"], real)),
                confidence_weights if self.confidence_weight_opd else None,
            )
            beta = self._opd_beta()
            loss = loss + beta * opd_loss
            metrics.update(self._opd_metrics(opd_loss, beta))

        if self.reference_kl_coef > 0:
            with torch.no_grad():
                ref_logits = split_rows_head(self._reference_model(**model_inputs).logits, real)
            ref_loss = self._reference_loss(student_logits, ref_logits, labels)
            loss = loss + self.reference_kl_coef * ref_loss
            metrics["reference_kl"] = ref_loss.detach()

        self.store_batch_metrics(metrics, "train" if self.model.training else "eval", real)

        if return_outputs:
            return loss, student_outputs
        return loss

    def _install_vision_reuse(self, model) -> bool:
        """Install a caching wrapper around the VLM's ``get_image_features`` (once).

        The wrapper records the student's image features and replays them on the teacher pass, gated
        by ``self._vision_record``/``self._vision_replay``. Returns whether reuse is active.
        """
        if self._vision_reuse_setup:
            return self._vision_reuse_active
        self._vision_reuse_setup = True

        inner = unwrap_framework_wrappers(model)
        mm = getattr(inner, "model", None)
        config = getattr(inner, "config", None)
        if not (model_type_matches(config, *_VISION_REUSE_MODEL_TYPES) and hasattr(mm, "get_image_features")):
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
        if is_global_main_process():
            logger.info(f"Vision-feature reuse enabled for OPD teacher forward (model_type={config.model_type})")
        return True

    def _weighted_cross_entropy(self, logits, labels, sample_weights):
        """Token-mean SFT cross-entropy with optional per-sample confidence weighting."""
        shift_labels = labels[:, 1:]
        token_ce = shifted_token_cross_entropy(logits[:, :-1, :], shift_labels)
        return masked_token_mean(token_ce, shift_labels != LABEL_IGNORE_INDEX, sample_weights)

    def _opd_mask(self, labels):
        """The positions OPD distils: supervised next tokens, less EOS/stop under ``opd_exclude_eos``."""
        targets = labels[:, 1:]
        mask = targets != LABEL_IGNORE_INDEX
        if self.opd_exclude_eos and self._stop_token_ids:
            if self._stop_ids_tensor is None:
                self._stop_ids_tensor = torch.tensor(sorted(self._stop_token_ids), device=targets.device)
            mask &= ~torch.isin(targets, self._stop_ids_tensor)
        return mask

    def _reference_loss(self, student_logits, reference_logits, labels):
        """Divergence to the frozen reference on the supervised tokens; the two share the sequence."""
        mask = labels[:, 1:] != LABEL_IGNORE_INDEX
        width = self._reference_vocab_width
        return self._response_divergence(
            self.reference_kl_loss_fn, student_logits[..., :width], mask, reference_logits[..., :width], mask
        )

    def _response_divergence(
        self, loss_fn, student_logits, student_mask, target_logits, target_mask, sample_weights=None
    ):
        """``loss_fn`` between the student's and the target's next-token rows at the masked positions,
        the k-th masked row of one paired with the k-th of the other, as a per-sample token mean.

        One gather per side, so the fp32 divergence planes cover the response rows only.
        """
        student_counts, target_counts = student_mask.sum(-1), target_mask.sum(-1)
        # One mask on both sides (the reference term) pairs trivially. A host check, not a device assert:
        # the boolean gathers below sync the host anyway, and a device assert would lose the counts.
        if student_mask is not target_mask and not torch.equal(student_counts, target_counts):
            raise RuntimeError(
                f"Self-distillation pairs per-row response counts {student_counts.tolist()} (student) with "
                f"{target_counts.tolist()} (target): the batch breaks the SelfDistill collators' response "
                f"alignment contract (require_aligned_responses), so rows would pair across samples."
            )
        student_rows = student_logits[:, :-1][student_mask]
        target_rows = target_logits[:, :-1][target_mask]
        per_row = loss_fn(student_rows, target_rows, self.sdpg_temperature).sum(-1)
        per_token = per_row.new_zeros(student_mask.shape).masked_scatter(student_mask, per_row)
        return masked_token_mean(per_token, student_mask, sample_weights)
