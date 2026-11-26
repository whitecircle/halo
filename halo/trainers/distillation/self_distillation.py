"""Offline privileged-context self-distillation, the offline approximation of SDPG. One model scores the fixed response
as student ``pi(. | x, y_<t)`` and, no-grad, as teacher ``pi(. | c, x, y_<t)`` (``c`` = gold-answer hint), with
``L = L_sft + beta(k) * L_OPD + alpha * L_ref``."""

import functools
import logging
from dataclasses import dataclass

import torch
from transformers.modeling_outputs import CausalLMOutputWithPast

from halo.configs.distillation import SelfDistillScriptArguments
from halo.data.turn_markers import resolve_eos_token_ids
from halo.data.vlm_render import SEQUENCE_ALIGNED_VISION_KEYS
from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from halo.model_loading.dtype import resolve_training_dtype
from halo.model_loading.parallel.frozen_models import ReferenceAlternatives, warn_unparallelized_reference
from halo.model_loading.tokenizer_setup import resolve_tokenizer
from halo.models.introspection.config_access import model_type_matches
from halo.models.introspection.unwrapping import unwrap_framework_wrappers
from halo.runtime.ranks import is_global_main_process
from halo.trainers.distillation.divergences import (
    ProjectedTarget,
    logits_forward_inputs,
    masked_token_mean,
    place_frozen_scorer,
    projected_distillation_terms,
    require_divergence,
    shifted_token_cross_entropy,
    token_divergence,
)
from halo.trainers.distillation.opd_term import OPDTermMixin, privileged_teacher_pass
from halo.trainers.full_logits_memory import LogitsWidth, warn_if_full_logits_large
from halo.trainers.hidden_state_forward import HiddenStateForwardMixin, ProjectedModel, reject_metrics_reading_logits
from halo.trainers.loop.evaluation import split_rows_head
from halo.trainers.metrics.stored_metrics import StoredMetricsMixin
from halo.trainers.sft import DistributedSFTTrainer

logger = logging.getLogger(__name__)


# Opt-in per family: replay is only sound when both passes call get_image_features once, identically.
_VISION_REUSE_MODEL_TYPES = ("lfm2_vl",)


@dataclass(frozen=True)
class _ChunkedLossModules:
    """The privileged teacher is the student itself, so it reads the student's modules."""

    student: ProjectedModel
    reference: ProjectedModel | None


def _teacher_inputs(model_inputs: dict, teacher: dict) -> dict:
    """The student's inputs with every per-token tensor taken from the ``teacher_*`` branch, which the hint
    lengthens."""
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


def _reject_misaligned_responses(student_mask: torch.Tensor, target_mask: torch.Tensor) -> None:
    """Refuse a batch whose rows supervise a different count of tokens per branch. A host check, not a device assert:
    the boolean gathers after it sync the host anyway, and a device assert would lose the counts."""
    student_counts, target_counts = student_mask.sum(-1), target_mask.sum(-1)
    if not torch.equal(student_counts, target_counts):
        raise RuntimeError(
            f"Self-distillation pairs per-row response counts {student_counts.tolist()} (student) with "
            f"{target_counts.tolist()} (target): the batch breaks the SelfDistill collators' response "
            f"alignment contract (require_aligned_responses), so rows would pair across samples."
        )


class DistributedSelfDistillationTrainer(
    HiddenStateForwardMixin, OPDTermMixin, StoredMetricsMixin, DistributedSFTTrainer
):
    SUPPORTS_CP = False  # privileged teacher uses a separate, longer sequence
    SUPPORTS_LOWP_PRECISION = False
    # Unlike the SFT parent's CP-only opt-out, the SDPG objective is its own mean on every axis.
    LOSS_IS_OWN_MEAN = True
    # Unlike its SFT base, this trainer forwards without labels and builds the CE itself.
    ROUTER_AUX_LOSS_ADDED_BY = None
    # compute_loss reads outputs.logits, so eval must not take the base's skip_logits path.
    LOSS_READS_LOGITS = True
    # Its loss stores its terms through StoredMetricsMixin and records none of the SFT loss metrics.
    RECORDS_LOSS_METRICS = False
    _chunked_loss_modules: _ChunkedLossModules | None = None
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
        use_chunked_distillation_loss: bool,
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
        self.use_chunked_distillation_loss = use_chunked_distillation_loss
        self._reference_model = reference_model
        self._vision_reuse_setup = False
        self._vision_reuse_active = False
        self._vision_cache = None
        self._vision_record = False
        self._vision_replay = False

        super().__init__(*args, **kwargs)

        # Every rank builds the same collator, so this refusal is world-uniform and needs no collective.
        if self.sdpg.sdpg_beta_base != 0.0 and not getattr(self.data_collator, "builds_teacher_branch", False):
            raise ValueError(
                f"sdpg_beta_base={self.sdpg.sdpg_beta_base} needs the privileged teacher branch (teacher_* keys) in "
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
        if self.use_chunked_distillation_loss:
            reject_metrics_reading_logits(self.compute_metrics, self.preprocess_logits_for_metrics)
            self._resolve_chunked_loss_modules()
        self._warn_if_full_logits_large()

    def _resolve_stop_token_ids(self):
        tok = resolve_tokenizer(self.processing_class)
        model = getattr(self, "model", None)
        self._stop_token_ids = set(
            resolve_eos_token_ids(tok, getattr(model, "config", None), getattr(model, "generation_config", None))
        )
        self._stop_ids_tensor = None

    def _warn_if_full_logits_large(self) -> None:
        """The full-logits path holds the student's logits and their gradient, the teacher's when OPD is on and the
        reference's when its term is. The collators refuse a teacher row past ``max_length`` too, so it bounds every
        plane. Skipped on the chunked path, with no CUDA device or with no ``max_length``."""
        if self.use_chunked_distillation_loss or not torch.cuda.is_available():
            return
        args = self.args
        if (args.max_length or 0) <= 0:
            return
        planes = 2 + (self.sdpg.sdpg_beta_base != 0.0) + (self.reference_kl_coef > 0)
        models = [self.accelerator.unwrap_model(self.model)]
        if self.reference_kl_coef > 0:
            models.append(self._reference_model)
        warn_if_full_logits_large(
            args,
            LogitsWidth(args.max_length, "max_length"),
            max(model.config.get_text_config().vocab_size for model in models),
            self.accelerator.device,
            bytes_per_logit=planes * resolve_training_dtype(args).itemsize,
            loss="self-distillation loss",
            remedy="Set use_chunked_distillation_loss: true to project the logits a chunk of positions at a time",
        )

    def _resolve_chunked_loss_modules(self) -> _ChunkedLossModules:
        """Resolved once, as each lookup unwraps a model through its whole module tree."""
        if self._chunked_loss_modules is None:
            reference = self._reference_model if self.reference_kl_coef > 0 else None
            self._chunked_loss_modules = _ChunkedLossModules(
                student=self._resolve_projected_model(self.model),
                reference=None if reference is None else self._resolve_projected_model(reference),
            )
        return self._chunked_loss_modules

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        self._warn_once_on_empty_labels(inputs)

        teacher = {k[len("teacher_") :]: inputs.pop(k) for k in list(inputs) if k.startswith("teacher_")}
        confidence_weights = inputs.pop("confidence_weights", None)

        labels = inputs["labels"]
        model_inputs = logits_forward_inputs(inputs)
        # Rows padding an eval split's final round repeat its first rows, so every term reads the split's own.
        real = self.eval_split_rows(labels.size(0))
        if confidence_weights is not None:
            confidence_weights = split_rows_head(confidence_weights, real)

        reuse = self._install_vision_reuse(model)
        self._vision_cache = None
        if self.use_chunked_distillation_loss:
            student_outputs = CausalLMOutputWithPast()
            sft_loss, opd_loss, ref_loss = self._projected_losses(
                model, model_inputs, labels, teacher, real, confidence_weights, reuse
            )
        else:
            student_outputs, sft_loss, opd_loss, ref_loss = self._full_logits_losses(
                model, model_inputs, labels, teacher, real, confidence_weights, reuse
            )

        loss = sft_loss
        metrics = {"sft_loss": sft_loss.detach()}
        if opd_loss is not None:
            beta = self._opd_beta()
            loss = loss + beta * opd_loss
            metrics.update(self._opd_metrics(opd_loss, beta))
        if ref_loss is not None:
            loss = loss + self.reference_kl_coef * ref_loss
            metrics["reference_kl"] = ref_loss.detach()

        self.store_batch_metrics(metrics, "train" if self.model.training else "eval", real)

        if return_outputs:
            return loss, student_outputs
        return loss

    def _full_logits_losses(self, model, model_inputs, labels, teacher, real, confidence_weights, reuse):
        """``(student outputs, sft, opd or None, reference or None)`` from every model's full logits."""
        self._vision_record, self._vision_replay = reuse, False
        student_outputs = model(**model_inputs)
        self._vision_record = False
        student_logits, labels = split_rows_head(student_outputs.logits, real), split_rows_head(labels, real)

        sft_loss = self._weighted_cross_entropy(student_logits, labels, confidence_weights)

        opd_loss = None
        # The teacher forward is a collective: config-gated, never batch-gated, so every rank enters it.
        if self.sdpg.sdpg_beta_base != 0.0:
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

        ref_loss = None
        if self.reference_kl_coef > 0:
            with torch.no_grad():
                ref_logits = split_rows_head(self._reference_model(**model_inputs).logits, real)
            ref_loss = self._reference_loss(student_logits, ref_logits, labels)
        return student_outputs, sft_loss, opd_loss, ref_loss

    def _projected_losses(self, model, model_inputs, labels, teacher, real, confidence_weights, reuse):
        """``(sft, opd or None, reference or None)``, every term taken at the supervised positions alone, the only ones
        projected, with no ``[tokens, V]`` logits built. A micro-batch with none still runs its backward through the
        head and the backbone, and every collective in them, through the empty selection."""
        modules = self._resolve_chunked_loss_modules()
        supervised = labels[:, 1:] != LABEL_IGNORE_INDEX
        supervised[real:] = False
        # One host read of each mask: integer positions keep every later selection, and its backward, off the host.
        rows, columns = supervised.nonzero().unbind(-1)
        positions = rows * labels.size(1) + columns

        teacher_inputs = teacher_positions = None
        # The teacher forward is a collective: config-gated, never batch-gated, so every rank enters it.
        if self.sdpg.sdpg_beta_base != 0.0:
            teacher_supervised = teacher["labels"][:, 1:] != LABEL_IGNORE_INDEX
            teacher_supervised[real:] = False
            # The k-th supervised position of one branch pairs with the k-th of the other.
            _reject_misaligned_responses(supervised, teacher_supervised)
            teacher_rows, teacher_columns = teacher_supervised.nonzero().unbind(-1)
            teacher_positions = teacher_rows * teacher["labels"].size(1) + teacher_columns
            teacher_inputs = _teacher_inputs(model_inputs, teacher)

        reference = None
        if modules.reference is not None:
            with torch.no_grad():
                hidden = modules.reference.backbone(**model_inputs).last_hidden_state
                head = modules.reference.head.projector(self._reference_vocab_width)
                reference = ProjectedTarget(
                    self.reference_kl_loss_fn,
                    head.project,
                    head.head_inputs_at(hidden, positions),
                    width=self._reference_vocab_width,
                )

        # The reference's tensors ride in a closure: as forward arguments FSDP2 would cast them to the student's dtype.
        student_terms = functools.partial(self._projected_student_terms, model=model, reference=reference, reuse=reuse)
        # Keyword arguments only: the wrapper's forward pre-hooks read the batch, the no-image refusal included.
        terms = self._chunked_forward_redirection(
            model,
            self.accelerator.unwrap_model(model),
            student_terms,
            hard_labels=labels[:, 1:][rows, columns],
            positions=positions,
            teacher_inputs=teacher_inputs,
            teacher_positions=teacher_positions,
            **model_inputs,
        ).unbind(-1)

        def per_token(values: torch.Tensor) -> torch.Tensor:
            """``[real, T - 1]``, 0 off the supervised positions."""
            placed = values.new_zeros(supervised.shape).index_put((rows, columns), values)
            return split_rows_head(placed, real)

        labels, supervised = split_rows_head(labels, real), split_rows_head(supervised, real)
        sft_loss = masked_token_mean(per_token(terms[0]), supervised, confidence_weights)
        opd_loss = ref_loss = None
        if teacher_inputs is not None:
            opd_loss = masked_token_mean(
                per_token(terms[1]),
                self._opd_mask(labels),
                confidence_weights if self.confidence_weight_opd else None,
            )
        if reference is not None:
            ref_loss = masked_token_mean(per_token(terms[-1]), supervised)
        return sft_loss, opd_loss, ref_loss

    def _projected_student_terms(
        self,
        *,
        model: torch.nn.Module,
        reference: ProjectedTarget | None,
        reuse: bool,
        hard_labels: torch.Tensor,
        positions: torch.Tensor,
        teacher_inputs: dict | None,
        teacher_positions: torch.Tensor | None,
        **model_inputs,
    ) -> torch.Tensor:
        """``[N, 1 + targets]`` cross-entropy, then the OPD and reference divergences that are on. Run inside the
        redirected forward, where FSDP2 has gathered the student's head; one projector serves the student and its
        privileged teacher, holding those tensors for the backward's recompute."""
        backbone = self._resolve_chunked_loss_modules().student.backbone
        self._vision_record, self._vision_replay = reuse, False
        hidden = self._backbone_last_hidden_state(backbone, **model_inputs)
        self._vision_record = False
        head = self._resolve_chunked_loss_modules().student.head.projector()
        student_rows = head.head_inputs_at(hidden, positions)

        targets = []
        if teacher_inputs is not None:
            self._vision_replay = reuse
            # Eval mode turns the adapters' dropout off for the teacher's head inputs too.
            with privileged_teacher_pass(model):
                teacher_hidden = self._backbone_last_hidden_state(backbone, **teacher_inputs)
                teacher_rows = head.head_inputs_at(teacher_hidden, teacher_positions)
            self._vision_replay = False
            self._vision_cache = None
            targets.append(ProjectedTarget(self.sdpg_loss_fn, head.project, teacher_rows))
        if reference is not None:
            targets.append(reference)
        return projected_distillation_terms(
            head.project,
            student_rows,
            targets,
            hard_labels,
            self.sdpg.sdpg_temperature,
            vocab=head.width,
            trains_cross_entropy=True,
        )

    def _install_vision_reuse(self, model) -> bool:
        """Cache the student's image features and replay them on the teacher pass; returns whether reuse is active."""
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
        """``loss_fn`` at the masked positions, the k-th masked row of one side paired with the k-th of the other, as a
        per-sample token mean. One gather per side, so the divergence covers the response rows only."""
        # One mask on both sides (the reference term) pairs trivially.
        if student_mask is not target_mask:
            _reject_misaligned_responses(student_mask, target_mask)
        student_rows = student_logits[:, :-1][student_mask]
        target_rows = target_logits[:, :-1][target_mask]
        per_row = token_divergence(loss_fn, student_rows, target_rows, self.sdpg.sdpg_temperature)
        per_token = per_row.new_zeros(student_mask.shape).masked_scatter(student_mask, per_row)
        return masked_token_mean(per_token, student_mask, sample_weights)
