# Copyright 2024 White Circle
#
# Licensed under the Halo License (the "License"): the Apache License, Version 2.0
# (a copy is provided in APACHE-2.0.txt), as modified by, and subject to, the
# Supplemental Terms in the LICENSE file at the root of this repository.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Distributed KTO (Kahneman-Tversky Optimization) trainer: EP, TP, and PP.

KTO learns from *unpaired* preference data — each example is a single
(prompt, completion, label) triple where ``label`` marks the completion as
desirable or not — instead of DPO's chosen/rejected pairs.

CP is NOT supported: like DPO, KTO needs global per-sequence log-prob sums over
full sequences (and a KL reference term), incompatible with sequence splitting.

Under EP/TP the reference model is NOT parallelized — use PEFT/LoRA
(ref_model=None) or precompute_ref_log_probs=True (then free the ref model).

Under PP the trainer is PRECOMPUTE-ONLY and restricted to ``loss_type:
apo_zero_unpaired``: that loss has no KL term, so with ``ref_logps`` already a
dataset column (in the eval dataset too, when evaluation is configured) the
objective is a per-row function of the last stage's logits and no reference model
ever runs. See ``_validate_pp_mode`` for each rejection and its mechanism.
"""

import torch
from trl import KTOTrainer

from src.data.spans import LABEL_IGNORE_INDEX
from src.distributed.pipeline_parallel.losses import (
    PPLossAdapter,
    completion_labels,
    pad_to_pipeline_length,
    row_count_normalizer,
    rows_with_labels,
    sequence_logprobs,
)
from src.trainers.mixins.base import DistributedTrainerMixin
from src.trainers.mixins.pp_gates import (
    reject_pp_compute_metrics,
    reject_pp_ref_model,
    require_model_and_args_kwargs,
    require_precomputed_reference,
)
from src.trainers.mixins.validation import ctor_config, ctor_positions, ctor_value, disable_trl_liger
from src.trainers.preference.precompute import PrecomputeRefLogpsRankConsistentMixin

# TRL KTOTrainer positional slot for the EP/TP reference gate — derived from the installed signature
# (its public class is a ``(*args, **kwargs)`` shim, so the derivation reads the class it forwards to).
_CTOR_POSITIONS = ctor_positions(KTOTrainer, "ref_model")

_REF_LOGPS_COLUMN = "ref_logps"


class DistributedKTOTrainer(DistributedTrainerMixin, PrecomputeRefLogpsRankConsistentMixin, KTOTrainer):
    """TRL's KTOTrainer plus EP/TP/PP via DistributedTrainerMixin. CP unsupported (see module docstring).

    ``PrecomputeRefLogpsRankConsistentMixin`` keeps ``precompute_ref_log_probs`` (the EP/TP
    full-finetune reference path) from deadlocking on its rank-divergent disk cache. PP is
    ``apo_zero_unpaired``-only and precompute-only with the reference log-probs shipped as a
    dataset column; the PP contract lives in ``_pp_loss_adapter``.
    """

    _tag_names = ["trl", "kto"]

    _supports_pp = True
    # TRL's KTO loss adds the aux term itself and this class does not override it, so aux_loss
    # balancing really does reach the graph here (unlike the other preference objectives).
    _consumes_router_aux_loss = True

    def __init__(self, *args, **kwargs):
        # TRL's fused KTO Liger path NPEs when ref_model is None; MUST precede _init_distributed_config.
        disable_trl_liger(
            ctor_config(args, kwargs),
            "Disabling TRL's use_liger_kernel for KTO: the experimental KTO Liger loss path is "
            "broken in TRL 1.6. Liger kernels are still applied at the model level by "
            "load_distributed_model.",
        )

        kwargs = self._init_distributed_config(kwargs, ctor_args=args)
        self._validate_reference_model(ctor_value(args, kwargs, "ref_model", _CTOR_POSITIONS))
        super().__init__(*args, **kwargs)
        # Post-super: TRL builds its own reference inside __init__ when beta != 0 without PEFT.
        self._validate_implicit_reference_model()
        self._setup_distributed_modes()

    def _validate_pp_mode(self, ctor_args: tuple, kwargs: dict) -> None:
        """Reject KTO configurations pipeline parallelism cannot honor (fail loud at construction).

        Runs BEFORE the model is split, so every raise here is cheap and rank-uniform. The KTO
        options the PP loss reproduces — ``beta``, ``desirable_weight``, ``undesirable_weight`` —
        need no gate; everything else that shapes the loss is the loss type itself, gated below.
        The remaining ctor parameters are read from ``kwargs`` only:
        ``require_model_and_args_kwargs`` forces ``model`` into kwargs, and any positional argument
        would rebind TRL's ``model`` slot and fail at ``super().__init__``.
        """
        require_model_and_args_kwargs(kwargs)
        training_args = kwargs["args"]
        if training_args.loss_type != "apo_zero_unpaired":
            raise ValueError(
                f"loss_type '{training_args.loss_type}' is not supported under pipeline "
                f"parallelism: the default 'kto' loss needs a world-global detached KL baseline "
                f"(per-rank batch mean, all-gathered, clamped) that no microbatch can compute, "
                f"plus a per-batch no-grad policy forward over the KL sequences — a full-model "
                f"pass even with precomputed reference log-probs. Only the KL-free "
                f"'apo_zero_unpaired' loss is decomposable and supported under PP."
            )
        reject_pp_ref_model(
            ctor_value(ctor_args, kwargs, "ref_model", _CTOR_POSITIONS), "the ref_logps dataset column"
        )
        require_precomputed_reference(
            "KTO",
            training_args,
            kwargs.get("train_dataset"),
            kwargs.get("eval_dataset"),
            (_REF_LOGPS_COLUMN,),
        )
        reject_pp_compute_metrics(
            kwargs.get("compute_metrics"),
            (
                "TRL's KTO "
                "prediction_step hands metrics the RAW logits plane (outputs.logits, paired with "
                "completion_input_ids), and a pipeline can only serve that by broadcasting "
                "[rows, max_length, vocab] logits from the last stage to every rank in the chain — "
                "the eval hop the adapter's predictions_fn exists to shrink, and multiple GB per "
                "row at a production vocabulary. Any smaller tensor would be a different "
                "prediction than the non-PP path's. Evaluate on eval_loss alone, or compute the "
                "metric in a non-PP run."
            ),
        )

    def _required_ref_logps_columns(self) -> tuple[str, ...]:
        """Columns TRL's KTO sweep would write; ``ref_KL_logps`` only when the loss carries a KL
        term (``calculate_KL``, never under PP)."""
        return (_REF_LOGPS_COLUMN, "ref_KL_logps") if self.calculate_KL else (_REF_LOGPS_COLUMN,)

    def _pp_loss_adapter(self) -> PPLossAdapter:
        """KTO's pipeline-loss contract: unpaired rows + dataset-precomputed reference log-probs.

        Rows are independent, so the microbatch split is safe at any size; ``ref_logps`` and the
        per-row desirability ``label`` ship as extra targets so the split keeps them row-aligned.
        The collator's ``completion_*`` keys are padded to the fixed shape in the batch transform,
        so no ``pad_spec`` is needed. ``pin_runtime_columns=False``: KTO's dataset columns feed
        TRL's unpaired-preference collator, so HF's column pruning must keep TRL's own signature
        set, not the runtime contract. ``eval_normalizer`` is training's pre-pad row count.
        """
        return PPLossAdapter(
            token_loss_fn=self._pp_kto_token_loss,
            batch_transform=self._pp_kto_batch_transform,
            normalizer=row_count_normalizer,
            extra_target_keys=("ref_logps", "label"),
            eval_normalizer=row_count_normalizer,
            pin_runtime_columns=False,
        )

    def _pp_kto_batch_transform(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """TRL's ``completion_*`` collator keys → the pipeline's fixed-shape batch contract.

        Labels are the completion token ids at their positions (ignore elsewhere) — exactly the
        positions TRL's shifted ``completion_mask`` scores. Right-pads to ``args.max_length``
        because the P2P buffer shapes freeze on the first step; ``label`` (a list of bools from
        the collator) becomes a per-row float tensor so the microbatch split can carry it.
        """
        input_ids = batch["completion_input_ids"]
        attention_mask = batch["completion_attention_mask"]
        completion_mask = batch["completion_mask"]
        labels = completion_labels(input_ids, completion_mask)
        # Over-length is unreachable with TRL's collator; the helper's raise guards a custom one.
        padded = pad_to_pipeline_length(
            {
                "input_ids": (input_ids, self._tokenizer.pad_token_id),
                "attention_mask": (attention_mask, 0),
                "labels": (labels, LABEL_IGNORE_INDEX),
            },
            self.args.max_length,
            "Truncate in the collator, or raise max_length.",
        )
        return {
            **padded,
            "ref_logps": batch["ref_logps"].float(),
            "label": torch.as_tensor(batch["label"], dtype=torch.float32, device=input_ids.device),
        }

    def _pp_kto_token_loss(self, logits: torch.Tensor, target: dict[str, torch.Tensor]) -> torch.Tensor:
        """Summed per-row apo_zero_unpaired losses over one microbatch (÷ row count by the runtime).

        Reproduces TRL's ``_compute_loss`` for the PP subset: per-row log-ratio against the
        precomputed reference, then ``desirable_weight · (1 − σ(β·logratio))`` for desirable rows
        and ``undesirable_weight · σ(β·logratio)`` for undesirable ones. The per-row ``where``
        makes a single-sided microbatch (all-desirable / all-undesirable) a plain partial sum —
        no empty-tensor branch. Deviation from TRL's reduction: TRL takes ``losses.nanmean()``,
        which would silently skip a NaN row; sum ÷ row count is equivalent under the assumption
        that no row is NaN — a NaN per-row loss means corrupted data/refs and should surface, not
        vanish from the mean.
        """
        seq_logps = sequence_logprobs(logits, target["labels"])
        logratios = seq_logps - target["ref_logps"].float()
        desirable = target["label"].bool()
        per_row = torch.where(
            desirable,
            self.desirable_weight * (1 - torch.sigmoid(self.beta * logratios)),
            self.undesirable_weight * torch.sigmoid(self.beta * logratios),
        )
        # Inert eval-padding rows must contribute exactly 0 (σ(0) = 0.5 otherwise) — train-time no-op.
        return (per_row * rows_with_labels(target["labels"]).float()).sum()
