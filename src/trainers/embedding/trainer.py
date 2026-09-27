"""Distributed embedding trainer (sentence-transformers losses) with EP/TP support.

CP is not supported: embedding models require full-sequence pooling.
"""

import dataclasses
import inspect
import json
import logging
import os
from collections.abc import Callable, Iterator
from typing import Any

import sentence_transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from datasets import Dataset, DatasetDict, IterableDataset
from peft.tuners.lora import LoraModel
from safetensors.torch import save_file
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
from sentence_transformers.base.sampler import BatchSamplers
from sentence_transformers.evaluation import SentenceEvaluator
from sentence_transformers.losses import (
    AnglELoss,
    BatchAllTripletLoss,
    BatchHardTripletLoss,
    CachedMultipleNegativesRankingLoss,
    ContrastiveLoss,
    CoSENTLoss,
    CosineSimilarityLoss,
    MatryoshkaLoss,
    MultipleNegativesRankingLoss,
    OnlineContrastiveLoss,
    TripletLoss,
)
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollator
from transformers.trainer_callback import TrainerCallback
from trl.trainer.utils import disable_dropout_in_model

import src.trainers.embedding.sentence_transformers_compat  # noqa: F401  installs ST's gradient-checkpointing signatures
from src.checkpoint.adapters import adapter_weight_paths, read_adapter_file
from src.checkpoint.format import (
    ADAPTER_SAFETENSORS_FILE,
    RESUME_ADAPTER_DIR,
    RESUME_ADAPTER_MARKER_FILE,
    resume_adapter_dir,
    write_gathered_checkpoint,
    write_resume_adapter_marker,
)
from src.configs.embedding_config import EmbeddingConfig
from src.distributed.checkpoint.context import CheckpointContext
from src.distributed.checkpoint.coordination import consensus_read
from src.distributed.checkpoint.loader import CheckpointLoader, built_from_checkpoint, weights_read_from
from src.distributed.checkpoint.peft import copy_full_tensor
from src.distributed.checkpoint.save import save_checkpoint
from src.distributed.checkpoint.write import gather_saveable_tensors, resolve_retained
from src.distributed.fsdp import reshard_fsdp2_modules
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import (
    barrier_on_exit,
    broadcast_from_rank0,
    fs_aware_makedirs,
    fs_aware_save_rank,
    is_global_main_process,
    reject_across_ranks,
)
from src.log import KEY_PREVIEW_COUNT
from src.models.loading.config_levels import restore_special_token_ids
from src.models.loading.tokenizer_setup import pristine_model_max_length
from src.models.structure import lora_fold_targets, lora_folded, normalize_peft_param_name, persistent_buffers
from src.trainers.mixins.base import DistributedTrainerMixin

logger = logging.getLogger(__name__)

# Metric encoding is a diagnostic re-forward under no_grad; a full batch would double peak activations.
_METRIC_MAX_SAMPLES = 256


LOSS_REGISTRY: dict[str, type] = {
    "mnrl": MultipleNegativesRankingLoss,
    "cached_mnrl": CachedMultipleNegativesRankingLoss,
    "cosent": CoSENTLoss,
    "angle": AnglELoss,
    "cosine_similarity": CosineSimilarityLoss,
    "triplet": TripletLoss,
    "contrastive": ContrastiveLoss,
    "online_contrastive": OnlineContrastiveLoss,
    "batch_all_triplet": BatchAllTripletLoss,
    "batch_hard_triplet": BatchHardTripletLoss,
}

# Read off each loss class's own signature, so the set tracks sentence-transformers adding or
# removing the parameter on a registered loss.
_SCALE_LOSSES = frozenset(
    name for name, loss_cls in LOSS_REGISTRY.items() if "scale" in inspect.signature(loss_cls.__init__).parameters
)


def create_loss(model: SentenceTransformer, config: EmbeddingConfig) -> nn.Module:
    """Create a loss function from config, optionally wrapping with MatryoshkaLoss."""
    loss_type = config.loss_type
    if loss_type not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss_type '{loss_type}'. Valid options: {sorted(LOSS_REGISTRY.keys())}")

    loss_cls = LOSS_REGISTRY[loss_type]
    loss_kwargs: dict = {}
    if loss_type in _SCALE_LOSSES:
        loss_kwargs["scale"] = config.loss_scale
    if loss_type == "cached_mnrl":
        loss_kwargs["mini_batch_size"] = config.cached_mnrl_mini_batch_size

    loss = loss_cls(model=model, **loss_kwargs)

    if config.matryoshka_dimensions:
        matryoshka_kwargs: dict = {"matryoshka_dims": config.matryoshka_dimensions}
        if config.matryoshka_weights:
            matryoshka_kwargs["matryoshka_weights"] = config.matryoshka_weights
        loss = MatryoshkaLoss(model=model, loss=loss, **matryoshka_kwargs)

    return loss


def _folded_backbone_items(backbone: nn.Module) -> Iterator[tuple[str, torch.Tensor]]:
    """The backbone's saveable tensors with its injected LoRA folded in, under the base model's names.

    The injected model is not a ``PeftModel``, so a plain save would write the adapter keys verbatim
    and reload as random base weights. Each base tensor a LoRA layer adapts is folded out of place
    with PEFT's merge (:func:`~src.models.structure.lora_folded`), the adapter tensors are dropped and
    ``base_layer`` is spelled out, one tensor at a time. Under FSDP2 each fold is a DTensor collective,
    issued in ``named_parameters`` order on every rank.
    """
    folds = lora_fold_targets(backbone)
    for name, param in backbone.named_parameters():
        plain = normalize_peft_param_name(name, LoraModel.prefix)
        if plain is not None:
            layers = folds.get(id(param))
            yield plain, lora_folded(param, layers) if layers else param.data
    for name, buffer in persistent_buffers(backbone):
        plain = normalize_peft_param_name(name, LoraModel.prefix)
        if plain is not None:
            yield plain, buffer


def _trainable_tensors(model: nn.Module) -> dict[str, torch.Tensor]:
    """The tensors an injected-LoRA run trains, which its fold cannot give back: every parameter with
    ``requires_grad``, the same set the optimizer steps. Structural, so rank-uniform in name and order."""
    return {name: param.data for name, param in model.named_parameters() if param.requires_grad}


def _resume_adapter_mismatch(saved: dict[str, torch.Tensor], live: dict[str, torch.Tensor], path: str) -> str | None:
    """Why the resume adapter at ``path`` cannot restore ``live`` exactly, or None.

    Names and full shapes must match one for one: a tensor left out would resume from
    initialization, and ``copy_`` would broadcast a saved rank-1 adapter into a wider one.
    """
    missing = sorted(live.keys() - saved.keys())
    unexpected = sorted(saved.keys() - live.keys())
    reshaped = [
        f"{name}: saved {tuple(saved[name].shape)}, live {tuple(live[name].shape)}"
        for name in sorted(live.keys() & saved.keys())
        if saved[name].shape != live[name].shape
    ]
    if not (missing or unexpected or reshaped):
        return None
    return (
        f"the resume adapter at {path} does not match this run's trainable tensors (was the run "
        f"relaunched with other lora_target_modules or lora_r?): missing {missing[:KEY_PREVIEW_COUNT]}, "
        f"unexpected {unexpected[:KEY_PREVIEW_COUNT]}, reshaped {reshaped[:KEY_PREVIEW_COUNT]}"
    )


class EmbeddingTrainer(DistributedTrainerMixin, SentenceTransformerTrainer):
    """SentenceTransformerTrainer + DistributedTrainerMixin (FSDP/EP/TP).

    CP is unsupported because pooling requires the full sequence.
    """

    _tag_names = ["trl", "embedding"]
    # SBERT losses return the batch's own mean and ignore num_items_in_batch, while
    # SentenceTransformer.forward's **kwargs would otherwise make HF infer the opposite.
    _loss_is_own_mean = True
    _supports_pp = False
    _pp_unsupported_reason = (
        "the default and most registered losses (MNRL, CoSENT/AnglE, online-contrastive, the "
        "batch-* triplet losses) are in-batch similarity matrices over the whole batch (optionally "
        "gathered across devices with gradient), so microbatching changes the negative set and "
        "therefore the loss — no exact per-microbatch form exists; every loss also runs one forward "
        "per sentence column (anchor, positive, each hard negative) rather than one per example, "
        "which the single-input, fused forward-backward pipeline step cannot drive; and the "
        "SentenceTransformer nn.Sequential has neither the backbone-with-layers layout nor the "
        "task head the stage split locates"
    )

    def __init__(
        self,
        model: SentenceTransformer | None = None,
        args: EmbeddingConfig | None = None,
        train_dataset: Dataset | DatasetDict | IterableDataset | None = None,
        eval_dataset: Dataset | DatasetDict | IterableDataset | None = None,
        loss: nn.Module | dict[str, nn.Module] | None = None,
        evaluator: SentenceEvaluator | list[SentenceEvaluator] | None = None,
        data_collator: DataCollator | None = None,
        tokenizer: PreTrainedTokenizerBase | Callable | None = None,
        callbacks: list[TrainerCallback] | None = None,
        optimizers: tuple = (None, None),
        parallelism_config: ParallelismConfig | None = None,
        save_sharded_ep: bool = False,
        dataset_presharded: bool = False,
        moe_balancing: str = "auto",
    ):
        self._init_distributed_config(
            {"args": args},
            parallelism_config=parallelism_config,
            save_sharded_ep=save_sharded_ep,
            dataset_presharded=dataset_presharded,
            moe_balancing=moe_balancing,
        )
        self._reject_batch_sampler_on_toolkit_loader(args)

        if loss is None and model is not None and args is not None:
            loss = create_loss(model, args)

        if args is not None and args.disable_dropout and model is not None:
            disable_dropout_in_model(model)

        # SentenceTransformerDataCollator does its own column conversion, so keep every column.
        if args is not None:
            args.remove_unused_columns = False

        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            loss=loss,
            evaluator=evaluator,
            data_collator=data_collator,
            processing_class=tokenizer,  # ST >=5 name; `tokenizer=` survives only on a deprecation shim
            callbacks=callbacks,
            optimizers=optimizers,
        )

        self._embedding_metrics: dict[str, float] = {}
        self._last_metrics_step: int = -1
        self._in_eval_loop: bool = False
        self._eval_embedding_accum: dict[str, list[float]] = {}
        self._setup_distributed_modes()
        self._validate_injected_lora_parallelism()
        self._validate_injected_lora_foldable()

    def _reject_batch_sampler_on_toolkit_loader(self, args: EmbeddingConfig | None) -> None:
        """Refuse a batch sampler the toolkit's loader would drop.

        The runs :meth:`get_train_dataloader` hands to the mixin's DP-sharded loader (TP/ETP, a
        pre-sharded dataset) batch a plain sampler and never read ``batch_sampler``, so
        ``no_duplicates`` would let in-batch duplicates through as false negatives, silently.
        """
        if args is None or not self._needs_custom_dataloader():
            return
        if BatchSamplers(args.batch_sampler) is BatchSamplers.BATCH_SAMPLER:
            return
        raise ValueError(
            f"batch_sampler: {BatchSamplers(args.batch_sampler).value} is not applied under tensor or "
            "expert-tensor parallelism or on a pre-sharded dataset: those runs batch through the toolkit's "
            "DP-sharded loader, which builds plain batches. Set batch_sampler: batch_sampler, or train on "
            "plain DP / pure EP, where the sentence-transformers loader applies it."
        )

    def _validate_injected_lora_parallelism(self) -> None:
        """Reject in-place-injected LoRA under EP, where the save path cannot fold it.

        The injected-LoRA branch of :meth:`_save_distributed_embedding_model` folds the adapters into
        the backbone's own parameter walk, which under EP holds this rank's expert shards under their
        local names, producing a checkpoint no loader accepts. TP is rejected one level up by the
        mixin's LoRA gate. Runs after wrapping, on the live model.
        """
        config = self.parallelism_config
        if not config.is_ep_mode or not self._has_injected_lora():
            return
        raise ValueError(
            "LoRA for embedding training is not supported with Expert Parallelism: the EP save path "
            "gathers the expert layout directly and has no adapter-merge step, so the checkpoint "
            "would carry adapter keys that reload as random base weights. Drop "
            "--expert_parallel_size, or full fine-tune this model."
        )

    def _validate_injected_lora_foldable(self) -> None:
        """Refuse an injected adapter the save cannot fold (:func:`~src.models.structure.lora_fold_targets`:
        ``nn.MultiheadAttention`` LoRA, trainable tokens, a variant other than DoRA). Structural, so it
        raises here on every rank rather than on the save rank alone at the first checkpoint, where its
        peers would block in the save's next collective.
        """
        lora_fold_targets(self._get_unwrapped_model())

    def _get_unwrapped_model(self) -> nn.Module:
        """The SentenceTransformer's transformer backbone (``auto_model``), for the backbone-specific
        introspection the base contract limits this to; whole-model work uses ``_top_level_model``.

        SentenceTransformer is an ``nn.Sequential`` whose first module is the
        Transformer (carrying ``.auto_model``), followed by Pooling/Normalize.
        """
        model = self._top_level_model()

        if isinstance(model, SentenceTransformer):
            first_module = list(model.children())[0]
            if hasattr(first_module, "auto_model"):
                return first_module.auto_model
            return first_module

        return super()._get_unwrapped_model()

    def _has_injected_lora(self, backbone: nn.Module | None = None) -> bool:
        """True if ``inject_adapter_in_model`` added LoRA layers (in-place, so not a PeftModel), of
        any kind and anywhere: an embedding target's ``lora_embedding_A``/``_B`` count as a linear
        one's do, and so does an adapter on a shared expert an EP layer adopted. The name test is
        ``inject_lora``'s own; this path builds no native expert LoRA for it to mistake."""
        backbone = backbone if backbone is not None else self._get_unwrapped_model()
        return any("lora_" in name for name, _ in backbone.named_parameters())

    def compute_loss(
        self,
        model: SentenceTransformer,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """Compute loss, optionally capturing embedding metrics via a separate no-grad encoding pass
        (decoupled so it works with cached losses like ``cached_mnrl``)."""
        result = SentenceTransformerTrainer.compute_loss(self, model, inputs, return_outputs, num_items_in_batch)

        is_eval = self._in_eval_loop
        should_capture = is_eval or self._should_capture_embedding_metrics()

        if should_capture:
            metrics = self._encode_and_compute_metrics(model, inputs)
            if not is_eval:
                # Advance unconditionally: this gate decides whether a collective forward runs.
                self._last_metrics_step = self.state.global_step
            if metrics:
                if is_eval:
                    for k, v in metrics.items():
                        self._eval_embedding_accum.setdefault(k, []).append(v)
                else:
                    self._embedding_metrics = metrics

        return result

    def _should_capture_embedding_metrics(self) -> bool:
        """True on steps where ``log()`` fires next; skips if already captured this step."""
        if self._last_metrics_step == self.state.global_step:
            return False
        next_step = self.state.global_step + 1
        # state.logging_steps, not args: HF resolves a ratio (0 < logging_steps < 1) against
        # max_steps there, and max(0.1, 1) would capture on every step, which is a no-grad encode
        # and a collective under EP/TP.
        return next_step % max(int(self.state.logging_steps), 1) == 0

    def _encode_and_compute_metrics(
        self,
        model: SentenceTransformer,
        inputs: dict[str, torch.Tensor | Any],
    ) -> dict[str, float]:
        """No-grad encode each text group in ``inputs`` and compute quality metrics, capped at
        ``_METRIC_MAX_SAMPLES`` samples per group.
        """
        prefixes = self._get_text_group_prefixes(inputs)
        if not prefixes:
            return {}

        # No try/except: ``model(features)`` is a collective, so swallowing on one rank hangs its peers.
        embeddings: list[torch.Tensor] = []
        with torch.no_grad():
            for prefix in prefixes:
                features = {k[len(prefix) :]: v for k, v in inputs.items() if k.startswith(prefix)}
                if not features:
                    continue
                output = model(self._cap_group_samples(features))
                embeddings.append(output["sentence_embedding"].detach())

        return self._compute_embedding_metrics(embeddings) if embeddings else {}

    def _cap_group_samples(self, features: dict[str, Any]) -> dict[str, Any]:
        """Cap one text group at ``_METRIC_MAX_SAMPLES`` samples, or return it whole.

        Only a padded group has a sliceable sample axis. For a flash-attention backbone
        sentence-transformers emits packed/varlen features (``cu_seq_lens_q`` offsets over a single
        flattened row, plus scalar and string entries), where dim 0 is not the sample axis and a
        positional slice would desynchronize the offsets from the tokens.
        """
        input_ids = features["input_ids"]
        if "cu_seq_lens_q" in features or input_ids.size(0) <= _METRIC_MAX_SAMPLES:
            return features
        return {
            key: value[:_METRIC_MAX_SAMPLES]
            if isinstance(value, torch.Tensor) and value.size(0) == input_ids.size(0)
            else value
            for key, value in features.items()
        }

    @staticmethod
    def _get_text_group_prefixes(inputs) -> list[str]:
        """Text group prefixes in the collated inputs (e.g. ``anchor_input_ids`` -> ``anchor_``)."""
        if not isinstance(inputs, dict):
            return []
        return [k[: -len("input_ids")] for k in inputs if k.endswith("_input_ids")]

    @staticmethod
    def _compute_embedding_metrics(embeddings: list[torch.Tensor]) -> dict[str, float]:
        """Embedding quality metrics, keyed by group count.

        Always: ``embed/norm``, ``embed/std``. 2+ groups add ``embed/cos_sim``,
        ``embed/mrr``, ``embed/recall@{1,3,10}``. 3+ groups add ``embed/neg_cos_sim``,
        ``embed/triplet_margin``.
        """
        normed = [F.normalize(e, p=2, dim=1) for e in embeddings]
        anchor_n = normed[0]

        metrics: dict[str, float] = {
            "embed/norm": embeddings[0].norm(dim=1).mean().item(),
            "embed/std": embeddings[0].std(dim=0).mean().item(),
        }

        if len(normed) >= 2:
            cos_sim = (anchor_n * normed[1]).sum(dim=1)
            metrics["embed/cos_sim"] = cos_sim.mean().item()

            if len(normed) >= 3:
                neg_cos_sim = (anchor_n * normed[2]).sum(dim=1)
                metrics["embed/neg_cos_sim"] = neg_cos_sim.mean().item()
                metrics["embed/triplet_margin"] = (cos_sim - neg_cos_sim).mean().item()

            candidates_n = torch.cat(normed[1:], dim=0)
            batch_size = anchor_n.size(0)
            sim_matrix = anchor_n @ candidates_n.T
            ranks = (-sim_matrix).argsort(dim=1).argsort(dim=1)
            target_ranks = ranks[torch.arange(batch_size), torch.arange(batch_size)] + 1
            target_ranks_f = target_ranks.float()

            metrics["embed/mrr"] = (1.0 / target_ranks_f).mean().item()
            for k in (1, 3, 10):
                if k <= candidates_n.size(0):
                    metrics[f"embed/recall@{k}"] = (target_ranks <= k).float().mean().item()

        return metrics

    def log(self, logs: dict[str, float], start_time: float | None = None):
        """Inject embedding metrics into logs before delegating to mixin/Trainer."""
        if self._embedding_metrics:
            logs.update(self._embedding_metrics)
            self._embedding_metrics = {}
        super().log(logs, start_time)

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: bool | None = None,
        ignore_keys: list[str] | None = None,
        metric_key_prefix: str = "eval",
    ):
        """Eval loop that captures embedding metrics every batch, averages them,
        and injects them into the output alongside ``eval_loss``.
        """
        self._in_eval_loop = True
        self._eval_embedding_accum = {}

        try:
            output = super().evaluation_loop(
                dataloader,
                description,
                prediction_loss_only,
                ignore_keys,
                metric_key_prefix,
            )
        finally:
            self._in_eval_loop = False

        for k, values in self._eval_embedding_accum.items():
            output.metrics[f"{metric_key_prefix}_{k}"] = sum(values) / len(values)

        self._eval_embedding_accum = {}
        return output

    def get_train_dataloader(self) -> DataLoader:
        """Train dataloader with TP-aware sharding.

        Plain DP and pure EP delegate to ST (preserves batch samplers like NO_DUPLICATES). TP/ETP take
        the mixin's dataloader: ST shards across ``world_size``, but ranks sharing a batch leave
        ``dp_size`` < ``world_size``. A pre-sharded dataset takes it too, since ST would re-shard an
        already-disjoint per-rank slice.
        """
        if self._needs_custom_dataloader():
            return DistributedTrainerMixin.get_train_dataloader(self)
        return SentenceTransformerTrainer.get_train_dataloader(self)

    def get_eval_dataloader(self, eval_dataset=None) -> DataLoader:
        """Eval dataloader with TP-aware sharding."""
        if self._needs_custom_dataloader():
            return DistributedTrainerMixin.get_eval_dataloader(self, eval_dataset)
        return SentenceTransformerTrainer.get_eval_dataloader(self, eval_dataset)

    def save_model(self, output_dir: str = None, _internal_call: bool = False):
        """Save model with parallelism-aware handling.

        EP/TP and mixin-managed FSDP2 hold the backbone as DTensors that ST's save_model would write
        un-gathered (unloadable), so they take the distributed path (gather + ST pipeline config).
        Single-GPU, DDP, accelerate-managed FSDP keep plain params → delegate to ST.
        """
        output_dir = output_dir or self.args.output_dir
        # A forward's transient unsharded params predate the last optimizer step; the resume adapter
        # written after this save reads the resharded ones, and the fold must read the same tensors.
        reshard_fsdp2_modules(self._top_level_model())
        fs_aware_makedirs(output_dir)

        # align_special_tokens collapsed the backbone eos list at train start; the mixin restore covers one branch.
        restore_special_token_ids(self._pristine_special_token_ids)

        ctx = self._checkpoint_context()
        with pristine_model_max_length(ctx.tokenizer):
            config = self.parallelism_config
            # _has_injected_lora: ST's save would write adapter keys that reload as random base weights.
            if config.is_ep_mode or config.is_tp_mode or self._fsdp_wrapped or self._has_injected_lora():
                self._save_distributed_embedding_model(ctx, output_dir, _internal_call=_internal_call)
            else:
                super().save_model(output_dir, _internal_call=_internal_call)
        # This override bypasses the mixin's sidecar; without it a resumed MoE run re-inits balancing.
        self._persist_router_balancing_biases(output_dir)
        self._mark_model_save_collectives_done()

    def _save_merged_checkpoint_resume_adapter(self, checkpoint_dir: str) -> None:
        """Write what an injected-LoRA checkpoint resumes from; the mixin's hook for any other run.

        Every injected-LoRA save folds the adapters into the weights it writes, and a fold cannot
        resume: it rounds the delta to the save dtype and leaves no adapter tensors for the restored
        optimizer moments to belong to. Each training checkpoint therefore also carries the run's
        trainable tensors unfolded, at their live dtype and under the top-level model's parameter
        names, in :data:`~src.checkpoint.format.RESUME_ADAPTER_DIR`; each save rank writes the marker
        the resume classifies on once its own copy is complete. With any older marker removed before
        the save began (:func:`~src.distributed.checkpoint.save.remove_stale_resume_marker`), a failed
        write leaves the checkpoint unmarked. The final ``save_model`` export carries neither.
        Collective.
        """
        if not self._has_injected_lora():
            super()._save_merged_checkpoint_resume_adapter(checkpoint_dir)
            return
        model = self._top_level_model()
        reshard_fsdp2_modules(model)
        is_save_rank = fs_aware_save_rank()
        state = resolve_retained(_trainable_tensors(model).items(), retain=is_save_rank)
        adapter_dir = os.path.join(checkpoint_dir, RESUME_ADAPTER_DIR)
        fs_aware_makedirs(adapter_dir)
        with barrier_on_exit():
            if is_save_rank:
                save_file(state, os.path.join(adapter_dir, ADAPTER_SAFETENSORS_FILE))
                write_resume_adapter_marker(checkpoint_dir)
                logger.info(f"Saved the resume adapter of folded checkpoint {checkpoint_dir} ({len(state)} tensors)")
        del state

    def _load_from_checkpoint(
        self, resume_from_checkpoint: str, model: nn.Module = None, *, for_best_model: bool = False
    ) -> None:
        """Resume an injected-LoRA run from its checkpoint's resume adapter; the mixin's loader otherwise.

        The checkpoint's weights are the fold and already hold the delta, so they are never read:
        the run was rebuilt from the base with fresh adapters, and its trainable tensors are
        restored exactly from :data:`RESUME_ADAPTER_DIR` before the optimizer state resumes onto
        them. The marker is the verdict, decided on rank 0: a run without injected LoRA refuses a
        marked checkpoint, and an injected-LoRA run refuses an unmarked one.
        """
        marked = broadcast_from_rank0(
            is_global_main_process() and resume_adapter_dir(resume_from_checkpoint) is not None
        )
        if not self._has_injected_lora():
            if marked:
                raise ValueError(
                    f"{resume_from_checkpoint} is an injected-LoRA checkpoint ({RESUME_ADAPTER_MARKER_FILE}): "
                    f"it resumes by restoring {RESUME_ADAPTER_DIR}/ onto the base model, but this run trains "
                    f"no injected adapters. Resume with the same use_peft / lora_* settings, or start a new "
                    f"run from its folded weights (model_name_or_path: {resume_from_checkpoint}, without "
                    f"resume_from_checkpoint)."
                )
            # The loader reshards what it is handed, the backbone; the FSDP2 root group is the whole
            # SentenceTransformer, whose eval forward before a best-model load leaves it unsharded.
            reshard_fsdp2_modules(self._top_level_model())
            super()._load_from_checkpoint(resume_from_checkpoint, model, for_best_model=for_best_model)
            return
        if not marked:
            raise ValueError(
                f"{resume_from_checkpoint} holds folded weights without a resume adapter (no "
                f"{RESUME_ADAPTER_MARKER_FILE}: a torn save, or one written without it), and this run "
                f"trains injected LoRA. Resuming would restart the adapters from initialization while "
                f"restoring their optimizer state. Resume from a checkpoint that carries its resume "
                f"adapter, or start a new run from the folded weights (model_name_or_path: "
                f"{resume_from_checkpoint}, without resume_from_checkpoint)."
            )
        self._restore_injected_lora(resume_from_checkpoint)
        self._restore_router_balancing_biases(resume_from_checkpoint)

    def _restore_injected_lora(self, checkpoint: str) -> None:
        """Copy the resume adapter into the live trainable tensors, bit-exact at an unchanged dtype.

        Refuses a model built from the checkpoint itself (its fold already holds the delta, so the
        restored adapters would apply it twice), a marked checkpoint whose adapter file is absent,
        and a file whose tensors differ from the live trainable set in name or shape (another
        ``lora_target_modules`` / ``lora_r``). Every verdict is rank-uniform. Each rank reads its own
        node's copy; plain tensors restore from it, and FSDP2 DTensors take mesh rank 0's through
        ``distribute_tensor``, a mesh collective issued in sorted key order on every rank.
        """
        adapter_dir = os.path.join(checkpoint, RESUME_ADAPTER_DIR)
        # realpath resolves node-locally, so rank 0's verdict is the world's.
        if broadcast_from_rank0(built_from_checkpoint(weights_read_from(self._get_unwrapped_model()), checkpoint)):
            raise ValueError(
                f"{checkpoint} holds folded weights, which already carry the adapter delta, and resume "
                f"restores the unfolded adapters from {adapter_dir} onto the BASE model. This model was "
                f"loaded from the checkpoint itself, so the delta would apply twice. Point "
                f"model_name_or_path at the base model the run started from."
            )
        saved, path = consensus_read(
            adapter_weight_paths(adapter_dir), read_adapter_file, what="Resume adapter", checkpoint=checkpoint
        )
        if path is None:
            raise RuntimeError(
                f"{checkpoint} is marked to resume from its adapter ({RESUME_ADAPTER_MARKER_FILE}), but "
                f"{adapter_dir} holds no adapter file on any rank, so the adapters would resume from "
                f"initialization. Resume from a complete checkpoint."
            )
        model = self._top_level_model()
        # Best-model loads follow an eval forward that left the FSDP2 tree unsharded.
        reshard_fsdp2_modules(model)
        live = _trainable_tensors(model)
        reject_across_ranks(_resume_adapter_mismatch(saved, live, path), "Injected-LoRA resume", ValueError)
        with torch.no_grad():
            for name in sorted(live):
                copy_full_tensor(live[name], saved[name])
        del saved
        if is_global_main_process():
            logger.info(f"Restored {len(live)} injected-LoRA tensors from {path}")

    def _checkpoint_loader(self) -> CheckpointLoader:
        """The mixin's weight loader, re-pointed at the backbone the savers write.

        Every save writes the ``auto_model`` backbone's names (:meth:`_checkpoint_context`), so a
        loader handed the ``SentenceTransformer``, whose names carry its ``0.<module>.`` prefix, would
        match none of them, and the FSDP2 / TP coverage gates would refuse every full fine-tune reload.
        The optimizer store keeps the ``SentenceTransformer``, whose parameters the optimizer steps.
        """
        return CheckpointLoader(
            dataclasses.replace(self._checkpoint_load_context(), model=self._get_unwrapped_model())
        )

    def _checkpoint_context(self) -> CheckpointContext:
        """The mixin's context, re-pointed at the backbone.

        The base factory snapshots ``_top_level_model()``, which for embedding training is the
        ``SentenceTransformer`` ``nn.Sequential``; the savers need the ``auto_model`` backbone
        instead, and the tokenizer is on the ST's first module when the trainer has none.
        """
        model = self._top_level_model()
        return dataclasses.replace(
            super()._checkpoint_context(),
            model=self._get_unwrapped_model(),
            tokenizer=self._resolve_tokenizer(model),
        )

    def _save_distributed_embedding_model(self, ctx: CheckpointContext, output_dir: str, _internal_call: bool = False):
        """Save the backbone in EP / TP / mixin-managed-FSDP2 mode, then the ST pipeline config.

        Everything but in-place-injected LoRA goes through the shared ``save_checkpoint`` ladder, so
        embedding exports get the same save dtype, hub expert layout, shard size and ``.bin``
        fallback as other trainers. Injected LoRA is not a ``PeftModel``, so its adapters are folded
        into the tensors as they are gathered (:func:`_folded_backbone_items`).

        Every branch gathers on every rank (collectives) and writes only on the save rank.
        """
        backbone = ctx.model

        # Fenced: one rank writes while all reach the trailing barrier, so a failed write (ENOSPC,
        # say) does not leave the peers blocked.
        with barrier_on_exit():
            if self._has_injected_lora(backbone):
                # Gather on all ranks (collective); only the writer retains (else N× host RAM per node).
                state_dict = gather_saveable_tensors(
                    backbone, retain=ctx.is_save_rank, items=_folded_backbone_items(backbone)
                )
                if ctx.is_save_rank:
                    write_gathered_checkpoint(backbone, state_dict, output_dir, max_shard_size=ctx.max_shard_size)
                    if ctx.tokenizer is not None:
                        ctx.tokenizer.save_pretrained(output_dir)
                del state_dict
            elif not save_checkpoint(ctx, output_dir):
                # No active parallelism left to gather: ST's own writer handles the plain-tensor
                # layout. Not the mixin's save_model, which would rebuild a context around the ST
                # Sequential.
                SentenceTransformerTrainer.save_model(self, output_dir, _internal_call=_internal_call)

            # Same FS-aware save rank, so each node's dir is complete on a non-shared filesystem.
            model = self._top_level_model()
            if fs_aware_save_rank() and isinstance(model, SentenceTransformer):
                self._save_st_pipeline_config(model, output_dir)

    def _resolve_tokenizer(self, model: nn.Module) -> PreTrainedTokenizerBase | None:
        """Resolve tokenizer from trainer or model for saving."""
        tokenizer = getattr(self, "processing_class", None)
        if tokenizer is None and isinstance(model, SentenceTransformer):
            first_module = list(model.children())[0]
            tokenizer = getattr(first_module, "tokenizer", None)
        return tokenizer

    def _save_st_pipeline_config(self, model: SentenceTransformer, output_dir: str):
        """Write ST pipeline config so output loads with ``SentenceTransformer(output_dir)``."""
        modules_config = []
        for idx, (name, module) in enumerate(model.named_children()):
            if idx == 0:
                get_config = getattr(module, "get_config_dict", None)
                if get_config is not None:
                    # Backbone save skips module.save(), so write the sentence_bert_config.json it reads back.
                    with open(os.path.join(output_dir, "sentence_bert_config.json"), "w") as f:
                        json.dump(get_config(), f, indent=2)
                modules_config.append(
                    {
                        "idx": idx,
                        "name": name,
                        "path": "",
                        "type": "sentence_transformers.models.Transformer",
                    }
                )
            else:
                module_dir_name = f"{idx}_{type(module).__name__}"
                module_path = os.path.join(output_dir, module_dir_name)
                os.makedirs(module_path, exist_ok=True)
                module.save(module_path)

                modules_config.append(
                    {
                        "idx": idx,
                        "name": name,
                        "path": module_dir_name,
                        "type": f"{type(module).__module__}.{type(module).__name__}",
                    }
                )

        with open(os.path.join(output_dir, "modules.json"), "w") as f:
            json.dump(modules_config, f, indent=2)

        st_config = {
            "model_type": "SentenceTransformer",
            "__version__": {
                "sentence_transformers": sentence_transformers.__version__,
                "transformers": transformers.__version__,
                "pytorch": torch.__version__,
            },
            "prompts": getattr(model, "prompts", {}),
            "default_prompt_name": getattr(model, "default_prompt_name", None),
            "similarity_fn_name": getattr(model, "similarity_fn_name", None),
        }
        with open(os.path.join(output_dir, "config_sentence_transformers.json"), "w") as f:
            json.dump(st_config, f, indent=2)
