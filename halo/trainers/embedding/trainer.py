"""Distributed embedding trainer (sentence-transformers losses) with EP/TP support.

CP is not supported: embedding models require full-sequence pooling.
"""

import dataclasses
import inspect
import json
import logging
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from functools import partial
from typing import Any

import sentence_transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from datasets import Dataset, DatasetDict, IterableDataset
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from sentence_transformers.base.sampler import NoDuplicatesBatchSampler
from sentence_transformers.evaluation import SentenceEvaluator
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerBase
from transformers.data.data_collator import DataCollator
from transformers.trainer_callback import TrainerCallback
from trl.trainer.utils import disable_dropout_in_model

from halo.checkpoint.adapters import (
    resume_adapter_dir,
    unmarked_merged_checkpoint_reason,
)
from halo.checkpoint.atomic import apply_umask_mode, save_safetensors_file
from halo.checkpoint.format import ADAPTER_SAFETENSORS_FILE, RESUME_ADAPTER_DIR, RESUME_ADAPTER_MARKER_FILE
from halo.checkpoint.parallel.completeness import mark_resume_adapter_complete
from halo.checkpoint.parallel.context import CheckpointContext, CheckpointLoadContext
from halo.checkpoint.parallel.save import save_checkpoint, save_tokenizer
from halo.checkpoint.parallel.writing import stream_gathered_checkpoint
from halo.checkpoint.writing import normalize_gathered_state_dict
from halo.configs.embedding import EmbeddingConfig
from halo.distributed.fsdp import reshard_fsdp2_modules
from halo.distributed.parallelism_config import ParallelismConfig
from halo.log import warn_once
from halo.models.peft_adapters import lora_fold_targets
from halo.models.unwrapping import unwrap_framework_wrappers
from halo.runtime.collectives import broadcast_from_rank0
from halo.runtime.consensus import barrier_on_exit
from halo.runtime.dtensor import gather_full_tensors
from halo.runtime.filesystem import fs_aware_makedirs, is_fs_save_rank
from halo.runtime.ranks import is_global_main_process
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from halo.trainers.embedding.embedding_rules import (
    reject_batch_sampler_on_toolkit_loader,
    reject_injected_lora_parallelism,
    reject_modules_outside_backbone,
    reject_uneven_sentence_transformers_batches,
    reject_unresolved_max_length,
)
from halo.trainers.embedding.injected_lora import (
    folded_backbone_chunks,
    has_injected_lora,
    require_injected_lora_config,
    restore_injected_lora,
    trainable_tensors,
)
from halo.trainers.embedding.preloaded_transformer import (
    PreloadedTransformer,
    patch_sentence_transformer_gradient_checkpointing,
)
from halo.trainers.training_args_overrides import keep_all_dataset_columns

logger = logging.getLogger(__name__)

# Rows per text group the embedding metrics read: their similarity ranking is quadratic in it.
_METRIC_MAX_SAMPLES = 256


LOSS_REGISTRY: dict[str, type] = {
    "mnrl": losses.MultipleNegativesRankingLoss,
    "cached_mnrl": losses.CachedMultipleNegativesRankingLoss,
    "cosent": losses.CoSENTLoss,
    "angle": losses.AnglELoss,
    "cosine_similarity": losses.CosineSimilarityLoss,
    "triplet": losses.TripletLoss,
    "contrastive": losses.ContrastiveLoss,
    "online_contrastive": losses.OnlineContrastiveLoss,
    "batch_all_triplet": losses.BatchAllTripletLoss,
    "batch_hard_triplet": losses.BatchHardTripletLoss,
}


def build_embedding_loss(model: SentenceTransformer, config: EmbeddingConfig) -> nn.Module:
    """The configured sentence-transformers loss, wrapped in ``MatryoshkaLoss`` when dimensions are set."""
    loss_cls = LOSS_REGISTRY[config.loss_type]
    loss_kwargs: dict = {}
    # Read off the loss class's own signature, so it tracks sentence-transformers adding or removing it.
    if "scale" in inspect.signature(loss_cls.__init__).parameters:
        loss_kwargs["scale"] = config.loss_scale
    if config.loss_type == "cached_mnrl":
        loss_kwargs["mini_batch_size"] = config.cached_mnrl_mini_batch_size

    loss = loss_cls(model=model, **loss_kwargs)

    if config.matryoshka_dimensions:
        matryoshka_kwargs: dict = {"matryoshka_dims": config.matryoshka_dimensions}
        if config.matryoshka_weights:
            matryoshka_kwargs["matryoshka_weights"] = config.matryoshka_weights
        loss = losses.MatryoshkaLoss(model=model, loss=loss, **matryoshka_kwargs)

    return loss


def _group_samples(inputs: dict[str, Any]) -> list[int]:
    """The sample count of each text group of a collated batch, in column order: a packed (varlen) group's
    sequences, a padded group's rows."""
    samples = []
    for key, input_ids in inputs.items():
        if key.endswith("_input_ids"):
            offsets = inputs.get(f"{key[: -len('input_ids')]}cu_seq_lens_q")
            samples.append(input_ids.size(0) if offsets is None else offsets.numel() - 1)
    return samples


def _group_heads(chunks: list[torch.Tensor], group_samples: list[int]) -> list[torch.Tensor] | None:
    """The first :data:`_METRIC_MAX_SAMPLES` rows of each text group, the groups laid back to back over
    ``chunks`` with ``group_samples`` rows each, or ``None`` when ``chunks`` do not hold exactly those rows."""
    if not group_samples or sum(chunk.size(0) for chunk in chunks) != sum(group_samples):
        return None
    heads: list[list[torch.Tensor]] = [[] for _ in group_samples]
    starts = [sum(group_samples[:group]) for group in range(len(group_samples))]
    offset = 0
    for chunk in chunks:
        for group, (start, samples) in enumerate(zip(starts, group_samples, strict=True)):
            begin, end = max(start, offset), min(start + min(samples, _METRIC_MAX_SAMPLES), offset + chunk.size(0))
            if begin < end:
                heads[group].append(chunk[begin - offset : end - offset])
        offset += chunk.size(0)
    return [torch.cat(parts) for parts in heads]


def _apply_umask_mode_to_module(output_dir: str, module_path: str) -> None:
    """Give the weights a sentence-transformers module saved at ``module_path`` (its ``modules.json`` path)
    the umask mode (:func:`~halo.checkpoint.atomic.apply_umask_mode`) in place of the ``0600`` it writes them at.

    A module past the first owns its folder whole, nested module folders included (a ``Router``'s
    ``2_Router/query_0_Dense/``); the first module's path ``""`` is ``output_dir`` itself, which also holds other
    saves' folders, so there only the files directly under it.
    """
    module_dir = os.path.join(output_dir, module_path)
    folders = [folder for folder, _, _ in os.walk(module_dir)] if module_path else [module_dir]
    for folder in folders:
        apply_umask_mode(folder)


class EmbeddingTrainer(DistributedTrainerMixin, SentenceTransformerTrainer):
    """SentenceTransformerTrainer + DistributedTrainerMixin (FSDP/EP/TP)."""

    # SBERT losses return the batch's own mean and ignore num_items_in_batch, while
    # SentenceTransformer.forward's **kwargs would otherwise make HF infer the opposite.
    LOSS_IS_OWN_MEAN = True
    # no_duplicates defers a row that repeats a batch's value to a later batch, never drops it.
    DATASET_DRAWING_BATCH_SAMPLERS = (NoDuplicatesBatchSampler,)
    PP_REFUSAL = (
        "the default and most registered losses (MNRL, CoSENT/AnglE, online-contrastive, the "
        "batch-* triplet losses) are in-batch similarity matrices over the whole batch (optionally "
        "gathered across devices with gradient), so microbatching changes the negative set and "
        "therefore the loss — no exact per-microbatch form exists; every loss also runs several "
        "forwards per batch (the anchor column, then the candidate columns) rather than one per "
        "example, which the single-input, fused forward-backward pipeline step cannot drive; and the "
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
        save_max_shard_size: str | None = None,
        allow_optimizer_warm_restart: bool = False,
    ):
        patch_sentence_transformer_gradient_checkpointing()
        # The SentenceTransformer's ``config`` is its backbone's, so the seam sees a MoE's experts.
        self._init_distributed_config(
            {"model": model, "args": args},
            parallelism_config=parallelism_config,
            save_sharded_ep=save_sharded_ep,
            dataset_presharded=dataset_presharded,
            moe_balancing=moe_balancing,
            save_max_shard_size=save_max_shard_size,
            allow_optimizer_warm_restart=allow_optimizer_warm_restart,
        )
        toolkit_loader = self._needs_custom_dataloader()
        reject_batch_sampler_on_toolkit_loader(args, toolkit_loader=toolkit_loader)
        reject_unresolved_max_length(args)
        reject_uneven_sentence_transformers_batches(args, train_dataset, toolkit_loader=toolkit_loader)

        if loss is None and model is not None and args is not None:
            loss = build_embedding_loss(model, args)

        if args is not None and args.disable_dropout and model is not None:
            disable_dropout_in_model(model)

        # SentenceTransformerDataCollator does its own column conversion, so keep every column.
        keep_all_dataset_columns(args)

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
        self._warned: set[str] = set()
        self._configure_distributed_modes()
        backbone = self._backbone_model()
        reject_modules_outside_backbone(
            self._top_level_model(), backbone, saves_rank_locally=self._saves_rank_locally()
        )
        reject_injected_lora_parallelism(self.parallelism_config, has_injected_lora=has_injected_lora(backbone))
        # Structural, so an adapter the save cannot fold raises here on every rank rather than on the save
        # rank alone at the first checkpoint, where its peers would block in the save's next collective.
        lora_fold_targets(backbone)

    def compute_loss(
        self,
        model: SentenceTransformer,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, Any]]:
        """The sentence-transformers loss; on every eval batch and before each log step, the embedding metrics
        of the sentence embeddings the loss forward computed (:meth:`_capturing_sentence_embeddings`)."""
        is_eval = self._in_eval_loop
        group_samples = _group_samples(inputs) if isinstance(inputs, dict) else []
        if not group_samples or not (is_eval or self._should_capture_embedding_metrics()):
            return SentenceTransformerTrainer.compute_loss(self, model, inputs, return_outputs, num_items_in_batch)

        with self._capturing_sentence_embeddings(model) as chunks:
            result = SentenceTransformerTrainer.compute_loss(self, model, inputs, return_outputs, num_items_in_batch)
        if not is_eval:
            self._last_metrics_step = self.state.global_step
        heads = _group_heads(chunks, group_samples)
        if heads is None:
            warn_once(
                logger,
                self._warned,
                "embedding_metrics",
                "embed/* metrics skipped: the loss forwarded %d sentence embeddings for text groups of %s samples, "
                "so its forwards cannot be read group by group.",
                sum(chunk.size(0) for chunk in chunks),
                group_samples,
            )
            return result
        metrics = self._compute_embedding_metrics(heads)
        if is_eval:
            for k, v in metrics.items():
                self._eval_embedding_accum.setdefault(k, []).append(v)
        else:
            self._embedding_metrics = metrics
        return result

    @contextmanager
    def _capturing_sentence_embeddings(self, model: SentenceTransformer) -> Iterator[list[torch.Tensor]]:
        """Yield the list the block's pipeline forwards append their detached sentence embeddings to, in call
        order. The hook sits on the pipeline's last module, so a forward served from a loss's cache (Matryoshka's
        smaller dimensions) adds nothing; sentence-transformers' losses forward the text groups in column order,
        merged candidates column-major and cached losses chunk by chunk, so the rows lie group after group."""
        chunks: list[torch.Tensor] = []
        handle = unwrap_framework_wrappers(model)[-1].register_forward_hook(
            lambda _module, _inputs, features: chunks.append(features["sentence_embedding"].detach())
        )
        try:
            yield chunks
        finally:
            handle.remove()

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
        """The eval loop, with the embedding metrics captured every batch averaged into its output."""
        self._in_eval_loop = True
        self._eval_embedding_accum = {}
        try:
            output = super().evaluation_loop(
                dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
            )
        finally:
            self._in_eval_loop = False

        for k, values in self._eval_embedding_accum.items():
            output.metrics[f"{metric_key_prefix}_{k}"] = sum(values) / len(values)

        self._eval_embedding_accum = {}
        return output

    def _backbone_model(self) -> nn.Module:
        """The transformer backbone (``auto_model``) of the SentenceTransformer's first module."""
        model = self._top_level_model()
        if isinstance(model, SentenceTransformer):
            first_module = model[0]
            if hasattr(first_module, "auto_model"):
                return first_module.auto_model
            return first_module
        return super()._backbone_model()

    def _should_capture_embedding_metrics(self) -> bool:
        """True on steps where ``log()`` fires next; skips if already captured this step."""
        if self._last_metrics_step == self.state.global_step:
            return False
        next_step = self.state.global_step + 1
        # state.logging_steps, not args: HF resolves a ratio (0 < logging_steps < 1) against
        # max_steps there, and max(0.1, 1) would capture on every step.
        return next_step % max(int(self.state.logging_steps), 1) == 0

    @staticmethod
    def _compute_embedding_metrics(embeddings: list[torch.Tensor]) -> dict[str, float]:
        """``embed/norm`` and ``embed/std``; 2+ groups add ``embed/cos_sim``, ``embed/mrr`` and
        ``embed/recall@{1,3,10}``, 3+ groups ``embed/neg_cos_sim`` and ``embed/triplet_margin``; read to the host
        in one transfer."""
        normed = [F.normalize(e, p=2, dim=1) for e in embeddings]
        anchor_n = normed[0]

        metrics: dict[str, torch.Tensor] = {
            "embed/norm": embeddings[0].norm(dim=1).mean(),
            "embed/std": embeddings[0].std(dim=0).mean(),
        }

        if len(normed) >= 2:
            cos_sim = (anchor_n * normed[1]).sum(dim=1)
            metrics["embed/cos_sim"] = cos_sim.mean()

            if len(normed) >= 3:
                neg_cos_sim = (anchor_n * normed[2]).sum(dim=1)
                metrics["embed/neg_cos_sim"] = neg_cos_sim.mean()
                metrics["embed/triplet_margin"] = (cos_sim - neg_cos_sim).mean()

            candidates_n = torch.cat(normed[1:], dim=0)
            batch_size = anchor_n.size(0)
            sim_matrix = anchor_n @ candidates_n.T
            ranks = (-sim_matrix).argsort(dim=1).argsort(dim=1)
            target_ranks = ranks[torch.arange(batch_size), torch.arange(batch_size)] + 1

            metrics["embed/mrr"] = (1.0 / target_ranks.float()).mean()
            for k in (1, 3, 10):
                if k <= candidates_n.size(0):
                    metrics[f"embed/recall@{k}"] = (target_ranks <= k).float().mean()

        return dict(zip(metrics, torch.stack([value.float() for value in metrics.values()]).tolist(), strict=True))

    def _load_from_checkpoint(
        self, resume_from_checkpoint: str, model: nn.Module = None, *, for_best_model: bool = False
    ) -> None:
        """Resume an injected-LoRA run from its checkpoint's resume adapter, any other run through the mixin.

        The folded weights are never read: the run was rebuilt from the base with fresh adapters, restored
        exactly from :data:`RESUME_ADAPTER_DIR`. The marker, read on rank 0, is the verdict: a run without
        injected LoRA refuses a marked checkpoint, and an injected-LoRA run an unmarked one.
        """
        marked = broadcast_from_rank0(
            is_global_main_process() and resume_adapter_dir(resume_from_checkpoint) is not None
        )
        if not has_injected_lora(self._backbone_model()):
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
            raise ValueError(unmarked_merged_checkpoint_reason(resume_from_checkpoint))
        restore_injected_lora(self._top_level_model(), self._backbone_model(), resume_from_checkpoint)
        self._restore_state_beside_weights(resume_from_checkpoint, for_best_model=for_best_model)

    def _weights_load_context(self) -> CheckpointLoadContext:
        """The load context over the backbone, whose names every save writes; an unsharded reload stays with
        sentence-transformers' own, which rebuilds the whole pipeline. The optimizer store keeps the
        SentenceTransformer (:meth:`_checkpoint_load_context`), whose parameters the optimizer steps."""
        return dataclasses.replace(
            self._checkpoint_load_context(), model=self._backbone_model(), base_owns_whole_weight_load=True
        )

    def _load_weights_through_base(self, checkpoint: str, model: nn.Module | None) -> None:
        # Called on the class, in its one-argument form: super() would re-enter the mixin's load.
        SentenceTransformerTrainer._load_from_checkpoint(self, checkpoint)

    def _save_merged_checkpoint_resume_adapter(self, checkpoint_dir: str) -> None:
        """COLLECTIVE. Write what an injected-LoRA checkpoint resumes from; the mixin's hook for any other run.

        A fold cannot resume: it rounds the delta to the save dtype and leaves no adapter tensors for the
        restored optimizer moments. Each training checkpoint therefore also carries the trainable tensors
        unfolded, at their live dtype and under the top-level model's names, in
        :data:`~halo.checkpoint.format.RESUME_ADAPTER_DIR`, and each save rank marks its copy complete; with
        older markers removed before the save began, a failed write leaves the checkpoint unmarked.
        """
        if not has_injected_lora(self._backbone_model()):
            super()._save_merged_checkpoint_resume_adapter(checkpoint_dir)
            return
        model = self._top_level_model()
        reshard_fsdp2_modules(model)
        is_save_rank = is_fs_save_rank()
        state = gather_full_tensors(trainable_tensors(model).items(), retain=is_save_rank)
        # The scaling the adapters trained at, which the resume holds its own to.
        peft_config = require_injected_lora_config(model)
        adapter_dir = os.path.join(checkpoint_dir, RESUME_ADAPTER_DIR)
        fs_aware_makedirs(adapter_dir)
        with barrier_on_exit(f"Writing the resume adapter to {adapter_dir}"):
            if is_save_rank:
                peft_config.save_pretrained(adapter_dir)
                save_safetensors_file(state, os.path.join(adapter_dir, ADAPTER_SAFETENSORS_FILE))
        del state
        mark_resume_adapter_complete(checkpoint_dir, is_save_rank=is_save_rank)

    def _checkpoint_context(self) -> CheckpointContext:
        """The mixin's context, re-pointed from the ``SentenceTransformer`` at its ``auto_model`` backbone, which
        the savers write, with the tokenizer of the pipeline's first module when the trainer has none."""
        model = self._top_level_model()
        return dataclasses.replace(
            super()._checkpoint_context(),
            model=self._backbone_model(),
            tokenizer=self._export_tokenizer(model),
        )

    def _export_tokenizer(self, model: nn.Module) -> PreTrainedTokenizerBase | None:
        """The tokenizer the export writes: the trainer's, else the pipeline's first module's."""
        tokenizer = getattr(self, "processing_class", None)
        if tokenizer is None and isinstance(model, SentenceTransformer):
            tokenizer = getattr(model[0], "tokenizer", None)
        return tokenizer

    def _write_model_payload(self, ctx: CheckpointContext, output_dir: str, _internal_call: bool) -> None:
        """Write the backbone through :meth:`_save_distributed_embedding_model` where ST's own writer cannot:
        EP/TP and mixin-managed FSDP2 DTensors it would write un-gathered, and injected-LoRA adapter keys that
        reload as random base weights. Every other run takes the mixin's payload, whose fallback is ST's."""
        if self._saves_rank_locally() or has_injected_lora(self._backbone_model()):
            self._save_distributed_embedding_model(ctx, output_dir, _internal_call=_internal_call)
        else:
            super()._write_model_payload(ctx, output_dir, _internal_call)

    def _save(self, output_dir: str | None = None, state_dict=None) -> None:
        """sentence-transformers' save, each module's weights (``2_Dense/``, say) given the umask mode
        (:func:`_apply_umask_mode_to_module`)."""
        super()._save(output_dir, state_dict=state_dict)
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        with open(os.path.join(output_dir, "modules.json")) as f:
            for module in json.load(f):
                _apply_umask_mode_to_module(output_dir, module["path"])

    def _saves_rank_locally(self) -> bool:
        """Whether the backbone saves through the distributed ladder, which writes the other pipeline modules
        rank-locally: FSDP2, TP or EP."""
        config = self.parallelism_config
        return self._fsdp_wrapped or config.is_tp_mode or config.has_ep_group

    def _save_distributed_embedding_model(self, ctx: CheckpointContext, output_dir: str, _internal_call: bool = False):
        """COLLECTIVE. Save the backbone through the ``save_checkpoint`` ladder, or, under injected LoRA (no
        ``PeftModel``), streamed through the gathered writer with its adapters folded in
        (:func:`folded_backbone_chunks`); then the ST pipeline config. Only the save rank writes.
        """
        backbone = ctx.model
        # Fenced: a failed write (ENOSPC, say) raises on every rank instead of leaving the peers blocked.
        with barrier_on_exit(f"Writing the embedding model to {output_dir}"):
            if has_injected_lora(backbone):
                save_tokenizer(ctx, output_dir)
                stream_gathered_checkpoint(
                    backbone,
                    folded_backbone_chunks(backbone, retain=ctx.is_save_rank),
                    output_dir,
                    is_save_rank=ctx.is_save_rank,
                    max_shard_size=ctx.max_shard_size,
                    normalize=partial(normalize_gathered_state_dict, backbone),
                )
            elif not save_checkpoint(ctx, output_dir):
                # No active parallelism left to gather: ST's own writer handles the plain-tensor
                # layout. Not the mixin's save_model, which would rebuild a context around the ST
                # Sequential.
                SentenceTransformerTrainer.save_model(self, output_dir, _internal_call=_internal_call)

            # Same FS-aware save rank, so each node's dir is complete on a non-shared filesystem.
            model = self._top_level_model()
            if is_fs_save_rank() and isinstance(model, SentenceTransformer):
                self._save_st_pipeline_config(model, output_dir)

    def _save_st_pipeline_config(self, model: SentenceTransformer, output_dir: str):
        """Write the ST pipeline config so the output loads with ``SentenceTransformer(output_dir)``."""
        modules_config = []
        for idx, (name, module) in enumerate(model.named_children()):
            if idx == 0:
                get_config = getattr(module, "get_config_dict", None)
                if get_config is not None:
                    # Backbone save skips module.save(), so write the sentence_bert_config.json it reads back.
                    with open(os.path.join(output_dir, PreloadedTransformer.config_file_name), "w") as f:
                        json.dump(get_config(), f, indent=2)
                path, module_type = "", "sentence_transformers.models.Transformer"
            else:
                path, module_type = (
                    f"{idx}_{type(module).__name__}",
                    f"{type(module).__module__}.{type(module).__name__}",
                )
                module_dir = os.path.join(output_dir, path)
                os.makedirs(module_dir, exist_ok=True)
                module.save(module_dir)
                _apply_umask_mode_to_module(output_dir, path)
            modules_config.append({"idx": idx, "name": name, "path": path, "type": module_type})

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
