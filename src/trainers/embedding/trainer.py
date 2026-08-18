"""Distributed embedding trainer (sentence-transformers losses) with EP/TP support.

CP is not supported: embedding models require full-sequence pooling.
"""

import dataclasses
import inspect
import json
import os
from collections.abc import Callable
from typing import Any

import sentence_transformers
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
from accelerate.logging import get_logger
from datasets import Dataset, DatasetDict, IterableDataset
from peft.tuners.lora import LoraLayer
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer
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
from src.checkpoint.format import write_gathered_checkpoint
from src.configs.embedding_config import EmbeddingConfig
from src.distributed.checkpoint.context import CheckpointContext
from src.distributed.checkpoint.save import save_checkpoint
from src.distributed.checkpoint.write import gather_saveable_tensors
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import (
    barrier_on_exit,
    fs_aware_makedirs,
    fs_aware_save_rank,
)
from src.models.loading.config_levels import restore_special_token_ids
from src.models.loading.tokenizer_setup import pristine_model_max_length
from src.trainers.mixins.base import DistributedTrainerMixin

logger = get_logger(__name__, log_level="INFO")

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


def _merge_injected_lora_state_dict(state_dict: dict, scaling: float) -> dict:
    """Fold ``inject_adapter_in_model`` LoRA into base weights within a gathered state dict.

    The injected model is not a ``PeftModel``, so a plain save would write the adapter keys verbatim
    and reload as random base weights. Emit plain ``<m>.weight = base + scaling * (B @ A)`` and drop
    the adapter/base_layer keys. DoRA unsupported (non-linear magnitude reparam).
    """
    if any(".lora_magnitude_vector." in k for k in state_dict):
        raise NotImplementedError("Merging DoRA adapters on save is not supported for embedding training.")

    lora_a = {k.split(".lora_A.")[0]: v for k, v in state_dict.items() if ".lora_A." in k}
    lora_b = {k.split(".lora_B.")[0]: v for k, v in state_dict.items() if ".lora_B." in k}

    merged: dict[str, torch.Tensor] = {}
    for k, v in state_dict.items():
        if ".lora_A." in k or ".lora_B." in k:
            continue
        if k.endswith(".base_layer.weight"):
            prefix = k[: -len(".base_layer.weight")]
            w = v
            if prefix in lora_a and prefix in lora_b:
                delta = scaling * (lora_b[prefix].float() @ lora_a[prefix].float())
                w = (v.float() + delta).to(v.dtype)
            merged[f"{prefix}.weight"] = w
        elif k.endswith(".base_layer.bias"):
            merged[f"{k[: -len('.base_layer.bias')]}.bias"] = v
        else:
            merged[k] = v
    return merged


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

    def _validate_injected_lora_parallelism(self) -> None:
        """Reject in-place-injected LoRA under EP, where the save path cannot fold it.

        The injected-LoRA branch of :meth:`_save_distributed_embedding_model` merges the adapters out
        of a plain gathered state dict, which under EP holds this rank's expert shards under their
        local names, producing a checkpoint no loader accepts. TP is rejected one level up by the
        mixin's LoRA gate. Runs after wrapping, on the live model.
        """
        config = self.parallelism_config
        if not (config.is_ep_mode or config.is_ep_tp_mode) or not self._has_injected_lora():
            return
        raise ValueError(
            "LoRA for embedding training is not supported with Expert Parallelism: the EP save path "
            "gathers the expert layout directly and has no adapter-merge step, so the checkpoint "
            "would carry adapter keys that reload as random base weights. Drop "
            "--expert_parallel_size, or full fine-tune this model."
        )

    def _get_unwrapped_model(self) -> nn.Module:
        """Return the backbone the mixin needs (EP grad-norm, TP FSDP2 setup, etc.).

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
        """True if ``inject_adapter_in_model`` added LoRA layers (in-place, so not a PeftModel)."""
        backbone = backbone if backbone is not None else self._get_unwrapped_model()
        return any((".lora_A." in n) or (".lora_B." in n) for n, _ in backbone.named_parameters())

    def _lora_scaling(self, backbone: nn.Module) -> float:
        """Active-adapter LoRA scaling read from a live LoraLayer (same factor the forward used)."""
        for module in backbone.modules():
            if isinstance(module, LoraLayer) and getattr(module, "scaling", None):
                adapters = list(module.active_adapters) or list(module.scaling.keys())
                if adapters:
                    return float(module.scaling[adapters[0]])
        return 1.0

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
        return next_step > 0 and next_step % max(int(self.state.logging_steps), 1) == 0

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
        fallback as other trainers. Injected LoRA is not a ``PeftModel``, so its adapters must be
        folded into the gathered dict before it is written.

        Every branch gathers on every rank (collectives) and writes only on the save rank.
        """
        backbone = ctx.model

        # Fenced: one rank writes while all reach the trailing barrier, so a failed write (ENOSPC,
        # say) does not leave the peers blocked.
        with barrier_on_exit():
            if self._has_injected_lora(backbone):
                # Gather on all ranks (collective); only the writer retains (else N× host RAM per node).
                state_dict = gather_saveable_tensors(backbone, retain=ctx.is_save_rank)
                if ctx.is_save_rank:
                    state_dict = _merge_injected_lora_state_dict(state_dict, self._lora_scaling(backbone))
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
