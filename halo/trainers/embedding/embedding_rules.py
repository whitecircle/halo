"""Embedding-training construction refusals: loader geometry, sequence length and where LoRA may sit."""

import torch.nn as nn
from datasets import Dataset, DatasetDict, IterableDataset
from peft.tuners.tuners_utils import BaseTunerLayer
from sentence_transformers.base.sampler import BatchSamplers
from torch.distributed.tensor import DTensor

from halo.configs.embedding import EmbeddingConfig
from halo.distributed.parallelism_config import ParallelismConfig
from halo.log import KEY_PREVIEW_COUNT
from halo.runtime.ranks import launcher_global_world_size


def reject_uneven_sentence_transformers_batches(
    args: EmbeddingConfig | None,
    train_dataset: Dataset | DatasetDict | IterableDataset | None,
    *,
    toolkit_loader: bool,
) -> None:
    """A sentence-transformers loader cannot give every rank its batches alike with ``even_batches`` off on a
    remainder-keeping map-style dataset, or an iterable one dispatched or split by more file shards than processes."""
    world_size = launcher_global_world_size()
    if args is None or toolkit_loader or world_size <= 1:
        return
    if isinstance(train_dataset, IterableDataset):
        loader_config = args.accelerator_config
        if loader_config.dispatch_batches is False and (
            loader_config.split_batches or train_dataset.n_shards <= world_size
        ):
            return
        raise ValueError(
            f"An iterable train dataset ({train_dataset.n_shards} shard(s), {world_size} processes) with "
            f"accelerator_config dispatch_batches={loader_config.dispatch_batches} and "
            f"split_batches={loader_config.split_batches} cannot reach every rank alike: dispatched, the "
            f"sentence-transformers collator's string modality fields cannot be concatenated; split by shard, "
            f"the ranks get different row counts and the run hangs. Set dispatch_batches: false, and either "
            f"reshard the dataset to at most {world_size} shards or set split_batches: true."
        )
    if not args.dataloader_drop_last:
        raise ValueError(
            "dataloader_drop_last: false is not supported for multi-process embedding training on plain DP "
            "or pure EP: the sentence-transformers loader shards batches with accelerate's even_batches off, "
            "so when the batch count is not a multiple of the world size some ranks run one step more than "
            "the others and the run hangs in that step's collectives. Set dataloader_drop_last: true."
        )


def reject_unresolved_max_length(args: EmbeddingConfig | None) -> None:
    """A non-positive ``max_length`` means the context window only once the script resolves it; unresolved, the
    collator would cut every text to its special tokens."""
    if args is None or args.max_length is None or args.max_length > 0:
        return
    raise ValueError(
        f"max_length={args.max_length} reached the embedding trainer unresolved, and sentence-transformers' "
        f"collator truncates every text column to it. Pass a positive length, or null to keep the model's "
        f"max_seq_length; only the embedding script resolves a non-positive one to the backbone's context window."
    )


def reject_batch_sampler_on_toolkit_loader(args: EmbeddingConfig | None, *, toolkit_loader: bool) -> None:
    """The toolkit's DP-sharded loader never reads ``batch_sampler``, so ``no_duplicates`` would let in-batch
    duplicates through as false negatives."""
    if args is None or not toolkit_loader:
        return
    if BatchSamplers(args.batch_sampler) is BatchSamplers.BATCH_SAMPLER:
        return
    raise ValueError(
        f"batch_sampler: {BatchSamplers(args.batch_sampler).value} is not applied under tensor or "
        "expert-tensor parallelism or on a pre-sharded dataset: those runs batch through the toolkit's "
        "DP-sharded loader, which builds plain batches. Set batch_sampler: batch_sampler, or train on "
        "plain DP / pure EP, where the sentence-transformers loader applies it."
    )


def reject_injected_lora_parallelism(
    parallelism_config: ParallelismConfig, *, has_injected_lora: bool, has_ep_layers: bool
) -> None:
    """The save folds adapters over the backbone's own parameters, which under EP layers (grouped-GEMM at
    ``expert_parallel_size`` 1 too) hold expert tensors in the layers' own layout and names."""
    if not has_injected_lora or not (parallelism_config.has_ep_group or has_ep_layers):
        return
    raise ValueError(
        "LoRA for embedding training is not supported with Expert Parallelism (EP or expert-TP) or on the "
        "grouped-GEMM EP layers (use_grouped_gemm, the default, puts them on a MoE backbone at "
        "expert_parallel_size 1): the save folds the adapters over the backbone's own parameters, which "
        "there hold the EP layers' expert tensors in the layers' own layout and names (this rank's shards "
        "under EP), so the checkpoint would load nowhere. Drop --expert_parallel_size / "
        "--expert_tensor_parallel_size and set use_grouped_gemm: false, or full fine-tune this model."
    )


def reject_modules_outside_backbone(model: nn.Module, backbone: nn.Module, *, saves_rank_locally: bool) -> None:
    """Refuse what the save and resume cannot carry outside the ``auto_model`` backbone: any tuner layer and, where the
    save writes the other modules rank-locally, a parameter that trains or that FSDP2 sharded."""
    backbone_modules = {id(module) for module in backbone.modules()}
    tuners = [
        name
        for name, module in model.named_modules()
        if isinstance(module, BaseTunerLayer) and id(module) not in backbone_modules
    ]
    if tuners:
        raise ValueError(
            f"LoRA adapters outside the SentenceTransformer's transformer backbone ({tuners[:KEY_PREVIEW_COUNT]}): "
            f"checkpoints fold and resume the backbone's adapters only, so these would leave the export "
            f"and the resume state. Inject into model[0].auto_model, as the embedding script does."
        )
    if not saves_rank_locally:
        return
    backbone_params = {id(param) for param in backbone.parameters()}
    outside = [
        name
        for name, param in model.named_parameters()
        if id(param) not in backbone_params and (param.requires_grad or isinstance(param.data, DTensor))
    ]
    if outside:
        raise ValueError(
            f"Parameters outside the SentenceTransformer's transformer backbone that train or that FSDP2 "
            f"shards ({outside[:KEY_PREVIEW_COUNT]}) are not supported under FSDP2 (torchrun), TP, EP or the "
            f"grouped-GEMM EP layers: the save writes their modules rank-locally, so a sharded one is saved "
            f"as this rank's shard, and resume restores the backbone alone. Train this pipeline on a single "
            f"GPU or with DDP (accelerate launch with a MULTI_GPU config), with use_grouped_gemm: false on a MoE."
        )
