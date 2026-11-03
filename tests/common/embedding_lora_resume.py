"""Two-phase resume of an embedding run: saves fold injected LoRA, so each checkpoint also carries the unfolded tensors
(``resume_adapter/``) and resume rebuilds from the base. ``--lora off`` is a full fine-tune; see :data:`MODES`. Under
LoRA, parallel shapes and ep1 MoEs only check the trainer's refusal."""

import os
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from accelerate.utils import extract_model_from_parallel
from datasets import Dataset
from peft.tuners.lora import LoraLayer
from safetensors.torch import load_file
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Dense
from tokenizers import Tokenizer, models, pre_tokenizers
from torch.distributed.tensor import DTensor
from transformers import AutoModel, PreTrainedTokenizerFast
from trl import get_peft_config

from halo.checkpoint.adapters import has_adapter_weight_file, resume_adapter_dir
from halo.checkpoint.format import ADAPTER_CONFIG_FILE, ADAPTER_SAFETENSORS_FILE, RESUME_ADAPTER_DIR
from halo.checkpoint.parallel.resume_source import resolve_resume_weights_source
from halo.checkpoint.reading import load_full_state_dict
from halo.configs.embedding import EmbeddingConfig, EmbeddingScriptArguments
from halo.configs.script_arguments import ModelArguments
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from halo.entrypoints.script import ScriptContext
from halo.trainers.embedding.trainer import EmbeddingTrainer
from scripts.training.embedding import build_sentence_transformer, inject_lora
from tests.common.adapters import LORA_ALPHA, LORA_R, injected_lora_fold
from tests.common.checkpoint_io import RestorePointSnapshot, loading_problems
from tests.common.distributed import pin_deterministic_ep_dispatch, shared_output_dir
from tests.common.dp_layout import dp_layout_checks, hsdp_layout, probe_pipeline_forward
from tests.common.memory import finish_phase
from tests.common.models import PARAPHRASE_MINILM
from tests.common.numerics import min_or_nan, relative_l2, tensors_equal_at_narrower_dtype
from tests.common.parallel_shape import parallel_shape_checks
from tests.common.rank_log import log
from tests.common.tiny_models import (
    TINY_DENSE_FAMILIES,
    TINY_GEMMA4_TEXT_FLEX_FAMILY,
    TINY_MOE_FAMILIES,
    TinyFamily,
    shared_tiny_family_checkpoint,
)
from tests.common.tolerances import TOL
from tests.common.training_checks import resumed_loss_deltas, snapshot_trainable, step_losses


@dataclass(frozen=True)
class Family:
    """One backbone family: the registry's random-init ``tiny`` model saved as a checkpoint, or (``tiny`` None) the hub
    checkpoint ``hub``."""

    targets: tuple[str, ...]
    embedding: str
    pooling: str
    tiny: TinyFamily | None = None
    hub: str | None = None
    attn_implementation: str | None = None
    moe: bool = False


FAMILIES = {
    "bert": Family(("query", "value"), "word_embeddings", "mean", hub=PARAPHRASE_MINILM),
    **{
        name: Family(("q_proj", "v_proj"), "embed_tokens", "lasttoken", tiny)
        for name, tiny in TINY_DENSE_FAMILIES.items()
    },
    # FA4, the EP/TP loader's Blackwell default, does not compile the tiny heads. A CUDA load builds Gemma 4 on
    # ``sdpa`` as flex-sliding, whose FlexAttention takes heads of 16 or more.
    "gemma4": Family(
        ("q_proj", "v_proj"),
        "embed_tokens",
        "lasttoken",
        TINY_GEMMA4_TEXT_FLEX_FAMILY,
        attn_implementation="sdpa",
        moe=True,
    ),
    "gpt_oss": Family(("q_proj", "v_proj"), "embed_tokens", "lasttoken", TINY_MOE_FAMILIES["gpt_oss"], moe=True),
}
LORA_TARGETS = ("attention", "mixed", "embedding", "dora", "off")
TRAIN_MODES = ("single", "fsdp", "ddp", "presharded")
# The FSDP2 shapes past the default. The ep1 shapes put a MoE on the grouped-GEMM EP layers, which refuse injected
# LoRA.
DP_MODES = {
    "zero3": {"fsdp_reshard_after_forward": True},
    "hsdp": hsdp_layout(4),
    "ep1": {"use_grouped_gemm": True},
    "ep1_unsharded": {"use_grouped_gemm": True, "fsdp_shard_ep1_experts": False},
    "ep1_single": {"use_grouped_gemm": True},
}
# Refused under LoRA, trained under a full fine-tune.
PARALLEL_MODES = {
    "tp": {"tp_size": 2},
    "tpdp": {"tp_size": 2},
    "ep": {"ep_size": 2},
    "etp": {"expert_tp_size": 2},
    "eptp": {"ep_size": 2, "tp_size": 2},
    "epetp": {"ep_size": 2, "expert_tp_size": 2},
    "ep2dp2": {"ep_size": 2},
    "ep2x2dom": {"ep_size": 2, "gpus_per_node": 2},
}
MODES = TRAIN_MODES + tuple(DP_MODES) + tuple(PARALLEL_MODES)
WORLD_SIZES = {
    mode: {"single": 1, "ep1_single": 1, "tpdp": 4, "hsdp": 4, "epetp": 4, "ep2dp2": 4, "ep2x2dom": 4}.get(mode, 2)
    for mode in MODES
}

SEED = 42
TOTAL_STEPS = 6
SAVE_AT_STEP = 3
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
# A fold whose delta DTensor contracts the sharded rank dim shard-locally measured off PEFT's merge on 9.6-9.9% of
# moved elements in BERT's 30522-row embedding, by up to 1.9e-3 of its largest value. Every other fold is bit for bit.
PARTIAL_FOLD_MAX_OFF_FRACTION = 0.25
PARTIAL_FOLD_MAX_STEP = 2**-7
# MNRL near zero would let the loss comparisons pass for any adapters.
MIN_INFORMATIVE_LOSS = 0.5
EP_LAYERS_LORA_REFUSAL = (
    "LoRA for embedding training is not supported with Expert Parallelism (EP or expert-TP) or on the grouped-GEMM "
    "EP layers"
)
# What ``accelerate launch`` with a MULTI_GPU config exports; the trainer then leaves DDP to accelerate.
ACCELERATE_LAUNCH_ENV = {"ACCELERATE_MIXED_PRECISION": "bf16"}
_SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[EOS]")


def _lora_targets(family: Family, lora: str) -> tuple[str, ...] | None:
    return {
        "attention": family.targets,
        "mixed": (family.embedding, *family.targets),
        "embedding": (family.embedding,),
        "dora": family.targets,
        "off": None,
    }[lora]


def _model_config(family: Family, source: str, lora: str) -> ModelArguments:
    targets = _lora_targets(family, lora)
    return ModelArguments(
        model_name_or_path=source,
        attn_implementation=family.attn_implementation,
        use_peft=targets is not None,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        lora_target_modules=list(targets) if targets is not None else None,
        use_dora=lora == "dora",
    )


def _requested_fields(family: Family, mode: str, lora: str) -> dict:
    if mode in PARALLEL_MODES:
        return PARALLEL_MODES[mode]
    # With no EP layers the resolver stays on the base, so the resume reads the checkpoint into a model that does not
    # hold it. Injected LoRA trains a MoE on its stock experts, since the EP layers refuse it.
    return {"use_grouped_gemm": lora != "off" and not family.moe, **DP_MODES.get(mode, {})}


def _parallelism_config(family: Family, mode: str, lora: str) -> ParallelismConfig:
    return ParallelismConfig(**_requested_fields(family, mode, lora))


def _refuses_injected_lora(family: Family, mode: str) -> bool:
    return mode in PARALLEL_MODES or (family.moe and DP_MODES.get(mode, {}).get("use_grouped_gemm", False))


def _pair_texts() -> tuple[list[str], list[str]]:
    """Pairs that differ only by an index through a fixed permutation, so in-batch negatives are as close as the
    positive and MNRL stays O(1)."""
    count = 256
    partner = torch.randperm(count, generator=torch.Generator().manual_seed(SEED)).tolist()
    anchors = [f"What is the capital of country {i}?" for i in range(count)]
    positives = [f"It is a large city located in country {partner[i]}." for i in range(count)]
    return anchors, positives


def _pairs(ctx, mode: str) -> Dataset:
    """``presharded`` hands each rank its own disjoint slice, as a pre-sharded dataset load does."""
    anchors, positives = _pair_texts()
    dataset = Dataset.from_dict({"anchor": anchors, "positive": positives})
    if mode == "presharded":
        return dataset.shard(num_shards=ctx.world_size, index=ctx.rank, contiguous=True)
    return dataset


def _word_tokenizer() -> PreTrainedTokenizerFast:
    """A word-level tokenizer over exactly the dataset's words, so a tiny vocabulary covers it."""
    pre_tokenizer = pre_tokenizers.Whitespace()
    anchors, positives = _pair_texts()
    words = sorted({word for text in anchors + positives for word, _span in pre_tokenizer.pre_tokenize_str(text)})
    vocab = {token: index for index, token in enumerate((*_SPECIAL_TOKENS, *words))}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizer
    return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", unk_token="[UNK]", eos_token="[EOS]")


def _base_source(ctx, family_name: str) -> str:
    """The family's base checkpoint: the hub one, or its seeded tiny one, which rank 0 saves for every rank."""
    family = FAMILIES[family_name]
    if family.hub is not None:
        return family.hub
    return shared_tiny_family_checkpoint(ctx, family.tiny, f"embedding_base_{family_name}", _word_tokenizer(), SEED)


def _config(output_dir: str, family: Family, *, save: bool, gradient_checkpointing: bool) -> EmbeddingConfig:
    return EmbeddingConfig(
        output_dir=output_dir,
        max_steps=TOTAL_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="constant",
        bf16=True,
        logging_steps=1,
        save_strategy="steps" if save else "no",
        save_steps=SAVE_AT_STEP,
        eval_strategy="no",
        report_to="none",
        logging_nan_inf_filter=False,
        loss_type="mnrl",
        pooling_mode=family.pooling,
        max_length=64,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        disable_dropout=True,
        use_liger_kernel=False,
        gradient_checkpointing=gradient_checkpointing,
        seed=SEED,
    )


def _build(ctx, model_config: ModelArguments, config: EmbeddingConfig, parallelism_config: ParallelismConfig):
    source = model_config.model_name_or_path
    args = EmbeddingScriptArguments()
    context = ScriptContext(parallelism_config, ctx.local_rank, None, source, EmbeddingTrainer)
    model = build_sentence_transformer(context, config, model_config, args)
    inject_lora(model, model_config, args)
    return model


def _make_trainer(
    ctx,
    family: Family,
    mode: str,
    lora: str,
    source: str,
    output_dir: str,
    *,
    save: bool,
    gradient_checkpointing: bool,
):
    config = _config(output_dir, family, save=save, gradient_checkpointing=gradient_checkpointing)
    parallelism_config = _parallelism_config(family, mode, lora)
    return EmbeddingTrainer(
        model=_build(ctx, _model_config(family, source, lora), config, parallelism_config),
        args=config,
        train_dataset=_pairs(ctx, mode),
        parallelism_config=parallelism_config,
        dataset_presharded=mode == "presharded",
    )


def backbone_prefix(model: SentenceTransformer) -> str:
    backbone = model[0].auto_model
    return next(name for name, module in model.named_modules() if module is backbone) + "."


def _partial_fold_targets(model) -> set[str]:
    """Backbone names of LoRA targets whose delta is a ``Partial`` DTensor, so the fold sums bf16 partials.
    Collective."""
    model = extract_model_from_parallel(model, recursive=True)
    prefix = backbone_prefix(model)
    partial = set()
    with torch.no_grad():
        for name, module in model.named_modules():
            if not isinstance(module, LoraLayer):
                continue
            for adapter in module.active_adapters:
                delta = module.get_delta_weight(adapter)
                if isinstance(delta, DTensor) and any(placement.is_partial() for placement in delta.placements):
                    partial.add(f"{name[len(prefix) :]}.weight")
    return partial


class _RestorePoint(RestorePointSnapshot):
    """The restore-point snapshot with whole trainable ``tensors`` and, at the save, the ``partial_folds``."""

    def __init__(self, event: str, trainer: EmbeddingTrainer):
        super().__init__(event, trainer, capture_optimizer=False)

    def extra(self) -> dict:
        extra = {"tensors": snapshot_trainable(self.trainer.model)}
        if self.event == "save":
            extra["partial_folds"] = _partial_fold_targets(self.trainer.model)
        return extra


def _mode_checks(ctx, family: Family, mode: str, lora: str, trainer: EmbeddingTrainer) -> dict[str, bool]:
    """The row runs the shape it names; otherwise every comparison below is vacuous."""
    params = list(trainer.model.named_parameters())
    adapters = [param for name, param in params if ".lora_" in name]
    if lora == "off":
        checks = {"model_trains_without_adapters": not adapters and any(param.requires_grad for _, param in params)}
    else:
        checks = {"model_carries_injected_adapters": bool(adapters)}
    trainable = [param for _, param in params if param.requires_grad]
    if mode == "single":
        checks["mode_is_single_process"] = ctx.world_size == 1 and not trainer._fsdp_wrapped
    elif mode == "ddp":
        checks["mode_is_accelerate_ddp"] = trainer._accelerate_manages_ddp and not trainer._fsdp_wrapped
    elif mode == "tpdp":
        checks["mode_is_tp_over_fsdp2"] = trainer.parallelism_config.is_tp_mode and trainer._fsdp_wrapped
    elif mode == "tp":
        checks["mode_is_pure_tp_with_dtensor_params"] = (
            trainer.parallelism_config.is_tp_mode
            and not trainer._fsdp_wrapped
            and any(isinstance(param.data, DTensor) for param in trainable)
        )
    elif mode == "ep":
        checks["mode_is_ep"] = trainer.parallelism_config.has_ep_group and bool(trainer._ep_layers())
    elif mode in PARALLEL_MODES or mode in DP_MODES:
        # Axes the shared probes read off the model before training; the data-parallel half after it.
        checks |= parallel_shape_checks(trainer._backbone_model(), _parallelism_config(family, mode, lora))
        if WORLD_SIZES[mode] == 1:
            checks["mode_is_single_process"] = ctx.world_size == 1 and not trainer._fsdp_wrapped
    else:
        checks["mode_is_fsdp2_with_dtensor_params"] = trainer._fsdp_wrapped and all(
            isinstance(param.data, DTensor) for param in trainable
        )
    return checks


def _loads_and_encodes(directory: str, device, pooling: str) -> dict[str, bool]:
    """``directory`` as stock ``AutoModel`` and ``SentenceTransformer`` load it; a save with no pipeline config would
    serve a mean-pooled bare backbone."""
    checks = {}
    _backbone, info = AutoModel.from_pretrained(directory, output_loading_info=True)
    problems = loading_problems(info)
    if problems:
        log(f"  from_pretrained loading info for {directory}: {problems}")
    checks["loads_with_stock_from_pretrained"] = not problems
    served = SentenceTransformer(directory, device=str(device))
    embeddings = served.encode(
        ["What is the capital of country 3?", "It is a large city located in country 7."], convert_to_tensor=True
    )
    checks["loads_and_encodes_with_stock_sentence_transformer"] = bool(
        embeddings.shape[0] == 2 and torch.isfinite(embeddings).all()
    )
    pipeline = [(type(module).__name__, getattr(module, "pooling_mode", None)) for module in list(served)[1:]]
    log(f"  {directory} serves the pipeline {pipeline}")
    checks["serves_the_configured_pipeline"] = pipeline == [("Pooling", pooling), ("Normalize", None)]
    return checks


def _serves(served: torch.Tensor, expected: torch.Tensor, base: torch.Tensor, partial: bool) -> bool:
    """Whether ``served`` is PEFT's single-device merge: bit for bit, or for a ``partial`` delta off by rounding on a
    minority of moved elements. Shard-local bf16 partials round differently wherever they cancel, so a per-element ulp
    bound does not hold. A misscaled delta changes nearly every moved element."""
    if torch.equal(served, expected):
        return True
    if not partial or served.shape != expected.shape:
        return False
    served, expected, base = served.float(), expected.float(), base.float()
    moved = (expected != base) | (served != base)
    return bool(
        (served != expected)[moved].float().mean() <= PARTIAL_FOLD_MAX_OFF_FRACTION
        and (served - expected).abs().max() <= PARTIAL_FOLD_MAX_STEP * expected.abs().max()
    )


def _serving_checks(
    checkpoint: str,
    model_config: ModelArguments,
    at_save: dict[str, torch.Tensor],
    prefix: str,
    embedding: str | None,
    partial_folds: set[str],
    pooling: str,
    device,
) -> dict[str, bool]:
    """The checkpoint as a server loads it, and the resume state beside it. Rank-local reads only."""
    adapter_dir = resume_adapter_dir(checkpoint)
    checks = {"checkpoint_is_marked_for_adapter_resume": adapter_dir is not None}
    adapter_file = os.path.join(adapter_dir, ADAPTER_SAFETENSORS_FILE) if adapter_dir is not None else None
    saved = load_file(adapter_file) if adapter_file is not None and os.path.isfile(adapter_file) else {}
    checks["resume_adapter_holds_the_live_tensors_bit_for_bit"] = (
        bool(at_save)
        and set(saved) == set(at_save)
        and all(torch.equal(saved[key], value) for key, value in at_save.items())
    )
    checks["no_adapter_file_or_config_at_the_root"] = not has_adapter_weight_file(checkpoint) and not os.path.exists(
        os.path.join(checkpoint, ADAPTER_CONFIG_FILE)
    )
    checks.update(_loads_and_encodes(checkpoint, device, pooling))
    base = AutoModel.from_pretrained(model_config.model_name_or_path, dtype=torch.bfloat16).state_dict()
    # TRL's LoRA builder, not the script's: its exclusion scan is collective, and this runs on rank 0 alone.
    backbone = AutoModel.from_pretrained(model_config.model_name_or_path, dtype=torch.bfloat16).to(device)
    expected = injected_lora_fold(backbone, get_peft_config(model_config), at_save, prefix)
    served = AutoModel.from_pretrained(checkpoint, dtype=torch.bfloat16).state_dict()
    unfolded = sorted(
        key
        for key, value in expected.items()
        if key not in served or not _serves(served[key], value, base[key], key in partial_folds)
    )
    unmoved = sorted(key for key, value in expected.items() if torch.equal(value, base[key]))
    embedding_folds = [key for key in expected if key.endswith(f".{embedding}.weight") or key == f"{embedding}.weight"]
    checks["lora_targets_serve_the_fold_of_the_live_adapters"] = bool(expected) and not unfolded
    checks["premise_every_fold_moves_its_base"] = bool(expected) and not unmoved
    if embedding is not None:
        checks["the_input_embedding_is_folded"] = len(embedding_folds) == 1
    checks["partial_folds_are_lora_targets"] = partial_folds <= set(expected)
    log(
        f"  {len(expected) - len(unfolded)}/{len(expected)} LoRA targets serve PEFT's merge "
        f"(embedding folds {embedding_folds}; partial deltas {sorted(partial_folds)}) {unfolded[:2]}; "
        f"unmoved {unmoved[:2]}"
    )
    return checks


def _bit_equal(expected: dict[str, torch.Tensor], actual: dict[str, torch.Tensor], what: str) -> bool:
    unequal = sorted(key for key in expected if key not in actual or not torch.equal(expected[key], actual[key]))
    log(f"  {what}: {len(expected) - len(unequal)}/{len(expected)} tensors bit-equal {unequal[:2]}")
    return bool(expected) and set(expected) == set(actual) and not unequal


def _refusal_row(ctx, family: Family, mode: str, lora: str, source: str, shared_dir: str) -> dict:
    """LoRA under TP / EP or on a MoE's ep1 EP layers: the script's loader for that shape, then the trainer must
    refuse."""
    config = _config(os.path.join(shared_dir, f"refused_{mode}"), family, save=False, gradient_checkpointing=False)
    parallelism_config = _parallelism_config(family, mode, lora)
    model = _build(ctx, _model_config(family, source, lora), config, parallelism_config)
    error = None
    try:
        trainer = EmbeddingTrainer(
            model=model, args=config, train_dataset=_pairs(ctx, mode), parallelism_config=parallelism_config
        )
        ctx.on_teardown(trainer.cleanup_ep)
    except ValueError as e:
        error = str(e)
    log(f"  {mode} construction: {'refused: ' + error.splitlines()[0] if error else 'ACCEPTED'}")
    expected = EP_LAYERS_LORA_REFUSAL if mode in DP_MODES else "LoRA"
    return {"checks": {f"{mode}_refuses_injected_lora_at_construction": error is not None and expected in error}}


def _head_row(ctx, family: Family, mode: str, lora: str, source: str, shared_dir: str) -> dict:
    """A projection head after the pooling: FSDP2, TP and EP save it rank-locally and resume the backbone alone, so the
    trainer must refuse it. A single process and DDP save and resume the whole pipeline, and accept it."""
    config = _config(os.path.join(shared_dir, f"head_{mode}"), family, save=False, gradient_checkpointing=False)
    parallelism_config = _parallelism_config(family, mode, lora)
    model = _build(ctx, _model_config(family, source, lora), config, parallelism_config)
    dim = model.get_embedding_dimension()
    head = Dense(in_features=dim, out_features=dim).to(device=ctx.device, dtype=torch.bfloat16)
    head.requires_grad_(lora == "off")
    model.append(head)
    error = None
    try:
        trainer = EmbeddingTrainer(
            model=model, args=config, train_dataset=_pairs(ctx, mode), parallelism_config=parallelism_config
        )
        ctx.on_teardown(trainer.cleanup_ep)
    except ValueError as e:
        error = str(e)
    log(f"  {mode} construction with a head: {'refused: ' + error.splitlines()[0][:160] if error else 'ACCEPTED'}")
    if mode in ("ddp", "single"):
        return {"checks": {f"{mode}_accepts_a_head_outside_the_backbone": error is None}}
    return {
        "checks": {
            f"{mode}_refuses_a_head_outside_the_backbone": error is not None
            and "outside the SentenceTransformer's transformer backbone" in error
        }
    }


def _loss_checks(uninterrupted: list[float], resumed: list[float], checks: dict, metrics: dict) -> None:
    """Every resumed loss replays the uninterrupted run's; the first reads only restored state, so it is reported on
    its own."""
    deltas = resumed_loss_deltas(uninterrupted, resumed, save_step=SAVE_AT_STEP, total_steps=TOTAL_STEPS)
    checks["resumed_ran_remaining_steps"] = deltas is not None
    if deltas is None:
        return
    metrics["first_resumed_loss_delta"] = deltas[0]
    metrics["resumed_loss_max_delta"] = max(deltas)
    checks["first_resumed_loss_matches"] = deltas[0] <= TOL.replayed_resume_loss_abs
    checks["resumed_losses_track_uninterrupted"] = max(deltas) <= TOL.replayed_resume_loss_abs


def _uninterrupted(
    ctx,
    family: Family,
    mode: str,
    lora: str,
    base_source: str,
    train_out: str,
    checks: dict,
    *,
    gradient_checkpointing: bool,
):
    log(f"\n[1/4] Uninterrupted {TOTAL_STEPS}-step run, checkpoint at step {SAVE_AT_STEP}...")
    trainer = _make_trainer(
        ctx, family, mode, lora, base_source, train_out, save=True, gradient_checkpointing=gradient_checkpointing
    )
    checks.update(_mode_checks(ctx, family, mode, lora, trainer))
    prefix = backbone_prefix(trainer.model)
    # Expert weights live in the EP layer's own layout (this rank's experts under EP), not the hub's.
    expert_weights = {
        f"{name}.{param_name}"[len(prefix) :]
        for name, layer in named_ep_layers(trainer.model)
        for param_name, _param in layer.named_parameters()
    }
    initial = snapshot_trainable(trainer.model) if gradient_checkpointing else {}
    at_save = _RestorePoint("save", trainer)
    trainer.add_callback(at_save)
    trainer.train()
    losses = step_losses(trainer)
    if mode in DP_MODES or mode in PARALLEL_MODES:
        # The module the FSDP2 wrap is rooted at.
        pipeline = trainer._top_level_model()
        checks |= dp_layout_checks(
            pipeline,
            _requested_fields(family, mode, lora),
            world_size=ctx.world_size,
            ep_config=trainer._ep_config,
            forward=lambda: probe_pipeline_forward(pipeline, ctx.device),
        )
    final = snapshot_trainable(trainer.model)
    if gradient_checkpointing:
        unmoved = sorted(name for name, tensor in initial.items() if torch.equal(tensor, final[name]))
        log(
            f"  {len(initial) - len(unmoved)}/{len(initial)} trainable tensors moved under checkpointing {unmoved[:2]}"
        )
        checks["every_trainable_tensor_moved_under_checkpointing"] = bool(initial) and not unmoved
    saved = at_save.captured or {}
    checks["uninterrupted_ran_all_steps"] = len(losses) == TOTAL_STEPS
    checks["losses_are_informative"] = bool(losses) and min_or_nan(losses) > MIN_INFORMATIVE_LOSS
    log(f"  uninterrupted losses {[f'{x:.6f}' for x in losses]}")
    finish_phase(trainer)
    return SimpleNamespace(
        losses=losses,
        at_save=saved.get("tensors", {}),
        partial_folds=saved.get("partial_folds", set()),
        final=final,
        prefix=prefix,
        expert_weights=expert_weights,
    )


def _full_finetune_row(
    ctx, family: Family, mode: str, base_source: str, shared_dir: str, *, gradient_checkpointing: bool
) -> dict:
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    train_out = os.path.join(shared_dir, "train_out")
    checkpoint = os.path.join(train_out, f"checkpoint-{SAVE_AT_STEP}")
    run = _uninterrupted(
        ctx, family, mode, "off", base_source, train_out, checks, gradient_checkpointing=gradient_checkpointing
    )

    log("\n[2/4] The checkpoint as a server loads it...")
    serving = {}
    if ctx.rank == 0:
        serving = _loads_and_encodes(checkpoint, ctx.device, family.pooling)
        saved = load_full_state_dict(checkpoint)
        live = {
            key[len(run.prefix) :]: value
            for key, value in run.at_save.items()
            if key[len(run.prefix) :] not in run.expert_weights
        }
        serving["checkpoint_holds_the_live_weights_at_the_save"] = (
            bool(live)
            and set(live) <= set(saved)
            and all(tensors_equal_at_narrower_dtype(saved[key], value) for key, value in live.items())
        )
        log(f"  {len(live)} live weights compared against the checkpoint ({len(run.expert_weights)} expert ones not)")
    checks.update(ctx.broadcast_checks(serving))
    ctx.barrier()

    log(f"\n[3/4] Resuming from {checkpoint}...")
    parallelism_config = _parallelism_config(family, mode, "off")
    source = resolve_resume_weights_source(
        checkpoint, SimpleNamespace(model_name_or_path=base_source), parallelism_config
    )
    rebuilds_from_checkpoint = mode in PARALLEL_MODES or bool(run.expert_weights)
    checks["policy_source_is_the_expected_one"] = source == (checkpoint if rebuilds_from_checkpoint else base_source)
    trainer = _make_trainer(
        ctx,
        family,
        mode,
        "off",
        source,
        os.path.join(shared_dir, "resume_out"),
        save=False,
        gradient_checkpointing=gradient_checkpointing,
    )
    restored = _RestorePoint("train_begin", trainer)
    trainer.add_callback(restored)
    trainer.train(resume_from_checkpoint=checkpoint)
    checks["weights_bit_equal_after_restore"] = _bit_equal(
        run.at_save, (restored.captured or {}).get("tensors", {}), "restored vs at save"
    )
    _loss_checks(run.losses, step_losses(trainer), checks, metrics)
    drift = relative_l2(snapshot_trainable(trainer.model), run.final)
    metrics["final_weight_relative_l2"] = drift
    checks["final_weights_match_uninterrupted"] = drift <= TOL.replayed_resume_weight_rtol
    log(f"  final weights, resumed vs uninterrupted: relative L2 {drift:.3e} (tol {TOL.replayed_resume_weight_rtol})")

    log(f"\n[4/4] Best-model load of {checkpoint} onto the resumed run...")
    trainer.state.best_model_checkpoint = checkpoint
    # A model wrapped for expert compute reloads base weights only at construction and TP+DP's stacked shards do not
    # invert, so the loader refuses both, as the startup gate refuses load_best_model_at_end.
    if run.expert_weights or mode == "tpdp":
        refusal = None
        try:
            trainer._load_best_model()
        except (ValueError, RuntimeError) as e:
            refusal = str(e)
        log(f"  best-model load: {'refused: ' + refusal.splitlines()[0][:120] if refusal else 'ACCEPTED'}")
        checks["best_model_load_is_refused"] = refusal is not None and "cannot reload" in refusal
    else:
        before = snapshot_trainable(trainer.model)
        checks["premise_resumed_run_trained_past_the_checkpoint"] = not _bit_equal(
            run.at_save, before, "live before the best-model load vs at save"
        )
        trainer._load_best_model()
        after = snapshot_trainable(trainer.model)
        checks["best_model_load_restores_the_checkpoint_bit_equal"] = _bit_equal(
            run.at_save, after, "after the best-model load vs at save"
        )
    finish_phase(trainer)
    return {"checks": checks, "metrics": metrics}


def run_embedding_lora_resume(
    ctx, family_name: str, mode: str, lora: str, *, head: bool, gradient_checkpointing: bool
) -> dict:
    """One family x mode x ``--lora`` row; returns the harness's ``checks`` and ``metrics``."""
    if gradient_checkpointing and (head or lora != "off"):
        raise ValueError("gradient_checkpointing rows are full fine-tunes (lora 'off', no head)")
    family = FAMILIES[family_name]
    log(
        f"\n{'=' * 70}\n  embedding resume: {family_name}, {mode}, lora {lora}{', head' if head else ''}"
        f"{', gradient checkpointing' if gradient_checkpointing else ''}, "
        f"world {ctx.world_size}\n{'=' * 70}"
    )
    if mode == "ddp":
        os.environ.update(ACCELERATE_LAUNCH_ENV)
    if PARALLEL_MODES.get(mode, {}).get("ep_size", 1) > 1:
        pin_deterministic_ep_dispatch()
    shared_dir = shared_output_dir(ctx)
    base_source = _base_source(ctx, family_name)
    if head:
        return _head_row(ctx, family, mode, lora, base_source, shared_dir)
    if lora == "off":
        return _full_finetune_row(
            ctx, family, mode, base_source, shared_dir, gradient_checkpointing=gradient_checkpointing
        )
    if _refuses_injected_lora(family, mode):
        return _refusal_row(ctx, family, mode, lora, base_source, shared_dir)

    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    train_out = os.path.join(shared_dir, "train_out")
    checkpoint = os.path.join(train_out, f"checkpoint-{SAVE_AT_STEP}")
    run = _uninterrupted(ctx, family, mode, lora, base_source, train_out, checks, gradient_checkpointing=False)

    log("\n[2/4] The checkpoint as a server loads it...")
    serving = (
        _serving_checks(
            checkpoint,
            _model_config(family, base_source, lora),
            run.at_save,
            run.prefix,
            family.embedding if family.embedding in _lora_targets(family, lora) else None,
            run.partial_folds,
            family.pooling,
            ctx.device,
        )
        if ctx.rank == 0
        else {}
    )
    checks.update(ctx.broadcast_checks(serving))
    ctx.barrier()

    log(f"\n[3/4] Resuming from {checkpoint}...")
    source = resolve_resume_weights_source(
        checkpoint, SimpleNamespace(model_name_or_path=base_source), ParallelismConfig()
    )
    checks["policy_source_is_the_base"] = source == base_source
    trainer = _make_trainer(
        ctx,
        family,
        mode,
        lora,
        source,
        os.path.join(shared_dir, "resume_out"),
        save=False,
        gradient_checkpointing=False,
    )
    restored = _RestorePoint("train_begin", trainer)
    trainer.add_callback(restored)
    trainer.train(resume_from_checkpoint=checkpoint)
    checks["adapters_bit_equal_after_restore"] = _bit_equal(
        run.at_save, (restored.captured or {}).get("tensors", {}), "restored vs at save"
    )
    _loss_checks(run.losses, step_losses(trainer), checks, metrics)
    drift = relative_l2(snapshot_trainable(trainer.model), run.final)
    metrics["final_adapter_relative_l2"] = drift
    checks["final_adapters_match_uninterrupted"] = drift <= TOL.replayed_resume_weight_rtol
    log(f"  final adapters, resumed vs uninterrupted: relative L2 {drift:.3e} (tol {TOL.replayed_resume_weight_rtol})")

    log("\n[4/4] The final export...")
    final_dir = os.path.join(shared_dir, "final")
    trainer.save_model(final_dir)
    finish_phase(trainer)
    export = {}
    if ctx.rank == 0:
        export = {
            f"final_export_{name}": ok
            for name, ok in _loads_and_encodes(final_dir, ctx.device, family.pooling).items()
        }
        export["final_export_carries_no_resume_state"] = resume_adapter_dir(final_dir) is None and not os.path.exists(
            os.path.join(final_dir, RESUME_ADAPTER_DIR)
        )
    checks.update(ctx.broadcast_checks(export))
    return {"checks": checks, "metrics": metrics}
