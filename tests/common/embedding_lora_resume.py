"""Two-phase resume of an embedding run, shared by the embedding resume GPU scripts.

The embedding script injects LoRA in place, and every save folds it into the weights it writes. Each
training checkpoint also carries the unfolded trainable tensors (``resume_adapter/``) and the marker
the resume classifies on, so resume builds from the base, re-injects, and restores them. The model is
built and adapted by the script's own ``build_sentence_transformer`` + ``inject_lora``, in bf16, with
dropout off, over each backbone family the plain data-parallel path takes (:data:`FAMILIES`), with
the adapters on the attention projections, on the input embedding beside them, on the input
embedding alone, or DoRA on the attention projections (:data:`LORA_TARGETS`):

  1. Uninterrupted: ``TOTAL_STEPS`` steps, checkpoint at ``SAVE_AT_STEP``; the trainable tensors are
     gathered right after that save (``on_save``).
  2. The checkpoint serves: stock ``SentenceTransformer`` and ``AutoModel.from_pretrained`` load it with
     no missing or unexpected keys and it encodes; its LoRA targets hold what PEFT's own in-place
     merge of the live adapters at the save writes, bit for bit, and move every target's base; the
     resume adapter holds those live tensors bit for bit; the root holds no adapter file or config.
  3. Resume through the production resolver: the policy source is the base; after the restore
     (``on_train_begin``) every trainable tensor is BIT-EQUAL to the saved one; the first resumed loss
     equals the uninterrupted one within ``FIRST_LOSS_TOL`` (its forward reads only restored state),
     later ones within ``LOSS_TOL`` and the final adapters within ``FINAL_ADAPTER_RTOL``.
  4. The resumed run's final ``save_model`` export loads and encodes and carries no resume state.

``--lora off`` is a full fine-tune of the same backbone, whose saves write the backbone's names:

  1. Uninterrupted, as above, with every parameter trainable.
  2. The checkpoint loads with stock ``AutoModel`` / ``SentenceTransformer`` and holds the live weights
     at the save.
  3. Resume through the production resolver. The data-parallel shapes run ``use_grouped_gemm: false``,
     where the resolver keeps the base, so the loader must read the checkpoint into a model that does
     not hold it; TP and EP build from the checkpoint. Every parameter is BIT-EQUAL to the saved one
     after the restore, and the losses and final weights track as above.
  4. The best-model load (``_load_best_model`` onto the resumed run, trained past the checkpoint)
     brings the checkpoint's weights back bit for bit. Skipped on a model wrapped for expert compute
     (EP, or a MoE under TP at the default grouped GEMM), whose base weights load only at
     construction and whose ``load_best_model_at_end`` the startup gate refuses.

Modes (:data:`MODES`): ``single`` (one process), ``fsdp`` (torchrun: mixin FSDP2, DTensor params),
``ddp`` (what ``accelerate launch`` with a MULTI_GPU config runs: accelerate's DDP over plain
tensors), ``presharded`` (FSDP2 over per-rank dataset slices, batched by the toolkit's loader), ``tp``
and ``ep`` (the script's TP / EP loader). Under LoRA, ``tp`` and ``ep`` only check that the trainer
refuses the injected adapters at construction.

The scripts stay separate because the manifest launches each at one world size and one tier.
"""

import math
import os
import random
from dataclasses import dataclass
from types import SimpleNamespace

import torch
import torch.distributed as dist
from accelerate.utils import extract_model_from_parallel
from datasets import Dataset
from safetensors.torch import load_file
from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer, models, pre_tokenizers
from torch.distributed.tensor import DTensor
from transformers import (
    AutoModel,
    Gemma4TextConfig,
    GptOssConfig,
    PreTrainedTokenizerFast,
    Qwen3_5TextConfig,
    Qwen3Config,
    TrainerCallback,
)
from trl import ModelConfig, get_peft_config

import src.optimizers.adamw_bf16 as adamw_bf16_mod
from scripts.training.embedding import build_sentence_transformer, inject_lora
from src.args.distributed_args import DistributedArguments
from src.checkpoint.format import (
    ADAPTER_CONFIG_FILE,
    ADAPTER_SAFETENSORS_FILE,
    RESUME_ADAPTER_DIR,
    cast_to_save_dtype,
    load_full_state_dict,
    resume_adapter_dir,
)
from src.configs.embedding_config import EmbeddingConfig
from src.distributed.expert_parallel.base_layer import find_ep_layers
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.embedding.trainer import EmbeddingTrainer
from src.training.environment import resolve_resume_weights_source
from src.training.script_runner import ScriptRuntime, apply_distributed_trainer_config
from tests.common.models import (
    PARAPHRASE_MINILM,
    TINY_GEMMA4_MOE_CONFIG,
    TINY_GPTOSS_CONFIG,
    TINY_QWEN3_CONFIG,
    TINY_QWEN35_CONFIG,
)
from tests.common.peft_helpers import injected_lora_merge
from tests.common.utils import cleanup_memory, log, step_losses, tensors_equal_at_narrower_dtype


@dataclass(frozen=True)
class Family:
    """One backbone family: a random-init tiny ``config_cls(**tiny)`` saved as a plain transformers
    checkpoint, or (``config_cls`` None) the hub sentence-transformers checkpoint ``hub``. ``targets``
    are its attention projections, ``embedding`` its input embedding."""

    targets: tuple[str, ...]
    embedding: str
    pooling: str
    config_cls: type | None = None
    tiny: dict | None = None
    hub: str | None = None
    attn_implementation: str | None = None


# The dense encoder ST ships, and the decoder families of the shipped examples/embedding/ recipes.
FAMILIES = {
    "bert": Family(("query", "value"), "word_embeddings", "mean", hub=PARAPHRASE_MINILM),
    "qwen3": Family(("q_proj", "v_proj"), "embed_tokens", "lasttoken", Qwen3Config, TINY_QWEN3_CONFIG),
    "qwen3_5": Family(("q_proj", "v_proj"), "embed_tokens", "lasttoken", Qwen3_5TextConfig, TINY_QWEN35_CONFIG),
    # FA4, the EP / TP loader's Blackwell default, does not compile the tiny config's head_dim of 8.
    "gemma4": Family(
        ("q_proj", "v_proj"), "embed_tokens", "lasttoken", Gemma4TextConfig, TINY_GEMMA4_MOE_CONFIG, None, "sdpa"
    ),
    "gpt_oss": Family(("q_proj", "v_proj"), "embed_tokens", "lasttoken", GptOssConfig, TINY_GPTOSS_CONFIG),
}
# What ``--lora`` adapts, and how (``dora``); ``off`` is a full fine-tune.
LORA_TARGETS = ("attention", "mixed", "embedding", "dora", "off")
TRAIN_MODES = ("single", "fsdp", "ddp", "presharded")
# Refused under LoRA, trained under a full fine-tune.
PARALLEL_MODES = {"tp": {"tp_size": 2}, "ep": {"ep_size": 2}}
MODES = TRAIN_MODES + tuple(PARALLEL_MODES)

LORA_R = 8
LORA_ALPHA = 16
SEED = 42
TOTAL_STEPS = 6
SAVE_AT_STEP = 3
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
# The bit-equal restores, the first resumed loss and the final adapters separate an exact resume from
# fresh adapters over the fold; the later losses and the full fine-tune's final weights only catch a
# gross divergence, since a bf16 MNRL loss near 2 moves in 7.8e-3 steps and fresh adapters over the
# fold forward almost like the trained ones.
# The first resumed forward reads only restored state: measured 0.0 on every row (fresh: 0 to 7.8e-3).
FIRST_LOSS_TOL = 1e-4
# Later steps carry AdamWBF16's stochastic-rounding stream, which restarts on resume: measured up to
# 4.7e-2, and 0.20 on Gemma 4 with its input embedding adapted, whose embed_scale multiplies that
# noise in the adapter (fresh adapters: 7.8e-3 to 0.34).
LOSS_TOL = 0.25
# Measured <=1.49e-2 (the same SR noise; 0.0 under DDP's plain AdamW), against 1.38-1.44 for fresh
# adapters.
FINAL_ADAPTER_RTOL = 2e-2
# Measured <=1.24e-2 (0.0 under DDP), against 1.0e-2 to 5.3e-2 of movement over the three resumed steps.
FINAL_WEIGHT_RTOL = 2e-2
# MNRL near zero would let the loss comparisons pass for any adapters.
MIN_INFORMATIVE_LOSS = 0.5
# What ``accelerate launch`` with a MULTI_GPU config exports; the trainer reads it to leave DDP to
# accelerate instead of FSDP2-wrapping the model itself.
ACCELERATE_LAUNCH_ENV = {"ACCELERATE_MIXED_PRECISION": "bf16"}
_SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[EOS]")


def _lora_targets(family: Family, lora: str) -> tuple[str, ...] | None:
    """The ``lora_target_modules`` of a ``--lora`` choice; None for a full fine-tune. TRL collapses a
    one-entry list into a string, which PEFT reads as a regex, so embedding-only is spelled as a user
    writes it: the embedding beside an ``lm_head`` the headless backbone lacks."""
    return {
        "attention": family.targets,
        "mixed": (family.embedding, *family.targets),
        "embedding": (family.embedding, "lm_head"),
        "dora": family.targets,
        "off": None,
    }[lora]


def _model_config(family: Family, source: str, lora: str) -> ModelConfig:
    """The script's model arguments for this row: the source, and the LoRA the ``--lora`` choice sets."""
    targets = _lora_targets(family, lora)
    return ModelConfig(
        model_name_or_path=source,
        attn_implementation=family.attn_implementation,
        use_peft=targets is not None,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        lora_target_modules=list(targets) if targets is not None else None,
        use_dora=lora == "dora",
    )


def _parallelism_config(mode: str, lora: str) -> ParallelismConfig:
    if mode in PARALLEL_MODES:
        return ParallelismConfig(**PARALLEL_MODES[mode])
    # A full fine-tune without expert wrappers keeps the resolver on the base, so its resume reads the
    # checkpoint into a model that does not hold it.
    return ParallelismConfig(use_grouped_gemm=lora != "off")


def _pair_texts() -> tuple[list[str], list[str]]:
    """Pairs that differ only by an index, paired through a fixed permutation, so the in-batch
    negatives are as close as the positive and MNRL stays O(1) for the whole run."""
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


def _base_source(ctx, family_name: str, shared_dir: str) -> str:
    """The family's base checkpoint: the hub one, or a seeded tiny one rank 0 saves for every rank."""
    family = FAMILIES[family_name]
    if family.hub is not None:
        return family.hub
    base_dir = os.path.join(shared_dir, f"tiny_{family_name}")
    if ctx.rank == 0:
        tokenizer = _word_tokenizer()
        sizes = {"vocab_size": len(tokenizer)}
        if "vocab_size_per_layer_input" in family.tiny:
            sizes["vocab_size_per_layer_input"] = len(tokenizer)
        config = family.config_cls(
            **{**family.tiny, **sizes}, pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id
        )
        torch.manual_seed(SEED)
        AutoModel.from_config(config).to(torch.bfloat16).save_pretrained(base_dir)
        tokenizer.save_pretrained(base_dir)
    ctx.barrier()
    return base_dir


def _config(output_dir: str, family: Family, *, save: bool) -> EmbeddingConfig:
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
        seed=SEED,
    )


def _build(ctx, model_config: ModelConfig, config: EmbeddingConfig, parallelism_config: ParallelismConfig):
    """The model exactly as the script builds and adapts it for this run shape."""
    source = model_config.model_name_or_path
    dist_args = DistributedArguments()
    mode_suffix = parallelism_config.mode_string or "standard"
    runtime = ScriptRuntime(parallelism_config, mode_suffix, ctx.local_rank, None, source)
    model = build_sentence_transformer(runtime, config, model_config, dist_args)
    inject_lora(model, model_config, dist_args)
    apply_distributed_trainer_config(config, parallelism_config)
    return model


def _make_trainer(ctx, family: Family, mode: str, lora: str, source: str, output_dir: str, *, save: bool):
    # Both phases start from the same stochastic-rounding stream.
    adamw_bf16_mod._SR_RNG = random.Random(0xB165EED)
    config = _config(output_dir, family, save=save)
    parallelism_config = _parallelism_config(mode, lora)
    return EmbeddingTrainer(
        model=_build(ctx, _model_config(family, source, lora), config, parallelism_config),
        args=config,
        train_dataset=_pairs(ctx, mode),
        parallelism_config=parallelism_config,
        dataset_presharded=mode == "presharded",
    )


def _release(ctx, trainer: EmbeddingTrainer) -> None:
    """Free a finished phase's trainer before the next one builds; DeepEP buffers are destroyed collectively."""
    if trainer._has_ep_layers:
        trainer.cleanup_ep()
    del trainer
    cleanup_memory()
    ctx.barrier()


def _trainable_snapshot(model) -> dict[str, torch.Tensor]:
    """Every trainable tensor, whole and on the host; FSDP2 / TP DTensors are gathered. Collective."""
    snapshot = {}
    for name, param in extract_model_from_parallel(model, recursive=True).named_parameters():
        if param.requires_grad:
            data = param.data.full_tensor() if isinstance(param.data, DTensor) else param.data
            snapshot[name] = data.detach().cpu().clone()
    return snapshot


class _SaveCapture(TrainerCallback):
    def __init__(self):
        self.tensors: dict[str, torch.Tensor] = {}

    def on_save(self, args, state, control, model=None, **kwargs):
        if state.global_step == SAVE_AT_STEP:
            self.tensors = _trainable_snapshot(model)


class _RestoreCapture(TrainerCallback):
    def __init__(self):
        self.tensors: dict[str, torch.Tensor] = {}

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.tensors = _trainable_snapshot(model)


def _mode_checks(ctx, mode: str, lora: str, trainer: EmbeddingTrainer) -> dict[str, bool]:
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
    elif mode == "tp":
        checks["mode_is_pure_tp_with_dtensor_params"] = (
            trainer.parallelism_config.is_tp_mode
            and not trainer._fsdp_wrapped
            and any(isinstance(param.data, DTensor) for param in trainable)
        )
    elif mode == "ep":
        checks["mode_is_ep"] = trainer.parallelism_config.is_ep_mode and trainer._has_ep_layers
    else:
        checks["mode_is_fsdp2_with_dtensor_params"] = trainer._fsdp_wrapped and all(
            isinstance(param.data, DTensor) for param in trainable
        )
    return checks


def backbone_prefix(model: SentenceTransformer) -> str:
    backbone = model[0].auto_model
    return next(name for name, module in model.named_modules() if module is backbone) + "."


def _expected_fold(
    model_config: ModelConfig, adapters: dict[str, torch.Tensor], prefix: str, device
) -> dict[str, torch.Tensor]:
    """What PEFT's own in-place merge writes per LoRA target, keyed by the backbone's names: the base in
    bf16 on the run's device, the script's LoRA injected, ``adapters`` (keyed by the ST's names) loaded
    and every LoRA layer merged."""
    model = AutoModel.from_pretrained(model_config.model_name_or_path, dtype=torch.bfloat16).to(device)
    # TRL's builder, not the script's: its exclusion scan is collective, and this runs on rank 0 alone.
    merged = injected_lora_merge(
        model, get_peft_config(model_config), {key[len(prefix) :]: value.to(device) for key, value in adapters.items()}
    )
    return {key: cast_to_save_dtype(value.cpu()) for key, value in merged.items() if key.endswith(".weight")}


def _loads_and_encodes(directory: str, device) -> dict[str, bool]:
    checks = {}
    _backbone, info = AutoModel.from_pretrained(directory, output_loading_info=True)
    problems = {kind: info[kind] for kind in ("missing_keys", "unexpected_keys", "mismatched_keys") if info[kind]}
    if problems:
        log(f"  from_pretrained loading info for {directory}: {problems}")
    checks["loads_with_stock_from_pretrained"] = not problems
    embeddings = SentenceTransformer(directory, device=str(device)).encode(
        ["What is the capital of country 3?", "It is a large city located in country 7."], convert_to_tensor=True
    )
    checks["loads_and_encodes_with_stock_sentence_transformer"] = bool(
        embeddings.shape[0] == 2 and torch.isfinite(embeddings).all()
    )
    return checks


def _serving_checks(
    checkpoint: str,
    model_config: ModelConfig,
    at_save: dict[str, torch.Tensor],
    prefix: str,
    embedding: str | None,
    device,
) -> dict[str, bool]:
    """The checkpoint as a server loads it, and the resume state beside it; ``embedding`` names the
    input embedding when it is a target. Rank-local reads only."""
    checks = {"checkpoint_is_marked_for_adapter_resume": resume_adapter_dir(checkpoint) is not None}
    adapter_file = os.path.join(checkpoint, RESUME_ADAPTER_DIR, ADAPTER_SAFETENSORS_FILE)
    saved = load_file(adapter_file) if os.path.isfile(adapter_file) else {}
    checks["resume_adapter_holds_the_live_tensors_bit_for_bit"] = (
        bool(at_save)
        and set(saved) == set(at_save)
        and all(torch.equal(saved[key], value) for key, value in at_save.items())
    )
    checks["no_adapter_file_or_config_at_the_root"] = not any(
        os.path.exists(os.path.join(checkpoint, name)) for name in (ADAPTER_CONFIG_FILE, ADAPTER_SAFETENSORS_FILE)
    )
    checks.update(_loads_and_encodes(checkpoint, device))
    base = AutoModel.from_pretrained(model_config.model_name_or_path, dtype=torch.bfloat16).state_dict()
    # From the live tensors at the save: the resume adapter's own are under test above.
    expected = _expected_fold(model_config, at_save, prefix, device)
    served = AutoModel.from_pretrained(checkpoint, dtype=torch.bfloat16).state_dict()
    unfolded = sorted(
        key for key, value in expected.items() if key not in served or not torch.equal(served[key], value)
    )
    unmoved = sorted(key for key, value in expected.items() if torch.equal(value, base[key]))
    embedding_folds = [key for key in expected if key.endswith(f".{embedding}.weight") or key == f"{embedding}.weight"]
    checks["lora_targets_serve_the_fold_of_the_resume_adapter"] = bool(expected) and not unfolded
    checks["every_fold_moves_its_base"] = bool(expected) and not unmoved
    if embedding is not None:
        checks["the_input_embedding_is_folded"] = len(embedding_folds) == 1
    log(
        f"  {len(expected) - len(unfolded)}/{len(expected)} LoRA targets serve PEFT's merge "
        f"(embedding folds {embedding_folds}) {unfolded[:2]}; unmoved {unmoved[:2]}"
    )
    return checks


def _relative_l2(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    if not a or set(a) != set(b):
        return math.inf
    num = sum(float((a[k].float() - b[k].float()).pow(2).sum()) for k in a)
    den = sum(float(a[k].float().pow(2).sum()) for k in a)
    return math.sqrt(num / den) if den else math.inf


def _bit_equal(expected: dict[str, torch.Tensor], actual: dict[str, torch.Tensor], what: str) -> bool:
    unequal = sorted(key for key in expected if key not in actual or not torch.equal(expected[key], actual[key]))
    log(f"  {what}: {len(expected) - len(unequal)}/{len(expected)} tensors bit-equal {unequal[:2]}")
    return bool(expected) and set(expected) == set(actual) and not unequal


def _refusal_row(ctx, family: Family, mode: str, lora: str, source: str, shared_dir: str) -> dict:
    """LoRA under TP / EP: the script's loader for that shape, then the trainer must refuse the adapters."""
    config = _config(os.path.join(shared_dir, f"refused_{mode}"), family, save=False)
    parallelism_config = _parallelism_config(mode, lora)
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
    return {"checks": {f"{mode}_refuses_injected_lora_at_construction": error is not None and "LoRA" in error}}


def _loss_checks(uninterrupted: list[float], resumed: list[float], checks: dict, metrics: dict) -> None:
    tail, reference = resumed[-(TOTAL_STEPS - SAVE_AT_STEP) :], uninterrupted[SAVE_AT_STEP:]
    checks["resumed_ran_remaining_steps"] = len(tail) == len(reference) == TOTAL_STEPS - SAVE_AT_STEP
    if not checks["resumed_ran_remaining_steps"]:
        return
    deltas = [
        abs(a - b) if math.isfinite(a) and math.isfinite(b) else math.inf for a, b in zip(tail, reference, strict=True)
    ]
    metrics["first_resumed_loss_delta"] = deltas[0]
    metrics["resumed_loss_max_delta"] = max(deltas)
    checks["first_resumed_loss_matches"] = deltas[0] < FIRST_LOSS_TOL
    checks["resumed_losses_track_uninterrupted"] = max(deltas) < LOSS_TOL
    log(
        f"  uninterrupted {[f'{x:.6f}' for x in reference]}  resumed {[f'{x:.6f}' for x in tail]}  "
        f"deltas {[f'{d:.2e}' for d in deltas]}"
    )


def _uninterrupted(ctx, family: Family, mode: str, lora: str, base_source: str, train_out: str, checks: dict):
    """Phase 1, shared by both flows: the run, its losses and trainable tensors at the save and at the end."""
    log(f"\n[1/4] Uninterrupted {TOTAL_STEPS}-step run, checkpoint at step {SAVE_AT_STEP}...")
    trainer = _make_trainer(ctx, family, mode, lora, base_source, train_out, save=True)
    checks.update(_mode_checks(ctx, mode, lora, trainer))
    prefix = backbone_prefix(trainer.model)
    # Expert weights live in the EP layer's own layout (this rank's experts under EP), not the hub's.
    expert_weights = {
        f"{name}.{param_name}"[len(prefix) :]
        for name, layer in find_ep_layers(trainer.model)
        for param_name, _param in layer.named_parameters()
    }
    at_save = _SaveCapture()
    trainer.add_callback(at_save)
    trainer.train()
    losses = step_losses(trainer)
    final = _trainable_snapshot(trainer.model)
    checks["uninterrupted_ran_all_steps"] = len(losses) == TOTAL_STEPS
    checks["losses_are_informative"] = bool(losses) and min(losses) > MIN_INFORMATIVE_LOSS
    log(f"  uninterrupted losses {[f'{x:.6f}' for x in losses]}")
    _release(ctx, trainer)
    return SimpleNamespace(
        losses=losses, at_save=at_save.tensors, final=final, prefix=prefix, expert_weights=expert_weights
    )


def _full_finetune_row(ctx, family: Family, mode: str, base_source: str, shared_dir: str) -> dict:
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    train_out = os.path.join(shared_dir, "train_out")
    checkpoint = os.path.join(train_out, f"checkpoint-{SAVE_AT_STEP}")
    run = _uninterrupted(ctx, family, mode, "off", base_source, train_out, checks)

    log("\n[2/4] The checkpoint as a server loads it...")
    serving = {}
    if ctx.rank == 0:
        serving = _loads_and_encodes(checkpoint, ctx.device)
        saved = load_full_state_dict(checkpoint)
        live = {
            key[len(run.prefix) :]: value
            for key, value in run.at_save.items()
            if key[len(run.prefix) :] not in run.expert_weights
        }
        serving["checkpoint_holds_the_live_weights_at_the_save"] = set(live) <= set(saved) and all(
            tensors_equal_at_narrower_dtype(saved[key], value) for key, value in live.items()
        )
        log(f"  {len(live)} live weights compared against the checkpoint ({len(run.expert_weights)} expert ones not)")
    checks.update(ctx.broadcast_checks(serving))
    ctx.barrier()

    log(f"\n[3/4] Resuming from {checkpoint}...")
    parallelism_config = _parallelism_config(mode, "off")
    source = resolve_resume_weights_source(
        checkpoint, SimpleNamespace(model_name_or_path=base_source), parallelism_config
    )
    rebuilds_from_checkpoint = mode in PARALLEL_MODES
    checks["policy_source_is_the_expected_one"] = source == (checkpoint if rebuilds_from_checkpoint else base_source)
    trainer = _make_trainer(ctx, family, mode, "off", source, os.path.join(shared_dir, "resume_out"), save=False)
    restored = _RestoreCapture()
    trainer.add_callback(restored)
    trainer.train(resume_from_checkpoint=checkpoint)
    checks["weights_bit_equal_after_restore"] = _bit_equal(run.at_save, restored.tensors, "restored vs at save")
    _loss_checks(run.losses, step_losses(trainer), checks, metrics)
    drift = _relative_l2(run.final, _trainable_snapshot(trainer.model))
    metrics["final_weight_relative_l2"] = drift
    checks["final_weights_match_uninterrupted"] = drift < FINAL_WEIGHT_RTOL
    log(f"  final weights, resumed vs uninterrupted: relative L2 {drift:.3e} (tol {FINAL_WEIGHT_RTOL})")

    # A model wrapped for expert compute (EP, or the default grouped GEMM on a MoE) reloads its base
    # weights only at construction, and the startup gate refuses load_best_model_at_end for it.
    if not run.expert_weights:
        log(f"\n[4/4] Best-model load of {checkpoint} onto the resumed run...")
        before = _trainable_snapshot(trainer.model)
        checks["premise_resumed_run_trained_past_the_checkpoint"] = not _bit_equal(
            run.at_save, before, "live before the best-model load vs at save"
        )
        trainer.state.best_model_checkpoint = checkpoint
        trainer._load_best_model()
        after = _trainable_snapshot(trainer.model)
        checks["best_model_load_restores_the_checkpoint_bit_equal"] = _bit_equal(
            run.at_save, after, "after the best-model load vs at save"
        )
    _release(ctx, trainer)
    return {"checks": checks, "metrics": metrics}


def run_embedding_lora_resume(ctx, family_name: str, mode: str, lora: str = "attention") -> dict:
    """One family x mode x ``--lora`` row; returns the harness's ``checks`` and ``metrics``."""
    family = FAMILIES[family_name]
    log(f"\n{'=' * 70}\n  embedding resume: {family_name}, {mode}, lora {lora}, world {ctx.world_size}\n{'=' * 70}")
    if mode == "ddp":
        os.environ.update(ACCELERATE_LAUNCH_ENV)
    shared = [ctx.output_dir]
    if dist.is_initialized():
        dist.broadcast_object_list(shared, src=0)
    base_source = _base_source(ctx, family_name, shared[0])
    if lora == "off":
        return _full_finetune_row(ctx, family, mode, base_source, shared[0])
    if mode in PARALLEL_MODES:
        return _refusal_row(ctx, family, mode, lora, base_source, shared[0])

    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    train_out = os.path.join(shared[0], "train_out")
    checkpoint = os.path.join(train_out, f"checkpoint-{SAVE_AT_STEP}")
    run = _uninterrupted(ctx, family, mode, lora, base_source, train_out, checks)

    log("\n[2/4] The checkpoint as a server loads it...")
    serving = (
        _serving_checks(
            checkpoint,
            _model_config(family, base_source, lora),
            run.at_save,
            run.prefix,
            family.embedding if family.embedding in _lora_targets(family, lora) else None,
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
    trainer = _make_trainer(ctx, family, mode, lora, source, os.path.join(shared[0], "resume_out"), save=False)
    restored = _RestoreCapture()
    trainer.add_callback(restored)
    trainer.train(resume_from_checkpoint=checkpoint)
    checks["adapters_bit_equal_after_restore"] = _bit_equal(run.at_save, restored.tensors, "restored vs at save")
    _loss_checks(run.losses, step_losses(trainer), checks, metrics)
    drift = _relative_l2(run.final, _trainable_snapshot(trainer.model))
    metrics["final_adapter_relative_l2"] = drift
    checks["final_adapters_match_uninterrupted"] = drift < FINAL_ADAPTER_RTOL
    log(f"  final adapters, resumed vs uninterrupted: relative L2 {drift:.3e} (tol {FINAL_ADAPTER_RTOL})")

    log("\n[4/4] The final export...")
    final_dir = os.path.join(shared[0], "final")
    trainer.save_model(final_dir)
    _release(ctx, trainer)
    export = {}
    if ctx.rank == 0:
        export = {f"final_export_{name}": ok for name, ok in _loads_and_encodes(final_dir, ctx.device).items()}
        export["final_export_carries_no_resume_state"] = resume_adapter_dir(final_dir) is None and not os.path.exists(
            os.path.join(final_dir, RESUME_ADAPTER_DIR)
        )
    checks.update(ctx.broadcast_checks(export))
    return {"checks": checks, "metrics": metrics}
