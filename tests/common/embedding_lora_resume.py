"""Two-phase resume of an injected-LoRA embedding run, shared by the embedding resume GPU scripts.

The embedding script injects LoRA in place, and every save folds it into the weights it writes. Each
training checkpoint also carries the unfolded trainable tensors (``resume_adapter/``) and the marker
the resume classifies on, so resume builds from the base, re-injects, and restores them. The model is
built and adapted by the script's own ``build_sentence_transformer`` + ``inject_lora``, in bf16, with
dropout off, over each backbone family the plain data-parallel path takes (:data:`FAMILIES`):

  1. Uninterrupted: ``TOTAL_STEPS`` steps, checkpoint at ``SAVE_AT_STEP``; the trainable tensors are
     gathered right after that save (``on_save``).
  2. The checkpoint serves: stock ``SentenceTransformer`` and ``AutoModel.from_pretrained`` load it with
     no missing or unexpected keys and it encodes; its LoRA targets hold ``base + scaling · B @ A`` of
     the resume adapter's tensors, which are the live ones at the save bit for bit; the root holds no
     adapter file or config.
  3. Resume through the production resolver: the policy source is the base; after the restore
     (``on_train_begin``) every trainable tensor is BIT-EQUAL to the saved one; the first resumed loss
     equals the uninterrupted one within ``FIRST_LOSS_TOL`` (its forward reads only restored state),
     later ones within ``LOSS_TOL`` and the final adapters within ``FINAL_ADAPTER_RTOL``.
  4. The resumed run's final ``save_model`` export loads and encodes and carries no resume state.

Modes (:data:`MODES`): ``single`` (one process), ``fsdp`` (torchrun: mixin FSDP2, DTensor adapters),
``ddp`` (what ``accelerate launch`` with a MULTI_GPU config runs: accelerate's DDP over plain
tensors), ``presharded`` (FSDP2 over per-rank dataset slices, batched by the toolkit's loader). The
refusal modes ``tp`` and ``ep`` build the model through the script's TP / EP loader and only check
that the trainer refuses the injected adapters at construction.

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
from trl import ModelConfig

import src.optimizers.adamw_bf16 as adamw_bf16_mod
from scripts.training.embedding import build_sentence_transformer, inject_lora
from src.args.distributed_args import DistributedArguments
from src.checkpoint.format import (
    ADAPTER_CONFIG_FILE,
    ADAPTER_SAFETENSORS_FILE,
    RESUME_ADAPTER_DIR,
    cast_to_save_dtype,
    resume_adapter_dir,
)
from src.configs.embedding_config import EmbeddingConfig
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
from tests.common.utils import cleanup_memory, log, step_losses


@dataclass(frozen=True)
class Family:
    """One backbone family: a random-init tiny ``config_cls(**tiny)`` saved as a plain transformers
    checkpoint, or (``config_cls`` None) the hub sentence-transformers checkpoint ``hub``."""

    targets: tuple[str, ...]
    pooling: str
    config_cls: type | None = None
    tiny: dict | None = None
    hub: str | None = None


# The dense encoder ST ships, and the decoder families of the shipped examples/embedding/ recipes.
FAMILIES = {
    "bert": Family(("query", "value"), "mean", hub=PARAPHRASE_MINILM),
    "qwen3": Family(("q_proj", "v_proj"), "lasttoken", Qwen3Config, TINY_QWEN3_CONFIG),
    "qwen3_5": Family(("q_proj", "v_proj"), "lasttoken", Qwen3_5TextConfig, TINY_QWEN35_CONFIG),
    "gemma4": Family(("q_proj", "v_proj"), "lasttoken", Gemma4TextConfig, TINY_GEMMA4_MOE_CONFIG),
    "gpt_oss": Family(("q_proj", "v_proj"), "lasttoken", GptOssConfig, TINY_GPTOSS_CONFIG),
}
TRAIN_MODES = ("single", "fsdp", "ddp", "presharded")
REFUSAL_MODES = {"tp": {"tp_size": 2}, "ep": {"ep_size": 2}}
MODES = TRAIN_MODES + tuple(REFUSAL_MODES)

LORA_R = 8
LORA_ALPHA = 16
SCALING = LORA_ALPHA / LORA_R
SEED = 42
TOTAL_STEPS = 6
SAVE_AT_STEP = 3
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
# The adapter checks separate an exact resume from fresh adapters over the fold; the loss bounds only
# catch a gross divergence, since a bf16 MNRL loss near 2 moves in 7.8e-3 steps and fresh adapters
# over the fold forward almost like the trained ones.
# The first resumed forward reads only restored state: measured 0.0 on every row (fresh: 0 to 7.8e-3).
FIRST_LOSS_TOL = 1e-4
# Later steps carry AdamWBF16's stochastic-rounding stream, which restarts on resume: measured up to
# 3.1e-2 (fresh adapters: 7.8e-3 to 0.34).
LOSS_TOL = 5e-2
# Measured <=1.14e-2 (the same SR noise; 0.0 under DDP's plain AdamW), against 1.38-1.44 for fresh
# adapters.
FINAL_ADAPTER_RTOL = 2e-2
# MNRL near zero would let the loss comparisons pass for any adapters.
MIN_INFORMATIVE_LOSS = 0.5
# What ``accelerate launch`` with a MULTI_GPU config exports; the trainer reads it to leave DDP to
# accelerate instead of FSDP2-wrapping the model itself.
ACCELERATE_LAUNCH_ENV = {"ACCELERATE_MIXED_PRECISION": "bf16"}
_SPECIAL_TOKENS = ("[PAD]", "[UNK]", "[EOS]")


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


def _build(ctx, family: Family, source: str, config: EmbeddingConfig, parallelism_config: ParallelismConfig):
    """The model exactly as the script builds and adapts it for this run shape."""
    model_config = ModelConfig(
        model_name_or_path=source,
        use_peft=True,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        lora_target_modules=list(family.targets),
    )
    dist_args = DistributedArguments()
    mode_suffix = parallelism_config.mode_string or "standard"
    runtime = ScriptRuntime(parallelism_config, mode_suffix, ctx.local_rank, None, source)
    model = build_sentence_transformer(runtime, config, model_config, dist_args)
    inject_lora(model, model_config, dist_args)
    apply_distributed_trainer_config(config, parallelism_config)
    return model


def _make_trainer(ctx, family: Family, mode: str, source: str, output_dir: str, *, save: bool) -> EmbeddingTrainer:
    # Both phases start from the same stochastic-rounding stream.
    adamw_bf16_mod._SR_RNG = random.Random(0xB165EED)
    config = _config(output_dir, family, save=save)
    parallelism_config = ParallelismConfig()
    return EmbeddingTrainer(
        model=_build(ctx, family, source, config, parallelism_config),
        args=config,
        train_dataset=_pairs(ctx, mode),
        parallelism_config=parallelism_config,
        dataset_presharded=mode == "presharded",
    )


def _trainable_snapshot(model) -> dict[str, torch.Tensor]:
    """Every trainable tensor, whole and on the host; FSDP2 DTensors are gathered. Collective."""
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


def _mode_checks(ctx, mode: str, trainer: EmbeddingTrainer) -> dict[str, bool]:
    """The row runs the data-parallel shape it names; otherwise every comparison below is vacuous."""
    adapters = [param for name, param in trainer.model.named_parameters() if ".lora_" in name]
    checks = {"model_carries_injected_adapters": bool(adapters)}
    if mode == "single":
        checks["mode_is_single_process"] = ctx.world_size == 1 and not trainer._fsdp_wrapped
    elif mode == "ddp":
        checks["mode_is_accelerate_ddp"] = trainer._accelerate_manages_ddp and not trainer._fsdp_wrapped
    else:
        checks["mode_is_fsdp2_with_dtensor_adapters"] = trainer._fsdp_wrapped and all(
            isinstance(param.data, DTensor) for param in adapters
        )
    return checks


def _backbone_prefix(model: SentenceTransformer) -> str:
    backbone = model[0].auto_model
    return next(name for name, module in model.named_modules() if module is backbone) + "."


def _expected_fold(base_source: str, adapters: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    """``base + scaling · B @ A`` per LoRA target, at the save dtype, keyed by the backbone's names."""
    base = AutoModel.from_pretrained(base_source, dtype=torch.bfloat16).state_dict()
    expected = {}
    for key, lora_a in adapters.items():
        if not key.endswith(".lora_A.default.weight"):
            continue
        module = key[len(prefix) : -len(".lora_A.default.weight")]
        lora_b = adapters[key.replace(".lora_A.", ".lora_B.")]
        weight = base[f"{module}.weight"]
        expected[f"{module}.weight"] = cast_to_save_dtype(
            (weight.float() + SCALING * (lora_b.float() @ lora_a.float())).to(weight.dtype)
        )
    return expected


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
    checkpoint: str, base_source: str, at_save: dict[str, torch.Tensor], prefix: str, device
) -> dict[str, bool]:
    """The checkpoint as a server loads it, and the resume state beside it. Rank-local reads only."""
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
    expected = _expected_fold(base_source, saved, prefix) if saved else {}
    served = AutoModel.from_pretrained(checkpoint, dtype=torch.bfloat16).state_dict()
    unfolded = sorted(
        key for key, value in expected.items() if key not in served or not torch.equal(served[key], value)
    )
    checks["lora_targets_serve_the_fold_of_the_resume_adapter"] = bool(expected) and not unfolded
    log(f"  {len(expected) - len(unfolded)}/{len(expected)} LoRA targets serve base + {SCALING:g}·B@A {unfolded[:2]}")
    return checks


def _relative_l2(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    if not a or set(a) != set(b):
        return math.inf
    num = sum(float((a[k].float() - b[k].float()).pow(2).sum()) for k in a)
    den = sum(float(a[k].float().pow(2).sum()) for k in a)
    return math.sqrt(num / den) if den else math.inf


def _refusal_row(ctx, family: Family, mode: str, source: str, shared_dir: str) -> dict:
    """TP / EP: the script's loader for that shape, then the trainer must refuse the injected adapters."""
    config = _config(os.path.join(shared_dir, f"refused_{mode}"), family, save=False)
    parallelism_config = ParallelismConfig(**REFUSAL_MODES[mode])
    model = _build(ctx, family, source, config, parallelism_config)
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


def run_embedding_lora_resume(ctx, family_name: str, mode: str) -> dict:
    """One family x mode row; returns the harness's ``checks`` and ``metrics``."""
    family = FAMILIES[family_name]
    log(f"\n{'=' * 70}\n  injected-LoRA embedding resume: {family_name}, {mode}, world {ctx.world_size}\n{'=' * 70}")
    if mode == "ddp":
        os.environ.update(ACCELERATE_LAUNCH_ENV)
    shared = [ctx.output_dir]
    if dist.is_initialized():
        dist.broadcast_object_list(shared, src=0)
    base_source = _base_source(ctx, family_name, shared[0])
    if mode in REFUSAL_MODES:
        return _refusal_row(ctx, family, mode, base_source, shared[0])

    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    train_out = os.path.join(shared[0], "train_out")
    checkpoint = os.path.join(train_out, f"checkpoint-{SAVE_AT_STEP}")

    log(f"\n[1/4] Uninterrupted {TOTAL_STEPS}-step run, checkpoint at step {SAVE_AT_STEP}...")
    trainer = _make_trainer(ctx, family, mode, base_source, train_out, save=True)
    checks.update(_mode_checks(ctx, mode, trainer))
    prefix = _backbone_prefix(trainer.model)
    at_save = _SaveCapture()
    trainer.add_callback(at_save)
    trainer.train()
    uninterrupted = step_losses(trainer)
    final_uninterrupted = _trainable_snapshot(trainer.model)
    checks["uninterrupted_ran_all_steps"] = len(uninterrupted) == TOTAL_STEPS
    checks["losses_are_informative"] = bool(uninterrupted) and min(uninterrupted) > MIN_INFORMATIVE_LOSS
    log(f"  uninterrupted losses {[f'{x:.6f}' for x in uninterrupted]}")
    del trainer
    cleanup_memory()
    ctx.barrier()

    log("\n[2/4] The checkpoint as a server loads it...")
    serving = _serving_checks(checkpoint, base_source, at_save.tensors, prefix, ctx.device) if ctx.rank == 0 else {}
    checks.update(ctx.broadcast_checks(serving))
    ctx.barrier()

    log(f"\n[3/4] Resuming from {checkpoint}...")
    source = resolve_resume_weights_source(
        checkpoint, SimpleNamespace(model_name_or_path=base_source), ParallelismConfig()
    )
    checks["policy_source_is_the_base"] = source == base_source
    trainer = _make_trainer(ctx, family, mode, source, os.path.join(shared[0], "resume_out"), save=False)
    restored = _RestoreCapture()
    trainer.add_callback(restored)
    trainer.train(resume_from_checkpoint=checkpoint)
    resumed = step_losses(trainer)
    final_resumed = _trainable_snapshot(trainer.model)

    saved = at_save.tensors
    unequal = sorted(
        key for key in saved if key not in restored.tensors or not torch.equal(saved[key], restored.tensors[key])
    )
    checks["adapters_bit_equal_after_restore"] = bool(saved) and set(saved) == set(restored.tensors) and not unequal
    log(f"  {len(saved) - len(unequal)}/{len(saved)} trainable tensors bit-equal after restore {unequal[:2]}")
    tail, reference = resumed[-(TOTAL_STEPS - SAVE_AT_STEP) :], uninterrupted[SAVE_AT_STEP:]
    checks["resumed_ran_remaining_steps"] = len(tail) == len(reference) == TOTAL_STEPS - SAVE_AT_STEP
    if checks["resumed_ran_remaining_steps"]:
        deltas = [
            abs(a - b) if math.isfinite(a) and math.isfinite(b) else math.inf
            for a, b in zip(tail, reference, strict=True)
        ]
        metrics["first_resumed_loss_delta"] = deltas[0]
        metrics["resumed_loss_max_delta"] = max(deltas)
        checks["first_resumed_loss_matches"] = deltas[0] < FIRST_LOSS_TOL
        checks["resumed_losses_track_uninterrupted"] = max(deltas) < LOSS_TOL
        log(
            f"  uninterrupted {[f'{x:.6f}' for x in reference]}  resumed {[f'{x:.6f}' for x in tail]}  "
            f"deltas {[f'{d:.2e}' for d in deltas]}"
        )
    drift = _relative_l2(final_uninterrupted, final_resumed)
    metrics["final_adapter_relative_l2"] = drift
    checks["final_adapters_match_uninterrupted"] = drift < FINAL_ADAPTER_RTOL
    log(f"  final adapters, resumed vs uninterrupted: relative L2 {drift:.3e} (tol {FINAL_ADAPTER_RTOL})")

    log("\n[4/4] The final export...")
    final_dir = os.path.join(shared[0], "final")
    trainer.save_model(final_dir)
    del trainer
    cleanup_memory()
    ctx.barrier()
    export = {}
    if ctx.rank == 0:
        export = {f"final_export_{name}": ok for name, ok in _loads_and_encodes(final_dir, ctx.device).items()}
        export["final_export_carries_no_resume_state"] = resume_adapter_dir(final_dir) is None and not os.path.exists(
            os.path.join(final_dir, RESUME_ADAPTER_DIR)
        )
    checks.update(ctx.broadcast_checks(export))
    return {"checks": checks, "metrics": metrics}
