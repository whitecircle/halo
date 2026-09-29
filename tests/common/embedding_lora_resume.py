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
     merge of the live adapters at the save writes, bit for bit (a target whose FSDP2 delta the save
     contracted shard-locally, within a rounding allowance), and move every target's base; the
     resume adapter holds those live tensors bit for bit; the root holds no adapter file or config.
  3. Resume through the production resolver: the policy source is the base; after the restore
     (``on_train_begin``) every trainable tensor is BIT-EQUAL to the saved one; with nothing reset
     between the runs, the resumed losses and the final adapters match the uninterrupted run's
     (``TOL.replayed_resume_loss_abs`` / ``TOL.replayed_resume_weight_rtol``).
  4. The resumed run's final ``save_model`` export loads and encodes and carries no resume state.

``--lora off`` is a full fine-tune of the same backbone, whose saves write the backbone's names:

  1. Uninterrupted, as above, with every parameter trainable.
  2. The checkpoint loads with stock ``AutoModel`` / ``SentenceTransformer`` and holds the live weights
     at the save.
  3. Resume through the production resolver. The data-parallel shapes run ``use_grouped_gemm: false``,
     where the resolver keeps the base, so the loader must read the checkpoint into a model that does
     not hold it; TP and EP build from the checkpoint. Every parameter is BIT-EQUAL to the saved one
     after the restore, and the losses and final weights match as above, within a bound under EP,
     where each rank's optimizer state is held BIT-EQUAL to the saved one instead.
  4. The best-model load (``_load_best_model`` onto the resumed run, trained past the checkpoint)
     brings the checkpoint's weights back bit for bit. A model wrapped for expert compute (EP, or a
     MoE under TP at the default grouped GEMM) loads its base weights only at construction, and TP
     over FSDP2 cannot invert its stacked shards: there the load must be refused, not keep the last
     weights.

Modes (:data:`MODES`): ``single`` (one process), ``fsdp`` (torchrun: mixin FSDP2, DTensor params),
``ddp`` (what ``accelerate launch`` with a MULTI_GPU config runs: accelerate's DDP over plain
tensors), ``presharded`` (FSDP2 over per-rank dataset slices, batched by the toolkit's loader), ``tp``,
``tpdp`` (TP over FSDP2, four ranks) and ``ep`` (the script's TP / EP loader). Under LoRA, ``tp``,
``tpdp`` and ``ep`` only check that the trainer refuses the injected adapters at construction. With ``head`` a row
only builds the trainer over a pipeline carrying a projection head after the pooling, which the
sharded and parallel shapes must refuse and a single process or DDP accept.

The scripts stay separate because the manifest launches each at one world size and one tier.
"""

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
from trl import ModelConfig, get_peft_config

from scripts.training.embedding import build_sentence_transformer, inject_lora
from src.args.distributed_args import DistributedArguments
from src.checkpoint.format import (
    ADAPTER_CONFIG_FILE,
    ADAPTER_SAFETENSORS_FILE,
    RESUME_ADAPTER_DIR,
    has_adapter_weight_file,
    load_full_state_dict,
    resume_adapter_dir,
)
from src.configs.embedding_config import EmbeddingConfig
from src.distributed.expert_parallel.base_layer import find_ep_layers
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.embedding.trainer import EmbeddingTrainer
from src.training.environment import resolve_resume_weights_source
from src.training.script_runner import ScriptRuntime, apply_distributed_trainer_config
from tests.common.checkpoint_io import RestorePointSnapshot, loading_problems
from tests.common.distributed import shared_output_dir, world_all
from tests.common.models import PARAPHRASE_MINILM
from tests.common.peft_helpers import LORA_ALPHA, LORA_R, injected_lora_fold
from tests.common.tiny_models import (
    TINY_DENSE_FAMILIES,
    TINY_MOE_FAMILIES,
    TinyFamily,
    shared_tiny_family_checkpoint,
)
from tests.common.tolerances import TOL
from tests.common.utils import (
    finish_phase,
    log,
    min_or_nan,
    optimizer_state_matches,
    relative_l2,
    resumed_loss_deltas,
    snapshot_trainable,
    step_losses,
    tensors_equal_at_narrower_dtype,
)


@dataclass(frozen=True)
class Family:
    """One backbone family: the shared registry's random-init ``tiny`` model saved as a checkpoint, or
    (``tiny`` None) the hub sentence-transformers checkpoint ``hub``. ``targets`` are its attention
    projections, ``embedding`` its input embedding."""

    targets: tuple[str, ...]
    embedding: str
    pooling: str
    tiny: TinyFamily | None = None
    hub: str | None = None
    attn_implementation: str | None = None


# The dense encoder ST ships, and the decoder families of the shipped examples/embedding/ recipes.
FAMILIES = {
    "bert": Family(("query", "value"), "word_embeddings", "mean", hub=PARAPHRASE_MINILM),
    **{
        name: Family(("q_proj", "v_proj"), "embed_tokens", "lasttoken", tiny)
        for name, tiny in TINY_DENSE_FAMILIES.items()
    },
    # FA4, the EP / TP loader's Blackwell default, does not compile the tiny config's head_dim of 8.
    "gemma4": Family(
        ("q_proj", "v_proj"), "embed_tokens", "lasttoken", TINY_MOE_FAMILIES["gemma4_text"], attn_implementation="sdpa"
    ),
    "gpt_oss": Family(("q_proj", "v_proj"), "embed_tokens", "lasttoken", TINY_MOE_FAMILIES["gpt_oss"]),
}
# What ``--lora`` adapts, and how (``dora``); ``off`` is a full fine-tune.
LORA_TARGETS = ("attention", "mixed", "embedding", "dora", "off")
TRAIN_MODES = ("single", "fsdp", "ddp", "presharded")
# Refused under LoRA, trained under a full fine-tune. ``tpdp`` is TP over FSDP2 (four ranks: tp 2, dp 2).
PARALLEL_MODES = {"tp": {"tp_size": 2}, "tpdp": {"tp_size": 2}, "ep": {"ep_size": 2}}
MODES = TRAIN_MODES + tuple(PARALLEL_MODES)

SEED = 42
TOTAL_STEPS = 6
SAVE_AT_STEP = 3
BATCH_SIZE = 8
LEARNING_RATE = 1e-3
# Every row but EP's replays the uninterrupted run (``TOL.replayed_resume_*``). EP's DeepEP combine
# sums in no fixed order, so after the first resumed step (which reads only restored state) its losses
# sit within one bf16 step of an MNRL loss in [2, 4), and its final weights measured 1.1e-4 to 1.2e-3
# off, against 9.7e-3 to 5.3e-2 of full fine-tune movement over the three resumed steps.
EP_LOSS_TOL = 2**-6
EP_FINAL_WEIGHT_RTOL = 5e-3
# A fold whose delta DTensor contracts the sharded rank dim shard-locally (a ``Partial`` placement)
# measured off PEFT's single-device merge on 9.6-9.9% of the elements the delta moves in BERT's
# 30522-row embedding, by up to 1.9e-3 of its largest value; every other fold is bit for bit.
PARTIAL_FOLD_MAX_OFF_FRACTION = 0.25
PARTIAL_FOLD_MAX_STEP = 2**-7
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


def _base_source(ctx, family_name: str) -> str:
    """The family's base checkpoint: the hub one, or its seeded tiny one at the word tokenizer's vocab,
    which rank 0 saves for every rank."""
    family = FAMILIES[family_name]
    if family.hub is not None:
        return family.hub
    return shared_tiny_family_checkpoint(ctx, family.tiny, f"embedding_base_{family_name}", _word_tokenizer(), SEED)


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
    config = _config(output_dir, family, save=save)
    parallelism_config = _parallelism_config(mode, lora)
    return EmbeddingTrainer(
        model=_build(ctx, _model_config(family, source, lora), config, parallelism_config),
        args=config,
        train_dataset=_pairs(ctx, mode),
        parallelism_config=parallelism_config,
        dataset_presharded=mode == "presharded",
    )


def backbone_prefix(model: SentenceTransformer) -> str:
    """The prefix the SentenceTransformer's names put before its transformer backbone's own."""
    backbone = model[0].auto_model
    return next(name for name, module in model.named_modules() if module is backbone) + "."


def _partial_fold_targets(model) -> set[str]:
    """Backbone names of the LoRA targets whose delta is a ``Partial`` DTensor: FSDP2 shards the rank dim
    of both factors, and DTensor contracts it shard-locally where A is wider than B is tall, so the
    fold sums bf16 partials. Collective: every rank calls it."""
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
    """The restore-point snapshot with every trainable tensor whole (``tensors``) and, at the save, the
    LoRA targets whose delta the save contracted shard-locally (``partial_folds``). This rank's
    optimizer state is captured under EP only, where the restore is held to it bit for bit."""

    def __init__(self, event: str, trainer: EmbeddingTrainer, mode: str):
        super().__init__(event, trainer, capture_optimizer=mode == "ep")

    def extra(self) -> dict:
        extra = {"tensors": snapshot_trainable(self.trainer.model)}
        if self.event == "save":
            extra["partial_folds"] = _partial_fold_targets(self.trainer.model)
        return extra


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
    elif mode == "tpdp":
        checks["mode_is_tp_over_fsdp2"] = trainer.parallelism_config.is_tp_mode and trainer._fsdp_wrapped
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


def _loads_and_encodes(directory: str, device) -> dict[str, bool]:
    checks = {}
    _backbone, info = AutoModel.from_pretrained(directory, output_loading_info=True)
    problems = loading_problems(info)
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


def _serves(served: torch.Tensor, expected: torch.Tensor, base: torch.Tensor, partial: bool) -> bool:
    """Whether ``served`` is the fold PEFT's single-device merge gives, ``expected``: bit for bit, or, for
    a ``partial`` delta (:func:`_partial_fold_targets`), off by rounding on a minority of the elements
    either fold moves off ``base``.

    The shard-local bf16 partials of such a delta round differently from one full contraction wherever
    they cancel, so a per-element ulp bound does not hold, and a delta near half a step of the base
    moves it in one fold only. A missing, partial or misscaled delta instead changes nearly every
    moved element."""
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
    model_config: ModelConfig,
    at_save: dict[str, torch.Tensor],
    prefix: str,
    embedding: str | None,
    partial_folds: set[str],
    device,
) -> dict[str, bool]:
    """The checkpoint as a server loads it, and the resume state beside it; ``embedding`` names the
    input embedding when it is a target, ``partial_folds`` the targets whose delta the save contracted
    shard-locally. Rank-local reads only."""
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
    checks.update(_loads_and_encodes(checkpoint, device))
    base = AutoModel.from_pretrained(model_config.model_name_or_path, dtype=torch.bfloat16).state_dict()
    # PEFT's own in-place merge of the live tensors at the save (the resume adapter's own are under test
    # above), in bf16 on the run's device. TRL's LoRA builder, not the script's: its exclusion scan is
    # collective, and this runs on rank 0 alone.
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


def _head_row(ctx, family: Family, mode: str, lora: str, source: str, shared_dir: str) -> dict:
    """A projection head after the pooling, trainable under a full fine-tune and frozen beside LoRA:
    FSDP2, TP and EP save it rank-locally and resume the backbone alone, so the trainer must refuse it
    at construction; a single process and DDP save and resume the whole pipeline, and accept it."""
    config = _config(os.path.join(shared_dir, f"head_{mode}"), family, save=False)
    parallelism_config = _parallelism_config(mode, lora)
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


def _loss_checks(uninterrupted: list[float], resumed: list[float], tol: float, checks: dict, metrics: dict) -> None:
    """The first resumed loss reads only restored state, so it replays the uninterrupted run's; the
    later ones match within ``tol``."""
    deltas = resumed_loss_deltas(uninterrupted, resumed, save_step=SAVE_AT_STEP, total_steps=TOTAL_STEPS)
    checks["resumed_ran_remaining_steps"] = deltas is not None
    if deltas is None:
        return
    metrics["first_resumed_loss_delta"] = deltas[0]
    metrics["resumed_loss_max_delta"] = max(deltas)
    checks["first_resumed_loss_matches"] = deltas[0] <= TOL.replayed_resume_loss_abs
    checks["resumed_losses_track_uninterrupted"] = max(deltas) <= tol


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
    at_save = _RestorePoint("save", trainer, mode)
    trainer.add_callback(at_save)
    trainer.train()
    losses = step_losses(trainer)
    final = snapshot_trainable(trainer.model)
    saved = at_save.captured or {}
    checks["uninterrupted_ran_all_steps"] = len(losses) == TOTAL_STEPS
    checks["losses_are_informative"] = bool(losses) and min_or_nan(losses) > MIN_INFORMATIVE_LOSS
    log(f"  uninterrupted losses {[f'{x:.6f}' for x in losses]}")
    finish_phase(trainer)
    return SimpleNamespace(
        losses=losses,
        at_save=saved.get("tensors", {}),
        optimizer=saved.get("optimizer"),
        partial_folds=saved.get("partial_folds", set()),
        final=final,
        prefix=prefix,
        expert_weights=expert_weights,
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
        serving["checkpoint_holds_the_live_weights_at_the_save"] = (
            bool(live)
            and set(live) <= set(saved)
            and all(tensors_equal_at_narrower_dtype(saved[key], value) for key, value in live.items())
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
    restored = _RestorePoint("train_begin", trainer, mode)
    trainer.add_callback(restored)
    trainer.train(resume_from_checkpoint=checkpoint)
    at_restore = restored.captured or {}
    checks["weights_bit_equal_after_restore"] = _bit_equal(
        run.at_save, at_restore.get("tensors", {}), "restored vs at save"
    )
    ep = mode == "ep"
    if ep:
        # The losses and final weights compare within a bound under EP, so the optimizer restore is held
        # to bit-equality on its own, per rank (expert state is rank-local).
        matches, why = optimizer_state_matches(run.optimizer, at_restore.get("optimizer"))
        log(f"  optimizer state restored vs at save: {'bit-equal' if matches else why}")
        checks["optimizer_state_bit_equal_after_restore"] = world_all(matches, ctx.device)
    _loss_checks(
        run.losses, step_losses(trainer), EP_LOSS_TOL if ep else TOL.replayed_resume_loss_abs, checks, metrics
    )
    drift = relative_l2(snapshot_trainable(trainer.model), run.final)
    tol = EP_FINAL_WEIGHT_RTOL if ep else TOL.replayed_resume_weight_rtol
    metrics["final_weight_relative_l2"] = drift
    checks["final_weights_match_uninterrupted"] = drift <= tol
    log(f"  final weights, resumed vs uninterrupted: relative L2 {drift:.3e} (tol {tol})")

    log(f"\n[4/4] Best-model load of {checkpoint} onto the resumed run...")
    trainer.state.best_model_checkpoint = checkpoint
    # A model wrapped for expert compute (EP, or the default grouped GEMM on a MoE) reloads its base
    # weights only at construction, and TP+DP's stacked shards do not invert: the loader refuses both
    # rather than keep the last weights, as the startup gate refuses load_best_model_at_end for them.
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


def run_embedding_lora_resume(ctx, family_name: str, mode: str, lora: str, head: bool = False) -> dict:
    """One family x mode x ``--lora`` row, or with ``head`` the construction check of a projection head
    outside the backbone; returns the harness's ``checks`` and ``metrics``."""
    family = FAMILIES[family_name]
    log(
        f"\n{'=' * 70}\n  embedding resume: {family_name}, {mode}, lora {lora}{', head' if head else ''}, "
        f"world {ctx.world_size}\n{'=' * 70}"
    )
    if mode == "ddp":
        os.environ.update(ACCELERATE_LAUNCH_ENV)
    shared_dir = shared_output_dir(ctx)
    base_source = _base_source(ctx, family_name)
    if head:
        return _head_row(ctx, family, mode, lora, base_source, shared_dir)
    if lora == "off":
        return _full_finetune_row(ctx, family, mode, base_source, shared_dir)
    if mode in PARALLEL_MODES:
        return _refusal_row(ctx, family, mode, lora, base_source, shared_dir)

    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    train_out = os.path.join(shared_dir, "train_out")
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
            run.partial_folds,
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
    trainer = _make_trainer(ctx, family, mode, lora, source, os.path.join(shared_dir, "resume_out"), save=False)
    restored = _RestorePoint("train_begin", trainer, mode)
    trainer.add_callback(restored)
    trainer.train(resume_from_checkpoint=checkpoint)
    checks["adapters_bit_equal_after_restore"] = _bit_equal(
        run.at_save, (restored.captured or {}).get("tensors", {}), "restored vs at save"
    )
    # LoRA is refused under EP, so every LoRA row replays the uninterrupted run.
    _loss_checks(run.losses, step_losses(trainer), TOL.replayed_resume_loss_abs, checks, metrics)
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
        export = {f"final_export_{name}": ok for name, ok in _loads_and_encodes(final_dir, ctx.device).items()}
        export["final_export_carries_no_resume_state"] = resume_adapter_dir(final_dir) is None and not os.path.exists(
            os.path.join(final_dir, RESUME_ADAPTER_DIR)
        )
    checks.update(ctx.broadcast_checks(export))
    return {"checks": checks, "metrics": metrics}
