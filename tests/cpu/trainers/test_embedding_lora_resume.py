#!/usr/bin/env python
"""An injected-LoRA embedding checkpoint serves from its folded weights and resumes from its adapters.

The embedding trainer injects LoRA in place, so every save folds the adapters into the weights it
writes. The fold serves, but a resume cannot come back from it: the adapters would restart from
initialization on top of the folded weights while their optimizer moments were restored. So every
training checkpoint also carries the run's unfolded trainable tensors in ``resume_adapter/`` plus the
root ``resume_adapter.json`` marker, and resume builds from the base, re-injects, and restores them.
Driven through the real trainer on a tiny random BERT (CPU, fp32, dropout off, so a resumed step is
bit-reproducible):

* layout: the adapter file holds exactly the live trainable tensors at save time, bit for bit, the
  root holds no adapter file or config, and the folded weights load with stock
  ``SentenceTransformer`` / ``AutoModel`` as PEFT's merge of those same tensors; a failed
  adapter write leaves no marker;
* the final ``save_model`` export carries no resume state;
* classification: the marker sends the policy load to the base;
* resume: the restored tensors equal the saved ones bit for bit, the frozen base is the base's, and
  the resumed steps reproduce the uninterrupted run's losses and final adapters exactly; the
  ``load_best_model_at_end`` load restores the best checkpoint's adapters the same way;
* refusals: a marked checkpoint missing its adapter file, an unmarked (folded-only) checkpoint under an
  injected-LoRA run, a model built from the folded weights, a run without injected LoRA, a resume
  adapter for other target modules or rank, and one trained at another LoRA scaling or without its
  recorded scaling;
* ordering: the resume adapter and its marker are on disk before rotation removes the previous
  checkpoint, with ``save_only_model`` too;
* a non-shared filesystem: two one-rank "nodes" each write a complete resume adapter to their own
  directory and each restores its own; one node missing its copy raises on both;
* other targets and variants (the input embedding ``word_embeddings`` beside the attention targets or
  alone, DoRA): the checkpoint serves PEFT's own merge, carries every trainable tensor in its resume
  adapter, and the run resumes exactly; an adapter the save cannot fold (``nn.MultiheadAttention``
  LoRA) is refused at construction;
* the resume-state rules every checkpoint follows: a save over a marked directory leaves no stale
  marker (a full fine-tune's checkpoint is full, a torn LoRA save unmarked); a marked checkpoint
  without its adapter is refused before the policy loads; a bf16 conversion keeps the resume state and
  resumes like its source, while a vocabulary patch, an adapter merge and an N-way merge carry none;
* the backbone's own ``lora_``-named parameters (jina's parametrizations) are base weights, neither
  taken for injected LoRA nor dropped by the fold nor trained; LoRA goes into the backbone only, an
  adapter outside it is refused, and under FSDP2 so is a head outside it that trains or is sharded;
* a full fine-tune: the weight loader the trainer resumes through covers the names its saves write.

    python tests/cpu/trainers/test_embedding_lora_resume.py
"""

import datetime
import json
import os
import shutil
import sys
from types import SimpleNamespace

import pytest
import torch
from accelerate import PartialState
from datasets import Dataset
from peft import LoraConfig, get_peft_model, inject_adapter_in_model
from peft.tuners.tuners_utils import BaseTunerLayer
from safetensors.torch import load_file
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Dense
from tokenizers import Tokenizer, models, pre_tokenizers
from torch.distributed.tensor import Shard, distribute_tensor
from torch.nn.utils.parametrize import register_parametrization
from transformers import (
    AutoModel,
    BertConfig,
    BertLMHeadModel,
    BertModel,
    PreTrainedTokenizerFast,
    TrainerCallback,
)
from transformers.trainer_utils import rotate_checkpoints
from trl import ModelConfig

import src.trainers.embedding.trainer as embedding_module
import src.trainers.mixins.checkpointing as checkpointing_mod
from scripts.after_training.convert_to_bf16 import convert_to_bf16
from scripts.after_training.merge_models import merge_models
from scripts.after_training.merge_peft_adapters import merge_peft_adapter
from scripts.training.embedding import inject_lora
from src.args.distributed_args import DistributedArguments
from src.checkpoint.format import (
    ADAPTER_CONFIG_FILE,
    ADAPTER_SAFETENSORS_FILE,
    RESUME_ADAPTER_DIR,
    RESUME_ADAPTER_MARKER_FILE,
    cast_to_save_dtype,
    read_checkpoint_key_set,
    resume_adapter_dir,
)
from src.configs.embedding_config import EmbeddingConfig
from src.distributed import runtime
from src.distributed.checkpoint.loader import resume_numel_coverage
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.embedding.trainer import EmbeddingTrainer
from src.training.environment import _classify_resume_checkpoint, resolve_resume_weights_source
from tests.common.distributed import fake_process_group_mesh
from tests.common.embedding_lora_resume import backbone_prefix
from tests.common.gloo import run_gloo_ranks
from tests.common.peft_helpers import injected_lora_merge
from tests.common.utils import load_script_module, step_losses

WORDS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", *(f"w{i}" for i in range(60))]
TARGET_MODULES = ("query", "value")
EMBEDDING_TARGET = "word_embeddings"
NUM_LAYERS = 2
LORA_R = 4
# alpha != r, so a fold at a guessed scaling of 1.0 would not match.
LORA_ALPHA = 8
SCALING = LORA_ALPHA / LORA_R
TOTAL_STEPS = 4
SAVE_AT_STEP = 2
# One rank per "node": every rank is its node's local main, hence a checkpoint writer.
patch_vocab = load_script_module("scripts/before_training/patch_vocab.py")
TWO_NODE_ENV = {"LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "1", "DIST_OUTPUT_SHARED_FILESYSTEM": "0"}
PG_TIMEOUT = datetime.timedelta(seconds=120)


def _tiny_base(path) -> str:
    """A random 2-layer BERT and a word-level tokenizer, saved as a plain transformers checkpoint."""
    backend = Tokenizer(models.WordLevel({word: i for i, word in enumerate(WORDS)}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]", cls_token="[CLS]", sep_token="[SEP]"
    )
    torch.manual_seed(0)
    config = BertConfig(
        vocab_size=len(WORDS),
        hidden_size=16,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=2,
        intermediate_size=32,
        max_position_embeddings=32,
    )
    BertModel(config).save_pretrained(str(path))
    tokenizer.save_pretrained(str(path))
    return str(path)


def _lora_model(
    source: str,
    *,
    seed: int,
    target_modules=TARGET_MODULES,
    r: int = LORA_R,
    lora_alpha: int = LORA_ALPHA,
    model: SentenceTransformer | None = None,
    **lora,
) -> SentenceTransformer:
    """``source`` loaded (or ``model``, already built from it), then LoRA-injected by the embedding
    script's own ``inject_lora``."""
    torch.manual_seed(seed)  # the adapter init draws from the global generator
    model = model if model is not None else SentenceTransformer(source, device="cpu")
    model_config = ModelConfig(
        model_name_or_path=source,
        use_peft=True,
        lora_r=r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        lora_target_modules=list(target_modules),
        **lora,
    )
    inject_lora(model, model_config, DistributedArguments())
    return model


def _pairs() -> Dataset:
    return Dataset.from_dict(
        {
            "anchor": [f"w{i} w{i + 1} w{i + 3}" for i in range(32)],
            "positive": [f"w{i + 2} w{i} w{i + 1}" for i in range(32)],
        }
    )


def _trainer(model, output_dir, *, callbacks=(), eval_dataset=None, **overrides) -> EmbeddingTrainer:
    args = EmbeddingConfig(
        **{
            "output_dir": str(output_dir),
            "max_steps": TOTAL_STEPS,
            "per_device_train_batch_size": 4,
            "learning_rate": 1e-2,
            "save_strategy": "steps",
            "save_steps": SAVE_AT_STEP,
            "logging_steps": 1,
            "report_to": "none",
            "bf16": False,
            "use_cpu": True,
            "use_liger_kernel": False,
            "disable_dropout": True,
            "max_length": 16,
            "seed": 42,
            **overrides,
        }
    )
    return EmbeddingTrainer(
        model=model,
        args=args,
        train_dataset=_pairs(),
        eval_dataset=eval_dataset,
        callbacks=list(callbacks),
        parallelism_config=ParallelismConfig(),
    )


def _host(model) -> EmbeddingTrainer:
    """The trainer methods under test over ``model``, without the trainer's construction."""
    host = object.__new__(EmbeddingTrainer)
    host.model = model
    return host


def _trainable(model) -> dict[str, torch.Tensor]:
    return {name: param.detach().clone() for name, param in model.named_parameters() if param.requires_grad}


def _frozen(model) -> dict[str, torch.Tensor]:
    return {name: param.detach().clone() for name, param in model.named_parameters() if not param.requires_grad}


class _Snapshot(TrainerCallback):
    """The trainable tensors right after the checkpoint save at ``SAVE_AT_STEP`` (``on_save``), or
    right after a resume restored them (``on_train_begin``)."""

    def __init__(self, event: str):
        self.event = event
        self.model = None
        self.tensors: dict[str, torch.Tensor] = {}

    def on_save(self, args, state, control, model=None, **kwargs):
        if self.event == "save" and state.global_step == SAVE_AT_STEP:
            self.tensors = _trainable(model)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.event == "train_begin":
            self.tensors = _trainable(model)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """One uninterrupted injected-LoRA run, checkpointed at ``SAVE_AT_STEP``; the tests read its artifacts."""
    PartialState()  # the script's LoRA injection logs through the accelerate logger
    root = tmp_path_factory.mktemp("embedding_lora_resume")
    base = _tiny_base(root / "base")
    at_save = _Snapshot("save")
    trainer = _trainer(_lora_model(base, seed=1), root / "out", callbacks=[at_save])
    trainer.train()
    return SimpleNamespace(
        root=root,
        base=base,
        checkpoint=str(root / "out" / f"checkpoint-{SAVE_AT_STEP}"),
        at_save=at_save.tensors,
        losses=step_losses(trainer),
        final=_trainable(trainer.model),
        trainer=trainer,
    )


def _copy(checkpoint: str, destination) -> str:
    return shutil.copytree(checkpoint, str(destination))


def _expected_fold(
    base: str, adapters: dict[str, torch.Tensor], prefix: str, lora: LoraConfig
) -> dict[str, torch.Tensor]:
    """What PEFT's own in-place merge writes per LoRA target, from ``adapters`` keyed by the ST's names:
    the base loaded, the same adapters injected and loaded, every LoRA layer merged."""
    merged = injected_lora_merge(
        BertModel.from_pretrained(base), lora, {key[len(prefix) :]: value for key, value in adapters.items()}
    )
    return {key: cast_to_save_dtype(value.clone()) for key, value in merged.items() if key.endswith(".weight")}


def _lora_config(target_modules=TARGET_MODULES, **lora) -> LoraConfig:
    """The ``LoraConfig`` ``inject_lora`` builds for :func:`_lora_model`'s settings."""
    return LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.0, target_modules=list(target_modules), **lora)


def _assert_serves_the_fold(
    directory: str,
    base: str,
    adapters: dict[str, torch.Tensor],
    prefix: str,
    folds: int = NUM_LAYERS * len(TARGET_MODULES),
    lora: LoraConfig | None = None,
) -> None:
    backbone, info = AutoModel.from_pretrained(directory, output_loading_info=True)
    problems = {kind: info[kind] for kind in ("missing_keys", "unexpected_keys", "mismatched_keys") if info[kind]}
    assert not problems, f"stock from_pretrained does not load the folded weights cleanly: {problems}"
    served = backbone.state_dict()
    assert not [key for key in served if ".lora_" in key or ".base_layer." in key]
    expected = _expected_fold(base, adapters, prefix, lora or _lora_config())
    assert len(expected) == folds, f"premise: one fold per LoRA target, got {sorted(expected)}"
    for key, value in expected.items():
        assert torch.equal(served[key].to(value.dtype), value), f"{key} is not PEFT's merge of the adapters"
    embeddings = SentenceTransformer(directory, device="cpu").encode(["w1 w2 w3", "w4 w5"], convert_to_tensor=True)
    assert embeddings.shape[0] == 2 and torch.isfinite(embeddings).all()


# --- layout ---------------------------------------------------------------------------------


def test_a_training_checkpoint_carries_its_resume_adapter_beside_the_fold(run):
    checkpoint = run.checkpoint
    assert resume_adapter_dir(checkpoint) == os.path.join(checkpoint, RESUME_ADAPTER_DIR)
    saved = load_file(os.path.join(checkpoint, RESUME_ADAPTER_DIR, ADAPTER_SAFETENSORS_FILE))
    assert run.at_save and set(saved) == set(run.at_save), "the adapter file must hold exactly the trainable tensors"
    changed = [key for key, value in run.at_save.items() if not torch.equal(saved[key], value)]
    assert not changed, f"trainable tensors changed on their way to the resume adapter: {changed[:3]}"
    assert all(saved[key].dtype == value.dtype for key, value in run.at_save.items()), "saved at the live dtype"
    lora_b = [value for key, value in run.at_save.items() if ".lora_B." in key]
    assert any(value.abs().sum() > 0 for value in lora_b), "premise: training moved the zero-init B matrices"
    for name in (ADAPTER_CONFIG_FILE, ADAPTER_SAFETENSORS_FILE):
        assert not os.path.exists(os.path.join(checkpoint, name)), (
            f"{name} at the root would make from_pretrained load the base it names instead of the fold"
        )
    _assert_serves_the_fold(checkpoint, run.base, saved, backbone_prefix(run.trainer.model))


def test_a_failed_adapter_write_leaves_the_checkpoint_unmarked(run, tmp_path, monkeypatch):
    """The marker is the resume's verdict, so it must never outrun its adapter: after a failed write
    the checkpoint stays unmarked, which an injected-LoRA resume refuses."""
    checkpoint = str(tmp_path / "checkpoint-2")
    os.makedirs(checkpoint)

    def full_disk(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(embedding_module, "save_file", full_disk)

    with pytest.raises(OSError, match="No space left"):
        EmbeddingTrainer._save_merged_checkpoint_resume_adapter(_host(_lora_model(run.base, seed=1)), checkpoint)
    assert resume_adapter_dir(checkpoint) is None


def test_the_final_export_is_a_serving_artifact_without_resume_state(run):
    final = str(run.root / "final")
    run.trainer.save_model(final)

    assert resume_adapter_dir(final) is None
    assert not os.path.exists(os.path.join(final, RESUME_ADAPTER_DIR))
    _assert_serves_the_fold(final, run.base, run.final, backbone_prefix(run.trainer.model))


# --- classification -------------------------------------------------------------------------


@pytest.mark.parametrize("use_grouped_gemm", [True, False], ids=["grouped-gemm-default", "grouped-gemm-off"])
def test_the_marker_sends_the_policy_load_to_the_base(run, use_grouped_gemm):
    """The fold is as loadable as a full fine-tune's weights; building from it and restoring the
    adapters would apply the delta twice. The default ``use_grouped_gemm`` takes the classifier."""
    assert _classify_resume_checkpoint(run.checkpoint) == "merged_adapter"
    model_config = SimpleNamespace(model_name_or_path=run.base)
    config = ParallelismConfig(use_grouped_gemm=use_grouped_gemm)
    assert resolve_resume_weights_source(run.checkpoint, model_config, config) == run.base


# --- resume ---------------------------------------------------------------------------------


def test_resume_restores_the_adapters_bit_equal_and_reproduces_the_run(run, tmp_path):
    source = resolve_resume_weights_source(
        run.checkpoint, SimpleNamespace(model_name_or_path=run.base), ParallelismConfig()
    )
    model = _lora_model(source, seed=2)
    fresh = _trainable(model)
    assert any(not torch.equal(fresh[key], run.at_save[key]) for key in fresh), "premise: fresh adapters differ"
    restored = _Snapshot("train_begin")
    trainer = _trainer(model, tmp_path / "resumed", callbacks=[restored])

    trainer.train(resume_from_checkpoint=run.checkpoint)

    assert set(restored.tensors) == set(run.at_save)
    unequal = [key for key, value in run.at_save.items() if not torch.equal(restored.tensors[key], value)]
    assert not unequal, f"adapters not restored bit-equal: {unequal[:3]}"
    base_frozen = _frozen(_lora_model(run.base, seed=3))
    moved = [key for key, value in _frozen(trainer.model).items() if not torch.equal(value, base_frozen[key])]
    assert not moved, f"the frozen base was overwritten on resume (the fold read back?): {moved[:3]}"
    resumed = step_losses(trainer)[-(TOTAL_STEPS - SAVE_AT_STEP) :]
    assert resumed == run.losses[SAVE_AT_STEP:], f"resumed {resumed} != uninterrupted {run.losses[SAVE_AT_STEP:]}"
    final = _trainable(trainer.model)
    drifted = [key for key, value in run.final.items() if not torch.equal(final[key], value)]
    assert not drifted, f"final adapters differ from the uninterrupted run's: {drifted[:3]}"


def test_load_best_model_at_end_restores_the_best_checkpoints_adapters(run, tmp_path):
    """The end-of-run best-model load goes through the same override: the best checkpoint's adapters
    come back from its resume adapter onto the live base, and the final export folds those."""
    trainer = _trainer(
        _lora_model(run.base, seed=1),
        tmp_path / "out",
        eval_dataset=_pairs(),
        eval_strategy="steps",
        eval_steps=SAVE_AT_STEP,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=True,
    )
    trainer.train()

    best = trainer.state.best_model_checkpoint
    assert best is not None and best.endswith(f"checkpoint-{SAVE_AT_STEP}"), (
        f"premise: the best checkpoint must predate the last step, or the load has nothing to restore ({best})"
    )
    saved = load_file(os.path.join(best, RESUME_ADAPTER_DIR, ADAPTER_SAFETENSORS_FILE))
    live = _trainable(trainer.model)
    assert set(live) == set(saved)
    unequal = [key for key, value in saved.items() if not torch.equal(live[key], value)]
    assert not unequal, f"the live adapters are not the best checkpoint's: {unequal[:3]}"
    final = str(tmp_path / "final")
    trainer.save_model(final)
    _assert_serves_the_fold(final, run.base, saved, backbone_prefix(trainer.model))


# --- refusals -------------------------------------------------------------------------------


def test_a_marked_checkpoint_without_its_adapter_file_raises(run, tmp_path):
    checkpoint = _copy(run.checkpoint, tmp_path / "checkpoint-2")
    os.remove(os.path.join(checkpoint, RESUME_ADAPTER_DIR, ADAPTER_SAFETENSORS_FILE))
    trainer = _trainer(_lora_model(run.base, seed=2), tmp_path / "out")

    with pytest.raises(RuntimeError, match="holds no adapter file"):
        trainer._load_from_checkpoint(checkpoint)


@pytest.mark.parametrize("built_from", ["resolver", "base"])
def test_an_unmarked_folded_checkpoint_refuses_an_injected_lora_run(run, tmp_path, built_from):
    """The layout a torn save leaves: folded weights, no marker. The resolver takes it for a full
    checkpoint and builds from the fold; fresh adapters on top of it under restored optimizer moments
    are the silent divergence, so the load raises. Built from the base too, where the adapter file the
    torn save left would otherwise restore with nothing else to refuse it."""
    checkpoint = _copy(run.checkpoint, tmp_path / "checkpoint-2")
    os.remove(os.path.join(checkpoint, RESUME_ADAPTER_MARKER_FILE))
    source = resolve_resume_weights_source(
        checkpoint, SimpleNamespace(model_name_or_path=run.base), ParallelismConfig()
    )
    assert source == checkpoint, "premise: an unmarked fold resolves as a full checkpoint"
    assert os.path.isfile(os.path.join(checkpoint, RESUME_ADAPTER_DIR, ADAPTER_SAFETENSORS_FILE)), (
        "premise: the adapter file is there to restore"
    )
    trainer = _trainer(_lora_model(source if built_from == "resolver" else run.base, seed=2), tmp_path / "out")

    with pytest.raises(ValueError, match="without its resume adapter"):
        trainer._load_from_checkpoint(checkpoint)


def test_a_model_built_from_the_folded_weights_is_refused(run, tmp_path):
    trainer = _trainer(_lora_model(run.checkpoint, seed=2), tmp_path / "out")
    before = _trainable(trainer.model)

    with pytest.raises(ValueError, match="delta would apply twice"):
        trainer._load_from_checkpoint(run.checkpoint)
    assert all(torch.equal(value, before[key]) for key, value in _trainable(trainer.model).items())


def test_a_run_without_injected_lora_refuses_a_marked_checkpoint(run, tmp_path):
    trainer = _trainer(SentenceTransformer(run.base, device="cpu"), tmp_path / "out")

    with pytest.raises(ValueError, match="trains no injected adapters"):
        trainer._load_from_checkpoint(run.checkpoint)


@pytest.mark.parametrize(
    "lora",
    [{"target_modules": ("query", "key")}, {"r": 2}],
    ids=["other-target-modules", "other-rank"],
)
def test_a_resume_adapter_for_another_adapter_layout_is_refused(run, tmp_path, lora):
    trainer = _trainer(_lora_model(run.base, seed=2, **lora), tmp_path / "out")

    with pytest.raises(ValueError, match="does not match this run's trainable tensors"):
        trainer._load_from_checkpoint(run.checkpoint)


@pytest.mark.parametrize("lora", [{"lora_alpha": 2 * LORA_ALPHA}, {"use_rslora": True}], ids=["other-alpha", "rslora"])
def test_a_resume_under_another_lora_scaling_is_refused(run, tmp_path, lora):
    """The adapters would restore by name and shape and every delta would be rescaled: the scaling the
    run saved with is recorded beside its resume adapter and held to the live one."""
    trainer = _trainer(_lora_model(run.base, seed=2, **lora), tmp_path / "out")
    before = _trainable(trainer.model)

    with pytest.raises(ValueError, match="another LoRA scaling"):
        trainer._load_from_checkpoint(run.checkpoint)
    assert all(torch.equal(value, before[key]) for key, value in _trainable(trainer.model).items())


def test_a_resume_adapter_without_its_recorded_scaling_is_refused(run, tmp_path):
    checkpoint = _copy(run.checkpoint, tmp_path / "checkpoint-2")
    with open(os.path.join(checkpoint, RESUME_ADAPTER_DIR, ADAPTER_CONFIG_FILE)) as fh:
        assert json.load(fh)["lora_alpha"] == LORA_ALPHA, "premise: the save records the run's scaling"
    os.remove(os.path.join(checkpoint, RESUME_ADAPTER_DIR, ADAPTER_CONFIG_FILE))
    trainer = _trainer(_lora_model(run.base, seed=2), tmp_path / "out")

    with pytest.raises(ValueError, match="is missing"):
        trainer._load_from_checkpoint(checkpoint)


# --- ordering -------------------------------------------------------------------------------


@pytest.mark.parametrize("save_only_model", [False, True], ids=["exact-resume", "save_only_model"])
def test_the_resume_adapter_is_on_disk_before_rotation(run, tmp_path, monkeypatch, save_only_model):
    """With ``save_total_limit: 1`` rotation deletes the previous checkpoint; a preemption between
    that and the resume-adapter write would leave one checkpoint no resume accepts."""
    output_dir = tmp_path / "out"
    seen = []

    def recording_rotate(**kwargs):
        steps = sorted(int(name.split("-")[1]) for name in os.listdir(output_dir) if name.startswith("checkpoint-"))
        newest = str(output_dir / f"checkpoint-{steps[-1]}")
        complete = resume_adapter_dir(newest) is not None and os.path.isfile(
            os.path.join(newest, RESUME_ADAPTER_DIR, ADAPTER_SAFETENSORS_FILE)
        )
        seen.append((steps, complete))
        rotate_checkpoints(**kwargs)

    monkeypatch.setattr(checkpointing_mod, "rotate_checkpoints", recording_rotate)
    trainer = _trainer(_lora_model(run.base, seed=1), output_dir, save_total_limit=1, save_only_model=save_only_model)
    trainer.train()

    assert seen == [([SAVE_AT_STEP], True), ([SAVE_AT_STEP, TOTAL_STEPS], True)]
    assert sorted(os.listdir(output_dir)) == [f"checkpoint-{TOTAL_STEPS}"]


# --- non-shared filesystem ------------------------------------------------------------------


def _node_checkpoint(root: str, rank: int) -> str:
    """This node's own checkpoint directory: on a non-shared FS a node sees only its own disk."""
    return os.path.join(root, f"node_{rank}", "checkpoint-2")


def _node_worker(rank: int, root: str, base: str) -> None:
    problems: list[str] = []
    try:
        runtime.resolve_shared_filesystem_consensus()  # what init_distributed does for a real run
        PartialState()
        if runtime.is_output_shared_filesystem() or not runtime.fs_aware_save_rank():
            problems.append("premise: this rank is not its node's checkpoint writer")
        checkpoint = _node_checkpoint(root, rank)
        os.makedirs(checkpoint)

        # Node-specific values: a rank restoring another node's copy must fail the comparison below.
        trained = _lora_model(base, seed=1)
        with torch.no_grad():
            generator = torch.Generator().manual_seed(100 + rank)
            for param in trained.parameters():
                if param.requires_grad:
                    param.copy_(torch.randn(param.shape, generator=generator))
        expected = _trainable(trained)
        EmbeddingTrainer._save_merged_checkpoint_resume_adapter(_host(trained), checkpoint)

        if resume_adapter_dir(checkpoint) is None:
            problems.append("this node's checkpoint carries no marker")
        saved = load_file(os.path.join(checkpoint, RESUME_ADAPTER_DIR, ADAPTER_SAFETENSORS_FILE))
        if set(saved) != set(expected) or any(not torch.equal(saved[k], v) for k, v in expected.items()):
            problems.append("this node's resume adapter does not hold its own trainable tensors")
        with open(os.path.join(root, f"first_tensor_{rank}.txt"), "w") as fh:
            fh.write(repr(saved[min(saved)].flatten()[:4].tolist()))

        fresh = _lora_model(base, seed=2)
        EmbeddingTrainer._load_from_checkpoint(_host(fresh), checkpoint)
        if any(not torch.equal(value, expected[key]) for key, value in _trainable(fresh).items()):
            problems.append("the restore did not reproduce this node's own tensors")

        if rank == 1:
            os.remove(os.path.join(checkpoint, RESUME_ADAPTER_DIR, ADAPTER_SAFETENSORS_FILE))
        try:
            EmbeddingTrainer._load_from_checkpoint(_host(_lora_model(base, seed=2)), checkpoint)
            problems.append("a resume adapter missing on one node did not raise")
        except RuntimeError as e:
            if "present on some ranks but missing on others" not in str(e):
                problems.append(f"a resume adapter missing on one node raised the wrong error: {e}")
    except Exception as e:
        problems.append(f"{type(e).__name__}: {str(e).splitlines()[0][:300]}")

    with open(os.path.join(root, f"result_{rank}.txt"), "w") as fh:
        fh.write("PASS" if not problems else "FAIL: " + "; ".join(problems))
    runtime.reset_shared_filesystem_consensus()


def test_each_node_writes_and_restores_its_own_resume_adapter(tmp_path):
    """Both ranks are writers, each on its own directory: a rank-0-only write leaves node 1 nothing to
    resume from, and a restore that read another node's copy fails the per-node comparison, since the
    two nodes hold different tensors. A copy missing on one node then raises on both, not one."""
    base = _tiny_base(tmp_path / "base")
    run_gloo_ranks(_node_worker, 2, str(tmp_path), base, pg_timeout=PG_TIMEOUT, env=TWO_NODE_ENV)

    for rank in range(2):
        result = (tmp_path / f"result_{rank}.txt").read_text()
        assert result == "PASS", f"rank {rank}: {result}"
    first = [(tmp_path / f"first_tensor_{rank}.txt").read_text() for rank in range(2)]
    assert first[0] != first[1], "premise: the two nodes hold different tensors"


# --- other targets and variants ---------------------------------------------------------------


@pytest.fixture(
    scope="module",
    # (lora_target_modules, other LoRA settings, adapted modules). TRL collapses a one-entry list into a
    # string, which PEFT reads as a regex, so embedding-only is spelled as a user writes it: the
    # embedding beside an lm_head the headless backbone lacks.
    params=[
        ((EMBEDDING_TARGET, *TARGET_MODULES), {}, 1 + NUM_LAYERS * len(TARGET_MODULES)),
        ((EMBEDDING_TARGET, "lm_head"), {}, 1),
        ((EMBEDDING_TARGET, *TARGET_MODULES), {"use_dora": True}, 1 + NUM_LAYERS * len(TARGET_MODULES)),
    ],
    ids=["embedding-and-attention", "embedding-only", "dora"],
)
def variant_run(request, tmp_path_factory):
    """``run`` with another target set or variant: PEFT adapts the input embedding as a
    ``lora.Embedding``, whose ``lora_embedding_A``/``_B`` are neither ``lora_A`` nor ``lora_B``, and DoRA
    adds a ``lora_magnitude_vector`` its merge rescales by."""
    PartialState()
    targets, lora, folds = request.param
    root = tmp_path_factory.mktemp("embedding_lora_variant")
    base = _tiny_base(root / "base")
    at_save = _Snapshot("save")
    trainer = _trainer(_lora_model(base, seed=1, target_modules=targets, **lora), root / "out", callbacks=[at_save])
    trainer.train()
    return SimpleNamespace(
        base=base,
        targets=targets,
        lora=lora,
        folds=folds,
        checkpoint=str(root / "out" / f"checkpoint-{SAVE_AT_STEP}"),
        at_save=at_save.tensors,
        losses=step_losses(trainer),
        final=_trainable(trainer.model),
        prefix=backbone_prefix(trainer.model),
    )


def test_every_target_serves_peft_merge_of_its_adapters(variant_run):
    """The checkpoint serves PEFT's merge (``(B @ A)ᵀ`` for the embedding, DoRA's rescale) under the plain
    names, with no adapter key left over, and its resume adapter holds every trainable tensor bit for bit."""
    moved = [
        key for key, value in variant_run.at_save.items() if key.endswith(".lora_embedding_A.default") and value.any()
    ]
    assert moved or EMBEDDING_TARGET not in variant_run.targets, (
        "premise: training moved the zero-init embedding A factor, so an unfolded table is detectable"
    )
    assert resume_adapter_dir(variant_run.checkpoint) is not None
    saved = load_file(os.path.join(variant_run.checkpoint, RESUME_ADAPTER_DIR, ADAPTER_SAFETENSORS_FILE))
    assert set(saved) == set(variant_run.at_save)
    assert all(torch.equal(saved[key], value) for key, value in variant_run.at_save.items())
    _assert_serves_the_fold(
        variant_run.checkpoint,
        variant_run.base,
        saved,
        variant_run.prefix,
        variant_run.folds,
        _lora_config(variant_run.targets, **variant_run.lora),
    )


def test_every_target_resumes_bit_equal_and_reproduces_the_run(variant_run, tmp_path):
    source = resolve_resume_weights_source(
        variant_run.checkpoint, SimpleNamespace(model_name_or_path=variant_run.base), ParallelismConfig()
    )
    assert source == variant_run.base
    restored = _Snapshot("train_begin")
    model = _lora_model(source, seed=2, target_modules=variant_run.targets, **variant_run.lora)
    trainer = _trainer(model, tmp_path / "resumed", callbacks=[restored])

    trainer.train(resume_from_checkpoint=variant_run.checkpoint)

    assert set(restored.tensors) == set(variant_run.at_save)
    unequal = [key for key, value in variant_run.at_save.items() if not torch.equal(restored.tensors[key], value)]
    assert not unequal, f"adapters not restored bit-equal: {unequal[:3]}"
    resumed = step_losses(trainer)[-(TOTAL_STEPS - SAVE_AT_STEP) :]
    assert resumed == variant_run.losses[SAVE_AT_STEP:]
    final = _trainable(trainer.model)
    assert all(torch.equal(final[key], value) for key, value in variant_run.final.items())


class _AttentionBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(8, 2)


def test_an_adapter_the_save_cannot_fold_is_refused_at_construction():
    """An ``nn.MultiheadAttention`` LoRA has no out-of-place fold. Refused when the trainer is built, on
    every rank, rather than on the save rank alone at the first checkpoint, where its peers would
    block in the save's collectives."""
    backbone = inject_adapter_in_model(LoraConfig(r=2, target_modules=["attn"]), _AttentionBackbone())
    host = _host(backbone)
    host._get_unwrapped_model = lambda: backbone

    with pytest.raises(ValueError, match="cannot be folded out of place"):
        EmbeddingTrainer._validate_injected_lora_foldable(host)


# --- the backbone's own lora_ parameters, and modules outside the backbone ---------------------


class _NativeLora(torch.nn.Module):
    """A remote-code backbone's own LoRA, as jina-embeddings-v3 parametrizes its weights: base
    parameters named ``lora_A`` / ``lora_B`` that no PEFT tuner layer owns."""

    def __init__(self, features: int):
        super().__init__()
        self.lora_A = torch.nn.Parameter(0.1 * torch.randn(2, features))
        self.lora_B = torch.nn.Parameter(0.1 * torch.randn(features, 2))

    def forward(self, weight):
        return weight + self.lora_B @ self.lora_A


def _with_native_lora(model: SentenceTransformer) -> list[str]:
    """Parametrize each attention ``key`` projection of ``model``'s backbone with :class:`_NativeLora`;
    returns the backbone's names of the native factors."""
    backbone = model[0].auto_model
    for layer in backbone.encoder.layer:
        key = layer.attention.self.key
        register_parametrization(key, "weight", _NativeLora(key.in_features))
    return [name for name, _ in backbone.named_parameters() if ".lora_" in name]


def test_a_backbones_own_lora_parameters_are_not_injected_lora(run):
    """Taken for injected LoRA, such a backbone's saves would fold and drop its own factors."""
    model = SentenceTransformer(run.base, device="cpu")
    native = _with_native_lora(model)

    assert native, "premise: the backbone carries lora_-named parameters of its own"
    assert not EmbeddingTrainer._has_injected_lora(_host(model))


def test_the_fold_keeps_a_backbones_own_lora_parameters_and_trains_only_the_injected_ones(run):
    """With injected LoRA on top, the fold drops the tensors the tuner layers own and nothing else, and
    the script trains those alone."""
    model = SentenceTransformer(run.base, device="cpu")
    native = _with_native_lora(model)
    _lora_model(run.base, seed=1, model=model)
    backbone = model[0].auto_model

    written = dict(embedding_module._folded_backbone_items(backbone))

    assert EmbeddingTrainer._has_injected_lora(_host(model))
    assert all(name in written for name in native), "the fold dropped the backbone's own LoRA factors"
    assert not [key for key in written if ".lora_A.default" in key or ".base_layer" in key]
    assert all(f"encoder.layer.{i}.attention.self.query.weight" in written for i in range(NUM_LAYERS))
    trainable = [name for name, param in backbone.named_parameters() if param.requires_grad]
    assert trainable and all(".default" in name for name in trainable), f"not the injected adapters: {trainable}"


def _with_dense_head(model: SentenceTransformer) -> Dense:
    """A trainable projection after the pooling, as some released embedding pipelines carry."""
    dim = model.get_sentence_embedding_dimension()
    dense = Dense(in_features=dim, out_features=dim)
    model.append(dense)
    return dense


def test_the_script_injects_lora_into_the_backbone_only(run):
    model = SentenceTransformer(run.base, device="cpu")
    dense = _with_dense_head(model)

    _lora_model(run.base, seed=1, target_modules=(*TARGET_MODULES, "linear"), model=model)

    assert any(isinstance(module, BaseTunerLayer) for module in model[0].auto_model.modules())
    assert not any(isinstance(module, BaseTunerLayer) for module in dense.modules()), "the head was adapted"


def test_lora_outside_the_backbone_is_refused_at_construction(run, tmp_path):
    """Checkpoints fold and resume the backbone's adapters only; one on the head would leave both."""
    model = _lora_model(run.base, seed=1)
    inject_adapter_in_model(LoraConfig(r=2, target_modules=["linear"]), _with_dense_head(model))

    with pytest.raises(ValueError, match="LoRA adapters outside"):
        _trainer(model, tmp_path / "out")


def _wrapped_host(model: SentenceTransformer, *, fsdp: bool) -> EmbeddingTrainer:
    host = _host(model)
    host._fsdp_wrapped = fsdp
    host.parallelism_config = ParallelismConfig()
    return host


def test_a_trainable_parameter_outside_the_backbone_is_refused_under_fsdp2(run):
    """The save writes the head rank-locally and resume restores the backbone alone, so the head's
    training is lost; a single process and DDP save and train the whole pipeline."""
    model = SentenceTransformer(run.base, device="cpu")
    _with_dense_head(model)

    with pytest.raises(ValueError, match="outside the SentenceTransformer's transformer backbone"):
        EmbeddingTrainer._validate_modules_outside_backbone(_wrapped_host(model, fsdp=True))
    EmbeddingTrainer._validate_modules_outside_backbone(_wrapped_host(model, fsdp=False))


def test_a_frozen_head_passes_unless_fsdp2_shards_it(run):
    """A frozen head FSDP2 leaves whole (a dtype exclusion) saves whole; a sharded one would be saved
    as this rank's shard."""
    model = SentenceTransformer(run.base, device="cpu")
    linear = _with_dense_head(model).linear
    linear.requires_grad_(False)
    EmbeddingTrainer._validate_modules_outside_backbone(_wrapped_host(model, fsdp=True))

    with fake_process_group_mesh(rank=0, world_size=2) as mesh:
        linear.weight = torch.nn.Parameter(
            distribute_tensor(linear.weight.data, mesh, [Shard(0)]), requires_grad=False
        )
        with pytest.raises(ValueError, match="that FSDP2 shards"):
            EmbeddingTrainer._validate_modules_outside_backbone(_wrapped_host(model, fsdp=True))


# --- the resume-state rules every checkpoint follows ---------------------------------------------


def test_a_save_over_a_marked_directory_leaves_no_stale_marker(run, tmp_path, monkeypatch):
    """A run resumed from an earlier checkpoint rewrites ``checkpoint-N`` in place. A marker the old save
    left there would vouch for the old adapter beside the new weights, so every save removes it first:
    a full fine-tune writes an unmarked, full checkpoint, and an injected-LoRA save torn before its own
    adapter write leaves none."""
    checkpoint = tmp_path / "out" / f"checkpoint-{SAVE_AT_STEP}"
    _copy(run.checkpoint, checkpoint)
    _trainer(SentenceTransformer(run.base, device="cpu"), tmp_path / "out", max_steps=SAVE_AT_STEP).train()
    assert resume_adapter_dir(str(checkpoint)) is None
    assert _classify_resume_checkpoint(str(checkpoint)) == "full"

    torn = tmp_path / "torn" / f"checkpoint-{SAVE_AT_STEP}"
    _copy(run.checkpoint, torn)

    def full_disk(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(embedding_module, "save_file", full_disk)
    with pytest.raises(OSError, match="No space left"):
        _trainer(_lora_model(run.base, seed=1), tmp_path / "torn", max_steps=SAVE_AT_STEP).train()
    assert resume_adapter_dir(str(torn)) is None


def test_a_marked_checkpoint_without_its_adapter_is_refused_before_the_policy_loads(run, tmp_path):
    checkpoint = _copy(run.checkpoint, tmp_path / "checkpoint-2")
    os.remove(os.path.join(checkpoint, RESUME_ADAPTER_DIR, ADAPTER_SAFETENSORS_FILE))

    with pytest.raises(ValueError, match="holds no adapter file"):
        resolve_resume_weights_source(checkpoint, SimpleNamespace(model_name_or_path=run.base), ParallelismConfig())


def _resume_state(directory) -> list[str]:
    return [
        name
        for name in (RESUME_ADAPTER_MARKER_FILE, RESUME_ADAPTER_DIR)
        if os.path.exists(os.path.join(directory, name))
    ]


def test_a_bf16_conversion_keeps_the_resume_state_and_resumes_like_its_source(run, tmp_path):
    out = str(tmp_path / "bf16")

    convert_to_bf16(run.checkpoint, out, "base")

    assert _classify_resume_checkpoint(out) == "merged_adapter"
    source_adapter = os.path.join(run.checkpoint, RESUME_ADAPTER_DIR, ADAPTER_SAFETENSORS_FILE)
    with (
        open(source_adapter, "rb") as source,
        open(os.path.join(out, RESUME_ADAPTER_DIR, ADAPTER_SAFETENSORS_FILE), "rb") as copy_,
    ):
        assert source.read() == copy_.read(), "the resume adapter changed in the conversion"
    assert os.path.isfile(os.path.join(out, "modules.json")), "the ST pipeline config was dropped"
    model = _lora_model(run.base, seed=2)
    _trainer(model, tmp_path / "resumed")._load_from_checkpoint(out)
    assert all(torch.equal(value, run.at_save[key]) for key, value in _trainable(model).items())


def test_a_new_base_built_from_the_checkpoint_carries_none_of_its_resume_state(run, tmp_path, monkeypatch):
    """A vocabulary patch, an adapter merged into it and an N-way merge are new models: resumed, the
    marker would rebuild them from the original base plus this run's adapter, dropping their weights."""
    patched = tmp_path / "patched"
    monkeypatch.setattr(
        sys,
        "argv",
        ["patch_vocab.py", "--model_id", run.checkpoint, "--output_dir", str(patched), "--patterns", '["w1 w2"]'],
    )
    patch_vocab.main()

    adapter = tmp_path / "adapter"
    # The tool loads the widest class, the LM head over the encoder, so the adapter addresses that.
    peft_model = get_peft_model(
        BertLMHeadModel.from_pretrained(run.checkpoint), LoraConfig(r=2, target_modules=["query"])
    )
    peft_model.peft_config["default"].base_model_name_or_path = run.checkpoint
    peft_model.save_pretrained(adapter)
    merged_adapter = tmp_path / "merged_adapter"
    merge_peft_adapter(adapter_dir=str(adapter), output_dir=str(merged_adapter), dtype=torch.float32, verbose=False)

    merged = tmp_path / "merged_models"
    merge_models(
        model_specs=[run.checkpoint, run.base],
        output_dir=str(merged),
        method="linear",
        dtype="float32",
        tokenizer_source=run.checkpoint,
        verbose=False,
    )

    for out in (patched, merged_adapter, merged):
        assert _resume_state(out) == [], f"{out.name} carries its source run's resume state"
        assert _classify_resume_checkpoint(str(out)) == "full"


# --- full fine-tune -------------------------------------------------------------------------


def test_the_weight_loader_covers_the_names_a_full_fine_tune_saves(tmp_path):
    """Every save writes the backbone's names, so the loader a full fine-tune resumes through must
    hold the backbone: handed the SentenceTransformer (``0.<module>.*``) it matches none of them, and
    the FSDP2 / TP reloads refuse the checkpoint at their coverage gate. The optimizer store keeps the
    SentenceTransformer, whose parameters the optimizer steps."""
    PartialState()
    base = _tiny_base(tmp_path / "base")
    _trainer(SentenceTransformer(base, device="cpu"), tmp_path / "out", max_steps=SAVE_AT_STEP).train()
    checkpoint = str(tmp_path / "out" / f"checkpoint-{SAVE_AT_STEP}")
    trainer = _trainer(SentenceTransformer(base, device="cpu"), tmp_path / "resumed")

    loader_model = trainer._checkpoint_loader().ctx.model
    covered, unmatched, _matched, _total = resume_numel_coverage(loader_model, read_checkpoint_key_set(checkpoint))

    assert covered and not unmatched, f"the loader's model misses the saved names: {sorted(unmatched)[:3]}"
    assert trainer._optimizer_store().ctx.model is trainer.model


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
