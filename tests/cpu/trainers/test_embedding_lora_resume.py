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
  ``SentenceTransformer`` / ``AutoModel`` as ``base + scaling · B @ A`` of those same tensors; a failed
  adapter write leaves no marker;
* the final ``save_model`` export carries no resume state;
* classification: the marker sends the policy load to the base;
* resume: the restored tensors equal the saved ones bit for bit, the frozen base is the base's, and
  the resumed steps reproduce the uninterrupted run's losses and final adapters exactly; the
  ``load_best_model_at_end`` load restores the best checkpoint's adapters the same way;
* refusals: a marked checkpoint missing its adapter file, an unmarked (folded-only) checkpoint under an
  injected-LoRA run, a model built from the folded weights, a run without injected LoRA, and a resume
  adapter for other target modules or rank;
* ordering: the resume adapter and its marker are on disk before rotation removes the previous
  checkpoint, with ``save_only_model`` too;
* a non-shared filesystem: two one-rank "nodes" each write a complete resume adapter to their own
  directory and each restores its own; one node missing its copy raises on both;
* an input-embedding target (``word_embeddings``, beside the attention targets or alone): its
  ``lora_embedding_A``/``_B`` fold as ``base + scaling · (B @ A)ᵀ``, the checkpoint carries them in its
  resume adapter, and the run resumes exactly; an adapter the fold cannot express (DoRA) is refused
  at construction;
* a full fine-tune: the weight loader the trainer resumes through covers the names its saves write.

    python tests/cpu/trainers/test_embedding_lora_resume.py
"""

import datetime
import os
import shutil
from types import SimpleNamespace

import pytest
import torch
from accelerate import PartialState
from datasets import Dataset
from safetensors.torch import load_file
from sentence_transformers import SentenceTransformer
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import AutoModel, BertConfig, BertModel, PreTrainedTokenizerFast, TrainerCallback
from transformers.trainer_utils import rotate_checkpoints
from trl import ModelConfig

import src.trainers.embedding.trainer as embedding_module
import src.trainers.mixins.checkpointing as checkpointing_mod
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
from tests.common.gloo import run_gloo_ranks
from tests.common.utils import step_losses

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
    source: str, *, seed: int, target_modules=TARGET_MODULES, r: int = LORA_R, **lora
) -> SentenceTransformer:
    """``source`` loaded, then LoRA-injected by the embedding script's own ``inject_lora``."""
    torch.manual_seed(seed)  # the adapter init draws from the global generator
    model = SentenceTransformer(source, device="cpu")
    model_config = ModelConfig(
        model_name_or_path=source,
        use_peft=True,
        lora_r=r,
        lora_alpha=LORA_ALPHA,
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


# PEFT's adapter spellings: the A factor's key suffix -> the B factor's, and whether the delta is
# ``(B @ A)ᵀ`` (an embedding's A is ``[r, vocab]``) rather than ``B @ A``.
_ADAPTER_SUFFIXES = {
    ".lora_A.default.weight": (".lora_B.default.weight", False),
    ".lora_embedding_A.default": (".lora_embedding_B.default", True),
}


def _expected_fold(base: str, adapters: dict[str, torch.Tensor], backbone_prefix: str) -> dict[str, torch.Tensor]:
    """PEFT's merge per LoRA target at the save dtype, from ``adapters`` keyed by the ST's names."""
    base_state = BertModel.from_pretrained(base).state_dict()
    expected = {}
    for key, lora_a in adapters.items():
        for a_suffix, (b_suffix, transposed) in _ADAPTER_SUFFIXES.items():
            if not key.endswith(a_suffix):
                continue
            module = key[len(backbone_prefix) : -len(a_suffix)]
            delta = adapters[key[: -len(a_suffix)] + b_suffix].float() @ lora_a.float()
            weight = base_state[f"{module}.weight"]
            expected[f"{module}.weight"] = cast_to_save_dtype(
                (weight.float() + SCALING * (delta.T if transposed else delta)).to(weight.dtype)
            )
    return expected


def _backbone_prefix(model: SentenceTransformer) -> str:
    backbone = model[0].auto_model
    return next(name for name, module in model.named_modules() if module is backbone) + "."


def _assert_serves_the_fold(
    directory: str,
    base: str,
    adapters: dict[str, torch.Tensor],
    prefix: str,
    folds: int = NUM_LAYERS * len(TARGET_MODULES),
) -> None:
    backbone, info = AutoModel.from_pretrained(directory, output_loading_info=True)
    problems = {kind: info[kind] for kind in ("missing_keys", "unexpected_keys", "mismatched_keys") if info[kind]}
    assert not problems, f"stock from_pretrained does not load the folded weights cleanly: {problems}"
    served = backbone.state_dict()
    assert not [key for key in served if ".lora_" in key or ".base_layer." in key]
    expected = _expected_fold(base, adapters, prefix)
    assert len(expected) == folds, f"premise: one fold per LoRA target, got {sorted(expected)}"
    for key, value in expected.items():
        assert torch.equal(served[key].to(value.dtype), value), f"{key} is not base + scaling · B @ A of the adapters"
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
    _assert_serves_the_fold(checkpoint, run.base, saved, _backbone_prefix(run.trainer.model))


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
    _assert_serves_the_fold(final, run.base, run.final, _backbone_prefix(run.trainer.model))


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
    _assert_serves_the_fold(final, run.base, saved, _backbone_prefix(trainer.model))


# --- refusals -------------------------------------------------------------------------------


def test_a_marked_checkpoint_without_its_adapter_file_raises(run, tmp_path):
    checkpoint = _copy(run.checkpoint, tmp_path / "checkpoint-2")
    os.remove(os.path.join(checkpoint, RESUME_ADAPTER_DIR, ADAPTER_SAFETENSORS_FILE))
    trainer = _trainer(_lora_model(run.base, seed=2), tmp_path / "out")

    with pytest.raises(RuntimeError, match="holds no adapter file"):
        trainer._load_from_checkpoint(checkpoint)


def test_an_unmarked_folded_checkpoint_refuses_an_injected_lora_run(run, tmp_path):
    """The layout a torn save leaves: folded weights, no marker. The resolver takes it for a full
    checkpoint and builds from the fold; fresh adapters on top of it under restored optimizer moments
    are the silent divergence, so the load raises."""
    checkpoint = _copy(run.checkpoint, tmp_path / "checkpoint-2")
    os.remove(os.path.join(checkpoint, RESUME_ADAPTER_MARKER_FILE))
    source = resolve_resume_weights_source(
        checkpoint, SimpleNamespace(model_name_or_path=run.base), ParallelismConfig()
    )
    assert source == checkpoint, "premise: an unmarked fold resolves as a full checkpoint"
    trainer = _trainer(_lora_model(source, seed=2), tmp_path / "out")

    with pytest.raises(ValueError, match="without a resume adapter"):
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


# --- input-embedding targets ----------------------------------------------------------------


@pytest.fixture(
    scope="module",
    # (targets, adapted modules). TRL collapses a one-entry list into a string, which PEFT reads as a
    # regex, so embedding-only is spelled as a user writes it: the embedding beside an lm_head the
    # headless backbone lacks.
    params=[
        ((EMBEDDING_TARGET, *TARGET_MODULES), 1 + NUM_LAYERS * len(TARGET_MODULES)),
        ((EMBEDDING_TARGET, "lm_head"), 1),
    ],
    ids=["embedding-and-attention", "embedding-only"],
)
def embedding_run(request, tmp_path_factory):
    """``run`` with the input embedding among the targets: PEFT adapts it as a ``lora.Embedding``,
    whose ``lora_embedding_A``/``_B`` are neither ``lora_A`` nor ``lora_B``."""
    PartialState()
    targets, folds = request.param
    root = tmp_path_factory.mktemp("embedding_lora_embedding_target")
    base = _tiny_base(root / "base")
    at_save = _Snapshot("save")
    trainer = _trainer(_lora_model(base, seed=1, target_modules=targets), root / "out", callbacks=[at_save])
    trainer.train()
    return SimpleNamespace(
        base=base,
        targets=targets,
        folds=folds,
        checkpoint=str(root / "out" / f"checkpoint-{SAVE_AT_STEP}"),
        at_save=at_save.tensors,
        losses=step_losses(trainer),
        final=_trainable(trainer.model),
        prefix=_backbone_prefix(trainer.model),
    )


def test_an_embedding_target_is_folded_into_the_served_table(embedding_run):
    """The checkpoint serves ``base + scaling · (B @ A)ᵀ`` as the plain ``word_embeddings.weight``, with no
    adapter key left over, and its resume adapter holds the embedding factors bit for bit."""
    embedding_a = [key for key in embedding_run.at_save if key.endswith(".lora_embedding_A.default")]
    assert len(embedding_a) == 1 and embedding_run.at_save[embedding_a[0]].abs().sum() > 0, (
        "premise: training moved the zero-init embedding A factor, so an unfolded table is detectable"
    )
    assert resume_adapter_dir(embedding_run.checkpoint) is not None
    saved = load_file(os.path.join(embedding_run.checkpoint, RESUME_ADAPTER_DIR, ADAPTER_SAFETENSORS_FILE))
    assert set(saved) == set(embedding_run.at_save)
    assert all(torch.equal(saved[key], value) for key, value in embedding_run.at_save.items())
    _assert_serves_the_fold(
        embedding_run.checkpoint, embedding_run.base, saved, embedding_run.prefix, embedding_run.folds
    )


def test_an_embedding_target_resumes_bit_equal_and_reproduces_the_run(embedding_run, tmp_path):
    source = resolve_resume_weights_source(
        embedding_run.checkpoint, SimpleNamespace(model_name_or_path=embedding_run.base), ParallelismConfig()
    )
    assert source == embedding_run.base
    restored = _Snapshot("train_begin")
    trainer = _trainer(
        _lora_model(source, seed=2, target_modules=embedding_run.targets), tmp_path / "resumed", callbacks=[restored]
    )

    trainer.train(resume_from_checkpoint=embedding_run.checkpoint)

    assert set(restored.tensors) == set(embedding_run.at_save)
    unequal = [key for key, value in embedding_run.at_save.items() if not torch.equal(restored.tensors[key], value)]
    assert not unequal, f"adapters not restored bit-equal: {unequal[:3]}"
    resumed = step_losses(trainer)[-(TOTAL_STEPS - SAVE_AT_STEP) :]
    assert resumed == embedding_run.losses[SAVE_AT_STEP:]
    final = _trainable(trainer.model)
    assert all(torch.equal(final[key], value) for key, value in embedding_run.final.items())


def test_an_adapter_the_fold_cannot_express_is_refused_at_construction(tmp_path):
    """DoRA's magnitude has no ``B @ A`` fold. Refused when the trainer is built, on every rank, rather
    than on the save rank alone at the first checkpoint, where its peers would block in the save."""
    PartialState()
    base = _tiny_base(tmp_path / "base")

    with pytest.raises(NotImplementedError, match="have no such fold"):
        _trainer(_lora_model(base, seed=1, use_dora=True), tmp_path / "out")


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
