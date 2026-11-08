#!/usr/bin/env python
"""An injected-LoRA embedding checkpoint serves from its folded weights and resumes from its adapters.
A resume from the fold would restart the adapters under restored moments, hence ``resume_adapter/`` and its marker."""

import ast
import datetime
import inspect
import json
import os
import shutil
import sys
import textwrap
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

import halo.checkpoint.parallel.adapters as peft_mod
import halo.trainers.embedding.trainer as embedding_module
import halo.trainers.loop.checkpointing as checkpointing_mod
from halo.checkpoint.adapters import resume_adapter_dir
from halo.checkpoint.format import (
    ADAPTER_CONFIG_FILE,
    ADAPTER_SAFETENSORS_FILE,
    RESUME_ADAPTER_DIR,
    RESUME_ADAPTER_MARKER_FILE,
)
from halo.checkpoint.parallel.resume_source import classify_resume_weights, resolve_resume_weights_source
from halo.checkpoint.parallel.resume_weights import resume_numel_coverage
from halo.checkpoint.reading import resolve_checkpoint_weights
from halo.checkpoint.tools.adapter_merge import merge_peft_adapter
from halo.checkpoint.tools.bf16_conversion import convert_to_bf16
from halo.checkpoint.tools.model_merge import merge_models
from halo.configs.embedding import EmbeddingConfig, EmbeddingScriptArguments
from halo.configs.script_arguments import ModelArguments
from halo.distributed.parallelism_config import ParallelismConfig
from halo.models.peft_adapters import lora_fold_targets
from halo.runtime import filesystem
from halo.trainers.embedding.embedding_rules import reject_modules_outside_backbone
from halo.trainers.embedding.injected_lora import folded_backbone_chunks, has_injected_lora
from halo.trainers.embedding.trainer import EmbeddingTrainer
from scripts.training.embedding import inject_lora
from tests.common.adapters import injected_lora_fold
from tests.common.checkpoint_io import loading_problems
from tests.common.distributed import fake_process_group_mesh, reset_shared_filesystem_consensus
from tests.common.embedding_lora_resume import backbone_prefix
from tests.common.gloo import run_gloo_ranks
from tests.common.repo import load_script_module
from tests.common.training_checks import snapshot_trainable, step_losses

WORDS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", *(f"w{i}" for i in range(60))]
TARGET_MODULES = ("query", "value")
EMBEDDING_TARGET = "word_embeddings"
NUM_LAYERS = 2
LORA_R = 4
# alpha != r, so a fold at a guessed scaling of 1.0 would not match.
LORA_ALPHA = 8
TOTAL_STEPS = 4
SAVE_AT_STEP = 2
# TWO_NODE_ENV: one rank per node, so every rank is its node's local main, hence a checkpoint writer.
patch_vocab = load_script_module("scripts/before_training/patch_vocab.py")
TWO_NODE_ENV = {"LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "1", "DIST_OUTPUT_SHARED_FILESYSTEM": "0"}
PG_TIMEOUT = datetime.timedelta(seconds=120)


def _tiny_base(path) -> str:
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
    torch.manual_seed(seed)  # the adapter init draws from the global generator
    model = model if model is not None else SentenceTransformer(source, device="cpu")
    model_config = ModelArguments(
        model_name_or_path=source,
        use_peft=True,
        lora_r=r,
        lora_alpha=lora_alpha,
        lora_dropout=0.0,
        lora_target_modules=list(target_modules),
        **lora,
    )
    inject_lora(model, model_config, EmbeddingScriptArguments())
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
    host = object.__new__(EmbeddingTrainer)
    host.model = model
    host.parallelism_config = ParallelismConfig()
    return host


def _frozen(model) -> dict[str, torch.Tensor]:
    return {name: param.detach().clone() for name, param in model.named_parameters() if not param.requires_grad}


class _Snapshot(TrainerCallback):
    """The trainable tensors after the save at ``SAVE_AT_STEP`` or after a resume restored them."""

    def __init__(self, event: str):
        self.event = event
        self.model = None
        self.tensors: dict[str, torch.Tensor] = {}

    def on_save(self, args, state, control, model=None, **kwargs):
        if self.event == "save" and state.global_step == SAVE_AT_STEP:
            self.tensors = snapshot_trainable(model)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.event == "train_begin":
            self.tensors = snapshot_trainable(model)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """One uninterrupted injected-LoRA run, checkpointed at ``SAVE_AT_STEP``; the tests read its artifacts."""
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
        final=snapshot_trainable(trainer.model),
        trainer=trainer,
    )


def _copy(checkpoint: str, destination) -> str:
    return shutil.copytree(checkpoint, str(destination))


def _lora_config(target_modules=TARGET_MODULES, **lora) -> LoraConfig:
    """The ``LoraConfig`` ``inject_lora`` builds."""
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
    problems = loading_problems(info)
    assert not problems, f"stock from_pretrained does not load the folded weights cleanly: {problems}"
    served = backbone.state_dict()
    assert not [key for key in served if ".lora_" in key or ".base_layer." in key]
    expected = injected_lora_fold(BertModel.from_pretrained(base), lora or _lora_config(), adapters, prefix)
    assert len(expected) == folds, f"premise: one fold per LoRA target, got {sorted(expected)}"
    for key, value in expected.items():
        assert torch.equal(served[key].to(value.dtype), value), f"{key} is not PEFT's merge of the adapters"
    embeddings = SentenceTransformer(directory, device="cpu").encode(["w1 w2 w3", "w4 w5"], convert_to_tensor=True)
    assert embeddings.shape[0] == 2 and torch.isfinite(embeddings).all()


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
    """The marker must never outrun its adapter: an injected-LoRA resume refuses an unmarked checkpoint."""
    checkpoint = str(tmp_path / f"checkpoint-{SAVE_AT_STEP}")
    os.makedirs(checkpoint)

    def full_disk(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(embedding_module, "save_safetensors_file", full_disk)

    with pytest.raises(OSError, match="No space left"):
        EmbeddingTrainer._save_merged_checkpoint_resume_adapter(_host(_lora_model(run.base, seed=1)), checkpoint)
    assert resume_adapter_dir(checkpoint) is None


def test_the_final_export_is_a_serving_artifact_without_resume_state(run):
    final = str(run.root / "final")
    run.trainer.save_model(final)

    assert resume_adapter_dir(final) is None
    assert not os.path.exists(os.path.join(final, RESUME_ADAPTER_DIR))
    _assert_serves_the_fold(final, run.base, run.final, backbone_prefix(run.trainer.model))


@pytest.mark.parametrize("use_grouped_gemm", [True, False], ids=["grouped-gemm-default", "grouped-gemm-off"])
def test_the_marker_sends_the_policy_load_to_the_base(run, use_grouped_gemm):
    """Building from the fold and restoring the adapters would apply the delta twice."""
    assert classify_resume_weights(run.checkpoint) == "merged_adapter"
    model_config = SimpleNamespace(model_name_or_path=run.base)
    config = ParallelismConfig(use_grouped_gemm=use_grouped_gemm)
    assert resolve_resume_weights_source(run.checkpoint, model_config, config) == run.base


def test_resume_restores_the_adapters_bit_equal_and_reproduces_the_run(run, tmp_path):
    source = resolve_resume_weights_source(
        run.checkpoint, SimpleNamespace(model_name_or_path=run.base), ParallelismConfig()
    )
    model = _lora_model(source, seed=2)
    fresh = snapshot_trainable(model)
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
    final = snapshot_trainable(trainer.model)
    drifted = [key for key, value in run.final.items() if not torch.equal(final[key], value)]
    assert not drifted, f"final adapters differ from the uninterrupted run's: {drifted[:3]}"


def test_load_best_model_at_end_restores_the_best_checkpoints_adapters(run, tmp_path):
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
    live = snapshot_trainable(trainer.model)
    assert set(live) == set(saved)
    unequal = [key for key, value in saved.items() if not torch.equal(live[key], value)]
    assert not unequal, f"the live adapters are not the best checkpoint's: {unequal[:3]}"
    final = str(tmp_path / "final")
    trainer.save_model(final)
    _assert_serves_the_fold(final, run.base, saved, backbone_prefix(trainer.model))


@pytest.mark.parametrize("for_best_model", [False, True], ids=["resume", "best-model"])
def test_an_injected_lora_resume_restores_the_state_beside_its_weights(run, monkeypatch, for_best_model):
    """Skipped, a resumed run re-imbalances its router and drops its sidecar state."""
    restored = []
    monkeypatch.setattr(
        EmbeddingTrainer,
        "_restore_state_beside_weights",
        lambda self, checkpoint, *, for_best_model: restored.append((checkpoint, for_best_model)),
    )
    host = _host(_lora_model(run.base, seed=2))
    EmbeddingTrainer._load_from_checkpoint(host, run.checkpoint, for_best_model=for_best_model)
    assert restored == [(run.checkpoint, for_best_model)]


def test_a_marked_checkpoint_without_its_adapter_file_raises(run, tmp_path):
    checkpoint = _copy(run.checkpoint, tmp_path / f"checkpoint-{SAVE_AT_STEP}")
    os.remove(os.path.join(checkpoint, RESUME_ADAPTER_DIR, ADAPTER_SAFETENSORS_FILE))
    trainer = _trainer(_lora_model(run.base, seed=2), tmp_path / "out")

    with pytest.raises(RuntimeError, match="holds no adapter file"):
        trainer._load_from_checkpoint(checkpoint)


@pytest.mark.parametrize("built_from", ["resolver", "base"])
def test_an_unmarked_folded_checkpoint_refuses_an_injected_lora_run(run, tmp_path, built_from):
    """A torn save's layout: fresh adapters on the fold under restored optimizer moments would diverge silently.
    Built from the base too, where the adapter file a torn save left would otherwise restore unrefused."""
    checkpoint = _copy(run.checkpoint, tmp_path / f"checkpoint-{SAVE_AT_STEP}")
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
    before = snapshot_trainable(trainer.model)

    with pytest.raises(ValueError, match="delta would apply twice"):
        trainer._load_from_checkpoint(run.checkpoint)
    assert all(torch.equal(value, before[key]) for key, value in snapshot_trainable(trainer.model).items())


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
def test_a_resume_under_another_lora_scaling_is_refused(run, tmp_path, lora, monkeypatch):
    """Adapters would restore by name and shape with every delta rescaled; the shared read reads the file once."""
    reads = []
    real_read = peft_mod.read_adapter_file
    monkeypatch.setattr(peft_mod, "read_adapter_file", lambda path: reads.append(path) or real_read(path))
    trainer = _trainer(_lora_model(run.base, seed=2, **lora), tmp_path / "out")
    before = snapshot_trainable(trainer.model)

    with pytest.raises(ValueError, match="another LoRA scaling"):
        trainer._load_from_checkpoint(run.checkpoint)
    assert all(torch.equal(value, before[key]) for key, value in snapshot_trainable(trainer.model).items())
    assert len(reads) == 1, f"the resume adapter was read {len(reads)} times through the shared read"


def test_a_resume_adapter_without_its_recorded_scaling_is_refused(run, tmp_path):
    checkpoint = _copy(run.checkpoint, tmp_path / f"checkpoint-{SAVE_AT_STEP}")
    with open(os.path.join(checkpoint, RESUME_ADAPTER_DIR, ADAPTER_CONFIG_FILE)) as fh:
        assert json.load(fh)["lora_alpha"] == LORA_ALPHA, "premise: the save records the run's scaling"
    os.remove(os.path.join(checkpoint, RESUME_ADAPTER_DIR, ADAPTER_CONFIG_FILE))
    trainer = _trainer(_lora_model(run.base, seed=2), tmp_path / "out")

    with pytest.raises(ValueError, match="is missing"):
        trainer._load_from_checkpoint(checkpoint)


@pytest.mark.parametrize("save_only_model", [False, True], ids=["exact-resume", "save_only_model"])
def test_the_resume_adapter_is_on_disk_before_rotation(run, tmp_path, monkeypatch, save_only_model):
    """A preemption between rotation and the resume-adapter write would leave one checkpoint no resume accepts."""
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


def _node_checkpoint(root: str, rank: int) -> str:
    return os.path.join(root, f"node_{rank}", f"checkpoint-{SAVE_AT_STEP}")


def _node_worker(rank: int, root: str, base: str) -> None:
    problems: list[str] = []
    try:
        filesystem.resolve_shared_filesystem_consensus()  # what init_distributed does for a real run
        PartialState()
        if filesystem.is_output_shared_filesystem() or not filesystem.is_fs_save_rank():
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
        expected = snapshot_trainable(trained)
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
        if any(not torch.equal(value, expected[key]) for key, value in snapshot_trainable(fresh).items()):
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
    reset_shared_filesystem_consensus()


def test_each_node_writes_and_restores_its_own_resume_adapter(tmp_path):
    """A rank-0-only write leaves node 1 nothing to resume from; a copy missing on one node raises on both."""
    base = _tiny_base(tmp_path / "base")
    run_gloo_ranks(_node_worker, 2, str(tmp_path), base, pg_timeout=PG_TIMEOUT, env=TWO_NODE_ENV)

    for rank in range(2):
        result = (tmp_path / f"result_{rank}.txt").read_text()
        assert result == "PASS", f"rank {rank}: {result}"
    first = [(tmp_path / f"first_tensor_{rank}.txt").read_text() for rank in range(2)]
    assert first[0] != first[1], "premise: the two nodes hold different tensors"


@pytest.fixture(
    scope="module",
    # (lora_target_modules, other LoRA settings, adapted modules).
    params=[
        ((EMBEDDING_TARGET, *TARGET_MODULES), {}, 1 + NUM_LAYERS * len(TARGET_MODULES)),
        ((EMBEDDING_TARGET,), {}, 1),
        ((EMBEDDING_TARGET, *TARGET_MODULES), {"use_dora": True}, 1 + NUM_LAYERS * len(TARGET_MODULES)),
    ],
    ids=["embedding-and-attention", "embedding-only", "dora"],
)
def variant_run(request, tmp_path_factory):
    """PEFT's ``lora.Embedding`` holds ``lora_embedding_A``/``_B``, neither ``lora_A`` nor ``lora_B``.
    DoRA adds a ``lora_magnitude_vector`` its merge rescales by."""
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
        final=snapshot_trainable(trainer.model),
        prefix=backbone_prefix(trainer.model),
    )


def test_every_target_serves_peft_merge_of_its_adapters(variant_run):
    """Served under the plain names: PEFT's merge, ``(B @ A)ᵀ`` for the embedding and DoRA's rescale."""
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
    final = snapshot_trainable(trainer.model)
    assert all(torch.equal(final[key], value) for key, value in variant_run.final.items())


class _AttentionBackbone(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(8, 2)


def test_an_adapter_the_save_cannot_fold_is_refused_at_construction():
    """An ``nn.MultiheadAttention`` LoRA has no out-of-place fold.
    Refused on the save rank alone, at the first checkpoint, its peers would block in the save's collectives."""
    backbone = inject_adapter_in_model(LoraConfig(r=2, target_modules=["attn"]), _AttentionBackbone())

    with pytest.raises(ValueError, match="cannot be folded out of place"):
        lora_fold_targets(backbone)
    constructor = ast.parse(textwrap.dedent(inspect.getsource(EmbeddingTrainer.__init__)))
    called = {
        node.func.id
        for node in ast.walk(constructor)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
    }
    assert "lora_fold_targets" in called, "the trainer's construction no longer checks the fold"


class _NativeLora(torch.nn.Module):
    """jina-embeddings-v3's parametrization: base parameters named ``lora_A`` / ``lora_B`` no PEFT tuner layer owns."""

    def __init__(self, features: int):
        super().__init__()
        self.lora_A = torch.nn.Parameter(0.1 * torch.randn(2, features))
        self.lora_B = torch.nn.Parameter(0.1 * torch.randn(features, 2))

    def forward(self, weight):
        return weight + self.lora_B @ self.lora_A


def _with_native_lora(model: SentenceTransformer) -> list[str]:
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
    assert not has_injected_lora(model[0].auto_model)


def test_the_fold_keeps_a_backbones_own_lora_parameters_and_trains_only_the_injected_ones(run):
    model = SentenceTransformer(run.base, device="cpu")
    native = _with_native_lora(model)
    _lora_model(run.base, seed=1, model=model)
    backbone = model[0].auto_model

    written = {key for chunk in folded_backbone_chunks(backbone, retain=True) for key in chunk}

    assert has_injected_lora(backbone)
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


def test_a_trainable_parameter_outside_the_backbone_is_refused_under_fsdp2(run):
    """The save writes the head rank-locally and resume restores the backbone alone, so the head's training is lost."""
    model = SentenceTransformer(run.base, device="cpu")
    _with_dense_head(model)

    with pytest.raises(ValueError, match="outside the SentenceTransformer's transformer backbone"):
        reject_modules_outside_backbone(model, model[0].auto_model, saves_rank_locally=True)
    reject_modules_outside_backbone(model, model[0].auto_model, saves_rank_locally=False)


def test_a_frozen_head_passes_unless_fsdp2_shards_it(run):
    """A frozen head FSDP2 leaves whole is saved whole; a sharded one would be saved as this rank's shard."""
    model = SentenceTransformer(run.base, device="cpu")
    linear = _with_dense_head(model).linear
    linear.requires_grad_(False)
    reject_modules_outside_backbone(model, model[0].auto_model, saves_rank_locally=True)

    with fake_process_group_mesh(rank=0, world_size=2) as mesh:
        linear.weight = torch.nn.Parameter(
            distribute_tensor(linear.weight.data, mesh, [Shard(0)]), requires_grad=False
        )
        with pytest.raises(ValueError, match="that FSDP2 shards"):
            reject_modules_outside_backbone(model, model[0].auto_model, saves_rank_locally=True)


def test_a_save_over_a_marked_directory_leaves_no_stale_marker(run, tmp_path, monkeypatch):
    """A resumed run rewrites ``checkpoint-N`` in place.
    A marker left there would vouch for the old adapter beside the new weights, so every save removes it first."""
    checkpoint = tmp_path / "out" / f"checkpoint-{SAVE_AT_STEP}"
    _copy(run.checkpoint, checkpoint)
    _trainer(SentenceTransformer(run.base, device="cpu"), tmp_path / "out", max_steps=SAVE_AT_STEP).train()
    assert resume_adapter_dir(str(checkpoint)) is None
    assert classify_resume_weights(str(checkpoint)) == "full"

    torn = tmp_path / "torn" / f"checkpoint-{SAVE_AT_STEP}"
    _copy(run.checkpoint, torn)

    def full_disk(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(embedding_module, "save_safetensors_file", full_disk)
    with pytest.raises(OSError, match="No space left"):
        _trainer(_lora_model(run.base, seed=1), tmp_path / "torn", max_steps=SAVE_AT_STEP).train()
    assert resume_adapter_dir(str(torn)) is None


def test_a_marked_checkpoint_without_its_adapter_is_refused_before_the_policy_loads(run, tmp_path):
    checkpoint = _copy(run.checkpoint, tmp_path / f"checkpoint-{SAVE_AT_STEP}")
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

    assert classify_resume_weights(out) == "merged_adapter"
    source_adapter = os.path.join(run.checkpoint, RESUME_ADAPTER_DIR, ADAPTER_SAFETENSORS_FILE)
    with (
        open(source_adapter, "rb") as source,
        open(os.path.join(out, RESUME_ADAPTER_DIR, ADAPTER_SAFETENSORS_FILE), "rb") as copy_,
    ):
        assert source.read() == copy_.read(), "the resume adapter changed in the conversion"
    assert os.path.isfile(os.path.join(out, "modules.json")), "the ST pipeline config was dropped"
    model = _lora_model(run.base, seed=2)
    _trainer(model, tmp_path / "resumed")._load_from_checkpoint(out)
    assert all(torch.equal(value, run.at_save[key]) for key, value in snapshot_trainable(model).items())


def test_a_new_base_built_from_the_checkpoint_carries_none_of_its_resume_state(run, tmp_path, monkeypatch):
    """These are new models: resumed, the marker would rebuild them from the original base plus this run's adapter."""
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
        assert classify_resume_weights(str(out)) == "full"


def test_the_weight_loader_covers_the_names_a_full_fine_tune_saves(tmp_path, monkeypatch):
    """On the SentenceTransformer (``0.<module>.*``) the loader matches no saved name, and FSDP2/TP reloads refuse.
    The optimizer store keeps the SentenceTransformer, whose parameters the optimizer steps."""
    PartialState()
    base = _tiny_base(tmp_path / "base")
    _trainer(SentenceTransformer(base, device="cpu"), tmp_path / "out", max_steps=SAVE_AT_STEP).train()
    checkpoint = str(tmp_path / "out" / f"checkpoint-{SAVE_AT_STEP}")
    trainer = _trainer(SentenceTransformer(base, device="cpu"), tmp_path / "resumed")

    contexts = []
    monkeypatch.setattr(
        checkpointing_mod, "load_resume_weights", lambda ctx, *args, **kwargs: contexts.append(ctx) or True
    )
    trainer._load_from_checkpoint(checkpoint)
    loader_model = contexts[0].model
    covered, unmatched, _matched, _total = resume_numel_coverage(
        loader_model, resolve_checkpoint_weights(checkpoint).key_set()
    )

    assert covered and not unmatched, f"the loader's model misses the saved names: {sorted(unmatched)[:3]}"
    assert trainer._optimizer_store().ctx.model is trainer.model


def test_a_single_process_resume_restores_the_whole_pipeline(tmp_path):
    """A single process saves every module; the mixin's reload of the backbone would leave a head at its init."""
    PartialState()
    base = _tiny_base(tmp_path / "base")
    torch.manual_seed(1)
    trained = SentenceTransformer(base, device="cpu")
    head = _with_dense_head(trained).linear.weight
    _trainer(trained, tmp_path / "out", max_steps=SAVE_AT_STEP).train()
    checkpoint = str(tmp_path / "out" / f"checkpoint-{SAVE_AT_STEP}")
    torch.manual_seed(2)
    model = SentenceTransformer(base, device="cpu")
    _with_dense_head(model)
    resumed = _trainer(model, tmp_path / "resumed")

    resumed._load_from_checkpoint(checkpoint)

    assert torch.equal(resumed.model[-1].linear.weight, head.detach()), "the trained head was not restored"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
