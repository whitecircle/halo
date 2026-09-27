#!/usr/bin/env python
"""A ``merge_expert_lora_on_save`` checkpoint serves from its merged weights and resumes from its adapters.

The merged weights cannot resume a run: the bf16 fold loses part of the delta, and the adapter
optimizer moments restored onto fresh adapters (``lora_B = 0``) push the run off its trajectory. So
every merge-on-save training checkpoint also carries the unmerged adapter in ``resume_adapter/``,
written by the non-merged save's own writer, plus a root marker the resume classifies on. Pinned
through the real writers, classifier and loader:

* layout: ``resume_adapter/`` holds the adapter file and config the non-merged save writes, for the
  expert-only and the mixed shape; the marker sits at the root, and the root holds no
  ``adapter_config.json``, which would make ``from_pretrained`` apply the adapter over the merge again;
* classification: the marker, not the files beside it, sends the policy load to the base;
* loader: the adapters come from ``resume_adapter/`` onto a base-built model; a model built from the
  merged weights (the delta twice), a marked checkpoint missing its adapter, and an unmarked merged
  checkpoint under an adapter run (adapters from init under restored moments) each raise;
* the merged save undoes its in-place attention merge exactly, so the run it checkpoints is the run
  a resume reproduces;
* the final ``save_model`` export carries no resume adapter.

    python tests/cpu/checkpoint/test_merged_checkpoint_resume.py
"""

import json
import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM

import src.distributed.checkpoint.loader as loader_mod
import src.distributed.checkpoint.peft as peft_mod
import src.distributed.checkpoint.save as save_mod
import src.distributed.expert_parallel.saving as saving_mod
import src.trainers.mixins.checkpointing as checkpointing_mod
from src.checkpoint.adapters import EXPERT_LORA_PEFT_TYPE, MIXED_EXPERT_LORA_PEFT_TYPE
from src.checkpoint.config_export import LOADED_WEIGHTS_FROM_ATTR
from src.checkpoint.format import (
    ADAPTER_CONFIG_FILE,
    ADAPTER_SAFETENSORS_FILE,
    RESUME_ADAPTER_DIR,
    RESUME_ADAPTER_MARKER_FILE,
    SAFETENSORS_WEIGHTS_FILE,
    resume_adapter_dir,
)
from src.distributed.checkpoint.context import CheckpointContext, CheckpointLoadContext
from src.distributed.checkpoint.loader import CheckpointLoader
from src.distributed.checkpoint.peft import restore_adapters
from src.distributed.checkpoint.save import save_resume_adapter
from src.distributed.expert_parallel.config import ExpertLoraSpec
from src.models.structure import merged_adapters
from src.trainers.mixins.base import DistributedTrainerMixin
from src.training.environment import _classify_resume_checkpoint, resolve_resume_weights_source
from tests.common.ep_stubs import StubEPLayerBase
from tests.common.parallelism import make_parallelism_config

# The resolver logs through the accelerate logger, which requires an initialized state.
PartialState()

BASE = "org/base-model"
_EP = make_parallelism_config(world_size=8, gpus_per_node=8, ep_size=8, use_grouped_gemm=True)
SPEC = ExpertLoraSpec(r=8, alpha=16.0, projections=frozenset({"gate", "up", "down"}))
# Values off the bf16 grid would compare against their own rounding; these sit on it, so the file
# and the restored adapters can be compared bit for bit.
EXPERT_STATE = {
    "model.layers.0.mlp.experts.gate_up_proj.lora_A": torch.arange(2 * 4 * 8, dtype=torch.float32).reshape(2, 4, 8)
    / 64,
    "model.layers.0.mlp.experts.gate_up_proj.lora_B": -torch.arange(2 * 8 * 6, dtype=torch.float32).reshape(2, 8, 6)
    / 32,
}


class _Recorder:
    def __init__(self, result=None):
        self.calls = []
        self.result = result

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result


class _TokenizerSpy:
    """Stands in for the run's tokenizer; the resume adapter must not carry a copy of it."""

    def __init__(self):
        self.saved_to = []

    def save_pretrained(self, path):
        self.saved_to.append(path)


class _ExpertLoraLayer(StubEPLayerBase):
    """An EP layer carrying native grouped expert LoRA, as an expert-only adapter run builds it."""

    def __init__(self):
        super().__init__()
        self._expert_lora_attrs = frozenset({"gate_up_proj"})


class _ExpertOnlyModel(nn.Module):
    """An expert-only run's model: no PeftModel, a config naming the base."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4, bias=False)
        self.config = SimpleNamespace(_name_or_path=BASE)


def _tiny_peft_model(
    seed: int,
    *,
    dtype: torch.dtype = torch.float32,
    tie_word_embeddings: bool = False,
    target_modules: tuple[str, ...] = ("q_proj", "v_proj"),
    lora_alpha: int = 8,
) -> PeftModel:
    config = AutoConfig.for_model(
        "qwen3",
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=tie_word_embeddings,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, dtype=dtype)
    torch.manual_seed(seed)
    return get_peft_model(model, LoraConfig(r=4, lora_alpha=lora_alpha, target_modules=list(target_modules)))


def _save_context(model, *, tokenizer=None) -> CheckpointContext:
    return CheckpointContext(
        model=model,
        parallelism_config=SimpleNamespace(expert_lora=SPEC),
        is_pp_mode=False,
        is_cp_mode=False,
        is_tp_mode=False,
        is_ep_tp_mode=False,
        has_ep_layers=True,
        fsdp_wrapped=True,
        accelerate_manages_fsdp=False,
        is_save_rank=True,
        max_shard_size="5GB",
        save_sharded_ep=False,
        has_expert_lora=True,
        merge_expert_lora_on_save=True,
        cp_wrapper=None,
        tokenizer=tokenizer,
    )


def _merged_checkpoint(path, *, marked: bool) -> str:
    """A checkpoint holding merged weights (a stand-in state dict), marked or not."""
    os.makedirs(path, exist_ok=True)
    save_file({"fc.weight": torch.ones(4, 4)}, os.path.join(path, SAFETENSORS_WEIGHTS_FILE))
    if marked:
        with open(os.path.join(path, RESUME_ADAPTER_MARKER_FILE), "w") as fh:
            json.dump({"adapter_dir": RESUME_ADAPTER_DIR}, fh)
    return str(path)


def _load_context(model) -> CheckpointLoadContext:
    def unreachable(*args, **kwargs):
        raise AssertionError("the EP/CP skip path must not fall through to the base Trainer loader")

    return CheckpointLoadContext(
        model=model,
        optimizer=None,
        lr_scheduler=None,
        parallelism_config=None,
        is_pp_mode=False,
        is_cp_mode=False,
        is_tp_mode=False,
        has_ep_layers=True,
        fsdp_wrapped=True,
        tp_rank=0,
        tp_size=1,
        super_load_from_checkpoint=unreachable,
        super_load_optimizer_and_scheduler=unreachable,
    )


def _built_from(model: nn.Module, source: str) -> nn.Module:
    """Stamp where ``model``'s weights were read, as ``load_distributed_model`` does."""
    setattr(model, LOADED_WEIGHTS_FROM_ATTR, source)
    return model


# --- layout ---------------------------------------------------------------------------------


def test_expert_only_resume_adapter_is_the_standalone_adapter_save(tmp_path, monkeypatch):
    """Expert-only runs have no PeftModel: the resume adapter is ``save_ep_lora_adapters``' artifact."""
    monkeypatch.setattr(saving_mod, "gather_ep_lora_adapters", lambda model, retain: dict(EXPERT_STATE))
    checkpoint = _merged_checkpoint(tmp_path / "checkpoint-3", marked=False)

    save_resume_adapter(_save_context(_ExpertOnlyModel()), checkpoint)

    adapter_dir = os.path.join(checkpoint, RESUME_ADAPTER_DIR)
    assert resume_adapter_dir(checkpoint) == adapter_dir, "the save must write the marker the resume reads"
    saved = load_file(os.path.join(adapter_dir, ADAPTER_SAFETENSORS_FILE))
    assert set(saved) == set(EXPERT_STATE)
    for key, value in EXPERT_STATE.items():
        assert torch.equal(saved[key].float(), value), f"{key} changed on its way to the resume adapter"
    with open(os.path.join(adapter_dir, ADAPTER_CONFIG_FILE)) as fh:
        config = json.load(fh)
    assert config["peft_type"] == EXPERT_LORA_PEFT_TYPE
    assert (config["r"], config["lora_alpha"]) == (SPEC.r, SPEC.alpha), "the scaling fields must travel"
    assert not os.path.exists(os.path.join(checkpoint, ADAPTER_CONFIG_FILE)), (
        "an adapter_config.json at the root makes from_pretrained apply the adapter over the merged weights"
    )
    assert not os.path.exists(os.path.join(checkpoint, ADAPTER_SAFETENSORS_FILE))


def test_a_failed_adapter_write_leaves_no_marker(tmp_path, monkeypatch):
    """The marker is the resume's verdict, so it must never outrun its adapter: a checkpoint whose
    adapter write failed has to stay unmarked, where the loader refuses it as a merged checkpoint
    without its adapter, rather than marked and missing the file it promises."""
    monkeypatch.setattr(saving_mod, "gather_ep_lora_adapters", lambda model, retain: dict(EXPERT_STATE))

    def full_disk(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(saving_mod, "save_file", full_disk)
    checkpoint = _merged_checkpoint(tmp_path / "checkpoint-3", marked=False)

    with pytest.raises(OSError, match="No space left"):
        save_resume_adapter(_save_context(_ExpertOnlyModel()), checkpoint)
    assert resume_adapter_dir(checkpoint) is None


class _StepSave:
    """Stands in for HF's ``Trainer._save_checkpoint``: rewrites the step's weights and trainer state
    in place, as a resumed run saving a step it already saved does."""

    def _save_checkpoint(self, model, trial):
        checkpoint = os.path.join(self.run_dir, f"checkpoint-{self.state.global_step}")
        _merged_checkpoint(checkpoint, marked=False)
        with open(os.path.join(checkpoint, "trainer_state.json"), "w") as fh:
            json.dump({"global_step": self.state.global_step, "trajectory": self.trajectory}, fh)


class _RunTrainer(DistributedTrainerMixin, _StepSave):
    """The real ``_save_checkpoint`` over a stub base save, merge-on-save or not."""

    def __init__(self, run_dir, trajectory: str, *, merge_expert_lora_on_save: bool = True):
        self.run_dir = run_dir
        self.trajectory = trajectory
        self.args = SimpleNamespace(save_total_limit=None, save_only_model=True, should_save=True)
        self.state = SimpleNamespace(global_step=3, best_model_checkpoint=None)
        self.parallelism_config = SimpleNamespace(
            is_tp_mode=False, merge_expert_lora_on_save=merge_expert_lora_on_save
        )
        self._fsdp_wrapped = True
        self.lr_scheduler = None

    def _get_output_dir(self, trial=None):
        return self.run_dir

    def _checkpoint_context(self):
        return _save_context(_ExpertOnlyModel())


def _abandoned_merged_step(run_dir: str, monkeypatch) -> str:
    """checkpoint-3 as a completed merge-on-save of a trajectory a resume then abandoned."""
    monkeypatch.setattr(saving_mod, "gather_ep_lora_adapters", lambda model, retain: dict(EXPERT_STATE))
    _RunTrainer(run_dir, "abandoned")._save_checkpoint(model=None, trial=None)
    checkpoint = os.path.join(run_dir, "checkpoint-3")
    assert _classify_resume_checkpoint(checkpoint) == "merged_adapter", "premise: the first save completed"
    return checkpoint


def _assert_resaved_unmarked(checkpoint: str) -> None:
    with open(os.path.join(checkpoint, "trainer_state.json")) as fh:
        assert json.load(fh)["trajectory"] == "resumed", "premise: the re-save rewrote the step's state"
    assert resume_adapter_dir(checkpoint) is None, "the abandoned run's marker survived the re-save"
    assert _classify_resume_checkpoint(checkpoint) == "full"


def test_a_resave_torn_before_its_adapter_leaves_the_step_unmarked(tmp_path, monkeypatch):
    """A run resumed from checkpoint-1 reaches step 3 again and saves over the checkpoint-3 an
    abandoned trajectory wrote. If that save stops after the new weights and trainer state but before
    the new adapter, the old marker must not survive: it would resume the abandoned run's adapter
    beside this run's state. Unmarked, the loader refuses the checkpoint as a merged one without its
    adapter instead."""
    checkpoint = _abandoned_merged_step(str(tmp_path), monkeypatch)

    def full_disk(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(saving_mod, "save_file", full_disk)
    with pytest.raises(OSError, match="No space left"):
        _RunTrainer(str(tmp_path), "resumed")._save_checkpoint(model=None, trial=None)

    _assert_resaved_unmarked(checkpoint)


def test_a_resave_by_a_run_that_writes_no_marker_removes_the_old_one(tmp_path, monkeypatch):
    """The marker left in a directory a save rewrites is stale whatever the saving run is: a resume
    of the same output_dir without merge_expert_lora_on_save writes no marker of its own, and the
    old one would send its resume to the abandoned run's adapter over its base."""
    checkpoint = _abandoned_merged_step(str(tmp_path), monkeypatch)

    _RunTrainer(str(tmp_path), "resumed", merge_expert_lora_on_save=False)._save_checkpoint(model=None, trial=None)

    _assert_resaved_unmarked(checkpoint)


def test_mixed_resume_adapter_round_trips_through_the_adapter_restore(tmp_path, monkeypatch):
    """A mixed run's resume adapter is ``PeftAdapterSaver``'s mixed artifact, and the restore reads it
    back bit-equal onto fresh adapters — the resume that merge-on-save checkpoints could not do."""
    monkeypatch.setattr(peft_mod, "gather_ep_lora_adapters", lambda model, retain: dict(EXPERT_STATE))
    trained = _tiny_peft_model(seed=1)
    with torch.no_grad():
        for name, param in trained.named_parameters():
            if ".lora_" in name:
                # On the bf16 grid, so the cast the adapter file applies is exact.
                param.copy_(torch.randn_like(param).to(torch.bfloat16).float())
    tokenizer = _TokenizerSpy()
    checkpoint = _merged_checkpoint(tmp_path / "checkpoint-3", marked=False)

    save_resume_adapter(_save_context(trained, tokenizer=tokenizer), checkpoint)

    adapter_dir = resume_adapter_dir(checkpoint)
    assert adapter_dir == os.path.join(checkpoint, RESUME_ADAPTER_DIR)
    with open(os.path.join(adapter_dir, ADAPTER_CONFIG_FILE)) as fh:
        assert json.load(fh)["peft_type"] == MIXED_EXPERT_LORA_PEFT_TYPE
    saved = load_file(os.path.join(adapter_dir, ADAPTER_SAFETENSORS_FILE))
    assert set(EXPERT_STATE) < set(saved), "the expert half is missing from the mixed resume adapter"
    assert not tokenizer.saved_to, "the tokenizer belongs to the checkpoint root, not the resume adapter"
    assert not os.path.exists(os.path.join(checkpoint, ADAPTER_CONFIG_FILE))

    fresh = _tiny_peft_model(seed=2)
    applied = _Recorder()
    monkeypatch.setattr(peft_mod, "apply_ep_lora_adapters", applied)
    assert restore_adapters(adapter_dir, fresh, is_cp_mode=False) is not None

    live = {name: param for name, param in trained.named_parameters() if ".lora_" in name}
    restored = dict(fresh.named_parameters())
    mismatched = [name for name, param in live.items() if not torch.equal(restored[name], param)]
    assert live and not mismatched, f"attention adapters not restored bit-equal: {mismatched[:3]}"
    (_model, expert_state), _ = applied.calls[0]
    assert set(expert_state) == set(EXPERT_STATE)
    assert all(torch.equal(expert_state[key].float(), value) for key, value in EXPERT_STATE.items())


# --- classification -------------------------------------------------------------------------


def test_the_marker_not_the_files_classifies_a_merged_checkpoint(tmp_path):
    """Merged weights are as loadable as a full fine-tune's, and the adapter directory beside them
    proves nothing about completeness: only the marker, written last, makes the checkpoint resume
    from the base."""
    checkpoint = _merged_checkpoint(tmp_path / "checkpoint-3", marked=False)
    os.makedirs(os.path.join(checkpoint, RESUME_ADAPTER_DIR))
    save_file(dict(EXPERT_STATE), os.path.join(checkpoint, RESUME_ADAPTER_DIR, ADAPTER_SAFETENSORS_FILE))
    assert _classify_resume_checkpoint(checkpoint) == "full"

    _merged_checkpoint(checkpoint, marked=True)
    assert _classify_resume_checkpoint(checkpoint) == "merged_adapter"


def test_a_marked_checkpoint_resumes_the_policy_from_the_base(tmp_path):
    """Building the policy from the merged weights and then restoring the adapter onto it would apply
    the delta twice; the base is what the resume adapter was trained on."""
    checkpoint = _merged_checkpoint(tmp_path / "checkpoint-3", marked=True)
    os.makedirs(os.path.join(checkpoint, RESUME_ADAPTER_DIR))
    save_file(dict(EXPERT_STATE), os.path.join(checkpoint, RESUME_ADAPTER_DIR, ADAPTER_SAFETENSORS_FILE))
    assert resolve_resume_weights_source(checkpoint, SimpleNamespace(model_name_or_path=BASE), _EP) == BASE


def test_a_marked_checkpoint_without_its_adapter_refuses_before_the_policy_loads(tmp_path):
    """The loader would refuse it too, but only after the whole base was built; the resolver sees the
    empty adapter directory first."""
    checkpoint = _merged_checkpoint(tmp_path / "checkpoint-3", marked=True)
    os.makedirs(os.path.join(checkpoint, RESUME_ADAPTER_DIR))

    with pytest.raises(ValueError, match="holds no adapter file"):
        resolve_resume_weights_source(checkpoint, SimpleNamespace(model_name_or_path=BASE), _EP)


# --- loader ---------------------------------------------------------------------------------


def test_a_marked_checkpoint_restores_its_resume_adapter_onto_the_base(tmp_path, monkeypatch):
    checkpoint = _merged_checkpoint(tmp_path / "checkpoint-3", marked=True)
    restored = _Recorder(result="adapter file")
    monkeypatch.setattr(loader_mod, "restore_adapters", restored)
    model = _built_from(nn.Linear(4, 4), BASE)

    CheckpointLoader(_load_context(model)).load_model(checkpoint, model)

    assert len(restored.calls) == 1
    (source, _model), _ = restored.calls[0]
    assert source == os.path.join(checkpoint, RESUME_ADAPTER_DIR), "the adapters must come from the resume adapter"


def test_a_marked_checkpoint_refuses_a_model_built_from_its_merged_weights(tmp_path, monkeypatch):
    checkpoint = _merged_checkpoint(tmp_path / "checkpoint-3", marked=True)
    restored = _Recorder(result="adapter file")
    monkeypatch.setattr(loader_mod, "restore_adapters", restored)
    model = _built_from(nn.Linear(4, 4), checkpoint)

    with pytest.raises(ValueError, match="delta would apply twice"):
        CheckpointLoader(_load_context(model)).load_model(checkpoint, model)
    assert not restored.calls


def test_a_marked_checkpoint_without_its_adapter_raises(tmp_path):
    """The real restore finds no adapter file under the marker's directory: the adapters would
    resume from init under restored moments, the failure the marker exists to prevent."""
    checkpoint = _merged_checkpoint(tmp_path / "checkpoint-3", marked=True)
    model = _built_from(nn.Linear(4, 4), BASE)

    with pytest.raises(RuntimeError, match="holds no adapter file"):
        CheckpointLoader(_load_context(model)).load_model(checkpoint, model)


def test_an_unmarked_merged_checkpoint_refuses_an_adapter_run(tmp_path):
    """Merged weights, no marker, and a run that trains adapters: the save was torn before its resume
    adapter, or never wrote one. Building from the merged weights with fresh adapters and then
    restoring their optimizer moments is the silent divergence; it must raise."""
    checkpoint = _merged_checkpoint(tmp_path / "checkpoint-3", marked=False)
    model = _tiny_peft_model(seed=1)
    _built_from(model.get_base_model(), checkpoint)

    with pytest.raises(ValueError, match="without its resume adapter"):
        CheckpointLoader(_load_context(model)).load_model(checkpoint, model)


def test_an_unmarked_merged_checkpoint_refuses_an_expert_only_adapter_run(tmp_path):
    """The same torn save under an expert-only run: its adapters live on the EP layers and no
    PeftModel is in the tree, so a PEFT lookup alone would read it as a full fine-tune."""
    checkpoint = _merged_checkpoint(tmp_path / "checkpoint-3", marked=False)
    model = _built_from(nn.Sequential(_ExpertLoraLayer()), checkpoint)

    with pytest.raises(ValueError, match="without its resume adapter"):
        CheckpointLoader(_load_context(model)).load_model(checkpoint, model)


def test_an_unmarked_full_checkpoint_still_resumes_a_full_fine_tune(tmp_path):
    """The other half: a full fine-tune built from its checkpoint trains no adapters, so the same
    layout resumes as before."""
    checkpoint = _merged_checkpoint(tmp_path / "checkpoint-3", marked=False)
    model = _built_from(nn.Linear(4, 4), checkpoint)

    CheckpointLoader(_load_context(model)).load_model(checkpoint, model)


# --- the merged save leaves the run untouched ------------------------------------------------


def _bf16_lora_linear() -> tuple[PeftModel, nn.Parameter]:
    """A bf16 LoRA'd linear whose adapter is large enough that a plain merge and unmerge moves the
    base weight, and that base weight."""
    torch.manual_seed(0)
    model = get_peft_model(
        nn.Sequential(nn.Linear(64, 64, bias=False)).to(torch.bfloat16),
        LoraConfig(r=4, lora_alpha=64, target_modules=["0"]),
    )
    with torch.no_grad():
        for name, param in model.named_parameters():
            if ".lora_" in name:
                param.normal_(std=0.5)
    return model, next(param for name, param in model.named_parameters() if ".base_layer." in name)


def test_the_merged_save_undoes_its_merge_exactly():
    """The save folds the attention adapters into their bf16 base weights in place and unfolds them
    after the write. ``(w + d) - d`` is not ``w`` in bf16, so without ``restore_base`` every merged
    save moves the frozen base of the run it checkpoints, and no resume of that checkpoint can
    reproduce the run."""
    model, base = _bf16_lora_linear()
    original = base.detach().clone()

    with merged_adapters(model):
        pass
    plain_unmerge_drifts = not torch.equal(base, original)
    with torch.no_grad():
        base.copy_(original)

    with merged_adapters(model, restore_base=True) as merged:
        folded = merged and not torch.equal(base, original)
    assert folded, "the body must see the merged weights"
    assert torch.equal(base, original), "the merged save moved the frozen base"
    assert plain_unmerge_drifts, "premise: a bf16 unmerge alone does not reverse the merge"


def test_the_merged_checkpoint_save_restores_the_base(tmp_path, monkeypatch):
    """``save_ep_checkpoint`` itself holds the merge open over the merged write, so it is the call that
    has to ask for the exact unfold; the premise that the plain one drifts is the test above's."""
    model, base = _bf16_lora_linear()
    original = base.detach().clone()
    written = []

    def merged_write(model, output_dir, **kwargs):
        written.append((kwargs["adapters_merged"], not torch.equal(base, original)))

    monkeypatch.setattr(save_mod, "save_ep_model", merged_write)
    save_mod.save_ep_checkpoint(_save_context(model), str(tmp_path / "checkpoint-1"))

    assert written == [(True, True)], "premise: the merged write sees the folded base"
    assert torch.equal(base, original), "the merged checkpoint save moved the frozen base"


def test_the_exact_unmerge_reaches_a_tied_base_weight():
    """A LoRA'd ``lm_head`` tied to ``embed_tokens`` is one tensor, which ``named_parameters()`` lists
    once, under the embedding's name rather than the ``.base_layer.`` one. A restore that walked the
    de-duplicated names would leave that weight to the bf16 unmerge."""
    model = _tiny_peft_model(
        seed=1, dtype=torch.bfloat16, tie_word_embeddings=True, target_modules=("lm_head", "q_proj"), lora_alpha=64
    )
    with torch.no_grad():
        for name, param in model.named_parameters():
            if ".lora_" in name:
                param.normal_(std=0.5)
    tied = model.base_model.model.lm_head.base_layer.weight
    assert tied is model.base_model.model.model.embed_tokens.weight, "premise: the head is tied"
    listed = [name for name, _ in model.named_parameters() if "lm_head.base_layer" in name]
    assert not listed, "premise: the tied weight is listed once, as the embedding"
    original = tied.detach().clone()

    for _ in range(20):
        with merged_adapters(model):
            pass
    plain_unmerge_drifts = not torch.equal(tied, original)
    with torch.no_grad():
        tied.copy_(original)
    for _ in range(20):
        with merged_adapters(model, restore_base=True):
            pass

    assert torch.equal(tied, original), "the tied base weight drifted through the merged saves"
    assert plain_unmerge_drifts, "premise: a bf16 unmerge alone moves the tied weight"


# --- the final export -----------------------------------------------------------------------


class _SaveModelTrainer(DistributedTrainerMixin):
    """The real ``save_model`` over a merge-on-save context, its state factories stubbed."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.parallelism_config = SimpleNamespace(is_pp_mode=False, merge_expert_lora_on_save=True)

    def _top_level_model(self):
        return self.ctx.model

    def _checkpoint_context(self):
        return self.ctx


def test_the_final_export_carries_no_resume_adapter(tmp_path, monkeypatch):
    """``save_model`` writes the serving artifact; the resume adapter is a checkpoint sidecar, which
    ``_save_checkpoint`` adds. Nothing resumes from the final export (no trainer or optimizer state),
    so an adapter copy there would only grow what gets served and pushed. The real saver ladder runs,
    down to the merged EP write, so a resume-adapter write moved into any of it is caught."""
    merged_writes, resume_writes = _Recorder(), _Recorder()
    monkeypatch.setattr(save_mod, "save_ep_model", merged_writes)
    monkeypatch.setattr(save_mod, "save_resume_adapter", resume_writes)
    monkeypatch.setattr(checkpointing_mod, "save_resume_adapter", resume_writes)

    _SaveModelTrainer(_save_context(_ExpertOnlyModel())).save_model(str(tmp_path / "final"))

    assert merged_writes.calls and merged_writes.calls[0][1]["merge_lora"], "premise: the merged save ran"
    assert not resume_writes.calls
    assert resume_adapter_dir(str(tmp_path / "final")) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
