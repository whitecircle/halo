#!/usr/bin/env python
"""A ``merge_expert_lora_on_save`` checkpoint serves from merged weights and resumes from its adapters in
``resume_adapter/``. The bf16 fold loses part of the delta, so merged weights cannot resume a run."""

import copy
import json
import os
import tempfile
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file, save_file
from transformers import AutoConfig, AutoModelForCausalLM, TrainerState

import halo.checkpoint.parallel.adapters as peft_mod
import halo.checkpoint.parallel.ep as saving_mod
import halo.checkpoint.parallel.resume_weights as loader_mod
import halo.checkpoint.parallel.save as save_mod
import halo.trainers.checkpointing as checkpointing_mod
from halo.checkpoint.adapters import EXPERT_LORA_PEFT_TYPE, MIXED_EXPERT_LORA_PEFT_TYPE, resume_adapter_dir
from halo.checkpoint.format import (
    ADAPTER_CONFIG_FILE,
    ADAPTER_SAFETENSORS_FILE,
    RESUME_ADAPTER_DIR,
    SAFETENSORS_WEIGHTS_FILE,
)
from halo.checkpoint.parallel.adapters import restore_adapters, save_resume_adapter
from halo.checkpoint.parallel.context import CheckpointLoadContext
from halo.checkpoint.parallel.resume_source import classify_resume_weights, resolve_resume_weights_source
from halo.checkpoint.parallel.resume_weights import load_resume_weights
from halo.checkpoint.reading import load_full_state_dict
from halo.checkpoint.weights_source import LOADED_WEIGHTS_FROM_ATTR
from halo.models.moe.expert_lora_spec import ExpertLoraSpec
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from tests.common.adapters import randomize_adapters
from tests.common.base_save import BaseTrainerSave
from tests.common.checkpoint_io import load_context, mark_resume_adapter, save_context
from tests.common.ep_stubs import StubEPLayerBase
from tests.common.parallelism import make_parallelism_config

BASE = "org/base-model"
_EP = make_parallelism_config(world_size=8, gpus_per_node=8, ep_size=8, use_grouped_gemm=True)
SPEC = ExpertLoraSpec(r=8, alpha=16.0, projections=frozenset({"gate", "up", "down"}))
# On the bf16 grid, so the file and the restored adapters compare bit for bit.
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
    """The resume adapter must not carry a copy of the tokenizer."""

    def __init__(self):
        self.saved_to = []

    def save_pretrained(self, path):
        self.saved_to.append(path)


class _ExpertLoraLayer(StubEPLayerBase):
    def __init__(self, spec: ExpertLoraSpec | None = None):
        super().__init__()
        self._expert_lora_attrs = frozenset({"gate_up_proj"})
        self.ep_config = SimpleNamespace(expert_lora=spec)


class _ExpertOnlyModel(nn.Module):
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


def _save_context(model, *, tokenizer=None):
    config = make_parallelism_config(world_size=1, gpus_per_node=1, expert_lora=SPEC, merge_expert_lora_on_save=True)
    return save_context(
        model,
        config,
        has_ep_layers=True,
        toolkit_fsdp2=True,
        is_save_rank=True,
        has_expert_lora=True,
        tokenizer=tokenizer,
    )


def _merged_checkpoint(path, *, marked: bool) -> str:
    os.makedirs(path, exist_ok=True)
    save_file({"fc.weight": torch.ones(4, 4)}, os.path.join(path, SAFETENSORS_WEIGHTS_FILE))
    if marked:
        mark_resume_adapter(str(path))
    return str(path)


def _load_context(model) -> CheckpointLoadContext:
    return load_context(model, has_ep_layers=True, fsdp_wrapped=True)


def _built_from(model: nn.Module, source: str) -> nn.Module:
    """As ``load_distributed_model`` does."""
    setattr(model, LOADED_WEIGHTS_FROM_ATTR, source)
    return model


def test_expert_only_resume_adapter_is_the_standalone_adapter_save(tmp_path, monkeypatch):
    """Expert-only runs have no PeftModel."""
    monkeypatch.setattr(peft_mod, "gather_expert_lora_adapters", lambda model, retain: dict(EXPERT_STATE))
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
    """The marker is the resume's verdict, so it must never outrun its adapter."""
    monkeypatch.setattr(peft_mod, "gather_expert_lora_adapters", lambda model, retain: dict(EXPERT_STATE))

    def full_disk(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(peft_mod, "save_safetensors_file", full_disk)
    checkpoint = _merged_checkpoint(tmp_path / "checkpoint-3", marked=False)

    with pytest.raises(OSError, match="No space left"):
        save_resume_adapter(_save_context(_ExpertOnlyModel()), checkpoint)
    assert resume_adapter_dir(checkpoint) is None


class _RunTrainer(DistributedTrainerMixin, BaseTrainerSave):
    """The base save rewrites the step in place, as a resumed run saving a step it already saved does. A save's
    trajectory is its trainer state's log history."""

    def __init__(self, run_dir, trajectory: str, *, merge_expert_lora_on_save: bool = True):
        self.run_dir = run_dir
        self.args = SimpleNamespace(
            save_total_limit=None,
            save_only_model=True,
            should_save=True,
            push_to_hub=False,
            include_num_input_tokens_seen="no",
        )
        self.state = TrainerState(global_step=3, log_history=[{"trajectory": trajectory}])
        self.parallelism_config = SimpleNamespace(
            is_tp_mode=False, merge_expert_lora_on_save=merge_expert_lora_on_save
        )
        self._fsdp_wrapped = True
        self.lr_scheduler = None

    def save_model(self, output_dir=None, _internal_call=False):
        _merged_checkpoint(output_dir, marked=False)
        self._mark_model_save_collectives_done()

    def _checkpoint_context(self):
        return _save_context(_ExpertOnlyModel())


def _abandoned_merged_step(run_dir: str, monkeypatch) -> str:
    monkeypatch.setattr(peft_mod, "gather_expert_lora_adapters", lambda model, retain: dict(EXPERT_STATE))
    _RunTrainer(run_dir, "abandoned")._save_checkpoint(model=None, trial=None)
    checkpoint = os.path.join(run_dir, "checkpoint-3")
    assert classify_resume_weights(checkpoint) == "merged_adapter", "premise: the first save completed"
    return checkpoint


def _assert_resaved_unmarked(checkpoint: str, *, completed: bool) -> None:
    """A stopped re-save keeps trainer state withheld, which makes resume detection pass the step over."""
    state_file = "trainer_state.json" if completed else ".trainer_state.json.uncommitted"
    with open(os.path.join(checkpoint, state_file)) as fh:
        assert json.load(fh)["log_history"] == [{"trajectory": "resumed"}], (
            "premise: the re-save rewrote the step's state"
        )
    if not completed:
        assert not os.path.exists(os.path.join(checkpoint, "trainer_state.json")), (
            "the stopped re-save publishes a trainer state that vouches for it"
        )
    assert resume_adapter_dir(checkpoint) is None, "the abandoned run's marker survived the re-save"
    assert classify_resume_weights(checkpoint) == "full"


def test_a_resave_torn_before_its_adapter_leaves_the_step_unmarked(tmp_path, monkeypatch):
    """The old marker would resume the abandoned run's adapter beside this run's state."""
    checkpoint = _abandoned_merged_step(str(tmp_path), monkeypatch)

    def full_disk(*args, **kwargs):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(peft_mod, "save_safetensors_file", full_disk)
    with pytest.raises(OSError, match="No space left"):
        _RunTrainer(str(tmp_path), "resumed")._save_checkpoint(model=None, trial=None)

    _assert_resaved_unmarked(checkpoint, completed=False)


def test_a_resave_by_a_run_that_writes_no_marker_removes_the_old_one(tmp_path, monkeypatch):
    """The stale marker would send a resume without merge_expert_lora_on_save to the abandoned run's adapter."""
    checkpoint = _abandoned_merged_step(str(tmp_path), monkeypatch)

    _RunTrainer(str(tmp_path), "resumed", merge_expert_lora_on_save=False)._save_checkpoint(model=None, trial=None)

    _assert_resaved_unmarked(checkpoint, completed=True)


def test_mixed_resume_adapter_round_trips_through_the_adapter_restore(tmp_path, monkeypatch):
    monkeypatch.setattr(peft_mod, "gather_expert_lora_adapters", lambda model, retain: dict(EXPERT_STATE))
    trained = _tiny_peft_model(seed=1)
    randomize_adapters(trained, dtype=torch.bfloat16)  # On the bf16 grid, so the adapter file's cast is exact.
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
    monkeypatch.setattr(peft_mod, "load_expert_lora_adapters", applied)
    assert restore_adapters(adapter_dir, fresh, is_cp_mode=False) is not None

    live = {name: param for name, param in trained.named_parameters() if ".lora_" in name}
    restored = dict(fresh.named_parameters())
    mismatched = [name for name, param in live.items() if not torch.equal(restored[name], param)]
    assert live and not mismatched, f"attention adapters not restored bit-equal: {mismatched[:3]}"
    (_model, expert_state), _ = applied.calls[0]
    assert set(expert_state) == set(EXPERT_STATE)
    assert all(torch.equal(expert_state[key].float(), value) for key, value in EXPERT_STATE.items())


def test_the_adapter_restore_refuses_another_attention_lora_scaling(tmp_path):
    """Names and shapes match, so without the recorded scaling the restore would rescale every delta."""
    adapter_dir = str(tmp_path / "adapter")
    _tiny_peft_model(seed=1, lora_alpha=8).save_pretrained(adapter_dir)

    with pytest.raises(ValueError, match=r"lora_alpha: saved 8, live 16\).* Resume with the lora_alpha it was"):
        restore_adapters(adapter_dir, _tiny_peft_model(seed=2, lora_alpha=16), is_cp_mode=False)
    assert restore_adapters(adapter_dir, _tiny_peft_model(seed=2, lora_alpha=8), is_cp_mode=False) is not None
    os.remove(os.path.join(adapter_dir, ADAPTER_CONFIG_FILE))
    with pytest.raises(ValueError, match="is missing"):
        restore_adapters(adapter_dir, _tiny_peft_model(seed=2, lora_alpha=8), is_cp_mode=False)


class _ExpertLoraModel(_ExpertOnlyModel):
    def __init__(self, spec: ExpertLoraSpec):
        super().__init__()
        self.experts = _ExpertLoraLayer(spec)


def test_the_adapter_restore_refuses_another_expert_lora_scaling(tmp_path, monkeypatch):
    monkeypatch.setattr(peft_mod, "gather_expert_lora_adapters", lambda model, retain: dict(EXPERT_STATE))
    applied = _Recorder()
    monkeypatch.setattr(peft_mod, "load_expert_lora_adapters", applied)
    checkpoint = _merged_checkpoint(tmp_path / "checkpoint-3", marked=False)
    save_resume_adapter(_save_context(_ExpertOnlyModel()), checkpoint)
    adapter_dir = resume_adapter_dir(checkpoint)
    rescaled = ExpertLoraSpec(r=SPEC.r, alpha=2 * SPEC.alpha, projections=SPEC.projections)

    with pytest.raises(ValueError, match=r"expert lora_alpha: .* Resume with the lora_alpha it was"):
        restore_adapters(adapter_dir, _ExpertLoraModel(rescaled), is_cp_mode=False)
    assert not applied.calls, "the expert adapters were restored before the refusal"
    assert restore_adapters(adapter_dir, _ExpertLoraModel(SPEC), is_cp_mode=False) is not None
    assert applied.calls


def test_the_marker_not_the_files_classifies_a_merged_checkpoint(tmp_path):
    """Merged weights are as loadable as a full fine-tune's, and only the marker, written last, proves completeness."""
    checkpoint = _merged_checkpoint(tmp_path / "checkpoint-3", marked=False)
    os.makedirs(os.path.join(checkpoint, RESUME_ADAPTER_DIR))
    save_file(dict(EXPERT_STATE), os.path.join(checkpoint, RESUME_ADAPTER_DIR, ADAPTER_SAFETENSORS_FILE))
    assert classify_resume_weights(checkpoint) == "full"

    _merged_checkpoint(checkpoint, marked=True)
    assert classify_resume_weights(checkpoint) == "merged_adapter"


def test_a_marked_checkpoint_resumes_the_policy_from_the_base(tmp_path):
    """Building from the merged weights and restoring the adapter would apply the delta twice."""
    checkpoint = _merged_checkpoint(tmp_path / "checkpoint-3", marked=True)
    os.makedirs(os.path.join(checkpoint, RESUME_ADAPTER_DIR))
    save_file(dict(EXPERT_STATE), os.path.join(checkpoint, RESUME_ADAPTER_DIR, ADAPTER_SAFETENSORS_FILE))
    assert resolve_resume_weights_source(checkpoint, SimpleNamespace(model_name_or_path=BASE), _EP) == BASE


def test_a_marked_checkpoint_without_its_adapter_refuses_before_the_policy_loads(tmp_path):
    """The loader would refuse too, but only after the whole base was built."""
    checkpoint = _merged_checkpoint(tmp_path / "checkpoint-3", marked=True)
    os.makedirs(os.path.join(checkpoint, RESUME_ADAPTER_DIR))

    with pytest.raises(ValueError, match="holds no adapter file"):
        resolve_resume_weights_source(checkpoint, SimpleNamespace(model_name_or_path=BASE), _EP)


def test_a_marked_checkpoint_restores_its_resume_adapter_onto_the_base(tmp_path, monkeypatch):
    checkpoint = _merged_checkpoint(tmp_path / "checkpoint-3", marked=True)
    restored = _Recorder(result="adapter file")
    monkeypatch.setattr(loader_mod, "restore_adapters", restored)
    model = _built_from(nn.Linear(4, 4), BASE)

    load_resume_weights(_load_context(model), checkpoint, model)

    assert len(restored.calls) == 1
    (source, _model), _ = restored.calls[0]
    assert source == os.path.join(checkpoint, RESUME_ADAPTER_DIR), "the adapters must come from the resume adapter"


def test_a_marked_checkpoint_refuses_a_model_built_from_its_merged_weights(tmp_path, monkeypatch):
    checkpoint = _merged_checkpoint(tmp_path / "checkpoint-3", marked=True)
    restored = _Recorder(result="adapter file")
    monkeypatch.setattr(loader_mod, "restore_adapters", restored)
    model = _built_from(nn.Linear(4, 4), checkpoint)

    with pytest.raises(ValueError, match="delta would apply twice"):
        load_resume_weights(_load_context(model), checkpoint, model)
    assert not restored.calls


def test_a_marked_checkpoint_without_its_adapter_raises(tmp_path):
    """The adapters would resume from init under restored moments."""
    checkpoint = _merged_checkpoint(tmp_path / "checkpoint-3", marked=True)
    model = _built_from(nn.Linear(4, 4), BASE)

    with pytest.raises(RuntimeError, match="holds no adapter file"):
        load_resume_weights(_load_context(model), checkpoint, model)


def test_an_unmarked_merged_checkpoint_refuses_an_adapter_run(tmp_path):
    """Fresh adapters under restored optimizer moments are the silent divergence."""
    checkpoint = _merged_checkpoint(tmp_path / "checkpoint-3", marked=False)
    model = _tiny_peft_model(seed=1)
    _built_from(model.get_base_model(), checkpoint)

    with pytest.raises(ValueError, match="without its resume adapter"):
        load_resume_weights(_load_context(model), checkpoint, model)


def test_an_unmarked_merged_checkpoint_refuses_an_expert_only_adapter_run(tmp_path):
    """Its adapters live on the EP layers and no PeftModel is in the tree, so a PEFT lookup alone would read it as a
    full fine-tune."""
    checkpoint = _merged_checkpoint(tmp_path / "checkpoint-3", marked=False)
    model = _built_from(nn.Sequential(_ExpertLoraLayer()), checkpoint)

    with pytest.raises(ValueError, match="without its resume adapter"):
        load_resume_weights(_load_context(model), checkpoint, model)


def test_an_unmarked_full_checkpoint_still_resumes_a_full_fine_tune(tmp_path):
    checkpoint = _merged_checkpoint(tmp_path / "checkpoint-3", marked=False)
    model = _built_from(nn.Linear(4, 4), checkpoint)

    load_resume_weights(_load_context(model), checkpoint, model)


def _trained_bf16_peft_model(**kwargs) -> PeftModel:
    """PEFT zero-inits ``lora_B``, so an untouched adapter folds to nothing."""
    model = _tiny_peft_model(seed=1, dtype=torch.bfloat16, lora_alpha=64, **kwargs)
    randomize_adapters(model, std=0.5)
    return model


def _merged_save(model: PeftModel, out_dir) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    expected = copy.deepcopy(model).merge_and_unload().state_dict()
    saving_mod.save_ep_gathered_checkpoint(_save_context(model), str(out_dir))
    return load_full_state_dict(str(out_dir)), expected


def test_the_merged_save_writes_the_peft_merge_and_no_weight_of_the_run(tmp_path):
    """Folded out of place, so no parameter of the run moves and a resume reproduces it."""
    model = _trained_bf16_peft_model()
    before = {name: param.detach().clone() for name, param in model.named_parameters()}

    written, expected = _merged_save(model, tmp_path)

    after = dict(model.named_parameters())
    assert not [name for name, value in before.items() if not torch.equal(value, after[name])], (
        "the save moved the run"
    )
    targets = sorted(key for key in expected if key.endswith(("q_proj.weight", "v_proj.weight")))
    base = {name.replace("base_model.model.", "").replace(".base_layer", ""): value for name, value in before.items()}
    assert targets and all(not torch.equal(expected[key], base[key]) for key in targets), "premise: the delta moves"
    assert not [key for key in targets if not torch.equal(written[key], expected[key])], "not PEFT's merge"
    assert not any("lora_" in key or "base_layer" in key for key in written), "adapter keys reached the checkpoint"


def test_the_merged_save_folds_a_tied_base_weight():
    """``named_parameters()`` lists the tied tensor under the embedding's name; the fold is keyed by the tensor, so
    the written weight carries the head's delta."""
    model = _trained_bf16_peft_model(tie_word_embeddings=True, target_modules=("lm_head", "q_proj"))
    tied = model.base_model.model.lm_head.base_layer.weight
    assert tied is model.base_model.model.model.embed_tokens.weight, "premise: the head is tied"
    assert not [name for name, _ in model.named_parameters() if "lm_head.base_layer" in name], (
        "premise: the tied weight is listed once, as the embedding"
    )
    original = tied.detach().clone()

    with tempfile.TemporaryDirectory() as out_dir:
        written, expected = _merged_save(model, out_dir)

    key = "model.embed_tokens.weight"
    assert not torch.equal(expected[key], original), "premise: the head's delta moves the tied weight"
    assert torch.equal(written[key], expected[key]), "the tied weight was written without the head's delta"
    assert torch.equal(tied, original), "the save moved the tied base weight"


class _SaveModelTrainer(DistributedTrainerMixin):
    """Its state factories are stubbed."""

    def __init__(self, ctx):
        self.ctx = ctx
        self.parallelism_config = SimpleNamespace(is_pp_mode=False, merge_expert_lora_on_save=True)

    def _top_level_model(self):
        return self.ctx.model

    def _checkpoint_context(self):
        return self.ctx


def test_the_final_export_carries_no_resume_adapter(tmp_path, monkeypatch):
    """Nothing resumes from the final export, so an adapter copy there would only grow what gets served and pushed."""
    merged_writes, resume_writes = _Recorder(), _Recorder()
    monkeypatch.setattr(save_mod, "save_ep_gathered_checkpoint", merged_writes)
    monkeypatch.setattr(checkpointing_mod, "save_resume_adapter", resume_writes)

    _SaveModelTrainer(_save_context(_ExpertOnlyModel())).save_model(str(tmp_path / "final"))

    assert merged_writes.calls and merged_writes.calls[0][0][0].merges_expert_lora, "premise: the merged save ran"
    assert not resume_writes.calls
    assert resume_adapter_dir(str(tmp_path / "final")) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
