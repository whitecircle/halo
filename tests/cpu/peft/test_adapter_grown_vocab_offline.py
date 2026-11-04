#!/usr/bin/env python
"""A LoRA run that grew its vocabulary saves, resumes and merges the added rows with the Hub out of reach.
PEFT's ``save_embedding_layers="auto"`` detects a grown vocabulary by rereading the base config, impossible offline."""

import json
import logging
import os
import shutil
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from huggingface_hub import constants as hub_constants
from peft import LoraConfig, PeftModel, get_peft_model
from peft.utils import get_peft_model_state_dict
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

import halo.checkpoint.parallel.adapters as adapters_mod
from halo.checkpoint.adapters import (
    MIXED_EXPERT_LORA_PEFT_TYPE,
    adapter_weight_paths,
    read_adapter_config,
    read_adapter_file,
    resume_adapter_dir,
    write_adapter_file,
)
from halo.checkpoint.format import ADAPTER_SAFETENSORS_FILE
from halo.checkpoint.parallel.adapters import (
    gather_adapter_state,
    restore_adapters,
    save_peft_adapters,
    save_resume_adapter,
)
from halo.checkpoint.tools.adapter_merge import merge_peft_adapter
from halo.distributed.context_parallel.config import CPConfig
from halo.distributed.context_parallel.model_rules import SUPPORTED_ATTN_IMPLEMENTATIONS
from halo.distributed.context_parallel.ulysses_model import UlyssesModel
from halo.models.moe.expert_lora_spec import ExpertLoraSpec
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from halo.trainers.parallelism.adapter_rules import reject_unsaved_grown_vocabulary
from tests.common.checkpoint_io import save_context
from tests.common.ep_stubs import StubEPLayerBase
from tests.common.grown_vocab import GROWING_TOKENS, WORDS, build_word_level_checkpoint, grow_vocabulary
from tests.common.parallelism import make_parallelism_config
from tests.common.tiny_models import TINY_QWEN3_VL_FAMILY, tiny_family_model, word_level_tokenizer

REPO = "halo-test/grown-vocab-base"
COMMIT = "a" * 40
EMBED_KEY = "base_model.model.model.embed_tokens.weight"
HEAD_KEY = "base_model.model.lm_head.weight"
LORA = {"r": 4, "lora_alpha": 8, "target_modules": ["q_proj", "v_proj"]}
EXPERT_SPEC = ExpertLoraSpec(r=4, alpha=8.0, projections=frozenset({"gate", "up"}))
# The expert tensors a mixed run's gather emits, which no live tensor of the tiny dense model takes.
EXPERT_STATE = {
    "model.layers.0.mlp.experts.gate_up_proj.lora_A": torch.ones(2, 4, 8),
    "model.layers.0.mlp.experts.gate_up_proj.lora_B": torch.zeros(2, 8, 6),
}


@pytest.fixture(scope="module", params=[True, False], ids=["tied", "untied"])
def checkpoint(request, tmp_path_factory) -> tuple[str, bool]:
    directory = str(tmp_path_factory.mktemp("tied" if request.param else "untied"))
    build_word_level_checkpoint(directory, tied=request.param)
    return directory, request.param


@pytest.fixture
def offline_base(checkpoint, tmp_path, monkeypatch) -> bool:
    """The checkpoint as :data:`REPO`'s cached snapshot, read with the Hub out of reach; returns whether it is tied."""
    directory, tied = checkpoint
    storage = tmp_path / "hub" / f"models--{REPO.replace('/', '--')}"
    (storage / "refs").mkdir(parents=True)
    (storage / "refs" / "main").write_text(COMMIT)
    shutil.copytree(directory, storage / "snapshots" / COMMIT)
    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    return tied


def _policy(*, seed: int | None, cp: bool = False, lora: dict = LORA) -> tuple[PeftModel, object, int]:
    """:data:`REPO` with a grown vocabulary and LoRA; the added rows are drawn from ``seed``, zeroed for ``None``."""
    model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32)
    rows = model.get_input_embeddings().num_embeddings
    if cp:
        # Built eager (flash-attn cannot instantiate on CPU) but declaring what the CP validator reads.
        model.config._attn_implementation = SUPPORTED_ATTN_IMPLEMENTATIONS[0]
        model = UlyssesModel(model, CPConfig(1, stage_world_size=1, stage_base_rank=0, nvlink_domain_size=1))
    tokenizer = grow_vocabulary(model, AutoTokenizer.from_pretrained(REPO), GROWING_TOKENS)
    peft_model = get_peft_model(model, LoraConfig(**lora))
    with torch.no_grad():
        for module in (peft_model.get_input_embeddings(), peft_model.get_output_embeddings()):
            added = module.weight[rows:]
            if seed is None:
                added.zero_()
            else:
                added.copy_(torch.randn(added.shape, generator=torch.Generator().manual_seed(seed)))
    return peft_model, tokenizer, rows


def _saved_keys(tied: bool) -> list[str]:
    return [EMBED_KEY] if tied else [EMBED_KEY, HEAD_KEY]


def _added_rows(peft_model, rows: int) -> list[torch.Tensor]:
    return [
        module.weight[rows:].detach().clone()
        for module in (peft_model.get_input_embeddings(), peft_model.get_output_embeddings())
    ]


def _checkpoint_context(peft_model, *, cp: bool = False):
    """A training checkpoint's save context, which keeps the live dtype for an exact resume; CP2 with ``cp``."""
    parallelism = make_parallelism_config(world_size=2, gpus_per_node=2, cp_size=2) if cp else None
    return save_context(peft_model, parallelism, training_checkpoint=True)


def test_peft_alone_leaves_the_grown_rows_out_offline(offline_base):
    peft_model, _, rows = _policy(seed=1)
    assert peft_model.get_input_embeddings().num_embeddings == len(WORDS) + 2 + len(GROWING_TOKENS) > rows
    with pytest.warns(UserWarning, match="Could not find a config file"):
        selected = get_peft_model_state_dict(peft_model)
    assert not {EMBED_KEY, HEAD_KEY} & set(selected)


@pytest.mark.parametrize("branch", ["plain", "gathered", "cp"])
def test_every_saver_writes_the_grown_rows_offline(offline_base, tmp_path, branch):
    peft_model, _, rows = _policy(seed=1, cp=branch == "cp")
    grown = len(WORDS) + 2 + len(GROWING_TOKENS)
    if branch == "gathered":
        saved = gather_adapter_state(peft_model)
    else:
        save_peft_adapters(_checkpoint_context(peft_model, cp=branch == "cp"), str(tmp_path))
        written = [os.path.basename(path) for path in adapter_weight_paths(str(tmp_path)) if os.path.isfile(path)]
        assert written == [ADAPTER_SAFETENSORS_FILE], written
        saved = read_adapter_file(os.path.join(tmp_path, ADAPTER_SAFETENSORS_FILE))
    for key in _saved_keys(offline_base):
        assert key in saved and saved[key].shape[0] == grown, (key, sorted(saved)[:4])
    for key, want in zip((EMBED_KEY, HEAD_KEY), _added_rows(peft_model, rows), strict=True):
        if key in saved:
            assert torch.equal(saved[key][rows:].float(), want), key


def _gathered_save(peft_model, directory: str) -> None:
    """What ``save_peft_adapters``'s gathered branch writes for FSDP2 and mixed EP runs; a tied pair is named once."""
    peft_model.peft_config["default"].save_pretrained(directory)
    write_adapter_file(gather_adapter_state(peft_model), directory)


@pytest.mark.parametrize("branch", ["plain", "gathered", "cp"])
def test_the_resume_restores_the_grown_rows_offline(offline_base, tmp_path, branch):
    cp = branch == "cp"
    trained, _, rows = _policy(seed=1, cp=cp)
    if branch == "gathered":
        _gathered_save(trained, str(tmp_path))
        saved = read_adapter_file(os.path.join(tmp_path, ADAPTER_SAFETENSORS_FILE))
        assert (HEAD_KEY in saved) is not offline_base, "premise: a tied pair is written under one name"
    else:
        save_peft_adapters(_checkpoint_context(trained, cp=cp), str(tmp_path))
    fresh, _, _ = _policy(seed=None, cp=cp)
    assert all(not added.any() for added in _added_rows(fresh, rows)), "premise: the fresh rows are zeroed"

    assert restore_adapters(str(tmp_path), fresh, is_cp_mode=cp) is not None

    for got, want in zip(_added_rows(fresh, rows), _added_rows(trained, rows), strict=True):
        assert torch.equal(got, want)
    if offline_base:
        assert fresh.get_input_embeddings().weight is fresh.get_output_embeddings().weight


def _lora_tensors(peft_model) -> dict[str, torch.Tensor]:
    return {name: param.detach().clone() for name, param in peft_model.named_parameters() if "lora_" in name}


def _resume_warnings(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == adapters_mod.__name__]


@pytest.mark.parametrize(
    "lora", [LORA, {**LORA, "modules_to_save": ["embed_tokens"]}], ids=["lora", "embedding-modules-to-save"]
)
def test_an_adapter_saved_without_the_grown_rows_resumes_on_the_regrown_rows(offline_base, tmp_path, caplog, lora):
    """Its run kept the rows frozen at their mean-resize init, so the resume keeps the rows this run's grow drew."""
    trained, _, rows = _policy(seed=1, lora=lora)
    trained.save_pretrained(str(tmp_path), save_embedding_layers=False)
    saved = set(load_file(tmp_path / ADAPTER_SAFETENSORS_FILE))
    # The trained modules_to_save copy is saved whatever the check decides, under the embedding's own name.
    frozen = {HEAD_KEY, EMBED_KEY.replace(".weight", ".original_module.weight")}
    assert not frozen & saved and (EMBED_KEY in saved) == ("modules_to_save" in lora), "premise: frozen rows left out"
    fresh, _, _ = _policy(seed=None, lora=lora)
    before = _added_rows(fresh, rows)

    with caplog.at_level(logging.WARNING, logger=adapters_mod.__name__):
        assert restore_adapters(str(tmp_path), fresh, is_cp_mode=False) is not None

    (warning,) = _resume_warnings(caplog)
    assert "saved before adapter saves carried them" in warning and HEAD_KEY in warning, warning
    trained_lora, fresh_lora = _lora_tensors(trained), _lora_tensors(fresh)
    assert trained_lora and all(torch.equal(fresh_lora[name], value) for name, value in trained_lora.items())
    if "modules_to_save" in lora:
        assert torch.equal(fresh.get_input_embeddings().weight, trained.get_input_embeddings().weight)
    else:
        assert all(torch.equal(got, want) for got, want in zip(_added_rows(fresh, rows), before, strict=True))


def test_an_adapter_missing_a_trained_tensor_is_still_refused(offline_base, tmp_path):
    """Only the grown vocabulary's frozen rows may be absent."""
    trained, _, _ = _policy(seed=1)
    trained.save_pretrained(str(tmp_path), save_embedding_layers=False)
    weights = tmp_path / ADAPTER_SAFETENSORS_FILE
    state = load_file(weights)
    dropped = next(key for key in sorted(state) if "lora_B" in key)
    del state[dropped]
    save_file(state, weights)
    fresh, _, _ = _policy(seed=None)

    with pytest.raises(RuntimeError, match="missing from the file") as excinfo:
        restore_adapters(str(tmp_path), fresh, is_cp_mode=False)
    message = str(excinfo.value)
    assert dropped in message and EMBED_KEY not in message and "added_special_tokens" in message, message


def test_ranks_disagreeing_on_the_unsaved_grown_rows_refuse_before_any_copy(offline_base, tmp_path, monkeypatch):
    """A kept key's copy is a mesh collective on a DTensor, so the missing rows are agreed across ranks first."""
    trained, _, _ = _policy(seed=1)
    trained.save_pretrained(str(tmp_path), save_embedding_layers=False)
    fresh, _, _ = _policy(seed=None)
    before = _lora_tensors(fresh)
    agreed: list[object] = []

    def a_peer_disagrees(obj) -> bool:
        agreed.append(obj)
        return False

    monkeypatch.setattr(adapters_mod, "agree_across_ranks", a_peer_disagrees)

    with pytest.raises(RuntimeError, match="on some ranks only"):
        restore_adapters(str(tmp_path), fresh, is_cp_mode=False)
    assert len(agreed) == 1 and HEAD_KEY in agreed[0], agreed
    assert all(torch.equal(param, before[name]) for name, param in _lora_tensors(fresh).items()), "a tensor was copied"


def test_a_mixed_resume_adapter_carries_the_grown_rows(offline_base, tmp_path, monkeypatch):
    """Also holds for the resume adapter of a ``merge_expert_lora_on_save`` checkpoint."""
    monkeypatch.setattr(adapters_mod, "gather_expert_lora_adapters", lambda model, retain: dict(EXPERT_STATE))
    trained, _, rows = _policy(seed=1)
    config = make_parallelism_config(
        world_size=1, gpus_per_node=1, expert_lora=EXPERT_SPEC, merge_expert_lora_on_save=True
    )
    checkpoint = str(tmp_path / "checkpoint-2")
    os.makedirs(checkpoint)

    save_resume_adapter(save_context(trained, config, has_expert_lora=True, has_ep_layers=True), checkpoint)

    adapter_dir = resume_adapter_dir(checkpoint)
    assert read_adapter_config(adapter_dir)["peft_type"] == MIXED_EXPERT_LORA_PEFT_TYPE
    saved = load_file(os.path.join(adapter_dir, ADAPTER_SAFETENSORS_FILE))
    assert set(EXPERT_STATE) < set(saved)
    for key in _saved_keys(offline_base):
        assert saved[key].shape[0] == trained.get_input_embeddings().num_embeddings, key
    fresh, _, _ = _policy(seed=None)
    expert_loads = []
    monkeypatch.setattr(adapters_mod, "load_expert_lora_adapters", lambda model, state: expert_loads.append(state))

    assert restore_adapters(adapter_dir, fresh, is_cp_mode=False) is not None

    for got, want in zip(_added_rows(fresh, rows), _added_rows(trained, rows), strict=True):
        assert torch.equal(got, want)
    assert set(expert_loads[0]) == set(EXPERT_STATE)


def test_the_merge_serves_the_rows_saved_offline(offline_base, tmp_path):
    adapter_dir, merged_dir = str(tmp_path / "adapter"), str(tmp_path / "merged")
    trained, tokenizer, rows = _policy(seed=1)
    save_peft_adapters(_checkpoint_context(trained), adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    with open(os.path.join(adapter_dir, "adapter_config.json")) as fh:
        assert json.load(fh)["base_model_name_or_path"] == REPO, "premise: the adapter names its base by Hub id"

    merge_peft_adapter(adapter_dir, merged_dir, dtype=torch.float32, verbose=False)

    merged = AutoModelForCausalLM.from_pretrained(merged_dir, dtype=torch.float32)
    for got, want in zip(_added_rows(merged, rows), _added_rows(trained, rows), strict=True):
        assert torch.equal(got, want)


def test_a_composite_config_saves_the_grown_rows_from_a_local_base(tmp_path):
    """The base config is on disk, but its ``vocab_size`` sits on the text config."""
    tokenizer = word_level_tokenizer(WORDS)
    torch.manual_seed(0)
    model = tiny_family_model(TINY_QWEN3_VL_FAMILY, tokenizer)
    model.save_pretrained(tmp_path / "base")
    model = type(model).from_pretrained(tmp_path / "base", dtype=torch.float32)
    rows = model.get_input_embeddings().num_embeddings
    grow_vocabulary(model, tokenizer, GROWING_TOKENS)
    peft_model = get_peft_model(model, LoraConfig(**LORA))
    assert not any("embed_tokens" in key for key in get_peft_model_state_dict(peft_model)), "premise: PEFT drops it"

    saved = gather_adapter_state(peft_model)

    embedding = [key for key in saved if key.endswith("embed_tokens.weight")]
    assert len(embedding) == 1 and saved[embedding[0]].shape[0] == len(tokenizer) > rows, sorted(saved)[:4]


def _expert_only(model) -> None:
    """Give ``model`` a native expert-adapted EP layer, as an expert-only LoRA run builds without a PEFT model."""
    layer = StubEPLayerBase()
    layer._expert_lora_attrs = frozenset({"down_proj"})
    model.model.layers[0].mlp = layer


def test_the_trainer_refuses_runs_whose_saves_would_leave_added_rows_out(offline_base):
    peft_model, _, _ = _policy(seed=1)
    with pytest.raises(ValueError, match="accelerate-managed FSDP"):
        reject_unsaved_grown_vocabulary(peft_model, accelerate_manages_fsdp=True)
    reject_unsaved_grown_vocabulary(peft_model, accelerate_manages_fsdp=False)

    embedding_lora = {**LORA, "target_modules": ["q_proj", "embed_tokens"]}
    one_layer, _, _ = _policy(seed=1, lora=embedding_lora)
    if offline_base:
        reject_unsaved_grown_vocabulary(one_layer, accelerate_manages_fsdp=False)
    else:
        with pytest.raises(ValueError, match="leaves out the output head"):
            reject_unsaved_grown_vocabulary(one_layer, accelerate_manages_fsdp=False)
        both_layers, _, _ = _policy(seed=1, lora={**LORA, "target_modules": ["q_proj", "embed_tokens", "lm_head"]})
        reject_unsaved_grown_vocabulary(both_layers, accelerate_manages_fsdp=False)


class _PastTheGates(Exception):
    """Carries control out of the trainer's mode setup once its pre-wrap gates have passed."""


class _ModeSetupHost(DistributedTrainerMixin):
    """The real ``_configure_distributed_modes`` on one rank, up to the first step past its pre-wrap gates."""

    _deferred_liger_kernel = False
    _accelerate_manages_ddp = False
    label_smoother = None
    compute_loss_func = None

    def __init__(self, model, *, accelerate_manages_fsdp: bool):
        self.model = model
        self.parallelism_config = make_parallelism_config(world_size=1, gpus_per_node=1)
        self.args = SimpleNamespace(
            load_best_model_at_end=False,
            optim="adamw_torch",
            per_device_train_batch_size=1,
            eval_strategy="no",
            eval_on_start=False,
        )
        self._bf16_optimizer = False
        self._ep_config = None
        self._accelerate_manages_fsdp = accelerate_manages_fsdp

    def _cast_peft_params_to_compute_dtype(self):
        raise _PastTheGates


def test_trainer_construction_runs_the_grown_vocabulary_refusal(offline_base):
    """The refusal runs before any wrap; otherwise the save it guards first fails at a checkpoint."""
    peft_model, _, _ = _policy(seed=1)
    with patch("halo.trainers.distributed_trainer.agree_rank_uniform_env"):
        with pytest.raises(ValueError, match="grew the vocabulary of an adapter run under accelerate-managed FSDP"):
            DistributedTrainerMixin._configure_distributed_modes(
                _ModeSetupHost(peft_model, accelerate_manages_fsdp=True)
            )
        with pytest.raises(_PastTheGates):
            DistributedTrainerMixin._configure_distributed_modes(
                _ModeSetupHost(peft_model, accelerate_manages_fsdp=False)
            )


def test_an_expert_only_run_that_grew_its_vocabulary_trains(offline_base):
    """The resume regrows the rows at the same init, and ``merge_expert_lora_on_save`` exports the embedding whole."""
    model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32)
    grow_vocabulary(model, AutoTokenizer.from_pretrained(REPO), GROWING_TOKENS)
    _expert_only(model)
    reject_unsaved_grown_vocabulary(model, accelerate_manages_fsdp=True)


def test_an_ungrown_vocabulary_is_never_refused(checkpoint):
    """The refusals key on the grow, so the same runs on the checkpoint's own rows train."""
    for lora in (LORA, {**LORA, "target_modules": ["q_proj", "embed_tokens"]}):
        model = AutoModelForCausalLM.from_pretrained(checkpoint[0], dtype=torch.float32)
        reject_unsaved_grown_vocabulary(get_peft_model(model, LoraConfig(**lora)), accelerate_manages_fsdp=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
