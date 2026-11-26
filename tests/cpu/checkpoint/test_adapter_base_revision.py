#!/usr/bin/env python
"""An adapter save records the Hub commit its base was read at, and every later read of that base (both merges, the
bf16 tool's adapter load, the engine export, the adapter sync's launch base) takes that commit after the repo's default
branch moved."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from huggingface_hub import constants as hub_constants
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

from halo.checkpoint.adapters import read_adapter_config
from halo.checkpoint.parallel.adapters import save_expert_lora_adapters, save_peft_adapters
from halo.checkpoint.tools import adapter_merge, bf16_conversion
from halo.checkpoint.tools.adapter_merge import merge_peft_adapter
from halo.checkpoint.tools.offline_export import resolve_adapter_base_revision
from halo.checkpoint.weights_source import LOADED_WEIGHTS_FROM_ATTR, LOADED_WEIGHTS_REVISION_ATTR, weights_read_at
from halo.distributed.context_parallel.config import CPConfig
from halo.distributed.context_parallel.model_rules import SUPPORTED_ATTN_IMPLEMENTATIONS
from halo.distributed.context_parallel.ulysses_model import UlyssesModel
from halo.distributed.expert_parallel.moe_layer.expert_lora import EXPERT_LORA_RANK_MULTIPLE
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.pipeline_parallel.stage import build_pipeline_stage
from halo.entrypoints import engine_adapter_export
from halo.model_loading import tokenizer_setup
from halo.model_loading.parallel import loader
from halo.model_loading.tokenizer_setup import load_processing_class, resolve_tokenizer
from halo.models.moe.expert_lora_spec import ExpertLoraSpec
from halo.trainers.grpo.rollout import adapter_sync
from halo.weight_sync.clients.base import ServedBase
from tests.common.checkpoint_io import safetensors_state_dict, save_context
from tests.common.models import TINY_QWEN3_CONFIG
from tests.common.parallelism import make_parallelism_config, single_process_ep_config
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model, word_level_tokenizer

REPO = "halo-test/moving-base"
# The commit the run trained on, and the one the default branch names by the time the adapter is merged.
TRAINED, MOVED = "1" * 40, "2" * 40
_TARGET = "q_proj"
_MOVED_WORD = "moved"
_RESOLVER_LOGGER = "halo.checkpoint.tools.offline_export"


def _offline_hub_repo(tmp_path, monkeypatch, repo: str, default_branch: str) -> Path:
    """An empty repo in an offline Hub cache of its own, its default branch on ``default_branch``; returns where its
    snapshots go."""
    storage = tmp_path / "hub" / f"models--{repo.replace('/', '--')}"
    (storage / "refs").mkdir(parents=True)
    (storage / "refs" / "main").write_text(default_branch)
    monkeypatch.setattr(hub_constants, "HF_HUB_CACHE", str(tmp_path / "hub"))
    monkeypatch.setattr(hub_constants, "HF_HUB_OFFLINE", True)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    return storage / "snapshots"


def _tiny_base(seed: int) -> Qwen3ForCausalLM:
    torch.manual_seed(seed)
    return Qwen3ForCausalLM(Qwen3Config(**{**TINY_QWEN3_CONFIG, "num_hidden_layers": 2, "vocab_size": 96}))


@pytest.fixture
def snapshots(tmp_path, monkeypatch) -> Path:
    """:data:`REPO` at two commits that differ in every weight and in their tokenizer, the default branch on
    :data:`MOVED`; returns the snapshots directory."""
    snapshots = _offline_hub_repo(tmp_path, monkeypatch, REPO, MOVED)
    for seed, (commit, words) in enumerate(((TRAINED, ("hello", "world")), (MOVED, ("hello", "world", _MOVED_WORD)))):
        _tiny_base(seed).save_pretrained(snapshots / commit)
        word_level_tokenizer(words).save_pretrained(snapshots / commit)
    return snapshots


def _trained_policy() -> PeftModel:
    """:data:`REPO` at :data:`TRAINED` under a randomized LoRA, stamped as ``load_distributed_model`` stamps a Hub
    load."""
    model = Qwen3ForCausalLM.from_pretrained(REPO, revision=TRAINED, dtype=torch.float32)
    setattr(model, LOADED_WEIGHTS_FROM_ATTR, REPO)
    setattr(model, LOADED_WEIGHTS_REVISION_ATTR, TRAINED)
    peft_model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=[_TARGET], task_type="CAUSAL_LM"))
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_B" in name:
                param.normal_(std=0.3)
    return peft_model


def _saved_adapter(directory: Path) -> Path:
    save_peft_adapters(save_context(_trained_policy()), str(directory))
    return directory


def _unrecorded_adapter(directory: Path) -> Path:
    """Saved by stock PEFT, which records no revision."""
    _trained_policy().save_pretrained(directory)
    assert read_adapter_config(str(directory))["revision"] is None
    return directory


def _base_commit(merged: Path, snapshots: Path) -> str:
    """The commit whose untargeted tensors ``merged`` carries, over those the two commits differ in; its targeted ones
    must have moved off that commit's."""
    got = safetensors_state_dict(merged)
    bases = {commit: safetensors_state_dict(snapshots / commit) for commit in (TRAINED, MOVED)}
    telling = [
        key for key in bases[TRAINED] if _TARGET not in key and not torch.equal(bases[TRAINED][key], bases[MOVED][key])
    ]
    assert telling, "premise: the commits differ outside the adapter's targets"
    matching = [commit for commit, base in bases.items() if all(torch.equal(got[key], base[key]) for key in telling)]
    assert len(matching) == 1, f"the merge carries the untargeted tensors of {matching or 'neither commit'}"
    targeted = [key for key in got if _TARGET in key]
    assert targeted and not any(torch.equal(got[key], bases[matching[0]][key]) for key in targeted)
    return matching[0]


def _vocabulary(directory: Path) -> set[str]:
    return set(resolve_tokenizer(load_processing_class(str(directory))).get_vocab())


def test_a_peft_save_records_the_commit_the_base_was_read_at(snapshots, tmp_path):
    config = read_adapter_config(str(_saved_adapter(tmp_path / "adapter")))

    assert (config["base_model_name_or_path"], config["revision"]) == (REPO, TRAINED)


def _under_cp(model: torch.nn.Module) -> UlyssesModel:
    """The CP wrapper the CP loaders return, built eager (flash-attn cannot instantiate on CPU)."""
    model.config._attn_implementation = SUPPORTED_ATTN_IMPLEMENTATIONS[0]
    return UlyssesModel(model, CPConfig(1, stage_world_size=1, stage_base_rank=0, nvlink_domain_size=1))


@pytest.mark.parametrize("cp", [False, True], ids=["plain", "cp-wrapped"])
def test_an_expert_only_save_records_the_commit_the_base_was_read_at(tmp_path, cp):
    spec = ExpertLoraSpec(r=EXPERT_LORA_RANK_MULTIPLE, alpha=2 * EXPERT_LORA_RANK_MULTIPLE)
    model = apply_ep_layers(
        tiny_family_model(TINY_MOE_FAMILIES["qwen3_moe"]), single_process_ep_config(expert_lora=spec)
    )
    model.config._name_or_path = REPO
    if cp:
        model = _under_cp(model)
    # On what the loader returns, the wrapper under CP.
    setattr(model, LOADED_WEIGHTS_REVISION_ATTR, TRAINED)
    config = make_parallelism_config(world_size=1, gpus_per_node=1, expert_lora=spec)

    save_expert_lora_adapters(save_context(model, config, has_ep_layers=True, has_expert_lora=True), str(tmp_path))

    saved = read_adapter_config(str(tmp_path))
    assert (saved["base_model_name_or_path"], saved["revision"]) == (REPO, TRAINED)


@pytest.mark.parametrize("stamped", ["wrapper", "wrapped model"])
@pytest.mark.parametrize("peft", [False, True], ids=["bare", "peft"])
def test_the_commit_is_read_through_a_cp_wrapper(stamped, peft):
    inner = Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG, pad_token_id=0, eos_token_id=1))
    model = _under_cp(inner)
    setattr(model if stamped == "wrapper" else inner, LOADED_WEIGHTS_REVISION_ATTR, TRAINED)
    if peft:
        model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=[_TARGET]))

    assert weights_read_at(model) == TRAINED


def test_an_unstamped_model_was_read_at_no_commit():
    assert weights_read_at(Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG, pad_token_id=0, eos_token_id=1))) is None


@pytest.mark.parametrize("streaming", [False, True], ids=["in-memory", "streaming"])
def test_a_merge_folds_into_the_recorded_commit_after_the_default_branch_moved(snapshots, tmp_path, streaming):
    adapter, merged = _saved_adapter(tmp_path / "adapter"), tmp_path / "merged"

    merge_peft_adapter(str(adapter), str(merged), dtype=torch.float32, streaming=streaming, verbose=False)

    assert _base_commit(merged, snapshots) == TRAINED
    assert _MOVED_WORD not in _vocabulary(merged), "the merged tokenizer came from the default branch"


def test_the_in_memory_merge_exports_the_bases_config_at_the_recorded_commit(snapshots, tmp_path, monkeypatch):
    """A source-schema family's export reads the base's ``config.json`` and remote code again, through this."""
    pinned: list[tuple[str, str | None]] = []
    save = adapter_merge.save_full_checkpoint

    def recording(model, output_dir, **kwargs):
        pinned.append((kwargs["source_dir"], kwargs["source_revision"]))
        return save(model, output_dir, **kwargs)

    monkeypatch.setattr(adapter_merge, "save_full_checkpoint", recording)
    adapter = _saved_adapter(tmp_path / "adapter")

    merge_peft_adapter(str(adapter), str(tmp_path / "merged"), dtype=torch.float32, verbose=False)

    assert pinned == [(REPO, TRAINED)]


@pytest.mark.parametrize("streaming", [False, True], ids=["in-memory", "streaming"])
def test_an_adapter_recording_no_revision_merges_into_the_default_branch_and_says_so(
    snapshots, tmp_path, streaming, caplog
):
    adapter, merged = _unrecorded_adapter(tmp_path / "adapter"), tmp_path / "merged"

    with caplog.at_level(logging.WARNING, logger=_RESOLVER_LOGGER):
        merge_peft_adapter(str(adapter), str(merged), dtype=torch.float32, streaming=streaming, verbose=False)

    assert _base_commit(merged, snapshots) == MOVED
    assert any("--base_revision" in record.message and REPO in record.message for record in caplog.records)


@pytest.mark.parametrize("streaming", [False, True], ids=["in-memory", "streaming"])
def test_base_revision_names_the_commit_for_an_adapter_recording_none(snapshots, tmp_path, streaming):
    adapter, merged = _unrecorded_adapter(tmp_path / "adapter"), tmp_path / "merged"

    merge_peft_adapter(
        str(adapter), str(merged), dtype=torch.float32, streaming=streaming, base_revision=TRAINED, verbose=False
    )

    assert _base_commit(merged, snapshots) == TRAINED
    assert _MOVED_WORD not in _vocabulary(merged)


@pytest.mark.parametrize(
    ("recorded", "override", "expected", "warns"),
    [
        (TRAINED, None, TRAINED, False),
        (None, TRAINED, TRAINED, False),
        (TRAINED, TRAINED, TRAINED, False),
        (TRAINED, MOVED, MOVED, True),
        (None, None, None, True),
    ],
    ids=["recorded", "override", "both agree", "override differs", "neither"],
)
def test_the_base_revision_of_a_hub_base(recorded, override, expected, warns, caplog):
    with caplog.at_level(logging.WARNING, logger=_RESOLVER_LOGGER):
        assert resolve_adapter_base_revision(REPO, recorded, override) == expected
    assert bool(caplog.records) is warns


def test_a_local_base_has_no_revision(tmp_path, caplog):
    with caplog.at_level(logging.WARNING, logger=_RESOLVER_LOGGER):
        assert resolve_adapter_base_revision(str(tmp_path), TRAINED, MOVED) is None
        assert resolve_adapter_base_revision(str(tmp_path), None) is None
    assert not caplog.records


def _record_processor_loads(monkeypatch, *, found: bool) -> list[tuple[str, str | None]]:
    """Every ``(source, revision)`` the bf16 tool reads a processing class at; ``found=False`` makes each read miss."""
    loads: list[tuple[str, str | None]] = []
    load = bf16_conversion.load_processing_class

    def recording(source, *, trust_remote_code=False, revision=None):
        loads.append((source, revision))
        return load(source, trust_remote_code=trust_remote_code, revision=revision) if found else None

    monkeypatch.setattr(bf16_conversion, "load_processing_class", recording)
    return loads


def test_the_bf16_tool_loads_an_adapters_base_at_the_recorded_commit(snapshots, tmp_path, monkeypatch):
    adapter = _saved_adapter(tmp_path / "adapter")
    loads = _record_processor_loads(monkeypatch, found=True)

    model = bf16_conversion.load_model(str(adapter), "causal_lm", is_peft=True, dtype=torch.float32)

    embeddings = {
        commit: safetensors_state_dict(snapshots / commit)["model.embed_tokens.weight"] for commit in (TRAINED, MOVED)
    }
    assert not torch.equal(embeddings[TRAINED], embeddings[MOVED])
    assert torch.equal(model.get_input_embeddings().weight, embeddings[TRAINED])
    assert loads == [(str(adapter), None), (REPO, TRAINED)]


def test_the_bf16_inference_check_reads_the_bases_tokenizer_at_the_recorded_commit(snapshots, tmp_path, monkeypatch):
    adapter = _saved_adapter(tmp_path / "adapter")
    loads = _record_processor_loads(monkeypatch, found=False)

    bf16_conversion.run_test_inference(str(adapter), "causal_lm", is_peft=True)

    assert loads == [(str(adapter), None), (REPO, TRAINED)]


@pytest.mark.parametrize(
    ("save", "base_revision"),
    [(_saved_adapter, None), (_unrecorded_adapter, TRAINED)],
    ids=["recorded", "--base_revision"],
)
def test_the_engine_export_reads_the_bases_config_at_the_adapters_commit(
    snapshots, tmp_path, monkeypatch, save, base_revision
):
    adapter = save(tmp_path / "adapter")
    reads: list[tuple[str, str | None]] = []
    resolve = engine_adapter_export.resolve_build_config

    def recording(source, *, revision, trust_remote_code):
        reads.append((source, revision))
        return resolve(source, revision=revision, trust_remote_code=trust_remote_code)

    monkeypatch.setattr(engine_adapter_export, "resolve_build_config", recording)

    engine_adapter_export.export_engine_adapter(
        str(adapter), str(tmp_path / "engine"), rollout_backend="sglang", base_revision=base_revision
    )

    assert reads == [(REPO, TRAINED)]


@pytest.mark.parametrize("processor", [True, False], ids=["processor", "tokenizer fallback"])
def test_the_processor_load_takes_a_hub_revision(snapshots, monkeypatch, processor):
    if not processor:

        def no_processor(*args, **kwargs):
            raise OSError("no processor config")

        monkeypatch.setattr(tokenizer_setup.AutoProcessor, "from_pretrained", no_processor)

    def vocabulary(revision):
        return set(resolve_tokenizer(load_processing_class(REPO, revision=revision)).get_vocab())

    assert _MOVED_WORD in vocabulary(None)
    assert _MOVED_WORD not in vocabulary(TRAINED)


def test_a_commit_carrying_no_tokenizer_loads_none_beside_a_default_branch_that_has_one(tmp_path, monkeypatch):
    """transformers builds the family's tokenizer class with no file behind it, so the presence check has to look at
    the commit that was loaded."""
    snapshots = _offline_hub_repo(tmp_path, monkeypatch, REPO, MOVED)
    for commit in (TRAINED, MOVED):
        _tiny_base(0).save_pretrained(snapshots / commit)
    word_level_tokenizer(("hello", "world")).save_pretrained(snapshots / MOVED)

    assert load_processing_class(REPO, revision=TRAINED) is None
    assert load_processing_class(REPO) is not None


def _stamp_after_load(monkeypatch, model_name_or_path: str, **load) -> str | None:
    """The revision stamp the real ``load_distributed_model`` leaves over a stubbed per-mode load."""
    model = SimpleNamespace(config=SimpleNamespace())
    monkeypatch.setattr(loader, "reject_unsupported_structure", lambda *args, **kwargs: None)
    monkeypatch.setattr(loader, "_dispatch_model_loading", lambda *args, **kwargs: model)
    monkeypatch.setattr(loader, "apply_run_model_policies", lambda *args, **kwargs: None)
    monkeypatch.setattr(loader, "warm_attention_kernels", lambda *args, **kwargs: None)
    loader.load_distributed_model(
        model_name_or_path,
        make_parallelism_config(world_size=1, gpus_per_node=1),
        attn_implementation="eager",
        trust_remote_code=False,
        **load,
    )
    return getattr(model, LOADED_WEIGHTS_REVISION_ATTR)


@pytest.mark.parametrize(("revision", "commit"), [(TRAINED, TRAINED), (None, MOVED)], ids=["pinned", "default branch"])
def test_the_loader_stamps_the_commit_a_hub_source_resolved_to(snapshots, monkeypatch, revision, commit):
    assert _stamp_after_load(monkeypatch, REPO, revision=revision) == commit


def test_the_loader_stamps_no_commit_on_a_local_directory(snapshots, monkeypatch):
    assert _stamp_after_load(monkeypatch, str(snapshots / TRAINED), revision="v1") is None


def test_the_loader_stamps_no_commit_on_a_random_init(snapshots, monkeypatch):
    assert _stamp_after_load(monkeypatch, REPO, revision=TRAINED, init_from_scratch=True) is None


def test_a_pipeline_stage_carries_the_commit_its_model_was_read_at():
    model = Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG, pad_token_id=0, eos_token_id=1))
    setattr(model, LOADED_WEIGHTS_REVISION_ATTR, TRAINED)

    assert weights_read_at(build_pipeline_stage(model, 0, 2)) == TRAINED


def test_the_adapter_sync_launch_base_is_the_commit_the_trainer_read(snapshots):
    """A server launched at the trained commit reports that snapshot; the default branch's would never match it."""
    launch = adapter_sync.resolve_launch_base(_trained_policy(), SimpleNamespace(model_name=REPO))

    assert launch == ServedBase(REPO, str(snapshots / TRAINED), TRAINED)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
