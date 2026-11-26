#!/usr/bin/env python
"""``export_engine_adapter.py`` writes a saved adapter as the directory adapter sync publishes for the same adapters,
tensor for tensor and config key for key, on a tiny Inkling 2 with attention and native expert LoRA, and refuses what
the engine cannot serve, with the reason and the path that remains, before any write."""

import json
import logging
import re
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file

from halo.checkpoint.adapters import EXPERT_LORA_CONFIG_KEY, read_adapter_config
from halo.checkpoint.format import ADAPTER_CONFIG_FILE, ADAPTER_SAFETENSORS_FILE
from halo.checkpoint.parallel.adapters import save_expert_lora_adapters, save_peft_adapters
from halo.distributed.expert_parallel.moe_layer.expert_lora import EXPERT_LORA_RANK_MULTIPLE
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.entrypoints import engine_adapter_export
from halo.entrypoints.engine_adapter_export import export_engine_adapter
from halo.model_loading.model_preparation import (
    resolve_auto_model_class,
    resolve_build_config,
    resolve_text_only_config,
)
from halo.models.moe.expert_lora_spec import ExpertLoraSpec
from halo.trainers.grpo.rollout.adapter_sync import AdapterSync, reject_unadaptable_model, resolve_adapter_serving
from halo.weight_sync.adapter_payload import ADAPTER_WIRE_DTYPE
from halo.weight_sync.adapter_routes import AdapterRoutes
from halo.weight_sync.adapter_servers import AdapterServers
from halo.weight_sync.adapter_versions import AdapterVersions
from halo.weight_sync.clients.base import ServedBase
from halo.weight_sync.clients.vllm import VLLMWeightSyncClient
from tests.common.checkpoint_io import save_context
from tests.common.fake_engines import FakeVLLMServer
from tests.common.parallelism import make_parallelism_config, single_process_ep_config
from tests.common.tiny_models import (
    TINY_DENSE_FAMILY,
    TINY_MOE_FAMILIES,
    TINY_MOE_VLM_FAMILIES,
    TinyFamily,
    tiny_family_model,
)

_RANK = EXPERT_LORA_RANK_MULTIPLE
_SPEC = ExpertLoraSpec(r=_RANK, alpha=2 * _RANK)
# The shipped Inkling 2 examples' PEFT targets: the text tower's attention, and o_proj beside the experts.
_ATTENTION = r".*language_model\.layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj|relative_position_bias|o_proj)\.linear"
_NO_PARALLELISM = SimpleNamespace(is_tp_mode=False, is_pp_mode=False, is_cp_mode=False)
_INKLING2 = TINY_MOE_VLM_FAMILIES["inkling2"]


def _output_projection(model: torch.nn.Module) -> str:
    """``o_proj``, or on Inkling 2 the linear inside its output projection module."""
    return "o_proj.linear" if any(name.endswith("o_proj.linear") for name, _ in model.named_modules()) else "o_proj"


def _trained(base_dir: Path, *, experts: ExpertLoraSpec | None, family: TinyFamily = _INKLING2, targets=_ATTENTION):
    """A PeftModel over ``family``'s tiny model saved at ``base_dir``, with LoRA on ``targets``, or on ``o_proj`` beside
    native expert LoRA under ``experts``, every adapter drawn away from its zero init."""
    torch.manual_seed(0)
    model = tiny_family_model(family).to(torch.bfloat16)
    model.save_pretrained(base_dir)
    model.name_or_path = str(base_dir)
    if experts is not None:
        model = apply_ep_layers(model, single_process_ep_config(expert_lora=experts))
        targets = [_output_projection(model)]
    # Adapter sync serves bf16 adapters only, the dtype the engine stores.
    peft_model = get_peft_model(model, LoraConfig(r=_RANK, lora_alpha=2 * _RANK, target_modules=targets))
    peft_model = peft_model.to(torch.bfloat16)
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_" in name:
                param.normal_(std=0.05)
    return peft_model


def _save(peft_model, adapter_dir: Path, experts: ExpertLoraSpec | None) -> None:
    run = make_parallelism_config(world_size=1, gpus_per_node=1, expert_lora=experts)
    mixed = experts is not None
    save_peft_adapters(save_context(peft_model, run, has_ep_layers=mixed, has_expert_lora=mixed), str(adapter_dir))


def _published_by_adapter_sync(peft_model, adapter_dir: Path) -> Path:
    """The version a live sync publishes to a vLLM server launched on the base, as Inkling 2's runs sync."""
    fake = FakeVLLMServer()
    fake.lora = {"max_lora_rank": 16, "max_loras": 2, "max_cpu_loras": 4}
    servers = AdapterServers([fake.url], client_cls=VLLMWeightSyncClient, connection_timeout=5.0)
    try:
        sync = AdapterSync(
            resolve_adapter_serving(peft_model),
            AdapterVersions(),
            AdapterRoutes(VLLMWeightSyncClient.is_adapter_missing),
            ServedBase(model_id="fake-model", path=fake.model_path, revision=None),
            "no full push into this family is validated",
            str(adapter_dir),
        )
        sync.servers = servers
        sync.sync(SimpleNamespace(model=peft_model), None)
        return adapter_dir / sync.versions.current
    finally:
        servers.close()
        fake.close()


@pytest.mark.parametrize("experts", [None, _SPEC], ids=["attention", "attention+experts"])
def test_the_export_of_a_saved_adapter_is_what_adapter_sync_publishes(tmp_path, experts):
    peft_model = _trained(tmp_path / "base", experts=experts)
    reject_unadaptable_model(peft_model, _NO_PARALLELISM, VLLMWeightSyncClient)
    published = _published_by_adapter_sync(peft_model, tmp_path / "published")
    _save(peft_model, tmp_path / "saved", experts)

    exported = export_engine_adapter(str(tmp_path / "saved"), str(tmp_path / "exported"), rollout_backend="vllm")

    want, saved = (load_file(str(path / ADAPTER_SAFETENSORS_FILE)) for path in (published, tmp_path / "saved"))
    got = load_file(str(exported / ADAPTER_SAFETENSORS_FILE))
    assert not saved.keys() & want.keys(), "premise: the saved directory is not the form the engine loads"
    assert any(".model.llm.layers." in key for key in want), "premise: the release's names"
    assert (".experts.w13_weight.lora_B.weight" in " ".join(want)) == (experts is not None)
    assert got.keys() == want.keys(), sorted(got.keys() ^ want.keys())[:4]
    differ = sorted(key for key in want if got[key].dtype != want[key].dtype or not torch.equal(got[key], want[key]))
    assert not differ, f"{len(differ)} tensor(s) differ from the published ones, e.g. {differ[:3]}"
    assert json.loads((exported / ADAPTER_CONFIG_FILE).read_text()) == json.loads(
        (published / ADAPTER_CONFIG_FILE).read_text()
    )


def _rewrite_config(adapter_dir: Path, **fields) -> None:
    config = read_adapter_config(str(adapter_dir))
    for key, value in fields.items():
        if isinstance(value, dict) and isinstance(config.get(key), dict):
            config[key] = {**config[key], **value}
        else:
            config[key] = value
    (adapter_dir / ADAPTER_CONFIG_FILE).write_text(json.dumps(config))


def _saved_mixed(tmp_path: Path, family: TinyFamily = _INKLING2) -> Path:
    peft_model = _trained(tmp_path / "base", experts=_SPEC, family=family)
    _save(peft_model, tmp_path / "saved", _SPEC)
    return tmp_path / "saved"


# GptOss's own class holds the ``gate_up_proj`` and ``down_proj`` its EP layer adapts. Qwen3 MoE's EP layer adapts
# ``gate_proj`` and ``up_proj`` apart where its own class fuses them, so those keys name nothing in the base. The rank
# case keeps the attention adapters' scaling.
@pytest.mark.parametrize(
    ("family", "backend", "fields", "reason"),
    [
        (TINY_MOE_FAMILIES["gpt_oss"], "sglang", {}, "serves no routed-expert adapter on this model"),
        (TINY_MOE_FAMILIES["qwen3_moe"], "sglang", {}, "serves no routed-expert adapter on this model"),
        (_INKLING2, "vllm", {EXPERT_LORA_CONFIG_KEY: {"lora_alpha": _RANK}}, "another rank or scaling than the"),
        (
            _INKLING2,
            "vllm",
            {EXPERT_LORA_CONFIG_KEY: {"r": 2 * _RANK, "lora_alpha": 4 * _RANK}},
            "another rank or scaling than the",
        ),
        (_INKLING2, "vllm", {"use_dora": True}, "DoRA: the engine refuses it"),
        (_INKLING2, "vllm", {"rank_pattern": {"o_proj.linear": 4}}, "rank_pattern: the engine reads one rank"),
    ],
    ids=["sglang-experts", "sglang-experts-the-base-fuses", "expert-scaling", "expert-rank", "dora", "rank-pattern"],
)
def test_what_the_engine_cannot_serve_is_refused_with_its_reason_before_any_write(
    tmp_path, family, backend, fields, reason
):
    saved = _saved_mixed(tmp_path, family)
    _rewrite_config(saved, **fields)

    with pytest.raises(ValueError, match=reason):
        export_engine_adapter(str(saved), str(tmp_path / "exported"), rollout_backend=backend)

    assert not (tmp_path / "exported").exists()


@pytest.mark.parametrize(
    ("family", "reason"),
    [
        (_INKLING2, "serves its LoRA adapters beside the launch base on rollout_backend: vllm"),
        (TINY_MOE_FAMILIES["laguna"], "rollout_backend='sglang' cannot serve model_type"),
    ],
    ids=["validated-on-the-other-engine", "unservable-model-type"],
)
def test_a_family_adapter_sync_refuses_on_the_engine_is_refused(tmp_path, family, reason):
    """SGLang's client refuses no adapter of its own; the family's spec and the client's unservable model types do,
    as they refuse the run that would sync these adapters."""
    targets = [_output_projection(tiny_family_model(family))]
    peft_model = _trained(tmp_path / "base", experts=None, family=family, targets=targets)
    _save(peft_model, tmp_path / "saved", None)

    with pytest.raises(ValueError, match=reason):
        export_engine_adapter(str(tmp_path / "saved"), str(tmp_path / "exported"), rollout_backend="sglang")

    assert not (tmp_path / "exported").exists()


def test_a_dense_adapter_is_refused_for_vllm_and_keeps_its_names_for_sglang(tmp_path):
    """vLLM serves no adapter of the family; Qwen3's hub names are its module names, so SGLang's directory holds the
    saved keys with the run's ``lora_alpha / r`` of 2 folded into B."""
    peft_model = _trained(tmp_path / "base", experts=None, family=TINY_DENSE_FAMILY, targets=["q_proj"])
    _save(peft_model, tmp_path / "saved", None)

    with pytest.raises(ValueError, match=re.escape(VLLMWeightSyncClient.ADAPTER_SYNC_REFUSAL)):
        export_engine_adapter(str(tmp_path / "saved"), str(tmp_path / "exported"), rollout_backend="vllm")
    assert not (tmp_path / "exported").exists()

    exported = export_engine_adapter(str(tmp_path / "saved"), str(tmp_path / "exported"), rollout_backend="sglang")

    saved, got = (load_file(str(path / ADAPTER_SAFETENSORS_FILE)) for path in (tmp_path / "saved", exported))
    assert got.keys() == saved.keys()
    for key, tensor in saved.items():
        folded = tensor.float() * 2 if ".lora_B." in key else tensor
        assert torch.equal(got[key], folded.to(ADAPTER_WIRE_DTYPE)), key
    engine_config = json.loads((exported / ADAPTER_CONFIG_FILE).read_text())
    wanted = {"r": _RANK, "lora_alpha": _RANK, "target_modules": ["q_proj"]}
    assert {field: engine_config[field] for field in wanted} == wanted


def test_whole_tensors_beside_the_adapters_are_refused(tmp_path):
    """A run that grew its vocabulary saves the added embedding rows with its adapters."""
    peft_model = _trained(tmp_path / "base", experts=None, family=TINY_DENSE_FAMILY, targets=["q_proj"])
    _save(peft_model, tmp_path / "saved", None)
    weights = tmp_path / "saved" / ADAPTER_SAFETENSORS_FILE
    save_file(
        {**load_file(str(weights)), "base_model.model.model.embed_tokens.weight": torch.zeros(2, 4)}, str(weights)
    )

    with pytest.raises(ValueError, match="whole tensors beside the adapters"):
        export_engine_adapter(str(tmp_path / "saved"), str(tmp_path / "exported"), rollout_backend="sglang")

    assert not (tmp_path / "exported").exists()


_ATTENTION_TARGET_REMEDY = "lora_target_modules: [o_proj, gate_up_proj, down_proj], with o_proj.linear on Inkling 2"
_MERGE_REMEDY = "merge_peft_adapters.py folds it into its base"


def _saved_experts_alone(tmp_path: Path, family: TinyFamily) -> Path:
    torch.manual_seed(0)
    model = tiny_family_model(family).to(torch.bfloat16)
    model.save_pretrained(tmp_path / "base")
    model.config._name_or_path = str(tmp_path / "base")
    model = apply_ep_layers(model, single_process_ep_config(expert_lora=_SPEC))
    run = make_parallelism_config(world_size=1, gpus_per_node=1, expert_lora=_SPEC)
    save_expert_lora_adapters(
        save_context(model, run, has_ep_layers=True, has_expert_lora=True), str(tmp_path / "saved")
    )
    return tmp_path / "saved"


# Only vLLM serves routed-expert adapters, and only on Inkling 2: there an attention target beside the experts is
# served; elsewhere the mixed adapter is refused too, so retraining would end in a second refusal.
@pytest.mark.parametrize(
    ("family", "backend", "reasons", "remedy", "not_remedy"),
    [
        (_INKLING2, "vllm", [], _ATTENTION_TARGET_REMEDY, _MERGE_REMEDY),
        (
            TINY_MOE_FAMILIES["gpt_oss"],
            "sglang",
            ["serves no routed-expert adapter on this model"],
            _MERGE_REMEDY,
            _ATTENTION_TARGET_REMEDY,
        ),
        (
            TINY_MOE_FAMILIES["qwen3_moe"],
            "vllm",
            [VLLMWeightSyncClient.ADAPTER_SYNC_REFUSAL, "serves no routed-expert adapter on this model"],
            _MERGE_REMEDY,
            _ATTENTION_TARGET_REMEDY,
        ),
    ],
    ids=["served-beside-attention", "sglang-serves-no-expert-adapter", "vllm-serves-no-adapter-of-the-family"],
)
def test_an_expert_only_adapter_is_refused_with_the_remedy_the_engine_serves(
    tmp_path, family, backend, reasons, remedy, not_remedy
):
    saved = _saved_experts_alone(tmp_path, family)

    with pytest.raises(ValueError, match="expert adapters alone") as refused:
        export_engine_adapter(str(saved), str(tmp_path / "exported"), rollout_backend=backend)

    message = str(refused.value)
    assert all(reason in message for reason in reasons), message
    assert remedy in message and not_remedy not in message, message
    assert not (tmp_path / "exported").exists()


def _saved_dense(tmp_path: Path) -> Path:
    peft_model = _trained(tmp_path / "base", experts=None, family=TINY_DENSE_FAMILY, targets=["q_proj"])
    _save(peft_model, tmp_path / "saved", None)
    return tmp_path / "saved"


def _saved_mixed_at_another_expert_scaling(tmp_path: Path) -> Path:
    saved = _saved_mixed(tmp_path)
    _rewrite_config(saved, **{EXPERT_LORA_CONFIG_KEY: {"lora_alpha": _RANK}})
    return saved


# vLLM serves a dense Qwen3 adapter as no adapter and SGLang does; no engine serves gpt-oss's expert adapters; SGLang
# refuses Inkling 2 at the family gate, so its rank refusal on vLLM names no other engine.
@pytest.mark.parametrize(
    ("build", "backend", "remedy", "not_remedy"),
    [
        (
            _saved_dense,
            "vllm",
            f"Export it with --rollout_backend sglang, whose engine serves it, or serve it merged: {_MERGE_REMEDY}.",
            None,
        ),
        (
            partial(_saved_mixed, family=TINY_MOE_FAMILIES["gpt_oss"]),
            "sglang",
            f"Serve it merged instead: {_MERGE_REMEDY} where its streaming merge finds each expert layer",
            "--rollout_backend",
        ),
        (
            _saved_mixed_at_another_expert_scaling,
            "vllm",
            "merge_expert_lora_on_save: true folds it at the run's save where it does not",
            "--rollout_backend",
        ),
    ],
    ids=["dense-served-by-the-other-engine", "experts-no-engine-serves", "the-other-engine-refuses-the-family"],
)
def test_a_refusal_ends_with_the_path_that_remains(tmp_path, build, backend, remedy, not_remedy):
    saved = build(tmp_path)

    with pytest.raises(ValueError, match="cannot be served as an adapter") as refused:
        export_engine_adapter(str(saved), str(tmp_path / "exported"), rollout_backend=backend)

    message = str(refused.value)
    assert remedy in message, message
    assert not_remedy is None or not_remedy not in message, message


def test_an_adapter_config_without_use_rslora_exports_as_plain_lora(tmp_path):
    """PEFT reads an absent ``use_rslora`` as false, so the scaling folded into B is ``lora_alpha / r``."""
    saved = _saved_dense(tmp_path)
    config = read_adapter_config(str(saved))
    del config["use_rslora"]
    (saved / ADAPTER_CONFIG_FILE).write_text(json.dumps(config))

    exported = export_engine_adapter(str(saved), str(tmp_path / "exported"), rollout_backend="sglang")

    tensors, got = (load_file(str(path / ADAPTER_SAFETENSORS_FILE)) for path in (saved, exported))
    for key, tensor in tensors.items():
        folded = tensor.float() * 2 if ".lora_B." in key else tensor
        assert torch.equal(got[key], folded.to(ADAPTER_WIRE_DTYPE)), key


def test_an_adapter_config_without_its_base_names_what_to_set(tmp_path):
    saved = _saved_dense(tmp_path)
    _rewrite_config(saved, base_model_name_or_path=None)

    with pytest.raises(ValueError, match="names no base_model_name_or_path: set it to the checkpoint the adapter"):
        export_engine_adapter(str(saved), str(tmp_path / "exported"), rollout_backend="sglang")


def test_an_adapter_on_a_multimodal_bases_text_only_class_exports_through_that_class(tmp_path, caplog):
    """A ``text_only_model`` run adapts the composite checkpoint's text-only class, whose module paths the composite
    lacks: the base is built as that class, and the directory carries the class's names with the run's scaling."""
    torch.manual_seed(0)
    composite = tiny_family_model(TINY_MOE_VLM_FAMILIES["qwen3_5_moe"]).to(torch.bfloat16)
    composite.save_pretrained(tmp_path / "base")
    build_config = resolve_build_config(str(tmp_path / "base"), revision=None, trust_remote_code=False)
    text_only = resolve_auto_model_class(build_config, text_only=True).from_pretrained(
        str(tmp_path / "base"), config=resolve_text_only_config(build_config), dtype=torch.bfloat16
    )
    peft_model = get_peft_model(text_only, LoraConfig(r=_RANK, lora_alpha=2 * _RANK, target_modules=["o_proj"]))
    peft_model = peft_model.to(torch.bfloat16)
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_" in name:
                param.normal_(std=0.05)
    _save(peft_model, tmp_path / "saved", None)

    with caplog.at_level(logging.INFO, logger=engine_adapter_export.__name__):
        exported = export_engine_adapter(str(tmp_path / "saved"), str(tmp_path / "exported"), rollout_backend="sglang")

    saved, got = (load_file(str(path / ADAPTER_SAFETENSORS_FILE)) for path in (tmp_path / "saved", exported))
    assert saved and all(key.startswith("base_model.model.model.layers.") for key in saved), sorted(saved)
    assert not hasattr(composite.model, "layers"), "premise: the composite holds no module the adapter addresses"
    assert got.keys() == saved.keys()
    for key, tensor in saved.items():
        folded = tensor.float() * 2 if ".lora_B." in key else tensor
        assert torch.equal(got[key], folded.to(ADAPTER_WIRE_DTYPE)), key
    engine_config = json.loads((exported / ADAPTER_CONFIG_FILE).read_text())
    wanted = {"r": _RANK, "lora_alpha": _RANK, "target_modules": ["o_proj"]}
    assert {field: engine_config[field] for field in wanted} == wanted
    assert f"reading the base as {type(text_only).__name__}" in caplog.text, caplog.text


def test_an_existing_output_is_refused(tmp_path):
    saved = _saved_mixed(tmp_path)
    (tmp_path / "exported").mkdir()

    with pytest.raises(FileExistsError, match="written whole into a new path: name an output_dir that does not"):
        export_engine_adapter(str(saved), str(tmp_path / "exported"), rollout_backend="vllm")


def test_an_output_that_is_the_adapter_directory_is_refused_as_existing(tmp_path):
    """The export writes beside its source and renames into place, so it destroys nothing; the true refusal is that the
    path exists."""
    saved = _saved_mixed(tmp_path)
    before = sorted(path.name for path in saved.iterdir())

    with pytest.raises(FileExistsError, match="written whole into a new path"):
        export_engine_adapter(str(saved), str(saved), rollout_backend="vllm")

    assert sorted(path.name for path in saved.iterdir()) == before


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
