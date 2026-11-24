#!/usr/bin/env python
"""The streaming adapter merge reproduces the in-memory merge tensor for tensor: PEFT's own fold for linear and DoRA
adapters (dense and composite bases, and ones whose hub layout respells the module tree), the gather's
merge-on-save fold for native expert LoRA on every expert layout, and the run's trained router balancing biases in
their native slots. A refusal leaves no output."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import pytest
import torch
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file, save_file
from transformers import (
    Qwen3_5Config,
    Qwen3_5ForCausalLM,
    Qwen3_5ForConditionalGeneration,
    Qwen3Config,
    Qwen3ForCausalLM,
)

from halo.checkpoint.adapters import is_expert_lora_key
from halo.checkpoint.format import (
    ADAPTER_SAFETENSORS_FILE,
    MERGED_ADAPTER_CONFIG_DIR,
    ROUTER_BALANCING_BIASES_FILE,
    TRAINING_PROVENANCE_FILE,
)
from halo.checkpoint.parallel.adapters import save_expert_lora_adapters, save_peft_adapters
from halo.checkpoint.parallel.ep import save_ep_gathered_checkpoint
from halo.checkpoint.sidecars import PROVENANCE_GPT_OSS_SINKS
from halo.checkpoint.tools import adapter_stream_merge
from halo.checkpoint.tools.adapter_merge import merge_peft_adapter
from halo.checkpoint.tools.adapter_stream_merge import _FUSED_GATE_UP_KINDS, _required_attrs
from halo.distributed.context_parallel.config import CPConfig
from halo.distributed.context_parallel.model_rules import SUPPORTED_ATTN_IMPLEMENTATIONS
from halo.distributed.context_parallel.ulysses_model import UlyssesModel
from halo.distributed.expert_parallel.moe_layer.expert_lora import EXPERT_LORA_RANK_MULTIPLE
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.model_loading.lazy_safetensors.meta_shell import MetaShellBuildError
from halo.models.attention.gpt_oss_sinks import SinksPolicy, apply_sinks_policy, is_sink_key, neutralized_sink_value
from halo.models.introspection.unwrapping import PEFT_BASE_MODEL_PREFIX
from halo.models.moe.balancing import is_balancing_state_key
from halo.models.moe.expert_layout import ExpertLayoutKind
from halo.models.moe.expert_lora_spec import ExpertLoraSpec
from tests.common.checkpoint_io import safetensors_state_dict, save_context
from tests.common.models import TINY_QWEN3_CONFIG, TINY_QWEN35_CONFIG
from tests.common.parallelism import make_parallelism_config, single_process_ep_config
from tests.common.tiny_models import (
    TINY_MOE_FAMILIES,
    TINY_MOE_VLM_FAMILIES,
    tiny_family_model,
    word_level_tokenizer,
)

# One family per expert layout: interleaved fused with biases, per-expert separate, fused, per-expert renamed.
_EXPERT_FAMILIES = ("gpt_oss", "qwen3_moe", "qwen3_5_moe_text", "laguna", "cohere2_moe")
_PROJECTIONS = {"all": None, "down only": frozenset({"down"}), "gate only": frozenset({"gate"})}
# The tiny families whose hub layout respells an expert layer's module path.
_HUB_RESPELLED_EXPERT_FAMILIES = (
    "deepseek_v4",
    "inkling_mm_model",
    "mistral3",
    "nemotron_h",
    "nemotron_h_omni",
    "step3p7",
)


def _assert_same_checkpoint(streamed: Path, reference: Path, *, changed: set[str] | None = None) -> None:
    got, want = safetensors_state_dict(streamed), safetensors_state_dict(reference)
    assert set(got) == set(want), f"key sets differ: {sorted(set(got) ^ set(want))[:6]}"
    for key, tensor in want.items():
        assert got[key].dtype == tensor.dtype, f"{key}: {got[key].dtype} vs {tensor.dtype}"
        assert torch.equal(got[key], tensor), f"{key} differs from the in-memory merge"
    if changed is not None:
        base = safetensors_state_dict(streamed.parent / "base")
        moved = {key for key in got if not torch.equal(got[key], base[key])}
        assert changed <= moved, f"premise: the adapter moves {sorted(changed - moved)[:4]}"


def _randomize_lora_b(model, std: float = 0.3) -> None:
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "lora_B" in name or "lora_magnitude" in name:
                param.normal_(std=std) if param.dim() > 1 else param.add_(torch.randn_like(param) * std)


def _dense_base(base_dir: Path, dtype: torch.dtype) -> None:
    torch.manual_seed(0)
    # Biased projections, so a LoRA bias has a base bias to fold into.
    config = Qwen3Config(**{**TINY_QWEN3_CONFIG, "num_hidden_layers": 3, "vocab_size": 96, "attention_bias": True})
    Qwen3ForCausalLM(config).to(dtype).save_pretrained(base_dir)


def _save_peft_adapter(model_cls, base_dir: Path, adapter_dir: Path, dtype: torch.dtype, **lora) -> set[str]:
    """A randomized adapter over ``base_dir``; returns the weight keys it targets."""
    model = model_cls.from_pretrained(base_dir, dtype=dtype)
    config = LoraConfig(r=4, lora_alpha=8, task_type="CAUSAL_LM", **{"target_modules": ["q_proj", "v_proj"], **lora})
    peft_model = get_peft_model(model, config)
    _randomize_lora_b(peft_model)
    peft_model.save_pretrained(adapter_dir)
    word_level_tokenizer(("hello", "world")).save_pretrained(adapter_dir)
    return {
        name.removeprefix("base_model.model.").replace(".base_layer", "")
        for name, _ in peft_model.named_parameters()
        if "base_layer.weight" in name
    }


def _merge_both(adapter_dir: Path, tmp_path: Path, dtype: torch.dtype) -> tuple[Path, Path]:
    streamed, reference = tmp_path / "streamed", tmp_path / "reference"
    merge_peft_adapter(str(adapter_dir), str(streamed), dtype=dtype, streaming=True, verbose=False)
    merge_peft_adapter(str(adapter_dir), str(reference), dtype=dtype, verbose=False)
    return streamed, reference


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16], ids=["fp32", "bf16"])
@pytest.mark.parametrize(
    "lora",
    [
        {},
        {"use_dora": True},
        {"use_rslora": True, "alpha_pattern": {"v_proj": 32}, "rank_pattern": {"q_proj": 2}},
        {"lora_bias": True},
    ],
    ids=["plain", "dora", "rslora-patterns", "lora-bias"],
)
def test_a_dense_linear_merge_equals_the_in_memory_merge(tmp_path, dtype, lora):
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    _dense_base(base, dtype)
    targets = _save_peft_adapter(Qwen3ForCausalLM, base, adapter, dtype, **lora)
    streamed, reference = _merge_both(adapter, tmp_path, dtype)

    _assert_same_checkpoint(streamed, reference, changed={f"{key}" for key in targets})
    assert (streamed / MERGED_ADAPTER_CONFIG_DIR / "adapter_config.json").is_file()


def test_a_merge_leaves_the_untargeted_tensors_as_the_base_stored_them(tmp_path):
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    _dense_base(base, torch.bfloat16)
    targets = _save_peft_adapter(Qwen3ForCausalLM, base, adapter, torch.bfloat16)
    streamed = tmp_path / "streamed"
    merge_peft_adapter(str(adapter), str(streamed), streaming=True, verbose=False)

    original, merged = safetensors_state_dict(base), safetensors_state_dict(streamed)
    assert targets and not any(torch.equal(merged[key], original[key]) for key in targets)
    assert all(torch.equal(merged[key], original[key]) for key in original if key not in targets)


def _composite_base(base_dir: Path) -> Qwen3_5Config:
    torch.manual_seed(0)
    vision = {"depth": 1, "hidden_size": 16, "intermediate_size": 16, "num_heads": 2}
    config = Qwen3_5Config(
        text_config={**TINY_QWEN35_CONFIG, "tie_word_embeddings": False},
        vision_config={**vision, "out_hidden_size": TINY_QWEN35_CONFIG["hidden_size"]},
    )
    Qwen3_5ForConditionalGeneration(config).save_pretrained(base_dir)
    word_level_tokenizer(("hello", "world")).save_pretrained(base_dir)
    return config


def test_a_composite_base_merge_equals_the_in_memory_merge(tmp_path):
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    _composite_base(base)
    targets = _save_peft_adapter(
        Qwen3_5ForConditionalGeneration, base, adapter, torch.float32, target_modules=["q_proj", "o_proj"]
    )
    assert any(".language_model." in key for key in targets), "premise: the adapter addresses the wrapper's tree"
    streamed, reference = _merge_both(adapter, tmp_path, torch.float32)

    _assert_same_checkpoint(streamed, reference, changed=targets)


def test_a_text_only_model_adapter_over_a_multimodal_base_is_refused(tmp_path):
    """A text_only_model run trains the composite's text class, whose module paths the multimodal base stores under
    its language model: no stored key is the adapter's own."""
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    text_only = Qwen3_5ForCausalLM(_composite_base(base).text_config)
    text_only.name_or_path = str(base)  # as the text_only_model load leaves it, where get_peft_model reads the path
    peft_model = get_peft_model(
        text_only, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    )
    _randomize_lora_b(peft_model)
    peft_model.save_pretrained(adapter)
    targets = sorted(
        name.removeprefix(PEFT_BASE_MODEL_PREFIX).replace(".base_layer", "")
        for name, _ in peft_model.named_parameters()
        if "base_layer.weight" in name
    )
    out = tmp_path / "out"

    with pytest.raises(ValueError, match=f"rewrites {len(targets)} tensor\\(s\\) the base") as refusal:
        merge_peft_adapter(str(adapter), str(out), streaming=True, verbose=False)
    assert repr(targets[0]) in str(refusal.value), str(refusal.value)
    assert not out.exists(), "a refused merge left an output directory behind"


@pytest.mark.parametrize(
    ("family", "target_modules"),
    [
        # The hub spells the KDA forget gate's projections without their ``forget_gate`` level, q_proj as the tree does.
        ("glm5_next", ["q_proj", "f_a_proj", "f_b_proj"]),
        # The hub's root is ``backbone``, the tree's ``model``.
        ("nemotron_h", ["q_proj", "in_proj"]),
        # The hub's ``attn.wq_a`` is the tree's ``self_attn.q_a_proj``.
        ("deepseek_v4", ["q_a_proj", "o_b_proj"]),
    ],
)
def test_a_module_the_base_stores_under_another_name_folds_into_that_key(tmp_path, family, target_modules):
    torch.manual_seed(0)
    tiny = TINY_MOE_FAMILIES[family]
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    tiny_family_model(tiny).save_pretrained(base)
    targets = _save_peft_adapter(tiny.load_class, base, adapter, torch.float32, target_modules=target_modules)
    streamed, reference = _merge_both(adapter, tmp_path, torch.float32)

    stored = safetensors_state_dict(base)
    moved = {key for key, tensor in safetensors_state_dict(reference).items() if not torch.equal(tensor, stored[key])}
    assert len(moved) == len(targets), f"premise: the in-memory merge moves every target, got {sorted(moved)}"
    assert moved - targets, "premise: the base stores a target under a key the adapter's module path does not spell"
    _assert_same_checkpoint(streamed, reference, changed=moved)


def _refuse_meta_shells(monkeypatch) -> None:
    """A base class the meta device cannot build from its config, as Ling-3.0's remote code reading the
    ``rope_scaling: null`` its hub config ships."""

    def refuse(*args, **kwargs):
        raise MetaShellBuildError("Could not build a config-only meta-device shell for the probe class.")

    monkeypatch.setattr(adapter_stream_merge, "instantiate_on_meta", refuse)


def test_a_base_the_meta_device_cannot_build_folds_under_the_keys_spelling_each_module_path(tmp_path, monkeypatch):
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    _dense_base(base, torch.float32)
    targets = _save_peft_adapter(Qwen3ForCausalLM, base, adapter, torch.float32)
    _refuse_meta_shells(monkeypatch)
    streamed, reference = _merge_both(adapter, tmp_path, torch.float32)

    _assert_same_checkpoint(streamed, reference, changed=targets)


def test_a_base_the_meta_device_cannot_build_is_refused_where_it_respells_a_module_path(tmp_path, monkeypatch):
    torch.manual_seed(0)
    tiny = TINY_MOE_FAMILIES["glm5_next"]
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    tiny_family_model(tiny).save_pretrained(base)
    _save_peft_adapter(tiny.load_class, base, adapter, torch.float32, target_modules=["f_a_proj"])
    _refuse_meta_shells(monkeypatch)
    out = tmp_path / "out"

    with pytest.raises(ValueError, match="module names off this shell"):
        merge_peft_adapter(str(adapter), str(out), streaming=True, verbose=False)
    assert not out.exists(), "a refused merge left an output directory behind"


def _expert_run(
    tmp_path: Path, family: str, projections, *, neutralize_sinks: bool = False
) -> tuple[Path, Path, Path]:
    """``(base, adapter, merged-on-save reference)`` of a tiny EP family with native expert LoRA. ``neutralize_sinks``
    applies the policy a ``reset_sinks`` run trains under after the base is saved, so the base keeps live sinks."""
    torch.manual_seed(0)
    rank = EXPERT_LORA_RANK_MULTIPLE
    spec = ExpertLoraSpec(r=rank, alpha=2 * rank, **({} if projections is None else {"projections": projections}))
    model = tiny_family_model({**TINY_MOE_FAMILIES, **TINY_MOE_VLM_FAMILIES}[family]).to(torch.bfloat16)
    model = apply_ep_layers(model, single_process_ep_config(expert_lora=spec))
    _randomize_lora_b(model)
    plain = make_parallelism_config(world_size=1, gpus_per_node=1, expert_lora=spec)
    merging = make_parallelism_config(world_size=1, gpus_per_node=1, expert_lora=spec, merge_expert_lora_on_save=True)
    base, adapter, reference = (tmp_path / name for name in ("base", "adapter", "reference"))
    save_ep_gathered_checkpoint(save_context(model, plain, has_ep_layers=True), str(base))
    model.config._name_or_path = str(base)
    if neutralize_sinks:
        apply_sinks_policy(model, model.config, policy=SinksPolicy.NEUTRALIZED)
    save_expert_lora_adapters(save_context(model, plain, has_ep_layers=True, has_expert_lora=True), str(adapter))
    save_ep_gathered_checkpoint(save_context(model, merging, has_ep_layers=True, has_expert_lora=True), str(reference))
    return base, adapter, reference


@pytest.mark.parametrize("projections", sorted(_PROJECTIONS))
@pytest.mark.parametrize("family", _EXPERT_FAMILIES)
def test_a_native_expert_lora_merge_equals_the_merge_on_save(tmp_path, family, projections):
    base, adapter, reference = _expert_run(tmp_path, family, _PROJECTIONS[projections])
    streamed = tmp_path / "streamed"
    # Automatic: the in-memory merge refuses this adapter.
    merge_peft_adapter(str(adapter), str(streamed), verbose=False)

    moved = {
        key
        for key, tensor in safetensors_state_dict(reference).items()
        if not torch.equal(tensor, safetensors_state_dict(base)[key])
    }
    assert moved and all(".experts." in key for key in moved), f"premise: only expert tensors move, got {moved}"
    _assert_same_checkpoint(streamed, reference, changed=moved)


@pytest.mark.parametrize("cp_size", [1, 2], ids=["no-cp", "cp"])
def test_a_mixed_run_adapter_folds_both_halves(tmp_path, cp_size):
    """The adapter the run's own saver writes: it walks from the PeftModel, so the expert keys carry its prefix, and
    under CP the wrapper's level too."""
    torch.manual_seed(0)
    spec = ExpertLoraSpec(r=EXPERT_LORA_RANK_MULTIPLE, alpha=2 * EXPERT_LORA_RANK_MULTIPLE)
    model = tiny_family_model(TINY_MOE_FAMILIES["qwen3_moe"]).to(torch.bfloat16)
    model = apply_ep_layers(model, single_process_ep_config(expert_lora=spec))
    plain = make_parallelism_config(world_size=1, gpus_per_node=1, expert_lora=spec)
    shape = {"world_size": cp_size, "gpus_per_node": cp_size, "cp_size": cp_size, "expert_lora": spec}
    run = make_parallelism_config(**shape)
    merging = make_parallelism_config(**shape, merge_expert_lora_on_save=True)
    base, adapter, reference, streamed = (tmp_path / name for name in ("base", "adapter", "reference", "streamed"))
    save_ep_gathered_checkpoint(save_context(model, plain, has_ep_layers=True), str(base))
    model.name_or_path = str(base)  # as from_pretrained leaves it, where get_peft_model reads the base's path
    if run.is_cp_mode:
        model.config._attn_implementation = SUPPORTED_ATTN_IMPLEMENTATIONS[0]
        model = UlyssesModel(model, CPConfig(1, stage_world_size=1, stage_base_rank=0, nvlink_domain_size=1))
    peft_model = get_peft_model(
        model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM")
    )
    _randomize_lora_b(peft_model)
    # A training checkpoint keeps the adapters' live dtype, so both references fold the tensors the file holds.
    mixed_run = {"has_ep_layers": True, "has_expert_lora": True, "training_checkpoint": True}
    save_peft_adapters(save_context(peft_model, run, **mixed_run), str(adapter))
    save_ep_gathered_checkpoint(save_context(peft_model, merging, **mixed_run), str(reference))
    saved = load_file(str(adapter / ADAPTER_SAFETENSORS_FILE))
    expert_keys = [key for key in saved if is_expert_lora_key(key)]
    prefix = f"{PEFT_BASE_MODEL_PREFIX}model.model." if run.is_cp_mode else PEFT_BASE_MODEL_PREFIX
    assert expert_keys and all(key.startswith(prefix) for key in expert_keys), sorted(saved)[:4]

    merge_peft_adapter(str(adapter), str(streamed), verbose=False)

    got, stored = safetensors_state_dict(streamed), safetensors_state_dict(base)
    expert_reference = safetensors_state_dict(reference)
    merged_attention = peft_model.merge_and_unload().state_dict()
    assert set(got) == set(stored)
    moved = {key for key in got if not torch.equal(got[key], stored[key])}
    attention = {key for key in got if key.endswith(("q_proj.weight", "v_proj.weight"))}
    experts = {key for key in got if ".experts." in key}
    assert attention and experts and moved == attention | experts, sorted(moved ^ (attention | experts))[:6]
    for key in attention:
        assert torch.equal(got[key], merged_attention[key]), f"{key} differs from PEFT's merge"
    for key in experts:
        assert torch.equal(got[key], expert_reference[key]), f"{key} differs from the merge-on-save"


@pytest.mark.parametrize("family", _HUB_RESPELLED_EXPERT_FAMILIES)
def test_an_expert_adapter_the_stream_cannot_fold_is_sent_to_the_merge_on_save(tmp_path, family):
    """The family's hub layout respells the expert layer's path, and the in-memory merge would drop expert deltas."""
    _, adapter, _ = _expert_run(tmp_path, family, None)
    expert_keys = [key for key in load_file(str(adapter / ADAPTER_SAFETENSORS_FILE)) if is_expert_lora_key(key)]
    assert expert_keys, "premise: the adapter holds expert LoRA"
    out = tmp_path / "out"

    with pytest.raises(ValueError, match="does not store") as refusal:
        merge_peft_adapter(str(adapter), str(out), verbose=False)
    assert "merge_expert_lora_on_save" in str(refusal.value), refusal.value
    assert "merges in memory" not in str(refusal.value), refusal.value
    assert not out.exists(), "a refused merge left an output directory behind"


def _write_balancing_sidecar(
    model, adapter_dir: Path, prefix: str = PEFT_BASE_MODEL_PREFIX
) -> dict[str, torch.Tensor]:
    """A bias_update run's ``router_balancing_biases.pt``: an fp32 bias per MoE block, keyed by the module path the
    run's saver walks (``prefix`` its PEFT wrapper's), at values bf16 does not hold."""
    generator = torch.Generator().manual_seed(1)
    biases = {
        f"{prefix}{name}": 0.125 + 1e-3 * torch.rand(module.experts.num_experts, generator=generator)
        for name, module in model.named_modules()
        if isinstance(getattr(module, "experts", None), torch.nn.Module)
    }
    torch.save(biases, adapter_dir / ROUTER_BALANCING_BIASES_FILE)
    return biases


def _adapter_with_balancing_sidecar(tmp_path: Path, family: str, dtype: torch.dtype, **overrides):
    """``(adapter, sidecar)``: attention LoRA over a tiny ``family`` base whose run trained balancing biases."""
    torch.manual_seed(0)
    tiny = TINY_MOE_FAMILIES[family]
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    model = tiny_family_model(tiny, overrides=overrides).to(dtype)
    model.save_pretrained(base)
    _save_peft_adapter(tiny.load_class, base, adapter, dtype)
    return adapter, _write_balancing_sidecar(model, adapter)


@pytest.mark.parametrize(
    ("family", "dtype"),
    # Laguna's hub respells the live slot gate.e_score_correction_bias as experts.e_score_correction_bias.
    [("gpt_oss", torch.bfloat16), ("glm5_next", torch.float32), ("laguna", torch.float32)],
)
def test_the_trained_balancing_biases_land_in_their_native_slots_as_the_in_memory_merge_writes_them(
    tmp_path, family, dtype
):
    adapter, sidecar = _adapter_with_balancing_sidecar(tmp_path, family, dtype)
    streamed, reference = _merge_both(adapter, tmp_path, dtype)

    slots = {key for key in safetensors_state_dict(tmp_path / "base") if is_balancing_state_key(key)}
    assert len(slots) == len(sidecar), f"premise: one stored slot per router, got {sorted(slots)}"
    _assert_same_checkpoint(streamed, reference, changed=slots)


def test_an_expert_adapters_trained_balancing_biases_land_in_the_native_slot_at_fp32(tmp_path):
    """A bias_update run training expert LoRA alone, whose adapter carries no tensor of the slot's module."""
    _, adapter, reference = _expert_run(tmp_path, "gpt_oss", None)
    sidecar = _write_balancing_sidecar(tiny_family_model(TINY_MOE_FAMILIES["gpt_oss"]), adapter, prefix="")
    streamed = tmp_path / "streamed"
    merge_peft_adapter(str(adapter), str(streamed), verbose=False)

    got, want = safetensors_state_dict(streamed), safetensors_state_dict(reference)
    trained = {f"{module}.router.bias": bias for module, bias in sidecar.items()}
    assert set(got) == set(want) and set(trained) <= set(want), sorted(set(trained) - set(want))[:4]
    for key, tensor in want.items():
        expected = trained.get(key, tensor)
        assert got[key].dtype == expected.dtype, f"{key}: {got[key].dtype} vs {expected.dtype}"
        assert torch.equal(got[key], expected), f"{key} differs"


def test_a_router_with_no_slot_drops_its_transient_bias_as_the_in_memory_merge_does(tmp_path, caplog):
    adapter, sidecar = _adapter_with_balancing_sidecar(tmp_path, "qwen3_moe", torch.float32)
    with caplog.at_level(logging.WARNING, logger=adapter_stream_merge.__name__):
        streamed, reference = _merge_both(adapter, tmp_path, torch.float32)

    _assert_same_checkpoint(streamed, reference)
    assert f"{len(sidecar)} router(s) trained a transient balancing bias" in caplog.text, caplog.text


@pytest.mark.parametrize(
    ("family", "overrides", "tamper", "match"),
    [
        # The in-memory merge builds the slot and sets the flag; the stream writes the base's config as stored.
        ("lfm2_moe", {"use_expert_bias": False}, None, "use_expert_bias: true"),
        ("gpt_oss", {}, "rename_module", "different model tree"),
        ("gpt_oss", {}, "reshape", "another expert count"),
    ],
    ids=["config-gated-slot", "absent-module", "reshaped"],
)
def test_a_balancing_sidecar_the_stream_cannot_fold_is_refused_before_anything_is_written(
    tmp_path, family, overrides, tamper, match
):
    adapter, sidecar = _adapter_with_balancing_sidecar(tmp_path, family, torch.float32, **overrides)
    if tamper == "rename_module":
        sidecar = {name.replace("layers.0.", "layers.99."): bias for name, bias in sidecar.items()}
    elif tamper == "reshape":
        sidecar = {name: torch.cat([bias, bias]) for name, bias in sidecar.items()}
    torch.save(sidecar, adapter / ROUTER_BALANCING_BIASES_FILE)
    out = tmp_path / "out"

    with pytest.raises(ValueError, match=match):
        merge_peft_adapter(str(adapter), str(out), streaming=True, verbose=False)
    assert not out.exists(), "a refused merge left an output directory behind"


_PROBE_AUTO_MAP = {"AutoConfig": "configuration_probe.ProbeConfig"}


@pytest.mark.parametrize(
    ("edit", "remote_code"),
    [
        ({"model_type": "halo_unregistered_probe", "auto_map": _PROBE_AUTO_MAP}, True),
        ({"model_type": "halo_unregistered_probe"}, False),
        # Untrusted, a registered type reads the config through transformers' own class, whose strict dataclass
        # refuses a field the remote class types otherwise (Ling-3.0's integer kda_lower_bound).
        ({"auto_map": _PROBE_AUTO_MAP, "rms_norm_eps": 1}, True),
    ],
    ids=["auto-map", "unknown-type", "auto-map-strict-native-class"],
)
def test_a_remote_code_base_is_refused_with_the_flag_that_trusts_it(tmp_path, edit, remote_code):
    """The dense merge builds the base's class, so a config only the base's own code reads needs that code trusted; a
    config transformers cannot read for another reason gets no such remedy."""
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    _dense_base(base, torch.float32)
    _save_peft_adapter(Qwen3ForCausalLM, base, adapter, torch.float32)
    config = json.loads((base / "config.json").read_text())
    (base / "config.json").write_text(json.dumps({**config, **edit}))
    out = tmp_path / "out"

    with pytest.raises(ValueError) as refusal:
        merge_peft_adapter(str(adapter), str(out), streaming=True, trust_remote_code=False, verbose=False)
    assert ("--trust_remote_code" in str(refusal.value)) is remote_code, refusal.value
    assert not out.exists(), "a refused merge left an output directory behind"


@pytest.mark.parametrize(
    ("tamper", "match"),
    [
        ("rename_module", "does not store"),
        ("embedding_lora", "does not fold"),
        ("grown_vocabulary", "resized"),
    ],
)
def test_a_merge_it_cannot_make_is_refused_before_anything_is_written(tmp_path, tamper, match):
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    _dense_base(base, torch.float32)
    _save_peft_adapter(Qwen3ForCausalLM, base, adapter, torch.float32)
    tensors = load_file(str(adapter / ADAPTER_SAFETENSORS_FILE))
    if tamper == "rename_module":
        tensors = {key.replace("layers.0.", "layers.9."): tensor for key, tensor in tensors.items()}
    elif tamper == "embedding_lora":
        tensors["base_model.model.model.embed_tokens.lora_embedding_A"] = torch.zeros(4, 96)
    else:
        tensors["base_model.model.lm_head.weight"] = torch.zeros(100, TINY_QWEN3_CONFIG["hidden_size"])
    save_file(tensors, str(adapter / ADAPTER_SAFETENSORS_FILE))
    out = tmp_path / "out"

    with pytest.raises(ValueError, match=match):
        merge_peft_adapter(str(adapter), str(out), streaming=True, verbose=False)
    assert not out.exists(), "a refused merge left an output directory behind"


def test_an_adapter_directory_without_a_weight_file_is_refused_by_name(tmp_path):
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    _dense_base(base, torch.float32)
    _save_peft_adapter(Qwen3ForCausalLM, base, adapter, torch.float32)
    (adapter / ADAPTER_SAFETENSORS_FILE).unlink()
    out = tmp_path / "out"

    with pytest.raises(FileNotFoundError, match=ADAPTER_SAFETENSORS_FILE):
        merge_peft_adapter(str(adapter), str(out), streaming=True, verbose=False)
    assert not out.exists()


def _adapter_recording_sinks(tmp_path: Path, policy: SinksPolicy) -> tuple[Path, Path]:
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    _dense_base(base, torch.float32)
    _save_peft_adapter(Qwen3ForCausalLM, base, adapter, torch.float32)
    (adapter / TRAINING_PROVENANCE_FILE).write_text(json.dumps({PROVENANCE_GPT_OSS_SINKS: str(policy)}))
    return adapter, tmp_path / "out"


def test_neutralized_sinks_recorded_for_a_base_without_sinks_refuse_the_merge_before_any_output(tmp_path):
    adapter, out = _adapter_recording_sinks(tmp_path, SinksPolicy.NEUTRALIZED)

    with pytest.raises(ValueError, match="stores no sink tensor"):
        merge_peft_adapter(str(adapter), str(out), streaming=True, verbose=False)
    assert not out.exists(), "a refused merge left an output directory behind"


def test_an_expert_only_save_records_the_sink_policy_the_run_trained_under(tmp_path):
    _base, adapter, _reference = _expert_run(tmp_path, "gpt_oss", None, neutralize_sinks=True)

    recorded = json.loads((adapter / TRAINING_PROVENANCE_FILE).read_text())
    assert recorded == {PROVENANCE_GPT_OSS_SINKS: SinksPolicy.NEUTRALIZED}


def test_an_expert_only_save_without_a_sink_policy_writes_no_provenance(tmp_path):
    _base, adapter, _reference = _expert_run(tmp_path, "qwen3_moe", None)

    assert not (adapter / TRAINING_PROVENANCE_FILE).exists()


def test_a_merge_writes_the_neutralized_sinks_the_run_trained_under(tmp_path):
    """The base stores live sinks and the merge-on-save of the same run stores neutralized ones, so the streamed merge
    must write each sink at its dtype minimum to give the trained model."""
    base, adapter, reference = _expert_run(tmp_path, "gpt_oss", None, neutralize_sinks=True)
    streamed = tmp_path / "streamed"
    merge_peft_adapter(str(adapter), str(streamed), verbose=False)

    stored, merged = safetensors_state_dict(base), safetensors_state_dict(streamed)
    sinks = [key for key in merged if is_sink_key(key)]
    assert sinks, "premise: the tiny GPT-OSS checkpoint stores sink tensors"
    for key in sinks:
        floor = neutralized_sink_value(merged[key].dtype)
        assert not (stored[key] == floor).all(), f"premise: the base stores {key} live"
        assert (merged[key] == floor).all(), f"{key} was merged live"
    _assert_same_checkpoint(streamed, reference)


def test_live_sinks_in_the_provenance_do_not_refuse_the_merge(tmp_path):
    adapter, out = _adapter_recording_sinks(tmp_path, SinksPolicy.LIVE)

    merge_peft_adapter(str(adapter), str(out), streaming=True, verbose=False)
    assert out.is_dir()


@pytest.mark.parametrize("option", [{"task": "classification"}, {"device_map": "cpu"}, {"fp32_norms": True}])
def test_options_only_a_loaded_model_honours_are_refused(tmp_path, option):
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    _dense_base(base, torch.float32)
    _save_peft_adapter(Qwen3ForCausalLM, base, adapter, torch.float32)

    with pytest.raises(ValueError, match="streaming merge loads no model"):
        merge_peft_adapter(str(adapter), str(tmp_path / "out"), streaming=True, verbose=False, **option)
    assert not (tmp_path / "out").exists()


def test_the_merged_checkpoint_carries_the_adapter_config_and_the_base_config(tmp_path):
    base, adapter = tmp_path / "base", tmp_path / "adapter"
    _dense_base(base, torch.float32)
    _save_peft_adapter(Qwen3ForCausalLM, base, adapter, torch.float32)
    out = tmp_path / "out"
    merge_peft_adapter(str(adapter), str(out), streaming=True, verbose=False)

    assert json.loads((out / "config.json").read_text())["model_type"] == "qwen3"
    assert (out / MERGED_ADAPTER_CONFIG_DIR / "adapter_config.json").is_file()
    assert not (out / "adapter_config.json").exists(), "an adapter config in the root reads the merge as an adapter"
    assert os.path.isfile(out / "tokenizer.json"), "the adapter's tokenizer did not ship"


def test_every_expert_layout_kind_has_a_streaming_attribute_set():
    """A new ``ExpertLayoutKind`` must be classified in the streaming merge, not fall into the fused-gate-up branch."""
    separate = {ExpertLayoutKind.SEPARATE_GLU, ExpertLayoutKind.SEPARATE_MLP}
    assert separate | _FUSED_GATE_UP_KINDS == set(ExpertLayoutKind)
    for kind in ExpertLayoutKind:
        assert _required_attrs(kind, {"down_proj"})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
