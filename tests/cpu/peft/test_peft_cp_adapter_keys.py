#!/usr/bin/env python
"""CP+LoRA adapter keys over a real CP-wrapped PeftModel: spelling, normalization, resume remap.
PEFT wraps outside CP, so ``UlyssesModel.state_dict`` runs nested and must not collapse its ``model.`` level."""

from __future__ import annotations

import itertools

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from peft.utils import get_peft_model_state_dict
from transformers import AutoConfig, AutoModelForCausalLM

from halo.checkpoint.parallel.adapters import (
    _load_peft_adapter_state,
    remap_cp_adapter_keys_to_live,
    saved_adapter_tensors,
)
from halo.distributed.context_parallel.config import CPConfig
from halo.distributed.context_parallel.model_rules import SUPPORTED_ATTN_IMPLEMENTATIONS
from halo.distributed.context_parallel.ulysses_model import UlyssesModel
from halo.models.param_roles import persistent_buffers
from halo.models.unwrapping import plain_adapter_key

_LORA_KWARGS = {"r": 4, "lora_alpha": 8, "target_modules": ["q_proj", "v_proj"], "bias": "none"}


def _plain_keys(state: dict) -> dict:
    return {plain_adapter_key(key): value for key, value in state.items()}


def _tiny_qwen3():
    config = AutoConfig.for_model(
        "qwen3",
        vocab_size=64,
        hidden_size=32,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=32,
        tie_word_embeddings=False,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config, dtype=torch.float32)
    # Built eager (flash-attn cannot instantiate on CPU) but declaring what the CP validator reads.
    model.config._attn_implementation = SUPPORTED_ATTN_IMPLEMENTATIONS[0]
    return model


def _cp_peft_model() -> PeftModel:
    """``PeftModel(LoraModel(UlyssesModel(Qwen3)))``, the runtime order. ``cp_size=1`` still patches attention."""
    wrapper = UlyssesModel(_tiny_qwen3(), CPConfig(1, stage_world_size=1, stage_base_rank=0, nvlink_domain_size=1))
    return get_peft_model(wrapper, LoraConfig(**_LORA_KWARGS))


def _live_lora_keys(peft_model: PeftModel) -> set[str]:
    return {name for name, _ in peft_model.named_parameters() if ".lora_" in name}


def test_root_state_dict_matches_named_parameters_spelling():
    """Recursion bypasses both overrides, so both must be raw; collapsing ``model.`` in one alone breaks resume."""
    peft_model = _cp_peft_model()
    sd_keys = set(peft_model.state_dict())
    np_keys = {name for name, _ in peft_model.named_parameters()}

    missing = sorted(np_keys - sd_keys)
    assert not missing, f"state_dict respells {len(missing)} parameter names, e.g. {missing[:3]}"

    # The CP artifacts must be present in the live spelling, or this test checks nothing.
    lora_keys = _live_lora_keys(peft_model)
    assert lora_keys, "no LoRA params were injected"
    assert all(".original_attention." in key for key in lora_keys)
    assert all(key.startswith("base_model.model.model.model.layers.") for key in lora_keys)


def test_saver_normalization_yields_vanilla_keys():
    """Keys equal the same LoRA's on the unwrapped model, so a CP-trained adapter loads for non-CP inference."""
    cp_saved = _plain_keys(get_peft_model_state_dict(_cp_peft_model()))
    vanilla_keys = set(get_peft_model_state_dict(get_peft_model(_tiny_qwen3(), LoraConfig(**_LORA_KWARGS))))
    assert set(cp_saved) == vanilla_keys


def test_normalize_strips_cp_artifacts_injectively():
    """Over the real live key set; injective, so the resume remap is unambiguous."""
    live = sorted(name.replace(".default.", ".") for name in _live_lora_keys(_cp_peft_model()))
    normalized = [plain_adapter_key(key) for key in live]

    assert len(set(normalized)) == len(live)
    assert all(n != k for n, k in zip(normalized, live, strict=True))
    assert all(".original_attention." not in key for key in normalized)
    assert all(key.startswith("base_model.model.model.layers.") for key in normalized)


def test_normalize_collapses_top_level_lm_head():
    """A ``modules_to_save`` lm_head has one fewer ``model.`` level, so the backbone collapse misses it and the head
    stays base weights on a non-CP load. A plain backbone key (3 ``model.`` levels) must not re-collapse."""
    cp_head = "base_model.model.model.lm_head.modules_to_save.default.weight"
    plain_head = "base_model.model.lm_head.modules_to_save.default.weight"
    assert plain_adapter_key(cp_head) == plain_head
    assert plain_adapter_key(plain_head) == plain_head
    plain_backbone = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    assert plain_adapter_key(plain_backbone) == plain_backbone


def test_resume_remap_recovers_live_keys():
    """Every saved key lands in the loader's namespace: ``named_parameters()`` spelling without the adapter name."""
    peft_model = _cp_peft_model()
    saved = _plain_keys(get_peft_model_state_dict(peft_model))
    assert saved, "no adapter tensors to save"

    remapped = remap_cp_adapter_keys_to_live(saved, saved_adapter_tensors(peft_model))

    expected_live = {name.replace(".default.", ".") for name in _live_lora_keys(peft_model)}
    assert set(remapped) == expected_live, "remap did not recover the exact live key set"
    assert set(remapped) != set(saved)


def test_resume_roundtrip_loads_every_saved_tensor():
    """A mis-spelled remap leaves the keys in portable spelling, which the copy cannot place."""
    trained = _cp_peft_model()
    with torch.no_grad():  # make adapters non-zero so a successful restore is detectable (B inits 0)
        for name, param in trained.named_parameters():
            if ".lora_" in name:
                param.add_(torch.randn_like(param))
    saved = _plain_keys(get_peft_model_state_dict(trained))

    fresh = _cp_peft_model()
    trained_lora = {n: p for n, p in trained.named_parameters() if ".lora_" in n}
    fresh_lora = {n: p for n, p in fresh.named_parameters() if ".lora_" in n}
    assert set(fresh_lora) == set(trained_lora)
    assert any(not torch.equal(fresh_lora[n], trained_lora[n]) for n in trained_lora), (
        "fresh adapters already equal trained — the restore below would be unfalsifiable"
    )

    saved_tensors = saved_adapter_tensors(fresh)
    remapped = remap_cp_adapter_keys_to_live(saved, saved_tensors)
    unplaced = sorted(set(remapped) - set(saved_tensors))
    assert not unplaced, f"{len(unplaced)}/{len(remapped)} adapter keys name no live tensor, e.g. {unplaced[:3]}"
    _load_peft_adapter_state(saved_tensors, remapped)

    not_restored = [n for n in trained_lora if not torch.equal(fresh_lora[n], trained_lora[n])]
    assert not not_restored, (
        f"{len(not_restored)}/{len(trained_lora)} adapter tensors not restored: {not_restored[:3]}"
    )


def test_resume_remap_covers_modules_to_save_buffers():
    """A live map from ``named_parameters()`` alone leaves a router's balancing bias unexpected, at its init."""
    model = _tiny_qwen3()
    model.lm_head.register_buffer("balancing_bias", torch.zeros(model.config.vocab_size))
    wrapper = UlyssesModel(model, CPConfig(1, stage_world_size=1, stage_base_rank=0, nvlink_domain_size=1))
    peft_model = get_peft_model(wrapper, LoraConfig(modules_to_save=["lm_head"], **_LORA_KWARGS))

    saved = _plain_keys(get_peft_model_state_dict(peft_model))
    assert any(key.endswith("balancing_bias") for key in saved), (
        "modules_to_save clone did not serialize its persistent buffer — the remap case is vacuous"
    )

    remapped = remap_cp_adapter_keys_to_live(saved, saved_adapter_tensors(peft_model))
    live = {
        name.replace(".modules_to_save.default.", ".").replace(".default.", ".")
        for name, _ in itertools.chain(peft_model.named_parameters(), persistent_buffers(peft_model))
    }
    stranded = sorted(key for key in remapped if key.endswith("balancing_bias") and key not in live)
    assert not stranded, f"buffer keys left in portable spelling (resume drops them): {stranded}"


def test_resume_remap_covers_embedding_adapters():
    """PEFT's embedding ``ParameterDict`` live name ends in a trailing ``.default``; replacing ``".default."`` alone
    would let the embedding adapter silently resume at initialization."""
    wrapper = UlyssesModel(_tiny_qwen3(), CPConfig(1, stage_world_size=1, stage_base_rank=0, nvlink_domain_size=1))
    peft_model = get_peft_model(wrapper, LoraConfig(**{**_LORA_KWARGS, "target_modules": ["embed_tokens"]}))
    live_embedding = {
        name for name, _ in peft_model.named_parameters() if "lora_embedding_" in name and name.endswith(".default")
    }
    assert live_embedding, "no ParameterDict embedding adapter was injected — the case is vacuous"

    saved = _plain_keys(get_peft_model_state_dict(peft_model))
    remapped = remap_cp_adapter_keys_to_live(saved, saved_adapter_tensors(peft_model))

    expected = {name.removesuffix(".default") for name in live_embedding}
    assert expected <= set(remapped), (
        f"embedding adapter keys never reached live spelling: {sorted(expected - set(remapped))} "
        f"(remapped: {sorted(k for k in remapped if 'lora_embedding_' in k)})"
    )


def test_resume_remap_covers_dora_magnitudes():
    """PEFT saves a DoRA magnitude as ``…lora_magnitude_vector``, without the adapter name and the live ``.weight``."""
    wrapper = UlyssesModel(_tiny_qwen3(), CPConfig(1, stage_world_size=1, stage_base_rank=0, nvlink_domain_size=1))
    peft_model = get_peft_model(wrapper, LoraConfig(**{**_LORA_KWARGS, "use_dora": True}))
    saved = _plain_keys(get_peft_model_state_dict(peft_model))
    magnitudes = [key for key in saved if key.endswith("lora_magnitude_vector")]
    assert magnitudes, "no DoRA magnitude was saved — the case is vacuous"

    saved_tensors = saved_adapter_tensors(peft_model)
    remapped = remap_cp_adapter_keys_to_live(saved, saved_tensors)
    assert set(remapped) == set(saved_tensors), sorted(set(remapped) ^ set(saved_tensors))[:4]


def test_resume_remap_passes_through_unmatched_saved_key():
    """The restore's layout check names and refuses it (``tests/cpu/peft/test_adapter_restore_strict.py``)."""
    orphan = "base_model.model.model.layers.99.self_attn.q_proj.lora_A.weight"
    assert remap_cp_adapter_keys_to_live({orphan: 123}, saved_adapter_tensors(_cp_peft_model())) == {orphan: 123}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
