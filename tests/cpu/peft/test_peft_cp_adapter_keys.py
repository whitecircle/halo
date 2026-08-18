#!/usr/bin/env python
"""CP+LoRA adapter keys over a REAL CP-wrapped PeftModel: spelling, normalization, resume remap.

PEFT wraps OUTSIDE the CP wrapper, so every adapter key flows through
``UlyssesCPModelWrapper.state_dict()`` in a NESTED position: ``nn.Module.state_dict`` recursion
passes ``destination``/``prefix`` into the override. The trap this pins: an override that forwards
them to the inner model writes raw keys at the wrapper's own prefix — collapsing the wrapper's
``model.`` level and discarding the cleaned dict it built — so from the PeftModel root
``state_dict()`` spells adapter keys differently from ``named_parameters()`` /
``load_state_dict()``, and every LoRA/QLoRA+CP resume dies at adapter restore (the loader's
"CP-remap regression" guard).

Everything here is real — tiny Qwen3, real attention patching, real ``get_peft_model`` — because
monkeypatching the live-key source with hand-written keys certifies the remap against spellings
runtime never produces.

Run: python tests/cpu/peft/test_peft_cp_adapter_keys.py
"""

from __future__ import annotations

import itertools

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoConfig, AutoModelForCausalLM

from src.distributed.checkpoint.peft import PeftAdapterSaver, remap_cp_adapter_keys_to_live
from src.distributed.context_parallel.config import CPConfig
from src.distributed.context_parallel.validation import SUPPORTED_ATTN_IMPLEMENTATIONS
from src.distributed.context_parallel.wrapper import UlyssesCPModelWrapper
from src.models.structure import persistent_buffers

_LORA_KWARGS = {"r": 4, "lora_alpha": 8, "target_modules": ["q_proj", "v_proj"], "bias": "none"}


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
    # Built eager (flash-attn cannot instantiate on CPU) but declaring the implementation the CP
    # validator reads; no attention kernel ever runs here.
    model.config._attn_implementation = SUPPORTED_ATTN_IMPLEMENTATIONS[0]
    return model


def _cp_peft_model() -> PeftModel:
    """``PeftModel(LoraModel(UlyssesCPModelWrapper(Qwen3)))`` — the runtime order (PEFT outside CP).

    ``cp_size=1`` still patches every attention layer (the patch walk is size-independent), so the
    live keys carry the real ``.original_attention.`` and wrapper ``model.`` artifacts without
    needing a process group.
    """
    wrapper = UlyssesCPModelWrapper(_tiny_qwen3(), CPConfig(cp_size=1, world_size=1, gpus_per_node=1))
    return get_peft_model(wrapper, LoraConfig(**_LORA_KWARGS))


def _live_lora_keys(peft_model: PeftModel) -> set[str]:
    return {name for name, _ in peft_model.named_parameters() if ".lora_" in name}


def test_root_state_dict_matches_named_parameters_spelling():
    """The nn.Module contract from the PEFT root: every parameter spells identically in
    ``state_dict()`` and ``named_parameters()`` (recursion bypasses both overrides, so both must be
    raw). A nested override that collapses the wrapper's ``model.`` level in ``state_dict`` only
    splits the two spellings and breaks resume."""
    peft_model = _cp_peft_model()
    sd_keys = set(peft_model.state_dict())
    np_keys = {name for name, _ in peft_model.named_parameters()}

    missing = sorted(np_keys - sd_keys)
    assert not missing, f"state_dict respells {len(missing)} parameter names, e.g. {missing[:3]}"

    # Anti-vacuity: the CP artifacts this file is about are present in the live spelling.
    lora_keys = _live_lora_keys(peft_model)
    assert lora_keys, "no LoRA params were injected"
    assert all(".original_attention." in key for key in lora_keys)
    assert all(key.startswith("base_model.model.model.model.layers.") for key in lora_keys)


def test_saver_normalization_yields_vanilla_keys():
    """The portability contract: normalized CP adapter keys equal exactly what the same LoRA on the
    UNWRAPPED model spells — what lets a CP-trained adapter load for non-CP inference."""
    cp_saved = PeftAdapterSaver._normalize_cp_adapter_keys(get_peft_model_state_dict(_cp_peft_model()))
    vanilla_keys = set(get_peft_model_state_dict(get_peft_model(_tiny_qwen3(), LoraConfig(**_LORA_KWARGS))))
    assert set(cp_saved) == vanilla_keys


def test_normalize_strips_cp_artifacts_injectively():
    """Over the REAL live key set: normalization removes ``.original_attention.`` and the extra
    ``model.`` level, changes every CP key, and stays injective (the resume remap is unambiguous)."""
    live = sorted(name.replace(".default.", ".") for name in _live_lora_keys(_cp_peft_model()))
    normalized = [PeftAdapterSaver._normalize_cp_adapter_key(key) for key in live]

    assert len(set(normalized)) == len(live)
    assert all(n != k for n, k in zip(normalized, live, strict=True))
    assert all(".original_attention." not in key for key in normalized)
    assert all(key.startswith("base_model.model.model.layers.") for key in normalized)


def test_normalize_collapses_top_level_lm_head():
    """A ``modules_to_save`` lm_head sits ABOVE the backbone: one fewer ``model.`` level than
    backbone keys, so the backbone collapse misses it and the saved key would be unexpected on
    every non-CP load — the head silently stays base weights. Idempotent: an already-plain
    backbone key (3 ``model.`` levels ending in ``layers.``) must NOT re-collapse."""
    cp_head = "base_model.model.model.lm_head.modules_to_save.default.weight"
    plain_head = "base_model.model.lm_head.modules_to_save.default.weight"
    assert PeftAdapterSaver._normalize_cp_adapter_key(cp_head) == plain_head
    assert PeftAdapterSaver._normalize_cp_adapter_key(plain_head) == plain_head
    plain_backbone = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"
    assert PeftAdapterSaver._normalize_cp_adapter_key(plain_backbone) == plain_backbone


def test_resume_remap_recovers_live_keys():
    """``remap_cp_adapter_keys_to_live`` must land every saved (normalized) key in the live
    namespace the adapter loader resolves — ``named_parameters()`` spelling sans the adapter-name
    segment — with none left in the portable namespace."""
    peft_model = _cp_peft_model()
    saved = PeftAdapterSaver._normalize_cp_adapter_keys(get_peft_model_state_dict(peft_model))
    assert saved, "no adapter tensors to save"

    remapped = remap_cp_adapter_keys_to_live(saved, peft_model)

    expected_live = {name.replace(".default.", ".") for name in _live_lora_keys(peft_model)}
    assert set(remapped) == expected_live, "remap did not recover the exact live key set"
    # The remap actually changed the spelling — a no-op here would mean nothing was exercised.
    assert set(remapped) != set(saved)


def test_resume_roundtrip_loads_every_saved_tensor():
    """Loader-level round trip — save-normalized keys → remap → ``set_peft_model_state_dict``, the
    exact non-DTensor path ``_load_peft_adapter_state`` runs on resume. Zero unexpected keys and
    every live LoRA tensor equal to the saved one. A mis-spelled remap leaves 100% unexpected and the
    loader raises "CP-remap regression"."""
    trained = _cp_peft_model()
    with torch.no_grad():  # make adapters non-zero so a successful restore is detectable (B inits 0)
        for name, param in trained.named_parameters():
            if ".lora_" in name:
                param.add_(torch.randn_like(param))
    saved = PeftAdapterSaver._normalize_cp_adapter_keys(get_peft_model_state_dict(trained))

    fresh = _cp_peft_model()
    trained_lora = {n: p for n, p in trained.named_parameters() if ".lora_" in n}
    fresh_lora = {n: p for n, p in fresh.named_parameters() if ".lora_" in n}
    assert set(fresh_lora) == set(trained_lora)
    assert any(not torch.equal(fresh_lora[n], trained_lora[n]) for n in trained_lora), (
        "fresh adapters already equal trained — the restore below would be unfalsifiable"
    )

    remapped = remap_cp_adapter_keys_to_live(saved, fresh)
    load_result = set_peft_model_state_dict(fresh, remapped)
    unexpected = list(getattr(load_result, "unexpected_keys", None) or [])
    assert not unexpected, f"{len(unexpected)}/{len(remapped)} adapter keys unexpected, e.g. {unexpected[:3]}"

    not_restored = [n for n in trained_lora if not torch.equal(fresh_lora[n], trained_lora[n])]
    assert not not_restored, (
        f"{len(not_restored)}/{len(trained_lora)} adapter tensors not restored: {not_restored[:3]}"
    )


def test_resume_remap_covers_modules_to_save_buffers():
    """A ``modules_to_save`` clone serializes its persistent buffers (a router's balancing bias —
    the reason ``_resolve_adapter_state`` chains ``persistent_buffers``). The resume remap must map
    them back to live spelling exactly like parameters; a live map built from ``named_parameters()``
    alone leaves the buffer in portable spelling, the loader reports it unexpected, and the bias
    silently resumes at init."""
    model = _tiny_qwen3()
    model.lm_head.register_buffer("balancing_bias", torch.zeros(model.config.vocab_size))
    wrapper = UlyssesCPModelWrapper(model, CPConfig(cp_size=1, world_size=1, gpus_per_node=1))
    peft_model = get_peft_model(wrapper, LoraConfig(modules_to_save=["lm_head"], **_LORA_KWARGS))

    saved = PeftAdapterSaver._normalize_cp_adapter_keys(get_peft_model_state_dict(peft_model))
    assert any(key.endswith("balancing_bias") for key in saved), (
        "modules_to_save clone did not serialize its persistent buffer — the remap case is vacuous"
    )

    remapped = remap_cp_adapter_keys_to_live(saved, peft_model)
    live = {
        name.replace(".modules_to_save.default.", ".").replace(".default.", ".")
        for name, _ in itertools.chain(peft_model.named_parameters(), persistent_buffers(peft_model))
    }
    stranded = sorted(key for key in remapped if key.endswith("balancing_bias") and key not in live)
    assert not stranded, f"buffer keys left in portable spelling (resume drops them): {stranded}"


def test_resume_remap_covers_embedding_adapters():
    """PEFT stores an embedding adapter in a ``ParameterDict``, so its live name ends in a TRAILING
    ``.default`` (``…lora_embedding_A.default``) while the saved key carries none. A live map built
    by replacing ``".default."`` alone never strips that spelling, so the saved key matches nothing,
    passes through in portable spelling, and the loader drops it — the embedding adapter silently
    resumes at initialization on every CP+LoRA-on-embeddings run."""
    wrapper = UlyssesCPModelWrapper(_tiny_qwen3(), CPConfig(cp_size=1, world_size=1, gpus_per_node=1))
    peft_model = get_peft_model(wrapper, LoraConfig(**{**_LORA_KWARGS, "target_modules": ["embed_tokens"]}))
    live_embedding = {
        name for name, _ in peft_model.named_parameters() if "lora_embedding_" in name and name.endswith(".default")
    }
    assert live_embedding, "no ParameterDict embedding adapter was injected — the case is vacuous"

    saved = PeftAdapterSaver._normalize_cp_adapter_keys(get_peft_model_state_dict(peft_model))
    remapped = remap_cp_adapter_keys_to_live(saved, peft_model)

    expected = {name.removesuffix(".default") for name in live_embedding}
    assert expected <= set(remapped), (
        f"embedding adapter keys never reached live spelling: {sorted(expected - set(remapped))} "
        f"(remapped: {sorted(k for k in remapped if 'lora_embedding_' in k)})"
    )


def test_resume_remap_passes_through_unmatched_saved_key():
    """A saved key with no live match falls back to identity pass-through; downstream reports it
    unexpected and drops it (documented gap — a future warn/raise updates this test)."""
    orphan = "base_model.model.model.layers.99.self_attn.q_proj.lora_A.weight"
    assert remap_cp_adapter_keys_to_live({orphan: 123}, _cp_peft_model()) == {orphan: 123}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
