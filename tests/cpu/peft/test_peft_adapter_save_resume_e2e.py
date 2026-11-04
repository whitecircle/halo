#!/usr/bin/env python
"""CP+PEFT adapter round trip through the real writer and loader, file on disk included.
Pins what only holds end to end; the key algebra is in ``tests/cpu/peft/test_peft_cp_adapter_keys.py``."""

from __future__ import annotations

import itertools
import json
import os
import tempfile

import pytest
import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM

from halo.checkpoint.format import ADAPTER_SAFETENSORS_FILE, TRAINING_PROVENANCE_FILE
from halo.checkpoint.parallel.adapters import restore_adapters, save_peft_adapters
from halo.checkpoint.sidecars import PROVENANCE_GPT_OSS_SINKS
from halo.distributed.context_parallel.config import CPConfig
from halo.distributed.context_parallel.model_rules import SUPPORTED_ATTN_IMPLEMENTATIONS
from halo.distributed.context_parallel.ulysses_model import UlyssesModel
from halo.models.attention.gpt_oss_sinks import SINKS_POLICY_ATTR, SinksPolicy
from halo.models.param_roles import persistent_buffers
from tests.common.checkpoint_io import save_context
from tests.common.parallelism import make_parallelism_config

_LORA_KWARGS = {"r": 4, "lora_alpha": 8, "target_modules": ["q_proj", "v_proj"], "bias": "none"}
NUM_EXPERTS = 6
# A bf16 round trip can't reproduce these (more than 8 mantissa bits), so an escaped cast shows as an inequality.
BALANCING_VALUES = [0.10009766 + 1e-4 * i for i in range(NUM_EXPERTS)]


class _Router(nn.Module):
    """A gate carrying the balancing slot GLM-4/DeepSeek-V4 export: ``e_score_correction_bias``."""

    def __init__(self, hidden: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(NUM_EXPERTS, hidden))
        self.register_buffer("e_score_correction_bias", torch.tensor(BALANCING_VALUES, dtype=torch.float32))

    def forward(self, hidden_states):
        return hidden_states


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
    for layer in model.model.layers:
        layer.mlp.gate = _Router(config.hidden_size)
    return model


def _cp_peft_model() -> PeftModel:
    """``PeftModel(LoraModel(UlyssesModel(Qwen3)))``, the runtime order. ``cp_size=1`` still patches attention."""
    wrapper = UlyssesModel(_tiny_qwen3(), CPConfig(1, stage_world_size=1, stage_base_rank=0, nvlink_domain_size=1))
    return get_peft_model(wrapper, LoraConfig(modules_to_save=["gate"], **_LORA_KWARGS))


def _context(peft_model: PeftModel, *, is_save_rank: bool = True):
    """The save context a CP+LoRA run builds: CP on, no EP, this rank writes."""
    return save_context(
        peft_model, make_parallelism_config(world_size=2, gpus_per_node=2, cp_size=2), is_save_rank=is_save_rank
    )


def _train_adapters(peft_model: PeftModel) -> None:
    """Move every trainable adapter tensor off its init so a successful restore is detectable."""
    torch.manual_seed(7)
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if ".lora_" in name or ".modules_to_save." in name:
                param.add_(torch.randn_like(param))


def _save_adapter(peft_model: PeftModel, output_dir: str, **ctx_kwargs) -> dict[str, torch.Tensor]:
    save_peft_adapters(_context(peft_model, **ctx_kwargs), output_dir)
    return load_file(os.path.join(output_dir, ADAPTER_SAFETENSORS_FILE))


def _through_save_dtype(name: str, tensor: torch.Tensor) -> torch.Tensor:
    """What the writer stores for ``name``: bf16-rounded, except the exempt balancing tensors."""
    if name.endswith("e_score_correction_bias"):
        return tensor
    return tensor.to(torch.bfloat16).to(tensor.dtype)


def _live_adapter_tensors(peft_model: PeftModel) -> dict[str, torch.Tensor]:
    """Trainable adapter params plus the ``modules_to_save`` clones' persistent buffers."""
    return {
        name: tensor
        for name, tensor in itertools.chain(peft_model.named_parameters(), persistent_buffers(peft_model))
        if ".lora_" in name or ".modules_to_save." in name
    }


def test_saved_adapter_file_carries_the_router_balancing_buffer():
    """Without it the adapter resumes, and serves after a merge, on the pretrained routing bias."""
    peft_model = _cp_peft_model()
    _train_adapters(peft_model)
    with tempfile.TemporaryDirectory() as tmp:
        state = _save_adapter(peft_model, tmp)

    balancing = [key for key in state if key.endswith("e_score_correction_bias")]
    assert len(balancing) == 2, f"expected one balancing buffer per layer, got {sorted(state)}"
    assert any(".lora_A." in key for key in state), "premise: the file must also hold LoRA tensors"
    for key in balancing:
        assert torch.equal(state[key], torch.tensor(BALANCING_VALUES, dtype=torch.float32))


def test_saved_adapter_keys_are_portable_cp_normalized_spellings():
    """No ``.original_attention.``, no extra ``model.`` level; only the writer's CP branch normalizes on this path."""
    with tempfile.TemporaryDirectory() as tmp:
        state = _save_adapter(_cp_peft_model(), tmp)

    assert state, "nothing was written"
    assert all(".original_attention." not in key for key in state), sorted(state)[:3]
    assert all(not key.startswith("base_model.model.model.model.") for key in state), sorted(state)[:3]


def test_writer_keeps_the_balancing_buffer_fp32_while_lora_casts_to_bf16():
    """1e-3 sign steps are sub-eps in bf16, so a rounded bias flips near-tied top-k picks. LoRA still casts."""
    peft_model = _cp_peft_model()
    _train_adapters(peft_model)
    with tempfile.TemporaryDirectory() as tmp:
        state = _save_adapter(peft_model, tmp)

    balancing = {key: value for key, value in state.items() if key.endswith("e_score_correction_bias")}
    lora = {key: value for key, value in state.items() if ".lora_" in key}
    assert balancing and lora, "premise: the file must hold both kinds"
    assert all(value.dtype == torch.float32 for value in balancing.values()), {
        k: str(v.dtype) for k, v in balancing.items()
    }
    assert all(value.dtype == torch.bfloat16 for value in lora.values()), {k: str(v.dtype) for k, v in lora.items()}
    # bf16 could not round-trip these, so the fp32 storage is what is under test.
    assert any(not torch.equal(value, value.to(torch.bfloat16).to(torch.float32)) for value in balancing.values()), (
        "the pinned values survive a bf16 round trip — this test cannot see a lost exemption"
    )


def test_resume_restores_every_saved_tensor_through_the_loader():
    """A key that missed the remap would fail the restore's layout check; every trained tensor equals the saved one."""
    trained = _cp_peft_model()
    _train_adapters(trained)
    fresh = _cp_peft_model()
    live_before = _live_adapter_tensors(fresh)
    live_trained = _live_adapter_tensors(trained)
    assert set(live_before) == set(live_trained)
    assert any(not torch.equal(live_before[name], live_trained[name]) for name in live_trained), (
        "the fresh model already matches the trained one — the restore below would be unfalsifiable"
    )

    with tempfile.TemporaryDirectory() as tmp:
        _save_adapter(trained, tmp)
        assert restore_adapters(tmp, fresh, is_cp_mode=True) is not None

    # Exact: the writer casts all but the balancing tensors to bf16, so resume owes the trained value after that trip.
    live_after = _live_adapter_tensors(fresh)
    not_restored = [
        name
        for name in live_trained
        if not torch.equal(live_after[name], _through_save_dtype(name, live_trained[name]))
    ]
    assert not not_restored, f"{len(not_restored)}/{len(live_trained)} tensors not restored: {not_restored[:3]}"


def test_resume_restores_the_balancing_buffer_at_full_precision():
    """A live map from ``named_parameters()`` alone reports it unexpected and leaves the router at its init bias."""
    trained = _cp_peft_model()
    _train_adapters(trained)
    fresh = _cp_peft_model()
    with torch.no_grad():  # clear it, so a restore that did nothing fails
        for _, buffer in persistent_buffers(fresh):
            buffer.zero_()

    with tempfile.TemporaryDirectory() as tmp:
        _save_adapter(trained, tmp)
        assert restore_adapters(tmp, fresh, is_cp_mode=True) is not None

    # The ``modules_to_save`` clone is the copy the wrapped forward runs; the frozen original is never consulted again.
    restored = {
        name: buffer for name, buffer in persistent_buffers(fresh) if ".modules_to_save." in name and "gate." in name
    }
    assert len(restored) == 2, sorted(restored)
    for name, bias in restored.items():
        assert torch.equal(bias, torch.tensor(BALANCING_VALUES, dtype=torch.float32)), f"{name}: {bias}"


def test_save_writes_the_sinks_provenance_only_when_the_policy_stamped_the_model():
    """A merge rebuilds the base from the hub (sinks live), so this record keeps merged attention matching training.
    An unstamped run must leave no file."""
    stamped = _cp_peft_model()
    setattr(stamped.get_base_model(), SINKS_POLICY_ATTR, SinksPolicy.NEUTRALIZED)
    with tempfile.TemporaryDirectory() as tmp:
        _save_adapter(stamped, tmp)
        path = os.path.join(tmp, TRAINING_PROVENANCE_FILE)
        assert os.path.isfile(path), "a stamped run wrote no provenance — the merge cannot recover it"
        with open(path) as fh:
            assert json.load(fh)[PROVENANCE_GPT_OSS_SINKS] == SinksPolicy.NEUTRALIZED

    with tempfile.TemporaryDirectory() as tmp:
        _save_adapter(_cp_peft_model(), tmp)
        assert not os.path.isfile(os.path.join(tmp, TRAINING_PROVENANCE_FILE))


def test_non_save_rank_writes_no_adapter_and_no_provenance():
    """Every rank enters ``save`` (the gathers are collective); a non-save rank's write would race on a shared FS."""
    peft_model = _cp_peft_model()
    setattr(peft_model.get_base_model(), SINKS_POLICY_ATTR, SinksPolicy.NEUTRALIZED)
    with tempfile.TemporaryDirectory() as tmp:
        save_peft_adapters(_context(peft_model, is_save_rank=False), tmp)
        assert not os.path.isfile(os.path.join(tmp, ADAPTER_SAFETENSORS_FILE))
        assert not os.path.isfile(os.path.join(tmp, TRAINING_PROVENANCE_FILE))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
