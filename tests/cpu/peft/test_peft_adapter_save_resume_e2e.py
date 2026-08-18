#!/usr/bin/env python
"""CP+PEFT adapter round trip through the REAL writer and the REAL loader, file on disk included.

``tests/cpu/peft/test_peft_cp_adapter_keys.py`` pins the key algebra (normalize / remap) and
``test_adapter_state_includes_buffers.py`` pins ``_resolve_adapter_state``, but both stop short of
the artifact: neither calls :meth:`PeftAdapterSaver.save`, and nothing anywhere calls
:func:`_load_peft_adapter_state`, the function resume actually runs. So the three
contracts that only hold END TO END are pinned nowhere else:

* a ``modules_to_save`` router's balancing buffer reaches ``adapter_model.safetensors`` — a filter
  that dropped buffers, or a writer branch that never saw them, leaves the serving bias at init;
* that buffer stays **fp32** in the adapter file while its LoRA neighbours cast to bf16 — the
  export-contract cast is applied by the writer, over keys the writer itself produced (the dtype
  test feeds hand-written keys, which cannot catch a respelling upstream of the cast);
* the saved keys load back onto a freshly built CP+PEFT model through the loader's own function,
  with zero unexpected keys — the shape the loader turns into a hard "adapter would resume from
  zero-init" failure.

Plus the provenance seam ``save`` owns: a stamped run must leave ``training_provenance.json`` next
to the adapter (a merge cannot recover the sinks policy from the tensors), an unstamped one must
leave none, and a non-save rank must write neither.

    python tests/cpu/peft/test_peft_adapter_save_resume_e2e.py
"""

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

from src.checkpoint.format import (
    ADAPTER_SAFETENSORS_FILE,
    PROVENANCE_GPT_OSS_SINKS,
    TRAINING_PROVENANCE_FILE,
)
from src.distributed.checkpoint.context import CheckpointContext
from src.distributed.checkpoint.peft import PeftAdapterSaver, _load_peft_adapter_state, remap_cp_adapter_keys_to_live
from src.distributed.context_parallel.config import CPConfig
from src.distributed.context_parallel.validation import SUPPORTED_ATTN_IMPLEMENTATIONS
from src.distributed.context_parallel.wrapper import UlyssesCPModelWrapper
from src.models.patches.gpt_oss_sinks import LIVE_SINKS_ATTR, SinksPolicy
from src.models.structure import persistent_buffers

_LORA_KWARGS = {"r": 4, "lora_alpha": 8, "target_modules": ["q_proj", "v_proj"], "bias": "none"}
NUM_EXPERTS = 6
# Values a bf16 round trip cannot reproduce: each needs more than 8 mantissa bits, so a cast that
# escaped the balancing exemption is visible as an inequality, not as a rounding of the last digit.
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
    """``PeftModel(LoraModel(UlyssesCPModelWrapper(Qwen3)))`` — the runtime order, PEFT outside CP.

    ``cp_size=1`` still patches every attention layer, so the live keys carry the real
    ``.original_attention.`` and wrapper ``model.`` artifacts without needing a process group.
    """
    wrapper = UlyssesCPModelWrapper(_tiny_qwen3(), CPConfig(cp_size=1, world_size=1, gpus_per_node=1))
    return get_peft_model(wrapper, LoraConfig(modules_to_save=["gate"], **_LORA_KWARGS))


def _context(peft_model: PeftModel, *, is_save_rank: bool = True) -> CheckpointContext:
    """The save context a CP+LoRA run builds: CP on, no EP, this rank writes."""
    return CheckpointContext(
        model=peft_model,
        parallelism_config=None,
        is_pp_mode=False,
        is_cp_mode=True,
        is_tp_mode=False,
        is_ep_tp_mode=False,
        has_ep_layers=False,
        fsdp_wrapped=False,
        accelerate_manages_fsdp=False,
        is_save_rank=is_save_rank,
        max_shard_size="5GB",
        save_sharded_ep=False,
        has_expert_lora=False,
        merge_expert_lora_on_save=False,
        cp_wrapper=None,
        tokenizer=None,
    )


def _train_adapters(peft_model: PeftModel) -> None:
    """Move every trainable adapter tensor off its init so a successful restore is detectable."""
    torch.manual_seed(7)
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if ".lora_" in name or ".modules_to_save." in name:
                param.add_(torch.randn_like(param))


def _save_adapter(peft_model: PeftModel, output_dir: str, **ctx_kwargs) -> dict[str, torch.Tensor]:
    assert PeftAdapterSaver().save(_context(peft_model, **ctx_kwargs), peft_model, output_dir)
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
    """The writer must put the ``modules_to_save`` router's balancing buffer in the adapter file.

    A saved adapter without it resumes — and serves, after a merge — on the pretrained routing bias
    while the run believed it had trained one.
    """
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
    """The written keys are what a non-CP load resolves: no ``.original_attention.``, no extra
    ``model.`` level. Asserted on the file, not on an intermediate dict — the CP branch of the
    writer is the only thing that applies the normalization on this path."""
    with tempfile.TemporaryDirectory() as tmp:
        state = _save_adapter(_cp_peft_model(), tmp)

    assert state, "nothing was written"
    assert all(".original_attention." not in key for key in state), sorted(state)[:3]
    assert all(not key.startswith("base_model.model.model.model.") for key in state), sorted(state)[:3]


def test_writer_keeps_the_balancing_buffer_fp32_while_lora_casts_to_bf16():
    """The export contract, measured on the artifact: 1e-3 sign steps are sub-eps in bf16, so a
    balancing bias rounded on the way out flips near-tied top-k picks on the served model. The LoRA
    half must still cast, or the exemption is indistinguishable from "the cast never ran"."""
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
    # bf16 could not round-trip these, so fp32 storage is load-bearing rather than incidental.
    assert any(not torch.equal(value, value.to(torch.bfloat16).to(torch.float32)) for value in balancing.values()), (
        "the pinned values survive a bf16 round trip — this test cannot see a lost exemption"
    )


def test_resume_restores_every_saved_tensor_through_the_loader():
    """Save → remap → :func:`_load_peft_adapter_state`, the exact resume path.

    Zero unexpected keys (the loader turns "all unexpected" into a hard raise and a partial miss
    into a warning that leaves those layers at init), and every trained tensor — LoRA weights AND
    the router's balancing buffer — equal to what was saved.
    """
    trained = _cp_peft_model()
    _train_adapters(trained)
    with tempfile.TemporaryDirectory() as tmp:
        saved = _save_adapter(trained, tmp)

    fresh = _cp_peft_model()
    live_before = _live_adapter_tensors(fresh)
    live_trained = _live_adapter_tensors(trained)
    assert set(live_before) == set(live_trained)
    assert any(not torch.equal(live_before[name], live_trained[name]) for name in live_trained), (
        "the fresh model already matches the trained one — the restore below would be unfalsifiable"
    )

    remapped = remap_cp_adapter_keys_to_live(saved, fresh)
    unexpected = _load_peft_adapter_state(fresh, remapped)
    assert not unexpected, f"{len(unexpected)}/{len(remapped)} saved keys unexpected, e.g. {unexpected[:3]}"

    # Exact, not approximate: the writer casts everything but the balancing tensors to the bf16 save
    # dtype, so the value resume owes back is the trained one put through that same round trip.
    live_after = _live_adapter_tensors(fresh)
    not_restored = [
        name
        for name in live_trained
        if not torch.equal(live_after[name], _through_save_dtype(name, live_trained[name]))
    ]
    assert not not_restored, f"{len(not_restored)}/{len(live_trained)} tensors not restored: {not_restored[:3]}"


def test_resume_restores_the_balancing_buffer_at_full_precision():
    """The buffer specifically — a loader whose live map came from ``named_parameters()`` alone
    reports it unexpected and silently leaves the router at its init bias."""
    trained = _cp_peft_model()
    _train_adapters(trained)
    with tempfile.TemporaryDirectory() as tmp:
        saved = _save_adapter(trained, tmp)

    fresh = _cp_peft_model()
    with torch.no_grad():  # clear it, so "restored" cannot mean "was already right"
        for _, buffer in persistent_buffers(fresh):
            buffer.zero_()

    unexpected = _load_peft_adapter_state(fresh, remap_cp_adapter_keys_to_live(saved, fresh))
    assert not unexpected

    # The ``modules_to_save`` clone is the copy the wrapped module's forward runs, so it is the one
    # that has to come back — the frozen original is never consulted again.
    restored = {
        name: buffer for name, buffer in persistent_buffers(fresh) if ".modules_to_save." in name and "gate." in name
    }
    assert len(restored) == 2, sorted(restored)
    for name, bias in restored.items():
        assert torch.equal(bias, torch.tensor(BALANCING_VALUES, dtype=torch.float32)), f"{name}: {bias}"


def test_save_writes_the_sinks_provenance_only_when_the_policy_stamped_the_model():
    """``save`` owns the provenance write: a merge rebuilds the base from the hub (sinks always
    live), so a neutralized-sinks run's record is the only thing that keeps the merged model's
    attention matching what the adapter trained under. An absent stamp means "nothing to record",
    never "neutralized", so an unstamped run must leave no file at all."""
    stamped = _cp_peft_model()
    setattr(stamped.get_base_model(), LIVE_SINKS_ATTR, SinksPolicy.NEUTRALIZED)
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
    """Every rank enters ``save`` (the gathers are collective) but only the save rank writes; a
    non-save rank writing would race the writer on a shared filesystem."""
    peft_model = _cp_peft_model()
    setattr(peft_model.get_base_model(), LIVE_SINKS_ATTR, SinksPolicy.NEUTRALIZED)
    with tempfile.TemporaryDirectory() as tmp:
        assert PeftAdapterSaver().save(_context(peft_model, is_save_rank=False), peft_model, tmp)
        assert not os.path.isfile(os.path.join(tmp, ADAPTER_SAFETENSORS_FILE))
        assert not os.path.isfile(os.path.join(tmp, TRAINING_PROVENANCE_FILE))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
