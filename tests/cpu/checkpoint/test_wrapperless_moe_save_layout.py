#!/usr/bin/env python
"""Wrapper-less MoE gathered saves must write the HUB expert layout, not the module-fused one.

transformers 5.16 loads several MoE families into a module-FUSED expert layout
(``experts.gate_up_proj [E,2M,H]``) and ``save_pretrained`` reverts to the per-expert hub layout at
write time. The toolkit's gathered writer bypasses ``save_pretrained``, so a wrapper-less run
(``use_grouped_gemm: false`` at ep1 — no EP layer, the FSDP2/CP/TP savers) that emits fused keys
produces an artifact vLLM 0.26.0 hard-fails on for GLM-4/LFM-2 and silently mis-loads for Laguna.
The writer applies ``revert_weight_conversion`` exactly when the model carries no EP layers.

That revert has a second, quieter failure mode. transformers filters ``model._weight_conversions``
down to the converters a load actually USED, and a source checkpoint already in the fused layout —
a toolkit save read back in the stage1 -> stage2 flow — matches the module tree key-for-key, so the
list comes back ``[]`` and the revert degenerates into an identity that re-emits the fused keys.
``None`` (not ``[]``) is the sentinel that makes transformers fall back to the family's DECLARED
mapping, so the writer swaps it in for an empty list and restores the model's own value afterwards.

    python tests/cpu/checkpoint/test_wrapperless_moe_save_layout.py
"""

import os
from types import SimpleNamespace

import pytest
import torch
from accelerate import PartialState
from safetensors.torch import save_file
from transformers import CONFIG_MAPPING, AutoModelForCausalLM

PartialState()  # save_model_config logs through accelerate's logger

from src.checkpoint.format import load_full_state_dict, write_gathered_checkpoint
from src.distributed.tensor_parallel.checkpoint import save_tp_model
from tests.common.checkpoint_io import written_keys


def _tiny_qwen3_moe():
    config = CONFIG_MAPPING["qwen3_moe"]()
    config.hidden_size = 32
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.head_dim = 8
    config.num_hidden_layers = 1
    config.intermediate_size = 64
    # Distinct from hidden_size and from 2x itself, so the fused (E, 2*moe_inter, hidden) layout
    # cannot be mistaken for its transpose when the per-expert split is checked.
    config.moe_intermediate_size = 24
    config.num_experts = 4
    config.num_experts_per_tok = 2
    config.vocab_size = 128
    config.tie_word_embeddings = False
    return AutoModelForCausalLM.from_config(config)


def _write_fused_checkpoint(model, output_dir: str) -> None:
    """Write the module-FUSED state dict as a checkpoint — what a save without the unfuse produces.

    Loading it back converts nothing (the keys already match the module tree), which is the whole
    point: it is the one source shape that leaves ``_weight_conversions`` empty.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.config.save_pretrained(output_dir)
    save_file(
        {k: v.contiguous().clone() for k, v in model.state_dict().items()},
        os.path.join(output_dir, "model.safetensors"),
        metadata={"format": "pt"},
    )


def _write_gathered(model, output_dir: str) -> None:
    ctx = SimpleNamespace(has_ep_layers=False, max_shard_size="5GB")
    state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    write_gathered_checkpoint(model, state_dict, output_dir, ctx.max_shard_size)


def _assert_hub_expert_layout(written: set[str]) -> None:
    fused = {k for k in written if k.endswith(("mlp.experts.gate_up_proj", "mlp.experts.down_proj"))}
    assert not fused, f"module-fused keys leaked into the artifact: {sorted(fused)}"
    assert any(k.endswith("mlp.experts.0.gate_proj.weight") for k in written), "hub per-expert keys missing"


def test_wrapperless_save_writes_hub_expert_layout(tmp_path):
    # bf16, matching the writer's save dtype, so the round-trip equality stays exact (this test
    # pins the LAYOUT — rounding noise from an fp32 fixture would mask a layout bug).
    model = _tiny_qwen3_moe().to(torch.bfloat16)
    state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    # Premise: 5.14 really stores this family module-fused — if this moves, the writer's revert
    # (and this test) must be re-decided, not silently skipped.
    assert any(k.endswith("mlp.experts.gate_up_proj") for k in state_dict), "premise: live module is fused"

    ctx = SimpleNamespace(has_ep_layers=False, max_shard_size="5GB")
    write_gathered_checkpoint(model, state_dict, str(tmp_path), ctx.max_shard_size)

    _assert_hub_expert_layout(written_keys(str(tmp_path)))

    # The artifact must round-trip bit-exact through from_pretrained (the conversion re-fuses).
    reloaded = AutoModelForCausalLM.from_pretrained(str(tmp_path), dtype=torch.bfloat16)
    fused_key = next(k for k in model.state_dict() if k.endswith("mlp.experts.gate_up_proj"))
    assert torch.equal(reloaded.state_dict()[fused_key], model.state_dict()[fused_key])


def test_a_fused_layout_source_still_writes_the_hub_layout(tmp_path):
    """The stage1 -> stage2 case: the source checkpoint is itself a fused-layout toolkit save.

    Nothing is converted at load, so ``_weight_conversions`` is ``[]`` — and an empty list is NOT the
    fallback sentinel, it is an empty transform set that renames every key to itself. Without the
    swap to ``None`` the revert runs and changes nothing, and stage 2 re-emits the same unservable
    fused artifact stage 1 produced.
    """
    source, out = tmp_path / "fused-source", tmp_path / "out"
    # bf16 end-to-end: the writer casts to the save dtype, and this test pins LAYOUT round-tripping —
    # a bf16 fixture keeps torch.equal exact instead of masking layout bugs behind rounding noise.
    origin = _tiny_qwen3_moe().to(torch.bfloat16)
    _write_fused_checkpoint(origin, str(source))

    model = AutoModelForCausalLM.from_pretrained(str(source), dtype=torch.bfloat16)
    conversions = model._weight_conversions
    assert conversions == [], f"premise: a fused-layout source uses no converter, got {conversions!r}"

    _write_gathered(model, str(out))

    _assert_hub_expert_layout(written_keys(str(out)))
    # Restored to the model's OWN value, not left on the sentinel: the model keeps training/saving
    # after this call, and a later save must see exactly what the load left behind.
    assert model._weight_conversions is conversions

    reloaded = AutoModelForCausalLM.from_pretrained(str(out), dtype=torch.bfloat16)
    for key, tensor in origin.state_dict().items():
        assert torch.equal(reloaded.state_dict()[key], tensor), f"{key} did not survive the unfuse/re-fuse"


def test_a_hub_layout_source_keeps_the_conversions_its_load_used(tmp_path):
    """The normal path is unchanged: a hub-layout source loads WITH converters, and the writer must
    leave that list alone (swapping in the sentinel would drop a conversion a real load needed)."""
    source, out = tmp_path / "hub-source", tmp_path / "out"
    origin = _tiny_qwen3_moe().to(torch.bfloat16)
    origin.save_pretrained(str(source))
    assert any(k.endswith("mlp.experts.0.gate_proj.weight") for k in written_keys(str(source))), (
        "premise: save_pretrained writes the per-expert hub layout"
    )

    model = AutoModelForCausalLM.from_pretrained(str(source), dtype=torch.bfloat16)
    conversions = model._weight_conversions
    assert conversions, "premise: a hub-layout load fuses the experts, so it USES converters"

    _write_gathered(model, str(out))

    _assert_hub_expert_layout(written_keys(str(out)))
    assert model._weight_conversions is conversions

    reloaded = AutoModelForCausalLM.from_pretrained(str(out), dtype=torch.bfloat16)
    for key, tensor in origin.state_dict().items():
        assert torch.equal(reloaded.state_dict()[key], tensor), f"{key} did not survive the unfuse/re-fuse"


def test_the_gathered_tp_writer_writes_the_hub_layout(tmp_path):
    """``save_tp_model`` streams through the shared gathered writer, so it applies
    the same revert seam — a wrapper-less MoE under pure TP otherwise exports the fused keys vLLM
    rejects (GLM-4/LFM-2) or silently drops (Laguna). Driven end to end here, not by asserting the
    call, so a TP path that stopped sharing the writer still fails."""
    model = _tiny_qwen3_moe()
    fused_key = next(k for k in model.state_dict() if k.endswith("mlp.experts.gate_up_proj"))
    expected = model.state_dict()[fused_key].to(torch.bfloat16)

    save_tp_model(model, str(tmp_path))

    _assert_hub_expert_layout(written_keys(str(tmp_path)))
    # The artifact round-trips through from_pretrained (the load conversion re-fuses the experts).
    reloaded = AutoModelForCausalLM.from_pretrained(str(tmp_path), dtype=torch.bfloat16)
    assert torch.equal(reloaded.state_dict()[fused_key], expected), "experts did not survive the unfuse/re-fuse"


def test_the_gathered_tp_writer_applies_the_save_dtype(tmp_path):
    """Under fp32 masters the TP save must not write raw fp32 while every other writer casts through
    save_dtype_caster — that makes a TP export differ from the same model's FSDP2/EP one. One policy
    everywhere: save dtype for weights, trained dtype for norm params."""
    model = _tiny_qwen3_moe()  # from_config → fp32 params
    assert model.get_input_embeddings().weight.dtype == torch.float32, "premise: fp32 masters"

    save_tp_model(model, str(tmp_path))

    state = load_full_state_dict(str(tmp_path))
    assert state["model.embed_tokens.weight"].dtype == torch.bfloat16, "weights must export at the save dtype"
    norm_key = next(k for k in state if k.endswith("input_layernorm.weight"))
    assert state[norm_key].dtype == torch.float32, "norm params keep their trained dtype"


def test_a_model_without_a_config_still_gets_the_normalized_safetensors_artifact(tmp_path):
    """Only the config write is gated on the model carrying a config. A ``torch.save`` short-circuit
    there would hand one caller a raw ``pytorch_model.bin`` at raw fp32 while every other gathered
    save writes safetensors at the save dtype — a silent per-path format and dtype split, in the one
    writer whose reason to exist is that the two cannot diverge. (The hub expert layout is the one
    thing such a model cannot get: the revert reads its config, and warns when it cannot.)"""
    model = _tiny_qwen3_moe()
    state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    del model.config  # a stage-like module: the tree, and its dtypes, without a config

    write_gathered_checkpoint(model, state_dict, str(tmp_path))

    assert not (tmp_path / "pytorch_model.bin").exists(), "a config-less model must not fall back to .bin"
    state = load_full_state_dict(str(tmp_path))
    assert state["model.embed_tokens.weight"].dtype == torch.bfloat16, "weights must export at the save dtype"
    norm_key = next(k for k in state if k.endswith("input_layernorm.weight"))
    assert state[norm_key].dtype == torch.float32, "norm params keep their trained dtype"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
