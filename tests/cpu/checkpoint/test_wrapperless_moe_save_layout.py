#!/usr/bin/env python
"""Wrapper-less MoE gathered saves must write the layout the model was loaded from.

transformers 5.16 loads several MoE families from a per-expert hub into a module-FUSED expert layout
(``experts.gate_up_proj [E,2M,H]``) and ``save_pretrained`` reverts to the per-expert hub layout at
write time. The toolkit's gathered writer bypasses ``save_pretrained``, so a wrapper-less run
(``use_grouped_gemm: false`` at ep1 — no EP layer, the FSDP2/CP/TP savers) that emits fused keys
produces an artifact vLLM 0.26.0 hard-fails on for GLM-4/LFM-2 and silently mis-loads for Laguna.
The writer applies ``revert_weight_conversion`` exactly when the model carries no EP layers.

transformers records in ``model._weight_conversions`` the converters a load actually USED. A source
already in the module layout converts nothing and records ``[]``: the Qwen3.5/3.6 hub ships its
experts fused, and transformers still registers the per-expert converter for the family. Reverting
an empty record through that registry mapping would save a fused-hub model per-expert, a layout
neither its hub, its ``save_pretrained`` nor its EP save writes, and one its own FSDP2 resume cannot
read back. Every saver writes what the load read; the registry mapping is only for a model no load
recorded (``None``: built from its config), as in transformers' own revert.

    python tests/cpu/checkpoint/test_wrapperless_moe_save_layout.py
"""

import os
from types import SimpleNamespace

import pytest
import torch
from accelerate import PartialState
from safetensors.torch import save_file
from transformers import (
    CONFIG_MAPPING,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    Qwen3_5MoeConfig,
    Qwen3_5MoeForConditionalGeneration,
)

PartialState()  # save_model_config logs through accelerate's logger

from src.checkpoint.format import (
    load_full_state_dict,
    save_pretrained_layout,
    write_gathered_checkpoint,
)
from src.distributed.checkpoint.save import save_fsdp2_checkpoint
from src.distributed.checkpoint.tp_save import save_tp_model
from tests.common.checkpoint_io import written_keys
from tests.common.models import TINY_QWEN35_MOE_CONFIG


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
    # Premise: the pinned transformers stores this family module-fused — if this moves, the writer's revert
    # (and this test) must be re-decided, not silently skipped.
    assert any(k.endswith("mlp.experts.gate_up_proj") for k in state_dict), "premise: live module is fused"

    ctx = SimpleNamespace(has_ep_layers=False, max_shard_size="5GB")
    write_gathered_checkpoint(model, state_dict, str(tmp_path), ctx.max_shard_size)

    _assert_hub_expert_layout(written_keys(str(tmp_path)))

    # The artifact must round-trip bit-exact through from_pretrained (the conversion re-fuses).
    reloaded = AutoModelForCausalLM.from_pretrained(str(tmp_path), dtype=torch.bfloat16)
    fused_key = next(k for k in model.state_dict() if k.endswith("mlp.experts.gate_up_proj"))
    assert torch.equal(reloaded.state_dict()[fused_key], model.state_dict()[fused_key])


def test_a_fused_layout_source_writes_the_layout_it_read(tmp_path):
    """A source already in the module layout records ``[]``: nothing was converted, so nothing is
    reverted. Swapping in the family's registry mapping would respell the experts into a layout the
    load never read, which is what turned the fused Qwen3.5/3.6 hub into a per-expert save."""
    source, out = tmp_path / "fused-source", tmp_path / "out"
    # bf16 end-to-end: the writer casts to the save dtype, and this test pins LAYOUT round-tripping —
    # a bf16 fixture keeps torch.equal exact instead of masking layout bugs behind rounding noise.
    origin = _tiny_qwen3_moe().to(torch.bfloat16)
    _write_fused_checkpoint(origin, str(source))

    model = AutoModelForCausalLM.from_pretrained(str(source), dtype=torch.bfloat16)
    conversions = model._weight_conversions
    assert conversions == [], f"premise: a fused-layout source uses no converter, got {conversions!r}"

    _write_gathered(model, str(out))

    assert written_keys(str(out)) == written_keys(str(source)), "the save must write the layout the load read"
    # The model keeps training and saving after this call; a later save must see what the load left.
    assert model._weight_conversions is conversions

    reloaded = AutoModelForCausalLM.from_pretrained(str(out), dtype=torch.bfloat16)
    for key, tensor in origin.state_dict().items():
        assert torch.equal(reloaded.state_dict()[key], tensor), f"{key} did not survive the round trip"


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


def _tiny_qwen3_5_moe_multimodal() -> Qwen3_5MoeForConditionalGeneration:
    vision = {"depth": 1, "hidden_size": 16, "intermediate_size": 16, "num_heads": 2}
    config = Qwen3_5MoeConfig(
        text_config=dict(TINY_QWEN35_MOE_CONFIG),
        vision_config={**vision, "out_hidden_size": TINY_QWEN35_MOE_CONFIG["hidden_size"]},
    )
    return Qwen3_5MoeForConditionalGeneration(config)


def _qwen3_5_hub_source(directory: str) -> None:
    """The Qwen3.5/3.6-35B-A3B hub layout: experts FUSED under the multimodal prefix."""
    _write_fused_checkpoint(_tiny_qwen3_5_moe_multimodal().to(torch.bfloat16), directory)


def _qwen3_moe_hub_source(directory: str) -> None:
    """The Qwen3-30B-A3B hub layout: one tensor per expert."""
    _tiny_qwen3_moe().to(torch.bfloat16).save_pretrained(directory)


def _text_only(keys: set[str]) -> set[str]:
    """``keys`` as a text-only load of the multimodal checkpoint saves them: no tower, no wrapper prefix."""
    return {k.replace("model.language_model.", "model.") for k in keys if not k.startswith("model.visual.")}


_SOURCES = {
    "qwen3_5_moe-hub-multimodal": (_qwen3_5_hub_source, AutoModelForImageTextToText, lambda keys: keys),
    "qwen3_5_moe-hub-text-only": (_qwen3_5_hub_source, AutoModelForCausalLM, _text_only),
    "qwen3_moe-hub": (_qwen3_moe_hub_source, AutoModelForCausalLM, lambda keys: keys),
}


def _save_fsdp2(model, output_dir: str) -> None:
    ctx = SimpleNamespace(
        model=model, is_save_rank=True, max_shard_size="5GB", training_checkpoint=False, tokenizer=None
    )
    save_fsdp2_checkpoint(ctx, output_dir)


def _save_pretrained(model, output_dir: str) -> None:
    with save_pretrained_layout(model):
        model.save_pretrained(output_dir)


_SAVERS = {
    "fsdp2": _save_fsdp2,
    "tp": save_tp_model,
    "gathered": _write_gathered,
    "save_pretrained": _save_pretrained,
}


@pytest.mark.parametrize("saver", _SAVERS)
@pytest.mark.parametrize("source", _SOURCES)
def test_every_saver_writes_the_layout_the_hub_ships(tmp_path, source, saver):
    """Each wrapper-less saver, and ``save_pretrained``, writes a hub-loaded model back in its hub's
    expert layout: fused for Qwen3.5/3.6 (whose load converts nothing), per-expert for Qwen3-MoE
    (whose load fuses). A Qwen3.5 save that came out per-expert is the layout its FSDP2 resume, which
    reads raw keys into the fused module, and the EP lazy loader under ``text_only_model`` cannot
    read back."""
    write_source, load_class, expected_keys = _SOURCES[source]
    source_dir, out = str(tmp_path / "hub"), str(tmp_path / "out")
    write_source(source_dir)
    hub_keys = written_keys(source_dir)
    fused_hub = any(k.endswith("mlp.experts.gate_up_proj") for k in hub_keys)
    assert fused_hub == source.startswith("qwen3_5_moe"), "premise: the hub layout this case stands for"

    model = load_class.from_pretrained(source_dir, dtype=torch.bfloat16)
    os.makedirs(out, exist_ok=True)
    _SAVERS[saver](model, out)

    assert written_keys(out) == expected_keys(hub_keys), f"{saver} wrote a layout the hub does not ship"
    reloaded = load_class.from_pretrained(out, dtype=torch.bfloat16)
    for key, tensor in model.state_dict().items():
        assert torch.equal(reloaded.state_dict()[key], tensor), f"{key} did not survive the {saver} round trip"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
