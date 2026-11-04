#!/usr/bin/env python
"""Wrapper-less MoE gathered saves write the layout the model was loaded from, via ``revert_weight_conversion``. A
``[]`` record means nothing converted; ``None`` means no load recorded."""

import os

import pytest
import torch
from safetensors.torch import save_file
from transformers import (
    CONFIG_MAPPING,
    AutoModelForCausalLM,
    AutoModelForImageTextToText,
    Qwen3_5MoeConfig,
    Qwen3_5MoeForConditionalGeneration,
)

from halo.checkpoint.hub_layout import save_pretrained_layout
from halo.checkpoint.parallel.save import save_fsdp2_checkpoint, save_tp_checkpoint
from halo.checkpoint.reading import load_full_state_dict
from tests.common.checkpoint_io import save_context, written_keys
from tests.common.models import TINY_QWEN35_MOE_CONFIG


def _tiny_qwen3_moe():
    config = CONFIG_MAPPING["qwen3_moe"]()
    config.hidden_size = 32
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.head_dim = 8
    config.num_hidden_layers = 1
    config.intermediate_size = 64
    # Distinct from hidden_size and from 2x itself, so the fused layout cannot be mistaken for its transpose.
    config.moe_intermediate_size = 24
    config.num_experts = 4
    config.num_experts_per_tok = 2
    config.vocab_size = 128
    config.tie_word_embeddings = False
    return AutoModelForCausalLM.from_config(config)


def _write_fused_checkpoint(model, output_dir: str) -> None:
    """Loading it back converts nothing, the one source shape that leaves ``_weight_conversions`` empty."""
    os.makedirs(output_dir, exist_ok=True)
    model.config.save_pretrained(output_dir)
    save_file(
        {k: v.contiguous().clone() for k, v in model.state_dict().items()},
        os.path.join(output_dir, "model.safetensors"),
        metadata={"format": "pt"},
    )


def _save_fsdp2(model, output_dir: str) -> None:
    save_fsdp2_checkpoint(save_context(model), output_dir)


def _assert_hub_expert_layout(written: set[str]) -> None:
    fused = {k for k in written if k.endswith(("mlp.experts.gate_up_proj", "mlp.experts.down_proj"))}
    assert not fused, f"module-fused keys leaked into the artifact: {sorted(fused)}"
    assert any(k.endswith("mlp.experts.0.gate_proj.weight") for k in written), "hub per-expert keys missing"


def test_wrapperless_save_writes_hub_expert_layout(tmp_path):
    # bf16 to match the save dtype, since rounding noise would mask a layout bug.
    model = _tiny_qwen3_moe().to(torch.bfloat16)
    state_dict = {k: v.clone() for k, v in model.state_dict().items()}
    # Premise: if the pinned transformers moves this, the writer's revert must be re-decided.
    assert any(k.endswith("mlp.experts.gate_up_proj") for k in state_dict), "premise: live module is fused"

    _save_fsdp2(model, str(tmp_path))

    _assert_hub_expert_layout(written_keys(str(tmp_path)))

    # The conversion re-fuses on load.
    reloaded = AutoModelForCausalLM.from_pretrained(str(tmp_path), dtype=torch.bfloat16)
    fused_key = next(k for k in model.state_dict() if k.endswith("mlp.experts.gate_up_proj"))
    assert torch.equal(reloaded.state_dict()[fused_key], model.state_dict()[fused_key])


def test_a_fused_layout_source_writes_the_layout_it_read(tmp_path):
    """Swapping in the registry mapping would turn the fused Qwen3.5/3.6 hub into a per-expert save."""
    source, out = tmp_path / "fused-source", tmp_path / "out"
    # bf16 keeps torch.equal exact.
    origin = _tiny_qwen3_moe().to(torch.bfloat16)
    _write_fused_checkpoint(origin, str(source))

    model = AutoModelForCausalLM.from_pretrained(str(source), dtype=torch.bfloat16)
    conversions = model._weight_conversions
    assert conversions == [], f"premise: a fused-layout source uses no converter, got {conversions!r}"

    _save_fsdp2(model, str(out))

    assert written_keys(str(out)) == written_keys(str(source)), "the save must write the layout the load read"
    # A later save must see what the load left.
    assert model._weight_conversions is conversions

    reloaded = AutoModelForCausalLM.from_pretrained(str(out), dtype=torch.bfloat16)
    for key, tensor in origin.state_dict().items():
        assert torch.equal(reloaded.state_dict()[key], tensor), f"{key} did not survive the round trip"


def test_a_hub_layout_source_keeps_the_conversions_its_load_used(tmp_path):
    """Swapping in the sentinel would drop a conversion a real load needed."""
    source, out = tmp_path / "hub-source", tmp_path / "out"
    origin = _tiny_qwen3_moe().to(torch.bfloat16)
    origin.save_pretrained(str(source))
    assert any(k.endswith("mlp.experts.0.gate_proj.weight") for k in written_keys(str(source))), (
        "premise: save_pretrained writes the per-expert hub layout"
    )

    model = AutoModelForCausalLM.from_pretrained(str(source), dtype=torch.bfloat16)
    conversions = model._weight_conversions
    assert conversions, "premise: a hub-layout load fuses the experts, so it USES converters"

    _save_fsdp2(model, str(out))

    _assert_hub_expert_layout(written_keys(str(out)))
    assert model._weight_conversions is conversions

    reloaded = AutoModelForCausalLM.from_pretrained(str(out), dtype=torch.bfloat16)
    for key, tensor in origin.state_dict().items():
        assert torch.equal(reloaded.state_dict()[key], tensor), f"{key} did not survive the unfuse/re-fuse"


def test_the_gathered_tp_writer_writes_the_hub_layout(tmp_path):
    """Otherwise it exports fused keys vLLM rejects (GLM-4/LFM-2) or silently drops (Laguna). End to end, so a TP
    path leaving the writer still fails."""
    model = _tiny_qwen3_moe()
    fused_key = next(k for k in model.state_dict() if k.endswith("mlp.experts.gate_up_proj"))
    expected = model.state_dict()[fused_key].to(torch.bfloat16)

    save_tp_checkpoint(save_context(model), str(tmp_path))

    _assert_hub_expert_layout(written_keys(str(tmp_path)))
    # The load conversion re-fuses the experts.
    reloaded = AutoModelForCausalLM.from_pretrained(str(tmp_path), dtype=torch.bfloat16)
    assert torch.equal(reloaded.state_dict()[fused_key], expected), "experts did not survive the unfuse/re-fuse"


def test_the_gathered_tp_writer_applies_the_save_dtype(tmp_path):
    """Raw fp32 would make a TP export differ from the same model's FSDP2/EP one."""
    model = _tiny_qwen3_moe()  # from_config → fp32 params.
    assert model.get_input_embeddings().weight.dtype == torch.float32, "premise: fp32 masters"

    save_tp_checkpoint(save_context(model), str(tmp_path))

    state = load_full_state_dict(str(tmp_path))
    assert state["model.embed_tokens.weight"].dtype == torch.bfloat16, "weights must export at the save dtype"
    norm_key = next(k for k in state if k.endswith("input_layernorm.weight"))
    assert state[norm_key].dtype == torch.float32, "norm params keep their trained dtype"


def test_a_model_without_a_config_still_gets_the_normalized_safetensors_artifact(tmp_path):
    """Only the config write is gated; the revert reads the config and warns when it cannot."""
    model = _tiny_qwen3_moe()
    del model.config  # A stage-like module: the tree and its dtypes, without a config.

    _save_fsdp2(model, str(tmp_path))

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
    """Experts fused under the multimodal prefix."""
    _write_fused_checkpoint(_tiny_qwen3_5_moe_multimodal().to(torch.bfloat16), directory)


def _qwen3_moe_hub_source(directory: str) -> None:
    """One tensor per expert."""
    _tiny_qwen3_moe().to(torch.bfloat16).save_pretrained(directory)


def _text_only(keys: set[str]) -> set[str]:
    return {k.replace("model.language_model.", "model.") for k in keys if not k.startswith("model.visual.")}


_SOURCES = {
    "qwen3_5_moe-hub-multimodal": (_qwen3_5_hub_source, AutoModelForImageTextToText, lambda keys: keys),
    "qwen3_5_moe-hub-text-only": (_qwen3_5_hub_source, AutoModelForCausalLM, _text_only),
    "qwen3_moe-hub": (_qwen3_moe_hub_source, AutoModelForCausalLM, lambda keys: keys),
}


def _save_tp(model, output_dir: str) -> None:
    save_tp_checkpoint(save_context(model), output_dir)


def _save_pretrained(model, output_dir: str) -> None:
    with save_pretrained_layout(model):
        model.save_pretrained(output_dir)


_SAVERS = {
    "fsdp2": _save_fsdp2,
    "tp": _save_tp,
    "save_pretrained": _save_pretrained,
}


@pytest.mark.parametrize("saver", _SAVERS)
@pytest.mark.parametrize("source", _SOURCES)
def test_every_saver_writes_the_layout_the_hub_ships(tmp_path, source, saver):
    """A per-expert Qwen3.5 save is a layout its FSDP2 resume and the EP lazy loader under ``text_only_model``
    cannot read back."""
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
