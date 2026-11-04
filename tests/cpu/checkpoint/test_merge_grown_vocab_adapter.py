#!/usr/bin/env python
"""The adapter tools grow the base to a grown-vocabulary adapter's tokenizer and fill every added row. Rows older
adapters lack are rebuilt as the checkpoint's mean, their run's frozen init."""

import logging
import os

import pytest
import torch
from peft import LoraConfig, PeftModel, get_peft_model
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer

from halo.checkpoint.format import ADAPTER_SAFETENSORS_FILE
from halo.checkpoint.tools.adapter_merge import merge_peft_adapter
from halo.checkpoint.tools.bf16_conversion import load_model
from tests.common.grown_vocab import (
    FITTING_TOKENS,
    GROWING_TOKENS,
    WORDS,
    build_word_level_checkpoint,
    grow_vocabulary,
)
from tests.common.tiny_models import (
    TINY_DENSE_FAMILY,
    TINY_QWEN3_VL_FAMILY,
    build_tiny_family_checkpoint,
    word_level_tokenizer,
)

HEAD_KEY = "base_model.model.lm_head.weight"


@pytest.fixture(scope="module", params=[True, False], ids=["tied", "untied"])
def base_dir(request, tmp_path_factory) -> str:
    directory = str(tmp_path_factory.mktemp("tied" if request.param else "untied"))
    build_word_level_checkpoint(directory, tied=request.param)
    return directory


def _train(
    base_dir: str,
    adapter_dir: str,
    added: list[str] | None,
    *,
    save_embedding_layers: bool,
    load_class=AutoModelForCausalLM,
) -> PeftModel:
    """Added rows and adapters moved off their init; returns the trained model."""
    model = load_class.from_pretrained(base_dir, dtype=torch.float32)
    rows = model.get_input_embeddings().num_embeddings
    tokenizer = grow_vocabulary(model, AutoTokenizer.from_pretrained(base_dir), added)
    peft_model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=["q_proj", "v_proj"]))
    generator = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for module in (peft_model.get_input_embeddings(), peft_model.get_output_embeddings()):
            grown = module.weight[rows:]
            grown.copy_(torch.randn(grown.shape, generator=generator))
        for name, param in peft_model.named_parameters():
            if "lora_B" in name:
                param.copy_(torch.randn(param.shape, generator=generator))
    peft_model.save_pretrained(adapter_dir, save_embedding_layers=save_embedding_layers)
    tokenizer.save_pretrained(adapter_dir)
    return peft_model


def _rows(module: torch.nn.Module, start: int) -> torch.Tensor:
    return module.weight[start:].detach().float()


def test_a_grown_vocabulary_merges_with_its_trained_added_rows(base_dir, tmp_path):
    adapter_dir, merged_dir = str(tmp_path / "adapter"), str(tmp_path / "merged")
    base_rows = AutoModelForCausalLM.from_pretrained(base_dir).get_input_embeddings().num_embeddings
    trained = _train(base_dir, adapter_dir, GROWING_TOKENS, save_embedding_layers=True)
    grown = trained.get_input_embeddings().num_embeddings
    assert grown == len(WORDS) + 2 + len(GROWING_TOKENS) > base_rows, "premise: the run grew the vocabulary"
    with pytest.raises(RuntimeError, match="size mismatch"):
        PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(base_dir), adapter_dir)

    merge_peft_adapter(adapter_dir, merged_dir, dtype=torch.float32, verbose=False)

    merged = AutoModelForCausalLM.from_pretrained(merged_dir, dtype=torch.float32)
    assert merged.get_input_embeddings().num_embeddings == grown == len(AutoTokenizer.from_pretrained(merged_dir))
    for role in ("get_input_embeddings", "get_output_embeddings"):
        want = _rows(getattr(trained, role)(), base_rows)
        assert torch.equal(_rows(getattr(merged, role)(), base_rows), want), role
    tied = trained.get_input_embeddings().weight is trained.get_output_embeddings().weight
    assert (merged.get_input_embeddings().weight is merged.get_output_embeddings().weight) == tied
    folded = merged.model.layers[0].self_attn.q_proj.weight
    unfolded = trained.get_base_model().model.layers[0].self_attn.q_proj.base_layer.weight
    assert not torch.equal(folded, unfolded), "the LoRA delta must be folded too"


def test_the_bf16_tool_loads_a_grown_vocabulary_adapter_onto_its_grown_base(base_dir, tmp_path):
    adapter_dir = str(tmp_path / "adapter")
    base_rows = AutoModelForCausalLM.from_pretrained(base_dir).get_input_embeddings().num_embeddings
    trained = _train(base_dir, adapter_dir, GROWING_TOKENS, save_embedding_layers=True)

    loaded = load_model(adapter_dir, "causal_lm", is_peft=True, dtype=torch.float32, trust_remote_code=False)

    assert torch.equal(
        _rows(loaded.get_input_embeddings(), base_rows), _rows(trained.get_input_embeddings(), base_rows)
    )
    assert torch.equal(
        _rows(loaded.get_output_embeddings(), base_rows), _rows(trained.get_output_embeddings(), base_rows)
    )


def _grow_lines(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records if "Growing the base's embeddings" in record.getMessage()]


@pytest.mark.parametrize(("added", "grows"), [(GROWING_TOKENS, True), (FITTING_TOKENS, False)], ids=["grown", "fits"])
def test_both_tools_say_when_they_grow_the_base(base_dir, tmp_path, caplog, added, grows):
    """The output holds more rows than its base_model_name_or_path, so each tool names the grow."""
    adapter_dir, merged_dir = str(tmp_path / "adapter"), str(tmp_path / "merged")
    base_rows = AutoModelForCausalLM.from_pretrained(base_dir).get_input_embeddings().num_embeddings
    trained = _train(base_dir, adapter_dir, added, save_embedding_layers=grows)
    prefix = f"Growing the base's embeddings from {base_rows} to {trained.get_input_embeddings().num_embeddings} rows"
    with caplog.at_level(logging.INFO, logger="halo.checkpoint.tools"):
        merge_peft_adapter(adapter_dir, merged_dir, dtype=torch.float32)
        merged = _grow_lines(caplog)
        caplog.clear()
        load_model(adapter_dir, "causal_lm", is_peft=True, dtype=torch.float32, trust_remote_code=False)
        loaded = _grow_lines(caplog)
    for lines in (merged, loaded):
        assert len(lines) == int(grows) and all(line.startswith(prefix) for line in lines), lines


def _checkpoint_mean(module: torch.nn.Module) -> torch.Tensor:
    return module.weight.detach().mean(dim=0, dtype=torch.float32)


def _rebuild_warnings(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records if record.name == "halo.checkpoint.tools.adapter_merge"]


def test_an_adapter_saved_without_its_added_rows_merges_them_rebuilt(base_dir, tmp_path, caplog):
    """PEFT writes this when it cannot tell the vocabulary grew (offline, before adapter saves carried the rows)."""
    adapter_dir, merged_dir = str(tmp_path / "adapter"), str(tmp_path / "merged")
    base = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.float32)
    rows = base.get_input_embeddings().num_embeddings
    _train(base_dir, adapter_dir, GROWING_TOKENS, save_embedding_layers=False)
    assert not any("embed_tokens" in key for key in load_file(os.path.join(adapter_dir, ADAPTER_SAFETENSORS_FILE)))

    with caplog.at_level(logging.WARNING, logger="halo.checkpoint.tools.adapter_merge"):
        merge_peft_adapter(adapter_dir, merged_dir, dtype=torch.float32, verbose=False)

    (warning,) = _rebuild_warnings(caplog)
    assert "holds no value for the added rows of the input embedding" in warning
    merged = AutoModelForCausalLM.from_pretrained(merged_dir, dtype=torch.float32)
    assert merged.get_input_embeddings().num_embeddings == len(AutoTokenizer.from_pretrained(merged_dir)) > rows
    for role in ("get_input_embeddings", "get_output_embeddings"):
        added = _rows(getattr(merged, role)(), rows)
        assert torch.equal(added, _checkpoint_mean(getattr(base, role)()).expand_as(added)), role


def test_an_adapter_missing_the_untied_head_rows_rebuilds_only_those(base_dir, tmp_path, caplog):
    """The rule is per role."""
    adapter_dir, merged_dir = str(tmp_path / "adapter"), str(tmp_path / "merged")
    base = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.float32)
    rows = base.get_input_embeddings().num_embeddings
    trained = _train(base_dir, adapter_dir, GROWING_TOKENS, save_embedding_layers=True)
    weights = os.path.join(adapter_dir, ADAPTER_SAFETENSORS_FILE)
    state = load_file(weights)
    assert state.pop(HEAD_KEY, None) is not None, "premise: the adapter saved the head under its own name"
    save_file(state, weights)
    tied = trained.get_input_embeddings().weight is trained.get_output_embeddings().weight

    with caplog.at_level(logging.WARNING, logger="halo.checkpoint.tools.adapter_merge"):
        merge_peft_adapter(adapter_dir, merged_dir, dtype=torch.float32, verbose=False)

    merged = AutoModelForCausalLM.from_pretrained(merged_dir, dtype=torch.float32)
    assert torch.equal(_rows(merged.get_input_embeddings(), rows), _rows(trained.get_input_embeddings(), rows))
    head = _rows(merged.get_output_embeddings(), rows)
    if tied:
        assert torch.equal(head, _rows(trained.get_output_embeddings(), rows)) and not _rebuild_warnings(caplog)
    else:
        assert torch.equal(head, _checkpoint_mean(base.get_output_embeddings()).expand_as(head))
        (warning,) = _rebuild_warnings(caplog)
        assert "added rows of the output head:" in warning and "input embedding" not in warning


def test_a_composite_config_merges_its_grown_vocabulary(tmp_path):
    """``vocab_size`` sits on its text config alone."""
    base_dir, adapter_dir, merged_dir = (str(tmp_path / name) for name in ("base", "adapter", "merged"))
    build_tiny_family_checkpoint(TINY_QWEN3_VL_FAMILY, base_dir, word_level_tokenizer(WORDS))
    load_class = TINY_QWEN3_VL_FAMILY.load_class
    rows = load_class.from_pretrained(base_dir).get_input_embeddings().num_embeddings
    trained = _train(base_dir, adapter_dir, GROWING_TOKENS, save_embedding_layers=True, load_class=load_class)

    merge_peft_adapter(adapter_dir, merged_dir, dtype=torch.float32, verbose=False)

    merged = load_class.from_pretrained(merged_dir, dtype=torch.float32)
    assert merged.get_input_embeddings().num_embeddings == len(AutoTokenizer.from_pretrained(merged_dir)) > rows
    for role in ("get_input_embeddings", "get_output_embeddings"):
        assert torch.equal(_rows(getattr(merged, role)(), rows), _rows(getattr(trained, role)(), rows)), role


def test_added_tokens_within_the_padding_merge_onto_the_base_unchanged(base_dir, tmp_path):
    """Anti-vacuity: the base grows only where the run did, past its padded rows."""
    adapter_dir, merged_dir = str(tmp_path / "adapter"), str(tmp_path / "merged")
    base = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.float32)
    trained = _train(base_dir, adapter_dir, FITTING_TOKENS, save_embedding_layers=False)
    assert trained.get_input_embeddings().num_embeddings == base.get_input_embeddings().num_embeddings

    merge_peft_adapter(adapter_dir, merged_dir, dtype=torch.float32, verbose=False)

    merged = AutoModelForCausalLM.from_pretrained(merged_dir, dtype=torch.float32)
    assert torch.equal(merged.get_input_embeddings().weight, base.get_input_embeddings().weight)


def test_a_base_whose_own_tokenizer_outruns_its_rows_merges_unchanged(tmp_path):
    """A truncated vocabulary's tokenizer already outruns its rows and trains without a grow, so the adapter carries
    no embedding."""
    base_dir, adapter_dir, merged_dir = (str(tmp_path / name) for name in ("base", "adapter", "merged"))
    build_tiny_family_checkpoint(
        TINY_DENSE_FAMILY, base_dir, overrides={"vocab_size": 64, "pad_token_id": 1, "eos_token_id": 1}
    )
    word_level_tokenizer(WORDS).save_pretrained(base_dir)
    base = AutoModelForCausalLM.from_pretrained(base_dir, dtype=torch.float32)
    rows = base.get_input_embeddings().num_embeddings
    assert len(AutoTokenizer.from_pretrained(base_dir)) > rows, "premise: the base's tokenizer outruns its rows"
    _train(base_dir, adapter_dir, None, save_embedding_layers=False)

    merge_peft_adapter(adapter_dir, merged_dir, dtype=torch.float32, verbose=False)

    merged = AutoModelForCausalLM.from_pretrained(merged_dir, dtype=torch.float32)
    assert torch.equal(merged.get_input_embeddings().weight, base.get_input_embeddings().weight)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
