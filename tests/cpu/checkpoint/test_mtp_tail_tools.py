#!/usr/bin/env python
"""What each checkpoint tool does to a stored MTP tail. The class-loading tools write back a head they have a spec for,
at the output's dtype, and on any other family drop the tail and say so with the config field they leave behind; the
streaming tools keep it. Real tools on tiny checkpoints."""

import json
import logging

import pytest
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoConfig

from halo.checkpoint.reading import iter_checkpoint_shard_entries
from halo.checkpoint.tools import model_merge, offline_export
from halo.checkpoint.tools.bf16_conversion import convert_to_bf16
from halo.checkpoint.tools.expert_unfuse import unfuse_checkpoint
from halo.checkpoint.tools.vocab_patch import patch_vocabulary
from halo.model_loading.model_preparation import auto_load_model
from tests.common.mtp_heads import run_entry, saved_tensors, write_entry_corpus, write_entry_run
from tests.common.tiny_models import (
    TINY_BAILING_LING3_MLA_FAMILY,
    TINY_MOE_FAMILIES,
    TINY_MOE_VLM_FAMILIES,
    tiny_family_model,
    word_level_tokenizer,
)

_TAIL = {
    "mtp.fc.weight": torch.full((128, 256), 1.5),
    "mtp.norm.weight": torch.full((128,), 1.5),
    "mtp.layers.0.mlp.experts.gate_up_proj": torch.full((8, 128, 128), 1.5),
    "mtp.layers.0.mlp.experts.down_proj": torch.full((8, 128, 64), 1.5),
}


def _checkpoint(path, *, tail: dict[str, torch.Tensor] | None = None, seed: int = 0) -> str:
    torch.manual_seed(seed)
    model = tiny_family_model(TINY_MOE_FAMILIES["qwen3_5_moe_text"])
    model.config.mtp_num_hidden_layers = 1
    model.save_pretrained(path)
    word_level_tokenizer(["hello", "world"]).save_pretrained(path)
    if tail:
        tensors = load_file(path / "model.safetensors")
        save_file({**tensors, **tail}, path / "model.safetensors", metadata={"format": "pt"})
    return str(path)


def _keys(path) -> set[str]:
    return {key for _shard, _reader, key in iter_checkpoint_shard_entries(str(path))}


def _dropped_tail_warnings(caplog) -> list[str]:
    return [record.getMessage() for record in caplog.records if "MTP head tensor(s)" in record.getMessage()]


@pytest.fixture(params=["convert_to_bf16", "patch_vocab"])
def class_loading_tool(request):
    if request.param == "convert_to_bf16":
        return lambda source, out: convert_to_bf16(source, out, "causal_lm", trust_remote_code=False)
    return lambda source, out: patch_vocabulary(source, out, patterns=[])


def test_a_class_loading_tool_carries_a_head_it_has_a_spec_for(class_loading_tool, tmp_path, caplog):
    source = _checkpoint(tmp_path / "source", tail=_TAIL)
    out = tmp_path / "out"
    with caplog.at_level(logging.WARNING, logger=offline_export.__name__):
        class_loading_tool(source, str(out))
    assert not _dropped_tail_warnings(caplog)
    # The save reverts the class's expert fusion across the state dict, so the bank lands as its trunk's does.
    written = saved_tensors(out)
    for key in ("mtp.fc.weight", "mtp.norm.weight", "mtp.layers.0.mlp.experts.7.gate_proj.weight"):
        assert written[key].dtype == torch.bfloat16, key
        assert torch.all(written[key] == 1.5), key
    assert json.loads((out / "config.json").read_text())["mtp_num_hidden_layers"] == 1
    reloaded = auto_load_model(str(out))
    assert not any(name.startswith("mtp.") for name, _param in reloaded.named_parameters())


def test_a_source_with_no_tail_converts_silently_and_gains_none(class_loading_tool, tmp_path, caplog):
    source = _checkpoint(tmp_path / "source")
    out = tmp_path / "out"
    with caplog.at_level(logging.WARNING, logger=offline_export.__name__):
        class_loading_tool(source, str(out))
    assert not _dropped_tail_warnings(caplog)
    assert not any(key.startswith("mtp.") for key in _keys(out))


def test_a_family_with_no_tail_never_looks_at_the_source(tmp_path, monkeypatch):
    """Off the MTP path a tool does what it did: a Hub id's snapshot would fetch files the load did not need."""
    looked = []

    def resolve(model_id):
        looked.append(model_id)
        return str(tmp_path)

    monkeypatch.setattr(offline_export, "resolve_checkpoint_source", resolve)
    for family, expected in (
        ("gpt_oss", []),
        # Its class names no tail, and its config's explicit 0 declares none.
        ("step3p7", []),
        ("qwen3_5_moe_text", ["org/repo"]),
        ("nemotron_h_omni", ["org/repo"]),
    ):
        looked.clear()
        model = tiny_family_model({**TINY_MOE_FAMILIES, **TINY_MOE_VLM_FAMILIES}[family])
        assert offline_export.carried_mtp_tail(model, "org/repo", str(tmp_path), tool="t", dtype=torch.bfloat16) == {}
        assert looked == expected, family


def test_a_tail_with_no_spec_is_dropped_with_the_count_and_the_field_it_leaves_behind(tmp_path, caplog):
    """A family refused by name: nothing says which of its tail's tensors follow the vocabulary."""
    torch.manual_seed(0)
    model = tiny_family_model(TINY_MOE_FAMILIES["deepseek_v4"])
    model.config.num_nextn_predict_layers = 1
    source = tmp_path / "source"
    model.save_pretrained(source, max_shard_size="100GB")
    word_level_tokenizer(["hello", "world"]).save_pretrained(source)
    tail_key = "mtp.0.e_proj.weight"
    tensors = load_file(source / "model.safetensors")
    save_file({**tensors, tail_key: torch.ones((2, 2))}, source / "model.safetensors", metadata={"format": "pt"})
    out = tmp_path / "out"
    with caplog.at_level(logging.WARNING, logger=offline_export.__name__):
        convert_to_bf16(str(source), str(out), "causal_lm", trust_remote_code=False)
    warned = _dropped_tail_warnings(caplog)
    assert len(warned) == 1 and "stores 1 MTP head tensor(s)" in warned[0], warned
    assert "num_nextn_predict_layers" in warned[0], warned
    assert tail_key not in _keys(out)


def _composite_source(path, family, tail_key: str, **overrides) -> str:
    torch.manual_seed(0)
    model = tiny_family_model(family, overrides=overrides)
    model.save_pretrained(path, max_shard_size="100GB")
    word_level_tokenizer(["hello", "world"]).save_pretrained(path)
    tensors = load_file(path / "model.safetensors")
    save_file({**tensors, tail_key: torch.ones((2, 2))}, path / "model.safetensors", metadata={"format": "pt"})
    return str(path)


def _bf16(source: str, out: str) -> None:
    convert_to_bf16(source, out, "causal_lm", trust_remote_code=False)


def _patched(source: str, out: str) -> None:
    patch_vocabulary(source, out, patterns=[], trust_remote_code=False)


@pytest.mark.parametrize(
    ("family", "tail_key", "overrides", "tool"),
    [
        (TINY_MOE_VLM_FAMILIES["nemotron_h_omni"], "mtp.layers.0.mixer.q_proj.weight", {}, _bf16),
        (TINY_MOE_VLM_FAMILIES["nemotron_h_omni"], "mtp.layers.1.mlp.gate.weight", {}, _patched),
        (TINY_MOE_FAMILIES["step3p7"], "model.layers.4.eh_proj.weight", {"num_nextn_predict_layers": 1}, _bf16),
    ],
    ids=["nemotron_omni_bf16", "nemotron_omni_patch_vocab", "step3p7_bf16"],
)
def test_a_composite_class_naming_no_tail_still_warns_on_the_tail_it_drops(
    family, tail_key, overrides, tool, tmp_path, caplog
):
    """A release's config declares the head and the composite class names none, so the output's config keeps a field
    with no head under it and a server drafting from it refuses."""
    source = _composite_source(tmp_path / "source", family, tail_key, **overrides)
    out = tmp_path / "out"
    with caplog.at_level(logging.WARNING, logger=offline_export.__name__):
        tool(source, str(out))
    warned = _dropped_tail_warnings(caplog)
    assert len(warned) == 1 and "stores 1 MTP head tensor(s)" in warned[0] and tail_key in warned[0], warned
    assert "num_nextn_predict_layers" in warned[0], warned
    assert tail_key not in _keys(out)


def test_a_config_declaring_no_tail_leaves_the_composites_keys_alone(tmp_path, caplog):
    source = _composite_source(
        tmp_path / "source",
        TINY_MOE_VLM_FAMILIES["nemotron_h_omni"],
        "mtp.layers.0.mixer.q_proj.weight",
        num_nextn_predict_layers=0,
    )
    with caplog.at_level(logging.WARNING, logger=offline_export.__name__):
        _bf16(source, str(tmp_path / "out"))
    assert not _dropped_tail_warnings(caplog)


def test_a_tail_the_class_builds_reads_as_no_dropped_head(tmp_path, caplog):
    """Ling-3.0's remote class holds its tail at the keys the depth declares, and its save writes them."""
    model = tiny_family_model(TINY_BAILING_LING3_MLA_FAMILY, overrides={"num_nextn_predict_layers": 1})
    source = tmp_path / "source"
    model.save_pretrained(source, max_shard_size="100GB")
    stored = _keys(source)
    assert any(f".layers.{model.config.num_hidden_layers}." in key for key in stored), "the tiny class built no tail"
    with caplog.at_level(logging.WARNING, logger=offline_export.__name__):
        carried = offline_export.carried_mtp_tail(
            model, str(source), str(tmp_path / "o"), tool="t", dtype=torch.bfloat16
        )
    assert carried == {}
    assert not _dropped_tail_warnings(caplog)


def test_a_quantized_tail_is_dropped_with_the_warning_as_it_holds_no_float_to_cast(tmp_path, caplog):
    source = _checkpoint(tmp_path / "source", tail=_TAIL)
    config_path = tmp_path / "source" / "config.json"
    config = json.loads(config_path.read_text())
    model = auto_load_model(source)
    config_path.write_text(json.dumps({**config, "quantization_config": {"quant_method": "fp8"}}))
    with caplog.at_level(logging.WARNING, logger=offline_export.__name__):
        carried = offline_export.carried_mtp_tail(model, source, str(tmp_path / "out"), tool="t", dtype=torch.bfloat16)
    assert carried == {}
    assert f"stores {len(_TAIL)} MTP head tensor(s)" in _dropped_tail_warnings(caplog)[0]


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    return write_entry_corpus(tmp_path_factory.mktemp("corpus") / "rows")


@pytest.mark.parametrize("name", ["qwen3_5_moe_text", "glm4_moe_lite"])
def test_a_trained_heads_conversion_holds_its_bytes_and_every_key(name, tmp_path, corpus):
    """Through a real run: the head a conversion is handed is the one it writes."""
    config = write_entry_run(name, tmp_path, corpus, steps=1, save_steps=1, mtp_loss_weight=0.3, mtp_loss_type="soft")
    done = run_entry(config)
    assert done.returncode == 0, (done.stderr + done.stdout)[-3000:]
    checkpoint = tmp_path / "run" / "checkpoint-1"
    trained = saved_tensors(checkpoint)
    out = tmp_path / "out"
    convert_to_bf16(str(checkpoint), str(out), "causal_lm", trust_remote_code=False, verify=True)
    written = saved_tensors(out)
    assert set(written) == set(trained)
    head = [key for key in trained if key.startswith("mtp." if name == "qwen3_5_moe_text" else "model.layers.2.")]
    assert head and all(torch.equal(written[key], trained[key].to(torch.bfloat16)) for key in head)


def test_a_vocabulary_patch_regrows_a_list_heads_trunk_copies(tmp_path, corpus):
    """The release's copies read the trunk's embeddings, which the patch grew; a stored copy at the old width fails the
    engine's load."""
    config = write_entry_run(
        "glm4_moe_lite", tmp_path, corpus, steps=1, save_steps=1, mtp_loss_weight=0.3, mtp_loss_type="soft"
    )
    done = run_entry(config)
    assert done.returncode == 0, (done.stderr + done.stdout)[-3000:]
    source = tmp_path / "run" / "checkpoint-1"
    trained = saved_tensors(source)
    out = tmp_path / "out"
    # Past the padding rows the tiny base's embedding carries beyond its tokenizer.
    patterns = [f"<|grown {index}|>" for index in range(64)]
    patch_vocabulary(str(source), str(out), patterns=patterns, trust_remote_code=False)
    written = saved_tensors(out)
    embedding = written["model.embed_tokens.weight"]
    assert embedding.shape[0] > trained["model.embed_tokens.weight"].shape[0], "the patch grew nothing"
    assert torch.equal(written["model.layers.2.embed_tokens.weight"], embedding)
    assert torch.equal(written["model.layers.2.shared_head.head.weight"], written["lm_head.weight"])
    assert torch.equal(
        written["model.layers.2.eh_proj.weight"], trained["model.layers.2.eh_proj.weight"].to(torch.bfloat16)
    )


def test_merged_fine_tunes_that_all_carry_a_head_merge_it(tmp_path):
    a = _checkpoint(tmp_path / "a", tail={key: value + 1 for key, value in _TAIL.items()}, seed=1)
    b = _checkpoint(tmp_path / "b", tail={key: value + 3 for key, value in _TAIL.items()}, seed=1)
    out = tmp_path / "merged"
    model_merge.merge_models([a, b], str(out), method="linear", allow_missing_tokenizer=True, verbose=False)
    merged = load_file(out / "model.safetensors")
    for key, value in _TAIL.items():
        assert torch.allclose(merged[key].float(), value + 2), key


def test_merged_fine_tunes_where_only_one_carries_a_head_are_refused(tmp_path):
    """Keeping it would mix one model's head with both trunks, and dropping it would lose a trained head silently."""
    a = _checkpoint(tmp_path / "a", tail=_TAIL, seed=1)
    b = _checkpoint(tmp_path / "b", seed=1)
    out = tmp_path / "merged"
    with pytest.raises(ValueError, match="does not carry the merge's reference key set"):
        model_merge.merge_models([a, b], str(out), method="linear", allow_missing_tokenizer=True, verbose=False)
    assert not out.exists()


def test_unfuse_splits_the_tails_experts_with_its_bytes(tmp_path):
    source = _checkpoint(tmp_path / "source", tail=_TAIL)
    out = tmp_path / "out"
    unfuse_checkpoint(source, str(out))
    written = load_file(out / "model.safetensors")
    assert "mtp.layers.0.mlp.experts.gate_up_proj" not in written
    assert torch.equal(written["mtp.layers.0.mlp.experts.7.gate_proj.weight"], torch.full((64, 128), 1.5))
    assert torch.equal(written["mtp.layers.0.mlp.experts.7.up_proj.weight"], torch.full((64, 128), 1.5))
    assert torch.equal(written["mtp.layers.0.mlp.experts.0.down_proj.weight"], torch.full((128, 64), 1.5))
    assert torch.equal(written["mtp.fc.weight"], _TAIL["mtp.fc.weight"])
    assert AutoConfig.from_pretrained(out).model_type == "qwen3_5_moe_text"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
