#!/usr/bin/env python
"""Nemotron-H's head through its save and the checkpoint tools, per expert at the release's keys: an untrained tail comes
back byte for byte with its router bias in fp32, on the text class, its no-latent shape and the composite, and a trained
head survives each tool the page lists. Real savers and tools on tiny checkpoints."""

import json
import logging
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file, save_file

from halo.checkpoint.parallel.ep import _reject_unmergeable_hub_keys, save_ep_gathered_checkpoint
from halo.checkpoint.parallel.save import save_unsharded_checkpoint
from halo.checkpoint.tools import model_merge, offline_export, vision_reattach
from halo.checkpoint.tools.bf16_conversion import convert_to_bf16
from halo.checkpoint.tools.expert_unfuse import unfuse_checkpoint
from halo.checkpoint.tools.lowp_export import quantize_checkpoint
from halo.checkpoint.tools.vision_reattach import reattach_vision_tower
from halo.checkpoint.tools.vocab_patch import patch_vocabulary
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.model_loading.parallel.model_source import ModelSource
from halo.model_loading.parallel.mtp_head import attach_mtp_head
from halo.models.moe.family_specs import NEMOTRON_H
from halo.models.mtp.spec import MtpHeadRequest, MtpObjective
from tests.common.checkpoint_io import save_context
from tests.common.mtp_heads import (
    MTP_TINY_FAMILIES,
    run_entry,
    saved_tensors,
    write_checkpoint_with_tail,
    write_entry_corpus,
    write_entry_run,
)
from tests.common.parallelism import single_process_ep_config
from tests.common.tiny_models import tiny_family_model

_PREFIXES = {"nemotron_h": "mtp.", "nemotron_h_no_latent": "mtp.", "nemotron_h_omni": "language_model.mtp."}
_ROUTES = sorted(_PREFIXES)
_BIAS = "layers.1.mixer.gate.e_score_correction_bias"
_COMPOSITE_CONFIG_MODULE = "configuration_nemotron_h_omni"
_PER_EXPERT_UP = "layers.1.mixer.experts.0.up_proj.weight"


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    return write_entry_corpus(tmp_path_factory.mktemp("corpus") / "rows")


def _head_of(tensors: dict, name: str) -> dict:
    return {key: value for key, value in tensors.items() if key.startswith(_PREFIXES[name])}


def _attached(name: str, path, *, ep_config=None):
    torch.manual_seed(1)
    model = tiny_family_model(MTP_TINY_FAMILIES[name]).to(torch.bfloat16)
    if ep_config is not None:
        apply_ep_layers(model, ep_config)
    attach_mtp_head(
        model,
        MtpHeadRequest(MtpObjective(weight=0.3)),
        pc=SimpleNamespace(fp32_non_ep_params=False, ep_config=lambda: ep_config),
        source=ModelSource(str(path), None, str(path)),
        dtype=torch.bfloat16,
        wraps_experts=ep_config is not None,
        init_from_scratch=False,
        preserve_checkpoint_precision=False,
    )
    return model


def _assert_byte_equal(written: dict, stored: dict) -> None:
    assert set(written) == set(stored), sorted(set(written) ^ set(stored))[:6]
    for key, value in stored.items():
        assert written[key].dtype == value.dtype, (key, written[key].dtype, value.dtype)
        assert torch.equal(written[key], value), key


def _save_gathered_ep(name: str, ckpt, out) -> None:
    model = _attached(name, ckpt, ep_config=single_process_ep_config())
    save_ep_gathered_checkpoint(save_context(model, has_ep_layers=True, is_save_rank=True), str(out))


def _save_unsharded(name: str, ckpt, out) -> None:
    save_unsharded_checkpoint(save_context(_attached(name, ckpt), is_save_rank=True), str(out))


def _save_pretrained(name: str, ckpt, out) -> None:
    _attached(name, ckpt).save_pretrained(out)


@pytest.mark.parametrize(
    "save", [_save_gathered_ep, _save_unsharded, _save_pretrained], ids=lambda save: save.__name__
)
@pytest.mark.parametrize("name", _ROUTES)
def test_an_untrained_tail_comes_back_byte_for_byte_per_expert(name, save, tmp_path):
    """What a served head is: the release's keys, its bytes, bf16 but for the fp32 router bias a bf16 cast would flatten."""
    write_checkpoint_with_tail(name, tmp_path / "ckpt", per_expert=True)
    stored = _head_of(load_file(tmp_path / "ckpt" / "model.safetensors"), name)
    assert stored[f"{_PREFIXES[name]}{_BIAS}"].dtype == torch.float32
    assert stored[f"{_PREFIXES[name]}{_PER_EXPERT_UP}"].dtype == torch.bfloat16
    save(name, tmp_path / "ckpt", tmp_path / "out")
    _assert_byte_equal(_head_of(saved_tensors(tmp_path / "out"), name), stored)


@pytest.mark.parametrize("name", _ROUTES)
def test_a_sharded_save_stays_refused_for_the_trunks_names(name, tmp_path):
    """The merge cannot respell ``backbone.*``, so a head changes nothing here, and the gathered save is the one route."""
    write_checkpoint_with_tail(name, tmp_path / "ckpt", per_expert=True)
    model = _attached(name, tmp_path / "ckpt", ep_config=single_process_ep_config())
    with pytest.raises(ValueError, match="merge_ep_shards.py's key-by-key stream cannot"):
        _reject_unmergeable_hub_keys(model)


def test_a_composite_export_holds_no_head_at_the_text_only_path(tmp_path):
    """An engine reading the composite strips ``language_model.`` from a key, so a root ``mtp.*`` beside it would double
    the head."""
    write_checkpoint_with_tail("nemotron_h_omni", tmp_path / "ckpt", per_expert=True)
    _save_gathered_ep("nemotron_h_omni", tmp_path / "ckpt", tmp_path / "out")
    assert not [key for key in saved_tensors(tmp_path / "out") if key.startswith("mtp.")]


def _store_whole(directory, tensors: dict) -> None:
    """A tiny base is sharded under an index, which a single file beside it would not join."""
    for part in directory.glob("model*"):
        part.unlink()
    save_file(tensors, directory / "model.safetensors", metadata={"format": "pt"})


def _trained(name: str, target, corpus: str, **fields):
    config = write_entry_run(
        name, target, corpus, steps=1, save_steps=1, mtp_loss_weight=0.3, mtp_loss_type="soft", **fields
    )
    done = run_entry(config)
    assert done.returncode == 0, (done.stderr + done.stdout)[-3000:]
    return target / "run" / "checkpoint-1"


@pytest.mark.parametrize(
    ("name", "fields"),
    [
        ("nemotron_h", {}),
        ("nemotron_h_no_latent", {}),
        ("nemotron_h_omni", {}),
        ("nemotron_h_omni", {"text_only_model": True}),
    ],
    ids=["text_class", "no_latent", "composite", "composite_text_route"],
)
def test_a_checkpoint_in_the_release_layout_fits_its_own_head(name, fields, tmp_path, corpus):
    """Hub spellings per expert with the tail beside them and no ``mtp_head_source``, as a release is: the head loads from
    the checkpoint the trunk does, and only it moves."""
    config = write_entry_run(
        name,
        tmp_path,
        corpus,
        steps=1,
        save_steps=1,
        mtp_loss_weight=1.0,
        mtp_loss_type="soft",
        mtp_train_scope="head_only",
        mtp_min_acceptance_gain=None,
        mtp_head_source=None,
        **fields,
    )
    base = saved_tensors(tmp_path / "base")
    assert any(".experts.0." in key for key in base), "the base is not in the hub's per-expert spelling"
    tail = _head_of(load_file(tmp_path / "tail" / "model.safetensors"), name)
    _store_whole(tmp_path / "base", {**base, **tail})
    done = run_entry(config)
    assert done.returncode == 0, (done.stderr + done.stdout)[-3000:]
    prefix = "mtp." if fields else _PREFIXES[name]
    written = saved_tensors(tmp_path / "run" / "checkpoint-1")
    head = {key: value for key, value in written.items() if key.startswith(prefix)}
    stored = {f"{prefix}{key[len(_PREFIXES[name]) :]}": value for key, value in tail.items()}
    assert set(head) == set(stored)
    assert [key for key in head if not torch.equal(head[key].float(), stored[key].float())], (
        "the head came back unchanged"
    )
    expected_level = "language_model." if not fields and name == "nemotron_h_omni" else ""
    assert f"{expected_level}backbone.norm_f.weight" in written


def _spell_as_the_release(directory) -> None:
    """The composite's ``config.json`` as the release spells it, with the ``auto_map`` that marks a servable source."""
    path = directory / "config.json"
    payload = json.loads(path.read_text())
    payload["llm_config"] = payload.pop("text_config")
    payload["auto_map"] = {"AutoConfig": f"{_COMPOSITE_CONFIG_MODULE}.NemotronH_Omni_Reasoning_V3_Config"}
    path.write_text(json.dumps(payload))
    (directory / f"{_COMPOSITE_CONFIG_MODULE}.py").write_text("")


@pytest.mark.parametrize(
    "fields",
    [{"mtp_loss_weight": 0.3, "mtp_loss_type": "soft"}, {"mtp_head_source": None}],
    ids=["with_a_head", "no_head"],
)
def test_a_trainer_saves_a_composite_loaded_off_the_release_schema(fields, tmp_path, corpus):
    """The Trainer's write to the composite's config is a field its class does not hold, which refused every such save at
    its finalize, head or no head. Its config keeps the release's spelling."""
    config = write_entry_run("nemotron_h_omni", tmp_path, corpus, steps=1, save_steps=1, **fields)
    _spell_as_the_release(tmp_path / "base")
    done = run_entry(config)
    assert done.returncode == 0, (done.stderr + done.stdout)[-3000:]
    saved = json.loads((tmp_path / "run" / "checkpoint-1" / "config.json").read_text())
    assert "llm_config" in saved and "text_config" not in saved
    assert "use_cache" not in saved


def test_the_tools_carry_the_head_off_the_release_schema(tmp_path, corpus):
    """The source's schema reaches each output's ``config.json``, and the head its keys."""
    config = write_entry_run(
        "nemotron_h_omni", tmp_path, corpus, steps=1, save_steps=1, mtp_loss_weight=0.3, mtp_loss_type="soft"
    )
    _spell_as_the_release(tmp_path / "base")
    done = run_entry(config)
    assert done.returncode == 0, (done.stderr + done.stdout)[-3000:]
    checkpoint = tmp_path / "run" / "checkpoint-1"
    head = set(_head_of(saved_tensors(checkpoint), "nemotron_h_omni"))
    assert head
    convert_to_bf16(str(checkpoint), str(tmp_path / "bf16"), "causal_lm", trust_remote_code=False)
    patch_vocabulary(str(checkpoint), str(tmp_path / "patched"), patterns=[], trust_remote_code=False)
    unfuse_checkpoint(str(checkpoint), str(tmp_path / "unfused"))
    for out in ("bf16", "patched", "unfused"):
        assert set(_head_of(saved_tensors(tmp_path / out), "nemotron_h_omni")) == head, out
        written = json.loads((tmp_path / out / "config.json").read_text())
        assert "llm_config" in written and "text_config" not in written, out


def _no_tail_warning(caplog) -> None:
    assert not [r.getMessage() for r in caplog.records if "MTP head tensor(s)" in r.getMessage()]


@pytest.mark.parametrize("name", _ROUTES)
def test_a_bf16_conversion_keeps_the_trained_head_and_its_fp32_bias(name, tmp_path, corpus, caplog):
    """The trunk's bias keeps its width through the class's pin, so the head's must, or a converted head flips near-ties
    its fit never saw."""
    checkpoint = _trained(name, tmp_path, corpus)
    trained = _head_of(saved_tensors(checkpoint), name)
    with caplog.at_level(logging.WARNING, logger=offline_export.__name__):
        convert_to_bf16(str(checkpoint), str(tmp_path / "out"), "causal_lm", trust_remote_code=False)
    _no_tail_warning(caplog)
    written = _head_of(saved_tensors(tmp_path / "out"), name)
    assert set(written) == set(trained)
    bias = f"{_PREFIXES[name]}{_BIAS}"
    assert written[bias].dtype == torch.float32 and torch.equal(written[bias], trained[bias])
    for key, value in trained.items():
        if key != bias:
            assert written[key].dtype == torch.bfloat16, key
            assert torch.equal(written[key], value.to(torch.bfloat16)), key


def test_a_text_class_tool_carries_a_composites_tail_to_its_own_level(tmp_path):
    """A text-only load of Super VL's checkpoint: its export stores ``mtp.*``, and a tail left at ``language_model.mtp.*``
    would sit under no module of that export."""
    write_checkpoint_with_tail("nemotron_h_omni", tmp_path / "ckpt", per_expert=True)
    stored = _head_of(load_file(tmp_path / "ckpt" / "model.safetensors"), "nemotron_h_omni")
    model = tiny_family_model(MTP_TINY_FAMILIES["nemotron_h"])
    carried = offline_export.carried_mtp_tail(
        model, str(tmp_path / "ckpt"), str(tmp_path / "out"), tool="t", dtype=torch.bfloat16
    )
    assert set(carried) == {key.removeprefix("language_model.") for key in stored}
    for key, value in stored.items():
        assert torch.equal(carried[key.removeprefix("language_model.")], value), key


@pytest.mark.parametrize("name", _ROUTES)
def test_a_vocabulary_patch_keeps_the_head_and_grows_only_what_it_shares(name, tmp_path, corpus, caplog):
    checkpoint = _trained(name, tmp_path, corpus)
    trained = _head_of(saved_tensors(checkpoint), name)
    with caplog.at_level(logging.WARNING, logger=offline_export.__name__):
        patch_vocabulary(str(checkpoint), str(tmp_path / "out"), patterns=[], trust_remote_code=False)
    _no_tail_warning(caplog)
    written = _head_of(saved_tensors(tmp_path / "out"), name)
    assert set(written) == set(trained)
    assert written[f"{_PREFIXES[name]}{_BIAS}"].dtype == torch.float32


@pytest.mark.parametrize("name", _ROUTES)
def test_unfusing_copies_a_stored_per_expert_head_through(name, tmp_path, corpus):
    checkpoint = _trained(name, tmp_path, corpus)
    unfuse_checkpoint(str(checkpoint), str(tmp_path / "out"))
    _assert_byte_equal(
        _head_of(saved_tensors(tmp_path / "out"), name),
        _head_of(saved_tensors(checkpoint), name),
    )


def test_quantizing_streams_the_head_with_the_trunk(tmp_path, corpus):
    """Its expert projections pack as the trunk's do, and its norms, router and bias stay what they were."""
    checkpoint = _trained("nemotron_h", tmp_path, corpus)
    trained = _head_of(saved_tensors(checkpoint), "nemotron_h")
    out = tmp_path / "out"
    quantize_checkpoint(str(checkpoint), str(out), "mxfp8", apply_moe_experts=True)
    written = saved_tensors(out)
    assert "mtp.layers.1.mixer.experts.0.down_proj.weight_packed" in written
    assert "backbone.layers.1.mixer.experts.0.down_proj.weight_packed" in written
    for key in ("mtp.layers.0.enorm.weight", "mtp.layers.1.final_layernorm.weight", f"mtp.{_BIAS}"):
        assert torch.equal(written[key], trained[key]), key


def _two_fine_tunes(name: str, target, corpus: str):
    first = _trained(name, target / "a", corpus, seed=1)
    second = _trained(name, target / "b", corpus, seed=1, learning_rate=2e-3)
    return first, second


@pytest.mark.parametrize("name", ["nemotron_h", "nemotron_h_omni"])
def test_merging_fine_tunes_that_all_carry_the_head_merges_it(name, tmp_path, corpus):
    first, second = _two_fine_tunes(name, tmp_path, corpus)
    model_merge.merge_models(
        [str(first), str(second)],
        str(tmp_path / "merged"),
        method="linear",
        allow_missing_tokenizer=True,
        verbose=False,
    )
    merged, a, b = (_head_of(saved_tensors(path), name) for path in (tmp_path / "merged", first, second))
    assert set(merged) == set(a)
    key = f"{_PREFIXES[name]}layers.0.eh_proj.weight"
    assert not torch.equal(a[key], b[key]), "the two runs share a head, so a merge proves nothing"
    assert torch.allclose(merged[key].float(), (a[key].float() + b[key].float()) / 2, atol=1e-2)


@pytest.mark.parametrize("name", ["nemotron_h", "nemotron_h_omni"])
def test_a_base_tail_no_fine_tune_stores_is_left_out_of_a_merge(name, tmp_path):
    """Both classes' load patterns name it: the composite names no tail where its text class does, and a base's stale head
    refused the merge on the composite alone."""
    write_checkpoint_with_tail(name, tmp_path / "base", per_expert=True)
    stored = load_file(tmp_path / "base" / "model.safetensors")
    prefix = _PREFIXES[name]
    for fine_tune in ("a", "b"):
        (tmp_path / fine_tune).mkdir()
        save_file(
            {key: value for key, value in stored.items() if not key.startswith(prefix)},
            tmp_path / fine_tune / "model.safetensors",
            metadata={"format": "pt"},
        )
        (tmp_path / fine_tune / "config.json").write_text((tmp_path / "base" / "config.json").read_text())
    model_merge.merge_models(
        [str(tmp_path / "a"), str(tmp_path / "b")],
        str(tmp_path / "merged"),
        method="task_arithmetic",
        base_model=str(tmp_path / "base"),
        allow_missing_tokenizer=True,
        trust_remote_code=False,
        verbose=False,
    )
    assert not _head_of(saved_tensors(tmp_path / "merged"), name)


def test_merging_where_only_one_carries_the_head_is_refused(tmp_path, corpus):
    first = _trained("nemotron_h", tmp_path / "a", corpus)
    second = tmp_path / "b"
    second.mkdir()
    stored = saved_tensors(first)
    save_file(
        {key: value for key, value in stored.items() if not key.startswith("mtp.")},
        second / "model.safetensors",
        metadata={"format": "pt"},
    )
    (second / "config.json").write_text((first / "config.json").read_text())
    with pytest.raises(ValueError, match="does not carry the merge's reference key set"):
        model_merge.merge_models(
            [str(first), str(second)],
            str(tmp_path / "merged"),
            method="linear",
            allow_missing_tokenizer=True,
            verbose=False,
        )


def test_a_text_only_fit_reattaches_with_the_head_it_trained_at_the_composites_path(tmp_path, corpus, caplog):
    """Super VL's own text route: it reads the composite's ``language_model.mtp.*``, exports ``mtp.*``, and the reattach
    writes that back where both engines read it, over the base's stale head and with no root ``mtp.*`` beside it."""
    checkpoint = _trained("nemotron_h_omni", tmp_path, corpus, text_only_model=True)
    export = saved_tensors(checkpoint)
    head = {key: value for key, value in export.items() if key.startswith("mtp.")}
    assert head and not any(key.startswith("language_model.") for key in export)
    tail = load_file(tmp_path / "tail" / "model.safetensors")
    base = saved_tensors(tmp_path / "base")
    _store_whole(
        tmp_path / "base",
        {**base, **{key: value for key, value in tail.items() if key.startswith("language_model.mtp.")}},
    )
    with caplog.at_level(logging.WARNING, logger=vision_reattach.__name__):
        reattach_vision_tower(str(checkpoint), str(tmp_path / "base"), str(tmp_path / "out"), trust_remote_code=False)
    assert not [record for record in caplog.records if "MTP head" in record.getMessage()]
    written = saved_tensors(tmp_path / "out")
    assert not [key for key in written if key.startswith("mtp.")]
    assert {key for key in written if key.startswith("language_model.mtp.")} == {
        f"language_model.{key}" for key in head
    }
    for key, value in head.items():
        assert torch.equal(written[f"language_model.{key}"], value), key
    moved = [key for key, value in head.items() if not torch.equal(value, tail[f"language_model.{key}"])]
    assert moved, "the fit moved nothing, so the base's stale head could have written these"
    assert any(key.startswith("vision_model.") for key in written), "the vision tower did not return"


def test_a_reattach_with_no_trained_head_carries_the_base_and_offers_a_fit(tmp_path, caplog):
    write_checkpoint_with_tail("nemotron_h_omni", tmp_path / "base", per_expert=True)
    base = load_file(tmp_path / "base" / "model.safetensors")
    composite = tiny_family_model(MTP_TINY_FAMILIES["nemotron_h_omni"])
    composite.save_pretrained(tmp_path / "hub")
    hub = load_file(tmp_path / "hub" / "model.safetensors")
    save_file(
        {**hub, **{key: value for key, value in base.items() if key.startswith("language_model.mtp.")}},
        tmp_path / "hub" / "model.safetensors",
        metadata={"format": "pt"},
    )
    export = tmp_path / "export"
    export.mkdir()
    save_file(
        {
            key.removeprefix("language_model."): value
            for key, value in hub.items()
            if key.startswith(("language_model.backbone.", "language_model.lm_head."))
        },
        export / "model.safetensors",
        metadata={"format": "pt"},
    )
    composite.config.text_config.save_pretrained(export)
    with caplog.at_level(logging.WARNING, logger=vision_reattach.__name__):
        reattach_vision_tower(str(export), str(tmp_path / "hub"), str(tmp_path / "out"), trust_remote_code=False)
    warned = [record.getMessage() for record in caplog.records if "MTP head" in record.getMessage()]
    assert len(warned) == 1 and "mtp_train_scope: head_only" in warned[0], warned
    written = saved_tensors(tmp_path / "out")
    assert torch.equal(written[f"language_model.mtp.{_BIAS}"], base[f"language_model.mtp.{_BIAS}"])


def test_the_composite_layout_writes_a_text_exports_tail_at_the_release_path():
    layout = NEMOTRON_H.text_only_layout
    assert layout.to_composite_key("mtp.layers.0.norm.weight") == "language_model.mtp.layers.0.norm.weight"
    assert layout.to_composite_key("backbone.norm_f.weight") == "language_model.backbone.norm_f.weight"
    assert layout.to_composite_key("lm_head.weight") == "language_model.lm_head.weight"
    assert layout.to_composite_key("vision_model.summary_idxs") == "vision_model.summary_idxs"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
