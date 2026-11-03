#!/usr/bin/env python
"""Gathered saves of a model holding an external table, Qwen3.8-Flash-Next's host n-gram table, written as table-only
files. A later save links the earlier one's files; the sharded EP save and ``save_pretrained`` are refused."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoModelForCausalLM

import halo.checkpoint.parallel.writing as gathered_writing
from halo.checkpoint.format import SAFETENSORS_INDEX_FILE
from halo.checkpoint.parallel.ep import reject_unmergeable_ep_sharded_save, save_ep_gathered_checkpoint
from halo.checkpoint.parallel.save import save_fsdp2_checkpoint
from halo.distributed.expert_parallel.families.qwen4_exp import EPQwen4ExpMoELayer
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.fsdp import IdentityParamSet, apply_fsdp2_per_layer, mixed_precision_policy
from halo.distributed.mesh import build_dp_mesh
from halo.model_loading.model_preparation import resolve_auto_model_class
from halo.model_loading.parallel.ep import load_ep_model_lazy
from halo.models.external_tables import EXTERNAL_TABLES_STAMP
from tests.common.checkpoint_io import loading_problems, safetensors_state_dict, save_context
from tests.common.file_links import refuse_hard_links
from tests.common.gloo import run_gloo_ranks
from tests.common.models import TINY_QWEN4_EXP_CONFIG
from tests.common.parallelism import single_process_ep_config
from tests.common.qwen4_exp import (
    NGRAM_TABLE_PATH,
    build_tiny_qwen4_exp_checkpoint,
    load_host_table_model,
    ngram_table,
    stamped_config,
    transformers_forwards,
)
from tests.common.tiny_models import TINY_MOE_FAMILIES, TINY_MOE_VLM_FAMILIES, TINY_SHARD_SIZE, tiny_family_model

PARTS = TINY_QWEN4_EXP_CONFIG["split_ngram_parts"]
TABLE_KEYS = {f"{NGRAM_TABLE_PATH}.shard_{index}.weight" for index in range(PARTS)}


def _ids(seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, TINY_QWEN4_EXP_CONFIG["vocab_size"], (2, 29), generator=generator)


@pytest.fixture
def source(tmp_path) -> str:
    directory = str(tmp_path / "source")
    build_tiny_qwen4_exp_checkpoint(directory)
    return directory


def _trained(model: torch.nn.Module) -> torch.nn.Module:
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(
                torch.randn(parameter.shape, generator=torch.Generator().manual_seed(1)).to(parameter) * 0.01
            )
    return model


def _logits(model: torch.nn.Module, ids: torch.Tensor) -> torch.Tensor:
    """Under transformers' own forwards, whichever forwards this process installed."""
    with torch.no_grad(), transformers_forwards():
        return model.eval()(input_ids=ids, use_cache=False).logits


def _weight_map(directory: str) -> dict[str, str]:
    return json.loads(Path(directory, SAFETENSORS_INDEX_FILE).read_text())["weight_map"]


def _table_files(directory: str) -> dict[str, str]:
    weight_map = _weight_map(directory)
    return {key: weight_map[key] for key in TABLE_KEYS}


def _assert_reloads_into(directory: str, expected_logits: torch.Tensor, dtype: torch.dtype) -> None:
    reloaded, info = AutoModelForCausalLM.from_pretrained(directory, dtype=dtype, output_loading_info=True)
    assert loading_problems(info) == {}
    assert torch.equal(_logits(reloaded, _ids()), expected_logits)


def test_the_gathered_save_writes_the_table_and_reloads_into_the_trained_model(source, tmp_path):
    model = _trained(load_host_table_model(source, dtype=torch.bfloat16))
    expected = _logits(model, _ids())
    out = str(tmp_path / "out")

    save_fsdp2_checkpoint(save_context(model), out)

    written = safetensors_state_dict(out)
    assert set(written) >= TABLE_KEYS and f"{NGRAM_TABLE_PATH}.weight" not in written
    files = set(_table_files(out).values())
    assert {key for key, name in _weight_map(out).items() if name in files} == TABLE_KEYS, "a table file holds more"
    config = json.loads(Path(out, "config.json").read_text())
    assert config["split_ngram_parts"] == PARTS
    assert EXTERNAL_TABLES_STAMP not in json.dumps(config), "the run-scoped stamp reached the exported config"
    _assert_reloads_into(out, expected, torch.bfloat16)


def test_the_ep_gathered_save_writes_the_table(source, tmp_path):
    model = _trained(load_host_table_model(source, dtype=torch.bfloat16))
    # The EP layer's expert compute rounds differently in bf16.
    expected = _logits(model, _ids())
    apply_ep_layers(model, single_process_ep_config())
    out = str(tmp_path / "out")

    save_ep_gathered_checkpoint(save_context(model, has_ep_layers=True), out)

    assert set(safetensors_state_dict(out)) >= TABLE_KEYS
    _assert_reloads_into(out, expected, torch.bfloat16)


def test_a_later_save_links_the_table_files_of_the_last(source, tmp_path):
    """A model attached from such a save links them on its first save."""
    model = load_host_table_model(source, dtype=torch.bfloat16)
    first, second, third = (str(tmp_path / name) for name in ("first", "second", "third"))
    save_fsdp2_checkpoint(save_context(model), first)
    save_fsdp2_checkpoint(save_context(model), second)
    resumed = load_host_table_model(second, dtype=torch.bfloat16)
    save_fsdp2_checkpoint(save_context(resumed), third)

    def inodes(directory: str) -> set[int]:
        return {os.stat(os.path.join(directory, name)).st_ino for name in set(_table_files(directory).values())}

    source_inodes = {os.stat(os.path.join(source, name)).st_ino for name in os.listdir(source)}
    assert not inodes(first) & source_inodes, "the release's files mix other tensors in and cannot be linked"
    assert inodes(second) == inodes(first) == inodes(third)
    assert torch.equal(_logits(resumed, _ids()), _logits(model, _ids()))


def test_a_save_after_the_last_ones_removal_writes_the_table_again(source, tmp_path):
    model = load_host_table_model(source, dtype=torch.bfloat16)
    expected = _logits(model, _ids())
    first, second = tmp_path / "first", str(tmp_path / "second")
    save_fsdp2_checkpoint(save_context(model), str(first))
    shutil.rmtree(first)
    save_fsdp2_checkpoint(save_context(model), second)
    _assert_reloads_into(second, expected, torch.bfloat16)


def test_a_save_where_links_are_refused_copies_the_table(source, tmp_path, monkeypatch):
    model = load_host_table_model(source, dtype=torch.bfloat16)
    first, second = str(tmp_path / "first"), str(tmp_path / "second")
    save_fsdp2_checkpoint(save_context(model), first)
    refuse_hard_links(monkeypatch)
    save_fsdp2_checkpoint(save_context(model), second)

    for key, name in _table_files(second).items():
        copied, original = os.path.join(second, name), os.path.join(first, _table_files(first)[key])
        assert os.stat(copied).st_ino != os.stat(original).st_ino
        assert Path(copied).read_bytes() == Path(original).read_bytes()


def test_the_release_layout_loads_text_only_and_exports_the_text_layout(tmp_path):
    """Under EP the layers go on before the attach, which reads the table off ``model.language_model.*``."""
    source = str(tmp_path / "release")
    torch.manual_seed(0)
    composite = tiny_family_model(TINY_MOE_VLM_FAMILIES["qwen4_exp"]).to(torch.bfloat16)
    composite.save_pretrained(source, max_shard_size=TINY_SHARD_SIZE)
    released = safetensors_state_dict(source)
    assert any(key.startswith("model.language_model.layers.") for key in released), "premise: the hub spelling"
    text_class = resolve_auto_model_class(AutoConfig.from_pretrained(source), text_only=True)

    eager = load_host_table_model(source, model_class=text_class)
    lazy = load_ep_model_lazy(
        source,
        single_process_ep_config(),
        stamped_config(source),
        dtype=torch.float32,
        trust_remote_code=False,
        model_class=text_class,
    )
    assert any(isinstance(module, EPQwen4ExpMoELayer) for module in lazy.modules()), "premise: the EP layers"
    stock_table = released[f"model.language_model.{NGRAM_TABLE_PATH.removeprefix('model.')}.shard_0.weight"]
    for model in (eager, lazy):
        assert type(model).__name__ == "Qwen4ExpForCausalLM"
        assert torch.equal(ngram_table(model).parts[0], stock_table)
    expected = _logits(eager, _ids())
    # The EP layer's expert compute rounds differently from the stock block's.
    torch.testing.assert_close(_logits(lazy, _ids()), expected, rtol=1e-5, atol=1e-5)

    eager_out, lazy_out = str(tmp_path / "eager"), str(tmp_path / "lazy")
    save_fsdp2_checkpoint(save_context(eager), eager_out)
    save_ep_gathered_checkpoint(save_context(lazy, has_ep_layers=True), lazy_out)
    for out in (eager_out, lazy_out):
        written = safetensors_state_dict(out)
        assert set(written) >= TABLE_KEYS
        assert not [key for key in written if key.startswith("model.language_model.") or ".visual." in key]
        assert json.loads(Path(out, "config.json").read_text())["model_type"] == "qwen4_exp_text"
        _assert_reloads_into(out, expected, torch.float32)


def test_save_pretrained_refuses_a_model_holding_the_table(source, tmp_path):
    """The trainer's saves route around it (``tests/cpu/trainers/test_qwen4_exp_trainer_saves.py``)."""
    model = load_host_table_model(source, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="external tables"):
        model.save_pretrained(str(tmp_path / "out"))
    assert not (tmp_path / "out").exists()


def test_a_save_of_a_table_never_attached_is_refused(source, tmp_path):
    model = AutoModelForCausalLM.from_config(stamped_config(source))
    with pytest.raises(RuntimeError, match="holds no table"):
        save_fsdp2_checkpoint(save_context(model), str(tmp_path / "out"))


def test_a_sharded_ep_save_is_refused(source):
    model = load_host_table_model(source, dtype=torch.bfloat16)
    apply_ep_layers(model, single_process_ep_config())
    with pytest.raises(ValueError, match="external tables"):
        reject_unmergeable_ep_sharded_save(model, world_size=1)


@pytest.mark.parametrize("max_shard_size", ["5GB", "200KB"])
def test_a_family_without_a_table_saves_byte_for_byte_as_before(tmp_path, monkeypatch, max_shard_size):
    torch.manual_seed(0)
    model = tiny_family_model(TINY_MOE_FAMILIES["qwen3_moe"]).to(torch.bfloat16)
    with_seam, without_seam = tmp_path / "with", tmp_path / "without"
    with_seam.mkdir()
    without_seam.mkdir()
    save_fsdp2_checkpoint(save_context(model, max_shard_size=max_shard_size), str(with_seam))
    monkeypatch.setattr(gathered_writing, "write_external_tables", lambda model, writer: None)
    save_fsdp2_checkpoint(save_context(model, max_shard_size=max_shard_size), str(without_seam))

    names = sorted(path.name for path in with_seam.iterdir() if path.suffix in (".safetensors", ".json"))
    assert names == sorted(path.name for path in without_seam.iterdir() if path.suffix in (".safetensors", ".json"))
    assert (max_shard_size == "200KB") == (SAFETENSORS_INDEX_FILE in names), f"premise: {names}"
    for name in names:
        assert (with_seam / name).read_bytes() == (without_seam / name).read_bytes(), name


def _fsdp2_save_worker(rank: int, source: str, out: str, expected_path: str) -> None:
    model = load_host_table_model(source)
    apply_fsdp2_per_layer(
        model,
        build_dp_mesh(dist.get_world_size(), device_type="cpu"),
        mixed_precision_policy(torch.float32, fp32_reduce=False, cast_forward_inputs=True),
        False,
        IdentityParamSet(),
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model(input_ids=_ids(seed=rank), labels=_ids(seed=rank), use_cache=False).loss.backward()
    optimizer.step()
    expected = _logits(model, _ids())
    save_fsdp2_checkpoint(
        save_context(model, toolkit_fsdp2=True, is_save_rank=rank == 0, training_checkpoint=True), out
    )
    if rank == 0:
        torch.save(expected, expected_path)
        assert ngram_table(model).files.directory == out, "the next save would not link this one's table"


def test_an_fsdp2_save_on_two_ranks_reloads_into_the_trained_model(source, tmp_path):
    out, expected_path = str(tmp_path / "out"), str(tmp_path / "expected.pt")
    run_gloo_ranks(_fsdp2_save_worker, 2, source, out, expected_path)
    _assert_reloads_into(out, torch.load(expected_path), torch.float32)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
