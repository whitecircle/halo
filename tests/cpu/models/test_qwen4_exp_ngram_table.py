#!/usr/bin/env python
"""Qwen3.8-Flash-Next's n-gram table in host memory (``HostNGramTable``), attached as views of the checkpoint's shard
tensors. No parameter, buffer or state-dict entry holds it, and its lookups equal the stock embedding's bit for bit."""

from __future__ import annotations

import copy
import logging
import os
import threading
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.core_model_loading import Concatenate

import halo.model_loading.parallel.ep as ep_lazy_loader
import halo.model_loading.parallel.loader as model_loading
import halo.model_loading.parallel.whole_model as whole_model
import halo.models.families.qwen4_exp.forwards as qwen4_exp_forwards
from halo.checkpoint.reading import resolve_checkpoint_weights
from halo.distributed.expert_parallel.config import EPConfig
from halo.distributed.fsdp import IdentityParamSet, apply_fsdp2_per_layer, mixed_precision_policy
from halo.distributed.mesh import build_dp_mesh
from halo.model_loading.finalization import finalize_loaded_model
from halo.model_loading.lazy_safetensors.conversion_selection import family_key_mapping
from halo.model_loading.parallel.ep import load_ep_model_lazy
from halo.model_loading.parallel.lazy_plan import WeightPlanner
from halo.model_loading.parallel.model_source import ModelSource
from halo.models.external_tables import EXTERNAL_TABLES_STAMP
from halo.models.families.qwen4_exp.ngram_table import HostNGramTable
from halo.models.gradient_checkpointing import install_recompute_scopes
from tests.common.gloo import run_gloo_ranks
from tests.common.models import TINY_QWEN4_EXP_CONFIG
from tests.common.parallelism import make_parallelism_config
from tests.common.qwen4_exp import (
    NGRAM_TABLE_PATH,
    build_tiny_qwen4_exp_checkpoint,
    load_host_table_model,
    ngram_table,
    stamped_config,
)

STOCK_TABLE_KEY = f"{NGRAM_TABLE_PATH}.weight"
HUB_ID = "Qwen/Qwen3.8-Flash-Next"
AGREED_COMMIT = "0123456789abcdef0123456789abcdef01234567"


def _ids(length: int = 37, batch: int = 2, seed: int = 0) -> torch.Tensor:
    return torch.randint(
        0, TINY_QWEN4_EXP_CONFIG["vocab_size"], (batch, length), generator=torch.Generator().manual_seed(seed)
    )


@pytest.fixture
def checkpoint(tmp_path) -> str:
    directory = str(tmp_path / "tiny")
    build_tiny_qwen4_exp_checkpoint(directory)
    return directory


def _stock(directory: str) -> torch.nn.Module:
    return AutoModelForCausalLM.from_pretrained(directory, dtype=torch.float32).eval()


def test_the_stock_build_trains_the_table_and_concatenates_it(checkpoint, monkeypatch):
    """Premise for every test here."""
    calls = []
    original = Concatenate.convert
    monkeypatch.setattr(Concatenate, "convert", lambda self, *a, **k: calls.append(self) or original(self, *a, **k))
    stock = _stock(checkpoint)
    assert stock.get_parameter(STOCK_TABLE_KEY).requires_grad
    assert any(call.num_shards_attribute == "split_ngram_parts" for call in calls)


def test_a_stamped_meta_shell_holds_no_table_and_runs_no_concatenation(checkpoint, monkeypatch):
    calls = []
    original = Concatenate.convert
    monkeypatch.setattr(Concatenate, "convert", lambda self, *a, **k: calls.append(self) or original(self, *a, **k))
    shell, info = AutoModelForCausalLM.from_pretrained(
        checkpoint, config=stamped_config(checkpoint), device_map="meta", output_loading_info=True
    )

    assert isinstance(shell.get_submodule(NGRAM_TABLE_PATH), HostNGramTable)
    assert not [name for name, _ in shell.named_parameters() if "ngram_embedding" in name]
    assert info["missing_keys"] == set() and info["unexpected_keys"] == set(), info
    assert not [call for call in calls if call.num_shards_attribute == "split_ngram_parts"]


def test_the_lazy_planner_plans_everything_but_the_table(tmp_path, caplog):
    """A key the model declares it ignores draws no warning; a key nothing claims still does."""
    directory = str(tmp_path / "tiny")
    build_tiny_qwen4_exp_checkpoint(directory, overrides={"split_ngram_parts": 128})
    weight_map = dict(resolve_checkpoint_weights(directory).weight_map)
    table_keys = {key for key in weight_map if ".ngram_embedding.shard_" in key}
    assert len(table_keys) == 128
    shell = AutoModelForCausalLM.from_pretrained(directory, config=stamped_config(directory), device_map="meta")
    weight_map["model.layers.0.unclaimed.weight"] = next(iter(weight_map.values()))

    disk_to_model, fanout = family_key_mapping(shell, list(weight_map))
    with caplog.at_level(logging.WARNING):
        plans = WeightPlanner(None).build(
            weight_map,
            disk_to_model,
            set(shell.state_dict()),
            fanout=fanout,
            ignore_unexpected=shell._keys_to_ignore_on_load_unexpected,
        )

    planned = {key for plan in plans for key in plan.disk_keys}
    assert planned == set(weight_map) - table_keys - {"model.layers.0.unclaimed.weight"}
    warned = [record.getMessage() for record in caplog.records if "align to no model tensor" in record.getMessage()]
    assert len(warned) == 1 and warned[0].startswith("1 checkpoint key(s)"), warned
    assert "model.layers.0.unclaimed.weight" in warned[0]


def test_the_stock_lazy_mapping_cannot_read_the_table(checkpoint):
    """Premise for the patch: the lazy loaders have no ranged read for transformers' table concatenation."""
    shell = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="meta")
    with pytest.raises(Exception, match="concat"):
        family_key_mapping(shell, list(resolve_checkpoint_weights(checkpoint).weight_map))


def test_the_host_lookup_equals_the_stock_rows_and_logits(checkpoint):
    stock = _stock(checkpoint)
    model = load_host_table_model(checkpoint).eval()
    table = ngram_table(model)

    assert torch.equal(torch.cat(table.parts).float(), stock.get_parameter(STOCK_TABLE_KEY))
    ids = _ids()
    # Mid-row eos tokens, where upstream restarts the n-gram history.
    ids[:, [5, 6, 19]] = TINY_QWEN4_EXP_CONFIG["eos_token_id"]
    with torch.no_grad():
        expected = stock(input_ids=ids, use_cache=False).logits
        assert torch.equal(model(input_ids=ids, use_cache=False).logits, expected)
        # The n-gram history rides the cache as upstream's does.
        stock_cache = stock(input_ids=ids[:, :-1], use_cache=True).past_key_values
        host_cache = model(input_ids=ids[:, :-1], use_cache=True).past_key_values
        step = ids[:, -1:]
        expected_step = stock(input_ids=step, past_key_values=stock_cache, use_cache=True).logits
        actual_step = model(input_ids=step, past_key_values=host_cache, use_cache=True).logits
    assert torch.equal(actual_step, expected_step)


def test_the_table_is_no_parameter_buffer_or_state_dict_entry(checkpoint):
    model = load_host_table_model(checkpoint)
    table = ngram_table(model)
    storages = {part.untyped_storage().data_ptr() for part in table.parts}

    def reaches_table(tensors) -> bool:
        return any(tensor.untyped_storage().data_ptr() in storages for tensor in tensors)

    assert not any("ngram_embedding" in key and "ngram_heads" not in key for key in model.state_dict())
    assert not reaches_table(model.parameters()) and not reaches_table(model.buffers())
    assert not reaches_table(model.state_dict().values())
    optimizer = torch.optim.AdamW(model.parameters())
    assert not reaches_table(p for group in optimizer.param_groups for p in group["params"])


def test_the_table_stays_host_resident_through_a_module_cast(checkpoint):
    model = load_host_table_model(checkpoint)
    table = ngram_table(model)
    parts = table.parts
    model.to(torch.bfloat16)
    assert table.parts is parts and table.compute_dtype == torch.bfloat16
    with torch.no_grad():
        assert model(input_ids=_ids(), use_cache=False).logits.dtype == torch.bfloat16


def _fsdp2_worker(rank: int, checkpoint: str) -> None:
    model = load_host_table_model(checkpoint)
    table = ngram_table(model)
    parts = table.parts
    apply_fsdp2_per_layer(
        model,
        build_dp_mesh(dist.get_world_size(), device_type="cpu"),
        mixed_precision_policy(torch.float32, fp32_reduce=False, cast_forward_inputs=True),
        False,
        IdentityParamSet(),
    )
    assert table.parts is parts, "the FSDP2 wrap touched the table"
    assert not any("ngram_embedding" in name for name, _ in model.named_parameters())
    loss = model(input_ids=_ids(seed=rank), labels=_ids(seed=rank), use_cache=False).loss
    loss.backward()
    assert all(part.grad is None for part in table.parts)


def test_no_fsdp2_unit_holds_the_table(checkpoint):
    run_gloo_ranks(_fsdp2_worker, 2, checkpoint)


def test_an_unstamped_build_in_the_same_process_stays_stock(checkpoint):
    """The patches are process-wide."""
    load_host_table_model(checkpoint)
    stock = _stock(checkpoint)
    finalize_loaded_model(stock)
    assert isinstance(stock.get_submodule(NGRAM_TABLE_PATH), torch.nn.Embedding)
    assert stock.get_parameter(STOCK_TABLE_KEY).requires_grad
    assert not hasattr(stock.config, EXTERNAL_TABLES_STAMP)


def test_a_stamped_model_without_a_table_source_is_refused(checkpoint, tmp_path):
    config = stamped_config(checkpoint)
    config._name_or_path = ""
    with pytest.raises(FileNotFoundError, match="names no checkpoint"):
        finalize_loaded_model(AutoModelForCausalLM.from_config(config))

    # A Hub id records no revision, so its cached snapshot could be another commit than the weights'.
    config._name_or_path = HUB_ID
    with pytest.raises(FileNotFoundError, match="no local directory"):
        finalize_loaded_model(AutoModelForCausalLM.from_config(config))

    config_only = tmp_path / "config_only"
    AutoConfig.from_pretrained(checkpoint).save_pretrained(config_only)
    config._name_or_path = str(config_only)
    with pytest.raises(FileNotFoundError, match="holds no keys"):
        finalize_loaded_model(AutoModelForCausalLM.from_config(config))


def test_an_eager_hub_load_attaches_the_table_from_the_snapshot_its_loader_agreed_on(checkpoint, monkeypatch):
    """The built config records only the Hub id, which alone is refused (above), so the loader names the snapshot."""

    def from_pretrained_verified(model_class, name, **kwargs):
        assert (name, kwargs.pop("revision")) == (HUB_ID, AGREED_COMMIT)
        model = model_class.from_pretrained(checkpoint, **kwargs)
        model.config._name_or_path = HUB_ID
        return model

    monkeypatch.setattr(whole_model, "from_pretrained_verified", from_pretrained_verified)
    model = whole_model.load_whole_model(
        AutoModelForCausalLM,
        ModelSource(HUB_ID, AGREED_COMMIT, checkpoint),
        make_parallelism_config(world_size=1, gpus_per_node=1, use_grouped_gemm=False),
        load_phase="Model load",
        ep_config=None,
        place_on_device=False,
        config=stamped_config(checkpoint),
        dtype=torch.float32,
        revision=AGREED_COMMIT,
    )

    table = ngram_table(model)
    assert table.is_attached and table.files.directory == checkpoint
    assert torch.equal(torch.cat(table.parts).float(), _stock(checkpoint).get_parameter(STOCK_TABLE_KEY))


class _Built(Exception):
    """Stops the policy load at its build."""


def test_a_random_init_policy_load_is_refused_before_the_build(checkpoint, monkeypatch):
    """``_name_or_path`` survives ``from_config``, so the attach would pair a pretrained table with random weights."""
    model = AutoModelForCausalLM.from_config(stamped_config(checkpoint))
    finalize_loaded_model(model)
    assert ngram_table(model).is_attached, "premise: the attach alone pairs the table with the random weights"

    builds = []

    def build(*args, **kwargs):
        builds.append(kwargs["init_from_scratch"])
        raise _Built

    monkeypatch.setattr(model_loading, "configure_float32_matmul_precision", lambda: None)
    monkeypatch.setattr(
        model_loading, "resolve_model_source", lambda name, revision, **k: ModelSource(name, None, name)
    )
    monkeypatch.setattr(model_loading, "resolve_attn_implementation", lambda *a, **k: "sdpa")
    monkeypatch.setattr(
        model_loading, "AutoTokenizer", SimpleNamespace(from_pretrained=lambda *a, **k: SimpleNamespace())
    )
    monkeypatch.setattr(model_loading, "_dispatch_model_loading", build)

    def load(init_from_scratch: bool):
        model_loading.load_distributed_model(
            checkpoint,
            parallelism_config=make_parallelism_config(world_size=1, gpus_per_node=1, use_grouped_gemm=False),
            dtype=torch.float32,
            trust_remote_code=False,
            init_from_scratch=init_from_scratch,
        )

    with pytest.raises(ValueError, match="init_from_scratch"):
        load(init_from_scratch=True)
    assert builds == [], "the refusal came after the build"
    with pytest.raises(_Built):
        load(init_from_scratch=False)
    assert builds == [False]


def test_a_lookup_without_a_table_or_past_its_rows_is_refused(checkpoint):
    unattached = HostNGramTable(rows=16, dim=4, part_count=2, compute_dtype=torch.float32)
    with pytest.raises(RuntimeError, match="no table attached"):
        unattached(torch.zeros(1, 3, dtype=torch.long))
    table = ngram_table(load_host_table_model(checkpoint))
    with pytest.raises(IndexError, match="holds"):
        table(torch.tensor([[0, table.rows]]))
    with pytest.raises(IndexError, match="holds"):
        table(torch.tensor([[-1, 0]]))


@pytest.mark.parametrize(
    ("parts", "keys", "error"),
    [
        ({"t.shard_1.weight": torch.zeros(8, 4), "t.shard_0.weight": torch.zeros(8, 4)}, None, "not its parts"),
        ({"t.shard_0.weight": torch.zeros(8, 4), "t.shard_1.weight": torch.zeros(7, 4)}, None, "15 rows"),
        ({"t.shard_0.weight": torch.zeros(8, 4), "t.shard_1.weight": torch.zeros(8, 3)}, None, r"\(3,\)"),
        ({"t.shard_0.weight": torch.zeros(8, 4), "t.shard_1.weight": torch.zeros(8, 4).half()}, None, "float16"),
        (
            {"t.shard_0.weight": torch.zeros(8, 4), "t.shard_1.weight": torch.zeros(8, 4)},
            ["t.shard_0.weight"],
            "2 parts",
        ),
    ],
)
def test_attach_refuses_parts_unlike_the_table(tmp_path, parts, keys, error):
    """Out of order, short of rows, of another width or dtype, or fewer than its parts."""
    save_file(parts, tmp_path / "model.safetensors")
    table = HostNGramTable(rows=16, dim=4, part_count=2, compute_dtype=torch.float32)
    with pytest.raises(ValueError, match=error):
        table.attach(str(tmp_path), dict.fromkeys(parts, "model.safetensors"), keys or list(parts))
    assert not table.is_attached


def test_a_table_unlike_the_config_is_refused(checkpoint):
    config = stamped_config(checkpoint)
    config.get_text_config().split_ngram_parts = 8
    model = AutoModelForCausalLM.from_config(config)
    model.config._name_or_path = checkpoint
    with pytest.raises(ValueError, match="built for 8 parts"):
        finalize_loaded_model(model)


def test_a_checkpointed_recompute_reads_the_first_pass_rows(checkpoint, monkeypatch):
    model = load_host_table_model(checkpoint).train()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    assert install_recompute_scopes(model) > 0
    gathers = []
    original = HostNGramTable._gather
    monkeypatch.setattr(HostNGramTable, "_gather", lambda self, ids: gathers.append(ids.shape) or original(self, ids))

    model(input_ids=_ids(), labels=_ids(), use_cache=False).loss.backward()

    assert len(gathers) == 1, f"the table was gathered {len(gathers)} times for one microbatch"


def test_a_deep_copy_shares_the_table(checkpoint):
    model = load_host_table_model(checkpoint)
    clone = copy.deepcopy(model)
    assert all(a is b for a, b in zip(ngram_table(model).parts, ngram_table(clone).parts, strict=True))
    with torch.no_grad():
        assert torch.equal(
            clone(input_ids=_ids(), use_cache=False).logits, model(input_ids=_ids(), use_cache=False).logits
        )


def test_the_local_main_process_prefetches_the_table_files(checkpoint, monkeypatch):
    advised = []
    monkeypatch.setattr(
        os, "posix_fadvise", lambda descriptor, *args: advised.append(os.readlink(f"/proc/self/fd/{descriptor}"))
    )
    table = ngram_table(load_host_table_model(checkpoint))
    for thread in threading.enumerate():
        if thread.name == "external-table-prefetch":
            thread.join()
    assert sorted(set(advised)) == sorted({os.path.join(checkpoint, name) for name in table.files.files})


def test_an_upstream_signature_change_fails_the_patch(monkeypatch):
    class _Renamed(torch.nn.Module):
        def __init__(self, config, embedding_dim, layer_idx, ple_layer_index=0):
            super().__init__()

        def forward(self, input_ids, cache_params):
            """The forward of a release that renamed its cache argument."""

    monkeypatch.setattr(qwen4_exp_forwards.qwen4_exp_modeling, "Qwen4ExpTextNGramEmbedding", _Renamed)
    with pytest.raises(RuntimeError, match="re-derive it"):
        qwen4_exp_forwards.patch_qwen4_exp_ngram_embedding()


def test_the_lazy_ep_load_attaches_the_table_and_matches_from_pretrained(checkpoint, monkeypatch, caplog):
    """The patched class declares the table keys ignored, so the lazy load warns about none."""
    monkeypatch.setattr(ep_lazy_loader, "apply_ep_layers", lambda model, *a, **k: model)
    reference = _stock(checkpoint)
    with caplog.at_level(logging.WARNING):
        lazy = load_ep_model_lazy(
            checkpoint,
            EPConfig(ep_size=1, stage_world_size=1, nvlink_domain_size=1, experts_fsdp_managed=True),
            stamped_config(checkpoint),
            dtype=torch.float32,
            trust_remote_code=False,
            model_class=AutoModelForCausalLM,
        )

    assert not [record for record in caplog.records if "align to no model tensor" in record.getMessage()]

    expected = {key: tensor for key, tensor in reference.state_dict().items() if key != STOCK_TABLE_KEY}
    actual = lazy.state_dict()
    assert actual.keys() == expected.keys()
    assert not [key for key, tensor in actual.items() if not torch.equal(tensor, expected[key])]
    assert torch.equal(torch.cat(ngram_table(lazy).parts).float(), reference.get_parameter(STOCK_TABLE_KEY))


def test_the_lazy_ep_load_attaches_the_table_from_the_directory_it_read(checkpoint, monkeypatch):
    """A shell recording the Hub id (as a config-only build does) would otherwise have its table refused."""

    def ep_layers_recording_the_hub_id(model, *args, **kwargs):
        model.config._name_or_path = "Qwen/Qwen3.8-Flash-Next"
        return model

    monkeypatch.setattr(ep_lazy_loader, "apply_ep_layers", ep_layers_recording_the_hub_id)
    lazy = load_ep_model_lazy(
        checkpoint,
        EPConfig(ep_size=1, stage_world_size=1, nvlink_domain_size=1, experts_fsdp_managed=True),
        stamped_config(checkpoint),
        dtype=torch.float32,
        trust_remote_code=False,
        model_class=AutoModelForCausalLM,
    )
    assert ngram_table(lazy).is_attached
    assert ngram_table(lazy).files.directory == checkpoint


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
