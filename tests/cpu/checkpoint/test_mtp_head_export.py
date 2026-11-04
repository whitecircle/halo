#!/usr/bin/env python
"""What a trained head's checkpoint holds: the release's keys through the real entry script on one process and two gloo
ranks, the trunk copies a list head's engines read, a stock reload that ignores it, and a sync-side load holding no
head. Per-rank EP shards merge to the gathered save, and two nodes' own disks each hold the whole head."""

import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from halo.checkpoint.parallel.ep import save_ep_gathered_checkpoint, save_ep_sharded_checkpoint
from halo.checkpoint.parallel.save import save_checkpoint, save_fsdp2_checkpoint, select_checkpoint_saver
from halo.checkpoint.parallel.save import save_unsharded_checkpoint as save_unsharded
from halo.checkpoint.tools.ep_shard_merge import merge_ep_shards
from halo.checkpoint.tools.vision_reattach import reattach_vision_tower
from halo.distributed import fsdp as fsdp_module
from halo.distributed.expert_parallel.config import EPConfig
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.fsdp import apply_fsdp2_for_dp, reshard_fsdp2_modules
from halo.distributed.mesh import build_dp_mesh
from halo.model_loading.finalization import finalize_loaded_model
from halo.model_loading.model_preparation import auto_load_model
from halo.model_loading.parallel.model_source import ModelSource
from halo.model_loading.parallel.mtp_head import attach_mtp_head
from halo.models.mtp.head import find_mtp_head, mtp_trunk_copy_keys
from halo.models.mtp.spec import MtpHeadRequest, MtpObjective
from halo.runtime import filesystem
from halo.runtime.filesystem import fs_aware_makedirs
from halo.trainers.grpo.rollout.weight_push import _dense_params
from tests.common.checkpoint_io import save_context
from tests.common.distributed import reset_shared_filesystem_consensus
from tests.common.gloo import run_gloo_ranks
from tests.common.models import (
    GLM4_FLASH,
    NEMOTRON_3_5_LIGHTNING,
    NEMOTRON_3_5_SUPER_VL,
    QWEN3_5_MOE_122B,
    QWEN3_NEXT_80B_A3B,
)
from tests.common.mtp_heads import (
    MTP_TINY_FAMILIES,
    run_entry,
    saved_tensors,
    tiny_model_with_head,
    write_checkpoint_with_tail,
    write_entry_corpus,
    write_entry_run,
)
from tests.common.parallelism import make_parallelism_config, single_process_ep_config
from tests.common.repo import REPO_ROOT
from tests.common.tiny_models import tiny_family_model

_RELEASE_TAILS = json.loads((REPO_ROOT / "tests/common/mtp_release_tails.json").read_text(encoding="utf-8"))
_RELEASE_OF_FAMILY = {
    "qwen3_5": "Qwen/Qwen3.8-27B",
    "qwen3_5_moe_text": QWEN3_5_MOE_122B,
    "qwen3_5_moe": QWEN3_5_MOE_122B,
    "qwen3_next": QWEN3_NEXT_80B_A3B,
    "glm4_moe_lite": GLM4_FLASH,
    "nemotron_h": NEMOTRON_3_5_SUPER_VL,
    "nemotron_h_no_latent": NEMOTRON_3_5_LIGHTNING,
    "nemotron_h_omni": NEMOTRON_3_5_SUPER_VL,
}
_HEAD_PATHS = {"glm4_moe_lite": "model.layers.2.", "nemotron_h_omni": "language_model.mtp."}
_RELEASE_PATHS = {
    "glm4_moe_lite": "model.layers.47.",
    "nemotron_h": "language_model.mtp.",
    "nemotron_h_omni": "language_model.mtp.",
}
_STEPS = 1
_FUSED_TO_PER_EXPERT = {
    "experts.gate_up_proj": ("experts.N.gate_proj.weight", "experts.N.up_proj.weight"),
    "experts.down_proj": ("experts.N.down_proj.weight",),
    "experts.up_proj": ("experts.N.up_proj.weight",),
}
_TINY_QUERY_PROJECTION = re.compile(r"self_attn\.q_(proj|a_proj|a_layernorm|b_proj)\.")
_WORLD = 2
_TWO_NODES = {"LOCAL_RANK": "0", "LOCAL_WORLD_SIZE": "1", "DIST_OUTPUT_SHARED_FILESYSTEM": "0"}
_ROOT_FAMILY = "qwen3_5_moe_text"
_LIST_FAMILY = "glm4_moe_lite"


@pytest.fixture(scope="module")
def corpus(tmp_path_factory):
    return write_entry_corpus(tmp_path_factory.mktemp("corpus") / "rows")


def _release_form(keys, prefix: str) -> set[str]:
    """Keys under ``prefix``, relative to it, in the one spelling of a fused and a per-expert bank."""
    spelled = set()
    for key in keys:
        # A tiny GLM-4 sets no q_lora_rank, so its query projection is the one key spelled off the release's config.
        if not key.startswith(prefix) or _TINY_QUERY_PROJECTION.search(key):
            continue
        relative = re.sub(r"experts\.\d+\.", "experts.N.", key[len(prefix) :])
        fused = next((fused for fused in _FUSED_TO_PER_EXPERT if relative.endswith(fused)), None)
        if fused is None:
            spelled.add(relative)
        else:
            spelled.update(relative[: -len(fused)] + split for split in _FUSED_TO_PER_EXPERT[fused])
    return spelled


def _run(name: str, target: Path, corpus: str, *, ranks: int, **fields) -> Path:
    config = write_entry_run(name, target, corpus, steps=_STEPS, mtp_loss_weight=0.3, mtp_loss_type="soft", **fields)
    done = run_entry(config, ranks=ranks)
    assert done.returncode == 0, (done.stderr + done.stdout)[-3000:]
    return target / "run" / f"checkpoint-{_STEPS}"


@pytest.mark.parametrize("ranks", [1, _WORLD], ids=["one_process", "two_ranks"])
@pytest.mark.parametrize("name", sorted(_RELEASE_OF_FAMILY))
def test_a_run_saves_its_head_under_the_releases_keys(name, ranks, tmp_path, corpus):
    saved = saved_tensors(_run(name, tmp_path, corpus, ranks=ranks))
    release_tail = _RELEASE_TAILS[_RELEASE_OF_FAMILY[name]]["tail"]
    prefix = _HEAD_PATHS.get(name, "mtp.")
    release_prefix = _RELEASE_PATHS.get(name, "mtp.")
    assert _release_form(saved, prefix) == _release_form(release_tail, release_prefix)


@pytest.mark.parametrize("ranks", [1, _WORLD], ids=["one_process", "two_ranks"])
def test_a_list_heads_trunk_copies_hold_the_trained_embeddings(ranks, tmp_path, corpus):
    """An engine's completeness check reads them, and the head shares the embeddings in training, so each is written off
    the trained tensor; the run moved both, or the bytes below say nothing."""
    checkpoint = _run(_LIST_FAMILY, tmp_path, corpus, ranks=ranks)
    saved = saved_tensors(checkpoint)
    base = saved_tensors(tmp_path / "base")
    for copy_key, trunk_key in (
        ("model.layers.2.embed_tokens.weight", "model.embed_tokens.weight"),
        ("model.layers.2.shared_head.head.weight", "lm_head.weight"),
    ):
        assert not torch.equal(saved[trunk_key], base[trunk_key]), f"{trunk_key} took no step"
        assert torch.equal(saved[copy_key], saved[trunk_key]), copy_key


@pytest.mark.parametrize(
    ("name", "field"), [(_ROOT_FAMILY, "mtp_num_hidden_layers"), (_LIST_FAMILY, "num_nextn_predict_layers")]
)
def test_the_export_keeps_the_depth_field_an_engine_builds_its_drafter_from(name, field, tmp_path, corpus):
    write_entry_run(name, tmp_path, corpus, steps=_STEPS)
    config_path = tmp_path / "base" / "config.json"
    config_path.write_text(json.dumps({**json.loads(config_path.read_text()), field: 1}))
    checkpoint = _run(name, tmp_path, corpus, ranks=1)
    assert json.loads((checkpoint / "config.json").read_text())[field] == 1


def test_a_tied_list_head_copies_the_one_embedding_under_both_keys():
    model, _head = tiny_model_with_head(_LIST_FAMILY)
    model.lm_head.weight = model.model.embed_tokens.weight
    assert mtp_trunk_copy_keys(model) == {
        "model.embed_tokens.weight": ["model.layers.2.embed_tokens.weight", "model.layers.2.shared_head.head.weight"]
    }


def test_only_a_head_with_trunk_copies_takes_the_gathered_saver_on_one_process():
    """``save_pretrained`` writes the state dict alone, so it would leave the copies out."""
    list_head, _ = tiny_model_with_head(_LIST_FAMILY)
    root_head, _ = tiny_model_with_head(_ROOT_FAMILY)
    no_head = tiny_family_model(MTP_TINY_FAMILIES[_LIST_FAMILY])
    assert select_checkpoint_saver(save_context(list_head)) is save_unsharded
    assert select_checkpoint_saver(save_context(root_head)) is None
    assert select_checkpoint_saver(save_context(no_head)) is None


@pytest.mark.parametrize("name", [_ROOT_FAMILY, _LIST_FAMILY, "qwen3_5", "nemotron_h", "nemotron_h_omni"])
def test_a_stock_load_of_the_export_ignores_the_head_and_a_sync_finds_nothing_to_gather(name, tmp_path, corpus):
    """An RL run starting from a trained head's export: its server reads the head at launch, and its trainer holds no
    live head for the sync's parameter walk to push."""
    checkpoint = _run(name, tmp_path, corpus, ranks=1)
    assert any(key.startswith(_HEAD_PATHS.get(name, "mtp.")) for key in saved_tensors(checkpoint))
    # Through the coverage gate, so a key the head's load left missing would raise.
    model = auto_load_model(str(checkpoint), dtype=torch.float32)
    finalize_loaded_model(model)
    assert find_mtp_head(model) is None
    trained_keys = saved_tensors(checkpoint)
    walked = {key for key, _param in _dense_params(model, {})}
    tail_keys = {key for key in trained_keys if key.startswith(_HEAD_PATHS.get(name, "mtp."))}
    assert not walked & tail_keys, sorted(walked & tail_keys)[:4]


@pytest.mark.parametrize("name", [_ROOT_FAMILY, _LIST_FAMILY, "qwen3_next", "nemotron_h", "nemotron_h_omni"])
def test_a_gathered_ep_save_writes_the_releases_keys_at_the_export_dtype(name, tmp_path):
    """The EP saver on one rank: the head's experts in the family's layout, its norms at the trunk's fp32 and the rest at
    bf16, the trunk copies beside them."""
    write_checkpoint_with_tail(name, tmp_path / "tail", per_expert=True)
    ep_config = single_process_ep_config()
    torch.manual_seed(1)
    model = tiny_family_model(MTP_TINY_FAMILIES[name]).float()
    apply_ep_layers(model, ep_config)
    attach_mtp_head(
        model,
        MtpHeadRequest(MtpObjective(0.3)),
        pc=SimpleNamespace(fp32_non_ep_params=False, ep_config=lambda: ep_config),
        source=ModelSource(f"{tmp_path}/tail", None, f"{tmp_path}/tail"),
        dtype=torch.float32,
        wraps_experts=True,
        init_from_scratch=False,
        preserve_checkpoint_precision=False,
    )
    save_ep_gathered_checkpoint(save_context(model, has_ep_layers=True, is_save_rank=True), str(tmp_path / "out"))
    saved = saved_tensors(tmp_path / "out")
    prefix = _HEAD_PATHS.get(name, "mtp.")
    assert _release_form(saved, prefix) == _release_form(
        _RELEASE_TAILS[_RELEASE_OF_FAMILY[name]]["tail"], _RELEASE_PATHS.get(name, "mtp.")
    )
    projection = next(key for key in saved if key.startswith(prefix) and key.endswith(("fc.weight", "eh_proj.weight")))
    assert saved[projection].dtype == torch.bfloat16
    norm = {_LIST_FAMILY: "enorm.weight", "nemotron_h": "layers.0.enorm.weight"}
    norm["nemotron_h_omni"] = norm["nemotron_h"]
    assert saved[f"{prefix}{norm.get(name, 'norm.weight')}"].dtype == torch.float32


def _rank_ep_config() -> EPConfig:
    config = EPConfig(
        ep_size=_WORLD,
        stage_world_size=_WORLD,
        nvlink_domain_size=_WORLD,
        use_grouped_gemm=False,
        experts_fsdp_managed=False,
    )
    return config


def _save_shards_and_gathered(rank: int, name: str, root: str) -> None:
    """Real EP groups and a real gather; no forward, so no transport."""
    ep_config = _rank_ep_config()
    torch.manual_seed(1)
    model = tiny_family_model(MTP_TINY_FAMILIES[name]).to(torch.bfloat16)
    apply_ep_layers(model, ep_config)
    attach_mtp_head(
        model,
        MtpHeadRequest(MtpObjective(0.3)),
        pc=SimpleNamespace(fp32_non_ep_params=False, ep_config=lambda: ep_config),
        source=ModelSource(f"{root}/tail", None, f"{root}/tail"),
        dtype=torch.bfloat16,
        wraps_experts=True,
        init_from_scratch=False,
        preserve_checkpoint_precision=False,
    )
    pc = make_parallelism_config(world_size=_WORLD, gpus_per_node=_WORLD, rank=rank, ep_size=_WORLD)
    fields = {"has_ep_layers": True, "training_checkpoint": True, "is_save_rank": rank == 0}
    save_ep_sharded_checkpoint(save_context(model, pc, save_sharded_ep=True, **fields), f"{root}/shards")
    save_ep_gathered_checkpoint(save_context(model, pc, **fields), f"{root}/gathered")


@pytest.mark.parametrize("name", [_ROOT_FAMILY, "qwen3_next"])
def test_per_rank_shards_merge_to_the_gathered_save_with_the_head(name, tmp_path):
    write_checkpoint_with_tail(name, tmp_path / "tail", per_expert=True)
    run_gloo_ranks(_save_shards_and_gathered, _WORLD, name, str(tmp_path))
    merge_ep_shards(str(tmp_path / "shards"), str(tmp_path / "merged"), verbose=False)
    merged, gathered = saved_tensors(tmp_path / "merged"), saved_tensors(tmp_path / "gathered")
    assert any(key.startswith("mtp.layers.0.mlp.experts") for key in merged), "the head's experts were not merged"
    assert set(merged) == set(gathered)
    differing = [key for key in merged if not torch.equal(merged[key], gathered[key])]
    assert not differing, differing[:4]


def _node_dir(root: str, rank: int) -> str:
    return os.path.join(root, f"node_{rank}", "checkpoint-1")


def _save_on_each_node(rank: int, name: str, root: str) -> None:
    filesystem.resolve_shared_filesystem_consensus()
    assert not filesystem.is_output_shared_filesystem() and filesystem.is_fs_save_rank()
    fsdp_module.build_dp_mesh = lambda *args, **kwargs: build_dp_mesh(*args, **{**kwargs, "device_type": "cpu"})
    model, _head = tiny_model_with_head(name, dtype=torch.float32)
    apply_fsdp2_for_dp(model, _WORLD, compute_dtype=torch.float32, fp32_reduce=True)
    reshard_fsdp2_modules(model)
    out = _node_dir(root, rank)
    fs_aware_makedirs(out)
    pc = make_parallelism_config(world_size=_WORLD, gpus_per_node=1, rank=rank)
    context = save_context(model, pc, toolkit_fsdp2=True, is_save_rank=True, training_checkpoint=True)
    assert select_checkpoint_saver(context) is save_fsdp2_checkpoint
    assert save_checkpoint(context, out)
    # A resume reads this node's own copy.
    fresh = tiny_family_model(MTP_TINY_FAMILIES[name]).float()
    head = attach_mtp_head(
        fresh,
        MtpHeadRequest(MtpObjective(0.3)),
        pc=SimpleNamespace(fp32_non_ep_params=False, ep_config=lambda: None),
        source=ModelSource(out, None, out),
        dtype=torch.float32,
        wraps_experts=False,
        init_from_scratch=False,
        preserve_checkpoint_precision=True,
    )
    torch.save(head.state_dict(), Path(root) / f"read_back_{rank}.pt")
    reset_shared_filesystem_consensus()


@pytest.mark.parametrize("name", [_ROOT_FAMILY, _LIST_FAMILY, "nemotron_h", "nemotron_h_omni"])
def test_on_per_node_storage_every_node_holds_the_whole_head(name, tmp_path):
    run_gloo_ranks(_save_on_each_node, _WORLD, name, str(tmp_path), env=_TWO_NODES)
    first, second = (saved_tensors(_node_dir(str(tmp_path), rank)) for rank in range(_WORLD))
    prefix = _HEAD_PATHS.get(name, "mtp.")
    assert any(key.startswith(prefix) for key in first), "no head reached the first node's disk"
    assert set(first) == set(second)
    differing = [key for key in first if not torch.equal(first[key], second[key])]
    assert not differing, differing[:4]
    for copied in mtp_trunk_copy_keys(tiny_model_with_head(name)[0]).values():
        assert set(copied) <= set(second), "a copy reached the first node's disk alone"
    read_back = [torch.load(tmp_path / f"read_back_{rank}.pt") for rank in range(_WORLD)]
    trained = {key[len(prefix) :]: value for key, value in first.items() if key.startswith(prefix)}
    assert read_back[0].keys() == read_back[1].keys()
    for key, value in read_back[0].items():
        assert torch.equal(value, read_back[1][key]), key
        if key in trained:
            assert torch.equal(value, trained[key]), key


def test_a_text_only_run_reattaches_with_the_head_it_trained(tmp_path, corpus):
    """A composite trained as its text sibling: the export's head supersedes the base's, whole, onto the wrapper
    layout, and the vision tower returns from the base."""
    checkpoint = _run("qwen3_5_moe", tmp_path, corpus, ranks=1, text_only_model=True)
    export = saved_tensors(checkpoint)
    base = saved_tensors(tmp_path / "base")
    out = tmp_path / "reattached"
    reattach_vision_tower(str(checkpoint), str(tmp_path / "base"), str(out), trust_remote_code=False)
    written = saved_tensors(out)
    head = {key for key in export if key.startswith("mtp.")}
    assert head and head == {key for key in written if key.startswith("mtp.")}
    for key in head:
        assert torch.equal(written[key], export[key]), key
    assert any(key.startswith("model.visual.") for key in written), "the vision tower did not return"
    assert not any(key.startswith("mtp.") for key in base), "the base held no head, so nothing was superseded"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
