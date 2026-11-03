#!/usr/bin/env python
"""Qwen3.8-Flash-Next (``qwen4_exp``) pipeline stages: the flat hyper-connection boundary, the token ids its per-layer
embeddings re-read on a later stage, the segment markers a packed microbatch's stage rebuilds, and the external tables
a stage saves and attaches under the unsplit model's names.

The tiny model carries a per-layer-embedding (PLE) layer on each stage of the ``[2, 2]`` split (``ple_layer_ids`` 2
and 3, layers 1 and 2, both GatedDeltaNet as the family requires), each with its own host n-gram table. The
load-bearing check is BITWISE chained equivalence in fp32: stage 0 → boundary stream → stage 1 runs the unsplit
model's ops in its order, so any drift means the boundary, the token ids or the markers lost state.

    python tests/cpu/parallelism/test_pp_qwen4_exp_stages.py
"""

import copy
import os
import shutil
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors import safe_open
from transformers import AutoModelForImageTextToText

from halo.checkpoint.external_tables import write_external_tables
from halo.checkpoint.format import PP_STAGE_PART_RE, pp_stage_part_prefix
from halo.checkpoint.parallel import pp as pp_checkpoint
from halo.checkpoint.parallel.pp import save_pp_checkpoint
from halo.checkpoint.reading import resolve_checkpoint_weights
from halo.checkpoint.tools.pp_stage_merge import merge_pp_stage_parts
from halo.checkpoint.writing import StreamingShardWriter
from halo.distributed.parallelism_config import ParallelismConfig
from halo.distributed.pipeline_parallel import stage as stage_module
from halo.distributed.pipeline_parallel.families import PP_SPEC_MAP
from halo.distributed.pipeline_parallel.model_rules import reject_unsupported_pp_model
from halo.distributed.pipeline_parallel.partition import validate_partition
from halo.distributed.pipeline_parallel.stage import build_pipeline_stage
from halo.model_loading.external_tables import attach_external_tables
from halo.model_loading.parallel.pp import _load_stage_lazy
from halo.models.external_tables import named_external_tables
from tests.common.checkpoint_io import save_context
from tests.common.gloo import run_gloo_ranks
from tests.common.qwen4_exp import (
    EOS_TOKEN_ID,
    build_tiny_qwen4_exp_checkpoint,
    load_host_table_model,
    packed_markers,
    packed_positions,
    stamped_config,
)
from tests.common.tiny_models import TINY_MOE_VLM_FAMILIES, tiny_family_model

# A PLE layer on each stage of the [2, 2] split, and both on stage 1 of the off-period [1, 3] one.
PLE_LAYER_IDS = [2, 3]
SPLITS = ([2, 2], [1, 3])
SEQ = 24
PACKED_DOCUMENTS = [[7, 9, 8]]
# pp2 x dp2 on two simulated nodes of two ranks, one stage per node.
WORLD_SIZE = 4
RANKS_PER_NODE = 2
SAVE_SPLIT = [2, 2]


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> str:
    directory = str(tmp_path_factory.mktemp("qwen4_exp_pp"))
    build_tiny_qwen4_exp_checkpoint(directory, overrides={"ple_layer_ids": PLE_LAYER_IDS})
    return directory


@pytest.fixture(scope="module")
def model(checkpoint):
    torch.manual_seed(0)
    return load_host_table_model(checkpoint).eval()


def _stages(model, pp_split):
    # A deep copy shares the attached table views, as every rank on a node shares the page cache.
    return [
        build_pipeline_stage(copy.deepcopy(model), rank, len(pp_split), moe_balancing="none", pp_split=pp_split)
        for rank in range(len(pp_split))
    ]


def _input_ids(model, batch: int, length: int) -> torch.Tensor:
    vocab_size = model.config.get_text_config().vocab_size
    ids = torch.randint(2, vocab_size, (batch, length), generator=torch.Generator().manual_seed(3))
    ids[:, length // 3] = EOS_TOKEN_ID  # the n-gram history restarts after eos on every stage alike
    return ids


def _padded_rows(model):
    input_ids = _input_ids(model, 2, SEQ)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, SEQ - 5 :] = 0
    return input_ids, {"attention_mask": attention_mask}


def _packed_row(model):
    position_ids = packed_positions(PACKED_DOCUMENTS)
    return _input_ids(model, 1, position_ids.shape[1]), {"position_ids": position_ids}


def _chain(stages, input_ids, token_ids=None, **kwargs):
    """``(boundary stream, logits)`` through the chained stages; later stages take ``token_ids`` (default: the
    microbatch's own ids)."""
    activation, boundary = stages[0](input_ids, **kwargs), None
    for stage in stages[1:]:
        boundary = activation
        activation = stage(activation, token_ids=input_ids if token_ids is None else token_ids, **kwargs)
    return boundary, activation


def _reference(model, input_ids, **kwargs):
    markers = packed_markers(model.config, kwargs["position_ids"]) if "position_ids" in kwargs else {}
    return model(input_ids=input_ids, **kwargs, **markers).logits


@pytest.mark.parametrize("pp_split", SPLITS, ids=str)
@pytest.mark.parametrize("rows", [_padded_rows, _packed_row], ids=["padded", "packed"])
def test_chained_stages_reproduce_the_unsplit_model_bitwise(model, pp_split, rows):
    input_ids, kwargs = rows(model)
    with torch.no_grad():
        reference = _reference(model, input_ids, **kwargs)
        boundary, logits = _chain(_stages(model, pp_split), input_ids, **kwargs)
    config = model.config
    assert boundary.shape == (*input_ids.shape, config.hc_count * config.hidden_size), tuple(boundary.shape)
    assert torch.equal(logits, reference), (logits - reference).abs().max().item()


def test_gradients_match_the_unsplit_model_through_the_boundary(model):
    """The stream stays on the graph and every stage's gradient is the unsplit model's, by unsplit name; the QSA
    indexer projection gets none on either side (its top-k is a boolean selection)."""
    input_ids, kwargs = _packed_row(model)
    reference_model = copy.deepcopy(model)
    _reference(reference_model, input_ids, **kwargs).pow(2).mean().backward()
    expected = {name: param.grad for name, param in reference_model.named_parameters() if param.grad is not None}

    stages = _stages(model, [2, 2])
    _chain(stages, input_ids, **kwargs)[1].pow(2).mean().backward()
    compared = set()
    for stage in stages:
        for local, param in stage.named_parameters():
            name = stage.global_parameter_name(local)
            if param.grad is None:
                assert name not in expected, f"{name} got no gradient through the chain"
                continue
            assert torch.equal(param.grad, expected[name]), name
            compared.add(name)
    assert compared == set(expected), sorted(set(expected) ^ compared)[:4]
    assert any(".ple." in name for name in compared if name.startswith("model.layers.2.")), "anti-vacuity"


def test_a_later_stage_reads_the_token_ids_it_is_handed(model):
    """Negative controls for the side channel: withheld, the stage refuses; others in their place change the
    logits, which only the stage-1 PLE layer reads."""
    input_ids, kwargs = _padded_rows(model)
    stages = _stages(model, [2, 2])
    stream = stages[0](input_ids, **kwargs)
    assert stages[1].model.layers[0].ple is not None, "premise: stage 1 holds a PLE layer"
    with pytest.raises(RuntimeError, match="token ids as 'ple_input_ids', but none arrived"):
        stages[1](stream, **kwargs)
    with pytest.raises(RuntimeError, match="token ids never, but some arrived"):
        stages[0](input_ids, token_ids=input_ids, **kwargs)
    with torch.no_grad():
        right = stages[1](stream, token_ids=input_ids, **kwargs)
        wrong = stages[1](stream, token_ids=input_ids.roll(1, dims=1), **kwargs)
    assert not torch.equal(right, wrong)


def test_a_packed_stage_runs_on_the_markers_it_rebuilds(model):
    """Without the stage's rebuilt ``seq_idx`` a packed row reaches the document forward unmarked, which refuses it."""
    input_ids, kwargs = _packed_row(model)
    stages = _stages(model, [2, 2])
    with mock.patch.object(stage_module, "packed_row_segment_kwargs", lambda position_ids, markers: {}):
        with pytest.raises(ValueError, match="arrived without seq_idx"):
            stages[0](input_ids, **kwargs)
    handed = stages[0]._segment_kwargs(input_ids, None, kwargs["position_ids"])
    expected = packed_markers(model.config, kwargs["position_ids"])
    assert handed.keys() == expected.keys() and {"seq_idx", "cu_seq_lens_q"} <= handed.keys()
    assert all(
        torch.equal(handed[key], value) if torch.is_tensor(value) else handed[key] == value
        for key, value in expected.items()
    )


@pytest.fixture(scope="module")
def composite(tmp_path_factory):
    """The multimodal class over the same tiny text tower, loaded as a run loads it."""
    torch.manual_seed(0)
    directory = str(tmp_path_factory.mktemp("qwen4_exp_composite"))
    tiny_family_model(TINY_MOE_VLM_FAMILIES["qwen4_exp"]).save_pretrained(directory)
    return load_host_table_model(directory, model_class=AutoModelForImageTextToText).eval()


def test_a_text_only_composite_splits_at_its_text_tower(composite):
    """The multimodal class splits at its text tower: stages drive the tower directly and its text-only logits are the
    composite's own, bitwise; the vision tower sits in no stage."""
    first, last = _stages(composite, [2, 2])
    assert first.layout.backbone_prefix == "model.language_model"
    assert not any(name.startswith("model.visual") for name in first.state_dict()), "vision tower inside a stage"
    input_ids, kwargs = _padded_rows(composite)
    with torch.no_grad():
        reference = composite(input_ids=input_ids, **kwargs).logits
        _, logits = _chain([first, last], input_ids, **kwargs)
    assert torch.equal(logits, reference), (logits - reference).abs().max().item()


def test_a_later_stage_refuses_a_collapsed_boundary(model):
    stage = _stages(model, [2, 2])[1]
    input_ids, kwargs = _padded_rows(model)
    with pytest.raises(RuntimeError, match="hyper-connection stream"):
        stage(torch.randn(*input_ids.shape, model.config.hidden_size), token_ids=input_ids, **kwargs)


def test_the_stream_collapse_is_last_stage_state_and_identity_elsewhere(model):
    first, last = _stages(model, [2, 2])
    assert isinstance(first.model.hyper_connection_mixer, nn.Identity)
    assert not isinstance(last.model.hyper_connection_mixer, nn.Identity)
    first_keys = {first.global_parameter_name(k) for k in first.state_dict()}
    last_keys = {last.global_parameter_name(k) for k in last.state_dict()}
    assert first_keys | last_keys == set(model.state_dict()) and not first_keys & last_keys


def test_the_spec_registers_a_rebase_safe_family(model):
    spec = PP_SPEC_MAP["Qwen4ExpTextModel"]
    assert spec.validated and spec.layer_types_rebase_safe and spec.token_ids_kwarg == "ple_input_ids"
    reject_unsupported_pp_model(model, "none")
    validate_partition(model, [(0, 1), (1, model.config.num_hidden_layers)])  # off the layer_types period


def test_a_stage_saves_its_tables_under_unsplit_names_in_files_of_its_own(model, tmp_path):
    """Stage 1's table is global layer 2's; its files carry the stage's part prefix, so stage 0's table00 files and
    stage 1's never share a name in one directory."""
    stage = _stages(model, [2, 2])[1]
    prefix = pp_stage_part_prefix(1, 2)
    writer = StreamingShardWriter(str(tmp_path), "1GB", enabled=True, filename_prefix=prefix)
    write_external_tables(stage, writer, checkpoint_name=stage.global_parameter_name, file_prefix=prefix)
    weight_map, _ = writer.close()
    (table,) = (table for _, table in named_external_tables(stage))
    keys = [f"model.layers.2.ple.ple_embedding.ngram_embedding.shard_{i}.weight" for i in range(table.part_count)]
    assert sorted(weight_map) == sorted(keys), sorted(weight_map)
    assert all(name.startswith(f"{prefix}-table00-") for name in weight_map.values()), set(weight_map.values())
    for key, name in weight_map.items():
        with safe_open(os.path.join(tmp_path, name), framework="pt") as handle:
            index = int(key.split(".shard_")[1].split(".")[0])
            assert torch.equal(handle.get_tensor(key), table.parts[index])


def _checkpoint_table(checkpoint: str, layer: int) -> list[torch.Tensor]:
    weight_map = resolve_checkpoint_weights(checkpoint).weight_map
    prefix = f"model.layers.{layer}.ple.ple_embedding.ngram_embedding.shard_"
    keys = sorted(
        (key for key in weight_map if key.startswith(prefix)), key=lambda k: int(k[len(prefix) :].split(".")[0])
    )
    parts = []
    for key in keys:
        with safe_open(os.path.join(checkpoint, weight_map[key]), framework="pt") as handle:
            parts.append(handle.get_tensor(key))
    return parts


def test_a_later_stage_attaches_its_own_layers_table(checkpoint):
    """The stage loader plans the attach in the checkpoint's key space: stage 1's local layer 0 is global layer 2. By
    its stage-local name the same table would look for global layer 0's keys, which hold no table."""
    stage = _load_stage_lazy(
        checkpoint,
        1,
        2,
        config=stamped_config(checkpoint),
        pp_split=[2, 2],
        dtype=torch.float32,
        trust_remote_code=False,
    )
    ((name, table),) = named_external_tables(stage)
    assert name == "model.layers.0.ple.ple_embedding.ngram_embedding"
    expected = _checkpoint_table(checkpoint, 2)
    assert len(table.parts) == len(expected) and all(map(torch.equal, table.parts, expected))

    table.parts = ()  # detached, as a fresh build is
    with pytest.raises(FileNotFoundError, match="holds no keys that load into"):
        attach_external_tables(stage, checkpoint)


def _save_dir(root: str, rank: int, per_node: bool) -> str:
    return os.path.join(root, f"node{rank // RANKS_PER_NODE}" if per_node else "shared")


def _save_worker(rank: int, checkpoint: str, root: str, per_node: bool, writes_tables: bool = True) -> None:
    """Save the stages through ``save_pp_checkpoint``, then rebuild each stage stage-aware from the directory its node
    holds, as a resume does: every stage attaches its own layer's table from the files its stage writer wrote.
    ``writes_tables=False`` is the save without its table write."""
    if not writes_tables:
        mock.patch.object(pp_checkpoint, "write_external_tables", lambda *args, **kwargs: None).start()
    os.environ["LOCAL_RANK"] = str(rank % RANKS_PER_NODE)
    config = ParallelismConfig(
        pp_size=2, nvlink_domain_size=RANKS_PER_NODE, max_concurrent_loading=0, pp_split=SAVE_SPLIT
    )
    out_dir = _save_dir(root, rank, per_node)
    if rank % RANKS_PER_NODE == 0:
        os.makedirs(out_dir, exist_ok=True)
    dist.barrier()
    stage = build_pipeline_stage(
        load_host_table_model(checkpoint), config.pp_rank, 2, moe_balancing="none", pp_split=SAVE_SPLIT
    )
    save_pp_checkpoint(save_context(stage, config), out_dir)

    resumed = _load_stage_lazy(
        out_dir,
        config.pp_rank,
        2,
        config=stamped_config(out_dir),
        pp_split=SAVE_SPLIT,
        dtype=torch.float32,
        trust_remote_code=False,
    )
    ((_, saved),), ((_, attached),) = named_external_tables(stage), named_external_tables(resumed)
    assert attached.files.directory == out_dir and all(map(torch.equal, attached.parts, saved.parts)), rank


def _table_files(directory: str) -> list[str]:
    return sorted(name for name in os.listdir(directory) if "-table" in name)


def _probe_logits(directory: str) -> torch.Tensor:
    model = load_host_table_model(directory)
    input_ids, kwargs = _padded_rows(model)
    with torch.no_grad():
        return model(input_ids=input_ids, **kwargs).logits


def test_a_pp_save_holds_every_stages_tables_and_reloads_unsplit(checkpoint, tmp_path):
    """On a shared output filesystem one directory holds both stages' tables, each in files under its own stage's part
    prefix and indexed under the unsplit model's names, and reloads with them as the source model."""
    run_gloo_ranks(_save_worker, WORLD_SIZE, checkpoint, str(tmp_path), False, env={"LOCAL_WORLD_SIZE": "2"})
    saved = str(tmp_path / "shared")
    weight_map = resolve_checkpoint_weights(saved).weight_map
    for layer, stage in ((1, 0), (2, 1)):
        files = {
            name
            for key, name in weight_map.items()
            if key.startswith(f"model.layers.{layer}.ple.ple_embedding") and ".ngram_embedding.shard_" in key
        }
        assert files and all(name.startswith(f"{pp_stage_part_prefix(stage, 2)}-table00-") for name in files), files
    assert torch.equal(_probe_logits(saved), _probe_logits(checkpoint))


def test_without_its_table_write_the_saved_stage_cannot_resume(checkpoint, tmp_path):
    """Negative control for the save: a PP save that writes no table leaves a checkpoint whose stages refuse to attach
    one on resume."""
    with pytest.raises(Exception, match="holds no keys that load into"):
        run_gloo_ranks(
            _save_worker, WORLD_SIZE, checkpoint, str(tmp_path), False, False, env={"LOCAL_WORLD_SIZE": "2"}
        )


def test_per_node_pp_saves_merge_with_their_tables(checkpoint, tmp_path):
    """Per-node storage: each node's directory holds only its own stage's table files, and the merge carries both into
    a directory that reloads as the source model. A node directory missing its stage's table fails the merge."""
    run_gloo_ranks(
        _save_worker,
        WORLD_SIZE,
        checkpoint,
        str(tmp_path),
        True,
        env={"LOCAL_WORLD_SIZE": "2", "DIST_OUTPUT_SHARED_FILESYSTEM": "0"},
    )
    nodes = [str(tmp_path / f"node{node}") for node in range(2)]
    for node, directory in enumerate(nodes):
        files = _table_files(directory)
        assert files and all(int(PP_STAGE_PART_RE.match(name)[1]) == node for name in files), (node, files)

    merged = str(tmp_path / "merged")
    merge_pp_stage_parts(nodes, merged)
    assert _table_files(merged) == sorted(_table_files(nodes[0]) + _table_files(nodes[1]))
    assert torch.equal(_probe_logits(merged), _probe_logits(checkpoint))

    torn = str(tmp_path / "node1-torn")
    shutil.copytree(nodes[1], torn)
    os.remove(os.path.join(torn, _table_files(torn)[0]))
    with pytest.raises(ValueError, match=r"table00.*\(stage 1\)"):
        merge_pp_stage_parts([nodes[0], torn], str(tmp_path / "merged-torn"))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
