#!/usr/bin/env python
"""Nemotron-H's text class split into real pipeline stages. Chained stages must match bitwise in fp32, so any drift means
state was lost at a cut; its blocks follow no period, so every split offset has to hold."""

import copy
import dataclasses
import os
import re
from unittest import mock

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM, NemotronHConfig, NemotronHForCausalLM
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHMamba2Mixer

from halo.checkpoint.parallel.pp import save_pp_checkpoint
from halo.checkpoint.reading import iter_checkpoint_tensors
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from halo.distributed.pipeline_parallel import families
from halo.distributed.pipeline_parallel import stage as stage_module
from halo.distributed.pipeline_parallel.families import NEMOTRON_H_PP, PP_SPEC_MAP
from halo.distributed.pipeline_parallel.model_rules import reject_unsupported_pp_structure
from halo.distributed.pipeline_parallel.partition import (
    head_cost_layer_equivalents,
    resolve_layer_partition,
    validate_partition,
)
from halo.distributed.pipeline_parallel.stage import PipelineStageModule, build_pipeline_stage
from halo.model_loading.parallel.pp import _load_stage_lazy
from halo.models.nemotron_h import apply_nemotron_h_patches
from tests.common.checkpoint_io import save_context
from tests.common.gloo import run_gloo_ranks
from tests.common.models import NEMOTRON_H_RELEASE_BLOCKS, NEMOTRON_H_RELEASE_LAYOUT, NEMOTRON_H_RELEASE_TEXT_CONFIG
from tests.common.nemotron_h import NEMOTRON_H_COMPOSITE, write_tiny_hub_checkpoint
from tests.common.parallelism import single_process_ep_config
from tests.common.qwen4_exp import packed_markers, packed_positions
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model

_FAMILY = TINY_MOE_FAMILIES["nemotron_h"]
# Each cut off any period the block list could have, the first with no MoE layer on stage 0 and the last with one layer
# on stage 1.
SPLITS = ([1, 5], [2, 4], [3, 3], [5, 1])
SEQ = 24
PACKED_DOCUMENTS = [[7, 9, 8]]
_SCAN_STAGE_SEEN = [2, 4]
# pp2 x dp2 on two simulated nodes of two ranks, one stage per node.
_WORLD_SIZE, _RANKS_PER_NODE = 4, 2
_RUN_DTYPE = torch.bfloat16
# ``in_proj`` scaled up where a row must cross: at stock init a crossed boundary moves the logits by rounding alone.
_IN_PROJ_SCALE = 50.0


@pytest.fixture(scope="module", autouse=True)
def _patched():
    apply_nemotron_h_patches()


@pytest.fixture
def model():
    torch.manual_seed(0)
    built = tiny_family_model(_FAMILY).eval()
    with torch.no_grad():
        for module in built.modules():
            if isinstance(module, NemotronHMamba2Mixer):
                module.in_proj.weight.mul_(_IN_PROJ_SCALE)
    return built


def _stages(model, pp_split, **kwargs) -> list[PipelineStageModule]:
    kwargs.setdefault("moe_balancing", "none")
    return [
        build_pipeline_stage(copy.deepcopy(model), rank, len(pp_split), pp_split=pp_split, **kwargs)
        for rank in range(len(pp_split))
    ]


def _input_ids(model, batch: int, length: int) -> torch.Tensor:
    return torch.randint(30, model.config.vocab_size, (batch, length), generator=torch.Generator().manual_seed(3))


def _padded_rows(model):
    input_ids = _input_ids(model, 2, SEQ)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, SEQ - 5 :] = 0
    return input_ids, {"attention_mask": attention_mask}


def _left_padded_rows(model):
    input_ids = _input_ids(model, 2, SEQ)
    attention_mask = torch.ones_like(input_ids)
    attention_mask[1, :5] = 0
    return input_ids, {"attention_mask": attention_mask}


def _packed_row(model):
    position_ids = packed_positions(PACKED_DOCUMENTS)
    return _input_ids(model, 1, position_ids.shape[1]), {"position_ids": position_ids}


def _chain(stages, input_ids, **kwargs) -> torch.Tensor:
    activation = input_ids
    for stage in stages:
        activation = stage(activation, **kwargs)
    return activation


def _reference(model, input_ids, **kwargs) -> torch.Tensor:
    markers = packed_markers(model.config, kwargs["position_ids"]) if "position_ids" in kwargs else {}
    return model(input_ids=input_ids, use_cache=False, **kwargs, **markers).logits


@pytest.mark.parametrize("pp_split", SPLITS, ids=str)
@pytest.mark.parametrize(
    "rows", [_padded_rows, _left_padded_rows, _packed_row], ids=["right_padded", "left_padded", "packed"]
)
def test_chained_stages_reproduce_the_unsplit_model_bitwise(model, pp_split, rows):
    input_ids, kwargs = rows(model)
    with torch.no_grad():
        reference = _reference(model, input_ids, **kwargs)
        logits = _chain(_stages(model, pp_split), input_ids, **kwargs)
    assert torch.equal(logits, reference), (logits - reference).abs().max().item()


def test_gradients_match_the_unsplit_model_through_the_boundary(model):
    input_ids, kwargs = _padded_rows(model)
    reference_model = copy.deepcopy(model)
    _reference(reference_model, input_ids, **kwargs).pow(2).mean().backward()
    expected = {name: param.grad for name, param in reference_model.named_parameters() if param.grad is not None}

    stages = _stages(model, _SCAN_STAGE_SEEN)
    _chain(stages, input_ids, **kwargs).pow(2).mean().backward()
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
    assert any(".mixer.in_proj." in name for name in compared if name.startswith("model.layers.4.")), "anti-vacuity"


def test_the_final_norm_lands_on_the_last_stage_and_each_key_on_one_stage(model):
    first, last = _stages(model, _SCAN_STAGE_SEEN)
    assert isinstance(first.model.norm_f, nn.Identity)
    assert not isinstance(last.model.norm_f, nn.Identity)
    assert first.model.embeddings is not None and last.model.embeddings is None
    first_keys = {first.global_parameter_name(key) for key in first.state_dict()}
    last_keys = {last.global_parameter_name(key) for key in last.state_dict()}
    assert first_keys | last_keys == set(model.state_dict()) and not first_keys & last_keys


def test_a_packed_stage_scans_on_the_markers_it_rebuilds(model):
    """Without them every later stage's Mamba layer carries one document into the next."""
    input_ids, kwargs = _packed_row(model)
    with torch.no_grad():
        reference = _reference(model, input_ids, **kwargs)
        with mock.patch.object(stage_module, "packed_row_segment_kwargs", lambda position_ids, markers: {}):
            crossed = _chain(_stages(model, _SCAN_STAGE_SEEN), input_ids, **kwargs)
    assert (crossed - reference).abs().max() > 1e-2


def test_expert_parallel_stages_reproduce_the_ep_model_and_resolve_the_bias(model):
    wrapped = apply_ep_layers(copy.deepcopy(model), single_process_ep_config())
    input_ids, kwargs = _padded_rows(wrapped)
    with torch.no_grad():
        reference = wrapped(input_ids=input_ids, use_cache=False, **kwargs).logits
        stages = _stages(wrapped, _SCAN_STAGE_SEEN, moe_balancing="auto")
        logits = _chain(stages, input_ids, **kwargs)
    assert all(named_ep_layers(stage) for stage in stages), "a MoE layer on each stage of this split"
    assert torch.equal(logits, reference), (logits - reference).abs().max().item()


def test_the_release_splits_by_its_head_cost_with_a_mixer_of_every_kind_on_each_stage():
    """On a meta shell of the real block list, which has no period to snap to."""
    with torch.device("meta"):
        shell = NemotronHForCausalLM(NemotronHConfig(**NEMOTRON_H_RELEASE_TEXT_CONFIG))
    assert head_cost_layer_equivalents(shell.config) > 0
    partition = resolve_layer_partition(shell, 2)
    validate_partition(shell, partition)
    assert partition[0][0] == 0 and partition[-1][1] == len(NEMOTRON_H_RELEASE_LAYOUT)
    for lo, hi in partition:
        assert set(NEMOTRON_H_RELEASE_LAYOUT[lo:hi]) == set(NEMOTRON_H_RELEASE_BLOCKS)


def test_the_spec_registers_a_rebase_safe_family():
    assert PP_SPEC_MAP["NemotronHModel"] is NEMOTRON_H_PP
    assert NEMOTRON_H_PP.validated and NEMOTRON_H_PP.layer_types_rebase_safe
    assert NEMOTRON_H_PP.final_norm_attrs == ("norm_f",)


def test_the_default_norm_probes_find_no_final_norm(model, monkeypatch):
    monkeypatch.setitem(PP_SPEC_MAP, "NemotronHModel", dataclasses.replace(NEMOTRON_H_PP, final_norm_attrs=("norm",)))
    with pytest.raises(ValueError, match="exposes no final norm"):
        build_pipeline_stage(copy.deepcopy(model), 0, 2, moe_balancing="none", pp_split=_SCAN_STAGE_SEEN)


def test_an_unsafe_rebase_refuses_every_split_of_the_aperiodic_list(model, monkeypatch):
    monkeypatch.setitem(
        PP_SPEC_MAP, "NemotronHModel", dataclasses.replace(NEMOTRON_H_PP, layer_types_rebase_safe=False)
    )
    for pp_split in SPLITS:
        with pytest.raises(ValueError, match=r"config\.layer_types"):
            validate_partition(model, [(0, pp_split[0]), (pp_split[0], sum(pp_split))])


def test_an_unvalidated_spec_is_refused(model, monkeypatch):
    monkeypatch.setitem(PP_SPEC_MAP, "NemotronHModel", dataclasses.replace(NEMOTRON_H_PP, validated=False))
    monkeypatch.setattr(families, "PP_SPEC_MAP", PP_SPEC_MAP)
    with pytest.raises(ValueError, match="not registered for pipeline parallelism"):
        reject_unsupported_pp_structure(model)


def test_the_composite_is_sent_to_its_text_class():
    composite = tiny_family_model(NEMOTRON_H_COMPOSITE)
    with pytest.raises(ValueError, match="text_only_model: true"):
        reject_unsupported_pp_structure(composite)


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> str:
    directory = tmp_path_factory.mktemp("nemotron_h_pp")
    write_tiny_hub_checkpoint(directory, weights_dtype=_RUN_DTYPE)
    return str(directory)


def _text_config(path: str):
    return AutoConfig.from_pretrained(path).text_config


def _load_stage(checkpoint: str, pp_rank: int):
    """As the trainer's loader does, its experts wrapped, the only form a MoE stage takes."""
    return _load_stage_lazy(
        checkpoint,
        pp_rank,
        2,
        config=_text_config(checkpoint),
        pp_split=_SCAN_STAGE_SEEN,
        ep_config=single_process_ep_config(),
        dtype=_RUN_DTYPE,
        trust_remote_code=False,
        model_class=AutoModelForCausalLM,
    )


@pytest.mark.parametrize("pp_rank", [0, 1])
def test_a_stage_aware_load_reads_its_blocks_off_the_release_spelling(checkpoint, pp_rank):
    """With per-expert keys, the composite's level and a multi-token head no stage holds. Each block's own norm and
    each of its experts tell the blocks apart."""
    stored = load_file(os.path.join(checkpoint, "model.safetensors"))
    stage = _load_stage(checkpoint, pp_rank)
    lo = sum(_SCAN_STAGE_SEEN[:pp_rank])
    held = stage.state_dict()
    norms = [key for key in held if re.fullmatch(r"model\.layers\.\d+\.norm\.weight", key)]
    assert len(norms) == _SCAN_STAGE_SEEN[pp_rank]
    for key in norms:
        index = int(key.split(".")[2])
        assert torch.equal(held[key], stored[f"language_model.backbone.layers.{index + lo}.norm.weight"]), key
    moe_layers = named_ep_layers(stage)
    assert moe_layers
    for name, layer in moe_layers:
        index = int(name.split(".")[2]) + lo
        first_expert = stored[f"language_model.backbone.layers.{index}.mixer.experts.0.up_proj.weight"]
        assert torch.equal(layer.up_proj[0].transpose(0, 1), first_expert), name
    assert not any("mtp" in key for key in held)


def _save_worker(rank: int, checkpoint: str, out_dir: str) -> None:
    apply_nemotron_h_patches()
    os.environ["LOCAL_RANK"] = str(rank % _RANKS_PER_NODE)
    config = ParallelismConfig(
        pp_size=2, nvlink_domain_size=_RANKS_PER_NODE, max_concurrent_loading=0, pp_split=_SCAN_STAGE_SEEN
    )
    if rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    dist.barrier()
    model = _load_stage(checkpoint, config.pp_rank)
    stage = build_pipeline_stage(model, config.pp_rank, 2, moe_balancing="none", pp_split=_SCAN_STAGE_SEEN)
    save_pp_checkpoint(save_context(stage, config), out_dir)


def test_a_pp_save_writes_the_towers_hub_spelling_and_reloads_unsplit(checkpoint, tmp_path):
    out_dir = str(tmp_path / "saved")
    run_gloo_ranks(_save_worker, _WORLD_SIZE, checkpoint, out_dir, env={"LOCAL_WORLD_SIZE": "2"})
    written = dict(iter_checkpoint_tensors(out_dir))
    stored = load_file(os.path.join(checkpoint, "model.safetensors"))
    expected = {
        key.removeprefix("language_model."): value
        for key, value in stored.items()
        if key.startswith(("language_model.backbone.", "language_model.lm_head."))
    }
    assert set(written) == set(expected), sorted(set(written) ^ set(expected))[:6]
    for key, value in expected.items():
        assert written[key].dtype == value.dtype and torch.equal(written[key], value), key

    ids = torch.arange(30, 46).unsqueeze(0)
    with torch.no_grad():
        saved = AutoModelForCausalLM.from_pretrained(out_dir, dtype=_RUN_DTYPE).eval()(input_ids=ids).logits
        whole = AutoModelForCausalLM.from_pretrained(checkpoint, config=_text_config(checkpoint), dtype=_RUN_DTYPE)
        reference = whole.eval()(input_ids=ids).logits
    assert torch.equal(saved, reference)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
