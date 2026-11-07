"""A composite release loaded through its text class under pipeline parallelism. Only the text sub-config reaches that
class, so the shell's config tree is not the one the load started from, and the tie the release declares at either level
must still reach the pipeline tie gate and the saved stage config."""

import copy
import json
import os
import shutil
from pathlib import Path

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, Qwen3VLConfig

from halo.checkpoint.config_export import save_model_config
from halo.checkpoint.format import pp_stage_part_prefix
from halo.checkpoint.parallel.pp import save_pp_checkpoint
from halo.checkpoint.reading import resolve_checkpoint_weights
from halo.checkpoint.tools.pp_stage_merge import merge_pp_stage_parts
from halo.checkpoint.tools.vision_reattach import reattach_vision_tower
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from halo.distributed.pipeline_parallel.families import PP_SPEC_MAP
from halo.distributed.pipeline_parallel.stage import PP_STAGE_PARTITION_ATTR, StageLayout, build_pipeline_stage
from halo.model_loading.model_preparation import auto_load_model, resolve_text_only_config
from halo.model_loading.parallel.loader import load_distributed_model
from halo.model_loading.parallel.pp import StageKeyFilter, _instantiate_untied_on_meta, _load_stage_lazy
from halo.models.introspection.config_access import TIE_WORD_EMBEDDINGS_FLAG, text_config, ties_word_embeddings
from halo.models.introspection.module_tree import backbone_with_layers, decoder_layers
from halo.models.moe import family_specs
from tests.common.checkpoint_io import loading_problems, safetensors_state_dict, save_context
from tests.common.gloo import run_gloo_ranks
from tests.common.models import TINY_QWEN3_VL_TEXT_CONFIG, TINY_QWEN3_VL_VISION_CONFIG
from tests.common.mtp_heads import write_checkpoint_with_tail
from tests.common.parallelism import built_ep_config, create_config
from tests.common.qwen4_exp import stamped_config
from tests.common.tiny_models import (
    TINY_MOE_FAMILIES,
    TINY_MOE_VLM_FAMILIES,
    TINY_PP_SPLIT_OVERRIDES,
    TINY_QWEN3_5_COMPOSITE_FAMILY,
    build_tiny_family_checkpoint,
    word_level_tokenizer,
)

DTYPE = torch.bfloat16
GLOO_WORLD = 4
ABSENT = "absent"
EIGHT_LAYERS = {"num_hidden_layers": 8}
# What ``text_only_model`` builds: transformers' Auto factory narrows the first three, the family spec the last.
COMPOSITES = {
    "qwen3_5": (TINY_QWEN3_5_COMPOSITE_FAMILY, EIGHT_LAYERS),
    "qwen3_5_moe": (TINY_MOE_VLM_FAMILIES["qwen3_5_moe"], EIGHT_LAYERS),
    "qwen4_exp": (TINY_MOE_VLM_FAMILIES["qwen4_exp"], {}),
    "nemotron_h_omni": (TINY_MOE_VLM_FAMILIES["nemotron_h_omni"], {}),
}
EXPERT_COMPOSITES = ("qwen3_5_moe", "qwen4_exp", "nemotron_h_omni")
# Root and text flags a release may store; Qwen3.5-35B's ``config.json`` states no text flag at all.
TIED_DECLARATIONS = [(True, True), (True, False), (False, True), (True, ABSENT)]
UNTIED_DECLARATIONS = [(False, False), (False, ABSENT)]
# Each composite on an expert axis where it has experts; every stage parameter moves by the offset, under which the bf16
# values a save and a reload pass through still tell a moved tensor from the release's.
SAVE_AXES = {
    "qwen3_5": {},
    "qwen3_5_moe": {"ep_size": 2},
    "qwen4_exp": {"expert_tp_size": 2},
    "nemotron_h_omni": {"ep_size": 2},
}
SAVE_OFFSET = 0.25
SAVE_TOLERANCE = 0.01


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory) -> dict[str, str]:
    root = tmp_path_factory.mktemp("composite_text_class")
    paths = {}
    for name, (family, overrides) in COMPOSITES.items():
        paths[name] = str(root / name)
        build_tiny_family_checkpoint(family, paths[name], overrides=overrides)
        word_level_tokenizer().save_pretrained(paths[name])
    return paths


def _text_key(raw: dict) -> str:
    return next(key for key in ("text_config", "llm_config") if key in raw)


def _declaring(checkpoint: str, tmp_path: Path, *, root, text) -> str:
    """``checkpoint``'s weights under a ``config.json`` declaring the tie as given."""
    declared = tmp_path / "declared"
    declared.mkdir()
    for entry in os.listdir(checkpoint):
        if entry != "config.json":
            os.symlink(os.path.join(checkpoint, entry), declared / entry)
    with open(os.path.join(checkpoint, "config.json")) as fh:
        raw = json.load(fh)
    for level, flag in ((raw, root), (raw[_text_key(raw)], text)):
        level.pop(TIE_WORD_EMBEDDINGS_FLAG, None)
        if flag != ABSENT:
            level[TIE_WORD_EMBEDDINGS_FLAG] = flag
    with open(declared / "config.json", "w") as fh:
        json.dump(raw, fh)
    return str(declared)


def _load_text_only(path: str, *, rank: int = 0, **axes):
    """``load_distributed_model`` on one simulated rank of a two-stage world."""
    config = create_config(world_size=4, gpus_per_node=2, nvlink_domain_size=2, pp_size=2, rank=rank, **axes)
    if config.needs_ep_layers:
        built_ep_config(config)
    return load_distributed_model(
        path, config, DTYPE, trust_remote_code=False, text_only_model=True, attn_implementation="sdpa"
    )[0]


def _levels(config) -> dict[str, object]:
    return {"root": config, "text": text_config(config)}


def _whole_text_class_state(path: str) -> dict[str, torch.Tensor]:
    return auto_load_model(path, text_only=True, dtype=DTYPE, attn_implementation="sdpa").state_dict()


@pytest.mark.parametrize("rank", [0, 2], ids=["first_stage", "last_stage"])
@pytest.mark.parametrize("name", sorted(COMPOSITES))
def test_an_untied_composite_splits_as_its_text_class(checkpoints, name, rank):
    path = checkpoints[name]
    stage = _load_text_only(path, rank=rank, use_grouped_gemm=False)

    composite = AutoConfig.from_pretrained(path, trust_remote_code=False)
    assert type(stage.config) is type(text_config(composite)), "the text class takes the text level alone"
    assert PP_SPEC_MAP[type(backbone_with_layers(stage)).__name__].validated
    assert not ties_word_embeddings(stage.config)
    assert stage.get_output_embeddings().weight is not stage.get_input_embeddings().weight

    partition = getattr(stage, PP_STAGE_PARTITION_ATTR)
    lo, hi = partition[rank // 2]
    assert len(decoder_layers(backbone_with_layers(stage))) == hi - lo
    # A stage key that re-bases onto the wrong global layer holds another layer's values under a matching shape.
    keys = StageKeyFilter(lo, hi, StageLayout.of(stage).layer_root)
    whole = _whole_text_class_state(path)
    state = stage.state_dict()
    assert not [key for key, tensor in state.items() if tensor.is_meta]
    compared = [key for key in state if keys.model_key(key) in whole]
    assert any(".layers." in key for key in compared), "anti-vacuity: the comparison reached the decoder layers"
    unequal = [key for key in compared if not torch.equal(state[key], whole[keys.model_key(key)])]
    assert not unequal


@pytest.mark.parametrize(
    "axes",
    [{"ep_size": 1}, {"ep_size": 2}, {"expert_tp_size": 2}],
    ids=["pp2", "pp2_ep2", "pp2_etp2"],
)
@pytest.mark.parametrize("name", EXPERT_COMPOSITES)
def test_expert_axes_load_the_text_class_on_every_stage(checkpoints, name, axes):
    for rank in (0, 2):
        stage = _load_text_only(checkpoints[name], rank=rank, **axes)
        assert not ties_word_embeddings(stage.config)
        assert not [key for key, tensor in stage.state_dict().items() if tensor.is_meta]


@pytest.mark.parametrize(("root", "text"), TIED_DECLARATIONS, ids=str)
@pytest.mark.parametrize("name", sorted(COMPOSITES))
def test_a_tie_at_either_level_is_refused_by_the_tie_gate(checkpoints, tmp_path, name, root, text):
    """A text class reads only its own flag: a tie stated above it would otherwise pass the gate and train a random
    head where the release ties it."""
    path = _declaring(checkpoints[name], tmp_path, root=root, text=text)

    with pytest.raises(ValueError, match=TIE_WORD_EMBEDDINGS_FLAG) as excinfo:
        _load_text_only(path, use_grouped_gemm=False)

    assert "pipeline parallelism does not support" in str(excinfo.value)


@pytest.mark.parametrize("config_only", [False, True], ids=["from_pretrained", "per_node_save"])
@pytest.mark.parametrize("name", sorted(COMPOSITES))
def test_a_sibling_towers_flag_is_not_a_tie_of_this_head(checkpoints, tmp_path, name, config_only):
    """An untied release whose ``config.json`` spells a default ``tie_word_embeddings`` on its vision tower, on the shell
    itself and through the whole load: that flag describes the tower, and the head on disk is its own."""
    path = _declaring(checkpoints[name], tmp_path, root=False, text=False)
    config_path = os.path.join(path, "config.json")
    with open(config_path) as fh:
        raw = json.load(fh)
    raw["vision_config"][TIE_WORD_EMBEDDINGS_FLAG] = True
    with open(config_path, "w") as fh:
        json.dump(raw, fh)
    config = AutoConfig.from_pretrained(path, trust_remote_code=False)

    shell = _instantiate_untied_on_meta(
        path,
        AutoModelForCausalLM,
        resolve_text_only_config(config),
        dtype=DTYPE,
        trust_remote_code=False,
        config_only=config_only,
    )
    assert getattr(shell.config, TIE_WORD_EMBEDDINGS_FLAG) is False

    stage = _load_text_only(path, use_grouped_gemm=False)
    assert not ties_word_embeddings(stage.config)


# Hub layouts one module per expert, against a class that stores them fused or module-wise, and the fused layouts too.
PP_MOE_TEXT_CLASSES = ("glm4_moe_lite", "gpt_oss", "laguna", "nemotron_h", "qwen3_5_moe_text", "qwen3_moe")


@pytest.mark.parametrize("name", PP_MOE_TEXT_CLASSES)
def test_a_stage_with_no_expert_axis_holds_the_checkpoints_experts(tmp_path, name):
    """No EP layer wraps a family under ``use_grouped_gemm: false``, and a stage has no ``from_pretrained`` to fuse a
    per-expert checkpoint for it: its experts read as absent from the checkpoint."""
    path = str(tmp_path / "ckpt")
    build_tiny_family_checkpoint(TINY_MOE_FAMILIES[name], path, overrides=TINY_PP_SPLIT_OVERRIDES.get(name))
    word_level_tokenizer().save_pretrained(path)

    stage = _load_text_only(path, use_grouped_gemm=False)

    lo, hi = getattr(stage, PP_STAGE_PARTITION_ATTR)[0]
    keys = StageKeyFilter(lo, hi, StageLayout.of(stage).layer_root)
    whole = _whole_text_class_state(path)
    state = stage.state_dict()
    experts = [key for key in state if "experts" in key]
    assert experts, "anti-vacuity: the stage holds expert tensors"
    assert not [key for key in experts if not torch.equal(state[key], whole[keys.model_key(key)])]


def _edit_saved_tied(model) -> None:
    model.config.tie_word_embeddings = True
    model.tie_weights()


@pytest.mark.parametrize("name", ["qwen3_5", "qwen3_5_moe"])
def test_a_release_saved_with_no_head_key_is_refused(tmp_path, name):
    """The shape Qwen3.5-2B and -4B ship: the head is the embedding on disk, so a passed gate means a random head."""
    family, overrides = COMPOSITES[name]
    path = str(tmp_path / "saved_tied")
    build_tiny_family_checkpoint(family, path, overrides=overrides, edit=_edit_saved_tied)
    word_level_tokenizer().save_pretrained(path)
    assert "lm_head.weight" not in resolve_checkpoint_weights(path).weight_map, (
        "anti-vacuity: the release stores no head"
    )

    with pytest.raises(ValueError, match=TIE_WORD_EMBEDDINGS_FLAG):
        _load_text_only(path, use_grouped_gemm=False)


@pytest.mark.parametrize(
    ("root", "text", "saved"),
    [(True, True, True), (True, False, True), (False, True, True), (True, ABSENT, True), (False, False, False)],
    ids=str,
)
@pytest.mark.parametrize("config_only", [False, True], ids=["from_pretrained", "per_node_save"])
@pytest.mark.parametrize("name", ["qwen3_5", "qwen3_5_moe", "qwen4_exp"])
def test_the_saved_stage_config_states_what_the_release_declared(
    checkpoints, tmp_path, name, config_only, root, text, saved
):
    """An untied flag in that file would reload the head as its own random parameter. A per-node save's absent shards
    build the shell from the config alone, through the factory's other narrowing."""
    path = _declaring(checkpoints[name], tmp_path, root=root, text=text)
    config = AutoConfig.from_pretrained(path, trust_remote_code=False)

    shell = _instantiate_untied_on_meta(
        path, AutoModelForCausalLM, config, dtype=DTYPE, trust_remote_code=False, config_only=config_only
    )

    assert type(shell.config) is type(text_config(config)), "the factory narrowed the composite"
    assert shell.get_output_embeddings().weight is not shell.get_input_embeddings().weight, "tying stays suppressed"
    assert getattr(shell.config, TIE_WORD_EMBEDDINGS_FLAG) is saved
    out = tmp_path / "stage"
    save_model_config(shell, str(out))
    with open(out / "config.json") as fh:
        assert json.load(fh)[TIE_WORD_EMBEDDINGS_FLAG] is saved


@pytest.mark.parametrize(("root", "text"), TIED_DECLARATIONS + UNTIED_DECLARATIONS, ids=str)
def test_the_load_leaves_the_source_config_as_declared(checkpoints, tmp_path, root, text):
    path = _declaring(checkpoints["qwen3_5_moe"], tmp_path, root=root, text=text)
    config = AutoConfig.from_pretrained(path, trust_remote_code=False)
    before = {level: bool(getattr(owner, TIE_WORD_EMBEDDINGS_FLAG)) for level, owner in _levels(config).items()}

    _instantiate_untied_on_meta(path, AutoModelForCausalLM, config, dtype=DTYPE, trust_remote_code=False)

    assert {level: getattr(owner, TIE_WORD_EMBEDDINGS_FLAG) for level, owner in _levels(config).items()} == before


class _ClassTakingTheTextLevel:
    """A text class under a factory that hands it the text sub-config alone."""

    @staticmethod
    def from_pretrained(_path, *, config, **_kwargs):
        module = torch.nn.Module()
        module.config = copy.deepcopy(text_config(config))
        return module


@pytest.mark.parametrize("declared", [True, False])
def test_a_text_level_with_no_flag_of_its_own_takes_the_composites(declared):
    """Qwen3-VL's text config states no flag at all, leaving only the root's to read."""
    text = {key: value for key, value in TINY_QWEN3_VL_TEXT_CONFIG.items() if key != TIE_WORD_EMBEDDINGS_FLAG}
    config = Qwen3VLConfig(
        text_config=text, vision_config=dict(TINY_QWEN3_VL_VISION_CONFIG), tie_word_embeddings=declared
    )
    assert not hasattr(text_config(config), TIE_WORD_EMBEDDINGS_FLAG), "the premise: the text level states nothing"

    shell = _instantiate_untied_on_meta(
        "unread", _ClassTakingTheTextLevel, config, dtype=DTYPE, trust_remote_code=False
    )

    assert type(shell.config) is type(text_config(config))
    assert ties_word_embeddings(shell.config) is declared


class _ShellDroppingTheTextLevel:
    """A class handed the whole tree that keeps its root and drops the level declaring the tie."""

    @staticmethod
    def from_pretrained(_path, *, config, **_kwargs):
        module = torch.nn.Module()
        module.config = AutoConfig.for_model(config.model_type)
        module.config.text_config = None
        return module


def test_a_shell_missing_a_level_it_should_carry_still_raises(checkpoints):
    """The narrowing is told apart from a lost level by the class it built from."""
    path = checkpoints["qwen3_5_moe"]
    config = AutoConfig.from_pretrained(path, trust_remote_code=False)

    with pytest.raises(RuntimeError, match=r"lost the config level\(s\) \['text_config\.'\]"):
        _instantiate_untied_on_meta(path, _ShellDroppingTheTextLevel, config, dtype=DTYPE, trust_remote_code=False)


def _record_gloo_load(rank: int, path: str, axes: dict, out: str) -> None:
    config = ParallelismConfig(world_size=GLOO_WORLD, gpus_per_node=2, nvlink_domain_size=2, pp_size=2, **axes)
    stage = load_distributed_model(
        path, config, DTYPE, trust_remote_code=False, text_only_model=True, attn_implementation="sdpa"
    )[0]
    record = {
        "rank": rank,
        "cls": type(stage).__name__,
        "partition": getattr(stage, PP_STAGE_PARTITION_ATTR),
        "layers": len(decoder_layers(backbone_with_layers(stage))),
        "meta": [key for key, tensor in stage.state_dict().items() if tensor.is_meta],
        "ties": ties_word_embeddings(stage.config),
    }
    with open(f"{out}.{rank}.json", "w") as fh:
        json.dump(record, fh)


@pytest.mark.parametrize("axes", [{"ep_size": 2}, {"expert_tp_size": 2}], ids=["pp2_ep2", "pp2_etp2"])
def test_the_real_groups_load_every_stage_through_the_text_class(checkpoints, tmp_path, axes):
    """Every rank reaches the end of the load on real process groups, so no collective strands a stage."""
    out = str(tmp_path / "record")
    run_gloo_ranks(_record_gloo_load, GLOO_WORLD, checkpoints["qwen3_5_moe"], axes, out)

    records = [json.loads(Path(f"{out}.{rank}.json").read_text()) for rank in range(GLOO_WORLD)]
    assert {record["cls"] for record in records} == {"Qwen3_5MoeForCausalLM"}
    assert all(not record["meta"] and not record["ties"] for record in records)
    partition = records[0]["partition"]
    assert all(record["partition"] == partition for record in records)
    assert [record["layers"] for record in records] == [
        hi - lo for lo, hi in partition for _ in range(GLOO_WORLD // len(partition))
    ]


def _gloo_config(axes: dict) -> ParallelismConfig:
    return ParallelismConfig(world_size=GLOO_WORLD, gpus_per_node=2, nvlink_domain_size=2, pp_size=2, **axes)


def _dump_path(root: str, rank: int) -> str:
    return f"{root}.{rank}.pt"


def _save_a_moved_stage(rank: int, path: str, axes: dict, out: str, dump: str) -> None:
    config = _gloo_config(axes)
    model = load_distributed_model(
        path, config, DTYPE, trust_remote_code=False, text_only_model=True, attn_implementation="sdpa"
    )[0]
    stage = build_pipeline_stage(model, config.pp_rank, config.pp_size, moe_balancing="none")
    with torch.no_grad():
        for parameter in stage.parameters():
            parameter.add_(SAVE_OFFSET)
    os.makedirs(out, exist_ok=True)
    save_pp_checkpoint(save_context(stage, config, has_ep_layers=bool(named_ep_layers(stage))), out)
    torch.save({key: tensor.clone() for key, tensor in stage.state_dict().items()}, _dump_path(dump, rank))


def _record_resumes(rank: int, shared: str, node_dirs: list[str], axes: dict, dump: str, out: str) -> None:
    """The resume a training script takes from one directory, and a per-node save's from each node's own, whose absent
    parts build the shell from the config alone, against the tensors this rank saved."""
    config = _gloo_config(axes)
    saved = torch.load(_dump_path(dump, rank), weights_only=True)
    resumed = load_distributed_model(
        shared, config, DTYPE, trust_remote_code=False, text_only_model=True, attn_implementation="sdpa"
    )[0]
    node_config = resolve_text_only_config(stamped_config(node_dirs[config.pp_rank]))
    wraps_experts = family_specs.wraps_experts(node_config, ep_layers_requested=config.needs_ep_layers)
    per_node = _load_stage_lazy(
        node_dirs[config.pp_rank],
        config.pp_rank,
        config.pp_size,
        config=node_config,
        ep_config=config.ep_config() if wraps_experts else None,
        dtype=DTYPE,
        trust_remote_code=False,
        model_class=AutoModelForCausalLM,
        attn_implementation="sdpa",
    )
    report = {}
    for label, model in (("shared", resumed), ("per_node", per_node)):
        state = build_pipeline_stage(model, config.pp_rank, config.pp_size, moe_balancing="none").state_dict()
        report[label] = sorted(set(state) ^ set(saved)) + [
            key for key in saved.keys() & state.keys() if not torch.equal(state[key], saved[key])
        ]
    with open(f"{out}.{rank}.json", "w") as fh:
        json.dump(report, fh)


def _load_cleanly(model_class, path: str):
    model, info = model_class.from_pretrained(
        path, dtype=DTYPE, attn_implementation="sdpa", output_loading_info=True, trust_remote_code=False
    )
    assert not loading_problems(info)
    return model


def _close(tensor: torch.Tensor, expected: torch.Tensor) -> bool:
    return torch.allclose(tensor.float(), expected.float(), atol=SAVE_TOLERANCE, rtol=0.0)


@pytest.mark.parametrize("name", sorted(SAVE_AXES))
def test_a_stage_save_of_a_composite_serves_every_later_path(checkpoints, tmp_path, name):
    """The save is the text class's own checkpoint, a merge of its per-node directories reloads, the composite comes back
    with the release's flags, and both resumes read back what each rank saved."""
    axes, release = SAVE_AXES[name], checkpoints[name]
    save, dump = str(tmp_path / "save"), str(tmp_path / "dump")
    run_gloo_ranks(_save_a_moved_stage, GLOO_WORLD, release, axes, save, dump)
    word_level_tokenizer().save_pretrained(save)

    node_dirs = []
    for pp_rank in range(2):
        node_dir = tmp_path / f"node{pp_rank}"
        node_dir.mkdir()
        for entry in os.listdir(save):
            if not entry.startswith(pp_stage_part_prefix(1 - pp_rank, 2)):
                shutil.copy2(os.path.join(save, entry), node_dir)
        node_dirs.append(str(node_dir))
    merged = str(tmp_path / "merged")
    merge_pp_stage_parts(node_dirs, merged)

    with open(os.path.join(release, "config.json")) as fh:
        declared = json.load(fh)
    with open(os.path.join(merged, "config.json")) as fh:
        exported = json.load(fh)
    assert exported["model_type"] == declared[_text_key(declared)]["model_type"]
    assert exported[TIE_WORD_EMBEDDINGS_FLAG] is False

    reloaded = _load_cleanly(AutoModelForCausalLM, merged)
    assert reloaded.get_output_embeddings().weight is not reloaded.get_input_embeddings().weight
    # Stamped as a stage's load is, so an external table on neither side leaves the comparison.
    base = AutoModelForCausalLM.from_pretrained(
        release, config=resolve_text_only_config(stamped_config(release)), dtype=DTYPE, attn_implementation="sdpa"
    )
    moved = {key for key, _ in base.named_parameters()}
    state = reloaded.state_dict()
    assert moved, "anti-vacuity: the base has parameters"
    assert not [
        key
        for key, tensor in base.state_dict().items()
        if not _close(state[key], tensor + SAVE_OFFSET if key in moved else tensor)
    ]

    composite_dir = str(tmp_path / "composite")
    reattach_vision_tower(merged, release, composite_dir, trust_remote_code=False)
    composite = _load_cleanly(AutoModelForImageTextToText, composite_dir)
    assert _close(composite.get_output_embeddings().weight, reloaded.get_output_embeddings().weight)
    with open(os.path.join(composite_dir, "config.json")) as fh:
        regrafted = json.load(fh)
    assert regrafted.get(TIE_WORD_EMBEDDINGS_FLAG) == declared.get(TIE_WORD_EMBEDDINGS_FLAG)
    assert regrafted["text_config"][TIE_WORD_EMBEDDINGS_FLAG] is False

    out = str(tmp_path / "resumes")
    run_gloo_ranks(_record_resumes, GLOO_WORLD, merged, node_dirs, axes, dump, out)
    for rank in range(GLOO_WORLD):
        assert json.loads(Path(f"{out}.{rank}.json").read_text()) == {"shared": [], "per_node": []}


@pytest.mark.parametrize("axes", [{"ep_size": 2}, {"use_grouped_gemm": False}], ids=["pp2_ep2", "pp2_no_expert_axis"])
def test_a_release_tail_stays_out_of_a_stage_and_returns_with_the_composite(tmp_path, axes):
    """Qwen3.5-35B's shape: a fused trunk beside a per-expert MTP tail, whose expert keys must not read as the trunk's
    layout, and which the text class drops at load."""
    release = tmp_path / "release"
    write_checkpoint_with_tail("qwen3_5_moe", release, per_expert=True, overrides=EIGHT_LAYERS)
    word_level_tokenizer().save_pretrained(release)
    tail = {key: tensor for key, tensor in safetensors_state_dict(release).items() if key.startswith("mtp.")}
    assert any(".experts.0." in key for key in tail), "anti-vacuity: the tail stores one module per expert"

    save = str(tmp_path / "save")
    run_gloo_ranks(_save_a_moved_stage, GLOO_WORLD, str(release), axes, save, str(tmp_path / "dump"))
    assert not [key for key in safetensors_state_dict(save) if "mtp" in key]

    composite_dir = str(tmp_path / "composite")
    reattach_vision_tower(save, str(release), composite_dir, trust_remote_code=False)
    stored = safetensors_state_dict(composite_dir)
    assert not [key for key, tensor in tail.items() if not torch.equal(stored[key], tensor)]
    _load_cleanly(AutoModelForImageTextToText, composite_dir)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
