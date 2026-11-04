#!/usr/bin/env python
"""Every EP MoE family has one tiny model and the per-family sweeps run each of them: the roster is held to the EP
registry and the manifest rows to the roster. A family the rows skip would leave its path untested with nothing
failing."""

import shlex

import pytest
from transformers import AutoConfig

from halo.distributed.tensor_parallel.parallelize_attention import TP_SHARDABLE_ATTENTION_CLASSES
from halo.models.moe.family_specs import find_moe_family_spec, find_moe_family_spec_for_config
from halo.weight_sync.registry import resolve_weight_sync_client, rollout_backends
from tests.common import ep_determinism
from tests.common.equivalence.cp_equivalence import FAMILIES as CP_FAMILIES
from tests.common.lora_sync_exactness import parse_row, row_families, syncable_moe_families
from tests.common.merged_resume_e2e import ADAPTER_MODES, LAYOUTS, merged_resume_parser
from tests.common.preference_precompute_e2e import DENSE, precompute_resume_parser, world_size
from tests.common.rosters import ep_layer_class_by_model_type
from tests.common.tiny_models import (
    TINY_BAILING_RING_FAMILY,
    TINY_DENSE_FAMILIES,
    TINY_MOE_FAMILIES,
    TINY_MOE_VLM_FAMILIES,
    build_tiny_family_checkpoint,
    tiny_family_model,
)
from tests.common.weight_push_exact import CHECKPOINTS as PUSH_EXACT_CHECKPOINTS
from tests.common.weight_push_exact import parse_row as parse_push_exact_row
from tests.gpu.manifest import MANIFEST

MERGED_RESUME_SUITE = "trainers/lora/test_lora_merged_save_resume.py"
PRECOMPUTE_RESUME_SUITE = "parallelism/ep/test_ep_preference_precompute_resume.py"
# What every MoE family runs through the precompute-resume body: DPO on every expert layout, KTO once;
# and DPO under attention TP, with and without EP, where the family's attention has a TP plan.
PRECOMPUTE_PER_FAMILY = (("dpo", "ep2"), ("dpo", "ep1"), ("kto", "ep2"), ("dpo", "etp2"), ("dpo", "ep2etp2"))
PRECOMPUTE_PER_TP_FAMILY = (("dpo", "tp2"), ("dpo", "ep2tp2"))
SYNC_EXACTNESS_SUITE = "trainers/lora/test_lora_weight_sync_exact.py"
PUSH_EXACT_SUITE = "trainers/grpo/test_weight_push_exact.py"
DETERMINISM_SUITE = "parallelism/ep/test_ep_deterministic_expert_grads.py"
CP_SUITES = ("parallelism/cp/test_cp_family_vs_reference.py", "parallelism/cp/test_cp_fused_head_loss.py")
REENTRANT_SUITE = "parallelism/test_fsdp2_reentrant_recompute.py"
OFFLINE_GRPO_CP_SUITE = "trainers/grpo/test_offline_grpo_cp_resume.py"
TP_REFERENCE_SUITE = "parallelism/tp/test_tp_family_vs_reference.py"
# The layouts every family's full fine-tune runs under FSDP2 against the unsharded reference: experts in the FSDP2
# shard, and two EP groups whose replicas the deferred sweep keeps in step.
REENTRANT_PER_FAMILY_LAYOUTS = ("ep1", "ep_multi")
# The family the determinism replay runs under every two-rank layout; every other family runs ep2.
DETERMINISM_LAYOUT_FAMILY = "gpt_oss"
# What every syncable MoE family runs through the sync-exactness body, as (mode, adapters): both expert
# layouts with attention PEFT alone and mixed with expert LoRA, and pure ETP, which refuses expert LoRA.
# A dense family runs fsdp alone.
SYNC_EXACTNESS_PER_MOE_FAMILY = (
    ("ep1", "peft"),
    ("ep1", "mixed"),
    ("ep2", "peft"),
    ("ep2", "mixed"),
    ("etp2", "peft"),
)


def _launches(suite: str, parse) -> list:
    """``(parsed row, the spec it launches under)`` for every row of ``suite``."""
    return [(parse(shlex.split(args)), spec) for args, spec in MANIFEST[suite].launches()]


def _rows(suite: str, parse) -> list:
    return [row for row, _ in _launches(suite, parse)]


def test_every_ep_family_has_one_tiny_model():
    registry = ep_layer_class_by_model_type()
    unregistered = sorted(name for name in TINY_MOE_FAMILIES if name not in registry)
    assert not unregistered, f"tiny families no EP layer class claims: {unregistered}"
    covered = [registry[name] for name in TINY_MOE_FAMILIES]
    missing = sorted(cls.__name__ for cls in set(registry.values()) - set(covered))
    assert not missing, f"EP families with no tiny model in TINY_MOE_FAMILIES: {missing}"
    assert len(set(covered)) == len(covered), "two tiny models stand for one EP family"


def test_the_merged_resume_body_reaches_recipes_liger_composites_and_chosen_targets():
    args = merged_resume_parser().parse_args(
        ["--family", "qwen3_5_moe", "--liger", "--attention-targets", "q_proj,in_proj_z"]
    )
    assert args.liger
    assert args.attention_targets == ["q_proj", "in_proj_z"]
    assert not merged_resume_parser().parse_args(["--family", "gpt_oss"]).liger


_RELOADED_FAMILIES = {**TINY_MOE_FAMILIES, "bailing_moe_linear": TINY_BAILING_RING_FAMILY}


@pytest.mark.parametrize("name", sorted(_RELOADED_FAMILIES))
def test_a_tiny_checkpoint_reloads_under_its_family_spec(name, tmp_path):
    """A vendor config leaves ``model_type`` off its class attribute, so a plain save reloads family-less and every row
    built on it skips the spec-keyed paths a release takes."""
    family = _RELOADED_FAMILIES[name]
    build_tiny_family_checkpoint(family, str(tmp_path))
    config = AutoConfig.from_pretrained(str(tmp_path), trust_remote_code=family.trust_remote_code)
    assert find_moe_family_spec_for_config(config) is find_moe_family_spec(name)


def test_the_merged_resume_rows_cover_every_family_adapter_shape_and_layout():
    rows = _rows(MERGED_RESUME_SUITE, merged_resume_parser().parse_args)
    covered = [(row.family, row.adapters, row.ep_size, row.cp_size, row.fp32_masters) for row in rows]
    expected = {
        (family, adapters, ep, cp, False)
        for family in TINY_MOE_FAMILIES
        for adapters in ADAPTER_MODES
        for ep, cp in LAYOUTS
    }
    assert not sorted(expected - set(covered)), f"merged-resume shapes no row runs: {sorted(expected - set(covered))}"
    assert len(set(covered)) == len(covered), "a merged-resume shape runs in two rows"


def _has_tp_plan(family: str) -> bool:
    """Whether ``family``'s attention is one the selective-TP path shards."""
    model = tiny_family_model(TINY_MOE_FAMILIES[family])
    return any(type(module).__name__ in TP_SHARDABLE_ATTENTION_CLASSES for module in model.modules())


def test_the_precompute_resume_rows_cover_every_family():
    launched = _launches(PRECOMPUTE_RESUME_SUITE, precompute_resume_parser().parse_args)
    misfit = sorted((row.mode, spec.nproc) for row, spec in launched if spec.nproc != world_size(row.mode))
    assert not misfit, f"precompute-resume rows launched at a width their layout does not run on: {misfit}"
    rows = [row for row, _ in launched]
    covered = {(row.trainer, row.family, row.mode) for row in rows if not row.peft}
    tp_families = {family for family in TINY_MOE_FAMILIES if _has_tp_plan(family)}
    expected = {(trainer, family, mode) for family in TINY_MOE_FAMILIES for trainer, mode in PRECOMPUTE_PER_FAMILY} | {
        (trainer, family, mode) for family in tp_families for trainer, mode in PRECOMPUTE_PER_TP_FAMILY
    }
    assert not sorted(expected - covered), f"precompute-resume shapes no row runs: {sorted(expected - covered)}"
    tp_modes = {mode for _, mode in PRECOMPUTE_PER_TP_FAMILY}
    off_plan = sorted(
        {(family, mode) for _, family, mode in covered if mode in tp_modes}
        - {(family, mode) for family in (*tp_families, DENSE) for mode in tp_modes}
    )
    assert not off_plan, f"precompute-resume TP rows on a family with no TP plan: {off_plan}"


def test_the_sync_exactness_rows_run_every_syncable_family_once_per_shape():
    roster = row_families()
    known = (*TINY_DENSE_FAMILIES, *TINY_MOE_FAMILIES)
    rows = _rows(SYNC_EXACTNESS_SUITE, lambda argv: parse_row(known, argv))
    outside = sorted({row.family for row in rows} - set(roster))
    assert not outside, f"sync-exactness rows naming a family outside the syncable roster: {outside}"
    covered = [(row.family, row.mode, row.adapters) for row in rows]
    expected = {(family, "fsdp", "peft") for family in TINY_DENSE_FAMILIES} | {
        (family, mode, adapters)
        for family in syncable_moe_families()
        for mode, adapters in SYNC_EXACTNESS_PER_MOE_FAMILY
    }
    missing = sorted(expected - set(covered))
    assert not missing, f"syncable family x shape no sync-exactness row runs: {missing}"
    assert set(covered) <= expected, f"sync-exactness rows off the roster's shapes: {sorted(set(covered) - expected)}"
    assert len(set(covered)) == len(covered), "a sync-exactness shape runs in two rows"


def test_the_determinism_rows_run_every_family_and_every_layout_once():
    rows = _rows(DETERMINISM_SUITE, ep_determinism.determinism_parser().parse_args)
    covered = [(row.family, row.mode) for row in rows]
    expected = {(family, "ep2") for family in TINY_MOE_FAMILIES} | {
        (DETERMINISM_LAYOUT_FAMILY, mode) for mode in ep_determinism.MODES
    }
    missing = sorted(expected - set(covered))
    assert not missing, f"family x layout no determinism row runs: {missing}"
    assert len(set(covered)) == len(covered), "a determinism shape runs in two rows"


def _flag(argv: list[str], flag: str, default: str | None = None) -> str | None:
    return argv[argv.index(flag) + 1] if flag in argv else default


def test_the_reentrant_rows_run_every_family_full_on_ep1_and_two_ep_groups():
    rows = [shlex.split(args) for args, _ in MANIFEST[REENTRANT_SUITE].launches()]
    covered = {
        (_flag(row, "--family"), _flag(row, "--layout")) for row in rows if _flag(row, "--adapter", "full") == "full"
    }
    expected = {(family, layout) for family in TINY_MOE_FAMILIES for layout in REENTRANT_PER_FAMILY_LAYOUTS}
    assert not sorted(expected - covered), f"family x layout no reentrant row trains: {sorted(expected - covered)}"


def test_the_offline_grpo_cp_rows_run_every_ulysses_moe_family_pure_and_under_ep():
    rows = [shlex.split(args) for args, _ in MANIFEST[OFFLINE_GRPO_CP_SUITE].launches()]
    covered = {
        (_flag(row, "--family"), _flag(row, "--ep-size", "1"))
        for row in rows
        if _flag(row, "--cp-size") == "2" and "--family" in row
    }
    expected = {
        (family, ep_size) for family, tiny in TINY_MOE_FAMILIES.items() if tiny.ulysses_cp for ep_size in ("1", "2")
    }
    assert covered == expected, (
        f"offline GRPO CP rows {sorted(covered)} != every Ulysses family x ep {sorted(expected)}"
    )


def test_the_tp_reference_rows_run_every_tp_planned_moe_family_alone_and_beside_ep():
    rows = [shlex.split(args) for args, _ in MANIFEST[TP_REFERENCE_SUITE].launches()]
    moe = {
        (_flag(row, "--family"), _flag(row, "--mode")) for row in rows if _flag(row, "--family") in TINY_MOE_FAMILIES
    }
    expected = {(family, mode) for family in TINY_MOE_FAMILIES if _has_tp_plan(family) for mode in ("tp2", "ep2tp2")}
    assert moe == expected, f"TP reference MoE rows {sorted(moe)} != every TP-planned family x mode {sorted(expected)}"


@pytest.mark.parametrize("suite", CP_SUITES)
def test_the_cp_rows_run_every_cp_roster_family_once(suite):
    families = [shlex.split(args)[1] for args, _ in MANIFEST[suite].launches()]
    assert sorted(families) == sorted(CP_FAMILIES), (
        f"CP rows {sorted(families)} != the CP roster {sorted(CP_FAMILIES)}"
    )


def test_every_engine_runs_the_push_exactness_rows_of_every_family_it_takes():
    """The live proof that an unchanged push leaves a server alone. A family the rows skip could corrupt it with every
    tensor forwarded, since only the engine's reload knows what it counts."""
    rows = [
        (parse_push_exact_row(shlex.split(args)), spec.markers) for args, spec in MANIFEST[PUSH_EXACT_SUITE].launches()
    ]
    assert {row.family for row, _ in rows} == set(PUSH_EXACT_CHECKPOINTS)
    # A real step's bar is gross on these two, so the name-by-name rounds are what catch a dropped name.
    assert {(row.family, row.backend) for row, _ in rows if row.reach} == {
        ("laguna", "vllm"),
        ("gemma4_text", "vllm"),
        ("gemma4_text", "sglang"),
    }
    registry = ep_layer_class_by_model_type()
    for backend in rollout_backends():
        refused = resolve_weight_sync_client(backend).UNSERVABLE_MODEL_TYPES
        served = {row.family for row, _ in rows if row.backend == backend}
        for family in syncable_moe_families():
            assert (family in served) == (registry[family].FAMILY.model_types[0] not in refused), (family, backend)
    for row, markers in rows:
        assert f"{row.backend}_server" in markers, f"{row} would run with no {row.backend} server gate"


def test_the_composite_gemma4_per_layer_table_follows_a_vocab_override():
    """A tokenizer-sized vocab past the table's own width raises at the first embedded id."""
    model = tiny_family_model(TINY_MOE_VLM_FAMILIES["gemma4"], overrides={"vocab_size": 2048})

    assert model.config.text_config.vocab_size_per_layer_input == 2048
    assert model.get_input_embeddings().num_embeddings == 2048


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
