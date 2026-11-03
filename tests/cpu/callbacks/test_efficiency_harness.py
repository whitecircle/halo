"""The efficiency harness finds nothing between two runs of today's ``EfficiencyCallback``, each scenario reaches its
path, each seeded change shows up as a delta, and the verdict fails when its negative control raises."""

from dataclasses import replace

import pytest

from halo.callbacks import efficiency, model_flops
from halo.models.attention.layout import attention_layout_from_config
from tests.common.equivalence import efficiency_callback
from tests.common.equivalence.efficiency_callback import (
    CASES,
    DENSE_FAMILIES,
    FAMILIES,
    OUTLIER_TOKENS,
    REPRESENTATIVE_FAMILIES,
    SCENARIO_FAMILY,
    SCENARIOS,
    SHAPES,
    WARMUP_STEPS,
    EfficiencyCase,
    Scenario,
    assess,
    produce,
    read_reports,
    run_case,
)
from tests.common.equivalence.revisions import RAISED, compare_outputs, run_in_tree
from tests.common.repo import REPO_ROOT

MODULE = produce.__module__
DEFAULT = EfficiencyCase(SCENARIO_FAMILY, "single", "default")
CP = EfficiencyCase(SCENARIO_FAMILY, "cp2", "default")
OUTLIER = EfficiencyCase(SCENARIO_FAMILY, "single", "token_outlier")
SAMPLE = (DEFAULT, CP, OUTLIER, EfficiencyCase("qwen3", "single", "default"))
STAGED = tuple(case for case in CASES if SHAPES[case.shape].fields.get("pp_size", 1) > 1)
LAYERLESS_STAGE = EfficiencyCase(SCENARIO_FAMILY, "pp2_first", "layerless")
# Each pipeline shape and the unsplit shape whose model its two stages share between them.
UNSPLIT_SHAPE = {"pp2": "single", "ep2_pp2": "ep2", "etp2_pp2": "etp2"}
# The step figures are rounded to two decimals, so a sum of two stages may sit a few hundredths off.
ROUNDING = 0.02


def _reports(output, cases, *, perturb: bool = False) -> dict[str, dict]:
    produce(output, cases, perturb=perturb)
    return read_reports(output)


def _scenario(reports: dict, scenario: str) -> dict:
    return reports[f"{SCENARIO_FAMILY}/single/{scenario}"]


def _last_step(reports: dict, family: str, shape: str) -> dict:
    return reports[f"{family}/{shape}/default"]["steps"][-1]


def _messages(report: dict, level: str) -> str:
    return "\n".join(message for record_level, message in report["records"] if record_level == level)


@pytest.fixture(scope="module")
def roster(tmp_path_factory):
    return _reports(tmp_path_factory.mktemp("roster"), CASES)


def test_the_cases_run_every_family_shape_and_scenario():
    assert {case.family for case in CASES} == set(FAMILIES)
    assert {case.shape for case in CASES} == set(SHAPES)
    assert {case.scenario for case in CASES} == set(SCENARIOS)
    assert len({case.path for case in CASES}) == len(CASES)


def test_every_case_reports_its_steps_and_the_warmup_reports_none(roster):
    assert {path for path, report in roster.items() if RAISED in report} == set()
    for path, report in roster.items():
        steps = report["steps"]
        assert all(step == {} for step in steps[:WARMUP_STEPS]), path
        assert all(step["step_tokens_per_second"] > 0 for step in steps[WARMUP_STEPS:]), path


def test_each_scenario_reaches_the_path_it_names(roster):
    default = _scenario(roster, "default")["steps"][-1]
    assert default["step_mfu_percent"] > default["step_smfu_percent"] > 0
    assert "MFU reporting disabled" in _messages(_scenario(roster, "estimate_raises"), "WARNING")
    assert "Could not detect the GPU model" in _messages(_scenario(roster, "no_device"), "WARNING")
    assert "has no fp8 peak for A100" in _messages(_scenario(roster, "no_peak"), "WARNING")
    assert "measured cluster delta (0)" in _messages(_scenario(roster, "token_fallback"), "WARNING")
    assert f"measured cluster delta ({OUTLIER_TOKENS})" in _messages(_scenario(roster, "token_outlier"), "WARNING")
    assert "no measured attention work" in _messages(_scenario(roster, "config_attention"), "INFO")
    assert "Full model params" in _messages(_scenario(roster, "full_params"), "INFO")
    assert "step_mfu_percent" not in _scenario(roster, "no_diagnostics")["steps"][-1]
    for scenario in ("single_expert", "no_top_k"):
        assert "step_smfu_percent" not in _scenario(roster, scenario)["steps"][-1], scenario
    assert _scenario(roster, "no_device")["steps"][-1]["allocated_gb"] == 0.0 < default["allocated_gb"]
    assert _scenario(roster, "lora")["steps"][-1]["step_mfu_percent"] < default["step_mfu_percent"]
    precisions = {
        scenario: _scenario(roster, scenario)["snapshot"]["diagnostics"]["precision"]
        for scenario in ("fp8", "fp8_native", "fp32", "tf32", "late_model")
    }
    assert precisions == {"fp8": "bf16", "fp8_native": "fp8", "fp32": "fp32", "tf32": "tf32", "late_model": "fp32"}
    # Before the model arrives the arguments alone decide, and fp32 arguments fall back to bf16.
    late = _messages(_scenario(roster, "late_model"), "INFO")
    assert "Precision: bf16" in late and "Precision: fp32" in late
    assert "Could not get model reference" in _messages(_scenario(roster, "late_model"), "WARNING")
    single, cp = roster[DEFAULT.path]["steps"][-1], roster[CP.path]["steps"][-1]
    assert cp["step_tokens_per_second"] == pytest.approx(single["step_tokens_per_second"] / 2, rel=1e-9)


def test_the_sharded_shapes_report_rank_local_figures(roster):
    """A rank holding 1/ep of the experts serves ep times their tokens, so its active FLOPs equal a rank holding them
    all while its dense FLOPs fall with the expert count."""
    for family in ("qwen3_moe", "gpt_oss"):
        steps = {shape: roster[f"{family}/{shape}/default"]["steps"][-1] for shape in SHAPES}
        active = {shape: step["step_smfu_tflops_per_sec"] for shape, step in steps.items()}
        dense = {shape: step["step_tflops_per_sec"] for shape, step in steps.items()}
        assert active["ep2"] == active["ep4"] == active["single"], family
        assert dense["ep4"] < dense["ep2"] < dense["single"], family
        assert all(active[shape] < active["single"] for shape in ("etp2", "tp2", "fsdp2", "ep2_fsdp2")), family
    dense = {
        shape: roster[f"qwen3/{shape}/default"]["steps"][-1]["step_tflops_per_sec"] for shape in ("single", "tp2")
    }
    assert dense["tp2"] == pytest.approx(dense["single"] / 2, rel=0.05)


def test_a_pipeline_stage_reports_the_layers_it_holds(roster):
    """The two stages' MFU and S-MFU add up to the unsplit model's on the same expert layout."""
    for family in REPRESENTATIVE_FAMILIES:
        pipelines = ("pp2",) if family in DENSE_FAMILIES else tuple(UNSPLIT_SHAPE)
        for pipeline in pipelines:
            whole = _last_step(roster, family, UNSPLIT_SHAPE[pipeline])
            stages = [_last_step(roster, family, f"{pipeline}_{stage}") for stage in ("first", "last")]
            for metric in ("step_mfu_percent", "step_smfu_percent"):
                split = sum(stage.get(metric, 0.0) for stage in stages)
                assert split == pytest.approx(whole.get(metric, 0.0), abs=ROUNDING), (family, pipeline, metric)


# Data-parallel replicas of each shape, counted by hand: the ranks of one TP, expert-TP or CP group and of one
# pipeline chain share a batch, every EP rank draws its own.
REPLICAS = {
    "single": 1,
    "ep2": 2,
    "ep4": 4,
    "etp2": 1,
    "ep2_etp2": 2,
    "tp2": 1,
    "cp2": 1,
    "ep2_cp2": 1,
    "ep2_tp2": 1,
    "fsdp2": 2,
    "ep2_fsdp2": 2,
    "pp2_first": 2,
    "pp2_last": 2,
    "ep2_pp2_first": 2,
    "ep2_pp2_last": 2,
    "etp2_pp2_first": 1,
    "etp2_pp2_last": 1,
}


def test_every_shape_counts_each_shared_batch_once(roster):
    """Each shape's cluster figure is its replica count times one rank's, and a GPU's share is that over the world."""
    assert set(REPLICAS) == set(SHAPES)
    one_replica = roster[DEFAULT.path]["snapshot"]["cluster_tokens_per_second"]
    for shape, replicas in REPLICAS.items():
        snapshot = roster[f"{SCENARIO_FAMILY}/{shape}/default"]["snapshot"]
        cluster = snapshot["cluster_tokens_per_second"]
        assert cluster == pytest.approx(replicas * one_replica, rel=1e-9), shape
        per_gpu = cluster / SHAPES[shape].world_size
        assert snapshot["tokens_per_second"] == pytest.approx(per_gpu, abs=ROUNDING), shape


def test_two_producer_processes_match(tmp_path):
    """Each tree's producer runs in its own process, as ``make verify-efficiency`` runs them."""
    outputs = [tmp_path / "first", tmp_path / "second"]
    for output in outputs:
        run_in_tree(REPO_ROOT, MODULE, "--produce", str(output), "--cases", *(case.path for case in SAMPLE))
    first, second = map(read_reports, outputs)

    assert set(first) == {case.path for case in SAMPLE}
    assert compare_outputs(first, second) == []


def test_unclassified_experts_are_a_delta_in_every_moe_case(monkeypatch, tmp_path):
    baseline = _reports(tmp_path / "baseline", SAMPLE)
    monkeypatch.setattr(model_flops, "_is_expert_param", lambda name: False)

    deltas = compare_outputs(baseline, _reports(tmp_path / "mutated", SAMPLE))

    assert {delta["path"] for delta in deltas} == {DEFAULT.path, CP.path, OUTLIER.path}


def test_a_changed_flops_rule_is_a_delta_in_every_case(monkeypatch, tmp_path):
    baseline = _reports(tmp_path / "baseline", SAMPLE)
    monkeypatch.setattr(
        model_flops, "_training_flops_per_token", lambda trainable, frozen: 6.0 * trainable + 2.0 * frozen + 1000.0
    )

    deltas = compare_outputs(baseline, _reports(tmp_path / "mutated", SAMPLE))

    assert {delta["path"] for delta in deltas} == {case.path for case in SAMPLE}


def test_a_pipeline_blind_attention_share_is_a_delta_in_the_staged_config_term_only(monkeypatch, tmp_path):
    """A stage whose layers are not reachable owns 1 / pp_size of the config's full depth, so dropping pp_size moves
    only the layerless stage."""
    baseline = _reports(tmp_path / "baseline", STAGED)
    term = model_flops._attention_term
    monkeypatch.setattr(
        model_flops, "_attention_term", lambda model, seq_len, pp_size, tp_size: term(model, seq_len, 1, tp_size)
    )

    deltas = compare_outputs(baseline, _reports(tmp_path / "mutated", STAGED))

    assert [delta["path"] for delta in deltas] == [LAYERLESS_STAGE.path]


def test_a_full_depth_layout_on_a_stage_is_a_delta_in_every_staged_case_only(monkeypatch, tmp_path):
    cases = SAMPLE + STAGED
    baseline = _reports(tmp_path / "baseline", cases)
    monkeypatch.setattr(
        model_flops,
        "attention_layout",
        lambda model: replace(attention_layout_from_config(model.config), source="layers"),
    )

    deltas = compare_outputs(baseline, _reports(tmp_path / "mutated", cases))

    assert {delta["path"] for delta in deltas} == {case.path for case in STAGED}


def test_a_looser_token_bound_is_a_delta_in_the_outlier_case_only(monkeypatch, tmp_path):
    baseline = _reports(tmp_path / "baseline", SAMPLE)
    monkeypatch.setattr(efficiency, "_TOKEN_DELTA_UPPER_BOUND", 2000.0)

    deltas = compare_outputs(baseline, _reports(tmp_path / "mutated", SAMPLE))

    assert [delta["path"] for delta in deltas] == [OUTLIER.path]


def test_a_changed_rounding_is_a_delta_in_the_logs(monkeypatch, tmp_path):
    baseline = _reports(tmp_path / "baseline", [DEFAULT])
    monkeypatch.setattr(efficiency, "_DISPLAY_DECIMALS", 3)

    [delta] = compare_outputs(baseline, _reports(tmp_path / "mutated", [DEFAULT]))

    assert any(entry.startswith("steps") for entry in delta["entries"])


def test_a_silenced_setup_summary_is_a_records_delta_only(monkeypatch, tmp_path):
    baseline = _reports(tmp_path / "baseline", [DEFAULT])
    monkeypatch.setattr(efficiency, "is_global_main_process", lambda: False)

    [delta] = compare_outputs(baseline, _reports(tmp_path / "mutated", [DEFAULT]))

    assert delta["path"] == DEFAULT.path
    assert all(entry.startswith("records") for entry in delta["entries"])


def test_the_negative_control_moves_the_steps_only(tmp_path):
    reports = {perturb: run_case(DEFAULT, str(tmp_path), perturb=perturb) for perturb in (False, True)}

    assert reports[False]["records"] == reports[True]["records"]
    assert reports[False]["steps"][:-1] == reports[True]["steps"][:-1]
    assert reports[False]["steps"][-1] != reports[True]["steps"][-1]


def test_an_accepted_case_keeps_its_delta_out_of_the_verdict(tmp_path):
    base = _reports(tmp_path / "base", [DEFAULT, CP])
    head = {**base, CP.path: {**base[CP.path], "snapshot": {}}}
    control = _reports(tmp_path / "control", [DEFAULT], perturb=True)

    assert assess({"base": base, "head": head, "control": control})[0] == "fail"
    verdict, findings = assess({"base": base, "head": head, "control": control}, accept=[CP.path])
    assert (verdict, [delta["path"] for delta in findings["accepted"]]) == ("pass", [CP.path])


def test_a_case_only_one_tree_runs_is_listed_not_compared():
    reports = {
        "base": {DEFAULT.path: {"steps": [1]}, CP.path: {"steps": [1]}},
        "head": {DEFAULT.path: {"steps": [2]}, OUTLIER.path: {"steps": [1]}},
        "control": {DEFAULT.path: {"steps": [3]}},
    }

    verdict, findings = assess(reports)

    assert verdict == "fail"
    assert [delta["path"] for delta in findings["deltas"]] == [DEFAULT.path]
    assert (findings["cases"], findings["new"], findings["dropped"]) == (1, [OUTLIER.path], [CP.path])
    assert assess(reports, accept=[DEFAULT.path])[0] == "pass"


def test_a_negative_control_that_raises_or_matches_fails_the_verdict(monkeypatch, tmp_path):
    base = _reports(tmp_path / "base", [DEFAULT])
    unmoved = _reports(tmp_path / "unmoved", [DEFAULT])
    monkeypatch.setattr(efficiency_callback, "run_case", lambda case, scratch, perturb=False: 1 / 0)
    raised = _reports(tmp_path / "raised", [DEFAULT], perturb=True)

    for control in (unmoved, raised):
        verdict, findings = assess({"base": base, "head": base, "control": control})
        assert (verdict, findings["negative_control"]["unseen"]) == ("fail", [DEFAULT.path])


def test_a_misspelled_scenario_field_is_refused():
    with pytest.raises(ValueError, match="layerles"):
        Scenario(model="layerles")
    with pytest.raises(ValueError, match="outliers"):
        Scenario(tokens="outliers")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
