"""The checkpoint artifact harness finds nothing between two runs of today's savers, every case but the known ones
saves what its mode owes at the dtypes it owes, and each seeded change to a saver shows up as a delta.

Run: python tests/cpu/checkpoint/test_checkpoint_artifacts.py
"""

import json

import pytest

import src.checkpoint.atomic as checkpoint_atomic
import src.checkpoint.parallel.ep as ep_save
import src.checkpoint.parallel.optimizer as checkpoint_optimizer
import src.checkpoint.parallel.save as checkpoint_save
import src.checkpoint.parallel.writing as checkpoint_write
import src.checkpoint.writing as checkpoint_writing
from tests.common.equivalence.checkpoint_artifacts import (
    ARTIFACT_CASES,
    EXPERT_LORA_MODES,
    FAMILIES,
    FULL_WEIGHT_MODES,
    KNOWN_RAISING,
    PRODUCER_ENV,
    RAISED_FILE,
    RUN_DTYPES,
    ArtifactCase,
    describe,
    produce,
    raised_cases,
    unexpected_raises,
)
from tests.common.equivalence.revisions import OUTPUT_ROOT_MARK, compare_outputs, run_in_tree
from tests.common.repo import REPO_ROOT

SAMPLE_FAMILIES = ("qwen3", "gpt_oss", "gemma4")
MODULE = produce.__module__
DENSE_EXPORT = ArtifactCase("qwen3", "fsdp2", "fp32", training=False)
DENSE_TRAINING = ArtifactCase("qwen3", "fsdp2", "fp32", training=True)
DENSE_BF16_TRAINING = ArtifactCase("qwen3", "fsdp2", "bf16", training=True)
# A family whose hub names differ from its module tree: the export reverts them per streamed chunk.
VENDOR_NAMESPACE_EXPORT = ArtifactCase("deepseek_v4", "ep", "fp32", training=False)
# A bf16 run's training checkpoint keeps the fp32 router master beside its bf16 weights.
ROUTER = "model.layers.0.mlp.gate.weight"
EMBEDDING = "model.embed_tokens.weight"


def _produced(output, cases) -> dict:
    produce(output, cases)
    return describe(output)


def _weights(description: dict, case: ArtifactCase) -> dict:
    return description[f"{case.path}/model.safetensors"]["content"]["tensors"]


def test_the_roster_runs_every_family_in_every_mode_at_every_dtype():
    assert {case.family for case in ARTIFACT_CASES} == set(FAMILIES)
    assert {case.mode for case in ARTIFACT_CASES} == {*FULL_WEIGHT_MODES, *EXPERT_LORA_MODES, "peft"}
    assert {case.dtype for case in ARTIFACT_CASES} == set(RUN_DTYPES)
    assert len({case.path for case in ARTIFACT_CASES}) == len(ARTIFACT_CASES)
    assert KNOWN_RAISING.issubset(case.path for case in ARTIFACT_CASES)


def test_every_case_but_the_known_ones_saves_what_its_mode_owes(tmp_path):
    """Weights in every case, the optimizer shard with its meta in exactly the full-weight training checkpoints,
    and a bf16 training checkpoint's fp32 router beside its bf16 weights. One test over the whole roster, which
    takes a minute and a half to save."""
    roster = _produced(tmp_path, ARTIFACT_CASES)
    assert set(raised_cases(roster)) == KNOWN_RAISING
    assert unexpected_raises(raised_cases(roster)) == []
    shards = {"optimizer_shard_00000.pt", "optimizer_meta.pt"}
    wrong = []
    for case in ARTIFACT_CASES:
        if case.path in KNOWN_RAISING:
            continue
        files = {path.removeprefix(f"{case.path}/") for path in roster if path.startswith(f"{case.path}/")}
        owed_shards = shards if case.training and case.mode in FULL_WEIGHT_MODES else set()
        if not any(name.endswith(".safetensors") for name in files) or shards & files != owed_shards:
            wrong.append(f"{case.path}: {sorted(files)}")
    assert wrong == []
    bf16_training = _weights(roster, ArtifactCase("qwen3_moe", "ep", "bf16", training=True))
    assert (bf16_training[ROUTER][0], bf16_training[EMBEDDING][0]) == ("F32", "BF16")
    fp32_training = _weights(roster, ArtifactCase("qwen3_moe", "ep", "fp32", training=True))
    assert {dtype for dtype, _, _ in fp32_training.values()} == {"F32"}


def test_two_producer_processes_match(tmp_path):
    """Base against base: each tree's producer runs in its own process, as ``make verify-artifacts`` runs
    them, so nothing a process draws afresh (a string hash seed, an object id) may reach an artifact."""
    outputs = [tmp_path / "first", tmp_path / "second"]
    for output in outputs:
        run_in_tree(REPO_ROOT, MODULE, "--produce", str(output), "--families", *SAMPLE_FAMILIES, env=PRODUCER_ENV)
    first, second = map(describe, outputs)

    assert any(path.endswith("adapter_config.json") for path in first), "the sample lost its PEFT adapter"
    assert len(first) > 100
    assert compare_outputs(first, second) == []


def test_a_norm_weight_cast_on_export_is_a_dtype_delta(monkeypatch, tmp_path):
    baseline = _produced(tmp_path / "baseline", [DENSE_EXPORT])
    monkeypatch.setattr(checkpoint_writing, "norm_param_keys", lambda model: frozenset())

    [delta] = compare_outputs(baseline, _produced(tmp_path / "mutated", [DENSE_EXPORT]))

    assert delta["path"] == f"{DENSE_EXPORT.path}/model.safetensors"
    assert any("norm.weight" in entry and '"F32"' in entry and '"BF16"' in entry for entry in delta["entries"])


def test_a_training_checkpoint_upcasting_live_bf16_is_a_dtype_delta_only_at_bf16(monkeypatch, tmp_path):
    """The fp32 run cannot show it: its live tensors are fp32 already."""
    cases = [DENSE_TRAINING, DENSE_BF16_TRAINING]
    baseline = _produced(tmp_path / "baseline", cases)
    monkeypatch.setattr(checkpoint_writing, "_as_live", lambda name, tensor: tensor.float())

    [delta] = compare_outputs(baseline, _produced(tmp_path / "mutated", cases))

    assert delta["path"] == f"{DENSE_BF16_TRAINING.path}/model.safetensors"
    assert all('"BF16"' in entry and '"F32"' in entry for entry in delta["entries"])


def test_a_changed_value_is_a_byte_delta(monkeypatch, tmp_path):
    baseline = _produced(tmp_path / "baseline", [DENSE_EXPORT])
    cast = checkpoint_writing.cast_to_save_dtype
    monkeypatch.setattr(checkpoint_writing, "cast_to_save_dtype", lambda tensor: cast(tensor * 2))

    [delta] = compare_outputs(baseline, _produced(tmp_path / "mutated", [DENSE_EXPORT]))

    assert delta["path"] == f"{DENSE_EXPORT.path}/model.safetensors"
    assert all(entry.startswith("content/tensors/") for entry in delta["entries"])


def test_a_config_field_is_a_json_delta(monkeypatch, tmp_path):
    baseline = _produced(tmp_path / "baseline", [DENSE_EXPORT])
    save_model_config = checkpoint_write.save_model_config

    def with_extra_field(model, output_dir, *args, **kwargs):
        save_model_config(model, output_dir, *args, **kwargs)
        path = f"{output_dir}/config.json"
        with open(path) as file:
            config = json.load(file)
        with open(path, "w") as file:
            json.dump({**config, "planted": True}, file)

    monkeypatch.setattr(checkpoint_write, "save_model_config", with_extra_field)

    [delta] = compare_outputs(baseline, _produced(tmp_path / "mutated", [DENSE_EXPORT]))

    assert delta == {
        "path": f"{DENSE_EXPORT.path}/config.json",
        "delta": "differs",
        "entries": ["content/planted: only in head"],
    }


def test_a_staged_file_readable_by_its_owner_only_is_a_mode_delta(monkeypatch, tmp_path):
    baseline = _produced(tmp_path / "baseline", [DENSE_TRAINING])
    meta = f"{DENSE_TRAINING.path}/optimizer_meta.pt"
    assert baseline[meta]["mode"] != "600", "premise: the umask leaves a staged file readable beyond its owner"
    create_staged_file = checkpoint_atomic.create_staged_file

    def owner_only(directory, filename):
        staged = create_staged_file(directory, filename)
        staged.chmod(0o600)
        return staged

    monkeypatch.setattr(checkpoint_atomic, "create_staged_file", owner_only)

    deltas = compare_outputs(baseline, _produced(tmp_path / "mutated", [DENSE_TRAINING]))

    staged = sorted((meta, f"{DENSE_TRAINING.path}/README.md"))
    assert sorted(deltas, key=lambda delta: delta["path"]) == [
        {"path": path, "delta": "differs", "entries": [f'mode: "{baseline[path]["mode"]}" != "600"']}
        for path in staged
    ]


def test_a_dropped_optimizer_moment_is_a_delta(monkeypatch, tmp_path):
    baseline = _produced(tmp_path / "baseline", [DENSE_TRAINING])
    state_dict = checkpoint_optimizer.get_optimizer_state_dict

    def without_one_moment(*args, **kwargs):
        osd = state_dict(*args, **kwargs)
        del next(iter(osd["state"].values()))["exp_avg_sq"]
        return osd

    monkeypatch.setattr(checkpoint_optimizer, "get_optimizer_state_dict", without_one_moment)

    [delta] = compare_outputs(baseline, _produced(tmp_path / "mutated", [DENSE_TRAINING]))

    assert delta["path"] == f"{DENSE_TRAINING.path}/optimizer_shard_00000.pt"
    assert len(delta["entries"]) == 1 and delta["entries"][0].endswith("/exp_avg_sq: only in base")


def test_an_export_left_in_the_module_namespace_is_a_key_delta(monkeypatch, tmp_path):
    baseline = _produced(tmp_path / "baseline", [VENDOR_NAMESPACE_EXPORT])
    monkeypatch.setattr(ep_save, "export_conversions", lambda model: [])

    [delta] = compare_outputs(baseline, _produced(tmp_path / "mutated", [VENDOR_NAMESPACE_EXPORT]))

    assert delta["path"] == f"{VENDOR_NAMESPACE_EXPORT.path}/model.safetensors"
    assert {"content/tensors/head.weight: only in base", "content/tensors/lm_head.weight: only in head"} <= set(
        delta["entries"]
    )


def test_a_saver_that_raises_is_a_delta_and_an_unexpected_raise(monkeypatch, tmp_path):
    baseline = _produced(tmp_path / "baseline", [DENSE_EXPORT])

    def refuse(ctx, output_dir):
        raise RuntimeError(f"refused to write {output_dir}")

    monkeypatch.setattr(checkpoint_save, "save_fsdp2_checkpoint", refuse)
    mutated = _produced(tmp_path / "mutated", [DENSE_EXPORT])

    assert raised_cases(mutated) == {
        DENSE_EXPORT.path: {
            "type": "RuntimeError",
            "message": f"refused to write {OUTPUT_ROOT_MARK}/{DENSE_EXPORT.path}",
        }
    }
    assert unexpected_raises(raised_cases(mutated)) == [DENSE_EXPORT.path]
    deltas = compare_outputs(baseline, mutated)
    assert {delta["delta"] for delta in deltas} == {"only in base", "only in head"}
    assert {"path": f"{DENSE_EXPORT.path}/{RAISED_FILE}", "delta": "only in head"} in deltas


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
