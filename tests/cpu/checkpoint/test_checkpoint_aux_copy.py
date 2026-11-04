"""copy_checkpoint_aux_files: weight artifacts are skipped, resume sidecars survive, module directories ride whole.
``.pth`` sits on both sides, so the sidecar exemption is by name."""

import pytest
import torch
from huggingface_hub.constants import REPOCARD_NAME
from safetensors.torch import load_file, save_file
from transformers.trainer import SCHEDULER_NAME, TRAINER_STATE_NAME

from halo.checkpoint.atomic import atomic_torch_save, create_staged_file
from halo.checkpoint.format import (
    ADAPTER_SAFETENSORS_FILE,
    OPTIMIZER_META_FILE,
    REFERENCE_LOGPS_FILE,
    RESUME_ADAPTER_DIR,
    RESUME_ADAPTER_MARKER_FILE,
    ROUTER_BALANCING_BIASES_FILE,
    RUN_PLAN_FILE,
    WEIGHT_DUMP_IGNORE_PATTERNS,
    prefetch_pending_filename,
)
from halo.checkpoint.tools.offline_export import copy_checkpoint_aux_files
from tests.common.checkpoint_io import mark_resume_adapter

SKIPPED = (
    "model.safetensors",
    "model-00001-of-00002.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "training_args.bin",
    "optimizer.pt",
    "optimizer_shard_0.pt",
    # Foreign-framework exports a hub repo ships beside the safetensors.
    "consolidated.00.pth",
    "model.gguf",
    "tf_model.h5",
    "flax_model.msgpack",
    "rust_model.ot",
    "64.tflite",
    "model.onnx",
    RUN_PLAN_FILE,
)
KEPT = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "modeling_remote.py",
)
SIDECARS = (
    TRAINER_STATE_NAME,
    "rng_state_0.pth",
    SCHEDULER_NAME,
    ROUTER_BALANCING_BIASES_FILE,
    REFERENCE_LOGPS_FILE,
    prefetch_pending_filename(0, 2),
)
# modules.json names these directories, and 2_Dense carries weights no merge rewrites.
MODULE_DIRS = ("1_Pooling", "2_Dense", "original_adapter_config")
# Distinct payloads for the two same-named trainer_state.json files, so a copy that flattened the nested directory in
# is visible.
ARTIFACT_AUX_BYTES = b"x"
NESTED_RUN_STATE = b"the nested run's state, not the artifact's"
# Spelled literally: each is a name a crash left on disk, which outlives a later rename of its writer.
_STAGE_TOKEN = "0123456789abcdef0123456789abcdef"
LEFTOVER_STAGES = (
    ".trainer_state.json.uncommitted",
    f".reference_logps.pt.{_STAGE_TOKEN}.tmp",
)


@pytest.fixture()
def checkpoint_dir(tmp_path):
    src = tmp_path / "checkpoint"
    src.mkdir()
    for name in SKIPPED + KEPT + SIDECARS:
        (src / name).write_bytes(ARTIFACT_AUX_BYTES)
    (src / "modules.json").write_bytes(b"[]")
    (src / "1_Pooling").mkdir()
    (src / "1_Pooling" / "config.json").write_bytes(b"{}")
    (src / "2_Dense").mkdir()
    (src / "2_Dense" / "config.json").write_bytes(b"{}")
    save_file({"linear.weight": torch.ones(2, 2)}, str(src / "2_Dense" / "model.safetensors"))
    (src / "original_adapter_config").mkdir()
    (src / "original_adapter_config" / "adapter_config.json").write_bytes(b"{}")
    # Its aux files shadow the artifact's own by name.
    (src / "checkpoint-500").mkdir()
    (src / "checkpoint-500" / "optimizer.pt").write_bytes(b"x")
    (src / "checkpoint-500" / "trainer_state.json").write_bytes(NESTED_RUN_STATE)
    (src / "original").mkdir()
    save_file({"tok_embeddings.weight": torch.ones(2, 2)}, str(src / "original" / "consolidated.safetensors"))
    (src / "metal").mkdir()
    (src / "metal" / "model.bin").write_bytes(b"x")
    return src


def test_weight_files_skipped_and_aux_kept(checkpoint_dir, tmp_path):
    out = tmp_path / "merged"
    out.mkdir()
    copy_checkpoint_aux_files(str(checkpoint_dir), str(out))
    copied = {p.name for p in out.iterdir()}
    # README.md is the export's Hub card, written tagged even though this source ships none.
    assert copied == set(KEPT) | set(SIDECARS) | set(MODULE_DIRS) | {"modules.json", "README.md"}, (
        f"unexpected copy set: {sorted(copied)}"
    )


def test_resume_sidecars_survive_merge_copy(checkpoint_dir, tmp_path):
    out = tmp_path / "merged"
    out.mkdir()
    copy_checkpoint_aux_files(str(checkpoint_dir), str(out))
    assert (out / SCHEDULER_NAME).exists(), "LR schedule must survive into the merged resume source"
    assert (out / ROUTER_BALANCING_BIASES_FILE).exists(), "router biases must survive into the merged resume source"
    assert (out / REFERENCE_LOGPS_FILE).exists(), "a precompute run's reference log-probs must survive the merge"
    # Same suffix as the foreign exports, opposite verdict: a bit-reproducible resume replays from this.
    assert (out / "rng_state_0.pth").exists(), "per-rank RNG state must survive into the resume source"
    assert not (out / "optimizer.pt").exists()
    assert not (out / "pytorch_model.bin").exists()


def test_resume_sidecars_can_be_excluded(checkpoint_dir, tmp_path):
    """An N-way merged artifact describes no single run."""
    out = tmp_path / "merged"
    out.mkdir()
    copy_checkpoint_aux_files(str(checkpoint_dir), str(out), include_resume_sidecars=False)
    for name in SIDECARS:
        assert not (out / name).exists(), f"{name} describes one run's state and must not ship in a merge"
    for name in KEPT:
        assert (out / name).exists(), f"{name} must still be carried"


@pytest.mark.parametrize("include_resume_sidecars", [True, False], ids=["resume-source", "n-way-merge"])
def test_a_merged_checkpoint_resume_adapter_travels_with_the_sidecars(tmp_path, include_resume_sidecars):
    """Carried into a merge of several runs, the marker would resume one run's adapters over the base and ignore the
    merged weights."""
    src = tmp_path / "checkpoint"
    (src / RESUME_ADAPTER_DIR).mkdir(parents=True)
    save_file(
        {"a.experts.down_proj.lora_A": torch.ones(2, 2)}, str(src / RESUME_ADAPTER_DIR / ADAPTER_SAFETENSORS_FILE)
    )
    mark_resume_adapter(str(src))
    (src / "config.json").write_text("{}")
    out = tmp_path / "out"
    out.mkdir()

    copy_checkpoint_aux_files(str(src), str(out), include_resume_sidecars=include_resume_sidecars)

    assert (out / "config.json").exists()
    assert (out / RESUME_ADAPTER_MARKER_FILE).exists() is include_resume_sidecars
    assert (out / RESUME_ADAPTER_DIR / ADAPTER_SAFETENSORS_FILE).exists() is include_resume_sidecars


def test_module_directories_copy_whole_with_their_weights(checkpoint_dir, tmp_path):
    """A copy that skips directories ships a modules.json naming modules that do not exist."""
    out = tmp_path / "merged"
    out.mkdir()
    copy_checkpoint_aux_files(str(checkpoint_dir), str(out))
    assert (out / "1_Pooling" / "config.json").exists()
    assert (out / "original_adapter_config" / "adapter_config.json").exists()
    dense_weights = out / "2_Dense" / "model.safetensors"
    assert dense_weights.exists(), "a module directory's own weights ARE the module — they must ride"
    assert torch.equal(load_file(str(dense_weights))["linear.weight"], torch.ones(2, 2))


def test_nested_checkpoint_directories_stay_behind(checkpoint_dir, tmp_path):
    """A copy that recursed in could shadow the artifact's ``trainer_state.json``."""
    out = tmp_path / "merged"
    out.mkdir()
    copy_checkpoint_aux_files(str(checkpoint_dir), str(out))

    assert not (out / "checkpoint-500").exists(), "a nested checkpoint is resume state, not artifact aux data"
    assert (out / "2_Dense" / "model.safetensors").exists(), "the exclusion blanketed every directory"
    assert (out / "trainer_state.json").read_bytes() == ARTIFACT_AUX_BYTES, (
        "the nested run's trainer_state.json was flattened over the artifact's own"
    )


def test_a_vendor_weight_dump_directory_stays_behind(checkpoint_dir, tmp_path):
    """Hundreds of GB on a Mistral/Llama source; the hub-download ignore list drops it off the same tuple."""
    out = tmp_path / "merged"
    out.mkdir()
    copy_checkpoint_aux_files(str(checkpoint_dir), str(out))
    for dump in ("original", "metal"):
        assert not (out / dump).exists(), f"the {dump}/ vendor weight dump is weights, not aux data"
        assert f"{dump}/*" in WEIGHT_DUMP_IGNORE_PATTERNS, "the hub download must drop what the local copy drops"
    # Not a prefix match: original_adapter_config/ is aux data.
    assert (out / "original_adapter_config").exists()


def test_a_leftover_staged_card_is_not_carried(checkpoint_dir, tmp_path):
    """A crash between staging and swap leaves the copy behind, which would ship in every export."""
    leftover = create_staged_file(checkpoint_dir, REPOCARD_NAME).name
    (checkpoint_dir / leftover).write_text("---\ntags:\n- halo\n---\n")
    out = tmp_path / "merged"
    out.mkdir()
    copy_checkpoint_aux_files(str(checkpoint_dir), str(out))
    assert not (out / leftover).exists(), "a staged card from a crashed write was carried into the export"
    assert (out / "config.json").exists(), "the skip took the real aux files with it"


@pytest.mark.parametrize("include_resume_sidecars", [True, False], ids=["resume-source", "n-way-merge"])
@pytest.mark.parametrize(
    "sidecar", [REFERENCE_LOGPS_FILE, prefetch_pending_filename(0, 2), OPTIMIZER_META_FILE], ids=lambda name: name
)
def test_an_interrupted_staged_write_is_not_exported(checkpoint_dir, tmp_path, include_resume_sidecars, sidecar):
    target = checkpoint_dir / sidecar
    atomic_torch_save(str(target), lambda: {"values": torch.tensor([-1.0, -2.0])})
    stage = create_staged_file(checkpoint_dir, sidecar)
    stage.write_bytes(b"an interrupted sidecar, not a complete one")
    (checkpoint_dir / ".gitattributes").write_text("*.safetensors filter=lfs\n")
    out = tmp_path / "merged"
    out.mkdir()

    copy_checkpoint_aux_files(str(checkpoint_dir), str(out), include_resume_sidecars=include_resume_sidecars)

    assert not (out / stage.name).exists(), "a crashed staged write was carried into the export"
    assert (out / ".gitattributes").exists(), "the staging skip took an ordinary hidden file with it"
    assert (out / "config.json").exists(), "the skip also removed real aux files"


@pytest.mark.parametrize("leftover", LEFTOVER_STAGES)
def test_every_staging_spelling_a_crash_leaves_stays_out_of_an_export(checkpoint_dir, tmp_path, leftover):
    """The withheld trainer state of a save that never committed, a current stage, and two spellings earlier
    releases staged under."""
    (checkpoint_dir / leftover).write_bytes(b"an unpublished stage")
    (checkpoint_dir / f".{REFERENCE_LOGPS_FILE}.notes").write_text("a hidden file no writer stages")
    out = tmp_path / "merged"
    out.mkdir()

    copy_checkpoint_aux_files(str(checkpoint_dir), str(out))

    assert not (out / leftover).exists(), f"{leftover} was carried into the export"
    assert (out / f".{REFERENCE_LOGPS_FILE}.notes").exists(), "the staging skip took an unstaged hidden file"
    assert (out / REFERENCE_LOGPS_FILE).exists(), "the skip took the published sidecar with it"


def test_foreign_framework_exports_are_not_carried(checkpoint_dir, tmp_path):
    """They bloat the converted checkpoint by tens of GB and shadow nothing useful."""
    out = tmp_path / "merged"
    out.mkdir()
    copy_checkpoint_aux_files(str(checkpoint_dir), str(out))
    for name in ("consolidated.00.pth", "model.gguf", "tf_model.h5", "rust_model.ot", "64.tflite", "model.onnx"):
        assert not (out / name).exists(), f"{name} is a weight export and must not be carried over"


def test_hidden_directories_stay_behind(checkpoint_dir, tmp_path):
    """A model source is often a working directory, so a whole repository history would land in every artifact."""
    (checkpoint_dir / ".git").mkdir()
    (checkpoint_dir / ".git" / "HEAD").write_text("ref: refs/heads/main")
    (checkpoint_dir / ".cache" / "huggingface").mkdir(parents=True)
    (checkpoint_dir / ".cache" / "huggingface" / "blob").write_text("x")

    out = tmp_path / "merged"
    out.mkdir()
    copy_checkpoint_aux_files(str(checkpoint_dir), str(out))

    assert not (out / ".git").exists(), "the workspace's git history is not artifact aux data"
    assert not (out / ".cache").exists()
    assert (out / "2_Dense" / "model.safetensors").exists(), "the exclusion blanketed every directory"


def test_a_runs_own_directories_stay_behind(checkpoint_dir, tmp_path):
    """A run's ``output_dir`` is also an export source; this state is gigabytes and not the artifact's."""
    (checkpoint_dir / "_reference_cache" / "launch").mkdir(parents=True)
    (checkpoint_dir / "_reference_cache" / "launch" / "merged.values").write_bytes(b"scores")
    (checkpoint_dir / "_incomplete_checkpoints" / "checkpoint-200").mkdir(parents=True)
    (checkpoint_dir / "_incomplete_checkpoints" / "checkpoint-200" / "config.json").write_text("{}")
    out = tmp_path / "merged"
    out.mkdir()

    copy_checkpoint_aux_files(str(checkpoint_dir), str(out))

    assert not (out / "_reference_cache").exists()
    assert not (out / "_incomplete_checkpoints").exists()
    assert (out / "1_Pooling" / "config.json").exists(), "the skip took a module directory with it"


def test_an_output_directory_inside_the_input_is_refused(checkpoint_dir):
    """It would copy itself into itself and fill the disk."""
    nested = checkpoint_dir / "merged"
    nested.mkdir()
    with pytest.raises(ValueError, match="inside input_dir"):
        copy_checkpoint_aux_files(str(checkpoint_dir), str(nested))


def test_a_sibling_output_directory_is_allowed(checkpoint_dir, tmp_path):
    out = tmp_path / "merged"
    out.mkdir()
    copy_checkpoint_aux_files(str(checkpoint_dir), str(out))
    assert (out / "config.json").exists()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
