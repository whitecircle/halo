"""copy_checkpoint_aux_files: weight artifacts are skipped, resume sidecars survive, module
directories ride whole.

A merged directory is the mandated resume source for sharded EP/TP checkpoints
(resolve_resume_weights_source), so the copy must keep ``scheduler.pt``,
``router_balancing_biases.pt`` and every ``rng_state_<rank>.pth`` — dropping them silently
re-warms the LR schedule from step 0, zeroes the router balancing biases, and re-draws every
shuffle and dropout mask on resume — while still refusing to carry weight files that would
shadow the freshly written safetensors. The refusal covers foreign-framework exports
(``consolidated.*.pth``, ``.gguf``, ``.h5``, ``rust_model.ot``, ``*.tflite``) a hub source ships
beside them: those are weights too, and copying them bloats every converted checkpoint. That
``.pth`` sits on both sides is exactly why the sidecar exemption is by name, not by suffix.

Subdirectories are part of the artifact and copy whole, their own weights included: a
SentenceTransformer module directory (``1_Pooling/``, ``2_Dense/``) or
``original_adapter_config/`` carries weights no merge rewrites, so skipping the directory (or
filtering weights inside it) leaves ``modules.json`` naming modules that no longer exist — exactly
what an embedding EP merge produces. Two kinds of directory stay behind: a nested
``checkpoint-N`` (a run's resume state, not the artifact's) and a vendor weight dump
(``original/`` — the same weights again, which the hub-download ignore list drops off the same
tuple). ``include_resume_sidecars=False`` is the seam for artifacts that describe no single
training run (an N-way model merge).

    python tests/cpu/checkpoint/test_checkpoint_aux_copy.py
"""

import sys

import pytest
import torch
from safetensors.torch import load_file, save_file

from src.checkpoint.format import WEIGHT_FILE_IGNORE_PATTERNS, copy_checkpoint_aux_files

SKIPPED = (
    "model.safetensors",
    "model-00001-of-00002.safetensors",
    "model.safetensors.index.json",
    "pytorch_model.bin",
    "training_args.bin",
    "optimizer.pt",
    "optimizer_shard_0.pt",
    # Foreign-framework exports a hub repo ships beside the safetensors — weights, not metadata.
    "consolidated.00.pth",
    "model.gguf",
    "tf_model.h5",
    "flax_model.msgpack",
    "rust_model.ot",
    "64.tflite",
    "model.onnx",
)
KEPT = (
    "config.json",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "preprocessor_config.json",
    "modeling_remote.py",
    "trainer_state.json",
)
SIDECARS = (
    "rng_state_0.pth",
    "scheduler.pt",
    "router_balancing_biases.pt",
)
# The SentenceTransformer module layout an embedding EP save produces: modules.json names these
# directories, and 2_Dense carries its OWN weights that no merge rewrites.
MODULE_DIRS = ("1_Pooling", "2_Dense", "original_adapter_config")
# Distinct payloads for the two same-named trainer_state.json files — the artifact's own and the one
# inside the nested checkpoint — so a copy that flattened the nested directory in is visible.
ARTIFACT_AUX_BYTES = b"x"
NESTED_RUN_STATE = b"the nested run's state, not the artifact's"


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
    # A nested checkpoint is a run's resume state, never the artifact's aux data. Its aux files
    # shadow the artifact's own by name, which is what a recurse-and-filter "fix" would flatten in.
    (src / "checkpoint-500").mkdir()
    (src / "checkpoint-500" / "optimizer.pt").write_bytes(b"x")
    (src / "checkpoint-500" / "trainer_state.json").write_bytes(NESTED_RUN_STATE)
    # A Llama/Mistral-style vendor weight dump: the raw format beside the transformers one.
    (src / "original").mkdir()
    save_file({"tok_embeddings.weight": torch.ones(2, 2)}, str(src / "original" / "consolidated.safetensors"))
    return src


def test_weight_files_skipped_and_aux_kept(checkpoint_dir, tmp_path):
    out = tmp_path / "merged"
    out.mkdir()
    copy_checkpoint_aux_files(str(checkpoint_dir), str(out))
    copied = {p.name for p in out.iterdir()}
    assert copied == set(KEPT) | set(SIDECARS) | set(MODULE_DIRS) | {"modules.json"}, (
        f"unexpected copy set: {sorted(copied)}"
    )


def test_resume_sidecars_survive_merge_copy(checkpoint_dir, tmp_path):
    out = tmp_path / "merged"
    out.mkdir()
    copy_checkpoint_aux_files(str(checkpoint_dir), str(out))
    assert (out / "scheduler.pt").exists(), "LR schedule must survive into the merged resume source"
    assert (out / "router_balancing_biases.pt").exists(), "router biases must survive into the merged resume source"
    # Same suffix as the foreign exports below, opposite verdict: a per-rank RNG state is what a
    # bit-reproducible resume replays from.
    assert (out / "rng_state_0.pth").exists(), "per-rank RNG state must survive into the resume source"
    assert not (out / "optimizer.pt").exists()
    assert not (out / "pytorch_model.bin").exists()


def test_resume_sidecars_can_be_excluded(checkpoint_dir, tmp_path):
    """The model-merge seam: an N-way merged artifact describes no single run, so its aux copy
    passes include_resume_sidecars=False — sidecars dropped, everything else identical."""
    out = tmp_path / "merged"
    out.mkdir()
    copy_checkpoint_aux_files(str(checkpoint_dir), str(out), include_resume_sidecars=False)
    for name in SIDECARS:
        assert not (out / name).exists(), f"{name} describes one run's state and must not ship in a merge"
    for name in KEPT:
        assert (out / name).exists(), f"{name} must still be carried"


def test_module_directories_copy_whole_with_their_weights(checkpoint_dir, tmp_path):
    """The embedding EP merge failure: modules.json points at 1_Pooling/ and 2_Dense/, and 2_Dense's
    weights are the module — a copy that skips directories (or strips weights inside them) ships a
    modules.json naming modules that do not exist."""
    out = tmp_path / "merged"
    out.mkdir()
    copy_checkpoint_aux_files(str(checkpoint_dir), str(out))
    assert (out / "1_Pooling" / "config.json").exists()
    assert (out / "original_adapter_config" / "adapter_config.json").exists()
    dense_weights = out / "2_Dense" / "model.safetensors"
    assert dense_weights.exists(), "a module directory's own weights ARE the module — they must ride"
    assert torch.equal(load_file(str(dense_weights))["linear.weight"], torch.ones(2, 2))


def test_nested_checkpoint_directories_stay_behind(checkpoint_dir, tmp_path):
    """The exclusion is one directory NAME, not a retreat from copying directories at all — the
    blanket skip the module-directory case above exists to forbid. Both
    halves are asserted off the same call, so a copy that simply skipped every directory cannot pass
    here. The content check is the third way this can go wrong: the nested run ships a
    ``trainer_state.json`` of its own, so a copy that recursed into the directory instead of leaving
    it can shadow the artifact's."""
    out = tmp_path / "merged"
    out.mkdir()
    copy_checkpoint_aux_files(str(checkpoint_dir), str(out))

    assert not (out / "checkpoint-500").exists(), "a nested checkpoint is resume state, not artifact aux data"
    assert (out / "2_Dense" / "model.safetensors").exists(), "the exclusion blanketed every directory"
    assert (out / "trainer_state.json").read_bytes() == ARTIFACT_AUX_BYTES, (
        "the nested run's trainer_state.json was flattened over the artifact's own"
    )


def test_a_vendor_weight_dump_directory_stays_behind(checkpoint_dir, tmp_path):
    """``original/`` is the vendor's own copy of the SAME weights (hundreds of GB on a Mistral/Llama
    source). Copying directories whole must not start duplicating it into every converted
    checkpoint — the hub-download ignore list drops it for exactly this reason, off the same tuple."""
    out = tmp_path / "merged"
    out.mkdir()
    copy_checkpoint_aux_files(str(checkpoint_dir), str(out))
    assert not (out / "original").exists(), "a vendor weight dump is weights, not aux data"
    assert "original/*" in WEIGHT_FILE_IGNORE_PATTERNS, "the hub download must drop what the local copy drops"
    # Not a prefix match: original_adapter_config/ IS aux data (asserted copied above).
    assert (out / "original_adapter_config").exists()


def test_foreign_framework_exports_are_not_carried(checkpoint_dir, tmp_path):
    """A hub source's TF/Flax/GGUF/ONNX/CoreML/rust exports are weights: copying them bloats the
    converted checkpoint by tens of GB and can shadow nothing useful."""
    out = tmp_path / "merged"
    out.mkdir()
    copy_checkpoint_aux_files(str(checkpoint_dir), str(out))
    for name in ("consolidated.00.pth", "model.gguf", "tf_model.h5", "rust_model.ot", "64.tflite", "model.onnx"):
        assert not (out / name).exists(), f"{name} is a weight export and must not be carried over"


# --- Directory-copy blast radius: a denylist of two names is not a filter on what may be copied. ---


def test_hidden_directories_stay_behind(checkpoint_dir, tmp_path):
    """A model source is often a working directory, so ``.git``/``.cache``/``.ipynb_checkpoints`` sit
    beside the weights. Copying subdirectories whole with only a two-name denylist duplicates a whole
    repository history into every converted artifact."""
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


def test_an_output_directory_inside_the_input_is_refused(checkpoint_dir):
    """The walk copies subdirectories whole, so a destination nested in the source copies itself into
    itself — it fills the disk instead of failing."""
    nested = checkpoint_dir / "merged"
    nested.mkdir()
    with pytest.raises(ValueError, match="inside input_dir"):
        copy_checkpoint_aux_files(str(checkpoint_dir), str(nested))


def test_a_sibling_output_directory_is_allowed(checkpoint_dir, tmp_path):
    """Anti-over-rejection: the guard is about NESTING, not about sharing a parent."""
    out = tmp_path / "merged"
    out.mkdir()
    copy_checkpoint_aux_files(str(checkpoint_dir), str(out))
    assert (out / "config.json").exists()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
