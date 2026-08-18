#!/usr/bin/env python
"""CPU tests for the distributed-resume weights-source resolution.

``resolve_resume_weights_source`` decides where the POLICY model loads weights from on resume, and is
what keeps a resume off silent base weights: modes needing EP wrappers or CP skip the Trainer's
checkpoint weight reload (the EP-fused / CP-wrapped structure can't ingest HF-format keys), so for
those modes the trained weights must be loaded at construction from the checkpoint. With the
production default ``use_grouped_gemm: true``, ``needs_ep_wrappers`` is True even without EP — EVERY
stock torchrun resume repoints; only ``use_grouped_gemm: false`` (and no EP/CP) keeps loading from
base and lets the loader reload the checkpoint. These tests lock that decision table against REAL
``ParallelismConfig`` instances (returning base for a wrapper/CP mode resumes training from the
untrained base, silently) and the fail-loud path for an unloadable checkpoint.

Run: python tests/cpu/checkpoint/test_resume_weights_source.py  (or: pytest -m cpu tests/cpu/checkpoint/test_resume_weights_source.py)
"""

import json
import sys

import pytest
from accelerate import PartialState

from src.training.environment import (
    _checkpoint_has_full_model_weights,
    _classify_resume_checkpoint,
    resolve_resume_weights_source,
)
from tests.common.parallelism import make_parallelism_config

# The resolver logs through the accelerate logger, which requires an initialized state.
PartialState()

BASE = "org/base-model"


def _mode(*, ep=False, cp=False, grouped_gemm=False):
    """A real ParallelismConfig on a single 8-GPU node.

    The resolver gates on ``needs_ep_wrappers or is_cp_mode``, and ``needs_ep_wrappers`` is derived
    (``ep_group_size > 1 or use_grouped_gemm``). ``grouped_gemm`` defaults to False here — against
    the production default — so ``ep``/``cp`` are the only thing that opens the gate; leaving it on
    would make every EP and CP case below pass with both of those terms deleted from the resolver.
    """
    return make_parallelism_config(
        world_size=8,
        gpus_per_node=8,
        ep_size=8 if ep else 1,
        cp_size=8 if cp else 1,
        use_grouped_gemm=grouped_gemm,
    )


def _model_config():
    return type("_MC", (), {"model_name_or_path": BASE})()


def _touch(path):
    with open(path, "w") as f:
        f.write("x")


def _write_index(path, fmt=None):
    meta = {} if fmt is None else {"format": fmt}
    with open(path, "w") as f:
        json.dump({"metadata": meta, "weight_map": {}}, f)


def test_full_weights_single_safetensors(tmp_path):
    _touch(tmp_path / "model.safetensors")
    assert _checkpoint_has_full_model_weights(str(tmp_path)) is True


def test_full_weights_gathered_index(tmp_path):
    # Gathered HF shard index (no per-rank "format") is directly loadable.
    _write_index(tmp_path / "model.safetensors.index.json", fmt=None)
    assert _checkpoint_has_full_model_weights(str(tmp_path)) is True


def test_full_weights_legacy_bin(tmp_path):
    _touch(tmp_path / "pytorch_model.bin")
    assert _checkpoint_has_full_model_weights(str(tmp_path)) is True


def test_sharded_index_is_not_full(tmp_path):
    # A per-rank EP save reuses the index filename but holds partial tensors: merge first.
    _write_index(tmp_path / "model.safetensors.index.json", fmt="ep_sharded")
    assert _checkpoint_has_full_model_weights(str(tmp_path)) is False


def test_sharded_index_beside_a_gathered_file_is_not_full(tmp_path):
    """A directory carrying BOTH is a gathered save torn by a later sharded one. The sharded
    rejection wins: probing ``model.safetensors`` first would call the stale gathered half loadable
    and resume off pre-sharded-save weights with no error anywhere."""
    _touch(tmp_path / "model.safetensors")
    _write_index(tmp_path / "model.safetensors.index.json", fmt="ep_sharded")
    assert _checkpoint_has_full_model_weights(str(tmp_path)) is False
    assert _classify_resume_checkpoint(str(tmp_path)) == "invalid"


def test_adapter_only_is_not_full(tmp_path):
    _touch(tmp_path / "adapter_model.safetensors")
    assert _checkpoint_has_full_model_weights(str(tmp_path)) is False


def test_empty_dir_is_not_full(tmp_path):
    assert _checkpoint_has_full_model_weights(str(tmp_path)) is False


def test_no_checkpoint_returns_base(tmp_path):
    assert resolve_resume_weights_source(None, _model_config(), _mode(ep=True)) == BASE


@pytest.mark.parametrize("mode", [_mode(ep=True), _mode(cp=True)])
def test_ep_cp_full_checkpoint_repoints_to_checkpoint(tmp_path, mode):
    """The crux: EP/CP full-finetune resume must load weights FROM the checkpoint, not base."""
    _touch(tmp_path / "model.safetensors")
    assert resolve_resume_weights_source(str(tmp_path), _model_config(), mode) == str(tmp_path)


def test_default_config_full_checkpoint_repoints(tmp_path):
    """The production default (use_grouped_gemm=True ⇒ needs_ep_wrappers=True even without EP): every
    stock torchrun resume loads policy weights FROM the checkpoint."""
    _touch(tmp_path / "model.safetensors")
    assert resolve_resume_weights_source(str(tmp_path), _model_config(), _mode(grouped_gemm=True)) == str(tmp_path)


def test_no_wrappers_no_cp_full_checkpoint_stays_base(tmp_path):
    """Without EP wrappers or CP (reachable only with use_grouped_gemm: false) the Trainer's loader
    reloads the checkpoint itself, so the policy still loads from base."""
    _touch(tmp_path / "model.safetensors")
    assert resolve_resume_weights_source(str(tmp_path), _model_config(), _mode()) == BASE


def test_ep_adapter_only_stays_base(tmp_path):
    """Adapter-only EP checkpoint: frozen base stays at model_name_or_path; loader restores adapter."""
    _touch(tmp_path / "adapter_model.safetensors")
    assert resolve_resume_weights_source(str(tmp_path), _model_config(), _mode(ep=True)) == BASE


def test_ep_adapter_bin_only_stays_base(tmp_path):
    """PeftAdapterSaver falls back to adapter_model.bin when the safetensors write fails; that
    checkpoint must resume like any adapter save, not raise as 'invalid'."""
    _touch(tmp_path / "adapter_model.bin")
    assert resolve_resume_weights_source(str(tmp_path), _model_config(), _mode(ep=True)) == BASE


@pytest.mark.parametrize("adapter_name", ["adapter_model.safetensors", "adapter_model.bin"])
def test_classify_accepts_both_adapter_filenames(tmp_path, adapter_name):
    """The classifier must accept every filename PeftAdapterSaver can write (the loader restores
    both); classifying the .bin fallback 'invalid' made those checkpoints un-resumable."""
    _touch(tmp_path / adapter_name)
    assert _classify_resume_checkpoint(str(tmp_path)) == "adapter"


def test_classify_full_beats_adapter(tmp_path):
    _touch(tmp_path / "model.safetensors")
    _touch(tmp_path / "adapter_model.bin")
    assert _classify_resume_checkpoint(str(tmp_path)) == "full"


def test_classify_empty_is_invalid(tmp_path):
    assert _classify_resume_checkpoint(str(tmp_path)) == "invalid"


def test_ep_unmerged_sharded_checkpoint_raises(tmp_path):
    """EP full-finetune resume from an unmerged sharded save (no loadable weights, no adapter) must
    fail loud — otherwise the loader skips the reload and training silently continues from base."""
    _write_index(tmp_path / "model.safetensors.index.json", fmt="ep_sharded")
    with pytest.raises(ValueError, match="silently continue from BASE"):
        resolve_resume_weights_source(str(tmp_path), _model_config(), _mode(ep=True))


def test_ep_empty_checkpoint_raises(tmp_path):
    with pytest.raises(ValueError, match="silently continue from BASE"):
        resolve_resume_weights_source(str(tmp_path), _model_config(), _mode(ep=True))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
