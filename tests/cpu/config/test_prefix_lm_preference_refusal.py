#!/usr/bin/env python
"""DPO, KTO and SMPO refuse image data on a prefix-LM checkpoint (PaliGemma) before any weight loads.

No vision preference collator marks a completion as the causal suffix in ``token_type_ids``: TRL's (DPO, KTO) calls
the PaliGemma processor on the completions without images, which raises a batch later, and SMPO's marks the whole
row as prefix, so every completion would score with its own next tokens in view. Driven through each script's
``main()`` with the checkpoint probe stubbed, so the refusal is pinned on the probe the script's run verdict reads.

Run: ``python tests/cpu/config/test_prefix_lm_preference_refusal.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import sys
import types
from unittest import mock

import pytest
from datasets import Dataset, DatasetDict
from transformers import Gemma3Config

from halo.entrypoints import data_setup
from halo.entrypoints.data_setup import reject_prefix_lm_image_preference
from tests.common.prefix_lm import tiny_paligemma_config, tiny_paligemma_processor
from tests.common.repo import load_script_module

_MODEL_ID = "stub/paligemma2-3b-pt-224"
_TURNS = [{"role": "user", "content": "hi"}]
# method -> (script, the loader it calls once its data verdict is in, its rows)
_SCRIPTS = {
    "DPO": ("scripts/training/preference/dpo.py", "load_preference_models", ("prompt", "chosen", "rejected")),
    "KTO": ("scripts/training/preference/kto.py", "load_preference_models", ("prompt", "completion")),
    "SMPO": ("scripts/training/preference/smpo.py", "load_policy_model", ("prompt", "chosen", "rejected")),
}


class _ModelLoadReached(Exception):
    """Raised by the stubbed model load: the run got past its data verdict."""


def _paligemma_config():
    return tiny_paligemma_config(tiny_paligemma_processor())


def _image_rows(text_columns: tuple[str, ...]) -> DatasetDict:
    data = {column: [_TURNS] for column in text_columns}
    data.update(images=[[]], label=[True])
    return DatasetDict({"train": Dataset.from_dict(data), "test": Dataset.from_dict(data)})


def _run(tmp_path, method: str, checkpoint_config) -> None:
    """Run ``method``'s ``main()`` on an image dataset up to its model load, the checkpoint probe answering
    ``checkpoint_config`` (multimodal)."""
    script, model_loader, text_columns = _SCRIPTS[method]
    module = load_script_module(script, f"halo_test_prefix_lm_{method.lower()}")
    config = tmp_path / "config.yaml"
    config.write_text(
        f"model_name_or_path: {_MODEL_ID}\ndataset:\n- dummy/dataset\noutput_dir: {tmp_path / 'out'}\n"
        f"bf16: false\nuse_cpu: true\n"
    )
    runtime = types.SimpleNamespace(
        parallelism_config=types.SimpleNamespace(cp_size=1, is_cp_mode=False, pp_size=1),
        model_source=_MODEL_ID,
        mode_suffix="",
        local_rank=0,
        resume_checkpoint=None,
        policy_from_checkpoint=False,
    )
    patches = [
        mock.patch.object(module, "init_training_script", return_value=runtime),
        mock.patch.object(module, "load_script_datasets", return_value=(_image_rows(text_columns), False)),
        mock.patch.object(data_setup, "probe_checkpoint", return_value=(checkpoint_config, True)),
        mock.patch.object(module, model_loader, side_effect=_ModelLoadReached),
        mock.patch("halo.entrypoints.script.install_log_tee"),
        mock.patch.object(sys, "argv", ["prog", str(config)]),
    ]
    with mock.patch.object(module, "with_distributed_teardown", lambda fn: fn):
        for patch in patches:
            patch.start()
        try:
            module.main()
        finally:
            for patch in reversed(patches):
                patch.stop()


@pytest.mark.parametrize("method", list(_SCRIPTS))
def test_the_script_refuses_a_prefix_lm_image_run_before_the_model_load(tmp_path, method):
    with pytest.raises(ValueError, match=f"{method} on image data cannot train '{_MODEL_ID}'"):
        _run(tmp_path, method, _paligemma_config())


@pytest.mark.parametrize("method", list(_SCRIPTS))
def test_another_vlm_reaches_the_model_load(tmp_path, method):
    """Anti-vacuity: the same image run on Gemma 3, which is no prefix LM, gets past the refusal."""
    with pytest.raises(_ModelLoadReached):
        _run(tmp_path, method, Gemma3Config())


@pytest.mark.parametrize("checkpoint_config", [None, Gemma3Config()], ids=["unread-config", "gemma3"])
def test_a_checkpoint_that_is_no_prefix_lm_passes(checkpoint_config):
    """A config the probe could not read (``None``) is no prefix LM: the probe's own name fallback decides the
    modality, and the refusal keys on the family alone."""
    reject_prefix_lm_image_preference(checkpoint_config, _MODEL_ID, "DPO")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
