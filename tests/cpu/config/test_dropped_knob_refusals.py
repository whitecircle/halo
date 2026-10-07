#!/usr/bin/env python
"""Knobs a training script would overwrite or never read are refused at startup, not dropped.

* SFT and self-distillation render ``conversation_field`` through their own processor and collator,
  and ``disable_trl_dataset_prep`` hands TRL a ``dataset_kwargs`` of its own, so ``dataset_text_field``
  and a YAML ``dataset_kwargs`` parse and do nothing.
* ``eval_packing`` only narrows packing on SFT's raw-dataset path: ``true`` without ``packing`` packs
  nothing, a pre-processed dataset is used as baked, and self-distillation refuses packing outright.
* Environmental GRPO pins ``max_completion_length`` to ``rollout_max_tokens``; a third value would be
  overwritten. The value equal to either the field default or ``rollout_max_tokens`` stays legal.
* SFT refuses ``packing`` / ``padding_free`` for a checkpoint with compressed-KV layers off the
  config its modality probe already read, before the training setup and the model load.

Each refusal must fire before the distributed init and the model load; the anti-over-rejection
cases reach the first stubbed step past the guards instead.

Usage:
    python tests/cpu/config/test_dropped_knob_refusals.py
"""

import contextlib
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from transformers.models.deepseek_v4 import DeepseekV4Config

from tests.common.models import TINY_DSV4_CONFIG
from tests.common.utils import load_script_module

_REFUSED = "does not support these config fields"


class _PastTheGuards(Exception):
    """Raised by the first stubbed step after the guards: main() got through them."""


def _stop(*args, **kwargs):
    raise _PastTheGuards


def _run_main(script: str, config_body: str, tmp_path: Path, **stubs) -> None:
    """Drive ``scripts/training/<script>:main()`` from a minimal YAML; each stub replaces that name in
    the script's namespace."""
    module = load_script_module(f"scripts/training/{script}", "halo_test_dropped_knobs_" + Path(script).stem)
    config = tmp_path / "config.yaml"
    config.write_text(
        f"model_name_or_path: stub/qwen3-4b\ndataset:\n- dummy/dataset\noutput_dir: {tmp_path / 'out'}\n"
        f"bf16: false\nuse_cpu: true\n{config_body}"
    )
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch("src.training.parser.install_log_tee"))
        stack.enter_context(mock.patch.object(sys, "argv", ["prog", str(config)]))
        for name, stub in stubs.items():
            stack.enter_context(mock.patch.object(module, name, stub))
        module.main()


# SFT: before init_distributed, the first step that touches the process group or the hub.


@pytest.mark.parametrize(
    ("config_body", "knob"),
    [
        ("dataset_text_field: messages\n", "dataset_text_field"),
        ("dataset_kwargs:\n  add_special_tokens: false\n", "dataset_kwargs"),
    ],
)
def test_sft_refuses_trl_dataset_prep_knobs(config_body, knob, tmp_path):
    with pytest.raises(ValueError, match=rf"{_REFUSED}.*{knob}"):
        _run_main("sft.py", config_body, tmp_path, init_distributed=_stop)


def test_sft_refuses_eval_packing_without_packing(tmp_path):
    with pytest.raises(ValueError, match="eval_packing=True packs nothing without packing=True"):
        _run_main("sft.py", "eval_packing: true\n", tmp_path, init_distributed=_stop)


@pytest.mark.parametrize(
    "config_body",
    [
        "",
        # The declared default spelled out is not a request.
        "dataset_text_field: text\n",
        # Narrowing is what eval_packing does honor.
        "packing: true\nmax_length: 64\neval_packing: false\n",
        "packing: true\nmax_length: 64\neval_packing: true\n",
        "eval_packing: false\n",
    ],
)
def test_sft_accepts_what_it_honors(config_body, tmp_path):
    with pytest.raises(_PastTheGuards):
        _run_main("sft.py", config_body, tmp_path, init_distributed=_stop)


def _run_sft_to_dataset(tmp_path, config_body: str, *, preprocessed: bool) -> None:
    """Run ``sft.py:main()`` through its dataset load, stopping at the step after it."""
    _run_main(
        "sft.py",
        config_body,
        tmp_path,
        init_distributed=lambda: None,
        probe_checkpoint=lambda *args, **kwargs: (None, False),
        init_training_script=lambda *args, **kwargs: SimpleNamespace(parallelism_config=None),
        load_script_datasets=lambda *args, **kwargs: ((None, preprocessed), False),
        resolve_vlm_run=_stop,
    )


def test_sft_refuses_eval_packing_on_a_preprocessed_dataset(tmp_path):
    """Both baked splits are used as-is: an unpacked eval requested over a packed artifact would not happen."""
    with pytest.raises(ValueError, match=rf"pre-processed dataset {_REFUSED}.*eval_packing"):
        _run_sft_to_dataset(tmp_path, "packing: true\nmax_length: 64\neval_packing: false\n", preprocessed=True)


def test_sft_keeps_eval_packing_on_a_raw_dataset(tmp_path):
    with pytest.raises(_PastTheGuards):
        _run_sft_to_dataset(tmp_path, "packing: true\nmax_length: 64\neval_packing: false\n", preprocessed=False)


@pytest.mark.parametrize("config_body", ["packing: true\nmax_length: 64\n", "padding_free: true\n"])
@pytest.mark.parametrize("compressed", [True, False], ids=["compressed-kv", "masked-attention-only"])
def test_sft_refuses_multi_document_rows_on_compressed_kv_before_the_setup(config_body, compressed, tmp_path):
    layer_types = TINY_DSV4_CONFIG["layer_types"] if compressed else ["sliding_attention"] * 3
    checkpoint_config = DeepseekV4Config(**{**TINY_DSV4_CONFIG, "layer_types": layer_types})
    run = partial(
        _run_main,
        "sft.py",
        config_body,
        tmp_path,
        init_distributed=lambda: None,
        probe_checkpoint=lambda *args, **kwargs: (checkpoint_config, False),
        init_training_script=_stop,
    )
    if compressed:
        with pytest.raises(ValueError, match="layers pool KV over windows cut at fixed indices"):
            run()
    else:
        with pytest.raises(_PastTheGuards):
            run()


# Self-distillation: before init_training_script.


@pytest.mark.parametrize(
    ("config_body", "knob"),
    [
        ("dataset_text_field: messages\n", "dataset_text_field"),
        ("dataset_kwargs:\n  add_special_tokens: false\n", "dataset_kwargs"),
        ("eval_packing: true\n", "eval_packing"),
        ("eval_packing: false\n", "eval_packing"),
    ],
)
def test_self_distill_refuses_trl_dataset_prep_knobs(config_body, knob, tmp_path):
    with pytest.raises(ValueError, match=rf"{_REFUSED}.*{knob}"):
        _run_main("distillation/self_distill.py", config_body, tmp_path, init_training_script=_stop)


def test_self_distill_accepts_the_defaults(tmp_path):
    with pytest.raises(_PastTheGuards):
        _run_main("distillation/self_distill.py", "dataset_text_field: text\n", tmp_path, init_training_script=_stop)


def test_self_distill_refuses_a_loss_type_its_own_cross_entropy_is_not(tmp_path):
    """The trainer computes its own nll, so TRL's loss_type never reaches the loss: ``dft`` would be
    silently ignored."""
    with pytest.raises(ValueError, match="loss_type='dft' is not implemented for self-distillation"):
        _run_main("distillation/self_distill.py", "loss_type: dft\n", tmp_path, init_training_script=_stop)


def test_self_distill_checks_a_separate_reference_tokenizer_before_any_load(tmp_path):
    """Only tokenizer files are read, so a mismatched anchor repo is refused before the dataset and
    both model loads."""
    parallelism = SimpleNamespace(is_ep_mode=False, is_tp_mode=False)
    vocabs = {"stub/qwen3-4b": {"a": 0, "b": 1}, "org/other-reference": {"a": 1, "b": 0}}
    tokenizers = SimpleNamespace(
        from_pretrained=lambda path, **_: SimpleNamespace(get_vocab=lambda: dict(vocabs[path]))
    )
    stubs = {
        "init_training_script": lambda *a, **k: SimpleNamespace(parallelism_config=parallelism),
        "AutoTokenizer": tokenizers,
        "load_script_datasets": _stop,
    }
    with pytest.raises(ValueError, match="disagree on"):
        _run_main(
            "distillation/self_distill.py",
            "reference_kl_coef: 0.1\nreference_model_name_or_path: org/other-reference\n",
            tmp_path,
            **stubs,
        )
    with pytest.raises(_PastTheGuards):
        _run_main("distillation/self_distill.py", "reference_kl_coef: 0.1\n", tmp_path, **stubs)


def test_self_distill_refuses_confidence_weighting_over_a_presharded_load(tmp_path):
    """The weights divide by the train split's mean confidence; per-rank shards would each divide by
    their own, so a row's weight would depend on the rank that drew it."""
    parallelism = SimpleNamespace(is_ep_mode=False, is_tp_mode=False)
    runtime = SimpleNamespace(parallelism_config=parallelism)
    with pytest.raises(ValueError, match="presharded dataset"):
        _run_main(
            "distillation/self_distill.py",
            "confidence_field: conf\n",
            tmp_path,
            init_training_script=lambda *a, **k: runtime,
            load_script_datasets=lambda *a, **k: (None, True),
        )


@pytest.mark.parametrize("loss_type", ["nll", "chunked_nll"])
def test_self_distill_accepts_both_spellings_of_its_own_nll(loss_type, tmp_path):
    """TRL's default either side of 1.7, and the loss the trainer computes."""
    with pytest.raises(_PastTheGuards):
        _run_main("distillation/self_distill.py", f"loss_type: {loss_type}\n", tmp_path, init_training_script=_stop)


# Environmental GRPO: before init_training_script.


def test_env_grpo_refuses_a_completion_length_it_would_overwrite(tmp_path):
    with pytest.raises(ValueError, match=r"max_completion_length=1024 is not a knob.*rollout_max_tokens \(512\)"):
        _run_main(
            "environmental_grpo.py",
            "rollout_max_tokens: 512\nmax_completion_length: 1024\n",
            tmp_path,
            init_training_script=_stop,
        )


@pytest.mark.parametrize(
    "config_body",
    [
        "rollout_max_tokens: 512\n",
        # Equal to the value the script writes: nothing is discarded.
        "rollout_max_tokens: 512\nmax_completion_length: 512\n",
    ],
)
def test_env_grpo_accepts_a_completion_length_it_would_not_change(config_body, tmp_path):
    with pytest.raises(_PastTheGuards):
        _run_main("environmental_grpo.py", config_body, tmp_path, init_training_script=_stop)


def test_env_grpo_refuses_a_react_system_prompt_before_the_model_load(tmp_path):
    """The registry refuses the key; the script must reach that refusal before any load."""
    with pytest.raises(ValueError, match="environment_kwargs.system_prompt"):
        _run_main(
            "environmental_grpo.py",
            "environment_type: react_math\nenvironment_kwargs:\n  system_prompt: Answer tersely.\n",
            tmp_path,
            init_training_script=_stop,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
