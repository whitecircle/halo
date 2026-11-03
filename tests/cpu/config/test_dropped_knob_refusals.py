#!/usr/bin/env python
"""Knobs a training script would overwrite or never read are refused at startup, before the training setup and the
model load; the anti-over-rejection cases reach the first stubbed step past the guards."""

import contextlib
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from transformers.models.deepseek_v4 import DeepseekV4Config
from transformers.models.glm5_next import Glm5NextConfig

from tests.common.models import BAILING_MOE_RING_MINI, TINY_DSV4_CONFIG, TINY_GPTOSS_CONFIG
from tests.common.repo import load_script_module
from tests.common.tiny_models import remote_config

_REFUSED = "does not support these config fields"
_SDPA = "attn_implementation: sdpa\n"
_FLASH = "attn_implementation: flash_attention_2\n"
_DENSE_LEAK = "this family's forward does not pass position_ids into its mask construction"
_UNREAD_SEQ_LENS = "padding_free requires a varlen attention kernel, but this model resolved to"


class _PastTheGuards(Exception):
    """Raised by the first stubbed step after the guards: main() got through them."""


def _stop(*args, **kwargs):
    raise _PastTheGuards


def _run_main(script: str, config_body: str, tmp_path: Path, *, bf16: bool = False, **stubs) -> None:
    """Drive ``scripts/training/<script>:main()`` from a minimal YAML; each stub replaces that name in
    the script's namespace. ``bf16`` keeps a flash request flash (an fp32 run falls back to a dense kernel)."""
    module = load_script_module(f"scripts/training/{script}", "halo_test_dropped_knobs_" + Path(script).stem)
    config = tmp_path / "config.yaml"
    config.write_text(
        f"model_name_or_path: stub/qwen3-4b\ndataset:\n- dummy/dataset\noutput_dir: {tmp_path / 'out'}\n"
        f"bf16: {str(bf16).lower()}\nuse_cpu: true\n{config_body}"
    )
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch("halo.entrypoints.script.install_log_tee"))
        stack.enter_context(mock.patch.object(sys, "argv", ["prog", str(config)]))
        for name, stub in stubs.items():
            stack.enter_context(mock.patch.object(module, name, stub))
        module.main()


# SFT: before the modality probe, the first step that touches the hub.


@pytest.mark.parametrize(
    ("config_body", "knob"),
    [
        ("dataset_text_field: messages\n", "dataset_text_field"),
        ("dataset_kwargs:\n  add_special_tokens: false\n", "dataset_kwargs"),
    ],
)
def test_sft_refuses_trl_dataset_prep_knobs(config_body, knob, tmp_path):
    with pytest.raises(ValueError, match=rf"{_REFUSED}.*{knob}"):
        _run_main("sft.py", config_body, tmp_path, probe_checkpoint=_stop)


def test_sft_refuses_eval_packing_without_packing(tmp_path):
    with pytest.raises(ValueError, match="eval_packing=True packs nothing without packing=True"):
        _run_main("sft.py", "eval_packing: true\n", tmp_path, probe_checkpoint=_stop)


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
        _run_main("sft.py", config_body, tmp_path, probe_checkpoint=_stop)


def _run_sft_to_dataset(tmp_path, config_body: str, *, preprocessed: bool) -> None:
    _run_main(
        "sft.py",
        config_body,
        tmp_path,
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
        probe_checkpoint=lambda *args, **kwargs: (checkpoint_config, False),
        init_training_script=_stop,
    )
    if compressed:
        with pytest.raises(ValueError, match="layers pool KV over windows cut at fixed indices"):
            run()
    elif "padding_free" in config_body:
        # DeepSeek-V4 runs eager only, so no kernel would read the flattened row's cu_seq_lens.
        with pytest.raises(ValueError, match=_UNREAD_SEQ_LENS):
            run()
    else:
        with pytest.raises(_PastTheGuards):
            run()


_FLAT_ROWS = {"packing": "packing: true\nmax_length: 64\n", "padding_free": "padding_free: true\n"}
_TREE_ROWS = "packing: true\nmax_length: 64\ntree_packing: true\ntree_share_markers: ['x']\n"
_CROSSING_OPT_IN = "allow_cross_document_linear_attention: true\n"


@pytest.mark.parametrize(
    ("config_body", "refusal"),
    [
        *((body, "allow_cross_document_linear_attention: true") for body in _FLAT_ROWS.values()),
        *((body + _CROSSING_OPT_IN, None) for body in _FLAT_ROWS.values()),
        (_TREE_ROWS, None),
        (_TREE_ROWS + _CROSSING_OPT_IN, rf"tree_packing on {_REFUSED}.*allow_cross_document_linear_attention"),
    ],
    ids=[*_FLAT_ROWS, *(f"{mode}-opted-in" for mode in _FLAT_ROWS), "tree_packing", "tree_packing-opted-in"],
)
def test_sft_refuses_unmarked_linear_attention_rows_before_the_setup(config_body, refusal, tmp_path):
    """Ring's Lightning Attention-2 layers read no document boundary, so a flat multi-document row is refused before
    the model load unless the run opts in. Tree packing brings its own plan, gets past this check without the
    opt-in and is refused the opt-in it never reads."""
    run = partial(
        _run_main,
        "sft.py",
        config_body,
        tmp_path,
        bf16=True,
        probe_checkpoint=lambda *args, **kwargs: (remote_config(BAILING_MOE_RING_MINI), False),
        init_training_script=_stop,
    )
    if refusal is None:
        with pytest.raises(_PastTheGuards):
            run()
    else:
        with pytest.raises(ValueError, match=refusal):
            run()


_GPT_OSS = partial(CONFIG_MAPPING["gpt_oss"], **TINY_GPTOSS_CONFIG)
_QWEN3 = CONFIG_MAPPING["qwen3"]
# name -> (builds the checkpoint config the probe read, the run's rows, the refusal or None: past the guards)
_PLANNED_BACKEND_ROWS = {
    "gpt_oss-sdpa-packing": (_GPT_OSS, _SDPA + _FLAT_ROWS["packing"], _DENSE_LEAK),
    "gpt_oss-flash-packing": (_GPT_OSS, _FLASH + _FLAT_ROWS["packing"], None),
    # Tree packing is checked against its own plan after the setup.
    "gpt_oss-sdpa-tree_packing": (_GPT_OSS, _SDPA + _TREE_ROWS, None),
    # A family whose dense forward isolates packs on SDPA; only the dense mask's cost is warned about, later.
    "qwen3-sdpa-packing": (_QWEN3, _SDPA + _FLAT_ROWS["packing"], None),
    "qwen3-sdpa-padding_free": (_QWEN3, _SDPA + _FLAT_ROWS["padding_free"], _UNREAD_SEQ_LENS),
    "qwen3-flash-padding_free": (_QWEN3, _FLASH + _FLAT_ROWS["padding_free"], None),
    # No flash backend, and its DSA layers read only the padding mask: refused whatever the opt-in says, before
    # the linear-attention check reads it.
    "glm5_next-packing": (Glm5NextConfig, _FLAT_ROWS["packing"], _DENSE_LEAK),
    "glm5_next-packing-opted-in": (Glm5NextConfig, _FLAT_ROWS["packing"] + _CROSSING_OPT_IN, _DENSE_LEAK),
    "glm5_next-padding_free": (Glm5NextConfig, _FLAT_ROWS["padding_free"], _UNREAD_SEQ_LENS),
    "glm5_next-padding_free-opted-in": (
        Glm5NextConfig,
        _FLAT_ROWS["padding_free"] + _CROSSING_OPT_IN,
        _UNREAD_SEQ_LENS,
    ),
}


@pytest.mark.parametrize("case", sorted(_PLANNED_BACKEND_ROWS))
def test_sft_refuses_rows_its_planned_backend_cannot_serve_before_the_setup(case, tmp_path):
    """The backend comes from the resolution the model load runs, so a row the collator factory would refuse after a
    minutes-long load is refused off the probed config first."""
    build_config, config_body, refusal = _PLANNED_BACKEND_ROWS[case]
    checkpoint_config = build_config()
    run = partial(
        _run_main,
        "sft.py",
        config_body,
        tmp_path,
        bf16=True,
        probe_checkpoint=lambda *args, **kwargs: (checkpoint_config, False),
        init_training_script=_stop,
    )
    with pytest.raises(ValueError, match=refusal) if refusal else pytest.raises(_PastTheGuards):
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


@pytest.mark.parametrize(
    "line",
    [
        "generate_eval_examples: false",
        "num_eval_examples: 50",
        "train_on_last_assistant_only: false",
        "allow_cross_document_linear_attention: false",
        "tree_packing: false",
        "tree_share_markers: []",
        "tree_share_requires: null",
        "tree_full_attention: null",
    ],
)
def test_self_distill_has_no_key_for_what_its_data_path_cannot_do(line, tmp_path):
    """Its dataset stays raw and its collators pad one row per conversation, masking every assistant turn. Each value
    is the SFT default, which an inherited field would accept."""
    with pytest.raises(ValueError, match=line.partition(":")[0]):
        _run_main("distillation/self_distill.py", f"{line}\n", tmp_path, init_training_script=_stop)


_OPD_KNOBS = {
    "sdpg_hint_template": "'Hint: {answer}'",
    "sdpg_loss": "forward_kl",
    "sdpg_temperature": 2.0,
    "sdpg_beta_warmup_steps": 5,
    "sdpg_beta_decay_steps": 5,
    "sdpg_answer_field": "gold",
    "privileged_solution_field": "reference",
    "opd_exclude_eos": "false",
}


@pytest.mark.parametrize(("knob", "value"), _OPD_KNOBS.items())
def test_self_distill_refuses_an_opd_knob_with_the_term_off(knob, value, tmp_path):
    body = f"{knob}: {value}\n"
    with pytest.raises(ValueError, match=rf"sdpg_beta_base: 0\) {_REFUSED}.*{knob}"):
        _run_main("distillation/self_distill.py", f"sdpg_beta_base: 0\n{body}", tmp_path, init_training_script=_stop)
    with pytest.raises(_PastTheGuards):
        _run_main("distillation/self_distill.py", body, tmp_path, init_training_script=_stop)


def test_self_distill_refuses_opd_confidence_weighting_with_the_term_off(tmp_path):
    """The SFT term still takes the confidence weights, so this needs a confidence_field to reach its own refusal."""
    body = "confidence_field: confidence\nconfidence_weight_opd: false\n"
    with pytest.raises(ValueError, match=rf"sdpg_beta_base: 0\) {_REFUSED}.*confidence_weight_opd"):
        _run_main("distillation/self_distill.py", f"sdpg_beta_base: 0\n{body}", tmp_path, init_training_script=_stop)
    with pytest.raises(_PastTheGuards):
        _run_main("distillation/self_distill.py", body, tmp_path, init_training_script=_stop)


def test_self_distill_keeps_the_temperature_its_reference_anchor_reads_with_the_opd_term_off(tmp_path):
    body = "sdpg_beta_base: 0\nreference_kl_coef: 0.1\nsdpg_temperature: 2.0\n"
    with pytest.raises(_PastTheGuards):
        _run_main("distillation/self_distill.py", body, tmp_path, init_training_script=_stop)
    # The rest stay unread beside the anchor.
    with pytest.raises(ValueError, match=rf"sdpg_beta_base: 0\) {_REFUSED}.*sdpg_loss"):
        _run_main(
            "distillation/self_distill.py", f"{body}sdpg_loss: forward_kl\n", tmp_path, init_training_script=_stop
        )


def test_self_distill_runs_with_the_opd_term_off(tmp_path):
    with pytest.raises(_PastTheGuards):
        _run_main("distillation/self_distill.py", "sdpg_beta_base: 0\n", tmp_path, init_training_script=_stop)


def _smpo_truncation_run(tmp_path, *, is_vlm: bool) -> None:
    runtime = SimpleNamespace(parallelism_config=SimpleNamespace(is_cp_mode=False))
    _run_main(
        "preference/smpo.py",
        "truncation_mode: keep_start\n",
        tmp_path,
        init_training_script=lambda *a, **k: runtime,
        load_script_datasets=lambda *a, **k: (None, False),
        resolve_preference_vlm_run=lambda *a, **k: is_vlm,
        load_policy_model=_stop,
    )


def test_smpo_refuses_a_truncation_mode_on_the_vlm_path(tmp_path):
    """The VLM collator refuses an over-length prompt instead of cutting one."""
    with pytest.raises(ValueError, match=rf"SMPO VLM mode {_REFUSED}.*truncation_mode"):
        _smpo_truncation_run(tmp_path, is_vlm=True)
    with pytest.raises(_PastTheGuards):
        _smpo_truncation_run(tmp_path, is_vlm=False)


def test_self_distill_accepts_the_defaults(tmp_path):
    with pytest.raises(_PastTheGuards):
        _run_main("distillation/self_distill.py", "dataset_text_field: text\n", tmp_path, init_training_script=_stop)


def test_self_distill_refuses_a_loss_type_its_own_cross_entropy_is_not(tmp_path):
    """The trainer computes its own nll, so TRL's loss_type never reaches the loss and ``dft`` would be silently
    ignored."""
    with pytest.raises(ValueError, match="loss_type='dft' is not implemented for self-distillation"):
        _run_main("distillation/self_distill.py", "loss_type: dft\n", tmp_path, init_training_script=_stop)


def test_self_distill_checks_a_separate_reference_tokenizer_before_any_load(tmp_path):
    """Only tokenizer files are read, so a mismatched anchor repo is refused before the dataset and
    both model loads."""
    parallelism = SimpleNamespace(has_ep_group=False, is_tp_mode=False)
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
    parallelism = SimpleNamespace(has_ep_group=False, is_tp_mode=False)
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


# Offline GRPO: at the parse.


@pytest.mark.parametrize(
    "key",
    [
        # Only masked positions ever held it; ``pad_token`` overrides the tokenizer's pad.
        "padding_value",
        # The budget is the two caps; pipeline parallelism pads to their sum.
        "max_length",
    ],
)
def test_offline_grpo_has_no_such_key(key, tmp_path):
    with pytest.raises(ValueError, match=key):
        _run_main("grpo/offline.py", f"{key}: 64\n", tmp_path, init_training_script=_stop)


# Environmental GRPO: before init_training_script.


def test_env_grpo_refuses_a_completion_length_it_would_overwrite(tmp_path):
    with pytest.raises(ValueError, match=r"max_completion_length=1024 is not a knob.*rollout_max_tokens \(512\)"):
        _run_main(
            "grpo/environmental.py",
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
        _run_main("grpo/environmental.py", config_body, tmp_path, init_training_script=_stop)


def test_env_grpo_refuses_a_react_system_prompt_before_the_model_load(tmp_path):
    """The registry refuses the key; the script must reach that refusal before any load."""
    with pytest.raises(ValueError, match="environment_kwargs.system_prompt"):
        _run_main(
            "grpo/environmental.py",
            "environment_type: react_math\nenvironment_kwargs:\n  system_prompt: Answer tersely.\n",
            tmp_path,
            init_training_script=_stop,
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
