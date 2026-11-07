#!/usr/bin/env python3
"""Pin the script-arg dataclass field surfaces: every field of ``CommonScriptArguments``, and for each subclass what
it declares or re-defaults. Moving fields or classes must not add, drop or re-default a field; order and metadata
are not pinned."""

import dataclasses
import importlib
import pkgutil
from unittest import mock

import pytest

import halo.configs
from halo import env
from halo.configs.script_arguments import CommonScriptArguments


def _script_argument_classes() -> dict[str, type]:
    """Every script-argument dataclass under ``halo/configs/``, by name, derived so a new one is pinned or fails."""
    found = {}
    for module_info in pkgutil.iter_modules(halo.configs.__path__):
        module = importlib.import_module(f"halo.configs.{module_info.name}")
        for name, obj in vars(module).items():
            if isinstance(obj, type) and obj.__module__ == module.__name__ and issubclass(obj, CommonScriptArguments):
                found[name] = obj
    return found


_CLASSES = _script_argument_classes()

# Do not regenerate casually: an intentional field change is edited here explicitly, with the diff called out in
# review.
_EXPECTED = {
    "ClassificationScriptArguments": {
        "text_field": "None",
    },
    "CommonScriptArguments": {
        "added_special_tokens": "None",
        "allow_optimizer_warm_restart": "False",
        "bos_token": "None",
        "chat_template": "None",
        "dataset": "'/path/to/dataset'",
        "dataset_ratio": "None",
        "gradient_checkpointing_plan": "'off'",
        "enable_efficiency_metrics": "False",
        "enable_moe_metrics": "True",
        "enable_torch_profiler": "False",
        "eos_token": "None",
        "force_chat_template": "False",
        "freeze_layers_patterns": "None",
        "log_decoded_samples": "False",
        "max_consecutive_nonfinite_steps": "3",
        "moe_balancing": "'auto'",
        "num_full_model_params": "None",
        "overwrite_output_dir": "False",
        "pad_token": "None",
        "profiler_active": "3",
        "profiler_output_dir": "factory:'<data_root>/profiling/torch'",
        "profiler_ranks": "'0'",
        "profiler_record_memory_snapshot": "False",
        "profiler_wait": "5",
        "profiler_warmup": "1",
        "project_name": "'default-project'",
        "report_mfu_diagnostics": "False",
        "reset_sinks": "True",
        "router_balancing_rate": "0.001",
        "save_completions": "True",
        "save_max_shard_size": "'5GB'",
        "save_sharded_ep": "False",
        "test_size": "None",
        "text_only_model": "False",
        "tokenizer_backend": "'hf'",
        "tools_field": "None",
        "train_sinks": "False",
        "unfreeze_layers_patterns": "None",
    },
    "DPOScriptArguments": {
        "freeze_vision_tower": "False",
        "max_pixels": "None",
        "min_pixels": "None",
        "generate_eval_examples": "True",
        "generation_max_prompt_length": "512",
        "images_field": "None",
        "num_eval_examples": "50",
    },
    "EmbeddingScriptArguments": {},
    "EnvironmentalGRPOScriptArguments": {
        "answer_field": "'answer'",
        "context_fields": "None",
        "max_prompt_length": "None",
        "prompt_field": "'prompt'",
    },
    "KTOScriptArguments": {
        "freeze_vision_tower": "False",
        "max_pixels": "None",
        "min_pixels": "None",
        "completion_field": "'completion'",
        "images_field": "None",
        "label_field": "'label'",
    },
    "OfflineGRPOScriptArguments": {
        "generate_eval_examples": "True",
        "num_eval_examples": "50",
    },
    "OnlineGRPOScriptArguments": {
        "answer_field": "'answer'",
        "balance_rank_tokens": "False",
        "balance_token_mass": "False",
        "drop_degenerate_groups": "False",
        "early_stop_entropy_band": "None",
        "early_stop_logratio_gap": "None",
        "early_stop_patience": "3",
        "log_phase_times": "False",
        "loss_forward_importance_weights": "False",
        "max_prompt_length": "None",
        "opd_positive_advantage_only": "True",
        "packed_microbatch_tokens": "None",
        "prompt_field": "'prompt'",
        "reasoning_effort": "None",
        "rewards": "factory:[{'source': 'accuracy'}]",
        "rlrr_correctness_clip": "True",
        "rlrr_correctness_threshold": "0.5",
        "rlrr_lambda": "2048.0",
        "rlrr_length_rerank": "True",
        "rlrr_mode": "'hrr'",
        "rlrr_std_normalize": "False",
        "rlrr_tau": "0.1",
        "rlrr_xi_neg": "-0.001",
        "rlrr_xi_pos": "0.001",
        "scale_rewards_std_floor": "0.0",
        "sdpg_beta_base": "1.0",
        "sdpg_beta_decay_steps": "0",
        "sdpg_beta_warmup_steps": "0",
        "sdpg_hint_template": "'\\n[Hint] The correct answer is: {answer}. Do NOT state that you were given the answer.\\n'",
        "sdpg_jsd_beta": "0.5",
        "sdpg_loss": "'reverse_kl'",
        "sdpg_temperature": "1.0",
        "system_prompt": "None",
        "use_chunked_grpo_logprobs": "False",
        "use_rlrr": "False",
        "use_sdpg": "False",
    },
    "RewardScriptArguments": {
        "freeze_vision_tower": "False",
        "max_pixels": "None",
        "min_pixels": "None",
        "images_field": "None",
    },
    "SFTScriptArguments": {
        "freeze_vision_tower": "False",
        "max_pixels": "None",
        "min_pixels": "None",
        "allow_cross_document_linear_attention": "False",
        "assistant_message_template": "None",
        "conversation_field": "'prompt'",
        "generate_eval_examples": "False",
        "images_field": "None",
        "interleaved_thinking": "False",
        "model_supports_system_role": "True",
        "mtp_head_source": "None",
        "mtp_loss_type": "'hard'",
        "mtp_loss_weight": "0.0",
        "mtp_min_acceptance_gain": "0.0",
        "mtp_train_scope": "'joint'",
        "num_eval_examples": "50",
        "system_prompt": "None",
        "train_on_completions_only": "True",
        "train_on_last_assistant_only": "False",
        "tree_full_attention": "None",
        "tree_packing": "False",
        "tree_share_markers": "factory:[]",
        "tree_share_requires": "None",
    },
    "SMPOScriptArguments": {
        "freeze_vision_tower": "False",
        "max_pixels": "None",
        "min_pixels": "None",
        "generate_eval_examples": "True",
        "num_eval_examples": "50",
    },
    "SelfDistillScriptArguments": {
        "freeze_vision_tower": "False",
        "max_pixels": "None",
        "min_pixels": "None",
        "assistant_message_template": "None",
        "confidence_field": "None",
        "confidence_power": "4.0",
        "confidence_weight_opd": "True",
        "conversation_field": "'prompt'",
        "images_field": "None",
        "interleaved_thinking": "False",
        "model_supports_system_role": "True",
        "opd_exclude_eos": "True",
        "privileged_solution_field": "'solution'",
        "reference_kl_coef": "0.0",
        "reference_kl_loss": "'unnormalized_kl'",
        "reference_model_name_or_path": "None",
        "sdpg_answer_field": "'answer'",
        "sdpg_beta_base": "1.0",
        "sdpg_beta_decay_steps": "0",
        "sdpg_beta_warmup_steps": "0",
        "sdpg_hint_template": "'\\n[Hint] The correct answer is: {answer}. Do NOT state that you were given the answer.\\n'",
        "sdpg_jsd_beta": "0.5",
        "sdpg_loss": "'reverse_kl'",
        "sdpg_temperature": "1.0",
        "system_prompt": "None",
        "train_on_completions_only": "True",
    },
    "TeacherDistillScriptArguments": {
        "freeze_vision_tower": "False",
        "max_pixels": "None",
        "min_pixels": "None",
        "assistant_message_template": "None",
        "conversation_field": "'messages'",
        "images_field": "None",
        "interleaved_thinking": "False",
        "model_supports_system_role": "True",
        "system_prompt": "None",
        "teacher_model": "None",
        "teacher_model_revision": "None",
        "train_on_completions_only": "True",
    },
}


def _field_defaults(cls):
    """Field name -> default repr. ``default_factory`` paths read ``env.HALO_DATA_ROOT`` at call time, so the pin
    sets that root for the call; rewriting the result cannot work, since a root of ``/`` prefixes every absolute
    default."""
    with mock.patch.object(env, "HALO_DATA_ROOT", "<data_root>"):
        out = {}
        for f in dataclasses.fields(cls):
            if f.default is not dataclasses.MISSING:
                out[f.name] = repr(f.default)
            elif f.default_factory is not dataclasses.MISSING:  # type: ignore[misc]
                out[f.name] = "factory:" + repr(f.default_factory())
            else:
                out[f.name] = "<required>"
        return out


def test_every_script_argument_class_is_pinned():
    assert set(_CLASSES) == set(_EXPECTED), (
        f"unpinned: {sorted(set(_CLASSES) - set(_EXPECTED))}; pinned but not found: "
        f"{sorted(set(_EXPECTED) - set(_CLASSES))}"
    )


def _own_field_defaults(cls) -> dict[str, str]:
    """:func:`_field_defaults` of ``cls`` past what it inherits unchanged from ``CommonScriptArguments``."""
    actual = _field_defaults(cls)
    if cls is CommonScriptArguments:
        return actual
    common = _field_defaults(CommonScriptArguments)
    return {name: default for name, default in actual.items() if common.get(name) != default}


@pytest.mark.parametrize("cls_name", sorted(_EXPECTED))
def test_fields_and_defaults_unchanged(cls_name):
    actual = _own_field_defaults(_CLASSES[cls_name])
    expected = _EXPECTED[cls_name]
    assert actual == expected, (
        f"{cls_name} field surface drifted.\n"
        f"missing: {sorted(set(expected) - set(actual))}\n"
        f"added: {sorted(set(actual) - set(expected))}\n"
        f"re-defaulted: { {k: (expected[k], actual[k]) for k in set(expected) & set(actual) if expected[k] != actual[k]} }"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
