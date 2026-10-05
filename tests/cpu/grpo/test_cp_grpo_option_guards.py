"""Unsupported CP anchors and adapters fail before the policy loader runs."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from accelerate import PartialState
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from src.configs.offline_grpo_config import OfflineGRPOConfig
from src.trainers.grpo.offline import OfflineGRPOTrainer
from tests.common.offline_grpo import make_offline_tokenizer, offline_grpo_dataset

PartialState()


def _cp_config(*, expert_lora=None):
    return SimpleNamespace(is_cp_mode=True, is_pp_mode=False, expert_lora=expert_lora)


def _peft_policy():
    config = AutoConfig.for_model(
        "llama",
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=32,
    )
    return get_peft_model(
        AutoModelForCausalLM.from_config(config), LoraConfig(r=2, target_modules=["q_proj", "v_proj"])
    )


@pytest.mark.parametrize("option", ["peft-config", "peft-policy", "expert-lora", "ref-model"])
@pytest.mark.parametrize("kl_beta", [0.0, 0.2])
def test_cp_unsupported_options_fail_before_loading_weights(tmp_path, option, kl_beta):
    model = _peft_policy() if option == "peft-policy" else "unused/local-policy"
    peft_config = LoraConfig() if option == "peft-config" else None
    parallelism = _cp_config(expert_lora=object() if option == "expert-lora" else None)
    reference = object() if option == "ref-model" else None
    args = OfflineGRPOConfig(
        output_dir=str(tmp_path),
        kl_beta=kl_beta,
        use_cpu=True,
        bf16=False,
        use_liger_kernel=False,
        report_to="none",
    )
    expected = "Drop ref_model" if option == "ref-model" else "requires a full policy fine-tune"
    with patch("src.trainers.grpo.offline.load_model_from_pretrained") as loader:
        with pytest.raises(ValueError, match=expected):
            OfflineGRPOTrainer(
                model=model,
                ref_model=reference,
                peft_config=peft_config,
                args=args,
                parallelism_config=parallelism,
                train_dataset=offline_grpo_dataset(1),
                processing_class=make_offline_tokenizer(),
            )
        loader.assert_not_called()


def test_cp_does_not_hold_an_external_kl_reference():
    policy, reference = nn.Linear(1, 1), object()
    assert not OfflineGRPOTrainer._holds_kl_reference(policy, 0.2, _cp_config(), None)
    assert OfflineGRPOTrainer._kl_reference(policy, None, 0.2, _cp_config(), None) is None
    with pytest.raises(ValueError, match="never be read.*Drop ref_model"):
        OfflineGRPOTrainer._kl_reference(policy, reference, 0.2, _cp_config(), None)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
