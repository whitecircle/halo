"""Run-start sweep restrictions must not change inactive-KL or adapter behavior."""

from types import SimpleNamespace

import pytest
from accelerate import PartialState
from datasets import Dataset, IterableDataset
from peft import LoraConfig
from transformers import LlamaConfig, LlamaForCausalLM

import src.trainers.grpo.offline as offline_mod
from src.configs.offline_grpo_config import OfflineGRPOConfig
from src.data.collators.offline_grpo import REF_PER_TOKEN_LOGPS_COLUMN
from src.distributed.expert_parallel.config import ExpertLoraSpec
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.grpo.offline import OfflineGRPOTrainer, tokenize_offline_grpo_rows
from src.trainers.grpo.reference_lifecycle import reject_unsupported_reference_input
from tests.common.gloo import run_gloo_ranks
from tests.common.offline_grpo import make_offline_tokenizer, offline_grpo_dataset

PartialState(cpu=True)


def test_string_policy_with_kl_reaches_the_model_loader_without_a_peft_probe_crash(tmp_path, monkeypatch):
    calls = []

    def stop_at_loader(source, args, **kwargs):
        calls.append(source)
        raise RuntimeError("model loader reached")

    monkeypatch.setattr(offline_mod, "load_model_from_pretrained", stop_at_loader)
    args = OfflineGRPOConfig(
        output_dir=str(tmp_path), kl_beta=0.2, use_cpu=True, bf16=False, use_liger_kernel=False, report_to="none"
    )
    with pytest.raises(RuntimeError, match="model loader reached"):
        OfflineGRPOTrainer(
            model="local-or-hub-policy",
            args=args,
            train_dataset=offline_grpo_dataset(2),
            processing_class=make_offline_tokenizer(),
            parallelism_config=ParallelismConfig(),
        )
    assert calls == ["local-or-hub-policy"]


@pytest.mark.parametrize("presharded", [False, True])
def test_constructor_passes_presharded_ownership_to_the_active_reference_gate(tmp_path, monkeypatch, presharded):
    def stop_at_loader(*args, **kwargs):
        raise RuntimeError("model loader reached")

    monkeypatch.setattr(offline_mod, "load_model_from_pretrained", stop_at_loader)
    args = OfflineGRPOConfig(
        output_dir=str(tmp_path), kl_beta=0.2, use_cpu=True, bf16=False, use_liger_kernel=False, report_to="none"
    )
    error = ValueError if presharded else RuntimeError
    message = "pre-sharded dataset" if presharded else "model loader reached"
    with pytest.raises(error, match=message):
        OfflineGRPOTrainer(
            model="local-or-hub-policy",
            args=args,
            train_dataset=offline_grpo_dataset(2),
            processing_class=make_offline_tokenizer(),
            parallelism_config=ParallelismConfig(),
            dataset_presharded=presharded,
        )


def _policy():
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
    )


@pytest.mark.parametrize("mode,beta", [("plain", 0.0), ("peft", 0.0), ("peft", 0.2), ("expert", 0.2)])
def test_unused_supplied_reference_column_is_discarded_instead_of_refused(tmp_path, monkeypatch, mode, beta):
    args = OfflineGRPOConfig(
        output_dir=str(tmp_path),
        kl_beta=beta,
        use_cpu=True,
        bf16=False,
        use_liger_kernel=False,
        report_to="none",
        max_prompt_length=16,
        max_completion_length=16,
        dataset_num_proc=1,
        remove_unused_columns=False,
    )
    dataset = offline_grpo_dataset(2).add_column(REF_PER_TOKEN_LOGPS_COLUMN, ["unused", "unused"])
    parallelism = (
        ParallelismConfig(expert_lora=ExpertLoraSpec(r=2, alpha=4)) if mode == "expert" else ParallelismConfig()
    )
    monkeypatch.setattr(OfflineGRPOTrainer, "_setup_distributed_modes", lambda self: None)
    trainer = OfflineGRPOTrainer(
        model=_policy(),
        args=args,
        train_dataset=dataset,
        eval_dataset=dataset,
        processing_class=make_offline_tokenizer(),
        parallelism_config=parallelism,
        peft_config=LoraConfig(target_modules=["q_proj", "v_proj"]) if mode == "peft" else None,
        ref_model=_policy() if mode == "expert" else None,
    )
    assert not trainer._precompute_reference
    assert REF_PER_TOKEN_LOGPS_COLUMN not in trainer.train_dataset.column_names
    assert REF_PER_TOKEN_LOGPS_COLUMN not in trainer.eval_dataset.column_names


def test_grouped_tokenization_discards_unused_supplied_scores():
    batch = {
        "prompt": ["question alpha"],
        "completions": [["yes alpha", "no bravo"]],
        "rewards": [[1.0, -1.0]],
        REF_PER_TOKEN_LOGPS_COLUMN: ["unused"],
    }
    result = tokenize_offline_grpo_rows(
        batch,
        [0],
        processing_class=make_offline_tokenizer(),
        max_prompt_length=16,
        max_completion_length=16,
        advantage_method="z_norm",
        best_completion_emphasis=0.0,
        is_encoder_decoder=False,
    )
    assert len(result["completion_input_ids"]) == 2
    assert REF_PER_TOKEN_LOGPS_COLUMN not in result


@pytest.mark.parametrize("kind", ["named", "streaming", "schema-less"])
def test_active_guard_reports_unsupported_split_shapes_without_attribute_errors(kind):
    train = Dataset.from_dict({"prompt": ["question"]})
    if kind == "named":
        evaluation = {"heldout": train}
        pattern = "one finite evaluation Dataset"
    elif kind == "streaming":
        evaluation = IterableDataset.from_generator(lambda: iter([{"prompt": "question"}]))
        assert evaluation.column_names is None
        pattern = "finite datasets.Dataset evaluation"
    else:
        evaluation = SimpleNamespace(column_names=None)
        pattern = "finite datasets.Dataset evaluation"
    with pytest.raises(ValueError, match=pattern):
        reject_unsupported_reference_input(train, evaluation)
    reject_unsupported_reference_input(train, evaluation, active=False, presharded=True)


def test_active_guard_still_refuses_supplied_columns_and_pre_sharded_data():
    train = Dataset.from_dict({"prompt": ["question"]})
    supplied = train.add_column(REF_PER_TOKEN_LOGPS_COLUMN, [[-1.0]])
    for first, second in ((supplied, None), (train, supplied)):
        with pytest.raises(ValueError, match="Supplied ref_per_token_logps"):
            reject_unsupported_reference_input(first, second)
    with pytest.raises(ValueError, match="pre-sharded dataset"):
        reject_unsupported_reference_input(train, None, presharded=True)


def _ranked_active_input(rank, root):
    train = Dataset.from_dict({"prompt": ["question"]})
    evaluation = {"heldout": train} if rank else train
    with pytest.raises(ValueError, match="one finite evaluation Dataset"):
        reject_unsupported_reference_input(train, evaluation)


def test_a_single_rank_unsupported_split_is_refused_before_collective_scoring(tmp_path):
    run_gloo_ranks(_ranked_active_input, 2, str(tmp_path))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
