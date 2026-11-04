"""Actual DPO entrypoints over tiny local models; CPU runs the native head tiles eagerly."""

from collections import defaultdict
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import Gemma2Config, Gemma2ForCausalLM, PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM
from trl import DPOConfig
from trl.models.utils import _ForwardRedirection
from trl.trainer.dpo_trainer import DataCollatorForPreference

import halo.trainers.preference.chunked_dpo as chunked_dpo
from halo.distributed.parallelism_config import ParallelismConfig
from halo.trainers.preference.chunked_dpo import CHUNKED_DPO_SCORER, DPO_SCORER_KEY, reject_chunked_dpo_options
from halo.trainers.preference.dpo import DistributedDPOTrainer
from tests.common.preference_precompute import column, precompute_trainer, token_rows


def _model(family="qwen3"):
    torch.manual_seed(31)
    kwargs = {
        "vocab_size": 32,
        "hidden_size": 16,
        "intermediate_size": 32,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "max_position_embeddings": 64,
        "attention_dropout": 0.0,
        "pad_token_id": 0,
        "eos_token_id": 1,
        "bos_token_id": 2,
        "tie_word_embeddings": False,
    }
    if family == "gemma2":
        config = Gemma2Config(**kwargs, final_logit_softcapping=0.7, attn_logit_softcapping=1.5)
        config._attn_implementation = "eager"
        return Gemma2ForCausalLM(config)
    config = Qwen3Config(**kwargs)
    config._attn_implementation = "eager"
    return Qwen3ForCausalLM(config)


def _rows():
    return [
        {"prompt_ids": [2, 4], "chosen_ids": [6, 7, 8, 1], "rejected_ids": [9, 1]},
        {"prompt_ids": [2, 5, 6], "chosen_ids": [10, 1], "rejected_ids": [11, 12, 13, 1]},
    ]


def _batch():
    return DataCollatorForPreference(pad_token_id=0)(_rows())


def _loss_trainer(model, *, bounded):
    trainer = precompute_trainer("dpo")
    del trainer.compute_ref_log_probs
    trainer.use_chunked_dpo_logprobs = bounded
    trainer._dpo_head_transforms = {}
    trainer._dpo_scored_datasets = {}
    trainer._dpo_forward_redirection = _ForwardRedirection()
    trainer.model = model
    trainer.ref_model = None
    trainer.accelerator.unwrap_model = lambda wrapped: wrapped
    trainer.accelerator.autocast = nullcontext
    trainer.accelerator.gather = lambda values: values
    trainer.beta = 0.7
    trainer.precompute_ref_logps = True
    trainer.loss_types = ["sigmoid"]
    trainer.loss_weights = [1.7]
    trainer.use_weighting = False
    trainer.f_divergence_type = "reverse_kl"
    trainer._metrics = {"train": defaultdict(list), "eval": defaultdict(list)}
    trainer.aux_loss_enabled = False
    trainer._tp_size = 1
    trainer._total_train_tokens = 0
    return trainer


def _reference_batch(trainer, model):
    batch = _batch()
    chosen, rejected = trainer.compute_ref_log_probs(model, batch)
    assert not chosen.requires_grad and not rejected.requires_grad
    assert chosen.dtype == rejected.dtype == torch.float32
    batch.update(
        ref_chosen_logps=chosen - torch.tensor([0.3, -0.8]), ref_rejected_logps=rejected + torch.tensor([0.4, 0.1])
    )
    return batch


@pytest.mark.parametrize("family", ["qwen3", "gemma2"])
def test_actual_loss_gradients_and_three_updates_match_full_logits(family):
    policy, oracle = _model(family), _model(family)
    bounded, full = _loss_trainer(policy, bounded=True), _loss_trainer(oracle, bounded=False)
    batch = _reference_batch(full, oracle)
    optimizers = [torch.optim.AdamW(model.parameters(), lr=0.002) for model in (policy, oracle)]
    initial = policy.lm_head.weight.detach().clone()
    for _ in range(3):
        for optimizer in optimizers:
            optimizer.zero_grad()
        losses = [trainer.compute_loss(model, batch) for trainer, model in ((bounded, policy), (full, oracle))]
        torch.testing.assert_close(*losses, rtol=2e-5, atol=2e-6)
        for loss in losses:
            loss.backward()
        for (name, parameter), (other_name, expected) in zip(
            policy.named_parameters(), oracle.named_parameters(), strict=True
        ):
            assert name == other_name
            assert parameter.grad is not None, name
            torch.testing.assert_close(parameter.grad, expected.grad, rtol=2e-4, atol=2e-5, msg=name)
        for optimizer in optimizers:
            optimizer.step()
        for parameter, expected in zip(policy.parameters(), oracle.parameters(), strict=True):
            torch.testing.assert_close(parameter, expected, rtol=2e-4, atol=2e-5)
    assert (policy.lm_head.weight - initial).abs().max() > 0.001
    for metric in ("entropy", "num_tokens", "logps/chosen", "logps/rejected", "rewards/margins", "rewards/accuracies"):
        torch.testing.assert_close(
            torch.tensor(bounded._metrics["train"][metric]),
            torch.tensor(full._metrics["train"][metric]),
            rtol=2e-5,
            atol=2e-5,
        )
    assert not any(name.startswith("logits/") for name in bounded._metrics["train"])


def test_no_grad_reference_sweep_matches_full_logits_and_stores_fp32():
    model = _model()
    bounded, full = (_loss_trainer(model, bounded=mode) for mode in (True, False))
    for trainer in (bounded, full):
        trainer.data_collator = DataCollatorForPreference(pad_token_id=0)
    swept = bounded._precompute_ref_logps(Dataset.from_list(_rows()), "train", 1)
    expected = full.compute_ref_log_probs(model, _batch())
    for name, want in zip(bounded.REFERENCE_COLUMNS, expected, strict=True):
        assert swept.features[name].dtype == "float32"
        torch.testing.assert_close(column(swept, name), want, rtol=2e-5, atol=2e-6)
    assert bounded._reference_logps_by_split["train"]["settings"][DPO_SCORER_KEY] == CHUNKED_DPO_SCORER
    assert bounded._precompute_ref_logps(swept, "train", 1) is swept
    assert all(parameter.grad is None for parameter in model.parameters())


def test_bf16_storage_matches_fp32_head_on_rounded_operands(monkeypatch):
    policy, oracle = _model().bfloat16(), _model().bfloat16()
    bounded, full = _loss_trainer(policy, bounded=True), _loss_trainer(oracle, bounded=False)
    monkeypatch.setattr(
        oracle.lm_head, "forward", lambda hidden: F.linear(hidden.float(), oracle.lm_head.weight.float())
    )
    batch = _reference_batch(full, oracle)
    tokens, entropy = bounded._chunked_dpo_logps(policy, batch, compute_entropy=True)
    logits = oracle(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False).logits[
        :, :-1
    ]
    log_softmax = logits.log_softmax(-1)
    mask = batch["completion_mask"][:, 1:].bool()
    expected = log_softmax.gather(-1, batch["input_ids"][:, 1:, None]).squeeze(-1).masked_fill(~mask, 0.0)
    torch.testing.assert_close(tokens, expected, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(
        tokens.sum(1, dtype=torch.float64), expected.sum(1, dtype=torch.float64), rtol=2e-5, atol=2e-6
    )
    torch.testing.assert_close(entropy[mask], -(log_softmax.exp() * log_softmax).sum(-1)[mask], rtol=2e-5, atol=2e-6)
    losses = [trainer.compute_loss(model, batch) for trainer, model in ((bounded, policy), (full, oracle))]
    torch.testing.assert_close(*losses, rtol=2e-5, atol=2e-6)
    for loss in losses:
        loss.backward()
    for (name, parameter), expected_parameter in zip(policy.named_parameters(), oracle.parameters(), strict=True):
        error = (parameter.grad.float() - expected_parameter.grad.float()).norm()
        scale = expected_parameter.grad.float().norm().clamp_min(1e-12)
        assert error / scale < 5e-3, name


def test_frozen_head_and_no_full_logits_forward(monkeypatch):
    model = _model()
    model.lm_head.weight.requires_grad_(False)
    trainer = _loss_trainer(model, bounded=True)

    def full_head(*args, **kwargs):
        raise AssertionError("bounded scoring called the vocabulary-wide head forward")

    monkeypatch.setattr(model.lm_head, "forward", full_head)
    batch = _reference_batch(trainer, model)
    trainer.compute_loss(model, batch).backward()
    assert model.lm_head.weight.grad is None
    assert model.model.embed_tokens.weight.grad.abs().max() > 1e-5


def test_explicit_reference_scores_its_own_weights_not_the_policy():
    policy, reference = _model(), _model()
    with torch.no_grad():
        reference.lm_head.weight.add_(torch.randn_like(reference.lm_head.weight) * 0.2)
    bounded = _loss_trainer(policy, bounded=True)
    full = _loss_trainer(reference, bounded=False)
    actual = bounded.compute_ref_log_probs(reference, _batch())
    expected = full.compute_ref_log_probs(reference, _batch())
    live = bounded.compute_ref_log_probs(policy, _batch())
    for got, want, wrong in zip(actual, expected, live, strict=True):
        torch.testing.assert_close(got, want, rtol=2e-5, atol=2e-6)
        assert (got - wrong).abs().max() > 0.01
    assert all(parameter.grad is None for model in (policy, reference) for parameter in model.parameters())


@pytest.mark.parametrize("empty", [True, False], ids=["empty", "truncated"])
def test_zero_target_batch_keeps_connected_zero_gradients(empty):
    model = _model()
    trainer = _loss_trainer(model, bounded=True)
    rows = [{"prompt_ids": [2, 4, 5], "chosen_ids": [] if empty else [6, 7], "rejected_ids": [] if empty else [8]}]
    batch = DataCollatorForPreference(pad_token_id=0, max_length=None if empty else 2)(rows)
    assert not batch["completion_mask"].any()
    batch.update(ref_chosen_logps=torch.zeros(1), ref_rejected_logps=torch.zeros(1))
    loss = trainer.compute_loss(model, batch)
    assert loss.requires_grad and torch.isfinite(loss)
    loss.backward()
    for name, parameter in model.named_parameters():
        assert parameter.grad is not None, name
        assert parameter.grad.count_nonzero() == 0, name
    assert trainer._metrics["train"]["entropy"] == [0.0]


def test_head_is_read_inside_wrapped_root_and_each_backbone_gets_fresh_scope(monkeypatch):
    model = _model()
    trainer = _loss_trainer(model, bounded=True)
    state = {"active": False, "roots": 0, "heads": 0, "generations": 0}

    class Root(torch.nn.Module):
        def forward(self, *args, **kwargs):
            assert all(not isinstance(arg, dict) for arg in args), "FP32 references must not enter root input casting"
            state["active"] = True
            state["roots"] += 1
            try:
                return model(*args, **kwargs)
            finally:
                state["active"] = False

    materialize = chunked_dpo.materialize_dtensor

    def read_head(parameter):
        assert state["active"], "head read outlived root FSDP unshard"
        state["heads"] += 1
        return materialize(parameter)

    def generation():
        state["generations"] += 1

    monkeypatch.setattr(chunked_dpo, "materialize_dtensor", read_head)
    monkeypatch.setattr(chunked_dpo, "bump_forward_generation", generation)
    root = Root()
    trainer.accelerator.unwrap_model = lambda wrapped: model
    batch = _reference_batch(trainer, root)
    trainer.compute_loss(root, batch).backward()
    assert state == {"active": False, "roots": 2, "heads": 4, "generations": 2}


def test_redirection_restores_forward_if_root_hook_fails():
    model = _model()
    trainer = _loss_trainer(model, bounded=True)
    original = model.forward

    def fail(*args):
        raise RuntimeError("root unshard failure")

    hook = model.register_forward_pre_hook(fail)
    with pytest.raises(RuntimeError, match="root unshard failure"):
        trainer.compute_ref_log_probs(model, _batch())
    hook.remove()
    assert model.forward == original
    assert torch.isfinite(model(**{k: v for k, v in _batch().items() if k != "completion_mask"}).logits).all()


@pytest.mark.parametrize("mutation", ["mask", "shift", "pairs", "transform"])
def test_wrong_scoring_changes_loss_and_parameter_gradients(mutation, monkeypatch):
    policy = _model("gemma2")
    trainer = _loss_trainer(policy, bounded=True)
    batch = _reference_batch(trainer, policy)
    correct = trainer.compute_loss(policy, batch)
    correct_gradient = torch.autograd.grad(correct, policy.lm_head.weight)[0]
    wrong = {name: value.clone() for name, value in batch.items()}
    if mutation == "mask":
        wrong["completion_mask"][0, 1] = 1
    elif mutation == "shift":
        wrong["completion_mask"] = wrong["completion_mask"].roll(1, dims=1)
    elif mutation == "pairs":
        wrong["ref_rejected_logps"] = wrong["ref_rejected_logps"].flip(0)
    else:
        transform = chunked_dpo.resolve_head_transform(policy)
        monkeypatch.setitem(trainer._dpo_head_transforms, id(policy), type(transform)())
    changed = trainer.compute_loss(policy, wrong)
    changed_gradient = torch.autograd.grad(changed, policy.lm_head.weight)[0]
    assert (changed - correct).abs().item() > 1e-5
    assert (changed_gradient - correct_gradient).abs().max().item() > 1e-5


@pytest.mark.parametrize("saved_bounded", [False, True], ids=["full-to-bounded", "bounded-to-full"])
def test_scorer_mismatch_is_refused_on_policy_from_checkpoint(tmp_path, saved_bounded):
    trainer = precompute_trainer("dpo")
    trainer.use_chunked_dpo_logprobs = saved_bounded
    trainer._dpo_scored_datasets = {}
    trainer._precompute_ref_logps(token_rows("dpo"), "train", 2)
    trainer._persist_trainer_sidecars(str(tmp_path))
    resumed = precompute_trainer("dpo", resume_checkpoint=str(tmp_path), policy_from_checkpoint=True)
    resumed.use_chunked_dpo_logprobs = not saved_bounded
    resumed._dpo_scored_datasets = {}
    with pytest.raises(ValueError, match="scores must not be mixed on resume"):
        resumed._precompute_ref_logps(token_rows("dpo"), "train", 2)
    assert resumed.compute_ref_log_probs.batches == 0


def test_same_scorer_restores_bit_exact_without_sweep(tmp_path):
    trainer = precompute_trainer("dpo")
    trainer.use_chunked_dpo_logprobs = True
    trainer._dpo_scored_datasets = {}
    expected = trainer._precompute_ref_logps(token_rows("dpo"), "train", 2)
    trainer._persist_trainer_sidecars(str(tmp_path))
    resumed = precompute_trainer("dpo", resume_checkpoint=str(tmp_path), policy_from_checkpoint=True)
    resumed.use_chunked_dpo_logprobs = True
    resumed._dpo_scored_datasets = {}
    restored = resumed._precompute_ref_logps(token_rows("dpo"), "train", 2)
    assert restored.to_dict() == expected.to_dict()
    assert resumed.compute_ref_log_probs.batches == 0
    assert resumed._precompute_ref_logps(restored, "train", 2) is restored


def test_supplied_references_are_refused_only_when_bounded():
    rows = token_rows("dpo").add_column("ref_chosen_logps", [0.0] * 4).add_column("ref_rejected_logps", [0.0] * 4)
    trainer = precompute_trainer("dpo")
    assert trainer._precompute_ref_logps(rows, "train", 2) is rows
    trainer.use_chunked_dpo_logprobs = True
    trainer._dpo_scored_datasets = {}
    with pytest.raises(ValueError, match="without scorer provenance"):
        trainer._precompute_ref_logps(rows, "train", 2)
    assert DPO_SCORER_KEY not in precompute_trainer("kto")._reference_settings()


def _args(**changes):
    values = {
        "loss_type": ["sigmoid"],
        "f_divergence_type": "reverse_kl",
        "use_weighting": False,
        "ld_alpha": None,
        "label_smoothing": 0.0,
        "padding_free": False,
        "precompute_ref_log_probs": True,
        "sync_ref_model": False,
    }
    return SimpleNamespace(**(values | changes))


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"loss_type": ["hinge"]}, "sigmoid"),
        ({"f_divergence_type": "js_divergence"}, "reverse_kl"),
        ({"use_weighting": True}, "weighting"),
        ({"ld_alpha": 0.5}, "ld_alpha"),
        ({"label_smoothing": 0.1}, "label_smoothing"),
        ({"padding_free": True}, "padded"),
        ({"precompute_ref_log_probs": False}, "precompute"),
        ({"sync_ref_model": True}, "sync_ref_model"),
        ({"deepspeed": "config.json"}, "FSDP2"),
        ({"router_aux_loss_coef": 0.1}, "aux_loss"),
    ],
)
def test_unsupported_objectives_fail_before_trainer_setup(changes, match):
    with pytest.raises(ValueError, match=match):
        reject_chunked_dpo_options(
            _model(), _args(**changes), SimpleNamespace(tp_size=1, expert_tp_size=1, cp_size=1, pp_size=1, ep_size=1)
        )


@pytest.mark.parametrize("axis", ["tp_size", "expert_tp_size", "cp_size", "pp_size"])
def test_unsupported_axes_fail_before_scoring(axis):
    axes = {"tp_size": 1, "expert_tp_size": 1, "cp_size": 1, "pp_size": 1, "ep_size": 1}
    axes[axis] = 2
    with pytest.raises(ValueError, match="TP, ETP, CP or PP"):
        reject_chunked_dpo_options(_model(), _args(), SimpleNamespace(**axes))


@pytest.mark.parametrize("option,match", [("peft_config", "full fine-tuning"), ("compute_metrics", "raw logits")])
def test_adapters_and_raw_logit_metrics_are_refused(option, match):
    kwargs = {option: LoraConfig() if option == "peft_config" else lambda predictions: {}}
    with pytest.raises(ValueError, match=match):
        reject_chunked_dpo_options(
            _model(), _args(), SimpleNamespace(tp_size=1, expert_tp_size=1, cp_size=1, pp_size=1, ep_size=1), **kwargs
        )


@pytest.mark.parametrize(
    "kind,match", [("quantized", "quantized"), ("adapted_head", "plain Linear"), ("launcher", "FSDP2")]
)
def test_unqualified_loading_paths_are_refused(kind, match, monkeypatch):
    model = _model()
    if kind == "quantized":
        model.is_loaded_in_4bit = True
    elif kind == "adapted_head":
        model.lm_head.forward = lambda hidden: F.linear(hidden, model.lm_head.weight) + 0.1
    else:
        monkeypatch.setattr(chunked_dpo, "is_accelerate_launch", lambda: True)
    with pytest.raises(ValueError, match=match):
        reject_chunked_dpo_options(
            model, _args(), SimpleNamespace(tp_size=1, expert_tp_size=1, cp_size=1, pp_size=1, ep_size=1)
        )


def _tokenizer():
    backend = Tokenizer(
        WordLevel(
            {"[PAD]": 0, "[EOS]": 1, "[BOS]": 2, "[UNK]": 3, **{f"t{i}": i for i in range(4, 32)}}, unk_token="[UNK]"
        )
    )
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]", bos_token="[BOS]", unk_token="[UNK]"
    )


def test_real_trainer_constructs_precomputes_and_evaluates_loss_only(tmp_path):
    args = DPOConfig(
        output_dir=str(tmp_path),
        use_cpu=True,
        bf16=False,
        report_to=[],
        max_steps=2,
        gradient_checkpointing=False,
        precompute_ref_log_probs=True,
        max_length=16,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        dataloader_num_workers=0,
        dataset_num_proc=1,
        disable_tqdm=True,
        use_liger_kernel=False,
    )
    rows = Dataset.from_list(
        [
            {"prompt": "t4 t5", "chosen": "t6 t7 t8", "rejected": "t9 t10"},
            {"prompt": "t11", "chosen": "t12", "rejected": "t13 t14 t15"},
        ]
    )
    trainer = DistributedDPOTrainer(
        model=_model(),
        args=args,
        train_dataset=rows,
        eval_dataset=rows,
        processing_class=_tokenizer(),
        parallelism_config=ParallelismConfig(),
        use_chunked_dpo_logprobs=True,
    )
    assert not args.prediction_loss_only, "exercise evaluate()'s normal no-metrics loss-only dispatch"
    assert trainer.train_dataset.features["ref_chosen_logps"].dtype == "float32"
    metrics = trainer.evaluate()
    assert torch.isfinite(torch.tensor(metrics["eval_loss"]))
    batch = trainer.data_collator([trainer.train_dataset[i] for i in range(2)])
    with pytest.raises(ValueError, match="does not return raw logits"):
        trainer.compute_loss(trainer.model, batch, return_outputs=True)


def test_real_cpu_training_checkpoint_resume_and_export(tmp_path):
    rows = Dataset.from_list(
        [
            {"prompt": "t4 t5", "chosen": "t6 t7 t8", "rejected": "t9 t10"},
            {"prompt": "t11", "chosen": "t12", "rejected": "t13 t14 t15"},
        ]
    )

    def build(model, directory, **resume):
        args = DPOConfig(
            output_dir=str(directory),
            use_cpu=True,
            bf16=False,
            report_to=[],
            max_steps=2,
            learning_rate=0.002,
            logging_steps=1,
            gradient_checkpointing=False,
            precompute_ref_log_probs=True,
            max_length=16,
            per_device_train_batch_size=2,
            dataloader_num_workers=0,
            dataset_num_proc=1,
            disable_tqdm=True,
            use_liger_kernel=False,
            save_strategy="steps",
            save_steps=1,
        )
        return DistributedDPOTrainer(
            model=model,
            args=args,
            train_dataset=rows,
            processing_class=_tokenizer(),
            parallelism_config=ParallelismConfig(),
            use_chunked_dpo_logprobs=True,
            **resume,
        )

    trainer = build(_model(), tmp_path / "continuous")
    initial = trainer.model.lm_head.weight.detach().clone()
    assert trainer.train().global_step == 2
    assert (trainer.model.lm_head.weight - initial).abs().max() > 1e-4
    checkpoint = tmp_path / "continuous" / "checkpoint-1"
    assert (checkpoint / "reference_logps.pt").is_file()
    model = Qwen3ForCausalLM.from_pretrained(checkpoint, dtype=torch.float32, attn_implementation="eager")
    resumed = build(model, tmp_path / "resumed", resume_checkpoint=str(checkpoint), policy_from_checkpoint=True)
    assert resumed.train_dataset.to_dict() == trainer.train_dataset.to_dict()
    assert resumed.train(resume_from_checkpoint=str(checkpoint)).global_step == 2
    for actual, expected in zip(resumed.model.parameters(), trainer.model.parameters(), strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    exported = tmp_path / "export"
    resumed.save_model(str(exported))
    stock, info = Qwen3ForCausalLM.from_pretrained(
        exported, dtype=torch.float32, attn_implementation="eager", output_loading_info=True
    )
    assert not info["missing_keys"] and not info["unexpected_keys"] and not info["mismatched_keys"]
    inputs = {key: value for key, value in _batch().items() if key != "completion_mask"}
    with torch.no_grad():
        torch.testing.assert_close(stock(**inputs).logits, resumed.model(**inputs).logits, rtol=0, atol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
