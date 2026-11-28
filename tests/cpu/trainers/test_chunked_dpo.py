"""Actual DPO entrypoints over tiny local models; CPU runs the native head tiles eagerly."""

import ast
import inspect
from collections import defaultdict
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn.functional as F
from datasets import Dataset
from peft import LoraConfig
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    Gemma2Config,
    Gemma2ForCausalLM,
    PreTrainedTokenizerFast,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoeConfig,
)
from trl import DPOConfig
from trl.models.utils import _ForwardRedirection
from trl.trainer.dpo_trainer import DataCollatorForPreference

import halo.trainers.preference.chunked_dpo as chunked_dpo
import tests.common.dpo_scoring as dpo_scoring
import tests.common.preference_precompute_e2e as precompute_e2e
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.load_rules import reject_unsupported_model, reject_unsupported_run
from halo.models.attention.gpt_oss_sinks import SinksPolicy
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from halo.trainers.parallelism.grad_sync import GradientSyncMixin
from halo.trainers.preference.chunked_dpo import CHUNKED_DPO_SCORER, DPO_SCORER_KEY, reject_chunked_dpo_options
from halo.trainers.preference.dpo import DistributedDPOTrainer
from tests.common.dpo_head_gradients import PinnedHeadGradients, full_head_derivative
from tests.common.dpo_scoring import (
    assert_dpo_gradients_match,
    assert_dpo_loss_matches,
    assert_dpo_updates_match,
    assert_dpo_weight_steps_match,
    assert_sgd_deltas_match,
    assert_sgd_update,
    same_state_dpo_oracle,
)
from tests.common.gloo import run_gloo_ranks
from tests.common.parallelism import make_parallelism_config
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


def test_distributed_scoring_fixture_pins_scored_ids_and_masks():
    batch = dpo_scoring._batch(0, "cpu")
    dpo_scoring._assert_fixture_tokens(batch)
    altered = {name: value.clone() for name, value in batch.items()}
    # A second terminator on a padded row can fool two scorers reading the same wrong batch.
    altered["input_ids"][2, 4] = 1
    altered["attention_mask"][2, 4] = 1
    altered["completion_mask"][2, 4] = 1
    with pytest.raises(AssertionError, match="completion mask changed"):
        dpo_scoring._assert_fixture_tokens(altered)


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


def _configure_cpu_backward(trainer):
    trainer.state = SimpleNamespace(global_step=1)
    trainer.optimizer = torch.optim.SGD(trainer.model.parameters(), lr=0.05)
    trainer.compute_loss_context_manager = nullcontext
    trainer._finish_fsdp2_backward = lambda model: None
    trainer._get_sharded_expert_param_ids = lambda: set()
    trainer.accelerator.backward = lambda loss: loss.backward()
    trainer._get_grad_norm = lambda model: torch.cat(
        [parameter.grad.detach().reshape(-1) for parameter in model.parameters() if parameter.grad is not None]
    ).norm()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_pinned_full_head_derivative_separates_logits_gradient_rounding(dtype):
    generator = torch.Generator().manual_seed(102)
    hidden = torch.randn(7, 16, generator=generator).to(dtype)
    weight = torch.randn(32, 16, generator=generator).to(dtype)
    targets = torch.arange(7) + 3
    upstream = torch.linspace(-0.2312345, 0.5432123, 7)
    exact = full_head_derivative(hidden, weight, targets, upstream)
    rounded = full_head_derivative(hidden, weight, targets, upstream, quantize_logits_gradient=True)
    logits = F.linear(hidden.float(), weight.float())
    derivative = (F.one_hot(targets, weight.shape[0]) - logits.softmax(-1)) * upstream[:, None]
    for output, dz in ((exact, derivative), (rounded, derivative.to(dtype).float())):
        torch.testing.assert_close(output["hidden"], (dz @ weight.float()).to(dtype), atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(output["weight"], (dz.T @ hidden.float()).to(dtype), atol=2e-6, rtol=2e-5)
    assert torch.equal(exact["logps"], rounded["logps"])
    assert torch.equal(exact["hidden"], rounded["hidden"]) == (dtype == torch.float32)


def test_bf16_matched_head_keeps_fp32_master_gradient_and_rejects_wrong_rounding():
    generator = torch.Generator().manual_seed(102)
    hidden = torch.randn(7, 16, generator=generator).to(torch.bfloat16).requires_grad_()
    head = torch.nn.Linear(16, 32, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.randn(head.weight.shape, generator=generator))
    targets = torch.arange(7) + 3
    upstream = torch.linspace(-0.2312345, 0.5432123, 7)
    logits = dpo_scoring._fp32_head_logits(head, hidden, backward_dtype=torch.bfloat16)
    logps = logits.log_softmax(-1).gather(1, targets[:, None]).squeeze(1)
    got_hidden, got_weight = torch.autograd.grad(logps, (hidden, head.weight), upstream)
    matched = full_head_derivative(hidden, head.weight, targets, upstream, quantize_logits_gradient=True)
    torch.testing.assert_close(logps, matched["logps"], rtol=0, atol=0)
    torch.testing.assert_close(got_hidden, matched["hidden"], rtol=0, atol=0)
    torch.testing.assert_close(got_weight, matched["weight"], rtol=0, atol=0)
    for wrong in (
        full_head_derivative(hidden, head.weight, targets, upstream, quantize_logits_gradient=torch.float16),
        full_head_derivative(hidden, head.weight, targets, upstream, round_forward_logits=True),
    ):
        relative = (got_weight - wrong["weight"]).norm() / got_weight.norm()
        assert relative > dpo_scoring.FP32_GRAD_REL_L2


def test_bf16_same_state_control_leaves_independent_trajectory_intact_on_cpu():
    bounded = _loss_trainer(_model().to(torch.bfloat16), bounded=True)
    full = _loss_trainer(_model().to(torch.bfloat16), bounded=False)
    for trainer in (bounded, full):
        _configure_cpu_backward(trainer)
    batch = _reference_batch(full, full.model)
    got, relative, independent = dpo_scoring._compare_backward(bounded, full, batch, family="dense", bf16=True)
    assert got.isfinite() and independent.isfinite() and relative < dpo_scoring.FP32_GRAD_REL_L2
    assert full.state.global_step == 1 and all(parameter.grad is not None for parameter in full.model.parameters())


def test_pinned_head_diagnosis_observes_real_loss_without_replacing_gradients():
    policy, oracle = _model(), _model()
    bounded, full = _loss_trainer(policy, bounded=True), _loss_trainer(oracle, bounded=False)
    batch = _reference_batch(full, oracle)
    original_forward = oracle.lm_head.forward
    with PinnedHeadGradients().record(full) as diagnosis:
        losses = [trainer.compute_loss(model, batch) for trainer, model in ((bounded, policy), (full, oracle))]
        for loss in losses:
            loss.backward()
        report = diagnosis.report({"head": policy.lm_head.weight.grad}, {"head": oracle.lm_head.weight.grad})
    assert oracle.lm_head.forward == original_forward and not diagnosis.hooks
    assert report["qualification"] == "DIAGNOSTIC_NOT_ACCEPTANCE"
    assert report["identical_supervised_hidden"] and report["identical_live_weight"]
    assert report["native_hidden_vs_fp32_derivative"]["relative_l2"] < 1e-5
    assert report["full_logits_gradient_vs_pinned_upstream"]["relative_l2"] < 1e-5
    assert report["full_hidden_vs_fp32_derivative"]["relative_l2"] < 1e-5


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


def test_bounded_warmup_covers_reference_and_training_and_clears_grads(monkeypatch):
    trainer = _loss_trainer(_model(), bounded=True)
    trainer.model.eval()
    trainer.compute_loss_context_manager = nullcontext
    events = []

    def score(model, batch, *, compute_entropy):
        events.append((compute_entropy, torch.is_grad_enabled()))
        assert batch["input_ids"].shape == (2, 3)
        return model.lm_head.weight.sum().expand(2, 2), None

    trainer._chunked_dpo_logps = score
    trainer._finish_fsdp2_backward = lambda model: events.append("backward finished")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        DistributedTrainerMixin, "_configure_distributed_modes", lambda self: events.append("configured")
    )
    monkeypatch.setattr(chunked_dpo, "barrier", lambda: events.append("barrier"))
    trainer._configure_distributed_modes()
    assert events == ["configured", (False, False), (True, True), "backward finished", "barrier"]
    assert not trainer.model.training
    assert all(parameter.grad is None for parameter in trainer.model.parameters())


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


def test_gradient_checkpointing_loss_and_all_gradients_match_full_head():
    policy, oracle = _model(), _model()
    for model in (policy, oracle):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    bounded, full = _loss_trainer(policy, bounded=True), _loss_trainer(oracle, bounded=False)
    for trainer in (bounded, full):
        trainer.args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    batch = _reference_batch(full, oracle)
    assert policy.is_gradient_checkpointing and oracle.is_gradient_checkpointing
    got, want = bounded.compute_loss(policy, batch), full.compute_loss(oracle, batch)
    torch.testing.assert_close(got, want, rtol=2e-5, atol=2e-6)
    got.backward()
    want.backward()
    assert_dpo_gradients_match(
        {name: parameter.grad for name, parameter in policy.named_parameters()},
        {name: parameter.grad for name, parameter in oracle.named_parameters()},
        bf16=False,
    )


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
    held_metrics = trainer.evaluate(eval_dataset=trainer.eval_dataset)
    torch.testing.assert_close(
        torch.tensor(held_metrics["eval_loss"]), torch.tensor(metrics["eval_loss"]), rtol=0, atol=0
    )
    with pytest.raises(ValueError, match="without scorer provenance"):
        trainer.evaluate(eval_dataset=trainer.eval_dataset.select(range(2)))
    trainer.state.global_step = 1
    held_metrics = trainer.evaluate(eval_dataset=trainer.eval_dataset)
    torch.testing.assert_close(
        torch.tensor(held_metrics["eval_loss"]), torch.tensor(metrics["eval_loss"]), rtol=0, atol=0
    )
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


@pytest.mark.parametrize("bf16", [False, True])
def test_distributed_gradient_comparison_rejects_missing_scaled_and_nonfinite_shards(bf16):
    expected = {"head.weight": torch.arange(1, 17).float().reshape(4, 4), "norm.weight": torch.zeros(4)}
    assert assert_dpo_gradients_match(expected, expected, bf16=bf16) == 0.0
    with pytest.raises(AssertionError, match="parameter set"):
        assert_dpo_gradients_match({"head.weight": expected["head.weight"]}, expected, bf16=bf16)
    with pytest.raises(AssertionError, match="Non-finite or wrong-shaped"):
        assert_dpo_gradients_match(expected | {"head.weight": None}, expected, bf16=bf16)
    with pytest.raises(AssertionError, match="relative L2"):
        assert_dpo_gradients_match(expected | {"head.weight": expected["head.weight"] * 2}, expected, bf16=bf16)
    with pytest.raises(AssertionError, match="Expected-zero"):
        assert_dpo_gradients_match(expected | {"norm.weight": torch.ones(4)}, expected, bf16=bf16)
    with pytest.raises(AssertionError, match="Non-finite"):
        assert_dpo_gradients_match(
            expected | {"head.weight": expected["head.weight"] * float("nan")}, expected, bf16=bf16
        )


def test_fp32_distributed_gradient_comparison_rejects_diluted_sparse_error():
    expected = {"head.weight": torch.ones(4096)}
    changed = expected["head.weight"].clone()
    changed[0] += 0.001
    assert (changed - expected["head.weight"]).norm() / expected["head.weight"].norm() < 5e-4
    with pytest.raises(AssertionError):
        assert_dpo_gradients_match({"head.weight": changed}, expected, bf16=False)


@pytest.mark.parametrize("family,limit", [("dense", 2.2e-2), ("qwen3_moe", 1.9e-2)])
def test_bf16_whole_model_gradient_comparison_pins_each_family_boundary(family, limit):
    expected = {"backbone.weight": torch.ones(8, dtype=torch.float64)}
    inside = {"backbone.weight": expected["backbone.weight"] * (1 + limit - 1e-4)}
    outside = {"backbone.weight": expected["backbone.weight"] * (1 + limit + 1e-4)}
    assert dpo_scoring.resolve_dpo_gradient_tolerance(family=family, bf16=True) == limit
    assert assert_dpo_gradients_match(inside, expected, bf16=True, family=family) == pytest.approx(limit - 1e-4)
    with pytest.raises(AssertionError, match="relative L2"):
        assert_dpo_gradients_match(outside, expected, bf16=True, family=family)
    with pytest.raises(AssertionError, match="relative L2"):
        assert_dpo_gradients_match(inside, expected, bf16=False, family=family)


def test_bf16_family_bounds_do_not_relax_fp32_comparison():
    assert dpo_scoring.FP32_GRAD_REL_L2 == 5e-4
    for family in ("dense", "qwen3_moe"):
        assert dpo_scoring.resolve_dpo_gradient_tolerance(family=family, bf16=False) == 5e-4
    with pytest.raises(ValueError, match="dense or qwen3_moe"):
        dpo_scoring.resolve_dpo_gradient_tolerance(family="unknown", bf16=True)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_weight_step_bound_keeps_prior_divergence_on_the_second_update(dtype):
    """The second step must inherit the first step's error rather than spend a fresh absolute tolerance."""
    before = torch.tensor([0.25, -0.75], dtype=dtype)
    other = before.clone()
    gradients = (torch.tensor([0.3, -0.1], dtype=dtype), torch.tensor([0.31, -0.11], dtype=dtype))
    for _ in range(2):
        after = before.clone().add_(gradients[0], alpha=-0.05)
        other_after = other.clone().add_(gradients[1], alpha=-0.05)
        assert_sgd_update(before, gradients[0], after, 0.05)
        assert_sgd_update(other, gradients[1], other_after, 0.05)
        assert_dpo_weight_steps_match((before, gradients[0], after), (other, gradients[1], other_after), 0.05)
        before, other = after, other_after
    assert not torch.equal(before, other)
    unchanged = torch.zeros_like(before)
    assert_dpo_weight_steps_match((before, unchanged, before), (other, unchanged, other), 0.05)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("sign", [-1, 1])
def test_weight_step_bound_uses_both_spacing_directions_at_a_power_of_two(dtype, sign):
    """Adjacent storage gaps differ at powers of two; the larger gap is the one-quantum ceiling."""
    before = torch.tensor([sign], dtype=dtype)
    zero = torch.zeros_like(before)
    away = torch.full_like(before, sign * float("inf"))
    one = torch.nextafter(before, away)
    two = torch.nextafter(one, away)
    assert_dpo_weight_steps_match((before, zero, one), (before, zero, before), 0.05)
    with pytest.raises(AssertionError, match="elementwise"):
        assert_dpo_weight_steps_match((before, zero, two), (before, zero, before), 0.05)
    with pytest.raises(AssertionError):
        assert_sgd_update(before, zero, one, 0.05)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_weight_step_budget_cannot_grow_with_a_corrupt_post_update_exponent(dtype):
    """Predict the quantum from the native SGD result, not a corrupted weight across a binade boundary."""
    changed = torch.ones(1, dtype=dtype)
    zero = torch.zeros_like(changed)
    before = torch.nextafter(torch.nextafter(changed, zero), zero)
    with pytest.raises(AssertionError, match="elementwise"):
        assert_dpo_weight_steps_match((before, zero, changed), (before, zero, before), 0.05)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("subnormal", [False, True], ids=["zero", "subnormal"])
def test_weight_step_bound_has_no_absolute_floor_at_zero_or_subnormal_weights(dtype, subnormal):
    """A small parameter gets its own representable spacing, not a normal-scale epsilon or absolute floor."""
    zero = torch.zeros(1, dtype=dtype)
    quantum = torch.nextafter(zero, torch.full_like(zero, float("inf")))
    before = quantum if subnormal else zero
    one = torch.nextafter(before, torch.full_like(before, float("inf")))
    two = torch.nextafter(one, torch.full_like(one, float("inf")))
    assert_dpo_weight_steps_match((before, zero, one), (before, zero, before), 0.05)
    with pytest.raises(AssertionError, match="elementwise"):
        assert_dpo_weight_steps_match((before, zero, two), (before, zero, before), 0.05)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_weight_step_bound_stays_finite_at_the_largest_stored_value(dtype):
    """An infinite adjacent representable value must not turn a finite weight's budget infinite."""
    before = torch.full((1,), torch.finfo(dtype).max, dtype=dtype)
    zero = torch.zeros_like(before)
    one = torch.nextafter(before, zero)
    two = torch.nextafter(one, zero)
    assert_dpo_weight_steps_match((before, zero, before), (before, zero, before), 0.05)
    with pytest.raises(AssertionError, match="elementwise"):
        assert_dpo_weight_steps_match((before, zero, two), (before, zero, before), 0.05)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_weight_step_bound_rejects_one_corrupt_element_without_norm_dilution(dtype):
    """A single invalid weight remains visible among thousands of otherwise matching elements."""
    before = torch.ones(4096, dtype=dtype)
    gradient = torch.full_like(before, 0.01)
    expected = before.clone().add_(gradient, alpha=-0.05)
    changed = expected.clone()
    changed[0] += 0.1
    with pytest.raises(AssertionError, match="elementwise"):
        assert_dpo_weight_steps_match((before, gradient, changed), (before, gradient, expected), 0.05)


@pytest.mark.parametrize("arm", [0, 1])
@pytest.mark.parametrize("slot", [0, 1, 2], ids=["before", "gradient", "after"])
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_weight_step_bound_refuses_nonfinite_snapshots(arm, slot, value):
    tensors = [torch.ones(2), torch.zeros(2), torch.ones(2)]
    first, second = [tensor.clone() for tensor in tensors], [tensor.clone() for tensor in tensors]
    (first if arm == 0 else second)[slot][0] = value
    with pytest.raises(AssertionError, match="finite matching"):
        assert_dpo_weight_steps_match(first, second, 0.05)


@pytest.mark.parametrize("arm", [0, 1])
@pytest.mark.parametrize("slot", [0, 1, 2], ids=["before", "gradient", "after"])
@pytest.mark.parametrize("fault", ["shape", "dtype"])
def test_weight_step_bound_refuses_broadcasting_or_mixed_storage(arm, slot, fault):
    first, second = [torch.ones(2), torch.zeros(2), torch.ones(2)], [torch.ones(2), torch.zeros(2), torch.ones(2)]
    target = first if arm == 0 else second
    target[slot] = target[slot][:1] if fault == "shape" else target[slot].bfloat16()
    with pytest.raises(AssertionError, match="finite matching"):
        assert_dpo_weight_steps_match(first, second, 0.05)


@pytest.mark.parametrize("learning_rate", [0.0, -0.05, float("nan"), float("inf")])
def test_weight_step_bound_refuses_invalid_learning_rates(learning_rate):
    snapshot = (torch.ones(2), torch.zeros(2), torch.ones(2))
    with pytest.raises(AssertionError, match="finite positive"):
        assert_dpo_weight_steps_match(snapshot, snapshot, learning_rate)


def test_weight_step_bound_refuses_nonfinite_native_predictions():
    before = torch.full((1,), torch.finfo(torch.float32).max)
    gradient = -before.clone()
    with pytest.raises(AssertionError, match="prediction is nonfinite"):
        assert_dpo_weight_steps_match((before, gradient, before), (before, gradient, before), 2.0)


@pytest.mark.parametrize("fail", [False, True], ids=["normal", "exception"])
def test_same_state_oracle_restores_bit_exact_weights_state_and_optimizer_references(fail):
    """A parity probe cannot change either independent trajectory or leave its gradients for the optimizer."""
    policy = SimpleNamespace(model=torch.nn.Linear(4, 3))
    model = torch.nn.Linear(4, 3)
    oracle = SimpleNamespace(
        model=model, state=SimpleNamespace(global_step=1), optimizer=torch.optim.SGD(model.parameters(), lr=0.05)
    )
    original = {name: value.detach().clone() for name, value in model.named_parameters()}
    source = {name: value.detach().clone() for name, value in policy.model.named_parameters()}
    parameters = tuple(model.parameters())
    optimizer_parameters = tuple(oracle.optimizer.param_groups[0]["params"])
    state = oracle.state
    with pytest.raises(RuntimeError, match="probe failed") if fail else nullcontext():
        with same_state_dpo_oracle(oracle, policy):
            assert oracle.state is not state and oracle.state.global_step == state.global_step
            assert all(actual is expected for actual, expected in zip(model.parameters(), parameters, strict=True))
            for name, value in model.named_parameters():
                torch.testing.assert_close(value, source[name], rtol=0, atol=0)
            sum(value.square().sum() for value in model.parameters()).backward()
            if fail:
                raise RuntimeError("probe failed")
    assert oracle.state is state
    assert all(actual is expected for actual, expected in zip(model.parameters(), parameters, strict=True))
    assert all(
        actual is expected
        for actual, expected in zip(oracle.optimizer.param_groups[0]["params"], optimizer_parameters, strict=True)
    )
    for name, value in model.named_parameters():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
        assert value.grad is None
    for name, value in policy.model.named_parameters():
        torch.testing.assert_close(value, source[name], rtol=0, atol=0)


@pytest.mark.parametrize("fault", ["names", "shape", "dtype", "aliased"])
def test_same_state_oracle_refuses_incompatible_parameters_before_mutation(fault):
    model = torch.nn.Linear(4, 3)
    source = torch.nn.Linear(4, 3, bias=fault != "names")
    if fault == "shape":
        source = torch.nn.Linear(5, 3)
    elif fault == "dtype":
        source.bfloat16()
    elif fault == "aliased":
        source = model
    oracle = SimpleNamespace(
        model=model, state=SimpleNamespace(global_step=1), optimizer=torch.optim.SGD(model.parameters(), lr=0.05)
    )
    original = {name: value.detach().clone() for name, value in model.named_parameters()}
    state = oracle.state
    with pytest.raises(AssertionError, match="DPO same-state oracle requires"):
        with same_state_dpo_oracle(oracle, SimpleNamespace(model=source)):
            pytest.fail("incompatible parameters entered a same-state probe")
    assert oracle.state is state
    for name, value in model.named_parameters():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)


def test_same_state_oracle_matches_actual_loss_and_gradients_without_overwriting_its_trajectory():
    """Strict parity compares one current state; independent SGD trajectories may have different losses."""
    policy, oracle_model = _model(), _model()
    with torch.no_grad():
        policy.lm_head.weight.mul_(2)
    bounded, full = _loss_trainer(policy, bounded=True), _loss_trainer(oracle_model, bounded=False)
    full.state = SimpleNamespace(global_step=1)
    full.optimizer = torch.optim.SGD(oracle_model.parameters(), lr=0.05)
    batch = _reference_batch(full, oracle_model)
    independent = full.compute_loss(oracle_model, batch)
    got = bounded.compute_loss(policy, batch)
    got.backward()
    with pytest.raises(AssertionError):
        assert_dpo_loss_matches(got, independent)
    original = {name: value.detach().clone() for name, value in oracle_model.named_parameters()}
    with same_state_dpo_oracle(full, bounded):
        observed = full.compute_loss(oracle_model, batch)
        assert_dpo_loss_matches(got, observed)
        observed.backward()
        assert_dpo_gradients_match(
            {name: value.grad for name, value in policy.named_parameters()},
            {name: value.grad for name, value in oracle_model.named_parameters()},
            bf16=False,
        )
    own = full.compute_loss(oracle_model, batch)
    torch.testing.assert_close(own, independent, rtol=0, atol=0)
    own.backward()
    gradients = {name: value.grad.detach().clone() for name, value in oracle_model.named_parameters()}
    full.optimizer.step()
    for name, value in oracle_model.named_parameters():
        assert_sgd_update(original[name], gradients[name], value.detach(), 0.05)


def test_same_state_oracle_wrong_copied_weights_still_fail_the_strict_loss_gate():
    policy, oracle_model = _model(), _model()
    bounded, full = _loss_trainer(policy, bounded=True), _loss_trainer(oracle_model, bounded=False)
    full.state = SimpleNamespace(global_step=1)
    full.optimizer = torch.optim.SGD(oracle_model.parameters(), lr=0.05)
    batch = _reference_batch(full, oracle_model)
    expected = bounded.compute_loss(policy, batch)
    with same_state_dpo_oracle(full, bounded):
        with torch.no_grad():
            oracle_model.lm_head.weight.zero_()
        with pytest.raises(AssertionError):
            assert_dpo_loss_matches(expected, full.compute_loss(oracle_model, batch))


def test_same_state_oracle_gives_probe_and_real_step_separate_deferred_ep_sweeps():
    """Both backwards divide their own expert gradients once, even at the same numerical global step."""
    model = torch.nn.Linear(4, 3)
    full = SimpleNamespace(
        model=model, state=SimpleNamespace(global_step=1), optimizer=torch.optim.SGD(model.parameters(), lr=0.05)
    )
    full._ep_config = SimpleNamespace(
        runs_post_backward_sweep=True,
        expert_replica_group=None,
        stage_world_size=2,
        expert_tp_size=1,
        fp32_grad_reduce=False,
        sweep_reduces_across_replicas=False,
    )
    full._classify_deferred_params = lambda config: (list(model.parameters()), [], [])
    full._deferred_sweep_last_step = (full.state, 0)
    state = full.state
    with same_state_dpo_oracle(full, SimpleNamespace(model=torch.nn.Linear(4, 3))):
        for parameter in model.parameters():
            parameter.grad = torch.full_like(parameter, 8)
        GradientSyncMixin._run_deferred_grad_sweep(full)
        for parameter in model.parameters():
            torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 4), rtol=0, atol=0)
        with pytest.raises(RuntimeError, match="ran twice"):
            GradientSyncMixin._run_deferred_grad_sweep(full)
    assert full.state is state
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, 8)
    GradientSyncMixin._run_deferred_grad_sweep(full)
    for parameter in model.parameters():
        torch.testing.assert_close(parameter.grad, torch.full_like(parameter, 4), rtol=0, atol=0)
    assert full._deferred_sweep_last_step[0] is state and full._deferred_sweep_last_step[1] == 1


@pytest.mark.parametrize("fault", [None, "stale", "wrong", "nonfinite_own"])
def test_same_state_backward_preserves_strict_parity_and_independent_full_gradients(monkeypatch, fault):
    """The full comparison uses current policy weights, then restores its own state for the actual update."""
    policy, oracle_model = _model(), _model()
    with torch.no_grad():
        policy.lm_head.weight.mul_(2)
    bounded, full = _loss_trainer(policy, bounded=True), _loss_trainer(oracle_model, bounded=False)
    norm_states, full_backwards = [], []
    for trainer in (bounded, full):
        _configure_cpu_backward(trainer)
    norm = full._get_grad_norm

    def full_norm(model):
        norm_states.append(full.state)
        return norm(model)

    def full_backward(loss):
        full_backwards.append(full.state)
        loss.backward()

    full._get_grad_norm = full_norm
    full.accelerator.backward = full_backward
    original_state = full.state
    original = {name: value.detach().clone() for name, value in oracle_model.named_parameters()}
    batch = _reference_batch(full, oracle_model)
    own = full.compute_loss(oracle_model, batch)
    own_gradients = torch.autograd.grad(own, tuple(oracle_model.parameters()))
    scope = dpo_scoring.same_state_dpo_oracle

    @contextmanager
    def corrupt_scope(oracle, source):
        with scope(oracle, source):
            with torch.no_grad():
                oracle.model.lm_head.weight.zero_()
            yield

    if fault == "stale":
        monkeypatch.setattr(dpo_scoring, "same_state_dpo_oracle", lambda oracle, source: nullcontext())
    elif fault == "wrong":
        monkeypatch.setattr(dpo_scoring, "same_state_dpo_oracle", corrupt_scope)
    elif fault == "nonfinite_own":
        compute = full.compute_loss
        calls = []

        def corrupt_own_loss(*args):
            calls.append(None)
            value = compute(*args)
            return value * float("nan") if len(calls) == 2 else value

        monkeypatch.setattr(full, "compute_loss", corrupt_own_loss)
    with pytest.raises(AssertionError) if fault else nullcontext():
        bounded_loss, relative, independent_loss = dpo_scoring._compare_backward(
            bounded, full, batch, family="dense", bf16=False
        )
    assert full.state is original_state
    for name, value in oracle_model.named_parameters():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    if fault == "nonfinite_own":
        assert len(full_backwards) == 1
    elif fault is None:
        assert len(norm_states) == len(full_backwards) == 2
        assert norm_states[0] is not original_state and norm_states[1] is original_state
        assert full_backwards[0] is not original_state and full_backwards[1] is original_state
        assert relative < dpo_scoring.FP32_GRAD_REL_L2
        torch.testing.assert_close(independent_loss, own, rtol=0, atol=0)
        with pytest.raises(AssertionError):
            assert_dpo_loss_matches(bounded_loss, independent_loss)
        for actual, expected in zip(oracle_model.parameters(), own_gradients, strict=True):
            torch.testing.assert_close(actual.grad, expected, rtol=0, atol=0)


def _dpo_validation_failure_worker(rank):
    for fault, fence in (
        (None, None),
        ("loss", "DPO same-state loss and routing checks"),
        ("gradient", "DPO same-state gradients, norm and expert-activity checks"),
        ("nonfinite_own", "DPO independent full-head loss check"),
        ("update", "DPO independent SGD and stored-weight checks"),
    ):
        bounded, full = _loss_trainer(_model(), bounded=True), _loss_trainer(_model(), bounded=False)
        for trainer in (bounded, full):
            _configure_cpu_backward(trainer)
        batch = _reference_batch(full, full.model)
        depth = 0
        barrier = dpo_scoring.barrier_on_exit

        @contextmanager
        def local_checks(what):
            nonlocal depth
            with barrier(what):
                depth += 1
                try:
                    yield
                finally:
                    depth -= 1

        def outside_validation(operation):
            def checked(*args, **kwargs):
                assert depth == 0, "A collective operation entered a local-validation fence"
                return operation(*args, **kwargs)

            return checked

        def injected_assertion(check):
            def checked(*args, **kwargs):
                result = check(*args, **kwargs)
                if rank == 1:
                    raise AssertionError(f"injected {fault} validation failure")
                return result

            return checked

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(dpo_scoring, "barrier_on_exit", local_checks)
            for trainer in (bounded, full):
                for name in ("compute_loss", "_finish_fsdp2_backward", "_get_grad_norm"):
                    patch.setattr(trainer, name, outside_validation(getattr(trainer, name)))
                patch.setattr(trainer.accelerator, "backward", outside_validation(trainer.accelerator.backward))
            if fault in ("loss", "gradient"):
                name = "assert_dpo_loss_matches" if fault == "loss" else "assert_dpo_gradients_match"
                patch.setattr(dpo_scoring, name, injected_assertion(getattr(dpo_scoring, name)))
            elif fault == "nonfinite_own":
                compute, calls = full.compute_loss, 0

                def corrupt_own_loss(*args):
                    nonlocal calls
                    calls += 1
                    value = compute(*args)
                    return value * float("nan") if calls == 2 and rank == 1 else value

                patch.setattr(full, "compute_loss", corrupt_own_loss)
            error, entered = None, False
            try:
                dpo_scoring._compare_backward(bounded, full, batch, family="dense", bf16=False)
                before = [
                    {name: value.detach().clone() for name, value in trainer.model.named_parameters()}
                    for trainer in (bounded, full)
                ]
                gradients = [
                    {name: value.grad.detach().clone() for name, value in trainer.model.named_parameters()}
                    for trainer in (bounded, full)
                ]
                for trainer in (bounded, full):
                    trainer.optimizer.step()
                updated = [
                    {name: value.detach().clone() for name, value in trainer.model.named_parameters()}
                    for trainer in (bounded, full)
                ]
                if fault == "update" and rank == 1:
                    next(iter(updated[1].values())).reshape(-1)[0] += 0.25
                assert_dpo_updates_match(before, gradients, updated, bf16=False)
                entered = True
                marker = torch.ones(1)
                dist.all_reduce(marker)
                assert marker.item() == 2
            except (AssertionError, RuntimeError) as exc:
                error = f"{type(exc).__name__}: {exc}"
        outcomes = [None, None]
        dist.all_gather_object(outcomes, (error, entered))
        if fault is None:
            assert outcomes == [(None, True), (None, True)], outcomes
        else:
            assert all(error and not entered for error, entered in outcomes), outcomes
            assert outcomes[1][0].startswith("AssertionError:"), outcomes
            assert outcomes[0][0].startswith("RuntimeError:") and fence in outcomes[0][0], outcomes


def test_dpo_local_validation_failures_stop_both_ranks_before_the_next_collective():
    """Real loss, gradient, independent-loss and SGD faults must fail both ranks without fencing collective work."""
    run_gloo_ranks(_dpo_validation_failure_worker, 2, pg_timeout=timedelta(seconds=30))


def test_fp32_objective_gate_rejects_constant_offset_even_when_gradients_match():
    margin = torch.tensor(0.3, requires_grad=True)
    loss = -F.logsigmoid(margin)
    changed = loss + 0.001
    assert torch.equal(
        torch.autograd.grad(loss, margin, retain_graph=True)[0], torch.autograd.grad(changed, margin)[0]
    )
    assert_dpo_loss_matches(loss, loss.detach().double())
    with pytest.raises(AssertionError):
        assert_dpo_loss_matches(changed, loss)


@pytest.mark.parametrize("omitted", ["head.weight", "backbone.weight", "expert.gate_up_proj"])
def test_sgd_proof_rejects_one_omitted_parameter_update(omitted):
    before = {
        "head.weight": torch.full((8,), 0.7),
        "backbone.weight": torch.full((4,), 0.25),
        "expert.gate_up_proj": torch.full((4, 8), 0.7, dtype=torch.bfloat16),
    }
    gradients = {name: torch.full_like(value, 0.3) for name, value in before.items()}
    after = {name: value.clone().add_(gradients[name], alpha=-0.05) for name, value in before.items()}
    for name in before:
        assert_sgd_update(before[name], gradients[name], after[name], 0.05)
        assert_sgd_deltas_match(
            (before[name], gradients[name], after[name]), (before[name], gradients[name], after[name]), 0.05
        )
    with pytest.raises(AssertionError):
        assert_sgd_update(before[omitted], gradients[omitted], before[omitted], 0.05)
    with pytest.raises(AssertionError, match="SGD delta"):
        assert_sgd_deltas_match(
            (before[omitted], gradients[omitted], before[omitted]),
            (before[omitted], gradients[omitted], after[omitted]),
            0.05,
        )


@pytest.mark.parametrize("family", ["dense", "qwen3_moe"])
def test_gpu_numerical_row_passes_real_preallocation_model_and_precision_guards(family, monkeypatch):
    config = Qwen3Config() if family == "dense" else Qwen3MoeConfig(num_experts=4, num_experts_per_tok=2)
    observed = []
    loader_signature = inspect.signature(dpo_scoring.load_distributed_model)

    def preflight(**kwargs):
        loader_signature.bind(**kwargs)
        pc = kwargs["parallelism_config"]
        reject_unsupported_run(pc, kwargs["dtype"])
        reject_unsupported_model(pc, config, sinks_policy=SinksPolicy.NEUTRALIZED, wraps_experts=family == "qwen3_moe")
        observed.append(pc)
        raise RuntimeError("configuration preflight completed before GPU allocation")

    monkeypatch.setattr(
        dpo_scoring,
        "ParallelismConfig",
        lambda **kwargs: make_parallelism_config(world_size=2, gpus_per_node=2, **kwargs),
    )
    monkeypatch.setattr(dpo_scoring, "load_distributed_model", preflight)
    with pytest.raises(RuntimeError, match="configuration preflight completed"):
        dpo_scoring._build_trainer(
            SimpleNamespace(), "/unread/local/fixture", family=family, dtype=torch.bfloat16, bounded=True
        )
    assert len(observed) == 1 and observed[0].fp32_non_ep_params == (family == "dense")
    assert not observed[0].ep_fp32_experts


@pytest.mark.parametrize("family,dtype", [("dense", torch.float32), ("qwen3_moe", torch.bfloat16)])
def test_gpu_numerical_storage_guard_rejects_one_wrong_parameter(family, dtype):
    model = torch.nn.Linear(4, 3).to(dtype)
    trainer = SimpleNamespace(model=model)
    dpo_scoring.verify_dpo_numerical_storage(trainer, family=family)
    model.bias.data = model.bias.data.to(torch.bfloat16 if dtype == torch.float32 else torch.float32)
    with pytest.raises(AssertionError, match=f"DPO {family} numerical fixture requires .*storage: bias"):
        dpo_scoring.verify_dpo_numerical_storage(trainer, family=family)


@pytest.mark.parametrize("module", [dpo_scoring, precompute_e2e], ids=["equivalence", "lifecycle"])
def test_gpu_fixture_loader_calls_bind_actual_release_signatures(module):
    calls = [
        node
        for node in ast.walk(ast.parse(inspect.getsource(module)))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in ("load_distributed_model", "load_peft_model")
    ]
    assert calls, "the contract no longer sees the GPU fixture loaders"
    for call in calls:
        assert all(not isinstance(arg, ast.Starred) for arg in call.args)
        assert all(keyword.arg is not None for keyword in call.keywords)
        signature = inspect.signature(getattr(module, call.func.id))
        signature.bind(*[object() for _ in call.args], **{keyword.arg: object() for keyword in call.keywords})
    required = inspect.signature(dpo_scoring.load_distributed_model)
    with pytest.raises(TypeError, match="trust_remote_code"):
        required.bind(model_name_or_path="/unread/local/fixture", parallelism_config=object())


@pytest.mark.parametrize("bounded", [True, False], ids=["bounded", "full-oracle"])
def test_gpu_numerical_builder_reaches_real_cpu_trainer_constructor(tmp_path, monkeypatch, bounded):
    loader_signature = inspect.signature(dpo_scoring.load_distributed_model)
    config_constructor = dpo_scoring.DPOConfig
    observed, teardown = [], []

    def load_cpu_fixture(**kwargs):
        loader_signature.bind(**kwargs)
        observed.append(kwargs)
        return _model(), _tokenizer()

    def cpu_config(**kwargs):
        inspect.signature(config_constructor).bind(**kwargs)
        return config_constructor(**kwargs, use_cpu=True)

    monkeypatch.setattr(dpo_scoring, "load_distributed_model", load_cpu_fixture)
    monkeypatch.setattr(dpo_scoring, "DPOConfig", cpu_config)
    try:
        trainer, hidden_dtypes = dpo_scoring._build_trainer(
            SimpleNamespace(output_dir=str(tmp_path), on_teardown=teardown.append),
            "/unread/local/fixture",
            family="dense",
            dtype=torch.float32,
            bounded=bounded,
        )
        assert isinstance(trainer, DistributedDPOTrainer) and trainer.optimizer is not None
        assert trainer.use_chunked_dpo_logprobs is bounded
        assert len(observed) == 1 and observed[0]["trust_remote_code"] is False
        assert hidden_dtypes and set(hidden_dtypes) == {torch.float32}
        assert trainer.train_dataset.features["ref_chosen_logps"].dtype == "float32"
    finally:
        for callback in reversed(teardown):
            callback()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
