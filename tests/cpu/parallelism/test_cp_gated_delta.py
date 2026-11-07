"""Native Torch-dispatched GatedDeltaNet on real CPU Gloo; not CUDA/FLA qualification."""

import copy
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist
from peft import LoraConfig, get_peft_model
from torch.utils.checkpoint import checkpoint
from transformers.models.qwen3_5 import modeling_qwen3_5

from halo.distributed.context_parallel import families, ulysses_attention
from halo.distributed.context_parallel.config import CPConfig, CPConfigError
from halo.distributed.context_parallel.families import ULYSSES_GATED_DELTA_MAP, Qwen3_5UlyssesGatedDelta
from halo.distributed.context_parallel.model_rules import reject_unsupported_ulysses_model
from halo.distributed.context_parallel.ulysses_model import UlyssesModel
from halo.models.unwrapping import find_peft_model, strip_cp_attention_level
from halo.trainers.parallelism.mode_setup import ModeSetupMixin
from halo.trainers.preference.smpo import SMPOTrainer
from halo.trainers.sft import DistributedSFTTrainer
from halo.trainers.trainer_rules import reject_hybrid_cp_training
from tests.common.equivalence.cp_equivalence import reference_flash_attn
from tests.common.gloo import run_gloo_ranks
from tests.common.hybrid_cp import (
    build_hybrid_model,
    hybrid_hidden_batch,
    hybrid_tensor_agreement,
    hybrid_text_config,
    hybrid_token_batch,
)


def _capture(layer, hidden, mask, targets, *, checkpointing=False):
    """Return detached outputs and fresh gradients for the masked objective, with optional reentrant recomputation."""
    layer.zero_grad(set_to_none=True)
    inputs = hidden.detach().clone().requires_grad_()
    output = checkpoint(layer, inputs, use_reentrant=True) if checkpointing else layer(inputs)
    loss = (output * targets * mask[..., None]).sum() / mask.sum().clamp_min(1)
    loss.backward()
    return output.detach(), inputs.grad, {name: parameter.grad for name, parameter in layer.named_parameters()}


def _layer_worker(rank):
    """COLLECTIVE. Match native real-token outputs and summed gradients; require layout/state faults to diverge."""
    for batch, length, checkpointing in ((1, 12, False), (3, 14, False), (3, 12, True)):
        torch.manual_seed(114)
        config = hybrid_text_config()
        native = modeling_qwen3_5.Qwen3_5GatedDeltaNet(config, 0)
        with torch.no_grad():
            native.dt_bias.copy_(torch.linspace(-0.5, 0.3, native.num_v_heads))
        wrapped = Qwen3_5UlyssesGatedDelta(copy.deepcopy(native), dist.group.WORLD, 2)
        hidden, mask, targets = hybrid_hidden_batch(batch, length, config.hidden_size)
        reference = _capture(native, hidden, mask, targets, checkpointing=checkpointing)
        part = slice(rank * (length // 2), (rank + 1) * (length // 2))
        local_hidden, local_mask, local_targets = hidden[:, part], mask[:, part], targets[:, part]
        # Every rank owns one unscaled loss contribution; SUM reconstructs the full native objective.
        local_targets = local_targets * local_mask.sum().clamp_min(1) / mask.sum()
        with patch.object(dist, "all_to_all_single", wraps=dist.all_to_all_single) as observed:
            output, input_gradient, gradients = _capture(
                wrapped, local_hidden, local_mask, local_targets, checkpointing=checkpointing
            )
            assert observed.call_count == (6 if checkpointing else 4)
        real = local_mask[..., None].expand_as(output)
        torch.testing.assert_close(output[real], reference[0][:, part][real], atol=2e-6, rtol=2e-5)
        torch.testing.assert_close(input_gradient, reference[1][:, part], atol=2e-6, rtol=2e-5)
        for name, gradient in gradients.items():
            assert gradient is not None, name
            dist.all_reduce(gradient)
            expected = reference[2][name.removeprefix("original_attention.")]
            torch.testing.assert_close(gradient, expected, atol=3e-6, rtol=3e-5, msg=name)
        original_keys = set(native.state_dict())
        assert {
            strip_cp_attention_level("layer." + name).removeprefix("layer.") for name in wrapped.state_dict()
        } == original_keys
        with patch.object(dist, "all_to_all_single", wraps=dist.all_to_all_single) as observed:
            for kwargs in (
                {"cache_params": object()},
                {"use_cache": True},
                {"seq_idx": torch.zeros(1, length // 2)},
                {"attention_mask": local_mask},
            ):
                with pytest.raises(CPConfigError):
                    wrapped(local_hidden, **kwargs)
            assert observed.call_count == 0

    native = modeling_qwen3_5.Qwen3_5GatedDeltaNet(hybrid_text_config(), 0)
    with torch.no_grad():
        native.dt_bias.copy_(torch.linspace(-0.5, 0.3, native.num_v_heads))
    hidden, mask, targets = hybrid_hidden_batch(3, 12, native.hidden_size)
    baseline = native(hidden).detach()
    part = slice(rank * 6, (rank + 1) * 6)
    real = mask[:, part, None].expand(3, 6, native.hidden_size)
    for fault in ("rank", "inverse", "boundary", "conv_channels", "decay", "dt_bias"):
        wrapped = Qwen3_5UlyssesGatedDelta(copy.deepcopy(native), dist.group.WORLD, 2)
        if fault == "rank":
            wrapped.cp_rank = 1 - rank
        if fault in ("decay", "dt_bias"):
            parameter = wrapped.original_attention.A_log if fault == "decay" else wrapped.original_attention.dt_bias
            with torch.no_grad():
                parameter.copy_(parameter.roll(native.num_v_heads // 2))
        original_conv = modeling_qwen3_5.causal_conv1d_fn
        original_inverse = families.heads_to_sequence

        def broken_conv(inputs, weight, *args, **kwargs):
            """Corrupt channel ownership or sever the causal convolution at the CP boundary."""
            if fault == "conv_channels":
                return original_conv(inputs, weight.roll(1, dims=0), *args, **kwargs)
            return torch.cat(
                [original_conv(chunk, weight, *args, **kwargs) for chunk in inputs.chunk(2, dim=-1)], dim=-1
            )

        def broken_inverse(core, group):
            return original_inverse(core, group).roll(1, dims=2)

        with (
            patch.object(
                modeling_qwen3_5,
                "causal_conv1d_fn",
                broken_conv if fault in ("boundary", "conv_channels") else original_conv,
            ),
            patch(
                "halo.distributed.context_parallel.families.heads_to_sequence",
                broken_inverse if fault == "inverse" else original_inverse,
            ),
        ):
            changed = wrapped(hidden[:, part]).detach()
        error = (changed[real] - baseline[:, part][real]).abs().max()
        dist.all_reduce(error, op=dist.ReduceOp.MAX)
        assert error > 1e-5, fault


def test_native_gated_delta_cp2_matches_outputs_and_every_gradient():
    run_gloo_ranks(_layer_worker, 2, pg_timeout=timedelta(seconds=20))


def _model_worker(rank):
    """Collectively match full-model loss, averaged gradients, SGD updates and native state-dict reloads."""
    for batch, checkpointing in ((1, False), (3, True)):
        baseline = build_hybrid_model()
        candidate = copy.deepcopy(baseline)
        candidate.config._attn_implementation = "flash_attention_2"
        cp_config = CPConfig(2, stage_world_size=2, stage_base_rank=0, nvlink_domain_size=2)
        wrapped = UlyssesModel(candidate, cp_config)
        if checkpointing:
            baseline.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
            candidate.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
        generator = torch.Generator().manual_seed(114)
        inputs = torch.randint(1, baseline.config.vocab_size, (batch, 14), generator=generator)
        mask = torch.arange(14)[None, :] < (13 - torch.arange(batch))[:, None]
        labels = inputs.masked_fill(~mask, -100)
        labels[:, : 8 if checkpointing else 7] = -100
        expected = baseline(input_ids=inputs, attention_mask=mask, labels=labels, use_cache=False)
        expected.loss.backward()
        # Only attention uses a CPU reference; GDN and CP loss are the actual model/production paths.
        with patch.object(ulysses_attention, "_flash_attn_func", return_value=reference_flash_attn):
            actual = wrapped(input_ids=inputs, attention_mask=mask, labels=labels, use_cache=False)
            actual.loss.backward()
        part = slice(rank * 7, (rank + 1) * 7)
        real = mask[:, part, None].expand_as(actual.logits)
        torch.testing.assert_close(actual.logits[real], expected.logits[:, part][real], atol=3e-6, rtol=3e-5)
        average_loss = actual.loss.detach().clone()
        dist.all_reduce(average_loss)
        torch.testing.assert_close(average_loss / 2, expected.loss, atol=2e-6, rtol=2e-5)
        expected_parameters = dict(baseline.named_parameters())
        for name, parameter in wrapped.named_parameters():
            assert parameter.grad is not None, name
            dist.all_reduce(parameter.grad)
            parameter.grad.div_(2)
            torch.testing.assert_close(parameter.grad, expected_parameters[name].grad, atol=3e-6, rtol=3e-5, msg=name)
        before = baseline.lm_head.weight.detach().clone()
        for model in (baseline, wrapped):
            torch.optim.SGD(model.parameters(), lr=0.01).step()
        assert not torch.equal(before, baseline.lm_head.weight)
        saved = wrapped.state_dict()
        assert saved.keys() == baseline.state_dict().keys()
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(saved[name], value, atol=3e-6, rtol=3e-5, msg=name)
        reload = build_hybrid_model()
        reload.load_state_dict(saved, strict=True)
        reload.eval()
        baseline.eval()
        with torch.no_grad():
            restored = reload(input_ids=inputs, attention_mask=mask, use_cache=False).logits
            expected_reloaded = baseline(input_ids=inputs, attention_mask=mask, use_cache=False).logits
        torch.testing.assert_close(restored[mask], expected_reloaded[mask], atol=3e-6, rtol=3e-5)


def test_whole_hybrid_cp_model_with_cpu_reference_attention_and_native_reload():
    run_gloo_ranks(_model_worker, 2, pg_timeout=timedelta(seconds=20))


@pytest.mark.parametrize(
    "overrides,cp_size,match",
    (
        ({}, 4, "CP2"),
        ({"linear_num_key_heads": 3, "linear_num_value_heads": 6}, 2, "key heads"),
        ({"linear_num_value_heads": 7}, 2, "value heads"),
        ({"linear_num_key_heads": 4, "linear_num_value_heads": 6}, 2, "complete groups of key heads"),
    ),
)
def test_gated_delta_geometry_refusals_before_group_lookup(overrides, cp_size, match):
    layer = modeling_qwen3_5.Qwen3_5GatedDeltaNet(hybrid_text_config(**overrides), 0)
    with pytest.raises(CPConfigError, match=match):
        Qwen3_5UlyssesGatedDelta(layer, None, cp_size)


def test_validator_and_patcher_share_dense_hybrid_claim():
    model = build_hybrid_model()
    model.config._attn_implementation = "flash_attention_2"
    reject_unsupported_ulysses_model(model, 2)
    assert set(ULYSSES_GATED_DELTA_MAP) == {"Qwen3_5GatedDeltaNet"}


def test_hybrid_multimodal_config_is_rejected_at_validation_and_adoption():
    model = build_hybrid_model()
    model.config.vision_config = SimpleNamespace()
    model.config._attn_implementation = "flash_attention_2"
    with pytest.raises(CPConfigError, match="text-only"):
        reject_unsupported_ulysses_model(model, 2)
    trainer = DistributedSFTTrainer.__new__(DistributedSFTTrainer)
    trainer.model = model
    trainer.args = SimpleNamespace()
    trainer.parallelism_config = SimpleNamespace(is_cp_mode=True, cp_size=2, expert_lora=False)
    with pytest.raises(CPConfigError, match="text-only"):
        trainer._adopt_cp_wrapper()


@pytest.mark.parametrize("supported,peft,packing", ((False, None, False), (True, object(), False), (True, None, True)))
def test_hybrid_training_gate_rejects_other_trainers_adapters_and_packing(supported, peft, packing):
    with pytest.raises(CPConfigError):
        reject_hybrid_cp_training(
            build_hybrid_model(),
            SimpleNamespace(packing=packing),
            SimpleNamespace(is_cp_mode=True, cp_size=2, expert_lora=False),
            supported=supported,
            peft_config=peft,
        )


@pytest.mark.parametrize(
    "layer_metadata",
    (
        pytest.param("missing", id="missing"),
        pytest.param(None, id="null"),
        pytest.param([], id="empty"),
        pytest.param(["full_attention", "full_attention"], id="misleading"),
    ),
)
def test_hybrid_training_gate_reads_live_layers_without_linear_attention_metadata(layer_metadata):
    """Live hybrid layers must trigger refusal even when their config metadata is absent or misleading."""
    model = build_hybrid_model()
    if layer_metadata == "missing":
        del model.config.layer_types
    else:
        model.config.layer_types = layer_metadata
    assert any(type(module).__name__ in ULYSSES_GATED_DELTA_MAP for module in model.modules())
    with pytest.raises(CPConfigError, match="SFT only"):
        reject_hybrid_cp_training(
            model,
            SimpleNamespace(),
            SimpleNamespace(is_cp_mode=True, cp_size=2, expert_lora=False),
            supported=False,
        )


def test_hybrid_training_gate_refuses_peft_tuner_layers_injected_in_place():
    """Refuse structural LoRA layers even after the enclosing PEFT model wrapper is removed."""
    model = get_peft_model(build_hybrid_model(), LoraConfig(r=2, target_modules=["in_proj_qkv"])).get_base_model()
    assert find_peft_model(model) is None
    assert any("lora_A" in name for name, _ in model.named_parameters())
    with pytest.raises(CPConfigError, match="full fine-tuning"):
        reject_hybrid_cp_training(
            model,
            SimpleNamespace(),
            SimpleNamespace(is_cp_mode=True, cp_size=2, expert_lora=False),
            supported=True,
        )


@pytest.mark.parametrize("is_cp_mode", [False, True])
def test_hybrid_training_gate_leaves_non_cp_and_attention_only_cp_unchanged(is_cp_mode):
    """Hybrid restrictions must not reject adapter requests outside CP or on attention-only CP models."""
    model = build_hybrid_model(**({"layer_types": ["full_attention", "full_attention"]} if is_cp_mode else {}))
    reject_hybrid_cp_training(
        model,
        SimpleNamespace(),
        SimpleNamespace(is_cp_mode=is_cp_mode, cp_size=2, expert_lora=False),
        supported=False,
        peft_config=object(),
    )


@pytest.mark.parametrize("model", [None, "tiny/checkpoint"])
def test_hybrid_training_gate_leaves_unmaterialized_models_for_hf_setup(model):
    reject_hybrid_cp_training(
        model,
        SimpleNamespace(),
        SimpleNamespace(is_cp_mode=True, cp_size=2, expert_lora=False),
        supported=False,
    )


@pytest.mark.parametrize("missing_layer_metadata", [False, True])
def test_adoption_gate_executes_for_a_cp_capable_non_sft_trainer(missing_layer_metadata):
    """Reject unsupported hybrid trainers before CP wrapper adoption, regardless of layer metadata."""
    model = build_hybrid_model()
    if missing_layer_metadata:
        del model.config.layer_types
    trainer = SimpleNamespace(
        args=SimpleNamespace(),
        parallelism_config=SimpleNamespace(is_cp_mode=True, cp_size=2, expert_lora=False),
        SUPPORTS_HYBRID_CP=False,
        _top_level_model=lambda: model,
    )
    with patch("halo.trainers.parallelism.mode_setup.require_ulysses_model") as wrapper_lookup:
        with pytest.raises(CPConfigError, match="SFT only"):
            ModeSetupMixin._adopt_cp_wrapper(trainer)
        wrapper_lookup.assert_not_called()


@pytest.mark.parametrize("missing_layer_metadata", [False, True])
@pytest.mark.parametrize(
    "trainer_class,peft,match",
    ((SMPOTrainer, None, "SFT only"), (DistributedSFTTrainer, object(), "full fine-tuning")),
)
def test_hybrid_trainer_initialization_refuses_before_hf_setup(trainer_class, peft, match, missing_layer_metadata):
    """Reject unsupported trainer/adapter combinations before HuggingFace setup can obscure the failure."""
    kwargs = {
        "model": build_hybrid_model(),
        "args": SimpleNamespace(),
        "parallelism_config": SimpleNamespace(is_cp_mode=True, cp_size=2, expert_lora=False),
    }
    if missing_layer_metadata:
        del kwargs["model"].config.layer_types
    if peft is not None:
        kwargs["peft_config"] = peft
    with pytest.raises(CPConfigError, match=match):
        if trainer_class is DistributedSFTTrainer:
            trainer_class(**kwargs)
        else:
            trainer_class.__new__(trainer_class)._init_distributed_config(kwargs)


def test_sft_subclasses_must_declare_hybrid_support_on_the_concrete_trainer():
    class UnqualifiedSFT(DistributedSFTTrainer):
        def _top_level_model(self):
            return self.model

    trainer = UnqualifiedSFT.__new__(UnqualifiedSFT)
    trainer.args = SimpleNamespace()
    trainer.parallelism_config = SimpleNamespace(is_cp_mode=True, cp_size=2, expert_lora=False)
    trainer.model = build_hybrid_model()
    assert trainer.SUPPORTS_HYBRID_CP
    with pytest.raises(CPConfigError, match="SFT only"):
        trainer._adopt_cp_wrapper()


def test_gpu_oracle_gradient_bars_keep_expected_zero_and_small_parameters_observable():
    zero = torch.zeros(4)
    assert hybrid_tensor_agreement(zero, zero, gradient=True)[0]
    assert not hybrid_tensor_agreement(zero + 1e-10, zero, gradient=True)[0]
    expected = torch.tensor([1e-9, -2e-9, 3e-9])
    assert hybrid_tensor_agreement(expected, expected, gradient=True)[0]
    assert not hybrid_tensor_agreement(expected * 2, expected, gradient=True)[0]
    assert not hybrid_tensor_agreement(-expected, expected, gradient=True)[0]


def test_gpu_logical_batches_pin_boundary_and_zero_supervision_shards():
    ordinary = hybrid_token_batch(3, 64)
    boundary = hybrid_token_batch(3, 64, boundary_only=True)
    assert torch.equal(ordinary["input_ids"], boundary["input_ids"])
    assert not (ordinary["labels"][:, 1:66] != -100).any()
    assert (ordinary["labels"][:, 66:] != -100).any()
    assert torch.equal((boundary["labels"] != -100).sum(1), torch.ones(3, dtype=torch.long))
    assert (boundary["labels"][:, 65] != -100).all()
    with pytest.raises(ValueError, match="excluded"):
        hybrid_token_batch(1, 64, length=16386)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
