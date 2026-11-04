#!/usr/bin/env python
"""Nemotron-H's family contract against the real ``NemotronHMoE``: gateless experts on a latent the wire carries, a
router scoring in fp32 from bf16 weights, and a selection bias at the release's scale, where bf16 erases it."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn
from transformers import NemotronHConfig, NemotronHForCausalLM
from transformers.models.nemotron_h.modeling_nemotron_h import NemotronHMoE
from trl.trainer.utils import selective_log_softmax

from halo.checkpoint.parallel.ep import _reject_unmergeable_hub_keys
from halo.distributed.context_parallel.config import CPConfigError
from halo.distributed.context_parallel.model_rules import reject_unsupported_ulysses_model
from halo.distributed.expert_parallel.deepep.capacity import padded_wire_hidden
from halo.distributed.expert_parallel.families.nemotron_h import EPNemotronHMoELayer
from halo.distributed.expert_parallel.families.roster import MOE_LAYER_MAP
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.expert_parallel.parallelize import apply_ep_layers, ep_claimed_blocks
from halo.distributed.tensor_parallel.parallelize_attention import reject_unshardable_tp_model
from halo.models.moe.balancing import accepts_native_balancing_bias, resolve_balancing_mode, severs_aux_loss
from halo.models.moe.expert_layout import ExpertLayout, ExpertLayoutKind, to_hub_expert_tensors
from halo.models.moe.family_specs import (
    NEMOTRON_H,
    find_moe_family_spec,
    find_moe_family_spec_for_config,
    per_expert_hub_model_types,
    wraps_experts,
)
from halo.trainers.grpo.logprob_scoring import LogprobScoringMixin
from halo.trainers.grpo.rollout.weight_push import reject_unsyncable_model
from tests.common.checkpoint_io import safetensors_state_dict
from tests.common.ep_reference import stores_expert_tp_split_experts, zero_filled_expert_banks
from tests.common.ep_stubs import stubbed_rank_ep_layer
from tests.common.models import NEMOTRON_H_RELEASE_TEXT_CONFIG, TINY_NEMOTRON_H_CONFIG
from tests.common.nemotron_h import wire_width_checks
from tests.common.parallelism import single_process_ep_config
from tests.common.rosters import ep_layer_class_by_model_type
from tests.common.tiny_models import (
    TINY_MOE_FAMILIES,
    TINY_MOE_VLM_FAMILIES,
    build_tiny_family_checkpoint,
    fill_release_scale_bias,
    tiny_family_model,
)
from tests.cpu.kernels.test_grouped_mm_backward import _reference_grouped_mm

E = TINY_NEMOTRON_H_CONFIG["n_routed_experts"]
H = TINY_NEMOTRON_H_CONFIG["hidden_size"]
L = TINY_NEMOTRON_H_CONFIG["moe_latent_size"]
M = TINY_NEMOTRON_H_CONFIG["moe_intermediate_size"]
TOKENS = (2, 32)
# ~6x the measured worst case, as the roster's own comparison; a real disagreement clears it by orders of magnitude.
_RELATIVE_BAND = 2e-6
_FAMILIES = {"text": TINY_MOE_FAMILIES["nemotron_h"], "composite": TINY_MOE_VLM_FAMILIES["nemotron_h_omni"]}
_SHARED_AND_LATENT = ("shared_experts", "fc1_latent_proj", "fc2_latent_proj")


def _block(dtype: torch.dtype = torch.float32) -> NemotronHMoE:
    """Live weights, as random init leaves the router near a tie, and the bias at fp32 whatever the dtype."""
    torch.manual_seed(0)
    block = NemotronHMoE(NemotronHConfig(**TINY_NEMOTRON_H_CONFIG), layer_idx=1).to(dtype)
    for parameter in block.parameters():
        nn.init.normal_(parameter, std=0.2)
    fill_release_scale_bias(block)
    return block


def _wrapped_block(dtype: torch.dtype = torch.float32, **ep_config) -> tuple[NemotronHMoE, EPNemotronHMoELayer]:
    """``(untouched reference block, EP layer around an identical copy)``."""
    block = _block(dtype)
    return copy.deepcopy(block), EPNemotronHMoELayer(block, single_process_ep_config(E, **ep_config)).cpu()


def _tokens(dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.randn(*TOKENS, H, dtype=dtype, generator=torch.Generator().manual_seed(11))


def _same_output(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    assert expected.abs().max().item() > 1e-2, "reference output is ~zero; nothing was compared"
    return (actual - expected).abs().max().item() <= _RELATIVE_BAND * expected.abs().max().item()


def _selected(indices: torch.Tensor) -> torch.Tensor:
    return indices.sort(dim=-1).values


@pytest.fixture
def cuda_autocast_available(monkeypatch):
    """Autocast's state on a host with no GPU, as CUDA tensors would read it. CPU tensors skip its dispatch key."""
    monkeypatch.setattr(torch.cuda.amp.common, "amp_definitely_not_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)


class _Fp32UnderAutocast(nn.Module):
    """What autocast makes of relu2's square on a GPU."""

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        squared = F.relu(tokens).square()
        return squared.float() if torch.is_autocast_enabled("cuda") else squared


def test_registrations():
    assert MOE_LAYER_MAP["NemotronHMoE"] is EPNemotronHMoELayer
    registry = ep_layer_class_by_model_type()
    assert registry["nemotron_h"] is registry["nemotron_h_omni"] is EPNemotronHMoELayer
    assert find_moe_family_spec("nemotron_h") is find_moe_family_spec("nemotron_h_omni") is NEMOTRON_H


@pytest.mark.parametrize("route", sorted(_FAMILIES))
def test_a_tiny_row_starts_from_the_releases_bias(route):
    """Random init zeroes it, and a row at zero sees neither its selection nor its precision."""
    biases = [
        value
        for name, value in tiny_family_model(_FAMILIES[route]).state_dict().items()
        if name.endswith("e_score_correction_bias")
    ]
    assert biases and all(bias.dtype == torch.float32 and bias.min() > 40.0 for bias in biases)
    assert all(not torch.equal(bias, bias.bfloat16().float()) for bias in biases)


@pytest.mark.parametrize(("fp32_pins", "dtype"), [(True, torch.float32), (False, torch.bfloat16)])
def test_a_tiny_checkpoint_stores_the_bias_as_asked(tmp_path, fp32_pins, dtype):
    """A release stores it in fp32, and a checkpoint cast to bf16 would have rounded it before any load could."""
    build_tiny_family_checkpoint(TINY_MOE_FAMILIES["nemotron_h"], str(tmp_path), fp32_pins=fp32_pins)
    stored = safetensors_state_dict(str(tmp_path))
    biases = [value for name, value in stored.items() if name.endswith("e_score_correction_bias")]
    assert biases and all(bias.dtype == dtype for bias in biases)
    assert stored["backbone.norm_f.weight"].dtype == torch.bfloat16


@pytest.mark.parametrize("route", sorted(_FAMILIES))
def test_both_routes_wrap_every_moe_layer_and_adopt_every_parameter(route):
    model = tiny_family_model(_FAMILIES[route])
    assert find_moe_family_spec_for_config(model.config) is NEMOTRON_H
    assert wraps_experts(model.config, ep_layers_requested=True)
    expected = len([kind for kind in TINY_NEMOTRON_H_CONFIG["layers_block_type"] if kind == "moe"])
    assert len(ep_claimed_blocks(model)) == expected
    parameters = sum(parameter.numel() for parameter in model.parameters())

    apply_ep_layers(model, single_process_ep_config())

    layers = named_ep_layers(model)
    assert len(layers) == expected and all(type(layer) is EPNemotronHMoELayer for _, layer in layers)
    assert sum(parameter.numel() for parameter in model.parameters()) == parameters


@pytest.mark.parametrize("route", sorted(_FAMILIES))
def test_the_chunked_path_scores_the_ep_model_as_its_forward(route):
    """Both routes enter the backbone directly past their forward, and the gradient reaches every rank-local bank."""
    torch.manual_seed(0)
    model = tiny_family_model(_FAMILIES[route]).float().eval()
    apply_ep_layers(model, single_process_ep_config())
    ids = torch.randint(0, TINY_NEMOTRON_H_CONFIG["vocab_size"], (2, 16), generator=torch.Generator().manual_seed(1))
    mask, keep = torch.ones_like(ids), 6
    with torch.no_grad():
        logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    forward = selective_log_softmax(logits[:, :-1][:, -keep:].float(), ids[:, -keep:])

    scorer = LogprobScoringMixin()
    scorer.temperature = 1.0
    logps, _ = scorer._chunked_logps_impl(model, ids, mask, keep, 2, False)
    torch.testing.assert_close(logps, forward, atol=1e-5, rtol=1e-5)
    logps.sum().backward()
    banks = [bank for _, layer in named_ep_layers(model) for bank in (layer.up_proj, layer.down_proj)]
    assert banks and all(bank.grad is not None and bool((bank.grad != 0).any()) for bank in banks)
    assert bool((model.get_output_embeddings().weight.grad != 0).any())


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64], ids=["fp32", "fp64"])
def test_every_parameter_gradient_matches_the_block(dtype):
    """The roster's comparison stops at the input gradient; the latent projections and the router sit past it."""
    reference, layer = _wrapped_block(dtype)
    tokens = _tokens(dtype)
    expected, actual = reference(tokens), layer(tokens)
    assert _same_output(actual, expected)
    expected.pow(2).sum().backward()
    actual.pow(2).sum().backward()

    pairs = {
        "up_proj": (layer.up_proj.grad.transpose(1, 2), reference.experts.up_proj.grad),
        "down_proj": (layer.down_proj.grad.transpose(1, 2), reference.experts.down_proj.grad),
        "gate.weight": (layer.gate.weight.grad, reference.gate.weight.grad),
    }
    for name in _SHARED_AND_LATENT:
        own, stock = getattr(layer, name), getattr(reference, name)
        for (key, parameter), (_, stock_parameter) in zip(
            own.named_parameters(), stock.named_parameters(), strict=True
        ):
            pairs[f"{name}.{key}"] = (parameter.grad, stock_parameter.grad)
    assert len(pairs) == 3 + 4, sorted(pairs)
    for name, (got, want) in pairs.items():
        assert want.abs().max().item() > 0, f"{name}: the reference gradient is zero"
        assert (got - want).abs().max().item() <= _RELATIVE_BAND * want.abs().max().item(), name


def test_a_block_with_no_latent_runs_its_experts_on_the_hidden_state():
    """``moe_latent_size`` is optional upstream, and Identity projections hold nothing to replicate."""
    torch.manual_seed(0)
    block = NemotronHMoE(NemotronHConfig(**{**TINY_NEMOTRON_H_CONFIG, "moe_latent_size": None}), layer_idx=1)
    for parameter in block.parameters():
        nn.init.normal_(parameter, std=0.2)
    reference = copy.deepcopy(block)
    layer = EPNemotronHMoELayer(block, single_process_ep_config(E)).cpu()
    assert layer.hidden_dim == H
    assert not [name for name, _ in layer.replicated_named_params() if "latent" in name]
    tokens = _tokens()
    assert _same_output(layer(tokens), reference(tokens))


def test_a_rank_adopts_its_own_experts_or_the_slice_the_loader_handed_it():
    reference = _block()
    for rank in range(2):
        layer = stubbed_rank_ep_layer(EPNemotronHMoELayer, _block(), world_size=2, ep_size=2, rank=rank, grouped=False)
        local = slice(rank * E // 2, (rank + 1) * E // 2)
        assert torch.equal(layer.up_proj.transpose(1, 2), reference.experts.up_proj[local])
        assert torch.equal(layer.down_proj.transpose(1, 2), reference.experts.down_proj[local])

    sliced = _block()
    sliced.experts.up_proj.data = sliced.experts.up_proj.data[E // 2 :]
    sliced.experts.down_proj.data = sliced.experts.down_proj.data[E // 2 :]
    layer = stubbed_rank_ep_layer(
        EPNemotronHMoELayer, sliced, world_size=2, ep_size=2, rank=1, grouped=False, weights_already_sharded=True
    )
    assert torch.equal(layer.up_proj.transpose(1, 2), reference.experts.up_proj[E // 2 :])


def test_rotated_experts_diverge():
    """The comparison reads each expert's own slot, so the adoption cannot swap or offset them unnoticed."""
    reference, layer = _wrapped_block()
    with torch.no_grad():
        layer.up_proj.copy_(layer.up_proj.roll(1, 0))
        layer.down_proj.copy_(layer.down_proj.roll(1, 0))
        assert not _same_output(layer(_tokens()), reference(_tokens()))


def test_a_zeroed_bias_diverges():
    """Selection reads the release's correction, which the router's gate weights never see."""
    reference, layer = _wrapped_block()
    layer.gate.e_score_correction_bias.zero_()
    with torch.no_grad():
        assert not _same_output(layer(_tokens()), reference(_tokens()))


def test_the_bias_selects_as_the_router_and_a_bf16_cast_would_not():
    """At 42 bf16 steps by 0.25, more than the bias's whole spread, so a lost fp32 pin re-routes tokens."""
    reference, layer = _wrapped_block()
    assert layer.gate.e_score_correction_bias.dtype == torch.float32
    flat = _tokens().reshape(-1, H)
    _, _, router_indices = reference.gate(flat)
    indices, _ = layer._route_hidden_states(flat)
    assert torch.equal(_selected(indices), _selected(router_indices))

    layer.gate.e_score_correction_bias.copy_(layer.gate.e_score_correction_bias.to(torch.bfloat16))
    cast, _ = layer._route_hidden_states(flat)
    assert (_selected(cast) != _selected(router_indices)).any(dim=-1).float().mean() > 0.1


def test_the_gate_weights_are_the_routers_own_in_fp32():
    """Bf16 logits select the same experts at this scale, so only the weights can tell the two apart."""
    reference, layer = _wrapped_block(torch.bfloat16)
    flat = _tokens(torch.bfloat16).reshape(-1, H)
    _, router_weights, router_indices = reference.gate(flat)
    indices, weights = layer._route_hidden_states(flat)
    assert weights.dtype == torch.float32
    assert torch.equal(indices, router_indices)
    assert torch.equal(weights, router_weights)

    in_bf16 = layer._linear_router_logits(flat)
    assert in_bf16.dtype == torch.bfloat16
    bf16_scores = in_bf16.float().sigmoid().gather(1, indices)
    scale = layer.routed_scaling_factor
    assert not torch.equal(scale * bf16_scores / (bf16_scores.sum(-1, keepdim=True) + 1e-20), router_weights)


def test_the_router_module_never_runs(monkeypatch):
    """It would select a second time per layer, and it is what makes ``SEVERS_AUX_LOSS`` a true statement."""
    _, layer = _wrapped_block()

    def refuse(*_args, **_kwargs):
        raise AssertionError("the layer called the router's own forward")

    monkeypatch.setattr(layer.gate, "forward", refuse)
    layer(_tokens())


def test_the_wire_carries_the_width_its_buffer_is_sized_for():
    """``hidden_dim`` sizes DeepEP's arena and its wire padding, so the dispatch has to hand it exactly that width."""
    _, layer = _wrapped_block()
    assert layer.dispatcher.hidden_dim == L != H
    dispatched = []
    dispatch = layer.dispatcher.dispatch

    def spy(x, *args, **kwargs):
        dispatched.append(x.shape[-1])
        return dispatch(x, *args, **kwargs)

    layer.dispatcher.dispatch = spy
    layer(_tokens())
    assert dispatched == [layer.dispatcher.hidden_dim]


def test_the_release_dispatches_its_latent_at_the_wire_alignment():
    config = NemotronHConfig(**NEMOTRON_H_RELEASE_TEXT_CONFIG)
    with torch.device("meta"):
        block = NemotronHMoE(config, layer_idx=1)
    latent = EPNemotronHMoELayer._detect_hidden_dim(block.gate, block.experts)
    assert latent == config.moe_latent_size == 1024
    assert padded_wire_hidden(latent) == latent
    assert block.gate.weight.shape[1] == 4096, "premise: the router reads a wider hidden state than the wire"


def test_the_grouped_path_matches_the_block_with_the_kernels_dtype_contract(monkeypatch):
    """The stand-in refuses mixed operands as the kernel does, so a layer running its experts through the loop alone, or
    one whose activation leaves bf16, cannot pass here."""
    monkeypatch.setattr(F, "grouped_mm", _reference_grouped_mm)
    reference, layer = _wrapped_block()
    layer._use_grouped_mm = True
    calls = []
    compute = layer._grouped_expert_ffn

    def spy(*args, **kwargs):
        calls.append(args[0].shape[-1])
        return compute(*args, **kwargs)

    layer._grouped_expert_ffn = spy
    tokens = _tokens()
    actual, expected = layer(tokens), reference(tokens)
    assert calls, "the grouped path never ran"
    assert set(calls) == {L}, f"the grouped FFN read widths {set(calls)}, not the latent"
    assert _same_output(actual, expected)


def test_the_grouped_path_at_bf16_holds_under_autocast(monkeypatch, cuda_autocast_available):
    """A relu2 under autocast comes back in fp32, which the next ``grouped_mm`` refuses at the first forward."""
    monkeypatch.setattr(F, "grouped_mm", _reference_grouped_mm)
    _, layer = _wrapped_block(torch.bfloat16)
    layer._use_grouped_mm = True
    layer.act_fn = _Fp32UnderAutocast()
    assert layer(_tokens(torch.bfloat16)).dtype == torch.bfloat16


def test_autocast_would_break_the_grouped_path(monkeypatch, cuda_autocast_available):
    """The control: the same fp32 activation, called with autocast left on, hits the kernel's refusal."""
    monkeypatch.setattr(F, "grouped_mm", _reference_grouped_mm)
    _, layer = _wrapped_block(torch.bfloat16)
    layer._use_grouped_mm = True
    layer.act_fn = _Fp32UnderAutocast()
    monkeypatch.setattr(EPNemotronHMoELayer, "_expert_activation", lambda self, tokens: self.act_fn(tokens))
    with pytest.raises(RuntimeError, match="matching operand dtypes"):
        layer(_tokens(torch.bfloat16))


def test_the_activation_warm_up_runs_with_autocast_off(cuda_autocast_available):
    """An outer autocast (the LoRA trainers') would otherwise warm the fp32 kernel the compute never launches."""
    _, layer = _wrapped_block(torch.bfloat16)
    layer.act_fn = _Fp32UnderAutocast()
    tokens = torch.zeros(4, M, dtype=torch.bfloat16)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        assert torch.is_autocast_enabled("cuda"), "premise: autocast is on around the warm-up"
        assert layer._warm_expert_activation(tokens).dtype == torch.bfloat16


def test_the_export_writes_every_expert_in_the_release_shape():
    reference, layer = _wrapped_block()
    exported = layer.gather_expert_state_dict()
    assert set(exported) == {f"experts.{e}.{name}_proj.weight" for e in range(E) for name in ("up", "down")}
    for expert in range(E):
        assert torch.equal(exported[f"experts.{expert}.up_proj.weight"], reference.experts.up_proj[expert])
        assert torch.equal(exported[f"experts.{expert}.down_proj.weight"], reference.experts.down_proj[expert])
    assert exported["experts.0.up_proj.weight"].shape == (M, L)
    assert exported["experts.0.down_proj.weight"].shape == (L, M)


def test_the_whole_layer_exports_the_release_keys():
    reference, layer = _wrapped_block()
    gathered = layer.gather_layer_state_dict("mixer")
    expected = {f"mixer.experts.{e}.{name}_proj.weight" for e in range(E) for name in ("up", "down")} | {
        f"mixer.{key}" for key in reference.state_dict() if not key.startswith("experts.")
    }
    assert set(gathered) == expected
    assert torch.equal(gathered["mixer.gate.e_score_correction_bias"], reference.gate.e_score_correction_bias)


def test_the_reference_comparison_reads_the_banks_as_the_stock_container_holds_them():
    """What the GPU rows score the expert gradients with, and what expert TP's storage check reads."""
    reference, layer = _wrapped_block()
    tokens = _tokens()
    reference(tokens).pow(2).sum().backward()
    layer(tokens).pow(2).sum().backward()
    banks = zero_filled_expert_banks(layer)
    assert set(banks) == {"up_proj", "down_proj"}
    for name, bank in banks.items():
        want = getattr(reference.experts, name).grad
        assert (bank - want).abs().max().item() <= _RELATIVE_BAND * want.abs().max().item(), name
    assert stores_expert_tp_split_experts(layer)
    fused = copy.copy(layer)
    fused.gate_up_proj = nn.Parameter(torch.zeros(1))
    assert not stores_expert_tp_split_experts(fused)


def test_the_layout_refuses_a_gate_it_has_no_branch_for():
    """A family's experts are all a layout writes, so a third tensor would be dropped from the export unseen."""
    layout = ExpertLayout(ExpertLayoutKind.SEPARATE_MLP)
    with pytest.raises(ValueError, match="unexpected expert params"):
        to_hub_expert_tensors(
            layout, dict.fromkeys(("gate_proj", "up_proj", "down_proj"), torch.zeros(1, 1, 1)), device="cpu"
        )


def test_the_unfuse_tool_has_no_layout_to_write():
    """The gathered save is already per expert, and a triple would name a gate this family does not have."""
    assert NEMOTRON_H.expert_layout.per_expert_keys is None
    assert not set(NEMOTRON_H.model_types) & set(per_expert_hub_model_types())


def _layer_on_the_wire(width: int, fc1: nn.Module, fc2: nn.Module) -> SimpleNamespace:
    return SimpleNamespace(dispatcher=SimpleNamespace(hidden_dim=width), fc1_latent_proj=fc1, fc2_latent_proj=fc2)


def test_the_wire_check_reads_the_rows_width_not_the_models():
    """A no-latent row whose override stopped reaching the model must fail, and the latent row on a model that lost its
    latent, where each would otherwise hold the width to itself."""
    latent_wire = [_layer_on_the_wire(L, nn.Linear(H, L), nn.Linear(L, H))]
    hidden_wire = [_layer_on_the_wire(H, nn.Identity(), nn.Identity())]
    assert wire_width_checks(hidden_wire, hidden_size=H, latent=None) == {"hidden_state_on_the_wire": True}
    assert wire_width_checks(latent_wire, hidden_size=H, latent=L) == {"latent_on_the_wire": True}

    assert wire_width_checks(latent_wire, hidden_size=H, latent=None) == {"hidden_state_on_the_wire": False}
    assert wire_width_checks(hidden_wire, hidden_size=H, latent=L) == {"latent_on_the_wire": False}


@pytest.mark.parametrize("stray", ["fc1", "fc2"])
def test_the_no_latent_wire_check_refuses_a_projection_left_around_the_dispatch(stray):
    projection = {"fc1": nn.Identity(), "fc2": nn.Identity(), stray: nn.Linear(H, H)}
    layers = [_layer_on_the_wire(H, projection["fc1"], projection["fc2"])]
    assert wire_width_checks(layers, hidden_size=H, latent=None) == {"hidden_state_on_the_wire": False}


def test_expert_tp_splits_the_intermediate_and_its_partial_outputs_sum_to_the_whole():
    """Relu2 is elementwise, so a partner's slice of the intermediate axis is its own term of the FFN."""
    _, whole = _wrapped_block()
    ranks = [
        stubbed_rank_ep_layer(EPNemotronHMoELayer, _block(), world_size=2, ep_size=1, rank=rank, grouped=False)
        for rank in range(2)
    ]
    assert [layer.expert_tp_rank for layer in ranks] == [0, 1]
    assert ranks[0].up_proj.shape == (E, L, M // 2)
    assert ranks[0].down_proj.shape == (E, M // 2, L)
    latent = torch.randn(5, L, generator=torch.Generator().manual_seed(2))
    for expert in range(E):
        summed = sum(layer._expert_forward(expert, latent) for layer in ranks)
        assert torch.allclose(summed, whole._expert_forward(expert, latent), atol=1e-5, rtol=0), expert

    gathered = {
        name: torch.cat([getattr(layer, name).detach() for layer in ranks], dim=whole.EXPERT_TP_SHARD_DIMS[name])
        for name in ("up_proj", "down_proj")
    }
    assert torch.equal(gathered["up_proj"], whole.up_proj)
    assert torch.equal(gathered["down_proj"], whole.down_proj)


def test_an_intermediate_expert_tp_cannot_divide_is_refused():
    """A floored shard would drop the top units of every expert."""
    with pytest.raises(ValueError, match="divisible by"):
        stubbed_rank_ep_layer(
            EPNemotronHMoELayer,
            NemotronHMoE(NemotronHConfig(**{**TINY_NEMOTRON_H_CONFIG, "moe_intermediate_size": M + 1}), layer_idx=1),
            world_size=2,
            ep_size=1,
            rank=0,
            grouped=False,
        )


def test_every_trainable_parameter_has_a_gradient_sync():
    """The latent projections sit outside the expert roots and the router, and FSDP2 ignores the whole layer, so an
    unlisted one drifts across data-parallel replicas."""
    layer = stubbed_rank_ep_layer(EPNemotronHMoELayer, _block(), world_size=4, ep_size=2, rank=0, grouped=False)
    names = {name for name, _ in layer.replicated_named_params()}
    assert {"fc1_latent_proj.weight", "fc2_latent_proj.weight"} <= names
    assert any(name.startswith("shared_experts.") for name in names)
    trainable = {id(parameter) for parameter in layer.parameters() if parameter.requires_grad}
    assert layer.ep_config.uses_backward_grad_hooks, "premise: the hooks, not the sweep, own this shape"
    assert layer.synced_trainable_param_ids() == trainable


@pytest.mark.parametrize("route", sorted(_FAMILIES))
def test_balancing_resolves_to_the_exported_bias_on_both_routes(route):
    """The modeling computes no aux loss, and the release's slot exports what the callback trains."""
    model = tiny_family_model(_FAMILIES[route])
    apply_ep_layers(model, single_process_ep_config())
    assert accepts_native_balancing_bias(model)
    assert severs_aux_loss(model)
    assert resolve_balancing_mode("auto", model, is_moe=True) == "bias_update"


def test_a_stock_text_model_resolves_no_aux_loss():
    """A forward that never declares the flag has no aux path to fall back on, wrapped or not."""
    model = NemotronHForCausalLM(NemotronHConfig(**TINY_NEMOTRON_H_CONFIG))
    assert resolve_balancing_mode("auto", model, is_moe=True) == "none"


def test_the_trained_bias_stays_in_the_slot_the_router_reads():
    _, layer = _wrapped_block(torch.bfloat16)
    slot = layer.gate.e_score_correction_bias
    assert layer.enable_bias_balancing() is True
    assert layer.balancing_biases is layer.gate.e_score_correction_bias
    assert layer.balancing_biases.dtype == torch.float32
    assert torch.equal(layer.balancing_biases, slot)
    assert "gate.e_score_correction_bias" in layer.state_dict()


@pytest.mark.parametrize("backend", ["vllm", "sglang"])
@pytest.mark.parametrize("route", sorted(_FAMILIES))
@pytest.mark.parametrize("wrapped", [True, False], ids=["ep_layers", "stock"])
def test_weight_sync_is_refused_on_both_engines(route, backend, wrapped):
    model = tiny_family_model(_FAMILIES[route])
    if wrapped:
        apply_ep_layers(model, single_process_ep_config())
    with pytest.raises(ValueError, match="does not support weight sync"):
        reject_unsyncable_model(model, backend)


@pytest.mark.parametrize("route", sorted(_FAMILIES))
def test_context_parallelism_is_refused_on_the_mamba_layers(route):
    with pytest.raises(CPConfigError, match="linear_attention"):
        reject_unsupported_ulysses_model(tiny_family_model(_FAMILIES[route]), cp_size=2)


@pytest.mark.parametrize("route", sorted(_FAMILIES))
def test_tensor_parallelism_is_refused_on_the_unplanned_attention(route):
    with pytest.raises(ValueError, match="shards ZERO attention layers"):
        reject_unshardable_tp_model(tiny_family_model(_FAMILIES[route]), tp_size=2)


@pytest.mark.parametrize("route", sorted(_FAMILIES))
def test_a_sharded_ep_save_is_refused(route):
    """The merge streams key by key inside the EP layer's spelling, and ``backbone.`` sits outside it."""
    with pytest.raises(ValueError, match="merge_ep_shards"):
        _reject_unmergeable_hub_keys(tiny_family_model(_FAMILIES[route]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
