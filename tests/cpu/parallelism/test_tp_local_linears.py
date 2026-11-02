#!/usr/bin/env python
"""TP linears on their local shards compute what DTensor dispatch computed, bit for bit, on every TP family.

Both TP mechanisms run each colwise / rowwise projection as ``F.linear`` on ``weight.to_local()`` with
explicit c10d collectives (``halo/distributed/tensor_parallel/local_linears.py``) in place of DTensor
dispatch. Two gloo ranks build each tiny-roster TP family twice, once on DTensor dispatch (transformers'
own forward for an HF-native load, torch's ``ColwiseParallel`` / ``RowwiseParallel`` for the selective
attention plan) and once on the local path, then compare the logits, the loss, the gradient into the
decoder stack, every parameter gradient (local shard and placement), the no-grad and inference-mode logits
and a three-step SGD loss trajectory, plain, under FSDP2 at dp 2 x tp 2, under activation checkpointing and
with the dense MLP projections retyped by ``lowp_precision``.
The selective plan's second rank starts from perturbed projections, which its shard must take from rank 0.

Bitwise wherever no rowwise projection carries a bias: the local path adds that bias once after the SUM,
where DTensor added half of it to each rank's partial (GPT-OSS ``o_proj``), so those families compare
to fp32 rounding. The negative controls drop one collective of the local path at a time, hand a rank
its neighbour's slice or shard each rank's own values, and must be caught.

    python tests/cpu/parallelism/test_tp_local_linears.py
"""

from __future__ import annotations

import copy
import functools
import json
from dataclasses import dataclass

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module
from transformers import DistributedConfig
from transformers.distributed.tensor_parallel import _get_parameter_plan

from halo.distributed.fsdp import apply_fsdp2_for_tp
from halo.distributed.mesh import build_dp_tp_mesh, tp_submesh
from halo.distributed.tensor_parallel import hf_plan, local_linears, parallelize_attention
from halo.distributed.tensor_parallel.plain_params import per_head_norm_param_names
from halo.kernels.lowp.linear import LowPrecisionLinear
from halo.kernels.lowp.spec import LOWP_RECIPES, LowpSpec
from halo.model_loading.buffer_repair import finalize_loaded_model
from halo.model_loading.parallel import lowp_compute
from halo.model_loading.parallel import tp as tp_loading
from halo.models.module_tree import backbone_with_layers, decoder_layers
from tests.common.distributed import fake_process_group, fake_process_group_mesh
from tests.common.gloo import run_gloo_ranks
from tests.common.tiny_models import (
    TINY_DENSE_ATTENTION_FAMILIES,
    TINY_DENSE_FAMILIES,
    TINY_MOE_FAMILIES,
    TINY_SLIDING_DENSE_FAMILIES,
    build_tiny_family_checkpoint,
    tiny_family_model,
)

TP = 2
DP = 2
STEPS = 3
LR = 0.1
BATCH, SEQ = 2, 16
# fp32 rounding of the one reassociated bias add, on activations and gradients of order one.
BIAS_REORDER_TOL = 1e-5

# Every dense family the roster carries, on the HF-native plan; qwen3 tied adds the vocab-parallel tie.
HF_FAMILIES = {
    **TINY_SLIDING_DENSE_FAMILIES,
    **TINY_DENSE_ATTENTION_FAMILIES,
    **TINY_DENSE_FAMILIES,
}
HF_OVERRIDES = {"qwen3_tied": ("qwen3", {"tie_word_embeddings": True})}
HF_CASES = sorted([*HF_FAMILIES, *HF_OVERRIDES])


def _hf_styles(model: nn.Module) -> dict[str, str]:
    """Each plain ``nn.Linear`` the model's HF plan styles under a local style, by module name."""
    plan = model._tp_plan
    return {
        name: style
        for name, module in model.named_modules()
        if (style := _get_parameter_plan(name, plan, is_weight=False)) in local_linears.LINEAR_TP_STYLES
        and type(module) is nn.Linear
    }


def _selective_styles(model: nn.Module) -> dict[str, str]:
    """Each attention projection the selective plan shards, with its style, by module name."""
    layers = set(decoder_layers(backbone_with_layers(model)))
    styles = {}
    for layer_name, layer in model.named_modules():
        attn, attn_name = parallelize_attention._find_attention(layer) if layer in layers else (None, None)
        if parallelize_attention._is_tp_shardable(attn):
            for proj, style in parallelize_attention._attention_tp_plan(attn).items():
                styles[f"{layer_name}.{attn_name}.{proj}"] = style
    return styles


def _has_rowwise_bias(model: nn.Module, styles: dict[str, str]) -> bool:
    return any(
        not local_linears.LINEAR_TP_STYLES[style].colwise
        and getattr(model.get_submodule(name), "bias", None) is not None
        for name, style in styles.items()
    )


def _expectation(model: nn.Module, styles: dict[str, str]) -> dict:
    """The styles a comparison runs and whether it compares bitwise, as its result reports them."""
    return {"styles": sorted(set(styles.values())), "bitwise": not _has_rowwise_bias(model, styles)}


def _meta_model(family, overrides: dict | None = None) -> nn.Module:
    with torch.device("meta"):
        return tiny_family_model(family, overrides=overrides)


def _hf_expected(case: str) -> dict:
    family_key, overrides = HF_OVERRIDES.get(case, (case, None))
    model = _meta_model(HF_FAMILIES[family_key], overrides)
    return _expectation(model, _hf_styles(model))


def _selective_expected(model_type: str) -> dict:
    model = _meta_model(TINY_MOE_FAMILIES[model_type])
    return _expectation(model, _selective_styles(model))


# The remote-code Bailing models, whose attention the plan names no class of, are not built.
SELECTIVE_CASES = sorted(
    model_type
    for model_type, family in TINY_MOE_FAMILIES.items()
    if not family.trust_remote_code and _selective_styles(_meta_model(family))
)


@dataclass(frozen=True)
class Variant:
    """How a comparison runs: ``dp`` FSDP2 replicas over the TP pair, ``gc`` activation checkpointing
    (``"reentrant"`` / ``"non_reentrant"``), ``lowp`` the ``lowp_precision`` the dense MLP projections are
    retyped to, ``broken`` a negative control from :data:`_BREAKS`."""

    dp: int = 1
    gc: str | None = None
    lowp: str | None = None
    broken: str | None = None


PLAIN = Variant()


class _Identity(torch.autograd.Function):
    """A collective-shaped no-op: identity forward and backward."""

    @staticmethod
    def forward(ctx, x, group):
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def _neighbour_shard(x, rank, world):
    return x.chunk(world, dim=-1)[(rank + 1) % world].contiguous()


# Each breaks one collective of the local path; the comparison must report it.
_BREAKS = {
    "input_grad_sum": ("SumGradAcrossGroup", _Identity),
    "output_sum": ("_SumAcrossGroup", _Identity),
    "neighbour_shard": ("_last_dim_shard", _neighbour_shard),
    "own_values": ("distribute_tensor", functools.partial(distribute_tensor, src_data_rank=None)),
}


def _break(broken: str | None) -> None:
    if broken is not None:
        attr, replacement = _BREAKS[broken]
        setattr(local_linears, attr, replacement)


def _randomize_biases(model: nn.Module) -> None:
    """Random-init zeroes every bias, which would hide where the rowwise bias is added; a release carries them
    learned."""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.endswith("bias"):
                param.normal_(std=0.5)


def _local(t: torch.Tensor | None) -> torch.Tensor | None:
    return t.to_local() if isinstance(t, DTensor) else t


def _forward_backward(model: nn.Module, ids: torch.Tensor) -> dict:
    """One forward and backward: logits, loss, the gradient into the decoder stack and every parameter's."""
    captured = {}

    def keep_grad(_module, _args, out):
        out.register_hook(lambda grad: captured.setdefault("stack_input_grad", grad.detach().clone()))

    handle = model.get_input_embeddings().register_forward_hook(keep_grad)
    out = model(input_ids=ids, labels=ids)
    out.loss.backward()
    handle.remove()
    result = {"logits": out.logits.detach(), "loss": out.loss.detach(), **captured}
    for name, param in model.named_parameters():
        result[f"grad:{name}"] = _local(param.grad)
        result[f"placement:{name}"] = str(getattr(param.grad, "placements", None))
    return result


def _train(model: nn.Module, ids: torch.Tensor, tp_group) -> list[torch.Tensor]:
    """``STEPS`` SGD steps, the per-head norms SUMmed over the TP group as the trainer does; the losses."""
    optimizer = torch.optim.SGD(model.parameters(), lr=LR, foreach=False)
    summed = set(per_head_norm_param_names(model))
    losses = []
    for _ in range(STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        for name, param in model.named_parameters():
            if name in summed:
                dist.all_reduce(_local(param.grad), group=tp_group)
        optimizer.step()
        losses.append(loss.detach())
    return losses


def _prepare(model: nn.Module, variant: Variant) -> None:
    if variant.gc is not None:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": variant.gc == "reentrant"})
        model.train()


def _compare(reference: nn.Module, local: nn.Module, tp_mesh, *, bitwise: bool, dp_rank: int = 0) -> dict:
    """``{"mismatches": [...], "max_abs": float}`` of ``local`` against ``reference`` on this DP rank's batch."""
    vocab = reference.config.get_text_config().vocab_size
    ids = torch.randint(0, vocab, (BATCH, SEQ), generator=torch.Generator().manual_seed(dp_rank))
    expected, actual = {}, {}
    for mode in (torch.no_grad, torch.inference_mode):
        with mode():
            expected[f"logits:{mode.__name__}"] = reference(input_ids=ids).logits.clone()
            actual[f"logits:{mode.__name__}"] = local(input_ids=ids).logits.clone()
    expected |= _forward_backward(reference, ids)
    actual |= _forward_backward(local, ids)
    tp_group = tp_mesh.get_group()
    expected |= {f"step{i}": loss for i, loss in enumerate(_train(reference, ids, tp_group))}
    actual |= {f"step{i}": loss for i, loss in enumerate(_train(local, ids, tp_group))}

    mismatches, max_abs = [], 0.0
    for key, want in expected.items():
        got = actual.get(key)
        if isinstance(want, str) or want is None or got is None:
            if want != got:
                mismatches.append(f"{key}: {got!r} != {want!r}")
            continue
        diff = (got - want).abs().max().item() if got.numel() else 0.0
        max_abs = max(max_abs, diff)
        if not (torch.equal(got, want) if bitwise else torch.allclose(got, want, rtol=0, atol=BIAS_REORDER_TOL)):
            mismatches.append(f"{key}: max |delta| {diff:.3e}")
    return {"mismatches": mismatches, "max_abs": max_abs, "bitwise": bitwise}


def _compare_counting(reference: nn.Module, local: nn.Module, styles: dict[str, str], tp_mesh, **kwargs) -> dict:
    """:func:`_compare`, also counting the forwards of ``local``'s first rowwise decoder projection, which a
    checkpointed layer reruns in backward."""
    name = next(name for name, style in styles.items() if style == "rowwise" and ".layers." in f".{name}")
    calls = []
    handle = local.get_submodule(name).register_forward_hook(lambda *_: calls.append(1))
    result = _compare(reference, local, tp_mesh, **kwargs)
    handle.remove()
    return result | {"rowwise_calls": len(calls)}


def _localized(model: nn.Module, names) -> list[str]:
    """The modules of ``names`` running the local-shard forward."""
    return [
        name
        for name in names
        if getattr(vars(model.get_submodule(name)).get("forward"), "__module__", None) == local_linears.__name__
    ]


def _lowp_localized(model: nn.Module) -> int:
    """How many low-precision projections run the local-shard forward."""
    return len(
        _localized(model, [name for name, module in model.named_modules() if isinstance(module, LowPrecisionLinear)])
    )


def _retype_lowp(reference: nn.Module, local: nn.Module, precision: str) -> dict:
    """Retype both models' dense MLP projections to ``precision``; the reference's conversion re-installs
    transformers' forward around them, as it did before the local path covered them."""
    spec = LowpSpec(precision=precision)
    local_styles, lowp_compute.LINEAR_TP_STYLES = lowp_compute.LINEAR_TP_STYLES, {}
    try:
        lowp_compute.apply_lowp_compute(reference, spec, expert_weight_cacheable=True)
    finally:
        lowp_compute.LINEAR_TP_STYLES = local_styles
    converted = lowp_compute.apply_lowp_compute(local, spec, expert_weight_cacheable=True)["dense_linears"]
    return {
        "lowp_converted": converted,
        "lowp_localized": _lowp_localized(local),
        "reference_lowp_localized": _lowp_localized(reference),
    }


def _load_hf(checkpoint: str, load_class, mesh, tp_mesh):
    with tp_loading._toolkit_tp_mesh(mesh, tp_mesh):
        model = load_class.from_pretrained(
            checkpoint, distributed_config=DistributedConfig(tp_plan="auto", tp_size=TP), dtype=torch.float32
        )
    finalize_loaded_model(model, weights_dir=checkpoint)
    return model


def _hf_worker(rank: int, checkpoint: str, load_class, variant: Variant, out_path: str) -> None:
    mesh = build_dp_tp_mesh(TP, variant.dp, device_type="cpu")
    tp_mesh = tp_submesh(mesh)
    reference = _load_hf(checkpoint, load_class, mesh, tp_mesh)
    # What the load finish did before the local path: the mesh stamp and the norm-hook retarget.
    reference._device_mesh = mesh
    hf_plan.retarget_hf_replicated_grad_hooks(reference)

    local = _load_hf(checkpoint, load_class, mesh, tp_mesh)
    hf_plan.finish_hf_tp_load(local, mesh, TP)
    _break(variant.broken)

    styles = _hf_styles(local)
    bitwise = not _has_rowwise_bias(local, styles)
    localized = {
        "localized": len(_localized(local, styles)),
        "reference_localized": len(_localized(reference, styles)),
    }
    if variant.lowp is not None:
        localized |= _retype_lowp(reference, local, variant.lowp)
    for model in (reference, local):
        _prepare(model, variant)
        if variant.dp > 1:
            apply_fsdp2_for_tp(model, TP, compute_dtype=torch.float32, fp32_reduce=False, dp_size=variant.dp)
    dp_rank = mesh["dp"].get_local_rank() if variant.dp > 1 else 0
    result = _compare_counting(reference, local, styles, tp_mesh, bitwise=bitwise, dp_rank=dp_rank)
    result |= {"styles": sorted(set(styles.values())), "planned": len(styles), **localized}
    with open(f"{out_path}.{rank}", "w") as fh:
        json.dump(result, fh)


def _dtensor_shard(linear: nn.Linear, style_name: str, tp_mesh, *, name: str) -> None:
    """The selective plan's former sharding: torch's own TP style on the projection."""
    style = ColwiseParallel() if local_linears.LINEAR_TP_STYLES[style_name].colwise else RowwiseParallel()
    parallelize_module(linear, tp_mesh, style)


def _selective_worker(rank: int, model_type: str, variant: Variant, out_path: str) -> None:
    torch.manual_seed(0)
    reference = tiny_family_model(TINY_MOE_FAMILIES[model_type])
    _randomize_biases(reference)
    local = copy.deepcopy(reference)
    mesh = build_dp_tp_mesh(TP, variant.dp, device_type="cpu")
    tp_mesh = tp_submesh(mesh)
    styles = _selective_styles(local)
    if tp_mesh.get_local_rank():
        # Off TP rank 0 the pre-shard values differ, as an unseeded init would leave them.
        with torch.no_grad():
            for name in styles:
                for param in local.get_submodule(name).parameters(recurse=False):
                    param.add_(1.0)

    parallelize_attention.shard_linear, shard_linear = _dtensor_shard, parallelize_attention.shard_linear
    try:
        parallelize_attention.apply_tp_to_attention_only(reference, mesh)
    finally:
        parallelize_attention.shard_linear = shard_linear
    _break(variant.broken)
    parallelize_attention.apply_tp_to_attention_only(local, mesh)
    for model in (reference, local):
        _prepare(model, variant)
        if variant.dp > 1:
            model._device_mesh = mesh
            apply_fsdp2_for_tp(model, TP, compute_dtype=torch.float32, fp32_reduce=False, dp_size=variant.dp)
    dp_rank = mesh["dp"].get_local_rank() if variant.dp > 1 else 0
    bitwise = not _has_rowwise_bias(local, styles)
    result = _compare_counting(reference, local, styles, tp_mesh, bitwise=bitwise, dp_rank=dp_rank)
    result |= {
        "styles": sorted(set(styles.values())),
        "localized": len(_localized(local, styles)),
        "planned": len(styles),
        "reference_localized": len(_localized(reference, styles)),
    }
    with open(f"{out_path}.{rank}", "w") as fh:
        json.dump(result, fh)


def _results(out_path: str, world: int = TP) -> list[dict]:
    results = []
    for rank in range(world):
        with open(f"{out_path}.{rank}") as fh:
            results.append(json.load(fh))
    return results


def _assert_matches(results: list[dict], expected: dict, *, checkpointed: bool = False) -> None:
    """Every rank matched, over the styles and in the comparison mode ``expected`` derives off the family."""
    # Two no-grad forwards, then one forward per backward, plus its recompute under checkpointing.
    forwards = 2 + (1 + STEPS) * (2 if checkpointed else 1)
    for rank, result in enumerate(results):
        assert result["rowwise_calls"] == forwards, f"rank {rank}: {result['rowwise_calls']} forwards, not {forwards}"
        assert result["planned"] > 0, "premise: the plan styles at least one linear"
        assert result["styles"] == expected["styles"], f"rank {rank}: ran {result['styles']}, not {expected}"
        assert result["bitwise"] == expected["bitwise"], f"rank {rank}: comparison mode differs from {expected}"
        assert result["reference_localized"] == 0, "premise: the reference computes through DTensor dispatch"
        assert result["localized"] == result["planned"], f"rank {rank}: {result}"
        assert not result["mismatches"], f"rank {rank}: {result['mismatches']}"
        # A rowwise bias the batch never moved would compare the reassociated add vacuously.
        assert result["bitwise"] or result["max_abs"] > 0, f"rank {rank}: the rowwise bias changed nothing"


@pytest.fixture(scope="module")
def hf_checkpoints(tmp_path_factory):
    def build(case: str) -> str:
        family_key, overrides = HF_OVERRIDES.get(case, (case, None))
        path = tmp_path_factory.mktemp(case)
        build_tiny_family_checkpoint(HF_FAMILIES[family_key], str(path), edit=_randomize_biases, overrides=overrides)
        return str(path)

    return build


@pytest.mark.parametrize("case", HF_CASES)
def test_hf_native_plan_matches_dtensor_dispatch(case, hf_checkpoints, tmp_path):
    family_key = HF_OVERRIDES.get(case, (case,))[0]
    out = str(tmp_path / "result")
    run_gloo_ranks(_hf_worker, TP, hf_checkpoints(case), HF_FAMILIES[family_key].load_class, PLAIN, out)
    _assert_matches(_results(out), _hf_expected(case))


@pytest.mark.parametrize("case", ["qwen3", "qwen3_tied"])
def test_hf_native_plan_under_fsdp2_matches_dtensor_dispatch(case, hf_checkpoints, tmp_path):
    """dp=2 x tp=2: FSDP2 hands the forward its unsharded TP DTensors and reduce-scatters the grads over dp."""
    family_key = HF_OVERRIDES.get(case, (case,))[0]
    out = str(tmp_path / "result")
    variant = Variant(dp=DP)
    run_gloo_ranks(_hf_worker, DP * TP, hf_checkpoints(case), HF_FAMILIES[family_key].load_class, variant, out)
    _assert_matches(_results(out, DP * TP), _hf_expected(case))


@pytest.mark.parametrize("gc", ["reentrant", "non_reentrant"])
def test_hf_native_plan_under_checkpointing_matches_dtensor_dispatch(gc, hf_checkpoints, tmp_path):
    """The recompute reruns the in-place output SUM inside backward."""
    out = str(tmp_path / "result")
    run_gloo_ranks(_hf_worker, TP, hf_checkpoints("qwen3"), HF_FAMILIES["qwen3"].load_class, Variant(gc=gc), out)
    _assert_matches(_results(out), _hf_expected("qwen3"), checkpointed=True)


@pytest.mark.parametrize("precision", sorted(LOWP_RECIPES))
def test_hf_native_plan_with_lowp_mlps_matches_dtensor_dispatch(precision, hf_checkpoints, tmp_path):
    """``lowp_precision`` retypes the MLP projections; the local path computes their fake-quant matmul on each
    rank's shard as transformers' forward did on the DTensors."""
    out = str(tmp_path / "result")
    variant = Variant(lowp=precision)
    run_gloo_ranks(_hf_worker, TP, hf_checkpoints("qwen3"), HF_FAMILIES["qwen3"].load_class, variant, out)
    results = _results(out)
    _assert_matches(results, _hf_expected("qwen3"))
    for rank, result in enumerate(results):
        assert result["lowp_converted"] > 0, "premise: the MLP projections are retyped"
        assert result["reference_lowp_localized"] == 0, "premise: the reference keeps transformers' forward"
        assert result["lowp_localized"] == result["lowp_converted"], f"rank {rank}: {result}"


@pytest.mark.parametrize("model_type", SELECTIVE_CASES)
def test_selective_attention_plan_matches_dtensor_dispatch(model_type, tmp_path):
    out = str(tmp_path / "result")
    run_gloo_ranks(_selective_worker, TP, model_type, PLAIN, out)
    _assert_matches(_results(out), _selective_expected(model_type))


@pytest.mark.parametrize("model_type", ["gpt_oss", "qwen3_moe"])
def test_selective_attention_plan_under_fsdp2_matches_dtensor_dispatch(model_type, tmp_path):
    """dp=2 x tp=2, the EP+TP layout: FSDP2 shards the projections over dp beside the replicated experts."""
    out = str(tmp_path / "result")
    run_gloo_ranks(_selective_worker, DP * TP, model_type, Variant(dp=DP), out)
    _assert_matches(_results(out, DP * TP), _selective_expected(model_type))


@pytest.mark.parametrize("gc", ["reentrant", "non_reentrant"])
def test_selective_attention_plan_under_checkpointing_matches_dtensor_dispatch(gc, tmp_path):
    out = str(tmp_path / "result")
    run_gloo_ranks(_selective_worker, TP, "qwen3_moe", Variant(gc=gc), out)
    _assert_matches(_results(out), _selective_expected("qwen3_moe"), checkpointed=True)


def test_the_roster_covers_every_style_and_a_rowwise_bias():
    """Anti-vacuity: the HF cases run all four local styles (only OLMo 3 splits a rowwise input), the selective
    cases include both MLA families, and GPT-OSS's biased ``o_proj`` runs the tolerance mode; every comparison
    asserts it ran the styles and mode derived here."""
    covered = set().union(*(_hf_expected(case)["styles"] for case in HF_CASES))
    assert covered == {"colwise", "colwise_gather_output", "rowwise", "rowwise_split_input"}, covered
    assert covered == set(local_linears.LINEAR_TP_STYLES), sorted(local_linears.LINEAR_TP_STYLES)
    assert {"gpt_oss", "qwen3_moe", "glm4_moe_lite", "mistral4"} <= set(SELECTIVE_CASES), SELECTIVE_CASES
    assert not _selective_expected("gpt_oss")["bitwise"], "GPT-OSS's o_proj carries no bias"


@pytest.mark.parametrize(
    ("case", "broken"), [("qwen3", "input_grad_sum"), ("qwen3", "output_sum"), ("olmo3", "neighbour_shard")]
)
def test_negative_control_hf_breaks_are_caught(case, broken, hf_checkpoints, tmp_path):
    out = str(tmp_path / "result")
    run_gloo_ranks(_hf_worker, TP, hf_checkpoints(case), HF_FAMILIES[case].load_class, Variant(broken=broken), out)
    for result in _results(out):
        assert result["mismatches"], f"{broken} went unnoticed"


@pytest.mark.parametrize("broken", ["input_grad_sum", "output_sum", "own_values"])
def test_negative_control_selective_breaks_are_caught(broken, tmp_path):
    out = str(tmp_path / "result")
    run_gloo_ranks(_selective_worker, TP, "qwen3_moe", Variant(broken=broken), out)
    for result in _results(out):
        assert result["mismatches"], f"{broken} went unnoticed"


def _placed(linear: nn.Linear, mesh, weight, bias) -> nn.Linear:
    """``linear`` with its weight and bias distributed into the given placements, as a load leaves them."""
    for pname, placement in (("weight", weight), ("bias", bias)):
        param = getattr(linear, pname)
        setattr(linear, pname, nn.Parameter(distribute_tensor(param, mesh, [placement], src_data_rank=None)))
    return linear


@pytest.mark.parametrize("style_name", sorted(local_linears.LINEAR_TP_STYLES))
def test_install_accepts_the_style_placements(style_name):
    style = local_linears.LINEAR_TP_STYLES[style_name]
    with fake_process_group_mesh(rank=0, world_size=TP) as mesh:
        linear = _placed(nn.Linear(8, 8), mesh, **style.placements)
        local_linears.install_local_forward(linear, style_name)
        assert vars(linear)["forward"].__module__ == local_linears.__name__


@pytest.mark.parametrize(
    ("style_name", "weight", "bias"),
    [("colwise", Shard(1), Shard(0)), ("rowwise", Shard(1), Shard(0)), ("rowwise", Shard(0), Replicate())],
)
def test_install_refuses_a_layout_the_style_does_not_name(style_name, weight, bias):
    """A placement the forward was not written for would compute the wrong slice of the matmul."""
    with fake_process_group_mesh(rank=0, world_size=TP) as mesh:
        linear = _placed(nn.Linear(8, 8), mesh, weight, bias)
        with pytest.raises(ValueError, match=f"TP style '{style_name}'"):
            local_linears.install_local_forward(linear, style_name)
        assert "forward" not in vars(linear)


def test_install_refuses_plain_parameters():
    with pytest.raises(ValueError, match="plain tensor"):
        local_linears.install_local_forward(nn.Linear(8, 8), "colwise")


def test_install_refuses_a_linear_with_its_own_forward():
    class _Scaled(nn.Linear):
        def forward(self, x):
            return 2 * super().forward(x)

    with fake_process_group_mesh(rank=0, world_size=TP) as mesh:
        linear = _placed(_Scaled(8, 8), mesh, Shard(0), Shard(0))
        with pytest.raises(TypeError, match="forward of its own"):
            local_linears.install_local_forward(linear, "colwise")


@pytest.mark.parametrize(
    ("style_name", "in_features", "out_features"), [("colwise_gather_output", 8, 6), ("rowwise_split_input", 6, 8)]
)
def test_install_refuses_a_full_width_dim_the_group_does_not_divide(style_name, in_features, out_features):
    """The gather and the split move equal slices; an uneven one would fail the first collective instead."""
    style = local_linears.LINEAR_TP_STYLES[style_name]
    with fake_process_group_mesh(rank=0, world_size=4) as mesh:
        linear = _placed(nn.Linear(in_features, out_features), mesh, **style.placements)
        with pytest.raises(ValueError, match="does not divide"):
            local_linears.install_local_forward(linear, style_name)


def test_rowwise_bias_keeps_the_matmul_dtype_under_autocast():
    """An fp32 rowwise bias under bf16 autocast leaves the output bf16, as a biased ``F.linear`` (DTensor's
    ``addmm``) returns it, rather than promoting the SUM to fp32."""
    with fake_process_group_mesh(rank=0, world_size=TP) as mesh:
        linear = _placed(nn.Linear(8, 8), mesh, **local_linears.LINEAR_TP_STYLES["rowwise"].placements)
        local_linears.install_local_forward(linear, "rowwise")
        with torch.autocast("cpu", dtype=torch.bfloat16):
            out = linear(torch.randn(2, 8 // TP))
    assert linear.bias.dtype == torch.float32
    assert out.dtype == torch.bfloat16


def test_a_deepcopy_computes_with_its_own_parameters():
    """The forward is bound to its module, so a copy's projection reads the copy's shards and its gradient
    lands on them, not on the original's."""
    with fake_process_group_mesh(rank=0, world_size=TP) as mesh:
        linear = _placed(nn.Linear(8, 8), mesh, **local_linears.LINEAR_TP_STYLES["colwise"].placements)
        local_linears.install_local_forward(linear, "colwise")
        clone = copy.deepcopy(linear)
        with torch.no_grad():
            clone.weight.to_local().mul_(2)
        x = torch.randn(2, 8)
        out = clone(x)
        out.sum().backward()
        want = nn.functional.linear(x, clone.weight.to_local().detach(), clone.bias.to_local().detach())
    assert torch.equal(out.detach(), want)
    assert clone.weight.grad is not None and linear.weight.grad is None


def test_forward_refuses_parameters_fsdp2_left_sharded():
    """Called past its FSDP unit's unshard, a projection holds FSDP2's (dp, tp) shard, whose local slice would
    compute part of the output; it raises instead."""
    with fake_process_group(rank=0, world_size=DP * TP):
        mesh = build_dp_tp_mesh(TP, DP, device_type="cpu")
        linear = _placed(nn.Linear(8, 8), tp_submesh(mesh), Shard(0), Shard(0))
        local_linears.install_local_forward(linear, "colwise", name="proj")
        linear(torch.randn(2, 8))
        full = nn.Linear(8, 8)
        for pname in ("weight", "bias"):
            sharded = distribute_tensor(getattr(full, pname).detach(), mesh, [Shard(0), Shard(0)], src_data_rank=None)
            setattr(linear, pname, nn.Parameter(sharded))
        with pytest.raises(RuntimeError, match="proj ran with its parameters FSDP2-sharded"):
            linear(torch.randn(2, 8))


def test_load_finish_diagnoses_a_tied_pair_the_plan_disagrees_with():
    """A planned head whose tied weight loaded replicated gets the tie check's diagnosis from the load finish,
    ahead of the local-forward install, which would only refuse the head's plain weight."""
    model = tiny_family_model(HF_FAMILIES["qwen3"], overrides={"tie_word_embeddings": True})
    assert model.lm_head.weight is model.model.embed_tokens.weight
    with fake_process_group_mesh(rank=0, world_size=TP) as mesh:
        for layer in model.model.layers:
            q_proj = layer.self_attn.q_proj
            q_proj.weight = nn.Parameter(
                distribute_tensor(q_proj.weight.detach(), mesh, [Shard(0)], src_data_rank=None)
            )
        model._tp_plan = {"model.layers.*.self_attn.q_proj": "colwise", "lm_head": "colwise_gather_output"}
        with pytest.raises(RuntimeError, match="tied embedding/head pair and the applied TP plan disagree"):
            hf_plan.finish_hf_tp_load(model, mesh, TP)


def test_install_planned_leaves_embeddings_and_other_styles_on_their_forward():
    """Only an ``nn.Linear`` under a local style moves; the vocab-parallel embedding keeps transformers' path."""

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(8, 8)
            self.proj = nn.Linear(8, 8)
            self.norm_proj = nn.Linear(8, 8)

    with fake_process_group_mesh(rank=0, world_size=TP) as mesh:
        model = _Model()
        _placed(model.proj, mesh, Shard(0), Shard(0))
        plan = {"embed": "embedding_rowwise", "proj": "colwise", "norm_proj": "sequence_parallel"}
        assert local_linears.install_planned_local_forwards(model, plan) == 1
        assert "forward" in vars(model.proj)
        assert "forward" not in vars(model.embed) and "forward" not in vars(model.norm_proj)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
