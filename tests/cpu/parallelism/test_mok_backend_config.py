"""``moe_backend: mok`` is refused at config time wherever its kernels cannot run the configuration, and at layer build
for a block whose experts they would compute wrong; the padded token count covers the declared budget, no routing
overflows the schedule MoK traps on, MXFP8 is the one low-precision recipe it takes, needing no ``deep_gemm``, and the
adapter hands MoK the layer's experts in MoK's layouts and back (against an eager stand-in for its kernels, as the
package runs on Blackwell GPUs only)."""

import dataclasses
import math
import sys
import types
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP
from transformers.models.qwen3_moe.configuration_qwen3_moe import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

from halo.configs.parallelism_arguments import ParallelismArguments, parallelism_config_from_args
from halo.distributed.expert_parallel import mok
from halo.distributed.expert_parallel.families.gpt_oss import EPGptOssMoELayer
from halo.distributed.expert_parallel.families.qwen3 import EPQwen3MoELayer
from halo.kernels.lowp import deepgemm
from halo.kernels.lowp.spec import LowpSpec
from halo.model_loading.parallel import load_rules
from halo.models.moe.expert_lora_spec import ExpertLoraSpec
from halo.models.moe.family_specs import MOE_FAMILY_SPECS
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from tests.common.ep_stubs import stubbed_rank_ep_layer
from tests.common.parallelism import create_config

BUDGET = {"ep_rows_per_device": 2, "ep_declared_max_length": 4096}


@pytest.fixture(autouse=True)
def no_visible_gpu(monkeypatch):
    """``require_mok`` refuses a visible GPU that is not SM103, which the CPU tier's container can be handed."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)


def _mok(**kw):
    return create_config(moe_backend="mok", **{"world_size": 4, "gpus_per_node": 4, "ep_size": 4, **BUDGET, **kw})


MXFP8 = LowpSpec(precision="fp8", backend="native", backward_precision="fp8", apply_dense_mlp=False)


@pytest.mark.parametrize("overrides", [{}, {"ep_size": 1, "fsdp_shard_ep1_experts": False}])
@pytest.mark.parametrize(
    "lowp",
    [LowpSpec(), MXFP8, LowpSpec(precision="fp8", apply_moe_experts=False)],
    ids=["bf16", "mxfp8", "simulated_on_the_dense_mlps_alone"],
)
def test_pure_node_local_ep_with_a_budget_is_accepted(overrides, lowp):
    assert _mok(**overrides, lowp=lowp).lowp == lowp


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"world_size": 2, "gpus_per_node": 2, "ep_size": 2}, "expert_parallel_size=2 is not one of"),
        ({"world_size": 16, "gpus_per_node": 8, "ep_size": 16}, "spans NVLink domains"),
        ({"world_size": 8, "gpus_per_node": 8, "ep_size": 4, "expert_tp_size": 2}, "expert_tensor_parallel_size=2"),
        ({"ep_size": 1, "cp_size": 4}, "context_parallel_size=4"),
        ({"world_size": 8, "gpus_per_node": 8, "ep_size": 8, "tp_size": 2}, "tensor_parallel_size=2"),
        ({"world_size": 16, "gpus_per_node": 8, "ep_size": 8, "pp_size": 2}, "pipeline_parallel_size=2"),
        ({"ep_size": 1}, "FSDP2-managed experts"),
        ({"expert_lora": ExpertLoraSpec(r=8, alpha=16)}, "expert LoRA"),
        ({"ep_fp32_experts": True}, "fp32_experts"),
        ({"lowp": LowpSpec(precision="fp8")}, "lowp_backend='simulated'"),
        (
            {"lowp": LowpSpec(precision="fp8", backend="native", apply_dense_mlp=False)},
            "lowp_backward_precision='bf16'",
        ),
        (
            {"lowp": LowpSpec(precision="mxfp4", backend="native", backward_precision="fp8", apply_dense_mlp=False)},
            "lowp_precision='mxfp4'",
        ),
        ({"ep_rows_per_device": 0}, "no declared token budget"),
        ({"use_grouped_gemm": False}, "use_grouped_gemm=False"),
        ({"ep_size": 1, "use_grouped_gemm": False}, "use_grouped_gemm=False"),
        ({"world_size": 8, "gpus_per_node": 8, "ep_size": 8, "ep_buffer_backend": "legacy"}, "ep_buffer_backend"),
        ({"ep_declared_max_length": 0}, "no declared token budget"),
        # DeepEP's refusal of the same layout offers expert_parallel_size=2 and expert TP, which MoK refuses next.
        (
            {"world_size": 8, "gpus_per_node": 8},
            "2 concurrent EP groups on one 8-GPU NVLink domain",
        ),
    ],
)
def test_a_configuration_mok_cannot_run_is_refused(overrides, message):
    with pytest.raises(ValueError, match="moe_backend='mok' cannot run") as raised:
        _mok(**overrides)
    assert message in str(raised.value)


def test_an_unknown_backend_is_refused():
    with pytest.raises(ValueError, match="moe_backend must be one of"):
        create_config(moe_backend="tutel")


@pytest.mark.parametrize(("tokens", "capacity"), [(0, 512), (100, 512), (1000, 1024), (1100, 1536), (8192, 8192)])
def test_the_token_capacity_is_the_budget_padded_to_mok_shapes(tokens, capacity):
    assert mok.mok_token_capacity(tokens) == capacity


@pytest.fixture
def mok_config(monkeypatch):
    """``_mok_config`` over a stand-in ``MoKConfig`` that keeps its fields."""
    monkeypatch.setattr(mok, "_mok", lambda: SimpleNamespace(MoKConfig=SimpleNamespace))
    mok._mok_config.cache_clear()
    yield mok._mok_config
    mok._mok_config.cache_clear()


@pytest.mark.parametrize(("intermediate", "sms"), [(768, 48), (1024, 48), (1536, 42), (2048, 42), (3072, 36)])
def test_narrow_experts_get_more_communication_sms(mok_config, intermediate, sms):
    """A token's dispatch scales with the hidden size and its compute with hidden x intermediate: narrow experts are
    communication-bound, and at a wide block's 36 SMs Qwen3-30B-A3B's 768-wide experts run 15% slower."""
    config = mok_config(16384, intermediate, 8, 4, 32)
    assert (config.fwd_num_comm_sms, config.bwd_num_comm_sms, config.minibatch_size) == (sms, sms, 8192)


@pytest.mark.parametrize(
    ("capacity", "top_k", "minibatch", "ring"),
    [(512, 4, 512, 4096), (4096, 8, 4096, 65536), (10240, 8, 8192, 163840), (16384, 8, 8192, 262144)],
)
def test_the_ring_holds_twice_a_ranks_even_share_in_whole_minibatches(mok_config, capacity, top_k, minibatch, ring):
    """MoK allocates every routed activation at the ring's size and saves it for the backward: its default 131072
    rows, or 32 minibatches, is four times a 4096-token rank's routed rows."""
    config = mok_config(capacity, 768, top_k, 4, 32)
    assert (config.minibatch_size, config.macrobatch_size) == (minibatch, ring)
    assert ring % minibatch == 0 and ring >= 2 * capacity * top_k


def _mok_schedule_rows(capacity: int, top_k: int, ep_size: int, multiplier: float) -> int:
    """A rank's schedule as MoK sizes it (``mok/functional.py``, ``get_workspace``)."""
    return capacity * top_k * max(2, math.ceil(ep_size * multiplier))


@pytest.mark.parametrize(
    ("capacity", "top_k", "ep_size", "local_experts"),
    [(512, 8, 1, 128), (2048, 8, 1, 128), (1024, 4, 1, 32), (512, 8, 4, 128), (4096, 8, 4, 32), (16384, 4, 4, 8)],
    ids=["qwen3_ep1_small", "qwen3_ep1", "tiny_ep1", "many_local_experts", "qwen3_ep4", "gpt_oss_ep4"],
)
def test_no_routing_overflows_the_schedule(capacity, top_k, ep_size, local_experts):
    """MoK traps in its kernel, with no message, when a rank's rows pass its schedule, each local expert's rounded up
    to 256. The worst routing puts every row of the group on one rank, one row past a multiple of 256 on every expert
    but the last. The controls: MoK's default 0.5 and a flat 1.0, which leaves the rounding out, both overflow."""
    rows = ep_size * capacity * min(top_k, local_experts)
    spread = [1] * (local_experts - 1) + [rows - (local_experts - 1)]
    worst = sum(-(-count // 256) * 256 for count in spread)
    multiplier = mok._schedule_capacity_multiplier(capacity, top_k, ep_size, local_experts)
    assert _mok_schedule_rows(capacity, top_k, ep_size, multiplier) >= worst
    assert _mok_schedule_rows(capacity, top_k, ep_size, multiplier) - capacity * top_k < max(
        worst, 2 * capacity * top_k
    )
    for control in (0.5, 1.0):
        assert _mok_schedule_rows(capacity, top_k, ep_size, control) < worst, f"the {control} control holds the worst"


def test_the_layer_config_carries_the_safe_schedule_multiplier(mok_config):
    config = mok_config(4096, 768, 8, 4, 32)
    assert config.schedule_capacity_multiplier == mok._schedule_capacity_multiplier(4096, 8, 4, 32) > 1.0


@pytest.fixture
def mok_import():
    mok._mok.cache_clear()
    saved = {name: sys.modules.pop(name) for name in [n for n in sys.modules if n == "mok" or n.startswith("mok.")]}
    yield
    for name in [n for n in sys.modules if n == "mok" or n.startswith("mok.")]:
        del sys.modules[name]
    sys.modules.update(saved)
    mok._mok.cache_clear()


def test_an_absent_package_reads_as_unavailable(mok_import):
    sys.modules["mok"] = None  # the import system's marker for a module that is not there
    assert not mok.is_mok_available()


def test_an_installed_package_that_does_not_import_raises(mok_import):
    """An ABI or CUDA mismatch in the extension must not read as "install mok"."""
    package = types.ModuleType("mok")
    package.__path__ = []  # installed, with a ``functional`` that fails to import
    sys.modules["mok"] = package
    with pytest.raises(ImportError):
        mok.is_mok_available()


def test_a_gpu_the_image_does_not_build_mok_for_is_refused(mok_import, monkeypatch):
    """The image's mok is SM103 code, which a B200 (SM100) cannot run."""
    package = types.ModuleType("mok")
    package.functional = sys.modules["mok.functional"] = types.ModuleType("mok.functional")
    sys.modules["mok"] = package
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (10, 0))
    with pytest.raises(RuntimeError, match="SM103 .* this GPU is SM100"):
        mok.require_mok()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: (10, 3))
    assert mok.require_mok() is package.functional


def test_a_model_with_no_wrapped_moe_block_is_refused():
    """At ``expert_parallel_size`` 1 nothing else would notice: the setting would run nothing."""
    dense = SimpleNamespace(model_type="qwen3")
    with pytest.raises(ValueError, match="moe_backend='mok' runs the routed experts .* would be ignored"):
        load_rules.reject_ep_without_family(_mok(ep_size=1, fsdp_shard_ep1_experts=False), dense, wraps_experts=False)


@pytest.mark.parametrize(("backend", "rows"), [("mok", 8), ("deepep", 1)])
def test_an_evaluating_run_sizes_mok_buffers_for_its_eval_batch(backend, rows, caplog):
    """HF's default eval batch is 8: a buffer sized for one training row would raise at the first evaluation. Every
    training forward is then padded to those 8 rows, which the config says."""
    training = SimpleNamespace(
        per_device_train_batch_size=1, per_device_eval_batch_size=8, max_length=1024, eval_strategy="steps"
    )
    with caplog.at_level("WARNING", logger="halo.configs.parallelism_arguments"):
        config = parallelism_config_from_args(
            ParallelismArguments(moe_backend=backend, fsdp_shard_ep1_experts=False),
            trainer_cls=DistributedTrainerMixin,
            training_config=training,
        )
    assert (config.ep_rows_per_device, config.ep_declared_max_length) == (rows, 1024)
    warned = [r.getMessage() for r in caplog.records if "pads every forward" in r.getMessage()]
    assert len(warned) == (backend == "mok") and all("8.0x its own tokens" in message for message in warned)


def test_an_eval_batch_no_larger_than_the_training_one_warns_nothing(caplog):
    training = SimpleNamespace(
        per_device_train_batch_size=2, per_device_eval_batch_size=2, max_length=1024, eval_strategy="steps"
    )
    with caplog.at_level("WARNING", logger="halo.configs.parallelism_arguments"):
        parallelism_config_from_args(
            ParallelismArguments(moe_backend="mok", fsdp_shard_ep1_experts=False),
            trainer_cls=DistributedTrainerMixin,
            training_config=training,
        )
    assert not caplog.records


def _qwen3_block(intermediate: int) -> nn.Module:
    return Qwen3MoeSparseMoeBlock(Qwen3MoeConfig(hidden_size=256, moe_intermediate_size=intermediate, num_experts=4))


def _gpt_oss_block() -> nn.Module:
    return GptOssMLP(GptOssConfig(hidden_size=256, intermediate_size=256, num_local_experts=4, num_experts_per_tok=2))


def _build(layer_cls, block, monkeypatch, *, available: bool = True):
    monkeypatch.setattr(mok, "_mok", lambda: object() if available else None)
    return stubbed_rank_ep_layer(layer_cls, block, 1, 1, 0, grouped=True, moe_backend="mok")


def test_a_supported_block_builds(monkeypatch):
    layer = _build(EPQwen3MoELayer, _qwen3_block(256), monkeypatch)
    assert isinstance(layer.dispatcher, mok.MoKExperts)


@pytest.mark.parametrize(
    ("layer_cls", "block", "message"),
    [
        (EPGptOssMoELayer, _gpt_oss_block, "EPGptOssMoELayer's family does not declare supports_mok"),
        (EPQwen3MoELayer, lambda: _qwen3_block(192), "intermediate 192 not multiples of 256"),
    ],
    ids=["biased-clamped-experts", "misaligned-intermediate"],
)
def test_experts_mok_would_compute_wrong_are_refused_at_build(monkeypatch, layer_cls, block, message):
    with pytest.raises(ValueError, match=message):
        _build(layer_cls, block(), monkeypatch)


def test_only_the_families_with_an_equivalence_row_support_mok():
    """Qwen3-Next and Qwen3.8-Flash-Next run Qwen3.5's EP layer under specs of their own, which declare nothing."""
    assert {spec.model_types[0] for spec in MOE_FAMILY_SPECS if spec.supports_mok} == {"qwen3_moe", "qwen3_5_moe"}


@pytest.mark.parametrize(
    ("model_type", "message"), [("qwen3_next", "does not declare supports_mok"), ("qwen3_moe", "")]
)
def test_an_unsupported_family_and_an_absent_package_are_refused_before_the_load(monkeypatch, model_type, message):
    config = SimpleNamespace(model_type=model_type)
    monkeypatch.setattr(mok, "_mok", lambda: None)
    expected = (
        pytest.raises(ValueError, match=message) if message else pytest.raises(RuntimeError, match="needs the mok")
    )
    with expected:
        load_rules.reject_ep_without_family(_mok(), config, wraps_experts=True)
    monkeypatch.setattr(mok, "_mok", object)
    if not message:
        load_rules.reject_ep_without_family(_mok(), config, wraps_experts=True)


def test_destroying_the_layers_backend_releases_mok_workspaces(monkeypatch):
    """``cleanup_ep`` destroys every EP layer's ``dispatcher``, which under mok is the MoK backend."""
    layer = _build(EPQwen3MoELayer, _qwen3_block(256), monkeypatch)
    released = []
    monkeypatch.setattr(mok, "_mok", lambda: SimpleNamespace(clear_workspace_cache=lambda: released.append(True)))
    layer.dispatcher.destroy()
    assert released == [True]


def test_mxfp8_under_mok_needs_no_deep_gemm(monkeypatch):
    """MoK's own kernels run the native recipe; the DeepGEMM gate is for the DeepEP path."""
    monkeypatch.setattr(deepgemm, "_deep_gemm", lambda: None)
    layer = _build(EPQwen3MoELayer, _qwen3_block(256), monkeypatch)
    assert layer.set_lowp_compute(MXFP8, weight_cacheable=True) and layer._lowp is MXFP8


def test_the_mxfp8_cast_is_redone_after_a_write_to_a_bank_and_only_then(monkeypatch):
    """The backward and every grad-accumulation microbatch reuse one cast per step; a stale one trains old weights. An
    optimizer step, a ``load_state_dict`` and a rebound bank each cast anew."""
    casts = []

    def quantize(w, normal, transposed):
        casts.append(w)
        return w, w, w, w

    monkeypatch.setattr(mok, "_mxfp8_quantize", lambda: quantize)
    holder = nn.ParameterList(nn.Parameter(torch.randn(2, 256, 256)) for _ in range(2))
    banks = tuple(holder)
    first = mok._mxfp8_banks(banks)
    assert mok._mxfp8_banks(banks).forward[0][0] is first.forward[0][0] and len(casts) == 3

    for bank in banks:
        bank.grad = torch.ones_like(bank)
    torch.optim.SGD(holder, lr=0.1).step()
    mok._mxfp8_banks(banks), mok._mxfp8_banks(banks)
    assert len(casts) == 6, "an optimizer step"

    holder.load_state_dict({name: torch.zeros_like(bank) for name, bank in holder.state_dict().items()})
    mok._mxfp8_banks(banks), mok._mxfp8_banks(banks)
    assert len(casts) == 9 and not casts[-1].any(), "a load_state_dict"

    rebound = (banks[0], nn.Parameter(banks[1].detach().clone()))
    with torch.no_grad():
        rebound[1].add_(0.0)
        rebound[1].add_(0.0)
    assert rebound[1]._version == banks[1]._version, "premise: another tensor at the same version"
    mok._mxfp8_banks(rebound)
    assert len(casts) == 12, "a rebound bank"


def test_an_absent_package_is_refused_at_build(monkeypatch):
    with pytest.raises(RuntimeError, match="needs the mok package"):
        _build(EPQwen3MoELayer, _qwen3_block(256), monkeypatch, available=False)


@dataclasses.dataclass
class _Schedule:
    experts: torch.Tensor
    peer_token_idx: torch.Tensor
    num_tokens: torch.Tensor
    tokens_per_expert: torch.Tensor


@dataclasses.dataclass
class _Context:
    x_routed: torch.Tensor
    gate_shared: torch.Tensor
    gate_routed: torch.Tensor
    up_shared: torch.Tensor
    up_routed: torch.Tensor
    hidden_shared: torch.Tensor
    hidden_routed: torch.Tensor


class _EagerMoK:
    """``mok.functional`` at EP 1 in eager fp32: a routed SwiGLU over MoK's ``F.linear`` layouts, gate and up
    ``[E, I, H]`` and down ``[E, H, I]``, plus the shared expert MoK always runs."""

    MoKSchedule, MoKForwardContext, MoKConfig = _Schedule, _Context, SimpleNamespace

    def __init__(self):
        self.workspaces, self.schedules = [], []

    def get_workspace(self, config, group, **shape):
        self.workspaces.append((config, shape))
        return SimpleNamespace(**shape)

    def build_schedule(self, workspace, config, experts, *, num_local_experts):
        self.schedules.append((experts.clone(), num_local_experts))
        counts = torch.bincount(experts.reshape(-1), minlength=num_local_experts)
        return _Schedule(experts.clone(), experts.new_zeros(1), torch.tensor([experts.numel()]), counts)

    @staticmethod
    def _ffn(experts, x, weights, shared_gate, shared_up, shared_down, gate, up, down):
        hidden = F.silu(torch.einsum("tkih,th->tki", gate[experts], x)) * torch.einsum("tkih,th->tki", up[experts], x)
        routed = (weights.unsqueeze(-1) * torch.einsum("tkhi,tki->tkh", down[experts], hidden)).sum(1)
        return routed + (F.silu(x @ shared_gate.T) * (x @ shared_up.T)) @ shared_down.T

    def forward(self, config, workspace, schedule, x, weights, *banks):
        out = self._ffn(schedule.experts, x.float(), weights, *(bank.float() for bank in banks))
        return out, _Context(*(x.new_zeros(1) for _ in dataclasses.fields(_Context)))

    def backward(self, config, workspace, schedule, context, grad, x, weights, *banks):
        leaves = [t.float().requires_grad_(True) for t in (x, weights, *banks)]
        with torch.enable_grad():
            out = self._ffn(schedule.experts, *leaves)
        d_x, d_weights, *d_shared, d_gate, d_up, d_down = torch.autograd.grad(out, leaves, grad.float())
        return d_x, d_weights, d_gate, d_up, d_down, *d_shared


def _transpose(src, out=None):
    """``transpose_last_two``, a Triton kernel."""
    return src.transpose(1, 2).contiguous() if out is None else out.copy_(src.transpose(1, 2))


def _bf16_exact(*shape) -> torch.Tensor:
    """Sixteenths in [-2, 2]: bf16 holds them, so the adapter's casts change nothing."""
    return torch.randint(-32, 33, shape).float() / 16


@pytest.mark.parametrize("fused", [True, False], ids=["fused_gate_up", "separate_gate_up"])
def test_the_adapter_runs_the_layers_experts_through_mok_layouts_and_back(monkeypatch, fused):
    """Fewer tokens than the capacity, through the stand-in: the output and every gradient equal the layer-layout
    SwiGLU's own, so the bank transposes, a fused bank's gate-first halves, the padding rows and the saved-tensor
    slicing of the backward are each what MoK's kernels are handed. The schedule covers the padded capacity with
    local expert ids only, and the config is sized for this group."""
    torch.manual_seed(0)
    functional = _EagerMoK()
    monkeypatch.setattr(mok, "_mok", lambda: functional)
    monkeypatch.setattr(mok, "transpose_last_two", _transpose)
    monkeypatch.setattr(mok.dist, "get_world_size", lambda group: 1)
    mok._mok_config.cache_clear()
    experts, hidden, intermediate, top_k, tokens, capacity = 4, 8, 16, 2, 5, 512
    x = _bf16_exact(tokens, hidden).requires_grad_(True)
    router_weights = torch.rand(tokens, top_k, requires_grad=True)
    top_experts = torch.stack([torch.randperm(experts)[:top_k] for _ in range(tokens)])
    gate, up = (_bf16_exact(experts, hidden, intermediate) / 4 for _ in range(2))
    down = _bf16_exact(experts, intermediate, hidden) / 4
    glu = (torch.cat([gate, up], dim=2),) if fused else (gate, up)
    banks = tuple(bank.requires_grad_(True) for bank in (*glu, down))
    seed_grad = _bf16_exact(tokens, hidden)

    ep_config = SimpleNamespace(mok_token_capacity=capacity, dispatch_ep_group=None)
    out = mok.MoKExperts(ep_config, experts, hidden)(x, top_experts, router_weights, banks)
    out.backward(seed_grad)
    got = [out.detach(), x.grad, router_weights.grad, *(bank.grad for bank in banks)]
    mok._mok_config.cache_clear()

    leaves = [t.detach().clone().requires_grad_(True) for t in (x, router_weights, *banks)]
    ref_x, ref_weights, *ref_banks = leaves
    ref_gate, ref_up = ref_banks[0].chunk(2, dim=2) if fused else ref_banks[:2]
    per_slot = F.silu(torch.einsum("th,tkhi->tki", ref_x, ref_gate[top_experts])) * torch.einsum(
        "th,tkhi->tki", ref_x, ref_up[top_experts]
    )
    expected = (ref_weights.unsqueeze(-1) * torch.einsum("tki,tkih->tkh", per_slot, ref_banks[-1][top_experts])).sum(1)
    expected.backward(seed_grad)
    for name, value, want in zip(
        ("output", "grad_x", "grad_router_weights", *(f"grad_bank_{i}" for i in range(len(banks)))),
        got,
        [expected.detach(), *(leaf.grad for leaf in leaves)],
        strict=True,
    ):
        assert value.shape == want.shape, f"{name}: {tuple(value.shape)}"
        torch.testing.assert_close(value, want, rtol=1e-4, atol=1e-4, msg=lambda m, name=name: f"{name}: {m}")
    assert x.grad.abs().sum() > 0 and all(bank.grad.abs().sum() > 0 for bank in banks)

    (scheduled, local_experts), (config, shape) = functional.schedules[0], functional.workspaces[0]
    assert scheduled.shape == (capacity, top_k) and local_experts == experts
    assert torch.equal(scheduled[:tokens], top_experts) and scheduled.min() >= 0 and scheduled.max() < experts
    assert shape == {"device": x.device, "num_local_tokens": capacity, "hidden_size": hidden, "topk": top_k}
    assert config.schedule_capacity_multiplier == mok._schedule_capacity_multiplier(capacity, top_k, 1, experts)


def test_a_forward_past_the_capacity_raises_before_any_kernel(monkeypatch):
    monkeypatch.setattr(mok, "_mok", lambda: pytest.fail("the over-capacity forward reached MoK"))
    with pytest.raises(RuntimeError, match="sized its buffers to 512 tokens per rank .* a forward carried 513"):
        mok.MoKExperts(SimpleNamespace(mok_token_capacity=512), 4, 8)(
            torch.zeros(513, 8), torch.zeros(513, 2, dtype=torch.long), torch.zeros(513, 2), ()
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
