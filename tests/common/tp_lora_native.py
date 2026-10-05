"""Real Hugging Face model fixtures for PEFT's native DTensor TP path."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from types import MethodType, SimpleNamespace

import peft
import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers
from packaging.version import Version
from peft import LoraConfig, get_peft_model
from peft.tuners.lora.layer import LoraLayer
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard
from transformers import DistributedConfig, LlamaConfig, LlamaForCausalLM
from transformers.distributed.tensor_parallel import ALL_PARALLEL_STYLES

from src.distributed.mesh import ParallelDims
from src.trainers.mixins.base import DistributedTrainerMixin

BASE_SEED = 117
ADAPTER_SEED = 301
INPUT_SEED = 902
LORA_RANK = 4
TARGETS = ("q_proj", "o_proj")


def require_native_runtime() -> None:
    assert Version(peft.__version__) >= Version("0.21.1"), "Native TP tests require PEFT >= 0.21.1"
    assert Version(transformers.__version__) >= Version("5.17.0"), "Native TP tests require Transformers >= 5.17.0"


def write_tiny_checkpoint(path: str) -> None:
    require_native_runtime()
    torch.manual_seed(BASE_SEED)
    config = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    LlamaForCausalLM(config).save_pretrained(path)


def native_model(
    path: str,
    world_size: int,
    *,
    initialization: bool | str = False,
    targets: Sequence[str] = TARGETS,
    adapter_seed: int = ADAPTER_SEED,
    checkpointing: bool = False,
    dtype: torch.dtype = torch.float32,
    parallel: bool = True,
    autocast_adapter_dtype: bool = True,
    device: torch.device | str = "cpu",
):
    require_native_runtime()
    load_kwargs = {}
    if parallel:
        mesh = init_device_mesh(torch.device(device).type, (world_size,), mesh_dim_names=("tp",))
        load_kwargs.update(
            distributed_config=DistributedConfig(tp_plan="auto", tp_size=world_size),
            device_mesh=mesh,
        )
    else:
        load_kwargs["device_map"] = {"": device}
    base = LlamaForCausalLM.from_pretrained(path, dtype=dtype, attn_implementation="eager", **load_kwargs)
    if parallel:
        assert base._device_mesh.mesh_dim_names == ("tp",)
        assert base.config.distributed_config.tp_size == world_size
        assert base.tp_plan["model.layers.*.self_attn.q_proj"] == "colwise"
        assert base.tp_plan["model.layers.*.self_attn.o_proj"] == "rowwise"
    torch.manual_seed(adapter_seed)
    model = get_peft_model(
        base,
        LoraConfig(
            r=LORA_RANK,
            lora_alpha=8,
            lora_dropout=0.0,
            target_modules=list(targets),
            init_lora_weights=initialization,
            bias="none",
            task_type="CAUSAL_LM",
        ),
        autocast_adapter_dtype=autocast_adapter_dtype,
    )
    if parallel:
        assert_native_factors(model)
    if checkpointing:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
        model.config.use_cache = False
    return model


def lora_layers(model) -> dict[str, LoraLayer]:
    return {name: module for name, module in model.get_base_model().named_modules() if isinstance(module, LoraLayer)}


def native_forward_parts(factor) -> dict:
    forward = factor.forward
    assert forward.__name__ == "tp_forward", "the native Transformers forward transform was not installed"
    parts = dict(zip(forward.__code__.co_freevars, (cell.cell_contents for cell in forward.__closure__), strict=True))
    assert {"self", "mesh", "original_forward"} <= parts.keys()
    assert type(parts["self"]).__module__ == "transformers.distributed.tensor_parallel"
    return parts


def assert_native_factors(model) -> None:
    layers = lora_layers(model)
    assert layers, "no LoRA factors were injected"
    for name, layer in layers.items():
        base = layer.get_base_layer()
        style_name = getattr(base, "_hf_tp_plan", None)
        assert style_name in ("colwise", "rowwise"), f"{name}: no native linear TP plan"
        colwise = style_name == "colwise"
        shard = layer.lora_B["default"] if colwise else layer.lora_A["default"]
        replica = layer.lora_A["default"] if colwise else layer.lora_B["default"]
        assert isinstance(base.weight, DTensor)
        assert base._hf_tp_plan == style_name
        assert base._hf_device_mesh.mesh_dim_names == ("tp",)
        assert isinstance(shard.weight, DTensor)
        assert isinstance(shard.weight.placements[0], Shard)
        assert shard.weight.placements[0].dim % 2 == (0 if colwise else 1)
        assert not isinstance(replica.weight, DTensor)
        parts = native_forward_parts(shard)
        assert parts["self"] is ALL_PARALLEL_STYLES[style_name]
        assert parts["mesh"] is base._hf_device_mesh


def full(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.full_tensor() if isinstance(tensor, DTensor) else tensor.detach().clone()


def align_reference(native, reference) -> None:
    """Isolate collective correctness; initialization has its own independent assertions."""
    reference_layers = lora_layers(reference)
    with torch.no_grad():
        for name, layer in lora_layers(native).items():
            for factor_name in ("lora_A", "lora_B"):
                source = getattr(layer, factor_name)["default"].weight
                getattr(reference_layers[name], factor_name)["default"].weight.copy_(full(source))


def assert_replicas_equal(model, world_size: int) -> None:
    for name, layer in lora_layers(model).items():
        colwise = layer.get_base_layer()._hf_tp_plan == "colwise"
        replica = layer.lora_A["default"] if colwise else layer.lora_B["default"]
        bits = replica.weight.detach().contiguous().view(torch.uint8)
        peers = [torch.empty_like(bits) for _ in range(world_size)]
        dist.all_gather(peers, bits)
        assert all(torch.equal(peer, peers[0]) for peer in peers[1:]), f"{name} replica drifted"


def tp_clip_trainer(model) -> SimpleNamespace:
    mesh = model.get_base_model()._device_mesh
    trainer = SimpleNamespace(
        _top_level_model=lambda: model,
        _get_tp_process_group=lambda: mesh.get_group(),
        _sharded_grad_bucket=DistributedTrainerMixin._sharded_grad_bucket,
        _pp_stage_group=None,
        _pp_chain_group=None,
        parallel_dims=ParallelDims(mesh),
        parallelism_config=SimpleNamespace(fp32_grad_reduce=False),
        state=SimpleNamespace(global_step=0),
        accelerator=SimpleNamespace(),
    )
    for name in (
        "_tp_sharded_plain_param_ids",
        "_tp_per_head_norm_param_ids",
        "_sync_tp_replicated_grads",
        "_reduce_shard_norm_buckets",
        "_compute_tp_grad_norm",
    ):
        setattr(trainer, name, MethodType(getattr(DistributedTrainerMixin, name), trainer))
    DistributedTrainerMixin._patch_gradient_clipping_for_tp(trainer)
    return trainer


def reference_grad_norm(model) -> torch.Tensor:
    grads = [param.grad.float() for param in model.parameters() if param.grad is not None]
    return torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(grad) for grad in grads]))


@contextmanager
def without_native_collective(factor) -> Iterator[None]:
    """Keep the native local weight, removing only its TP forward/backward boundary."""
    native_forward_parts(factor)
    forward = factor.forward
    factor.forward = lambda x: F.linear(x, factor.weight.to_local(), factor.bias)
    try:
        yield
    finally:
        factor.forward = forward
