"""The wire form of a LoRA adapter for an engine's load-from-tensors route: the PEFT config the engine validates,
the tensors with the scaling folded into B, and the serialized payload."""

import base64
import pickle
from collections.abc import Iterable, Mapping
from typing import Any

import torch

# What the engine stores adapters in; a trainer adapter in another dtype would be rounded on the wire.
ADAPTER_WIRE_DTYPE = torch.bfloat16

_LORA_B_MARKER = ".lora_B."


def build_adapter_config(*, rank: int, target_modules: Iterable[str]) -> dict[str, Any]:
    """The ``adapter_config.json`` the engine reads. ``lora_alpha == rank`` makes the engine's scaling exactly 1, since
    :func:`wire_adapter_tensors` has already folded the trainer's into B."""
    return {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": rank,
        "lora_alpha": rank,
        "target_modules": sorted(set(target_modules)),
        "use_dora": False,
        "use_rslora": False,
        "lora_dropout": 0.0,
        "bias": "none",
    }


def wire_adapter_tensors(state: Mapping[str, torch.Tensor], scaling: float) -> dict[str, torch.Tensor]:
    """CPU copies of ``state`` in the wire dtype. B is scaled in fp32 and rounded once: the engines scale an adapter by
    ``alpha / r`` alone (SGLang ignores rsLoRA) and one of them rounds B a second time when it does, so the scaling the
    trainer applies must not reach them."""
    wire = {}
    for key, tensor in state.items():
        value = tensor.detach()
        if _LORA_B_MARKER in key:
            value = value.float() * scaling
        wire[key] = value.to(device="cpu", dtype=ADAPTER_WIRE_DTYPE, copy=True).contiguous()
    return wire


def serialize_adapter_tensors(tensors: Mapping[str, torch.Tensor]) -> str:
    """Base64 of a plain pickle. Unlike ``ForkingPickler``, which hands CPU storage over by file descriptor and CUDA
    storage by IPC handle, it carries the bytes inline, so the engine can be on another host."""
    return base64.b64encode(pickle.dumps(dict(tensors), protocol=pickle.HIGHEST_PROTOCOL)).decode("ascii")
