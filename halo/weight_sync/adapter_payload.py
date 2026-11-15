"""The wire form of a LoRA adapter for an engine: the PEFT config the engine validates, the tensors with the scaling
folded into B, and either the serialized payload of a load-from-tensors route or a published adapter directory."""

import base64
import json
import os
import pickle
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file

from halo.checkpoint.format import ADAPTER_CONFIG_FILE, ADAPTER_SAFETENSORS_FILE

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


def publish_adapter_directory(root: str | Path, name: str, tensors: Mapping[str, torch.Tensor], config: dict) -> Path:
    """``tensors`` and ``config`` as the PEFT adapter directory ``root/name``, staged beside it and renamed into place, so
    an engine never reads a half-written one. ``root`` is read by every engine at the same path."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    staged = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=root))
    save_file(dict(tensors), staged / ADAPTER_SAFETENSORS_FILE)
    (staged / ADAPTER_CONFIG_FILE).write_text(json.dumps(config))
    staged.chmod(0o755)
    published = root / name
    os.rename(staged, published)
    return published


def remove_adapter_directory(root: str | Path, name: str) -> None:
    shutil.rmtree(Path(root) / name, ignore_errors=True)
