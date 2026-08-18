"""Shared LoRA / QLoRA / native-expert-LoRA helpers for the GRPO + distillation GPU tests.

These drive the same production path the training scripts use (``split_expert_lora_targets`` →
``load_distributed_model(quantization_config=...)`` → ``setup_peft_model``), so the tests exercise the
real adapter wiring (attention PEFT, native grouped expert-LoRA, QLoRA 4-bit base) as the GRPO
scripts drive it rather than a hand-rolled approximation. The trainer families differ (offline GRPO,
teacher-distill, self-distill), so each test file builds its own dataset and trainer; this module
holds the model/adapter build and the adapter-invariant assertions shared across them.

Modes:
    lora        — attention PEFT LoRA (q_proj/v_proj). Dense under FSDP, MoE attention under EP.
    qlora       — LoRA on a 4-bit bitsandbytes base (dense, FSDP/DDP only; QLoRA+EP/TP is rejected).
    expert_lora — native grouped LoRA on the MoE expert FFNs (gate/up/down), built inside EP layers.
"""

from __future__ import annotations

import json
import os
import re
from types import SimpleNamespace

import torch
from accelerate.utils import extract_model_from_parallel
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from torch.distributed.tensor import DTensor
from trl import ModelConfig, get_quantization_config

from src.checkpoint.format import SAFETENSORS_INDEX_FILE
from src.distributed.expert_parallel.expert_weights import gather_ep_lora_adapters, has_ep_lora
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.loading.peft_setup import setup_peft_model, split_expert_lora_targets
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from tests.common.ep_reference import ep_layers
from tests.common.models import GPT_OSS_20B, QWEN3_0_6B
from tests.common.utils import log

DENSE_MODEL = QWEN3_0_6B
MOE_MODEL = GPT_OSS_20B

LORA_R = 8
LORA_ALPHA = 16

# setup_peft_model only reads these two attrs off ``args``; no freeze/unfreeze patterns in tests.
_PEFT_ARGS = SimpleNamespace(unfreeze_layers_patterns=None, freeze_layers_patterns=None)


# Logical expert projections. Family-independent by design: split_expert_lora_targets resolves them
# against whatever 3-D expert tensors the family actually stores.
_EXPERT_TARGETS = ["gate_proj", "up_proj", "down_proj"]
# Fallback attention projections for the dense/QLoRA modes, whose model is fixed (Qwen3-0.6B).
_DEFAULT_ATTENTION_TARGETS = ["q_proj", "v_proj"]
_MOE_MODES = frozenset({"expert_lora", "mixed"})

# Attention projections live under one of these containers depending on the family.
_ATTENTION_PROJ_RE = re.compile(r"\.(?:self_attn|self_attention|attention|attn)\.([A-Za-z0-9_]+)\.weight$")


def _materialize(t: torch.Tensor) -> torch.Tensor:
    """Clone a param to full CPU float, unfolding an FSDP2/TP DTensor shard (collective) first."""
    if isinstance(t, DTensor):
        t = t.full_tensor()
    return t.detach().float().cpu()


def attention_target_modules(model_name: str, revision: str | None = None) -> list[str]:
    """Attention projection leaf names for this family, read off the checkpoint's own parameter names.

    Derived rather than listed, because no list generalizes: MLA families (GLM-4.7, DeepSeek-V4)
    spell them ``q_a_proj``/``q_b_proj``/``kv_a_proj_with_mqa``, Zaya differs again, and PEFT ships no
    ``TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING`` entry for any MoE model_type in this
    roster, so a fixed ``["q_proj", "v_proj"]`` makes PEFT raise "Target modules not found" on
    everything but the Qwen/GptOss spelling. Reading the shipped weight map means a new family needs
    no entry anywhere. Falls back when the checkpoint is single-file (no index), as the small dense
    test models are.
    """
    try:
        index = hf_hub_download(model_name, SAFETENSORS_INDEX_FILE, revision=revision)
    except Exception:
        return list(_DEFAULT_ATTENTION_TARGETS)
    with open(index) as fh:
        weight_map = json.load(fh)["weight_map"]
    found = {m.group(1) for key in weight_map if (m := _ATTENTION_PROJ_RE.search(key))}
    # ``_proj`` only: the same container also holds q/k norms and GptOss's ``sinks``.
    return sorted(name for name in found if name.endswith("_proj")) or list(_DEFAULT_ATTENTION_TARGETS)


def _mode_targets(mode: str, model_name: str, revision: str | None) -> list[str]:
    """``lora_target_modules`` for a mode. ``mixed`` names both namespaces in one list, which
    split_expert_lora_targets peels apart: attention names stay with stock PEFT, expert names become
    the native grouped adapters. That shape trains two adapter kinds at once and is the only one whose
    checkpoint needs both folds."""
    if mode == "expert_lora":
        return list(_EXPERT_TARGETS)
    if mode == "mixed":
        return attention_target_modules(model_name, revision) + _EXPERT_TARGETS
    return list(_DEFAULT_ATTENTION_TARGETS)


def model_name_for(mode: str, parallelism_config) -> str:
    """The model a mode trains: MoE (GptOss-20B) for expert-bearing modes or any EP run, else dense."""
    return MOE_MODEL if (mode in _MOE_MODES or parallelism_config.is_ep_mode) else DENSE_MODEL


def parallelism_config_for(mode: str, size: int):
    """The sharding a mode names, as its ``_ep``/``_etp`` suffix spells it (``size`` = the axis width).

    Kept beside :func:`model_name_for`, which resolves the other half of a mode: two copies of this
    map would drift on their fall-through (one landing on plain FSDP2, the other on EP) and the two
    tests would disagree about what an unsuffixed mode means.
    """
    if mode.endswith("_etp"):
        return ParallelismConfig(ep_size=1, expert_tp_size=size)
    if mode.endswith("_ep") or mode in _MOE_MODES:
        return ParallelismConfig(ep_size=size)
    return ParallelismConfig()


def _build_model_config(mode: str, model_name: str, revision: str | None = None) -> ModelConfig:
    targets = _mode_targets(mode, model_name, revision)
    return ModelConfig(
        model_name_or_path=model_name,
        # Carried here as well as on the load: split_expert_lora_targets reads the config off the hub
        # to decide whether the model has experts, and a pinned family (Zaya, whose `main` serves an
        # incompatible merged layout) would otherwise peel against the wrong revision.
        model_revision=revision,
        use_peft=True,
        lora_r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0.0,
        lora_target_modules=targets,
        lora_task_type="CAUSAL_LM",
        load_in_4bit=(mode == "qlora"),
        bnb_4bit_quant_type="nf4",
        use_bnb_nested_quant=True,
        dtype="bfloat16",
        trust_remote_code=True,
    )


def load_peft_model(
    mode: str,
    parallelism_config,
    *,
    model_name: str | None = None,
    revision: str | None = None,
    attn_implementation: str | None = None,
    use_liger_kernel: bool = True,
    reset_sinks: bool = True,
):
    """Load a model + adapters through the production path. Returns (model, tokenizer, peft_config).

    ``expert_lora`` and any EP mode use the MoE model; ``lora``/``qlora`` without EP use the dense
    model. ``peft_config`` is the attention PEFT config (or None for expert-LoRA-only); the caller
    wires it into the trainer (``peft_config=`` kwarg, or ``get_peft_model`` for the plain-Trainer
    distillation trainer). QLoRA quantization is derived from ``model_config`` exactly as the scripts
    do.

    ``model_name``/``revision`` override the mode's default model, so one test file can sweep the
    other MoE families (each with its own expert storage layout and export renames) rather than being
    pinned to GptOss, and a family that requires a pinned revision can be swept at all.

    ``reset_sinks`` is forwarded to the loader: on-policy RL keeps GptOss's pretrained sinks live and
    frozen, and a run that reset them would train a different model from the one the engine samples.
    """
    model_name = model_name or model_name_for(mode, parallelism_config)
    model_config = _build_model_config(mode, model_name, revision)

    # Peel expert targets to native grouped-LoRA before the load (no-op for attention-only modes).
    parallelism_config.expert_lora = split_expert_lora_targets(model_config)
    quantization_config = get_quantization_config(model_config)

    model, tokenizer = load_distributed_model(
        model_name_or_path=model_name,
        parallelism_config=parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        revision=revision,
        attn_implementation=attn_implementation,
        use_liger_kernel=use_liger_kernel,
        quantization_config=quantization_config,
        reset_sinks=reset_sinks,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = setup_peft_model(_PEFT_ARGS, model, model_config, "CAUSAL_LM")
    return model, tokenizer, peft_config


def adapter_param_items(model) -> list[tuple[str, torch.Tensor]]:
    """(name, param) for every adapter parameter — both PEFT ``lora_*`` and native ``*_lora_{A,B}``."""
    return [(n, p) for n, p in model.named_parameters() if "lora_" in n]


def is_lora_b_key(key: str) -> bool:
    """Whether a key names a zero-init ``lora_B`` matrix, in any of the three spellings it takes.

    PEFT parameters and checkpoint keys carry it mid-path (``...q_proj.lora_B.default.weight``,
    ``...lora_B.weight``); the native grouped expert adapters end on it
    (``...experts.down_proj.lora_B``). B is the half that starts at zero, so both the "did it move"
    and "is this fresh" checks key off this set.
    """
    return key.endswith((".lora_B", "_lora_B")) or ".lora_B." in key


def _is_trainable_adapter(name: str) -> bool:
    """A param that LoRA legitimately trains: a ``lora_*`` adapter, or a ``modules_to_save`` copy
    (e.g. the randomly-initialised SEQ_CLS ``score``/``classifier`` head, which must train)."""
    return "lora_" in name or "modules_to_save" in name


def assert_only_adapters_trainable(model) -> tuple[bool, str]:
    """Static invariant: every trainable param is an adapter (or a modules_to_save head), and at
    least one ``lora_*`` adapter is trainable.

    This is the frozen-base guarantee: setup_peft_model / get_peft_model freeze the base (the 4-bit
    QLoRA weights included) and leave only adapters (plus any modules_to_save) trainable, so the
    optimizer can only move adapters."""
    trainable_non_adapter = [
        n for n, p in model.named_parameters() if p.requires_grad and not _is_trainable_adapter(n)
    ]
    trainable_loras = [n for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    if trainable_non_adapter:
        return False, f"non-adapter params are trainable: {trainable_non_adapter[:5]}"
    if not trainable_loras:
        return False, "no lora_* adapter params are trainable"
    return True, f"{len(trainable_loras)} lora adapter params trainable, base frozen"


def snapshot_adapters(model, *, expert_lora: bool) -> dict[str, torch.Tensor]:
    """Clone adapter values for a before/after comparison.

    Expert-LoRA tensors are EP-sharded, so gather across the EP group (collective: every rank must
    call this). Attention PEFT adapters are replicated or local, so a local clone suffices.
    """
    if expert_lora:
        return {k: _materialize(v) for k, v in gather_ep_lora_adapters(model).items()}
    return {n: _materialize(p) for n, p in adapter_param_items(model) if p.requires_grad}


def assert_adapters_moved(before: dict, after: dict) -> tuple[bool, str]:
    """At least one adapter changed and at least one zero-init B matrix became non-zero."""
    moved = [k for k in before if k in after and not torch.equal(before[k], after[k])]
    b_moved = [k for k in moved if is_lora_b_key(k)]
    if not moved:
        return False, "no adapter tensors changed after training"
    if not b_moved:
        return False, f"{len(moved)} adapters changed but no lora_B moved (B is zero-init; must move)"
    return True, f"{len(moved)}/{len(before)} adapters changed (lora_B moved: {len(b_moved)})"


def assert_adapter_checkpoint(trainer, output_dir: str, rank: int, *, expert_lora: bool) -> tuple[bool, str]:
    """Save an adapter-only checkpoint and verify it round-trips on disk.

    ``trainer.save_model`` is collective (gathers EP/TP shards); only rank 0 reads the file back. The
    written ``adapter_model.safetensors`` must contain LoRA keys, all finite, with at least one
    non-zero B tensor, so the trained adapter actually serialized.
    """
    adapter_dir = os.path.join(output_dir, "adapter_ckpt")
    trainer.save_model(adapter_dir)
    barrier()
    if rank != 0:
        return True, "non-zero rank (save participated)"

    path = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not os.path.exists(path):
        return False, f"adapter_model.safetensors not written to {adapter_dir}"
    saved = load_file(path)
    lora_keys = [k for k in saved if "lora_" in k]
    if not lora_keys:
        return False, f"no lora keys in saved adapter (keys: {list(saved)[:5]})"
    if not all(torch.isfinite(saved[k]).all() for k in lora_keys):
        return False, "saved adapter has non-finite values"
    b_keys = [k for k in lora_keys if is_lora_b_key(k)]
    if not b_keys:
        return False, f"no lora_B tensors in the saved adapter (keys: {lora_keys[:5]})"
    if not any(saved[k].abs().sum() > 0 for k in b_keys):
        return False, "all saved lora_B tensors are zero (training did not serialize)"
    if expert_lora and not any("experts." in k for k in lora_keys):
        return False, f"expert-LoRA checkpoint has no expert keys (keys: {lora_keys[:5]})"
    return True, f"adapter checkpoint OK ({len(lora_keys)} lora keys, {len(b_keys)} B tensors)"


def freeze_base_keep_expert_adapters(model) -> None:
    """Replicate setup_peft_model's adapter-run freeze: only the native expert adapters stay trainable.

    Used by the tests that build the grouped expert adapters directly (no ``setup_peft_model`` call),
    so the optimizer sees the same parameter set a production expert-LoRA run does.
    """
    for p in model.parameters():
        p.requires_grad = False
    for module in ep_layers(model):
        for attr in module._expert_lora_attrs:
            getattr(module, f"{attr}_lora_A").requires_grad_(True)
            getattr(module, f"{attr}_lora_B").requires_grad_(True)


def adapter_save_checks(save_dir: str, rank: int) -> dict[str, bool]:
    """Named checks that an already-written adapter save has the files PEFT needs to reload it.

    Complements :func:`assert_adapter_checkpoint`, which drives the save itself and reads the tensors
    back; this one only inspects a directory the test already saved. Only rank 0 reads the disk.
    """
    if rank != 0:
        return {}

    checks = {}
    dir_exists = os.path.isdir(save_dir)
    checks["save_dir_exists"] = dir_exists
    log(f"  Save dir exists: {'PASS' if dir_exists else 'FAIL'} ({save_dir})")

    if not dir_exists:
        return checks

    contents = os.listdir(save_dir)
    log(f"  Save contents: {sorted(contents)}")

    has_adapter_config = "adapter_config.json" in contents
    checks["has_adapter_config"] = has_adapter_config
    log(f"  Has adapter_config.json: {'PASS' if has_adapter_config else 'FAIL'}")

    has_adapter_weights = any("adapter" in f and (f.endswith(".safetensors") or f.endswith(".bin")) for f in contents)
    checks["has_adapter_weights"] = has_adapter_weights
    log(f"  Has adapter weights: {'PASS' if has_adapter_weights else 'FAIL'}")

    return checks


def unwrap(trainer_model):
    """Unwrap FSDP/DDP/compile wrappers, keeping the (possibly PEFT) model for adapter access."""
    return extract_model_from_parallel(trainer_model, recursive=True)


def is_expert_lora_active(model) -> bool:
    return has_ep_lora(model)
