"""Shared LoRA, QLoRA and native-expert-LoRA helpers for the adapter GPU tests, driving the training scripts'
production path. Modes: ``lora`` (attention PEFT), ``qlora`` (LoRA on a 4-bit base, dense FSDP/DDP only),
``expert_lora`` (native grouped), ``mixed`` (both)."""

from __future__ import annotations

import copy
import glob
import json
import os
import re
import traceback
from types import SimpleNamespace

import torch
from accelerate.utils import extract_model_from_parallel
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from peft import PeftModel, inject_adapter_in_model
from peft.tuners.lora import LoraLayer
from safetensors.torch import load_file
from torch.distributed.tensor import DTensor
from transformers import AutoModelForCausalLM
from trl import get_quantization_config

from halo.checkpoint.adapters import has_adapter_weight_file
from halo.checkpoint.format import ADAPTER_CONFIG_FILE, ADAPTER_SAFETENSORS_FILE, SAFETENSORS_INDEX_FILE
from halo.checkpoint.reading import read_checkpoint_index
from halo.checkpoint.writing import cast_to_save_dtype
from halo.configs.script_arguments import ModelArguments
from halo.distributed.expert_parallel.moe_layer.expert_lora import gather_expert_lora_adapters
from halo.distributed.parallelism_config import ParallelismConfig
from halo.env import env_flag
from halo.model_loading.finalization import finalize_loaded_model
from halo.model_loading.parallel.loader import load_distributed_model
from halo.model_loading.parallel.peft_setup import configure_adapter_training, split_expert_lora_targets
from halo.models.introspection.unwrapping import plain_adapter_key, strip_peft_adapter_segment
from halo.runtime.collectives import barrier
from tests.common.ep_reference import ep_layers
from tests.common.memory import cleanup_memory
from tests.common.models import GPT_OSS_20B, QWEN3_0_6B
from tests.common.rank_log import log
from tests.common.tolerances import TOL

DENSE_MODEL = QWEN3_0_6B
MOE_MODEL = GPT_OSS_20B

LORA_R = 8
LORA_ALPHA = 16

_PEFT_ARGS = SimpleNamespace(unfreeze_layers_patterns=None, freeze_layers_patterns=None)


# Logical names: split_expert_lora_targets resolves them against whatever 3-D tensors the family stores.
_EXPERT_TARGETS = ["gate_proj", "up_proj", "down_proj"]
# Fallback for the dense/QLoRA modes, whose model is fixed (Qwen3-0.6B).
_DEFAULT_ATTENTION_TARGETS = ["q_proj", "v_proj"]
_MOE_MODES = frozenset({"expert_lora", "mixed"})

_ATTENTION_CONTAINERS = ("self_attn", "self_attention", "attention", "attn")
_ATTENTION_PROJ_RE = re.compile(rf"\.(?:{'|'.join(_ATTENTION_CONTAINERS)})\.([A-Za-z0-9_]+)\.weight$")
# Token mixers a hybrid family interleaves with attention (Qwen3.5's GatedDeltaNet).
_MIXER_MODULE_RE = re.compile(rf"\.(?:{'|'.join((*_ATTENTION_CONTAINERS, 'linear_attn'))})\.([A-Za-z0-9_]+)$")


def _materialize(t: torch.Tensor) -> torch.Tensor:
    """Clone a param to full CPU float, unfolding an FSDP2/TP DTensor shard (collective) first."""
    if isinstance(t, DTensor):
        t = t.full_tensor()
    return t.detach().float().cpu()


def attention_target_modules(model_name: str, revision: str | None = None) -> list[str]:
    """Attention projection leaf names for this family, read off the checkpoint's weight map. No fixed list generalizes
    (MLA families, Zaya) and PEFT ships no mapping for these model types. Falls back only for a single-file checkpoint;
    on the hub a missing repo or bad revision raises, while offline an uncached repo reads as single-file."""
    if os.path.isdir(model_name):
        weight_map = read_checkpoint_index(model_name, missing_ok=True).get("weight_map", {})
    else:
        try:
            index = hf_hub_download(model_name, SAFETENSORS_INDEX_FILE, revision=revision)
        except EntryNotFoundError:
            return list(_DEFAULT_ATTENTION_TARGETS)
        with open(index) as fh:
            weight_map = json.load(fh)["weight_map"]
    found = {m.group(1) for key in weight_map if (m := _ATTENTION_PROJ_RE.search(key))}
    # ``_proj`` only: the same container also holds q/k norms and GptOss's ``sinks``.
    return sorted(name for name in found if name.endswith("_proj")) or list(_DEFAULT_ATTENTION_TARGETS)


def mixer_linear_leaves(model: torch.nn.Module) -> list[str]:
    """Leaf names of every ``nn.Linear`` under a token mixer: the targets of a tiny single-file checkpoint with no
    index to read. A leaf a convolution also ends in (a vision patch embedding's ``proj`` convolution) is
    dropped, since PEFT matches by suffix and would adapt that module too."""
    named = list(model.named_modules())
    linear_leaves = {
        match.group(1)
        for name, module in named
        if isinstance(module, torch.nn.Linear) and (match := _MIXER_MODULE_RE.search(name))
    }
    shared = {
        leaf
        for name, module in named
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d))
        for leaf in linear_leaves
        if name.endswith(f".{leaf}")
    }
    return sorted(linear_leaves - shared)


def _mode_targets(mode: str, model_name: str, revision: str | None) -> list[str]:
    """``lora_target_modules`` for a mode; ``mixed`` names both namespaces, which split_expert_lora_targets peels
    apart."""
    if mode == "expert_lora":
        return list(_EXPERT_TARGETS)
    if mode == "mixed":
        return attention_target_modules(model_name, revision) + _EXPERT_TARGETS
    return attention_target_modules(model_name, revision)


def mixed_targets(attention: list[str]) -> list[str]:
    """``mixed`` targets over explicit ``attention``, for a family whose index does not spell projections as the module
    tree does."""
    return [*attention, *_EXPERT_TARGETS]


def model_name_for(mode: str, parallelism_config) -> str:
    return MOE_MODEL if (mode in _MOE_MODES or parallelism_config.has_ep_group) else DENSE_MODEL


def parallelism_config_for(mode: str, size: int):
    """The sharding a mode names by its ``_ep``/``_etp`` suffix (``size`` = the axis width)."""
    if mode.endswith("_etp"):
        return ParallelismConfig(ep_size=1, expert_tp_size=size)
    if mode.endswith("_ep") or mode in _MOE_MODES:
        return ParallelismConfig(ep_size=size)
    return ParallelismConfig()


def _build_model_config(
    mode: str, model_name: str, revision: str | None = None, targets: list[str] | None = None
) -> ModelArguments:
    targets = targets or _mode_targets(mode, model_name, revision)
    return ModelArguments(
        model_name_or_path=model_name,
        # split_expert_lora_targets reads the config off the hub, and a pinned family would peel against the wrong
        # revision.
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


def peft_model_config(
    mode: str,
    parallelism_config,
    *,
    model_name: str | None = None,
    revision: str | None = None,
    lora_target_modules: list[str] | None = None,
) -> ModelArguments:
    """The ``ModelConfig`` a mode trains with, its expert targets peeled into ``parallelism_config``. The peel runs
    before any load, as ``init_training_script`` does, leaving only attention targets on the config."""
    model_name = model_name or model_name_for(mode, parallelism_config)
    model_config = _build_model_config(mode, model_name, revision, targets=lora_target_modules)
    parallelism_config.expert_lora = split_expert_lora_targets(model_config)
    return model_config


def load_peft_model_from_config(
    model_config: ModelArguments,
    parallelism_config,
    *,
    attn_implementation: str | None = None,
    use_liger_kernel: bool = True,
    reset_sinks: bool = True,
):
    """:func:`load_peft_model` for a config the caller holds. Returns (model, tokenizer, peft_config)."""
    model, tokenizer = load_distributed_model(
        model_name_or_path=model_config.model_name_or_path,
        parallelism_config=parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=env_flag("HALO_TEST_ENV_GRPO_TRUST_REMOTE_CODE", True),
        revision=model_config.model_revision,
        attn_implementation=attn_implementation,
        use_liger_kernel=use_liger_kernel,
        quantization_config=get_quantization_config(model_config),
        reset_sinks=reset_sinks,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = configure_adapter_training(_PEFT_ARGS, model, model_config, "CAUSAL_LM")
    return model, tokenizer, peft_config


def load_peft_model(
    mode: str,
    parallelism_config,
    *,
    model_name: str | None = None,
    revision: str | None = None,
    attn_implementation: str | None = None,
    use_liger_kernel: bool = True,
    reset_sinks: bool = True,
    lora_target_modules: list[str] | None = None,
):
    """Load a model and adapters through the production path. Returns (model, tokenizer, peft_config). ``peft_config``
    is the attention config, or None for expert-LoRA-only. ``reset_sinks`` is forwarded: on-policy RL keeps GptOss's
    pretrained sinks live, and resetting them would train a different model from the one the engine samples."""
    model_config = peft_model_config(
        mode, parallelism_config, model_name=model_name, revision=revision, lora_target_modules=lora_target_modules
    )
    return load_peft_model_from_config(
        model_config,
        parallelism_config,
        attn_implementation=attn_implementation,
        use_liger_kernel=use_liger_kernel,
        reset_sinks=reset_sinks,
    )


def adapter_param_items(model) -> list[tuple[str, torch.Tensor]]:
    return [(n, p) for n, p in model.named_parameters() if "lora_" in n]


def is_lora_b_key(key: str) -> bool:
    """Whether a key names a zero-init ``lora_B`` matrix: mid-path in PEFT and checkpoint keys, at the end on native
    expert adapters."""
    return key.endswith((".lora_B", "_lora_B")) or ".lora_B." in key


def randomize_adapters(
    model: torch.nn.Module, *, std: float = 1.0, lora_b_only: bool = False, dtype: torch.dtype | None = None
) -> None:
    """Redraw ``model``'s adapter tensors from ``N(0, std²)`` in place, on ``dtype``'s grid (default: each tensor's
    own). ``lora_B`` starts at zero, so an untouched adapter folds to its base and a fold test could not tell the fold
    from a no-op."""
    with torch.no_grad():
        for name, param in adapter_param_items(model):
            if not lora_b_only or is_lora_b_key(name):
                param.copy_(torch.empty_like(param, dtype=dtype).normal_(std=std))


def check_only_adapters_trainable(model) -> tuple[bool, str]:
    """Every trainable param is an adapter or modules_to_save copy (a SEQ_CLS head must train), and at least one
    ``lora_*`` is trainable."""
    trainable_non_adapter = [
        n for n, p in model.named_parameters() if p.requires_grad and "lora_" not in n and "modules_to_save" not in n
    ]
    trainable_loras = [n for n, p in model.named_parameters() if p.requires_grad and "lora_" in n]
    if trainable_non_adapter:
        return False, f"non-adapter params are trainable: {trainable_non_adapter[:5]}"
    if not trainable_loras:
        return False, "no lora_* adapter params are trainable"
    return True, f"{len(trainable_loras)} lora adapter params trainable, base frozen"


def snapshot_adapters(model, *, expert_lora: bool) -> dict[str, torch.Tensor]:
    """Clone adapter values for a before/after comparison. Collective with ``expert_lora``: EP-sharded tensors gather
    on every rank."""
    if expert_lora:
        return {k: _materialize(v) for k, v in gather_expert_lora_adapters(model).items()}
    return {n: _materialize(p) for n, p in adapter_param_items(model) if p.requires_grad}


def check_adapters_moved(before: dict, after: dict, *, require_all: bool = False) -> tuple[bool, str]:
    """Every adapter is finite, at least one changed (all with ``require_all``, which a partly wired adapter fails),
    and a B moved."""
    nonfinite = [k for k, t in after.items() if not torch.isfinite(t).all()]
    if nonfinite:
        return False, f"{len(nonfinite)} adapters are non-finite after training (e.g. {nonfinite[:3]})"
    moved = [k for k in before if k in after and not torch.equal(before[k], after[k])]
    b_moved = [k for k in moved if is_lora_b_key(k)]
    if not moved:
        return False, "no adapter tensors changed after training"
    if require_all and len(moved) < len(before):
        unmoved = [k for k in before if k not in moved]
        return False, f"{len(unmoved)}/{len(before)} adapters did not move (e.g. {unmoved[:3]})"
    if not b_moved:
        return False, f"{len(moved)} adapters changed but no lora_B moved (B is zero-init; must move)"
    return True, f"{len(moved)}/{len(before)} adapters changed (lora_B moved: {len(b_moved)})"


def save_and_check_adapter_checkpoint(trainer, output_dir: str, rank: int, *, expert_lora: bool) -> tuple[bool, str]:
    """Save an adapter-only checkpoint and check it on disk: LoRA keys, all finite, one non-zero B.
    ``trainer.save_model`` is collective; only rank 0 reads the file back."""
    adapter_dir = os.path.join(output_dir, "adapter_ckpt")
    trainer.save_model(adapter_dir)
    barrier()
    if rank != 0:
        return True, "non-zero rank (save participated)"

    path = os.path.join(adapter_dir, ADAPTER_SAFETENSORS_FILE)
    if not os.path.exists(path):
        return False, f"{ADAPTER_SAFETENSORS_FILE} not written to {adapter_dir}"
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
    """Replicate configure_adapter_training's freeze for tests that build grouped expert adapters directly."""
    for p in model.parameters():
        p.requires_grad = False
    for module in ep_layers(model):
        for attr in module._expert_lora_attrs:
            getattr(module, f"{attr}_lora_A").requires_grad_(True)
            getattr(module, f"{attr}_lora_B").requires_grad_(True)


def adapter_save_checks(save_dir: str, rank: int) -> dict[str, bool]:
    """Whether an already-written adapter save has the files PEFT needs to reload it; only rank 0 reads the disk."""
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

    has_adapter_config = ADAPTER_CONFIG_FILE in contents
    checks["has_adapter_config"] = has_adapter_config
    log(f"  Has {ADAPTER_CONFIG_FILE}: {'PASS' if has_adapter_config else 'FAIL'}")

    has_adapter_weights = has_adapter_weight_file(save_dir)
    checks["has_adapter_weights"] = has_adapter_weights
    log(f"  Has adapter weights: {'PASS' if has_adapter_weights else 'FAIL'}")

    return checks


def portable_adapter_key(name: str) -> str:
    """A live adapter param name as the saver writes it: adapter segment and CP-wrapper artifacts dropped.
    Idempotent."""
    return plain_adapter_key(strip_peft_adapter_segment(name))


def verify_adapter_reload(
    save_dir: str,
    trained_lora: dict[str, torch.Tensor],
    *,
    model_name: str,
    tokenizer,
    rank: int,
    local_rank: int,
    quantization_config=None,
) -> dict[str, bool]:
    """Reload a saved attention adapter onto a fresh unsharded base and check every tensor restores its trained value.
    Rank 0 only. A clean ``from_pretrained`` is vacuous, since an all-zero adapter loads with finite logits.
    ``trained_lora`` is gathered on every rank."""
    if rank != 0:
        return {}
    checks = {}
    try:
        log("  Reloading base model for checkpoint verification...")
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map={"": local_rank},
            quantization_config=quantization_config,
        )
        # Without this seam a remote-code family's non-persistent buffers sit on uninitialized memory, and the check
        # measures the allocator.
        finalize_loaded_model(base_model)
        log(f"  Loading adapter from {save_dir}...")
        reloaded = PeftModel.from_pretrained(base_model, save_dir)

        restored = {
            portable_adapter_key(n): p.detach().float().cpu() for n, p in reloaded.named_parameters() if "lora_" in n
        }
        trained = {portable_adapter_key(n): t.float() for n, t in trained_lora.items()}
        missing = sorted(k for k in trained if k not in restored)
        differ = sorted(
            k for k in trained if k in restored and not torch.allclose(trained[k], restored[k], atol=TOL.weight_atol)
        )
        # A lora_B training never moved reloads as zero too, so equality alone could pass an adapter that carries
        # nothing.
        trained_b_live = any(is_lora_b_key(k) and bool(t.abs().amax() > 0) for k, t in trained.items())
        checks["adapter_reload"] = trained_b_live and not missing and not differ
        log(
            f"  Adapter reload restores trained values: {'PASS' if checks['adapter_reload'] else 'FAIL'} "
            f"({len(trained)} trained, live lora_B: {trained_b_live}, {len(missing)} missing {missing[:3]}, "
            f"{len(differ)} differ {differ[:3]})"
        )

        reloaded.eval()
        test_input = tokenizer("What is 2 + 2?", return_tensors="pt").to(f"cuda:{local_rank}")
        with torch.no_grad():
            output = reloaded(**test_input)
        logits_finite = torch.isfinite(output.logits).all().item()
        checks["reload_logits_finite"] = logits_finite
        log(f"  Reload logits finite: {'PASS' if logits_finite else 'FAIL'}")

        # Finiteness misses a zeroed buffer, whose model has a dead RoPE. Swept off the model's own non-persistent set.
        unset = [
            name
            for name, tensor in base_model.named_non_persistent_buffers()
            if tensor.is_floating_point()
            and tensor.numel()
            and not (torch.isfinite(tensor).all().item() and tensor.abs().amax().item() > 0)
        ]
        checks["reload_buffers_initialized"] = not unset
        log(f"  Reload buffers initialized: {'PASS' if not unset else f'FAIL {unset[:5]}'}")
        if not logits_finite:
            # Separates a bad gathered save (non-finite saved tensor) from a bad reloaded forward.
            for shard in sorted(glob.glob(os.path.join(save_dir, "*.safetensors"))):
                tensors = load_file(shard)
                bad = {k: v for k, v in tensors.items() if not torch.isfinite(v).all()}
                log(f"  [diag] {os.path.basename(shard)}: {len(bad)} non-finite of {len(tensors)} tensors")
                for key, val in list(bad.items())[:10]:
                    log(f"  [diag]   {key} shape={tuple(val.shape)} nan={val.isnan().sum().item()}")
            log(f"  [diag] logits nan={output.logits.isnan().sum().item()} inf={output.logits.isinf().sum().item()}")

        del reloaded, base_model
        cleanup_memory()
    except Exception as e:
        log(f"  Checkpoint reload FAILED: {e}")
        traceback.print_exc()
        checks["adapter_reload"] = False
    return checks


def unwrap(trainer_model):
    """Unwrap FSDP/DDP/compile wrappers, keeping the (possibly PEFT) model for adapter access."""
    return extract_model_from_parallel(trainer_model, recursive=True)


def merged_lora_targets(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """What PEFT's in-place merge writes into each LoRA target of ``model``, which it modifies, keyed by plain module
    names."""
    merged = {}
    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            module.merge()
            for key, value in module.get_base_layer().named_parameters():
                merged[f"{name}.{key}"] = value.detach()
    return merged


def injected_lora_merge(
    model: torch.nn.Module, lora_config, adapters: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """:func:`merged_lora_targets` with ``lora_config`` injected and ``adapters`` loaded: the oracle for a save that
    folds them."""
    inject_adapter_in_model(copy.deepcopy(lora_config), model)
    loaded = model.load_state_dict(adapters, strict=False)
    if loaded.unexpected_keys:
        raise AssertionError(f"the oracle does not take every adapter tensor: {loaded.unexpected_keys[:3]}")
    return merged_lora_targets(model)


def injected_lora_fold(
    model: torch.nn.Module, lora_config, adapters: dict[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    """What a save folding ``adapters`` writes per LoRA target of the backbone: its merge at the save dtype, on the
    host. ``adapters`` are keyed by the SentenceTransformer's names under ``prefix``; the result by the backbone's
    own."""
    device = next(model.parameters()).device
    merged = injected_lora_merge(
        model, lora_config, {key[len(prefix) :]: value.to(device) for key, value in adapters.items()}
    )
    return {key: cast_to_save_dtype(value.cpu()) for key, value in merged.items() if key.endswith(".weight")}
