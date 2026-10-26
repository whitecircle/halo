"""The LoRA SFT rows: adapters trained on one model under one two-rank parallel shape through the
production adapter path (:func:`~tests.common.sft_modes.run_sft_mode`), then checked on the adapters: only
they train and every one moves, and per row a save reloads to them on a fresh unsharded base, the save
restores onto the zeroed live adapters (the resume seam's key remap), or a 25-step run's loss falls window
over window.
"""

import os
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch
from safetensors.torch import load_file

from halo.checkpoint.format import ADAPTER_SAFETENSORS_FILE
from halo.checkpoint.parallel.adapters import restore_adapters
from halo.distributed.fsdp import reshard_fsdp2_modules
from halo.runtime.collectives import barrier
from tests.common.adapters import (
    adapter_param_items,
    adapter_save_checks,
    assert_adapters_moved,
    assert_only_adapters_trainable,
    portable_adapter_key,
    snapshot_adapters,
    unwrap,
    verify_adapter_reload,
)
from tests.common.distributed import shared_output_dir
from tests.common.models import BAILING_MOE_RING_MINI, GPT_OSS_20B, QWEN3_0_6B
from tests.common.numerics import max_or_nan
from tests.common.rank_log import log
from tests.common.sft_modes import SFTMode, SFTSuite, TrainerChecks, load_suite_tokenizer, run_sft_mode
from tests.common.training_checks import LM_TRAINING_LOSS_BAND, step_losses

WORLD_SIZE = 2


@dataclass(frozen=True)
class LoraModel:
    """A checkpoint the rows train: its attention kernel, the modules LoRA targets, and its SFT arguments over
    :data:`~tests.common.sft_modes.BASE_SFT_ARGS`."""

    name: str
    attn_implementation: str
    lora_target_modules: tuple[str, ...]
    sft_args: Mapping[str, Any]


MODELS = {
    "gpt_oss_20b": LoraModel(
        GPT_OSS_20B, "flex_attention", ("q_proj", "v_proj"), {"learning_rate": 2e-4, "max_length": 2048}
    ),
    # Bailing fuses QKV and names its attention output projection `dense`.
    "bailing": LoraModel(
        BAILING_MOE_RING_MINI,
        "sdpa",
        ("query_key_value", "dense", "g_proj"),
        {"gradient_accumulation_steps": 2, "learning_rate": 1e-5, "max_length": 4096},
    ),
    # Every linear layer, so a reload covers the MLP adapters too.
    "qwen3_0_6b": LoraModel(
        QWEN3_0_6B,
        "flash_attention_2",
        ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"),
        {"learning_rate": 2e-4, "max_length": 2048},
    ),
}
# ``ParallelismConfig`` kwargs per mode, on WORLD_SIZE ranks.
MODES = {
    "ep": {"ep_size": 2},
    "ep_cp": {"ep_size": 2, "cp_size": 2},
    "etp": {"ep_size": 1, "expert_tp_size": 2},
    "cp": {"cp_size": 2},
    "fsdp": {},
}


def _loss_window_checks(trainer) -> dict[str, bool]:
    """The step losses in 5-step window means: the last below the first and none above 1.5x the first; and the
    last step under the language-model band's ceiling, above the uniform baseline (ln of the vocabulary, about
    12), so only a late blow-up fails it."""
    losses = step_losses(trainer)
    windows = [sum(losses[start : start + 5]) / 5 for start in range(0, len(losses) - 4, 5)]
    log(f"  5-step window means: {[f'{window:.4f}' for window in windows]}")
    return {
        "window_loss_decreased": len(windows) >= 2 and windows[-1] < windows[0],
        "no_window_diverged": bool(windows) and max_or_nan(windows) <= 1.5 * windows[0],
        "final_loss_reasonable": bool(losses) and losses[-1] < LM_TRAINING_LOSS_BAND[1],
    }


def _saved_adapter_checks(
    trainer, save_dir: str, trained: dict[str, torch.Tensor], model_name: str
) -> dict[str, bool]:
    """COLLECTIVE: save the adapters; on rank 0, the files PEFT reloads are there, every saved key is a LoRA key
    in its portable PEFT spelling, and a fresh unsharded base (quantized as the trained one was) reloads them to
    their trained values."""
    trainer.save_model(save_dir)
    barrier()
    rank = trainer.args.process_index
    checks = adapter_save_checks(save_dir, rank)
    if rank == 0:
        path = os.path.join(save_dir, ADAPTER_SAFETENSORS_FILE)
        keys = list(load_file(path)) if os.path.isfile(path) else []
        log(f"  Saved adapter keys: {len(keys)}, e.g. {keys[:2]}")
        checks["saved_keys_lora"] = bool(keys) and all("lora_" in key for key in keys)
        # No DTensor, EP or gather artifact may leak into a key PEFT reloads by name.
        checks["saved_keys_portable"] = all(
            portable_adapter_key(key) == key and not any(part in key for part in ("DTensor", "ep_", "full_tensor"))
            for key in keys
        )
    checks |= verify_adapter_reload(
        save_dir,
        trained,
        model_name=model_name,
        tokenizer=trainer.processing_class,
        rank=rank,
        local_rank=trainer.args.local_process_index,
        quantization_config=getattr(unwrap(trainer.model).config, "quantization_config", None),
    )
    barrier()
    return checks


def _adapter_restore_checks(trainer, save_dir: str) -> dict[str, bool]:
    """COLLECTIVE: the resume seam onto this run's live adapters, every one zeroed first. ``restore_adapters``
    remaps the saved (portable) keys onto the live names, CP-wrapped under CP, and each live adapter must come
    back equal to its saved tensor."""
    peft_model = unwrap(trainer.model)
    with torch.no_grad():
        for _, param in adapter_param_items(peft_model):
            param.zero_()
    restore_adapters(save_dir, trainer.model, is_cp_mode=trainer.parallelism_config.is_cp_mode)
    restored = {portable_adapter_key(name): t for name, t in snapshot_adapters(peft_model, expert_lora=False).items()}
    saved = load_file(os.path.join(save_dir, ADAPTER_SAFETENSORS_FILE))
    covered = bool(saved) and restored.keys() == saved.keys()
    log(f"  Restored {len(restored)} live adapters from {len(saved)} saved tensors")
    return {
        "restore_covers_saved_keys": covered,
        "restored_equal_saved": covered and all(torch.equal(restored[k].to(saved[k].dtype), saved[k]) for k in saved),
        "restored_nonzero": any(bool(tensor.abs().amax() > 0) for tensor in restored.values()),
    }


def _adapter_checks(
    trainer, *, model_name: str, save_dir: str | None, resume: bool, convergence: bool
) -> TrainerChecks:
    """Read the built run's adapters (only they train) and return the checks to run after training: every adapter
    moved, and the row's own checks."""
    peft_model = unwrap(trainer.model)
    trainable, why = assert_only_adapters_trainable(peft_model)
    log(f"  Only adapters trainable: {'PASS' if trainable else 'FAIL'} ({why})")
    before = snapshot_adapters(peft_model, expert_lora=False)

    def after_train(trainer) -> dict[str, bool]:
        reshard_fsdp2_modules(peft_model)
        trained = snapshot_adapters(peft_model, expert_lora=False)
        moved, why = assert_adapters_moved(before, trained, require_all=True)
        log(f"  Every adapter moved: {'PASS' if moved else 'FAIL'} ({why})")
        checks = {"only_adapters_trainable": trainable, "every_adapter_moved": moved}
        if convergence:
            checks |= _loss_window_checks(trainer)
        if save_dir is not None:
            checks |= _saved_adapter_checks(trainer, save_dir, trained, model_name)
        if resume:
            checks |= _adapter_restore_checks(trainer, save_dir)
        return checks

    return after_train


def run_lora_sft(
    ctx,
    *,
    model: str,
    mode: str,
    attn: str | None = None,
    qlora: bool = False,
    convergence: bool = False,
    save_reload: bool = False,
    resume: bool = False,
) -> dict:
    """Train ``model`` under ``mode`` with LoRA on its target modules (on a 4-bit base with ``qlora``, on the
    ``attn`` kernel when given), check its training loss against the language-model band and return the harness
    result. ``convergence`` runs 25 steps on 256 samples and judges the loss curve; ``save_reload`` saves and
    reloads the adapters; ``resume`` restores that save onto the run.
    """
    spec = MODELS[model]
    save_dir = os.path.join(shared_output_dir(ctx), "adapter") if save_reload else None
    suite = SFTSuite(
        model_name=spec.name,
        attn_implementation=spec.attn_implementation,
        sft_args={**spec.sft_args, "max_steps": 25 if convergence else 5},
        num_samples=(256 if convergence else 32, 0),
        evaluate=False,
        loss_band=LM_TRAINING_LOSS_BAND,
        # Liger cannot patch a 4-bit base.
        use_liger_kernel=not qlora,
    )
    sft_mode = SFTMode(
        MODES[mode],
        attn_implementation=attn,
        adapters={"mode": "qlora" if qlora else "lora", "lora_target_modules": list(spec.lora_target_modules)},
        before_train=partial(
            _adapter_checks, model_name=spec.name, save_dir=save_dir, resume=resume, convergence=convergence
        ),
    )
    return {"checks": run_sft_mode(ctx, suite, mode, sft_mode, load_suite_tokenizer(ctx, spec.name))}
