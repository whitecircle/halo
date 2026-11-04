"""Training-state sidecars on an export: the provenance a merge re-applies, balancing at its trained precision."""

import json
import os
import shutil

import torch

from halo.checkpoint.format import EXPORT_SIDECAR_FILES, ROUTER_BALANCING_BIASES_FILE, TRAINING_PROVENANCE_FILE
from halo.checkpoint.reading import checkpoint_shard_files, iter_checkpoint_tensors
from halo.models.attention.gpt_oss_sinks import SinksPolicy, apply_sinks_policy, stamped_sinks_policy
from halo.models.moe.balancing import apply_balancing_bias_sidecar, is_balancing_state_key

PROVENANCE_GPT_OSS_SINKS = "gpt_oss_attention_sinks"


def write_training_provenance(model, output_dir: str) -> None:
    """Record the sink policy a merge cannot recover: it rebuilds from the hub, whose GptOss sinks are always live,
    while a ``reset_sinks`` run trained neutralized ones. No stamp means nothing to record, never neutralized."""
    provenance = {}
    policy = stamped_sinks_policy(model)
    if policy is not None:
        provenance[PROVENANCE_GPT_OSS_SINKS] = policy
    if not provenance:
        return
    with open(os.path.join(output_dir, TRAINING_PROVENANCE_FILE), "w") as fh:
        json.dump(provenance, fh, indent=2)


def apply_training_provenance(model, adapter_dir: str) -> list[str]:
    """Re-apply what ``adapter_dir``'s provenance records; returns the actions taken, for the tool to report."""
    provenance_path = os.path.join(adapter_dir, TRAINING_PROVENANCE_FILE)
    if not os.path.isfile(provenance_path):
        return []
    with open(provenance_path) as fh:
        provenance = json.load(fh)
    actions = []
    if provenance.get(PROVENANCE_GPT_OSS_SINKS) == SinksPolicy.NEUTRALIZED:
        apply_sinks_policy(model, model.config, policy=SinksPolicy.NEUTRALIZED)
        actions.append(
            "Adapter trained with neutralized GptOss attention sinks — neutralized the merged "
            "model's sinks to match (dtype.min, contributes ~0, exactly as trained)."
        )
    return actions


def _balancing_tensors_at_trained_precision(model, source_dir: str) -> dict[str, torch.Tensor]:
    """The balancing tensors as they should be stored, where that differs from now: read from the source's shards (the
    load may have downcast them), else fp32 so the sidecar lands unrounded. Matched by module path, since an export may
    rename the slot and a router carries exactly one."""
    state = model.state_dict()
    balancing_keys = sorted(key for key in state if is_balancing_state_key(key))
    if not balancing_keys:
        return {}
    try:
        stored = dict(iter_checkpoint_tensors(source_dir, predicate=is_balancing_state_key))
    except (FileNotFoundError, NotADirectoryError):
        stored = {}
    stored_by_module = {key.rsplit(".", 1)[0]: tensor for key, tensor in stored.items()}
    restored: dict[str, torch.Tensor] = {}
    for key in balancing_keys:
        current = state[key]
        tensor = stored_by_module.get(key.rsplit(".", 1)[0], current.float())
        if tensor.dtype != current.dtype:
            restored[key] = tensor.to(device=current.device)
    return restored


def apply_training_sidecars(model, source_dir: str) -> list[str]:
    """Re-apply what ``source_dir``'s sidecars record; returns the actions taken. Every tool assembling an export
    follows this order: provenance, then balancing state at trained precision (a bf16 round trip loses the ~1e-3 sign
    steps), filled from ``router_balancing_biases.pt`` where the source ships no weights."""
    actions = apply_training_provenance(model, source_dir)

    restored = _balancing_tensors_at_trained_precision(model, source_dir)
    if restored:
        model.load_state_dict(restored, strict=False, assign=True)

    # A source shipping weights already holds the trained bias in its native slot; its sidecar is the
    # resume copy, keyed by the trainer's module tree, and travels with the artifact instead.
    sidecar_path = os.path.join(source_dir, ROUTER_BALANCING_BIASES_FILE)
    try:
        checkpoint_shard_files(source_dir)
        ships_weights = True
    except (FileNotFoundError, NotADirectoryError):
        ships_weights = False
    if ships_weights or not os.path.isfile(sidecar_path):
        return actions
    sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=True)
    applied, skipped = apply_balancing_bias_sidecar(model, sidecar)
    if applied:
        actions.append(f"Applied trained router balancing biases into {len(applied)} native slots.")
    if skipped:
        actions.append(
            f"WARNING: {len(skipped)} routers trained a TRANSIENT balancing bias this architecture "
            f"has no checkpoint slot for — the exported model serves without it (near-tied top-k "
            f"picks flip vs training): {skipped[:4]}{'...' if len(skipped) > 4 else ''}"
        )
    return actions


def copy_training_sidecars(source_dir: str, output_dir: str) -> None:
    """For a write whose aux copy skips ``source_dir`` (an unmerged PEFT save, a merge copying its base's files)."""
    for name in EXPORT_SIDECAR_FILES:
        source = os.path.join(source_dir, name)
        if os.path.isfile(source):
            shutil.copy2(source, os.path.join(output_dir, name))
