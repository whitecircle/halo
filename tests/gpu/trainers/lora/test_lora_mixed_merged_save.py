#!/usr/bin/env python
"""``merge_expert_lora_on_save`` gives a mixed attention+expert LoRA run a servable checkpoint.

``PeftModel.merge_and_unload`` drops every expert delta, so the save routes through ``save_ep_gathered_checkpoint``,
which folds both halves. A leaked PEFT prefix makes ``from_pretrained`` load nothing and leave those weights randomly
initialized. Needs scratch for one merged checkpoint, so TMPDIR must point at the large volume.
"""

import glob
import json
import os

import torch
from accelerate.utils import extract_model_from_parallel
from safetensors import safe_open
from trl import SFTConfig

from halo.checkpoint.format import SAFETENSORS_INDEX_FILE
from halo.distributed.expert_parallel.moe_layer.expert_lora import model_has_expert_lora
from halo.distributed.fsdp import reshard_fsdp2_modules
from halo.distributed.parallelism_config import ParallelismConfig
from halo.env import env_flag, env_int, env_str
from halo.models.introspection.unwrapping import find_peft_model, plain_peft_param_name
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.adapters import load_peft_model
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import ensure_model_downloaded
from tests.common.harness import gpu_test_main
from tests.common.models import GPT_OSS_20B
from tests.common.rank_log import log
from tests.common.weight_sync import lora_bases_and_merges

# One file sweeps the other MoE storage layouts: each family's gather_expert_state_dict does the fold, so the
# assertions are family-agnostic. Default flex_attention keeps GptOss sinks live (the FA2 path sets
# self_attn.sinks=None, which the save re-emits neutralized, a different tensor set).
MODEL_NAME = env_str("HALO_TEST_MODEL", GPT_OSS_20B)
MODEL_REVISION = env_str("HALO_TEST_REVISION") or None
ATTN_IMPL = env_str("HALO_TEST_ATTN", "flex_attention")
EP_SIZE = env_int("HALO_TEST_EP", 2)
# Zaya rejects EP + gradient checkpointing (its cross-layer EDA/CCA state recurses under recompute).
GRADIENT_CHECKPOINTING = env_flag("HALO_TEST_GC", default=True)
MAX_STEPS = 3
MAX_SEQ_LENGTH = 512
LEARNING_RATE = 1e-3  # Large so both adapter halves move clearly in 3 steps.
SEED = 42
PROBE_TOKENS = 64

# Each is a distinct failure: a PEFT wrapper level, a LoRA-wrapped base weight, an adapter tensor, and the frozen
# duplicate PEFT keeps beside a modules_to_save copy.
_PEFT_KEY_ARTIFACTS = ("base_model.", ".base_layer", "lora_A", "lora_B", "original_module", "modules_to_save")

_MIN_ADAPTER_EFFECT = 1e-2


def _probe_batch(tokenizer, device):
    """Rank-identical, so the EP dispatch collectives stay aligned and each rank computes the same logits."""
    text = "The capital of France is Paris, and the capital of Japan is Tokyo. " * 8
    ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=PROBE_TOKENS)
    return {k: v.to(device) for k, v in ids.items()}


@torch.no_grad()
def _logits(model, batch):
    was_training = model.training
    model.eval()
    try:
        return model(**batch).logits.float().cpu()
    finally:
        model.train(was_training)


def _weight_snapshot(model) -> dict[str, torch.Tensor]:
    """Local-shard copies of LoRA-wrapped base weights and every adapter. ``to_local()``: a gather would inject
    collectives around the save the production path does not have."""
    snapshot = {}
    for name, param in model.named_parameters():
        if ".base_layer." not in name and "lora_" not in name:
            continue
        data = param.data
        snapshot[name] = (data.to_local() if isinstance(data, torch.distributed.tensor.DTensor) else data).clone()
    return snapshot


def _checkpoint_keys(checkpoint_dir: str) -> list[str]:
    """From the index when there is one, for the reason in :func:`_read_checkpoint_tensors`."""
    index_path = os.path.join(checkpoint_dir, SAFETENSORS_INDEX_FILE)
    if os.path.isfile(index_path):
        with open(index_path) as fh:
            return list(json.load(fh)["weight_map"])
    keys: list[str] = []
    for path in sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors"))):
        with safe_open(path, framework="pt") as f:
            keys.extend(f.keys())
    return keys


def _read_checkpoint_tensors(checkpoint_dir: str, wanted: list[str]) -> dict[str, torch.Tensor]:
    """Read via the index: at 60 GB across 16 shards a scan runs on both ranks after the last collective, and DeepEP's
    100 s GPU-side timeout aborts the job when the first rank at teardown's destroy barrier waits out its peer."""
    index_path = os.path.join(checkpoint_dir, SAFETENSORS_INDEX_FILE)
    if os.path.isfile(index_path):
        with open(index_path) as fh:
            weight_map = json.load(fh)["weight_map"]
        by_file: dict[str, list[str]] = {}
        for key in wanted:
            if key in weight_map:
                by_file.setdefault(weight_map[key], []).append(key)
        shards = {os.path.join(checkpoint_dir, name): keys for name, keys in by_file.items()}
    else:  # Single-file checkpoint: one shard holds everything.
        shards = {path: list(wanted) for path in glob.glob(os.path.join(checkpoint_dir, "*.safetensors"))}

    found: dict[str, torch.Tensor] = {}
    for path, keys in sorted(shards.items()):
        with safe_open(path, framework="pt") as f:
            available = set(f.keys())
            found.update({key: f.get_tensor(key) for key in keys if key in available})
    return found


def _sample_merged_weights(model, checkpoint_keys: list[str]) -> dict[str, torch.Tensor]:
    """Full tensors under checkpoint names, resolved through the seams the save uses (``plain_peft_param_name`` +
    ``full_tensor()``): the live tree has PEFT's ``.base_layer`` infix and, at ep>1, FSDP2 DTensor shards."""
    out: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        hf_name = plain_peft_param_name(name, model.prefix)
        if hf_name not in checkpoint_keys:
            continue
        data = param.data
        full = data.full_tensor() if isinstance(data, torch.distributed.tensor.DTensor) else data
        out[hf_name] = full.detach().to(torch.bfloat16).cpu().clone()
    return out


def _pick_sample_keys(model) -> list[str]:
    """Checkpoint names of a few LoRA-wrapped weights. ``named_parameters()`` order is identical on every rank, which
    keeps the gather below collective-safe."""
    picked = [
        plain_peft_param_name(name, model.prefix)
        for name, _ in model.named_parameters()
        if ".base_layer.weight" in name
    ]
    return [name for name in picked if name is not None][:4]


def run(ctx) -> dict:
    log(f"\n{'=' * 70}\n  Mixed attention+expert LoRA -> merged servable checkpoint (ep{EP_SIZE})\n{'=' * 70}")

    shared_out = [ctx.output_dir]
    torch.distributed.broadcast_object_list(shared_out, src=0)
    output_dir = shared_out[0]
    merged_dir = os.path.join(output_dir, "merged")
    checks: dict[str, bool] = {}

    ensure_model_downloaded(MODEL_NAME, ctx.rank)

    log(f"\n[1/5] Loading {MODEL_NAME} with MIXED targets (attention PEFT + native expert LoRA)...")
    parallelism_config = ParallelismConfig(ep_size=EP_SIZE, merge_expert_lora_on_save=True)
    model, tokenizer, peft_config = load_peft_model(
        "mixed",
        parallelism_config,
        model_name=MODEL_NAME,
        revision=MODEL_REVISION,
        attn_implementation=ATTN_IMPL,
        use_liger_kernel=False,
    )

    config = SFTConfig(
        output_dir=output_dir,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="constant",
        bf16=True,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        use_liger_kernel=False,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        logging_nan_inf_filter=False,
        max_length=MAX_SEQ_LENGTH,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    trainer = DistributedSFTTrainer(
        model=model,
        args=config,
        train_dataset=create_sft_dataset(64, tokenizer, seed=SEED),
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
        peft_config=peft_config,
    )

    unwrapped = extract_model_from_parallel(trainer.model, recursive=True)
    peft_model = find_peft_model(unwrapped)
    # Every assertion below is vacuous unless both adapter kinds exist.
    checks["run_is_mixed"] = peft_model is not None and model_has_expert_lora(unwrapped)
    log(f"  peft_model={peft_model is not None}, native expert adapters={model_has_expert_lora(unwrapped)}")

    log(f"\n[2/5] Training {MAX_STEPS} steps...")
    trainer.train()

    device = next(trainer.model.parameters()).device
    batch = _probe_batch(tokenizer, device)
    trained_logits = _logits(trainer.model, batch)
    # EP-aware: install_ep_aware_disable_adapter drops the expert half too.
    with peft_model.disable_adapter():
        base_logits = _logits(trainer.model, batch)
    adapter_effect = (trained_logits - base_logits).abs().max().item()
    checks["adapters_changed_the_model"] = adapter_effect > _MIN_ADAPTER_EFFECT
    log(f"  adapter effect on logits: {adapter_effect:.4g} (floor {_MIN_ADAPTER_EFFECT})")

    # The probe forwards left transient unsharded params and the save reshards before it reads, so everything sampled
    # below must read the same resharded tensors.
    reshard_fsdp2_modules(unwrapped)

    # Every collective happens before the save. After it only local reads and file I/O remain, so a failure on one rank
    # reports itself instead of leaving its peer parked in a gather.
    sample_keys = _pick_sample_keys(unwrapped)
    _bases, merges = lora_bases_and_merges(peft_model)
    expected = {
        key: merges[name].to(torch.bfloat16).cpu()
        for name in merges
        if (key := plain_peft_param_name(name, peft_model.prefix)) in sample_keys
    }
    del merges
    unmerged = _sample_merged_weights(unwrapped, sample_keys)  # The frozen base, for non-vacuity.
    before = _weight_snapshot(unwrapped)

    log("\n[3/5] Saving merged checkpoint (both halves folded)...")
    # No barrier: the gathered EP save ends on DeferredRankFailure.reject(), a collective.
    trainer.save_model(merged_dir)

    # Asserted on the weights: a post-save EP forward would add a DeepEP dispatch minutes after the previous one, which
    # the NVLink barrier can time out on. Bit-identical, since an unmerge's (w+d)-d misses w by a rounding step in
    # bf16.
    after = _weight_snapshot(unwrapped)
    changed = sorted(k for k, v in after.items() if not torch.equal(v, before[k]))
    checks["save_left_weights_unchanged"] = not changed
    checks["adapters_still_trainable"] = any(p.requires_grad for n, p in unwrapped.named_parameters() if "lora_" in n)
    log(
        f"  {len(before)} weights compared; changed: {changed[:3] or 'none'}; "
        f"adapters trainable={checks['adapters_still_trainable']}"
    )

    # Both ranks run the rest: a rank-0-only block would leave the peer parked in a verdict broadcast.
    log("\n[4/5] Checking merged checkpoint key hygiene...")
    keys = _checkpoint_keys(merged_dir)
    offenders = sorted({a for a in _PEFT_KEY_ARTIFACTS if any(a in k for k in keys)})
    checks["no_peft_artifacts_in_keys"] = not offenders and len(keys) > 0
    log(f"  {len(keys)} tensors, PEFT artifacts: {offenders or 'none'}")

    log("\n[5/5] Comparing the written tensors against the live MERGED weights...")
    written = _read_checkpoint_tensors(merged_dir, sample_keys)

    # Bitwise. A key absent from the checkpoint counts as a mismatch, since the save dropping it is one of the
    # failures.
    def _is_the_merged_weight(key: str) -> bool:
        return key in written and torch.equal(written[key], expected[key])

    mismatched = sorted(k for k in sample_keys if not _is_the_merged_weight(k))
    checks["written_tensors_are_the_merged_weights"] = bool(sample_keys) and not mismatched
    log(f"  {len(sample_keys)} sampled tensors ({sample_keys[:2]}…), mismatched: {mismatched or 'none'}")
    # Non-vacuity: an unmerged save would write the frozen base.
    checks["merge_actually_changed_the_weights"] = any(not torch.equal(unmerged[k], expected[k]) for k in sample_keys)
    log(f"  merge moved at least one sampled tensor: {checks['merge_actually_changed_the_weights']}")

    # Every rank reads rank 0's merged dir, and rank 0's teardown removes it.
    ctx.barrier()
    return {"checks": checks}


main = gpu_test_main(min_world_size=2, prefix="lora_mixed_merged")(run)

if __name__ == "__main__":
    main()
