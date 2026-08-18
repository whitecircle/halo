#!/usr/bin/env python
"""
Test: merge_expert_lora_on_save gives a MIXED attention+expert LoRA run a SERVABLE checkpoint.

A mixed run trains two adapter kinds at once: stock PEFT on the attention projections, and native
grouped LoRA on the EP experts (grouped ``[E, K, r]`` tensors PEFT cannot wrap). Its adapter-only
save is resume-only — no merge tool folds it, because ``PeftModel.merge_and_unload`` drops every
expert delta. ``merge_expert_lora_on_save`` therefore routes the write through
``save_ep_checkpoint``, which folds BOTH halves: the expert deltas inside each family's gather,
and the attention deltas via ``merge_adapter`` held open for the duration of the write.

Each check fails when a piece of that wiring breaks:

  1. The run is genuinely mixed — a PeftModel AND native expert adapters. Everything below is
     vacuous otherwise.
  2. Both halves are live: disabling the adapters moves the logits by a measurable amount. Without
     this, check 4 would also pass against a base-quality checkpoint.
  3. Key hygiene: the merged checkpoint carries no ``base_model.`` prefix, no ``.base_layer`` infix,
     no ``lora_`` tensor and no ``original_module`` duplicate. A leaked PEFT prefix is exactly what
     makes ``from_pretrained`` load nothing and leave those weights randomly initialised.
  4. Value correctness: the tensors on disk are the live model's MERGED weights, gathered through the
     same seams the save uses. By distance, not bitwise — bf16 merge is not exactly reversible, and
     the save merges a second time from an already-rounded base — so each written tensor must sit far
     closer to the merged weight than to the frozen base, which an unmerged save lands exactly on.
     The base is compared too, to prove the merge moved something and the check is not vacuous.
  5. The save is non-destructive: every LoRA-wrapped base weight and every adapter is bit-identical
     afterwards, and the adapters are still trainable — so an intermediate merged checkpoint does not
     silently end the run's training (``merge_adapter`` is paired with ``unmerge_adapter``, unlike
     ``merge_and_unload``, which would dissolve the PeftModel).

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/lora/test_lora_mixed_merged_save.py

Requirements:
    - 2x GPUs (>=80GB), DeepEP installed. Model: unsloth/gpt-oss-20b-BF16 by default.
    - Scratch for one merged checkpoint (~model size) — TMPDIR must point at the large volume.
    - HALO_TEST_MODEL / HALO_TEST_REVISION / HALO_TEST_ATTN / HALO_TEST_EP / HALO_TEST_GC sweep the
      other MoE families, whose expert storage layouts and export renames differ.
"""

import glob
import json
import os
import sys
import traceback

import torch
from accelerate import PartialState
from accelerate.utils import extract_model_from_parallel
from safetensors import safe_open
from trl import SFTConfig

from src.checkpoint.format import SAFETENSORS_INDEX_FILE
from src.distributed.checkpoint.peft import find_peft_model
from src.distributed.expert_parallel.expert_weights import has_ep_lora
from src.distributed.fsdp import reshard_fsdp2_modules
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.env import env_flag, env_int, env_str
from src.models.structure import merged_adapters, normalize_peft_param_name
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import (
    cleanup_dirs,
    ensure_model_downloaded,
    init_distributed,
    setup_cache_dirs,
    teardown_distributed,
)
from tests.common.models import GPT_OSS_20B
from tests.common.peft_helpers import load_peft_model
from tests.common.utils import cleanup_memory, log

# Model/attn/EP are env-overridable exactly as in test_lora_ep_experts.py, so one file sweeps the
# other MoE storage layouts (fused-GLU, separate-GLU, per-expert-unfused) and the ep1 grouped-GEMM
# shape without a copy: each family's own gather_expert_state_dict does the fold, so the assertions
# below are family-agnostic. Default flex_attention keeps GptOss sinks live (the FA2 path sets
# self_attn.sinks=None, which the save then re-emits neutralized — a different tensor set).
MODEL_NAME = env_str("HALO_TEST_MODEL", GPT_OSS_20B)
MODEL_REVISION = env_str("HALO_TEST_REVISION") or None
ATTN_IMPL = env_str("HALO_TEST_ATTN", "flex_attention")
EP_SIZE = env_int("HALO_TEST_EP", 2)
# Zaya rejects EP + gradient checkpointing (its cross-layer EDA/CCA state recurses under recompute),
# so the sweep turns it off for that family rather than skipping the family.
GRADIENT_CHECKPOINTING = env_flag("HALO_TEST_GC", default=True)
MAX_STEPS = 3
MAX_SEQ_LENGTH = 512
LEARNING_RATE = 1e-3  # large so both adapter halves move clearly in 3 steps
SEED = 42
PROBE_TOKENS = 64

# Tensor-name fragments that must not survive into a servable checkpoint. Each is a distinct failure:
# a PEFT wrapper level, a LoRA-wrapped module's base weight, an adapter tensor, and the frozen
# duplicate PEFT keeps beside a modules_to_save copy.
_PEFT_KEY_ARTIFACTS = ("base_model.", ".base_layer", "lora_A", "lora_B", "original_module", "modules_to_save")

# Below this the adapters did not move enough in MAX_STEPS for anything downstream to be meaningful.
_MIN_ADAPTER_EFFECT = 1e-2
# One in-place bf16 merge -> unmerge on a merged base weight leaves (w+d)-d != w — the same rounding
# step every on-policy weight sync pays. Measured on this run's shapes: drift 0.0 (the round-trip
# reversed exactly), one rounding step ~2.4e-4, sampled LoRA delta 9.8e-4 — so the budget sits one
# step up while merge_drift_budget_below_lora_delta keeps the delta 3x above it, tolerance-proof.
_MERGE_ROUNDTRIP_TOL = 2.5e-4
# How much closer a written tensor must sit to the merged weight than to the frozen base. bf16 merge
# is not exactly reversible, so the checkpoint lands a rounding step from `expected` rather than on
# it; measured separation is 4x, and an UNMERGED save sits exactly on base (ratio 0), so anything
# above 1 both tolerates the rounding and still fails the bug this samples for.
MERGE_DISTANCE_MARGIN = 2.0


def _probe_batch(tokenizer, device):
    """A fixed, rank-identical batch. Every rank runs the same forward, so the EP dispatch
    collectives stay aligned and each rank computes the same logits."""
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
    """Local-shard copies of what ``merge_adapter`` touches (LoRA-wrapped base weights) and what it
    must leave alone (every adapter tensor, PEFT and native alike).

    ``to_local()`` rather than ``full_tensor()``: the comparison only needs each rank's own view, and
    a gather here would inject collectives around the save the production path does not have.
    """
    snapshot = {}
    for name, param in model.named_parameters():
        if ".base_layer." not in name and "lora_" not in name:
            continue
        data = param.data
        snapshot[name] = (data.to_local() if isinstance(data, torch.distributed.tensor.DTensor) else data).clone()
    return snapshot


def _checkpoint_keys(checkpoint_dir: str) -> list[str]:
    """Every tensor name in the checkpoint, from the index when there is one (same reason as
    :func:`_read_checkpoint_tensors`: no per-shard walk after the last collective)."""
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
    """Read just ``wanted`` from the checkpoint, via the index rather than by scanning every shard.

    Scanning matters: at 60 GB across 16 shards this runs on both ranks after the last collective,
    and DeepEP's ElasticBuffer carries a 100 s GPU-side timeout — the first rank to reach teardown's
    destroy barrier waits out that timeout for its peer and the job aborts. The index makes the read
    O(files actually holding a wanted key), which is one or two.
    """
    index_path = os.path.join(checkpoint_dir, SAFETENSORS_INDEX_FILE)
    if os.path.isfile(index_path):
        with open(index_path) as fh:
            weight_map = json.load(fh)["weight_map"]
        by_file: dict[str, list[str]] = {}
        for key in wanted:
            if key in weight_map:
                by_file.setdefault(weight_map[key], []).append(key)
        shards = {os.path.join(checkpoint_dir, name): keys for name, keys in by_file.items()}
    else:  # single-file checkpoint (small model): one shard holds everything
        shards = {path: list(wanted) for path in glob.glob(os.path.join(checkpoint_dir, "*.safetensors"))}

    found: dict[str, torch.Tensor] = {}
    for path, keys in sorted(shards.items()):
        with safe_open(path, framework="pt") as f:
            available = set(f.keys())
            found.update({key: f.get_tensor(key) for key in keys if key in available})
    return found


def _sample_merged_weights(model, checkpoint_keys: list[str]) -> dict[str, torch.Tensor]:
    """Full tensors for ``checkpoint_keys``, read off the live model under its CHECKPOINT names.

    The live tree spells them with PEFT's ``.base_layer`` infix and, at ep>1, holds them as FSDP2
    DTensor shards — so this resolves both through the very seams the save uses
    (``normalize_peft_param_name`` + ``full_tensor()``), which is what makes an equality comparison
    against the written bytes meaningful rather than a coincidence of naming.
    """
    out: dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        hf_name = normalize_peft_param_name(name, model.prefix)
        if hf_name not in checkpoint_keys:
            continue
        data = param.data
        full = data.full_tensor() if isinstance(data, torch.distributed.tensor.DTensor) else data
        out[hf_name] = full.detach().to(torch.bfloat16).cpu().clone()
    return out


def _pick_sample_keys(model) -> list[str]:
    """Checkpoint names of a few LoRA-wrapped weights, chosen from what the model actually adapted.

    Derived from the live tree rather than hardcoded module names, so this holds for any family
    whose projections the run targets. ``named_parameters()`` order is deterministic and identical
    on every rank, which is what lets the gather below stay collective-safe.
    """
    picked = [
        normalize_peft_param_name(name, model.prefix)
        for name, _ in model.named_parameters()
        if ".base_layer.weight" in name
    ]
    return [name for name in picked if name is not None][:4]


def run_mixed_merged_save_test() -> bool:
    rank, world_size, local_rank = init_distributed()
    log(f"\n{'=' * 70}\n  Mixed attention+expert LoRA -> merged servable checkpoint (ep{EP_SIZE})\n{'=' * 70}")
    if world_size < 2:
        log(f"\nERROR: requires at least 2 GPUs, got {world_size}")
        teardown_distributed()
        return False

    output_dir, cache_dir = setup_cache_dirs("lora_mixed_merged", rank)
    shared_out = [output_dir]
    torch.distributed.broadcast_object_list(shared_out, src=0)
    output_dir = shared_out[0]
    merged_dir = os.path.join(output_dir, "merged")
    checks: dict[str, bool] = {}

    try:
        ensure_model_downloaded(MODEL_NAME, rank)
        PartialState()

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
            max_length=MAX_SEQ_LENGTH,
            dataloader_drop_last=True,
            remove_unused_columns=False,
            dataloader_num_workers=0,
            fsdp="",
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
        # 1. Both adapter kinds must exist, or every assertion below is about a run that is not mixed.
        checks["run_is_mixed"] = peft_model is not None and has_ep_lora(unwrapped)
        log(f"  peft_model={peft_model is not None}, native expert adapters={has_ep_lora(unwrapped)}")

        log(f"\n[2/5] Training {MAX_STEPS} steps...")
        trainer.train()

        device = next(trainer.model.parameters()).device
        batch = _probe_batch(tokenizer, device)
        trained_logits = _logits(trainer.model, batch)
        # 2. Adapters off (EP-aware: make_disable_adapter_ep_aware drops the expert half too).
        with peft_model.disable_adapter():
            base_logits = _logits(trainer.model, batch)
        adapter_effect = (trained_logits - base_logits).abs().max().item()
        checks["adapters_changed_the_model"] = adapter_effect > _MIN_ADAPTER_EFFECT
        log(f"  adapter effect on logits: {adapter_effect:.4g} (floor {_MIN_ADAPTER_EFFECT})")

        # The probe forwards above left the FSDP2 modules holding their transient unsharded params;
        # the save reshards before it reads, so everything sampled or snapshotted below must read
        # the same resharded tensors — else the before/after comparison spans two registrations.
        reshard_fsdp2_modules(unwrapped)

        # Everything that needs a collective happens HERE, before the save. After it, the checks are
        # local reads and file I/O only — so a failure on one rank reports itself instead of leaving
        # its peer parked in a gather that will never complete.
        sample_keys = _pick_sample_keys(unwrapped)
        with merged_adapters(peft_model):
            expected = _sample_merged_weights(unwrapped, sample_keys)  # what the save must write
        unmerged = _sample_merged_weights(unwrapped, sample_keys)  # the frozen base, for non-vacuity
        # What merge_adapter mutates (LoRA-wrapped base weights) and what it must not (the adapters).
        before = _weight_snapshot(unwrapped)

        log("\n[3/5] Saving merged checkpoint (both halves folded)...")
        # No barrier after this: save_ep_model ends on DeferredRankFailure.reject(), a collective, so
        # the ranks are already in step. An extra one here is not just redundant — if any rank leaves
        # the body early, that spare barrier pairs with the one in `finally` and the survivors block
        # forever instead of reporting the real failure.
        trainer.save_model(merged_dir)

        # 5. merge_adapter must have been undone. Asserted on the weights rather than on a second
        # forward: it is the direct statement of the invariant, and a post-save EP forward would add
        # a DeepEP dispatch minutes after the previous one, which the NVLink barrier can time out on.
        after = _weight_snapshot(unwrapped)
        # Adapters and biases must come back bit-identical; each merged base WEIGHT carries the
        # save's own in-place merge -> unmerge rounding step, bounded by _MERGE_ROUNDTRIP_TOL.
        adapters_changed = sorted(k for k, v in after.items() if "lora_" in k and not torch.equal(v, before[k]))
        bias_changed = sorted(
            k for k, v in after.items() if "lora_" not in k and k.endswith(".bias") and not torch.equal(v, before[k])
        )
        weight_drift = {
            k: (v.float() - before[k].float()).abs().max().item()
            for k, v in after.items()
            if "lora_" not in k and not k.endswith(".bias")
        }
        drifted = sorted(k for k, d in weight_drift.items() if d > _MERGE_ROUNDTRIP_TOL)
        changed = adapters_changed + bias_changed + drifted
        checks["save_left_weights_unchanged"] = not changed
        # An unmerge that never ran leaves the whole LoRA delta on the base, which must sit clearly
        # above the drift budget — else the tolerance could mask exactly the failure this pins.
        sampled_delta = max((expected[k].float() - unmerged[k].float()).abs().max().item() for k in expected)
        checks["merge_drift_budget_below_lora_delta"] = sampled_delta > 3 * _MERGE_ROUNDTRIP_TOL
        checks["adapters_still_trainable"] = any(
            p.requires_grad for n, p in unwrapped.named_parameters() if "lora_" in n
        )
        log(
            f"  {len(before)} weights compared; changed: {changed[:3] or 'none'}; "
            f"max base-weight drift {max(weight_drift.values()):.2e} (tol {_MERGE_ROUNDTRIP_TOL}), "
            f"sampled LoRA delta {sampled_delta:.2e}; adapters trainable={checks['adapters_still_trainable']}"
        )

        # Both ranks run the remaining checks, on the same shared-FS directory and on their own
        # (identical) view of the model. Deliberately symmetric: a rank-0-only block would need the
        # verdict broadcast back, adding cross-rank machinery to the assertions rather than to the
        # code under test — and leaving the peer parked in that collective while rank 0 works.
        log("\n[4/5] Checking merged checkpoint key hygiene...")
        keys = _checkpoint_keys(merged_dir)
        offenders = sorted({a for a in _PEFT_KEY_ARTIFACTS if any(a in k for k in keys)})
        checks["no_peft_artifacts_in_keys"] = not offenders and len(keys) > 0
        log(f"  {len(keys)} tensors, PEFT artifacts: {offenders or 'none'}")

        log("\n[5/5] Comparing the written tensors against the live MERGED weights...")
        written = _read_checkpoint_tensors(merged_dir, sample_keys)

        # Compared by DISTANCE, not bitwise: `expected` comes from this test's own merge, which is then
        # undone, and save_model merges a SECOND time from the already-rounded base — in bf16
        # (w+d)-d+d != w+d, so the two agree to a rounding step. The frozen base is the discriminator:
        # an unmerged save lands exactly ON it, a merged one sits a rounding step from `expected` and a
        # whole LoRA delta away from base (measured: 4x). A key absent from the checkpoint counts as a
        # mismatch, not a KeyError — "the save dropped it" is one of the failures this catches.
        def _is_the_merged_weight(key: str) -> bool:
            if key not in written:
                return False
            to_merged = (written[key].float() - expected[key].float().cpu()).abs().max()
            to_base = (written[key].float() - unmerged[key].float().cpu()).abs().max()
            return bool(to_merged * MERGE_DISTANCE_MARGIN < to_base)

        mismatched = sorted(k for k in sample_keys if not _is_the_merged_weight(k))
        checks["written_tensors_are_the_merged_weights"] = bool(sample_keys) and not mismatched
        log(f"  {len(sample_keys)} sampled tensors ({sample_keys[:2]}…), mismatched: {mismatched or 'none'}")
        # Non-vacuity: an unmerged save would write the frozen base, which is NOT what we compared to.
        checks["merge_actually_changed_the_weights"] = any(
            not torch.equal(unmerged[k], expected[k]) for k in sample_keys
        )
        log(f"  merge moved at least one sampled tensor: {checks['merge_actually_changed_the_weights']}")

        all_passed = all(checks.values())

        log(f"\n{'=' * 70}\n  MIXED MERGED-SAVE TEST " + ("PASSED" if all_passed else "FAILED"))
        for name, ok in checks.items():
            log(f"  - {'PASS' if ok else 'FAIL'}: {name}")
        log(f"{'=' * 70}")
        return all_passed

    except Exception as e:
        print(f"\n  [rank{rank}] MIXED MERGED-SAVE TEST FAILED WITH ERROR: {e}", flush=True)
        traceback.print_exc()
        return False

    finally:
        barrier()
        cleanup_dirs(output_dir, cache_dir)
        cleanup_memory()


def main():
    success = run_mixed_merged_save_test()
    teardown_distributed()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
