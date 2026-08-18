#!/usr/bin/env python
"""
Test: NATIVE grouped-LoRA on EP-distributed MoE EXPERTS (EP=2, GptOss-20B).

Unlike test_lora_ep.py (attention-only PEFT LoRA under EP), this validates LoRA adapters on the
expert FFNs themselves — grouped A/B tensors built inside the EP layers, applied in the grouped-GEMM
compute, and gradient-synced/checkpointed via the existing EP machinery. Checks, each of which FAILS
when a piece of the wiring breaks:

  1. Adapters exist (has_ep_lora) and only ``*_lora_A``/``*_lora_B`` are trainable inside EP layers.
  2. Init delta == 0: a forward with adapters present is bit-identical to one with them disabled
     (B initialized to zeros), proving _expert_proj adds exactly the LoRA delta and nothing else.
  3. After 3 steps: the frozen base experts are byte-unchanged while the adapters moved (B left zero
     at init, so it MUST become non-zero) — the adapters, not the base, carry the update.
  4. Save -> reload round-trip: gather adapters, zero the live ones, re-apply the gathered state, and
     confirm the adapters are restored exactly (gather/load symmetry across the EP group).

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/lora/test_lora_ep_experts.py

Requirements:
    - 2x GPUs (>=80GB), DeepEP installed. Model: unsloth/gpt-oss-20b-BF16 (auto-downloaded).
"""

import math
import os

import torch
from accelerate.utils import extract_model_from_parallel
from safetensors.torch import load_file
from transformers import AutoTokenizer
from trl import SFTConfig

from src.distributed.expert_parallel.base_layer import EPMoELayerBase, disable_expert_adapters
from src.distributed.expert_parallel.config import ExpertLoraSpec
from src.distributed.expert_parallel.expert_weights import apply_ep_lora_adapters, gather_ep_lora_adapters, has_ep_lora
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.env import env_int, env_str
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import ensure_model_downloaded
from tests.common.ep_reference import ep_layers
from tests.common.harness import gpu_test_main
from tests.common.models import GPT_OSS_20B
from tests.common.peft_helpers import freeze_base_keep_expert_adapters
from tests.common.utils import gpu_mem_gb, log

# Model is GptOss-20B by default; the cross-architecture sweep overrides via env (HALO_TEST_MODEL /
# HALO_TEST_ATTN) to exercise the other storage layouts (fused-GLU, separate-GLU) without new files.
MODEL_NAME = env_str("HALO_TEST_MODEL", GPT_OSS_20B)
ATTN_IMPL = env_str("HALO_TEST_ATTN", "flex_attention")
# Parallelism is env-overridable so one file covers EP-only / pure-ETP / EP+ETP / EP+CP. EP=2 default;
# the sweep runs ep8 (nproc=8), etp2 (ep1), ep2+etp2 (nproc=4), ep2+cp2 (nproc=4). Under any ETP shape
# EPConfig refuses expert adapters outright, so those rows assert the refusal, not a sharded gather.
EP_SIZE = env_int("HALO_TEST_EP", 2)
EXPERT_TP_SIZE = env_int("HALO_TEST_ETP", 1)
CP_SIZE = env_int("HALO_TEST_CP", 1)
MAX_STEPS = 3
BATCH_SIZE = 1
MAX_SEQ_LENGTH = 1024
LEARNING_RATE = 1e-3  # large so the adapters move clearly in 3 steps
LORA_R = 8
LORA_ALPHA = 16
NUM_TRAIN_SAMPLES = 32
SEED = 42


def _adapter_param_pairs(model):
    """(name, param) for every native grouped-LoRA adapter parameter in the model."""
    out = []
    for layer_name, module in model.named_modules():
        if isinstance(module, EPMoELayerBase):
            for attr in module._expert_lora_attrs:
                for which in ("A", "B"):
                    pname = f"{attr}_lora_{which}"
                    out.append((f"{layer_name}.{pname}", getattr(module, pname)))
    return out


def run(ctx) -> dict:
    log(f"\n{'=' * 70}")
    log("  Native expert-LoRA test")
    log(f"{'=' * 70}")
    log(f"  World size: {ctx.world_size}, ep={EP_SIZE}, expert_tp={EXPERT_TP_SIZE}, cp={CP_SIZE}, model: {MODEL_NAME}")
    log(f"  LoRA r={LORA_R}, alpha={LORA_ALPHA}, projections=gate/up/down")

    log("\n[1/8] Ensuring model downloaded...")
    ensure_model_downloaded(MODEL_NAME, ctx.rank)

    log("\n[2/8] Loading tokenizer + dataset...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)

    log("\n[3/8] Loading model with EP=2 + native expert-LoRA spec...")
    parallelism_config = ParallelismConfig(ep_size=EP_SIZE, expert_tp_size=EXPERT_TP_SIZE, cp_size=CP_SIZE)
    # Mirror the trainer script: attach the spec BEFORE load so EP layers build adapters at init.
    parallelism_config.expert_lora = ExpertLoraSpec(
        r=LORA_R, alpha=LORA_ALPHA, projections=frozenset({"gate", "up", "down"})
    )
    model, _ = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation=ATTN_IMPL,
        use_liger_kernel=True,
    )
    freeze_base_keep_expert_adapters(model)
    log(f"  GPU memory after load: {gpu_mem_gb():.1f}GB")

    checks = {}

    # --- Check 1: adapters present; only adapters trainable inside EP layers ---
    log("\n[4/8] Checking adapter presence + trainability...")
    checks["has_expert_lora"] = has_ep_lora(model)
    adapters = _adapter_param_pairs(model)
    checks["adapters_built"] = len(adapters) > 0
    log(f"  has_ep_lora={checks['has_expert_lora']}, adapter params={len(adapters)}")

    non_adapter_trainable = []
    for module in ep_layers(model):
        native = {f"{a}_lora_{w}" for a in module._expert_lora_attrs for w in ("A", "B")}
        for pname, p in module.named_parameters():
            if p.requires_grad and pname not in native:
                non_adapter_trainable.append(pname)
    checks["only_adapters_trainable_in_ep"] = not non_adapter_trainable
    if non_adapter_trainable:
        log(f"  WARNING non-adapter trainable in EP: {non_adapter_trainable[:5]}")

    # init recipe: every B zeros, every A non-zero
    b_all_zero = all(p.detach().abs().sum().item() == 0.0 for n, p in adapters if n.endswith("_lora_B"))
    a_any_nonzero = all(p.detach().abs().sum().item() > 0.0 for n, p in adapters if n.endswith("_lora_A"))
    checks["init_B_zero_A_nonzero"] = b_all_zero and a_any_nonzero
    log(f"  init B==0: {b_all_zero}, init A!=0: {a_any_nonzero}")

    # A must be drawn on PEFT's scale: its 2-D [r, K] kaiming_uniform_(a=sqrt(5)) bound reduces to
    # 1/sqrt(K). Running that on the 3-D [E, K, r] tensor folds r into fan_in and shrinks the bound
    # by sqrt(r), so expert and attention adapters would warm up on different scales in one run.
    a_scales_ok, a_scale_detail = True, []
    for name, param in adapters:
        if not name.endswith("_lora_A"):
            continue
        _, k, r = param.shape
        bound, observed = 1.0 / math.sqrt(k), param.detach().float().abs().max().item()
        # The two candidate bounds are 1/sqrt(k) and 1/sqrt(k*r) — a factor of sqrt(r) apart. Split
        # them at 0.7x rather than at the wrong bound itself: these are bf16 draws, and a value one
        # bf16 ulp above 1/sqrt(k*r) satisfies a strict `> bound/sqrt(r)` test while still being
        # drawn on the sqrt(r)-shrunk bound.
        in_band = bound * 0.7 < observed <= bound * 1.05
        a_scales_ok &= in_band
        a_scale_detail.append(f"{name}: max={observed:.5f} vs peft_bound={bound:.5f}")
    checks["init_A_matches_peft_scale"] = a_scales_ok
    for detail in a_scale_detail:
        log(f"  {detail}")

    # Against the independently computed value, not against spec.scaling — comparing the layer to
    # the spec it was assigned from would hold whatever the spec computed.
    spec = parallelism_config.expert_lora
    expected_scaling = LORA_ALPHA / math.sqrt(LORA_R) if spec.use_rslora else LORA_ALPHA / LORA_R
    checks["scaling_matches_spec"] = all(
        abs(m.expert_lora_scaling - expected_scaling) < 1e-6 for m in ep_layers(model)
    )
    log(f"  expert_lora_scaling={spec.scaling} (rslora={spec.use_rslora}, expected={expected_scaling})")

    # --- Check 2: init delta == 0 (adapters present vs disabled produce the same output) ---
    # B is zero-initialized so the LoRA delta is structurally 0; _expert_proj must therefore be a
    # no-op at init. We can't assert bit-equality directly because the EP scatter-back uses a
    # non-deterministic atomic index_add_ when top_k < ep_size (e.g. GptOss top-4 at EP=8), so two
    # forward passes of the SAME model differ by run-to-run noise. Measure that noise floor (two
    # adapters-on passes) and require the adapters-on-vs-off difference to stay within it.
    log("\n[5/8] Checking init delta == 0 (within the EP forward's run-to-run noise floor)...")
    enc = tokenizer(["The quick brown fox jumps over the lazy dog."], return_tensors="pt")
    input_ids = enc["input_ids"].to(ctx.local_rank)
    attn = enc["attention_mask"].to(ctx.local_rank)
    model.eval()

    def _fwd():
        return model(input_ids=input_ids, attention_mask=attn).logits.float().cpu()

    with torch.no_grad():
        out_with_a = _fwd()
        out_with_b = _fwd()  # same config twice → intrinsic nondeterminism noise floor
        with disable_expert_adapters(model):  # base-only path (the EP ref-pass complement)
            out_without = _fwd()
    model.train()
    noise = (out_with_a - out_with_b).abs().max().item()
    delta = (out_with_a - out_without).abs().max().item()
    # The adapter delta must be the SAME ORDER as the forward's run-to-run noise (≈0 ⇒ no-op).
    # On a deterministic scatter (top_k >= ep_size) noise≈0 and this is effectively bit-equality;
    # on the atomic-scatter path (top_k < ep_size, e.g. EP=8) toggling adapters changes the compute
    # graph and thus the atomic accumulation order, so allow a few× the 2-sample noise floor. A real
    # wiring bug (nonzero/duplicated delta) would make `delta` orders of magnitude larger than noise.
    checks["init_delta_zero"] = delta <= max(4.0 * noise, 1e-3)
    log(f"  adapter delta={delta:.3e} vs forward noise floor={noise:.3e} (same order ⇒ no-op)")

    # --- Snapshot gathered base + adapters before training (collective: all ranks) ---
    log("\n[6/8] Snapshotting gathered base + adapters...")
    sample_layer = ep_layers(model)[0]
    base_before = {k: v.clone() for k, v in sample_layer.gather_expert_state_dict().items()}
    adapters_before = gather_ep_lora_adapters(model)

    # --- Train ---
    log("\n[7/8] Training (3 steps)...")
    config = SFTConfig(
        output_dir=ctx.output_dir,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=LEARNING_RATE,
        bf16=True,
        gradient_checkpointing=True,
        use_liger_kernel=False,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        max_length=MAX_SEQ_LENGTH,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=True,
        fsdp="",
    )
    trainer = DistributedSFTTrainer(
        model=model,
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
    )
    barrier()
    train_result = trainer.train()
    log(f"  Training loss: {train_result.training_loss:.6f}, steps: {train_result.global_step}")
    checks["loss_finite"] = math.isfinite(train_result.training_loss)
    checks["steps_completed"] = train_result.global_step == MAX_STEPS

    # --- Check 3: base unchanged, adapters moved ---
    log("\n--- Validating updates ---")
    unwrapped = extract_model_from_parallel(trainer.model, recursive=True)
    sample_layer_after = ep_layers(unwrapped)[0]
    base_after = sample_layer_after.gather_expert_state_dict()
    base_unchanged = all(torch.equal(base_before[k].cpu(), base_after[k].cpu()) for k in base_before)
    checks["base_frozen"] = base_unchanged
    log(f"  base experts unchanged: {base_unchanged}")

    adapters_after = gather_ep_lora_adapters(unwrapped)
    moved = sum(1 for k in adapters_before if not torch.equal(adapters_before[k].cpu(), adapters_after[k].cpu()))
    # B started at zero, so the B tensors in particular must have moved.
    b_moved = any(
        k.endswith(".lora_B") and not torch.equal(adapters_before[k].cpu(), adapters_after[k].cpu())
        for k in adapters_before
    )
    checks["adapters_moved"] = moved > 0 and b_moved
    log(f"  adapter tensors changed: {moved}/{len(adapters_before)} (B moved: {b_moved})")

    # --- Check 4: gather -> zero -> apply round-trip restores adapters exactly ---
    log("\n[8/8] Save/reload (gather->load) round-trip...")
    gathered = gather_ep_lora_adapters(unwrapped)
    for _, p in _adapter_param_pairs(unwrapped):
        p.data.zero_()
    apply_ep_lora_adapters(unwrapped, gathered)
    restored = gather_ep_lora_adapters(unwrapped)
    roundtrip_ok = all(torch.equal(gathered[k].cpu(), restored[k].cpu()) for k in gathered)
    checks["roundtrip_restores_adapters"] = roundtrip_ok
    log(f"  gather/load round-trip exact: {roundtrip_ok}")

    # --- Check 5: trainer.save_model() writes a standalone adapter file (save_ep_checkpoint Case B) ---
    log("\n[+] trainer.save_model() adapter-only write...")
    apply_ep_lora_adapters(unwrapped, gathered)  # restore the trained adapters (zeroed above)
    save_dir = os.path.join(ctx.output_dir, "adapter")
    trainer.save_model(save_dir)  # collective: every rank participates in the gather
    barrier()
    if ctx.rank == 0:
        adapter_file = os.path.join(save_dir, "adapter_model.safetensors")
        expert_keys = (
            [k for k in load_file(adapter_file) if "experts." in k and ".lora_" in k]
            if os.path.exists(adapter_file)
            else []
        )
        checks["save_model_adapter_file"] = len(expert_keys) > 0
        log(f"  adapter_model.safetensors written with {len(expert_keys)} expert-lora keys")

    return {"checks": checks}


main = gpu_test_main(min_world_size=2, prefix="lora_ep_experts")(run)

if __name__ == "__main__":
    main()
