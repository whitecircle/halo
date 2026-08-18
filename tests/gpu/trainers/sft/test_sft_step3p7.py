#!/usr/bin/env python
"""Step-3.7 EP=2 SFT smoke + gathered-save round-trip on a tiny random-init composite model.

Rank 0 materializes a tiny ``Step3p7ForConditionalGeneration`` checkpoint (real tokenizer vocab;
the family ships no text-only CausalLM sibling) on disk via ``save_pretrained`` — transformers'
HUB layout: no ``language_model`` prefix, per-layer fused-but-split ``moe.gate_proj`` /
``moe.up_proj`` / ``moe.down_proj`` tensors, ``moe.gate.weight`` + ``moe.router_bias``,
``share_expert.*``, the vendor-namespace vision tower. Then all ranks:

  1. Load it through ``load_distributed_model`` with EP=2 — the lazy loader route, which replays
     the family's hub conversion per key (``_HUB_CONVERSION_KEYS``: the prefix renames, the
     ``moe.*`` → ``mlp.*`` renames, the two-source ``moe.gate_proj + moe.up_proj → gate_up_proj``
     fan-in sliced through both sources, the scoped Step-3.5 vision tower) on a HETEROGENEOUS
     config (per-layer 4-vs-2 attention heads); its bit-exactness against ``from_pretrained`` is
     pinned by ``tests/gpu/parallelism/ep/test_lazy_load_converted.py --family step3p7``.
  2. Run a short DistributedSFTTrainer run over text-only conversations (forward + backward +
     optimizer under FSDP2 + EP, dense+sparse MLP span, full/sliding attention interleave,
     per-layer clamps) with the script's own callback wiring, whose ``moe_balancing: auto``
     resolves to ``bias_update`` here and adopts the native ``e_score_correction_bias`` slot (fp32).
  3. Save via the gathered EP save, which for this family (``_EXPORTS_HUB_NAMESPACE``) runs
     transformers' save-side conversion revert per streamed chunk — the artifact must land in the
     hub layout the serving engines read: the on-disk key set equals the plain ``save_pretrained``
     key set of step 0, the expert halves are bit-exact against the live gathered fused tensor, the
     ``moe.router_bias`` carries a distinctive value written before the save at trained fp32 (a
     dropped key would reload as a zero buffer — so zeros prove nothing), and no module-tree
     spelling survives. Then reload the checkpoint as a PLAIN HF model — its loss must match the EP
     model's post-training loss (a transposed expert axis, a swapped gate/up half, or a dropped
     shared expert shifts it by >>1).

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_step3p7.py
"""

import os
import shutil

import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer
from transformers.models.step3p7.configuration_step3p7 import Step3p7Config
from transformers.models.step3p7.modeling_step3p7 import Step3p7ForConditionalGeneration
from trl import SFTConfig

from src.args.common_script_args import CommonScriptArguments
from src.callbacks.wiring import build_perf_callbacks
from src.distributed.expert_parallel.layers.step3p7 import EPStep3p7MoELayer
from src.distributed.expert_parallel.lazy_loader import lazy_loader_supports_checkpoint
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.models.moe_balancing import NATIVE_BALANCING_BIAS_ADOPTED_ATTR
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import ensure_model_downloaded
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B, TINY_STEP3P7_CONFIG, TINY_STEP3P7_VISION_CONFIG
from tests.common.utils import cleanup_memory, log, safetensors_state_dict

SEED = 42
NUM_TRAIN_STEPS = 3
MAX_SEQ_LENGTH = 256
LOSS_TOL = 5e-2  # EP-vs-plain reload forward noise (bf16 + grouped-GEMM vs loop); dropped experts shift >>1

_SPARSE_LAYERS = [i for i, kind in enumerate(TINY_STEP3P7_CONFIG["mlp_layer_types"]) if kind == "sparse"]
_HUB_MOE_KEYS = {
    f"model.layers.{i}.moe.{name}"
    for i in _SPARSE_LAYERS
    for name in ("gate.weight", "router_bias", "gate_proj.weight", "up_proj.weight", "down_proj.weight")
} | {f"model.layers.{i}.share_expert.{proj}_proj.weight" for i in _SPARSE_LAYERS for proj in ("gate", "up", "down")}
_MODULE_TREE_SPELLINGS = ("language_model", ".mlp.experts.", "shared_experts", "multi_modal_projector")


def _materialize_checkpoint(base_dir: str, tokenizer) -> None:
    """Rank 0: build + save the tiny composite model with the real tokenizer's vocab.

    ``image_token_id`` sits inside that vocab (the family default 151679 need not)."""
    torch.manual_seed(SEED)
    config = Step3p7Config(
        text_config={
            **TINY_STEP3P7_CONFIG,
            "vocab_size": len(tokenizer),
            "pad_token_id": tokenizer.pad_token_id,
        },
        vision_config=dict(TINY_STEP3P7_VISION_CONFIG),
        image_token_id=2000,
    )
    config._attn_implementation = "eager"
    model = Step3p7ForConditionalGeneration(config).to(torch.bfloat16)
    model.save_pretrained(base_dir)
    tokenizer.save_pretrained(base_dir)


def _fixed_batch(device, vocab_size: int):
    torch.manual_seed(SEED + 7)
    ids = torch.randint(0, vocab_size, (2, 64), device=device)
    return ids, ids.clone()


def _forward_loss(model, ids, labels) -> float:
    model.eval()
    with torch.no_grad():
        return model(input_ids=ids, labels=labels).loss.item()


def run(ctx):
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    device = ctx.device
    torch.cuda.set_device(device)

    ensure_model_downloaded(QWEN3_0_6B, ctx.rank)  # tokenizer only
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # setup_cache_dirs is per-rank; these must be rank-shared, keyed by port to avoid collisions.
    tag = os.environ.get("MASTER_PORT", "0")
    temp_root = os.path.dirname(ctx.output_dir.rstrip("/"))
    base_dir = os.path.join(temp_root, f"step3p7_tiny_base_{tag}")
    save_dir = os.path.join(temp_root, f"step3p7_tiny_trained_{tag}")
    if ctx.rank == 0:
        for d in (base_dir, save_dir):
            shutil.rmtree(d, ignore_errors=True)
        ctx.on_teardown(lambda: [shutil.rmtree(d, ignore_errors=True) for d in (base_dir, save_dir)])
        _materialize_checkpoint(base_dir, tokenizer)
    barrier()

    # The hub layout is lazy-loadable (declared conversion keys), so every rank takes the lazy route.
    checks["lazy_loading_admitted"] = lazy_loader_supports_checkpoint(base_dir) is True

    pc = ParallelismConfig(ep_size=ctx.world_size)
    model, _ = load_distributed_model(
        model_name_or_path=base_dir,
        parallelism_config=pc,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        use_liger_kernel=False,  # no step3p7 Liger applier
    )

    ep_layers = [m for m in model.modules() if isinstance(m, EPStep3p7MoELayer)]
    checks["ep_layers_patched"] = len(ep_layers) == len(_SPARSE_LAYERS)
    checks["native_bias_adoptable"] = all(ep.can_adopt_native_balancing() for ep in ep_layers)
    vocab_size = model.config.text_config.vocab_size

    train_dataset = create_sft_dataset(16, tokenizer, seed=SEED)
    sft_config = SFTConfig(
        output_dir=ctx.output_dir,
        max_steps=NUM_TRAIN_STEPS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        learning_rate=1e-5,
        warmup_steps=1,
        max_length=MAX_SEQ_LENGTH,
        bf16=True,
        gradient_checkpointing=False,
        use_liger_kernel=False,
        logging_steps=1,
        eval_strategy="no",
        save_strategy="no",
        report_to=[],
        dataloader_num_workers=0,
        dataloader_drop_last=True,
        remove_unused_columns=False,
        fsdp="",
        ddp_find_unused_parameters=True,
    )
    # The script's own callback wiring (``build_training_callbacks`` → ``build_perf_callbacks`` at
    # the script-args default ``moe_balancing: auto``): a trainer built bare never enables balancing,
    # and this family's export contract is the adopted native slot.
    callbacks = build_perf_callbacks(CommonScriptArguments(), sft_config, model, pc)
    trainer = DistributedSFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        parallelism_config=pc,
        callbacks=callbacks,
    )
    result = trainer.train()
    losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
    log(f"train losses: {losses}")
    metrics["final_train_loss"] = result.training_loss
    checks["trained_all_steps"] = result.global_step == NUM_TRAIN_STEPS
    checks["train_losses_finite"] = all(torch.isfinite(torch.tensor(losses)).tolist()) and len(losses) > 0

    # ``auto`` resolves to ``bias_update`` on this family (no aux machinery, native exported slot),
    # so the wiring above must have adopted the slot on every EP layer and upcast it to fp32 — the
    # trained-dtype keep-set the fp32-on-disk check below rides on.
    checks["bias_slot_adopted_by_trainer"] = all(
        getattr(ep, NATIVE_BALANCING_BIAS_ADOPTED_ATTR, False) for ep in ep_layers
    )
    checks["bias_slot_fp32_live"] = all(ep.gate.e_score_correction_bias.dtype == torch.float32 for ep in ep_layers)
    log(f"router bias dtypes: {[str(ep.gate.e_score_correction_bias.dtype) for ep in ep_layers]}")

    # A distinctive, bf16-exact router bias (replicated state — identical on every rank) BEFORE the
    # fixed-batch loss and the save, so the round-trip below proves the values landed and steered
    # routing, rather than two zero buffers agreeing.
    with torch.no_grad():
        for offset, ep in enumerate(ep_layers):
            ep.gate.e_score_correction_bias.copy_(torch.arange(ep.num_experts, device=device) * 0.125 + offset)

    ids, labels = _fixed_batch(device, vocab_size)
    ep_loss = _forward_loss(model, ids, labels)
    metrics["ep_loss_post_train"] = ep_loss
    checks["ep_loss_finite"] = bool(torch.isfinite(torch.tensor(ep_loss)))

    barrier()
    trainer.save_model(save_dir)
    barrier()

    # Collective on every rank; the F.linear-convention fused tensors the save splits into the hub's two.
    gathered = [ep.gather_expert_state_dict(device="cpu") for ep in ep_layers]
    written = safetensors_state_dict(save_dir)
    checks["hub_keyset_matches_plain_save"] = set(written) == set(safetensors_state_dict(base_dir))
    checks["hub_moe_keys_present"] = set(written) >= _HUB_MOE_KEYS
    checks["no_module_tree_spelling"] = not any(s in key for key in written for s in _MODULE_TREE_SPELLINGS)
    halves_exact, bias_fp32, bias_values = [], [], []
    for i, ep, layer_state in zip(_SPARSE_LAYERS, ep_layers, gathered, strict=True):
        fused = layer_state["experts.gate_up_proj"]  # [E, 2M, H], halves [gate; up]
        half = fused.shape[1] // 2
        halves_exact.append(
            torch.equal(written[f"model.layers.{i}.moe.gate_proj.weight"], fused[:, :half])
            and torch.equal(written[f"model.layers.{i}.moe.up_proj.weight"], fused[:, half:])
            and torch.equal(written[f"model.layers.{i}.moe.down_proj.weight"], layer_state["experts.down_proj"])
        )
        bias = written[f"model.layers.{i}.moe.router_bias"]
        bias_fp32.append(bias.dtype == torch.float32)
        bias_values.append(torch.equal(bias.float(), ep.gate.e_score_correction_bias.float().cpu()))
    checks["expert_halves_bit_exact"] = all(halves_exact)
    checks["router_bias_fp32_on_disk"] = all(bias_fp32)
    checks["router_bias_values_on_disk"] = all(bias_values)
    del written, gathered

    reloaded = (
        AutoModelForImageTextToText.from_pretrained(save_dir, dtype=torch.bfloat16, attn_implementation="eager")
        .to(device=device)
        .eval()
    )
    reloaded_bias = reloaded.model.language_model.layers[_SPARSE_LAYERS[0]].mlp.gate.e_score_correction_bias
    checks["e_score_bias_roundtrip"] = torch.equal(
        reloaded_bias.float().cpu(), ep_layers[0].gate.e_score_correction_bias.float().cpu()
    )
    rl_loss = _forward_loss(reloaded, ids, labels)
    metrics["reload_loss"] = rl_loss
    delta = abs(rl_loss - ep_loss)
    metrics["reload_loss_delta"] = delta
    log(f"EP loss {ep_loss:.6f} vs reloaded plain-HF loss {rl_loss:.6f} (|Δ|={delta:.2e})")
    checks["reload_loss_matches"] = delta < LOSS_TOL

    del reloaded
    cleanup_memory()
    ctx.on_teardown(lambda: trainer.cleanup_ep() if hasattr(trainer, "cleanup_ep") else None)
    return {"checks": checks, "metrics": metrics}


main = gpu_test_main(min_world_size=2, exact_world_size=2, prefix="sft_step3p7")(run)

if __name__ == "__main__":
    main()
