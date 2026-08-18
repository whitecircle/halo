#!/usr/bin/env python
"""
Test suite for EmbeddingTrainer.

Tests multiple loss types (MNRL, CoSENT) as well as import and capability
checks. Uses sentence-transformers/paraphrase-MiniLM-L3-v2 for speed;
production workloads use Qwen/Qwen3-Embedding-0.6B.

Every leg runs collectives (``dist.barrier``, ``gather_saveable_tensors``, ``trainer.train``), so a
failure must end the process rather than fall through to the next leg: a rank that swallowed its own
exception would enter the NEXT leg's collective while its peer is still in the previous one and hang
until the NCCL timeout. :func:`gpu_test_main` owns that — a raise is a reported failure and an exit.

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/other/test_embedding.py
"""

import os

import torch
import torch.distributed as dist
from datasets import Dataset
from peft import LoraConfig, TaskType, inject_adapter_in_model
from sentence_transformers import SentenceTransformer

from src.checkpoint.format import save_dtype_caster
from src.configs.embedding_config import EmbeddingConfig
from src.distributed.checkpoint.write import gather_saveable_tensors
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.embedding.trainer import EmbeddingTrainer
from tests.common.harness import gpu_test_main
from tests.common.models import PARAPHRASE_MINILM
from tests.common.peft_helpers import (
    assert_adapters_moved,
    assert_only_adapters_trainable,
    snapshot_adapters,
    unwrap,
)
from tests.common.utils import cleanup_memory, log

MODEL_NAME = PARAPHRASE_MINILM
NUM_TRAIN_STEPS = 10
BATCH_SIZE = 4
# deliberately not ST's own default (20.0) — only a differing value can tell a wired scale from an unwired one
LOSS_SCALE = 30.0
# finite losses alone pass even when the optimizer never reaches the encoder; require real movement
MIN_LOSS_DROP = 0.25

LORA_TARGET_MODULES = ["query", "value"]  # MiniLM is BERT-arch (query/key/value projections)


def create_pairs_dataset(num_samples: int = 64) -> Dataset:
    """Create synthetic positive-pairs dataset for MNRL loss.

    Each sample contains an ``anchor`` question and a ``positive`` answer. Rows differ ONLY by the
    country index and share every other token, so the in-batch negatives are genuinely confusable and
    the MNRL loss stays O(1) instead of collapsing to ~0 — which is the regime where the objective pin
    and its scale control can be told apart at all.
    """
    anchors = [f"What is the capital of country {i}?" for i in range(num_samples)]
    positives = [f"It is a large city located in country {i}." for i in range(num_samples)]
    return Dataset.from_dict({"anchor": anchors, "positive": positives})


def create_scored_pairs_dataset(num_samples: int = 64) -> Dataset:
    """Create synthetic scored-pairs dataset for CoSENT loss.

    Even-indexed samples are the same fact restated (score=1.0); odd-indexed samples are the same
    sentence about a DIFFERENT index (score=0.0). Both levels share their wording, so the cosine
    similarities are not saturated and the CoSENT ranking hinge stays O(1) — the regime where the
    objective pin can be separated from its controls.
    """
    sentence1 = [f"Country {i} has a large capital city." for i in range(num_samples)]
    sentence2 = [f"The capital city of country {i if i % 2 == 0 else i + 1} is large." for i in range(num_samples)]
    scores = [1.0 if i % 2 == 0 else 0.0 for i in range(num_samples)]
    return Dataset.from_dict({"sentence1": sentence1, "sentence2": sentence2, "score": scores})


def make_config(output_dir: str, **overrides) -> EmbeddingConfig:
    """The common EmbeddingConfig; ``fsdp=""`` because the mixin owns FSDP wrapping."""
    kwargs = {
        "output_dir": output_dir,
        "max_steps": NUM_TRAIN_STEPS,
        "per_device_train_batch_size": BATCH_SIZE,
        "learning_rate": 2e-5,
        "bf16": True,
        "gradient_checkpointing": False,
        "use_liger_kernel": False,
        "logging_steps": 1,
        "save_strategy": "no",
        "eval_strategy": "no",
        "report_to": "none",
        "loss_type": "mnrl",
        "max_length": 128,
        "dataloader_drop_last": True,
        "fsdp": "",
        **overrides,
    }
    return EmbeddingConfig(**kwargs)


def lora_model() -> SentenceTransformer:
    """A SentenceTransformer with LoRA injected exactly the way the embedding script does it.

    ``inject_adapter_in_model`` (not ``ST.add_adapter``, which needs peft>=0.18.2) adds the adapters
    in place without globally freezing, so the freeze-by-name below is part of the wiring under test.
    """
    model = SentenceTransformer(MODEL_NAME)
    inject_adapter_in_model(
        LoraConfig(r=8, lora_alpha=16, target_modules=LORA_TARGET_MODULES, task_type=TaskType.FEATURE_EXTRACTION),
        model,
    )
    for name, param in model.named_parameters():
        param.requires_grad = "lora_" in name
    return model


def check_basic_import() -> bool:
    """Verify EmbeddingTrainer imports correctly and the re-export is the same class."""
    log("Check: Basic Import")

    from src.trainers.embedding.trainer import EmbeddingTrainer as ET

    assert ET is EmbeddingTrainer, "EmbeddingTrainer re-exported from src.trainers should be identical"
    return True


def check_capability_flags() -> bool:
    """Verify parallelism capability flags on EmbeddingTrainer."""
    log("Check: capability flags (CP unsupported, EP/TP supported)")

    assert hasattr(EmbeddingTrainer, "_supports_cp"), "Missing _supports_cp"
    assert EmbeddingTrainer._supports_cp is False, "_supports_cp should be False"
    assert EmbeddingTrainer._supports_ep is True, "_supports_ep should be True"
    assert EmbeddingTrainer._supports_tp is True, "_supports_tp should be True"
    return True


def check_loss_leg(ctx, loss_type: str, train_dataset: Dataset, eval_dataset: Dataset) -> bool:
    """Train one loss type: halo's configured ``loss_scale`` must reach the loss, and the loss must
    actually come down.

    The loss arithmetic itself is sentence-transformers'; what halo owns is ``create_loss`` — the
    registry lookup and the ``scale`` forwarding — so that is pinned directly against a value chosen
    to differ from the library default. A single ``isfinite`` assertion would pass on an unwired
    scale and on a run whose loss never moved.
    """
    log(f"Check: training with {loss_type}")

    trainer = EmbeddingTrainer(
        model=SentenceTransformer(MODEL_NAME),
        args=make_config(
            os.path.join(ctx.output_dir, loss_type),
            loss_type=loss_type,
            loss_scale=LOSS_SCALE,
            gradient_checkpointing=True,
            use_liger_kernel=True,
        ),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        parallelism_config=ParallelismConfig(),
    )

    for attr in ("is_ep_mode", "is_cp_mode", "is_tp_mode"):
        assert hasattr(trainer, attr), f"Missing {attr}"

    # halo's create_loss is what forwards config.loss_scale; an unwired scale silently uses 20.0.
    assert trainer.loss.scale == LOSS_SCALE, (
        f"configured loss_scale {LOSS_SCALE} did not reach the loss object (got {trainer.loss.scale})"
    )

    ctx.barrier()
    result = trainer.train()

    losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
    log(f"  Steps: {result.global_step}  Loss: {losses[0]:.6f} -> {losses[-1]:.6f}")
    assert len(losses) == NUM_TRAIN_STEPS, f"expected {NUM_TRAIN_STEPS} logged steps, got {len(losses)}"
    assert torch.isfinite(torch.tensor(losses)).all(), f"non-finite loss in {losses}"
    assert losses[0] > 0, f"{loss_type} loss starts at {losses[0]}; nothing to learn, the drop below is vacuous"
    assert losses[-1] < (1 - MIN_LOSS_DROP) * losses[0], (
        f"{loss_type} loss did not drop by {MIN_LOSS_DROP:.0%}: {losses[0]} -> {losses[-1]}"
    )
    cleanup_memory()
    return True


def check_training_lora(ctx) -> bool:
    """Train a SentenceTransformer with LoRA adapters (mirrors the embedding script's inject path).

    Validates the embedding PEFT wiring: inject + freeze-by-name leaves only adapters trainable, and
    they move under MNRL training. (Embedding LoRA is plain LoRA — no 4-bit QLoRA on the ST loader.)
    """
    log("Check: training with LoRA adapters")

    model = lora_model()
    ok, detail = assert_only_adapters_trainable(model)
    log(f"  [trainable] {detail}")
    if not ok:
        return False
    before = snapshot_adapters(model, expert_lora=False)

    trainer = EmbeddingTrainer(
        model=model,
        args=make_config(os.path.join(ctx.output_dir, "lora"), learning_rate=1e-3),
        train_dataset=create_pairs_dataset(64),
        parallelism_config=ParallelismConfig(),
    )
    ctx.barrier()
    result = trainer.train()
    assert torch.isfinite(torch.tensor(result.training_loss)), f"non-finite loss {result.training_loss}"

    after = snapshot_adapters(unwrap(trainer.model), expert_lora=False)
    ok, detail = assert_adapters_moved(before, after)
    log(f"  [moved] {detail}")
    cleanup_memory()
    return ok


def check_save_roundtrip_fsdp2(ctx, shared_dir: str) -> bool:
    """FSDP2 (torchrun DP) save must gather DTensors into a full, reloadable ST checkpoint.

    The trainer FSDP2-wraps the backbone under torchrun DP, so its params are DTensors. A naive
    delegation to SentenceTransformerTrainer.save_model would write them un-gathered (half-width,
    unloadable). Train a few steps, save, reload with ``SentenceTransformer(path)``, and assert the
    reloaded backbone matches the gathered trained weights at FULL shape (the shard discriminator)
    and encodes finite embeddings.
    """
    log("Check: FSDP2 save roundtrip")

    output_dir = os.path.join(shared_dir, "fsdp2_save")
    trainer = EmbeddingTrainer(
        model=SentenceTransformer(MODEL_NAME),
        args=make_config(output_dir, max_steps=3, learning_rate=1e-3),
        train_dataset=create_pairs_dataset(64),
        parallelism_config=ParallelismConfig(),
    )
    # vacuous unless the mixin-managed FSDP2 path is actually hit
    assert trainer._fsdp_wrapped, "expected mixin-managed FSDP2 under torchrun DP"
    ctx.barrier()
    trainer.train()

    # collective — must run on ALL ranks
    ref_sd = gather_saveable_tensors(trainer._get_unwrapped_model(), retain=True)
    cast_expected = save_dtype_caster(trainer._get_unwrapped_model())

    trainer.save_model(output_dir)
    ctx.barrier()

    ok = True
    if ctx.rank == 0:
        reloaded = SentenceTransformer(output_dir)
        rl_sd = dict(list(reloaded.children())[0].auto_model.state_dict())
        compared = 0
        for name, ref in ref_sd.items():
            if name not in rl_sd or not ref.is_floating_point():
                continue
            if tuple(rl_sd[name].shape) != tuple(ref.shape):
                log(f"  FAIL: {name} shape {tuple(rl_sd[name].shape)} != full {tuple(ref.shape)} (sharded save)")
                ok = False
                break
            # The export carries the toolkit save dtype (bf16 except norms / fp32-pinned tensors), so
            # the reload must equal the caster's view of the gathered weight exactly — a looser
            # tolerance would also pass a half-gathered or wrong-tensor write.
            expected = cast_expected(name, ref).float().cpu()
            if not torch.equal(rl_sd[name].float().cpu(), expected):
                log(
                    f"  FAIL: {name} values differ after reload (max|diff|={(rl_sd[name].float().cpu() - expected).abs().max():.3e})"
                )
                ok = False
                break
            compared += 1
        if ok:
            emb = reloaded.encode(["hello world", "embedding roundtrip"])
            ok = compared > 0 and bool(torch.isfinite(torch.tensor(emb)).all())
        log(f"  compared {compared} full-shape params; reloaded encode finite={ok}")

    cleanup_memory()
    return broadcast_verdict(ctx, ok)


def check_lora_save_roundtrip(ctx, shared_dir: str) -> bool:
    """A LoRA-injected embedding checkpoint must save MERGED, plain, and reloadable.

    The script injects LoRA in place (the model stays a SentenceTransformer, not a PeftModel), so a
    naive save writes ``base_layer``/``lora_*`` keys that reload as RANDOM base weights. Train LoRA,
    save, reload with ``SentenceTransformer(path)`` and assert: (1) a target projection reloads as
    ``base + scaling*(B @ A)`` — the trained merge, neither the original base nor random; (2) no
    ``base_layer``/``lora_*`` keys survive. Fails on a plain save (target weight missing →
    reinitialized).
    """
    log("Check: LoRA save roundtrip (merge-on-save)")

    output_dir = os.path.join(shared_dir, "lora_save")
    trainer = EmbeddingTrainer(
        model=lora_model(),
        args=make_config(output_dir, learning_rate=1e-2),
        train_dataset=create_pairs_dataset(64),
        parallelism_config=ParallelismConfig(),
    )
    ctx.barrier()
    trainer.train()  # lora_B starts at 0; training makes the delta (and the merge) non-trivial

    # collective gather — must run on ALL ranks
    backbone = trainer._get_unwrapped_model()
    scaling = trainer._lora_scaling(backbone)
    full = gather_saveable_tensors(backbone, retain=True)
    target = next(
        k[: -len(".base_layer.weight")]
        for k in full
        if k.endswith(".base_layer.weight") and f"{k[: -len('.base_layer.weight')]}.lora_A.default.weight" in full
    )
    base_w = full[f"{target}.base_layer.weight"].float().cpu()
    a = full[f"{target}.lora_A.default.weight"].float().cpu()
    b = full[f"{target}.lora_B.default.weight"].float().cpu()
    expected = base_w + scaling * (b @ a)

    trainer.save_model(output_dir)
    ctx.barrier()

    ok = True
    if ctx.rank == 0:
        reloaded = SentenceTransformer(output_dir)
        rl_sd = dict(list(reloaded.children())[0].auto_model.state_dict())
        leaked = [k for k in rl_sd if ".lora_" in k or ".base_layer." in k]
        if leaked:
            log(f"  FAIL: adapter keys leaked into checkpoint: {leaked[:3]}")
            ok = False
        plain_key = f"{target}.weight"
        if ok and plain_key not in rl_sd:
            log(f"  FAIL: merged key {plain_key} missing from reload")
            ok = False
        if ok:
            got = rl_sd[plain_key].float().cpu()
            if torch.allclose(got, base_w, atol=1e-4):
                log("  FAIL: merged weight equals base — trained LoRA delta lost")
                ok = False
            elif not torch.allclose(got, expected, atol=2e-2, rtol=1e-2):
                log(f"  FAIL: {plain_key} != base + {scaling:g}*(B@A) (merge incorrect)")
                ok = False
            else:
                log(f"  reload matches base + {scaling:g}*(B@A); trained delta captured, no adapter keys")

    cleanup_memory()
    return broadcast_verdict(ctx, ok)


def broadcast_verdict(ctx, ok: bool) -> bool:
    """Agree world-wide on a rank-0-only verdict — collective, every rank must call it."""
    verdict = torch.tensor([1 if ok else 0], device=ctx.device, dtype=torch.int32)
    dist.broadcast(verdict, src=0)
    return verdict.item() == 1


@gpu_test_main(exact_world_size=2, prefix="test_embedding")
def run(ctx):
    log(f"Model: {MODEL_NAME}  Steps: {NUM_TRAIN_STEPS}  Batch size: {BATCH_SIZE}")

    # rank 0's dir, so a rank-0 reload sees whichever rank the save strategy elects
    dirs = [ctx.output_dir]
    dist.broadcast_object_list(dirs, src=0)
    shared_dir = dirs[0]

    return {
        "checks": {
            "basic_import": check_basic_import(),
            "capability_flags": check_capability_flags(),
            "training_pairs_mnrl": check_loss_leg(ctx, "mnrl", create_pairs_dataset(64), create_pairs_dataset(16)),
            "training_scored_pairs_cosent": check_loss_leg(
                ctx, "cosent", create_scored_pairs_dataset(64), create_scored_pairs_dataset(16)
            ),
            "training_lora": check_training_lora(ctx),
            "save_roundtrip_fsdp2": check_save_roundtrip_fsdp2(ctx, shared_dir),
            "lora_save_roundtrip": check_lora_save_roundtrip(ctx, shared_dir),
        }
    }


if __name__ == "__main__":
    run()
