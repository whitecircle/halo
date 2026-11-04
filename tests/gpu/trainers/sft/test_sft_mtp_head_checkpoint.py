#!/usr/bin/env python
"""A released MTP head's save, exact resume and export under real DeepEP and the trainer's wrap, the rows no CPU rank
reaches. A run stopped at step 3 and resumed writes the bytes of one that ran to step 5, the head's included. Its
checkpoint reloads in the stock class and carries the head under the release's keys. Per-rank shards merge to the
gathered save, a resume that widens what trains raises the optimizer gate on every rank, and the acceptance gate reads
every rank's held-out dispatches. EP groups that replicate each other train different rows and hold byte-equal head
experts of a head that moved.

    torchrun --nproc_per_node=2 tests/gpu/trainers/sft/test_sft_mtp_head_checkpoint.py --family qwen3_5_moe_text --mode ep

Two machines, by hand, EP groups as replicas and then plain FSDP2 across both, with each node's own disk:
    DIST_SHARED_FILESYSTEM=0 torchrun --nnodes=2 --node_rank=$NODE_RANK --nproc_per_node=8 \\
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \\
        tests/gpu/trainers/sft/test_sft_mtp_head_checkpoint.py --family qwen3_5_moe_text --mode multi_node --ep_size 8
    DIST_SHARED_FILESYSTEM=0 torchrun --nnodes=2 --node_rank=$NODE_RANK --nproc_per_node=8 \\
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \\
        tests/gpu/trainers/sft/test_sft_mtp_head_checkpoint.py --family qwen3_5_moe_text --mode multi_node
"""

import argparse
import hashlib
import os
import shutil
from pathlib import Path

import torch
import torch.distributed as dist
from trl import SFTConfig

from halo.callbacks.mtp import MtpAcceptanceGate
from halo.checkpoint.tools.ep_shard_merge import merge_ep_shards
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.finalization import finalize_loaded_model
from halo.model_loading.liger.dispatch import finalize_liger_after_direct_load
from halo.model_loading.model_preparation import auto_load_model
from halo.model_loading.parallel.loader import load_distributed_model
from halo.models.mtp.head import find_mtp_head, mtp_head_path, mtp_trunk_copy_keys
from halo.models.mtp.spec import MtpHeadRequest, MtpObjective, require_mtp_head_spec
from halo.runtime.collectives import barrier
from halo.runtime.consensus import rank_consensus
from halo.runtime.dtensor import to_local
from halo.runtime.filesystem import is_fs_save_rank
from halo.runtime.ranks import is_global_main_process
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.distributed import shared_scratch_dir
from tests.common.harness import gpu_test_main
from tests.common.memory import cleanup_memory
from tests.common.models import QWEN3_0_6B
from tests.common.mtp_heads import (
    MTP_GPU_MODES,
    MTP_TINY_FAMILIES,
    collate_tokenized_rows,
    saved_tensors,
    tokenized_rows,
    write_checkpoint_with_tail,
)
from tests.common.rank_log import log
from tests.common.tiny_models import build_tiny_family_checkpoint
from tests.common.tokenizers import load_cached_tokenizer

_MIDDLE, _LAST = 3, 5
_WEIGHT = 0.7
_LENGTH = 64
_PROMPT = 8
_LEARNING_RATE = 1e-3
_SEED = 42
# Any world: EP groups of ``--ep_size`` that replicate each other, or plain FSDP2 at 1.
_MULTI_NODE_MODE = "multi_node"
# Where a rank holds a copy of an expert another rank trains from its own row, a sweep's divide all that keeps them one.
_REPLICA_MODES = (_MULTI_NODE_MODE, "ep1_ignored")
# fp32 masters on a frozen trunk nothing casts, which the entry script refuses.
_FP32_MASTERS_MODE = "ep_fp32_masters"
_SCRATCH = "sft_mtp_head_checkpoint"
_FAILING_GAIN = 2.0
_PASSING_GAIN = -1.0
_ARM_SCOPES = {"joint": "soft", "detached": "soft", "head_only": "hard"}


def _parallelism(cli) -> ParallelismConfig:
    if cli.mode == _MULTI_NODE_MODE:
        domains = {"gpus_per_node": cli.simulated_domain, "ep_scope": "node"} if cli.simulated_domain else {}
        return ParallelismConfig(ep_size=cli.ep_size, **domains)
    return ParallelismConfig(**MTP_GPU_MODES[cli.mode][0])


def _trainer(
    ctx,
    cli,
    output: Path,
    *,
    source: str,
    tail: str | None,
    scope: str,
    resuming: bool = False,
    fused: bool = False,
    **fields,
):
    """COLLECTIVE: a trainer over a fresh load of ``source``, its head read from ``tail`` or, resuming, ``source``.
    ``fused`` takes Liger's fused kernel, whose ``skip_logits`` forward an evaluation runs too."""
    parallelism = _parallelism(cli)
    liger = {"use_liger_kernel": fused, "liger_kernel_config": {"fused_linear_cross_entropy": True} if fused else None}
    objective = MtpObjective(_WEIGHT, _ARM_SCOPES[scope], scope)
    model, _ = load_distributed_model(
        source,
        parallelism,
        dtype=torch.bfloat16,
        attn_implementation=cli.attn,
        trust_remote_code=False,
        preserve_checkpoint_precision=resuming,
        mtp_head=MtpHeadRequest(objective, tail),
        **liger,
    )
    gate = fields.pop("gate", None)
    trainer_fields = {"save_sharded_ep": fields.pop("save_sharded_ep", False)}
    config = SFTConfig(
        output_dir=str(output),
        max_steps=_LAST,
        save_steps=_MIDDLE,
        save_strategy="steps",
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        learning_rate=_LEARNING_RATE,
        lr_scheduler_type="constant",
        max_grad_norm=0,
        bf16=True,
        gradient_checkpointing=cli.gc and scope != "head_only",
        logging_steps=1,
        report_to="none",
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        seed=_SEED,
        full_determinism=True,
        **liger,
        **fields,
    )
    finalize_liger_after_direct_load(config, fused, model)
    vocab_size = model.config.get_text_config().vocab_size
    rows = {"seed": _SEED, "length": _LENGTH, "prompt": _PROMPT}
    trainer = DistributedSFTTrainer(
        model=model,
        args=config,
        train_dataset=tokenized_rows(vocab_size, 8 * ctx.world_size, **rows),
        eval_dataset=tokenized_rows(vocab_size, 2 * ctx.world_size, **{**rows, "seed": _SEED + 1}),
        processing_class=load_cached_tokenizer(QWEN3_0_6B),
        data_collator=collate_tokenized_rows,
        parallelism_config=parallelism,
        callbacks=[gate] if gate is not None else None,
        **trainer_fields,
    )
    return trainer, model


def _finished(trainer, model) -> None:
    trainer.cleanup_ep()
    del trainer, model
    cleanup_memory()
    barrier()


def _stored(checkpoint: Path) -> dict[str, torch.Tensor]:
    """Empty off a save rank, which holds no copy to read on per-node storage."""
    return saved_tensors(checkpoint) if is_fs_save_rank() else {}


def _agrees(ok: bool) -> bool:
    return rank_consensus(ok).all_ok


def _head_prefix(model) -> str:
    return f"{mtp_head_path(model, require_mtp_head_spec(model.config))}."


def _digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().float().cpu().numpy().tobytes()).hexdigest()


def _replica_experts_agree(model, ep_config) -> bool:
    """COLLECTIVE. Ranks holding one slice in different EP groups hold one set of bytes, of a head that moved."""
    digests = {
        f"{path}.{name}": _digest(to_local(param))
        for path, layer in named_ep_layers(find_mtp_head(model))
        for name, param in layer.expert_named_params()
    }
    gathered: list = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, (ep_config.ep_rank, digests))
    by_slice: dict[tuple[int, str], set[str]] = {}
    for ep_rank, rank_digests in gathered:
        for name, digest in rank_digests.items():
            by_slice.setdefault((ep_rank, name), set()).add(digest)
    return bool(by_slice) and all(len(digests) == 1 for digests in by_slice.values())


def _every_rank_trained_its_own_row(row: torch.Tensor) -> bool:
    """COLLECTIVE. Replicas fed one row would hold one set of bytes however the sweep divided their gradients."""
    gathered: list = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, tuple(row.flatten().tolist()))
    return len(set(gathered)) == len(gathered)


def _replay(ctx, cli, paths: dict[str, str], scope: str, checks: dict[str, bool]) -> dict[str, torch.Tensor]:
    """COLLECTIVE: the uninterrupted run and the resumed one; the first's last checkpoint as stored."""
    tag = f"{scope}_{_ARM_SCOPES[scope]}"
    output = Path(paths["root"]) / tag
    trainer, model = _trainer(ctx, cli, output, source=paths["base"], tail=paths["tail"], scope=scope)
    prefix = _head_prefix(model)
    copies = mtp_trunk_copy_keys(model)
    seen_rows: list[torch.Tensor] = []
    capture = model.register_forward_pre_hook(
        lambda _module, _args, kwargs: seen_rows.append(kwargs["input_ids"].detach().cpu()) if not seen_rows else None,
        with_kwargs=True,
    )
    trainer.train()
    capture.remove()
    if cli.mode in _REPLICA_MODES:
        checks[f"{tag}_every_rank_trained_its_own_row"] = _every_rank_trained_its_own_row(seen_rows[0])
        config = trainer.parallelism_config
        if named_ep_layers(find_mtp_head(model)) and not config.experts_fsdp_managed:
            checks[f"{tag}_replica_experts_byte_equal"] = _replica_experts_agree(model, config.ep_config())
    _finished(trainer, model)

    straight = _stored(output / f"checkpoint-{_LAST}")
    middle = _stored(output / f"checkpoint-{_MIDDLE}")
    if is_fs_save_rank():
        shutil.rmtree(output / f"checkpoint-{_LAST}")
    barrier()
    checkpoint = output / f"checkpoint-{_MIDDLE}"
    trainer, model = _trainer(ctx, cli, output, source=str(checkpoint), tail=None, scope=scope, resuming=True)
    trainer.train(resume_from_checkpoint=str(checkpoint))
    _finished(trainer, model)

    resumed = _stored(output / f"checkpoint-{_LAST}")
    head_keys = [key for key in straight if key.startswith(prefix)]
    differing = [key for key in straight if not torch.equal(straight[key], resumed[key])]
    if differing:
        log(f"  {tag}: {len(differing)} tensor(s) left the uninterrupted run's bytes: {differing[:4]}")
    checks[f"{tag}_resume_replays_every_tensor"] = _agrees(set(straight) == set(resumed) and not differing)
    checks[f"{tag}_the_head_moved_past_the_resume_point"] = _agrees(
        not is_fs_save_rank() or any(not torch.equal(straight[key], middle[key]) for key in head_keys)
    )
    checks[f"{tag}_trunk_copies_hold_the_trained_embeddings"] = _agrees(
        all(
            torch.equal(straight[copied], straight[source])
            for source, copied_keys in copies.items()
            for copied in copied_keys
        )
        if is_fs_save_rank()
        else True
    )
    return straight


def _stock_reload_holds_no_head(ctx, output: Path) -> bool:
    """COLLECTIVE through its barrier. On CPU on the main rank: the coverage gate raises on a key the head left missing."""
    held = True
    if is_global_main_process():
        model = auto_load_model(str(output), dtype=torch.float32, device_map="cpu")
        finalize_loaded_model(model)
        held = find_mtp_head(model) is None
        del model
    barrier()
    return _agrees(held)


def _merged_equals_gathered(ctx, cli, paths: dict[str, str], gathered: dict[str, torch.Tensor]) -> bool:
    """COLLECTIVE: a second run saving per-rank shards, merged on the main rank."""
    output = Path(paths["root"]) / "sharded"
    trainer, model = _trainer(
        ctx, cli, output, source=paths["base"], tail=paths["tail"], scope="joint", save_sharded_ep=True
    )
    trainer.train()
    _finished(trainer, model)
    equal = True
    if is_global_main_process():
        merged_dir = output / "merged"
        merge_ep_shards(str(output / f"checkpoint-{_LAST}"), str(merged_dir), verbose=False)
        merged = saved_tensors(merged_dir)
        equal = set(merged) == set(gathered) and all(torch.equal(merged[key], gathered[key]) for key in merged)
    barrier()
    return _agrees(equal)


def _a_wider_resume_raises(ctx, cli, paths: dict[str, str]) -> bool:
    """COLLECTIVE: a head_only fit's middle checkpoint resumed under joint raises on every rank."""
    fit = Path(paths["root"]) / "head_only_hard" / f"checkpoint-{_MIDDLE}"
    output = Path(paths["root"]) / "widened"
    trainer, model = _trainer(ctx, cli, output, source=str(fit), tail=None, scope="joint", resuming=True)
    raised = False
    try:
        trainer.train(resume_from_checkpoint=str(fit))
    except RuntimeError as error:
        raised = "have no saved optimizer state" in str(error)
    _finished(trainer, model)
    return _agrees(raised)


def _a_narrower_resume_trains_the_head_alone(ctx, cli, paths: dict[str, str]) -> bool:
    """COLLECTIVE: a joint run's middle checkpoint resumed under head_only passes the gate; only the head moves on."""
    start = Path(paths["root"]) / "joint_soft" / f"checkpoint-{_MIDDLE}"
    output = Path(paths["root"]) / "narrowed"
    trainer, model = _trainer(ctx, cli, output, source=str(start), tail=None, scope="head_only", resuming=True)
    prefix = _head_prefix(model)
    trainer.train(resume_from_checkpoint=str(start))
    _finished(trainer, model)
    if not is_fs_save_rank():
        return _agrees(True)
    before, after = saved_tensors(start), saved_tensors(output / f"checkpoint-{_LAST}")
    moved = [key for key in before if not torch.equal(before[key], after[key])]
    return _agrees(bool(moved) and all(key.startswith(prefix) for key in moved))


def _gate_reads_every_rank(ctx, cli, paths: dict[str, str], checks: dict[str, bool]) -> None:
    """COLLECTIVE: eval forwards dispatch the head's experts on every rank; the verdict comes off their world sums."""
    for gain, tag in ((_PASSING_GAIN, "passing"), (_FAILING_GAIN, "failing")):
        output = Path(paths["root"]) / f"gate_{tag}"
        gate = MtpAcceptanceGate(gain)
        trainer, model = _trainer(
            ctx,
            cli,
            output,
            source=paths["base"],
            tail=paths["tail"],
            scope="head_only",
            gate=gate,
            fused=cli.fused_gate,
            eval_strategy="steps",
            eval_steps=_MIDDLE,
            eval_on_start=True,
        )
        trainer.train()
        readings = [entry for entry in trainer.state.log_history if "eval_mtp_acceptance_tv" in entry]
        _finished(trainer, model)
        checks[f"gate_{tag}_verdict"] = _agrees(gate.stopped == (gain == _FAILING_GAIN))
        checks[f"gate_{tag}_scored_at_start_and_end"] = _agrees(
            len(readings) >= 2
            and readings[0]["step"] == 0
            and readings[-1]["step"] == _LAST
            and all(entry["eval_mtp_scored_positions"] > 0 for entry in readings)
        )
        if gain == _FAILING_GAIN:
            named = f"{output}/checkpoint-{_LAST} holds the rejected head"
            checks["gate_failing_names_the_checkpoint"] = _agrees(
                gate.stop_reason is not None and named in gate.stop_reason
            )


@gpu_test_main(min_world_size=1)
def run(ctx) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=sorted(MTP_TINY_FAMILIES), required=True)
    parser.add_argument("--mode", choices=(*sorted(MTP_GPU_MODES), _MULTI_NODE_MODE), required=True)
    parser.add_argument(
        "--ep_size", type=int, default=1, help=f"--mode {_MULTI_NODE_MODE}: the ranks of one EP group."
    )
    parser.add_argument(
        "--simulated_domain",
        type=int,
        default=0,
        help=f"--mode {_MULTI_NODE_MODE}: this many ranks count as one NVLink domain, on a single machine.",
    )
    parser.add_argument("--gc", action="store_true")
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--sharded", action="store_true", help="Also merge a per-rank save.")
    parser.add_argument("--flip", action="store_true", help="Also resume a head_only fit under joint.")
    parser.add_argument("--gate", action="store_true", help="Also run the acceptance gate both ways.")
    parser.add_argument("--fused_gate", action="store_true", help="Run the gate's fits on Liger's fused kernel.")
    cli = parser.parse_args()
    if cli.mode != _MULTI_NODE_MODE and ctx.world_size != MTP_GPU_MODES[cli.mode][1]:
        raise ValueError(
            f"--mode {cli.mode} runs on {MTP_GPU_MODES[cli.mode][1]} ranks, launched on {ctx.world_size}."
        )

    tokenizer = load_cached_tokenizer(QWEN3_0_6B)
    # Per node, as a multi-node launch's save rank on each machine builds its own copy.
    root = shared_scratch_dir(_SCRATCH)
    if ctx.local_rank == 0:
        ctx.on_teardown(lambda: shutil.rmtree(root, ignore_errors=True))
    paths = {"root": root, "base": os.path.join(root, "base"), "tail": os.path.join(root, "tail")}
    if is_fs_save_rank():
        build_tiny_family_checkpoint(MTP_TINY_FAMILIES[cli.family], paths["base"], tokenizer)
        write_checkpoint_with_tail(cli.family, Path(paths["tail"]), per_expert=True)
    barrier()

    checks: dict[str, bool] = {}
    joint = _replay(ctx, cli, paths, "joint", checks)
    checks["a_stock_reload_holds_no_head"] = _stock_reload_holds_no_head(
        ctx, Path(root) / "joint_soft" / f"checkpoint-{_LAST}"
    )
    _replay(ctx, cli, paths, "detached", checks)
    if cli.mode != _FP32_MASTERS_MODE:
        _replay(ctx, cli, paths, "head_only", checks)
    if cli.sharded:
        checks["per_rank_shards_merge_to_the_gathered_save"] = _merged_equals_gathered(ctx, cli, paths, joint)
    if cli.flip:
        checks["a_wider_resume_raises_on_every_rank"] = _a_wider_resume_raises(ctx, cli, paths)
        checks["a_narrower_resume_trains_the_head_alone"] = _a_narrower_resume_trains_the_head_alone(ctx, cli, paths)
    if cli.gate:
        _gate_reads_every_rank(ctx, cli, paths, checks)
    return {"checks": checks, "metrics": {}}


if __name__ == "__main__":
    raise SystemExit(run())
