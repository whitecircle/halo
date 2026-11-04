#!/usr/bin/env python
"""A released MTP head through the trainer's real wrap, with what no CPU row reaches: Liger's fused kernel on the routed
term and real DeepEP. Each rank trains its own row. After the first step's reduce, every head gradient but a distributed
expert bank's is compared with the mean of those rows' gradients on a plain model scored by the chunked path. A fp32 run
holds every tensor to 1e-3. A bf16 run, which is every EP row, holds the whole head: its distance from the fp32 reference
may be at most 1.1x a plain bf16 model's on the same rows, or 1%. The plain model reads 14-18% soft and 1-2% hard on a
tiny row's near-uniform logits, so soft's bar sees a 25% error and not a 5% one; the fp32 rows carry the logic check. The
worst tensor is logged only, since one small tensor's gradient flips with the router. Then the run must log both
cross-entropies, move every head parameter, and move the trunk or not as its scope says.

A launch sweeps the arms of ``--arms``: ``all`` is joint hard and soft, detached soft, and head_only hard and soft,
``soft`` is joint soft alone, ``trunk_trains`` leaves out head_only. Hard and soft must issue one dispatch per step on the head and one capacity generation per
forward, and leave the run's first loss equal. ``--starve`` sends every head token to rank 0's experts, so rank 1's
backward through the head has no local token, and rank 1's experts must take no gradient while rank 0's do. ``--packed``
trains rows of two documents through the packing collator, so the layer call the head reuses carries the varlen set.
``--attn auto`` takes the backend the family selects. ``--accumulation`` takes that many forwards per optimizer step, the
first step's gradient held to the mean over all of them.

    torchrun --nproc_per_node=2 tests/gpu/trainers/sft/test_sft_mtp_head.py --family qwen3_5_moe_text --mode ep --gc
"""

import argparse
import contextlib
import math
import os
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from transformers import TrainerCallback
from trl import SFTConfig

from halo.data.collators.factory import select_data_collator
from halo.distributed.expert_parallel.deepep import capacity as ep_capacity
from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase, named_ep_layers
from halo.distributed.fsdp import reshard_fsdp2_modules
from halo.distributed.parallelism_config import ParallelismConfig
from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from halo.model_loading import finalization
from halo.model_loading.liger import cross_entropy
from halo.model_loading.liger.dispatch import LIGER_APPLIED_CONFIG_ATTR, finalize_liger_after_direct_load
from halo.model_loading.model_preparation import auto_load_model
from halo.model_loading.parallel.loader import load_distributed_model
from halo.model_loading.parallel.model_source import ModelSource
from halo.model_loading.parallel.mtp_head import attach_mtp_head
from halo.models.mtp.head import find_mtp_head
from halo.models.mtp.spec import MtpHeadRequest, MtpObjective
from halo.models.segment_markers import document_ids
from halo.runtime.collectives import barrier
from halo.runtime.consensus import rank_consensus
from halo.runtime.dtensor import to_local
from halo.runtime.ranks import is_global_main_process
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.distributed import shared_output_dir
from tests.common.harness import gpu_test_main, skip_unless_blackwell
from tests.common.memory import cleanup_memory
from tests.common.models import QWEN3_0_6B
from tests.common.mtp_heads import (
    MTP_GPU_MODES,
    MTP_TINY_FAMILIES,
    collate_tokenized_rows,
    mtp_mode_needs_bf16,
    packed_token_rows,
    tokenized_rows,
    write_checkpoint_with_tail,
)
from tests.common.rank_log import log, log_all
from tests.common.tiny_models import build_tiny_family_checkpoint
from tests.common.tokenizers import load_cached_tokenizer

_ARMS = {
    "all": (
        ("joint", "hard"),
        ("joint", "soft"),
        ("detached", "soft"),
        ("head_only", "hard"),
        ("head_only", "soft"),
    ),
    "soft": (("joint", "soft"),),
    "trunk_trains": (("joint", "hard"), ("joint", "soft"), ("detached", "soft")),
}
_WEIGHT = 0.7
_STEPS = 3
_LENGTH = 64
_PROMPT = 8
_PACKED_DOCUMENTS = 2
_LEARNING_RATE = 1e-3
_SEED = 42
# Between a plain model on the chunked path and a shard layout on the fused kernel, at one dtype.
_FP32_BAR = 1e-3
_BF16_OVER_PLAIN_BAR = 1.1
_BF16_GAP_FLOOR = 0.01
_LOSS_BAR = 1e-6


@dataclass(frozen=True)
class _Paths:
    root: str
    base: str
    tail: str


def _source(path: str) -> ModelSource:
    return ModelSource(path, None, path)


def _whole(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.full_tensor().detach() if hasattr(tensor, "full_tensor") else tensor.detach()


def _whole_head_gap(gradients: dict[str, torch.Tensor], reference: dict[str, torch.Tensor], names: list[str]) -> float:
    gap = sum(float((gradients[name] - reference[name]).norm() ** 2) for name in names)
    return math.sqrt(gap / sum(float(reference[name].norm() ** 2) for name in names))


def _expert_ids(head) -> set[int]:
    return {id(param) for _path, layer in named_ep_layers(head) for _name, param in layer.expert_named_params()}


class _FirstStepGradients(TrainerCallback):
    """COLLECTIVE on every rank in one order: the head's gradients as the optimizer would read them, past the reduce.
    ``whole`` holds what the reduce leaves comparable to one plain model: everything but a distributed expert bank, whose
    own sweep runs past this point."""

    def __init__(self, head, *, whole_experts: bool):
        self._head = head
        self._whole_experts = whole_experts
        self.whole: dict[str, torch.Tensor] = {}
        self.experts_with_gradient: list[bool] = []

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if state.global_step != 0:
            return
        experts = _expert_ids(self._head)
        for name, param in self._head.named_parameters():
            if id(param) not in experts:
                self.whole[name] = _whole(param.grad).float().cpu()
        for path, layer in named_ep_layers(self._head):
            for name, param in layer.expert_named_params():
                self.experts_with_gradient.append(
                    param.grad is not None and bool(to_local(param.grad).abs().sum() > 0)
                )
                if self._whole_experts:
                    # The spelling and axes of the bank a plain model holds.
                    self.whole[f"{path}.experts.{name}"] = _whole(param.grad).transpose(1, 2).float().cpu()


def _plain_model(paths: _Paths, objective: MtpObjective, dtype: torch.dtype, device: torch.device):
    """COLLECTIVE: no wrap and no fused kernel, every expert local, the head read the same way. The closed form the
    shard layouts are held to."""
    model = auto_load_model(paths.base, dtype=dtype, attn_implementation="sdpa").to(device)
    finalization.finalize_loaded_model(model)
    attach_mtp_head(
        model,
        MtpHeadRequest(objective, paths.tail),
        pc=argparse.Namespace(fp32_non_ep_params=False, ep_config=lambda: None),
        source=_source(paths.base),
        dtype=dtype,
        wraps_experts=False,
        init_from_scratch=False,
        preserve_checkpoint_precision=False,
    )
    return model.train()


def _documents(row: dict[str, torch.Tensor]) -> list[dict[str, torch.Tensor]]:
    if "position_ids" not in row:
        return [row]
    lengths = torch.unique_consecutive(document_ids(row["position_ids"]), return_counts=True)[1].tolist()
    return [
        dict(zip(row, parts, strict=True))
        for parts in zip(*(t.split(lengths, dim=1) for t in row.values()), strict=True)
    ]


def _mean_gradients(paths, objective, dtype, device, rows: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """COLLECTIVE through the load. Over the rows the ranks trained, at the normalizer the trainer gives one row. SDPA
    scores each document of a packed row alone, as the plain SDPA forward would let one attend across another."""
    total: dict[str, torch.Tensor] = {}
    for row in rows:
        model = _plain_model(paths, objective, dtype, device)
        inputs = {key: tensor.to(device) for key, tensor in row.items()}
        targets = (inputs["labels"] != LABEL_IGNORE_INDEX).sum()
        for document in _documents(inputs):
            model(**document, use_cache=False, num_items_in_batch=targets).loss.backward()
        for name, param in find_mtp_head(model).named_parameters():
            if param.grad is not None:
                total[name] = total.get(name, 0) + param.grad.float().cpu() / len(rows)
        del model
        cleanup_memory()
    return total


@contextlib.contextmanager
def _starved_rank_one(head):
    """Every head token picks experts on rank 0; gate weights stay the router's own."""
    seam = EPMoELayerBase._dispatch_compute_combine
    head_layers = {id(layer) for _path, layer in named_ep_layers(head)}

    def forced(self, flat, experts, weights, input_dtype):
        if id(self) not in head_layers:
            return seam(self, flat, experts, weights, input_dtype)
        span = self.num_experts // self.ep_size
        rows = torch.arange(flat.shape[0], device=flat.device)[:, None]
        slots = torch.arange(experts.shape[1], device=flat.device)[None, :]
        return seam(self, flat, (rows + slots) % span, weights, input_dtype)

    EPMoELayerBase._dispatch_compute_combine = forced
    try:
        yield
    finally:
        EPMoELayerBase._dispatch_compute_combine = seam


def _rows_of_every_rank(ctx, rows: list[dict[str, torch.Tensor]]) -> list[dict[str, torch.Tensor]]:
    if ctx.world_size == 1:
        return rows
    gathered: list = [None] * ctx.world_size
    dist.all_gather_object(gathered, rows)
    return [row for rank_rows in gathered for row in rank_rows]


def _run_arm(ctx, cli, paths: _Paths, tokenizer, scope: str, loss_type: str) -> tuple[dict[str, bool], list[float]]:
    tag = f"{scope}_{loss_type}"
    bf16 = cli.bf16 or mtp_mode_needs_bf16(cli.mode) or cli.attn != "sdpa"
    dtype = torch.bfloat16 if bf16 else torch.float32
    objective = MtpObjective(_WEIGHT, loss_type, scope)
    fused = cli.kernel == "fused"
    parallelism = ParallelismConfig(**MTP_GPU_MODES[cli.mode][0])
    model, _ = load_distributed_model(
        paths.base,
        parallelism,
        dtype=dtype,
        attn_implementation=None if cli.attn == "auto" else cli.attn,
        trust_remote_code=False,
        use_liger_kernel=fused,
        liger_kernel_config={"fused_linear_cross_entropy": True} if fused else None,
        mtp_head=MtpHeadRequest(objective, paths.tail),
    )
    head = find_mtp_head(model)
    config = SFTConfig(
        output_dir=os.path.join(paths.root, tag),
        max_steps=_STEPS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=cli.accumulation,
        learning_rate=_LEARNING_RATE,
        lr_scheduler_type="constant",
        max_grad_norm=0,
        bf16=bf16,
        gradient_checkpointing=cli.gc and scope != "head_only",
        use_liger_kernel=fused,
        liger_kernel_config={"fused_linear_cross_entropy": True} if fused else None,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        seed=_SEED,
    )
    finalize_liger_after_direct_load(config, fused, model)
    vocab_size = model.config.get_text_config().vocab_size
    if cli.packed:
        rows = packed_token_rows(
            vocab_size, 8 * ctx.world_size, seed=_SEED, length=_LENGTH, documents=_PACKED_DOCUMENTS
        )
        collator = select_data_collator(
            tokenizer,
            train_on_completions_only=False,
            packing=True,
            model_config=model.config,
            emit_varlen_one_document=True,
        )
    else:
        rows = tokenized_rows(vocab_size, 8 * ctx.world_size, seed=_SEED, length=_LENGTH, prompt=_PROMPT)
        collator = collate_tokenized_rows
    trainer = DistributedSFTTrainer(
        model=model,
        args=config,
        train_dataset=rows,
        processing_class=tokenizer,
        data_collator=collator,
        parallelism_config=parallelism,
    )
    gradients = _FirstStepGradients(head, whole_experts=parallelism.experts_fsdp_managed)
    trainer.add_callback(gradients)

    seen_rows: list[dict[str, torch.Tensor]] = []
    capture = model.register_forward_pre_hook(
        lambda _module, _args, kwargs: seen_rows.append(
            {key: kwargs[key].detach().cpu() for key in ("input_ids", "labels", "position_ids") if key in kwargs}
        )
        if len(seen_rows) < cli.accumulation
        else None,
        with_kwargs=True,
    )
    fused_on_this_config = bool(
        (getattr(model.config, LIGER_APPLIED_CONFIG_ATTR, None) or {}).get("fused_linear_cross_entropy")
    )
    kernel_calls = []
    fused_kernel = cross_entropy.liger_fused_linear_cross_entropy

    def counted_kernel(*args, **kwargs):
        kernel_calls.append(1)
        return fused_kernel(*args, **kwargs)

    cross_entropy.liger_fused_linear_cross_entropy = counted_kernel
    head_layers = [layer for _path, layer in named_ep_layers(head)]
    dispatches = []
    for layer in head_layers:
        if layer.dispatcher.backend is not None:
            real = layer.dispatcher.backend.dispatch_forward

            def counted(*args, _real=real, **kwargs):
                dispatches.append(1)
                return _real(*args, **kwargs)

            layer.dispatcher.backend.dispatch_forward = counted

    trunk_before = {
        name: _whole(param).float().cpu().clone()
        for name, param in model.named_parameters()
        if id(param) not in {id(p) for p in head.parameters()} and scope == "head_only"
    }
    head_before = {name: to_local(param).float().cpu().clone() for name, param in head.named_parameters()}
    generation = ep_capacity.capacity_scope.generation
    barrier()
    with _starved_rank_one(head) if cli.starve else contextlib.nullcontext():
        trainer.train()
    capture.remove()
    cross_entropy.liger_fused_linear_cross_entropy = fused_kernel

    checks: dict[str, bool] = {}
    forwards = _STEPS * cli.accumulation
    checks[f"{tag}_term_took_the_expected_kernel"] = fused_on_this_config == fused and len(kernel_calls) == (
        forwards if fused else 0
    )
    history = [entry for entry in trainer.state.log_history if "loss" in entry]
    checks[f"{tag}_both_terms_logged"] = (
        all(math.isfinite(entry["ntp_ce"]) and entry["mtp_ce"] > 0 for entry in history) and len(history) == _STEPS
    )
    reshard_fsdp2_modules(model)
    starved_rank = cli.starve and ctx.rank > 0
    expert_ids = _expert_ids(head)
    moved = []
    for name, param in head.named_parameters():
        changed = not torch.equal(to_local(param).float().cpu(), head_before[name])
        if id(param) in expert_ids:
            # A rank's own experts; a starved rank's see no token, so nothing moves them.
            moved.append(changed or to_local(param).numel() == 0 or starved_rank)
        else:
            # One rank's slice of a small tensor can keep its bf16 bytes through a few rounded steps.
            moved.append(rank_consensus(changed).any_ok)
    if not all(moved):
        unmoved = [name for (name, _param), ok in zip(head.named_parameters(), moved, strict=True) if not ok]
        log_all(f"  {tag}: head parameters that did not move: {unmoved}")
    checks[f"{tag}_every_head_param_moved"] = all(moved)
    if head_layers:
        if starved_rank:
            checks[f"{tag}_a_starved_rank_took_no_expert_gradient"] = not any(gradients.experts_with_gradient)
        else:
            checks[f"{tag}_experts_took_gradient"] = all(gradients.experts_with_gradient)
        if head_layers[0].dispatcher.backend is not None:
            checks[f"{tag}_one_head_dispatch_per_forward"] = len(dispatches) == forwards
        checks[f"{tag}_one_generation_per_forward"] = ep_capacity.capacity_scope.generation - generation == forwards
    if scope == "head_only":
        checks[f"{tag}_trunk_bytes_equal"] = all(
            torch.equal(_whole(param).float().cpu(), trunk_before[name])
            for name, param in model.named_parameters()
            if name in trunk_before
        )
    first_loss = [float(history[0]["loss"])]

    if not cli.starve:
        # Ranks sharing a data-parallel slot (expert TP) saw one row.
        unique = {tuple(row["input_ids"].flatten().tolist()): row for row in _rows_of_every_rank(ctx, seen_rows)}
        rows = list(unique.values())
        reference = _mean_gradients(paths, objective, dtype, ctx.device, rows)
        compared = sorted(set(gradients.whole) & set(reference))
        checks[f"{tag}_compared_the_projection"] = any(
            name.endswith(("fc.weight", "eh_proj.weight")) for name in compared
        )
        if head_layers and parallelism.experts_fsdp_managed:
            checks[f"{tag}_compared_the_experts"] = any(".experts." in name for name in compared)
        worst = max(
            float((gradients.whole[name] - reference[name]).norm() / reference[name].norm().clamp_min(1e-20))
            for name in compared
            if reference[name].abs().sum() > 0
        )
        log(f"  {tag}: worst head gradient error {worst:.3e} over {len(compared)} tensors")
        if bf16:
            truth = _mean_gradients(paths, objective, torch.float32, ctx.device, rows)
            run_gap = _whole_head_gap(gradients.whole, truth, compared)
            plain_gap = _whole_head_gap(reference, truth, compared)
            log(f"  {tag}: whole head from fp32: run {run_gap:.3e}, plain bf16 {plain_gap:.3e}")
            checks[f"{tag}_head_gradient_within_bar"] = run_gap <= max(
                _BF16_OVER_PLAIN_BAR * plain_gap, _BF16_GAP_FLOOR
            )
        else:
            checks[f"{tag}_head_gradient_within_bar"] = worst < _FP32_BAR

    trainer.cleanup_ep()
    del trainer, model, head
    cleanup_memory()
    return checks, first_loss


@gpu_test_main(min_world_size=1)
def run(ctx) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=sorted(MTP_TINY_FAMILIES), required=True)
    parser.add_argument("--mode", choices=sorted(MTP_GPU_MODES), required=True)
    parser.add_argument("--arms", choices=sorted(_ARMS), default="all")
    parser.add_argument("--gc", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--starve", action="store_true")
    parser.add_argument("--kernel", choices=("fused", "chunked"), default="fused")
    parser.add_argument("--attn", default="sdpa")
    parser.add_argument("--packed", action="store_true")
    parser.add_argument("--accumulation", type=int, default=1)
    cli = parser.parse_args()
    if cli.attn in ("auto", "flash_attention_4"):
        skip_unless_blackwell(f"--attn {cli.attn}")
    required = MTP_GPU_MODES[cli.mode][1]
    if ctx.world_size != required:
        raise ValueError(f"--mode {cli.mode} runs on {required} ranks, launched on {ctx.world_size}.")
    if cli.kernel == "chunked":
        # Past the row length, so each row's plane really splits.
        cross_entropy.logit_chunk_rows = lambda _vocab: _LENGTH // 4

    tokenizer = load_cached_tokenizer(QWEN3_0_6B)
    root = shared_output_dir(ctx)
    paths = _Paths(root, os.path.join(root, "base"), os.path.join(root, "tail"))
    if is_global_main_process():
        build_tiny_family_checkpoint(MTP_TINY_FAMILIES[cli.family], paths.base, tokenizer)
        write_checkpoint_with_tail(cli.family, Path(paths.tail), per_expert=True)
    barrier()

    checks: dict[str, bool] = {}
    first_losses: dict[tuple[str, str], float] = {}
    for scope, loss_type in _ARMS[cli.arms]:
        log(f"\n--- {cli.family} {cli.mode} {scope} {loss_type} ---")
        arm_checks, first_loss = _run_arm(ctx, cli, paths, tokenizer, scope, loss_type)
        checks |= arm_checks
        first_losses[(scope, loss_type)] = first_loss[0]
    for scope in ("joint", "head_only"):
        if (scope, "hard") in first_losses and (scope, "soft") in first_losses:
            hard, soft = first_losses[(scope, "hard")], first_losses[(scope, "soft")]
            checks[f"{scope}_soft_keeps_the_hard_loss"] = abs(hard - soft) <= _LOSS_BAR * abs(hard)
    return {
        "checks": checks,
        "metrics": {"first_losses": {"_".join(key): value for key, value in first_losses.items()}},
    }


if __name__ == "__main__":
    raise SystemExit(run())
