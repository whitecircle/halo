#!/usr/bin/env python
"""Dense per-row scoring under FSDP2 and reentrant checkpointing trains on every row's gradient, on a tiny MoE.

env-GRPO scores each turn row in its own forward, so one backward recomputes every checkpointed decoder
layer once per row; the trainer runs MoE gradient checkpointing reentrant. torch 2.11's FSDP2 hooks only
one of those recomputes, and with one EP group a post-accumulate expert divide repeated per recompute.
``DistributedSFTTrainer`` here scores each of its 4 rows per microbatch in its own forward and runs the
production wrap, checkpointing and EP clip through three of HF's steps (``training_step``, clip,
``model.zero_grad()``) with no optimizer update, so the weights stay fixed and steps 2 and 3 expose any
gradient carried across ``model.zero_grad()``. Each step's gradients, read off the sharded parameters an
optimizer holds, are compared with an unsharded ep1 model of the same checkpoint run without checkpointing
over every DP replica's rows: non-expert parameters whole (TP shards gathered), experts per rank slice, and
expert-TP banks gathered into the family's hub layout. A trainable parameter no gradient reaches (a composite's
vision tower under text rows, a sparse-attention indexer whose top-k indices only mask) keeps none, and the
parameters without one must be the same on both sides, None and exact zeros one state: an EP rank keeps None for
experts no token reached, where the unsharded reference sums zeros. Each rank holding trainable experts must score
some expert gradient against the reference, so no rank's expert sync passes on nothing. Every shard group must be
sharded after each step.

A family whose EP layers hand routing replay the logits of a router returning HF's ``(logits, scores, indices)``
(``routing_can_be_pinned``) runs both references on the policy's own top-k picks, captured off each forward: every
layout's bf16 reduction order flips near-tied picks, each moving a token's whole expert gradient, which the references
would resolve their own way. Each pick must sit within ``TOL.router_near_tie_logit_abs`` of the bf16 reference router's
own k-th logit, so a wrong selection, short by a top-k gap, still fails, and that bar must sit under every layer's gap
to a token's (k+1)-th best expert; each token's picks must name ``top_k`` distinct experts, which the gap alone cannot
see, and the TP and expert-TP partners of a DP replica must pick alike, since the references follow one of them.

A near-cancelled gradient outside the direction and norm bars (its reference under
``TOL.bf16_noise_excusable_layer_fraction`` of its layer's largest) is excused when it sits within
``TOL.bf16_noise_excuse_factor`` of its parameter's measured bf16 noise from an fp32 copy of the reference
(``score_against_bf16_noise``): the 1-8 element hyper-connection vectors of DeepSeek-V4 and GLM-5 Next cancel so
far that the bf16 reference itself misses fp32 by more than the bars. A router's or an expert bank's gradient is
never excused. Each pair outside the bars is logged with its numbers, every parameter must land inside the bars at
the step its fp32 reference gradient is largest (the closest of them to a bar there recorded as
``min_best_step_bar_margin``), and on every kind with an excused pair the same scoring must fail that kind's gradients
doubled (a sync summed twice) and, over several DP replicas, one replica's own (a dropped sync).

Each backward's leftovers are read before the trainer completes them. A one-module group a backward recomputes more
than once (each layer under ZeRO-3, a nested fp32-router group, the one-layer unit an odd layer count leaves under
ZeRO-2) gets FSDP2's post-backward in only one recompute, so every such group must be left to the trainer's
completion; a unit of several layers never is, since torch's final callback runs its post-backward itself.
Removing the completion therefore shows on the rows that recompute one-module groups alone.

Layouts: ``ep1`` (2 GPUs, experts FSDP-sharded), ``ep_single`` (4 GPUs, one EP group of 4),
``ep_multi`` (4 GPUs, two EP groups of 2), ``etp`` (2 GPUs, pure expert TP), ``ep_etp`` (4 GPUs, EP2 x ETP2),
``ep_tp`` (4 GPUs, two EP2 groups under attention TP2), ``tp_dp`` (4 GPUs, TP2 x DP2) and ``hsdp`` (4 GPUs, HSDP
over two simulated NVLink domains); ``--adapter lora`` trains attention LoRA over frozen experts;
``--reshard-after-forward`` (ep1 only) runs ZeRO-3; ``--fp32-router`` splits each layer's router into a nested
shard group (ep1) or keeps it an fp32 master. ``--family`` takes every tiny EP family; one whose spec refuses
gradient checkpointing (``MoEFamilySpec.supports_gradient_checkpointing``) runs the same comparison without it,
which still holds its FSDP2 wrap and multi-group expert sweep to the unsharded reference.

    torchrun --nproc_per_node=2 tests/gpu/parallelism/test_fsdp2_reentrant_recompute.py --family qwen3_moe --layout ep1
    torchrun --nproc_per_node=4 tests/gpu/parallelism/test_fsdp2_reentrant_recompute.py --family gpt_oss --layout ep_single
"""

import argparse
import contextlib
import copy
import os
from collections import defaultdict

import torch
import torch.distributed as dist
import torch.nn.functional as F
from peft import get_peft_model
from torch.distributed.fsdp._fully_shard._fsdp_common import is_bw
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer
from trl import SFTConfig

from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.fsdp import _holds_unsharded_grad, fsdp2_param_groups
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.loader import load_distributed_model
from halo.models.module_tree import backbone_with_layers, decoder_layers
from halo.models.moe.family_specs import find_moe_family_spec
from halo.models.unwrapping import base_transformers_model
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.adapters import load_peft_model
from tests.common.datasets import create_sft_dataset
from tests.common.dp_layout import fsdp2_groups_sharded, hsdp_layout
from tests.common.ep_reference import (
    NOISE_SCORED_CHECK,
    PinnedRoute,
    best_resolved_scores,
    captured_picks,
    fault_passes_bf16_noise_scoring,
    pair_step_grads,
    parameter_kind,
    parameters_outside_the_bars,
    router_parameter_names,
    routing_can_be_pinned,
    routing_pinned,
    score_against_bf16_noise,
    scores_any,
)
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.pp_equivalence import hub_layout_expert_grads
from tests.common.rank_log import log, log_all
from tests.common.tiny_models import TINY_MOE_FAMILIES, shared_tiny_family_checkpoint
from tests.common.tolerances import TOL

SEED = 42
STEPS = 3
ROWS = 4
ROW_LENGTHS = (24, 64)
# Sum-reduced row cross entropy over ~4 x 40 tokens, scaled to keep bf16 gradients well inside range.
LOSS_SCALE = 256.0
# Far above any norm here: the clip runs its EP sweep and scales by exactly 1.
NO_CLIP = 1e9
LAYOUTS = {
    "ep1": (2, {"ep_size": 1}),
    "ep_single": (4, {"ep_size": 4}),
    "ep_multi": (4, {"ep_size": 2}),
    "etp": (2, {"expert_tp_size": 2}),
    "ep_etp": (4, {"ep_size": 2, "expert_tp_size": 2}),
    "ep_tp": (4, {"ep_size": 2, "tp_size": 2}),
    "tp_dp": (4, {"tp_size": 2}),
    "hsdp": (4, hsdp_layout(4)),
}
FAMILIES = tuple(TINY_MOE_FAMILIES)
LORA_TARGETS = ["q_proj", "o_proj"]


def rows_loss(model, inputs) -> torch.Tensor:
    loss = 0.0
    for ids, mask, labels in zip(inputs["input_ids"], inputs["attention_mask"], inputs["labels"], strict=True):
        keep = mask.bool()
        logits = model(input_ids=ids[keep][None]).logits.float()
        loss = loss + F.cross_entropy(logits[0, :-1], labels[keep][1:], ignore_index=-100, reduction="sum")
    return loss / LOSS_SCALE


class RowsTrainer(DistributedSFTTrainer):
    """SFT whose loss scores each row in its own forward, as env-GRPO scores turn rows."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        return rows_loss(model, inputs)


def make_batch(vocab: int, step: int, rank: int, device) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(SEED + 1000 * step + rank)
    lengths = torch.randint(*ROW_LENGTHS, (ROWS,), generator=generator).tolist()
    width = max(lengths)
    ids = torch.randint(0, vocab, (ROWS, width), generator=generator)
    mask = (torch.arange(width)[None] < torch.tensor(lengths)[:, None]).long()
    labels = ids.masked_fill(mask == 0, -100)
    return {"input_ids": ids.to(device), "attention_mask": mask.to(device), "labels": labels.to(device)}


def full(tensor: torch.Tensor) -> torch.Tensor:
    """COLLECTIVE for a DTensor."""
    return (tensor.full_tensor() if isinstance(tensor, DTensor) else tensor).detach().float()


def expert_names(model) -> set[str]:
    """Expert parameters compared under their own name: EP rank slices and FSDP-sharded ep1 banks. Expert-TP banks
    compare by hub key (:func:`etp_expert_grads`)."""
    names = {id(param): name for name, param in model.named_parameters()}
    return {
        names[id(param)]
        for _, layer in named_ep_layers(model)
        if layer.expert_tp_size == 1
        for _, param in layer.expert_named_params()
        if param.requires_grad
    }


def expert_slices(model) -> dict[str, slice]:
    """Each EP-sharded expert's name and this rank's range. FSDP-sharded ep1 experts are whole once gathered, and
    expert-TP banks compare through :func:`etp_expert_grads`: both absent."""
    names = {id(param): name for name, param in model.named_parameters()}
    return {
        names[id(param)]: slice(layer.expert_start, layer.expert_end)
        for _, layer in named_ep_layers(model)
        if not layer.ep_config.experts_fsdp_managed and layer.expert_tp_size == 1
        for _, param in layer.expert_named_params()
        if param.requires_grad
    }


def etp_layer_names(model) -> list[str]:
    return [
        name
        for name, layer in named_ep_layers(model)
        if layer.expert_tp_size > 1 and any(param.requires_grad for _, param in layer.expert_named_params())
    ]


def etp_expert_grads(model, layer_names: list[str], device) -> dict[str, torch.Tensor]:
    """COLLECTIVE (each layer's expert gather): expert gradients in the family's hub layout, keyed ``{layer}.{hub
    key}``. On an unwrapped ep1 model this gathers nothing and only converts the layout, so the two compare key by
    key."""
    layers = dict(named_ep_layers(model))
    return {
        f"{name}.{key}": grad.float()
        for name in layer_names
        for key, grad in hub_layout_expert_grads(layers[name], device=device).items()
    }


def captured_grads(named_params: list[tuple[str, torch.nn.Parameter]]) -> dict[str, torch.Tensor | None]:
    """COLLECTIVE: each parameter's gradient, DTensors gathered (every rank in one order)."""
    return {name: None if param.grad is None else full(param.grad).clone() for name, param in named_params}


def recomputed_groups(model) -> tuple[set[int], set[int]]:
    """The ``id`` of every FSDP2 group inside the decoder layers checkpointing recomputes: those of one module (a
    ZeRO-3 layer, a nested shard group) and the units of several layers."""
    layers = decoder_layers(backbone_with_layers(base_transformers_model(model)))
    inside = {id(module) for layer in layers for module in layer.modules()}
    one_module, units = set(), set()
    for group in fsdp2_param_groups(model):
        if all(id(module) in inside for module in group.modules):
            (one_module if len(group.modules) == 1 else units).add(id(group))
    return one_module, units


def count_backward_forwards(model, group_ids: set[int]) -> dict[int, int]:
    """Per one-module group, a running count of its module's forwards inside a backward. A module whose weights are
    read without calling it (a router weight a layer reads directly) counts none."""
    counts = dict.fromkeys(group_ids, 0)

    def counter(key: int):
        def count(module, args) -> None:
            if is_bw():
                counts[key] += 1

        return count

    for group in fsdp2_param_groups(model):
        if id(group) in counts:
            group.modules[0].register_forward_pre_hook(counter(id(group)))
    return counts


def groups_left_after_backward(model) -> list:
    """The FSDP2 groups a finished backward left gathered or holding a gradient it has not reduce-scattered: the
    post-backward torch's own hooks and final callback skipped."""
    return [group for group in fsdp2_param_groups(model) if group.is_unsharded or _holds_unsharded_grad(group)]


def load_reference(checkpoint: str, dtype: torch.dtype):
    """The unwrapped ep1 model of ``checkpoint`` at ``dtype``. At the run dtype its router computes as the policy's
    FSDP2-cast router does; the fp32 copy measures that dtype's rounding, and its routers may pick other experts."""
    model, _ = load_distributed_model(
        checkpoint,
        ParallelismConfig(),
        dtype,
        trust_remote_code=True,
        attn_implementation="eager",
        use_liger_kernel=False,
    )
    return model


def with_policy_adapters(reference, policy, peft_config):
    """COLLECTIVE (the policy's DTensors gathered): ``reference`` under ``peft_config`` with the policy's adapter
    weights."""
    reference = get_peft_model(reference, copy.deepcopy(peft_config))
    reference_params = dict(reference.named_parameters())
    with torch.no_grad():
        for name, param in policy.named_parameters():
            if param.requires_grad:
                reference_params[name].copy_(full(param))
    return reference


def load_policy(args, checkpoint: str, pc: ParallelismConfig):
    if args.adapter == "lora":
        model, tokenizer, peft_config = load_peft_model(
            "lora",
            pc,
            model_name=checkpoint,
            attn_implementation="eager",
            use_liger_kernel=False,
            lora_target_modules=list(TINY_MOE_FAMILIES[args.family].attention_targets) or LORA_TARGETS,
        )
        # Nonzero B, so A trains too: at B == 0 every A gradient is zero and compares vacuously.
        peft_config.init_lora_weights = False
        return model, tokenizer, peft_config
    model, tokenizer = load_distributed_model(
        checkpoint, pc, torch.bfloat16, trust_remote_code=True, attn_implementation="eager", use_liger_kernel=False
    )
    return model, tokenizer, None


def reference_grads(
    reference, batches: list[list[dict]], picks: list[list[list[torch.Tensor]]] | None, etp_layers: list[str], device
) -> tuple[list[dict], list[list[PinnedRoute]]]:
    """Per step, the gradient of the DP replicas' mean loss over every replica's rows, without checkpointing; None for
    a parameter no gradient reaches. With ``picks``, routed on them, each step also reporting each EP layer's
    :class:`PinnedRoute`."""
    per_step, routes = [], []
    named = [(name, param) for name, param in reference.named_parameters() if param.requires_grad]
    for step, step_batches in enumerate(batches):
        reference.zero_grad(set_to_none=True)
        pinned = (
            routing_pinned(reference, [torch.cat(layer).to(device) for layer in zip(*picks[step], strict=True)])
            if picks is not None
            else contextlib.nullcontext([])
        )
        with pinned as step_routes:
            loss = sum(rows_loss(reference, batch) for batch in step_batches) / len(step_batches)
        loss.backward()
        grads = {name: None if param.grad is None else param.grad.detach().float().clone() for name, param in named}
        per_step.append(grads | etp_expert_grads(reference, etp_layers, device))
        routes.append(step_routes)
    return per_step, routes


def step_pairs(step, grads, ref_grads, fp32_grads, slices, checks: dict, metrics: dict) -> dict:
    """Pair each gradient with the bf16 and fp32 references'; gradient-less parameters must match on both sides."""
    paired = pair_step_grads(grads, ref_grads, fp32_grads, slices)
    gradless, ref_gradless = paired.gradless, paired.reference_gradless
    checks[f"step{step}_every_parameter_has_a_reference"] = not paired.missing
    checks[f"step{step}_gradless_parameters_match_reference"] = gradless == ref_gradless
    checks[f"step{step}_grads_compared"] = len(paired.pairs) > 0
    metrics[f"step{step}_gradless_parameters"] = len(gradless)
    if paired.missing:
        log_all(f"  step {step}: no reference for {paired.missing[:4]}")
    if gradless and gradless == ref_gradless and step == 0:
        log(f"  {len(gradless)} trainable parameters without a gradient on both sides, e.g. {gradless[:4]}")
    if gradless != ref_gradless:
        only_policy, only_reference = set(gradless) - set(ref_gradless), set(ref_gradless) - set(gradless)
        log_all(
            f"  step {step}: without a gradient only here {sorted(only_policy)[:4]}, only in the reference "
            f"{sorted(only_reference)[:4]}"
        )
    return paired.pairs


def record_pinned_routes(
    routes: list[list[PinnedRoute]], fp32_routes: list[list[PinnedRoute]], checks: dict, metrics: dict
) -> None:
    """Picks distinct and within the near-tie bar of the bf16 reference router's own, that bar under each layer's gap
    to a token's next-best expert, and each reference router selecting the top-k of the logits these read."""
    bf16 = [route for step_routes in routes for route in step_routes]
    fp32 = [route for step_routes in fp32_routes for route in step_routes]
    bar = TOL.router_near_tie_logit_abs
    regret = metrics["pinned_pick_regret_max"] = max(route.regret for route in bf16)
    metrics["pinned_pick_regret_max_fp32"] = max(route.regret for route in fp32)
    nearest = metrics["next_pick_regret_min"] = min(route.next_regret for route in bf16)
    flipped = metrics["pinned_flipped_tokens"] = sum(route.flipped for route in bf16)
    duplicated = metrics["pinned_duplicate_pick_tokens"] = sum(route.duplicated for route in bf16)
    own = metrics["reference_own_pick_regret_max"] = max(route.own_regret for route in bf16 + fp32)
    log(
        f"  pinned routing: the policy picked other experts than the bf16 reference router for {flipped} of "
        f"{sum(route.tokens for route in bf16)} layer-tokens ({duplicated} naming one expert twice), worst shortfall "
        f"{regret:.3e} below its k-th logit (fp32 {metrics['pinned_pick_regret_max_fp32']:.3e}; bar {bar:g}); a "
        f"(k+1)-th best pick falls {nearest:.3e} short on the closest layer"
    )
    checks["policy_picks_are_reference_near_ties"] = regret <= bar
    checks["policy_picks_are_distinct_experts"] = duplicated == 0
    checks["the_next_best_pick_falls_outside_the_bar"] = nearest > bar
    checks["reference_routers_select_their_logit_top_k"] = own <= TOL.router_own_pick_logit_abs


def record_scores(scored: list[dict], checks: dict, metrics: dict) -> None:
    """Each step's direction and norm checks; every parameter must score inside the bars at its best resolved step."""
    for step, scores in enumerate(scored):
        for name, score in scores.items():
            if not score.in_bars:
                verdict = "excused by its bf16 noise" if score.excused else "GRAD MISMATCH"
                log_all(
                    f"  step {step} {name}: {verdict} (cos {score.cosine:.4f}, norm ratio {score.ratio:.4f}; "
                    f"reference at {score.layer_fraction:.2e} of its layer's largest, excusable under "
                    f"{TOL.bf16_noise_excusable_layer_fraction:g}; |grad - fp32 grad| {score.fp32_error:.3e} is "
                    f"{score.noise_multiple:.2f}x its bf16 noise {score.noise:.3e}, excusable to "
                    f"{TOL.bf16_noise_excuse_factor:g}x)"
                )
        values = scores.values()
        checks[f"step{step}_grad_norms_match_reference"] = all(s.norm_in_band or s.excused for s in values)
        checks[f"step{step}_grad_directions_match_reference"] = all(s.direction_in_bars or s.excused for s in values)
        metrics[f"step{step}_noise_excused_pairs"] = sum(s.excused for s in values)
        metrics[f"step{step}_min_ratio"] = min((s.ratio for s in values), default=float("nan"))
        metrics[f"step{step}_max_ratio"] = max((s.ratio for s in values), default=float("nan"))
        metrics[f"step{step}_min_cos"] = min((s.cosine for s in values), default=float("nan"))
    excused = [s for scores in scored for s in scores.values() if s.excused]
    metrics["max_noise_excused_layer_fraction"] = max((s.layer_fraction for s in excused), default=0.0)
    metrics["max_noise_excused_multiple"] = max((s.noise_multiple for s in excused), default=0.0)
    closest = min(best_resolved_scores(scored).items(), key=lambda item: item[1].bar_margin, default=None)
    if closest is not None:
        name, score = closest
        metrics["min_best_step_bar_margin"] = score.bar_margin
        log_all(
            f"  closest to a bar at its best resolved step: {name}, {score.bar_margin:.3f}x inside (cos "
            f"{score.cosine:.4f}, norm ratio {score.ratio:.4f})"
        )
    unscored = parameters_outside_the_bars(scored)
    checks[NOISE_SCORED_CHECK] = not unscored
    if unscored:
        log_all(
            f"  GRAD MISMATCH: {unscored} land outside the bars at the step their fp32 reference gradient is largest"
        )


def by_dp_replica(gathered: list[tuple]) -> dict[int, list[tuple]]:
    """Every rank's ``(DP replica, rows, per-EP-layer picks)``, grouped with its TP and expert-TP partners'."""
    replicas = defaultdict(list)
    for replica, rows, layer_picks in gathered:
        replicas[replica].append((rows, layer_picks))
    return replicas


def partners_pick_alike(replicas: dict[int, list[tuple]]) -> bool:
    return all(
        all(torch.equal(mine, theirs) for mine, theirs in zip(partners[0][1], partner[1], strict=True))
        for partners in replicas.values()
        for partner in partners[1:]
    )


def check_sync_faults_fail(steps: list[dict], kinds: list[str], replica_grads, slices, checks: dict) -> None:
    """Negative controls on each noise-excused parameter kind: its gradients doubled, and its gradients replaced by one
    DP replica's own (``replica_grads``, None without a second replica), must each fail the scoring."""
    faults = {"doubled": lambda step, name, grad: 2 * grad}
    if replica_grads is not None:
        faults["dropped"] = lambda step, name, grad: replica_grads[step][name][slices.get(name, slice(None))]
    for fault, apply in faults.items():
        hidden = [
            kind
            for kind in kinds
            if fault_passes_bf16_noise_scoring(steps, kind, apply, cos_min=TOL.ep_grad_cosine_min)
        ]
        checks[f"{fault}_sync_fails_on_every_noise_excused_kind"] = not hidden
        if hidden:
            log_all(f"  NEGATIVE CONTROL MISSED: a {fault} sync on {hidden} still matches")


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--family", choices=FAMILIES, required=True)
parser.add_argument("--layout", choices=sorted(LAYOUTS), required=True)
parser.add_argument("--adapter", choices=("full", "lora"), default="full")
parser.add_argument("--reshard-after-forward", action="store_true")
parser.add_argument("--fp32-router", action="store_true")
ARGS = parser.parse_args()
GRADIENT_CHECKPOINTING = find_moe_family_spec(ARGS.family).supports_gradient_checkpointing


@gpu_test_main(exact_world_size=LAYOUTS[ARGS.layout][0], prefix="fsdp2_reentrant_recompute")
def run(ctx):
    torch.cuda.set_device(ctx.device)
    checks, metrics = {}, {}
    world, fields = LAYOUTS[ARGS.layout]
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    checkpoint = shared_tiny_family_checkpoint(
        ctx, TINY_MOE_FAMILIES[ARGS.family], f"reentrant_{ARGS.family}", tokenizer, SEED
    )
    pc = ParallelismConfig(
        **fields, ep_fp32_router=ARGS.fp32_router, fsdp_reshard_after_forward=ARGS.reshard_after_forward
    )
    log(
        f"{ARGS.family} {ARGS.layout} {ARGS.adapter}: {pc.mode_string}, gradient checkpointing {GRADIENT_CHECKPOINTING}"
    )
    model, tokenizer, peft_config = load_policy(ARGS, checkpoint, pc)
    reference, reference_fp32 = load_reference(checkpoint, torch.bfloat16), load_reference(checkpoint, torch.float32)
    pinned = routing_can_be_pinned(reference, torch.arange(ROW_LENGTHS[0], device=ctx.device)[None])

    config = SFTConfig(
        output_dir=os.path.join(ctx.output_dir, "out"),
        max_steps=STEPS,
        per_device_train_batch_size=ROWS,
        max_grad_norm=NO_CLIP,
        bf16=True,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        gradient_checkpointing_kwargs={"use_reentrant": True} if GRADIENT_CHECKPOINTING else None,
        use_liger_kernel=False,
        save_strategy="no",
        report_to="none",
        logging_nan_inf_filter=False,
        average_tokens_across_devices=False,
    )
    trainer = RowsTrainer(
        model=model,
        args=config,
        train_dataset=create_sft_dataset(8, tokenizer, seed=SEED),
        processing_class=tokenizer,
        parallelism_config=pc,
        peft_config=peft_config,
    )
    ctx.on_teardown(trainer.cleanup_ep)
    policy = trainer.model
    if peft_config is not None:
        reference = with_policy_adapters(reference, policy, peft_config)
        reference_fp32 = with_policy_adapters(reference_fp32, policy, peft_config)
    reference.train()
    reference_fp32.train()
    etp_layers = etp_layer_names(policy)
    etp_params = {
        id(param)
        for name, layer in named_ep_layers(policy)
        if name in etp_layers
        for _, param in layer.expert_named_params()
    }
    # Expert-TP banks carry no sharding-independent name and compare through the hub-layout gather instead.
    trainable = [
        (name, param)
        for name, param in policy.named_parameters()
        if param.requires_grad and id(param) not in etp_params
    ]
    slices = expert_slices(policy)
    experts = expert_names(policy)
    vocab = policy.get_input_embeddings().num_embeddings
    # TP and expert-TP partners read one DP replica's rows; the reference averages over the replicas.
    dp_rank = pc.get_data_parallel_rank()
    # FSDP2 wraps the dense parameters wherever the layout leaves DP replicas to shard over.
    wrapped = bool(fsdp2_param_groups(policy))
    one_module, units = recomputed_groups(policy) if wrapped and GRADIENT_CHECKPOINTING else (set(), set())
    log(
        f"  DP rank {dp_rank} of {pc.data_parallel_size}; {len(etp_layers)} expert-TP layers; FSDP2 wrap {wrapped}; "
        f"recomputed groups: {len(one_module)} of one module, {len(units)} units; references routed on the "
        f"policy's picks {pinned}"
    )
    # What each backward leaves for the trainer's post-backward completion, read before it runs.
    left_after_backward = []
    finish_backward = trainer._finish_fsdp2_backward

    def observe_then_finish(model) -> None:
        left_after_backward.append({id(group) for group in groups_left_after_backward(model)})
        finish_backward(model)

    trainer._finish_fsdp2_backward = observe_then_finish
    recomputes = count_backward_forwards(policy, one_module)

    # HF's per-step order, with gradient accumulation 1.
    trainer.current_gradient_accumulation_steps = 1
    step_grads, batches, picks = [], [], []
    for step in range(STEPS):
        trainer.state.global_step = step
        trainer.accelerator.gradient_state._set_sync_gradients(True)
        batch = make_batch(vocab, step, dp_rank, ctx.device)
        recomputes.update(dict.fromkeys(recomputes, 0))
        with captured_picks(policy) if pinned else contextlib.nullcontext([]) as captured:
            trainer.training_step(policy, batch)
        # Each row's forward, before a reentrant backward's recomputes.
        forwards = [layer_picks[:ROWS] for layer_picks in captured]
        if pinned:
            row_lengths = batch["attention_mask"].sum(dim=1).tolist()
            checks[f"step{step}_policy_picks_cover_every_row"] = all(
                [rows.shape[0] for rows in layer_picks] == row_lengths for layer_picks in forwards
            )
        left = left_after_backward[-1]
        recomputed = {key for key, count in recomputes.items() if count > 1}
        metrics[f"step{step}_groups_left_after_backward"] = len(left)
        metrics[f"step{step}_one_module_groups_recomputed"] = len(recomputed)
        if units:
            checks[f"step{step}_torch_completed_every_unit"] = not (left & units)
        if one_module:
            checks[f"step{step}_recomputed_one_module_groups_left_to_the_completion"] = recomputed <= left
        trainer.accelerator.clip_grad_norm_(policy.parameters(), NO_CLIP)
        etp_grads = etp_expert_grads(policy, etp_layers, ctx.device)
        experts |= set(etp_grads)
        step_grads.append(captured_grads(trainable) | etp_grads)
        policy.zero_grad()
        if wrapped:
            checks[f"step{step}_every_group_sharded"] = fsdp2_groups_sharded(policy)
        gathered = [None] * world
        replica_picks = [torch.cat(rows).cpu() for rows in forwards]
        dist.all_gather_object(gathered, (dp_rank, {key: value.cpu() for key, value in batch.items()}, replica_picks))
        replicas = by_dp_replica(gathered)
        if pinned:
            checks[f"step{step}_dp_replica_partners_pick_alike"] = partners_pick_alike(replicas)
        order = sorted(replicas)
        batches.append([{key: value.to(ctx.device) for key, value in replicas[r][0][0].items()} for r in order])
        picks.append([replicas[r][0][1] for r in order])
    checks["every_dp_replica_held_its_own_rows"] = len(batches[0]) == pc.data_parallel_size

    pinned_picks = picks if pinned else None
    ref_steps, ref_routes = reference_grads(reference, batches, pinned_picks, etp_layers, ctx.device)
    fp32_steps, fp32_routes = reference_grads(reference_fp32, batches, pinned_picks, etp_layers, ctx.device)
    if pinned:
        record_pinned_routes(ref_routes, fp32_routes, checks, metrics)
    steps = [
        step_pairs(step, step_grads[step], ref_steps[step], fp32_steps[step], slices, checks, metrics)
        for step in range(STEPS)
    ]
    if experts:
        checks["this_rank_scores_an_expert_gradient"] = scores_any(steps, experts)
    routed = frozenset(experts | router_parameter_names(policy))
    scored = score_against_bf16_noise(steps, cos_min=TOL.ep_grad_cosine_min, routed=routed)
    record_scores(scored, checks, metrics)
    excused = sorted({parameter_kind(name) for scores in scored for name, score in scores.items() if score.excused})
    metrics["noise_excused_kinds"] = len(excused)
    if excused:
        # The first replica's rows alone: the gradient a rank keeps when its DP sync is dropped.
        replica_grads = None
        if pc.data_parallel_size > 1:
            first_rows = [step_batches[:1] for step_batches in batches]
            first_picks = None if pinned_picks is None else [step_picks[:1] for step_picks in pinned_picks]
            replica_grads = reference_grads(reference, first_rows, first_picks, etp_layers, ctx.device)[0]
        check_sync_faults_fail(steps, excused, replica_grads, slices, checks)
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
