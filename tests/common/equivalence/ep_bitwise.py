"""EP bitwise harness: every EP family's forward, backward, gradients and construction, compared bit for bit between
two commits. ``make verify-ep BASE=<sha>``. CPU runs the per-expert loop; GPU runs grouped GEMM in bf16 under
``full_determinism`` on 1 rank, ep2 and ep2+etp2."""

from __future__ import annotations

import argparse
import contextlib
import functools
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import patch

import torch
import torch.distributed as dist
import transformers
from accelerate import PartialState
from accelerate.state import GradientState
from peft import LoraConfig, get_peft_model
from transformers.trainer_utils import enable_full_determinism

from halo.distributed.expert_parallel.config import EPConfig
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from halo.models.moe.expert_lora_spec import ExpertLoraSpec
from halo.models.moe.family_specs import find_moe_family_spec
from halo.optimizers.param_groups import decay_groups
from halo.trainers.grpo.environmental.routing_replay import build_routing_replay_injector
from tests.common.adapters import mixer_linear_leaves, randomize_adapters
from tests.common.distributed import init_distributed, teardown_distributed
from tests.common.ep_reference import random_token_batch, roll_expert_banks, sweep_ep_grads
from tests.common.equivalence.revisions import (
    common_families,
    export_tree,
    quiet_logs,
    resolve_commit,
    run_in_tree,
    split_accepted,
    unseen_controls,
    write_evidence,
)
from tests.common.equivalence.tensor_records import (
    GRADIENT_PREFIX,
    Orders,
    Tensors,
    case_of,
    compare_records,
    raised_records,
    record_case,
    record_paths,
)
from tests.common.parallelism import single_process_ep_config, stubbed_process_groups
from tests.common.repo import REPO_ROOT
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model

FAMILIES = tuple(TINY_MOE_FAMILIES)
SEED = 1234
# Reseeded after the build, so adapters and dropout masks do not hang on how much randomness the build consumed.
ADAPTER_SEED = 1
FORWARD_SEED = 2
BATCH = 2
SEQ = {"cpu": 16, "gpu": 256}
ADAPTER_STD = 0.02
BIAS_SCALE = 0.05
WEIGHT_DECAY = 0.1
LORA_RANK = 8
# The variants that run a forward and backward, by the EPConfig fields each sets on top of the defaults.
FORWARD_VARIANTS = {
    "defaults": {},
    "fp32_masters": {"experts_fsdp_managed": False, "fp32_router": True},
    "expert_lora": {"expert_lora": ExpertLoraSpec(r=LORA_RANK, alpha=16.0, dropout=0.1)},
    "routing_replay": {},
    "bias_update": {},
}
# Grouped path only: the per-expert loop's fp32-expert matmul relies on CUDA autocast, which does nothing on CPU.
GROUPED_VARIANT_FIELDS = {"fp32_masters": {"fp32_experts": True}}
SHARED_OVERLAP_ENV = "HALO_EP_SHARED_OVERLAP"
GPU_EP_MODES = {"ep2": {"ep_size": 2}, "ep2_etp2": {"ep_size": 2, "expert_tp_size": 2}}


@dataclass(frozen=True)
class ConstructionLayout:
    """A construction-only variant: built on stubbed process groups per rank, as ``EPConfig`` fields, optionally with a
    PEFT-saved router."""

    world_size: int
    fields: dict
    saved_router: bool = False


CONSTRUCTION_VARIANTS = {
    # One EP group with peers: the in-backward router hooks.
    "in_backward_grad_sync": ConstructionLayout(2, {"ep_size": 2}),
    # Two EP groups: the trainer's post-backward sweep syncs, so no hook may register.
    "deferred_grad_sync": ConstructionLayout(4, {"ep_size": 2}),
    # PEFT training a saved copy of the router, which each EP layer re-hooks.
    "modules_to_save": ConstructionLayout(2, {"ep_size": 2}, saved_router=True),
    # Pure expert TP: the shard of every expert each rank stores.
    "etp_construction": ConstructionLayout(2, {"ep_size": 1, "expert_tp_size": 2}),
}
VARIANTS = (*FORWARD_VARIANTS, *CONSTRUCTION_VARIANTS)
# Cases that raise today since the variant does not apply: routing replay for families that cannot re-derive gate
# weights at a forced selection.
KNOWN_RAISING = frozenset(
    f"{launch}/{family}/routing_replay"
    for launch in ("cpu", "grouped")
    for family in FAMILIES
    if not find_moe_family_spec(family).supports_routing_replay
)


@dataclass(frozen=True)
class Launch:
    """One producer run in a tree: output label, mode, rank count (``None``: one plain process) and environment."""

    label: str
    mode: str
    nproc: int | None = None
    env: dict[str, str] = field(default_factory=dict)


LAUNCHES = {
    "cpu": (Launch("cpu", "cpu"),),
    "gpu": (
        Launch("grouped", "grouped", nproc=1),
        Launch("ep2-overlap0", "ep2", nproc=2, env={SHARED_OVERLAP_ENV: "0"}),
        Launch("ep2-overlap1", "ep2", nproc=2, env={SHARED_OVERLAP_ENV: "1"}),
        Launch("ep2_etp2-overlap0", "ep2_etp2", nproc=4, env={SHARED_OVERLAP_ENV: "0"}),
        Launch("ep2_etp2-overlap1", "ep2_etp2", nproc=4, env={SHARED_OVERLAP_ENV: "1"}),
    ),
}
LAUNCH_BY_LABEL = {launch.label: launch for tier in LAUNCHES.values() for launch in tier}


def launch_variants(launch: Launch, variants) -> list[str]:
    """The variants ``launch`` records: all on CPU, those that forward on grouped, the defaults alone in a multi-rank
    mode."""
    if launch.mode == "cpu":
        return list(variants)
    if launch.mode == "grouped":
        return [variant for variant in variants if variant in FORWARD_VARIANTS]
    return ["defaults"]


def control_case(launch: Launch, families, variants) -> tuple[str, str] | None:
    """The ``(family, variant)`` the negative control reruns: the first case outside :data:`KNOWN_RAISING`, or None."""
    return next(
        (
            (family, variant)
            for family in families
            for variant in launch_variants(launch, variants)
            if f"{launch.label}/{family}/{variant}" not in KNOWN_RAISING
        ),
        None,
    )


def _model(family: str, dtype: torch.dtype, device: torch.device | str = "cpu") -> torch.nn.Module:
    torch.manual_seed(SEED)
    return tiny_family_model(TINY_MOE_FAMILIES[family]).to(device=device, dtype=dtype)


def _wrap(model: torch.nn.Module, ep_config: EPConfig, *, perturb: bool) -> None:
    """The EP patch and expert adapters; ``perturb`` is the negative control, every expert bank rotated by one."""
    apply_ep_layers(model, ep_config)
    if ep_config.expert_lora is not None:
        torch.manual_seed(ADAPTER_SEED)
        randomize_adapters(model, std=ADAPTER_STD)
    if perturb and not roll_expert_banks(model):
        raise RuntimeError("the negative control rotated no expert bank")


def _hook_effect(hook, param: torch.Tensor) -> str:
    """What a hook does to a gradient of ones: each all-reduce (op, dtype, group ranks) and the scale, so hooks match
    when they act alike."""
    probe = torch.zeros((), dtype=param.dtype, requires_grad=True)
    probe.grad = torch.ones((), dtype=param.dtype)
    reduces = []

    def all_reduce(tensor, op=dist.ReduceOp.SUM, group=None, async_op=False):
        ranks = "default group" if group is None else dist.get_process_group_ranks(group)
        reduces.append(f"all_reduce({op}, {tensor.dtype}, {ranks})")

    with patch.object(dist, "all_reduce", all_reduce):
        hook(probe)
    return " ".join([*reduces, f"scale {probe.grad.item()!r}"])


def _orders(model: torch.nn.Module) -> Orders:
    params = list(model.named_parameters())
    # HF's decay rule, which reads nothing off the trainer it is a method of.
    decay = transformers.Trainer.get_decay_parameter_names(None, model)
    groups = decay_groups(params, decay, WEIGHT_DECAY)
    return {
        "named_parameters": [
            f"{name} {param.dtype} {list(param.shape)} {param.requires_grad}" for name, param in params
        ],
        "state_dict": list(model.state_dict()),
        "optimizer_param_groups": [
            f"{index} {name}" for index, group in enumerate(groups) for name in group["param_names"]
        ],
        "grad_hooks": [
            f"{name} {_hook_effect(hook, param)}"
            for name, param in params
            for hook in (param._post_accumulate_grad_hooks or {}).values()
        ],
    }


def _replay_mask(injector, seq: int, device) -> torch.Tensor:
    """``[rows, seq, EP layers, top_k]`` distinct experts per token, from a fixed seed."""
    generator = torch.Generator().manual_seed(SEED)
    scores = torch.rand(BATCH, seq, injector.num_ep_layers, injector.num_experts, generator=generator)
    return scores.argsort(dim=-1)[..., : injector.top_k].to(device)


def _enable_bias_update(model: torch.nn.Module) -> None:
    generator = torch.Generator().manual_seed(SEED)
    for _, layer in named_ep_layers(model):
        if layer.enable_bias_balancing():
            bias = torch.randn(layer.num_experts, generator=generator) * BIAS_SCALE
            layer.balancing_biases = bias.to(layer.balancing_biases.device)


def _forward_backward(model: torch.nn.Module, variant: str, seq: int, device, batch_index: int) -> Tensors:
    """One forward and backward on batch ``batch_index``, armed with the replay mask for ``routing_replay``, capturing
    routing where supported."""
    model.train()
    layers = named_ep_layers(model)
    replays = variant == "routing_replay"
    injector = (
        build_routing_replay_injector(model)
        if replays or all(layer.FAMILY.supports_routing_replay for _, layer in layers)
        else None
    )
    if replays:
        injector.arm(_replay_mask(injector, seq, device))
    vocab_size = model.get_input_embeddings().num_embeddings
    ids, labels = random_token_batch(vocab_size, BATCH, seq, device, seed=SEED + batch_index)
    torch.manual_seed(FORWARD_SEED)
    with injector.capture(BATCH, seq) if injector is not None else contextlib.nullcontext([]) as routing:
        outputs = model(input_ids=ids, labels=labels)
    outputs.loss.backward()
    # A no-op in one process.
    sweep_ep_grads(model)
    if replays:
        injector.disarm()
    tensors = {"loss": outputs.loss, "logits": outputs.logits}
    if routing:
        tensors["routing"] = routing[0]
    tensors |= {
        f"{GRADIENT_PREFIX}{name}": param.grad for name, param in model.named_parameters() if param.grad is not None
    }
    for path, layer in layers:
        if getattr(layer, "expert_load_counter", None) is not None:
            tensors[f"expert_load/{path}"] = layer.expert_load_counter
    return tensors


def _forward_case(family: str, variant: str, *, tier: str, perturb: bool, device):
    """One variant that forwards: per-expert fp32 on CPU, grouped bf16 on GPU, and bf16 for ``fp32_masters`` so its
    upcasts change a dtype."""
    grouped = tier == "gpu"
    dtype = torch.bfloat16 if grouped or variant == "fp32_masters" else torch.float32
    fields = FORWARD_VARIANTS[variant] | (GROUPED_VARIANT_FIELDS.get(variant, {}) if grouped else {})
    model = _model(family, dtype, device)
    _wrap(model, single_process_ep_config(use_grouped_gemm=grouped, **fields), perturb=perturb)
    if variant == "bias_update":
        _enable_bias_update(model)
    return _forward_backward(model, variant, SEQ[tier], device, batch_index=0), _orders(model)


def _save_router(model: torch.nn.Module) -> torch.nn.Module:
    """``model`` under token-mixer LoRA with each router in ``modules_to_save``, re-hooked as the trainer does."""
    routers = [f"{path}.{layer._ROUTER_ATTR}" for path, layer in named_ep_layers(model) if layer._ROUTER_ATTR]
    targets = mixer_linear_leaves(model)
    torch.manual_seed(ADAPTER_SEED)
    model = get_peft_model(model, LoraConfig(r=LORA_RANK, target_modules=targets, modules_to_save=routers))
    for _, layer in named_ep_layers(model):
        layer.reattach_router_grad_sync()
    return model


def _construction_case(family: str, variant: str, *, perturb: bool):
    """Every rank of ``variant``'s layout built in turn: each rank's parameters and orders."""
    layout = CONSTRUCTION_VARIANTS[variant]
    tensors, orders = {}, {}
    for rank in range(layout.world_size):
        model = _model(family, torch.float32)
        with stubbed_process_groups(layout.world_size, rank):
            ep_config = EPConfig(
                stage_world_size=layout.world_size,
                nvlink_domain_size=layout.world_size,
                use_grouped_gemm=False,
                experts_fsdp_managed=False,
                **layout.fields,
            )
            _wrap(model, ep_config, perturb=perturb)
            if layout.saved_router:
                model = _save_router(model)
            rank_orders = _orders(model)
        tensors |= {f"rank{rank}/{name}": param for name, param in model.named_parameters()}
        orders |= {f"rank{rank}/{name}": order for name, order in rank_orders.items()}
    return tensors, orders


def _cpu_case(family: str, variant: str, *, perturb: bool):
    if variant in CONSTRUCTION_VARIANTS:
        return _construction_case(family, variant, perturb=perturb)
    return _forward_case(family, variant, tier="cpu", perturb=perturb, device="cpu")


def _gpu_ep_case(family: str, mode: str, *, perturb: bool, device):
    parallelism = ParallelismConfig(**GPU_EP_MODES[mode])
    model = _model(family, torch.bfloat16, device)
    _wrap(model, parallelism.ep_config(), perturb=perturb)
    # The expert-TP partners share a data-parallel batch.
    batch_index = parallelism.get_data_parallel_rank()
    return _forward_backward(model, "defaults", SEQ["gpu"], device, batch_index), _orders(model)


def produce(output: Path, launch: Launch, families, variants, *, perturb: bool = False) -> None:
    """Record ``launch``'s cases under ``output/<launch label>/<family>/<variant>``."""
    with quiet_logs():
        if launch.mode == "cpu":
            for family in families:
                for variant in launch_variants(launch, variants):
                    body = functools.partial(_cpu_case, family, variant, perturb=perturb)
                    record_case(output, f"{launch.label}/{family}/{variant}", 0, body)
            return
        rank, _, local_rank = init_distributed()
        device = torch.device("cuda", local_rank)
        PartialState()
        GradientState()._set_sync_gradients(True)
        # Before any dispatch builds a DeepEP buffer, which reads the mode then.
        enable_full_determinism(SEED)
        try:
            for family in families:
                for variant in launch_variants(launch, variants):
                    if launch.mode == "grouped":
                        body = functools.partial(
                            _forward_case, family, variant, tier="gpu", perturb=perturb, device=device
                        )
                    else:
                        body = functools.partial(_gpu_ep_case, family, launch.mode, perturb=perturb, device=device)
                    record_case(output, f"{launch.label}/{family}/{variant}", rank, body)
        finally:
            teardown_distributed()


def unexpected_raises(raised: dict[str, dict]) -> list[str]:
    return sorted({case_of(record) for record in raised} - KNOWN_RAISING)


def assess(base: Path, head: Path, control: Path, accept=()) -> tuple[str, dict]:
    """The verdict and findings: fails on a delta outside ``accept``, a head raise outside :data:`KNOWN_RAISING`, or a
    control that matches or raised."""
    deltas, accepted = split_accepted(compare_records(base, head), accept)
    raised = raised_records(head)
    unexpected = unexpected_raises(raised)
    control_records = sorted(record_paths(control))
    unseen = unseen_controls(compare_records(base, control), control_records, raised_records(control))
    findings = {
        "raised": raised,
        "unexpected_raised": unexpected,
        "negative_control": {"records": control_records, "unseen": unseen},
        "accepted": accepted,
        "deltas": deltas,
    }
    return "fail" if deltas or unexpected or unseen else "pass", findings


def _produce_in(tree: Path, output: Path, launch: Launch, families, variants, *, perturb: bool = False) -> None:
    args = ["--produce", str(output), "--launch", launch.label, "--families", *families, "--variants", *variants]
    run_in_tree(tree, __spec__.name, *args, *(["--perturb"] if perturb else []), nproc=launch.nproc, env=launch.env)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare EP forward, backward and gradients bit for bit.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--base-ref", help="the commit the change is compared with")
    mode.add_argument("--produce", type=Path, help="record this tree's cases of one --launch under the directory")
    parser.add_argument("--head-ref", default="HEAD", help="the commit under test (default: HEAD)")
    parser.add_argument("--tier", choices=sorted(LAUNCHES), default="cpu", help="the launches to compare")
    parser.add_argument("--launch", choices=sorted(LAUNCH_BY_LABEL), help="the launch --produce records")
    parser.add_argument("--families", nargs="+", choices=FAMILIES, default=list(FAMILIES))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument(
        "--accept", nargs="+", default=[], metavar="CASE", help="cases whose deltas a flagged change declares"
    )
    parser.add_argument("--perturb", action="store_true", help="rotate every expert bank (the negative control)")
    parser.add_argument("--image", help="the image the run used, recorded in the evidence")
    args = parser.parse_args(argv)
    if args.produce is not None:
        if args.launch is None:
            parser.error("--produce records one launch: name it with --launch")
        produce(args.produce, LAUNCH_BY_LABEL[args.launch], args.families, args.variants, perturb=args.perturb)
        return 0

    base, head = resolve_commit(REPO_ROOT, args.base_ref), resolve_commit(REPO_ROOT, args.head_ref)
    with tempfile.TemporaryDirectory() as scratch:
        trees = {
            label: export_tree(REPO_ROOT, commit, Path(scratch, label))
            for label, commit in (("base", base), ("head", head))
        }
        families, skipped = common_families(trees, __spec__.name, args.families)
        controls = [(launch, control_case(launch, families, args.variants)) for launch in LAUNCHES[args.tier]]
        if bare := [launch.label for launch, case in controls if case is None]:
            parser.error(
                f"the families {families} and --variants give {bare} no case outside KNOWN_RAISING to carry the "
                f"negative control: add a family or variant it runs (the grouped launch runs only "
                f"{list(FORWARD_VARIANTS)})"
            )
        outputs = {label: Path(scratch, f"{label}-records") for label in ("base", "head", "control")}
        for launch, (family, variant) in controls:
            for label in ("base", "head"):
                _produce_in(trees[label], outputs[label], launch, families, args.variants)
            _produce_in(trees["head"], outputs["control"], launch, [family], [variant], perturb=True)
        verdict, assessed = assess(outputs["base"], outputs["head"], outputs["control"], args.accept)
    findings = {
        "tier": args.tier,
        "families": families,
        "skipped": skipped,
        "variants": args.variants,
        "launches": [launch.label for launch, _ in controls],
        **assessed,
    }
    path = write_evidence("ep_bitwise", base=base, head=head, image=args.image, verdict=verdict, findings=findings)
    print(
        f"EP bitwise ({args.tier}): {verdict}, {len(assessed['deltas'])} deltas, {len(assessed['accepted'])} "
        f"accepted, {len(assessed['raised'])} records raising, {len(assessed['unexpected_raised'])} cases "
        f"unexpected, negative control unseen in {len(assessed['negative_control']['unseen'])}, skipped as absent "
        f"from a tree's roster: {skipped or 'none'} ({path})"
    )
    return 0 if verdict == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
