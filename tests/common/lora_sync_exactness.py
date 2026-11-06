"""Shared body of the LoRA weight-sync exactness suite: syncs leave the frozen base bit-identical. Folds are out of
place, since in-place fold and unfold misses ``w`` by a rounding step. No server: a recording sender takes the pushes,
with ``folded_in_place`` as the negative control. TP is no mode, since it refuses every LoRA shape."""

import argparse
import os
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from unittest.mock import patch

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer, TrainerCallback
from trl import SFTConfig

import halo.models.peft_adapters as peft_adapters
from halo.distributed.expert_parallel.moe_layer.expert_lora import model_has_expert_lora
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.fsdp import reshard_fsdp2_modules
from halo.distributed.parallelism_config import ParallelismConfig
from halo.models.peft_adapters import lora_fold_targets
from halo.models.unwrapping import find_peft_model, unwrap_framework_wrappers
from halo.runtime.collectives import broadcast_from_rank0
from halo.trainers.grpo.rollout.weight_push import (
    gather_and_send_weights,
    reject_unsyncable_model,
    sync_trainer_weights,
)
from halo.trainers.sft import DistributedSFTTrainer
from halo.weight_sync.registry import resolve_weight_sync_client, rollout_backends
from tests.common.adapters import load_peft_model, mixed_targets, mixer_linear_leaves
from tests.common.checkpoint_io import fixed_text_batch
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import shared_output_dir, world_all
from tests.common.models import QWEN3_0_6B
from tests.common.rank_log import log, log_all
from tests.common.rosters import ep_layer_class_by_model_type
from tests.common.tiny_models import (
    TINY_DENSE_FAMILIES,
    TINY_MOE_FAMILIES,
    shared_tiny_family_checkpoint,
    tiny_family_model,
)
from tests.common.weight_sync import (
    RecordingSender,
    as_pushed,
    folded_in_place,
    local_parameters,
    lora_bases_and_merges,
    moved_parameters,
)

SEED = 42
WORLD_SIZE = 2
# One sync per optimizer step; then, at the next step, PUSHES syncs of its adapters; then CONTROL_STEPS in-place syncs.
STEP_SYNCS = 3
PUSHES = 4
CONTROL_STEPS = 3
# Large enough that each step moves lora_B by far more than a bf16 rounding step of the base.
LEARNING_RATE = 5e-3
# The out-of-place fold's allowance, in sizes of the largest folded tensor: its shard copy and a full delta. Measured:
# each fold at 1.5 sizes, and held bytes never move; keeping every folded tensor adds the whole LoRA'd shard (1.1-1.5
# MiB).
FOLD_PEAK_BOUND = 2
# The same sequence on every rank, so an EP forward's dispatch collectives stay aligned.
PROBE_TEXT = "The capital of France is Paris. " * 8
MODES = {
    "fsdp": {},
    "ep1": {"ep_size": 1},
    "ep2": {"ep_size": 2},
    "etp2": {"ep_size": 1, "expert_tp_size": 2},
}
DENSE_MODES = ("fsdp",)


def syncable_ep_classes() -> set[type]:
    """Every EP family class some engine takes an online update for: declared syncable and its own ``model_type``
    unrefused by a client."""
    clients = [resolve_weight_sync_client(backend) for backend in rollout_backends()]
    return {
        cls
        for cls in set(ep_layer_class_by_model_type().values())
        if cls.FAMILY.weight_sync_refusal is None
        and any(cls.FAMILY.model_types[0] not in client.UNSERVABLE_MODEL_TYPES for client in clients)
    }


def syncable_moe_families() -> tuple[str, ...]:
    registry, syncable = ep_layer_class_by_model_type(), syncable_ep_classes()
    return tuple(sorted(name for name in TINY_MOE_FAMILIES if registry[name] in syncable))


def row_families() -> tuple[str, ...]:
    """Every ``--family`` a row may name: dense families (``DENSE_MODES`` only) and the syncable MoE roster."""
    return (*TINY_DENSE_FAMILIES, *syncable_moe_families())


def parse_row(families: Iterable[str], argv: list[str] | None = None) -> argparse.Namespace:
    """Parse ``--family``, ``--mode`` and ``--adapters``, refusing a shape no run takes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=sorted(families), required=True)
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument("--adapters", choices=("peft", "mixed"), default="peft")
    args = parser.parse_args(argv)
    dense = args.family in TINY_DENSE_FAMILIES
    if dense != (args.mode in DENSE_MODES):
        parser.error(f"--mode {args.mode} is not a {'dense' if dense else 'MoE'} sharding")
    if args.adapters == "mixed" and (dense or MODES[args.mode].get("expert_tp_size", 1) > 1):
        parser.error("--adapters mixed needs EP experts without expert TP (expert LoRA is refused under ETP)")
    return args


def _is_adapter(name: str) -> bool:
    """PEFT's ``...lora_A.default.weight`` and the native expert ``..._lora_A`` alike."""
    return "lora_" in name


def _expert_bases_and_merges(peft_model) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Every EP layer's experts as the push gathers them, without and with the native delta. Collective within the EP
    group."""
    bases, merges = {}, {}
    for layer_name, layer in named_ep_layers(peft_model):
        for merge, out in ((False, bases), (True, merges)):
            for key, tensor in layer.gather_expert_state_dict(device="cuda", merge_lora=merge).items():
                out[f"{layer_name}.{key}"] = tensor.detach().clone()
    return bases, merges


def _requested_bytes(stat: str) -> int:
    return torch.cuda.memory_stats()[f"requested_bytes.all.{stat}"]


def _gather_without_forwarding(trainer) -> None:
    """The step sync's gathers and folds with no rank forwarding; a sync proper refuses a forwarding rank with no
    client."""
    gather_and_send_weights(unwrap_framework_wrappers(trainer.model), None)


def _sync_peak(model, sync: Callable[[], None]) -> int:
    """Bytes ``sync`` requests at its peak over what was requested before it, on this rank. Requested, not allocated:
    the caching allocator may count a block up to 1 MiB bigger, swamping a tiny row. Resharded first, so the sync's own
    reshard cannot free the step's unsharded params inside the window."""
    reshard_fsdp2_modules(model)
    torch.cuda.synchronize()
    before = _requested_bytes("current")
    torch.cuda.reset_peak_memory_stats()
    sync()
    torch.cuda.synchronize()
    return _requested_bytes("peak") - before


@contextmanager
def _recording_folds(record: list[tuple[int, int]]) -> Iterator[None]:
    """Record each fold's starting bytes and own peak; a fold keeping anything past its tensor shows as a start that
    grows along the walk."""
    fold = peft_adapters.lora_folded

    def recorded(param, layers, **kwargs):
        torch.cuda.synchronize()
        start = _requested_bytes("current")
        torch.cuda.reset_peak_memory_stats()
        folded = fold(param, layers, **kwargs)
        torch.cuda.synchronize()
        record.append((start, _requested_bytes("peak") - start))
        return folded

    with patch.object(peft_adapters, "lora_folded", recorded):
        yield


class _SyncEachStep(TrainerCallback):
    """A sync after every optimizer step, measuring each step sync's peak; every rank runs callbacks, so gathers stay
    collective. Fixed-adapter pushes replace a step's sync: a bf16 fold/unfold of unchanged adapters mostly lands back
    where the last one left the base."""

    def __init__(self, state: dict):
        self.state = state

    def _parameters(self) -> dict[str, torch.Tensor]:
        return local_parameters(self.state["unwrapped"])

    def on_train_begin(self, args, state, control, **kwargs):
        self.state["start"] = self._parameters()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        trainer = self.state["trainer"]
        if state.global_step < STEP_SYNCS:
            self.state["peaks"]["out_of_place"].append(
                _sync_peak(self.state["unwrapped"], lambda: _gather_without_forwarding(trainer))
            )
        elif state.global_step == STEP_SYNCS:
            with _recording_folds(self.state["folds"]):
                _gather_without_forwarding(trainer)
        elif state.global_step == STEP_SYNCS + 1:
            self._fixed_adapter_pushes(trainer)
        else:
            self.state["peaks"]["in_place"].append(
                _sync_peak(self.state["unwrapped"], lambda: self._in_place_sync(trainer))
            )
            self._count_drift()
        return control

    def _in_place_sync(self, trainer) -> None:
        with folded_in_place(self.state["peft_model"]):
            _gather_without_forwarding(trainer)

    def _fixed_adapter_pushes(self, trainer) -> None:
        """``PUSHES`` syncs, each after a no-grad forward; expected values are spelled on every rank, so a refusal
        raises everywhere."""
        peft_model, rank = self.state["peft_model"], self.state["rank"]
        self.state["before_pushes"] = self._parameters()
        lora_base, lora_merged = lora_bases_and_merges(peft_model)
        expert_base, expert_merged = _expert_bases_and_merges(peft_model)
        self.state["expected"] = {
            "lora_base": as_pushed(peft_model, lora_base),
            "lora_merged": as_pushed(peft_model, lora_merged),
            "expert_base": as_pushed(peft_model, expert_base),
            "expert_merged": as_pushed(peft_model, expert_merged),
        }
        pushes = []
        for _ in range(PUSHES):
            with torch.no_grad():
                trainer.model(**self.state["batch"])
            sender = RecordingSender(keep_values=True) if rank == 0 else None
            sync_trainer_weights(trainer, sender)
            pushes.append({param.name: param.value for param in sender.params} if sender else {})
        self.state["pushes"] = pushes
        self.state["after_pushes"] = self._parameters()

    def _count_drift(self) -> None:
        """LoRA'd base elements the in-place syncs have moved since the pushes, summed over ranks."""
        reference, now = self.state["after_pushes"], self._parameters()
        drift = [now[name].float() - reference[name].float() for name in reference if ".base_layer." in name]
        drifted = torch.tensor(float(sum(int(d.count_nonzero()) for d in drift)), device=drift[0].device)
        dist.all_reduce(drifted)
        self.state["drifted"].append(int(drifted.item()))
        self.state["max_abs_drift"] = max([self.state["max_abs_drift"]] + [float(d.abs().max()) for d in drift])


def _layout_checks(family: str, mode: str, adapters: str, unwrapped, peft_model) -> dict[str, bool]:
    """The roster, the gate and the sharding the row names, read off the live tree (rank-local)."""
    params = dict(unwrapped.named_parameters())
    lora_bases = [name for name in params if ".base_layer." in name]
    layers = dict(named_ep_layers(unwrapped))
    experts = [
        param for layer in layers.values() for attr, param in layer.expert_named_params() if not _is_adapter(attr)
    ]
    model_type = None if family in TINY_DENSE_FAMILIES else family
    admitted = []
    for backend in rollout_backends():
        try:
            reject_unsyncable_model(unwrapped, backend)
            admitted.append(backend)
        except ValueError as refusal:
            log(f"  {backend} refuses the sync: {str(refusal).splitlines()[0][:160]}")
    layout = {
        "family_table_covers_the_syncable_roster": {
            ep_layer_class_by_model_type()[name] for name in syncable_moe_families()
        }
        == syncable_ep_classes(),
        "weight_sync_gate_admits_the_family": bool(admitted),
        "peft_lora_present": peft_model is not None and bool(lora_bases),
        "lora_base_is_fsdp2_sharded": all(isinstance(params[name].data, DTensor) for name in lora_bases),
        "ep_layers_are_the_family_class": model_type is None
        or (bool(layers) and all(type(m) is ep_layer_class_by_model_type()[model_type] for m in layers.values())),
        "experts_dtensor_iff_ep1": model_type is None
        or (bool(experts) and all(isinstance(p.data, DTensor) == (mode == "ep1") for p in experts)),
        "native_expert_lora_iff_mixed": model_has_expert_lora(unwrapped) == (adapters == "mixed"),
    }
    log(
        f"  layout: {layout} ({len(lora_bases)} PEFT-LoRA'd weights, {len(experts)} expert tensors, "
        f"engines admitting the sync: {admitted})"
    )
    return layout


def _push_checks(state: dict, adapters: str) -> dict[str, bool]:
    """What the forwarding rank handed its client across the fixed-adapter pushes. Rank 0 only."""
    pushes, expected = state["pushes"], state["expected"]
    merged, base = expected["lora_merged"], expected["lora_base"]
    checks = {"every_push_forwarded_the_lora_weights": bool(merged) and all(set(merged) <= set(p) for p in pushes)}
    checks["pushed_lora_weights_equal_base_plus_delta"] = checks["every_push_forwarded_the_lora_weights"] and all(
        torch.equal(push[key], value) for push in pushes for key, value in merged.items()
    )
    # Any suffices: an adapter the batches never reach (a vision tower on text-only rows) stays at lora_B = 0.
    moved = [key for key in base if not torch.equal(merged[key], base[key])]
    checks["fold_moved_lora_weights"] = bool(moved)
    log(f"  the fold moved {len(moved)}/{len(base)} pushed LoRA'd weights")
    checks["pushes_bit_identical"] = all(
        push.keys() == pushes[0].keys() and all(torch.equal(push[k], pushes[0][k]) for k in push)
        for push in pushes[1:]
    )
    experts, expert_base = expected["expert_merged"], expected["expert_base"]
    if adapters == "mixed":
        checks["pushed_experts_carry_their_delta"] = (
            bool(experts)
            and all(key in pushes[0] and torch.equal(pushes[0][key], value) for key, value in experts.items())
            and any(not torch.equal(experts[key], expert_base[key]) for key in expert_base)
        )
    elif expert_base:
        checks["pushed_experts_equal_their_base"] = all(
            key in pushes[0] and torch.equal(pushes[0][key], value) for key, value in expert_base.items()
        )
    return checks


def run_lora_sync_exactness(ctx, *, family: str, mode: str, adapters: str) -> dict:
    """Drive one (family x mode x adapters) row; every rank runs it."""
    log(f"\n{'=' * 70}\n  LoRA weight sync leaves the base exact: {family}, {mode}, {adapters}\n{'=' * 70}")
    checks: dict[str, bool] = {}
    output_dir = shared_output_dir(ctx)
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    dense = family in TINY_DENSE_FAMILIES
    tiny = TINY_DENSE_FAMILIES[family] if dense else TINY_MOE_FAMILIES[family]
    base_dir = shared_tiny_family_checkpoint(ctx, tiny, f"lora_sync_exactness_{family}", tokenizer, SEED)
    # Off the module tree, since MLA, fused QKV, gated and linear attention spell them differently. The logical expert
    # names become native expert LoRA on a MoE; on a dense model they are its MLP, which stock PEFT adapts.
    leaves = broadcast_from_rank0(mixer_linear_leaves(tiny_family_model(tiny, tokenizer)) if ctx.rank == 0 else None)
    targets = mixed_targets(leaves) if dense or adapters == "mixed" else leaves

    parallelism_config = ParallelismConfig(**MODES[mode])
    model, tokenizer, peft_config = load_peft_model(
        "mixed" if adapters == "mixed" else "lora",
        parallelism_config,
        model_name=base_dir,
        attn_implementation="eager",
        use_liger_kernel=False,
        # On-policy RL keeps GptOss's sinks live: the sync gate refuses a policy whose sinks were reset.
        reset_sinks=False,
        lora_target_modules=targets,
    )
    trainer = DistributedSFTTrainer(
        model=model,
        args=SFTConfig(
            output_dir=os.path.join(output_dir, "train_out"),
            max_steps=STEP_SYNCS + 1 + CONTROL_STEPS,
            per_device_train_batch_size=2,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type="constant",
            bf16=True,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            logging_nan_inf_filter=False,
            max_length=128,
            dataloader_drop_last=True,
            dataloader_num_workers=0,
            use_liger_kernel=False,
            seed=SEED,
        ),
        train_dataset=create_sft_dataset(32, tokenizer, seed=SEED),
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
        peft_config=peft_config,
    )
    ctx.on_teardown(trainer.cleanup_ep)
    unwrapped = unwrap_framework_wrappers(trainer.model)
    peft_model = find_peft_model(unwrapped)
    layout = _layout_checks(family, mode, adapters, unwrapped, peft_model)
    checks.update({name: world_all(ok, ctx.device) for name, ok in layout.items()})

    log(
        f"\n  {STEP_SYNCS} step syncs, {PUSHES} fixed-adapter pushes, {CONTROL_STEPS} step syncs inside an "
        f"in-place merge/unmerge..."
    )
    state = {
        "trainer": trainer,
        "unwrapped": unwrapped,
        "peft_model": peft_model,
        "rank": ctx.rank,
        "batch": {"input_ids": fixed_text_batch(tokenizer, ctx.device, PROBE_TEXT)[0]},
        "drifted": [],
        "max_abs_drift": 0.0,
        "peaks": {"out_of_place": [], "in_place": []},
        "folds": [],
    }
    trainer.add_callback(_SyncEachStep(state))
    trainer.train()

    trained = moved_parameters(state["start"], state["before_pushes"])
    checks["frozen_base_bit_identical_across_step_syncs"] = world_all(
        not any(not _is_adapter(name) for name in trained), ctx.device
    )
    checks["adapters_trained_between_syncs"] = world_all(any(map(_is_adapter, trained)), ctx.device)
    moved = moved_parameters(state["before_pushes"], state["after_pushes"])
    checks["frozen_base_bit_identical_after_pushes"] = world_all(
        not any(not _is_adapter(name) for name in moved), ctx.device
    )
    checks["adapters_bit_identical_after_pushes"] = world_all(not any(map(_is_adapter, moved)), ctx.device)
    checks["negative_control_moves_the_base"] = bool(state["drifted"]) and state["drifted"][-1] > 0
    log(f"  step syncs moved {len(trained)} local parameters, pushes moved {len(moved)} {moved[:3]}")

    lora_bases = [name for name in state["start"] if ".base_layer." in name]
    total = torch.tensor(float(sum(state["start"][name].numel() for name in lora_bases)), device=ctx.device)
    dist.all_reduce(total)
    params = dict(unwrapped.named_parameters())
    largest_fold = max(params[name].numel() * params[name].element_size() for name in lora_bases)
    out_of_place = max(state["peaks"]["out_of_place"][1:])
    lora_local = sum(state["start"][name].numel() * state["start"][name].element_size() for name in lora_bases)
    starts = [start for start, _ in state["folds"]]
    checks["folds_recorded"] = world_all(len(state["folds"]) == len(lora_fold_targets(unwrapped)), ctx.device)
    checks["each_fold_within_bound"] = world_all(
        all(peak <= FOLD_PEAK_BOUND * largest_fold for _, peak in state["folds"]), ctx.device
    )
    checks["folds_keep_nothing_across_tensors"] = world_all(
        bool(starts) and max(starts) - min(starts) <= largest_fold, ctx.device
    )
    # Premise: a copy of the whole LoRA'd shard would exceed the bound on this row.
    checks["memory_bound_discriminates"] = world_all(lora_local > FOLD_PEAK_BOUND * largest_fold, ctx.device)
    log_all(f"  step-sync peaks: out of place {state['peaks']['out_of_place']}, in place {state['peaks']['in_place']}")
    extra = torch.tensor(float(out_of_place - max(state["peaks"]["in_place"])), device=ctx.device)
    dist.all_reduce(extra, op=dist.ReduceOp.MAX)
    checks["sync_peak_within_bound_of_in_place"] = extra.item() <= FOLD_PEAK_BOUND * largest_fold
    mib = 2**20
    metrics = {
        "lora_base_elements_summed_over_ranks": int(total.item()),
        **{f"in_place_drifted_after_step_sync_{i + 1}": n for i, n in enumerate(state["drifted"])},
        "in_place_max_abs_drift_rank0": state["max_abs_drift"],
        "lora_base_mean_abs_rank0": float(
            torch.cat([state["start"][name].float().abs().flatten() for name in lora_bases]).mean()
        ),
        "sync_peak_out_of_place_mib_rank0": out_of_place / mib,
        "sync_peak_in_place_mib_rank0": max(state["peaks"]["in_place"]) / mib,
        "sync_peak_extra_mib_worst_rank": extra.item() / mib,
        "largest_folded_tensor_mib": largest_fold / mib,
        "lora_base_local_mib_rank0": lora_local / mib,
        "fold_start_spread_kib_rank0": (max(starts) - min(starts)) / 1024 if starts else -1.0,
        "fold_peak_over_largest_max_rank0": max((peak for _, peak in state["folds"]), default=-1) / largest_fold,
    }
    log(f"  in-place step syncs moved {state['drifted']} of {int(total.item())} LoRA'd base elements")
    log(
        f"  step-sync peak over allocated: out of place {metrics['sync_peak_out_of_place_mib_rank0']:.3f} MiB, "
        f"in place {metrics['sync_peak_in_place_mib_rank0']:.3f} MiB (bound {FOLD_PEAK_BOUND} x "
        f"{metrics['largest_folded_tensor_mib']:.3f} MiB; the LoRA'd base shard is "
        f"{metrics['lora_base_local_mib_rank0']:.3f} MiB)"
    )

    # Rank-local reads only from here: the pushes are on rank 0.
    local = _push_checks(state, adapters) if ctx.rank == 0 else {}
    log(f"  rank-0 push checks: {local}")
    checks.update(ctx.broadcast_checks(local))
    return {"checks": checks, "metrics": metrics}
