"""Shared body of the LoRA weight-sync exactness suites: the syncs leave the frozen base bit-identical.

Every rollout-engine sync (online GRPO, async GRPO with environments, SDPG; vLLM or SGLang) folds the
PEFT adapters into their bf16 base weights in place, gathers and forwards the merged policy, and
unfolds. The unfold alone, ``(w + d) - d``, misses ``w`` by a rounding step wherever the two roundings
do not cancel, and the sync repeats every few steps for the whole run, so the sync writes the base
back after it. ``test_lora_weight_sync_exact.py`` runs a representative dense and MoE family per mode;
``test_lora_weight_sync_exact_families.py`` sweeps every other family the sync serves.

A row is one tiny random-init family (``--family``) under one sharding (``--mode``: ``fsdp`` for a
dense model; ``ep1`` with FSDP-sharded DTensor experts, ``ep2`` with plain experts, or pure ETP
``etp2`` for a MoE) with stock PEFT on the attention projections (every linear layer on a dense
model), alone or mixed with native expert LoRA (``--adapters``). Syncs run through the trainers' own
entry (``sync_trainer_weights``, no server), and the row must:

  1. Cover the roster and the layout it names: the family table holds every EP family some engine
     takes an online update for, derived from the layer registry and the clients' refusals; the
     production gate (``validate_weight_sync_support``) admits the model for at least one engine; the
     LoRA'd base weights are FSDP2 DTensor shards (pure ETP included, whose FSDP2 spans the world at
     data parallel 1); experts are DTensor at ep1 and plain otherwise; a mixed row carries native
     expert adapters.
  2. Across a sync after each of ``STEP_SYNCS`` optimizer steps, leave every frozen parameter
     bit-identical while the adapters train.
  3. With the adapters of the next step fixed, ``PUSHES`` syncs (a recording sender on the forwarding
     rank), each after a forward that leaves FSDP2's unsharded params registered, leave the base and
     adapters bit-identical and forward the same bytes every push: each PEFT-LoRA'd weight as
     ``w + delta`` exactly as PEFT's merge adds it, the fold moving the trained ones off ``w``; a mixed
     row's experts with their delta folded, off the base experts, and a PEFT-only row's as their base.
     Expected tensors are spelled by the sync's own forwarder, so export renames and hub-namespace
     reverts name them as a push does.
  4. Negative control: ``CONTROL_STEPS`` further step syncs with the write-back dropped move the base
     (per-sync counts and the largest drift are reported as metrics), so checks 2 and 3 cannot pass
     vacuously on this row.

TP is no mode: every LoRA shape is refused at trainer construction under TP.
"""

import argparse
import os
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    Gemma4ForCausalLM,
    Gemma4TextConfig,
    Glm4MoeLiteConfig,
    Glm4MoeLiteForCausalLM,
    GptOssConfig,
    GptOssForCausalLM,
    LagunaConfig,
    LagunaForCausalLM,
    Lfm2MoeConfig,
    Lfm2MoeForCausalLM,
    Qwen3_5ForCausalLM,
    Qwen3_5MoeForCausalLM,
    Qwen3_5MoeTextConfig,
    Qwen3_5TextConfig,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
    TrainerCallback,
)
from transformers.models.step3p7.configuration_step3p7 import Step3p7Config
from transformers.models.step3p7.modeling_step3p7 import Step3p7ForConditionalGeneration
from trl import SFTConfig

from src.distributed.checkpoint.peft import find_peft_model
from src.distributed.expert_parallel.expert_weights import ep_layer_class_by_model_type, ep_layer_classes, has_ep_lora
from src.distributed.nccl.registry import resolve_weight_sync_client, rollout_backends
from src.distributed.parallelism_config import ParallelismConfig
from src.models.patches.remote_code_compat import apply_remote_code_compat_shims
from src.models.structure import unwrap_framework_wrappers
from src.trainers.grpo.rollout.weight_sync import sync_trainer_weights, validate_weight_sync_support
from src.trainers.mixins.ep_introspection import named_ep_layers
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.models import (
    BAILING_MOE_LING_MINI,
    QWEN3_0_6B,
    TINY_BAILING_MOE_CONFIG,
    TINY_GEMMA4_MOE_CONFIG,
    TINY_GLM4_MOE_LITE_CONFIG,
    TINY_GPTOSS_CONFIG,
    TINY_LAGUNA_CONFIG,
    TINY_LFM2_MOE_CONFIG,
    TINY_QWEN3_CONFIG,
    TINY_QWEN3_MOE_CONFIG,
    TINY_QWEN35_CONFIG,
    TINY_QWEN35_MOE_CONFIG,
    TINY_STEP3P7_CONFIG,
    TINY_STEP3P7_VISION_CONFIG,
)
from tests.common.peft_helpers import attention_linear_leaves, load_peft_model
from tests.common.utils import log
from tests.common.weight_sync import (
    RecordingSender,
    as_pushed,
    local_parameters,
    lora_bases_and_merges,
    moved_parameters,
    without_the_base_write_back,
)

SEED = 42
WORLD_SIZE = 2
# Optimizer steps followed by one sync each, as shipped; then, at the next step, PUSHES syncs of the
# adapters that step left; then CONTROL_STEPS step syncs with the base write-back dropped.
STEP_SYNCS = 2
PUSHES = 4
CONTROL_STEPS = 3
# Large enough that each step moves lora_B by far more than a bf16 rounding step of the base.
LEARNING_RATE = 5e-3
PROBE_TOKENS = 32
MODES = {
    "fsdp": {},
    "ep1": {"ep_size": 1},
    "ep2": {"ep_size": 2},
    "etp2": {"ep_size": 1, "expert_tp_size": 2},
}
DENSE_MODES = ("fsdp",)
# Logical expert projections, resolved per family by split_expert_lora_targets; on a dense model the
# same names are its MLP, which stock PEFT adapts.
MLP_TARGETS = ["gate_proj", "up_proj", "down_proj"]


@dataclass(frozen=True)
class Family:
    """A tiny random-init model at the tokenizer's vocab, and the EP registry key it builds (``None``
    for a dense family)."""

    build: Callable[[AutoTokenizer], torch.nn.Module]
    model_type: str | None = None


def _vocab(tokenizer) -> dict:
    return {
        "vocab_size": len(tokenizer),
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }


def _bailing(tokenizer) -> torch.nn.Module:
    """Remote code: the config class is the one the hub checkpoint ships, shrunk to the tiny shape."""
    apply_remote_code_compat_shims()
    config = AutoConfig.from_pretrained(BAILING_MOE_LING_MINI, trust_remote_code=True)
    for key, value in {**TINY_BAILING_MOE_CONFIG, **_vocab(tokenizer)}.items():
        setattr(config, key, value)
    return AutoModelForCausalLM.from_config(config, trust_remote_code=True)


def _step3p7(tokenizer) -> torch.nn.Module:
    """The family ships no text-only CausalLM, so the composite, its image token inside the vocab."""
    config = Step3p7Config(
        text_config={**TINY_STEP3P7_CONFIG, **_vocab(tokenizer)},
        vision_config=dict(TINY_STEP3P7_VISION_CONFIG),
        image_token_id=2000,
    )
    return Step3p7ForConditionalGeneration(config)


FAMILIES: dict[str, Family] = {
    "qwen3": Family(lambda tok: Qwen3ForCausalLM(Qwen3Config(**{**TINY_QWEN3_CONFIG, **_vocab(tok)}))),
    "qwen3_5": Family(lambda tok: Qwen3_5ForCausalLM(Qwen3_5TextConfig(**{**TINY_QWEN35_CONFIG, **_vocab(tok)}))),
    "gpt_oss": Family(lambda tok: GptOssForCausalLM(GptOssConfig(**{**TINY_GPTOSS_CONFIG, **_vocab(tok)})), "gpt_oss"),
    "qwen3_moe": Family(
        lambda tok: Qwen3MoeForCausalLM(Qwen3MoeConfig(**{**TINY_QWEN3_MOE_CONFIG, **_vocab(tok)})), "qwen3_moe"
    ),
    "qwen3_5_moe": Family(
        lambda tok: Qwen3_5MoeForCausalLM(Qwen3_5MoeTextConfig(**{**TINY_QWEN35_MOE_CONFIG, **_vocab(tok)})),
        "qwen3_5_moe_text",
    ),
    "glm4_moe_lite": Family(
        lambda tok: Glm4MoeLiteForCausalLM(Glm4MoeLiteConfig(**{**TINY_GLM4_MOE_LITE_CONFIG, **_vocab(tok)})),
        "glm4_moe_lite",
    ),
    "laguna": Family(lambda tok: LagunaForCausalLM(LagunaConfig(**{**TINY_LAGUNA_CONFIG, **_vocab(tok)})), "laguna"),
    # The per-layer-input table is indexed by the same token ids as the embedding.
    "gemma4": Family(
        lambda tok: Gemma4ForCausalLM(
            Gemma4TextConfig(**{**TINY_GEMMA4_MOE_CONFIG, **_vocab(tok), "vocab_size_per_layer_input": len(tok)})
        ),
        "gemma4_text",
    ),
    "lfm2_moe": Family(
        lambda tok: Lfm2MoeForCausalLM(Lfm2MoeConfig(**{**TINY_LFM2_MOE_CONFIG, **_vocab(tok)})), "lfm2_moe"
    ),
    "bailing_moe": Family(_bailing, "bailing_moe"),
    "step3p7": Family(_step3p7, "step3p7"),
}
# The core-tier rows: one dense and one MoE family; the full tier sweeps the rest.
REPRESENTATIVE_FAMILIES = ("qwen3", "qwen3_moe")


def syncable_ep_classes() -> set[type]:
    """Every EP family class some rollout engine takes an online update for: declared syncable, and
    its own ``model_type`` spelling (``HF_MODEL_TYPES[0]``) left unrefused by at least one client; the
    composite wrappers and sibling spellings behind it are refused under their own entries."""
    clients = [resolve_weight_sync_client(backend) for backend in rollout_backends()]
    return {
        cls
        for cls in ep_layer_classes()
        if cls.HF_MODEL_TYPES
        and cls._supports_weight_sync
        and any(cls.HF_MODEL_TYPES[0] not in client.UNSERVABLE_MODEL_TYPES for client in clients)
    }


def parse_row(families: tuple[str, ...]) -> argparse.Namespace:
    """``--family`` (one of ``families``), ``--mode`` and ``--adapters``, refusing a shape no run takes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=families, required=True)
    parser.add_argument("--mode", choices=sorted(MODES), required=True)
    parser.add_argument("--adapters", choices=("peft", "mixed"), default="peft")
    args, _ = parser.parse_known_args()
    dense = FAMILIES[args.family].model_type is None
    if dense != (args.mode in DENSE_MODES):
        parser.error(f"--mode {args.mode} is not a {'dense' if dense else 'MoE'} sharding")
    if args.adapters == "mixed" and (dense or MODES[args.mode].get("expert_tp_size", 1) > 1):
        parser.error("--adapters mixed needs EP experts without expert TP (expert LoRA is refused under ETP)")
    return args


def _all_ranks(local: bool, device) -> bool:
    flag = torch.tensor([1 if local else 0], device=device)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def _is_adapter(name: str) -> bool:
    """PEFT's ``...lora_A.default.weight`` and the native expert ``..._lora_A`` alike."""
    return "lora_" in name


def _build_tiny_checkpoint(family: str, target_dir: str, tokenizer) -> list[str]:
    """Rank 0: the family's seeded tiny model and the tokenizer, saved; returns its attention leaves,
    the PEFT targets (MLA, fused QKV and gated attention spell them differently)."""
    torch.manual_seed(SEED)
    model = FAMILIES[family].build(tokenizer).to(torch.bfloat16)
    model.save_pretrained(target_dir)
    tokenizer.save_pretrained(target_dir)
    return attention_linear_leaves(model)


def _expert_bases_and_merges(peft_model) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Every EP layer's experts gathered as the push gathers them, without and with the native delta,
    under their live names. Collective within the EP group; empty on a dense model."""
    bases, merges = {}, {}
    for layer_name, layer in named_ep_layers(peft_model).items():
        for merge, out in ((False, bases), (True, merges)):
            for key, tensor in layer.gather_expert_state_dict("cuda", merge_lora=merge).items():
                out[f"{layer_name}.{key}"] = tensor.detach().clone()
    return bases, merges


def _probe_batch(tokenizer, device) -> dict[str, torch.Tensor]:
    """A fixed, rank-identical batch, so an EP forward keeps its dispatch collectives aligned."""
    ids = tokenizer("The capital of France is Paris. " * 8, return_tensors="pt").input_ids[:, :PROBE_TOKENS]
    return {"input_ids": ids.to(device)}


class _SyncEachStep(TrainerCallback):
    """The online trainers' cadence, a sync after every optimizer step while the adapters train, in
    the three phases of checks 2-4. The fixed-adapter pushes replace a step's sync rather than follow
    it: a bf16 fold/unfold of unchanged adapters mostly lands back where the previous one left the
    base, so pushes that followed a sync of the same adapters would barely move it even without the
    write-back. Every rank runs callbacks, so the sync's gathers stay collective."""

    def __init__(self, state: dict):
        self.state = state

    def _parameters(self) -> dict[str, torch.Tensor]:
        return local_parameters(self.state["unwrapped"])

    def on_train_begin(self, args, state, control, **kwargs):
        self.state["start"] = self._parameters()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        trainer = self.state["trainer"]
        if state.global_step <= STEP_SYNCS:
            sync_trainer_weights(trainer, None)
        elif state.global_step == STEP_SYNCS + 1:
            self._fixed_adapter_pushes(trainer)
        else:
            with without_the_base_write_back():
                sync_trainer_weights(trainer, None)
            self._count_drift()
        return control

    def _fixed_adapter_pushes(self, trainer) -> None:
        """``PUSHES`` syncs, each after a no-grad forward; what the forwarding rank handed its client,
        and the expected values. Those are spelled on every rank, so a refusal raises everywhere
        rather than stranding the peers in the next gather."""
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
        """LoRA'd base elements an unrestored sync has moved since the pushes, summed over ranks."""
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
    layers = named_ep_layers(unwrapped)
    experts = [
        param for layer in layers.values() for attr, param in layer.expert_named_params() if not _is_adapter(attr)
    ]
    model_type = FAMILIES[family].model_type
    admitted = []
    for backend in rollout_backends():
        try:
            validate_weight_sync_support(unwrapped, backend)
            admitted.append(backend)
        except ValueError as refusal:
            log(f"  {backend} refuses the sync: {str(refusal).splitlines()[0][:160]}")
    layout = {
        "family_table_covers_the_syncable_roster": {
            ep_layer_class_by_model_type()[f.model_type] for f in FAMILIES.values() if f.model_type
        }
        == syncable_ep_classes(),
        "weight_sync_gate_admits_the_family": bool(admitted),
        "peft_lora_present": peft_model is not None and bool(lora_bases),
        "lora_base_is_fsdp2_sharded": all(isinstance(params[name].data, DTensor) for name in lora_bases),
        "ep_layers_are_the_family_class": model_type is None
        or (bool(layers) and all(type(m) is ep_layer_class_by_model_type()[model_type] for m in layers.values())),
        "experts_dtensor_iff_ep1": model_type is None
        or (bool(experts) and all(isinstance(p.data, DTensor) == (mode == "ep1") for p in experts)),
        "native_expert_lora_iff_mixed": has_ep_lora(unwrapped) == (adapters == "mixed"),
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
    # Any, not every: an adapter the batches never reach (a vision tower on text-only rows) stays at
    # lora_B = 0 and folds nothing.
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
        # Without expert adapters the push folds nothing into the experts.
        checks["pushed_experts_equal_their_base"] = all(
            key in pushes[0] and torch.equal(pushes[0][key], value) for key, value in expert_base.items()
        )
    return checks


def run_lora_sync_exactness(ctx, *, family: str, mode: str, adapters: str) -> dict:
    """Drive one (family x mode x adapters) row; every rank runs it."""
    log(f"\n{'=' * 70}\n  LoRA weight sync leaves the base exact: {family}, {mode}, {adapters}\n{'=' * 70}")
    checks: dict[str, bool] = {}
    shared = [ctx.output_dir]
    dist.broadcast_object_list(shared, src=0)
    base_dir = os.path.join(shared[0], "tiny_base")
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    leaves = [_build_tiny_checkpoint(family, base_dir, tokenizer) if ctx.rank == 0 else None]
    dist.broadcast_object_list(leaves, src=0)
    dense = FAMILIES[family].model_type is None
    targets = leaves[0] + (MLP_TARGETS if dense or adapters == "mixed" else [])

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
            output_dir=os.path.join(shared[0], "train_out"),
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
    checks.update({name: _all_ranks(ok, ctx.device) for name, ok in layout.items()})

    log(
        f"\n  {STEP_SYNCS} step syncs, {PUSHES} fixed-adapter pushes, {CONTROL_STEPS} step syncs without "
        f"the base write-back..."
    )
    state = {
        "trainer": trainer,
        "unwrapped": unwrapped,
        "peft_model": peft_model,
        "rank": ctx.rank,
        "batch": _probe_batch(tokenizer, ctx.device),
        "drifted": [],
        "max_abs_drift": 0.0,
    }
    trainer.add_callback(_SyncEachStep(state))
    trainer.train()

    trained = moved_parameters(state["start"], state["before_pushes"])
    checks["frozen_base_bit_identical_across_step_syncs"] = _all_ranks(
        not any(not _is_adapter(name) for name in trained), ctx.device
    )
    checks["adapters_trained_between_syncs"] = _all_ranks(any(map(_is_adapter, trained)), ctx.device)
    moved = moved_parameters(state["before_pushes"], state["after_pushes"])
    checks["frozen_base_bit_identical_after_pushes"] = _all_ranks(
        not any(not _is_adapter(name) for name in moved), ctx.device
    )
    checks["adapters_bit_identical_after_pushes"] = _all_ranks(not any(map(_is_adapter, moved)), ctx.device)
    checks["negative_control_moves_the_base"] = bool(state["drifted"]) and state["drifted"][-1] > 0
    log(f"  step syncs moved {len(trained)} local parameters, pushes moved {len(moved)} {moved[:3]}")

    lora_bases = [name for name in state["start"] if ".base_layer." in name]
    total = torch.tensor(float(sum(state["start"][name].numel() for name in lora_bases)), device=ctx.device)
    dist.all_reduce(total)
    metrics = {
        "lora_base_elements_summed_over_ranks": int(total.item()),
        **{f"unrestored_drifted_after_step_sync_{i + 1}": n for i, n in enumerate(state["drifted"])},
        "unrestored_max_abs_drift_rank0": state["max_abs_drift"],
        "lora_base_mean_abs_rank0": float(
            torch.cat([state["start"][name].float().abs().flatten() for name in lora_bases]).mean()
        ),
    }
    log(f"  unrestored step syncs moved {state['drifted']} of {int(total.item())} LoRA'd base elements")

    # Rank-local reads only from here: the pushes are on rank 0.
    local = _push_checks(state, adapters) if ctx.rank == 0 else {}
    log(f"  rank-0 push checks: {local}")
    checks.update(ctx.broadcast_checks(local))
    return {"checks": checks, "metrics": metrics}
