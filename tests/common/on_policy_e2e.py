"""Trainer-agnostic half of an on-policy e2e row: probe the served policy, move it, check it moved. A resume row's
phases share one process, server and group port; the trainer-side close is what frees that port."""

import math
import os
from dataclasses import dataclass

import requests
import torch
from datasets import Dataset
from safetensors.torch import load_file
from torch.distributed.tensor import DTensor

from halo.checkpoint.format import ADAPTER_SAFETENSORS_FILE
from halo.distributed.fsdp import reshard_fsdp2_modules
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.loader import load_distributed_model
from halo.models.attention.gpt_oss_sinks import has_live_attention_sinks, is_sink_key
from halo.models.moe.config_fields import has_routed_experts
from halo.models.moe.expert_lora_spec import ExpertLoraSpec
from tests.common.adapters import (
    LORA_ALPHA,
    LORA_R,
    adapter_param_items,
    assert_adapters_moved,
    load_peft_model,
    snapshot_adapters,
    unwrap,
)
from tests.common.dp_layout import dp_layout_checks, probe_forward
from tests.common.ep_reference import ep_layers, named_router_weights
from tests.common.numerics import max_or_nan
from tests.common.optimizer_state import optimizer_state_matches
from tests.common.rank_log import log
from tests.common.training_checks import step_losses
from tests.common.weight_sync import local_parameters, moved_parameters

# Greedy, one token, top-k: the assertion is that these numbers moved, so the probe must be the least noisy.
PROBE_PROMPT = "The capital of France is"
PROBE_LOGPROBS = 5
PROBE_TIMEOUT_S = 180
_MODEL_LIST_TIMEOUT_S = 60

# Exactly one step follows the checkpoint, so the resume's own push is all that reaches the engine before the resumed
# rollout. A longer run lets the per-step cadence push again and the comparison no longer isolates it.
RESUME_SAVE_STEP = 2
RESUME_MAX_STEPS = RESUME_SAVE_STEP + 1

# Additive on adapters: a freshly trained lora_B can be small enough that a scale factor moves nothing.
WEIGHT_PERTURBATION = 1.05
ADAPTER_PERTURBATION = 0.05
# Sinks enter softmax beside the attention scores; a change small against those is invisible in served logprobs.
SINK_PERTURBATION = 1.0

_QA_PAIRS = (
    ("What is 2 + 2?", "4"),
    ("What is 10 - 3?", "7"),
    ("What is 6 * 7?", "42"),
    ("What is 100 / 4?", "25"),
)


def qa_dataset(num_samples: int) -> Dataset:
    """Short arithmetic prompts with answers; ``answer`` is what an environment grades and SDPG's teacher conditions
    on."""
    pairs = [_QA_PAIRS[i % len(_QA_PAIRS)] for i in range(num_samples)]
    return Dataset.from_dict({"prompt": [p for p, _ in pairs], "answer": [a for _, a in pairs]})


def probe_top_logprobs(server_url: str, model_name: str) -> dict[str, float]:
    """Top-``PROBE_LOGPROBS`` next-token logprobs for ``PROBE_PROMPT``; ``model_name`` must be the id the server
    serves."""
    response = requests.post(
        f"{server_url}/v1/completions",
        json={
            "model": model_name,
            "prompt": PROBE_PROMPT,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": PROBE_LOGPROBS,
        },
        timeout=PROBE_TIMEOUT_S,
    )
    response.raise_for_status()
    top = response.json()["choices"][0]["logprobs"]["top_logprobs"][0]
    return {token: float(value) for token, value in top.items()}


def served_policy_delta(after: dict[str, float], before: dict[str, float]) -> float:
    """Largest logprob gap over the tokens two probes share; ``inf`` when they share none, NaN when any gap is NaN."""
    shared = after.keys() & before.keys()
    if not shared:
        return float("inf")
    return max_or_nan(abs(after[token] - before[token]) for token in shared)


def served_policy_moved(after: dict[str, float], before: dict[str, float]) -> bool:
    """Whether ``after`` probes a different policy: both finite and not identical, since NaN != NaN would read as
    moved."""
    finite = all(bool(probe) and all(math.isfinite(v) for v in probe.values()) for probe in (after, before))
    return finite and after != before


def record_served_baseline(server_url: str, model_name: str, checks: dict[str, bool]) -> dict[str, float]:
    """Rank 0 only: the server serves the checkpoint under test and its probe is reproducible (noise floor at zero)."""
    served = {
        entry["id"] for entry in requests.get(f"{server_url}/v1/models", timeout=_MODEL_LIST_TIMEOUT_S).json()["data"]
    }
    checks["server_serves_the_trained_model"] = model_name in served
    baseline = probe_top_logprobs(server_url, model_name)
    checks["baseline_logprobs_reproducible"] = baseline == probe_top_logprobs(server_url, model_name)
    log(
        f"  {server_url} serves {served} (need {model_name}); baseline { {k: round(v, 4) for k, v in baseline.items()} }"
    )
    return baseline


@dataclass(frozen=True)
class PolicyLayout:
    """The parallel shape an on-policy row trains under; ``gpus_per_node`` simulates NVLink domains of that width (0:
    the launch's own)."""

    ep_size: int = 1
    tp_size: int = 1
    expert_tp_size: int = 1
    reshard_after_forward: bool = False
    shard_ep1_experts: bool = True
    use_hsdp: bool = False
    gpus_per_node: int = 0
    fp32_router: bool = False
    fp32_experts: bool = False

    def config_fields(self) -> dict[str, object]:
        return {
            "ep_size": self.ep_size,
            "tp_size": self.tp_size,
            "expert_tp_size": self.expert_tp_size,
            "fsdp_reshard_after_forward": self.reshard_after_forward,
            "fsdp_shard_ep1_experts": self.shard_ep1_experts,
            "use_hsdp": self.use_hsdp,
            "gpus_per_node": self.gpus_per_node,
            "ep_fp32_router": self.fp32_router,
            "ep_fp32_experts": self.fp32_experts,
        }

    def parallelism_config(self) -> ParallelismConfig:
        """Fresh per phase: ``ep_config`` caches its ``EPConfig``, so a reused object binds a resumed model to phase
        1's groups."""
        return ParallelismConfig(**self.config_fields())


def expert_lora_under_etp_refusal(expert_tp_size: int) -> str:
    """``ParallelismConfig``'s refusal of expert LoRA under expert TP, or ``""``; only construction runs
    ``__post_init__``."""
    try:
        ParallelismConfig(ep_size=1, expert_tp_size=expert_tp_size, expert_lora=ExpertLoraSpec(LORA_R, LORA_ALPHA))
    except ValueError as e:
        return str(e)
    return ""


def load_policy(
    weights_source: str,
    parallelism_config: ParallelismConfig,
    peft: str | None,
    *,
    attn_implementation: str | None = None,
    lora_target_modules: list[str] | None = None,
):
    """Load the policy and adapters through the production path; returns ``(model, peft_config)``. ``weights_source``
    is the base checkpoint, or the resume checkpoint where the EP/CP loader constructs from it."""
    if peft is None:
        model, _ = load_distributed_model(
            model_name_or_path=weights_source,
            parallelism_config=parallelism_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            # The engine serves pretrained sinks; the fine-tuning reset would train a different model and sync
            # neutralized sinks.
            reset_sinks=False,
        )
        return model, None
    model, _, peft_config = load_peft_model(
        peft,
        parallelism_config,
        model_name=weights_source,
        reset_sinks=False,
        attn_implementation=attn_implementation,
        lora_target_modules=lora_target_modules,
    )
    return model, peft_config


def parallelism_engaged(trainer, layout: PolicyLayout) -> bool:
    """Whether every axis ``layout`` requests is realized on the built model; an ignored size would rerun the ep1 path
    and pass. Conjunctive, since a 4-rank row asks for two axes and the EP half could otherwise go missing."""
    config = trainer.parallelism_config
    realized = []
    if layout.tp_size > 1:
        # The config alone would not prove TP was applied.
        realized.append(config.is_tp_mode and any(isinstance(p.data, DTensor) for p in trainer.model.parameters()))
    if layout.expert_tp_size > 1:
        # Only the layers' own expert-TP width separates a sharded expert FFN from a replicated one.
        realized.append(
            config.is_expert_tp_mode
            and any(getattr(layer, "expert_tp_size", 1) == layout.expert_tp_size for layer in ep_layers(trainer.model))
        )
    if layout.ep_size > 1:
        realized.append(trainer._ep_config is not None and trainer._ep_config.ep_size == layout.ep_size)
    if not realized:
        return any(isinstance(p.data, DTensor) for p in trainer.model.parameters())
    return all(realized)


def local_view(tensor: torch.Tensor) -> torch.Tensor:
    """This rank's own storage for a tensor that may be an FSDP2/TP DTensor; comparable with no collective."""
    local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
    return local.detach().clone()


def _adapter_b_params(model) -> list[tuple[str, torch.nn.Parameter]]:
    """Every ``lora_B``: the zero-init half, so a perturbation cannot be cancelled by an untrained factor on the other
    side."""
    return [(name, param) for name, param in adapter_param_items(model) if "lora_B" in name]


def _perturbation_targets(model, adapter: str | None) -> tuple[str, list[torch.nn.Parameter], int]:
    """``(stream, tensors to move, how many of that stream they carry)``; handles, since the write comes after a probe
    forward."""
    if adapter is not None:
        adapters = [param for _, param in _adapter_b_params(model)]
        return "adapter", adapters, len(adapters)

    dense = [
        param
        for name, param in model.named_parameters()
        if "layers.0" in name and name.endswith(".weight") and param.dtype.is_floating_point
    ]
    experts = _expert_params(model)
    return "expert", dense + experts, len(experts)


def _expert_params(model) -> list[torch.nn.Parameter]:
    """Every EP layer's float expert tensors, walked by layer: fused 3D params are bare attributes with no ``.weight``
    or ``experts`` in the name."""
    return [
        param
        for layer in ep_layers(model)
        for _, param in layer.expert_named_params()
        if param.dtype.is_floating_point
    ]


def _apply_perturbation(targets: list[torch.nn.Parameter], adapter: str | None) -> None:
    """Move every target. Multiplicative on weights; additive on ``lora_B``, which starts at zero."""
    with torch.no_grad():
        for param in targets:
            if adapter is None:
                param.mul_(WEIGHT_PERTURBATION)
            else:
                param.add_(ADAPTER_PERTURBATION)


def frozen_base_weights(model) -> dict[str, torch.Tensor]:
    """This rank's frozen weights an adapter fold must leave bit-identical: every PEFT-wrapped base weight and the rest
    of layer 0. No served-policy probe can see a fold that wrote into the base, so the trainer's own weights are the
    only witness."""
    return local_parameters(
        model,
        keep=lambda name, param: ("layers.0." in name or ".base_layer." in name)
        and not param.requires_grad
        and param.dtype.is_floating_point,
    )


def perturbation_round(
    ctx,
    model,
    *,
    adapter: str | None,
    push,
    checks: dict[str, bool],
    check_base_untouched: bool = True,
) -> str:
    """Move what this row's sync must carry, push it with ``push``, and return the stream's name. Collective. Perturbed
    because a GRPO group whose samples tie has zero advantage and leaves the weights bit-identical. The checkpoint
    config, never the EP layers the count guards, decides whether expert tensors are owed."""
    expert_stream = has_routed_experts(unwrap(model).config)
    reshard_fsdp2_modules(unwrap(model))
    what, targets, stream_count = _perturbation_targets(model, adapter)
    base = frozen_base_weights(model) if adapter is not None and check_base_untouched else {}
    # The state a training step ends in. A sync that reads the forward's transient unsharded params ships the old copy.
    probe_forward(model, ctx.device)
    _apply_perturbation(targets, adapter)
    # Zero expert tensors or adapters here would leave the served-policy check resting on the dense delta alone.
    counted = stream_count if adapter is not None or expert_stream else len(targets)
    checks[f"{what}_perturbation_engaged"] = counted > 0
    log(f"  perturbed {len(targets)} tensor(s); {what} count {counted}")

    push()

    if base:
        moved = moved_parameters(base, frozen_base_weights(model))
        checks["sync_left_the_base_weights_alone"] = not moved
        log(f"  frozen base weights the push moved: {len(moved)}/{len(base)} {moved[:3]}")
    return what


def _push_moves_served_policy(
    ctx, stream: str, *, push, server_url: str, model_name: str, checks: dict[str, bool], check: str
) -> None:
    """Push a perturbation of ``stream`` alone and record under ``check`` whether the served logprobs moved.
    Collective."""
    before = probe_top_logprobs(server_url, model_name) if ctx.rank == 0 else {}
    push()
    ctx.barrier()
    if ctx.rank == 0:
        after = probe_top_logprobs(server_url, model_name)
        checks[check] = served_policy_moved(after, before)
        if not checks[check]:
            log(
                f"  IDENTICAL or non-finite logprobs after a {stream}-only perturbation: the {stream} stream did not land"
            )
        log(f"  post-{stream}-sync: { {k: round(v, 4) for k, v in after.items()} }")
    ctx.barrier()


def expert_round(ctx, model, *, server_url: str, model_name: str, push, checks: dict[str, bool]) -> bool:
    """The expert stream in its own round; returns whether this policy carries one. Collective.
    :func:`perturbation_round` passes on the dense delta even if an engine drops every expert tensor of the sync."""
    if not ep_layers(model):
        return False
    reshard_fsdp2_modules(unwrap(model))
    experts = _expert_params(model)
    # An EP layer that stopped exposing its expert params would otherwise skip the round silently.
    checks["expert_perturbation_engaged"] = bool(experts)
    if not experts:
        return False
    probe_forward(model, ctx.device)
    _apply_perturbation(experts, None)
    _push_moves_served_policy(
        ctx,
        "expert",
        push=push,
        server_url=server_url,
        model_name=model_name,
        checks=checks,
        check="forced_sync_moved_the_served_experts",
    )
    return True


def sink_round(ctx, model, *, server_url: str, model_name: str, push, checks: dict[str, bool]) -> bool:
    """GptOss attention sinks in their own round; returns whether this policy carries any. Collective. An engine reload
    that does not skip the module that ``copy_``s in place (vLLM's ``OAIAttention``) reverts sinks with no error, and
    the dense delta hides it."""
    if not has_live_attention_sinks(model):
        return False
    # A preceding push left the registered params unsharded.
    reshard_fsdp2_modules(unwrap(model))
    with torch.no_grad():
        perturbed = 0
        for name, param in model.named_parameters():
            if "layers.0" in name and is_sink_key(name):
                # Additive: a pretrained sink may sit at zero, where a scale factor changes nothing.
                param.add_(SINK_PERTURBATION)
                perturbed += 1
    # A live-sinks policy whose sinks left named_parameters() (the FA2 reset) has nothing in the sync stream to move.
    checks["sink_perturbation_engaged"] = perturbed > 0
    _push_moves_served_policy(
        ctx,
        "sink",
        push=push,
        server_url=server_url,
        model_name=model_name,
        checks=checks,
        check="forced_sync_moved_the_served_sinks",
    )
    return True


def record_step_losses(trainer, checks: dict[str, bool]) -> tuple[list[float], bool]:
    """``losses_finite`` over the per-step losses; returns ``(losses, the run had a gradient)``. A zero gradient is
    routine (a tied GRPO group has no advantage), so it is reported with no assert."""
    losses = [float(value) for value in step_losses(trainer)]
    checks["losses_finite"] = bool(losses) and all(math.isfinite(value) for value in losses)
    trained = any(value != 0.0 for value in losses)
    log(f"  per-step losses: {losses} (non-zero gradient: {trained})")
    return losses, trained


def record_adapter_training(
    trainer, checks: dict[str, bool], *, before: dict, expert_lora: bool, trained: bool
) -> None:
    """The optimizer stepped the adapters the model forwards through, and they moved. Collective. An optimizer holding
    the wrong side of FSDP2's param swap steps orphans. Movement is asserted only where the run had a gradient."""
    model = unwrap(trainer.model)
    optimizer_params = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    live_adapters = adapter_param_items(model)
    owned = sum(1 for _, param in live_adapters if id(param) in optimizer_params)
    checks["optimizer_owns_the_live_adapters"] = bool(live_adapters) and owned == len(live_adapters)
    ok, detail = assert_adapters_moved(before, snapshot_adapters(model, expert_lora=expert_lora))
    checks["adapters_moved_when_the_run_had_a_gradient"] = ok or not trained
    log(f"  adapters: {detail} (run had a gradient: {trained}); optimizer owns {owned}/{len(live_adapters)}")


def logged_lrs(trainer) -> dict[int, float]:
    return {int(e["step"]): float(e["learning_rate"]) for e in trainer.state.log_history if "learning_rate" in e}


def adapter_file_agreement(restored: dict[str, torch.Tensor], checkpoint: str) -> tuple[int, int, int]:
    """``(matching, compared, non-zero B)`` for restored adapters against the checkpoint's file, the resume's source of
    truth. PEFT drops its active-adapter infix on disk, so attention keys match with it stripped."""
    path = os.path.join(checkpoint, ADAPTER_SAFETENSORS_FILE)
    if not os.path.isfile(path):
        return 0, 0, 0
    saved = load_file(path)
    matching = compared = 0
    for name, value in restored.items():
        key = name.replace(".default", "")
        if key not in saved:
            continue
        compared += 1
        matching += int(torch.equal(value, saved[key].float()))
    nonzero_b = sum(1 for name, value in restored.items() if "lora_B" in name and value.abs().sum() > 0)
    return matching, compared, nonzero_b


def record_resume_continuity(
    checks: dict[str, bool], *, restored: dict | None, phase1_lrs: dict, phase2_lrs: dict
) -> None:
    """Where the resumed run picked up: the checkpoint's step, and phase 1's schedule continued."""
    checks["resume_restored_state_before_the_first_step"] = (
        restored is not None and restored["global_step"] == RESUME_SAVE_STEP
    )
    # A restarted scheduler only lands elsewhere while the schedule varies over these steps.
    checks["resume_schedule_varies_across_steps"] = phase1_lrs.get(RESUME_SAVE_STEP) != phase1_lrs.get(
        RESUME_MAX_STEPS
    )
    checks["resume_continued_the_schedule"] = phase2_lrs.get(RESUME_MAX_STEPS) == phase1_lrs.get(RESUME_MAX_STEPS)
    log(f"  learning rates: phase1={phase1_lrs} phase2={phase2_lrs}")


def record_scheduler_restore(checks: dict[str, bool], restored: dict | None) -> None:
    checks["resume_restored_the_scheduler"] = restored is not None and restored["sched_last_epoch"] == RESUME_SAVE_STEP


def record_optimizer_restore(checks: dict[str, bool], saved: dict, restored: dict) -> None:
    """Optimizer state, bit-exact across the resume. Both snapshots must have captured it."""
    ok, detail = optimizer_state_matches(saved["optimizer"], restored["optimizer"])
    checks["resume_restored_the_optimizer_state"] = ok
    if not ok:
        log(f"  optimizer state was NOT continued: {detail}")


def record_parallelism_engaged(ctx, trainer, layout: PolicyLayout, checks: dict) -> None:
    """Record :func:`parallelism_engaged`; a single process shards nothing, so it has no axis to realize."""
    if ctx.world_size > 1:
        checks["parallelism_engaged"] = parallelism_engaged(trainer, layout)
        log(f"  parallelism engaged: {checks['parallelism_engaged']}")


def _fp32_master_checks(model, layout: PolicyLayout) -> dict[str, bool]:
    """The fp32 masters the layout asks for are still fp32 after the optimizer steps."""
    checks = {}
    if layout.fp32_router:
        routers = named_router_weights(model)
        checks["router_masters_fp32"] = bool(routers) and all(p.dtype == torch.float32 for _, p in routers)
    if layout.fp32_experts:
        experts = [param for layer in ep_layers(model) for _, param in layer.expert_named_params()]
        checks["expert_masters_fp32"] = bool(experts) and all(p.dtype == torch.float32 for p in experts)
    if checks:
        log(f"  fp32 masters: {checks}")
    return checks


def record_dp_layout(ctx, trainer, layout: PolicyLayout, checks: dict) -> None:
    """The data-parallel half of ``layout``, read off the trained policy (:mod:`tests.common.dp_layout`). Collective;
    run after ``train()``."""
    model = unwrap(trainer.model)
    checks |= dp_layout_checks(
        model,
        layout.config_fields(),
        world_size=ctx.world_size,
        ep_config=trainer._ep_config,
        forward=lambda: probe_forward(trainer.model, ctx.device),
    )
    checks |= _fp32_master_checks(model, layout)
