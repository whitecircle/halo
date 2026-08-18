"""The shape every on-policy end-to-end row shares: probe the SERVED policy, move it, prove it moved.

Environmental GRPO (:mod:`tests.common.env_grpo_e2e`) and online GRPO / SDPG
(:mod:`tests.common.online_grpo_e2e`) drive different trainers over different rollout paths, but they
assert the same properties about the same thing — the policy an inference engine actually serves.
This module owns that trainer-agnostic half: the served-policy probe, the policy loader, the
parallelism verdict, the perturbation round (move what the sync must carry, push it, then prove an
adapter fold gave the base weights back), the sink round the dense delta cannot speak for, the
restore-point snapshot a two-phase resume compares, and the constants those need. Each body keeps
only what differs — its trainer classes, its config, its rollout wiring, and the refusals only it
exercises.

Both phases of a resume row run in ONE process against ONE server on the same weight-transfer group
port: a second ``init_communicator`` after ``close_communicator`` succeeds and its sync lands
(measured on vLLM 0.26.0). The trainer-side close is what makes that possible — it aborts an
interrupted update, lifts the pause so the server generates again, and frees the group port
(``src/distributed/nccl/clients/vllm.py``); it does not reach into the server's engine. The SERVER's
previous communicator is destroyed by the NEXT ``/init_weight_transfer_engine``, before that call
builds the new one (``docker/vllm/patches/vllm_weight_transfer_reinit_patch.py``). A leg that never
closes holds its trainer-side port for the life of the process.
"""

import math
import os

import requests
import torch
from datasets import Dataset
from safetensors.torch import load_file
from torch.distributed.tensor import DTensor
from transformers.trainer_callback import TrainerCallback

from src.distributed.expert_parallel.config import ExpertLoraSpec
from src.distributed.fsdp import reshard_fsdp2_modules
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import broadcast_from_rank0
from src.models.patches.gpt_oss_sinks import has_live_attention_sinks, is_sink_key
from tests.common.ep_reference import ep_layers
from tests.common.peft_helpers import (
    LORA_ALPHA,
    LORA_R,
    adapter_param_items,
    assert_adapters_moved,
    load_peft_model,
    snapshot_adapters,
    unwrap,
)
from tests.common.utils import local_optimizer_state, log, optimizer_state_matches, step_losses

# Greedy, one token, top-k: the assertion is "these numbers moved", so the probe must be the least
# noisy generation an engine can give. Every supported engine returns this exact shape from
# /v1/completions.
PROBE_PROMPT = "The capital of France is"
PROBE_LOGPROBS = 5
PROBE_TIMEOUT_S = 180
_MODEL_LIST_TIMEOUT_S = 60

# A resume row's shape, not a budget: the checkpoint needs exactly ONE step after it, so the
# resume's own push is the only thing that can have reached the engine before the resumed rollout.
# A shorter run has no step after the checkpoint; a longer one lets the per-step cadence push again
# and the served-policy comparison stops proving anything.
RESUME_SAVE_STEP = 2
RESUME_MAX_STEPS = RESUME_SAVE_STEP + 1

# Full fine-tuning moves the weights themselves; an adapter run moves the zero-init half of the
# adapter, whose only route into the engine is the sync's merge. Additive there because a freshly
# trained lora_B can still be small enough that a scale factor moves nothing.
WEIGHT_PERTURBATION = 1.05
ADAPTER_PERTURBATION = 0.05
# Attention sinks move by a whole logit: they enter softmax alongside the attention scores, so a
# change small against those is invisible in the served logprobs the sink round compares.
SINK_PERTURBATION = 1.0
# Tokens in the throwaway forward that puts the FSDP2 modules back in their end-of-step state.
UNSHARD_FORWARD_TOKENS = 8
# Ceiling on how far the sync's merge/unmerge round-trip may leave a base weight from where it found
# it, relative to that tensor's own scale. bf16 rounds twice across the round-trip (~2^-8 each); an
# adapter that stayed merged writes the whole perturbed delta in, orders of magnitude above this.
BASE_MERGE_DRIFT = 0.01

_QA_PAIRS = (
    ("What is 2 + 2?", "4"),
    ("What is 10 - 3?", "7"),
    ("What is 6 * 7?", "42"),
    ("What is 100 / 4?", "25"),
)


def qa_dataset(num_samples: int) -> Dataset:
    """Short arithmetic prompts with their answers.

    One builder for both families: ``prompt`` is what every row generates from, and ``answer`` is
    what an environment grades against and what SDPG's privileged teacher conditions on — a column
    the rows that ignore it simply carry.
    """
    pairs = [_QA_PAIRS[i % len(_QA_PAIRS)] for i in range(num_samples)]
    return Dataset.from_dict({"prompt": [p for p, _ in pairs], "answer": [a for _, a in pairs]})


def probe_top_logprobs(server_url: str, model_name: str) -> dict[str, float]:
    """Top-``PROBE_LOGPROBS`` next-token logprobs for ``PROBE_PROMPT``, greedily, from the server.

    ``model_name`` must be the id the server serves — an engine answers a completions request for a
    model it does not host with an error, not with the hosted one's logprobs.
    """
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
    """Largest logprob gap between two probes over the tokens they share; ``inf`` when they share none.

    Two probes of the same policy are bit-identical, so this is zero for them and a real number
    otherwise — a magnitude to compare against another magnitude, never a tolerance to pass.
    """
    shared = after.keys() & before.keys()
    if not shared:
        return float("inf")
    return max(abs(after[token] - before[token]) for token in shared)


def record_served_baseline(server_url: str, model_name: str, checks: dict[str, bool]) -> dict[str, float]:
    """RANK 0 only: the server serves the checkpoint under test, and its probe is reproducible.

    The second probe pins the engine's noise floor at zero, so every later "it moved" means something.
    """
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


def shared_output_dir(ctx) -> str:
    """Rank 0's output dir, on every rank.

    The EP/FSDP2 saver writes ONE directory; a per-rank ``mkdtemp`` would leave every peer with no
    checkpoint to resume from, and its restore would go untested.
    """
    return broadcast_from_rank0(ctx.output_dir if ctx.rank == 0 else None)


def fresh_parallelism_config(ep_size: int, tp_size: int = 1, expert_tp_size: int = 1) -> ParallelismConfig:
    """A FRESH config per phase: ``create_ep_config`` caches the ``EPConfig`` it builds, so a resumed
    phase reusing phase 1's object would rebuild its model against phase 1's expert groups."""
    return ParallelismConfig(ep_size=ep_size, tp_size=tp_size, expert_tp_size=expert_tp_size)


def expert_lora_under_etp_refusal(expert_tp_size: int) -> str:
    """``ParallelismConfig``'s refusal of native expert LoRA under expert TP, or ``""`` if it allowed it.

    Built at CONSTRUCTION, which is the only place ``__post_init__`` runs — assigning ``expert_lora``
    onto a finished config is exactly the spelling that skips this gate.
    """
    try:
        ParallelismConfig(ep_size=1, expert_tp_size=expert_tp_size, expert_lora=ExpertLoraSpec(LORA_R, LORA_ALPHA))
    except ValueError as e:
        return str(e)
    return ""


def load_policy(weights_source: str, parallelism_config: ParallelismConfig, peft: str | None):
    """Load the policy (plus adapters) through the production path; ``(model, peft_config)``.

    ``weights_source`` is what ``model_name_or_path`` would be on this phase — the base checkpoint,
    or the resume checkpoint where the EP/CP loader expects the model to be constructed from it.
    """
    if peft is None:
        model, _ = load_distributed_model(
            model_name_or_path=weights_source,
            parallelism_config=parallelism_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            # On-policy RL trains the policy the ENGINE runs, which serves the checkpoint's pretrained
            # sinks — the loader's fine-tuning reset would train a different model from the one sampled,
            # and the sync would then push neutralized sinks into the server (agent-docs/models/gpt-oss.md).
            # Inert for families without sinks.
            reset_sinks=False,
        )
        return model, None
    model, _, peft_config = load_peft_model(peft, parallelism_config, model_name=weights_source, reset_sinks=False)
    return model, peft_config


def parallelism_engaged(trainer, ep_size: int, tp_size: int, expert_tp_size: int) -> bool:
    """Whether EVERY requested axis is REALIZED on the built model, not merely requested.

    The modes must be told apart by what the model actually IS — a silently-ignored size would
    otherwise let a row re-run the ep1 path and pass. Conjunctive, not first-match: a 4-rank row asks
    for two axes at once (EP+ETP, EP+TP), and grading only the outer one would let the EP half go
    missing while the row still reported its combination engaged.
    """
    config = trainer.parallelism_config
    realized = []
    if tp_size > 1:
        # Toolkit TP shards attention into DTensors; the config alone would not prove it was applied.
        realized.append(config.is_tp_mode and any(isinstance(p.data, DTensor) for p in trainer.model.parameters()))
    if expert_tp_size > 1:
        # Only the layers' own expert-TP width separates a sharded expert FFN from a replicated one.
        realized.append(
            config.is_expert_tp_mode
            and any(getattr(layer, "expert_tp_size", 1) == expert_tp_size for layer in ep_layers(trainer.model))
        )
    if ep_size > 1:
        realized.append(trainer._ep_config is not None and trainer._ep_config.ep_size == ep_size)
    if not realized:
        # ep1 keeps experts in the FSDP2 shard (fsdp_shard_ep1_experts), so they are DTensors.
        return any(isinstance(p.data, DTensor) for p in trainer.model.parameters())
    return all(realized)


def local_view(tensor: torch.Tensor) -> torch.Tensor:
    """This rank's own storage for a tensor that may be an FSDP2/TP DTensor — comparable with no collective."""
    local = tensor.to_local() if isinstance(tensor, DTensor) else tensor
    return local.detach().clone()


def _adapter_b_params(model) -> list[tuple[str, torch.nn.Parameter]]:
    """Every ``lora_B`` — PEFT's attention adapters and the native grouped expert ones alike.

    ``lora_B`` and not ``lora_A``: B is the zero-init half, so a perturbation there cannot be
    cancelled by an untrained factor on the other side.
    """
    return [(name, param) for name, param in adapter_param_items(model) if "lora_B" in name]


def _perturbation_targets(model, adapter: str | None) -> tuple[str, list[torch.nn.Parameter], int]:
    """``(stream, tensors to move, how many of that stream they carry)``.

    Handles, not an in-place walk, because they are taken while the model is RESHARDED and written
    through after :func:`_unshard_with_a_forward` — see :func:`perturbation_round`.
    """
    if adapter is not None:
        adapters = [param for _, param in _adapter_b_params(model)]
        return "adapter", adapters, len(adapters)

    dense = [
        param
        for name, param in model.named_parameters()
        if "layers.0" in name and name.endswith(".weight") and param.dtype.is_floating_point
    ]
    # Expert tensors through the EP layers themselves, never a name match: the wrapper REPLACES the
    # mapped module and registers its fused 3D params as bare attributes
    # (``model.layers.0.mlp.gate_up_proj``), so they carry neither a ``.weight`` suffix nor an
    # ``experts`` path component on most families. A sync whose EXPERT stream silently no-ops while
    # dense lands is exactly what this must catch (SGLang's per-expert loader drops fused keys with
    # no log line), and a dense-only perturbation would pass it.
    experts = [
        param
        for layer in ep_layers(model)
        for _, param in layer.expert_named_params()
        if param.dtype.is_floating_point
    ]
    return "expert", dense + experts, len(experts)


def _apply_perturbation(targets: list[torch.nn.Parameter], adapter: str | None) -> None:
    """Move every target. Multiplicative on weights; additive on ``lora_B``, which starts at zero."""
    with torch.no_grad():
        for param in targets:
            if adapter is None:
                param.mul_(WEIGHT_PERTURBATION)
            else:
                param.add_(ADAPTER_PERTURBATION)


def _unshard_with_a_forward(model, device: torch.device) -> None:
    """One tiny no-grad forward, to leave the FSDP2 modules holding their transient UNSHARDED params.

    That is the state every training step ends in and the state the next step's weight sync reads,
    and it is what makes the sync check discriminating: the perturbation is written through handles
    onto the SHARDS, so a sync that forwards the registered params instead of resharding first ships
    the pre-perturbation copy and the served policy does not move.
    """
    with torch.no_grad():
        model(input_ids=torch.ones(1, UNSHARD_FORWARD_TOKENS, dtype=torch.long, device=device))


def _frozen_base_handles(model) -> list[tuple[str, torch.nn.Parameter]]:
    """The layer-0 base weights the sync's adapter fold must give back untouched.

    No served-policy probe can see a merge that never unmerged — the engine is SUPPOSED to receive
    base+adapter — so the only witness is the trainer's own weights before and after the push.
    """
    return [
        (name, param)
        for name, param in model.named_parameters()
        if "layers.0." in name and "lora_" not in name and param.dtype.is_floating_point
    ]


def _worst_relative_drift(handles: list[tuple[str, torch.nn.Parameter]], before: list[torch.Tensor]) -> float:
    """Largest change in ``handles`` since ``before``, relative to each tensor's own scale."""
    worst = 0.0
    for (_, param), reference in zip(handles, before, strict=True):
        moved = (local_view(param.data) - reference).abs().max()
        worst = max(worst, float((moved / reference.abs().max().clamp(min=1e-6)).item()))
    return worst


def perturbation_round(
    ctx,
    model,
    *,
    adapter: str | None,
    push,
    checks: dict[str, bool],
    expert_stream: bool = True,
    check_base_untouched: bool = True,
) -> str:
    """Move what this row's sync must carry, push it with ``push``, and return the stream's name.

    Perturbed deliberately rather than trusting training to have moved the weights: a GRPO group
    whose samples tie has zero advantage and leaves them bit-identical, and "the logprobs did not
    change" would then say nothing about the sync. Handles are taken while the model is RESHARDED;
    the forward then leaves the FSDP2 modules holding a transient unsharded copy and the write lands
    on the SHARDS — the state an optimizer step ends in and the one the sync reads after its own
    reshard. Every rank runs it: the gathers inside ``push`` are collective.

    ``expert_stream`` says this policy's sync must carry EXPERT tensors, which is where the
    anti-vacuity count then goes; a dense policy has none and counts the tensors it does move.
    ``check_base_untouched`` adds the adapter fold's other half — the engine is SUPPOSED to receive
    base+adapter, so only the trainer's own weights can witness a merge that never unmerged.
    """
    reshard_fsdp2_modules(unwrap(model))
    what, targets, stream_count = _perturbation_targets(model, adapter)
    base_handles = _frozen_base_handles(model) if adapter is not None and check_base_untouched else []
    base_before = [local_view(param.data) for _, param in base_handles]
    _unshard_with_a_forward(model, ctx.device)
    _apply_perturbation(targets, adapter)
    # ANTI-VACUITY, on the number that can actually go missing. A full fine-tune of a MoE policy must
    # move EXPERT tensors — zero of them means no EP wrapper carried any and the served-policy check
    # would rest on the dense delta alone, the blind spot this round exists to close. On an adapter
    # run the same holds of the adapters: nothing perturbed means nothing the fold could carry.
    counted = stream_count if adapter is not None or expert_stream else len(targets)
    checks[f"{what}_perturbation_engaged"] = counted > 0
    log(f"  perturbed {len(targets)} tensor(s); {what} count {counted}")

    push()

    if base_handles:
        drift = _worst_relative_drift(base_handles, base_before)
        checks["sync_left_the_base_weights_alone"] = drift <= BASE_MERGE_DRIFT
        log(f"  base-weight drift across the merge/unmerge round-trip: {drift:.2e} (max {BASE_MERGE_DRIFT})")
    return what


def sink_round(ctx, model, *, server_url: str, model_name: str, push, checks: dict[str, bool]) -> bool:
    """GptOss attention sinks, in a round of their own; returns whether this policy carries any.

    The per-head sink logit is a plain 1-D parameter riding the DENSE stream, and an engine whose
    layerwise reload does not skip the module that ``copy_``s its weights in place (vLLM's
    ``OAIAttention``) reverts exactly it, silently. It needs its own round because
    :func:`perturbation_round` cannot see that: the dense delta moves the served policy whether or
    not the sinks landed. Inert for a sink-less family, and for reset sinks, which the sync gate
    refuses precisely because they are not in the stream at all.

    Every rank runs it: ``push`` is collective.
    """
    if not has_live_attention_sinks(model):
        return False
    # Every write below goes through the registered params, which a preceding push left unsharded.
    reshard_fsdp2_modules(unwrap(model))
    before = probe_top_logprobs(server_url, model_name) if ctx.rank == 0 else {}
    with torch.no_grad():
        perturbed = 0
        for name, param in model.named_parameters():
            if "layers.0" in name and is_sink_key(name):
                # Additive: a pretrained sink may sit at zero, where a scale factor changes nothing.
                param.add_(SINK_PERTURBATION)
                perturbed += 1
    # ANTI-VACUITY: a live-sinks policy whose sinks left named_parameters() (the FA2 reset) has
    # nothing in the sync stream to move, which is what the trainer's own sync gate refuses.
    checks["sink_perturbation_engaged"] = perturbed > 0
    push()
    ctx.barrier()
    if ctx.rank == 0:
        after = probe_top_logprobs(server_url, model_name)
        checks["forced_sync_moved_the_served_sinks"] = after != before
        if after == before:
            log("  IDENTICAL logprobs after a sink-only perturbation: the sinks were not synced")
        log(f"  post-sink-sync: { {k: round(v, 4) for k, v in after.items()} }")
    ctx.barrier()
    return True


def record_step_losses(trainer, checks: dict[str, bool]) -> tuple[list[float], bool]:
    """``losses_finite`` over the per-step losses; returns ``(losses, the run had a gradient)``.

    Whether it had one is reported, not asserted: a GRPO group whose samples all earn the same reward
    has zero advantage and so a genuinely zero gradient — routine on a small policy answering easy
    prompts, and not a defect. It does mean a post-training probe proves nothing, which
    :func:`perturbation_round` handles by forcing a change of its own rather than leaving the
    assertion to sampling luck.
    """
    losses = [float(value) for value in step_losses(trainer)]
    checks["losses_finite"] = bool(losses) and all(math.isfinite(value) for value in losses)
    trained = any(value != 0.0 for value in losses)
    log(f"  per-step losses: {losses} (non-zero gradient: {trained})")
    return losses, trained


def record_adapter_training(
    trainer, checks: dict[str, bool], *, before: dict, expert_lora: bool, trained: bool
) -> None:
    """The optimizer stepped the adapters the model forwards through, and they moved. COLLECTIVE.

    The optimizer must hold the very tensors the model forwards through. It is built from
    ``model.parameters()``, but FSDP2 swaps a module's registered params between its sharded DTensors
    and a transient unsharded copy — an optimizer (or a reader) that captured the wrong side steps
    orphans, and the adapters never move however large the gradient.

    Movement is asserted only where the run HAD a gradient: with a tied group nothing moves, and
    asserting it there would assert sampling luck. The ownership check holds on every run.
    """
    model = unwrap(trainer.model)
    optimizer_params = {id(p) for group in trainer.optimizer.param_groups for p in group["params"]}
    live_adapters = adapter_param_items(model)
    owned = sum(1 for _, param in live_adapters if id(param) in optimizer_params)
    checks["optimizer_owns_the_live_adapters"] = bool(live_adapters) and owned == len(live_adapters)
    ok, detail = assert_adapters_moved(before, snapshot_adapters(model, expert_lora=expert_lora))
    checks["adapters_moved_when_the_run_had_a_gradient"] = ok or not trained
    log(f"  adapters: {detail} (run had a gradient: {trained}); optimizer owns {owned}/{len(live_adapters)}")


class RestorePointSnapshot(TrainerCallback):
    """Trainer/optimizer/scheduler state at ONE lifecycle point.

    ``"save"`` fires when the checkpoint is written, ``"train_begin"`` after the resume restore and
    before the first resumed step — so the two snapshots describe the SAME step and a warm-restarted
    optimizer cannot hide behind the steps that follow.

    The FIRST occurrence wins: a run that stops on ``max_steps`` writes a final checkpoint too, and
    a snapshot overwritten there would describe a step the resume never restores.

    ``capture_optimizer`` is off for a full fine-tune of a large policy: ``local_optimizer_state``
    offloads the whole state to host RAM, which is 6 B/param of AdamWBF16 moments. ``expert_lora``
    is the adapter-gather flag (``None`` = do not capture adapters); the capture has to happen here
    because a resumed run takes a step of its own before the body can look. A subclass adds its own
    entries through :meth:`extra`.
    """

    def __init__(self, event: str, trainer, *, capture_optimizer: bool, expert_lora: bool | None = None):
        self.event = event
        self.trainer = trainer
        self.capture_optimizer = capture_optimizer
        self.expert_lora = expert_lora
        self.captured: dict | None = None

    def extra(self) -> dict:
        """Extra entries for the snapshot, taken at the same point. Empty here."""
        return {}

    def _capture(self, state) -> None:
        if self.captured is not None:
            return
        # Every reader below goes by parameter identity, and a hook can fire while the FSDP2 modules
        # still hold the transient unsharded params an eval-only forward left registered.
        reshard_fsdp2_modules(unwrap(self.trainer.model))
        self.captured = {
            "global_step": state.global_step,
            "sched_last_epoch": self.trainer.lr_scheduler.last_epoch,
            "optimizer": local_optimizer_state(self.trainer.model, self.trainer.optimizer)
            if self.capture_optimizer
            else None,
            "adapters": snapshot_adapters(unwrap(self.trainer.model), expert_lora=self.expert_lora)
            if self.expert_lora is not None
            else None,
            **self.extra(),
        }

    def on_save(self, args, state, control, **kwargs):
        if self.event == "save":
            self._capture(state)

    def on_train_begin(self, args, state, control, **kwargs):
        if self.event == "train_begin":
            self._capture(state)


def logged_lrs(trainer) -> dict[int, float]:
    """Per-step learning rates from this ``train()`` call's log history."""
    return {int(e["step"]): float(e["learning_rate"]) for e in trainer.state.log_history if "learning_rate" in e}


def adapter_file_agreement(restored: dict[str, torch.Tensor], checkpoint: str) -> tuple[int, int, int]:
    """``(matching, compared, non-zero B)`` for restored adapters against the checkpoint's own file.

    The saved file is the resume's source of truth; comparing against a pre-save in-memory copy
    would conflate the save's cast with the restore. PEFT drops its active-adapter infix on the way
    to disk, so the attention keys are matched with it stripped.
    """
    path = os.path.join(checkpoint, "adapter_model.safetensors")
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
    # ANTI-VACUITY for the line below: a scheduler restarted from zero only lands elsewhere while
    # the schedule actually varies over these steps.
    checks["resume_schedule_varies_across_steps"] = phase1_lrs.get(RESUME_SAVE_STEP) != phase1_lrs.get(
        RESUME_MAX_STEPS
    )
    checks["resume_continued_the_schedule"] = phase2_lrs.get(RESUME_MAX_STEPS) == phase1_lrs.get(RESUME_MAX_STEPS)
    log(f"  learning rates: phase1={phase1_lrs} phase2={phase2_lrs}")


def record_scheduler_restore(checks: dict[str, bool], restored: dict | None) -> None:
    """The LR scheduler's own counter came back at the step the checkpoint was written on."""
    checks["resume_restored_the_scheduler"] = restored is not None and restored["sched_last_epoch"] == RESUME_SAVE_STEP


def record_optimizer_restore(checks: dict[str, bool], saved: dict, restored: dict) -> None:
    """Optimizer state, bit-exact across the resume. Both snapshots must have captured it."""
    ok, detail = optimizer_state_matches(saved["optimizer"], restored["optimizer"])
    checks["resume_restored_the_optimizer_state"] = ok
    if not ok:
        log(f"  optimizer state was NOT continued: {detail}")
