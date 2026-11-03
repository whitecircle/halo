"""``gradient_checkpointing_plan: auto``: the three-phase prediction, policy switch, refusals, consensus and record.
Exact live bytes come from a ``TorchDispatchMode`` ledger, the CPU stand-in for ``torch.cuda.memory_allocated``."""

import dataclasses
import functools
import gc
import json
import os
import sys
import weakref
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from datasets import Dataset, DatasetDict
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map
from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM
from transformers.modeling_layers import GradientCheckpointingLayer
from trl import SFTConfig

from halo.callbacks import gradient_checkpointing_plan as plan_callback
from halo.callbacks.gradient_checkpointing_plan import (
    GradientCheckpointingPlanCallback,
    MemoryReader,
    StepMemoryProbe,
    agree_plan,
    build_gradient_checkpointing_plan_callback,
    build_plan_context,
    build_plan_fingerprint,
    restore_recompute_floor,
    save_run_plan,
    step_memory_across_ranks,
)
from halo.checkpoint.format import RUN_PLAN_FILE
from halo.distributed.parallelism_config import ParallelismConfig
from halo.entrypoints import script as script_entry
from halo.entrypoints.script import ScriptContext, configure_balancing_and_build_callbacks
from halo.models.gradient_checkpointing import enable_scoped_gradient_checkpointing
from halo.models.gradient_checkpointing_plan import (
    CHECKPOINTING_OFF,
    DENSE_BUDGET_FRACTION,
    GradientCheckpointingPlan,
    PlanContext,
    StepMemory,
    apply_checkpointing_policy,
    candidate_policies,
    checkpointed_layer_count,
    plan_budget_bytes,
    plan_layers,
    predict_peak,
    recompute_floor_index,
    refusal_reasons,
    reject_unplannable_run,
    resolve_policy,
)
from halo.models.moe.router_aux_gradient import install_router_aux_gradient
from tests.common.gloo import run_gloo_ranks
from tests.common.repo import load_script_module
from tests.common.tiny_models import (
    TINY_BAILING_LING3_MLA_FAMILY,
    TINY_BAILING_RING_FAMILY,
    TINY_DENSE_ATTENTION_FAMILIES,
    TINY_DENSE_FAMILIES,
    TINY_MOE_FAMILIES,
    TINY_MOE_VLM_FAMILIES,
    TINY_QWEN3_VL_FAMILY,
    TINY_SLIDING_DENSE_FAMILIES,
    TINY_UNWRAPPED_MOE_FAMILY,
    tiny_family_model,
)

LAYERS = 8
WORLD = 4
VOCAB = 4096
SEQ = 512
FINGERPRINT = {"model": "tiny", "max_length": SEQ, "world_size": WORLD}
# Above any tiny step's peak: the allocator peak a transient allocation leaves before the measured step.
SPIKE_BYTES = 256 * 2**20
# The Bailing remote code checkpoints at the model level, so no plan applies.
ROSTER = {
    **{f"moe/{name}": family for name, family in TINY_MOE_FAMILIES.items()},
    **{f"vlm/{name}": family for name, family in TINY_MOE_VLM_FAMILIES.items()},
    **{f"dense/{name}": family for name, family in TINY_DENSE_FAMILIES.items()},
    **{f"sliding/{name}": family for name, family in TINY_SLIDING_DENSE_FAMILIES.items()},
    **{f"attn/{name}": family for name, family in TINY_DENSE_ATTENTION_FAMILIES.items()},
    "moe/mixtral": TINY_UNWRAPPED_MOE_FAMILY,
    "moe/bailing_ring": TINY_BAILING_RING_FAMILY,
    "moe/bailing_ling3_mla": TINY_BAILING_LING3_MLA_FAMILY,
    "vlm/qwen3_vl": TINY_QWEN3_VL_FAMILY,
}
MODEL_LEVEL_CHECKPOINTING = {"moe/bailing_moe", "moe/bailing_ring", "moe/bailing_ling3_mla"}


class _LiveBytes(TorchDispatchMode):
    """Exact live bytes of every op output (released by storage weakref)."""

    def __init__(self):
        super().__init__()
        self.live, self.cur, self.peak = {}, 0, 0

    def _add(self, t):
        if isinstance(t, torch.Tensor):
            storage = t.untyped_storage()
            key = storage.data_ptr()
            if key not in self.live and storage.nbytes():
                self.live[key] = storage.nbytes()
                self.cur += storage.nbytes()
                self.peak = max(self.peak, self.cur)
                weakref.finalize(storage, self._free, key)

    def _free(self, key):
        self.cur -= self.live.pop(key, 0)

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        tree_map(self._add, out)
        return out

    def reset_peak(self):
        self.peak = self.cur


def _reader(tracker: _LiveBytes) -> MemoryReader:
    # The ledger has no caching allocator, so reserved equals allocated.
    return MemoryReader(lambda: tracker.cur, lambda: tracker.peak, lambda: tracker.peak, tracker.reset_peak, lambda: 0)


def _tiny(kind: str, layers: int = LAYERS):
    torch.manual_seed(0)
    common = {
        "vocab_size": VOCAB,
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_hidden_layers": layers,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 32,
        "max_position_embeddings": 2048,
        "tie_word_embeddings": False,
        "attn_implementation": "sdpa",
    }
    if kind == "dense":
        return Qwen3ForCausalLM(Qwen3Config(**common)).train()
    aux = {"router_aux_loss_coef": 0.01, "output_router_logits": True} if kind == "moe_aux" else {}
    return Qwen3MoeForCausalLM(
        Qwen3MoeConfig(**common, moe_intermediate_size=64, num_experts=8, num_experts_per_tok=2, **aux)
    ).train()


def _enabled(kind: str, aux_gradient: bool = True):
    """Reentrant where Halo forces it (every MoE), HF's non-reentrant default otherwise. The aux-loss row carries the
    router aux gradient through the reentrant checkpoints, as balancing installs it."""
    model = _tiny(kind)
    reentrant = kind != "dense"
    enable_scoped_gradient_checkpointing(model, gradient_checkpointing_kwargs={"use_reentrant": reentrant})
    if reentrant:
        model.enable_input_require_grads()
    if kind == "moe_aux" and aux_gradient:
        assert install_router_aux_gradient(model) > 0
    return model


def _grads(model):
    model.zero_grad(set_to_none=True)
    ids = torch.randint(0, VOCAB, (1, SEQ), generator=torch.Generator().manual_seed(1))
    loss = model(input_ids=ids, labels=ids).loss
    loss.backward()
    return loss.item(), {n: p.grad.clone() for n, p in model.named_parameters()}


def _optimizer(model):
    return torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=True)


def _train_step(
    model,
    optimizer,
    tracker: _LiveBytes,
    before_optimizer=None,
    *,
    seed: int = 1,
    batch: int = 1,
    seq: int = SEQ,
    accumulation: int = 1,
    reset_peak: bool = True,
) -> int:
    """One optimizer step in the Trainer's order (``before_optimizer`` where ``on_pre_optimizer_step`` fires)."""
    if reset_peak:
        tracker.reset_peak()
    with tracker:
        for micro in range(accumulation):
            generator = torch.Generator().manual_seed(seed * accumulation + micro)
            ids = torch.randint(0, VOCAB, (batch, seq), generator=generator)
            (model(input_ids=ids, labels=ids).loss / accumulation).backward()
        if before_optimizer is not None:
            before_optimizer()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    return tracker.peak


def _measured(kind: str, tracker: _LiveBytes, batch: int = 1, seq: int = SEQ, accumulation: int = 1):
    """Reads the second step, since the first allocates the optimizer's moments. A transient allocation before the
    probe sets the ledger's peak above the step's, and the measured step leaves it to the probe, as production does."""
    model = _enabled(kind)
    optimizer = _optimizer(model)
    _train_step(model, optimizer, tracker, batch=batch, seq=seq, accumulation=accumulation)
    with tracker:
        torch.empty(SPIKE_BYTES, dtype=torch.uint8).fill_(0)
    probe = StepMemoryProbe(model, _reader(tracker))
    peak = _train_step(
        model,
        optimizer,
        tracker,
        probe.mark_backward_end,
        batch=batch,
        seq=seq,
        accumulation=accumulation,
        reset_peak=False,
    )
    reading = probe.close()
    assert reading.peak == peak < SPIKE_BYTES
    return model, optimizer, reading


@pytest.mark.parametrize("kind", ["dense", "moe", "moe_aux"])
def test_one_checkpointed_step_predicts_every_relaxed_policy_and_numerics_do_not_move(kind):
    tracker = _LiveBytes()
    model, optimizer, reading = _measured(kind, tracker)
    # The measuring step's early-stop-off recompute moves no peak.
    assert _train_step(model, optimizer, tracker) == reading.peak
    for k in (CHECKPOINTING_OFF, 4, 2):
        apply_checkpointing_policy(model, k)
        measured = _train_step(model, optimizer, tracker)
        predicted = predict_peak(reading, LAYERS, k)
        assert measured <= predicted <= 1.005 * measured, (k, predicted, measured)
    # A switch that never reached the layers would read as full checkpointing.
    apply_checkpointing_policy(model, 1)
    assert _train_step(model, optimizer, tracker) < 0.8 * predict_peak(reading, LAYERS, CHECKPOINTING_OFF)
    # Every policy replays the same forward, bitwise against a fully checkpointed copy.
    reference = _enabled(kind)
    reference.load_state_dict(model.state_dict())
    loss_full, grads_full = _grads(reference)
    for k in (CHECKPOINTING_OFF, 4, 2):
        apply_checkpointing_policy(model, k)
        loss, grads = _grads(model)
        assert loss == loss_full and all(torch.equal(grads[n], grads_full[n]) for n in grads_full), k


@pytest.mark.parametrize("kind", ["dense", "moe"])
def test_the_prediction_holds_under_gradient_accumulation(kind):
    """Each micro-step's backward starts on the gradients it accumulated, so the head phase is read at every
    micro-step's first recompute."""
    tracker = _LiveBytes()
    model, optimizer, reading = _measured(kind, tracker, accumulation=2)
    for k in (CHECKPOINTING_OFF, 4, 2):
        apply_checkpointing_policy(model, k)
        measured = _train_step(model, optimizer, tracker, accumulation=2)
        predicted = predict_peak(reading, LAYERS, k)
        assert measured <= predicted <= 1.005 * measured, (k, predicted, measured)


def test_the_backward_peak_keeps_the_optimizer_phase_out_of_the_relaxed_prediction():
    """Priced from the whole step's peak, the backward phase over-reads a relaxed policy once the optimizer phase is
    the fully checkpointed step's peak (short rows, few activations)."""
    tracker = _LiveBytes()
    model, optimizer, reading = _measured("dense", tracker, seq=SEQ // 4)
    assert reading.peak > reading.backward_peak
    unsplit = dataclasses.replace(reading, backward_peak=reading.peak)
    for k in (CHECKPOINTING_OFF, 2):
        apply_checkpointing_policy(model, k)
        measured = _train_step(model, optimizer, tracker, seq=SEQ // 4)
        assert measured <= predict_peak(reading, LAYERS, k) <= 1.005 * measured, k
        assert predict_peak(unsplit, LAYERS, k) > 1.02 * measured, k


def test_aux_loss_row_needs_the_router_gradient_hand_off_to_stay_bitwise():
    """Without the hand-off, full checkpointing loses the aux gradient the off policy keeps."""
    model = _enabled("moe_aux", aux_gradient=False)
    _, grads_full = _grads(model)
    apply_checkpointing_policy(model, CHECKPOINTING_OFF)
    _, grads_off = _grads(model)
    router = [n for n in grads_full if n.endswith("gate.weight")]
    assert router and not all(torch.equal(grads_full[n], grads_off[n]) for n in router)


def test_policy_switch_keeps_the_scoped_checkpoint_function_hf_re_enable_drops():
    model = _enabled("moe")
    assert apply_checkpointing_policy(model, 2) == LAYERS // 2 == checkpointed_layer_count(LAYERS, 2)
    layers = [m for m in model.modules() if isinstance(m, GradientCheckpointingLayer)]
    assert [layer.gradient_checkpointing for layer in layers] == [i % 2 == 0 for i in range(LAYERS)]
    assert all(getattr(layer._gradient_checkpointing_func, "_ep_scoped", False) for layer in layers)
    # HF's own disable installs the bare torch checkpoint (the EP dispatch replay would be gone).
    model.gradient_checkpointing_disable()
    assert not any(getattr(layer._gradient_checkpointing_func, "_ep_scoped", False) for layer in layers)


def test_a_layer_count_no_policy_divides_prices_what_the_switch_checkpoints():
    layers = 9
    model = _tiny("dense", layers=layers)
    for k in candidate_policies(layers):
        assert apply_checkpointing_policy(model, k) == checkpointed_layer_count(layers, k), k


@pytest.mark.parametrize("name", sorted(ROSTER))
def test_the_plan_counts_and_sets_the_text_decoder_layers_on_every_family(name):
    """A multimodal checkpoint's vision blocks are GradientCheckpointingLayer too and come first in ``modules()``.
    Counted with the decoder layers they would be priced as text layers and take the ratchet's first step."""
    torch.manual_seed(0)
    model = tiny_family_model(ROSTER[name])
    layers = plan_layers(model)
    if name in MODEL_LEVEL_CHECKPOINTING:
        assert layers == []
        return
    text_layers = model.config.get_text_config().num_hidden_layers
    assert len(layers) == text_layers, [type(layer).__name__ for layer in layers]
    model.gradient_checkpointing_enable()
    others = [
        m for m in model.modules() if isinstance(m, GradientCheckpointingLayer) and not any(m is x for x in layers)
    ]
    for k in candidate_policies(text_layers):
        assert apply_checkpointing_policy(model, k) == checkpointed_layer_count(text_layers, k), k
        assert [layer.gradient_checkpointing for layer in layers] == [
            k != CHECKPOINTING_OFF and i % k == 0 for i in range(text_layers)
        ], k
        assert all(m.gradient_checkpointing for m in others)


def _memory(**fields) -> StepMemory:
    base = {
        "peak": 100,
        "head_phase_peak": 90,
        "backward_peak": 100,
        "layer_recompute_bytes": 11,
        "layer_input_bytes": 1,
        "reserved_peak": 100,
        "non_torch_bytes": 0,
    }
    return StepMemory(**{**base, **fields})


def test_resolve_policy_takes_the_least_recompute_that_fits_a_reserved_and_non_torch_net_budget():
    step = _memory()
    assert candidate_policies(LAYERS) == [CHECKPOINTING_OFF, 8, 4, 3, 2, 1]
    # Past off every policy checkpoints at least an eighth of the layers.
    assert candidate_policies(40) == [CHECKPOINTING_OFF, 8, 4, 3, 2, 1]
    assert candidate_policies(3) == [CHECKPOINTING_OFF, 3, 2, 1]
    assert candidate_policies(1) == [CHECKPOINTING_OFF, 1]
    # Every step checkpoints more layers than the last; at 4, 5 and 9 layers two policies would checkpoint as many.
    for layers, policies in {4: [4, 3], 5: [4, 2], 6: [4, 2], 9: [8, 4, 2]}.items():
        assert candidate_policies(layers) == [CHECKPOINTING_OFF, *policies, 1], layers
    for layers in range(1, 65):
        counts = [checkpointed_layer_count(layers, k) for k in candidate_policies(layers)]
        assert counts == sorted(set(counts)) and counts[-1] == layers and counts[1] >= layers / 8, layers
    assert resolve_policy(step, LAYERS, 10**9) == (CHECKPOINTING_OFF, predict_peak(step, LAYERS, CHECKPOINTING_OFF))
    # off adds 8 * 10 to the head phase (170); every 4 checkpoints 2 layers: +60 -> head 150, backward 160.
    assert resolve_policy(step, LAYERS, 160) == (4, 160)
    assert resolve_policy(step, LAYERS, 99)[0] == 1  # nothing lighter fits
    # A floor bounds the choice from below and never relaxes what the budget needs.
    assert resolve_policy(step, LAYERS, 10**9, floor_index=1)[0] == 8
    assert resolve_policy(step, LAYERS, 160, floor_index=1)[0] == 4
    assert resolve_policy(step, LAYERS, 10**9, floor_index=5)[0] == 1
    # The optimizer phase is the same under every policy, so it floors the prediction.
    assert predict_peak(_memory(peak=300), LAYERS, CHECKPOINTING_OFF) == 300
    assert plan_budget_bytes(step, 1000, ep_group_size=1) == 850
    assert plan_budget_bytes(step, 1000, ep_group_size=8) == 750
    assert plan_budget_bytes(_memory(non_torch_bytes=100), 1000, ep_group_size=1) == 750
    # 25% fragmentation at the measured step: the device holds reserved bytes, so the allocated budget shrinks.
    assert plan_budget_bytes(_memory(reserved_peak=125), 1000, ep_group_size=1) == 680


_CLEAN = PlanContext(
    num_layers=LAYERS,
    gradient_checkpointing=True,
    every_n_layers_configured=False,
    offload=False,
    pp_size=1,
    torch_compile=False,
    trains_on_generated_rollouts=False,
    auto_find_batch_size=False,
    packed_rows=True,
    multimodal_rows=False,
    ep_group_size=1,
    fingerprint=FINGERPRINT,
)


@pytest.mark.parametrize(
    "change",
    [
        {"gradient_checkpointing": False},
        {"num_layers": 0},
        {"every_n_layers_configured": True},
        {"offload": True},
        {"pp_size": 2},
        {"torch_compile": True},
        {"trains_on_generated_rollouts": True},
        {"auto_find_batch_size": True},
        {"packed_rows": False},
        {"multimodal_rows": True},
    ],
)
def test_each_refusal_fires_alone(change):
    """The refusal names the setting that runs without a plan, since a reason can be a property of the user's run."""
    assert refusal_reasons(_CLEAN) == []
    reject_unplannable_run(_CLEAN)
    refused = dataclasses.replace(_CLEAN, **change)
    assert len(refusal_reasons(refused)) == 1
    with pytest.raises(ValueError, match="refused for this run") as refusal:
        reject_unplannable_run(refused)
    assert "Set gradient_checkpointing_plan: off" in str(refusal.value)


def _consensus_worker(rank: int, out_dir: str):
    # Routing skew: each rank's hot-rank receive count, hence its readings, differs.
    local = _memory(
        peak=2600 + 150 * rank,
        head_phase_peak=2500 + 150 * rank,
        backward_peak=2600 + 150 * rank,
        layer_recompute_bytes=60 + 10 * rank,
        layer_input_bytes=8 - rank % 2,
        reserved_peak=2600 + 150 * rank,
        non_torch_bytes=30 * rank,
    )
    naive = resolve_policy(local, LAYERS, plan_budget_bytes(local, 4800, 8))[0]
    world = step_memory_across_ranks(local)
    every_n = resolve_policy(world, LAYERS, plan_budget_bytes(world, 4800, 8))[0]
    plan = GradientCheckpointingPlan(every_n_layers=every_n, fingerprint=FINGERPRINT)
    agree_plan(plan)  # passes: the inputs were world-reduced first
    try:
        agree_plan(dataclasses.replace(plan, every_n_layers=1) if rank == 3 else plan)
        diverged = None
    except ValueError as error:
        diverged = str(error)
    with open(os.path.join(out_dir, f"rank{rank}.json"), "w") as f:
        json.dump({"naive": naive, "world": dataclasses.asdict(world), "plan": every_n, "diverged": diverged}, f)


def test_world_reduced_readings_give_every_rank_one_plan_and_a_divergence_raises_everywhere(tmp_path):
    run_gloo_ranks(_consensus_worker, WORLD, str(tmp_path))
    results = [json.loads((tmp_path / f"rank{r}.json").read_text()) for r in range(WORLD)]
    assert len({r["plan"] for r in results}) == 1
    assert len({r["naive"] for r in results}) > 1  # without the reduction the ranks would disagree
    assert results[0]["world"]["peak"] == 2600 + 150 * (WORLD - 1)  # the hottest rank governs
    assert results[0]["world"]["layer_input_bytes"] == 7  # and the smallest checkpoint input
    assert results[0]["world"]["reserved_peak"] == 2600 + 150 * (WORLD - 1)
    # The most memory outside any rank's allocator: every 4 fits the min-reduced budget, not this one.
    assert results[0]["world"]["non_torch_bytes"] == 30 * (WORLD - 1)
    assert results[0]["plan"] == 3  # relaxed, not the full-checkpointing fallback
    # It names the field and the bypass that works, since an explicit every_n_layers is itself refused under auto.
    assert all(r["diverged"] and "every_n_layers" in r["diverged"] for r in results)
    assert all("gradient_checkpointing_plan: off" in r["diverged"] for r in results)


def _ratio_worker(rank: int, out_dir: str):
    # Rank 0 is the hottest; rank 1 allocates less but reserves the most per allocated byte (90 against 120).
    peak = 100 - 10 * rank
    local = _memory(peak=peak, head_phase_peak=peak, backward_peak=peak, reserved_peak=100 + 20 * rank)
    world = step_memory_across_ranks(local)
    with open(os.path.join(out_dir, f"rank{rank}.json"), "w") as f:
        json.dump(
            {"peak": world.peak, "reserved": world.reserved_peak, "budget": plan_budget_bytes(world, 1000, 1)}, f
        )


def test_the_budget_takes_the_reserve_ratio_of_the_most_fragmented_rank(tmp_path):
    """The world view prices the hottest rank's peak at the largest reserved-to-allocated ratio any rank measured,
    since each device holds its own reserve."""
    run_gloo_ranks(_ratio_worker, 2, str(tmp_path))
    results = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)]
    assert all(r == {"peak": 100, "reserved": 134, "budget": 850 * 100 // 134} for r in results)
    # At this budget the fragmented rank's reserve fits its 850-byte share; at max(peak) / max(reserved) it would not.
    assert results[0]["budget"] * 120 / 90 <= 850 < plan_budget_bytes(_memory(reserved_peak=120), 1000, 1) * 120 / 90


def test_an_unfinished_plan_floors_the_next_launch_one_step_tighter_and_a_finished_one_sets_none():
    policies = candidate_policies(LAYERS)
    assert recompute_floor_index(None, FINGERPRINT, LAYERS) == 0
    for index, every_n in enumerate(policies):
        unfinished = GradientCheckpointingPlan(every_n_layers=every_n, fingerprint=FINGERPRINT)
        assert recompute_floor_index(unfinished, FINGERPRINT, LAYERS) == min(index + 1, len(policies) - 1)
        # A finished plan is no OOM evidence, and a record made for other inputs prices another footprint.
        assert recompute_floor_index(dataclasses.replace(unfinished, finished=True), FINGERPRINT, LAYERS) == 0
        assert recompute_floor_index(unfinished, {**FINGERPRINT, "max_length": 4096}, LAYERS) == 0
    with pytest.raises(ValueError, match="every_n_layers=5"):
        recompute_floor_index(
            GradientCheckpointingPlan(every_n_layers=5, fingerprint=FINGERPRINT), FINGERPRINT, LAYERS
        )


class _Args:
    def __init__(self, output_dir):
        self.output_dir = output_dir


class _State:
    def __init__(self, global_step: int = 0):
        self.global_step = global_step


def _drive(callback, kind: str, tracker, output_dir: str, steps: int, *, state=None, batch=1, seq=SEQ, finish=True):
    """One launch through the callback's hooks in the Trainer's order, with train end only when the run ``finish``es.
    The model dies with the call because the ledger counts the weights a forward views, and an earlier launch's would
    sit under this one's peaks."""
    gc.collect()
    model = _enabled(kind)
    optimizer = _optimizer(model)
    args, state = _Args(output_dir), state or _State()
    callback.on_train_begin(args, state, None, model=model)
    trace = []
    for _ in range(steps):
        callback.on_step_begin(args, state, None, model=model)
        before_optimizer = functools.partial(callback.on_pre_optimizer_step, args, state, None)
        peak = _train_step(model, optimizer, tracker, before_optimizer, seed=state.global_step, batch=batch, seq=seq)
        state.global_step += 1
        callback.on_step_end(args, state, None, model=model)
        trace.append((peak, callback.plan.every_n_layers if callback.plan else None))
    if finish:
        callback.on_train_end(args, state, None)
    return trace, [m.gradient_checkpointing for m in model.modules() if isinstance(m, GradientCheckpointingLayer)]


def _launch(tracker, output_dir: str, *, finish: bool, total_device_bytes: int = 2**40, steps: int = 3) -> int:
    callback = GradientCheckpointingPlanCallback(
        _CLEAN,
        total_device_bytes=total_device_bytes,
        floor_index=restore_recompute_floor(output_dir, FINGERPRINT, LAYERS),
        reader=_reader(tracker),
    )
    _, flags = _drive(callback, "dense", tracker, output_dir, steps, finish=finish)
    every_n = callback.plan.every_n_layers
    assert flags == [every_n != CHECKPOINTING_OFF and i % every_n == 0 for i in range(LAYERS)]  # it reached the layers
    return every_n


def test_restarts_after_unfinished_runs_tighten_to_full_and_a_finished_run_lifts_the_floor(tmp_path):
    """The OOM-loop guard: each launch after an unfinished plan checkpoints at least one step more, down to full."""
    tracker, out = _LiveBytes(), str(tmp_path)
    plans = [_launch(tracker, out, finish=False) for _ in range(7)]  # roomy: each measures off, then it crashes
    assert plans == [CHECKPOINTING_OFF, 8, 4, 3, 2, 1, 1]
    assert _launch(tracker, out, finish=True) == 1
    assert json.loads((tmp_path / RUN_PLAN_FILE).read_text())["finished"]
    assert _launch(tracker, out, finish=False) == CHECKPOINTING_OFF
    # Negative control: without the record every launch replays the plan that crashed.
    replays = []
    for _ in range(3):
        os.remove(tmp_path / RUN_PLAN_FILE)
        replays.append(_launch(tracker, out, finish=False))
    assert replays == [CHECKPOINTING_OFF] * 3


def test_a_floor_never_relaxes_the_plan_the_launch_measures(tmp_path):
    """A relaunch whose own step needs more recompute than the floor gets what its budget needs, and the floor still
    rises one step over a tighter unfinished plan. The fingerprint does not hold the optimizer, adapters or kernels."""
    reading = _measured("dense", _LiveBytes())[2]
    # Between the every-4 and every-8 predictions, so every 4 is what this launch's step fits.
    target = (predict_peak(reading, LAYERS, 4) + predict_peak(reading, LAYERS, 8)) // 2
    tight = int(target / DENSE_BUDGET_FRACTION)
    tracker, out = _LiveBytes(), str(tmp_path)
    save_run_plan(out, GradientCheckpointingPlan(every_n_layers=CHECKPOINTING_OFF, fingerprint=FINGERPRINT))
    assert _launch(tracker, out, finish=False, total_device_bytes=tight) == 4  # not the floor's every 8
    assert _launch(tracker, out, finish=False, total_device_bytes=tight) == 3  # every 4 did not finish
    # The same budget with no record measures every 4.
    os.remove(tmp_path / RUN_PLAN_FILE)
    assert _launch(tracker, out, finish=False, total_device_bytes=tight) == 4


def _record_worker(rank: int, out_dir: str):
    outcome = {}
    writers = []
    staged_file = plan_callback.staged_file

    def counting_staged_file(*args, **kwargs):
        writers.append(args[1])
        return staged_file(*args, **kwargs)

    shared = os.path.join(out_dir, "shared")
    unfinished = GradientCheckpointingPlan(every_n_layers=CHECKPOINTING_OFF, fingerprint=FINGERPRINT)
    with mock.patch.object(plan_callback, "staged_file", counting_staged_file):
        save_run_plan(shared, unfinished)
    outcome["writes"] = len(writers)
    outcome["floor"] = restore_recompute_floor(shared, FINGERPRINT, LAYERS)
    save_run_plan(shared, dataclasses.replace(unfinished, finished=True))
    outcome["finished"] = restore_recompute_floor(shared, FINGERPRINT, LAYERS)
    save_run_plan(shared, unfinished)
    outcome["other_inputs"] = restore_recompute_floor(shared, {**FINGERPRINT, "max_length": 4096}, LAYERS)
    outcome["absent"] = restore_recompute_floor(os.path.join(out_dir, "empty"), FINGERPRINT, LAYERS)
    # A record another version wrote sets no floor, and is not refused.
    other_version = os.path.join(out_dir, "other_version")
    if rank == 0:
        os.makedirs(other_version)
        with open(os.path.join(other_version, RUN_PLAN_FILE), "w") as f:
            json.dump({**dataclasses.asdict(unfinished), "version": 0, "field_of_another_version": 1}, f)
    torch.distributed.barrier()
    outcome["other_version"] = restore_recompute_floor(other_version, FINGERPRINT, LAYERS)
    # A per-node output filesystem where only node 0 kept an unfinished record: every rank takes its floor.
    node = os.path.join(out_dir, f"node{rank}")
    os.makedirs(node, exist_ok=True)
    if rank == 0:
        with open(os.path.join(node, RUN_PLAN_FILE), "w") as f:
            json.dump(dataclasses.asdict(GradientCheckpointingPlan(every_n_layers=4, fingerprint=FINGERPRINT)), f)
    outcome["torn"] = restore_recompute_floor(node, FINGERPRINT, LAYERS)
    # A copy unreadable on one node raises on every rank instead of stranding the others in the reduce.
    if rank == 2:
        with open(os.path.join(node, RUN_PLAN_FILE), "w") as f:
            f.write("{not json")
    try:
        restore_recompute_floor(node, FINGERPRINT, LAYERS)
        outcome["unreadable"] = None
    except ValueError as error:
        outcome["unreadable"] = str(error)
    with open(os.path.join(out_dir, f"rank{rank}.json"), "w") as f:
        json.dump(outcome, f)


def test_record_is_written_by_the_save_rank_read_on_every_rank_and_torn_copies_resolve_or_raise_everywhere(tmp_path):
    run_gloo_ranks(_record_worker, WORLD, str(tmp_path))
    assert sorted(os.listdir(tmp_path / "shared")) == [RUN_PLAN_FILE]  # no stage left
    policies = candidate_policies(LAYERS)
    for rank in range(WORLD):
        outcome = json.loads((tmp_path / f"rank{rank}.json").read_text())
        # The shared filesystem's one save rank writes; every other rank only joins the barrier.
        assert outcome["writes"] == (1 if rank == 0 else 0), rank
        assert outcome["floor"] == policies.index(8) and outcome["finished"] == 0
        assert outcome["other_inputs"] == 0 and outcome["absent"] == 0 and outcome["other_version"] == 0
        assert outcome["torn"] == policies.index(3)  # node 0's every 4, one step tighter
        assert outcome["unreadable"] and "failed on 1 of 4 rank(s) [2]" in outcome["unreadable"]


class _PlainTrainer:
    SYNCS_TO_EXTERNAL_GENERATOR = False
    POLICY_GRADIENT_LOSS = False


class _RolloutTrainer:
    SYNCS_TO_EXTERNAL_GENERATOR = True
    POLICY_GRADIENT_LOSS = True


def test_run_context_refuses_what_the_config_and_trainer_rule_out(tmp_path):
    model = _enabled("moe")
    sft = functools.partial(
        SFTConfig, output_dir=str(tmp_path), bf16=False, use_cpu=True, gradient_checkpointing=True, max_length=SEQ
    )
    build = functools.partial(
        build_plan_context,
        model=model,
        parallelism_config=ParallelismConfig(),
        trainer_cls=_PlainTrainer,
        packed_rows=True,
        multimodal_rows=False,
    )
    clean = build(sft(packing=True))
    assert refusal_reasons(clean) == [] and clean.num_layers == LAYERS
    # The rows the collator builds decide: here a corpus pre-processed packed under packing: false.
    assert refusal_reasons(build(sft(packing=False))) == []
    fingerprint = build_plan_fingerprint(
        sft(packing=True), model, ParallelismConfig(), num_layers=clean.num_layers, total_device_bytes=2**30
    )
    assert fingerprint["max_length"] == SEQ and fingerprint["num_layers"] == LAYERS
    assert fingerprint["model_type"] == "qwen3_moe" and fingerprint["total_device_bytes"] == 2**30
    refused = {
        # packing: true over a corpus pre-processed unpacked trains padded rows of varying length.
        "does not pack": build(sft(packing=True), packed_rows=False),
        "generation": build(sft(packing=True), trainer_cls=_RolloutTrainer),
        "explicit": build(sft(packing=True, gradient_checkpointing_kwargs={"every_n_layers": 2})),
        "offload": build(sft(packing=True, gradient_checkpointing_kwargs={"offload": True})),
        "multimodal": build(sft(packing=True), multimodal_rows=True),
        "is off": build(sft(packing=True, gradient_checkpointing=False)),
        "torch_compile": build(sft(packing=True, torch_compile=True)),
        "auto_find_batch_size": build(sft(packing=True, auto_find_batch_size=True)),
        "pipeline": build(
            sft(packing=True), parallelism_config=ParallelismConfig(pp_size=2, world_size=4, gpus_per_node=2)
        ),
    }
    for word, context in refused.items():
        reasons = refusal_reasons(context)
        assert len(reasons) == 1 and word in reasons[0], (word, reasons)


def _callback_worker(rank: int, out_dir: str, budget_case: str):
    torch.set_num_threads(2)
    batch = 1 + rank  # rank 1 runs twice the rows, the hot rank of a skewed EP group
    # The world reading the callback's own second step takes: the same model, rows and seeds.
    world = step_memory_across_ranks(_measured("moe", _LiveBytes(), batch=batch)[2])
    if budget_case == "roomy":
        total_device_bytes = 2**40
    else:
        # Between the world's every-4 and every-8 predictions, so every 4 is the one answer that fits.
        target = (predict_peak(world, LAYERS, 4) + predict_peak(world, LAYERS, 8)) // 2
        total_device_bytes = int(target / 0.75)
    context = dataclasses.replace(_CLEAN, ep_group_size=8)
    tracker = _LiveBytes()

    def callback(floor_index=0):
        return GradientCheckpointingPlanCallback(
            context, total_device_bytes=total_device_bytes, floor_index=floor_index, reader=_reader(tracker)
        )

    measured = callback()
    trace, _ = _drive(measured, "moe", tracker, out_dir, 4, batch=batch)
    restarted = callback(restore_recompute_floor(out_dir, FINGERPRINT, LAYERS))
    restarted_trace, _ = _drive(restarted, "moe", tracker, out_dir, 3, batch=batch)
    # A launch with no record measures at its own second step, wherever global_step stands.
    late_trace, _ = _drive(
        callback(), "moe", tracker, os.path.join(out_dir, "late"), 2, state=_State(100), batch=batch
    )
    plan = measured.plan.every_n_layers
    with open(os.path.join(out_dir, f"rank{rank}.json"), "w") as f:
        json.dump(
            {
                "plan": plan,
                "restarted": restarted.plan.every_n_layers,
                "restarted_floor": restarted.floor_index,
                "late": [every_n for _peak, every_n in late_trace],
                "predicted": predict_peak(world, LAYERS, plan),
                "budget": plan_budget_bytes(world, total_device_bytes, 8),
                "peaks": [peak for peak, _every_n in trace],
                "restarted_peaks": [peak for peak, _every_n in restarted_trace],
            },
            f,
        )


@pytest.mark.parametrize("budget_case", ["roomy", "tight"])
def test_callback_measures_step_two_relaxes_every_rank_alike_and_a_relaunch_after_it_finished_measures_the_same(
    tmp_path, budget_case
):
    run_gloo_ranks(_callback_worker, 2, str(tmp_path), budget_case)
    results = [json.loads((tmp_path / f"rank{r}.json").read_text()) for r in range(2)]
    print(json.dumps({"case": budget_case, "ranks": results}))
    plan = results[0]["plan"]
    assert plan == (CHECKPOINTING_OFF if budget_case == "roomy" else 4)
    assert all(r["plan"] == plan and r["restarted"] == plan and r["restarted_floor"] == 0 for r in results)
    assert all(r["late"][0] is None and r["late"][1] in candidate_policies(LAYERS) for r in results)
    hot = results[1]
    # Steps 1-2 fully checkpointed, steps 3-4 at the plan. The hot rank stays under the world prediction.
    assert hot["peaks"][1] < hot["peaks"][2] <= hot["predicted"] <= hot["budget"]
    # The world view over-reads the hot rank only by the min-reduced checkpoint input.
    assert hot["peaks"][2] >= 0.97 * hot["predicted"]
    # The relaunch runs its first two steps fully checkpointed again, then the same plan.
    assert all(r["restarted_peaks"] == r["peaks"][:3] for r in results)


def test_the_callback_prices_the_backward_phase_at_its_own_pre_optimizer_mark(tmp_path):
    """On short rows the optimizer phase is the fully checkpointed step's peak, so a budget between the two off
    predictions relaxes to off only when ``on_pre_optimizer_step`` marks the backward end."""
    seq = SEQ // 4
    reading = _measured("dense", _LiveBytes(), seq=seq)[2]
    marked = predict_peak(reading, LAYERS, CHECKPOINTING_OFF)
    unmarked = predict_peak(dataclasses.replace(reading, backward_peak=reading.peak), LAYERS, CHECKPOINTING_OFF)
    assert unmarked > 1.02 * marked
    tracker = _LiveBytes()
    callback = GradientCheckpointingPlanCallback(
        _CLEAN, total_device_bytes=int((marked + unmarked) / 2 / DENSE_BUDGET_FRACTION), reader=_reader(tracker)
    )
    _drive(callback, "dense", tracker, str(tmp_path), 3, seq=seq)
    assert callback.plan.every_n_layers == CHECKPOINTING_OFF


def test_cuda_non_torch_bytes_are_the_device_bytes_the_allocator_does_not_reserve():
    gib = 2**30
    with (
        mock.patch.object(torch.cuda, "mem_get_info", return_value=(30 * gib, 100 * gib)),
        mock.patch.object(torch.cuda, "memory_reserved", return_value=50 * gib),
    ):
        assert plan_callback.CUDA_MEMORY.non_torch() == 20 * gib


def _builder_worker(rank: int, out_dir: str, case: str):
    torch.set_num_threads(1)
    model = _enabled("dense")
    config = SFTConfig(output_dir=out_dir, bf16=False, use_cpu=True, gradient_checkpointing=True, max_length=SEQ)
    # One GPU model reporting two totals (ECC mode), or two GPU models.
    total = 2**34 - (2**30 if case == "mixed_memory" and rank == 1 else 0)
    gpu_model = "H100" if case == "mixed_model" and rank == 1 else "B300"
    build = functools.partial(
        build_gradient_checkpointing_plan_callback,
        config,
        model,
        ParallelismConfig(),
        _PlainTrainer,
        packed_rows=case != "unpacked",
        multimodal_rows=False,
    )
    outcome = {}
    with (
        mock.patch.object(torch.cuda, "is_available", return_value=True),
        mock.patch.object(torch.cuda, "current_device", return_value=0),
        mock.patch.object(torch.cuda, "get_device_properties", return_value=SimpleNamespace(total_memory=total)),
        mock.patch.object(plan_callback, "detect_gpu_model", return_value=gpu_model),
    ):
        try:
            fresh = build()
        except ValueError as error:
            outcome["error"] = str(error)
        else:
            outcome["floor"] = fresh.floor_index
            outcome["layers"] = fresh.context.num_layers
            outcome["total_device_bytes"] = [fresh.total_device_bytes, fresh.context.fingerprint["total_device_bytes"]]
            save_run_plan(out_dir, GradientCheckpointingPlan(every_n_layers=4, fingerprint=fresh.context.fingerprint))
            outcome["after_unfinished"] = build().floor_index
            save_run_plan(
                out_dir,
                GradientCheckpointingPlan(every_n_layers=4, fingerprint=fresh.context.fingerprint, finished=True),
            )
            outcome["after_finished"] = build().floor_index
    with open(os.path.join(out_dir, f"rank{rank}.json"), "w") as f:
        json.dump(outcome, f)


@pytest.mark.parametrize("case", ["agreed", "mixed_memory", "mixed_model", "unpacked"])
def test_the_builder_agrees_one_plan_input_across_ranks_and_floors_on_an_unfinished_record(tmp_path, case):
    run_gloo_ranks(_builder_worker, 2, str(tmp_path), case)
    results = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)]
    if case in ("agreed", "mixed_memory"):
        # Every rank budgets on the smallest device, so one plan fits them all.
        smallest = 2**34 - (2**30 if case == "mixed_memory" else 0)
        for outcome in results:
            assert outcome["floor"] == 0 and outcome["layers"] == LAYERS
            assert outcome["total_device_bytes"] == [smallest, smallest]
            assert outcome["after_unfinished"] == candidate_policies(LAYERS).index(3)
            assert outcome["after_finished"] == 0
    elif case == "mixed_model":
        # Every rank raises, naming the field, instead of resolving a plan for a cluster it does not describe.
        assert all("plan's inputs differs across ranks" in r["error"] and "gpu_model" in r["error"] for r in results)
    else:
        assert all("refused for this run" in r["error"] and "does not pack" in r["error"] for r in results)


def test_the_builder_needs_a_gpu(tmp_path):
    config = SFTConfig(output_dir=str(tmp_path), bf16=False, use_cpu=True, gradient_checkpointing=True)
    with mock.patch.object(torch.cuda, "is_available", return_value=False), pytest.raises(ValueError, match="GPU"):
        build_gradient_checkpointing_plan_callback(
            config,
            _enabled("dense"),
            ParallelismConfig(),
            _PlainTrainer,
            packed_rows=True,
            multimodal_rows=False,
        )


def test_the_script_hands_the_plan_the_rows_its_caller_names():
    context = ScriptContext(
        parallelism_config=ParallelismConfig(),
        local_rank=0,
        resume_checkpoint=None,
        model_source="model",
        trainer_cls=_PlainTrainer,
    )
    with (
        mock.patch.object(script_entry, "configure_moe_balancing", return_value="none"),
        mock.patch.object(script_entry, "build_run_callbacks", side_effect=lambda *_args, **_kwargs: []),
        mock.patch.object(script_entry, "build_gradient_checkpointing_plan_callback", return_value="plan") as build,
    ):
        args = SimpleNamespace(gradient_checkpointing_plan="auto")
        assert configure_balancing_and_build_callbacks(
            args, "config", "model", context, packed_rows=True, multimodal_rows=True
        ) == ["plan"]
        assert build.call_args.kwargs == {"packed_rows": True, "multimodal_rows": True}
        # A script that names no rows (every trainer but SFT) hands the plan unpacked text rows, which it refuses.
        configure_balancing_and_build_callbacks(args, "config", "model", context)
        assert build.call_args.kwargs == {"packed_rows": False, "multimodal_rows": False}
        args.gradient_checkpointing_plan = "off"
        assert configure_balancing_and_build_callbacks(args, "config", "model", context) == []
        assert build.call_count == 2


class _Stop(Exception):
    """Raised by the stubbed callback builder, so ``main()`` runs only as far as it."""


_TURNS = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]


def _sft_main_rows(tmp_path, yaml_body: str, *, preprocessed_packed: bool | None = None, vlm: bool = False) -> dict:
    """The rows ``sft.py:main()`` hands the callback builder; ``preprocessed_packed=None`` keeps the dataset raw."""
    module = load_script_module("scripts/training/sft.py", "halo_test_sft_checkpointing_plan")
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        f"model_name_or_path: stub/model\ndataset:\n- dummy/dataset\noutput_dir: {tmp_path / 'out'}\n"
        f"bf16: false\nuse_cpu: true\nmax_length: {SEQ}\ngradient_checkpointing_plan: auto\n{yaml_body}"
    )
    rows = {"prompt": [_TURNS]}
    dataset = DatasetDict({"train": Dataset.from_dict(rows), "test": Dataset.from_dict(rows)})
    runtime = SimpleNamespace(
        parallelism_config=ParallelismConfig(),
        model_source="stub/model",
        local_rank=0,
        resume_checkpoint=None,
        trainer_cls=_PlainTrainer,
    )
    policy = SimpleNamespace(model=mock.Mock(), tokenizer=mock.Mock(), processing_class=mock.Mock())
    captured = {}

    def build_callbacks(*_args, **kwargs):
        captured.update(packed_rows=kwargs["packed_rows"], multimodal_rows=kwargs["multimodal_rows"])
        raise _Stop

    no_op = {
        name: mock.DEFAULT
        for name in (
            "reject_multi_document_rows",
            "reject_unsupported_vlm_sft_options",
            "install_resolved_tokenizer",
            "apply_text_path_padding_side",
            "log_model_info",
            "log_script_dataset_examples",
        )
    }
    data = (dataset["train"], dataset["test"], None, mock.Mock())
    with (
        mock.patch.multiple(module, **no_op),
        mock.patch.object(module, "probe_checkpoint", return_value=(Qwen3Config(), vlm)),
        mock.patch.object(module, "has_dense_kernel_row_refusal", return_value=False),
        mock.patch.object(module, "init_training_script", return_value=runtime),
        mock.patch.object(
            module, "load_script_datasets", return_value=((dataset, preprocessed_packed is not None), False)
        ),
        mock.patch.object(module, "resolve_vlm_run", return_value=vlm),
        mock.patch.object(
            module, "load_preprocessed_metadata", return_value=SimpleNamespace(packed=preprocessed_packed, is_vlm=vlm)
        ),
        mock.patch.object(module, "plan_row_attention", return_value=SimpleNamespace(attn_default=None)),
        mock.patch.object(module, "load_policy_model", return_value=policy),
        mock.patch.object(module, "apply_max_length", return_value=policy.tokenizer),
        mock.patch.object(module, "configure_adapter_training", return_value=None),
        mock.patch.object(module, "_prepare_text_data", return_value=data),
        mock.patch.object(module, "_prepare_vlm_data", return_value=data),
        mock.patch.object(module.GenerateExamplesCallback, "from_config", return_value=None),
        mock.patch.object(module, "configure_balancing_and_build_callbacks", side_effect=build_callbacks),
        mock.patch("halo.entrypoints.script.install_log_tee"),
        mock.patch.object(sys, "argv", ["prog", str(config_file)]),
        pytest.raises(_Stop),
    ):
        module.main()
    return captured


@pytest.mark.parametrize(
    ("yaml_body", "preprocessed_packed", "vlm", "packed_rows"),
    [
        ("packing: true\n", None, False, True),
        ("", None, False, False),
        # The collator builds what the corpus was baked as, whatever ``packing`` says.
        ("", True, False, True),
        ("packing: true\n", False, False, False),
        ("packing: true\n", None, True, True),
    ],
    ids=["raw_packed", "raw_unpacked", "preprocessed_packed", "preprocessed_unpacked", "vlm"],
)
def test_sft_hands_the_plan_the_rows_its_collator_builds(tmp_path, yaml_body, preprocessed_packed, vlm, packed_rows):
    captured = _sft_main_rows(tmp_path, yaml_body, preprocessed_packed=preprocessed_packed, vlm=vlm)
    assert captured == {"packed_rows": packed_rows, "multimodal_rows": vlm}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
