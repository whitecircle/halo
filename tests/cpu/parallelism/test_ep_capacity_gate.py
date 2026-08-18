#!/usr/bin/env python
"""The run's declared per-rank token budget is judged against DeepEP's dispatch ceilings at config time.

The dispatcher sizes its arena from the all-reduced MAX tokens/rank of the EP group, aligned up by
``ep_dispatch_capacity``, and refuses a capacity past DeepEP's 32-bit wire index or past the validated
cross-node Gin ceiling. Both refusals are correct and both are late: they land after the whole
multi-hundred-GB load, and — for a corpus whose early batches are short — only at the step whose batch
first reaches ``max_length``. ``per_device_train_batch_size × max_length`` is that MAX, known before a
weight is read, so ``ParallelismConfig.validate_against_model_config`` applies the same two ceilings
through the same function the dispatcher calls.

What the gate must get right at scale:

* the ceiling that applies is the NVLink DOMAIN's question, not the OS node's — an NVL72 rack-wide
  ep72 group stays on MNNVL and is not Gin-bound, while a 4-node NVL8 ep32 group is;
* the per-rank buffer does not shrink with EP size, so a 512-GPU job is judged on the same budget as
  an 8-GPU one;
* CP is the one axis that divides a rank's token count;
* ``max_length: null`` — the documented "use the model's own limit" spelling, and the LARGEST budget a
  run can declare — is resolved against ``config.json``, not read as an absent budget;
* it stays silent on what it cannot judge (no declared budget, a config stating no context window,
  pure ETP, a dense config), because a false raise here blocks a legal run.

The runtime backstop is unchanged and still owns the batch actually in hand
(``test_ep_capacity_dedup_generation.py``).

Run: ``python tests/cpu/parallelism/test_ep_capacity_gate.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import pytest

from src.args.distributed_args import DistributedArguments
from src.distributed.expert_parallel.config import GIN_MAX_TOKENS_PER_RANK, ep_dispatch_capacity
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.mixins.base import DistributedTrainerMixin
from src.training.parallelism_args import (
    FORWARD_ROWS_PER_EXAMPLE_ATTR,
    forward_rows_per_example,
    parallelism_config_from_args,
)
from tests.common.parallelism import make_parallelism_config

# The subclass walk that derives the trainer roster, imported rather than restated so both contracts
# read the same hierarchy (importing it also populates the roster by importing ``src.trainers``).
from tests.cpu.trainers.test_loss_kwargs_contract import _subclasses

# Wide model: hidden 8192 (already wire-aligned) × top-8 puts the 32-bit index ceiling at a capacity
# of 2**31 / (8 × 8192) = 32768 tokens/rank — a budget a long-context run can genuinely declare.
_WIDE_HIDDEN = 8192
_WIDE_TOPK = 8
_WIDE_INDEX_CEILING = 2**31 // (_WIDE_TOPK * _WIDE_HIDDEN)
# gpt-oss shaped: hidden 2880 pads to 2944 on the wire, top-4.
_GPTOSS = {"hidden_size": 2880, "num_experts_per_tok": 4}


class _Cfg:
    """A stand-in HF config carrying only the fields the gate reads."""

    def __init__(self, num_experts: int, **fields):
        self.num_experts = num_experts
        for name, value in fields.items():
            setattr(self, name, value)


def _wide(num_experts: int = 128, **fields) -> _Cfg:
    return _Cfg(num_experts, hidden_size=_WIDE_HIDDEN, num_experts_per_tok=_WIDE_TOPK, **fields)


def _config(*, budget: int | None = None, **kwargs) -> ParallelismConfig:
    """A config on the simulated topology; ``world_size``/``gpus_per_node`` state the shape.

    ``budget`` states a declared per-rank token budget as one number — the trainers stamp it as the
    two factors the gate multiplies out (rows per device × ``max_length``), so one row per device
    makes the number itself the budget.
    """
    if budget is not None:
        kwargs.update(ep_rows_per_device=1, ep_declared_max_length=budget)
    return make_parallelism_config(**kwargs)


def test_a_budget_over_the_wire_index_ceiling_is_refused_with_its_numbers():
    """Single node, ep8: 32768 tokens/rank × top-8 × hidden 8192 reaches 2**31 exactly."""
    config = _config(world_size=8, gpus_per_node=8, ep_size=8, budget=_WIDE_INDEX_CEILING)
    with pytest.raises(ValueError) as err:
        config.validate_against_model_config(_wide())
    text = str(err.value)
    assert "32-bit wire-index limit" in text, text
    assert f"{_WIDE_INDEX_CEILING} tokens/rank" in text, "the refusal must name the budget it computed"
    assert "EP group of 8" in text, "and the group the budget was judged on"


def test_the_refusal_leads_with_the_escape_hatches():
    """The first sentence must carry the two levers, not bury them behind the arithmetic.

    An operator reads the head of a config-time raise; ``max_length`` and the Gin-ceiling override are
    what they can act on, and the override is scoped to the ceiling it actually lifts (it does not
    move the 32-bit wire index).
    """
    config = _config(world_size=8, gpus_per_node=8, ep_size=8, budget=_WIDE_INDEX_CEILING)
    with pytest.raises(ValueError) as err:
        config.validate_against_model_config(_wide())
    first_sentence = str(err.value).split(". ")[0]
    assert "max_length" in first_sentence, first_sentence
    assert "HALO_DEEPEP_GIN_MAX_TOKENS_PER_RANK" in first_sentence, first_sentence
    assert "cross-node Gin ceiling only" in first_sentence, "the override must not be offered for both ceilings"


def test_a_budget_under_the_ceiling_passes():
    """Anti-over-rejection: half that budget is the shape a long-context run actually wants."""
    _config(world_size=8, gpus_per_node=8, ep_size=8, budget=_WIDE_INDEX_CEILING // 2).validate_against_model_config(
        _wide()
    )


def test_the_gate_aligns_the_budget_the_way_the_dispatcher_does():
    """A budget under the ceiling that its ×256 alignment pushes over it must still be refused.

    ``_WIDE_INDEX_CEILING - 255`` fits the wire index as declared and does not fit as sized, so a gate
    that skipped ``ep_dispatch_capacity`` would pass a run the dispatcher then refuses.
    """
    budget = _WIDE_INDEX_CEILING - 255
    assert ep_dispatch_capacity(budget) == _WIDE_INDEX_CEILING, "precondition: the alignment is what bites"
    config = _config(world_size=8, gpus_per_node=8, ep_size=8, budget=budget)
    with pytest.raises(ValueError, match="32-bit wire-index limit"):
        config.validate_against_model_config(_wide())


def test_context_parallelism_divides_the_per_rank_budget():
    """A CP rank holds one sequence chunk, so cp2 halves what its MoE layers present.

    Same budget the first test refuses without CP — the division is the whole difference.
    """
    config = _config(world_size=8, gpus_per_node=8, ep_size=8, cp_size=2, budget=_WIDE_INDEX_CEILING)
    config.validate_against_model_config(_wide())


def test_nvl72_rack_wide_ep_is_not_bound_by_the_cross_node_ceiling():
    """ep72 inside a 72-GPU NVLink domain dispatches over MNNVL, not Gin — the 8k cap must not apply.

    Two racks (world 144) with ``NVLINK_DOMAIN_SIZE=72``: the EP group is a whole rack, so
    ``requires_rdma`` is False and a 16k-token/rank budget is legal. Judging by OS node instead of by
    NVLink domain would refuse it — the group spans nine 8-GPU nodes.
    """
    config = _config(
        world_size=144,
        gpus_per_node=8,
        nvlink_domain_size=72,
        ep_size=72,
        budget=2 * GIN_MAX_TOKENS_PER_RANK,
    )
    assert config.ep_scope == "node" and not config.requires_rdma, "precondition: the group is one rack"
    config.validate_against_model_config(_Cfg(144, **_GPTOSS))


def test_cross_node_ep_over_four_nvl8_nodes_is_refused_above_the_gin_ceiling():
    """ep32 spanning four 8-GPU NVLink domains dispatches over Gin, which wedges above the cap."""
    config = _config(
        world_size=32,
        gpus_per_node=8,
        ep_size=32,
        ep_scope="global",
        budget=2 * GIN_MAX_TOKENS_PER_RANK,
    )
    assert config.requires_rdma, "precondition: the group spans NVLink domains"
    with pytest.raises(ValueError) as err:
        config.validate_against_model_config(_Cfg(128, **_GPTOSS))
    text = str(err.value)
    assert "proxy-GIN ceiling" in text, text
    assert "cross-domain — Gin/RDMA" in text, "the refusal must say which ceiling applied and why"


def test_cross_node_ep_at_the_validated_ceiling_passes():
    """The measured boundary is inclusive — the shipped Inkling recipe runs mbs 2 × 4096 exactly on it."""
    _config(
        world_size=32,
        gpus_per_node=8,
        ep_size=32,
        ep_scope="global",
        budget=GIN_MAX_TOKENS_PER_RANK,
    ).validate_against_model_config(_Cfg(128, **_GPTOSS))


def test_a_512_gpu_job_is_judged_on_the_same_per_rank_budget():
    """The DeepEP arena is per-rank, so 64 nodes of node-local ep8 buy no headroom over one node."""
    shape = {"world_size": 512, "gpus_per_node": 8, "ep_size": 8}
    _config(**shape, budget=_WIDE_INDEX_CEILING // 2).validate_against_model_config(_wide())
    with pytest.raises(ValueError, match="32-bit wire-index limit"):
        _config(**shape, budget=_WIDE_INDEX_CEILING).validate_against_model_config(_wide())


def test_an_undeclared_budget_leaves_the_ceilings_to_the_dispatcher():
    """No rows = the trainer declares no max_length knob at all; the gate must not invent a bound.

    The model config states a 32768-token window here, which is exactly the budget the wire-index test
    refuses — so a gate that resolved a window without a declared shape would raise on a run whose
    real per-rank token count nothing here knows.
    """
    _config(world_size=8, gpus_per_node=8, ep_size=8, ep_rows_per_device=0).validate_against_model_config(
        _wide(max_position_embeddings=_WIDE_INDEX_CEILING)
    )


def test_pure_expert_tp_never_reaches_the_transport():
    """ep_size=1 + expert_tp replicates the experts and dispatches nothing, whatever the budget."""
    _config(
        world_size=8, gpus_per_node=8, expert_tp_size=8, budget=8 * _WIDE_INDEX_CEILING
    ).validate_against_model_config(_wide(num_experts=8))


# max_length: null — the largest budget a run can declare, resolved off config.json


def test_a_null_max_length_is_judged_against_the_models_own_context_window():
    """``max_length: null`` means "use the model's own limit", so the gate resolves the window rather
    than reading the run as having declared nothing.

    Left unresolved, the product ``rows × per_device_train_batch_size × 0`` switched the gate OFF for
    the one spelling that reaches the ceilings first — the refusal then landed at the first MoE
    dispatch, after the whole load.
    """
    config = _config(world_size=8, gpus_per_node=8, ep_size=8, ep_rows_per_device=1, ep_declared_max_length=0)
    with pytest.raises(ValueError) as err:
        config.validate_against_model_config(_wide(max_position_embeddings=_WIDE_INDEX_CEILING))
    text = str(err.value)
    assert "32-bit wire-index limit" in text, text
    assert f"max_length={_WIDE_INDEX_CEILING}" in text, "the refusal must name the length it resolved"
    assert "max_length: null" in text, "and say the number came from the model, not from the config"


def test_a_null_max_length_under_the_ceiling_still_passes():
    """Anti-over-rejection: resolving the window must not refuse a model whose window fits.

    Same declaration as the test above with a model half as long — the resolved number is the whole
    difference, so a gate that refused every null budget would fail here.
    """
    _config(
        world_size=8, gpus_per_node=8, ep_size=8, ep_rows_per_device=1, ep_declared_max_length=0
    ).validate_against_model_config(_wide(max_position_embeddings=_WIDE_INDEX_CEILING // 2))


def test_a_null_max_length_on_a_config_stating_no_window_is_not_guessed_at():
    """No ``max_position_embeddings``/``max_seq_length``/``n_positions`` means no window to resolve —
    silence, not a fabricated verdict. The dispatcher's runtime backstop still owns that run."""
    _config(
        world_size=8, gpus_per_node=8, ep_size=8, ep_rows_per_device=1, ep_declared_max_length=0
    ).validate_against_model_config(_wide())


def test_the_paired_row_count_multiplies_the_resolved_window_too():
    """The doubling and the resolution compose: a paired forward on a null budget is judged on
    ``2 × context_window``, the same as it would be on an explicit one."""
    shape = {"world_size": 8, "gpus_per_node": 8, "ep_size": 8}
    model = _wide(max_position_embeddings=_WIDE_INDEX_CEILING // 2)
    _config(**shape, ep_rows_per_device=1, ep_declared_max_length=0).validate_against_model_config(model)
    with pytest.raises(ValueError, match="32-bit wire-index limit"):
        _config(**shape, ep_rows_per_device=2, ep_declared_max_length=0).validate_against_model_config(model)


class _PairedTrainer:
    """Stands in for SMPO / DPO / the BT reward trainer: one concatenated chosen+rejected forward."""


setattr(_PairedTrainer, FORWARD_ROWS_PER_EXAMPLE_ATTR, 2)


class _TrainingConfig:
    """The two fields the builder reads for the declared per-rank token budget."""

    def __init__(self, per_device_train_batch_size: int, max_length: int | None):
        self.per_device_train_batch_size = per_device_train_batch_size
        self.max_length = max_length


class _GenerationTrainingConfig:
    """A trainer config with no ``max_length`` knob at all (GRPO's budget is two other knobs)."""

    def __init__(self, per_device_train_batch_size: int):
        self.per_device_train_batch_size = per_device_train_batch_size


def _built(trainer_cls, training_config) -> ParallelismConfig:
    """The config the entry-script prologue builds for this trainer, through the real builder."""
    return parallelism_config_from_args(
        DistributedArguments(),
        training_config=training_config,
        trainer_cls=trainer_cls,
        supports_cp=False,
        supports_pp=False,
    )


def _budget(trainer_cls, batch_size: int, max_length: int) -> int:
    """The per-rank token budget the prologue's stamped factors multiply out to."""
    config = _built(trainer_cls, _TrainingConfig(batch_size, max_length))
    return config.ep_rows_per_device * config.ep_declared_max_length


def test_a_concatenated_preference_forward_declares_twice_the_rows():
    """SMPO / DPO / BT reward run chosen+rejected in ONE model call (``torch.cat(..., dim=0)``), so a
    budget of ``per_device_train_batch_size x max_length`` names half the tokens the MoE layer sees.

    Judged at exactly the boundary: at 2 x 4096 the unpaired budget (8192) clears the Gin ceiling and
    the paired one (16384) does not, so a gate that missed the doubling would pass a cross-node EP
    preference run that wedges at step 1 — the failure this gate exists to pre-empt.
    """
    batch_size, max_length = 2, GIN_MAX_TOKENS_PER_RANK // 2
    assert _budget(None, batch_size, max_length) == GIN_MAX_TOKENS_PER_RANK
    assert _budget(_PairedTrainer, batch_size, max_length) == 2 * GIN_MAX_TOKENS_PER_RANK

    shape = {"world_size": 32, "gpus_per_node": 8, "ep_size": 32, "ep_scope": "global"}
    _config(**shape, budget=_budget(None, batch_size, max_length)).validate_against_model_config(_Cfg(128, **_GPTOSS))
    with pytest.raises(ValueError, match="proxy-GIN ceiling"):
        _config(**shape, budget=_budget(_PairedTrainer, batch_size, max_length)).validate_against_model_config(
            _Cfg(128, **_GPTOSS)
        )


def test_the_prologue_separates_a_null_max_length_from_an_absent_one():
    """Both spelled 0 in the old single-product field, and they mean opposite things: ``null`` is the
    model's whole context window, while a trainer with no such knob has declared no budget at all."""
    null_length = _built(_PairedTrainer, _TrainingConfig(2, None))
    assert (null_length.ep_rows_per_device, null_length.ep_declared_max_length) == (4, 0)

    no_knob = _built(_PairedTrainer, _GenerationTrainingConfig(2))
    assert (no_knob.ep_rows_per_device, no_knob.ep_declared_max_length) == (0, 0)


# Rows one model call carries per dataset example. Not derivable — it is a fact about each trainer's
# own forward — so every trainer on the derived roster states its count here, and one that states
# nothing fails rather than silently inheriting the 1 that ``forward_rows_per_example`` defaults to.
_ROWS_PER_EXAMPLE: dict[str, int] = {
    # One concatenated forward over the chosen|rejected pair: twice the rows in one model call.
    "SmoothMarginPOTrainer": 2,
    "DistributedDPOTrainer": 2,
    "DistributedRewardTrainer": 2,
    # One row per example. KTO's KL completion is a SEPARATE forward, never a dim-0 concat; the
    # distillation trainers run teacher and student as separate models; offline GRPO expands each
    # group into one dataset row per completion before the batch is built; the embedding losses call
    # the model once per text column.
    "ClassificationTrainer": 1,
    "DistributedAsyncEnvironmentalGRPOTrainer": 1,
    "DistributedDistillationTrainer": 1,
    "DistributedGRPOTrainer": 1,
    "DistributedKTOTrainer": 1,
    "DistributedSDPGTrainer": 1,
    "DistributedSFTTrainer": 1,
    "DistributedSelfDistillationTrainer": 1,
    "EmbeddingTrainer": 1,
    "OfflineGRPOTrainer": 1,
}

# Derived from the class hierarchy, not listed: a new trainer joins the parametrization by existing.
TRAINER_ROSTER = sorted(
    (cls for cls in _subclasses(DistributedTrainerMixin) if cls.__module__.startswith("src.trainers")),
    key=lambda cls: cls.__name__,
)


def test_the_derived_roster_covers_the_trainer_families():
    """Anti-vacuity: an empty or collapsed roster would make the coverage test below assert nothing."""
    names = {cls.__name__ for cls in TRAINER_ROSTER}
    assert len(names) >= 12, f"the roster collapsed to {sorted(names)} — the derivation stopped finding trainers"
    assert {"DistributedSFTTrainer", "SmoothMarginPOTrainer", "OfflineGRPOTrainer"} <= names, sorted(names)


@pytest.mark.parametrize("trainer", TRAINER_ROSTER, ids=lambda cls: cls.__name__)
def test_every_trainer_states_the_rows_its_forward_carries(trainer):
    """The budget the gate judges is ``rows × batch × max_length``, and ``rows`` is an opt-in the base
    defaults to 1 — so a paired forward that declares nothing is judged on HALF the tokens its MoE
    layers present, with no error anywhere. Nothing derives that count, so this demands one per
    roster member: a new trainer fails here until its author has read its own forward.
    """
    expected = _ROWS_PER_EXAMPLE.get(trainer.__name__)
    assert expected is not None, (
        f"{trainer.__name__} is on the trainer roster with no row count stated here. Read its forward: "
        f"one that scores two sequences in a single model call (torch.cat(..., dim=0)) must declare "
        f"{FORWARD_ROWS_PER_EXAMPLE_ATTR} = 2 on the class, or the config-time EP dispatch gate judges "
        f"it on half its tokens. Record the value either way."
    )
    assert forward_rows_per_example(trainer) == expected


def test_a_config_that_declares_no_router_width_is_not_guessed_at():
    """No top-k spelling means no wire extent to compute — silence, not a fabricated verdict."""
    _config(world_size=8, gpus_per_node=8, ep_size=8, budget=_WIDE_INDEX_CEILING).validate_against_model_config(
        _Cfg(128, hidden_size=_WIDE_HIDDEN)
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
