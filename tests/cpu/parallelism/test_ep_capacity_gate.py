#!/usr/bin/env python
"""The declared per-rank token budget against DeepEP's dispatch ceilings at config time. The dispatcher refuses the
same capacities only after the whole load, or at the step whose batch first reaches ``max_length``."""

from __future__ import annotations

import pytest

from halo.configs.parallelism_arguments import ParallelismArguments, parallelism_config_from_args
from halo.distributed.expert_parallel.deepep.capacity import GIN_MAX_TOKENS_PER_RANK, ep_dispatch_capacity
from halo.distributed.parallelism_config import ParallelismConfig
from halo.distributed.pipeline_parallel.runner import resolve_microbatch_count
from halo.model_loading.parallel.load_rules import reject_unsupported_model
from halo.models.attention.gpt_oss_sinks import SinksPolicy
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.parallelism import make_parallelism_config, simulated_world
from tests.common.rosters import distributed_trainer_classes

# Hidden 8192 x top-8 puts the 32-bit index ceiling at 2**31 / (8 x 8192) = 32768 tokens/rank, a budget a long-context
# run can declare.
_WIDE_HIDDEN = 8192
_WIDE_TOPK = 8
_WIDE_INDEX_CEILING = 2**31 // (_WIDE_TOPK * _WIDE_HIDDEN)
# gpt-oss shaped: hidden 2880 pads to 3072 on the wire, top-4.
_GPTOSS = {"hidden_size": 2880, "num_experts_per_tok": 4}


class _Cfg:
    def __init__(self, num_experts: int, **fields):
        self.num_experts = num_experts
        for name, value in fields.items():
            setattr(self, name, value)


def _wide(num_experts: int = 128, **fields) -> _Cfg:
    return _Cfg(num_experts, hidden_size=_WIDE_HIDDEN, num_experts_per_tok=_WIDE_TOPK, **fields)


def _config(*, budget: int | None = None, **kwargs) -> ParallelismConfig:
    """``budget`` is one number; trainers stamp it as two factors the gate multiplies out, so one row per device
    makes it the budget."""
    if budget is not None:
        kwargs.update(ep_rows_per_device=1, ep_declared_max_length=budget)
    return make_parallelism_config(**kwargs)


def _reject_unsupported_model(config: ParallelismConfig, model_config) -> None:
    reject_unsupported_model(config, model_config, sinks_policy=SinksPolicy.NEUTRALIZED, wraps_experts=False)


def test_a_budget_over_the_wire_index_ceiling_is_refused_with_its_numbers():
    """32768 tokens/rank x top-8 x hidden 8192 reaches 2**31 exactly."""
    config = _config(world_size=8, gpus_per_node=8, ep_size=8, budget=_WIDE_INDEX_CEILING)
    with pytest.raises(ValueError) as err:
        _reject_unsupported_model(config, _wide())
    text = str(err.value)
    assert "32-bit wire-index limit" in text, text
    assert f"{_WIDE_INDEX_CEILING} tokens/rank" in text, "the refusal must name the budget it computed"
    assert "EP group of 8" in text, "and the group the budget was judged on"


def test_the_refusal_leads_with_the_escape_hatches():
    """An operator reads the head of a config-time raise, and the override lifts only the Gin ceiling, not the
    32-bit wire index."""
    config = _config(world_size=8, gpus_per_node=8, ep_size=8, budget=_WIDE_INDEX_CEILING)
    with pytest.raises(ValueError) as err:
        _reject_unsupported_model(config, _wide())
    first_sentence = str(err.value).split(". ")[0]
    assert "max_length" in first_sentence, first_sentence
    assert "HALO_DEEPEP_GIN_MAX_TOKENS_PER_RANK" in first_sentence, first_sentence
    assert "cross-node Gin ceiling only" in first_sentence, "the override must not be offered for both ceilings"


def test_a_budget_under_the_ceiling_passes():
    _reject_unsupported_model(
        _config(world_size=8, gpus_per_node=8, ep_size=8, budget=_WIDE_INDEX_CEILING // 2), _wide()
    )


def test_the_gate_aligns_the_budget_the_way_the_dispatcher_does():
    """``_WIDE_INDEX_CEILING - 255`` fits the wire index as declared and not as sized."""
    budget = _WIDE_INDEX_CEILING - 255
    assert ep_dispatch_capacity(budget) == _WIDE_INDEX_CEILING, "precondition: the alignment is what bites"
    config = _config(world_size=8, gpus_per_node=8, ep_size=8, budget=budget)
    with pytest.raises(ValueError, match="32-bit wire-index limit"):
        _reject_unsupported_model(config, _wide())


def test_context_parallelism_divides_the_per_rank_budget():
    """A CP rank holds one sequence chunk, so cp2 halves what its MoE layers present."""
    config = _config(world_size=8, gpus_per_node=8, ep_size=8, cp_size=2, budget=_WIDE_INDEX_CEILING)
    _reject_unsupported_model(config, _wide())


def test_nvl72_rack_wide_ep_is_not_bound_by_the_cross_node_ceiling():
    """The EP group is a whole rack, so ``requires_rdma`` is False; judging by OS node would refuse it."""
    config = _config(
        world_size=144,
        gpus_per_node=8,
        nvlink_domain_size=72,
        ep_size=72,
        budget=2 * GIN_MAX_TOKENS_PER_RANK,
    )
    assert config.ep_scope == "node" and not config.requires_rdma, "precondition: the group is one rack"
    _reject_unsupported_model(config, _Cfg(144, **_GPTOSS))


def test_cross_node_ep_over_four_nvl8_nodes_is_refused_above_the_gin_ceiling():
    """Gin wedges above the cap."""
    config = _config(
        world_size=32,
        gpus_per_node=8,
        ep_size=32,
        ep_scope="global",
        budget=2 * GIN_MAX_TOKENS_PER_RANK,
    )
    assert config.requires_rdma, "precondition: the group spans NVLink domains"
    with pytest.raises(ValueError) as err:
        _reject_unsupported_model(config, _Cfg(128, **_GPTOSS))
    text = str(err.value)
    assert "proxy-GIN ceiling" in text, text
    assert "cross-domain — Gin/RDMA" in text, "the refusal must say which ceiling applied and why"


def test_cross_node_ep_at_the_validated_ceiling_passes():
    """The measured boundary is inclusive; the shipped Inkling recipe runs mbs 2 x 4096 exactly on it."""
    _reject_unsupported_model(
        _config(
            world_size=32,
            gpus_per_node=8,
            ep_size=32,
            ep_scope="global",
            budget=GIN_MAX_TOKENS_PER_RANK,
        ),
        _Cfg(128, **_GPTOSS),
    )


def test_a_512_gpu_job_is_judged_on_the_same_per_rank_budget():
    """The DeepEP arena is per-rank, so 64 nodes of node-local ep8 buy no headroom over one node."""
    shape = {"world_size": 512, "gpus_per_node": 8, "ep_size": 8}
    _reject_unsupported_model(_config(**shape, budget=_WIDE_INDEX_CEILING // 2), _wide())
    with pytest.raises(ValueError, match="32-bit wire-index limit"):
        _reject_unsupported_model(_config(**shape, budget=_WIDE_INDEX_CEILING), _wide())


def test_an_undeclared_budget_leaves_the_ceilings_to_the_dispatcher():
    """The config states a 32768-token window, the budget the wire-index test refuses, so a resolved window here
    would be a false raise."""
    _reject_unsupported_model(
        _config(world_size=8, gpus_per_node=8, ep_size=8, ep_rows_per_device=0),
        _wide(max_position_embeddings=_WIDE_INDEX_CEILING),
    )


def test_pure_expert_tp_never_reaches_the_transport():
    _reject_unsupported_model(
        _config(world_size=8, gpus_per_node=8, expert_tp_size=8, budget=8 * _WIDE_INDEX_CEILING), _wide(num_experts=8)
    )


def test_a_null_max_length_is_judged_against_the_models_own_context_window():
    """Left unresolved, the product is 0 and switches the gate off for the spelling that reaches the ceilings first."""
    config = _config(world_size=8, gpus_per_node=8, ep_size=8, ep_rows_per_device=1, ep_declared_max_length=0)
    with pytest.raises(ValueError) as err:
        _reject_unsupported_model(config, _wide(max_position_embeddings=_WIDE_INDEX_CEILING))
    text = str(err.value)
    assert "32-bit wire-index limit" in text, text
    assert f"{_WIDE_INDEX_CEILING} tokens/row" in text, "the refusal must name the length it resolved"
    assert "no length cap" in text, "and say the number came from the model, not from the config"


def test_a_null_max_length_under_the_ceiling_still_passes():
    """Anti-over-rejection: a gate that refused every null budget would fail here."""
    _reject_unsupported_model(
        _config(world_size=8, gpus_per_node=8, ep_size=8, ep_rows_per_device=1, ep_declared_max_length=0),
        _wide(max_position_embeddings=_WIDE_INDEX_CEILING // 2),
    )


def test_a_null_max_length_on_a_config_stating_no_window_is_not_guessed_at():
    """The dispatcher's runtime backstop owns that run."""
    _reject_unsupported_model(
        _config(world_size=8, gpus_per_node=8, ep_size=8, ep_rows_per_device=1, ep_declared_max_length=0), _wide()
    )


def test_the_paired_row_count_multiplies_the_resolved_window_too():
    shape = {"world_size": 8, "gpus_per_node": 8, "ep_size": 8}
    model = _wide(max_position_embeddings=_WIDE_INDEX_CEILING // 2)
    _reject_unsupported_model(_config(**shape, ep_rows_per_device=1, ep_declared_max_length=0), model)
    with pytest.raises(ValueError, match="32-bit wire-index limit"):
        _reject_unsupported_model(_config(**shape, ep_rows_per_device=2, ep_declared_max_length=0), model)


class _PairedTrainer(DistributedTrainerMixin):
    """Stands in for SMPO, DPO and the BT reward trainer: one concatenated chosen+rejected forward."""

    FORWARD_ROWS_PER_EXAMPLE = 2


class _TrainingConfig:
    def __init__(self, per_device_train_batch_size: int, max_length: int | None, gradient_accumulation_steps: int = 1):
        self.per_device_train_batch_size = per_device_train_batch_size
        self.max_length = max_length
        self.gradient_accumulation_steps = gradient_accumulation_steps


class _GenerationTrainingConfig:
    """GRPO's budget is two other knobs."""

    def __init__(self, per_device_train_batch_size: int):
        self.per_device_train_batch_size = per_device_train_batch_size


def _built(trainer_cls, training_config) -> ParallelismConfig:
    return parallelism_config_from_args(
        ParallelismArguments(),
        training_config=training_config,
        trainer_cls=trainer_cls,
    )


def _budget(trainer_cls, batch_size: int, max_length: int) -> int:
    config = _built(trainer_cls, _TrainingConfig(batch_size, max_length))
    return config.ep_rows_per_device * config.ep_declared_max_length


def test_a_concatenated_preference_forward_declares_twice_the_rows():
    """At 2 x 4096 the unpaired budget (8192) clears the Gin ceiling and the paired one (16384) does not; missing
    the doubling wedges at step 1."""
    batch_size, max_length = 2, GIN_MAX_TOKENS_PER_RANK // 2
    assert _budget(DistributedTrainerMixin, batch_size, max_length) == GIN_MAX_TOKENS_PER_RANK
    assert _budget(_PairedTrainer, batch_size, max_length) == 2 * GIN_MAX_TOKENS_PER_RANK

    shape = {"world_size": 32, "gpus_per_node": 8, "ep_size": 32, "ep_scope": "global"}
    _reject_unsupported_model(
        _config(**shape, budget=_budget(DistributedTrainerMixin, batch_size, max_length)), _Cfg(128, **_GPTOSS)
    )
    with pytest.raises(ValueError, match="proxy-GIN ceiling"):
        _reject_unsupported_model(
            _config(**shape, budget=_budget(_PairedTrainer, batch_size, max_length)), _Cfg(128, **_GPTOSS)
        )


def test_the_prologue_separates_a_null_max_length_from_an_absent_one():
    """A single product would spell both 0, yet ``null`` is the whole window and an absent knob declares no budget."""
    null_length = _built(_PairedTrainer, _TrainingConfig(2, None))
    assert (null_length.ep_rows_per_device, null_length.ep_declared_max_length) == (4, 0)

    no_knob = _built(_PairedTrainer, _GenerationTrainingConfig(2))
    assert (no_knob.ep_rows_per_device, no_knob.ep_declared_max_length) == (0, 0)


def _pp_built(batch_size: int, *, microbatches: int = 0, accumulation: int = 1, schedule: str = "1f1b"):
    """pp2 × ep8 on two 8-GPU nodes at 16384 tokens/row, through the real builder."""
    with simulated_world(16, 8):
        return parallelism_config_from_args(
            ParallelismArguments(
                pipeline_parallel_size=2,
                expert_parallel_size=8,
                pipeline_microbatches=microbatches,
                pipeline_schedule=schedule,
                nvlink_domain_size=8,
            ),
            training_config=_TrainingConfig(batch_size, _WIDE_INDEX_CEILING // 2, accumulation),
            trainer_cls=DistributedSFTTrainer,
        )


@pytest.mark.parametrize(
    "microbatches, accumulation, schedule, rows",
    [
        (8, 1, "1f1b", 1),
        (0, 8, "1f1b", 1),
        (2, 1, "1f1b", 4),
        (0, 1, "1f1b", 4),
        (8, 1, "gpipe", 1),
        (0, 1, "gpipe", 8),
    ],
)
def test_pipeline_rows_are_one_microbatch_as_the_schedule_resolves_it(microbatches, accumulation, schedule, rows):
    """The gate's row count and the schedule's split must come from one count, 1F1B's full-pipeline floor included."""
    config = _pp_built(8, microbatches=microbatches, accumulation=accumulation, schedule=schedule)
    assert config.ep_rows_per_device == rows
    assert resolve_microbatch_count(config, accumulation, 8) == 8 // rows


def test_a_pipeline_is_judged_on_one_microbatch_not_the_whole_batch():
    """8 × 16384 tokens over-runs the wire index as one forward; split into 8 microbatches each one fits."""
    _reject_unsupported_model(_pp_built(8, microbatches=8), _wide())


def test_a_microbatch_over_the_ceiling_is_still_refused_and_names_the_pipeline_lever():
    """4 rows × 16384 is twice the ceiling: the gate still binds, and more microbatches is what clears it."""
    with pytest.raises(ValueError) as err:
        _reject_unsupported_model(_pp_built(8, microbatches=2), _wide())
    assert "raise pipeline_microbatches" in str(err.value).split(". ")[0]
    assert "4 row(s)/forward" in str(err.value)


# A fact about each trainer's own forward, so not derivable; a trainer stating nothing fails instead of inheriting the
# base's 1.
_ROWS_PER_EXAMPLE: dict[str, int] = {
    "SMPOTrainer": 2,
    "DistributedDPOTrainer": 2,
    "DistributedRewardTrainer": 2,
    # KTO's KL completion is a separate forward, distillation runs separate models, offline GRPO expands one row per
    # completion, embedding losses call the model per text column.
    "ClassificationTrainer": 1,
    "DistributedAsyncEnvironmentalGRPOTrainer": 1,
    "DistributedTeacherDistillationTrainer": 1,
    "DistributedOnlineGRPOTrainer": 1,
    "DistributedKTOTrainer": 1,
    "DistributedSDPGTrainer": 1,
    "DistributedSFTTrainer": 1,
    "DistributedSelfDistillationTrainer": 1,
    "EmbeddingTrainer": 1,
    "OfflineGRPOTrainer": 1,
}

TRAINER_ROSTER = distributed_trainer_classes()


def test_the_derived_roster_covers_the_trainer_families():
    names = {cls.__name__ for cls in TRAINER_ROSTER}
    assert len(names) >= 12, f"the roster collapsed to {sorted(names)} — the derivation stopped finding trainers"
    assert {"DistributedSFTTrainer", "SMPOTrainer", "OfflineGRPOTrainer"} <= names, sorted(names)


@pytest.mark.parametrize("trainer", TRAINER_ROSTER, ids=lambda cls: cls.__name__)
def test_every_trainer_states_the_rows_its_forward_carries(trainer):
    """A paired forward declaring nothing is judged on half the tokens with no error, so a new trainer fails here
    until its author has read its forward."""
    expected = _ROWS_PER_EXAMPLE.get(trainer.__name__)
    assert expected is not None, (
        f"{trainer.__name__} is on the trainer roster with no row count stated here. Read its forward: "
        f"one that scores two sequences in a single model call (torch.cat(..., dim=0)) must declare "
        f"FORWARD_ROWS_PER_EXAMPLE = 2 on the class, or the config-time EP dispatch gate judges "
        f"it on half its tokens. Record the value either way."
    )
    assert expected == trainer.FORWARD_ROWS_PER_EXAMPLE


def test_a_config_that_declares_no_router_width_is_not_guessed_at():
    _reject_unsupported_model(
        _config(world_size=8, gpus_per_node=8, ep_size=8, budget=_WIDE_INDEX_CEILING),
        _Cfg(128, hidden_size=_WIDE_HIDDEN),
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
