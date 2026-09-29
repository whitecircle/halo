#!/usr/bin/env python
"""Per-trainer pipeline-parallelism gate: only trainers that declare ``_supports_pp`` may run PP.

PP's failure modes are silent — a trainer whose loss needs a second forward, couples the whole
batch, or normalizes by a whole-batch denominator produces a WRONG NUMBER under a pipeline split,
not a crash. So the gate defaults to off (declare-to-enable, mirroring ``_supports_cp``) and every
rejection must name the mechanism so it is actionable.

These tests drive the REAL ``_validate_parallelism_modes``; a re-implementation of the predicate
could not catch a regression in the production guard.

Run: python tests/cpu/parallelism/test_pp_trainer_gates.py
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from accelerate import PartialState
from datasets import Dataset
from transformers import Glm5NextConfig, Glm5NextForConditionalGeneration, Qwen3Config, Qwen3ForCausalLM

from src.checkpoint.format import save_dtype_caster
from src.distributed.loading.peft_setup import _reject_layer_indexed_patterns_under_pp as guard
from src.distributed.pipeline_parallel.stage import PP_STAGE_PARTITION_ATTR
from src.trainers.mixins.base import DistributedTrainerMixin
from src.trainers.mixins.pipeline import PipelineTrainerMixin, wrapper_state_outside_stages
from src.trainers.mixins.validation import ParallelismValidationMixin
from tests.common.models import TINY_GLM5_CONFIG, TINY_GLM5_VISION_CONFIG, TINY_QWEN3_CONFIG
from tests.common.parallelism import make_parallelism_config
from tests.common.rosters import distributed_trainer_classes

# Derived from the class hierarchy: a new trainer joins at the default off and must name its reason.
_TRAINERS = distributed_trainer_classes()

# PP-enabled: single-forward, per-sequence-or-token losses whose whole-batch denominators can be
# precomputed into a step normalizer (DPO/KTO precompute-only, offline GRPO at any kl_beta through a
# construction-time reference sweep).
_PP_ENABLED: set[str] = {
    "DistributedSFTTrainer",
    "SmoothMarginPOTrainer",
    "DistributedRewardTrainer",
    "ClassificationTrainer",
    "OfflineGRPOTrainer",
    "DistributedDPOTrainer",
    "DistributedKTOTrainer",
}


def _pc(**kwargs):
    """A PP-shaped ParallelismConfig: 4 ranks as 2 domains of 2, pipeline across the boundary."""
    return make_parallelism_config(world_size=4, gpus_per_node=2, **kwargs)


def _pipeline_stub(**attrs):
    """A trainer-shaped stand-in for the split gate.

    ``_maybe_prepare_pipeline_model`` dispatches the per-trainer ``_validate_pp_mode`` hook through
    ``self``, so a bare namespace cannot stand in for a trainer; the mixin's own no-op hook is what
    a trainer that declares no extra gates inherits.
    """
    stub = object.__new__(PipelineTrainerMixin)
    stub.__dict__.update(attrs)
    return stub


class _Validating(ParallelismValidationMixin):
    """Minimal stub carrying exactly what the real validator reads."""

    def __init__(self, pc, supports_pp, reason=""):
        self.parallelism_config = pc
        self._supports_tp = self._supports_ep = self._supports_cp = True
        self._supports_pp = supports_pp
        self._pp_unsupported_reason = reason


def test_base_default_is_off():
    """Declare-to-enable: a trainer that never considered PP must not silently get it."""
    assert DistributedTrainerMixin._supports_pp is False
    assert DistributedTrainerMixin._pp_unsupported_reason == ""


def test_every_trainer_declares_a_pp_stance():
    """The PP-enabled trainers are exactly the audited set, and every rejection names a mechanism."""
    enabled = {cls.__name__ for cls in _TRAINERS if cls._supports_pp}
    assert enabled == _PP_ENABLED, f"_supports_pp is set on {sorted(enabled)}, expected {sorted(_PP_ENABLED)}"
    for cls in _TRAINERS:
        if not cls._supports_pp:
            reason = cls._pp_unsupported_reason
            assert reason and len(reason) > 40, (
                f"{cls.__name__} rejects PP without an actionable reason (got {reason!r})"
            )


def test_only_sft_rides_the_base_causal_lm_contract():
    """Every PP-enabled trainer but SFT declares its own PP loss contract.

    A trainer inheriting the base (causal-LM) ``_pp_loss_adapter`` gets token-level cross-entropy
    over ``labels``. That is right for SFT and wrong for every objective built from per-sequence
    quantities, so a second name appearing here means a trainer picked up the causal-LM loss
    silently instead of declaring its own.
    """
    on_base_contract = {
        cls.__name__
        for cls in _TRAINERS
        if cls._supports_pp and cls._pp_loss_adapter is DistributedTrainerMixin._pp_loss_adapter
    }
    assert on_base_contract == {"DistributedSFTTrainer"}, on_base_contract


def test_validator_rejects_unsupported_trainer_under_pp():
    """The REAL guard raises, and the message carries the trainer's own reason."""
    pc = _pc(pp_size=2)
    stub = _Validating(pc, supports_pp=False, reason="the loss couples the whole batch")
    with pytest.raises(ValueError, match="does not support Pipeline Parallelism"):
        stub._validate_parallelism_modes()
    with pytest.raises(ValueError, match="couples the whole batch"):
        stub._validate_parallelism_modes()


def test_validator_accepts_supported_trainer_under_pp():
    """A validator that rejects everything would also pass the rejection test — pin the accept."""
    _Validating(_pc(pp_size=2), supports_pp=True)._validate_parallelism_modes()


def test_gate_is_inert_without_pp():
    """pp_size=1 must not reject a trainer that does not support PP."""
    _Validating(_pc(pp_size=1), supports_pp=False, reason="irrelevant")._validate_parallelism_modes()


def test_rejection_falls_back_to_a_generic_reason():
    """A trainer that declares no reason still gets a message explaining the constraint."""
    stub = _Validating(_pc(pp_size=2), supports_pp=False, reason="")
    with pytest.raises(ValueError, match="contiguous subset of the layers"):
        stub._validate_parallelism_modes()


def test_layer_indexed_freeze_patterns_rejected_on_a_stage():
    """A global layer index means something different on every pipeline stage.

    The stage-aware loader hands back only this stage's layers, re-based to index 0, so
    ``model.layers.30.*`` matches nothing on most stages and a DIFFERENT layer on the rest — and the
    freeze path has no matched-nothing check, so it is silent. Index-free patterns are stage-invariant
    and must keep working, and nothing may change off PP.
    """
    model = nn.Sequential(nn.Linear(2, 2))
    # Off PP the patterns are resolved against the whole model, so indices are meaningful.
    guard(model, ["model.layers.30.*"], "freeze_layers_patterns")

    setattr(model, PP_STAGE_PARTITION_ATTR, (0, 4))
    for flag in ("freeze_layers_patterns", "unfreeze_layers_patterns"):
        with pytest.raises(ValueError, match="re-based to index 0"):
            guard(model, ["model.layers.30.*"], flag)
    guard(model, ["*.self_attn.sinks", "score"], "freeze_layers_patterns")


def test_layer_ranges_written_as_glob_character_classes_are_rejected():
    """The patterns are fnmatch globs, so real configs write layer RANGES as character classes.

    A config unfreezing layers 56-93 writes them as
    ``model.layers.5[6-9].*`` / ``model.layers.[6-8][0-9].*`` / ``model.layers.9[0-3].*``. A
    per-segment ``str.isdigit`` reads every one as index-free (``"5[6-9]".isdigit()`` is False), so
    they pass the gate and then match nothing on a re-based stage — training only whatever a sibling
    index-free pattern catches, silently, because a co-occurring ``*.o_proj*`` keeps the
    matched-nothing raise from firing.
    """
    model = nn.Sequential(nn.Linear(2, 2))
    setattr(model, PP_STAGE_PARTITION_ATTR, (0, 47))
    for pattern in ("model.layers.5[6-9].*", "model.layers.[6-8][0-9].*", "model.layers.9[0-3].*"):
        with pytest.raises(ValueError, match="re-based to index 0"):
            guard(model, [pattern], "unfreeze_layers_patterns")

    # The converse: a non-layer index and a wildcard are stage-invariant and must not be refused.
    guard(model, ["*.mlp.experts.0.*", "model.layers.*.mlp", "*.self_attn.sinks"], "freeze_layers_patterns")


def _pp_args(**overrides):
    base = {
        "max_length": 32,
        "gradient_checkpointing": False,
        "gradient_checkpointing_kwargs": None,
        "eval_strategy": "no",
        "eval_on_start": False,
        "per_device_eval_batch_size": 2,
        "per_device_train_batch_size": 2,
        "gradient_accumulation_steps": 2,
        "torch_compile": False,
        "activation_offloading": False,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _tiny_composite():
    """A tiny GLM-5 composite wrapper — a family that ships NO text-only CausalLM sibling."""
    torch.manual_seed(0)
    config = Glm5NextConfig(
        text_config=dict(TINY_GLM5_CONFIG), vision_config=dict(TINY_GLM5_VISION_CONFIG), attn_implementation="sdpa"
    )
    return Glm5NextForConditionalGeneration(config)


def _drive_composite_split(train_dataset=None, eval_dataset=None, data_collator=None):
    """``_maybe_prepare_pipeline_model`` on the tiny composite, as stage 0 of a pp2 x dp2 topology."""
    PartialState()  # the gate's accelerate logger needs an initialized state
    mixin = _pipeline_stub(parallelism_config=_pc(pp_size=2), save_sharded_ep=False, _moe_balancing="none")
    kwargs = {"model": _tiny_composite(), "train_dataset": train_dataset, "eval_dataset": eval_dataset}
    if data_collator is not None:
        kwargs["data_collator"] = data_collator
    return mixin, PipelineTrainerMixin._maybe_prepare_pipeline_model(mixin, kwargs, _pp_args())


_IMAGE_TURN = [
    {"role": "user", "content": [{"type": "image", "image": "not-a-real-image"}, {"type": "text", "text": "?"}]}
]


@pytest.mark.parametrize(
    "evidence",
    [
        pytest.param({"train_dataset": {"images": ["x"], "input_ids": [[1]]}}, id="image_column"),
        pytest.param({"eval_dataset": {"pixel_values": [[0.0]], "input_ids": [[1]]}}, id="baked_pixels_in_eval"),
        pytest.param({"train_dataset": {"messages": [_IMAGE_TURN]}}, id="embedded_image_parts"),
        pytest.param(
            {"data_collator": SimpleNamespace(required_dataset_columns=("input_ids", "pixel_values"))}, id="collator"
        ),
    ],
)
def test_image_bearing_vlm_run_is_rejected_under_pp(evidence):
    """A run that FEEDS images to a multimodal wrapper stays refused, naming what it saw.

    The split keeps the backbone and head, and the backbone probe descends to ``language_model`` —
    the tower and projector are its siblings, in no stage: images would never reach the model
    (``pixel_values`` is pruned by the runtime's column pin) while their placeholder tokens train as
    text. Silent, so the gate is a raise; every declaration the VLM data path accepts must trip it.

    Driven through ``_maybe_prepare_pipeline_model`` itself. Asserting on ``is_vlm_model`` instead
    would pass with the entire gate deleted.
    """
    kwargs = {key: (Dataset.from_dict(value) if isinstance(value, dict) else value) for key, value in evidence.items()}
    with pytest.raises(ValueError, match="Vision-language training is not supported") as err:
        _drive_composite_split(**kwargs)
    assert "means this run feeds images" in str(err.value)


def test_text_only_run_of_a_multimodal_wrapper_splits_and_keeps_its_vision_tower_for_the_save():
    """The same wrapper with text data is admitted: the text tower becomes the pipeline stage and
    the tensors no stage holds — the vision tower — are stashed, untouched, for the PP save to
    re-emit so the export keeps the wrapper layout (a resume plans for them too)."""
    mixin, kwargs = _drive_composite_split(
        train_dataset=Dataset.from_dict({"input_ids": [[1, 2]], "labels": [[1, 2]]})
    )
    stage = kwargs["model"]
    assert type(stage).__name__ == "PipelineStageModule"
    assert not any("visual" in name for name, _ in stage.named_parameters()), "the vision tower leaked into a stage"

    composite = _tiny_composite()
    reference = {k: v for k, v in composite.state_dict().items() if k.startswith("model.visual.")}
    assert reference, "the fixture composite carries no vision tower — the test would prove nothing"
    stash = mixin._pp_wrapper_state
    assert set(stash) == set(reference), sorted(set(stash) ^ set(reference))[:5]
    # Held at the artifact's save dtype (the wrapper's own norm keep-set applies), on the host.
    cast = save_dtype_caster(composite)
    assert all(torch.equal(stash[k], cast(k, reference[k])) and stash[k].device.type == "cpu" for k in reference)
    assert any(stash[k].dtype != reference[k].dtype for k in reference), "the save-dtype cast never applied"


def test_plain_causal_lm_drops_nothing_at_the_split():
    """Anti-vacuity for the stash: a model whose every tensor is stage-owned yields an empty set,
    so no PP save of a plain causal LM ever writes a wrapper part."""
    assert wrapper_state_outside_stages(Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG))) == {}


def test_sharded_ep_save_is_rejected_under_pp():
    """The per-rank EP format keys tensors by UNSPLIT-model names with no stage layer offset, so two
    stages would write ``layers.0.*`` shards that collide in the merged index — or merge under the
    wrong names. Documented as rejected in four places and enforced here, at the same gate the other
    PP blockers use, so the flag cannot quietly survive into a run's first save."""
    args = SimpleNamespace(
        max_length=128,
        gradient_checkpointing=False,
        gradient_checkpointing_kwargs=None,
        eval_strategy="no",
        eval_on_start=False,
        per_device_eval_batch_size=1,
        per_device_train_batch_size=1,
        torch_compile=False,
    )
    model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3"))
    sharded = _pipeline_stub(parallelism_config=SimpleNamespace(is_pp_mode=True), save_sharded_ep=True)

    with pytest.raises(ValueError, match="save_sharded_ep is not supported under pipeline parallelism"):
        PipelineTrainerMixin._maybe_prepare_pipeline_model(sharded, {"model": model}, args)

    # The converse: the gathered save must get PAST this gate (it fails later, for other reasons).
    gathered = _pipeline_stub(parallelism_config=SimpleNamespace(is_pp_mode=True), save_sharded_ep=False)
    try:
        PipelineTrainerMixin._maybe_prepare_pipeline_model(gathered, {"model": model}, args)
    except Exception as exc:  # noqa: BLE001 — any later failure is fine; the sharded-save one is not
        assert "save_sharded_ep" not in str(exc), f"the gathered save tripped the PP sharded-save gate: {exc}"


@pytest.mark.parametrize(
    ("eval_strategy", "eval_on_start"),
    [pytest.param("steps", False, id="eval_strategy"), pytest.param("no", True, id="eval_on_start_only")],
)
def test_eval_batch_mismatch_is_rejected_whenever_an_evaluation_runs(eval_strategy, eval_on_start):
    """The P2P buffers freeze to the training microbatch, so an eval batch of another size cannot be
    split into matching chunks. ``eval_on_start`` evaluates under ``eval_strategy: no`` too, so the
    gate must key on whether any evaluation runs, not on the strategy alone."""
    stub = _pipeline_stub(parallelism_config=SimpleNamespace(is_pp_mode=True), save_sharded_ep=False)
    args = _pp_args(eval_strategy=eval_strategy, eval_on_start=eval_on_start, per_device_eval_batch_size=4)
    with pytest.raises(ValueError, match=r"per_device_eval_batch_size \(4\) must equal per_device_train_batch_size"):
        PipelineTrainerMixin._maybe_prepare_pipeline_model(stub, {}, args)


def test_eval_batch_mismatch_passes_when_no_evaluation_runs():
    """The converse: with no evaluation scheduled the eval batch size is never used, so the run gets
    past the gate to the next one (the missing ``model``)."""
    stub = _pipeline_stub(parallelism_config=SimpleNamespace(is_pp_mode=True), save_sharded_ep=False)
    with pytest.raises(ValueError, match="requires the model to be passed"):
        PipelineTrainerMixin._maybe_prepare_pipeline_model(stub, {}, _pp_args(per_device_eval_batch_size=4))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
