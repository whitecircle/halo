#!/usr/bin/env python
"""Only trainers declaring ``PP_REFUSAL = None`` may run PP: a whole-batch or second-forward loss gives a wrong
number with no crash."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import Glm5NextConfig, Glm5NextForConditionalGeneration, Qwen3Config, Qwen3ForCausalLM
from transformers.training_args import OptimizerNames

from halo.checkpoint.writing import save_dtype_caster
from halo.distributed.pipeline_parallel.stage import PP_STAGE_PARTITION_ATTR
from halo.model_loading.parallel.freeze_patterns import _reject_layer_indexed_patterns_under_pp as guard
from halo.optimizers.registry import NAMED_OPTIMIZER_BUILDERS
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from halo.trainers.parallelism.pipeline_rules import reject_pp_named_optimizer
from halo.trainers.parallelism.pipeline_trainer import PipelineTrainerMixin, wrapper_state_outside_stages
from tests.common.models import TINY_GLM5_CONFIG, TINY_GLM5_VISION_CONFIG, TINY_QWEN3_CONFIG
from tests.common.parallelism import make_parallelism_config
from tests.common.rosters import distributed_trainer_classes

# A new trainer joins at the default off and must name its reason.
_TRAINERS = distributed_trainer_classes()

# Single-forward, per-sequence-or-token losses whose whole-batch denominators can be precomputed into a step
# normalizer.
_PP_ENABLED: set[str] = {
    "DistributedSFTTrainer",
    "SMPOTrainer",
    "DistributedRewardTrainer",
    "ClassificationTrainer",
    "OfflineGRPOTrainer",
    "DistributedDPOTrainer",
    "DistributedKTOTrainer",
}


def _pc(**kwargs):
    """4 ranks as 2 domains of 2, pipeline across the boundary."""
    return make_parallelism_config(world_size=4, gpus_per_node=2, **kwargs)


def _pipeline_stub(**attrs):
    """``_split_pipeline_stage`` dispatches its hook through ``self``, so a bare namespace cannot stand in for a
    trainer."""
    stub = object.__new__(PipelineTrainerMixin)
    stub.__dict__.update(attrs)
    return stub


def _declaring(pp_refusal):
    return type("_Declaring", (DistributedTrainerMixin,), {"PP_REFUSAL": pp_refusal})


def test_base_default_is_off():
    """Declare-to-enable."""
    assert DistributedTrainerMixin.PP_REFUSAL is not None


def test_every_trainer_declares_a_pp_stance():
    enabled = {cls.__name__ for cls in _TRAINERS if cls.PP_REFUSAL is None}
    assert enabled == _PP_ENABLED, f"PP_REFUSAL is None on {sorted(enabled)}, expected {sorted(_PP_ENABLED)}"
    for cls in _TRAINERS:
        if cls.PP_REFUSAL is not None:
            assert cls.PP_REFUSAL != DistributedTrainerMixin.PP_REFUSAL and len(cls.PP_REFUSAL) > 40, (
                f"{cls.__name__} rejects PP without a reason of its own (got {cls.PP_REFUSAL!r})"
            )


def test_only_sft_rides_the_base_causal_lm_contract():
    """The base ``_pp_loss_adapter`` is token-level CE, wrong for per-sequence objectives; a second name here picked
    it up silently."""
    on_base_contract = {
        cls.__name__
        for cls in _TRAINERS
        if cls.PP_REFUSAL is None and cls._pp_loss_adapter is DistributedTrainerMixin._pp_loss_adapter
    }
    assert on_base_contract == {"DistributedSFTTrainer"}, on_base_contract


def test_validator_rejects_unsupported_trainer_under_pp():
    trainer_cls = _declaring("the loss couples the whole batch")
    with pytest.raises(ValueError, match="does not support Pipeline Parallelism.*couples the whole batch"):
        trainer_cls.reject_undeclared_axes(cp_size=1, pp_size=2)


def test_validator_accepts_supported_trainer_under_pp():
    _declaring(None).reject_undeclared_axes(cp_size=1, pp_size=2)


def test_gate_is_inert_without_pp():
    _declaring("irrelevant").reject_undeclared_axes(cp_size=1, pp_size=1)


def test_rejection_falls_back_to_a_generic_reason():
    class _Undeclared(DistributedTrainerMixin):
        pass

    with pytest.raises(ValueError, match="_Undeclared does not support Pipeline Parallelism.*contiguous subset"):
        _Undeclared.reject_undeclared_axes(cp_size=1, pp_size=2)


@pytest.mark.parametrize("optim", sorted(NAMED_OPTIMIZER_BUILDERS))
def test_named_optimizers_are_refused_under_pp(optim):
    """No pipeline run validates a toolkit-named optimizer; post-init ``args.optim`` is an ``OptimizerNames``
    member."""
    with pytest.raises(ValueError, match=rf"optim='{optim}' is not supported under pipeline parallelism"):
        reject_pp_named_optimizer(SimpleNamespace(optim=OptimizerNames(optim)), _pc(pp_size=2))


def test_the_optimizer_refusal_spares_adamw_under_pp_and_named_optimizers_off_pp():
    reject_pp_named_optimizer(SimpleNamespace(optim="adamw_torch"), _pc(pp_size=2))
    reject_pp_named_optimizer(SimpleNamespace(optim="muon"), _pc(pp_size=1))


def test_layer_indexed_freeze_patterns_rejected_on_a_stage():
    """The loader re-bases stage layers to 0, so ``model.layers.30.*`` matches nothing or a different layer, and the
    freeze path has no matched-nothing check."""
    model = nn.Sequential(nn.Linear(2, 2))
    guard(model, ["model.layers.30.*"], "freeze_layers_patterns")

    setattr(model, PP_STAGE_PARTITION_ATTR, (0, 4))
    for flag in ("freeze_layers_patterns", "unfreeze_layers_patterns"):
        with pytest.raises(ValueError, match="re-based to index 0"):
            guard(model, ["model.layers.30.*"], flag)
    guard(model, ["*.self_attn.sinks", "score"], "freeze_layers_patterns")


def test_layer_ranges_written_as_glob_character_classes_are_rejected():
    """``"5[6-9]".isdigit()`` is False, so a per-segment check reads these as index-free; a sibling ``*.o_proj*``
    keeps the matched-nothing raise quiet."""
    model = nn.Sequential(nn.Linear(2, 2))
    setattr(model, PP_STAGE_PARTITION_ATTR, (0, 47))
    for pattern in ("model.layers.5[6-9].*", "model.layers.[6-8][0-9].*", "model.layers.9[0-3].*"):
        with pytest.raises(ValueError, match="re-based to index 0"):
            guard(model, [pattern], "unfreeze_layers_patterns")

    # A non-layer index and a wildcard are stage-invariant.
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
    """A family that ships no text-only CausalLM sibling."""
    torch.manual_seed(0)
    config = Glm5NextConfig(
        text_config=dict(TINY_GLM5_CONFIG), vision_config=dict(TINY_GLM5_VISION_CONFIG), attn_implementation="sdpa"
    )
    return Glm5NextForConditionalGeneration(config)


def _drive_composite_split(train_dataset=None, eval_dataset=None, data_collator=None):
    """As stage 0 of a pp2 x dp2 topology."""
    mixin = _pipeline_stub(parallelism_config=_pc(pp_size=2), save_sharded_ep=False, _moe_balancing="none")
    kwargs = {"model": _tiny_composite(), "train_dataset": train_dataset, "eval_dataset": eval_dataset}
    if data_collator is not None:
        kwargs["data_collator"] = data_collator
    return mixin, PipelineTrainerMixin._split_pipeline_stage(mixin, kwargs, _pp_args())


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
    """Tower and projector sit in no stage, so placeholder tokens would train as text. Driven through the split
    since ``is_vlm_model`` passes with the gate deleted."""
    kwargs = {key: (Dataset.from_dict(value) if isinstance(value, dict) else value) for key, value in evidence.items()}
    with pytest.raises(ValueError, match="Vision-language training is not supported") as err:
        _drive_composite_split(**kwargs)
    assert "means this run feeds images" in str(err.value)


def test_text_only_run_of_a_multimodal_wrapper_splits_and_keeps_its_vision_tower_for_the_save():
    """The tensors no stage holds are stashed for the PP save to re-emit, so the export keeps the wrapper layout."""
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
    # At the artifact's save dtype (the wrapper's own norm keep-set applies), on the host.
    cast = save_dtype_caster(composite)
    assert all(torch.equal(stash[k], cast(k, reference[k])) and stash[k].device.type == "cpu" for k in reference)
    assert any(stash[k].dtype != reference[k].dtype for k in reference), "the save-dtype cast never applied"


def test_plain_causal_lm_drops_nothing_at_the_split():
    """Anti-vacuity: no PP save of a plain causal LM ever writes a wrapper part."""
    assert wrapper_state_outside_stages(Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG))) == {}


def test_sharded_ep_save_is_rejected_under_pp():
    """The per-rank format has no stage layer offset, so two stages write ``layers.0.*`` shards that collide in the
    merged index."""
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
        PipelineTrainerMixin._split_pipeline_stage(sharded, {"model": model}, args)

    # The gathered save gets past this gate (it fails later, for other reasons).
    gathered = _pipeline_stub(parallelism_config=SimpleNamespace(is_pp_mode=True), save_sharded_ep=False)
    try:
        PipelineTrainerMixin._split_pipeline_stage(gathered, {"model": model}, args)
    except Exception as exc:  # noqa: BLE001 — any later failure is fine; the sharded-save one is not
        assert "save_sharded_ep" not in str(exc), f"the gathered save tripped the PP sharded-save gate: {exc}"


@pytest.mark.parametrize(
    ("eval_strategy", "eval_on_start"),
    [pytest.param("steps", False, id="eval_strategy"), pytest.param("no", True, id="eval_on_start_only")],
)
def test_eval_batch_mismatch_is_rejected_whenever_an_evaluation_runs(eval_strategy, eval_on_start):
    """The P2P buffers freeze to the training microbatch, and ``eval_on_start`` evaluates under ``eval_strategy:
    no`` too."""
    stub = _pipeline_stub(parallelism_config=SimpleNamespace(is_pp_mode=True), save_sharded_ep=False)
    args = _pp_args(eval_strategy=eval_strategy, eval_on_start=eval_on_start, per_device_eval_batch_size=4)
    with pytest.raises(ValueError, match=r"per_device_eval_batch_size \(4\) must equal per_device_train_batch_size"):
        PipelineTrainerMixin._split_pipeline_stage(stub, {}, args)


@pytest.mark.parametrize(
    ("output_shared", "local_world_size", "refused"),
    [("0", "2", True), ("0", "4", False), ("1", "2", False)],
    ids=["per-node-across-nodes", "per-node-single-node", "shared-across-nodes"],
)
def test_push_to_hub_is_refused_only_where_nodes_hold_different_stages(
    monkeypatch, output_shared, local_world_size, refused
):
    """The Trainer uploads from global rank 0, so per-node storage would publish an index naming files the Hub repo
    never receives."""
    monkeypatch.setenv("DIST_OUTPUT_SHARED_FILESYSTEM", output_shared)
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", local_world_size)
    stub = _pipeline_stub(parallelism_config=SimpleNamespace(is_pp_mode=True), save_sharded_ep=False)
    if refused:
        with pytest.raises(ValueError, match="push_to_hub is not supported under pipeline parallelism"):
            PipelineTrainerMixin._split_pipeline_stage(stub, {}, _pp_args(push_to_hub=True))
    else:
        with pytest.raises(ValueError, match="requires the model to be passed"):
            PipelineTrainerMixin._split_pipeline_stage(stub, {}, _pp_args(push_to_hub=True))


def test_eval_batch_mismatch_passes_when_no_evaluation_runs():
    stub = _pipeline_stub(parallelism_config=SimpleNamespace(is_pp_mode=True), save_sharded_ep=False)
    with pytest.raises(ValueError, match="requires the model to be passed"):
        PipelineTrainerMixin._split_pipeline_stage(stub, {}, _pp_args(per_device_eval_batch_size=4))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
