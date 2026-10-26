#!/usr/bin/env python
"""CPU tests for three PEFT seams on ``DistributedTrainerMixin``:

* ``reject_unmergeable_expert_lora_save`` — the flag must be rejected exactly where it cannot deliver a
  merged checkpoint, and nowhere else. All three gates read the LIVE model rather than a mode list,
  so every parallelism/model combination is covered by construction: no native expert adapters means
  there is nothing to fold, accelerate-managed FSDP means the base Trainer owns the save, and a PEFT
  layer the out-of-place fold cannot reproduce would otherwise raise only at the first checkpoint
  save. A mixed attention+expert run is the flag's headline case and must pass —
  ``save_ep_gathered_checkpoint`` folds both halves — so a blanket rejection of the mixed shape fails here.
* ``save_model``'s routing — passing that guard is only half the promise. With the flag set, a mixed
  run must skip ``save_peft_adapters`` (which never merges) and reach the EP strategy's merged write
  with the attention delta already folded into the base weights being gathered. This is what makes
  "re-save with ``merge_expert_lora_on_save=True``" a real export route for the mixed shape rather
  than an adapter file no tool folds, and it is the behavior
  ``halo.checkpoint.adapters.assert_no_expert_lora_adapter`` points users at.
* the QLoRA post-backward sweep — it must honor
  ``parallelism_config.fp32_grad_reduce`` (the QLoRA path bypasses FSDP2's ``reduce_dtype``, so
  dropping the flag silently reduces adapter grads in bf16).

Run: ``python tests/cpu/peft/test_merge_expert_lora_save_guard.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from peft import LoraConfig, get_peft_model
from torch import nn

import halo.checkpoint.parallel.ep as ep_mod
import halo.checkpoint.parallel.save as save_mod
import halo.trainers.checkpointing as checkpointing_mod
import halo.trainers.parallelism.adapter_rules as adapter_rules
import halo.trainers.parallelism.grad_sync as grad_sync_mod
import halo.trainers.parallelism.mode_setup as mode_setup_mod
from halo.checkpoint.reading import load_full_state_dict
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from halo.trainers.parallelism.adapter_rules import reject_unmergeable_expert_lora_save
from tests.common.adapters import randomize_adapters


class _TinyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(8, 8, bias=False)

    def forward(self, x):
        return self.q_proj(x)


def _attention_peft_model():
    return get_peft_model(_TinyLM(), LoraConfig(target_modules=["q_proj"]))


class _StubTrainer:
    _accelerate_manages_fsdp = False

    def __init__(self, model, merge_expert_lora_on_save):
        self.model = model
        self.parallelism_config = SimpleNamespace(
            merge_expert_lora_on_save=merge_expert_lora_on_save, fp32_grad_reduce=False
        )

    _install_qlora_grad_sync = DistributedTrainerMixin._install_qlora_grad_sync
    _sync_qlora_grads = DistributedTrainerMixin._sync_qlora_grads

    state = SimpleNamespace(global_step=0)

    def _top_level_model(self):
        return self.model


def _reject(trainer: _StubTrainer) -> None:
    reject_unmergeable_expert_lora_save(
        trainer.model,
        merge_expert_lora_on_save=trainer.parallelism_config.merge_expert_lora_on_save,
        accelerate_manages_fsdp=trainer._accelerate_manages_fsdp,
    )


def test_merge_flag_with_attention_peft_passes():
    """A MIXED run is the flag's headline case: the gathered EP save folds the expert deltas in the
    family gather and the attention deltas via merge_adapter, so it must NOT be rejected."""
    trainer = _StubTrainer(_attention_peft_model(), merge_expert_lora_on_save=True)
    with patch.object(adapter_rules, "model_has_expert_lora", return_value=True):
        _reject(trainer)


def test_merge_flag_without_expert_adapters_raises():
    """Nothing to fold: dense runs, MoE runs naming no expert projection, and expert_tp_size > 1
    (where EPConfig rejects expert LoRA) all land here, and the flag would be silently ignored."""
    trainer = _StubTrainer(_attention_peft_model(), merge_expert_lora_on_save=True)
    with patch.object(adapter_rules, "model_has_expert_lora", return_value=False):
        with pytest.raises(ValueError, match="no native grouped expert"):
            _reject(trainer)


def test_merge_flag_under_accelerate_fsdp_raises():
    """Flat-param shards: the base Trainer owns that save, so the merged gather would write shard views."""
    trainer = _StubTrainer(_attention_peft_model(), merge_expert_lora_on_save=True)
    trainer._accelerate_manages_fsdp = True
    with patch.object(adapter_rules, "model_has_expert_lora", return_value=True):
        with pytest.raises(ValueError, match="accelerate-managed FSDP"):
            _reject(trainer)


class _UnfoldableLM(nn.Module):
    """A foldable ``q_proj`` beside the two layer kinds whose merge the out-of-place fold has no formula for."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 8)
        self.mha = nn.MultiheadAttention(8, 2)
        self.q_proj = nn.Linear(8, 8, bias=False)

    def forward(self, x):  # pragma: no cover - never called
        return self.q_proj(x)


@pytest.mark.parametrize(
    "config",
    [
        {"target_modules": ["q_proj"], "trainable_token_indices": {"embed_tokens": [0, 1]}},
        {"target_modules": ["q_proj", "mha"]},
    ],
    ids=["trainable_tokens", "multihead_attention"],
)
def test_merge_flag_with_a_layer_the_fold_cannot_reproduce_raises(config):
    """The merged save folds the attention half out of place and would raise on these at the first
    checkpoint, after the run has trained; construction refuses them instead."""
    trainer = _StubTrainer(get_peft_model(_UnfoldableLM(), LoraConfig(**config)), merge_expert_lora_on_save=True)
    with patch.object(adapter_rules, "model_has_expert_lora", return_value=True):
        with pytest.raises(ValueError, match="cannot be folded out of place"):
            _reject(trainer)


def test_merge_flag_expert_only_run_passes():
    """Expert-only native EP grouped-LoRA has no PeftModel wrapper — the merged-save path works."""
    with patch.object(adapter_rules, "model_has_expert_lora", return_value=True):
        _reject(_StubTrainer(_TinyLM(), merge_expert_lora_on_save=True))


def test_no_merge_flag_skips_every_check():
    """Flag off: adapter-only saves, so neither gate applies even with no expert adapters at all."""
    with patch.object(adapter_rules, "model_has_expert_lora", return_value=False):
        _reject(_StubTrainer(_attention_peft_model(), merge_expert_lora_on_save=False))


# --------------------------------------------------------------------------- #
# save_model routing: the flag the merge guard's remedy names must reach the fold
# --------------------------------------------------------------------------- #


class _SaveRoutingTrainer(_StubTrainer):
    """A mixed EP+LoRA run, carrying only the DATA ``save_model`` reads.

    Every method below is the real mixin implementation, so the routing decision under test — and
    the context it reads it from — is the production one.
    """

    def _ep_layers(self):
        return [object()]

    _fsdp_wrapped = True
    save_sharded_ep = False
    save_max_shard_size = None
    processing_class = None
    _pp_wrapper_state = None
    _writing_training_checkpoint = False

    def __init__(self, model, merge_expert_lora_on_save, output_dir):
        super().__init__(model, merge_expert_lora_on_save)
        self.parallelism_config.is_pp_mode = False
        self.parallelism_config.is_cp_mode = False
        self.parallelism_config.is_tp_mode = False
        self.parallelism_config.has_ep_group = False
        self.parallelism_config.tp_size = 1
        self.args = SimpleNamespace(output_dir=output_dir)
        self._device_mesh = None
        self.accelerator = SimpleNamespace(process_index=0)

    save_model = DistributedTrainerMixin.save_model
    _write_model_payload = DistributedTrainerMixin._write_model_payload
    _mark_model_save_collectives_done = DistributedTrainerMixin._mark_model_save_collectives_done
    _checkpoint_context = DistributedTrainerMixin._checkpoint_context
    _find_cp_wrapper = DistributedTrainerMixin._find_cp_wrapper
    _get_tp_rank = DistributedTrainerMixin._get_tp_rank


def _mixed_run_model():
    """The attention half of a mixed run, with a delta that is not zero.

    PEFT zero-inits ``lora_B``, so an untouched adapter merges to a no-op and the fold assertion
    below could not tell a merged save from one that never merged.
    """
    peft_model = _attention_peft_model()
    randomize_adapters(peft_model, lora_b_only=True)
    return peft_model


def _run_save(trainer):
    """Drive ``save_model`` with the adapter writer faked and the real merged EP save recorded.

    The record holds the merge flag the EP save saw, the LoRA-wrapped projection's weight it wrote, and
    the live weight after it, which is what distinguishes a merged write from an adapter-only one.
    """
    ep_calls: list[dict] = []
    wrapped = trainer.model.base_model.model.q_proj

    def _recorded_ep_save(ctx, output_dir):
        ep_mod.save_ep_gathered_checkpoint(ctx, output_dir)
        ep_calls.append(
            {
                "merge_lora": ctx.merges_expert_lora,
                "written_weight": load_full_state_dict(output_dir)["q_proj.weight"],
                "live_weight": wrapped.base_layer.weight.detach().clone(),
            }
        )

    adapter_saver = MagicMock()
    with (
        patch.object(checkpointing_mod, "model_has_expert_lora", return_value=True),
        patch.object(save_mod, "save_peft_adapters", adapter_saver),
        patch.object(save_mod, "save_ep_gathered_checkpoint", _recorded_ep_save),
    ):
        trainer.save_model(trainer.args.output_dir)
    return ep_calls, adapter_saver


def test_merged_save_routes_a_mixed_run_to_the_ep_merge_path(tmp_path):
    """The mixed shape's export route: the flag reaches the fold, with BOTH halves in it.

    ``save_peft_adapters`` never merges, so a mixed run that lands there gets the resume-only adapter
    file. The flag must route past it to ``save_ep_gathered_checkpoint``, which asks the gather for the
    expert deltas (``merge_lora``) and folds the attention adapters into the base weights it writes —
    and the assertion on the written weight is what proves that second half is not just a flag being
    passed: the base weight on disk carries the attention delta, the live one does not.
    """
    model = _mixed_run_model()
    trainer = _SaveRoutingTrainer(model, True, str(tmp_path))
    unmerged = model.base_model.model.q_proj.base_layer.weight.detach().clone()

    ep_calls, adapter_saver = _run_save(trainer)

    adapter_saver.assert_not_called()
    assert len(ep_calls) == 1, "the mixed run never reached the merged EP save"
    assert ep_calls[0]["merge_lora"] is True, "expert deltas would not be folded into the gather"
    written = ep_calls[0]["written_weight"]
    assert not torch.equal(written, unmerged.to(written.dtype)), (
        "the base weight on disk is the frozen one — the attention delta was not folded"
    )
    # Folded out of place: the live base is the frozen one during and after the save, and the
    # PeftModel stays unmerged, so the re-save the merge guard recommends is repeatable mid-run.
    assert torch.equal(ep_calls[0]["live_weight"], unmerged), "the save wrote the fold into the live base"
    wrapped = model.base_model.model.q_proj
    assert not wrapped.merged
    assert torch.equal(wrapped.base_layer.weight, unmerged)


def test_adapter_only_save_of_a_mixed_run_goes_to_the_peft_saver(tmp_path):
    """Flag off is the branch that produces the shape ``assert_no_expert_lora_adapter`` refuses.

    Without it the same mixed run writes an adapter file, which is exactly why the flag — not a
    retrain — is the remedy that message hands out.
    """
    trainer = _SaveRoutingTrainer(_mixed_run_model(), False, str(tmp_path))

    ep_calls, adapter_saver = _run_save(trainer)

    assert adapter_saver.call_count == 1
    assert not ep_calls


# --------------------------------------------------------------------------- #
# The QLoRA post-backward sweep must thread fp32_grad_reduce into reduce_grads_bucketed
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fp32_flag", [True, False])
def test_qlora_grad_sync_threads_fp32_grad_reduce(fp32_flag):
    """bf16 sums lose precision, so the reduce dtype must follow the run's knob, not a default."""
    trainer = _StubTrainer(_TinyLM(), merge_expert_lora_on_save=False)
    trainer.parallelism_config.fp32_grad_reduce = fp32_flag

    calls: list[dict] = []

    trainer._install_qlora_clip = lambda: None
    with patch.object(mode_setup_mod, "reject_fsdp_knobs_under_qlora"):
        trainer._install_qlora_grad_sync("data parallel")
    assert getattr(trainer, "_qlora_grad_sync", False), "the sweep was never enabled — the test is vacuous"

    trainer.model(torch.randn(2, 8)).sum().backward()

    with (
        patch.object(grad_sync_mod.dist, "all_reduce", side_effect=lambda t, op=None, group=None: None),
        patch.object(
            grad_sync_mod,
            "reduce_grads_bucketed",
            side_effect=lambda grads, **kw: calls.append({"fp32": kw.get("fp32"), "n": len(grads)}),
        ),
    ):
        trainer._sync_qlora_grads()

    assert calls and calls[0]["n"], "the sweep reduced nothing — the test is vacuous"
    assert all(c["fp32"] is fp32_flag for c in calls), (
        f"reduce_grads_bucketed called with fp32={calls[0]['fp32']}, expected {fp32_flag} "
        f"(fp32_grad_reduce not threaded through the QLoRA sweep)"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
