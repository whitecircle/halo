#!/usr/bin/env python
"""CPU tests for three PEFT seams on ``DistributedTrainerMixin``:

* ``_validate_merge_expert_lora_save`` — the flag must be rejected exactly where it cannot deliver a
  merged checkpoint, and nowhere else. Both gates read the LIVE model rather than a mode list, so
  every parallelism/model combination is covered by construction: no native expert adapters means
  there is nothing to fold, and accelerate-managed FSDP means the base Trainer owns the save. A mixed
  attention+expert run is the flag's headline case and must pass — ``save_ep_checkpoint`` folds
  both halves — so a blanket rejection of the mixed shape fails here.
* ``save_model``'s routing — passing that guard is only half the promise. With the flag set, a mixed
  run must skip ``PeftAdapterSaver`` (which never merges) and reach the EP strategy's merged write
  with the attention delta already folded into the base weights being gathered. This is what makes
  "re-save with ``merge_expert_lora_on_save=True``" a real export route for the mixed shape rather
  than an adapter file no tool folds, and it is the behavior
  ``src.distributed.checkpoint.peft._expert_lora_merge_remedy`` points users at.
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

import src.distributed.checkpoint.save as save_mod
import src.trainers.mixins.base as mixin_mod
import src.trainers.mixins.checkpointing as checkpointing_mod
import src.trainers.mixins.grad_sync as grad_sync_mod
from src.trainers.mixins.base import DistributedTrainerMixin


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

    _validate_merge_expert_lora_save = DistributedTrainerMixin._validate_merge_expert_lora_save
    _setup_qlora_gradient_sync = DistributedTrainerMixin._setup_qlora_gradient_sync
    _sync_qlora_grads = DistributedTrainerMixin._sync_qlora_grads

    state = SimpleNamespace(global_step=0)

    def _top_level_model(self):
        return self.model


def test_merge_flag_with_attention_peft_passes():
    """A MIXED run is the flag's headline case: save_ep_checkpoint folds the expert deltas in the
    family gather and the attention deltas via merge_adapter, so it must NOT be rejected."""
    trainer = _StubTrainer(_attention_peft_model(), merge_expert_lora_on_save=True)
    with patch.object(mixin_mod, "has_ep_lora", return_value=True):
        trainer._validate_merge_expert_lora_save()


def test_merge_flag_without_expert_adapters_raises():
    """Nothing to fold: dense runs, MoE runs naming no expert projection, and expert_tp_size > 1
    (where EPConfig rejects expert LoRA) all land here, and the flag would be silently ignored."""
    trainer = _StubTrainer(_attention_peft_model(), merge_expert_lora_on_save=True)
    with patch.object(mixin_mod, "has_ep_lora", return_value=False):
        with pytest.raises(ValueError, match="no native grouped expert"):
            trainer._validate_merge_expert_lora_save()


def test_merge_flag_under_accelerate_fsdp_raises():
    """Flat-param shards: the base Trainer owns that save, so the merged gather would write shard views."""
    trainer = _StubTrainer(_attention_peft_model(), merge_expert_lora_on_save=True)
    trainer._accelerate_manages_fsdp = True
    with patch.object(mixin_mod, "has_ep_lora", return_value=True):
        with pytest.raises(ValueError, match="accelerate-managed FSDP"):
            trainer._validate_merge_expert_lora_save()


def test_merge_flag_expert_only_run_passes():
    """Expert-only native EP grouped-LoRA has no PeftModel wrapper — the merged-save path works."""
    with patch.object(mixin_mod, "has_ep_lora", return_value=True):
        _StubTrainer(_TinyLM(), merge_expert_lora_on_save=True)._validate_merge_expert_lora_save()


def test_no_merge_flag_skips_every_check():
    """Flag off: adapter-only saves, so neither gate applies even with no expert adapters at all."""
    with patch.object(mixin_mod, "has_ep_lora", return_value=False):
        _StubTrainer(_attention_peft_model(), merge_expert_lora_on_save=False)._validate_merge_expert_lora_save()


# --------------------------------------------------------------------------- #
# save_model routing: the flag the merge guard's remedy names must reach the fold
# --------------------------------------------------------------------------- #


class _SaveRoutingTrainer(_StubTrainer):
    """A mixed EP+LoRA run, carrying only the DATA ``save_model`` reads.

    Every method below is the real mixin implementation, so the routing decision under test — and
    the context it reads it from — is the production one.
    """

    _has_ep_layers = True
    _fsdp_wrapped = True
    save_sharded_ep = False
    processing_class = None
    _pp_wrapper_state = None

    def __init__(self, model, merge_expert_lora_on_save, output_dir):
        super().__init__(model, merge_expert_lora_on_save)
        self.parallelism_config.is_pp_mode = False
        self.parallelism_config.is_cp_mode = False
        self.parallelism_config.is_tp_mode = False
        self.parallelism_config.is_ep_tp_mode = False
        self.parallelism_config.tp_size = 1
        self.args = SimpleNamespace(output_dir=output_dir, save_max_shard_size=None)
        self.parallel_dims = SimpleNamespace(tp_local_rank=lambda: 0)

    save_model = DistributedTrainerMixin.save_model
    _mark_model_save_collectives_done = DistributedTrainerMixin._mark_model_save_collectives_done
    _checkpoint_context = DistributedTrainerMixin._checkpoint_context
    _persist_router_balancing_biases = DistributedTrainerMixin._persist_router_balancing_biases
    _find_cp_wrapper = DistributedTrainerMixin._find_cp_wrapper
    _get_tp_rank = DistributedTrainerMixin._get_tp_rank


def _mixed_run_model():
    """The attention half of a mixed run, with a delta that is not zero.

    PEFT zero-inits ``lora_B``, so an untouched adapter merges to a no-op and the fold assertion
    below could not tell a merged save from one that never merged.
    """
    peft_model = _attention_peft_model()
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if "lora_B" in name:
                param.copy_(torch.randn_like(param))
    return peft_model


def _run_save(trainer):
    """Drive ``save_model`` with the two terminal writers faked, and report which one ran.

    ``save_ep_model`` records the LIVE weight of the LoRA-wrapped projection at call time — the
    tensor its gather would put on disk — which is what distinguishes a merged write from an
    adapter-only one.
    """
    ep_calls: list[dict] = []
    wrapped = trainer.model.base_model.model.q_proj

    def _fake_save_ep_model(_model, _output_dir, **kwargs):
        ep_calls.append({**kwargs, "written_weight": wrapped.base_layer.weight.detach().clone()})

    adapter_saver = MagicMock()
    with (
        patch.object(checkpointing_mod, "has_ep_lora", return_value=True),
        patch.object(checkpointing_mod, "PeftAdapterSaver", return_value=adapter_saver),
        patch.object(save_mod, "save_ep_model", _fake_save_ep_model),
    ):
        trainer.save_model(trainer.args.output_dir)
    return ep_calls, adapter_saver


def test_merged_save_routes_a_mixed_run_to_the_ep_merge_path(tmp_path):
    """The mixed shape's export route: the flag reaches the fold, with BOTH halves in it.

    ``PeftAdapterSaver`` never merges, so a mixed run that lands there gets the resume-only adapter
    file. The flag must route past it to ``save_ep_checkpoint``, which asks the gather for the
    expert deltas (``merge_lora``) while holding ``merged_adapters`` open (``adapters_merged``) — and
    the assertion on the live weight is what proves that second half is not just a flag being passed:
    the base weight the gather would write carries the attention delta.
    """
    model = _mixed_run_model()
    trainer = _SaveRoutingTrainer(model, True, str(tmp_path))
    unmerged = model.base_model.model.q_proj.base_layer.weight.detach().clone()

    ep_calls, adapter_saver = _run_save(trainer)

    adapter_saver.save.assert_not_called()
    assert len(ep_calls) == 1, "the mixed run never reached the merged EP save"
    assert ep_calls[0]["merge_lora"] is True, "expert deltas would not be folded into the gather"
    assert ep_calls[0]["adapters_merged"] is True
    assert not torch.equal(ep_calls[0]["written_weight"], unmerged), (
        "the base weight handed to the gather is the frozen one — the attention delta was not folded"
    )
    # Unmerged again afterwards, so the re-save the merge guard recommends is repeatable mid-run.
    # By state and by value, to a rounding step: (w+d)-d != w exactly, which is why merge_adapter is
    # paired with unmerge_adapter rather than merge_and_unload dissolving the PeftModel.
    wrapped = model.base_model.model.q_proj
    assert not wrapped.merged
    assert torch.allclose(wrapped.base_layer.weight, unmerged, atol=1e-6)


def test_adapter_only_save_of_a_mixed_run_goes_to_the_peft_saver(tmp_path):
    """Flag off is the branch that produces the shape ``assert_no_expert_lora_adapter`` refuses.

    Without it the same mixed run writes an adapter file, which is exactly why the flag — not a
    retrain — is the remedy that message hands out.
    """
    trainer = _SaveRoutingTrainer(_mixed_run_model(), False, str(tmp_path))

    ep_calls, adapter_saver = _run_save(trainer)

    assert adapter_saver.save.call_count == 1
    assert not ep_calls


# --------------------------------------------------------------------------- #
# PEFT-4: QLoRA grad-sync hooks must thread fp32_grad_reduce into reduce_grad
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("fp32_flag", [True, False])
def test_qlora_grad_sync_threads_fp32_grad_reduce(fp32_flag):
    """bf16 sums lose precision, so the reduce dtype must follow the run's knob, not a default."""
    trainer = _StubTrainer(_TinyLM(), merge_expert_lora_on_save=False)
    trainer.parallelism_config.fp32_grad_reduce = fp32_flag

    calls: list[dict] = []

    with patch.object(grad_sync_mod, "get_global_world_size", return_value=2):
        trainer._setup_qlora_gradient_sync()
    assert getattr(trainer, "_qlora_grad_sync", False), "the sweep was never enabled — the test is vacuous"

    trainer.model(torch.randn(2, 8)).sum().backward()

    with (
        patch.object(grad_sync_mod, "current_device", return_value=torch.device("cpu")),
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
