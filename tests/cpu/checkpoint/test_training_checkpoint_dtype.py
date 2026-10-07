#!/usr/bin/env python
"""A training checkpoint writes every tensor at its live dtype; an export casts to the save dtype.

fp32 master weights (``fp32_router``, ``fp32_experts``, ``fp32_non_ep_params``) rounded to bf16 in a
training checkpoint stay rounded through every load that reads them back exactly (Path A, the PP
stage load, an adapter restore), so the writers keep the live dtype when the trainer's
``_save_checkpoint`` is the caller (``CheckpointContext.training_checkpoint``) and cast to the save
dtype for the ``save_model`` export. Pinned here: the gathered writer the FSDP2 / CP / TP saves
share, the TP save's own hand-off to it, the EP gathered writer, the PP stage writer (on a real
gloo group), the hand-written adapter file, and the trainer seam that marks which save is which.

    python tests/cpu/checkpoint/test_training_checkpoint_dtype.py
"""

import dataclasses
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate import PartialState
from transformers import CONFIG_MAPPING, AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM, TrainerState

PartialState()  # save_model_config logs through accelerate's logger

from src.checkpoint.format import load_full_state_dict
from src.distributed.checkpoint.ep_save import save_ep_model
from src.distributed.checkpoint.peft import _adapter_file_state
from src.distributed.checkpoint.save import save_pp_checkpoint
from src.distributed.checkpoint.tp_save import save_tp_model
from src.distributed.checkpoint.write import chunked_saveable_tensors, stream_gathered_checkpoint
from src.distributed.pipeline_parallel.stage import build_pipeline_stage
from src.trainers.mixins.base import DistributedTrainerMixin
from tests.common.base_save import BaseTrainerSave
from tests.common.gloo import run_gloo_ranks
from tests.common.models import TINY_QWEN3_CONFIG
from tests.cpu.checkpoint.test_pp_sharded_save_roundtrip import PP_SIZE, WORLD_SIZE, _context, _parallelism_config

SHARD_SIZE = "64KB"
# The live parameters the tiny MoE trains as fp32 masters: a router, an attention projection and a
# fused expert bank, whose hub keys are the per-expert split the save reverts it to.
FP32_MASTERS = ("model.layers.0.mlp.gate.weight", "model.layers.0.self_attn.q_proj.weight")
FP32_EXPERTS = "model.layers.1.mlp.experts.gate_up_proj"
# One fp32 master in each of the two pipeline stages of the 8-layer tiny Qwen3.
PP_FP32_MASTERS = ("model.layers.0.self_attn.q_proj.weight", "model.layers.7.mlp.down_proj.weight")


def _tiny_moe_with_fp32_masters():
    config = CONFIG_MAPPING["qwen3_moe"]()
    config.hidden_size = 32
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.head_dim = 8
    config.num_hidden_layers = 2
    config.intermediate_size = 64
    config.moe_intermediate_size = 24
    config.num_experts = 4
    config.num_experts_per_tok = 2
    config.vocab_size = 128
    config.tie_word_embeddings = False
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config).to(torch.bfloat16)
    named = dict(model.named_parameters())
    for name in (*FP32_MASTERS, FP32_EXPERTS):
        # Off the bf16 grid, as a trained fp32 master is: a round trip through bf16 would move it.
        named[name].data = named[name].data.float() + 1e-5
    return model


def _tiny_dense_with_fp32_masters():
    """Identically initialized on every rank, as the PP replicas of one stage must be."""
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG, pad_token_id=0, eos_token_id=1)).to(torch.bfloat16)
    named = dict(model.named_parameters())
    for name in PP_FP32_MASTERS:
        named[name].data = named[name].data.float() + 1e-5
    return model


def _stream(model, output_dir, *, keep_live_dtype):
    stream_gathered_checkpoint(
        model,
        chunked_saveable_tensors(model, retain=True),
        output_dir,
        is_save_rank=True,
        max_shard_size=SHARD_SIZE,
        keep_live_dtype=keep_live_dtype,
    )


def _expert_keys(state: dict) -> list[str]:
    """The keys the fp32 ``gate_up_proj`` bank is written under: per-expert hub keys where the writer
    reverts the fusion (the gathered one), the live fused key where it does not (EP without EP layers)."""
    prefix = "model.layers.1.mlp.experts."
    suffixes = ("gate_proj.weight", "up_proj.weight", "gate_up_proj")
    return [key for key in state if key.startswith(prefix) and key.endswith(suffixes)]


@pytest.mark.parametrize("writer", ["gathered", "tp", "ep_gathered"])
def test_a_training_checkpoint_keeps_fp32_masters_and_an_export_casts_them(tmp_path, writer):
    model = _tiny_moe_with_fp32_masters()
    live = {name: param.detach().clone() for name, param in model.named_parameters()}
    for keep_live_dtype in (True, False):
        out = str(tmp_path / f"{writer}-{keep_live_dtype}")
        os.makedirs(out)
        if writer == "gathered":
            _stream(model, out, keep_live_dtype=keep_live_dtype)
        elif writer == "tp":
            save_tp_model(model, out, keep_live_dtype=keep_live_dtype)
        else:
            save_ep_model(model, out, keep_live_dtype=keep_live_dtype)
        state = load_full_state_dict(out)
        experts = _expert_keys(state)
        assert experts, "premise: the fp32 expert bank reached the checkpoint"
        if keep_live_dtype:
            for name in FP32_MASTERS:
                assert state[name].dtype == torch.float32 and torch.equal(state[name], live[name]), name
            # Decided on the tensor, not its name: a per-expert hub key names no live parameter.
            assert all(state[key].dtype == torch.float32 for key in experts), experts
        else:
            assert all(state[name].dtype == torch.bfloat16 for name in FP32_MASTERS)
            assert all(state[key].dtype == torch.bfloat16 for key in experts)
        # bf16 parameters are written as they live either way.
        assert state["model.embed_tokens.weight"].dtype == torch.bfloat16


def _pp_save_worker(rank: int, out_dir: str, training_checkpoint: bool) -> None:
    config = _parallelism_config()
    stage = build_pipeline_stage(_tiny_dense_with_fp32_masters(), config.pp_rank, PP_SIZE)
    if rank == 0:
        os.makedirs(out_dir)
    dist.barrier()
    save_pp_checkpoint(dataclasses.replace(_context(stage, config), training_checkpoint=training_checkpoint), out_dir)


def test_the_pp_stage_writer_keeps_fp32_masters_only_in_a_training_checkpoint(tmp_path):
    live = {name: param.detach().clone() for name, param in _tiny_dense_with_fp32_masters().named_parameters()}
    for training_checkpoint in (True, False):
        out = str(tmp_path / f"pp-{training_checkpoint}")
        run_gloo_ranks(_pp_save_worker, WORLD_SIZE, out, training_checkpoint)
        state = load_full_state_dict(out)
        for name in PP_FP32_MASTERS:
            if training_checkpoint:
                assert state[name].dtype == torch.float32 and torch.equal(state[name], live[name]), name
            else:
                assert state[name].dtype == torch.bfloat16, name
        assert state["model.embed_tokens.weight"].dtype == torch.bfloat16


def test_the_adapter_file_keeps_fp32_adapters_only_in_a_training_checkpoint():
    state = {
        "base_model.model.q_proj.lora_A.weight": torch.randn(2, 4),
        "base_model.model.gate.modules_to_save.default.e_score_correction_bias": torch.randn(4),
    }
    kept = _adapter_file_state(SimpleNamespace(training_checkpoint=True), state)
    assert all(kept[key] is tensor for key, tensor in state.items())
    exported = _adapter_file_state(SimpleNamespace(training_checkpoint=False), state)
    assert exported["base_model.model.q_proj.lora_A.weight"].dtype == torch.bfloat16
    assert exported["base_model.model.gate.modules_to_save.default.e_score_correction_bias"].dtype == torch.float32


class _Trainer(DistributedTrainerMixin, BaseTrainerSave):
    """The mixin's ``_save_checkpoint`` and ``_checkpoint_context`` over HF's save, which saves the model
    through ``save_model``; ``save_model`` records the context each save is handed. The base's
    optimizer write after it fails on ``base_save_fails``."""

    _has_ep_layers = False

    def __init__(self, run_dir, *, base_save_fails=False):
        self.run_dir = run_dir
        self.base_save_fails = base_save_fails
        self.model = nn.Linear(2, 2)
        self.args = SimpleNamespace(save_total_limit=None, save_only_model=False, should_save=True, push_to_hub=False)
        self.state = TrainerState(global_step=2)
        self.parallelism_config = SimpleNamespace(
            is_pp_mode=False, is_cp_mode=False, is_tp_mode=False, is_ep_tp_mode=False, merge_expert_lora_on_save=False
        )
        self._fsdp_wrapped = False
        self._accelerate_manages_fsdp = False
        self.save_sharded_ep = False
        self.lr_scheduler = None
        self.saves: list[bool] = []

    def _top_level_model(self):
        return self.model

    def _find_cp_wrapper(self):
        return None

    def save_model(self, output_dir=None, _internal_call=False):
        self.saves.append(self._checkpoint_context().training_checkpoint)
        self._model_save_collectives_done = True

    def _save_optimizer_and_scheduler(self, output_dir):
        if self.base_save_fails:
            raise OSError(28, "No space left on device")


def test_the_checkpoint_save_is_a_training_checkpoint_and_the_export_is_not(tmp_path):
    trainer = _Trainer(str(tmp_path))
    trainer._save_checkpoint(model=None, trial=None)
    trainer.save_model(str(tmp_path / "export"))
    assert trainer.saves == [True, False]


def test_a_failed_checkpoint_save_leaves_later_exports_casting(tmp_path):
    trainer = _Trainer(str(tmp_path), base_save_fails=True)
    with pytest.raises(RuntimeError, match="No space left on device"):
        trainer._save_checkpoint(model=None, trial=None)
    trainer.save_model(str(tmp_path / "export"))
    assert trainer.saves == [True, False]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
