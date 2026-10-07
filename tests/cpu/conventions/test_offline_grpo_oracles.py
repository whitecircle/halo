#!/usr/bin/env python
"""The offline-GRPO GPU suites' shared oracles grade the real trainer, on CPU.

The suites compare the trainer's run-start reference sweep and its pure-KL loss with independent
full-logits scores (:mod:`tests.common.offline_grpo`). Here the same helpers run against a real
single-process trainer: they must agree with it, reject a wrong objective, leave the policy as they
found it, and read a resumed export's move at the export's dtype.
"""

from types import SimpleNamespace

import pytest
import torch
from accelerate import PartialState
from safetensors.torch import save_file
from transformers import Qwen3Config, Qwen3ForCausalLM

from src.data.collators.offline_grpo import REF_PER_TOKEN_LOGPS_COLUMN
from src.distributed.parallelism_config import ParallelismConfig
from tests.common.offline_grpo import (
    OFFLINE_VOCAB,
    doubled_head_kl_verdict,
    full_logits_kl,
    offline_grpo_config,
    offline_grpo_dataset,
    offline_grpo_trainer,
    pure_kl_batch,
    reference_oracle_rows,
    resumed_export_verdict,
    swept_reference_error,
)
from tests.common.tolerances import TOL

PartialState()

BETA = 0.2


def _tiny_policy(seed):
    torch.manual_seed(seed)
    config = Qwen3Config(
        vocab_size=len(OFFLINE_VOCAB),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        pad_token_id=OFFLINE_VOCAB["<pad>"],
        eos_token_id=OFFLINE_VOCAB["<eos>"],
        tie_word_embeddings=False,
    )
    return Qwen3ForCausalLM(config)


def _cpu_trainer(tmp_path, name, *, beta, finalizers):
    args = offline_grpo_config(
        str(tmp_path / name),
        steps=1,
        save_steps=1,
        seed=0,
        kl_beta=beta,
        sequence_length=16,
        evaluate=False,
        save=False,
        use_cpu=True,
        bf16=False,
        gradient_checkpointing=False,
        dataset_num_proc=1,
    )
    ctx = SimpleNamespace(on_teardown=finalizers.append)
    return offline_grpo_trainer(ctx, _tiny_policy(7), ParallelismConfig(), args, offline_grpo_dataset(4))


def test_the_shared_oracles_agree_with_the_trainer_and_reject_a_wrong_objective(tmp_path):
    finalizers = []
    expected = reference_oracle_rows(_cpu_trainer(tmp_path, "oracle", beta=0.0, finalizers=finalizers))
    trainer = _cpu_trainer(tmp_path, "train", beta=BETA, finalizers=finalizers)
    assert finalizers[-1] == trainer.cleanup_ep, "teardown must release the trainer's DeepEP buffers"
    assert swept_reference_error(trainer.train_dataset[REF_PER_TOKEN_LOGPS_COLUMN], expected) < TOL.logprob_atol

    head = trainer.model.get_output_embeddings().weight
    before = head.detach().clone()
    trainer.model.train()
    batch = pure_kl_batch(trainer)
    assert not batch["advantage"].any()
    nonzero, matches = doubled_head_kl_verdict(trainer, batch, lambda model: full_logits_kl(model, batch, BETA), "cpu")
    assert nonzero and matches
    _, wrong_beta_matches = doubled_head_kl_verdict(
        trainer, batch, lambda model: full_logits_kl(model, batch, BETA / 2), "cpu control"
    )
    assert not wrong_beta_matches, "the oracle comparison cannot tell a halved KL coefficient apart"
    assert trainer.model.get_output_embeddings().weight is head and torch.equal(head.detach(), before)
    assert trainer.model.training


def _export(path, state):
    path.mkdir()
    model = _tiny_policy(7).to(torch.bfloat16)
    model.load_state_dict(state)
    model.save_pretrained(path)
    return path


def test_the_resumed_export_verdict_reads_a_move_at_the_export_dtype(tmp_path):
    state = {name: tensor.detach().clone() for name, tensor in _tiny_policy(7).to(torch.bfloat16).state_dict().items()}
    continuous = _export(tmp_path / "continuous", state)
    resumed = _export(tmp_path / "resumed", state)
    # The checkpoint stores fp32 masters a bf16 round trip loses; at the export dtype nothing moved.
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    masters = {name: tensor.float() * (1 + 2**-10) for name, tensor in state.items()}
    assert all(torch.equal(masters[name].bfloat16(), tensor) for name, tensor in state.items())
    assert any(not torch.equal(master, master.bfloat16().float()) for master in masters.values())
    save_file(masters, str(checkpoint / "model.safetensors"))
    assert resumed_export_verdict(continuous, resumed, checkpoint, "cpu") == (True, False, True)

    moved = dict(state)
    moved["lm_head.weight"] = state["lm_head.weight"] * 2
    diverged = _export(tmp_path / "diverged", moved)
    assert resumed_export_verdict(continuous, diverged, checkpoint, "cpu") == (False, True, True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
