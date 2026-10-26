#!/usr/bin/env python
"""``state.total_flos`` keeps HF's estimate while the parameter walk behind it runs once per model.

HF's ``Trainer.floating_point_ops`` multiplies each micro-batch's input tokens by
``model.num_parameters(exclude_embeddings=True)``, which walks every module and parameter: 1.6 ms per
micro-batch on an 8B model, 7 ms with all-linear LoRA, exposed on host-bound steps. The spine's override must
return the same number for every batch, walk the model once, and walk again only for a different model.

Run: python tests/cpu/trainers/test_total_flos_estimate.py
"""

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, Trainer

from halo.trainers.sft import DistributedSFTTrainer


def _model(layers: int) -> Qwen3ForCausalLM:
    return Qwen3ForCausalLM(
        Qwen3Config(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            vocab_size=64,
        )
    )


def _counting_walks(model, monkeypatch) -> list[int]:
    walks = []
    walk = model.num_parameters

    def counted(*args, **kwargs):
        walks.append(1)
        return walk(*args, **kwargs)

    monkeypatch.setattr(model, "num_parameters", counted)
    return walks


def test_the_estimate_is_hfs_and_the_model_is_walked_once(monkeypatch):
    trainer = object.__new__(DistributedSFTTrainer)
    trainer.model = _model(layers=2)
    batches = [{"input_ids": torch.zeros(rows, 9, dtype=torch.long)} for rows in (1, 3, 2)]
    expected = [Trainer.floating_point_ops(trainer, batch) for batch in batches]
    walks = _counting_walks(trainer.model, monkeypatch)

    assert [trainer.floating_point_ops(batch) for batch in batches] == expected
    assert len(walks) == 1, "the parameter walk must run once, not per micro-batch"

    trainer.model = _model(layers=3)
    assert trainer.floating_point_ops(batches[0]) == Trainer.floating_point_ops(trainer, batches[0])
    assert trainer.floating_point_ops(batches[0]) > expected[0], "a different model is walked again"


def test_a_batch_without_the_main_input_counts_nothing():
    trainer = object.__new__(DistributedSFTTrainer)
    trainer.model = _model(layers=2)
    assert trainer.floating_point_ops({"inputs_embeds": torch.zeros(1, 4, 32)}) == 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
