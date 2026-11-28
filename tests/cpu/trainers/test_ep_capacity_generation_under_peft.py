#!/usr/bin/env python
"""The EP capacity generation advances exactly once per forward of the model the training loop calls, whatever wraps
the EP-patched model. EP patching hooks the model it patched and trainer construction the top-level model; a hooked
forward inside another opens no scope, so a wrapper reaching the patched model through ``.forward()``, which runs none
of its hooks, still advances it once, and one calling it as a module does not advance it twice. Frozen, the generation
keeps the first forward's capacity until a longer batch fails in the dedup guard."""

from __future__ import annotations

import inspect

import pytest
import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from tokenizers import Tokenizer, models, pre_tokenizers
from torch import nn
from transformers import PreTrainedTokenizerFast, Trainer
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
from trl import SFTConfig

from halo.distributed.expert_parallel.deepep.capacity import capacity_scope, register_forward_generation_hook
from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.cp_model import unpatched_cp_wrapper
from tests.common.models import TINY_GEMMA4_MOE_CONFIG
from tests.common.parallelism import make_parallelism_config, single_process_ep_config

_INPUT_IDS = [3, 4, 5, 6]


def _tokenizer() -> PreTrainedTokenizerFast:
    """TRL demands a processing class; this run never tokenizes."""
    vocab = {f"t{index}": index for index in range(TINY_GEMMA4_MOE_CONFIG["vocab_size"])}
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="t0"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="t0", pad_token="t1", eos_token="t2")


def _ep_patched_model() -> Gemma4ForCausalLM:
    torch.manual_seed(0)
    model = Gemma4ForCausalLM(Gemma4TextConfig(**TINY_GEMMA4_MOE_CONFIG))
    apply_ep_layers(model, single_process_ep_config(TINY_GEMMA4_MOE_CONFIG["num_experts"]))
    # A zero-patch is only a warning at ep_group_size 1, and nothing below would notice.
    assert [name for name, module in model.named_modules() if isinstance(module, EPMoELayerBase)]
    return model


class _ForwardCaller(nn.Module):
    """A wrapper reaching the model it holds through ``.forward()``, which runs none of that model's hooks."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, **kwargs):
        return self.model.forward(**kwargs)

    def __getattr__(self, name):
        """The held model's attributes (``config``, the generation API) for the trainer, as the CP wrapper forwards."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "model":
                raise
            return getattr(self.model, name)


class _ModuleCaller(_ForwardCaller):
    def forward(self, **kwargs):
        return self.model(**kwargs)


_LORA = {"r": 4, "target_modules": ["q_proj", "v_proj"]}


def _peft(model: nn.Module, **task) -> PeftModel:
    return get_peft_model(model, LoraConfig(**_LORA, **task))


# Every layout a run's top-level model takes above the EP-patched model.
_LAYOUTS = {
    "ep_model": _ep_patched_model,
    "peft_typed": lambda: _peft(_ep_patched_model(), task_type="CAUSAL_LM"),
    "peft_untyped": lambda: _peft(_ep_patched_model()),
    "peft_over_cp": lambda: _peft(unpatched_cp_wrapper(_ep_patched_model()), task_type="CAUSAL_LM"),
    "cp_over_peft": lambda: unpatched_cp_wrapper(_peft(_ep_patched_model(), task_type="CAUSAL_LM")),
    "wrapper_calling_forward": lambda: _ForwardCaller(_ep_patched_model()),
    "wrapper_calling_module": lambda: _ModuleCaller(_ep_patched_model()),
}


def _trainer(tmp_path, model: nn.Module, peft_config: LoraConfig | None = None, **args) -> DistributedSFTTrainer:
    args = SFTConfig(
        output_dir=str(tmp_path),
        use_cpu=True,
        bf16=False,
        max_length=len(_INPUT_IDS),
        max_steps=1,
        per_device_train_batch_size=1,
        use_liger_kernel=False,
        remove_unused_columns=False,
        dataloader_num_workers=0,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        **{"gradient_checkpointing": False, **args},
    )
    return DistributedSFTTrainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_dict({"input_ids": [_INPUT_IDS] * 2, "labels": [_INPUT_IDS] * 2}),
        processing_class=_tokenizer(),
        parallelism_config=make_parallelism_config(world_size=1, gpus_per_node=1),
        peft_config=peft_config,
        moe_balancing="none",
    )


def _generation_delta(module: torch.nn.Module, forwards: int) -> int:
    before = capacity_scope.generation
    for _ in range(forwards):
        module(input_ids=torch.tensor([_INPUT_IDS]))
    return capacity_scope.generation - before


@pytest.mark.parametrize("layout", sorted(_LAYOUTS))
def test_the_top_level_model_advances_the_generation_once_per_forward(layout):
    """Registered as trainer construction registers it."""
    model = _LAYOUTS[layout]()
    register_forward_generation_hook(model)

    assert _generation_delta(model, forwards=2) == 2, (
        "the capacity generation did not advance exactly once per forward: frozen, it keeps the first forward's "
        "capacity; advanced twice, a hooked forward inside another opened a scope of its own"
    )


@pytest.mark.parametrize("layout", ["peft_config", "no_adapter", "wrapper_calling_forward", "wrapper_calling_module"])
def test_trainer_construction_hooks_the_model_the_loop_calls(tmp_path, layout):
    if layout == "peft_config":
        trainer = _trainer(tmp_path, _ep_patched_model(), LoraConfig(task_type="CAUSAL_LM", **_LORA))
        # TRL wrapped the patched model, and the loop calls that wrapper, not what it wraps.
        assert isinstance(trainer.model, PeftModel), type(trainer.model).__name__
    else:
        trainer = _trainer(tmp_path, _LAYOUTS["ep_model" if layout == "no_adapter" else layout]())
    assert "self._wrap_model(self.model_wrapped)" in inspect.getsource(Trainer._prepare_for_training)
    assert trainer.model_wrapped is trainer.model

    assert _generation_delta(trainer.model_wrapped, forwards=2) == 2


def test_a_recomputed_layer_opens_no_scope(tmp_path):
    """Gradient checkpointing reruns each decoder layer in the backward, outside every hooked forward."""
    trainer = _trainer(tmp_path, _peft(_ep_patched_model(), task_type="CAUSAL_LM"), gradient_checkpointing=True)
    model = trainer.model_wrapped.train()
    layer = next(module for module in model.modules() if isinstance(module, GradientCheckpointingLayer))
    runs = []
    # transformers checkpoints the layer's forward, so a submodule's hook runs again in the recompute.
    layer.self_attn.register_forward_pre_hook(lambda *_: runs.append(capacity_scope.generation))
    ids = torch.tensor([_INPUT_IDS])

    before = capacity_scope.generation
    model(input_ids=ids, labels=ids).loss.backward()

    assert len(runs) == 2, "premise: the backward recomputed the layer"
    assert capacity_scope.generation - before == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
