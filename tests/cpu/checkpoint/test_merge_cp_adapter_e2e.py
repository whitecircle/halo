#!/usr/bin/env python
"""A CP-trained LoRA adapter must merge into a plain base model — the whole export path, on disk.

Context parallelism respells every adapter key: the Ulysses wrapper adds a ``model.`` level and an
``.original_attention.`` segment, so the tensors a CP run holds are not the tensors stock PEFT
resolves. :meth:`PeftAdapterSaver._normalize_cp_adapter_key` exists to undo that; unit coverage of
the normalization alone does not reach the claim it is FOR — that
``scripts/after_training/merge_peft_adapters.py`` can take the written directory and produce a
loadable, actually-adapted HF checkpoint. Every intermediate step can be individually correct while
the artifact is still unusable, and PEFT drops unresolved adapter keys *silently* — a merge that
matched nothing writes a perfectly valid checkpoint holding the untouched base weights.

So this runs the real script function over a real (tiny) base and asserts the merged weights equal
``base + (B @ A) * alpha/r`` on the targeted projections and are byte-identical to the base on the
untargeted ones. It also pins the two seams the merge owns for a CP run:

* ``base_model_name_or_path`` — ``get_peft_model`` reads it from the wrapped module's ``__dict__``,
  which under CP is the wrapper's (empty), so without the saver's backfill the adapter records
  ``null`` and the merge cannot find its base at all;
* the training sidecars — ``training_provenance.json`` and ``router_balancing_biases.pt`` live in
  the adapter directory, not in the base the save copies from, so the merge has to carry them
  forward itself and warn when a trained bias has no slot the merged model can serve it from.

    python tests/cpu/checkpoint/test_merge_cp_adapter_e2e.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch
from accelerate import PartialState
from peft import LoraConfig, get_peft_model
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from src.checkpoint.format import (
    ADAPTER_CONFIG_FILE,
    PROVENANCE_GPT_OSS_SINKS,
    ROUTER_BALANCING_BIASES_FILE,
    TRAINING_PROVENANCE_FILE,
)
from src.checkpoint.tool_io import apply_training_sidecars
from src.distributed.checkpoint.context import CheckpointContext
from src.distributed.checkpoint.peft import PeftAdapterSaver
from src.distributed.context_parallel.config import CPConfig
from src.distributed.context_parallel.validation import SUPPORTED_ATTN_IMPLEMENTATIONS
from src.distributed.context_parallel.wrapper import UlyssesCPModelWrapper
from src.models.patches.gpt_oss_sinks import SinksPolicy
from tests.common.utils import load_script_module

_TINY_QWEN3 = {
    "vocab_size": 64,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "max_position_embeddings": 64,
    "tie_word_embeddings": False,
}
_LORA_R = 4
_LORA_ALPHA = 8
_TARGETS = ["q_proj", "v_proj"]
_UNTARGETED = "k_proj"


def _load_merge_script():
    return load_script_module("scripts/after_training/merge_peft_adapters.py", "merge_peft_adapters_e2e")


def _tiny_tokenizer() -> PreTrainedTokenizerFast:
    """A real fast tokenizer built in-process: the merge refuses a checkpoint without one, and a hub
    download would make this test network-bound."""
    backend = Tokenizer(models.WordLevel({"<unk>": 0, "<eos>": 1, "hello": 2, "world": 3}, unk_token="<unk>"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", eos_token="<eos>", pad_token="<eos>")


def _build_base(base_dir: Path) -> None:
    torch.manual_seed(0)
    Qwen3ForCausalLM(Qwen3Config(**_TINY_QWEN3)).save_pretrained(base_dir)
    _tiny_tokenizer().save_pretrained(base_dir)


def _cp_context(model) -> CheckpointContext:
    """The save context a CP+LoRA run builds."""
    return CheckpointContext(
        model=model,
        parallelism_config=None,
        is_pp_mode=False,
        is_cp_mode=True,
        is_tp_mode=False,
        is_ep_tp_mode=False,
        has_ep_layers=False,
        fsdp_wrapped=False,
        accelerate_manages_fsdp=False,
        is_save_rank=True,
        max_shard_size="5GB",
        save_sharded_ep=False,
        has_expert_lora=False,
        merge_expert_lora_on_save=False,
        cp_wrapper=None,
        tokenizer=None,
    )


def _train_cp_adapter(base_dir: Path, adapter_dir: Path) -> dict[str, torch.Tensor]:
    """Train (perturb) a LoRA under the CP wrapper and save it the way the trainer does.

    ``cp_size=1`` still patches every attention layer, so the saved keys carry the real CP artifacts
    without needing a process group. Returns the per-target expected weight delta, keyed by the
    plain (non-CP) module path the merged checkpoint spells.
    """
    inner = Qwen3ForCausalLM.from_pretrained(base_dir, dtype=torch.float32)
    inner.config._attn_implementation = SUPPORTED_ATTN_IMPLEMENTATIONS[0]
    wrapper = UlyssesCPModelWrapper(inner, CPConfig(cp_size=1, world_size=1, gpus_per_node=1))
    peft_model = get_peft_model(wrapper, LoraConfig(r=_LORA_R, lora_alpha=_LORA_ALPHA, target_modules=_TARGETS))

    # lora_B initializes to zero, so an untrained adapter merges to a no-op and every check below
    # would pass on a merge that resolved nothing.
    torch.manual_seed(11)
    with torch.no_grad():
        for name, param in peft_model.named_parameters():
            if ".lora_B." in name:
                param.add_(torch.randn_like(param))

    # The adapter file stores LoRA at the bf16 save dtype, so the delta the merge can reconstruct is
    # the one built from the ROUNDED factors — comparing against the fp32 originals would fail by
    # ~1e-3 and say nothing about key resolution.
    deltas = {}
    scaling = _LORA_ALPHA / _LORA_R
    for layer_index, layer in enumerate(peft_model.base_model.model.model.model.layers):
        for target in _TARGETS:
            module = getattr(layer.self_attn.original_attention, target)
            lora_a = module.lora_A["default"].weight.detach().to(torch.bfloat16).float()
            lora_b = module.lora_B["default"].weight.detach().to(torch.bfloat16).float()
            deltas[f"model.layers.{layer_index}.self_attn.{target}.weight"] = (lora_b @ lora_a) * scaling

    assert PeftAdapterSaver().save(_cp_context(peft_model), peft_model, str(adapter_dir))
    return deltas


def _write_sidecars(adapter_dir: Path) -> None:
    """The two files a training run leaves beside an adapter, neither of which the base carries."""
    (adapter_dir / TRAINING_PROVENANCE_FILE).write_text(json.dumps({PROVENANCE_GPT_OSS_SINKS: SinksPolicy.LIVE}))
    torch.save(
        {"model.layers.0.mlp": torch.arange(4, dtype=torch.float32)}, adapter_dir / ROUTER_BALANCING_BIASES_FILE
    )


def test_cp_trained_adapter_merges_into_a_plain_model():
    """The portability contract, end to end: CP-spelled adapter in, adapted plain checkpoint out."""
    PartialState()
    merge_script = _load_merge_script()
    with tempfile.TemporaryDirectory() as tmp:
        base, adapter, out = Path(tmp) / "base", Path(tmp) / "adapter", Path(tmp) / "out"
        _build_base(base)
        deltas = _train_cp_adapter(base, adapter)

        merge_script.merge_peft_adapter(
            adapter_dir=str(adapter), output_dir=str(out), dtype=torch.float32, verbose=False
        )

        merged = Qwen3ForCausalLM.from_pretrained(out, dtype=torch.float32).state_dict()
        original = Qwen3ForCausalLM.from_pretrained(base, dtype=torch.float32).state_dict()

    assert deltas, "premise: no targeted projections were found"
    assert all(delta.abs().max() > 1e-3 for delta in deltas.values()), "premise: the adapter is a no-op"

    for key, delta in deltas.items():
        assert torch.allclose(merged[key], original[key] + delta, atol=1e-5), (
            f"{key} did not receive its CP-trained LoRA delta — PEFT resolved no key for it and "
            f"merged silently, so the 'merged' model still serves base weights"
        )
    untargeted = [key for key in original if key.endswith(f"{_UNTARGETED}.weight")]
    assert untargeted, "premise: no untargeted control projections"
    assert all(torch.equal(merged[key], original[key]) for key in untargeted), (
        "an untargeted projection changed — the merge is not measuring what it claims"
    )


def test_saved_cp_adapter_records_its_base_model():
    """Without the saver's backfill the CP adapter records ``base_model_name_or_path: null`` and the
    merge has no base to load — the whole path dies before the keys ever matter."""
    with tempfile.TemporaryDirectory() as tmp:
        base, adapter = Path(tmp) / "base", Path(tmp) / "adapter"
        _build_base(base)
        _train_cp_adapter(base, adapter)
        with open(adapter / ADAPTER_CONFIG_FILE) as fh:
            recorded = json.load(fh)["base_model_name_or_path"]
    assert recorded == str(base), f"adapter points at {recorded!r}"


def test_merge_carries_the_training_sidecars_and_warns_about_an_unservable_bias():
    """``save_full_checkpoint`` copies aux files from the BASE directory, so the adapter's own
    sidecars reach the merged model only through the merge's explicit carry. And a bias trained on
    an architecture with no checkpoint slot for it cannot be served: the merge must say so rather
    than write an artifact that quietly routes on the pretrained gate."""
    PartialState()
    merge_script = _load_merge_script()
    with tempfile.TemporaryDirectory() as tmp:
        base, adapter, out = Path(tmp) / "base", Path(tmp) / "adapter", Path(tmp) / "out"
        _build_base(base)
        _train_cp_adapter(base, adapter)
        _write_sidecars(adapter)

        merged_model = Qwen3ForCausalLM.from_pretrained(base, dtype=torch.float32)
        actions = apply_training_sidecars(merged_model, str(adapter))
        assert any("TRANSIENT" in action for action in actions), (
            f"a dense model has no balancing slot — the merge must warn, got {actions}"
        )

        merge_script.merge_peft_adapter(
            adapter_dir=str(adapter), output_dir=str(out), dtype=torch.float32, verbose=False
        )

        for name in (TRAINING_PROVENANCE_FILE, ROUTER_BALANCING_BIASES_FILE):
            assert (out / name).is_file(), f"{name} did not travel with the merged model"
        assert json.loads((out / TRAINING_PROVENANCE_FILE).read_text())[PROVENANCE_GPT_OSS_SINKS] == SinksPolicy.LIVE


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
