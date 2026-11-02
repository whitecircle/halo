#!/usr/bin/env python
"""CPU tests for prefix-LM (PaliGemma) training attention: no trained token attends to a later token, the image
and prompt keep PaliGemma's bidirectional prefix, and no backend that cannot carry that mask is selected.

transformers sets PaliGemma's ``use_bidirectional_attention`` whenever a checkpoint leaves it unset, so its decoder
is non-causal wherever no mask is materialized, and the processor marks no suffix in a rendered conversation. A
trained answer that sees its own next tokens fits to a plausible loss and nothing downstream says so.

Run: ``python tests/cpu/models/test_prefix_lm_attention.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import contextlib
from unittest.mock import patch

import pytest
import torch
from transformers import Gemma3Config, GemmaConfig, GemmaForCausalLM, PaliGemmaForConditionalGeneration

from halo.model_loading.model_preparation import apply_run_model_policies
from halo.models.attention import selection as attention_mod
from halo.models.attention.gpt_oss_sinks import SinksPolicy
from tests.common.prefix_lm import (
    digit_row,
    first_scored_position,
    paligemma_collator,
    tiny_paligemma_config,
    tiny_paligemma_processor,
)

FA2, FA3, FA4, SDPA, FLEX, EAGER = (
    "flash_attention_2",
    "flash_attention_3",
    "flash_attention_4",
    "sdpa",
    "flex_attention",
    "eager",
)
_FLASH = (FA2, FA3, FA4)
_DETECTED = {"blackwell": FA4, "hopper": FA3}
# Logit movement that counts as attention reaching a token; untouched positions move by exactly 0.
_REACHED = 1e-4
# The in-vocabulary token (``The``) a perturbation swaps in.
_SWAP_IN = 9


def _tiny(text_model_type: str = "gemma"):
    processor = tiny_paligemma_processor()
    return processor, tiny_paligemma_config(processor, text_model_type=text_model_type)


def _training_model(config, attn_implementation: str):
    """A random-init PaliGemma on ``attn_implementation`` after the run policies every training load applies."""
    torch.manual_seed(0)
    model = PaliGemmaForConditionalGeneration._from_config(config, attn_implementation=attn_implementation).eval()
    apply_run_model_policies(
        model, config, sinks_policy=SinksPolicy.NEUTRALIZED, attn_implementation=attn_implementation
    )
    return model


def _logits(model, batch: dict, **overrides) -> torch.Tensor:
    inputs = {key: value for key, value in batch.items() if key != "labels"}
    inputs.update(overrides)
    with torch.no_grad():
        return model(**{key: value for key, value in inputs.items() if value is not None}).logits[0]


def _movement(model, batch: dict, position: int, **overrides) -> torch.Tensor:
    """Per-position max logit change when token ``position`` is swapped, under ``overrides``."""
    swapped = batch["input_ids"].clone()
    swapped[0, position] = _SWAP_IN
    before = _logits(model, batch, **overrides)
    after = _logits(model, batch, **{**overrides, "input_ids": swapped})
    return (before - after).abs().amax(dim=-1)


_BACKENDS = [("gemma", EAGER), ("gemma", SDPA), ("gemma2", EAGER), ("gemma2", FLEX)]


@pytest.mark.parametrize(("text_model_type", "backend"), _BACKENDS, ids=[f"{t}-{b}" for t, b in _BACKENDS])
def test_no_trained_token_attends_to_a_later_token(text_model_type, backend):
    """The SFT batch on every backend the selector can pick for the family: swapping the second-to-last answer token
    moves no earlier logit, while swapping a prompt token still moves an earlier prefix position (PaliGemma's
    bidirectional prefix is kept, not flattened to causal)."""
    processor, config = _tiny(text_model_type)
    batch = paligemma_collator(processor, config, completions_only=True)([digit_row()])
    model = _training_model(config, backend)
    length = batch["input_ids"].shape[1]
    start = first_scored_position(batch["labels"][0])

    assert _movement(model, batch, length - 2)[: length - 2].max() == 0
    prompt_token = start - 2
    assert _movement(model, batch, prompt_token)[1] > _REACHED, "the prefix lost its bidirectional attention"
    # Anti-vacuity: the processor's all-zero token_type_ids put the answer in the prefix, and the probe sees it.
    all_prefix = torch.zeros_like(batch["token_type_ids"])
    assert _movement(model, batch, length - 2, token_type_ids=all_prefix)[start : length - 2].max() > _REACHED


@pytest.mark.parametrize("backend", [EAGER, SDPA])
def test_a_batch_without_token_type_ids_scores_causally(backend):
    """A text row (no ``token_type_ids``, no padding) after the training policies: no position sees a later one."""
    processor, config = _tiny()
    batch = paligemma_collator(processor, config, completions_only=False)([digit_row()])
    batch["token_type_ids"] = None
    length = batch["input_ids"].shape[1]
    assert _movement(_training_model(config, backend), batch, length - 2)[: length - 2].max() == 0


def test_the_untouched_sdpa_decoder_is_bidirectional():
    """Anti-vacuity for the causal default: transformers builds PaliGemma's decoder non-causal, so an unpadded SDPA
    batch without ``token_type_ids`` reaches later tokens until the training policies run."""
    processor, config = _tiny()
    batch = paligemma_collator(processor, config, completions_only=False)([digit_row()])
    torch.manual_seed(0)
    model = PaliGemmaForConditionalGeneration._from_config(config, attn_implementation=SDPA).eval()
    length = batch["input_ids"].shape[1]
    assert _movement(model, batch, length - 2, token_type_ids=None)[: length - 2].max() > _REACHED


def test_the_causal_default_reaches_the_decoder_only():
    processor, config = _tiny()
    model = _training_model(config, EAGER)
    decoder = [m.is_causal for m in model.model.language_model.modules() if hasattr(m, "is_causal")]
    tower = [m.is_causal for m in model.model.vision_tower.modules() if hasattr(m, "is_causal")]
    assert decoder and all(decoder)
    assert tower and not any(tower), "the SigLIP tower attends bidirectionally"


def test_a_bidirectional_decoder_of_another_family_is_left_alone():
    """Anti-vacuity: the default is keyed on the prefix-LM family, not on ``use_bidirectional_attention``, which an
    encoder-style Gemma sets on purpose."""
    config = GemmaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        use_bidirectional_attention=True,
    )
    model = GemmaForCausalLM(config)
    apply_run_model_policies(model, config, sinks_policy=SinksPolicy.NEUTRALIZED, attn_implementation=EAGER)
    assert not model.model.layers[0].self_attn.is_causal


@contextlib.contextmanager
def _on(arch: str):
    """A host whose GPU is ``arch`` and whose flash builds all import."""
    with (
        patch.object(attention_mod, "_detect_attention_impl", lambda: _DETECTED[arch]),
        patch.object(attention_mod, "is_blackwell_gpu", lambda: arch == "blackwell"),
        patch.object(attention_mod, "is_hopper_gpu", lambda: arch == "hopper"),
        patch.object(attention_mod, "is_flash_attn_2_available", lambda: True),
        patch.object(attention_mod, "_flash_build_imports", lambda _impl: True),
        patch.object(attention_mod.logger, "info"),
    ):
        yield


def _resolve(config, request) -> str:
    return attention_mod.resolve_attn_implementation(
        config, request, torch.bfloat16, sinks_policy=SinksPolicy.NEUTRALIZED
    )


# Without a softcap SDPA carries the materialized mask; PaliGemma 2's softcap rules SDPA out, so flex carries both.
_MASK_CARRIER = {"gemma": SDPA, "gemma2": FLEX}


@pytest.mark.parametrize("arch", ["blackwell", "hopper"])
@pytest.mark.parametrize("request_", [None, *_FLASH, SDPA], ids=lambda r: str(r))
@pytest.mark.parametrize("text_model_type", ["gemma", "gemma2"])
def test_a_prefix_lm_never_resolves_to_a_flash_kernel(text_model_type, request_, arch):
    """Auto-detected or pinned: no flash kernel reads ``token_type_ids``, and PaliGemma's decoder is built
    non-causal, so a flash run would train every answer bidirectionally."""
    _, config = _tiny(text_model_type)
    with _on(arch), patch.object(attention_mod.logger, "warning"):
        assert _resolve(config, request_) == _MASK_CARRIER[text_model_type]


@pytest.mark.parametrize("request_", [FLEX, EAGER])
def test_a_mask_carrying_pin_on_a_prefix_lm_stands(request_):
    _, config = _tiny()
    with _on("blackwell"):
        assert _resolve(config, request_) == request_


def test_the_flash_demotion_names_the_prefix_lm():
    _, config = _tiny()
    with _on("blackwell"), patch.object(attention_mod.logger, "warning") as warn:
        _resolve(config, FA4)
    assert any("prefix-LM" in call.args[0] for call in warn.call_args_list)


def test_a_multimodal_gemma_that_is_no_prefix_lm_keeps_flash():
    """Anti-vacuity: the refusal is keyed on the prefix-LM family, so Gemma 3's VLM keeps its flash pick."""
    with _on("hopper"), patch.object(attention_mod.logger, "warning"):
        assert _resolve(Gemma3Config(), None) in _FLASH


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
