#!/usr/bin/env python
"""CPU test: an SMPO row scores alike whatever its batch's padding, because every row is flushed left.

The collator left-pads prompts, and leading pads move a row's real tokens off the positions and
indices they hold unpadded, which two of the roster's attentions read:

* Mistral4 scales its queries by the llama-4 factor of each token's absolute position;
* DeepSeek-V4's CSA/HCA compressors pool KV over windows cut at fixed tensor indices, with no pad mask,
  and judge causality per window, so no choice of ``position_ids`` realigns them.

``concatenated_inputs`` moves every pad behind its completion, after which a model's default positions
and windows are the unpadded row's. Pinned through the real ``concatenated_forward``:

* a ragged batch's left-padded pair scores the log-probs that pair scores alone, on a tiny Mistral4
  whose scale steps every 2 positions and on a tiny DeepSeek-V4 whose CSA windows are 4 tokens;
* the same for an image pair on a tiny Gemma 4, whose bidirectional image-block mask is built from the
  ``mm_token_type_ids`` the flush moves with their tokens;
* the CP layout scores the padded batch alike: one full-sequence forward with no mask and the
  ``arange`` positions the CP wrapper passes, the attention the Ulysses all-to-all reconstructs;
* the flush reads nothing back from the device, so it adds no host sync to a forward.

    python tests/cpu/trainers/test_smpo_trailing_pads.py
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from transformers import Gemma4Config, Gemma4ForConditionalGeneration
from transformers.models.mistral4 import Mistral4Config, Mistral4ForCausalLM

from src.trainers.preference.smpo import SmoothMarginPOTrainer
from tests.common.models import TINY_GEMMA4_MOE_CONFIG, TINY_MISTRAL4_CONFIG
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model

PAD_TOKEN_ID = 0
LOGP_KEYS = ("chosen_logps", "rejected_logps")


def _mistral4() -> nn.Module:
    rope = {
        **TINY_MISTRAL4_CONFIG["rope_parameters"],
        "original_max_position_embeddings": 2,
        "llama_4_scaling_beta": 0.5,
    }
    config = Mistral4Config(**{**TINY_MISTRAL4_CONFIG, "num_hidden_layers": 2, "rope_parameters": rope})
    config._attn_implementation = "eager"
    torch.manual_seed(0)
    return Mistral4ForCausalLM(config).float().eval()


def _deepseek_v4() -> nn.Module:
    torch.manual_seed(0)
    return tiny_family_model(TINY_MOE_FAMILIES["deepseek_v4"]).float().eval()


_FAMILIES = {"mistral4": _mistral4, "deepseek_v4": _deepseek_v4}


class _CPAttention(nn.Module):
    """What the CP path runs: dense causal attention over the whole row, the padding mask unread and the
    positions an ``arange``, as the CP wrapper builds them."""

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask=None, **kwargs):
        positions = torch.arange(input_ids.size(1)).expand_as(input_ids)
        return self.model(input_ids=input_ids, position_ids=positions, **kwargs)


def _host(cp_layout: bool) -> SmoothMarginPOTrainer:
    """A construction-free SMPO; ``cp_layout`` lays the batch out as CP does, on one process."""
    host = object.__new__(SmoothMarginPOTrainer)
    host.parallelism_config = SimpleNamespace(cp_size=2 if cp_layout else 1)
    host.cp_config = None
    host.pad_token_id = PAD_TOKEN_ID
    host.padding_free = False
    host.lower_clip_percentile = host.upper_clip_percentile = host.min_log_prob = None
    return host


def _batch(prompts: list[list[int]], chosen: list[list[int]], rejected: list[list[int]]) -> dict[str, torch.Tensor]:
    """The collator's layout: prompts left-padded, completions right-padded."""

    def pad(rows, left):
        width = max(map(len, rows))
        ids = torch.full((len(rows), width), PAD_TOKEN_ID)
        mask = torch.zeros(len(rows), width, dtype=torch.long)
        for i, row in enumerate(rows):
            span = slice(width - len(row), width) if left else slice(0, len(row))
            ids[i, span], mask[i, span] = torch.tensor(row), 1
        return ids, mask

    batch = {}
    for side, rows, left in (("prompt", prompts, True), ("chosen", chosen, False), ("rejected", rejected, False)):
        batch[f"{side}_input_ids"], batch[f"{side}_attention_mask"] = pad(rows, left)
    return batch


_PROMPTS = [[11, 12, 13], [21, 22, 23, 24, 25, 26]]
_CHOSEN = [[31, 32, 33, 34], [41, 42]]
_REJECTED = [[51, 52, 53], [61, 62, 63, 64, 65]]


@pytest.mark.parametrize("family", list(_FAMILIES))
@torch.no_grad()
def test_a_left_padded_pair_scores_as_it_does_alone(family):
    model = _FAMILIES[family]()
    ragged = _host(cp_layout=False).concatenated_forward(model, _batch(_PROMPTS, _CHOSEN, _REJECTED))
    alone = _host(cp_layout=False).concatenated_forward(model, _batch(_PROMPTS[:1], _CHOSEN[:1], _REJECTED[:1]))

    for key in LOGP_KEYS:
        torch.testing.assert_close(ragged[key][:1], alone[key], rtol=1e-5, atol=1e-5, msg=key)


_IMAGE = 5
# A 4x4 patch grid pooled 2x2: four soft tokens, so four image placeholders per prompt.
_GEMMA4_VISION = {
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "patch_size": 4,
    "position_embedding_size": 16,
    "pooling_kernel_size": 2,
}


def _gemma4_vlm() -> nn.Module:
    torch.manual_seed(0)
    text = {**TINY_GEMMA4_MOE_CONFIG, "use_bidirectional_attention": "vision", "pad_token_id": PAD_TOKEN_ID}
    model = Gemma4ForConditionalGeneration(
        Gemma4Config(text_config=text, vision_config=_GEMMA4_VISION, image_token_id=_IMAGE)
    )
    model.config._attn_implementation = "eager"
    return model.float().eval()


def _image_batch(prompts, chosen, rejected) -> dict[str, torch.Tensor]:
    """``_batch`` plus one image per prompt and the prompt-aligned token types the collator emits."""
    batch = _batch(prompts, chosen, rejected)
    batch["mm_token_type_ids"] = (batch["prompt_input_ids"] == _IMAGE).long()
    generator = torch.Generator().manual_seed(1)
    batch["pixel_values"] = torch.randn(len(prompts), 16, 3 * 4 * 4, generator=generator)
    grid = torch.tensor([[x, y] for y in range(4) for x in range(4)])
    batch["image_position_ids"] = grid.expand(len(prompts), -1, -1).contiguous()
    return batch


@torch.no_grad()
def test_a_left_padded_image_pair_scores_as_it_does_alone():
    """Token types left behind by the flush would put the bidirectional block on the wrong tokens."""
    model = _gemma4_vlm()
    prompts = [[11, *[_IMAGE] * 4, 12], [21, 22, 23, *[_IMAGE] * 4, 24, 25, 26]]
    ragged = _host(cp_layout=False).concatenated_forward(model, _image_batch(prompts, _CHOSEN, _REJECTED))
    alone = _host(cp_layout=False).concatenated_forward(model, _image_batch(prompts[:1], _CHOSEN[:1], _REJECTED[:1]))

    for key in LOGP_KEYS:
        torch.testing.assert_close(ragged[key][:1], alone[key], rtol=1e-5, atol=1e-5, msg=key)


@torch.no_grad()
def test_the_cp_layout_scores_the_padded_batch_alike():
    model = _mistral4()
    batch = _batch(_PROMPTS, _CHOSEN, _REJECTED)
    padded = _host(cp_layout=False).concatenated_forward(model, batch)
    cp_layout = _host(cp_layout=True).concatenated_forward(_CPAttention(model), batch)

    for key in LOGP_KEYS:
        torch.testing.assert_close(cp_layout[key], padded[key], rtol=1e-5, atol=1e-5, msg=key)


@pytest.mark.parametrize("cp_size", [1, 2])
def test_the_flush_reads_nothing_back_from_the_device(cp_size):
    """On meta tensors any host read of a value (``.item()``, ``int()``, a tensor in an ``if``) raises,
    so the concat completing there proves the flush issues no device-to-host copy."""
    batch = {key: value.to("meta") for key, value in _batch(_PROMPTS, _CHOSEN, _REJECTED).items()}
    batch["mm_token_type_ids"] = torch.zeros_like(batch["prompt_input_ids"])

    out = SmoothMarginPOTrainer.concatenated_inputs(batch, pad_token_id=PAD_TOKEN_ID, cp_size=cp_size)

    width = batch["prompt_input_ids"].size(1) + max(
        batch[f"{side}_input_ids"].size(1) for side in ("chosen", "rejected")
    )
    width += -width % cp_size
    for key in ("input_ids", "attention_mask", "labels", "mm_token_type_ids"):
        assert out[key].is_meta and out[key].shape == (2 * len(_PROMPTS), width), key


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
