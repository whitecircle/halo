#!/usr/bin/env python
"""CPU tests for the VLM preference pair used by Bradley-Terry reward modeling.

The map (:func:`render_vlm_preference_row`) and the collator
(:class:`DataCollatorForVLMPreference`) are one contract, so they are tested as one:

* the map renders BOTH sides whole (the reward head scores the full sequence), extracts the prompt's
  images out of the conversation, and persists no pixels;
* the collator emits TRL's ``[chosen ⧺ rejected]`` layout — pinned against TRL's OWN
  ``DataCollatorForPreference`` on a text-only batch rather than against a restatement of it, because
  a swapped half inverts every preference while every shape stays right;
* vision tensors ride that same layout row-major, repeated once per half (both sides share the
  prompt, hence the prompt's images);
* an over-length batch raises instead of truncating expanded image placeholders.

The stub processor mimics the real contract: ``apply_chat_template`` renders un-expanded
placeholders, ``__call__`` takes the whole ``2 * batch_size`` text block, expands each placeholder to
a fixed patch span, right-pads, and emits Qwen-style flat ``pixel_values`` in the order the images
were handed to it (filled with each image's width, so patch order is observable).

Run: python tests/cpu/data/test_vlm_preference_collator.py  (or pytest)
"""

import re
import sys

import pytest
import torch
from datasets import Dataset
from PIL import Image
from trl.trainer.reward_trainer import DataCollatorForPreference

from src.data.collators.vlm_preference import DataCollatorForVLMPreference
from src.data.pipeline.preferences import render_vlm_preference_row, vlm_preference_features

IMAGE_TOKEN = "<image>"
IMAGE_PAD_TOKEN = "<imgpad>"
PATCHES_PER_IMAGE = 4
PATCH_DIM = 8
PAD_ID = 0


class StubTokenizer:
    """Whitespace tokenizer with an image placeholder token and a growable vocab."""

    def __init__(self):
        self.pad_token_id = PAD_ID
        self.eos_token_id = 2
        self.vocab = {IMAGE_TOKEN: 5, IMAGE_PAD_TOKEN: 6}
        self._next_id = 10

    def _encode(self, text):
        ids = []
        for part in re.split(rf"({re.escape(IMAGE_TOKEN)}|{re.escape(IMAGE_PAD_TOKEN)})", text):
            if not part:
                continue
            if part in (IMAGE_TOKEN, IMAGE_PAD_TOKEN):
                ids.append(self.vocab[part])
                continue
            for token in part.split():
                if token not in self.vocab:
                    self.vocab[token] = self._next_id
                    self._next_id += 1
                ids.append(self.vocab[token])
        return ids

    def __call__(self, text, add_special_tokens=False, **kwargs):
        return {"input_ids": self._encode(text)}

    def decode(self, ids):
        reverse = {value: key for key, value in self.vocab.items()}
        return " ".join(reverse.get(int(i), f"t{int(i)}") for i in ids if int(i) != PAD_ID)


class StubProcessor:
    """Batched VLM processor: placeholder expansion, right padding, flat pixel patches."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        assert not tokenize
        rendered = []
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                content = " ".join(
                    IMAGE_TOKEN if part.get("type") == "image" else part.get("text", "") for part in content
                )
            rendered.append(f"[{message['role']}] {content} [end]\n")
        return "".join(rendered)

    def __call__(self, text, images=None, return_tensors="pt", add_special_tokens=True, padding=False, **kwargs):
        assert return_tensors == "pt"
        assert add_special_tokens is False, "collator must disable special tokens (the template carries them)"
        assert padding is True, "the whole chosen⧺rejected block is padded in one call"
        assert isinstance(text, list), "the collator must hand the processor the whole text block"

        image_token_id = self.tokenizer.vocab[IMAGE_TOKEN]
        expanded, placeholders = [], 0
        for one in text:
            row = []
            for token_id in self.tokenizer(one)["input_ids"]:
                if token_id == image_token_id:
                    row.extend([self.tokenizer.vocab[IMAGE_PAD_TOKEN]] * PATCHES_PER_IMAGE)
                    placeholders += 1
                else:
                    row.append(token_id)
            expanded.append(row)
        images = images or []
        if placeholders != len(images):
            raise ValueError(f"{placeholders} image placeholders but {len(images)} images")

        width = max(len(row) for row in expanded)
        out = {
            "input_ids": torch.tensor([row + [PAD_ID] * (width - len(row)) for row in expanded], dtype=torch.long),
            "attention_mask": torch.tensor(
                [[1] * len(row) + [0] * (width - len(row)) for row in expanded], dtype=torch.long
            ),
        }
        if images:
            # Fill value = image width → the patch block's order is directly observable.
            out["pixel_values"] = torch.cat(
                [torch.full((PATCHES_PER_IMAGE, PATCH_DIM), float(image.size[0])) for image in images], dim=0
            )
            out["image_grid_thw"] = torch.tensor([[1, 2, 2]] * len(images), dtype=torch.long)
        return out


def make_row(chosen="A red square", rejected="A blue circle", prompt="Describe the picture", widths=(7,)):
    return {
        "prompt": [{"role": "user", "content": prompt}],
        "chosen": [{"role": "assistant", "content": chosen}],
        "rejected": [{"role": "assistant", "content": rejected}],
        "images": [Image.new("RGB", (width, 5)) for width in widths],
    }


def render(row, processor):
    return render_vlm_preference_row(row, processor)


# --- the map -------------------------------------------------------------------------------------


def test_render_produces_both_sides_whole_and_extracts_the_images():
    processor = StubProcessor(StubTokenizer())
    row = make_row()

    rendered = render(row, processor)

    assert set(rendered) == {"chosen_text", "rejected_text", "images"}
    # Whole conversations, not completions: the reward head pools the full sequence.
    for side, completion in (("chosen", "A red square"), ("rejected", "A blue circle")):
        text = rendered[f"{side}_text"]
        assert text == f"[user] {IMAGE_TOKEN} Describe the picture [end]\n[assistant] {completion} [end]\n"
    assert rendered["images"] == row["images"]
    assert not {key for key in rendered if "pixel" in key}, "pixels must never reach the Arrow row"


def test_render_merges_a_multi_image_column_into_the_shared_prompt():
    processor = StubProcessor(StubTokenizer())

    rendered = render(make_row(widths=(7, 9)), processor)

    # Both sides carry the prompt, so both carry every placeholder.
    assert rendered["chosen_text"].count(IMAGE_TOKEN) == 2
    assert rendered["rejected_text"].count(IMAGE_TOKEN) == 2
    assert len(rendered["images"]) == 2


def test_render_reads_the_singular_image_column_too():
    processor = StubProcessor(StubTokenizer())
    row = make_row()
    row["image"] = row.pop("images")

    rendered = render(row, processor)

    assert rendered["chosen_text"].count(IMAGE_TOKEN) == 1
    assert len(rendered["images"]) == 1


def test_render_normalizes_an_implicit_prompt_dataset():
    """Skywork shape: no prompt column, the shared turns live inside chosen/rejected."""
    processor = StubProcessor(StubTokenizer())
    shared = {"role": "user", "content": "Describe the picture"}
    row = {
        "chosen": [dict(shared), {"role": "assistant", "content": "A red square"}],
        "rejected": [dict(shared), {"role": "assistant", "content": "A blue circle"}],
        "images": [Image.new("RGB", (7, 5))],
    }

    rendered = render(row, processor)

    assert rendered["chosen_text"].count(IMAGE_TOKEN) == 1
    assert rendered["chosen_text"].endswith("[assistant] A red square [end]\n")
    assert rendered["rejected_text"].endswith("[assistant] A blue circle [end]\n")


def test_render_rejects_images_inside_a_completion():
    processor = StubProcessor(StubTokenizer())
    row = make_row()
    row["rejected"] = [{"role": "assistant", "content": [{"type": "image", "image": Image.new("RGB", (7, 5))}]}]

    with pytest.raises(ValueError, match="images inside the 'rejected' completion"):
        render(row, processor)


def test_features_pin_margin_only_when_the_dataset_carries_one():
    plain = Dataset.from_dict({"chosen": [[]], "rejected": [[]]})
    with_margin = Dataset.from_dict({"chosen": [[]], "rejected": [[]], "margin": [0.5]})

    assert set(vlm_preference_features(plain)) == {"chosen_text", "rejected_text", "images"}
    assert "margin" in vlm_preference_features(with_margin)


def test_map_roundtrip_hands_back_pil_images_and_no_pixels():
    processor = StubProcessor(StubTokenizer())
    dataset = Dataset.from_list([make_row(), make_row(chosen="Two", rejected="Three")])

    mapped = dataset.map(
        render_vlm_preference_row,
        fn_kwargs={"processing_class": processor},
        remove_columns=dataset.column_names,
        features=vlm_preference_features(dataset),
    )

    assert set(mapped.column_names) == {"chosen_text", "rejected_text", "images"}
    assert isinstance(mapped[0]["images"][0], Image.Image)


# --- the collator --------------------------------------------------------------------------------


def collate(processor, rows, **kwargs):
    return DataCollatorForVLMPreference(processor=processor, **kwargs)(rows)


def test_layout_is_trls_own_chosen_then_rejected_block():
    """Pinned against TRL's collator, not a restatement of it.

    ``RewardTrainer.compute_loss`` chunks the pooled scores in two and reads the FIRST half as
    chosen. On a text-only batch this collator must therefore produce byte-identical ids to the
    collator that contract was written for — a swapped half changes no shape and no dtype.
    """
    tokenizer = StubTokenizer()
    processor = StubProcessor(tokenizer)
    rows = [
        {**render(make_row(chosen="First good", rejected="First bad", widths=()), processor), "margin": 0.5},
        {**render(make_row(chosen="Second good longer", rejected="Second bad", widths=()), processor), "margin": 0.0},
    ]

    batch = collate(processor, rows)
    reference = DataCollatorForPreference(pad_token_id=PAD_ID)(
        [
            {
                "chosen_ids": tokenizer._encode(row["chosen_text"]),
                "rejected_ids": tokenizer._encode(row["rejected_text"]),
                "margin": row["margin"],
            }
            for row in rows
        ]
    )

    assert torch.equal(batch["input_ids"], reference["input_ids"])
    assert torch.equal(batch["attention_mask"], reference["attention_mask"])
    assert torch.equal(batch["margin"], reference["margin"])


def test_the_first_half_really_is_the_chosen_half():
    """The layout read the way the loss reads it: ``chunk(scores, 2)[0]`` must be the chosen rows."""
    tokenizer = StubTokenizer()
    processor = StubProcessor(tokenizer)
    rows = [render(make_row(chosen="GOOD", rejected="BAD"), processor) for _ in range(3)]

    batch = collate(processor, rows)

    good_id = tokenizer.vocab["GOOD"]
    scores = (batch["input_ids"] == good_id).any(dim=-1).float()
    chosen_scores, rejected_scores = torch.chunk(scores, 2)
    assert bool((chosen_scores > rejected_scores).all()), "chunk(.., 2)[0] must hold the chosen sequences"


def test_vision_tensors_are_row_major_and_repeated_once_per_half():
    processor = StubProcessor(StubTokenizer())
    rows = [
        render(make_row(widths=(7,)), processor),
        render(make_row(prompt="Compare the two pictures", widths=(9, 11)), processor),
    ]

    batch = collate(processor, rows)

    assert batch["pixel_values"].shape == (6 * PATCHES_PER_IMAGE, PATCH_DIM)
    # Row-major within a half (7 | 9, 11), then the same block again for the rejected half.
    expected_widths = [7.0, 9.0, 11.0, 7.0, 9.0, 11.0]
    for index, width in enumerate(expected_widths):
        span = batch["pixel_values"][index * PATCHES_PER_IMAGE : (index + 1) * PATCHES_PER_IMAGE]
        assert bool((span == width).all()), f"patch block {index} is not image width {width}"
    assert batch["image_grid_thw"].shape == (6, 3)


def test_mixed_batch_with_a_text_only_row():
    processor = StubProcessor(StubTokenizer())
    rows = [render(make_row(widths=(7,)), processor), render(make_row(widths=()), processor)]

    batch = collate(processor, rows)

    assert batch["input_ids"].shape[0] == 4
    assert batch["pixel_values"].shape == (2 * PATCHES_PER_IMAGE, PATCH_DIM)


def test_over_length_batch_raises_instead_of_truncating():
    processor = StubProcessor(StubTokenizer())
    rows = [render(make_row(), processor)]

    with pytest.raises(ValueError, match="over max_length=5"):
        collate(processor, rows, max_length=5)

    # Under budget the same batch collates: the guard is a budget check, not a blanket refusal.
    assert collate(processor, rows, max_length=512)["input_ids"].shape[0] == 2


def test_collator_requires_a_processor():
    with pytest.raises(ValueError, match="requires a processor"):
        DataCollatorForVLMPreference()


def test_collator_declares_the_columns_pruning_would_drop():
    """TRL's reward signature set is chosen_ids / rejected_ids / margin — every column this collator
    reads has to be declared, or the dataloader mixin cannot keep it past column pruning."""
    declared = set(DataCollatorForVLMPreference.required_dataset_columns)
    consumed = {"chosen_text", "rejected_text", "images"}
    assert consumed <= declared


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
