#!/usr/bin/env python
"""The VLM preference pair for Bradley-Terry reward modeling: the map renders both sides whole and persists no
pixels, and the collator emits TRL's chosen-then-rejected layout, pinned against TRL's own since a swapped half
changes no shape. The stub's flat ``pixel_values`` hold each image's width, so patch order shows."""

import re

import pytest
import torch
from datasets import Dataset
from PIL import Image
from trl.trainer.reward_trainer import DataCollatorForPreference

from halo.data.collators.vlm_preference import DataCollatorForVLMPreference
from halo.data.pipeline.preferences import render_vlm_preference_row, vlm_preference_features

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

        row_images = images or [[] for _ in text]
        assert len(row_images) == len(text), "one image list per text, as transformers' processors batch them"
        image_token_id = self.tokenizer.vocab[IMAGE_TOKEN]
        expanded = []
        for one, one_images in zip(text, row_images, strict=True):
            row, placeholders = [], 0
            for token_id in self.tokenizer(one)["input_ids"]:
                if token_id == image_token_id:
                    row.extend([self.tokenizer.vocab[IMAGE_PAD_TOKEN]] * PATCHES_PER_IMAGE)
                    placeholders += 1
                else:
                    row.append(token_id)
            if placeholders != len(one_images):
                raise ValueError(f"{placeholders} image placeholders but {len(one_images)} images in a row")
            expanded.append(row)
        images = [image for one_images in row_images for image in one_images]

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


def collate(processor, rows, **kwargs):
    return DataCollatorForVLMPreference(processor=processor, **kwargs)(rows)


def test_layout_is_trls_own_chosen_then_rejected_block():
    """``RewardTrainer.compute_loss`` reads the first half as chosen, so on a text-only batch the ids match the
    collator that contract was written for."""
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
    """Every column this collator reads must be declared, or the dataloader mixin cannot keep it past column
    pruning."""
    declared = set(DataCollatorForVLMPreference.required_dataset_columns)
    consumed = {"chosen_text", "rejected_text", "images"}
    assert consumed <= declared


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
