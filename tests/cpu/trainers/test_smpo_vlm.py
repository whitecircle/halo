#!/usr/bin/env python
"""
CPU tests for SMPO VLM support (SmoothMarginPOTrainer VLM mode + DataCollatorForVLMSMPO).

Covers the pieces GPU tests can't cheaply exercise:
  - ``tokenize_vlm_preference_row``: raw-row normalization, image injection/extraction, prefix-strip
    invariant, completion tokenization parity with the text path (EOS append, truncation);
  - ``DataCollatorForVLMSMPO``: placeholder expansion at collation, row-major vision stacking,
    left-padded prompts, BOS handling, over-length prompt fail-loud;
  - ``concatenated_inputs``: vision-key duplication for the chosen|rejected concat (odd/even
    batches) with the text keys byte-identical to the no-vision batch;
  - ``concatenated_forward``: vision tensors reach the model forward; padding_free rejects them;
  - the VLM-mode guards (padding_free, CP);
  - an Arrow round-trip (dataset.map keeps PIL images collatable);
  - the real Qwen3.5 processor end-to-end (skipped when not cached locally).

The stub processor mimics the real contract: ``apply_chat_template`` renders un-expanded image
placeholders, ``__call__`` expands each placeholder to a fixed patch span and emits Qwen-style
``pixel_values`` ([total_patches, dim], row-major) + ``image_grid_thw``.

Run: python tests/cpu/trainers/test_smpo_vlm.py
"""

import re
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from datasets import Dataset
from PIL import Image
from transformers import AutoProcessor

from src.data.collators.smpo import DataCollatorForSMPO, DataCollatorForVLMSMPO
from src.trainers.preference.smpo import SmoothMarginPOTrainer, tokenize_vlm_preference_row

IMAGE_TOKEN = "<image>"
IMAGE_PAD_TOKEN = "<imgpad>"
PATCHES_PER_IMAGE = 4
PATCH_DIM = 8
EOS_ID = 2


class StubTokenizer:
    """Whitespace tokenizer with special image tokens and a growable vocab."""

    def __init__(self, bos_token_id=None):
        self.bos_token_id = bos_token_id
        self.eos_token_id = EOS_ID
        self.pad_token_id = 0
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
        # The terminator walk decodes one id at a time; this stub emits content ids only.
        reverse = {v: k for k, v in self.vocab.items()}
        return " ".join(reverse.get(i, f"t{i}") for i in ids)

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        assert not tokenize
        rendered = []
        for message in messages:
            content = message["content"]
            if isinstance(content, list):
                parts = [IMAGE_TOKEN if p.get("type") == "image" else p.get("text", "") for p in content]
                content = " ".join(parts)
            rendered.append(f"[{message['role']}] {content} [end]\n")
        return "".join(rendered)


class StubProcessor:
    """Mimics a VLM processor: placeholder expansion + Qwen-style flat pixel patches."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        return self.tokenizer.apply_chat_template(messages, tokenize=tokenize, **kwargs)

    def __call__(self, text, images=None, return_tensors="pt", add_special_tokens=True, **kwargs):
        assert return_tensors == "pt"
        assert add_special_tokens is False, "collator must disable special tokens (template carries them)"
        ids = self.tokenizer(text)["input_ids"]
        images = images or []
        image_token_id = self.tokenizer.vocab[IMAGE_TOKEN]
        pad_id = self.tokenizer.vocab[IMAGE_PAD_TOKEN]
        expanded, placeholders = [], 0
        for token_id in ids:
            if token_id == image_token_id:
                expanded.extend([pad_id] * PATCHES_PER_IMAGE)
                placeholders += 1
            else:
                expanded.append(token_id)
        if placeholders != len(images):
            raise ValueError(f"{placeholders} image placeholders but {len(images)} images")
        out = {
            "input_ids": torch.tensor([expanded], dtype=torch.long),
            "attention_mask": torch.ones(1, len(expanded), dtype=torch.long),
        }
        if images:
            # Fill value = image width → tests can verify row-major patch ordering.
            out["pixel_values"] = torch.cat(
                [torch.full((PATCHES_PER_IMAGE, PATCH_DIM), float(img.size[0])) for img in images], dim=0
            )
            out["image_grid_thw"] = torch.tensor([[1, 2, 2]] * len(images), dtype=torch.long)
        return out


def make_trainer(**attrs):
    """Bare SmoothMarginPOTrainer with just the attributes the tested methods read."""
    trainer = SmoothMarginPOTrainer.__new__(SmoothMarginPOTrainer)
    defaults = {
        "is_vlm": True,
        "padding_free": False,
        "max_prompt_length": 512,
        "max_completion_length": 64,
        "truncation_mode": "keep_start",
        "dataset_num_proc": None,
        # The real trainer resolves this once from tokenizer + model config.
        "_eos_token_ids": frozenset({EOS_ID}),
        "label_pad_token_id": -100,
        "pad_token_id": 0,
        # cp_size is a read-only mixin property; is_ep_mode gates dataset-map workers (EP pins 1).
        "parallelism_config": SimpleNamespace(cp_size=1, is_ep_mode=False),
        "cp_config": None,
        "lower_clip_percentile": None,
        "upper_clip_percentile": None,
        "min_log_prob": None,
    }
    defaults.update(attrs)
    for key, value in defaults.items():
        setattr(trainer, key, value)
    return trainer


def make_features(width=7, prompt_text="Describe the picture", n_images=1):
    images = [Image.new("RGB", (width + 2 * i, 5)) for i in range(n_images)]
    return {
        "prompt": [{"role": "user", "content": prompt_text}],
        "chosen": [{"role": "assistant", "content": "A red square"}],
        "rejected": [{"role": "assistant", "content": "A blue circle"}],
        "images": images,
    }


def prep_row(trainer, features, processor):
    """Run the VLM row prep the trainer's dataset map runs.

    ``tokenize_vlm_preference_row`` is module-level (a bound method would pickle the trainer, and
    with it the EP process groups, into every ``dataset.map`` worker), so its tunables come from the
    trainer's attributes exactly as ``_prepare_dataset`` threads them through ``fn_kwargs``.
    """
    return tokenize_vlm_preference_row(
        features,
        processor,
        max_prompt_length=trainer.max_prompt_length,
        max_completion_length=trainer.max_completion_length,
        truncation_mode=trainer.truncation_mode,
    )


def test_vlm_row_prep_basic():
    tokenizer = StubTokenizer()
    processor = StubProcessor(tokenizer)
    trainer = make_trainer()
    features = make_features()

    row = prep_row(trainer, features, processor)

    assert IMAGE_TOKEN in row["prompt_text"]
    assert row["prompt_text"] == f"[user] {IMAGE_TOKEN} Describe the picture [end]\n"
    assert row["images"] == features["images"]
    assert "prompt_input_ids" not in row  # tokenized at collation, not in the dataset
    assert "pixel_values" not in row  # pixels never land in the Arrow rows

    assert row["chosen_input_ids"][:-1] == tokenizer._encode("[assistant] A red square [end]\n")
    assert row["chosen_input_ids"][-1] == EOS_ID
    assert row["rejected_input_ids"][:-1] == tokenizer._encode("[assistant] A blue circle [end]\n")
    assert row["chosen_input_ids"] != row["rejected_input_ids"]


def test_vlm_row_prep_string_prompt_and_prefix_strip():
    tokenizer = StubTokenizer()
    processor = StubProcessor(tokenizer)
    trainer = make_trainer()
    prompt_msg = {"role": "user", "content": "What is shown"}
    features = {
        "prompt": "What is shown",  # string prompt → wrapped as a user turn
        "chosen": [dict(prompt_msg), {"role": "assistant", "content": "A cat"}],  # repeats the prompt
        "rejected": [dict(prompt_msg), {"role": "assistant", "content": "A dog"}],
        "images": [Image.new("RGB", (7, 5))],
    }

    row = prep_row(trainer, features, processor)

    assert row["prompt_text"] == f"[user] {IMAGE_TOKEN} What is shown [end]\n"
    assert row["chosen_input_ids"][:-1] == tokenizer._encode("[assistant] A cat [end]\n")
    assert row["rejected_input_ids"][:-1] == tokenizer._encode("[assistant] A dog [end]\n")


def test_vlm_row_prep_images_embedded_in_prompt():
    tokenizer = StubTokenizer()
    processor = StubProcessor(tokenizer)
    trainer = make_trainer()
    img = Image.new("RGB", (9, 5))
    features = {
        "prompt": [
            {
                "role": "user",
                "content": [{"type": "image", "image": img}, {"type": "text", "text": "Caption this"}],
            }
        ],
        "chosen": [{"role": "assistant", "content": "Yes"}],
        "rejected": [{"role": "assistant", "content": "No"}],
    }

    row = prep_row(trainer, features, processor)

    assert row["images"] == [img]
    assert IMAGE_TOKEN in row["prompt_text"]


def test_vlm_row_prep_completion_images_raise():
    tokenizer = StubTokenizer()
    processor = StubProcessor(tokenizer)
    trainer = make_trainer()
    features = make_features()
    features["chosen"] = [{"role": "assistant", "content": [{"type": "image", "image": Image.new("RGB", (7, 5))}]}]

    with pytest.raises(ValueError, match="images inside the 'chosen' completion"):
        prep_row(trainer, features, processor)


def test_vlm_row_prep_prefix_invariant_raises():
    class BrokenTemplateProcessor(StubProcessor):
        def apply_chat_template(self, messages, tokenize=False, **kwargs):
            # Message-count-dependent output: template(prompt+completion) loses the prompt prefix.
            return f"{len(messages)}:: " + super().apply_chat_template(messages, tokenize=tokenize, **kwargs)

    trainer = make_trainer()
    with pytest.raises(ValueError, match="prefix invariant"):
        prep_row(trainer, make_features(), BrokenTemplateProcessor(StubTokenizer()))


def test_vlm_row_prep_completion_truncation_keeps_eos():
    tokenizer = StubTokenizer()
    processor = StubProcessor(tokenizer)
    trainer = make_trainer(max_completion_length=3)

    row = prep_row(trainer, make_features(), processor)

    assert len(row["chosen_input_ids"]) == 3
    assert row["chosen_input_ids"][-1] == EOS_ID


def make_collator(processor, **kwargs):
    return DataCollatorForVLMSMPO(pad_token_id=0, processor=processor, **kwargs)


def test_collator_expansion_and_row_major_vision():
    tokenizer = StubTokenizer()
    processor = StubProcessor(tokenizer)
    trainer = make_trainer()
    row1 = prep_row(trainer, make_features(width=7), processor)
    row2 = prep_row(
        trainer, make_features(width=9, prompt_text="Compare the two pictures shown", n_images=2), processor
    )
    # Two placeholders only if both images landed in the first user turn.
    assert row2["prompt_text"].count(IMAGE_TOKEN) == 2

    batch = make_collator(processor)([row1, row2])

    std_keys = {
        "prompt_input_ids",
        "prompt_attention_mask",
        "chosen_input_ids",
        "chosen_attention_mask",
        "rejected_input_ids",
        "rejected_attention_mask",
    }
    assert std_keys | {"pixel_values", "image_grid_thw"} == set(batch.keys())

    pad_id = tokenizer.vocab[IMAGE_PAD_TOKEN]
    assert (batch["prompt_input_ids"][0] == pad_id).sum() == PATCHES_PER_IMAGE
    assert (batch["prompt_input_ids"][1] == pad_id).sum() == 2 * PATCHES_PER_IMAGE
    assert batch["prompt_input_ids"].shape == batch["prompt_attention_mask"].shape
    lengths = batch["prompt_attention_mask"].sum(dim=1)
    shorter = int(lengths.argmin())
    assert batch["prompt_attention_mask"][shorter, 0] == 0  # left padding
    assert batch["prompt_attention_mask"][shorter, -1] == 1

    # Row-major stacking: row1's image (w=7), then row2's (w=9, w=11).
    assert batch["pixel_values"].shape == (3 * PATCHES_PER_IMAGE, PATCH_DIM)
    expected_widths = [7.0, 9.0, 11.0]
    for i, width in enumerate(expected_widths):
        patch_span = batch["pixel_values"][i * PATCHES_PER_IMAGE : (i + 1) * PATCHES_PER_IMAGE]
        assert (patch_span == width).all()
    assert batch["image_grid_thw"].shape == (3, 3)

    assert batch["chosen_input_ids"].shape[0] == 2
    assert batch["chosen_attention_mask"][0, 0] == 1


def test_collator_mixed_batch_with_textonly_row():
    tokenizer = StubTokenizer()
    processor = StubProcessor(tokenizer)
    trainer = make_trainer()
    row_img = prep_row(trainer, make_features(width=7), processor)
    features_text = make_features(n_images=0)
    features_text["prompt"] = [{"role": "user", "content": "Just words"}]
    row_text = prep_row(trainer, features_text, processor)

    batch = make_collator(processor)([row_img, row_text])

    assert batch["pixel_values"].shape == (PATCHES_PER_IMAGE, PATCH_DIM)
    assert batch["image_grid_thw"].shape == (1, 3)
    assert batch["prompt_input_ids"].shape[0] == 2


def test_collator_overlength_prompt_raises():
    tokenizer = StubTokenizer()
    processor = StubProcessor(tokenizer)
    trainer = make_trainer()
    row = prep_row(trainer, make_features(), processor)

    with pytest.raises(ValueError, match="cannot be truncated"):
        make_collator(processor, max_prompt_length=5)([row])


class BosPostProcessingStubTokenizer(StubTokenizer):
    """A tokenizer whose ``add_special_tokens=True`` post-processor really prepends BOS.

    The probe keys on this, not on ``bos_token_id`` — the VLM prompt is tokenized with
    ``add_special_tokens=False``, so BOS is owed exactly when the post-processor owns it.
    """

    def __call__(self, text, add_special_tokens=False, **kwargs):
        ids = self._encode(text)
        if add_special_tokens:
            ids = [self.bos_token_id] + ids
        return {"input_ids": ids}


def test_collator_bos_prepend_when_post_processor_owns_it():
    tokenizer = BosPostProcessingStubTokenizer(bos_token_id=1)
    processor = StubProcessor(tokenizer)
    trainer = make_trainer()
    row = prep_row(trainer, make_features(), processor)

    batch = make_collator(processor)([row])

    assert batch["prompt_input_ids"][0, 0] == 1


def test_collator_no_bos_for_nominal_only_bos_token():
    """A declared-but-never-emitted bos_token must NOT be injected (gpt-oss/Bailing shape).

    ``StubTokenizer`` adds nothing on ``add_special_tokens=True``, so the probe reports no
    post-processor BOS; prepending anyway would train VLM SMPO prompts on a token the served
    policy never produces — the text path (`tokenize_preference_row`) already gates on the probe.
    """
    tokenizer = StubTokenizer(bos_token_id=1)
    processor = StubProcessor(tokenizer)
    trainer = make_trainer()
    row = prep_row(trainer, make_features(), processor)

    batch = make_collator(processor)([row])

    assert batch["prompt_input_ids"][0, 0] != 1, "a nominal-only bos_token must not be prepended"


def test_collator_requires_processor():
    with pytest.raises(ValueError, match="requires a processor"):
        DataCollatorForVLMSMPO(pad_token_id=0)


def make_text_batch(batch_size):
    g = torch.Generator().manual_seed(batch_size)
    return {
        "prompt_input_ids": torch.randint(10, 50, (batch_size, 6), generator=g),
        "prompt_attention_mask": torch.ones(batch_size, 6, dtype=torch.long),
        "chosen_input_ids": torch.randint(10, 50, (batch_size, 4), generator=g),
        "chosen_attention_mask": torch.ones(batch_size, 4, dtype=torch.long),
        "rejected_input_ids": torch.randint(10, 50, (batch_size, 5), generator=g),
        "rejected_attention_mask": torch.ones(batch_size, 5, dtype=torch.long),
    }


def test_concatenated_inputs_duplicates_vision_keys():
    for batch_size in (1, 2, 3):  # odd and even
        batch = make_text_batch(batch_size)
        pixel_values = torch.randn(batch_size * PATCHES_PER_IMAGE, PATCH_DIM)
        grid = torch.tensor([[1, 2, 2]] * batch_size)
        vision_batch = {**batch, "pixel_values": pixel_values, "image_grid_thw": grid}

        out = SmoothMarginPOTrainer.concatenated_inputs(vision_batch, pad_token_id=0)

        assert torch.equal(out["pixel_values"], torch.cat([pixel_values, pixel_values], dim=0))
        assert torch.equal(out["image_grid_thw"], torch.cat([grid, grid], dim=0))
        assert out["num_chosen"] == batch_size

        # Text keys must stay byte-identical to the no-vision batch.
        out_text = SmoothMarginPOTrainer.concatenated_inputs(batch, pad_token_id=0)
        assert set(out_text.keys()) == {"input_ids", "attention_mask", "labels", "num_chosen"}
        for key in ("input_ids", "attention_mask", "labels"):
            assert torch.equal(out[key], out_text[key])


class RecordingModel(nn.Module):
    def __init__(self, vocab_size=64):
        super().__init__()
        self.vocab_size = vocab_size
        self.device = torch.device("cpu")
        self.received = {}

    def forward(self, input_ids=None, attention_mask=None, use_cache=None, **kwargs):
        self.received = {"input_ids": input_ids, "attention_mask": attention_mask, **kwargs}
        g = torch.Generator().manual_seed(0)
        logits = torch.randn(input_ids.shape[0], input_ids.shape[1], self.vocab_size, generator=g)
        return SimpleNamespace(logits=logits)


def test_concatenated_forward_passes_vision_inputs():
    trainer = make_trainer()
    model = RecordingModel()
    batch_size = 2
    batch = make_text_batch(batch_size)
    batch["pixel_values"] = torch.randn(batch_size * PATCHES_PER_IMAGE, PATCH_DIM)
    batch["image_grid_thw"] = torch.tensor([[1, 2, 2]] * batch_size)

    out = trainer.concatenated_forward(model, batch)

    assert model.received["pixel_values"].shape == (2 * batch_size * PATCHES_PER_IMAGE, PATCH_DIM)
    assert model.received["image_grid_thw"].shape == (2 * batch_size, 3)
    assert out["chosen_logps"].shape == (batch_size,)
    assert out["rejected_logps"].shape == (batch_size,)
    assert torch.isfinite(out["chosen_sft_loss"])

    # A text batch must reach the model with NO extra kwargs.
    model_text = RecordingModel()
    trainer.concatenated_forward(model_text, make_text_batch(batch_size))
    assert set(model_text.received.keys()) == {"input_ids", "attention_mask"}


def test_concatenated_forward_padding_free_rejects_vision():
    trainer = make_trainer(padding_free=True)
    batch = make_text_batch(1)
    batch["pixel_values"] = torch.randn(PATCHES_PER_IMAGE, PATCH_DIM)

    with pytest.raises(ValueError, match="padding_free"):
        trainer.concatenated_forward(RecordingModel(), batch)


def test_vlm_mode_guards():
    with pytest.raises(ValueError, match="padding_free"):
        make_trainer(padding_free=True)._validate_vlm_mode(None)

    with pytest.raises(ValueError, match="Context Parallelism"):
        make_trainer()._validate_vlm_mode(SimpleNamespace(cp_size=2))

    make_trainer()._validate_vlm_mode(None)
    make_trainer()._validate_vlm_mode(SimpleNamespace(cp_size=1))

    # Text mode must never trip the VLM guards.
    make_trainer(is_vlm=False, padding_free=True)._validate_vlm_mode(SimpleNamespace(cp_size=2))


def test_prepare_dataset_skips_prepared_rows():
    from accelerate import PartialState

    PartialState()  # the accelerate logger in _prepare_dataset requires an initialized state
    vlm_ds = Dataset.from_dict({"prompt_text": ["p"], "chosen_input_ids": [[1, 2]], "rejected_input_ids": [[3, 2]]})
    assert make_trainer()._prepare_dataset(vlm_ds, None, "train") is vlm_ds

    text_ds = Dataset.from_dict(
        {"prompt_input_ids": [[1]], "chosen_input_ids": [[1, 2]], "rejected_input_ids": [[3, 2]]}
    )
    assert make_trainer(is_vlm=False)._prepare_dataset(text_ds, None, "train") is text_ds

    # A prepared TEXT dataset must not satisfy the VLM check, else a pixel-less run passes silently.
    assert not {"prompt_text"}.issubset(text_ds.column_names)


def test_dataset_map_arrow_roundtrip():
    from accelerate import PartialState

    PartialState()  # _prepare_dataset's map/logging path needs an initialized accelerate state
    tokenizer = StubTokenizer()
    processor = StubProcessor(tokenizer)
    trainer = make_trainer()

    with tempfile.TemporaryDirectory() as tmp:
        image_path = str(Path(tmp) / "img.png")
        Image.new("RGB", (7, 5), color=(200, 10, 10)).save(image_path)
        ds = Dataset.from_list(
            [
                {
                    "prompt": [{"role": "user", "content": "Describe the picture"}],
                    "chosen": [{"role": "assistant", "content": "A red square"}],
                    "rejected": [{"role": "assistant", "content": "A blue circle"}],
                    "images": [image_path],  # file-path form; process_image loads it
                }
            ]
        )
        mapped = trainer._prepare_dataset(ds, processor, "train")

        row = mapped[0]
        assert IMAGE_TOKEN in row["prompt_text"]
        assert isinstance(row["images"][0], Image.Image)

        batch = make_collator(processor)([mapped[0]])
        assert batch["pixel_values"].shape == (PATCHES_PER_IMAGE, PATCH_DIM)
        assert (batch["prompt_input_ids"][0] == tokenizer.vocab[IMAGE_PAD_TOKEN]).sum() == PATCHES_PER_IMAGE


def test_real_qwen35_processor_end_to_end():
    # The CPU tier mounts the HF cache (Makefile DOCKER_RUN_CPU), so an unavailable processor is a
    # broken environment, not a reason to report this check green as a skip.
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3.5-9B")
    assert hasattr(processor, "tokenizer"), "AutoProcessor resolved to a bare tokenizer (no vision processor)"

    trainer = make_trainer(max_completion_length=128)
    features = {
        "prompt": [{"role": "user", "content": "What color dominates this image?"}],
        "chosen": [{"role": "assistant", "content": "Red dominates the image."}],
        "rejected": [{"role": "assistant", "content": "It is mostly blue."}],
        "images": [Image.new("RGB", (112, 112), color=(200, 10, 10))],
    }

    row = prep_row(trainer, features, processor)
    assert row["chosen_input_ids"] != row["rejected_input_ids"]

    batch = DataCollatorForVLMSMPO(
        pad_token_id=processor.tokenizer.pad_token_id or processor.tokenizer.eos_token_id,
        processor=processor,
    )([row])
    assert "pixel_values" in batch
    image_token_id = getattr(processor, "image_token_id", None) or processor.tokenizer.convert_tokens_to_ids(
        processor.image_token
    )
    assert (batch["prompt_input_ids"][0] == image_token_id).sum() > 1

    concat = SmoothMarginPOTrainer.concatenated_inputs(batch, pad_token_id=0)
    assert concat["pixel_values"].shape[0] == 2 * batch["pixel_values"].shape[0]


def test_text_collator_unchanged():
    examples = [
        {"prompt_input_ids": [11, 12], "chosen_input_ids": [13, 2], "rejected_input_ids": [14, 15, 2]},
        {"prompt_input_ids": [16], "chosen_input_ids": [17, 18, 2], "rejected_input_ids": [19, 2]},
    ]
    batch = DataCollatorForSMPO(pad_token_id=0)(examples)
    assert set(batch.keys()) == {
        "prompt_input_ids",
        "prompt_attention_mask",
        "chosen_input_ids",
        "chosen_attention_mask",
        "rejected_input_ids",
        "rejected_attention_mask",
    }
    assert torch.equal(batch["prompt_input_ids"], torch.tensor([[11, 12], [0, 16]]))
    assert torch.equal(batch["prompt_attention_mask"], torch.tensor([[1, 1], [0, 1]]))
    assert torch.equal(batch["chosen_input_ids"], torch.tensor([[13, 2, 0], [17, 18, 2]]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
