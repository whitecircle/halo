"""Dataset-time preparation of preference corpora (DPO, SMPO, reward): row normalization, the chat-template map
for both sides of a pair, the vision pair's render, and SMPO's row tokenizers."""

from functools import partial
from typing import Any

from datasets import Dataset, Features, Sequence, Value
from datasets import Image as ImageFeature
from transformers import PreTrainedTokenizer, PreTrainedTokenizerBase, ProcessorMixin

from src.data.pipeline.chat_render import lacks_emitted_bos
from src.data.pipeline.conversation import as_conversation, chat_template_kwargs, reject_image_content
from src.data.pipeline.coordinated_map import DATASET_NUM_PROC, coordinated_map
from src.data.pipeline.text_rows import prepare_generative_row
from src.data.pipeline.vlm_rows import normalize_vlm_conversation
from src.data.turn_markers import lacks_terminator
from src.data.vlm_render import extract_conversation_images, render_vlm_text
from src.data.vlm_routing import VLM_RAW_IMAGE_COLUMNS
from src.model_loading.tokenizer_setup import resolve_tokenizer

# The columns :func:`render_vlm_preference_row` writes and the VLM preference collator consumes.
VLM_PREFERENCE_COLUMNS = ("chosen_text", "rejected_text", "images")

# TRL's optional per-pair margin, carried through the map unchanged (its collator reads this name).
MARGIN_COLUMN = "margin"


def _shared_message_prefix_len(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> int:
    """Length of the longest leading span where ``a`` and ``b`` agree on role and content."""
    n = 0
    for x, y in zip(a, b, strict=False):
        if x.get("role") != y.get("role") or x.get("content") != y.get("content"):
            break
        n += 1
    return n


def split_rendered_completion(prompt_text: str, full_text: str, field: str) -> str:
    """``full_text`` (``template(prompt + completion)``) minus its ``prompt_text`` prefix, raising on a template
    that breaks the prefix invariant. A completion never renders alone: strict templates (Qwen3.5) refuse
    assistant-only message lists."""
    if not full_text.startswith(prompt_text):
        raise ValueError(
            f"Chat template broke the prefix invariant: template(prompt + {field}) does not start "
            "with template(prompt), so the completion cannot be split off. "
            f"prompt head: {prompt_text[:120]!r} / full head: {full_text[:120]!r}"
        )
    return full_text[len(prompt_text) :]


def normalize_preference_row(row: dict[str, Any]) -> dict[str, Any]:
    """A hub preference row in the pipeline contract: ``prompt`` a message list, ``chosen``/``rejected``
    continuation-only message lists. A string prompt becomes a user turn, a repeated prompt prefix is stripped,
    and a missing prompt (Skywork-Reward style) is the shared leading span; a continuation that is empty or does
    not end on an assistant turn raises as a mis-mapped dataset."""
    chosen, rejected = row["chosen"], row["rejected"]
    prompt = as_conversation(row.get("prompt"))

    if not prompt:
        prompt = list(chosen[: _shared_message_prefix_len(chosen, rejected)])
        if not prompt:
            raise ValueError(
                "Preference row has no 'prompt' and chosen/rejected share no leading messages to extract one from."
            )

    for field in ("chosen", "rejected"):
        completion = row[field]
        # A completion that repeats every prompt turn is a full conversation, so strip the prefix.
        if _shared_message_prefix_len(completion, prompt) == len(prompt):
            completion = completion[len(prompt) :]
        if not completion or completion[-1].get("role") != "assistant":
            raise ValueError(
                f"Preference row '{field}' is not an assistant continuation after prompt normalization: "
                f"roles={[m.get('role') for m in row[field]]}"
            )
        row[field] = completion
    row["prompt"] = prompt
    return row


def split_vlm_preference_row(
    features: dict[str, Any], subject: str
) -> tuple[list[dict[str, Any]], list, dict[str, list[dict[str, Any]]]]:
    """One raw VLM ``(prompt, chosen, rejected[, images])`` row to prompt history, images and completions, the
    pre-render step every VLM preference route shares. Images may sit only in the prompt, the one region both
    sides render identically; ``subject`` names the route in that refusal."""
    row = normalize_preference_row({key: features[key] for key in ("prompt", "chosen", "rejected") if key in features})

    images = next((features[column] for column in VLM_RAW_IMAGE_COLUMNS if features.get(column) is not None), None)
    prompt_history, pil_images = extract_conversation_images(normalize_vlm_conversation(row["prompt"], images))

    completions = {}
    for field in ("chosen", "rejected"):
        completion_history, completion_images = extract_conversation_images(row[field])
        if completion_images:
            raise ValueError(
                f"{subject} carries images inside the '{field}' completion — completions must be "
                f"text-only (images belong in the prompt, which both sides share)."
            )
        completions[field] = completion_history
    return prompt_history, pil_images, completions


def vlm_preference_features(dataset: Dataset) -> Features:
    """The Arrow schema of :func:`render_vlm_preference_row` rows, pinned like :func:`vlm_map_features` (an
    all-text shard infers ``images`` as ``List(null)``); the ``Image`` sequence also hands back PIL objects.
    ``margin``, the one source column the map keeps, is declared exactly when the dataset carries it."""
    chosen, rejected, images = VLM_PREFERENCE_COLUMNS
    features = {chosen: Value("string"), rejected: Value("string"), images: Sequence(ImageFeature())}
    if MARGIN_COLUMN in dataset.column_names:
        features[MARGIN_COLUMN] = Value("float32")
    return Features(features)


def render_vlm_preference_row(features: dict[str, Any], processing_class: ProcessorMixin) -> dict[str, Any]:
    """Render one raw VLM preference row for the reward head, each side chat-templated whole through
    :func:`render_vlm_text`; the images travel as PIL objects, so no pixels reach the Arrow cache."""
    prompt_history, pil_images, completions = split_vlm_preference_row(features, "VLM preference row")
    rendered = {
        f"{field}_text": render_vlm_text(processing_class, prompt_history + history)
        for field, history in completions.items()
    }
    return {**rendered, "images": pil_images}


def _aligned_prompt_split(prompt_ids: list[int], full_chosen_ids: list[int], full_rejected_ids: list[int]) -> int:
    """Where the prompt ends in both full sequences: one token earlier when a boundary merge shifted it on either
    side, so the trainer's verbatim ``prompt ⧺ completion`` never duplicates the merged token and both sides
    condition on one prompt."""
    split = len(prompt_ids)
    if full_chosen_ids[:split] != prompt_ids or full_rejected_ids[:split] != prompt_ids:
        # Exact on the non-merging side too: its first split - 1 tokens still match the prompt.
        split -= 1
    return split


def _truncate_keep_eos(ids: list[int], max_length: int, eos_id: int | None) -> list[int]:
    """``ids`` capped at ``max_length``, keeping their final EOS so the model still learns to stop."""
    if len(ids) <= max_length:
        return ids
    if eos_id is None:
        return ids[:max_length]
    return ids[: max_length - 1] + [eos_id]


def tokenize_preference_row(
    features: dict[str, str],
    processing_class: PreTrainedTokenizerBase,
    *,
    max_prompt_length: int | None,
    max_completion_length: int | None,
    truncation_mode: str,
    eos_token_ids: frozenset[int] = frozenset(),
) -> dict[str, list[int]]:
    """Tokenize one rendered (prompt, chosen, rejected) example from its full prompt+completion sequences, so a
    token merged across the boundary splits the same way on both sides."""
    prompt = features["prompt"]
    prompt_ids = processing_class(prompt, add_special_tokens=False)["input_ids"]
    full_chosen = processing_class(prompt + features["chosen"], add_special_tokens=False)["input_ids"]
    full_rejected = processing_class(prompt + features["rejected"], add_special_tokens=False)["input_ids"]
    split = _aligned_prompt_split(prompt_ids, full_chosen, full_rejected)
    prompt_ids, chosen_input_ids, rejected_input_ids = prompt_ids[:split], full_chosen[split:], full_rejected[split:]

    if lacks_emitted_bos(prompt_ids, processing_class):
        prompt_ids = [processing_class.bos_token_id] + prompt_ids

    # A spurious ender would sit inside the mean log-prob the SMPO margin is computed from.
    eos_id = processing_class.eos_token_id
    if lacks_terminator(chosen_input_ids, processing_class, eos_token_ids):
        chosen_input_ids = chosen_input_ids + [eos_id]
    if lacks_terminator(rejected_input_ids, processing_class, eos_token_ids):
        rejected_input_ids = rejected_input_ids + [eos_id]

    if max_prompt_length and len(prompt_ids) > max_prompt_length:
        keep_start = truncation_mode == "keep_start"
        prompt_ids = prompt_ids[:max_prompt_length] if keep_start else prompt_ids[-max_prompt_length:]

    if max_completion_length:
        chosen_input_ids = _truncate_keep_eos(chosen_input_ids, max_completion_length, eos_id)
        rejected_input_ids = _truncate_keep_eos(rejected_input_ids, max_completion_length, eos_id)

    return {
        "prompt_input_ids": prompt_ids,
        "chosen_input_ids": chosen_input_ids,
        "rejected_input_ids": rejected_input_ids,
    }


def tokenize_vlm_preference_row(
    features: dict[str, Any],
    processing_class: ProcessorMixin,
    *,
    max_prompt_length: int | None,
    max_completion_length: int | None,
    truncation_mode: str,
    eos_token_ids: frozenset[int] = frozenset(),
) -> dict[str, Any]:
    """Prepare one raw VLM (prompt, chosen, rejected[, images]) example for SMPO: completions split off by the
    text pipeline's prefix strip and tokenized through :func:`tokenize_preference_row`; the prompt stays text plus
    PIL images for the collator to expand."""
    tokenizer = resolve_tokenizer(processing_class)
    prompt_history, pil_images, completions = split_vlm_preference_row(features, "VLM SMPO row")

    prompt_text = render_vlm_text(processing_class, prompt_history)
    completion_texts = {
        side: split_rendered_completion(prompt_text, render_vlm_text(processing_class, prompt_history + history), side)
        for side, history in completions.items()
    }

    tokenized = tokenize_preference_row(
        {"prompt": prompt_text, "chosen": completion_texts["chosen"], "rejected": completion_texts["rejected"]},
        tokenizer,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        truncation_mode=truncation_mode,
        eos_token_ids=eos_token_ids,
    )
    return {
        "prompt_text": prompt_text,
        "images": pil_images,
        "chosen_input_ids": tokenized["chosen_input_ids"],
        "rejected_input_ids": tokenized["rejected_input_ids"],
    }


def render_preference_row(
    row: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    tools_field: str | None = None,
) -> dict[str, str]:
    """Chat-template a normalized preference row's prompt, then each completion as ``template(prompt +
    completion)`` minus the rendered prompt (:func:`split_rendered_completion`); ``tools_field`` reaches the
    template."""
    row = normalize_preference_row(row)

    # DPO's run-intent dispatch reads image columns only, so an embedded-image row reaches here undetected.
    for field in ("prompt", "chosen", "rejected"):
        reject_image_content(row[field], f"preference field '{field}'")

    template_kwargs = chat_template_kwargs(row, interleaved_thinking=False, tools_field=tools_field)

    prompt_messages = row["prompt"]
    prompt_text = tokenizer.apply_chat_template(prompt_messages, tokenize=False, **template_kwargs)
    for field in ("chosen", "rejected"):
        full_text = tokenizer.apply_chat_template(prompt_messages + row[field], tokenize=False, **template_kwargs)
        row[field] = split_rendered_completion(prompt_text, full_text, field)
    row["prompt"] = prompt_text
    return row


def prepare_preference_datasets(
    train_dataset,
    eval_dataset,
    tokenizer: PreTrainedTokenizer,
    num_proc: int = DATASET_NUM_PROC,
    tools_field: str | None = None,
):
    """Apply chat templates to train and eval preference datasets; returns (train, eval)."""
    process_fn = partial(render_preference_row, tokenizer=tokenizer, tools_field=tools_field)
    train_dataset = coordinated_map(
        train_dataset, process_fn, desc="Applying chat template to train dataset", num_proc=num_proc
    )
    eval_dataset = coordinated_map(
        eval_dataset, process_fn, desc="Applying chat template to eval dataset", num_proc=num_proc
    )
    return train_dataset, eval_dataset


def _normalized_generative_row(row, tokenizer, max_length, tools_field=None):
    """A raw preference row to a tokenized generation prompt: normalized first, since the raw test split still
    carries hub shapes :func:`prepare_generative_row` (also offline GRPO's) cannot template."""
    return prepare_generative_row(
        normalize_preference_row(row), tokenizer=tokenizer, max_length=max_length, tools_field=tools_field
    )


def prepare_generative_dataset(
    dataset,
    tokenizer: PreTrainedTokenizer,
    max_prompt_length: int,
    num_proc: int = DATASET_NUM_PROC,
    tools_field: str | None = None,
):
    """Tokenize prompts (with generation prompt) for eval-time generation."""
    return coordinated_map(
        dataset,
        partial(
            _normalized_generative_row,
            tokenizer=tokenizer,
            max_length=max_prompt_length,
            tools_field=tools_field,
        ),
        num_proc=num_proc,
        desc="Preparing dataset for generation",
        cache_key_extras={"tools_field": tools_field},
    )
