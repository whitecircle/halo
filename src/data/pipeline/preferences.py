"""Dataset-time preparation of preference corpora (DPO, SMPO, reward): row normalization, the
chat-template map for both sides of a pair, the vision pair's render, and SMPO's row tokenizers.

Batch-time collation of the vision pair is :mod:`src.data.collators.vlm_preference`.
"""

from functools import partial
from typing import Any

from datasets import Dataset, Features, Sequence, Value
from datasets import Image as ImageFeature
from transformers import PreTrainedTokenizer, PreTrainedTokenizerBase, ProcessorMixin

from src.data.pipeline.conversation import as_conversation, chat_template_kwargs, reject_image_content
from src.data.pipeline.processing import DATASET_NUM_PROC, coordinated_map
from src.data.pipeline.rendered import lacks_emitted_bos
from src.data.pipeline.row_processors import normalize_vlm_conversation, prepare_generative_row
from src.data.spans import lacks_terminator
from src.data.vlm import VLM_RAW_IMAGE_COLUMNS, process_vlm_conversation, render_vlm_text
from src.models.structure import resolve_tokenizer

__all__ = [
    "MARGIN_COLUMN",
    "VLM_PREFERENCE_COLUMNS",
    "apply_chat_template_to_preference_data",
    "normalize_preference_row",
    "prepare_preference_datasets",
    "prepare_generative_dataset",
    "render_vlm_preference_row",
    "split_rendered_completion",
    "split_vlm_preference_row",
    "tokenize_preference_row",
    "tokenize_vlm_preference_row",
    "vlm_preference_features",
]

# The columns :func:`render_vlm_preference_row` writes and the VLM preference collator consumes.
VLM_PREFERENCE_COLUMNS = ("chosen_text", "rejected_text", "images")

# TRL's optional per-pair margin, carried through the map unchanged (its collator reads this name).
MARGIN_COLUMN = "margin"


def _shared_message_prefix_len(a: list[dict[str, Any]], b: list[dict[str, Any]]) -> int:
    """Length of the longest leading span where ``a`` and ``b`` agree on role + content."""
    n = 0
    for x, y in zip(a, b, strict=False):
        if x.get("role") != y.get("role") or x.get("content") != y.get("content"):
            break
        n += 1
    return n


def split_rendered_completion(prompt_text: str, full_text: str, field: str) -> str:
    """Return ``full_text`` (= template(prompt + completion)) minus its ``prompt_text`` prefix.

    Completions are never rendered standalone: strict templates (Qwen3.5) raise on assistant-only
    message lists, and the prefix strip guarantees ``prompt_text + completion_text`` reconstructs the
    full rendered conversation exactly. Raises on templates that break the prefix invariant.
    """
    if not full_text.startswith(prompt_text):
        raise ValueError(
            f"Chat template broke the prefix invariant: template(prompt + {field}) does not start "
            "with template(prompt), so the completion cannot be split off. "
            f"prompt head: {prompt_text[:120]!r} / full head: {full_text[:120]!r}"
        )
    return full_text[len(prompt_text) :]


def normalize_preference_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize hub preference shapes to the pipeline contract (prompt = message list,
    chosen/rejected = continuation-only message lists).

    The three hub shapes: a plain-string ``prompt`` becomes a user turn; chosen/rejected that repeat
    the prompt turns have that prefix stripped; a missing prompt column (Skywork-Reward style) is
    extracted as the longest shared leading span of chosen and rejected. Contract-shaped rows pass
    through. A continuation that ends up empty or not on an assistant turn raises, since that
    indicates a mis-mapped dataset rather than a shape to infer.
    """
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
    """One raw VLM ``(prompt, chosen, rejected[, images])`` row to prompt history, images, completions.

    The pre-render step every VLM preference route shares: normalize to the preference contract,
    merge a raw image column into the prompt conversation, then extract the images back out, leaving
    the placeholders the processor expands at collation. ``subject`` names the route in the
    images-in-completion refusal, which confines images to the shared prompt prefix, the only region
    both sides of a pair render identically.
    """
    row = normalize_preference_row({key: features[key] for key in ("prompt", "chosen", "rejected") if key in features})

    images = next((features[column] for column in VLM_RAW_IMAGE_COLUMNS if features.get(column) is not None), None)
    prompt_history, pil_images = process_vlm_conversation(normalize_vlm_conversation(row["prompt"], images))

    completions = {}
    for field in ("chosen", "rejected"):
        completion_history, completion_images = process_vlm_conversation(row[field])
        if completion_images:
            raise ValueError(
                f"{subject} carries images inside the '{field}' completion — completions must be "
                f"text-only (images belong in the prompt, which both sides share)."
            )
        completions[field] = completion_history
    return prompt_history, pil_images, completions


def vlm_preference_features(dataset: Dataset) -> Features:
    """Arrow schema for the :func:`render_vlm_preference_row` map output.

    Pinned rather than inferred, for the same reason the SFT VLM map pins its own: shard-wise
    inference diverges on a mixed dataset (an all-text shard infers ``images`` as ``List(null)``
    while an image-bearing one infers ``List(Image)``), and a multiprocess map then fails to align
    the shards. Declaring ``images`` as an ``Image`` sequence is also what makes a mapped row hand
    back PIL objects instead of the encoded bytes.

    ``margin`` is the only source column the map keeps, so it belongs in the schema exactly when the
    dataset carries it: a pinned schema must name every column of the mapped table.
    """
    features = {
        "chosen_text": Value("string"),
        "rejected_text": Value("string"),
        "images": Sequence(ImageFeature()),
    }
    if MARGIN_COLUMN in dataset.column_names:
        features[MARGIN_COLUMN] = Value("float32")
    return Features(features)


def render_vlm_preference_row(features: dict[str, Any], processing_class: ProcessorMixin) -> dict[str, Any]:
    """Render one raw VLM ``(prompt, chosen, rejected[, images])`` preference row.

    The pre-render step is :func:`split_vlm_preference_row`, shared with the SMPO vision route. Each
    side is then chat-templated whole: the reward head scores the full sequence, so there is no
    prompt/completion split to preserve and no prefix-strip invariant to rely on. The render goes
    through :func:`~src.data.vlm.render_vlm_text`, shared by every VLM path, so a conversation
    tokenizes identically here and on the SFT path. Pixels never reach the Arrow cache: the images
    travel as PIL objects and the patch geometry is a property of the processor call.
    """
    prompt_history, pil_images, completions = split_vlm_preference_row(features, "VLM preference row")
    rendered = {
        f"{field}_text": render_vlm_text(processing_class, prompt_history + history)
        for field, history in completions.items()
    }
    return {**rendered, "images": pil_images}


def tokenize_preference_row(
    features: dict[str, str],
    processing_class: PreTrainedTokenizerBase,
    *,
    max_prompt_length: int | None,
    max_completion_length: int | None,
    truncation_mode: str,
    eos_token_ids: frozenset[int] = frozenset(),
) -> dict[str, list[int]]:
    """Tokenize one (prompt, chosen, rejected) example.

    Tokenizes full prompt+completion sequences to handle boundary token-merging correctly.
    """
    prompt = features["prompt"]
    chosen = features["chosen"]
    rejected = features["rejected"]

    prompt_tokens = processing_class(prompt, add_special_tokens=False)["input_ids"]

    full_chosen = processing_class(prompt + chosen, add_special_tokens=False)
    full_rejected = processing_class(prompt + rejected, add_special_tokens=False)

    # Token merging at the prompt/completion boundary shifts the split point by one. The prompt must
    # move with it: the trainer concatenates prompt_input_ids ⧺ completion_input_ids verbatim, so
    # keeping the full prompt while the completion starts one token earlier duplicates the boundary
    # token. The split must be the same for both completions — they share one prompt field, and a
    # per-side split would condition chosen and rejected on different contexts, making the margin a
    # comparison between two different prompts.
    split = len(prompt_tokens)
    if full_chosen["input_ids"][:split] != prompt_tokens or full_rejected["input_ids"][:split] != prompt_tokens:
        # full_*[: split - 1] == prompt_tokens[: split - 1] holds on the non-merging side too, so
        # rolling both back stays exact there.
        split -= 1
        prompt_tokens = prompt_tokens[:split]

    chosen_input_ids = full_chosen["input_ids"][split:]
    rejected_input_ids = full_rejected["input_ids"][split:]

    if lacks_emitted_bos(prompt_tokens, processing_class):
        prompt_tokens = [processing_class.bos_token_id] + prompt_tokens

    # A spurious ender would sit inside the mean log-prob the SMPO margin is computed from.
    eos_id = processing_class.eos_token_id
    if lacks_terminator(chosen_input_ids, processing_class, eos_token_ids):
        chosen_input_ids = chosen_input_ids + [eos_id]
    if lacks_terminator(rejected_input_ids, processing_class, eos_token_ids):
        rejected_input_ids = rejected_input_ids + [eos_id]

    if max_prompt_length and len(prompt_tokens) > max_prompt_length:
        if truncation_mode == "keep_start":
            prompt_tokens = prompt_tokens[:max_prompt_length]
        else:  # keep_end
            prompt_tokens = prompt_tokens[-max_prompt_length:]

    if max_completion_length:
        # A plain tail slice would cut the EOS appended above, so the model never learns to stop there.
        def _truncate_keep_eos(ids: list[int]) -> list[int]:
            if len(ids) <= max_completion_length:
                return ids
            if eos_id is None:
                return ids[:max_completion_length]
            return ids[: max_completion_length - 1] + [eos_id]

        chosen_input_ids = _truncate_keep_eos(chosen_input_ids)
        rejected_input_ids = _truncate_keep_eos(rejected_input_ids)

    return {
        "prompt_input_ids": prompt_tokens,
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
    """Prepare one raw VLM (prompt, chosen, rejected [, images]) example.

    The pre-render half is :func:`split_vlm_preference_row`, shared with the VLM reward map.
    Completions are then rendered by the same prefix-strip invariant as the text pipeline
    (``template(prompt + completion)`` minus ``template(prompt)``) and tokenized through
    :func:`tokenize_preference_row`, so boundary merges, EOS appending and truncation stay
    byte-identical to it. The prompt stays text + PIL images; the collator expands placeholders per
    batch.
    """
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


def apply_chat_template_to_preference_data(
    row: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    tools_field: str | None = None,
) -> dict[str, str]:
    """Apply the chat template to the prompt/chosen/rejected fields.

    Rows are normalized first (:func:`normalize_preference_row`). Completions render as
    ``template(prompt + completion)`` minus the rendered-prompt prefix, never standalone: strict
    templates raise on assistant-only message lists, and the prefix strip is what makes
    ``prompt_text + completion_text`` reconstruct the conversation the trainers tokenize.
    ``tools_field`` (list of dicts or JSON string) is forwarded into apply_chat_template.
    """
    row = normalize_preference_row(row)

    # An image-carrying preference dataset belongs on TRL's vision path. DPO declares no
    # conversation_field, so its run-intent dispatch reads the image columns only and an
    # embedded-image row reaches here undetected.
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
    process_fn = partial(
        apply_chat_template_to_preference_data,
        tokenizer=tokenizer,
        tools_field=tools_field,
    )

    train_dataset = coordinated_map(
        train_dataset,
        process_fn,
        desc="Applying chat template to train dataset",
        num_proc=num_proc,
    )

    eval_dataset = coordinated_map(
        eval_dataset,
        process_fn,
        desc="Applying chat template to eval dataset",
        num_proc=num_proc,
    )

    return train_dataset, eval_dataset


def _normalized_generative_row(row, tokenizer, max_length, tools_field=None):
    """A raw preference row to a tokenized generation prompt, through the contract normalizer.

    ``prepare_generative_row`` templates ``row["prompt"]`` as a message list, but the raw test split
    still carries the hub shapes, so a string prompt would fail inside the chat template. The
    normalization is applied here rather than inside that helper, which is family-generic (offline
    GRPO maps it over rows with no chosen/rejected). Module-level so ``num_proc`` maps can pickle it.
    """
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
