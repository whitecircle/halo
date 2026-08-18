"""Dataset-time preparation of preference corpora (DPO, SMPO, reward): row normalization, the
chat-template map for both sides of a pair, and the vision pair's render.

Batch-time collation of the vision pair is :mod:`src.data.collators.vlm_preference`.
"""

from collections.abc import Callable
from functools import partial
from typing import Any

from accelerate.logging import get_logger
from datasets import Dataset, Features, Sequence, Value
from datasets import Image as ImageFeature
from transformers import PreTrainedTokenizer, ProcessorMixin

from src.data.pipeline.conversation import chat_template_kwargs, reject_image_content
from src.data.pipeline.processing import DATASET_NUM_PROC, coordinated_map
from src.data.pipeline.rendered import tokenize_rendered
from src.data.pipeline.row_processors import normalize_vlm_conversation, prepare_generative_row
from src.data.vlm import VLM_RAW_IMAGE_COLUMNS, process_vlm_conversation, render_vlm_text

logger = get_logger(__name__)

__all__ = [
    "MARGIN_COLUMN",
    "VLM_PREFERENCE_COLUMNS",
    "apply_chat_template_to_preference_data",
    "build_reward_preprocess_fn",
    "normalize_preference_row",
    "prepare_preference_datasets",
    "prepare_generative_dataset",
    "render_vlm_preference_row",
    "split_rendered_completion",
    "split_vlm_preference_row",
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
    prompt = row.get("prompt")

    if isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]
    elif not prompt:
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


def build_reward_preprocess_fn(
    tokenizer: PreTrainedTokenizer,
    max_length: int,
    tools_field: str | None = None,
) -> Callable[[dict], dict]:
    """Build the batched Bradley-Terry reward tokenization map function.

    Chat-templates ``prompt + chosen`` and ``prompt + rejected`` (optional per-row tools from
    ``tools_field``), tokenizing each to ``max_length``. Datasets without a ``prompt`` column
    (implicit-prompt, e.g. Skywork-Reward: the shared turns live inside chosen/rejected) template
    chosen/rejected whole.
    """

    def preprocess_function(examples):
        new_examples = {
            "input_ids_chosen": [],
            "attention_mask_chosen": [],
            "input_ids_rejected": [],
            "attention_mask_rejected": [],
        }
        batch_size = len(examples["chosen"])
        prompts_batch = examples["prompt"] if "prompt" in examples else [[]] * batch_size
        tools_batch = examples[tools_field] if tools_field and tools_field in examples else [None] * batch_size
        for prompt, chosen, rejected, tools_raw in zip(
            prompts_batch,
            examples["chosen"],
            examples["rejected"],
            tools_batch,
            strict=False,
        ):
            # Batched columns carry the raw tools value, so wrap it as a single-key pseudo-row.
            template_kwargs = chat_template_kwargs({tools_field: tools_raw} if tools_field else {}, False, tools_field)

            for field, messages in (("prompt", prompt), ("chosen", chosen), ("rejected", rejected)):
                reject_image_content(messages, f"reward field '{field}'")

            chosen = tokenizer.apply_chat_template(
                prompt + chosen,
                tokenize=False,
                add_generation_prompt=False,
                **template_kwargs,
            )
            rejected = tokenizer.apply_chat_template(
                prompt + rejected,
                tokenize=False,
                add_generation_prompt=False,
                **template_kwargs,
            )

            tokenized_chosen = tokenize_rendered(tokenizer, chosen, truncation=True, max_length=max_length)
            tokenized_rejected = tokenize_rendered(tokenizer, rejected, truncation=True, max_length=max_length)

            new_examples["input_ids_chosen"].append(tokenized_chosen["input_ids"])
            new_examples["attention_mask_chosen"].append(tokenized_chosen["attention_mask"])
            new_examples["input_ids_rejected"].append(tokenized_rejected["input_ids"])
            new_examples["attention_mask_rejected"].append(tokenized_rejected["attention_mask"])

        return new_examples

    return preprocess_function


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
