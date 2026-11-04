"""Synthetic dataset generators and one real-text source shared across test files; a generator only one test needs
stays in that file."""

import random
from datetime import datetime

from datasets import Dataset, DatasetDict, Sequence
from datasets import Image as HFImage
from PIL import Image, ImageDraw

from halo.data.chat_render import tokenize_rendered
from halo.data.preprocessed.metadata import PreprocessingConfig
from halo.data.preprocessed.preparation import preprocess_dataset
from tests.common.models import QWEN3_0_6B
from tests.common.repo import REPO_ROOT
from tests.common.tokenizers import load_cached_tokenizer

# gpt-oss's harmony template stamps ``strftime_now`` into its system message, so an unpinned render and every loss
# threshold drift with the date. The value matters: at 2025-01-01 the rotated-expert control falls under
# ``control_min_loss_shift``.
CHAT_TEMPLATE_NOW = datetime(2026, 8, 26)

MATH_TEMPLATES = [
    {"q": "What is {a} + {b}?", "a": "The answer is {r}.", "op": "+"},
    {"q": "Calculate {a} * {b}.", "a": "{a} times {b} equals {r}.", "op": "*"},
    {"q": "What is {a} - {b}?", "a": "The result of {a} minus {b} is {r}.", "op": "-"},
    {"q": "What is the sum of {a} and {b}?", "a": "The sum is {r}.", "op": "+"},
    {"q": "Solve: {a} + {b} = ?", "a": "The solution is {r}.", "op": "+"},
]

FOLLOWUP_TEMPLATES = [
    ("Now what is {a} + {b}?", "That equals {r}."),
    ("And {a} * {b}?", "That gives {r}."),
    ("How about {a} - {b}?", "That is {r}."),
]


def _compute_result(a: int, b: int, op: str) -> int:
    if op == "+":
        return a + b
    elif op == "*":
        return a * b
    else:
        return a - b


def _apply_chat_template(tokenizer, messages, add_generation_prompt=False):
    """Apply the chat template, with a fallback for tokenizers lacking one; a render kwarg ``strftime_now`` shadows the
    jinja global to pin the clock."""
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            strftime_now=CHAT_TEMPLATE_NOW.strftime,
        )
    sep = "\n"
    parts = []
    for m in messages:
        role = "User" if m["role"] == "user" else "Assistant"
        parts.append(f"{role}: {m['content']}")
    return sep.join(parts)


def create_sft_dataset(
    num_samples: int,
    tokenizer,
    seed: int = 42,
    multi_turn_ratio: float = 0.3,
) -> Dataset:
    """Synthetic math SFT dataset of ``{"text": str}`` records, 30% multi-turn by default."""
    random.seed(seed)
    data = []
    for _ in range(num_samples):
        t = random.choice(MATH_TEMPLATES)
        a, b = random.randint(1, 100), random.randint(1, 100)
        r = _compute_result(a, b, t["op"])
        messages = [
            {"role": "user", "content": t["q"].format(a=a, b=b, r=r)},
            {"role": "assistant", "content": t["a"].format(a=a, b=b, r=r)},
        ]
        if random.random() < multi_turn_ratio:
            ft = random.choice(FOLLOWUP_TEMPLATES)
            a2, b2 = random.randint(1, 50), random.randint(1, 50)
            op2 = "+" if "+" in ft[0] else ("*" if "*" in ft[0] else "-")
            r2 = _compute_result(a2, b2, op2)
            messages.extend(
                [
                    {"role": "user", "content": ft[0].format(a=a2, b=b2, r=r2)},
                    {"role": "assistant", "content": ft[1].format(a=a2, b=b2, r=r2)},
                ]
            )
        text = _apply_chat_template(tokenizer, messages)
        data.append({"text": text})
    return Dataset.from_list(data)


def create_tokenized_sft_dataset(num_samples: int, tokenizer, seed: int = 42) -> Dataset:
    """:func:`create_sft_dataset`'s rows tokenized as the SFT script does, unpadded, for its packing pass and
    padding-free collator."""
    rows = create_sft_dataset(num_samples, tokenizer, seed=seed)
    return rows.map(lambda row: dict(tokenize_rendered(tokenizer, row["text"])), remove_columns=rows.column_names)


# Separate from ``create_sft_dataset``: multi-turn rows change token counts and so the loss trajectory of suites pinned
# on this data.

SINGLE_TURN_TEMPLATES = (
    ("What is {a} + {b}?", "The answer is {result}."),
    ("Calculate {a} * {b}.", "{a} times {b} equals {result}."),
    ("What is {a} - {b}?", "The result of {a} minus {b} is {result}."),
    ("What is the sum of {a} and {b}?", "The sum of {a} and {b} is {result}."),
)

# Longer answers for suites that need more tokens per row.
VERBOSE_MATH_TEMPLATES = (
    ("What is {a} + {b}?", "The answer is {result}. To calculate this, I added {a} and {b} together."),
    ("Calculate {a} * {b}.", "{a} times {b} equals {result}. This is found by multiplying the two numbers."),
    ("What is {a} - {b}?", "The result of {a} minus {b} is {result}."),
    ("Explain how to compute {a} + {b}.", "To compute {a} + {b}, simply add the two numbers: {a} + {b} = {result}."),
    ("What is the sum of {a} and {b}?", "The sum of {a} and {b} is {result}."),
    ("Multiply {a} by {b} and explain.", "Multiplying {a} by {b}: {a} * {b} = {result}. I multiplied the two values."),
    (
        "What do you get when you subtract {b} from {a}?",
        "When you subtract {b} from {a}, you get {result}. This is basic subtraction.",
    ),
)


def create_single_turn_sft_dataset(
    num_samples: int,
    tokenizer,
    seed: int = 42,
    templates: tuple[tuple[str, str], ...] = SINGLE_TURN_TEMPLATES,
) -> Dataset:
    """Synthetic single-turn math SFT dataset; the operator comes from the instruction wording, so a new template must
    spell it the same way."""
    random.seed(seed)
    data = []
    for _ in range(num_samples):
        instruction_template, response_template = random.choice(templates)
        a, b = random.randint(1, 100), random.randint(1, 100)
        if "+" in instruction_template or "sum" in instruction_template:
            result = a + b
        elif "*" in instruction_template or "Multiply" in instruction_template:
            result = a * b
        else:
            result = a - b
        instruction = instruction_template.format(a=a, b=b, result=result)
        response = response_template.format(a=a, b=b, result=result)
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
        data.append({"text": _apply_chat_template(tokenizer, messages)})
    return Dataset.from_list(data)


PREFERENCE_TEMPLATES = [
    {
        "q": "What is {a} + {b}?",
        "chosen": "The answer is {correct}. Adding {a} and {b} gives {correct}.",
        "rejected": "The answer is {incorrect}.",
        "op": "+",
    },
    {
        "q": "Calculate {a} * {b}.",
        "chosen": "{a} times {b} equals {correct}.",
        "rejected": "{a} times {b} equals {incorrect}.",
        "op": "*",
    },
    {
        "q": "What is {a} - {b}?",
        "chosen": "The result of {a} minus {b} is {correct}.",
        "rejected": "The result is {incorrect}.",
        "op": "-",
    },
]


def create_preference_dataset(
    num_samples: int,
    tokenizer,
    seed: int = 42,
    multi_turn_ratio: float = 0.3,
) -> Dataset:
    """Synthetic math preference dataset of ``{"prompt", "chosen", "rejected"}`` strings."""
    random.seed(seed)
    data = []
    for _ in range(num_samples):
        t = random.choice(PREFERENCE_TEMPLATES)
        a, b = random.randint(1, 100), random.randint(1, 100)
        correct = _compute_result(a, b, t["op"])
        incorrect = correct + random.randint(1, 10) * random.choice([-1, 1])

        messages = [{"role": "user", "content": t["q"].format(a=a, b=b)}]

        if random.random() < multi_turn_ratio:
            messages.append({"role": "assistant", "content": f"Let me compute that. The answer is {correct}."})
            a2, b2 = random.randint(1, 50), random.randint(1, 50)
            messages.append({"role": "user", "content": f"Now what is {a2} + {b2}?"})
            correct = a2 + b2
            incorrect = correct + random.randint(1, 5)

        prompt = _apply_chat_template(tokenizer, messages, add_generation_prompt=True)
        chosen = t["chosen"].format(a=a, b=b, correct=correct, incorrect=incorrect)
        rejected = t["rejected"].format(a=a, b=b, correct=correct, incorrect=incorrect)
        data.append({"prompt": prompt, "chosen": chosen, "rejected": rejected})
    return Dataset.from_list(data)


# A group samples them without replacement, so no two completions in a group share a wording.
GRPO_ANSWER_TEMPLATES = [
    "The answer is {val}.",
    "The result equals {val}.",
    "That gives us {val}.",
    "I calculate the answer as {val}.",
    "It computes to {val}.",
    "After calculation, I get {val}.",
    "The final result is {val}.",
    "Working it out, the answer is {val}.",
]

# Best to worst: completion ``i`` drifts further from the correct value, so a group's z-normalized advantages carry
# both signs.
GRPO_REWARDS = (1.0, 0.7, 0.3, 0.0)

GRPO_FOLLOWUP_TEMPLATES = (
    ("Now add {c} to the result.", "+"),
    ("Subtract {c} from that.", "-"),
    ("Multiply the result by {c}.", "*"),
)


def create_offline_grpo_dataset(
    tokenizer,
    num_samples: int,
    seed: int = 42,
    num_completions: int = 4,
    multi_turn_ratio: float = 0.0,
) -> Dataset:
    """Synthetic math offline-GRPO dataset; completions and rewards shuffle together, so position never encodes
    quality."""
    rng = random.Random(seed)
    data = []
    for _ in range(num_samples):
        t = rng.choice(MATH_TEMPLATES)
        a, b = rng.randint(1, 100), rng.randint(1, 100)
        correct = _compute_result(a, b, t["op"])
        messages = [{"role": "user", "content": t["q"].format(a=a, b=b)}]

        if rng.random() < multi_turn_ratio:
            a2, b2 = rng.randint(1, 50), rng.randint(1, 50)
            first = _compute_result(a2, b2, "+")
            c = rng.randint(1, 30)
            followup, op2 = rng.choice(GRPO_FOLLOWUP_TEMPLATES)
            correct = _compute_result(first, c, op2)
            messages = [
                {"role": "user", "content": f"What is {a2} + {b2}?"},
                {"role": "assistant", "content": f"The sum is {first}."},
                {"role": "user", "content": followup.format(c=c)},
            ]

        prompt = _apply_chat_template(tokenizer, messages, add_generation_prompt=True)

        offsets = [0, rng.choice([1, -1]), rng.choice([5, -5, 10, -10]), rng.randint(20, 50)]
        answers = rng.sample(GRPO_ANSWER_TEMPLATES, num_completions)
        completions = [answers[i].format(val=correct + offsets[i % len(offsets)]) for i in range(num_completions)]
        rewards = [GRPO_REWARDS[i % len(GRPO_REWARDS)] for i in range(num_completions)]

        combined = list(zip(completions, rewards, strict=False))
        rng.shuffle(combined)
        completions, rewards = zip(*combined, strict=False)
        data.append({"prompt": prompt, "completions": list(completions), "rewards": list(rewards)})
    return Dataset.from_list(data)


_FILLER_SENTENCE = " The quick brown fox jumps over the lazy dog."


def create_variable_length_sft_dataset(
    tokenizer,
    max_length: int,
    num_samples: int = 256,
    avg_ratio: float = 0.25,
    seed: int = 42,
) -> Dataset:
    """Variable-length SFT dataset for collator benchmarks; lengths are uniform over 0.5 to 1.5 of ``avg_ratio *
    max_length`` tokens."""
    random.seed(seed)

    avg_tokens = int(max_length * avg_ratio)
    min_tokens = max(50, int(avg_tokens * 0.5))
    max_tokens = min(max_length - 50, int(avg_tokens * 1.5))

    filler_toks = len(tokenizer.encode(_FILLER_SENTENCE))
    chars_per_token = len(_FILLER_SENTENCE) / max(filler_toks, 1)

    data = []
    for _ in range(num_samples):
        target_tokens = random.randint(min_tokens, max_tokens)
        t = random.choice(MATH_TEMPLATES)
        a, b = random.randint(1, 100), random.randint(1, 100)
        r = _compute_result(a, b, t["op"])

        instruction = t["q"].format(a=a, b=b, r=r)
        response = t["a"].format(a=a, b=b, r=r)

        base_messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
        base_text = _apply_chat_template(tokenizer, base_messages)
        base_tokens = len(tokenizer.encode(base_text))

        remaining = max(0, target_tokens - base_tokens)
        filler_chars = int(remaining * chars_per_token)
        filler = (_FILLER_SENTENCE * ((filler_chars // len(_FILLER_SENTENCE)) + 1))[:filler_chars]

        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response + filler},
        ]
        text = _apply_chat_template(tokenizer, messages)
        data.append({"text": text})

    return Dataset.from_list(data)


def create_benchmark_dataset(tokenizer, seq_len: int, num_samples: int = 64, seed: int = 0) -> Dataset:
    """The profiling corpus: ``num_samples`` rows of ``seq_len`` uniform token ids from the regular vocabulary,
    identical on every rank. Uniform ids give little to fit, though an MoE's routing still moves over the first steps,
    so measure past them."""
    rng = random.Random(seed)
    rows = [[rng.randrange(tokenizer.vocab_size) for _ in range(seq_len)] for _ in range(num_samples)]
    return Dataset.from_dict(
        {"input_ids": rows, "attention_mask": [[1] * seq_len for _ in rows], "labels": [list(row) for row in rows]}
    )


def repository_text_rows(tokenizer, rows: int, length: int) -> list[list[int]]:
    """``rows`` consecutive windows of ``length`` token ids over the tree's ``agent-docs`` prose in path order. Raises
    if too few tokens."""
    ids: list[int] = []
    for page in sorted((REPO_ROOT / "agent-docs").rglob("*.md"), key=str):
        ids += tokenizer(page.read_text(encoding="utf-8"), add_special_tokens=False)["input_ids"]
        if len(ids) >= rows * length:
            return [ids[row * length : (row + 1) * length] for row in range(rows)]
    raise ValueError(f"agent-docs holds {len(ids)} tokens, fewer than {rows} rows of {length}")


def digit_image(d: int, size: int = 64) -> Image.Image:
    """A white tile carrying the digit ``d``, the image half of the read-a-digit VLM tasks."""
    img = Image.new("RGB", (size, size), "white")
    ImageDraw.Draw(img).text((size // 3, size // 4), str(d), fill="black")
    return img


def create_vlm_preference_dataset(n: int) -> Dataset:
    """Synthetic read-a-digit vision preference triples: chosen states the digit, rejected lies."""
    rows = []
    for i in range(n):
        d = i % 10
        wrong = (d + 1) % 10
        rows.append(
            {
                "images": [digit_image(d)],
                "prompt": [
                    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "What digit is shown?"}]}
                ],
                "chosen": [{"role": "assistant", "content": [{"type": "text", "text": f"The digit is {d}."}]}],
                "rejected": [{"role": "assistant", "content": [{"type": "text", "text": f"The digit is {wrong}."}]}],
            }
        )
    return Dataset.from_list(rows).cast_column("images", Sequence(HFImage()))


def write_prepared_text_dataset(output_dir, data: Dataset | DatasetDict, *, num_shards: int = 1) -> None:
    """Write ``data`` as a text-mode ``prepare_dataset`` artifact; uses the cached Qwen3-0.6B tokenizer, so the test
    skips where it is not cached."""
    config = PreprocessingConfig(
        model_name_or_path=QWEN3_0_6B, mode="text", max_length=64, num_shards=num_shards, num_proc=1
    )
    preprocess_dataset(data, load_cached_tokenizer(QWEN3_0_6B), config, output_dir=str(output_dir))
