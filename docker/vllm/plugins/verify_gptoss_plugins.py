"""Build-time checks for the gpt-oss parser plugins.

No test suite can cover these: they import vLLM, which the training image does not have
(ABI-incompatible stacks), so the image build is the only gate. Each check covers a failure mode that
produces no error at serve time, so a regression fails the build instead of the run.

Run against the installed plugins: ``python3 verify_gptoss_plugins.py /opt``
"""

import re
import sys

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else "/opt")

import gpt_oss_reasoning_parser as reasoning_plugin  # noqa: E402
import gpt_oss_text_tool_parser as tool_plugin  # noqa: E402
import torch  # noqa: E402
import vllm.utils.torch_utils as vllm_torch_utils  # noqa: E402
from vllm import SamplingParams  # noqa: E402
from vllm.reasoning import ReasoningParserManager  # noqa: E402
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager  # noqa: E402
from vllm.v1.sample.logits_processor.interface import BatchUpdate  # noqa: E402
from vllm.v1.sample.thinking_budget_state import ThinkingBudgetStateHolder  # noqa: E402

# vLLM's host->device helper pins memory, which needs a CUDA driver an image build does not have.
# Forcing is plain indexing, so unpinned CPU tensors drive the real state machine unchanged.
vllm_torch_utils.PIN_MEMORY = False

# ``tokenizer.encode(marker, add_special_tokens=False)`` on the o200k_harmony vocabulary
# (``unsloth/gpt-oss-20b-BF16``). Pinned rather than tokenized: the build has no model, and a render
# or vocabulary drift has to fail here rather than as an unbounded CoT in a training run.
HARMONY_MARKER_IDS = {
    "<|start|>assistant": [200006, 173781],
    "<|channel|>": [200005],
    "<|channel|>analysis<|message|>": [200005, 35644, 200008],
    "<|channel|>commentary<|message|>": [200005, 12606, 815, 200008],
    "<|channel|>final<|message|>": [200005, 17196, 200008],
    "<|start|>assistant<|channel|>final<|message|>": [200006, 173781, 200005, 17196, 200008],
    "<|end|>": [200007],
}
# gpt-oss ships a padded embedding; any ordinary (non-marker) id stands in for CoT and answer text.
VOCAB_SIZE = 201088
PROSE_ID = 1234
BUDGET = 16

# Prompt tails the harmony template renders with ``add_generation_prompt=True``. Both end in
# ``<|start|>assistant``; the tool-result one opens a channel after the last ``final`` marker, which
# a bare ``<|channel|>`` arming marker reads as reasoning already in flight and cuts immediately.
_USER_TURN = [200006, 1000, 200008, PROSE_ID, 200007]
SINGLE_TURN_PROMPT = _USER_TURN + HARMONY_MARKER_IDS["<|start|>assistant"]
TOOL_RESULT_PROMPT = (
    _USER_TURN
    + HARMONY_MARKER_IDS["<|channel|>commentary<|message|>"]
    + [PROSE_ID] * (3 * BUDGET)
    + HARMONY_MARKER_IDS["<|end|>"]
    + HARMONY_MARKER_IDS["<|start|>assistant"]
)

failures: list[str] = []


def check(label: str, ok: bool) -> None:
    print(f"    {'OK  ' if ok else 'FAIL'} {label}")
    if not ok:
        failures.append(label)


class _ReasoningTokenIds:
    """The two fields ``ThinkingBudgetStateHolder`` reads off a ``ReasoningConfig``.

    vLLM derives them by tokenizing the parser's marker strings
    (``ReasoningConfig.initialize_token_ids``), which needs a model the build does not have.
    """

    def __init__(self, start_ids: list[int], end_ids: list[int]):
        self.reasoning_start_token_ids = start_ids
        self.reasoning_end_token_ids = end_ids


def simulate_generation(
    start_ids: list[int],
    end_ids: list[int],
    prompt_ids: list[int],
    opening_ids: list[int],
    budget: int | None,
    max_new: int,
    natural_final_at: int | None = None,
) -> tuple[list[int], int | None]:
    """Decode ``max_new`` tokens through the real budget state machine.

    Mirrors the sampler's per-step order (``update_state`` over the committed output, then
    ``apply_to_logits`` for the token about to be sampled) and appends whatever the budget forces.
    ``natural_final_at`` makes the model open its own ``final`` channel once it has generated that
    many tokens, which distinguishes a run that finishes reasoning inside its budget from one that is
    cut. Returns the forced ids and how many tokens the model generated before the first of them.
    """
    output: list[int] = []
    holder = ThinkingBudgetStateHolder(
        _ReasoningTokenIds(start_ids, end_ids),
        max_num_seqs=1,
        num_spec_tokens=0,
        device=torch.device("cpu"),
        is_pin_memory=False,
    )
    # The holder keeps a reference to ``output``, as the batch does to a request's ids.
    holder.sync_batch(
        BatchUpdate(
            batch_size=1,
            removed=(),
            added=[(0, SamplingParams(thinking_token_budget=budget), list(prompt_ids), output)],
            moved=(),
        )
    )
    pending = list(opening_ids)
    forced: list[int] = []
    cut_at: int | None = None
    for _ in range(max_new):
        boosted: list[int] = []
        if holder.has_tracked_requests():
            holder.update_state([output], None)
            logits = holder.apply_to_logits(torch.zeros(1, VOCAB_SIZE), False, None)
            boosted = torch.nonzero(logits[0] > 1e8).flatten().tolist()
        if boosted:
            if cut_at is None:
                cut_at = len(output)
            forced.append(boosted[0])
            output.append(boosted[0])
            continue
        if not pending and natural_final_at is not None and len(output) >= natural_final_at:
            pending = (
                HARMONY_MARKER_IDS["<|end|>"]
                + HARMONY_MARKER_IDS["<|start|>assistant"]
                + HARMONY_MARKER_IDS["<|channel|>final<|message|>"]
            )
            natural_final_at = None
        output.append(pending.pop(0) if pending else PROSE_ID)
    return forced, cut_at


# Both plugins must resolve through the real managers under the name the serve command passes.
check(
    "reasoning parser registers as openai_gptoss",
    ReasoningParserManager.get_reasoning_parser("openai_gptoss") is reasoning_plugin.GptOssBudgetReasoningParser,
)
check(
    "tool parser registers as gpt_oss_text",
    ToolParserManager.get_tool_parser("gpt_oss_text") is tool_plugin.GptOssTextToolParser,
)

# tool_choice="required"/named must not take vLLM's standard-JSON branch: it parses the harmony
# channel text as a JSON tool list, swallows the ValidationError, and returns an empty message.
check(
    "tool parser opts out of the required/named JSON branch",
    tool_plugin.GptOssTextToolParser.supports_required_and_named is False,
)

parser_cls = reasoning_plugin.GptOssBudgetReasoningParser

# The base implementations raise (they assume harmony parses the output), so without these
# overrides every stream=true request 501s.
for name in ("extract_reasoning_streaming", "extract_content_ids"):
    check(
        f"{name} overridden (base raises NotImplementedError)",
        getattr(parser_cls, name).__qualname__.startswith(parser_cls.__name__),
    )


split = parser_cls._split
instance = object.__new__(parser_cls)  # _split reads no instance state

# Normal path: the model emits its own `<|start|>assistant<|channel|>final<|message|>`, which decodes
# to `assistantfinal`. The budget-forced cut appends the same ids, so it decodes the same way and
# takes this path, which holds because the marker carries the role tokens.
_, content = split(instance, "analysis17*20=340, 17*3=51.assistantfinal391")
check("assistantfinal split puts the answer in content", content == "391")

# A CoT cut mid-word, mid-space and while echoing the prompt's own "final", observed at budget 48 on
# unsloth/gpt-oss-20b-BF16. A marker without the role tokens decodes to a bare `final` here,
# indistinguishable from the prose one, leaving the whole answer inside reasoning.
reasoning, content = split(
    instance,
    'analysisThe user asks "showing every intermediate product before the final answer." Likely '
    "meaning to lay out theassistantfinalTo multiply 3847 by 2913, start with",
)
check("a cut CoT that quotes 'final' still splits at the marker", content == "To multiply 3847 by 2913, start with")
check("the quoted 'final' stays in reasoning", reasoning is not None and "before the final answer" in reasoning)

# The engine appends the end marker's ids verbatim, so the served text carries that marker minus its
# special tokens (the detokenizer skips them). _split has to find the answer behind it wherever the
# cut lands, whitespace included, where a role-less marker decodes to a bare `final` that no anchored
# pattern can distinguish from prose.
forced_text = re.sub(r"<\|[^|]*\|>", "", instance.reasoning_end_str)
_, content = split(instance, f"analysisthe intermediate products: {forced_text}Here is the answer.")
check("a budget cut landing on whitespace still splits at the forced marker", content == "Here is the answer.")

# Prose `final` alone is not a boundary: without the `assistant` anchor there is nothing to split on.
_, content = split(instance, "analysisI will give a short final answer soon.")
check("prose 'final' in the CoT does not split", content is None)

# A tool call has to reach content, since vLLM extracts tool calls from content only.
_, content = split(instance, 'analysisthinking commentary to=functions.get_weather json{"city":"NYC"}')
check("commentary tool call routed to content", content is not None and "to=functions." in content)

# ── thinking_token_budget arming ────────────────────────────────────────────────────────────────
# The parser's markers are what vLLM tokenizes into the budget's arm/force ids, and an arming marker
# the render never emits produces no error: the server takes the field and the CoT runs to
# max_tokens. The rows below drive vLLM's own state machine over a simulated harmony-disabled stream,
# so a marker or render drift fails the build.
start_ids = HARMONY_MARKER_IDS.get(instance.reasoning_start_str, [])
end_ids = HARMONY_MARKER_IDS.get(instance.reasoning_end_str, [])
check("arming marker is a pinned harmony marker", bool(start_ids))
check("end marker is a pinned harmony marker", bool(end_ids))

# The cut must be the end marker, once, at the budget: vLLM's continue_thinking count runs one token
# behind the arming marker, so it lands at budget+1. The window stays tight enough that an unarmed
# budget (no cut at all) and a prompt-armed one (cut at 0) both fail.
CUT_WINDOW = range(BUDGET, BUDGET + 3)
COMMENTARY_OPENING = HARMONY_MARKER_IDS["<|channel|>commentary<|message|>"]
for prompt_label, prompt in (("single-turn", SINGLE_TURN_PROMPT), ("tool-result", TOOL_RESULT_PROMPT)):
    for channel in ("analysis", "commentary"):
        opening = HARMONY_MARKER_IDS[f"<|channel|>{channel}<|message|>"]
        forced, cut_at = simulate_generation(start_ids, end_ids, prompt, opening, BUDGET, 4 * BUDGET)
        check(
            f"budget cuts a {channel} CoT after a {prompt_label} prompt (cut at {cut_at})",
            forced == end_ids and cut_at in CUT_WINDOW,
        )

# A run that opens its own `final` channel inside the budget is left alone: forcing there would cut
# the answer, and the arming marker must not re-arm on the answer text either.
forced, _ = simulate_generation(
    start_ids, end_ids, SINGLE_TURN_PROMPT, COMMENTARY_OPENING, BUDGET, 4 * BUDGET, natural_final_at=BUDGET // 2
)
check("a CoT that ends inside the budget is not cut", forced == [])

# Negative control: without the field nothing is forced, so the rows above measure the budget rather
# than the simulated stream.
forced, _ = simulate_generation(start_ids, end_ids, SINGLE_TURN_PROMPT, COMMENTARY_OPENING, None, 4 * BUDGET)
check("no thinking_token_budget forces nothing", forced == [])

if failures:
    print(f"\ngpt-oss plugin verification FAILED: {failures}")
    sys.exit(1)
print("gpt-oss plugin verification: OK")
