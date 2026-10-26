"""Build-time checks for the gpt-oss parser plugins and the thinking-budget patches they rely on.

No test suite can cover these: they import vLLM, which the training image does not have
(ABI-incompatible stacks), so the image build is the only gate. Each check covers a failure mode that
produces no error at serve time, so a regression fails the build instead of the run.

The budget rows drive Model Runner V2's own Triton kernels on the Triton interpreter, since the build
has no GPU: ``CUDA_VISIBLE_DEVICES= TRITON_INTERPRET=1 python3 verify_gptoss_plugins.py /opt``
"""

import re
import sys

sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else "/opt")

import gpt_oss_reasoning_parser as reasoning_plugin  # noqa: E402
import gpt_oss_text_tool_parser as tool_plugin  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import vllm.config.reasoning as reasoning_config_module  # noqa: E402
import vllm.v1.worker.gpu.sample.thinking_budget as thinking_budget  # noqa: E402
import vllm_reasoning_forced_end_patch as forced_end_patch  # noqa: E402
import vllm_thinking_budget_completion_patch as completion_patch  # noqa: E402
from vllm import SamplingParams  # noqa: E402
from vllm.reasoning import ReasoningParserManager  # noqa: E402
from vllm.tool_parsers.abstract_tool_parser import ToolParserManager  # noqa: E402
from vllm.v1.worker.gpu.sample.logits_processor import LogitsContext  # noqa: E402

# ``tokenizer.encode(text, add_special_tokens=False)`` on the o200k_harmony vocabulary
# (``unsloth/gpt-oss-20b-BF16``). Pinned rather than tokenized: the build has no model, and a render
# or vocabulary drift has to fail here rather than as an unbounded CoT in a training run.
HARMONY_IDS = {
    "<|start|>assistant": [200006, 173781],
    "<|channel|>": [200005],
    "<|message|>": [200008],
    "<|end|>": [200007],
    "<|call|>": [200012],
    "<|return|>": [200002],
    "analysis<|message|>": [35644, 200008],
    "<|channel|>final": [200005, 17196],
    "<|channel|>analysis<|message|>": [200005, 35644, 200008],
    "<|channel|>final<|message|>": [200005, 17196, 200008],
    "<|channel|>commentary<|message|>": [200005, 12606, 815, 200008],
    "<|end|><|start|>assistant<|channel|>final<|message|>": [200007, 200006, 173781, 200005, 17196, 200008],
}
H = HARMONY_IDS
# A tool call in the channel-first spelling and in the recipient-first one the chat template renders
# for earlier calls, which the model also writes:
# ``<|channel|>commentary to=functions.get_weather <|constrain|>json<|message|>`` and
# `` to=functions.get_weather<|channel|>commentary json<|message|>``.
CALL_CHANNEL_FIRST = [200005, 12606, 815, 316, 28, 44580, 775, 170154, 220, 200003, 4108, 200008]
CALL_RECIPIENT_FIRST = [316, 28, 44580, 775, 170154, 200005, 12606, 815, 5701, 200008]
# gpt-oss ships a padded embedding; any ordinary (non-marker) id stands in for CoT, call and answer text.
VOCAB_SIZE = 201088
PROSE = [1234]
BUDGET = 12
STOP_IDS = {H["<|call|>"][0], H["<|return|>"][0]}

# Prompt tails the harmony template renders with ``add_generation_prompt=True``; both end in
# ``<|start|>assistant``, the second after an earlier call and its tool result.
_USER_TURN = [200006, 1000, 200008, 1234, 200007]
SINGLE_TURN_PROMPT = _USER_TURN + H["<|start|>assistant"]
TOOL_RESULT_PROMPT = (
    _USER_TURN
    + H["<|start|>assistant"]
    + CALL_RECIPIENT_FIRST
    + PROSE * 4
    + H["<|call|>"]
    + [200006, 44580, 316, 28, 173781]
    + H["<|channel|>commentary<|message|>"]
    + PROSE * 4
    + H["<|end|>"]
    + H["<|start|>assistant"]
)
TO_FINAL = H["<|end|><|start|>assistant<|channel|>final<|message|>"]
ANSWER = PROSE * 4 + H["<|return|>"]

failures: list[str] = []


def check(label: str, ok: bool) -> None:
    print(f"    {'OK  ' if ok else 'FAIL'} {label}")
    if not ok:
        failures.append(label)


def analysis(n: int) -> list[int]:
    return H["<|channel|>analysis<|message|>"] + PROSE * n


class _PinnedTokenizer:
    """``encode`` off the pinned table, the one tokenizer call marker resolution makes."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return list(HARMONY_IDS[text])


class _HostTensor:
    """Host stand-in for ``UvaBackedTensor``, whose buffer pool needs a CUDA driver."""

    def __init__(self, size, dtype, max_concurrency=None):
        self.gpu = torch.zeros(size, dtype=dtype)
        self.np = self.gpu.numpy()

    def copy_to_uva(self, n=None):
        return self.gpu


class _Slots:
    """The two per-slot buffers ``ThinkingBudgetState`` reads off the runner's request state."""

    def __init__(self, max_num_reqs: int, max_len: int):
        self.max_num_reqs = max_num_reqs
        self.device = torch.device("cpu")
        self.all_token_ids = _HostTensor((max_num_reqs, max_len), torch.int32)
        self.total_len = _HostTensor(max_num_reqs, torch.int32)


thinking_budget.UvaBackedTensor = _HostTensor
thinking_budget.async_tensor_h2d = lambda data, device=None, dtype=None, out=None: torch.tensor(data, dtype=dtype)


class _Model:
    """Deterministic decoder: writes ``plan``; once cut, closes the cut channel and opens ``choice``."""

    def __init__(self, plan: list[int], choice: str = "final"):
        self.plan = list(plan)
        self.choice = choice
        self.cut = False
        self.cut_at = 0

    def next_tokens(self, written: list[int], count: int) -> list[int]:
        if not self.cut:
            return self.plan[len(written) : len(written) + count]
        header = H["<|end|>"] + H["<|start|>assistant"] + H["<|channel|>"]
        tail = written[self.cut_at :]
        if len(tail) < len(header):
            return header[len(tail) :][:count]
        # The channel the model would open; if something else was forced there, it answers.
        if len(tail) == len(header):
            channel = [17196] if self.choice == "final" else [35644]
            return (channel + H["<|message|>"] + ANSWER)[:count]
        return (H["<|message|>"] + ANSWER)[len(tail) - len(header) - 1 :][:count] or ANSWER[-1:]


def simulate(prompt, plan, budget, drafts=0, choice="final", state=None, slots=None):
    """Decode through V2's budget state, a perfect drafter proposing ``drafts`` tokens per step.

    Returns the forced ids and the output position of the first. A forced token wins its row; a draft
    that disagrees with its row's token ends the step, as rejection sampling does.
    """
    max_new = 8 * BUDGET
    slots = slots or _Slots(1, len(prompt) + max_new + drafts + 8)
    state = state or thinking_budget.ThinkingBudgetState(slots, _RESOLVED)
    model = _Model(plan, choice)
    state.add_request(0, SamplingParams(thinking_token_budget=budget, max_tokens=max_new))
    seq = list(prompt)
    slots.all_token_ids.gpu[0].zero_()
    slots.all_token_ids.gpu[0, : len(seq)] = torch.tensor(seq)
    slots.total_len.gpu[0] = len(seq)
    state.apply_staged_writes()
    forced, cut_at = [], None
    while len(seq) - len(prompt) < max_new:
        written = seq[len(prompt) :]
        proposal = model.next_tokens(written, drafts)
        rows = 1 + len(proposal)
        ctx = LogitsContext(
            expanded_idx_mapping=torch.zeros(rows, dtype=torch.int32),
            idx_mapping=torch.zeros(1, dtype=torch.int32),
            idx_mapping_np=np.zeros(1, dtype=np.int64),
            expanded_local_pos=torch.arange(rows, dtype=torch.int32),
            input_ids=torch.tensor(seq[-1:] + proposal, dtype=torch.int32),
            pos=torch.arange(len(seq) - 1, len(seq) - 1 + rows),
            seq_lens_upper_bound_np=np.array([len(seq) + len(proposal)]),
        )
        logits = torch.zeros(rows, VOCAB_SIZE)
        state.apply(logits, ctx)
        for row in range(rows):
            boosted = torch.nonzero(logits[row] > 1e8).flatten().tolist()
            natural = model.next_tokens(seq[len(prompt) :], 1)
            token = boosted[0] if boosted else (natural[0] if natural else H["<|return|>"][0])
            if boosted:
                if cut_at is None:
                    cut_at = len(seq) - len(prompt)
                if not model.cut:
                    model.cut, model.cut_at = True, cut_at
                forced.append(token)
            seq.append(token)
            if token in STOP_IDS or (row < len(proposal) and proposal[row] != token):
                break
        slots.all_token_ids.gpu[0, : len(seq)] = torch.tensor(seq)
        slots.total_len.gpu[0] = len(seq)
        if seq[-1] in STOP_IDS:
            break
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

# Normal path: the model emits its own `<|end|><|start|>assistant<|channel|>final<|message|>`, which
# decodes to `assistantfinal`. The budget-forced cut appends the same ids, so it decodes the same way.
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

# The engine appends the forced ids verbatim, so the served text carries the forced sequence minus its
# special tokens (the detokenizer skips them); the answer behind it splits wherever the cut lands.
forced_end = getattr(parser_cls, "reasoning_forced_end_str", "")
check("the parser names the sequence the budget forces", bool(forced_end))
forced_text = re.sub(r"<\|[^|]*\|>", "", forced_end)
_, content = split(instance, f"analysisthe intermediate products: {forced_text}Here is the answer.")
check("a budget cut landing on whitespace still splits at the forced marker", content == "Here is the answer.")

# Prose `final` alone is not a boundary: without the `assistant` anchor there is nothing to split on.
_, content = split(instance, "analysisI will give a short final answer soon.")
check("prose 'final' in the CoT does not split", content is None)

# A tool call has to reach content, since vLLM extracts tool calls from content only.
_, content = split(instance, 'analysisthinking commentary to=functions.get_weather json{"city":"NYC"}')
check("commentary tool call routed to content", content is not None and "to=functions." in content)

# ── token-level reasoning end ──────────────────────────────────────────────────────────────────
# vLLM reads is_reasoning_end off the prompt to decide whether a stream starts in reasoning, and
# structured outputs gate on it; True at the prompt sends the whole CoT to content without an error.
parser = parser_cls(_PinnedTokenizer())
ANALYSIS_TURN = analysis(3) + H["<|end|>"]
FINAL_OPENING = H["<|start|>assistant"] + H["<|channel|>final<|message|>"]
check("a single-turn prompt leaves reasoning open", not parser.is_reasoning_end(SINGLE_TURN_PROMPT))
check("a tool-result prompt leaves reasoning open", not parser.is_reasoning_end(TOOL_RESULT_PROMPT))
check(
    "the final channel opening ends reasoning",
    parser.is_reasoning_end(SINGLE_TURN_PROMPT + ANALYSIS_TURN + FINAL_OPENING + PROSE),
)
check(
    "a streamed final opening ends reasoning",
    parser.is_reasoning_end_streaming(SINGLE_TURN_PROMPT + ANALYSIS_TURN + FINAL_OPENING, FINAL_OPENING),
)
check(
    "content ids start after the final opening",
    parser.extract_content_ids(ANALYSIS_TURN + FINAL_OPENING + PROSE * 2) == PROSE * 2,
)

# ── thinking_token_budget ───────────────────────────────────────────────────────────────────────
# The parser's markers are what vLLM tokenizes into the budget's ids, and a wrong one produces no
# error: the server takes the field and the CoT runs to max_tokens, or the cut lands inside a tool
# call. The ids resolve here exactly as the server resolves them, through the forced-end patch.
check(
    "forced-end patch took",
    getattr(reasoning_config_module.ReasoningConfig.initialize_token_ids, forced_end_patch.PATCH_MARKER, False),
)
check(
    "completion patch took", getattr(thinking_budget.ThinkingBudgetState.apply, completion_patch.PATCH_MARKER, False)
)
reasoning_config_module.cached_tokenizer_from_config = lambda model_config: _PinnedTokenizer()
_RESOLVED = reasoning_config_module.ReasoningConfig(reasoning_parser="openai_gptoss")
_RESOLVED.initialize_token_ids(None)
check("count starts at an analysis opening", _RESOLVED.reasoning_start_token_ids == H["analysis<|message|>"])
check("count stops at any channel header", _RESOLVED.natural_reasoning_end_token_ids == H["<|channel|>"])
check("the cut forces the move into the final channel", _RESOLVED.reasoning_end_token_ids == TO_FINAL)

# The cut has to be the whole final-channel move, once, right after BUDGET analysis tokens, with and
# without speculative drafts (a forced token that rejects a draft must still be finished).
CUT_AT = len(H["<|channel|>analysis<|message|>"]) + BUDGET
for drafts in (0, 3):
    tag = f"{drafts} drafts"
    for prompt_label, prompt in (("single-turn", SINGLE_TURN_PROMPT), ("tool-result", TOOL_RESULT_PROMPT)):
        forced, cut_at = simulate(prompt, analysis(4 * BUDGET) + TO_FINAL + ANSWER, BUDGET, drafts)
        check(
            f"[{tag}] an overrunning analysis after a {prompt_label} prompt is cut into final (at {cut_at})",
            forced == TO_FINAL and cut_at == CUT_AT,
        )
    # The model would reopen analysis after the cut; the forced sequence decides the channel.
    forced, _ = simulate(SINGLE_TURN_PROMPT, analysis(4 * BUDGET) + TO_FINAL + ANSWER, BUDGET, drafts, "analysis")
    check(f"[{tag}] a cut model cannot reopen analysis", forced == TO_FINAL)
    forced, cut_at = simulate(SINGLE_TURN_PROMPT, analysis(4 * BUDGET) + TO_FINAL + ANSWER, 0, drafts)
    check(f"[{tag}] budget 0 closes the analysis at its opening", forced == TO_FINAL and cut_at == CUT_AT - BUDGET)

    # Nothing outside an analysis channel counts, and nothing there is cut.
    long_tail = PROSE * (4 * BUDGET)
    untouched = {
        "an analysis that ends inside the budget": analysis(BUDGET // 2) + TO_FINAL + long_tail + ANSWER,
        "a turn that goes straight to final": H["<|channel|>final<|message|>"] + long_tail + ANSWER,
        "a long tool call after a short analysis": analysis(BUDGET // 2)
        + H["<|end|>"]
        + H["<|start|>assistant"]
        + CALL_CHANNEL_FIRST
        + long_tail
        + H["<|call|>"],
        "a turn that goes straight to a long tool call": CALL_RECIPIENT_FIRST + long_tail + H["<|call|>"],
        "a long commentary preamble": H["<|channel|>commentary<|message|>"] + long_tail + TO_FINAL + ANSWER,
    }
    for label, plan in untouched.items():
        forced, _ = simulate(SINGLE_TURN_PROMPT, plan, BUDGET, drafts)
        check(f"[{tag}] {label} is not cut", forced == [])

    # Negative control: without the field nothing is forced, so the rows above measure the budget.
    forced, _ = simulate(SINGLE_TURN_PROMPT, analysis(4 * BUDGET) + TO_FINAL + ANSWER, None, drafts)
    check(f"[{tag}] no thinking_token_budget forces nothing", forced == [])

    # A recycled slot forgets the cut it held: the next request writes the same header at the same
    # position of its own accord, under a budget it has not used up, and must keep its tool call.
    slots = _Slots(1, len(SINGLE_TURN_PROMPT) + 8 * BUDGET + drafts + 8)
    state = thinking_budget.ThinkingBudgetState(slots, _RESOLVED)
    simulate(SINGLE_TURN_PROMPT, analysis(4 * BUDGET) + TO_FINAL + ANSWER, BUDGET, drafts, state=state, slots=slots)
    natural_header = analysis(BUDGET) + H["<|end|>"] + H["<|start|>assistant"]
    natural_header += CALL_CHANNEL_FIRST + PROSE * 8
    forced, _ = simulate(
        SINGLE_TURN_PROMPT, natural_header + H["<|call|>"], 4 * BUDGET, drafts, state=state, slots=slots
    )
    check(f"[{tag}] a recycled slot does not finish its previous request's cut", forced == [])

if failures:
    print(f"\ngpt-oss plugin verification FAILED: {failures}")
    sys.exit(1)
print("gpt-oss plugin verification: OK")
