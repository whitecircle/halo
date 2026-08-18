"""Single-GPU dense reference and expert-identity negative control for EP/ETP/TP tests.

A parallel MoE forward that dispatches every token to the wrong expert still produces a finite loss
and finite gradients, so finiteness checks cannot detect it. The same math on the same weights with
no distribution can. This module builds that reference (loss plus the first router gradient from a
plain ``from_pretrained`` forward/backward) and the control showing the comparison is sensitive to
expert identity: :func:`roll_expert_banks` rotates each rank's local expert bank by one expert, the
corruption a dispatch or ownership bug produces, and that must move the loss well outside the
tolerance the match is asserted at.

Reference and parallel model must be built with the same ``attn_implementation``. For a sinks model
(GptOss) that matters twice over: a sink-dropping backend shifts the loss by ~3 nats on live sinks,
which would swamp the signal this module measures.
"""

from __future__ import annotations

from datetime import datetime

import torch
import torch.distributed as dist
from transformers import AutoModelForCausalLM

from tests.common.utils import cleanup_memory, cos_sim

# Router parameter suffixes across the MoE roster: GptOss/DeepSeek-V4 name it ``router``,
# Qwen3/GLM4/Mistral4 name it ``gate``. The EP wrapper adopts the HF layer's own module
# (``EPMoELayerBase._find_gate_or_router``), so the parameter path is identical in the sharded and
# the dense model, which makes the two directly comparable.
_ROUTER_SUFFIXES = ("router.weight", "gate.weight")

# The clock every fixed batch is rendered against. gpt-oss's harmony template stamps
# ``strftime_now("%Y-%m-%d")`` into its system message, so an unpinned batch, and every loss threshold
# measured against it, drifts with the calendar date. The value matters: it moves the rotated-expert
# control's loss shift, which at 2025-01-01 falls under ``control_min_loss_shift``.
CHAT_TEMPLATE_NOW = datetime(2026, 8, 26)

# Deterministic conversations for the fixed batches, as (user, assistant) turns. Index 0 is the
# default short exchange; the rest are longer and mutually distinct, for callers that fill more of
# the sequence window or need several independent inputs.
FIXED_CONVERSATIONS = (
    ("What is 42 + 58?", "The sum of 42 and 58 is 100."),
    (
        "What is 42 + 58? Please explain step by step.",
        "To calculate 42 + 58, I add the ones place first: 2 + 8 = 10, "
        "carry the 1. Then tens place: 4 + 5 + 1 = 10. So the answer is 100.",
    ),
    (
        "Explain the concept of prime numbers with examples.",
        "A prime number is only divisible by 1 and itself. Examples: 2, 3, 5, 7, 11. "
        "The number 4 is not prime because 4 = 2 x 2.",
    ),
    (
        "How do neural networks learn from data?",
        "Neural networks learn by adjusting weights through backpropagation. "
        "The loss function measures error, and gradients flow backwards to update parameters.",
    ),
    (
        "What is the difference between TCP and UDP protocols?",
        "TCP is connection-oriented and reliable with ordered delivery. "
        "UDP is connectionless and faster but without delivery guarantees.",
    ),
    (
        "Describe the water cycle in detail.",
        "Water evaporates from surfaces, rises as vapor, condenses into clouds, "
        "and falls as precipitation. It then flows through rivers back to oceans.",
    ),
)


def ep_layers(model) -> list:
    """Every EP/ETP-wrapped MoE layer in ``model`` (the wrapper is the thing that owns ``ep_config``)."""
    return [m for m in model.modules() if hasattr(m, "ep_config")]


def find_router_weight(model) -> tuple[str, torch.nn.Parameter]:
    """First MoE router weight in module order, by parameter name."""
    for name, param in model.named_parameters():
        if name.endswith(_ROUTER_SUFFIXES):
            return name, param
    raise AssertionError(f"no router/gate weight found in {type(model).__name__} — nothing to compare")


def full_grad(param: torch.nn.Parameter) -> torch.Tensor:
    """A parameter's gradient as a full fp32 tensor, whether it is plain or a TP DTensor."""
    grad = param.grad
    if grad is None:
        raise AssertionError("parameter has no gradient — backward did not reach it")
    if hasattr(grad, "full_tensor"):
        grad = grad.full_tensor()
    return grad.detach().float().clone()


def compare_grad(got: torch.Tensor, reference: torch.Tensor) -> tuple[float, float]:
    """Compare a gradient to its single-GPU reference as (norm ratio, cosine similarity).

    Split rather than collapsed into one relative error because the two bug classes are independent:
    a mis-scaled cross-rank reduction moves the ratio and leaves the cosine at 1, while a
    routing/permutation corruption moves the cosine and can leave the ratio at 1. A single bound would
    have to be loosened past the weaker signal to absorb the other's noise.
    """
    ref_norm = reference.norm().item()
    got_norm = got.norm().item()
    ratio = got_norm / max(ref_norm, 1e-12)
    return ratio, cos_sim(got, reference)


def fixed_chat_batch(
    tokenizer,
    seq_len: int,
    device: str,
    seed: int = 42,
    *,
    conversation: int = 0,
    broadcast: bool = False,
):
    """Deterministic right-padded chat batch. Padding is labelled ``-100`` so it never scores.

    ``conversation`` indexes :data:`FIXED_CONVERSATIONS` (modulo its length), ``broadcast`` shares
    rank 0's tensors with every rank, and the seed pins the tokenizer-independent parts so a rerun on
    one rank reproduces the same batch. The rendered date is pinned to :data:`CHAT_TEMPLATE_NOW`:
    ``strftime_now`` is a jinja global transformers injects into every chat template, and a same-named
    render kwarg shadows it, which is a no-op for templates that never call it.
    """
    torch.manual_seed(seed)
    user, assistant = FIXED_CONVERSATIONS[conversation % len(FIXED_CONVERSATIONS)]
    text = tokenizer.apply_chat_template(
        [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
        tokenize=False,
        add_generation_prompt=False,
        strftime_now=CHAT_TEMPLATE_NOW.strftime,
    )
    tokens = tokenizer(text, return_tensors="pt", padding="max_length", max_length=seq_len, truncation=True)
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)
    labels = input_ids.clone()
    labels[attention_mask == 0] = -100
    if broadcast:
        for tensor in (input_ids, attention_mask, labels):
            dist.broadcast(tensor, src=0)
    return input_ids, attention_mask, labels


def dense_reference(
    model_name: str,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    device: str,
    *,
    attn_implementation: str,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[float, torch.Tensor]:
    """Plain undistributed forward+backward on the same checkpoint. Returns (loss, router grad).

    The gradient half is what catches a mis-scaled cross-rank reduction: that bug leaves the
    forward exact and only rescales the router gradient.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=dtype, trust_remote_code=True, attn_implementation=attn_implementation
    ).to(device)
    model.train()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False)
    outputs.loss.backward()
    loss = outputs.loss.item()
    grad = full_grad(find_router_weight(model)[1])
    del model, outputs
    cleanup_memory()
    return loss, grad


def broadcast_reference(
    loss: float, grad: torch.Tensor | None, device: str, rank: int, *, with_grad: bool = True
) -> tuple[float, torch.Tensor | None]:
    """Share a rank-0-only reference with every rank (they all compare their own result against it).

    ``with_grad=False`` shares the loss alone, for a caller whose gradient checks are not anchored
    here; every rank must pass the same value, since it decides whether the grad collective runs.
    """
    loss_t = torch.tensor([loss], device=device, dtype=torch.float32)
    dist.broadcast(loss_t, src=0)
    if not with_grad:
        return float(loss_t.item()), None

    shape = torch.zeros(2, dtype=torch.long, device=device)
    if rank == 0:
        shape[0], shape[1] = grad.shape
    dist.broadcast(shape, src=0)
    if grad is None:
        grad = torch.zeros(int(shape[0]), int(shape[1]), dtype=torch.float32, device=device)
    dist.broadcast(grad, src=0)
    return float(loss_t.item()), grad


def roll_expert_banks(model, shift: int = 1) -> int:
    """Negative control: rotate every EP layer's local expert bank by ``shift`` experts.

    Router, dispatch and combine are untouched; only which FFN a dispatched token lands in changes,
    which is what a wrong-expert routing or ownership bug does. Every rank applies the same rotation
    to its own bank, so the perturbed model stays rank-consistent and its reduced loss is well
    defined. Returns the number of tensors rotated; 0 means the control never engaged, which must
    fail the test.
    """
    rotated = 0
    with torch.no_grad():
        for layer in ep_layers(model):
            for _, param in layer.expert_named_params():
                # Leading dim is the expert axis for every family's expert weights and biases.
                if param.shape[0] < 2:
                    continue
                param.data.copy_(torch.roll(param.data, shifts=shift, dims=0))
                rotated += 1
    return rotated
