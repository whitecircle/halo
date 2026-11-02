"""``sdpa_flex_sliding`` test helpers: the rows a warm-up check trains on, how many attention layers each runs on
FlexAttention, and what a CPU run of the implementation needs: the registered function serving a CPU call as it
serves a CUDA one, and FlexAttention's device check lifted.

The implementation serves every masked causal call: on a packed row each layer gets the document mask, so full
layers run FlexAttention beside the sliding ones, while on a single document transformers hands a full layer SDPA's
``is_causal`` with no mask and only the sliding layers, whose window binds past its length, run it.
"""

import torch

from halo.models.attention import sdpa_flex_sliding
from halo.models.attention.layout import attention_call
from halo.models.attention.sdpa_flex_sliding import flex_attention_calls
from tests.common.fake_cuda import CudaLike

# (documents, train) per step: a packed row, then one document trained and forward-only, each past the window.
FLEX_ROWS = (([300, 150, 40], True), ([600], True), ([400], False))


def no_device_check(*tensors) -> None:
    """FlexAttention's device check (``_validate_device``), lifted: it refuses a CPU backward, which the kernel's
    eager math runs. One object for every patch, since Dynamo guards on it."""


def flex_layers(model: torch.nn.Module, lengths: list[int]) -> int:
    """The attention modules of ``model`` a row of ``lengths`` documents runs on FlexAttention: each one it serves
    on a packed row, only the windowed ones on a single document."""
    served = flex_attention_calls(model)
    calls = [call for call in map(attention_call, model.modules()) if call in served]
    assert all(sum(lengths) > (call.window or 0) for call in calls), "premise: the row is past every window"
    return sum(1 for call in calls if len(lengths) > 1 or call.window)


def train_row(model: torch.nn.Module, lengths: list[int], *, train: bool = True) -> None:
    """A forward over one row of ``lengths`` packed documents on ``model``'s device, and its backward when
    ``train``."""
    device = next(model.parameters()).device
    generator = torch.Generator().manual_seed(sum(lengths))
    vocab = model.config.get_text_config().vocab_size
    ids = torch.randint(3, vocab, (1, sum(lengths)), generator=generator).to(device)
    position_ids = torch.cat([torch.arange(n) for n in lengths]).unsqueeze(0).to(device)
    with torch.set_grad_enabled(train):
        out = model(input_ids=ids, position_ids=position_ids, use_cache=False)
    if train:
        hidden = out.logits if hasattr(out, "logits") else out.last_hidden_state
        hidden.float().pow(2).mean().backward()


def flex_sliding_on_cpu(module, query, key, value, attention_mask, **kwargs):
    """The registered function as it serves a CUDA call, the only call it takes; the kernel it reaches receives
    :class:`~tests.common.fake_cuda.CudaLike` inputs."""
    out, weights = sdpa_flex_sliding.flex_sliding_attention(
        module, *(t.as_subclass(CudaLike) for t in (query, key, value)), attention_mask, **kwargs
    )
    return out.as_subclass(torch.Tensor), weights
