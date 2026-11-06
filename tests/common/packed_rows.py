"""Forward and gradient helpers of the packed-row gate (tests/gpu/kernels/test_packed_rl_forward_mrope.py): several
documents scored as one flattened row against each alone. A tiny CPU model runs them too."""

import torch
import torch.nn.functional as F

from halo.model_loading.model_preparation import apply_family_patches
from halo.models.segment_markers import packed_row_segment_kwargs, packed_row_varlen_kwargs, segment_markers_for
from tests.common.numerics import max_abs_rel_err

SEED = 0
# Of the reference's peak, where the per-row error is below what a ratio can resolve.
ERROR_FLOOR = 2e-3


def emission_kwargs(emission: str, position_ids: torch.Tensor, config) -> dict:
    """What the micro-batch hands the model beside ``input_ids``: ``position_ids`` alone; those with the family's
    segment markers (what SFT ``packing`` emits); the flash varlen set alone (what ``padding_free`` emits); the markers
    with the M-RoPE positions spelled out as the ``[3, 1, N]`` plane the decoder would build itself."""
    if emission == "position_ids":
        return {"position_ids": position_ids}
    if emission == "markers":
        return {"position_ids": position_ids, **packed_row_segment_kwargs(position_ids, segment_markers_for(config))}
    if emission == "flash_varlen":
        return {"position_ids": position_ids, **packed_row_varlen_kwargs(position_ids)}
    if emission == "mrope_plane":
        return {
            "position_ids": position_ids.expand(3, -1, -1).contiguous(),
            **packed_row_segment_kwargs(position_ids, segment_markers_for(config)),
        }
    raise ValueError(f"unknown emission {emission!r}")


def set_attention(model, attn_implementation: str) -> None:
    resolved = apply_family_patches(model.config, attn_implementation)
    for config in (model.config, getattr(model.config, "text_config", None)):
        if config is not None:
            config._attn_implementation = resolved


def build_documents(lengths: tuple[int, ...], vocab_size: int, device) -> list[torch.Tensor]:
    generator = torch.Generator().manual_seed(SEED)
    return [torch.randint(1, vocab_size, (1, length), generator=generator).to(device) for length in lengths]


def next_token_targets(documents: list[torch.Tensor]) -> torch.Tensor:
    """Each document's next token, -100 at its last position, which has none."""
    return torch.cat([F.pad(doc[0, 1:], (0, 1), value=-100) for doc in documents])


def row_logits(model, documents: list[torch.Tensor], emission: str | None) -> torch.Tensor:
    """Logits over all documents' positions, in order: one forward per document, or one flattened forward under
    ``emission``."""
    if emission is None:
        return torch.cat([model(input_ids=doc, use_cache=False).logits[0] for doc in documents])
    ids = torch.cat(documents, dim=1)
    positions = torch.cat([torch.arange(doc.size(1)) for doc in documents]).unsqueeze(0).to(ids.device)
    return model(input_ids=ids, use_cache=False, **emission_kwargs(emission, positions, model.config)).logits[0]


def row_step(model, documents: list[torch.Tensor], emission: str | None):
    """The logits of ``row_logits`` and the parameter gradients of their summed cross entropy."""
    model.zero_grad(set_to_none=True)
    logits = row_logits(model, documents, emission)
    F.cross_entropy(logits.float(), next_token_targets(documents), ignore_index=-100, reduction="sum").backward()
    grads = {name: p.grad.detach().float() for name, p in model.named_parameters() if p.grad is not None}
    return logits.detach().float(), grads


def grad_error(grads: dict[str, torch.Tensor], reference: dict[str, torch.Tensor]) -> float:
    return max(max_abs_rel_err(grads[name], ref) for name, ref in reference.items() if name in grads)


def check_isolation(model, attn: str, emission: str, lengths, checks) -> None:
    """Changing one document leaves every other document's logits where they were, up to kernel noise."""
    set_attention(model, attn)
    device = next(model.parameters()).device
    documents = build_documents(lengths, model.config.get_text_config().vocab_size, device)
    changed = [documents[0], (documents[1] + 1) % model.config.get_text_config().vocab_size, *documents[2:]]
    with torch.no_grad():
        before = row_logits(model, documents, emission).float()
        after = row_logits(model, changed, emission).float()
    first = lengths[0]
    untouched = max_abs_rel_err(after[:first], before[:first])
    third_start = lengths[0] + lengths[1]
    checks[f"{attn}_a_neighbour_does_not_move_the_documents_around_it"] = (
        untouched <= ERROR_FLOOR and max_abs_rel_err(after[third_start:], before[third_start:]) <= ERROR_FLOOR
    )
