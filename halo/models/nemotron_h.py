"""Toolkit patches for upstream Nemotron-H and its Radio vision tower: the Mamba scan rewritten per group and per
document, and the two values no serving engine reads the way upstream's trainer does. Idempotent; a moved upstream
signature fails loud. The composite's head accessors, which change no forward, are set at import."""

import functools
import logging
from types import ModuleType
from typing import NamedTuple

import torch
import torch.nn.functional as F
import transformers.models.nemotron_h.modeling_nemotron_h as nemotron_modeling
import transformers.models.nemotron_h_omni.modeling_nemotron_h_omni as omni_modeling
import transformers.models.radio.modeling_radio as radio_modeling
from torch import nn

from halo.models.attention.tensor_memo import TensorMemo
from halo.models.conversion_overrides import NEMOTRON_H_COMPOSITE_CLASS
from halo.models.row_documents import row_documents
from halo.models.upstream_guard import verify_upstream_parameters
from halo.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)

# Documents share a scan while the padded batch they make costs at most this over their tokens. A short document's cost
# is mostly the scan's own, so a row of them gains from sharing, and a skewed row pays the longest for every member.
_MAX_PADDED_AREA = 1.25


class _RowLayout(NamedTuple):
    """Every document in row-major order: each row's lengths, and the documents sharing a padded batch, longest
    first."""

    lengths_per_row: tuple[tuple[int, ...], ...]
    batches: tuple[tuple[int, ...], ...]


def _padded_batches(lengths: list[int]) -> tuple[tuple[int, ...], ...]:
    batches: list[list[int]] = []
    tokens = 0
    for index in sorted(range(len(lengths)), key=lengths.__getitem__, reverse=True):
        if batches and lengths[batches[-1][0]] * (len(batches[-1]) + 1) <= _MAX_PADDED_AREA * (
            tokens + lengths[index]
        ):
            batches[-1].append(index)
            tokens += lengths[index]
        else:
            batches.append([index])
            tokens = lengths[index]
    return tuple(tuple(batch) for batch in batches)


def _resolve_row_layout(seq_idx: torch.Tensor) -> _RowLayout:
    batch, length = seq_idx.shape
    per_row: list[list[int]] = [[] for _ in range(batch)]
    for document in row_documents(seq_idx, None, batch, length, seq_idx.device):
        per_row[document.row].append(document.stop - document.start)
    return _RowLayout(
        tuple(tuple(lengths) for lengths in per_row),
        _padded_batches([length for lengths in per_row for length in lengths]),
    )


# Every Mamba layer of a forward and its recompute read one host copy of the row's boundaries.
_row_layout = TensorMemo(_resolve_row_layout)


def _grouped_chunk_scan(hidden_states, dt, A, B, C, chunk_size, D, dt_bias, initial_states, dt_softplus, dt_limit):
    """Upstream's chunk scan, with the state contractions per group, as a group's heads share B and C and upstream
    copies them per head. Letters as upstream's: batch, chunk, l and s positions in a chunk, head, p head_dim, n state;
    g a group and q a head within it. Returns the output and the state past the last chunk."""
    batch, length, heads, head_dim = hidden_states.shape
    groups, state_size = B.shape[2:]
    heads_per_group = heads // groups

    dt = dt + dt_bias.to(dt.dtype)
    if dt_softplus:
        dt = F.softplus(dt)
    dt = torch.clamp(dt, min=dt_limit[0], max=dt_limit[1]).float()

    hidden_states = hidden_states.float()
    pad_size = (chunk_size - length % chunk_size) % chunk_size
    residual = D[..., None] * nemotron_modeling.pad_tensor_by_size(hidden_states, pad_size)

    x, A, B, C = (
        nemotron_modeling.reshape_into_chunks(tensor, pad_size, chunk_size)
        for tensor in (hidden_states * dt[..., None], A.to(hidden_states.dtype) * dt, B.float(), C.float())
    )
    chunks = x.shape[1]
    A = A.permute(0, 3, 1, 2)
    A_cumsum = torch.cumsum(A, dim=-1)

    L = torch.exp(nemotron_modeling.segment_sum(A))
    G = torch.einsum("bclgn,bcsgn->bclsg", C, B)
    M = G.repeat_interleave(heads_per_group, dim=-1) * L.permute(0, 2, 3, 4, 1)
    del G, L
    y_diag = torch.einsum("bclsh,bcshp->bclhp", M, x)
    del M

    decay = torch.exp(A_cumsum[:, :, :, -1:] - A_cumsum).permute(0, 2, 3, 1)[..., None]
    x_decay = (x * decay).reshape(batch, chunks, chunk_size, groups, heads_per_group, head_dim)
    states = torch.einsum("bclgn,bclgqp->bcgqpn", B, x_decay).reshape(batch, chunks, heads, head_dim, state_size)
    del x_decay

    previous = (
        torch.zeros_like(states[:, :1])
        if initial_states is None
        else initial_states[:, None].to(dtype=states.dtype, device=states.device)
    )
    decay_chunk = nemotron_modeling.segment_sum(F.pad(A_cumsum[:, :, :, -1], (1, 0)))
    joined = torch.cat([previous, states], dim=1)
    new_states = torch.einsum("bzch,bzhpn->bchpn", torch.exp(decay_chunk).transpose(1, 3), joined)
    states, final_state = new_states[:, :-1], new_states[:, -1]

    grouped_states = states.reshape(batch, chunks, groups, heads_per_group, head_dim, state_size)
    y_off = (
        torch.einsum("bclgn,bcgqpn->bclgqp", C, grouped_states).reshape(batch, chunks, chunk_size, heads, head_dim)
        * torch.exp(A_cumsum).permute(0, 2, 3, 1)[..., None]
    )

    output = (y_diag + y_off).reshape(batch, -1, heads, head_dim) + residual
    return (output[:, :length] if pad_size > 0 else output), final_state


def _split_documents(tensor: torch.Tensor, lengths_per_row: tuple[tuple[int, ...], ...]) -> list[torch.Tensor]:
    """One split per row, as one slice per document would zero-fill the whole row per document in the backward."""
    return [
        piece for row, lengths in zip(tensor.unbind(0), lengths_per_row, strict=True) for piece in row.split(lengths)
    ]


def _padded_stack(pieces: list[torch.Tensor]) -> torch.Tensor:
    longest = pieces[0].shape[0]
    return torch.stack(
        [
            piece
            if piece.shape[0] == longest
            else F.pad(piece, (0, 0) * (piece.dim() - 1) + (0, longest - piece.shape[0]))
            for piece in pieces
        ]
    )


def _scan_documents(layout: _RowLayout, hidden_states, dt, A, B, C, **scan_arguments) -> torch.Tensor:
    """Each document alone from a zero state, chunks counted from its first token, whichever row and batch it takes."""
    pieces = [_split_documents(tensor, layout.lengths_per_row) for tensor in (hidden_states, dt, B, C)]
    lengths = [length for row in layout.lengths_per_row for length in row]
    outputs: list[torch.Tensor | None] = [None] * len(lengths)
    for batch in layout.batches:
        hidden_batch, dt_batch, b_batch, c_batch = (_padded_stack([part[i] for i in batch]) for part in pieces)
        scanned, _ = _grouped_chunk_scan(hidden_batch, dt_batch, A, b_batch, c_batch, **scan_arguments)
        for index, row in zip(batch, scanned.unbind(0), strict=True):
            outputs[index] = row[: lengths[index]]
    return torch.cat(outputs).reshape(hidden_states.shape)


@functools.wraps(nemotron_modeling.mamba2_chunk_scan)
def _document_chunk_scan(
    hidden_states,
    dt,
    A,
    B,
    C,
    chunk_size,
    D,
    dt_bias,
    initial_states,
    dt_softplus,
    dt_limit,
    return_final_states,
    seq_idx=None,
    **_kwargs,
):
    """Autocast off, as it would run the state contractions of this fp32 body in bf16."""
    scan_arguments = {
        "chunk_size": chunk_size,
        "D": D,
        "dt_bias": dt_bias,
        "dt_softplus": dt_softplus,
        "dt_limit": dt_limit,
    }
    with torch.autocast(hidden_states.device.type, enabled=False):
        layout = None if seq_idx is None else _row_layout(seq_idx)
        if layout is not None and sum(map(len, layout.lengths_per_row)) > hidden_states.shape[0]:
            if initial_states is not None or return_final_states:
                raise ValueError(
                    "A row holding several documents has no one state to carry through a cache; seq_idx marks "
                    "training rows, which take none."
                )
            return _scan_documents(layout, hidden_states, dt, A, B, C, initial_states=None, **scan_arguments)
        output, final_state = _grouped_chunk_scan(
            hidden_states, dt, A, B, C, initial_states=initial_states, **scan_arguments
        )
    return (output, final_state) if return_final_states else output


def _no_fused_scan(*_args, **_kwargs) -> None:
    """The mixer's own path, so the fused kernel a later ``mamba-ssm`` install would select cannot read ``in_proj`` and
    ``out_proj`` past their adapters, or refuse ``seq_idx``."""
    return None


def _patch_mamba_mixer() -> None:
    mixer_cls = nemotron_modeling.NemotronHMamba2Mixer
    original_init, original_forward = mixer_cls.__init__, mixer_cls.forward

    @functools.wraps(original_init)
    def __init__(self, config, *args, **kwargs):
        original_init(self, config, *args, **kwargs)
        # Upstream clamps at ``time_step_min``; both engines apply (0, inf) whether a release declares it or not.
        self.time_step_limit = tuple(config.time_step_limit)

    @functools.wraps(original_forward)
    def forward(self, *args, **kwargs):
        if not isinstance(self.conv1d, nn.Conv1d):
            raise TypeError(
                f"Nemotron-H's Mamba mixer reads conv1d.weight directly, so the {type(self.conv1d).__name__} on it "
                f"would be bypassed. Leave conv1d out of lora_target_modules; in_proj and out_proj take adapters."
            )
        return original_forward(self, *args, **kwargs)

    mixer_cls.__init__ = __init__
    mixer_cls.forward = forward


def _patch_radio_layer_scale() -> None:
    """The release stores no ``lambda1`` and neither engine reads one, so a trained value would serve as 1.0. A buffer,
    since an eager load of a checkpoint carrying it would turn a frozen parameter trainable again."""
    scale_cls = radio_modeling.RadioLayerScale
    original_init = scale_cls.__init__

    @functools.wraps(original_init)
    def __init__(self, config):
        original_init(self, config)
        self.register_buffer("lambda1", self._parameters.pop("lambda1").data)

    scale_cls.__init__ = __init__


def _composite_output_embeddings(self) -> nn.Module:
    return self.language_model.get_output_embeddings()


def _set_composite_output_embeddings(self, new_embeddings: nn.Module) -> None:
    self.language_model.set_output_embeddings(new_embeddings)


def set_composite_head_accessors(module: ModuleType) -> None:
    """On the composite ``module`` defines, transformers' or the release's own file an export carries. Upstream's
    accessors look for ``lm_head`` on the composite, which holds it under ``language_model``: a reader would find no
    head, and a resize would grow the embedding alone."""
    composite = getattr(module, NEMOTRON_H_COMPOSITE_CLASS, None)
    if isinstance(composite, type):
        composite.get_output_embeddings = _composite_output_embeddings
        composite.set_output_embeddings = _set_composite_output_embeddings


# At import, as a tool resizing a checkpoint takes no family patch.
set_composite_head_accessors(omni_modeling)


def apply_nemotron_h_patches() -> None:
    if nemotron_modeling.mamba2_chunk_scan is _document_chunk_scan:
        return
    patch = "apply_nemotron_h_patches"
    verify_upstream_parameters(
        nemotron_modeling.mamba2_chunk_scan,
        (
            "hidden_states",
            "dt",
            "A",
            "B",
            "C",
            "chunk_size",
            "D",
            "dt_bias",
            "initial_states",
            "dt_softplus",
            "dt_limit",
            "return_final_states",
            "kwargs",
        ),
        patch,
    )
    mixer_cls = nemotron_modeling.NemotronHMamba2Mixer
    verify_upstream_parameters(mixer_cls.__init__, ("self", "config", "layer_idx", "initialize_mixer_weights"), patch)
    verify_upstream_parameters(
        mixer_cls.forward, ("self", "hidden_states", "cache_params", "attention_mask", "kwargs"), patch
    )
    verify_upstream_parameters(radio_modeling.RadioLayerScale.__init__, ("self", "config"), patch)
    nemotron_modeling.mamba2_chunk_scan = _document_chunk_scan
    nemotron_modeling.mamba2_split_conv1d_scan_combined = _no_fused_scan
    _patch_mamba_mixer()
    _patch_radio_layer_scale()
    if is_global_main_process():
        logger.info("Patched Nemotron-H's Mamba scan per group and document, its time-step limit, and Radio's lambda1")
