"""A family's released MTP head built from its own decoder layers at the release's key path, and the hooks that run it
inside the model's forward so its term shares the next-token loss's normalizer."""

from __future__ import annotations

import copy
from collections import defaultdict
from collections.abc import Callable
from functools import partial
from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import init_empty_weights

from halo.kernels.logprobs import LABEL_IGNORE_INDEX, logit_chunk_rows
from halo.models.config_access import text_config
from halo.models.head_transform import resolve_head_transform
from halo.models.module_tree import (
    backbone_with_layers,
    decoder_layers,
    decoder_layers_attr,
    find_module_path,
    module_path,
)
from halo.models.moe.config_fields import require_router_aux_loss_coef
from halo.models.mtp.acceptance import score_drafts
from halo.models.mtp.spec import (
    ROOT_HEAD_ATTR,
    ROOT_HEAD_LAYERS_ATTR,
    MtpHeadSpec,
    MtpObjective,
    mtp_draft_layer_index,
)


class MtpLossSplit(NamedTuple):
    """A forward's two cross-entropies as this rank's sums over its own targets, set on the model's output: its
    ``loss`` holds them weighted over one normalizer, and under ``detached`` the next-token one is what the trunk fits."""

    next_token_sum: torch.Tensor
    next_token_targets: torch.Tensor
    head_sum: torch.Tensor
    head_targets: torch.Tensor


class MtpHeadChain:
    """``forward(e_next, h_in, *layer_args, **layer_kwargs)``: norms, projection, the family's decoder layers and the
    output norm as one call, so a wrapper that gathers on a forward holds the whole head for it."""

    _mtp_spec: MtpHeadSpec
    # Set by a ``head_only`` run's freeze.
    trunk_frozen: bool = False

    def forward(self, e_next: torch.Tensor, h_in: torch.Tensor, *layer_args, **layer_kwargs) -> torch.Tensor:
        spec = self._mtp_spec
        projected = self.get_submodule(spec.projection)(
            torch.cat(
                [self.get_submodule(spec.embedding_norm)(e_next), self.get_submodule(spec.hidden_norm)(h_in)], -1
            )
        )
        output = self._run_draft_layer(projected, *layer_args, **layer_kwargs)
        return self.get_submodule(spec.output_norm)(output[0] if isinstance(output, tuple) else output)

    def _run_draft_layer(self, hidden: torch.Tensor, *layer_args, **layer_kwargs):
        return super().forward(hidden, *layer_args, **layer_kwargs)


class RootMtpHead(MtpHeadChain, nn.Module):
    """``mtp.*``: the draft blocks under ``mtp.layers``, the projection set where the release stores it. Each block takes
    the attention layer's call, which a block of another kind ignores as the trunk's does."""

    def __init__(self, spec: MtpHeadSpec, blocks: list[nn.Module]):
        super().__init__()
        self._mtp_spec = spec
        self.add_module(ROOT_HEAD_LAYERS_ATTR, nn.ModuleList(blocks))

    def _run_draft_layer(self, hidden: torch.Tensor, *layer_args, **layer_kwargs):
        for block in getattr(self, ROOT_HEAD_LAYERS_ATTR):
            output = block(hidden, *layer_args, **layer_kwargs)
            hidden = output[0] if isinstance(output, tuple) else output
        return hidden


def _list_head_class(layer_class: type[nn.Module]) -> type[nn.Module]:
    """The decoder layer holding the projection set among its own children, as a ``list`` release stores them. A class
    of its own, so the layer's forward stays what it was for every trunk layer."""
    return type(f"Mtp{layer_class.__name__}", (MtpHeadChain, layer_class), {})


def set_nested_submodule(root: nn.Module, path: str, module: nn.Module) -> None:
    *parents, name = path.split(".")
    for parent in parents:
        if parent not in root._modules:
            root.add_module(parent, nn.Module())
        root = root._modules[parent]
    root.add_module(name, module)


def build_mtp_head(model: nn.Module, spec: MtpHeadSpec, *, device: str | torch.device = "meta") -> nn.Module:
    """The head of ``model``'s family on ``device``, detached from ``model``. Off meta its tensors are uninitialized, for
    the caller's init. A copy of the trunk's final norm keeps its class and epsilon."""
    text = text_config(model.config)
    backbone = backbone_with_layers(model)
    layers = decoder_layers(backbone)
    hidden_size = text.hidden_size

    def build_block(cls: type[nn.Module], index: int) -> nn.Module:
        if str(device) == "meta":
            # Parameters on meta, buffers real, as the lazy shell holds them.
            with init_empty_weights(include_buffers=False):
                return cls(text, index)
        with torch.device(device):
            return cls(text, index)

    if spec.placement == "root":
        indices = [mtp_draft_layer_index(text, kind) for kind in spec.chain_kinds]
        head = RootMtpHead(spec, [build_block(type(layers[index]), index) for index in indices])
    else:
        index = mtp_draft_layer_index(text)
        head = build_block(_list_head_class(type(layers[index])), index)
        head._mtp_spec = spec
    with torch.device(device):
        set_nested_submodule(head, spec.projection, nn.Linear(2 * hidden_size, hidden_size, bias=False))
    trunk_norm = getattr(backbone, spec.trunk_norm)
    for path in (spec.embedding_norm, spec.hidden_norm, spec.output_norm):
        set_nested_submodule(head, path, copy.deepcopy(trunk_norm).to(device))
    return head


def mtp_head_path(model: nn.Module, spec: MtpHeadSpec) -> str:
    """Where ``spec``'s release stores the head, relative to ``model``'s state dict: a ``root`` head beside the output
    embedding it shares, whichever level of a composite holds that."""
    if spec.placement == "root":
        owner = module_path(model, model.get_output_embeddings()).rpartition(".")[0]
        return f"{owner}.{ROOT_HEAD_ATTR}" if owner else ROOT_HEAD_ATTR
    backbone = backbone_with_layers(model)
    parent = find_module_path(model, backbone)
    layers = f"{parent}.{decoder_layers_attr(backbone)}" if parent else decoder_layers_attr(backbone)
    return f"{layers}.{text_config(model.config).num_hidden_layers}"


def attach_mtp_head_module(model: nn.Module, head: nn.Module, spec: MtpHeadSpec) -> None:
    """A ``list`` head joins the decoder list past ``num_hidden_layers``, which the family's forward slices off."""
    if spec.placement == "root":
        model.get_submodule(mtp_head_path(model, spec).rpartition(".")[0]).add_module(ROOT_HEAD_ATTR, head)
    else:
        decoder_layers(backbone_with_layers(model)).append(head)


def find_mtp_head(model: nn.Module) -> nn.Module | None:
    return next((module for module in model.modules() if isinstance(module, MtpHeadChain)), None)


def freeze_trunk_for_head(model: nn.Module, head: nn.Module) -> None:
    """Every parameter outside ``head`` stops training, and the backbone runs with no graph behind the head's leaves."""
    head_params = {id(param) for param in head.parameters()}
    for param in model.parameters():
        param.requires_grad_(id(param) in head_params)
    head.trunk_frozen = True
    backbone = backbone_with_layers(model)
    backbone.forward = torch.no_grad()(backbone.forward)


def mtp_trunk_copy_keys(model: nn.Module) -> dict[str, list[str]]:
    """The tail keys a release stores as copies of a trunk embedding, by the live key of the embedding each copies. The
    head shares those embeddings in training, and an engine's completeness check reads the copies. Empty with no head
    or on a spec declaring none."""
    head = find_mtp_head(model)
    if head is None or not head._mtp_spec.trunk_copies:
        return {}
    spec = head._mtp_spec
    path = mtp_head_path(model, spec)
    live_names = {id(param): name for name, param in model.named_parameters()}
    embeddings = {"input": model.get_input_embeddings(), "output": model.get_output_embeddings()}
    copies: dict[str, list[str]] = defaultdict(list)
    for key, copied in spec.trunk_copies:
        copies[live_names[id(embeddings[copied].weight)]].append(f"{path}.{key}")
    return dict(copies)


def frozen_trunk_params(model: nn.Module) -> list[nn.Parameter]:
    """What :func:`freeze_trunk_for_head` froze on ``model``; none on any other run."""
    head = find_mtp_head(model)
    if head is None or not head.trunk_frozen:
        return []
    head_params = {id(param) for param in head.parameters()}
    return [param for param in model.parameters() if id(param) not in head_params]


class MtpForwardTerm:
    """Five hooks, installed before any parallel wrap so theirs run first. A training forward with labels takes the
    trunk's input embeddings and its draft-kind layer's call, runs the head once the backbone returns, and adds the term
    to ``outputs.loss``. Inside the root forward HF's later loss scale multiplies both cross-entropies alike, so the
    soft route's scale is exactly ``weight`` on every batch and rank. An eval forward with labels leaves the loss alone
    and sets the drafts' acceptance on the output, every rank walking the head's ops as it does in training."""

    def __init__(
        self,
        model: nn.Module,
        head: nn.Module,
        spec: MtpHeadSpec,
        objective: MtpObjective,
        ce_sum: Callable[..., torch.Tensor],
    ):
        self._head = head
        self._spec = spec
        self._objective = objective
        self._ce_sum = ce_sum
        output_embedding = model.get_output_embeddings()
        self._project = partial(resolve_head_transform(model).project, output_embedding)
        self._chunk_rows = logit_chunk_rows(output_embedding.out_features)
        self._clear()
        backbone = backbone_with_layers(model)
        layers = decoder_layers(backbone)
        model.register_forward_pre_hook(self._capture_call, with_kwargs=True)
        layers[0].register_forward_pre_hook(self._capture_embeddings, with_kwargs=True)
        layers[mtp_draft_layer_index(text_config(model.config))].register_forward_pre_hook(
            self._capture_layer_call, with_kwargs=True
        )
        backbone.register_forward_hook(self._run_head, with_kwargs=True)
        model.register_forward_hook(self._add_term, with_kwargs=True)

    def _clear(self) -> None:
        """After the term, so a checkpoint's recompute during backward finds nothing to capture."""
        self._labels = self._position_ids = self._num_items = None
        self._embeddings = self._layer_args = self._layer_kwargs = None
        self._term = self._term_targets = self._scale = self._acceptance = None

    def _capture_call(self, _module, _args, kwargs) -> None:
        self._clear()
        self._labels = kwargs.get("labels")
        self._position_ids = kwargs.get("position_ids")
        self._num_items = kwargs.get("num_items_in_batch")

    def _capture_embeddings(self, _module, args, _kwargs) -> None:
        if self._labels is not None:
            self._embeddings = args[0]

    def _capture_layer_call(self, _module, args, kwargs) -> None:
        if self._labels is not None:
            self._layer_args, self._layer_kwargs = args[1:], dict(kwargs)

    def _positions(self, like: torch.Tensor) -> torch.Tensor:
        """Per-document positions; a row with none given is one document."""
        if self._position_ids is None:
            batch, length = like.shape[:2]
            return torch.arange(length, device=like.device).expand(batch, -1)
        if self._position_ids.dim() != 2:
            raise ValueError(
                f"The MTP head reads document starts off [batch, length] position_ids, got shape "
                f"{tuple(self._position_ids.shape)}. Train it on text rows."
            )
        return self._position_ids

    def _scored_targets(self, positions: torch.Tensor) -> torch.Tensor:
        """The target at t+2 stays in the document t+1 opened; the collator masks a document's first label."""
        return (self._labels[:, 2:] != LABEL_IGNORE_INDEX) & (positions[:, 1:-1] != 0)

    @torch.no_grad()
    def _score_held_out(self, hidden: torch.Tensor, next_embedding: torch.Tensor, positions: torch.Tensor) -> None:
        draft = self._head(next_embedding, hidden, *self._layer_args, **self._layer_kwargs)
        self._embeddings = self._layer_args = self._layer_kwargs = None
        keep = self._scored_targets(positions)
        # The trunk's hidden at t+1 predicts what the draft at t predicts: token t+2.
        self._acceptance = score_drafts(
            self._project,
            hidden[:, 1:-1][keep],
            draft[:, :-2][keep],
            self._labels[:, 2:][keep],
            chunk_rows=self._chunk_rows,
        )

    def _run_head(self, module, _args, _kwargs, output):
        if self._labels is None:
            return output
        hidden = output.last_hidden_state
        positions = self._positions(hidden)
        # Index t reads token t+1 and hidden t, so it predicts token t+2. The last index repeats, and scores nothing.
        next_embedding = torch.cat([self._embeddings[:, 1:], self._embeddings[:, -1:]], dim=1)
        if self._spec.zero_first_embedding:
            next_embedding = torch.where((positions == 0).unsqueeze(-1), 0, next_embedding)
        if not module.training:
            self._score_held_out(hidden, next_embedding, positions)
            return output
        scoped = self._objective.scope != "joint"
        # Leaves, so a reentrant checkpoint around the head still finds a grad-requiring input.
        head_input = hidden.detach().requires_grad_() if scoped else hidden
        embedding_input = next_embedding.detach().requires_grad_() if scoped else next_embedding
        draft = self._head(embedding_input, head_input, *self._layer_args, **self._layer_kwargs)
        self._embeddings = self._layer_args = self._layer_kwargs = None

        # One rounded value on both sides of the soft identity.
        self._scale = draft.new_tensor(self._objective.weight)
        keep = self._scored_targets(positions)
        targets = self._labels[:, 2:].masked_fill(~keep, LABEL_IGNORE_INDEX)
        self._term = self._ce_sum(draft[:, :-2].reshape(-1, draft.size(-1)), targets.reshape(-1), frozen_head=scoped)
        self._term_targets = keep.sum()
        if self._objective.loss_type == "soft":
            # Pairs each draft with the next-token loss on the same target; a document start has no such pair.
            route = (draft[:, :-1].detach() - draft[:, :-1]) * (positions[:, 1:] != 0).unsqueeze(-1)
            output.last_hidden_state = hidden + self._scale * F.pad(route, (0, 0, 1, 0)).to(hidden.dtype)
        return output

    def _add_term(self, module, _args, _kwargs, output):
        if self._acceptance is not None:
            output.mtp_acceptance = self._acceptance
            self._clear()
            return output
        if self._term is None:
            return output
        targets = (self._labels[:, 1:] != LABEL_IGNORE_INDEX).sum()
        normalizer = targets.clamp_min(1) if self._num_items is None else self._num_items
        next_token = output.loss.detach()
        aux = getattr(output, "aux_loss", None)
        if aux is not None:
            next_token = next_token - require_router_aux_loss_coef(module.config) * aux.detach().to(next_token.dtype)
        output.mtp_loss_split = MtpLossSplit(next_token * normalizer, targets, self._term.detach(), self._term_targets)
        output.loss = output.loss + self._scale.to(self._term.dtype) * self._term / normalizer
        self._clear()
        return output
