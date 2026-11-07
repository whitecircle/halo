"""The tiny Flash-Next family the ``qwen4_exp`` tests share, in the release's checkpoint layout with a host-resident
n-gram table."""

import contextlib
import dataclasses

import torch
import transformers.models.qwen4_exp.modeling_qwen4_exp as qwen4_exp_modeling
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_leaves
from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedModel
from transformers.conversion_mapping import get_model_conversion_mapping
from transformers.core_model_loading import PrefixChange
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig

from halo.model_loading.checkpoint_keys import is_expert_bank_merge
from halo.model_loading.finalization import finalize_loaded_model
from halo.model_loading.model_preparation import apply_family_patches
from halo.models.external_tables import ExternalTable, named_external_tables
from halo.models.families.qwen4_exp.forwards import patch_qwen4_exp_document_forwards
from halo.models.families.qwen4_exp.sparse_attention import QSASelection
from halo.models.segment_markers import document_ids, segment_marker_kwargs, segment_markers_for
from tests.common.models import TINY_QWEN4_EXP_CONFIG
from tests.common.tiny_models import TINY_MOE_FAMILIES, TINY_SHARD_SIZE, tiny_family_model

EOS_TOKEN_ID = TINY_QWEN4_EXP_CONFIG["eos_token_id"]
QSA_LAYER_IDX = TINY_QWEN4_EXP_CONFIG["layer_types"].index("full_attention")
# The classes whose forwards the document patch replaces.
_DOCUMENT_FORWARD_CLASSES = (
    qwen4_exp_modeling.Qwen4ExpTextModel,
    qwen4_exp_modeling.Qwen4ExpTextDecoderLayer,
    qwen4_exp_modeling.Qwen4ExpTextAttention,
)
TINY_QWEN4_EXP_FAMILY = TINY_MOE_FAMILIES["qwen4_exp_text"]
# Where the tiny text model's one PLE layer holds its table.
NGRAM_TABLE_PATH = "model.layers.1.ple.ple_embedding.ngram_embedding"


def tiny_qwen4_exp_text_config(**overrides) -> Qwen4ExpTextConfig:
    """The tiny text config on SDPA; ``indexer_budget=8`` makes every document past two 4-token blocks plus its tail
    sparse."""
    config = Qwen4ExpTextConfig(**{**TINY_QWEN4_EXP_CONFIG, **overrides})
    config._attn_implementation = "sdpa"
    return config


def rotary_embeddings(
    config: Qwen4ExpTextConfig, hidden_states: torch.Tensor, position_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The model's ``(cos, sin)`` for ``[batch, length]`` text positions, the same on every M-RoPE axis."""
    rotary = qwen4_exp_modeling.Qwen4ExpTextRotaryEmbedding(config)
    return rotary(hidden_states, position_ids.expand(3, -1, -1))


def packed_positions(rows: list[list[int]]) -> torch.Tensor:
    """``[len(rows), length]`` position ids restarting at 0 at each document."""
    return torch.stack([torch.cat([torch.arange(n) for n in lengths]) for lengths in rows])


def packed_seq_idx(position_ids: torch.Tensor) -> torch.Tensor:
    return document_ids(position_ids).to(torch.int32)


def packed_markers(config, position_ids: torch.Tensor, *, cu_seq_lens: bool = True) -> dict:
    """Every segment marker the packing collators emit on a ``[1, total]`` row; ``cu_seq_lens=False`` leaves out
    GatedDeltaNet's varlen set."""
    markers = segment_markers_for(config)
    if not cu_seq_lens:
        markers = dataclasses.replace(markers, cu_seq_lens=False)
    return segment_marker_kwargs(position_ids, markers)


def selection_keys(selection: QSASelection, batch: int, length: int) -> torch.Tensor:
    """``[batch, length, length]`` on the CPU: the keys ``selection`` lets each query attend, to compare with
    transformers' dense mask."""
    keys = torch.zeros(batch, length, length, dtype=torch.bool)
    blocks = selection.blocks.cpu().long()
    ratio = selection.compress_ratio
    for doc in selection.documents:
        span = doc.key_stop - doc.key_start
        for query in range(doc.start, doc.stop):
            visible = min(max(query - doc.key_start + 1, 0), span)
            keys[doc.row, query, doc.key_start + visible // ratio * ratio : doc.key_start + visible] = True
            chosen = blocks[doc.row, query]
            chosen = chosen[chosen >= 0]
            keys[doc.row, query, (doc.key_start + chosen[:, None] * ratio + torch.arange(ratio)).flatten()] = True
    return keys


@contextlib.contextmanager
def transformers_forwards():
    """transformers' own forwards in place of the toolkit's document forwards, for a stock reference."""
    patch_qwen4_exp_document_forwards()
    installed = {cls: cls.forward for cls in _DOCUMENT_FORWARD_CLASSES}
    for cls, forward in installed.items():
        cls.forward = forward.__wrapped__
    try:
        yield
    finally:
        for cls, forward in installed.items():
            cls.forward = forward


class LargestTensor(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.numel, self.shape = 0, ()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        output = func(*args, **(kwargs or {}))
        for tensor in tree_leaves(output):
            if isinstance(tensor, torch.Tensor) and tensor.numel() > self.numel:
                self.numel, self.shape = tensor.numel(), tuple(tensor.shape)
        return output


def build_tiny_qwen4_exp_checkpoint(directory: str, *, seed: int = 0, overrides: dict | None = None) -> None:
    """The seeded tiny text model in bf16 as the release stores it: fused banks beside the table's ``shard_*`` keys."""
    torch.manual_seed(seed)
    model = tiny_family_model(TINY_QWEN4_EXP_FAMILY, overrides=overrides).to(torch.bfloat16)
    model._weight_conversions = [
        conversion
        for conversion in get_model_conversion_mapping(model, add_legacy=False)
        if not is_expert_bank_merge(conversion) and not isinstance(conversion, PrefixChange)
    ]
    model.save_pretrained(directory, max_shard_size=TINY_SHARD_SIZE)


def stamped_config(directory: str):
    """``directory``'s config after the family patches a run's load applies, which stamp it for host tables."""
    config = AutoConfig.from_pretrained(directory)
    apply_family_patches(config, "sdpa")
    return config


def load_host_table_model(
    directory: str, *, dtype: torch.dtype = torch.float32, model_class=AutoModelForCausalLM
) -> PreTrainedModel:
    """``directory`` through the family patches and ``finalize_loaded_model``, as the loaders do."""
    model = model_class.from_pretrained(directory, config=stamped_config(directory), dtype=dtype)
    finalize_loaded_model(model)
    return model


def ngram_table(model: torch.nn.Module) -> ExternalTable:
    (table,) = (table for _, table in named_external_tables(model))
    return table
