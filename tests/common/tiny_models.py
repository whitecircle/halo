"""Builders a tiny random-init model needs before it can run: a synthetic on-disk checkpoint, a filled
routing table.

Kept apart from :mod:`tests.common.models`, the torch-free catalogue of names and configs that the CPU
tier and the image build scripts import.
"""

import shutil
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers.models.mistral4 import Mistral4Config, Mistral4ForCausalLM

from tests.common.models import MISTRAL3_119B_MOE, TINY_MISTRAL4_CONFIG

# Seed ``randomize_tid2eid`` fills the hash table from unless a test pins its own.
DSV4_TID2EID_SEED = 1234


def randomize_tid2eid(model, seed: int = DSV4_TID2EID_SEED) -> None:
    """Fill every DeepSeek-V4 hash layer's ``tid2eid`` with DISTINCT experts per token id.

    Random init leaves the table all-zero, and DeepEP dispatch and the EP wrapper's init guard both
    require distinct top-k experts per token.
    """
    gen = torch.Generator().manual_seed(seed)
    num_experts = model.config.n_routed_experts
    for layer in model.model.layers:
        if layer.mlp.is_hash:
            table = layer.mlp.gate.tid2eid
            perm = torch.rand(table.shape[0], num_experts, generator=gen).argsort(dim=-1)
            table.copy_(perm[:, : table.shape[1]])


# The files a synthetic checkpoint copies from its release so ``AutoTokenizer`` loads it offline.
TOKENIZER_FILE_PREFIXES = ("tokenizer", "special_tokens", "chat_template")


def build_tiny_mistral4_checkpoint(out_dir: Path, seed: int = 0) -> Path:
    """Write a random-init :data:`TINY_MISTRAL4_CONFIG` model to ``out_dir`` and return the path.

    The layout ``save_pretrained`` writes (``model.safetensors`` + ``config.json``), plus the release's
    tokenizer files, so the lazy loader and ``load_distributed_model`` run end to end without the
    119B download.
    """
    tokenizer_dir = Path(
        snapshot_download(MISTRAL3_119B_MOE, allow_patterns=[f"{p}*" for p in TOKENIZER_FILE_PREFIXES])
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for src in tokenizer_dir.iterdir():
        if src.is_file() and src.name.startswith(TOKENIZER_FILE_PREFIXES):
            shutil.copy2(src, out_dir / src.name)

    torch.manual_seed(seed)
    model = Mistral4ForCausalLM(Mistral4Config(**TINY_MISTRAL4_CONFIG)).to(torch.bfloat16)
    model.save_pretrained(out_dir, safe_serialization=True)
    return out_dir
