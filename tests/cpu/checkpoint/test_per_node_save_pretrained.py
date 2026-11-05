"""``save_pretrained`` writes weights on the filesystem-aware save rank, so each node of a per-node output filesystem
keeps a complete copy. Two gloo ranks stand for two nodes of one process each. transformers writes only on global rank
0, which left node 1 with a config and no weights."""

import glob
import os

import pytest
from tokenizers import Tokenizer, models
from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
from transformers.models.llama.modeling_llama import LlamaModel

from halo.checkpoint.format import SAFETENSORS_FAMILY_GLOB
from halo.checkpoint.parallel.writing import save_pretrained_on_save_rank
from halo.trainers.embedding.preloaded_transformer import PreloadedTransformer
from tests.common.gloo import run_gloo_ranks

WORLD_SIZE = 2
_TINY = {
    "vocab_size": 32,
    "hidden_size": 16,
    "intermediate_size": 32,
    "num_hidden_layers": 1,
    "num_attention_heads": 2,
}


def _weight_files(directory: str) -> list[str]:
    return glob.glob(os.path.join(directory, SAFETENSORS_FAMILY_GLOB))


def _be_a_node_of_one() -> None:
    """The harness numbers LOCAL_RANK by global rank; a node of one process has local rank 0."""
    os.environ["LOCAL_RANK"] = "0"


def _worker(rank: int, root: str) -> None:
    _be_a_node_of_one()
    model = LlamaForCausalLM(LlamaConfig(**_TINY))
    stock = os.path.join(root, f"stock-node{rank}")
    model.save_pretrained(stock)
    routed = os.path.join(root, f"routed-node{rank}")
    with save_pretrained_on_save_rank(model):
        model.save_pretrained(routed)
    assert "should_save_on_this_rank" not in vars(model), "the override outlived its block"


def _node_files(tmp_path, env: dict[str, str]) -> dict[str, list[str]]:
    run_gloo_ranks(_worker, WORLD_SIZE, str(tmp_path), env={"LOCAL_WORLD_SIZE": "1", **env})
    return {
        f"{kind}-node{rank}": _weight_files(str(tmp_path / f"{kind}-node{rank}"))
        for kind in ("stock", "routed")
        for rank in range(WORLD_SIZE)
    }


def test_each_node_of_a_per_node_filesystem_gets_the_weights(tmp_path):
    files = _node_files(tmp_path, {"DIST_OUTPUT_SHARED_FILESYSTEM": "0"})
    assert files["stock-node0"], "control: global rank 0 writes through the stock path"
    assert not files["stock-node1"], "control: the stock path must leave node 1 without weights"
    assert files["routed-node0"] and files["routed-node1"], files


def test_a_shared_filesystem_keeps_one_writer(tmp_path):
    files = _node_files(tmp_path, {"DIST_OUTPUT_SHARED_FILESYSTEM": "1"})
    assert files["routed-node0"], files
    assert not files["routed-node1"], "a second writer on shared storage"


def _embedding_worker(rank: int, root: str) -> None:
    _be_a_node_of_one()
    backbone = LlamaModel(LlamaConfig(**_TINY))
    vocab = {f"t{i}": i for i in range(_TINY["vocab_size"])}
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(models.WordLevel(vocab, unk_token="t0")))
    PreloadedTransformer(backbone, tokenizer).save(os.path.join(root, f"embedding-node{rank}"))


def test_the_embedding_backbone_save_reaches_every_node(tmp_path):
    run_gloo_ranks(
        _embedding_worker,
        WORLD_SIZE,
        str(tmp_path),
        env={"LOCAL_WORLD_SIZE": "1", "DIST_OUTPUT_SHARED_FILESYSTEM": "0"},
    )
    for rank in range(WORLD_SIZE):
        assert _weight_files(str(tmp_path / f"embedding-node{rank}")), f"node {rank} holds no weights"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
