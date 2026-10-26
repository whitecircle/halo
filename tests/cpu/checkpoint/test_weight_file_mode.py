"""Safetensors weight and adapter files follow the umask like every other checkpoint file.

``safetensors.torch.save_file`` creates its file ``0600`` whatever the umask, so a checkpoint whose JSON and
``torch.save`` files are group-readable would hold weights only their writer can read. Each writer below must
leave ``0o666 & ~umask``; the ``077`` case is the control where the two agree.
"""

import os
import stat

import pytest
import torch
from accelerate import PartialState
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from safetensors.torch import save_file
from sentence_transformers import SentenceTransformer
from sentence_transformers.models import Dense, Pooling, Router
from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3Model

from scripts.after_training.convert_to_bf16 import convert_to_bf16
from src.checkpoint.adapters import write_adapter_file
from src.checkpoint.offline_export import save_full_checkpoint
from src.checkpoint.writing import StreamingShardWriter, save_sharded_state_dict
from src.configs.embedding import EmbeddingConfig
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.embedding.preloaded_transformer import PreloadedTransformer
from src.trainers.embedding.trainer import EmbeddingTrainer
from tests.common.models import TINY_QWEN3_CONFIG
from tests.common.tiny_models import word_level_tokenizer


@pytest.fixture(params=[0o022, 0o002, 0o077], ids=lambda umask: f"umask-{umask:03o}")
def umask(request):
    previous = os.umask(request.param)
    try:
        yield request.param
    finally:
        os.umask(previous)


def _modes(directory) -> dict[str, str]:
    return {
        name: f"{stat.S_IMODE(os.stat(os.path.join(directory, name)).st_mode):o}"
        for name in sorted(os.listdir(directory))
        if name.endswith(".safetensors")
    }


def _expect(umask: int, names) -> dict[str, str]:
    return dict.fromkeys(sorted(names), f"{0o666 & ~umask:o}")


def _tensors(count: int = 4) -> dict[str, torch.Tensor]:
    return {f"layer.{index}.weight": torch.full((64, 64), float(index)) for index in range(count)}


def test_streamed_checkpoint_files_follow_the_umask(umask, tmp_path):
    writer = StreamingShardWriter(str(tmp_path), "5GB", enabled=True)
    for key, tensor in _tensors().items():
        writer.add(key, tensor)
    writer.close_as_hf_checkpoint()
    assert _modes(tmp_path) == _expect(umask, ["model.safetensors"])


def test_sharded_state_dict_files_follow_the_umask(umask, tmp_path):
    save_sharded_state_dict(_tensors(), str(tmp_path), max_shard_size="20KB")
    shards = [name for name in os.listdir(tmp_path) if name.endswith(".safetensors")]
    assert len(shards) > 1, f"premise: the save is sharded: {shards}"
    assert _modes(tmp_path) == _expect(umask, shards)


def test_adapter_file_follows_the_umask(umask, tmp_path):
    write_adapter_file({"base_model.model.q_proj.lora_A.weight": torch.ones(4, 8)}, str(tmp_path))
    assert _modes(tmp_path) == _expect(umask, ["adapter_model.safetensors"])


def test_full_checkpoint_files_follow_the_umask(umask, tmp_path):
    torch.manual_seed(0)
    save_full_checkpoint(Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG)), str(tmp_path), max_shard_size="1MB")
    shards = [name for name in os.listdir(tmp_path) if name.endswith(".safetensors")]
    assert len(shards) > 1, f"premise: the save is sharded: {shards}"
    assert _modes(tmp_path) == _expect(umask, shards)


def test_an_adapter_conversion_changes_the_mode_of_its_own_file_only(umask, tmp_path):
    """``convert_to_bf16 --peft`` writes adapter files and removes nothing, so an earlier save's weights in its
    output directory keep their mode (another user's file would refuse the chmod)."""
    PartialState()  # the tool's loads log through accelerate's logger
    Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG)).save_pretrained(tmp_path / "base")
    base = Qwen3ForCausalLM.from_pretrained(tmp_path / "base")
    get_peft_model(base, LoraConfig(r=4, target_modules=["q_proj"])).save_pretrained(tmp_path / "adapter")
    output = tmp_path / "output"
    output.mkdir()
    save_file({"layer.0.weight": torch.ones(4)}, output / "model.safetensors")

    convert_to_bf16(str(tmp_path / "adapter"), str(output), "causal_lm", is_peft=True)

    assert _modes(output) == {**_expect(umask, ["adapter_model.safetensors"]), "model.safetensors": "600"}


# A head past the pooling, and the module folders its save writes weights into: a ``Router`` nests one per route.
_EMBEDDING_HEADS = [
    pytest.param(lambda dimension: Dense(dimension, dimension), ["2_Dense"], id="dense"),
    pytest.param(
        lambda dimension: Router.for_query_document([Dense(dimension, dimension)], [Dense(dimension, dimension)]),
        ["2_Router/query_0_Dense", "2_Router/document_0_Dense"],
        id="router",
    ),
]


def _embedding_pipeline(head) -> SentenceTransformer:
    """The toolkit's backbone wrapper, mean pooling and ``head``."""
    torch.manual_seed(0)
    backbone = PreloadedTransformer(Qwen3Model(Qwen3Config(**TINY_QWEN3_CONFIG)), word_level_tokenizer(["a", "b"]))
    dimension = backbone.get_embedding_dimension()
    return SentenceTransformer(modules=[backbone, Pooling(dimension), head(dimension)], device="cpu")


@pytest.mark.parametrize(("head", "weight_folders"), _EMBEDDING_HEADS)
def test_an_embedding_save_follows_the_umask_in_every_module_folder(umask, head, weight_folders, tmp_path):
    """sentence-transformers writes each module's weights into its own folder (``2_Dense/``) at ``0600``,
    under the trainer's save of the whole pipeline."""
    args = EmbeddingConfig(
        output_dir=str(tmp_path / "run"),
        use_cpu=True,
        report_to="none",
        bf16=False,
        use_liger_kernel=False,
        max_length=16,
        router_mapping={"anchor": "query", "positive": "document"},  # the Router head's routes
    )
    pairs = Dataset.from_dict({"anchor": ["a b"], "positive": ["b a"]})
    trainer = EmbeddingTrainer(
        model=_embedding_pipeline(head), args=args, train_dataset=pairs, parallelism_config=ParallelismConfig()
    )

    trainer.save_model(str(tmp_path / "out"))

    assert _modes(tmp_path / "out") == _expect(umask, ["model.safetensors"])
    for folder in weight_folders:
        assert _modes(tmp_path / "out" / folder) == _expect(umask, ["model.safetensors"]), folder


@pytest.mark.parametrize(("head", "weight_folders"), _EMBEDDING_HEADS)
def test_a_distributed_embedding_save_follows_the_umask_in_every_module_folder(umask, head, weight_folders, tmp_path):
    """The pipeline modules past the backbone, which a sharded run's save writes one by one."""
    host = object.__new__(EmbeddingTrainer)

    host._save_st_pipeline_config(_embedding_pipeline(head), str(tmp_path))

    for folder in weight_folders:
        assert _modes(tmp_path / folder) == _expect(umask, ["model.safetensors"]), folder


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
