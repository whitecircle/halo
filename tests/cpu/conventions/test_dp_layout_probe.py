#!/usr/bin/env python
"""The FULL_SHARD layout probe of ``tests.common.dp_layout`` forwards the module a run's FSDP2 wrap is rooted at.

The data-parallel GPU suites read ZeRO-3 off the trained model through a probe forward, and they run only on the GPU
tiers. The root's own pre-forward unshards the root's group, so a probe entering below it meets the parameters that
group holds as sharded DTensors: a backbone probe fails that way on the SentenceTransformer the embedding trainer
wraps, while the pipeline probe enters at the root and its embedding reads gathered weights. The wrap passes the
reshard mode explicitly, so FSDP2 reshards the root after its forward too: the probe tells FULL_SHARD, whose groups
are back on their shards after it, from SHARD_GRAD_OP, which leaves them gathered, on that pipeline (whose one group
is the root's) as on a causal LM wrapped per decoder layer. One rank, gloo.

Run: python tests/cpu/conventions/test_dp_layout_probe.py
"""

import pytest
import torch
import torch.distributed as dist
from sentence_transformers import SentenceTransformer, models
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

from halo.distributed.fsdp import IdentityParamSet, apply_fsdp2_per_layer, mixed_precision_policy
from tests.common.dp_layout import full_shard_checks, probe_forward, probe_pipeline_forward
from tests.common.offline_grpo import make_offline_tokenizer
from tests.common.tiny_models import TINY_DENSE_FAMILIES, build_tiny_family_checkpoint, tiny_family_model

DEVICE = torch.device("cpu")
POLICY = mixed_precision_policy(torch.bfloat16, fp32_reduce=False, cast_forward_inputs=True)


@pytest.fixture
def mesh(tmp_path):
    dist.init_process_group("gloo", rank=0, world_size=1, init_method=f"file://{tmp_path / 'pg'}")
    try:
        yield init_device_mesh("cpu", (1,))
    finally:
        dist.destroy_process_group()


def _wrap(model, mesh, *, reshard_after_forward: bool):
    apply_fsdp2_per_layer(model, mesh, POLICY, reshard_after_forward, IdentityParamSet())
    return model


def _pipeline(tmp_path) -> SentenceTransformer:
    """The tiny Qwen3 as sentence-transformers' own load assembles it: the transformer backbone, then pooling."""
    source = str(tmp_path / "backbone")
    torch.manual_seed(0)
    build_tiny_family_checkpoint(TINY_DENSE_FAMILIES["qwen3"], source, make_offline_tokenizer(), seed=0)
    backbone = models.Transformer(source)
    pooling = models.Pooling(backbone.get_embedding_dimension(), "lasttoken")
    return SentenceTransformer(modules=[backbone, pooling]).to(torch.bfloat16)


def test_a_probe_below_the_root_meets_its_sharded_embedding(mesh, tmp_path):
    """The wrap roots at the pipeline, whose group holds the backbone's embedding."""
    pipeline = _wrap(_pipeline(tmp_path), mesh, reshard_after_forward=True)
    with pytest.raises(RuntimeError, match="mixed torch.Tensor and DTensor"):
        full_shard_checks(pipeline, lambda: probe_forward(pipeline[0].auto_model, DEVICE))


@pytest.mark.parametrize("reshard_after_forward", [True, False], ids=["full_shard", "shard_grad_op"])
def test_the_pipeline_probe_runs_through_the_root(mesh, tmp_path, reshard_after_forward):
    pipeline = _wrap(_pipeline(tmp_path), mesh, reshard_after_forward=reshard_after_forward)
    seen: dict[str, bool] = {}

    def entered_root(module, args) -> None:
        seen["root"] = True

    def embedded(module, args) -> None:
        seen["embedding_weight_sharded"] = isinstance(module.weight, DTensor)

    pipeline.register_forward_pre_hook(entered_root)
    pipeline[0].auto_model.get_input_embeddings().register_forward_pre_hook(embedded)
    checks = full_shard_checks(pipeline, lambda: probe_pipeline_forward(pipeline, DEVICE))
    assert seen == {"root": True, "embedding_weight_sharded": False}
    assert checks == {"fsdp_groups_resharded_after_forward": reshard_after_forward}


@pytest.mark.parametrize("reshard_after_forward", [True, False], ids=["full_shard", "shard_grad_op"])
def test_the_probe_tells_full_shard_from_shard_grad_op(mesh, reshard_after_forward):
    torch.manual_seed(0)
    model = tiny_family_model(TINY_DENSE_FAMILIES["qwen3"]).to(torch.bfloat16)
    _wrap(model, mesh, reshard_after_forward=reshard_after_forward)
    checks = full_shard_checks(model, lambda: probe_forward(model, DEVICE))
    assert checks == {"fsdp_groups_resharded_after_forward": reshard_after_forward}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
