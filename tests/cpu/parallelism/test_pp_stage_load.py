"""CPU tests for the stage-aware pipeline loader against a REAL safetensors checkpoint.

``tests/cpu/parallelism/test_pp_lazy_key_map.py`` pins the key algebra; this file runs
:func:`~src.distributed.pipeline_parallel.lazy_loader.load_pp_stage_model` end to end on a tiny
checkpoint written to disk, which is the only way to catch what the key map cannot:

  * every stage's tensors carry the VALUES of the global layers it claims (a correct-looking
    partition that streams the wrong slice loads plausible garbage);
  * nothing is left on the meta device;
  * a task head absent from the checkpoint (``score`` for reward / classification) is initialized —
    at the RUN's dtype, not the meta shell's. The shell is built at the checkpoint config's dtype
    while every loaded tensor is cast to the run's, so inheriting the shell's leaves a bf16 ``score``
    inside an fp32 model.

    python tests/cpu/parallelism/test_pp_stage_load.py
"""

import json
import os
import shutil

import pytest
import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForSequenceClassification, Qwen3Config, Qwen3ForCausalLM

from src.distributed.pipeline_parallel.lazy_loader import load_pp_stage_model
from src.distributed.pipeline_parallel.stage import PP_STAGE_PARTITION_ATTR
from tests.common.models import TINY_QWEN3_CONFIG

PP_SIZE = 2
NUM_LAYERS = TINY_QWEN3_CONFIG["num_hidden_layers"]
# Read off the config rather than restated, so the distribution gate below tracks the model it tests.
INITIALIZER_RANGE = Qwen3Config(**TINY_QWEN3_CONFIG).initializer_range


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> tuple[str, dict[str, torch.Tensor]]:
    """A tiny bf16 safetensors checkpoint plus its reference state dict."""
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG)).to(torch.bfloat16)
    path = tmp_path_factory.mktemp("pp_ckpt")
    model.save_pretrained(path, safe_serialization=True)
    return str(path), {k: v.clone() for k, v in model.state_dict().items()}


def _load(path: str, pp_rank: int, **kwargs):
    return load_pp_stage_model(
        path,
        pp_rank,
        PP_SIZE,
        config=AutoConfig.from_pretrained(path, trust_remote_code=False),
        trust_remote_code=False,
        **kwargs,
    )


@pytest.mark.parametrize("pp_rank", range(PP_SIZE))
def test_stage_holds_the_values_of_the_layers_it_claims(checkpoint, pp_rank):
    """What the key map alone cannot see: every layer shares its shapes, so streaming the wrong
    global slice into a stage is invisible until the loss curve is quietly wrong."""
    path, reference = checkpoint
    stage = _load(path, pp_rank, dtype=torch.bfloat16)
    lo, hi = getattr(stage, PP_STAGE_PARTITION_ATTR)[pp_rank]
    state = stage.state_dict()

    assert len(stage.model.layers) == hi - lo
    for local_index, global_index in enumerate(range(lo, hi)):
        for suffix in ("self_attn.q_proj.weight", "mlp.down_proj.weight", "input_layernorm.weight"):
            local = state[f"model.layers.{local_index}.{suffix}"]
            expected = reference[f"model.layers.{global_index}.{suffix}"]
            assert torch.equal(local.cpu(), expected.cpu()), (
                f"stage {pp_rank} local layer {local_index} does not hold global layer {global_index}"
            )


@pytest.mark.parametrize("pp_rank", range(PP_SIZE))
def test_no_tensor_is_left_on_meta(checkpoint, pp_rank):
    path, _ = checkpoint
    state = _load(path, pp_rank, dtype=torch.bfloat16).state_dict()
    assert [name for name, tensor in state.items() if tensor is not None and tensor.is_meta] == []


def test_the_partition_covers_every_layer_exactly_once(checkpoint):
    path, _ = checkpoint
    stages = [_load(path, rank, dtype=torch.bfloat16) for rank in range(PP_SIZE)]
    partition = getattr(stages[0], PP_STAGE_PARTITION_ATTR)
    assert partition == getattr(stages[1], PP_STAGE_PARTITION_ATTR)
    assert partition[0][0] == 0 and partition[-1][1] == NUM_LAYERS
    assert sum(hi - lo for lo, hi in partition) == NUM_LAYERS


@pytest.mark.parametrize("run_dtype", [torch.bfloat16, torch.float32])
def test_checkpoint_absent_head_is_initialized_at_the_run_dtype(checkpoint, run_dtype):
    """``score`` has no tensor on disk. It must be initialized (not meta, not the uninitialized
    memory a later meta sweep would leave) and share the dtype of the weights around it."""
    path, _ = checkpoint
    stage = _load(path, 1, dtype=run_dtype, model_class=AutoModelForSequenceClassification)
    state = stage.state_dict()

    score = state["score.weight"]
    assert not score.is_meta
    assert score.dtype == run_dtype
    assert score.dtype == state["model.layers.0.self_attn.q_proj.weight"].dtype
    # _init_weights ran: only the N(0, initializer_range) distribution separates it from torch.empty.
    assert torch.isfinite(score.float()).all()
    std = score.float().std().item()
    assert 0.5 * INITIALIZER_RANGE < std < 2.0 * INITIALIZER_RANGE, f"score std {std} is not an N(0, 0.02) draw"
    assert abs(score.float().mean().item()) < INITIALIZER_RANGE


def _load_head(path: str, pp_rank: int, seed: int = 0) -> torch.Tensor:
    torch.manual_seed(seed)
    stage = _load(path, pp_rank, dtype=torch.bfloat16, model_class=AutoModelForSequenceClassification)
    return stage.state_dict()["score.weight"].clone()


def test_absent_head_comes_from_the_rng_so_every_stage_draws_the_same(checkpoint):
    """Every stage materializes the head and the last one keeps it, so the kept copy must not depend
    on which rank survived ``build_pipeline_stage``.

    Same seed → bit-identical across stages (what ranks do). Different seed → different (what proves
    the values are an RNG draw and not whatever the allocator handed back, which no seed would
    change). Without the second half, a per-stage draw at the same seed would pass unnoticed.
    """
    path, _ = checkpoint
    heads = [_load_head(path, pp_rank) for pp_rank in range(PP_SIZE)]
    reseeded = _load_head(path, PP_SIZE - 1, seed=1)

    assert torch.equal(heads[0], heads[1])
    assert not torch.equal(heads[-1], reseeded)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))


# Per-node PP checkpoints (DIST_OUTPUT_SHARED_FILESYSTEM=0)
# A per-node PP save leaves each node its own stage's shards plus the FULL index, so an absent
# other-stage shard is legitimate only when its planned keys are cross-stage modules the stage drops.


@pytest.fixture(scope="module")
def per_node_dirs(checkpoint, tmp_path_factory) -> tuple[str, str]:
    """Two directories mimicking a per-node PP save of the tiny checkpoint (full index, one shard)."""
    path, reference = checkpoint
    partition = getattr(_load(path, 0, dtype=torch.bfloat16), PP_STAGE_PARTITION_ATTR)
    hi0 = partition[0][1]

    def stage_of(key: str) -> int:
        if key.startswith("model.layers."):
            return 0 if int(key.split(".")[2]) < hi0 else 1
        return 0 if key.startswith("model.embed_tokens") else 1  # norm + lm_head belong to the last

    shards: dict[int, dict[str, torch.Tensor]] = {0: {}, 1: {}}
    for key, value in reference.items():
        shards[stage_of(key)][key] = value.contiguous()
    shard_name = {s: f"model-pp{s:05d}-of-00002-00001.safetensors" for s in (0, 1)}
    weight_map = {key: shard_name[stage_of(key)] for key in reference}

    dirs = []
    for node in (0, 1):
        d = str(tmp_path_factory.mktemp(f"pp_node{node}"))
        shutil.copy(os.path.join(path, "config.json"), os.path.join(d, "config.json"))
        save_file(shards[node], os.path.join(d, shard_name[node]))
        with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
            json.dump({"metadata": {}, "weight_map": weight_map}, f)
        dirs.append(d)
    return dirs[0], dirs[1]


def test_per_node_dir_resumes_its_own_stage(checkpoint, per_node_dirs):
    """Each node's directory loads its own stage: real values for what the stage retains,
    materialized (never meta) storage for the cross-stage modules the stage build drops."""
    _, reference = checkpoint
    node0, node1 = per_node_dirs

    stage0 = _load(node0, 0, dtype=torch.bfloat16)
    state0 = stage0.state_dict()
    assert torch.equal(state0["model.embed_tokens.weight"].cpu(), reference["model.embed_tokens.weight"].cpu())
    assert torch.equal(
        state0["model.layers.0.self_attn.q_proj.weight"].cpu(),
        reference["model.layers.0.self_attn.q_proj.weight"].cpu(),
    )
    # Droppable cross-stage tensors: storage exists (the coverage gate and FSDP need it), values unread.
    assert not state0["lm_head.weight"].is_meta
    assert not state0["model.norm.weight"].is_meta

    lo1 = getattr(stage0, PP_STAGE_PARTITION_ATTR)[1][0]
    state1 = _load(node1, 1, dtype=torch.bfloat16).state_dict()
    assert torch.equal(state1["lm_head.weight"].cpu(), reference["lm_head.weight"].cpu())
    assert torch.equal(state1["model.norm.weight"].cpu(), reference["model.norm.weight"].cpu())
    assert torch.equal(
        state1["model.layers.0.self_attn.q_proj.weight"].cpu(),
        reference[f"model.layers.{lo1}.self_attn.q_proj.weight"].cpu(),
    )
    assert not state1["model.embed_tokens.weight"].is_meta


def test_per_node_dir_rejects_a_stage_it_does_not_hold(per_node_dirs):
    """Loading stage 1 from node 0's directory needs stage-1 LAYERS from the absent shard — that is
    missing trained state, not a droppable cross-stage module, and must fail loud."""
    node0, _ = per_node_dirs
    with pytest.raises(RuntimeError, match="NOT the cross-stage modules"):
        _load(node0, 1, dtype=torch.bfloat16)
