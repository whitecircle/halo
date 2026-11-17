#!/usr/bin/env python
"""``openai/gpt-oss-20b``, the natively MXFP4 release, lazily loaded at EP=8 with no third-party BF16 mirror:

- every EP layer, gathered, equals the release decoded by transformers' ``convert_moe_packed_tensors``, bit for bit:
  experts from ``_blocks``/``_scales``, biases and routers as stored,
- the loaded config no longer claims MXFP4,
- the loss on a fixed batch matches ``from_pretrained(..., quantization_config=Mxfp4Config(dequantize=True))`` run
  undistributed on rank 0,
- each rank's peak host RAM during the load is reported, as each decodes only its own experts.

    torchrun --nproc_per_node=8 tests/gpu/parallelism/ep/test_gptoss_native_mxfp4_ep8.py
"""

import resource

import torch
import torch.distributed as dist
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer, GptOssForCausalLM, Mxfp4Config
from transformers.integrations.mxfp4 import convert_moe_packed_tensors

from halo.checkpoint.reading import resolve_checkpoint_weights
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.lazy_plan import is_lazy_loadable
from halo.model_loading.parallel.loader import load_distributed_model
from tests.common.distributed import ensure_model_downloaded, world_all
from tests.common.ep_reference import fixed_chat_batch
from tests.common.harness import gpu_test_main
from tests.common.memory import cleanup_memory
from tests.common.models import GPT_OSS_20B_OPENAI
from tests.common.rank_log import log, log_all
from tests.common.tolerances import TOL

EP_SIZE = 8
SEQ_LEN = 128
# EP reorders the bf16 expert sum, and later routers score that reordered output, flipping near-tied top-4-of-32 picks
# with no bug: the bound test_ep_modes_vs_reference.py holds gpt-oss to. On 8x H200 the release and its BF16 mirror
# both load to an EP8 loss of 4.158315 on this batch, 0.058 from stock transformers at 4.216044.
LOSS_TOL = TOL.router_pick_flip_loss_abs


def _peak_host_rss_gb() -> float:
    """This process's host RAM high-water mark (Linux reports ``ru_maxrss`` in KiB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def _world_max(value: float, device) -> float:
    t = torch.tensor([value], device=device, dtype=torch.float64)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def _loss(model, batch) -> float:
    input_ids, attention_mask, labels = batch
    model.eval()
    with torch.no_grad():
        return model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, use_cache=False).loss.item()


class _Release:
    """Rank 0's reader of the release as stored: a decoded expert bank, or any other tensor as written."""

    def __init__(self, local_dir: str):
        self._dir = local_dir
        self._weight_map = resolve_checkpoint_weights(local_dir).require_safetensors("checkpoint").weight_map
        self._handles: dict = {}

    def _read(self, key: str) -> torch.Tensor:
        shard = self._weight_map[key]
        if shard not in self._handles:
            self._handles[shard] = safe_open(f"{self._dir}/{shard}", framework="pt", device="cpu")
        return self._handles[shard].get_tensor(key)

    def expected(self, key: str) -> torch.Tensor | None:
        if key in self._weight_map:
            return self._read(key)
        if f"{key}_blocks" in self._weight_map:
            return convert_moe_packed_tensors(self._read(f"{key}_blocks"), self._read(f"{key}_scales"))
        return None


def _layer_mismatches(release: _Release | None, gathered: dict) -> list[str]:
    bad = []
    for key, tensor in gathered.items():
        expected = release.expected(key)
        if expected is None:
            bad.append(f"{key}: not in the release")
        elif expected.shape != tensor.shape or not torch.equal(expected.to(tensor.dtype), tensor):
            bad.append(f"{key}: differs from the release")
    return bad


@gpu_test_main(exact_world_size=EP_SIZE, prefix="gptoss_native_mxfp4_ep8")
def run(ctx) -> dict:
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    ensure_model_downloaded(GPT_OSS_20B_OPENAI, ctx.rank)
    if ctx.rank == 0:
        snapshot_download(GPT_OSS_20B_OPENAI, allow_patterns=["*.json", "*.safetensors"])
    ctx.barrier()
    local_dir = snapshot_download(GPT_OSS_20B_OPENAI, allow_patterns=["*.json", "*.safetensors"])
    checks["lazy_gate_admits_the_release"] = is_lazy_loadable(local_dir)

    baseline_gb = _peak_host_rss_gb()
    model, _ = load_distributed_model(
        model_name_or_path=GPT_OSS_20B_OPENAI,
        parallelism_config=ParallelismConfig(ep_size=EP_SIZE),
        dtype=torch.bfloat16,
        attn_implementation="eager",
        use_liger_kernel=False,
        trust_remote_code=True,
        # The reference below is the release as transformers loads it, sinks live; the loader neutralizes them by
        # default, and eager is the one backend that carries live ones.
        reset_sinks=False,
    )
    peak_gb = _peak_host_rss_gb()
    log_all(f"host RAM high-water mark: {baseline_gb:.2f} GB before the load, {peak_gb:.2f} GB after")
    metrics["peak_host_rss_gb_max_rank"] = _world_max(peak_gb, ctx.device)
    metrics["load_host_rss_growth_gb_max_rank"] = _world_max(peak_gb - baseline_gb, ctx.device)
    checks["config_no_longer_claims_mxfp4"] = getattr(model.config, "quantization_config", None) is None

    tokenizer = AutoTokenizer.from_pretrained(GPT_OSS_20B_OPENAI)
    batch = fixed_chat_batch(tokenizer, SEQ_LEN, ctx.device, broadcast=True)
    ep_loss = _loss(model, batch)
    metrics["ep8_loss"] = ep_loss
    # Pure EP repeats bit for bit, so a difference means state survived between the two forwards.
    checks["repeated_forward_is_deterministic"] = _loss(model, batch) == ep_loss

    release = _Release(local_dir) if ctx.rank == 0 else None
    mismatches: list[str] = []
    layers = named_ep_layers(model)
    for name, layer in layers:
        gathered = layer.gather_layer_state_dict(name, retain=ctx.rank == 0)
        if ctx.rank == 0:
            mismatches += _layer_mismatches(release, gathered)
        del gathered
    verdict = [mismatches]
    dist.broadcast_object_list(verdict, src=0)
    if verdict[0]:
        log(f"{len(verdict[0])} tensor(s) differ from the release, e.g. {verdict[0][:4]}")
    checks["every_ep_layer_equals_the_decoded_release"] = bool(layers) and not verdict[0]
    del model, layers
    cleanup_memory()
    ctx.barrier()

    ref_loss = torch.zeros(1, device=ctx.device, dtype=torch.float64)
    if ctx.rank == 0:
        reference = GptOssForCausalLM.from_pretrained(
            local_dir,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            quantization_config=Mxfp4Config(dequantize=True),
        ).to(ctx.device)
        ref_loss[0] = _loss(reference, batch)
        del reference
        cleanup_memory()
    dist.broadcast(ref_loss, src=0)
    metrics["reference_loss"] = float(ref_loss.item())
    delta = abs(ep_loss - metrics["reference_loss"])
    log(f"EP8 loss {ep_loss:.6f} vs dequantized from_pretrained {metrics['reference_loss']:.6f} (|Δ|={delta:.2e})")
    checks["ep8_loss_matches_dequantized_reference"] = delta < LOSS_TOL
    checks = {name: world_all(ok, ctx.device) for name, ok in checks.items()}
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
