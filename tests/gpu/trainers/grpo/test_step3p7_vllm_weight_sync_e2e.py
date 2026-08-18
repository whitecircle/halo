#!/usr/bin/env python
"""Step-3.7 live vLLM weight sync: the served policy must move, and move TO the trainer's policy.

Step-3.7's live tree is spelled behind transformers' from_pretrained-side conversion while vLLM
0.26.0's ``step3p5`` loader maps the hub namespace and silently drops every other name — so a sync
that ships module-tree names returns ``200 OK`` and changes nothing. The gather-side pin
(``tests/cpu/grpo/test_weight_sync_hub_namespace.py``) proves the forwarded keys equal the hub
checkpoint's; only a live server proves the engine consumes them. This drives the production sync
(``sync_weights_to_client`` over the EP-wrapped model) against a served tiny checkpoint and compares
the server's next-token logprobs with the trainer's own forward, per weight group:

  1. before any sync, server and trainer agree (both loaded the same hub checkpoint);
  2. routed experts perturbed (the hub split of ``gate_up_proj``), sync → server moved, and agrees;
  3. router gate + shared expert perturbed (``moe.gate`` / ``share_expert.*`` renames), same;
  4. embeddings + lm_head, then the attention projections, then the attention norms perturbed (the
     top-level prefix renames, split so a projection-only or norm-only miss names itself), same.

Each group is a distinct conversion, so a rename that silently misses lands as "server did not move"
on exactly that step. The agreement tolerance is the bf16 kernel gap between the two stacks; every
perturbation moves the probe by an order of magnitude more.

The served checkpoint is a random-init Step-3.7 in the HUB repo layout (vendor ``config.json`` and
config class, hub-namespace safetensors, the real tokenizer) sized to what vLLM 0.26.0 accepts: its
fused-MoE clamp supports ``swiglu_limits`` of exactly 7, the vision tower keeps the hub's 728/14
geometry (vLLM's image processor is fixed to it), and the vocabulary is the real tokenizer's.

Prerequisites (the server must own a GPU the trainer does not):
    python tests/gpu/trainers/grpo/test_step3p7_vllm_weight_sync_e2e.py --write-checkpoint $DIR
    # all GPUs visible, masked by CUDA_VISIBLE_DEVICES: NCCL P2P to the trainer's GPU needs the peer
    # device present in the server container (``--gpus device=7`` fails init with 'unhandled cuda error')
    docker run -d --gpus all -e CUDA_VISIBLE_DEVICES=7 --network host --ipc host -v $DIR:$DIR \\
        -e NCCL_IB_DISABLE=1 -e NCCL_P2P_LEVEL=NVL vllm-server:0.26.0 $DIR --port 8000 \\
        --trust-remote-code --moe-backend triton --weight-transfer-config '{"backend": "nccl"}' \\
        --return-tokens-as-token-ids --max-model-len 1024 --gpu-memory-utilization 0.2 \\
        --max-logprobs 64 --enforce-eager
    HALO_TEST_STEP3P7_MODEL=$DIR VLLM_SERVER_URL=http://localhost:8000 VLLM_GROUP_HOST=127.0.0.1 \\
        NCCL_IB_DISABLE=1 NCCL_NET=Socket CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node=1 \\
        tests/gpu/trainers/grpo/test_step3p7_vllm_weight_sync_e2e.py
"""

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import requests
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from transformers.models.step3p7.configuration_step3p7 import Step3p7Config
from transformers.models.step3p7.modeling_step3p7 import Step3p7ForConditionalGeneration

from src.distributed.expert_parallel.base_layer import has_grouped_mm
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.nccl.clients.vllm import VLLMWeightSyncClient
from src.distributed.parallelism_config import ParallelismConfig
from src.env import env_int, env_str
from src.trainers.grpo.rollout.weight_sync import sync_weights_to_client, validate_weight_sync_support
from src.trainers.mixins.ep_introspection import named_ep_layers
from tests.common.harness import gpu_test_main
from tests.common.utils import log

VLLM_SERVER_URL = env_str("VLLM_SERVER_URL") or "http://localhost:8000"
CHECKPOINT = env_str("HALO_TEST_STEP3P7_MODEL")
GROUP_PORT = env_int("HALO_TEST_VLLM_GROUP_PORT", 51220)
HUB_REPO = "stepfun-ai/Step-3.7-Flash"
# Tokenizer, chat template and the vendor config class the served checkpoint copies from the hub.
HUB_FILES = (
    "configuration_step3p7.py",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
)
SEED = 0
PROBE_PROMPT = "The capital of France is"
PROBE_LOGPROBS = 16
# Bounds the bf16 kernel gap between the two stacks on this model. MOVE_MIN, the shift every
# perturbation below has to produce, sits 5x above it, so a match and a miss cannot overlap.
MATCH_TOL = 0.06
MOVE_MIN = 0.3
NOISE_STD = 0.05

LAYER_TYPES = ["full_attention", "sliding_attention", "full_attention", "sliding_attention"]
# The vendor spelling vLLM 0.26.0's ``Step3p5Config`` reads (transformers 5.16 loads it natively too).
SERVED_CONFIG = {
    "architectures": ["Step3p7ForConditionalGeneration"],
    "auto_map": {"AutoConfig": "configuration_step3p7.Step3p7Config"},
    "model_type": "step3p7",
    "image_token_id": 128001,
    "image_token_len": 169,
    "patch_token_len": 81,
    "understand_projector_stride": 2,
    "projector_bias": False,
    "vision_config": {
        "model_type": "perception_encoder",
        "image_size": 728,
        "patch_size": 14,
        "width": 32,
        "layers": 2,
        "heads": 2,
        "mlp_ratio": 2.0,
        "use_cls_token": False,
        "ls_init_value": 0.1,
        "use_ln_post": False,
        "hidden_act": "quick_gelu",
        "output_dim": None,
        "pool_type": "none",
    },
    "text_config": {
        "architectures": ["Step3p5ForCausalLM"],
        "model_type": "step3p5",
        "hidden_size": 256,
        "intermediate_size": 256,
        "num_hidden_layers": 4,
        "max_position_embeddings": 4096,
        "vocab_size": 128896,
        "torch_dtype": "bfloat16",
        "moe_layers_enum": "1,2,3",
        "num_attention_heads": 4,
        "num_attention_groups": 2,
        "head_dim": 64,
        "use_moe": True,
        "moe_num_experts": 8,
        "moe_top_k": 2,
        "moe_intermediate_size": 128,
        "share_expert_dim": 64,
        "norm_expert_weight": True,
        "moe_router_activation": "sigmoid",
        "moe_router_scaling_factor": 2.5,
        "att_impl_type": "GQA",
        "num_nextn_predict_layers": 0,
        "rope_theta": [5000000.0, 10000.0, 5000000.0, 10000.0],
        "rope_scaling": {
            "rope_type": "llama3",
            "factor": 2.0,
            "original_max_position_embeddings": 2048,
            "low_freq_factor": 1.0,
            "high_freq_factor": 32.0,
        },
        "yarn_only_types": ["full_attention"],
        "use_head_wise_attn_gate": True,
        "sliding_window": 64,
        "use_moe_router_bias": True,
        "need_fp32_gate": True,
        "layer_types": LAYER_TYPES,
        "partial_rotary_factors": [0.5, 1.0, 0.5, 1.0],
        "attention_other_setting": {
            "attention_type": "sliding_attention",
            "num_attention_heads": 2,
            "num_attention_groups": 2,
            "head_dim": 64,
        },
        "swiglu_limits": [0.0, 0.0, 7, 7],
        "swiglu_limits_shared": [0.0, 0.0, 16, 16],
        "eos_token_id": [1, 2, 128007],
        "bos_token_id": 0,
    },
}


def write_checkpoint(out: Path) -> None:
    """Materialize the served checkpoint: hub-layout files, then the seed-0 model's own save."""
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(SERVED_CONFIG, indent=2) + "\n")
    for name in HUB_FILES:
        shutil.copyfile(hf_hub_download(HUB_REPO, name), out / name)
    torch.manual_seed(SEED)
    model = Step3p7ForConditionalGeneration(Step3p7Config.from_pretrained(out)).to(torch.bfloat16)
    # save_pretrained writes the weights in the hub namespace but the config in transformers' own
    # spelling, which the vLLM image's transformers 5.14 cannot read — keep only the weight files.
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        model.save_pretrained(tmp)
        for part in Path(tmp).glob("*.safetensors*"):
            shutil.move(str(part), out / part.name)
    print(f"wrote {out}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params")


def _served_model_id() -> str:
    served = requests.get(f"{VLLM_SERVER_URL}/v1/models", timeout=60).json()["data"]
    return served[0]["id"]


def _server_logprobs(model_id: str, prompt_ids: list[int]) -> dict[int, float]:
    """Top-``PROBE_LOGPROBS`` next-token logprobs from the server, keyed by token id."""
    response = requests.post(
        f"{VLLM_SERVER_URL}/v1/completions",
        json={
            "model": model_id,
            "prompt": prompt_ids,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": PROBE_LOGPROBS,
        },
        timeout=120,
    )
    response.raise_for_status()
    top = response.json()["choices"][0]["logprobs"]["top_logprobs"][0]
    # ``--return-tokens-as-token-ids`` spells the keys ``token_id:<id>``.
    return {int(token.split(":")[1]): float(value) for token, value in top.items()}


@torch.no_grad()
def _trainer_logprobs(model, prompt_ids: list[int]) -> torch.Tensor:
    ids = torch.tensor([prompt_ids], device="cuda")
    logits = model(input_ids=ids, use_cache=False).logits[0, -1].float()
    return torch.log_softmax(logits, dim=-1)


def _gap(server: dict[int, float], trainer: torch.Tensor) -> float:
    return max(abs(value - trainer[token].item()) for token, value in server.items())


def _shift(before: dict[int, float], after: dict[int, float]) -> float:
    """How far the server's distribution moved, over the tokens both probes reported."""
    common = set(before) & set(after)
    return max(abs(before[t] - after[t]) for t in common) if common else float("inf")


def _trainer_shift(before: torch.Tensor, after: torch.Tensor, tokens) -> float:
    """How far the trainer's own distribution moved, over the tokens the server probe reports."""
    return max(abs(before[t].item() - after[t].item()) for t in tokens)


@torch.no_grad()
def _perturb(params: list[torch.Tensor], generator: torch.Generator) -> None:
    for param in params:
        param.add_(
            torch.randn(param.shape, generator=generator, device=param.device, dtype=torch.float32).to(param.dtype)
            * NOISE_STD
        )


def _weight_groups(model) -> dict[str, list[torch.Tensor]]:
    """The three conversion classes the sync must carry, as disjoint parameter groups."""
    ep_layers = named_ep_layers(model)
    experts = [param for layer in ep_layers.values() for _name, param in layer.expert_named_params()]
    router_and_shared = [
        param
        for layer in ep_layers.values()
        for name, param in layer.named_parameters()
        if name.startswith(("gate.", "shared_experts."))
    ]
    text = model.model.language_model
    attention = [
        (name, param)
        for layer in text.layers
        for name, param in layer.self_attn.named_parameters()
        if name.endswith("weight")
    ]

    def attention_group(*keys: str) -> list[torch.Tensor]:
        return [param for name, param in attention if any(key in name for key in keys)]

    return {
        "experts": experts,
        "router_and_shared_expert": router_and_shared,
        "embeddings_lm_head": [text.embed_tokens.weight, model.lm_head.weight],
        "attention_qk": attention_group("q_proj", "k_proj"),
        "attention_v": attention_group("v_proj"),
        "attention_o": attention_group("o_proj"),
        "attention_gate": attention_group("g_proj"),
        "attention_norms": attention_group("norm"),
    }


def run(ctx) -> dict:
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    assert CHECKPOINT, "HALO_TEST_STEP3P7_MODEL must name the served checkpoint (--write-checkpoint)"
    model_id = _served_model_id()
    log(f"server {VLLM_SERVER_URL} serves {model_id}; trainer loads {CHECKPOINT}")

    # The production load: the hub checkpoint through the native class, EP-wrapped at ep_size 1.
    parallelism_config = ParallelismConfig(ep_size=1, use_grouped_gemm=has_grouped_mm())
    model, _ = load_distributed_model(
        CHECKPOINT, parallelism_config, dtype=torch.bfloat16, trust_remote_code=False, attn_implementation="sdpa"
    )
    model.eval()
    checks["ep_wrapped"] = bool(named_ep_layers(model))
    validate_weight_sync_support(model)  # the construction gate must admit the family
    checks["construction_gate_admits_step3p7"] = True

    tokenizer = AutoTokenizer.from_pretrained(CHECKPOINT)
    prompt_ids = tokenizer(PROBE_PROMPT)["input_ids"]

    baseline = _server_logprobs(model_id, prompt_ids)
    checks["baseline_reproducible"] = baseline == _server_logprobs(model_id, prompt_ids)
    gap = _gap(baseline, _trainer_logprobs(model, prompt_ids))
    metrics["pre_sync_gap"] = gap
    checks["server_matches_trainer_before_sync"] = gap < MATCH_TOL
    log(f"  pre-sync gap (same checkpoint both sides): {gap:.4f}")

    client = VLLMWeightSyncClient(base_url=VLLM_SERVER_URL, group_port=GROUP_PORT, connection_timeout=120)
    client.init_communicator(device=torch.device("cuda", torch.cuda.current_device()))
    generator = torch.Generator(device="cuda").manual_seed(SEED)
    previous, previous_expected = baseline, _trainer_logprobs(model, prompt_ids)
    try:
        for group, params in _weight_groups(model).items():
            assert params, f"{group}: empty group — the perturbation would be a no-op"
            _perturb(params, generator)
            expected = _trainer_logprobs(model, prompt_ids)
            sync_weights_to_client(model, client, is_main=True, is_tp_main=True)
            after = _server_logprobs(model_id, prompt_ids)
            # The server must move about as far as the trainer's own forward moved: a group whose
            # tensors miss their slots leaves the server where it was (or moves it by the residual
            # of an earlier group alone).
            trainer_moved = _trainer_shift(previous_expected, expected, previous)
            moved, gap = _shift(previous, after), _gap(after, expected)
            metrics[f"{group}/trainer_moved"] = trainer_moved
            metrics[f"{group}/moved"] = moved
            metrics[f"{group}/gap"] = gap
            checks[f"{group}_sync_moved_the_server"] = moved > MOVE_MIN * min(1.0, trainer_moved / MOVE_MIN)
            checks[f"{group}_server_matches_trainer"] = gap < MATCH_TOL
            log(
                f"  {group}: trainer moved {trainer_moved:.3f}, server moved {moved:.3f}, "
                f"gap to trainer {gap:.4f} (tol {MATCH_TOL})"
            )
            previous, previous_expected = after, expected
    finally:
        client.close_communicator()
    return {"checks": checks, "metrics": metrics}


main = gpu_test_main(exact_world_size=1, prefix="step3p7_vllm_weight_sync_e2e")(run)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-checkpoint", type=Path, help="materialize the served checkpoint here and exit")
    args = parser.parse_args()
    if args.write_checkpoint is not None:
        write_checkpoint(args.write_checkpoint)
        sys.exit(0)
    main()
