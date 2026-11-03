#!/usr/bin/env python
"""ZAYA1-8B load, forward, backward and optimizer step on one GPU, isolating the modeling + patch path.

Each stage builds on the one before. Memory on a B300: ~17 GB bf16 weights, ~70 GB AdamW state at fp32, ~90 GB total.
"""

import torch
from transformers import AutoConfig

from halo.distributed.expert_parallel.families.zaya import EPZayaMoELayer
from halo.distributed.parallelism_config import ParallelismConfig
from halo.env import env_str
from halo.model_loading.parallel.loader import load_distributed_model
from tests.common.harness import gpu_test_main
from tests.common.models import ZAYA_8B
from tests.common.rank_log import log

MODEL = env_str("HALO_TEST_ZAYA_MODEL", ZAYA_8B)
# Keeps the activation memory bounded.
MAX_PROMPT_TOKENS = 64
# More parameters than any truncated or partially-loaded ZAYA1-8B checkpoint holds.
MIN_PARAMS = 8_000_000_000
# Every one must be found: a renamed module would otherwise shrink the sample silently.
SAMPLE_PARAM_SUFFIXES = (
    "self_attn.qkv_proj.q_proj.weight",
    "mlp.gate.down_proj.weight",
    "post_mlp_residual_scale.hidden_states_bias",
    "input_layernorm.weight",
)


@gpu_test_main(exact_world_size=1, prefix="zaya_load_forward_backward")
def run(ctx):
    log(f"  ZAYA1-8B load + fwd + bwd smoke test: {MODEL} on {torch.cuda.get_device_name(ctx.local_rank)}")
    checks: dict[str, bool] = {}

    cfg = AutoConfig.from_pretrained(MODEL)
    checks["native_config"] = type(cfg).__name__ == "ZayaConfig" and type(cfg).__module__.startswith("transformers.")
    log(f"  Config class: {type(cfg).__name__} ({type(cfg).__module__})")
    if not checks["native_config"]:
        return {"checks": checks}

    # The toolkit loader applies the Zaya patches (load recording, the GC refusal, flash position_ids), the path every
    # run takes.
    model, tokenizer = load_distributed_model(
        model_name_or_path=MODEL,
        parallelism_config=ParallelismConfig(),
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        trust_remote_code=True,
    )
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  Model loaded: {n_params / 1e9:.2f}B params, HBM {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    checks["all_params_loaded"] = n_params > MIN_PARAMS
    # EPZayaMoELayer stores the fused experts in matmul convention: gate_up [E, H, 2M], down [E, M, H].
    block = next((module for module in model.modules() if isinstance(module, EPZayaMoELayer)), None)
    checks["experts_wrapped_fused"] = block is not None and all(
        weight.dim() == 3 and weight.shape[0] == cfg.num_experts for weight in (block.gate_up_proj, block.down_proj)
    )
    if not (checks["all_params_loaded"] and checks["experts_wrapped_fused"]):
        return {"checks": checks}

    # A real prompt, since CCA's Conv1d cares about contiguous structure.
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Say hi briefly."}],
        add_generation_prompt=True,
        tokenize=False,
        enable_thinking=False,
    )
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids[:, :MAX_PROMPT_TOKENS].to(ctx.device)
    model.train()
    out = model(input_ids=input_ids, labels=input_ids.clone(), use_cache=False)
    checks["forward_finite"] = (
        out.loss is not None and bool(torch.isfinite(out.loss)) and bool(torch.isfinite(out.logits).all())
    )
    log(f"  Forward: loss {out.loss.item() if out.loss is not None else None}, logits {tuple(out.logits.shape)}")
    if not checks["forward_finite"]:
        return {"checks": checks}

    out.loss.backward()
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    missing = [name for name, p in trainable if p.grad is None]
    nonfinite = [name for name, p in trainable if p.grad is not None and not torch.isfinite(p.grad).all()]
    max_abs_grad = max((p.grad.detach().abs().max().item() for _, p in trainable if p.grad is not None), default=0.0)
    log(
        f"  Backward: {len(trainable) - len(missing)}/{len(trainable)} params got grads, max |grad| {max_abs_grad:.3e}"
    )
    checks["every_param_gets_a_grad"] = not missing
    checks["grads_finite"] = not nonfinite
    checks["grads_nonzero"] = max_abs_grad > 0
    if missing or nonfinite:
        log(f"  Missing grads: {missing[:5]}; non-finite grads: {nonfinite[:5]}")

    # lr=1.0 for visibility: at 1e-4 Adam updates on slow-moving bf16 weights round below bf16's epsilon.
    snapshot = {name: p.detach().clone() for name, p in trainable if name.endswith(SAMPLE_PARAM_SUFFIXES)}
    checks["sampled_params_present"] = all(any(name.endswith(s) for name in snapshot) for s in SAMPLE_PARAM_SUFFIXES)
    torch.optim.AdamW(model.parameters(), lr=1.0).step()
    after = dict(model.named_parameters())
    moved = sum((after[name].detach() - before).abs().max().item() > 0 for name, before in snapshot.items())
    checks["optimizer_step_moves_params"] = bool(snapshot) and moved == len(snapshot)
    log(f"  Optimizer step moved {moved}/{len(snapshot)} sampled params")

    # The patch makes transformers raise up front, over faulting mid-backward in cuDNN on the CCA Conv1d pair.
    try:
        model.gradient_checkpointing_enable()
    except ValueError as exc:
        checks["gradient_checkpointing_refused"] = True
        log(f"  Gradient checkpointing refused: {exc}")
    else:
        checks["gradient_checkpointing_refused"] = False
        log("  gradient_checkpointing_enable() was accepted: apply_zaya_patches did not reach this model")
    return {"checks": checks}


if __name__ == "__main__":
    run()
