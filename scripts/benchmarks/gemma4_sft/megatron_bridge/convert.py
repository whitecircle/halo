"""Convert google/gemma-4-26B-A4B-it to a Megatron torch_dist checkpoint via AutoBridge.

Bridge 0.6.2 has no text-only MoE path for Gemma4ForConditionalGeneration (GEMMA4_CONVERSION_MODE=text
only applies to Dense E4B), so the MoE checkpoint converts to Gemma4VLModel (language_model + vision tower).
Same as AutoBridge.import_ckpt except gradient_accumulation_fusion=False (APEX not installed; the non-TE
output layer requires APEX's fused_weight_gradient_mlp_cuda for that flag).
"""

import os

import torch  # noqa: F401  (preload CUDA libs before TE)
from megatron.bridge import AutoBridge

BENCH_ROOT = os.environ["BENCH_ROOT"]  # see paths.env


def main():
    HF = "google/gemma-4-26B-A4B-it"
    OUT = f"{BENCH_ROOT}/megatron/ckpt/gemma4-26b-a4b-it"
    te_act = os.environ.get("MB_TE_ACT", "0") == "1"
    if te_act:  # checkpoint variant whose sharded state includes TE GEGLU op _extra_state
        OUT += "-teact"

    bridge = AutoBridge.from_hf_pretrained(HF, torch_dtype=torch.bfloat16)
    provider = bridge.to_megatron_provider(load_weights=True)
    provider.gradient_accumulation_fusion = False
    if te_act:
        provider.activation_func = torch.nn.functional.gelu  # -> te.ops.GEGLU (tanh approx, == gelu_pytorch_tanh)
        provider.use_te_activation_func = True
    provider.finalize()
    model = provider.provide_distributed_model(wrap_with_ddp=False, use_cpu_initialization=True)
    bridge.save_megatron_model(model, OUT, hf_tokenizer_path=HF)
    print("DONE", OUT)


if __name__ == "__main__":
    main()
