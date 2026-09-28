#!/usr/bin/env python
"""DeepSeek-V4 EP=2 SFT smoke + gathered-save round-trip on a tiny random-init model.

The shared round trip (``tests/common/ep_sft_roundtrip.py``) on the registry's tiny DeepSeek-V4, whose
hash layers' tid2eid table the registry fills with distinct experts per token id:

  1. Load it through ``load_distributed_model`` with EP=2 — exercising the eager-force guard (a
     flash_attention request must be overridden to eager), the lazy fused-expert loader, and the
     per-rope-type inv_freq recompute.
  2. Run a short DistributedSFTTrainer run (forward + backward + optimizer under FSDP2 + EP).
  3. Save via the gathered EP save and reload the checkpoint as a PLAIN HF model — the reloaded
     loss must match the EP model's post-training loss, and the hash tid2eid table must survive the
     round-trip.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_deepseek_v4_moe.py
"""

import torch

from src.distributed.expert_parallel.layers.deepseek_v4 import EPDeepseekV4MoELayer
from tests.common.ep_sft_roundtrip import EPSftRoundTrip
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_DSV4_CONFIG
from tests.common.utils import log


class DeepseekV4RoundTrip(EPSftRoundTrip):
    family = "deepseek_v4"
    ep_layer_cls = EPDeepseekV4MoELayer
    num_ep_layers = len(TINY_DSV4_CONFIG["layer_types"])
    attn_implementation = "flash_attention_2"  # the guard must override it to eager
    load_liger_kernel = True  # FLCE-only applier
    # FLCE-only forward returns no logits: with the flag off, TRL's metric path slices None.
    train_liger_kernel = True
    # Flattens the ``_keep_in_fp32_modules_strict`` norms, whose fp32 output otherwise crashes the stock
    # eager forward; those modules upcast internally, and the EP loader does the same.
    reload_in_bf16 = True

    def after_load(self, model, ep_layers):
        attn = getattr(model.config, "_attn_implementation", None)
        log(f"attn_implementation resolved to: {attn}")
        return {"eager_forced": attn == "eager", "hash_layer_present": any(ep.is_hash for ep in ep_layers)}

    def after_reload(self, reloaded, ep_layers):
        # tid2eid must survive train + gathered save (frozen buffer, not trained).
        hash_ep = next(ep for ep in ep_layers if ep.is_hash)
        return {
            "tid2eid_roundtrip": torch.equal(
                reloaded.model.layers[0].mlp.gate.tid2eid.cpu(), hash_ep.gate.tid2eid.cpu()
            )
        }


main = gpu_test_main(exact_world_size=2, prefix="sft_dsv4_moe")(DeepseekV4RoundTrip().run)

if __name__ == "__main__":
    main()
