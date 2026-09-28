#!/usr/bin/env python
"""Cohere2 MoE EP=2 SFT smoke + gathered-save round-trip on a tiny random-init model.

The shared round trip (``tests/common/ep_sft_roundtrip.py``) on the registry's tiny Cohere2 MoE:

  1. Load it through ``load_distributed_model`` with EP=2 — the family declares
     ``_supports_lazy_loading = False``, so this exercises the ``from_pretrained`` route (including
     the per-expert→fused checkpoint conversion) plus the generic inv_freq recompute.
  2. Run a short DistributedSFTTrainer run (forward + backward + optimizer under FSDP2 + EP, tied
     embeddings, parallel-residual blocks, interleaved sliding/NoPE attention).
  3. Save via the gathered EP save and reload the checkpoint as a PLAIN HF model — the reloaded
     loss must match the EP model's post-training loss, and ``logit_scale`` must survive the save.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_cohere2_moe.py
"""

from src.distributed.expert_parallel.layers.cohere2_moe import EPCohere2MoELayer
from tests.common.ep_sft_roundtrip import EPSftRoundTrip
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_COHERE2_MOE_CONFIG


class Cohere2MoeRoundTrip(EPSftRoundTrip):
    family = "cohere2_moe"
    ep_layer_cls = EPCohere2MoELayer
    num_ep_layers = TINY_COHERE2_MOE_CONFIG["num_hidden_layers"]
    # The family's Liger spec patches Cohere2MoeMLP, which under EP is the SHARED expert the wrapper
    # adopts unchanged — so this run is the one that exercises the fused GLU surviving EP.
    load_liger_kernel = True

    def after_load(self, model, ep_layers):
        return {"average_combination_scaled": all(ep._output_scale == 0.5 for ep in ep_layers)}

    def after_reload(self, reloaded, ep_layers):
        return {"logit_scale_survives_save": reloaded.config.logit_scale == TINY_COHERE2_MOE_CONFIG["logit_scale"]}


main = gpu_test_main(exact_world_size=2, prefix="sft_cohere2_moe")(Cohere2MoeRoundTrip().run)

if __name__ == "__main__":
    main()
