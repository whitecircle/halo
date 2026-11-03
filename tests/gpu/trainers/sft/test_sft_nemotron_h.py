#!/usr/bin/env python
"""NemotronH EP=2 SFT smoke + gathered-save round-trip on a tiny random-init model.

The shared round trip (``tests/common/ep_sft_roundtrip.py``) on the registry's tiny NemotronH:

  1. Load it through ``load_distributed_model`` with EP=2 — the family declares
     ``_supports_lazy_loading = False``, so this exercises the ``from_pretrained`` route (including
     the per-expert→fused checkpoint conversion) with the Mamba-2 layers on the Hub kernels.
  2. Run a short DistributedSFTTrainer run (forward + backward + optimizer under FSDP2 + EP).
  3. Save via the gathered EP save and reload the checkpoint as a PLAIN HF model — the experts are
     written per expert, and the trained correction bias survives at fp32.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_nemotron_h.py
"""

from src.distributed.expert_parallel.layers.nemotron_h import EPNemotronHMoELayer
from tests.common.ep_sft_roundtrip import EPSftRoundTrip
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_NEMOTRON_H_CONFIG
from tests.common.utils import safetensors_state_dict

_BLOCK_TYPES = TINY_NEMOTRON_H_CONFIG["layers_block_type"]


class NemotronHRoundTrip(EPSftRoundTrip):
    family = "nemotron_h"
    ep_layer_cls = EPNemotronHMoELayer
    num_ep_layers = _BLOCK_TYPES.count("moe")
    native_bias_layer = _BLOCK_TYPES.index("moe")
    moe_block_attr = "mixer"
    load_liger_kernel = True
    train_liger_kernel = True

    def after_save(self, ep_layers, base_dir, save_dir):
        keys = set(safetensors_state_dict(save_dir))
        layer = f"layers.{self.native_bias_layer}.mixer.experts"
        per_expert = {
            f"{layer}.{e}.{proj}.weight"
            for e in range(TINY_NEMOTRON_H_CONFIG["n_routed_experts"])
            for proj in ("up_proj", "down_proj")
        }
        written = {key.split(".", 1)[1] for key in keys if f"{layer}." in key}
        return {
            "experts_saved_per_expert": per_expert <= written,
            "no_fused_expert_keys": not any(k.endswith(("experts.up_proj", "experts.down_proj")) for k in keys),
        }


main = gpu_test_main(exact_world_size=2, prefix="sft_nemotron_h")(NemotronHRoundTrip().run)

if __name__ == "__main__":
    main()
