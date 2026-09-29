#!/usr/bin/env python
"""GLM-5 Next EP=2 SFT smoke + gathered-save round-trip on a tiny random-init composite model.

The shared round trip (``tests/common/ep_sft_roundtrip.py``) on the registry's tiny
``Glm5NextForConditionalGeneration`` (the family ships no text-only CausalLM sibling), whose
``save_pretrained`` reverts to the HUB layout: per-expert ``experts.{i}.{gate,up,down}_proj`` tensors
plus the vendor-namespace KDA/hyper-connection keys (``hc_attn_fn``, ``self_attn.f_a_proj``, split
``q/k/v_conv1d``). Then all ranks:

  1. Load it through ``load_distributed_model`` with EP=2 — the lazy loader route, which replays
     the family's hub conversion per key (``_HUB_CONVERSION_KEYS``: the vendor-namespace renames,
     the three-source ``q/k/v_conv1d → conv1d`` fan-in) and fuses the per-expert projections
     locally; its bit-exactness against ``from_pretrained`` is pinned by
     ``tests/gpu/parallelism/ep/test_lazy_load_converted.py``.
  2. Run a short DistributedSFTTrainer run over text-only conversations (forward + backward +
     optimizer under FSDP2 + EP, dense+sparse MLP span, KDA/DSA attention interleave).
  3. Save via the gathered EP save and reload the checkpoint as a PLAIN HF model — the reloaded
     loss must match the EP model's post-training loss, and the fp32 ``e_score_correction_bias``
     buffer, set to a distinctive value after training, must land on disk at fp32 and survive the
     round-trip.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_glm5_next.py
"""

from src.distributed.expert_parallel.layers.glm5_next import EPGlm5NextMoELayer
from tests.common.ep_sft_roundtrip import EPSftRoundTrip
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_GLM5_CONFIG


class Glm5NextRoundTrip(EPSftRoundTrip):
    family = "glm5_next"
    ep_layer_cls = EPGlm5NextMoELayer
    num_ep_layers = TINY_GLM5_CONFIG["mlp_layer_types"].count("sparse")
    composite_token_ids = {
        "image_token_id": 2000,
        "video_token_id": 2001,
        "image_start_token_id": 2002,
        "image_end_token_id": 2003,
        "video_start_token_id": 2004,
        "video_end_token_id": 2005,
    }
    hub_conversion = True
    native_bias_layer = TINY_GLM5_CONFIG["mlp_layer_types"].index("sparse")


main = gpu_test_main(exact_world_size=2, prefix="sft_glm5_next")(Glm5NextRoundTrip().run)

if __name__ == "__main__":
    main()
