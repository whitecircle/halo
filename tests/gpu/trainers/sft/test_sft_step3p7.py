#!/usr/bin/env python
"""Step-3.7 EP=2 SFT smoke + gathered-save round-trip on a tiny random-init composite model.

The shared round trip (``tests/common/ep_sft_roundtrip.py``) on the registry's tiny
``Step3p7ForConditionalGeneration`` (the family ships no text-only CausalLM sibling), whose
``save_pretrained`` writes transformers' HUB layout: no ``language_model`` prefix, per-layer
fused-but-split ``moe.gate_proj`` / ``moe.up_proj`` / ``moe.down_proj`` tensors, ``moe.gate.weight`` +
``moe.router_bias``, ``share_expert.*``, the vendor-namespace vision tower. Then all ranks:

  1. Load it through ``load_distributed_model`` with EP=2 — the lazy loader route, which replays
     the family's hub conversion per key (``_HUB_CONVERSION_KEYS``: the prefix renames, the
     ``moe.*`` → ``mlp.*`` renames, the two-source ``moe.gate_proj + moe.up_proj → gate_up_proj``
     fan-in sliced through both sources, the scoped Step-3.5 vision tower) on a HETEROGENEOUS
     config (per-layer 4-vs-2 attention heads); its bit-exactness against ``from_pretrained`` is
     pinned by ``tests/gpu/parallelism/ep/test_lazy_load_converted.py --family step3p7``.
  2. Run a short DistributedSFTTrainer run over text-only conversations (forward + backward +
     optimizer under FSDP2 + EP, dense+sparse MLP span, full/sliding attention interleave,
     per-layer clamps) with the script's own callback wiring, whose ``moe_balancing: auto``
     resolves to ``bias_update`` here and adopts the native ``e_score_correction_bias`` slot (fp32).
  3. Save via the gathered EP save, which for this family (``_EXPORTS_HUB_NAMESPACE``) runs
     transformers' save-side conversion revert per streamed chunk — the artifact must land in the
     hub layout the serving engines read: the on-disk key set equals the plain ``save_pretrained``
     key set of step 0, the expert halves are bit-exact against the live gathered fused tensor, the
     ``moe.router_bias`` carries a distinctive value written before the save at trained fp32 (a
     dropped key would reload as a zero buffer — so zeros prove nothing), and no module-tree
     spelling survives. Then reload the checkpoint as a PLAIN HF model — its loss must match the EP
     model's post-training loss.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/sft/test_sft_step3p7.py
"""

import torch

from src.distributed.expert_parallel.layers.step3p7 import EPStep3p7MoELayer
from src.models.moe_balancing import NATIVE_BALANCING_BIAS_ADOPTED_ATTR
from tests.common.ep_sft_roundtrip import EPSftRoundTrip
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_STEP3P7_CONFIG
from tests.common.utils import log, safetensors_state_dict

_SPARSE_LAYERS = [i for i, kind in enumerate(TINY_STEP3P7_CONFIG["mlp_layer_types"]) if kind == "sparse"]
_HUB_MOE_KEYS = {
    f"model.layers.{i}.moe.{name}"
    for i in _SPARSE_LAYERS
    for name in ("gate.weight", "router_bias", "gate_proj.weight", "up_proj.weight", "down_proj.weight")
} | {f"model.layers.{i}.share_expert.{proj}_proj.weight" for i in _SPARSE_LAYERS for proj in ("gate", "up", "down")}
_MODULE_TREE_SPELLINGS = ("language_model", ".mlp.experts.", "shared_experts", "multi_modal_projector")


class Step3p7RoundTrip(EPSftRoundTrip):
    family = "step3p7"
    ep_layer_cls = EPStep3p7MoELayer
    num_ep_layers = len(_SPARSE_LAYERS)
    # The family default 151679 is no id of this tokenizer.
    composite_token_ids = {"image_token_id": 2000}
    hub_conversion = True
    native_bias_layer = _SPARSE_LAYERS[0]
    # This family's export contract is the adopted native slot, which only the wiring enables.
    balancing_callbacks = True

    def after_train(self, ep_layers, device):
        # ``auto`` resolves to ``bias_update`` on this family (no aux machinery, native exported slot),
        # so the wiring must have adopted the slot on every EP layer and upcast it to fp32 — the
        # trained-dtype keep-set the fp32-on-disk check rides on.
        checks = {
            "bias_slot_adopted_by_trainer": all(
                getattr(ep, NATIVE_BALANCING_BIAS_ADOPTED_ATTR, False) for ep in ep_layers
            ),
            "bias_slot_fp32_live": all(ep.gate.e_score_correction_bias.dtype == torch.float32 for ep in ep_layers),
        }
        log(f"router bias dtypes: {[str(ep.gate.e_score_correction_bias.dtype) for ep in ep_layers]}")
        return checks

    def after_save(self, ep_layers, base_dir, save_dir):
        # Collective on every rank; the F.linear-convention fused tensors the save splits into the hub's two.
        gathered = [ep.gather_expert_state_dict(device="cpu") for ep in ep_layers]
        written = safetensors_state_dict(save_dir)
        checks = {
            "hub_keyset_matches_plain_save": set(written) == set(safetensors_state_dict(base_dir)),
            "hub_moe_keys_present": set(written) >= _HUB_MOE_KEYS,
            "no_module_tree_spelling": not any(s in key for key in written for s in _MODULE_TREE_SPELLINGS),
        }
        halves_exact, bias_values = [], []
        for i, ep, layer_state in zip(_SPARSE_LAYERS, ep_layers, gathered, strict=True):
            fused = layer_state["experts.gate_up_proj"]  # [E, 2M, H], halves [gate; up]
            half = fused.shape[1] // 2
            halves_exact.append(
                torch.equal(written[f"model.layers.{i}.moe.gate_proj.weight"], fused[:, :half])
                and torch.equal(written[f"model.layers.{i}.moe.up_proj.weight"], fused[:, half:])
                and torch.equal(written[f"model.layers.{i}.moe.down_proj.weight"], layer_state["experts.down_proj"])
            )
            bias = written[f"model.layers.{i}.moe.router_bias"]
            bias_values.append(torch.equal(bias.float(), ep.gate.e_score_correction_bias.float().cpu()))
        checks["expert_halves_bit_exact"] = all(halves_exact)
        # The shared round trip checks the bias's fp32 dtype on disk; this pins it to the hub key.
        checks["router_bias_values_on_disk"] = all(bias_values)
        return checks


main = gpu_test_main(exact_world_size=2, prefix="sft_step3p7")(Step3p7RoundTrip().run)

if __name__ == "__main__":
    main()
