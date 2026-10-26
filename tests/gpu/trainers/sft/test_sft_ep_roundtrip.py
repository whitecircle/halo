#!/usr/bin/env python
"""EP=2 SFT smoke + gathered-save round trip of the families whose save or load path is their own.

The shared round trip (``tests/common/ep_sft_roundtrip.py``) on the registry's tiny random-init model of
each ``--family``: load through ``load_distributed_model`` at EP=2, run a short ``DistributedSFTTrainer``
run (forward + backward + optimizer under FSDP2 + EP), save via the gathered EP save and reload the
checkpoint as a PLAIN HF model, whose loss must match the EP model's post-training loss. Per family:

  * ``cohere2_moe``: ``supports_lazy_loading=False``, so the ``from_pretrained`` route (with the
    per-expert→fused checkpoint conversion) plus the generic inv_freq recompute; tied embeddings,
    parallel-residual blocks, interleaved sliding/NoPE attention; ``logit_scale`` survives the save.
  * ``deepseek_v4``: the eager-force guard (a flash_attention request overridden to eager), the lazy
    fused-expert loader, the per-rope-type inv_freq recompute; the hash layers' tid2eid table, which the
    registry fills with distinct experts per token id, survives the round trip.
  * ``glm5_next``: the tiny ``Glm5NextForConditionalGeneration`` (no text-only CausalLM sibling), whose
    ``save_pretrained`` writes the hub layout (per-expert ``experts.{i}.{gate,up,down}_proj``, the
    vendor-namespace KDA/hyper-connection keys, split ``q/k/v_conv1d``); the lazy loader plans its keys on
    the family's hub conversion (bit-exact against ``from_pretrained`` in
    ``tests/gpu/parallelism/ep/test_lazy_load_converted.py``), trains over text-only conversations
    (dense+sparse MLP span, KDA/DSA interleave), and the fp32 ``e_score_correction_bias``, set to a
    distinctive value after training, lands on disk at fp32 and survives.
  * ``step3p7``: the tiny ``Step3p7ForConditionalGeneration``, saved in transformers' hub layout (no
    ``language_model`` prefix, per-layer fused-but-split ``moe.{gate,up,down}_proj``, ``moe.gate.weight`` +
    ``moe.router_bias``, ``share_expert.*``, the vendor-namespace vision tower) on a heterogeneous config
    (per-layer 4-vs-2 attention heads); the lazy loader fans ``moe.gate_proj + moe.up_proj`` into
    ``gate_up_proj``. The script's own callback wiring resolves ``moe_balancing: auto`` to ``bias_update``,
    adopting the native ``e_score_correction_bias`` slot at fp32. The gathered save reverts the whole vendor
    namespace per streamed chunk: its key set equals step 0's plain ``save_pretrained`` key set, the expert
    halves are bit-exact against the live gathered fused tensor, ``moe.router_bias`` carries a distinctive
    value written before the save at trained fp32 (a dropped key would reload as a zero buffer), and no
    module-tree spelling survives.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 tests/gpu/trainers/sft/test_sft_ep_roundtrip.py --family step3p7
"""

import argparse

import torch

from src.distributed.expert_parallel.families.cohere2_moe import EPCohere2MoELayer
from src.distributed.expert_parallel.families.deepseek_v4 import EPDeepseekV4MoELayer
from src.distributed.expert_parallel.families.glm5_next import EPGlm5NextMoELayer
from src.distributed.expert_parallel.families.step3p7 import EPStep3p7MoELayer
from src.models.moe.balancing import NATIVE_BALANCING_BIAS_ADOPTED_ATTR
from tests.common.ep_sft_roundtrip import EPSftRoundTrip
from tests.common.harness import gpu_test_main
from tests.common.models import TINY_COHERE2_MOE_CONFIG, TINY_DSV4_CONFIG, TINY_GLM5_CONFIG, TINY_STEP3P7_CONFIG
from tests.common.utils import log, safetensors_state_dict

_STEP3P7_SPARSE_LAYERS = [i for i, kind in enumerate(TINY_STEP3P7_CONFIG["mlp_layer_types"]) if kind == "sparse"]
_STEP3P7_HUB_MOE_KEYS = {
    f"model.layers.{i}.moe.{name}"
    for i in _STEP3P7_SPARSE_LAYERS
    for name in ("gate.weight", "router_bias", "gate_proj.weight", "up_proj.weight", "down_proj.weight")
} | {
    f"model.layers.{i}.share_expert.{proj}_proj.weight"
    for i in _STEP3P7_SPARSE_LAYERS
    for proj in ("gate", "up", "down")
}
_MODULE_TREE_SPELLINGS = ("language_model", ".mlp.experts.", "shared_experts", "multi_modal_projector")


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


class Step3p7RoundTrip(EPSftRoundTrip):
    family = "step3p7"
    ep_layer_cls = EPStep3p7MoELayer
    num_ep_layers = len(_STEP3P7_SPARSE_LAYERS)
    # The family default 151679 is no id of this tokenizer.
    composite_token_ids = {"image_token_id": 2000}
    hub_conversion = True
    native_bias_layer = _STEP3P7_SPARSE_LAYERS[0]
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
            "hub_moe_keys_present": set(written) >= _STEP3P7_HUB_MOE_KEYS,
            "no_module_tree_spelling": not any(s in key for key in written for s in _MODULE_TREE_SPELLINGS),
        }
        halves_exact, bias_values = [], []
        for i, ep, layer_state in zip(_STEP3P7_SPARSE_LAYERS, ep_layers, gathered, strict=True):
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


ROUND_TRIPS = {round_trip.family: round_trip for round_trip in EPSftRoundTrip.__subclasses__()}


def run(ctx) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=sorted(ROUND_TRIPS), required=True)
    return ROUND_TRIPS[parser.parse_args().family]().run(ctx)


main = gpu_test_main(exact_world_size=2, prefix="sft_ep_roundtrip")(run)

if __name__ == "__main__":
    main()
