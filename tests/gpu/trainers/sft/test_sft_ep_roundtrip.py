#!/usr/bin/env python
"""EP=2 SFT smoke plus gathered-save round trip of the families whose save or load path is their own.

The reloaded plain HF model's loss must match the EP model's post-training loss (``tests/common/ep_sft_roundtrip.py``).
"""

import argparse
import os

import torch

from halo.distributed.expert_parallel.families.cohere2_moe import EPCohere2MoELayer
from halo.distributed.expert_parallel.families.deepseek_v4 import EPDeepseekV4MoELayer
from halo.distributed.expert_parallel.families.glm5_next import EPGlm5NextMoELayer
from halo.distributed.expert_parallel.families.nemotron_h import EPNemotronHMoELayer
from halo.distributed.expert_parallel.families.qwen3_next import EPQwen3NextMoELayer
from halo.distributed.expert_parallel.families.qwen4_exp import EPQwen4ExpMoELayer
from halo.distributed.expert_parallel.families.step3p7 import EPStep3p7MoELayer
from halo.model_loading.parallel.lazy_plan import is_lazy_loadable
from halo.models.external_tables import EXTERNAL_TABLES_STAMP, named_external_tables
from halo.models.moe.balancing import NATIVE_BALANCING_BIAS_ADOPTED_ATTR
from tests.common.checkpoint_io import safetensors_state_dict
from tests.common.ep_sft_roundtrip import EPSftRoundTrip
from tests.common.harness import gpu_test_main
from tests.common.models import (
    TINY_COHERE2_MOE_CONFIG,
    TINY_DSV4_CONFIG,
    TINY_GLM5_CONFIG,
    TINY_NEMOTRON_H_CONFIG,
    TINY_QWEN3_NEXT_CONFIG,
    TINY_QWEN4_EXP_CONFIG,
    TINY_STEP3P7_CONFIG,
)
from tests.common.rank_log import log

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
    # The Liger spec patches Cohere2MoeMLP, under EP the shared expert the wrapper adopts: this exercises the fused
    # GLU.
    load_liger_kernel = True

    def after_load(self, model, ep_layers):
        return {"average_combination_scaled": all(ep._output_scale == 0.5 for ep in ep_layers)}

    def after_reload(self, reloaded, ep_layers):
        return {"logit_scale_survives_save": reloaded.config.logit_scale == TINY_COHERE2_MOE_CONFIG["logit_scale"]}


class DeepseekV4RoundTrip(EPSftRoundTrip):
    family = "deepseek_v4"
    ep_layer_cls = EPDeepseekV4MoELayer
    num_ep_layers = len(TINY_DSV4_CONFIG["layer_types"])
    attn_implementation = "flash_attention_2"  # The guard must override it to eager.
    load_liger_kernel = True  # FLCE-only applier.
    # FLCE-only forward returns no logits: with the flag off, TRL's metric path slices None.
    train_liger_kernel = True
    # Flattens the ``_keep_in_fp32_modules_strict`` norms, whose fp32 output otherwise crashes the stock eager forward.
    reload_in_bf16 = True

    def after_load(self, model, ep_layers):
        attn = getattr(model.config, "_attn_implementation", None)
        log(f"attn_implementation resolved to: {attn}")
        return {"eager_forced": attn == "eager", "hash_layer_present": any(ep.is_hash for ep in ep_layers)}

    def after_reload(self, reloaded, ep_layers):
        # tid2eid must survive train + gathered save (frozen buffer).
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


class _NemotronHChecks:
    ep_layer_cls = EPNemotronHMoELayer
    num_ep_layers = TINY_NEMOTRON_H_CONFIG["layers_block_type"].count("moe")
    hub_conversion = True
    native_bias_layer = TINY_NEMOTRON_H_CONFIG["layers_block_type"].index("moe")
    # ``auto`` adopts the exported slot, which only the wiring enables.
    balancing_callbacks = True
    # The release stores its bias in fp32, where a bf16 cast would round it by more than its spread.
    fp32_pins = True
    # Where the checkpoint keeps the tower.
    tower: str

    def after_load(self, model, ep_layers):
        biases = [ep.gate.e_score_correction_bias.detach() for ep in ep_layers]
        return {
            "bias_loaded_in_fp32": all(bias.dtype == torch.float32 for bias in biases),
            "bias_finer_than_bf16": all(not torch.equal(bias, bias.bfloat16().float()) for bias in biases),
        }

    def after_train(self, ep_layers, device):
        return {
            "bias_slot_adopted_by_trainer": all(
                getattr(ep, NATIVE_BALANCING_BIAS_ADOPTED_ATTR, False) for ep in ep_layers
            ),
            "bias_slot_fp32_after_training": all(
                ep.gate.e_score_correction_bias.dtype == torch.float32 for ep in ep_layers
            ),
        }

    def after_save(self, ep_layers, base_dir, save_dir):
        # COLLECTIVE
        gathered = [ep.gather_expert_state_dict(device="cpu") for ep in ep_layers]
        written = safetensors_state_dict(save_dir)
        moe_layers = [index for index, kind in enumerate(TINY_NEMOTRON_H_CONFIG["layers_block_type"]) if kind == "moe"]
        return {
            "hub_keyset_matches_plain_save": set(written) == set(safetensors_state_dict(base_dir)),
            "one_expert_tensor_per_projection": all(
                f"{self.tower}layers.{index}.mixer.experts.0.up_proj.weight" in written for index in moe_layers
            ),
            "experts_bit_exact": all(
                torch.equal(written[f"{self.tower}layers.{index}.mixer.{key}"], tensor)
                for index, layer_state in zip(moe_layers, gathered, strict=True)
                for key, tensor in layer_state.items()
            ),
        }


class NemotronHRoundTrip(_NemotronHChecks, EPSftRoundTrip):
    family = "nemotron_h"
    tower = "backbone."


class NemotronHCompositeRoundTrip(_NemotronHChecks, EPSftRoundTrip):
    """The save under the wrap that groups the embedding at the composite's root, on text rows: every rank skips the
    tower alike."""

    family = "nemotron_h_omni"
    tower = "language_model.backbone."

    def after_save(self, ep_layers, base_dir, save_dir):
        checks = super().after_save(ep_layers, base_dir, save_dir)
        written = safetensors_state_dict(save_dir)
        checks["vision_norm_in_the_release_spelling"] = "vision_projector.vision_final_layernorm.weight" in written
        return checks


class Qwen3NextRoundTrip(EPSftRoundTrip):
    family = "qwen3_next"
    ep_layer_cls = EPQwen3NextMoELayer
    num_ep_layers = TINY_QWEN3_NEXT_CONFIG["num_hidden_layers"]

    def after_save(self, ep_layers, base_dir, save_dir):
        # COLLECTIVE
        gathered = [ep.gather_expert_state_dict(device="cpu") for ep in ep_layers]
        written = safetensors_state_dict(save_dir)
        experts_exact = all(
            torch.equal(written[f"model.layers.{i}.mlp.{key}"], tensor)
            for i, layer_state in enumerate(gathered)
            for key, tensor in layer_state.items()
        )
        return {
            "lazy_loading_admitted": is_lazy_loadable(base_dir) is True,
            "hub_keyset_matches_plain_save": set(written) == set(safetensors_state_dict(base_dir)),
            "per_expert_hub_keys": all(
                f"model.layers.{i}.mlp.experts.0.gate_proj.weight" in written for i in range(self.num_ep_layers)
            ),
            "experts_bit_exact": experts_exact,
        }


class Qwen4ExpRoundTrip(EPSftRoundTrip):
    family = "qwen4_exp_text"
    ep_layer_cls = EPQwen4ExpMoELayer
    num_ep_layers = TINY_QWEN4_EXP_CONFIG["num_hidden_layers"]

    def after_load(self, model, ep_layers):
        tables = [table for _, table in named_external_tables(model)]
        return {
            "ngram_table_attached": len(tables) == 1 and tables[0].is_attached,
            "ngram_table_outside_parameters": not any(
                "ngram_embedding" in name for name, _ in model.named_parameters()
            ),
        }

    def after_save(self, ep_layers, base_dir, save_dir):
        written, base = safetensors_state_dict(save_dir), safetensors_state_dict(base_dir)
        table_keys = {key for key in base if ".ngram_embedding.shard_" in key}
        with open(os.path.join(save_dir, "config.json")) as handle:
            config = handle.read()
        return {
            "ngram_table_keys_written": bool(table_keys)
            and {key for key in written if ".ngram_embedding." in key} == table_keys,
            "ngram_table_bit_exact": all(torch.equal(written[key], base[key]) for key in table_keys),
            "fused_hub_expert_banks": all(
                f"model.layers.{i}.mlp.experts.gate_up_proj" in written for i in range(self.num_ep_layers)
            ),
            "no_run_stamp_in_config": EXTERNAL_TABLES_STAMP not in config,
        }


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
        # ``auto`` resolves to ``bias_update`` here, so the wiring must have adopted the slot on every EP layer at
        # fp32, which the fp32-on-disk check rides on.
        checks = {
            "bias_slot_adopted_by_trainer": all(
                getattr(ep, NATIVE_BALANCING_BIAS_ADOPTED_ATTR, False) for ep in ep_layers
            ),
            "bias_slot_fp32_live": all(ep.gate.e_score_correction_bias.dtype == torch.float32 for ep in ep_layers),
        }
        log(f"router bias dtypes: {[str(ep.gate.e_score_correction_bias.dtype) for ep in ep_layers]}")
        return checks

    def after_save(self, ep_layers, base_dir, save_dir):
        # COLLECTIVE; the F.linear-convention fused tensors the save splits into the hub's two.
        gathered = [ep.gather_expert_state_dict(device="cpu") for ep in ep_layers]
        written = safetensors_state_dict(save_dir)
        checks = {
            "hub_keyset_matches_plain_save": set(written) == set(safetensors_state_dict(base_dir)),
            "hub_moe_keys_present": set(written) >= _STEP3P7_HUB_MOE_KEYS,
            "no_module_tree_spelling": not any(s in key for key in written for s in _MODULE_TREE_SPELLINGS),
        }
        halves_exact, bias_values = [], []
        for i, ep, layer_state in zip(_STEP3P7_SPARSE_LAYERS, ep_layers, gathered, strict=True):
            fused = layer_state["experts.gate_up_proj"]  # [E, 2M, H], halves [gate; up].
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
