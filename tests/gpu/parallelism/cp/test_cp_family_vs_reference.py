#!/usr/bin/env python
"""Each tiny model of the CP roster under CP=2 against the same model unsplit: the first attention layer, every
attention layer wrapped, the loss, and every gradient.

The roster is ``FAMILIES`` of ``tests/common/equivalence/cp_equivalence.py``, which covers every wrapper class.
Every rank builds the same seeded model. The first attention layer under its wrapper over each rank's sequence
shard, gathered, must match the unwrapped layer over the full sequence, each fed what the decoder hands it: on
a random model the loss and the gradient direction can average away a wrong position-dependent term such as
Mistral4's llama-4 query scale read off chunk-local positions, which this output does not. Every rank then
scores one shared batch over the full sequence as the reference, wraps the model in ``UlyssesModel`` and scores
its sequence shard: the CP loss averaged over the ranks must match the reference loss, and the gradients averaged
over the ranks (the reduction FSDP2 runs) must match the reference gradients in direction and norm. Each comparison
runs at batch 1, where flash takes Q, K and V as views of the all-to-all's receive buffer, and at batch 2, where
they reach it as contiguous copies. A family adds the checks only its wrapper needs:

  * ``glm4_moe_lite``: MLA, whose ``kv_a_proj_with_mqa`` emits a compressed KV split into a normed nope part
    and one rotary key shared by every head; run with LoRA-Q under both ``rope_interleave`` settings, the
    wrapper's flag pinned to the config's, since the loss cannot tell the two RoPE pairings apart on a random
    model, and at GLM-4.7-Flash's head geometry, whose 256-wide latent heads CP runs on FA2 even on Blackwell.
  * ``qwen3_5_moe_text``: a double-width ``q_proj`` (query and a sigmoid output gate), every layer full
    attention, also at Qwen3.5's head geometry, which puts CP on FA4's head_dim-256 kernel on Blackwell. The
    router aux term takes no extra ``cp_size`` factor: its component, the eval loss at a loud coefficient minus
    the loss at zero, matches the reference and is the same on every rank. A hybrid ``layer_types`` config, whose
    GatedDeltaNet layers scan the sequence axis, is refused by name.
  * ``bailing_moe``: remote code with a fused ``query_key_value``, an output projection named ``dense``,
    Q/K RMSNorm before a partial rotary and a softmax scale the module never stores. An sdpa label is
    accepted (the wrapper calls flash itself) while a Lightning-Attention-2 layer, whose file reuses the
    full-attention class names, is refused by name, and the wrapper reads its scale, Q/K norm and output
    projection off the config.

Run: torchrun --nproc_per_node=2 tests/gpu/parallelism/cp/test_cp_family_vs_reference.py --family bailing_moe
"""

import argparse
import copy

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from halo.distributed.context_parallel.config import CPConfigError
from halo.distributed.context_parallel.families import (
    ULYSSES_ATTENTION_MAP,
    BailingMoeV2UlyssesAttention,
    Glm4MoeLiteUlyssesAttention,
)
from halo.distributed.context_parallel.model_rules import reject_unsupported_ulysses_model
from halo.distributed.context_parallel.sequence_chunks import split_sequence_for_cp
from halo.distributed.context_parallel.ulysses_attention import UlyssesAttention
from halo.distributed.context_parallel.ulysses_model import UlyssesModel
from halo.distributed.parallelism_config import ParallelismConfig
from tests.common.distributed import world_mean, world_spread
from tests.common.ep_reference import random_token_batch
from tests.common.equivalence.cp_equivalence import (
    BATCH,
    CP_SIZE,
    FAMILIES,
    SEQ,
    decoder_attention_inputs,
    tiny_model,
)
from tests.common.first_step import parameter_gradient_checks
from tests.common.harness import gpu_test_main
from tests.common.memory import cleanup_memory
from tests.common.numerics import cos_sim
from tests.common.rank_log import log, log_all
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model
from tests.common.tolerances import TOL

# The aux components are differences of two eval forwards on one batch, so the CE term and the CP-vs-reference
# normalization gap cancel, leaving coef * aux ~ 0.2 plus bf16 noise; a dropped or doubled cp_size factor moves
# the CP component 2x or 0.5x, and the floor keeps a vanished component from passing.
QWEN3_5_LOSS_TOLERANCE = 8e-2
AUX_COMPONENT_TOLERANCE = QWEN3_5_LOSS_TOLERANCE / 4
AUX_COMPONENT_MIN = 5 * AUX_COMPONENT_TOLERANCE
# The attention check's copy of the layer runs its matrices at this gain: at the tiny init's scale the softmax is
# near uniform and hides a wrong position-dependent query scale under bf16 noise, and a much higher gain saturates it.
ATTENTION_WEIGHT_GAIN = 8.0
# One bf16 attention layer against itself: CP's flash kernel over the gathered heads against the family's own
# attention over the full sequence, the max error relative to the reference's largest output.
ATTENTION_RTOL = 5e-2
ATTENTION_COSINE_MIN = 0.9995
# At batch 1 the swap to heads returns views of its receive buffer, beyond it contiguous copies.
BATCHES = (1, BATCH)


def causal_lm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Shifted causal-LM cross entropy in fp32, the objective the CP wrapper reproduces per chunk."""
    return F.cross_entropy(
        logits[:, :-1].float().reshape(-1, logits.size(-1)), labels[:, 1:].reshape(-1), ignore_index=-100
    )


def attention_checks(model: nn.Module, cp_config, ids, metrics: dict) -> dict[str, bool]:
    """COLLECTIVE. The first attention layer under its wrapper over this rank's shard of one shared hidden state,
    gathered, against the unwrapped layer over the full sequence, each fed what the decoder hands it."""
    batch = ids.shape[0]
    positions = torch.arange(SEQ, device=ids.device).unsqueeze(0).expand(batch, -1)
    full_inputs = decoder_attention_inputs(model, ids, positions)
    local_inputs = decoder_attention_inputs(
        model, split_sequence_for_cp(ids, cp_config), split_sequence_for_cp(positions, cp_config)
    )
    attention = copy.deepcopy(next(m for m in model.modules() if type(m).__name__ in ULYSSES_ATTENTION_MAP))
    with torch.no_grad():
        for weight in attention.parameters():
            if weight.dim() == 2:
                weight.mul_(ATTENTION_WEIGHT_GAIN)
    wrapper = ULYSSES_ATTENTION_MAP[type(attention).__name__](attention, cp_config.process_group, CP_SIZE)
    hidden = torch.randn(batch, SEQ, model.get_input_embeddings().embedding_dim, device=ids.device, dtype=model.dtype)
    dist.broadcast(hidden, src=0)
    with torch.no_grad():
        reference = attention(**(full_inputs | {"hidden_states": hidden}))[0]
        local = wrapper(
            hidden_states=split_sequence_for_cp(hidden, cp_config),
            position_embeddings=local_inputs["position_embeddings"],
            position_ids=local_inputs.get("position_ids"),
        )[0]
    gathered = [torch.empty_like(local) for _ in range(CP_SIZE)]
    dist.all_gather(gathered, local.contiguous(), group=cp_config.process_group)
    cp_output = torch.cat(gathered, dim=1)
    relative = (cp_output.float() - reference.float()).abs().max().item() / reference.float().abs().max().item()
    cosine = cos_sim(cp_output, reference, label="attention output")
    log(f"  first attention layer: max|Δ|/max|ref|={relative:.3e}  cosine={cosine:.6f}")
    metrics |= {"attention_relative_max_error": relative, "attention_cosine": cosine}
    return {"attention_matches_unwrapped": relative <= ATTENTION_RTOL and cosine >= ATTENTION_COSINE_MIN}


def gradient_checks(reference: dict, cp_model: nn.Module, metrics: dict) -> dict[str, bool]:
    """The rank-averaged CP gradients against the reference ones, named as the reference model names them."""
    cp_grads = {name: param.grad for name, param in cp_model.named_parameters() if param.grad is not None}
    matches, grad_metrics = parameter_gradient_checks(cp_grads, reference)
    metrics |= grad_metrics
    return {"cp_grads_cover_reference": cp_grads.keys() == reference.keys(), "cp_grads_match_reference": matches}


class FamilyChecks:
    """A family's runs and what only its wrapper needs beside the shared comparison: ``variants`` maps the suffix a
    run's checks carry to its config overrides, ``loss_tolerance`` bounds the CP loss against the reference, and the
    hooks check the unwrapped model before CP wraps it and the wrapped one after its backward."""

    variants: dict[str, dict] = {"": {}}
    loss_tolerance = TOL.parallel_vs_baseline_loss_abs

    def before_wrap(self, ctx, model: nn.Module, ids, labels, cp_group) -> dict[str, bool]:
        return {}

    def after_wrap(self, ctx, cp_model: nn.Module, model: nn.Module, ids, labels) -> dict[str, bool]:
        return {}


class Glm4MoeLiteChecks(FamilyChecks):
    family = "glm4_moe_lite"
    variants = {
        "interleaved": {"rope_interleave": True, "q_lora_rank": 48},
        "rotate_half": {"rope_interleave": False, "q_lora_rank": 48},
        "glm_4_7_flash_heads": {
            "rope_interleave": True,
            "q_lora_rank": 48,
            "num_attention_heads": 20,
            "num_key_value_heads": 20,
            "qk_nope_head_dim": 192,
            "qk_rope_head_dim": 64,
            "v_head_dim": 256,
            "kv_lora_rank": 64,
        },
    }
    # MLA adds an all-to-all and a V pad/crop per layer on top of bf16 ordering noise; a wrong projection or
    # interleave lands an order of magnitude off.
    loss_tolerance = 1e-1

    def after_wrap(self, ctx, cp_model, model, ids, labels):
        return {
            "wrapper_rope_interleave_matches_config": all(
                isinstance(module, Glm4MoeLiteUlyssesAttention)
                and module.rope_interleave == model.config.rope_interleave
                for module in cp_model.modules()
                if isinstance(module, UlyssesAttention)
            )
        }


class Qwen3_5MoeChecks(FamilyChecks):
    family = "qwen3_5_moe_text"
    # Loud enough that the aux component clears bf16 noise.
    variants = {
        "": {"router_aux_loss_coef": 0.1},
        "qwen3_5_heads": {
            "router_aux_loss_coef": 0.1,
            "num_attention_heads": 16,
            "num_key_value_heads": 2,
            "head_dim": 256,
        },
    }
    # The sigmoid gate's multiply adds to the noise; a split that drops the gate lands ~10x off.
    loss_tolerance = QWEN3_5_LOSS_TOLERANCE

    @staticmethod
    def aux_component(runner: nn.Module, model: nn.Module, ids, labels) -> float:
        """``runner``'s eval loss at ``model``'s router aux coefficient minus at zero. The stock forward reads the
        coefficient it latched at init; the CP wrapper must read the config per forward, so its run zeroes the
        config alone."""
        coef = model.config.router_aux_loss_coef
        runner.eval()
        with torch.no_grad():
            with_aux = runner(input_ids=ids, labels=labels, use_cache=False, output_router_logits=True).loss.item()
            model.config.router_aux_loss_coef = 0.0
            if runner is model:
                model.router_aux_loss_coef = 0.0
            without_aux = runner(input_ids=ids, labels=labels, use_cache=False, output_router_logits=True).loss.item()
        model.router_aux_loss_coef = model.config.router_aux_loss_coef = coef
        runner.train()
        return with_aux - without_aux

    def before_wrap(self, ctx, model, ids, labels, cp_group):
        self.reference_aux = self.aux_component(model, model, ids, labels)
        return {}

    def after_wrap(self, ctx, cp_model, model, ids, labels):
        local_aux = self.aux_component(cp_model, model, ids, labels)
        cp_aux = world_mean(local_aux)
        log(f"  aux component: reference={self.reference_aux:.6f}  cp_mean={cp_aux:.6f}")
        hybrid = tiny_family_model(TINY_MOE_FAMILIES[self.family]).to(device=ctx.device, dtype=model.dtype)
        hybrid.config._attn_implementation = "flash_attention_2"
        try:
            reject_unsupported_ulysses_model(hybrid, cp_size=CP_SIZE)
            refused = False
        except CPConfigError as error:
            refused = "linear_attention" in str(error) or "GatedDeltaNet" in str(error)
        return {
            "aux_component_nonvacuous": self.reference_aux > AUX_COMPONENT_MIN,
            "cp_aux_component_matches_reference": abs(cp_aux - self.reference_aux) <= AUX_COMPONENT_TOLERANCE,
            # The eval wrapper all-reduces the aux term to the CP-group mean, so a chunk-local term would pass the
            # average above while biasing eval_loss through the one-representative gather.
            "cp_aux_component_rank_uniform": world_spread(local_aux) <= 1e-6,
            # Its default hybrid layer_types: the GatedDeltaNet layers scan the sequence axis.
            "hybrid_layer_types_refused": refused,
        }


class BailingMoeChecks(FamilyChecks):
    family = "bailing_moe"

    def before_wrap(self, ctx, model, ids, labels, cp_group):
        return self._validation_checks(model) | self._seam_checks(model, cp_group)

    @staticmethod
    def _validation_checks(model: nn.Module) -> dict[str, bool]:
        """An sdpa label is accepted; a Lightning-Attention-2 layer, injected by the class name the validator and
        the patcher both match on, is refused by name."""
        checks = {"sdpa_label_accepted": True, "linear_attention_refused_by_name": False}
        label = model.config._attn_implementation
        model.config._attn_implementation = "sdpa"
        try:
            reject_unsupported_ulysses_model(model, cp_size=CP_SIZE)
        except CPConfigError as error:
            log(f"  FAIL: validation rejected an sdpa-labeled Bailing model: {error}")
            checks["sdpa_label_accepted"] = False
        finally:
            model.config._attn_implementation = label
        model.model.layers[0].linear_attn_probe = type("BailingMoeV2LinearAttention", (nn.Module,), {})()
        try:
            reject_unsupported_ulysses_model(model, cp_size=CP_SIZE)
        except CPConfigError as error:
            checks["linear_attention_refused_by_name"] = "BailingMoeV2LinearAttention" in str(error)
        finally:
            del model.model.layers[0].linear_attn_probe
        return checks

    @staticmethod
    def _seam_checks(model: nn.Module, cp_group) -> dict[str, bool]:
        """The wrapper seams the loss cannot isolate on a random model, and the partial rotary the tiny config
        must exercise."""
        attn, config = model.model.layers[0].attention, model.config
        wrapper = BailingMoeV2UlyssesAttention(attn, cp_group, CP_SIZE)
        return {
            "wrapper_scaling_matches_config": wrapper.scaling == config.head_dim**-0.5,
            "wrapper_qk_norm_matches_config": wrapper.use_qk_norm == config.use_qk_norm,
            "wrapper_output_projection_is_dense": getattr(wrapper.original_attention, wrapper.OUTPUT_PROJECTION)
            is attn.dense,
            "rotary_dim_is_partial": 2 * model.model.rotary_emb.inv_freq.numel()
            == int(config.head_dim * config.rope_parameters["partial_rotary_factor"]),
        }


FAMILY_CHECKS = {checks.family: checks for checks in FamilyChecks.__subclasses__()}


def compare_with_reference(
    ctx, family: str, checker: FamilyChecks, overrides: dict, batch: int, metrics: dict
) -> dict[str, bool]:
    """COLLECTIVE. ``family``'s tiny model under CP against itself unsplit on ``batch`` rows, with ``checker``'s
    family checks."""
    model = tiny_model(family, "gpu", ctx.device, **overrides)
    # The seeded init already agrees across ranks; the broadcast keeps a weight mismatch from passing as a
    # wrapper error.
    for tensor in [*model.parameters(), *model.buffers()]:
        if tensor.is_floating_point():
            dist.broadcast(tensor.data, src=0)
    ids, labels = random_token_batch(model.get_input_embeddings().num_embeddings, batch, SEQ, ctx.device)
    cp_config = ParallelismConfig(cp_size=CP_SIZE).cp_config()

    model.train()
    reference_loss = causal_lm_loss(model(input_ids=ids, use_cache=False).logits, labels)
    reference_loss.backward()
    reference_grads = {name: p.grad.detach().clone() for name, p in model.named_parameters() if p.grad is not None}
    model.zero_grad(set_to_none=True)
    attention_layers = sum(type(module).__name__ in ULYSSES_ATTENTION_MAP for module in model.modules())
    checks = attention_checks(model, cp_config, ids, metrics)
    checks |= checker.before_wrap(ctx, model, ids, labels, cp_config.process_group)

    cp_model = UlyssesModel(model, cp_config)
    wrapped = sum(isinstance(module, UlyssesAttention) for module in cp_model.modules())
    log(f"  wrapped attention layers: {wrapped}/{attention_layers}")
    checks["every_attention_layer_wrapped"] = attention_layers > 0 and wrapped == attention_layers
    cp_loss = cp_model(input_ids=ids, labels=labels, use_cache=False).loss
    log_all(f"  cp loss: {cp_loss.item():.6f}")
    cp_mean = world_mean(cp_loss.item())
    delta = abs(cp_mean - reference_loss.item())
    log(f"  reference={reference_loss.item():.6f}  cp_mean={cp_mean:.6f}  |Δ|={delta:.3e}")
    metrics |= {"reference_loss": reference_loss.item(), "cp_mean_loss": cp_mean, "loss_delta": delta}
    checks["cp_loss_matches_reference"] = delta <= checker.loss_tolerance

    cp_loss.backward()
    for param in model.parameters():
        if param.grad is not None:
            dist.all_reduce(param.grad, op=dist.ReduceOp.AVG)
    checks |= gradient_checks(reference_grads, cp_model, metrics)
    checks |= checker.after_wrap(ctx, cp_model, model, ids, labels)

    del cp_model, model
    cleanup_memory()
    return checks


@gpu_test_main(exact_world_size=CP_SIZE, prefix="cp_family_vs_reference")
def run(ctx):
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
    family = parser.parse_args().family
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    checker = FAMILY_CHECKS.get(family, FamilyChecks)()
    for batch in BATCHES:
        for variant, overrides in checker.variants.items():
            log(f"\n--- {family} {variant} batch {batch} ---")
            suffix = f"_{variant}" if variant else ""
            suffix += f"_batch{batch}"
            variant_metrics: dict[str, float] = {}
            variant_checks = compare_with_reference(ctx, family, checker, overrides, batch, variant_metrics)
            checks |= {f"{name}{suffix}": ok for name, ok in variant_checks.items()}
            metrics |= {f"{name}{suffix}": value for name, value in variant_metrics.items()}
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
