#!/usr/bin/env python
"""GPT-OSS EP with DeepSeek-V3 bias-update router balancing, on a tiny model.

Adoption turns `router.bias` into a persistent buffer under the same key: frozen out of gradient training, still in the
`state_dict()` engines load. `_route_with_bias` reproduces the stock router's selection and gate weights, so trainer
and server compute the same route. Counting through a live DeepEP buffer is ``test_ep_gc_bias_balancing.py``'s job.
"""

from types import SimpleNamespace

import torch
from transformers import GptOssConfig, GptOssForCausalLM, TrainerState

from halo.callbacks.balancing import BalancingBiasCallback, apply_balancing_mode
from halo.distributed.expert_parallel.families.glm4 import EPGlm4MoELayer
from halo.distributed.expert_parallel.families.gpt_oss import EPGptOssMoELayer
from halo.distributed.expert_parallel.families.lfm2 import EPLfm2MoELayer
from halo.distributed.expert_parallel.families.mistral4 import EPMistral4MoELayer
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from halo.models.moe.balancing import iter_balancing_routers, resolve_balancing_mode
from tests.common.harness import gpu_test_main
from tests.common.rank_log import log

NUM_EXPERTS = 8
TOP_K = 2
HIDDEN = 64
SEQ_LEN = 16
BATCH = 1
SEED = 17


def build_tiny_gptoss(device):
    config = GptOssConfig(
        vocab_size=256,
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        num_local_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        max_position_embeddings=128,
        layer_types=["full_attention", "full_attention"],
        sliding_window=0,
        attn_implementation="eager",
    )
    torch.manual_seed(SEED)
    model = GptOssForCausalLM(config).to(device)
    return model


def check(checks, name, cond, detail=""):
    """Log a check and AND it into ``checks``: a repeated name must hold every time."""
    ok = bool(cond)
    log(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    checks[name] = ok and checks.get(name, True)


def check_route_injection(checks, cls, score_fn, device, **attrs):
    """A family's ``route_tokens_to_experts`` injection on an ``object.__new__`` stub, normalization off so the gate
    weights equal the gathered unbiased scores."""
    name = cls.__name__
    layer = cls.__new__(cls)
    layer.training = True
    layer.balancing_biases = None
    # __init__ is bypassed, so mirror the routing-replay defaults the selection helpers read.
    layer._forced_topk_indices = None
    layer._forced_cursor = 0
    layer._capture_routing = False
    layer._captured_routing_chunks = []
    layer.num_experts = NUM_EXPERTS
    layer.top_k = TOP_K
    layer.norm_topk_prob = False
    layer.routed_scaling_factor = 1.0
    for k, v in attrs.items():
        setattr(layer, k, v)

    T = 32
    torch.manual_seed(SEED)
    logits = torch.randn(T, NUM_EXPERTS, device=device)
    scores = score_fn(logits)  # Gate-space scores the weights are gathered from.

    idx_s, _ = layer.route_tokens_to_experts(logits)
    layer.balancing_biases = torch.zeros(NUM_EXPERTS, device=device)
    idx0, _ = layer.route_tokens_to_experts(logits)
    check(checks, f"{name} zero-bias selection == stock", torch.equal(idx0.sort(-1).values, idx_s.sort(-1).values))

    # Forces an otherwise-unselected expert in; gate weights stay unbiased.
    unsel = ~torch.isin(torch.arange(NUM_EXPERTS, device=device), idx0.flatten())
    free = int(unsel.nonzero()[0].item()) if unsel.any() else 0
    big = torch.zeros(NUM_EXPERTS, device=device)
    big[free] = 1e3
    layer.balancing_biases = big
    idxb, wb = layer.route_tokens_to_experts(logits)
    check(checks, f"{name} large +bias forces expert {free}", bool((idxb == free).any(-1).all()))
    check(checks, f"{name} gate weights bias-free", (wb - scores.gather(1, idxb)).abs().max().item() < 1e-4)


def run(ctx) -> dict:
    checks: dict[str, bool] = {}
    device = ctx.device
    torch.cuda.set_device(device)

    model = build_tiny_gptoss(device)
    pc = ParallelismConfig(ep_size=ctx.world_size)
    model = apply_ep_layers(model, pc.ep_config())

    ep_layers = [m for m in model.modules() if isinstance(m, EPGptOssMoELayer)]
    check(checks, "EP layers patched", len(ep_layers) > 0, f"{len(ep_layers)} EPGptOssMoELayer(s)")

    # Opt-in preserved.
    pre_mode = resolve_balancing_mode("auto", model, is_moe=True)
    check(checks, "auto resolves to aux_loss before enable", pre_mode == "aux_loss", f"got {pre_mode!r}")
    check(checks, "no balancing routers before enable", len(list(iter_balancing_routers(model))) == 0)

    # The EP bias path bypasses the router module HF's OutputRecorder hooks, so output_router_logits left on makes
    # load_balancing_loss_func crash on an empty tuple. apply_balancing_mode must force it off.
    model.config.output_router_logits = True
    apply_balancing_mode(model, "bias_update")
    check(
        checks,
        "bias_update forces output_router_logits=False",
        model.config.output_router_logits is False,
        f"got {model.config.output_router_logits!r}",
    )
    for layer in ep_layers:
        bias = getattr(layer, "balancing_biases", None)
        check(checks, "balancing_biases attached", bias is not None)
        check(checks, "bias IS the native router.bias", bias is layer.router.bias)
        check(checks, "bias shape == num_experts", bias.numel() == NUM_EXPERTS, f"{bias.numel()}")
        check(checks, "bias is fp32", bias.dtype == torch.float32, str(bias.dtype))
        check(checks, "counter slot present (None)", getattr(layer, "expert_load_counter", "missing") is None)

    detected = list(iter_balancing_routers(model))
    check(checks, "callback detects all routers", len(detected) == len(ep_layers), f"{len(detected)}/{len(ep_layers)}")
    post_mode = resolve_balancing_mode("auto", model, is_moe=True)
    check(checks, "auto resolves to bias_update after enable", post_mode == "bias_update", f"got {post_mode!r}")

    l0 = ep_layers[0]
    check(checks, "router.bias not a parameter post-adoption", "bias" not in dict(l0.router.named_parameters()))
    check(checks, "router.bias is a buffer post-adoption", "bias" in dict(l0.router.named_buffers()))
    check(checks, "router.bias stays in state_dict (export path)", "bias" in l0.router.state_dict())

    # Both read the same adopted router.bias through the same F.linear, so selection and gate weights coincide exactly:
    # the train/serve parity this adoption exists for.
    def check_matches_stock(tag: str):
        logits, weights, experts = l0._route_with_bias(x)
        s_logits, s_scores, s_idx = l0.router(x)  # Stock 3-tuple (logits, softmax-over-topk, indices).
        same_sel = torch.equal(experts.sort(-1).values, s_idx.sort(-1).values.to(experts.dtype))
        check(checks, f"{tag}: selection == stock selection", same_sel)
        eo = experts.argsort(-1)
        so = s_idx.argsort(-1)
        wdiff = (torch.gather(weights, -1, eo) - torch.gather(s_scores, -1, so)).abs().max().item()
        check(checks, f"{tag}: gate weights == stock", wdiff < 1e-4, f"maxdiff={wdiff:.2e}")
        check(
            checks,
            f"{tag}: gate weights sum to 1",
            torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)), atol=1e-4),
        )
        return experts

    model.train()
    torch.manual_seed(SEED + ctx.rank)
    x = torch.randn(BATCH * SEQ_LEN, HIDDEN, device=device, dtype=l0.router.weight.dtype)
    l0.balancing_biases.zero_()  # Writes into router.bias, which both paths read.
    experts = check_matches_stock("zero bias")

    # The stock router, what an engine serves, steers identically: the bias is model state.
    free = (
        int((~torch.isin(torch.arange(NUM_EXPERTS, device=device), experts.flatten())).nonzero()[0].item())
        if (~torch.isin(torch.arange(NUM_EXPERTS, device=device), experts.flatten())).any()
        else 0
    )
    l0.balancing_biases.zero_()
    l0.balancing_biases[free] = 1e3
    e_b = check_matches_stock("steering bias")
    check(checks, "large +bias forces target expert", bool((e_b == free).any(-1).all()), f"expert {free}")
    l0.balancing_biases.zero_()

    gamma = 0.1
    forced = torch.zeros(NUM_EXPERTS, device=device, dtype=torch.float32)
    forced[0] = 100.0  # Expert 0 overloaded: bias[0] should go down, others up.
    for layer in ep_layers:
        layer.balancing_biases.zero_()
        layer.expert_load_counter = forced.clone()
    BalancingBiasCallback(update_rate=gamma).on_step_end(None, TrainerState(global_step=1), None, model=model)
    b0 = ep_layers[0].balancing_biases
    check(checks, "overloaded expert bias decreased", b0[0].item() < 0, f"bias[0]={b0[0].item():.3f}")
    check(checks, "underloaded expert bias increased", (b0[1:] > 0).all().item(), f"bias[1:]={b0[1:].tolist()}")
    check(checks, "update magnitude == gamma", torch.allclose(b0.abs(), torch.full_like(b0, gamma), atol=1e-6))
    counter_after = ep_layers[0].expert_load_counter
    check(checks, "counter zeroed after step", float(counter_after.abs().sum().item()) == 0.0)

    # Other families' injection needs only plain attributes, so it runs on lightweight stubs.
    log("\n--- route_tokens_to_experts bias injection (GLM4 / LFM2 / Mistral4) ---")
    check_route_injection(
        checks,
        EPGlm4MoELayer,
        lambda l: l.sigmoid(),
        device,
        gate=SimpleNamespace(),
        n_group=1,
        topk_group=1,
    )
    check_route_injection(checks, EPLfm2MoELayer, lambda l: l.sigmoid(), device, use_expert_bias=False)
    check_route_injection(
        checks,
        EPMistral4MoELayer,
        lambda l: l.softmax(dim=-1),
        device,
        n_group=1,
        topk_group=1,
    )

    return {"checks": checks}


main = gpu_test_main(min_world_size=1, prefix="gptoss_bias_balancing", partial_state=False)(run)

if __name__ == "__main__":
    main()
