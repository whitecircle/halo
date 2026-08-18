#!/usr/bin/env python
"""``MoEMetricsCallback`` must count the routing the model performed, not ``topk(router_logits)``.

DeepSeek-style routers pick over ``score(logits) + e_score_correction_bias`` inside a group mask
(``n_group`` / ``topk_group``), then publish the RAW pre-bias, pre-mask logits as ``router_logits``.
Re-ranking those describes a load distribution the model never routed: a balancing bias flips
near-ties onto a different expert, and group-limiting excludes experts holding the largest raw logit
in the layer. Under those routers the ``moe/*`` metrics are the only signal a run has about expert
collapse, so counting the wrong selection is a silent mis-report, not a rounding error.

The vehicle is transformers' own ``Glm5NextTextTopkRouter`` — bias and group limit in one forward —
so these tests track the family's real routing rather than a re-implementation of it. The families
whose routers carry the same two mechanisms (GLM-4/5, Laguna, DeepSeek-V4, Inkling, Step-3.7,
Bailing, LFM-2, GPT-OSS) all reach the callback through the same seam.

Run::

    python tests/cpu/callbacks/test_moe_metrics_router_selection.py
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers.models.glm5_next.configuration_glm5_next import Glm5NextTextConfig
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextTopkRouter
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssForCausalLM, GptOssTopKRouter
from transformers.utils.output_capturing import OutputRecorder

import src.callbacks.moe_metrics as moe_metrics
from src.callbacks.moe_metrics import MoEMetricsCallback, _hooked_routers
from tests.common.models import TINY_GLM5_CONFIG, TINY_GPTOSS_CONFIG

NUM_EXPERTS = TINY_GLM5_CONFIG["n_routed_experts"]
TOP_K = TINY_GLM5_CONFIG["num_experts_per_tok"]
HIDDEN = TINY_GLM5_CONFIG["hidden_size"]


def _router_config(**overrides) -> Glm5NextTextConfig:
    return Glm5NextTextConfig(**{**TINY_GLM5_CONFIG, **overrides})


class _RouterModel(nn.Module):
    """A model whose one routing module is a real HF router, declared the way transformers declares it.

    ``_can_record_outputs`` is what tells the toolkit (and transformers' own capture) which module
    produces ``router_logits``; the callback discovers its routers through exactly that declaration.
    """

    def __init__(self, router: nn.Module, model_type: str = "glm5_next"):
        super().__init__()
        self.gate = router
        self.config = SimpleNamespace(output_router_logits=True, model_type=model_type)
        self._can_record_outputs = {"router_logits": OutputRecorder(type(router), index=0)}


def _one_hot_router(logits: list[float], bias: list[float] | None = None, **config_overrides):
    """A router whose logits for the probe token are exactly ``logits``.

    ``F.linear`` against a one-hot hidden state selects column 0 of the weight, so writing the desired
    logits there makes the routing decision fully determined and readable in the assertions.
    """
    config = _router_config(**config_overrides)
    router = Glm5NextTextTopkRouter(config)
    with torch.no_grad():
        router.weight.zero_()
        router.weight[:, 0] = torch.tensor(logits)
        router.e_score_correction_bias.copy_(torch.tensor(bias if bias is not None else [0.0] * NUM_EXPERTS))
    hidden = torch.zeros(1, HIDDEN)
    hidden[0, 0] = 1.0
    return router, hidden


def _counts(indices: torch.Tensor) -> list[float]:
    return torch.bincount(indices.flatten(), minlength=NUM_EXPERTS).float().tolist()


def _collect(router: nn.Module, hidden: torch.Tensor) -> list[float]:
    """Drive the callback over one router forward and return its per-expert counts."""
    model = _RouterModel(router)
    model.train()
    callback = MoEMetricsCallback(topk=TOP_K)
    callback.on_train_begin(args=None, state=None, control=None, model=model)
    assert callback._hook_handles, "the callback hooked no router at all"
    model.gate(hidden)
    counter = callback._counters[0]
    callback.on_train_end(args=None, state=None, control=None)
    assert counter is not None, "the router forward produced no counts"
    return counter.tolist()


# The two mechanisms that move the selection away from topk(router_logits).

# expert 1 and 2 are a near-tie one bias step apart; the bias hands the slot to expert 2.
_TIE_LOGITS = [1.0, 0.99, 0.98, -5.0, -5.0, -5.0, -5.0, -5.0]
_TIE_BIAS = [0.0, 0.0, 0.05, 0.0, 0.0, 0.0, 0.0, 0.0]

# expert 0 holds the largest raw logit, but its group loses the group score, so it routes NO tokens.
_GROUP_LOGITS = [3.0, -5.0, -5.0, -5.0, 2.0, 1.9, 1.8, 1.7]
_GROUP_OVERRIDES = {"n_group": 2, "topk_group": 1}


def test_premise_bias_and_group_limit_really_move_the_selection():
    """Guard the premise: if these fixtures ever stopped diverging, the tests below would prove nothing."""
    for logits, bias, overrides in (
        (_TIE_LOGITS, _TIE_BIAS, {}),
        (_GROUP_LOGITS, None, _GROUP_OVERRIDES),
    ):
        router, hidden = _one_hot_router(logits, bias, **overrides)
        router_logits, _, topk_indices = router(hidden)
        naive = torch.topk(router_logits, TOP_K, dim=-1).indices
        assert set(topk_indices.flatten().tolist()) != set(naive.flatten().tolist()), (
            f"router selected {topk_indices.tolist()} and topk(router_logits) selected {naive.tolist()} "
            "— the fixture no longer separates the two readings"
        )


def test_balancing_bias_flipping_a_near_tie_is_reflected_in_the_load():
    """The token the bias moved must be counted on the expert that ran it."""
    router, hidden = _one_hot_router(_TIE_LOGITS, _TIE_BIAS)
    _, _, topk_indices = router(hidden)

    counts = _collect(router, hidden)

    assert counts == _counts(topk_indices), (
        f"moe/* counted {counts} against the router's own selection {topk_indices.tolist()}; the "
        "balancing bias added before the top-k was ignored"
    )
    assert counts[2] == 1.0 and counts[1] == 0.0, (
        f"expert 2 won the near-tie through the bias and expert 1 lost it, but the load reads {counts} "
        "— this is the topk(router_logits) reading"
    )


def test_group_limited_routing_leaves_the_excluded_expert_dead():
    """An expert masked out by group-limited routing receives no tokens and must read as dead."""
    router, hidden = _one_hot_router(_GROUP_LOGITS, None, **_GROUP_OVERRIDES)
    _, _, topk_indices = router(hidden)

    counts = _collect(router, hidden)

    assert counts == _counts(topk_indices), (
        f"moe/* counted {counts} against the router's own selection {topk_indices.tolist()}; the "
        "group mask applied before the top-k was ignored"
    )
    assert counts[0] == 0.0, (
        f"expert 0 holds the layer's largest raw logit but its group was masked out, so it ran no "
        f"tokens — the load reads {counts}, which credits it anyway"
    )


def test_counts_accumulate_across_microbatches():
    """Counters span an optimizer step, so a second forward adds to the first."""
    router, hidden = _one_hot_router(_GROUP_LOGITS, None, **_GROUP_OVERRIDES)
    _, _, topk_indices = router(hidden)

    model = _RouterModel(router)
    model.train()
    callback = MoEMetricsCallback(topk=TOP_K)
    callback.on_train_begin(args=None, state=None, control=None, model=model)
    model.gate(hidden)
    model.gate(hidden)

    doubled = [2.0 * c for c in _counts(topk_indices)]
    assert callback._counters[0].tolist() == doubled


def test_rewiring_does_not_stack_hooks():
    """A second ``train()`` on the same callback must re-hook, not add a second hook per router."""
    router, hidden = _one_hot_router(_GROUP_LOGITS, None, **_GROUP_OVERRIDES)
    _, _, topk_indices = router(hidden)

    model = _RouterModel(router)
    model.train()
    callback = MoEMetricsCallback(topk=TOP_K)
    callback.on_train_begin(args=None, state=None, control=None, model=model)
    callback.on_train_begin(args=None, state=None, control=None, model=model)
    model.gate(hidden)

    assert callback._counters[0].tolist() == _counts(topk_indices), (
        "one forward was counted more than once — the first wiring's hooks were left installed"
    )


def test_eval_forwards_are_not_counted():
    """The hot path pays nothing outside training, and eval routing must not pollute the step's load."""
    router, hidden = _one_hot_router(_GROUP_LOGITS, None, **_GROUP_OVERRIDES)
    model = _RouterModel(router)
    model.eval()
    callback = MoEMetricsCallback(topk=TOP_K)
    callback.on_train_begin(args=None, state=None, control=None, model=model)
    model.gate(hidden)
    assert callback._counters == [None], f"eval forward accumulated {callback._counters}"


# Discovery, and the two readings that are NOT the router's index tensor.


def test_routers_are_discovered_from_the_transformers_declaration():
    """Every MoE layer of a real family model contributes exactly one hooked router, in layer order."""
    model = GptOssForCausalLM(GptOssConfig(**TINY_GPTOSS_CONFIG))
    routers = _hooked_routers(model)

    assert [r.name for r in routers] == [
        f"model.layers.{i}.mlp.router" for i in range(TINY_GPTOSS_CONFIG["num_hidden_layers"])
    ], f"discovered {[r.name for r in routers]}"
    assert all(isinstance(r.module, GptOssTopKRouter) for r in routers)
    assert all(r.logits_index == 0 and not r.folds_discard_slot for r in routers)


class _LogitsOnlyRouter(nn.Module):
    """A router publishing logits and nothing else — the only case the top-k fallback is for."""

    def forward(self, hidden_states):
        return (hidden_states @ torch.eye(NUM_EXPERTS),)


def test_logits_only_router_falls_back_to_topk_and_says_so_once(monkeypatch):
    """The fallback must stay reachable, and must name the model type it is approximating."""
    warnings: list[tuple] = []
    monkeypatch.setattr(moe_metrics.logger, "warning", lambda msg, *args, **kwargs: warnings.append((msg,) + args))

    model = _RouterModel(_LogitsOnlyRouter(), model_type="some_exotic_moe")
    model.train()
    callback = MoEMetricsCallback(topk=1)
    callback.on_train_begin(args=None, state=None, control=None, model=model)

    hidden = torch.zeros(1, NUM_EXPERTS)
    hidden[0, 3] = 1.0
    model.gate(hidden)
    model.gate(hidden)

    assert callback._counters[0].tolist() == [0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0]
    approximation = [w for w in warnings if "topk(router_logits)" in str(w[0])]
    assert len(approximation) == 1, f"expected exactly one fallback warning, got {warnings}"
    assert "some_exotic_moe" in approximation[0], (
        f"the fallback warning does not name the model type: {approximation[0]}"
    )


class _DiscardSlotRouter(nn.Module):
    """A router whose trailing logit column is a discard slot (Zaya): skipped tokens come back as an
    expert id, so only the logits still carry the discard decision."""

    _has_discard_expert_slot = True

    def forward(self, hidden_states):
        logits = hidden_states @ torch.eye(NUM_EXPERTS)
        return logits, logits[:, :1], torch.zeros(hidden_states.shape[0], 1, dtype=torch.long)


def test_discard_slot_router_is_read_from_its_logits():
    """Its index tensor cannot express "discarded", so counting it would credit a real expert."""
    model = _RouterModel(_DiscardSlotRouter())
    model.train()
    callback = MoEMetricsCallback(topk=1)
    callback.on_train_begin(args=None, state=None, control=None, model=model)
    assert callback._hook_handles and _hooked_routers(model)[0].folds_discard_slot

    hidden = torch.zeros(1, NUM_EXPERTS)
    hidden[0, NUM_EXPERTS - 1] = 1.0  # this token takes the discard slot
    model.gate(hidden)

    counts = callback._counters[0].tolist()
    assert counts[-1] == 1.0 and counts[0] == 0.0, (
        f"the discard slot took the token but the load reads {counts}, crediting expert 0 — the "
        "router's masked index tensor was counted instead of its logits"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
