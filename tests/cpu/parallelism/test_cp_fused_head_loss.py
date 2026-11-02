"""The CP wrapper's fused head loss: under ``fused_linear_cross_entropy`` a training forward scores each rank's
chunk from its backbone hidden states, and must reproduce the fp32 logits loss it replaces.

Two gloo ranks run every CP roster family twice on one seeded model, once without the request (the logits
loss) and once with it, Liger's Triton kernel replaced by a plain-torch stand-in of its contract (Liger has no CPU
kernel; the GPU tier runs the real one). Per rank the two runs must agree on the loss, every gradient and the router
aux term, over labels that supervise the chunk boundary alone or leave one rank's chunk empty; the fused run must
return no logits and the chunk's next-token accuracy. In process, the head refusals (an adapted output embedding, a
forward without ``logits_to_keep``, an unverifiable head path, a logit scale over a bias) fall back to the logits
loss with one warning, eval keeps the logits loss, and the kernel receives the family's head path.

Run: python tests/cpu/parallelism/test_cp_fused_head_loss.py
"""

import inspect
import logging
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import PartialState
from liger_kernel.transformers.functional import CrossEntropyOutput, liger_fused_linear_cross_entropy
from peft import LoraConfig, get_peft_model

from halo.distributed.context_parallel.sequence_chunks import cp_local_targets
from halo.distributed.context_parallel.ulysses_model import UlyssesModel
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.distributed.parallelism_config import ParallelismConfig
from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from halo.model_loading.liger import cross_entropy
from halo.model_loading.liger.cross_entropy import FusedHeadLossUnavailable, fused_head_ce_sum, fused_head_loss
from halo.model_loading.liger.dispatch import LIGER_CP_FUSED_LOSS_ATTR
from halo.models.config_access import set_config_field_run_scoped
from halo.models.head_transform import HeadTransform
from tests.common.cp_model import head_input_lengths, unpatched_cp_wrapper
from tests.common.ep_reference import random_token_batch
from tests.common.equivalence.cp_equivalence import BATCH, CP_SIZE, FAMILIES, SEED, SEQ, tier_controls, tiny_model
from tests.common.gloo import run_gloo_ranks
from tests.common.parallelism import single_process_ep_config

# Loud enough that a dropped or doubled aux term moves the loss far past the paths' fp32 agreement.
MOE_AUX_OVERRIDES = {"output_router_logits": True, "router_aux_loss_coef": 0.1}
# The two paths run the same fp32 math in different op orders (a logit scale folded onto the hidden states).
RELATIVE_TOLERANCE = 1e-5
AUX_SHARE_MIN = 1e3 * RELATIVE_TOLERANCE
# Remote-code forwards without ``logits_to_keep``, which keep the logits loss.
LOGITS_LOSS_FAMILIES = ("bailing_moe",)
# The tiny MoE families whose forward returns ``aux_loss`` under ``output_router_logits``.
AUX_LOSS_FAMILIES = ("gpt_oss", "qwen3_5_moe_text", "qwen3_moe")


def torch_fused_linear_cross_entropy(
    input, weight, target, bias=None, *, ignore_index, reduction, softcap, accum_dtype, return_predicted_tokens
):
    """Liger's ``liger_fused_linear_cross_entropy`` contract, for the keywords :func:`fused_head_ce_sum` passes, in
    plain torch: the logits of ``input @ weight.T + bias`` softcapped, their fp32 cross-entropy and argmax, which
    the kernel writes as ``-1`` at an ignored target."""
    assert accum_dtype is torch.float32 and return_predicted_tokens
    logits = F.linear(input, weight, bias).float()
    if softcap is not None:
        logits = softcap * torch.tanh(logits / softcap)
    loss = F.cross_entropy(logits, target, ignore_index=ignore_index, reduction=reduction)
    predicted = torch.where(target == ignore_index, -1, logits.argmax(dim=-1))
    return CrossEntropyOutput(loss=loss, predicted_tokens=predicted)


def _labels(pattern: str, vocab_size: int):
    """``(input_ids, labels)``: every fourth label masked; or every label of rank 0's chunk masked bar the next
    chunk's first, which only rank 0's boundary position targets; or rank 0's every target masked."""
    ids, labels = random_token_batch(vocab_size, BATCH, SEQ, "cpu", seed=SEED)
    if pattern == "boundary_only":
        labels[:, : SEQ // CP_SIZE] = LABEL_IGNORE_INDEX
        labels[:, SEQ // CP_SIZE] = ids[:, SEQ // CP_SIZE]
    elif pattern == "empty_chunk":
        labels[:, : SEQ // CP_SIZE + 1] = LABEL_IGNORE_INDEX
    return ids, labels


LABEL_PATTERNS = ("interleaved", "boundary_only", "empty_chunk")


def _step(wrapper: UlyssesModel, ids, labels) -> dict:
    with head_input_lengths(wrapper.model) as head_lengths:
        outputs = wrapper(input_ids=ids, labels=labels)
    outputs.loss.backward()
    aux = getattr(outputs, "aux_loss", None)
    return {
        "head_lengths": head_lengths,
        "loss": outputs.loss.detach(),
        "logits": None if outputs.logits is None else outputs.logits.detach(),
        "token_accuracy": getattr(outputs, "token_accuracy", None),
        "aux": None if aux is None else aux.detach(),
        "grads": {name: p.grad.detach().clone() for name, p in wrapper.named_parameters() if p.grad is not None},
    }


def _wrapped(name: str, cp_config, *, fused: bool) -> UlyssesModel:
    family = FAMILIES[name]
    model = tiny_model(name, "cpu", torch.device("cpu"), **(MOE_AUX_OVERRIDES if family.moe else {}))
    if family.moe:
        apply_ep_layers(model, single_process_ep_config())
    if fused:
        set_config_field_run_scoped(model.config, LIGER_CP_FUSED_LOSS_ATTR, True)
    return UlyssesModel(model, cp_config).train()


def _record_worker(rank: int, name: str, output: str) -> None:
    PartialState()
    cp_config = ParallelismConfig(cp_size=CP_SIZE).cp_config()
    stand_in = patch.object(cross_entropy, "liger_fused_linear_cross_entropy", torch_fused_linear_cross_entropy)
    with tier_controls("cpu"), stand_in:
        for pattern in LABEL_PATTERNS:
            runs = {}
            for path in ("logits", "fused"):
                wrapper = _wrapped(name, cp_config, fused=path == "fused")
                ids, labels = _labels(pattern, wrapper.get_input_embeddings().num_embeddings)
                runs[path] = _step(wrapper, ids, labels)
            runs["targets"] = cp_local_targets(labels, rank, CP_SIZE)
            runs["takes_logits_to_keep"] = "logits_to_keep" in inspect.signature(wrapper.model.forward).parameters
            torch.save(runs, Path(output, f"{pattern}-{rank}.pt"))


def _relative_gap(head: torch.Tensor, reference: torch.Tensor) -> float:
    return ((head - reference).abs().max() / reference.abs().max().clamp(min=1e-12)).item()


def _path_problems(runs: dict) -> list[str]:
    """How the fused run departs from the logits run on one rank: loss, gradients, logits and the chunk accuracy.
    A family whose forward takes no ``logits_to_keep`` (Bailing's remote code) keeps the logits loss."""
    logits_run, fused_run, targets = runs["logits"], runs["fused"], runs["targets"]
    problems = []
    if (
        not torch.isfinite(fused_run["loss"])
        or _relative_gap(fused_run["loss"], logits_run["loss"]) > RELATIVE_TOLERANCE
    ):
        problems.append(f"fused loss {fused_run['loss'].item():.8f} vs logits loss {logits_run['loss'].item():.8f}")
    if fused_run["grads"].keys() != logits_run["grads"].keys():
        problems.append(f"gradients differ in {sorted(fused_run['grads'].keys() ^ logits_run['grads'].keys())}")
    else:
        gaps = {name: _relative_gap(fused_run["grads"][name], g) for name, g in logits_run["grads"].items()}
        problems += [f"{name} gradient off by {gap:.3e}" for name, gap in gaps.items() if gap > RELATIVE_TOLERANCE]
    if logits_run["logits"] is None or logits_run["token_accuracy"] is not None:
        problems.append("the logits run did not return logits alone")
    chunk = [SEQ // CP_SIZE]
    if logits_run["head_lengths"] != chunk:
        problems.append(f"the logits run projected {logits_run['head_lengths']} positions, not the chunk's {chunk}")
    if not runs["takes_logits_to_keep"]:
        if fused_run["logits"] is None or fused_run["token_accuracy"] is not None:
            problems.append("a forward without logits_to_keep did not keep the logits loss")
        return problems
    if fused_run["logits"] is not None:
        problems.append("the fused run returned logits")
    # The memory saving itself: the family forward honours logits_to_keep, so no [N, V] plane is built.
    if fused_run["head_lengths"] != [1]:
        problems.append(f"the fused run projected {fused_run['head_lengths']} positions through the head, not [1]")
    supervised = targets != LABEL_IGNORE_INDEX
    correct = ((logits_run["logits"].argmax(dim=-1) == targets) & supervised).sum()
    expected = correct / supervised.sum().clamp(min=1)
    if fused_run["token_accuracy"] is None or not torch.equal(fused_run["token_accuracy"], expected):
        problems.append(f"token accuracy {fused_run['token_accuracy']} vs {expected}")
    return problems


def _aux_problems(name: str, runs: dict) -> list[str]:
    """The router aux term on both paths alike, at a share of the loss a dropped or doubled term would move past
    the loss comparison."""
    logits_aux, fused_aux = runs["logits"]["aux"], runs["fused"]["aux"]
    if (logits_aux is None) != (fused_aux is None):
        return [f"{name}: aux term on one path only"]
    if fused_aux is None:
        return []
    share = MOE_AUX_OVERRIDES["router_aux_loss_coef"] * fused_aux.abs().item() / runs["fused"]["loss"].item()
    problems = [] if torch.equal(fused_aux, logits_aux) else [f"{name}: aux terms differ"]
    if share < AUX_SHARE_MIN:
        problems.append(f"{name}: aux share {share:.2e} too small to tell once from twice")
    return problems


@pytest.mark.parametrize("name", list(FAMILIES))
def test_every_family_and_label_pattern_matches_the_logits_loss_on_two_gloo_ranks(name, tmp_path):
    """One spawn per family records its every label pattern on both paths."""
    run_gloo_ranks(_record_worker, CP_SIZE, name, str(tmp_path))
    recorded = {
        pattern: [torch.load(tmp_path / f"{pattern}-{rank}.pt", weights_only=False) for rank in range(CP_SIZE)]
        for pattern in LABEL_PATTERNS
    }
    problems = [
        f"{pattern}/rank {rank}: {problem}"
        for pattern, ranks in recorded.items()
        for rank, runs in enumerate(ranks)
        for problem in _path_problems(runs)
    ]
    interleaved = recorded["interleaved"]
    problems += [problem for runs in interleaved for problem in _aux_problems(name, runs)]
    assert problems == []
    assert (interleaved[0]["fused"]["aux"] is not None) == (name in AUX_LOSS_FAMILIES)
    assert interleaved[0]["takes_logits_to_keep"] == (name not in LOGITS_LOSS_FAMILIES)
    # The patterns are what they claim: rank 0 scores the boundary target alone in each row, then nothing.
    boundary = recorded["boundary_only"][0]["targets"]
    assert ((boundary != LABEL_IGNORE_INDEX).nonzero()[:, 1] == SEQ // CP_SIZE - 1).all()
    assert (boundary != LABEL_IGNORE_INDEX).sum(dim=1).tolist() == [1] * BATCH
    empty = recorded["empty_chunk"][0]
    assert not (empty["targets"] != LABEL_IGNORE_INDEX).any()
    if empty["fused"]["aux"] is None:
        assert empty["fused"]["loss"].item() == 0.0


def _tiny_qwen3(**overrides) -> nn.Module:
    model = tiny_model("qwen3", "cpu", torch.device("cpu"), **overrides)
    model.config._attn_implementation = "eager"
    return model


def _requested(model: nn.Module) -> nn.Module:
    set_config_field_run_scoped(model.config, LIGER_CP_FUSED_LOSS_ATTR, True)
    return model


def _single_rank_step(model: nn.Module) -> dict:
    ids, labels = random_token_batch(model.get_input_embeddings().num_embeddings, BATCH, SEQ, "cpu", seed=SEED)
    with patch.object(cross_entropy, "liger_fused_linear_cross_entropy", torch_fused_linear_cross_entropy):
        return _step(unpatched_cp_wrapper(model).train(), ids, labels)


def test_the_wrapper_scores_through_the_kernel_only_when_the_run_asks():
    assert _single_rank_step(_tiny_qwen3())["logits"] is not None
    assert _single_rank_step(_requested(_tiny_qwen3()))["logits"] is None


def test_eval_keeps_the_logits_loss_under_the_request():
    model = _requested(_tiny_qwen3())
    ids, labels = random_token_batch(model.get_input_embeddings().num_embeddings, BATCH, SEQ, "cpu", seed=SEED)
    with torch.no_grad():
        outputs = unpatched_cp_wrapper(model).eval()(input_ids=ids, labels=labels)
    assert outputs.logits is not None and outputs.logits.shape[1] == SEQ


def _unlabelled_warnings(model: nn.Module, caplog, *, grad: bool) -> list[str]:
    """The no-labels warnings two forwards without labels draw (a labelled one in between), each asserted to return
    the chunk's full logits."""
    wrapper = unpatched_cp_wrapper(model).train()
    ids, labels = random_token_batch(model.get_input_embeddings().num_embeddings, BATCH, SEQ, "cpu", seed=SEED)
    caplog.clear()
    with (
        caplog.at_level(logging.WARNING),
        torch.set_grad_enabled(grad),
        patch.object(cross_entropy, "liger_fused_linear_cross_entropy", torch_fused_linear_cross_entropy),
    ):
        for call_labels in (None, labels, None):
            outputs = wrapper(input_ids=ids, labels=call_labels)
            assert call_labels is not None or outputs.logits.shape[1] == SEQ
    return [r.getMessage() for r in caplog.records if "passed no labels" in r.getMessage()]


def test_an_unlabelled_training_forward_under_the_request_warns_once(caplog):
    """SMPO's forward under CP: the request cannot run without labels, which the log says once."""
    (warning,) = _unlabelled_warnings(_requested(_tiny_qwen3()), caplog, grad=True)
    assert "only a labelled forward (SFT) runs the fused loss" in warning
    assert _unlabelled_warnings(_tiny_qwen3(), caplog, grad=True) == []
    assert _unlabelled_warnings(_requested(_tiny_qwen3()), caplog, grad=False) == []


def _assert_falls_back_with_one_warning(model: nn.Module, reason: str, caplog) -> None:
    reference = _single_rank_step(_tiny_qwen3())
    wrapper = unpatched_cp_wrapper(model).train()
    ids, labels = random_token_batch(model.get_input_embeddings().num_embeddings, BATCH, SEQ, "cpu", seed=SEED)
    with caplog.at_level(logging.WARNING):
        first, second = (wrapper(input_ids=ids, labels=labels) for _ in range(2))
    warnings = [r.getMessage() for r in caplog.records if "fused_linear_cross_entropy is on" in r.getMessage()]
    assert len(warnings) == 1 and reason in warnings[0], warnings
    assert first.logits is not None and second.logits is not None
    assert _relative_gap(first.loss.detach(), reference["loss"]) <= RELATIVE_TOLERANCE


def test_a_forward_without_logits_to_keep_falls_back(caplog):
    model = _requested(_tiny_qwen3())
    stock_forward = type(model).forward

    def forward_without_logits_to_keep(self, input_ids=None, attention_mask=None, labels=None, **kwargs):
        return stock_forward(self, input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)

    with patch.object(type(model), "forward", forward_without_logits_to_keep):
        _assert_falls_back_with_one_warning(model, "takes no logits_to_keep", caplog)


@pytest.mark.parametrize("peft_kwargs", [{"target_modules": ["lm_head"]}, {"modules_to_save": ["lm_head"]}])
def test_an_adapted_output_embedding_falls_back(peft_kwargs, caplog):
    model = _requested(_tiny_qwen3())
    peft_model = get_peft_model(model, LoraConfig(r=2, **({"target_modules": ["q_proj"]} | peft_kwargs)))
    with pytest.raises(FusedHeadLossUnavailable):
        fused_head_loss(peft_model)
    _assert_falls_back_with_one_warning(peft_model.get_base_model(), "output embedding", caplog)


def test_a_linear_subclass_with_its_own_forward_is_refused():
    """A low-precision or otherwise re-implemented head computes what ``hidden @ weight.T`` would not."""

    class OwnForwardLinear(nn.Linear):
        def forward(self, input):
            return super().forward(input) * 2

    model = _tiny_qwen3()
    head = model.get_output_embeddings()
    model.lm_head = OwnForwardLinear(head.in_features, head.out_features, bias=False)
    with pytest.raises(FusedHeadLossUnavailable, match="not a plain nn.Linear"):
        fused_head_loss(model)


def test_a_tuner_over_the_plain_output_embedding_is_refused():
    """A PEFT-style tuner whose ``base_layer`` is the very linear ``get_output_embeddings`` returns: the tuner's delta
    lives outside that linear."""

    class Tuner(nn.Module):
        def __init__(self, base_layer: nn.Linear):
            super().__init__()
            self.base_layer = base_layer

    model = _tiny_qwen3()
    model.head_tuner = Tuner(model.get_output_embeddings())
    with pytest.raises(FusedHeadLossUnavailable, match="PEFT tuner wraps"):
        fused_head_loss(model)


def test_the_kernel_call_binds_to_the_real_liger_signature():
    """The CPU stand-in only checks the keywords against itself; the real function must accept the same call."""
    calls = []

    def recorder(*args, **kwargs):
        calls.append((args, kwargs))
        return torch_fused_linear_cross_entropy(*args, **kwargs)

    head = nn.Linear(16, 32, bias=False)
    with patch.object(cross_entropy, "liger_fused_linear_cross_entropy", recorder):
        fused_head_ce_sum(head, HeadTransform(softcap=3.0), torch.randn(4, 16), torch.randint(0, 32, (4,)))
    ((args, kwargs),) = calls
    inspect.signature(liger_fused_linear_cross_entropy).bind(*args, **kwargs)


def test_an_unverifiable_head_path_is_refused_with_its_reason():
    def unverifiable(_model):
        raise ValueError("the probe could not verify it")

    with patch.object(cross_entropy, "resolve_head_transform", unverifiable):
        with pytest.raises(FusedHeadLossUnavailable, match="the probe could not verify it"):
            fused_head_loss(_tiny_qwen3())


def test_a_logit_scale_over_a_biased_head_is_refused():
    model = _tiny_qwen3()
    head = model.get_output_embeddings()
    model.lm_head = nn.Linear(head.in_features, head.out_features, bias=True)
    with patch.object(cross_entropy, "resolve_head_transform", lambda _model: HeadTransform(logit_scale=0.5)):
        with pytest.raises(FusedHeadLossUnavailable, match="biased head"):
            fused_head_loss(model)


@pytest.mark.parametrize(
    "transform",
    [
        HeadTransform(),
        HeadTransform(hidden_scale=0.5),
        HeadTransform(logit_scale=0.25),
        HeadTransform(softcap=3.0),
        HeadTransform(vocab_size=29),
        HeadTransform(hidden_scale=2.0, logit_scale=0.5, softcap=5.0, vocab_size=29),
    ],
    ids=["identity", "hidden_scale", "logit_scale", "softcap", "vocab_cut", "all"],
)
def test_the_kernel_scores_the_family_head_path(transform):
    """:func:`fused_head_ce_sum` against ``transform.project`` and fp32 cross-entropy: loss, the argmax of every
    supervised row and the hidden and weight gradients."""
    generator = torch.Generator().manual_seed(SEED)
    head = nn.Linear(16, 32, bias=False)
    hidden = (4 * torch.randn(24, 16, generator=generator)).requires_grad_()
    targets = torch.randint(0, transform.vocab_size or 32, (24,), generator=generator)
    targets[::5] = LABEL_IGNORE_INDEX

    logits = transform.project(head, hidden).float()
    reference = F.cross_entropy(logits, targets, ignore_index=LABEL_IGNORE_INDEX, reduction="sum")
    reference.backward()
    expected_grads = hidden.grad.clone(), head.weight.grad.clone()
    hidden.grad, head.weight.grad = None, None

    with patch.object(cross_entropy, "liger_fused_linear_cross_entropy", torch_fused_linear_cross_entropy):
        loss, predicted = fused_head_ce_sum(head, transform, hidden, targets)
    loss.backward()
    assert _relative_gap(loss.detach(), reference.detach()) <= RELATIVE_TOLERANCE
    supervised = targets != LABEL_IGNORE_INDEX
    assert torch.equal(predicted[supervised], logits.argmax(dim=-1)[supervised])
    for got, expected in zip((hidden.grad, head.weight.grad), expected_grads, strict=True):
        assert _relative_gap(got, expected) <= RELATIVE_TOLERANCE


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
