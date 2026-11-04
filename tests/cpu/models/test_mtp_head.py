#!/usr/bin/env python
"""The head's forward on real tiny family classes: it predicts two tokens ahead from the embedding of the next one, its
term scales with the next-token loss on one normalizer, and the soft route and the scopes reach what they claim."""

import json
import re

import pytest
import torch
import torch.nn.functional as F
from accelerate import init_empty_weights
from transformers import (
    CONFIG_MAPPING,
    AutoModelForCausalLM,
    NemotronH_Omni_Reasoning_V3,
    NemotronH_Omni_Reasoning_V3_Config,
)

from halo.model_loading.liger import cross_entropy
from halo.models.config_access import text_config
from halo.models.module_tree import backbone_with_layers, decoder_layers
from halo.models.mtp.head import (
    build_mtp_head,
    find_mtp_head,
    freeze_trunk_for_head,
    frozen_trunk_params,
    mtp_head_path,
)
from halo.models.mtp.spec import mtp_draft_layer_index, require_mtp_head_spec
from tests.common.models import NEMOTRON_H_RELEASE_TEXT_CONFIG, NEMOTRON_H_RELEASE_VISION_CONFIG
from tests.common.mtp_heads import (
    LABEL_IGNORE,
    MTP_TINY_FAMILIES,
    install_term,
    packed_rows,
    padded_rows,
    tiny_model_with_head,
)
from tests.common.nemotron_h_hub_keys import nemotron_h_hub_keys
from tests.common.repo import REPO_ROOT

_WEIGHT = 0.7
# Read from each release's safetensors headers and config.json.
_RELEASE_TAILS = json.loads((REPO_ROOT / "tests/common/mtp_release_tails.json").read_text(encoding="utf-8"))
_FAMILIES = sorted(MTP_TINY_FAMILIES)
_MOE_TEXT_FAMILIES = ("qwen3_5_moe_text", "qwen3_next", "glm4_moe_lite", "nemotron_h")
_NEMOTRON_ROUTES = ("nemotron_h", "nemotron_h_omni")
_LIGHTNING = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
_SUPER_VL = "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
_SUPER_VL_TAIL_LEVEL = "language_model."


def _grads(module) -> dict[str, torch.Tensor]:
    return {name: param.grad.clone() for name, param in module.named_parameters() if param.grad is not None}


def _relative(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def _worst(a: dict, b: dict, names=None) -> float:
    return max(_relative(a[name], b[name]) for name in (names or b) if b[name].abs().sum() > 0)


def _arm(name: str, rows: dict, **objective):
    """The head's weights are fixed per family, so two arms differ in the objective alone."""
    model, head = tiny_model_with_head(name)
    if objective:
        install_term(model, head, **objective)
    loss = model(**rows).loss
    loss.backward()
    return model, head, loss.detach()


def _head_and_trunk(model, head):
    head_ids = {id(param) for param in head.parameters()}
    head_names = [n for n, p in model.named_parameters() if id(p) in head_ids and p.grad is not None]
    trunk_names = [n for n, p in model.named_parameters() if id(p) not in head_ids and p.grad is not None]
    return head_names, trunk_names


def _draft_of(model, head, rows: dict) -> torch.Tensor:
    """The head's output for ``rows``, with its hidden and embedding inputs."""
    seen = []
    handle = head.register_forward_hook(lambda _module, args, output: seen.append((output.detach(), args)))
    model(**rows)
    handle.remove()
    assert len(seen) == 1, "the head runs once per labelled training forward"
    return seen[0][0]


@pytest.mark.parametrize("name", _FAMILIES)
def test_the_term_is_the_cross_entropy_two_tokens_ahead_over_the_next_token_normalizer(name):
    rows = padded_rows()
    model, head = tiny_model_with_head(name)
    baseline = model(**rows).loss.detach()
    install_term(model, head, weight=_WEIGHT)
    draft_seen = []
    handle = head.register_forward_hook(lambda _module, _args, output: draft_seen.append(output))
    loss = model(**rows).loss
    handle.remove()
    weight = model.get_output_embeddings().weight
    labels = rows["labels"]
    expected = (
        F.cross_entropy(
            F.linear(draft_seen[0][:, :-2], weight).reshape(-1, weight.size(0)),
            labels[:, 2:].reshape(-1),
            ignore_index=LABEL_IGNORE,
            reduction="sum",
        )
        / (labels[:, 1:] != LABEL_IGNORE).sum()
    )
    assert expected > 0
    assert torch.isclose(loss - baseline, _WEIGHT * expected, rtol=1e-9), "wrong target offset, weight or normalizer"


@pytest.mark.parametrize("name", _FAMILIES)
def test_a_given_normalizer_divides_the_term_too(name):
    """HF's per-batch token count, which every rank shares, not this rank's own."""
    rows = padded_rows()
    given = torch.tensor(1000)
    plain, _ = tiny_model_with_head(name)
    model, head = tiny_model_with_head(name)
    install_term(model, head, weight=1.0)
    term_over_given = model(**rows, num_items_in_batch=given).loss - plain(**rows, num_items_in_batch=given).loss
    term_over_own = model(**rows).loss - plain(**rows).loss
    tokens = (rows["labels"][:, 1:] != LABEL_IGNORE).sum()
    assert term_over_own > 0
    assert torch.isclose(term_over_given * given, term_over_own * tokens, rtol=1e-9)


@pytest.mark.parametrize("name", _FAMILIES)
def test_soft_keeps_the_loss_and_the_output_embedding_gradient_of_hard(name):
    rows = padded_rows()
    hard_model, _, hard_loss = _arm(name, rows, weight=_WEIGHT, loss_type="hard")
    soft_model, _, soft_loss = _arm(name, rows, weight=_WEIGHT, loss_type="soft")
    assert soft_loss == hard_loss
    assert torch.equal(soft_model.get_output_embeddings().weight.grad, hard_model.get_output_embeddings().weight.grad)


@pytest.mark.parametrize("name", _FAMILIES)
def test_soft_gives_the_head_the_kl_gradient_with_the_next_token_distribution_as_its_target(name):
    rows = padded_rows()
    soft_model, soft_head, _ = _arm(name, rows, weight=_WEIGHT, loss_type="soft", scope="detached")
    hard_model, hard_head, _ = _arm(name, rows, weight=_WEIGHT, loss_type="hard", scope="detached")

    model, head = tiny_model_with_head(name)
    install_term(model, head, weight=0.0, scope="detached")
    hidden_seen, draft_seen = [], []
    handles = (
        head.register_forward_pre_hook(lambda _module, args: hidden_seen.append(args[1].detach())),
        head.register_forward_hook(lambda _module, _args, output: draft_seen.append(output)),
    )
    model(**rows)
    for handle in handles:
        handle.remove()
    model.zero_grad()
    weight = model.get_output_embeddings().weight.detach()
    labels = rows["labels"]
    keep = labels[:, 2:] != LABEL_IGNORE
    teacher = F.softmax(F.linear(hidden_seen[0][:, 1:-1], weight), dim=-1)
    student = F.log_softmax(F.linear(draft_seen[0][:, :-2], weight), dim=-1)
    (_WEIGHT * (-(teacher * student).sum(-1) * keep).sum() / (labels[:, 1:] != LABEL_IGNORE).sum()).backward()

    kl = _grads(head)
    assert kl, "the direct KL reached no head parameter"
    assert _worst(_grads(soft_head), kl) < 1e-5
    assert _worst(_grads(hard_head), kl) > 0.05, "hard and soft agree, so the comparison above proves nothing"


def _next_token_plus_kl(name: str, rows: dict, *, teacher_stopped: bool):
    """The next-token loss plus the direct KL at the head's step, off an arm with no term of the head's own."""
    model, head = tiny_model_with_head(name)
    install_term(model, head, weight=0.0)
    hidden_seen, draft_seen = [], []
    handles = (
        head.register_forward_pre_hook(lambda _module, args: hidden_seen.append(args[1])),
        head.register_forward_hook(lambda _module, _args, output: draft_seen.append(output)),
    )
    loss = model(**rows).loss
    for handle in handles:
        handle.remove()
    weight = model.get_output_embeddings().weight.detach()
    labels = rows["labels"]
    teacher_hidden = hidden_seen[0][:, 1:-1]
    teacher = F.softmax(F.linear(teacher_hidden.detach() if teacher_stopped else teacher_hidden, weight), dim=-1)
    student = F.log_softmax(F.linear(draft_seen[0][:, :-2], weight), dim=-1)
    kl = -(teacher * student).sum(-1) * (labels[:, 2:] != LABEL_IGNORE)
    (loss + _WEIGHT * kl.sum() / (labels[:, 1:] != LABEL_IGNORE).sum()).backward()
    written = model.get_output_embeddings().weight
    return {key: value for key, value in _grads(model).items() if model.get_parameter(key) is not written}


@pytest.mark.parametrize("name", _FAMILIES)
def test_joint_soft_is_the_next_token_loss_plus_a_kl_with_the_teacher_stopped(name):
    """Soft-MTP's identity on what the term reaches: head, trunk and embedding take the KL's gradient with none through
    the teacher, and the output embedding keeps the hard push the test above pins."""
    rows = padded_rows()
    soft_model, _, _ = _arm(name, rows, weight=_WEIGHT, loss_type="soft")
    expected = _next_token_plus_kl(name, rows, teacher_stopped=True)
    # The fused CE reduces in fp32 on this fp64 model, which a low-norm parameter reads as this much.
    assert _worst(_grads(soft_model), expected) < 1e-3
    hard_model, _, _ = _arm(name, rows, weight=_WEIGHT, loss_type="hard")
    assert _worst(_grads(hard_model), expected) > 0.01, "hard agrees with the KL, so the comparison proves nothing"
    assert _worst(_grads(soft_model), _next_token_plus_kl(name, rows, teacher_stopped=False)) > 0.01


@pytest.mark.parametrize("loss_type", ["hard", "soft"])
@pytest.mark.parametrize("name", _FAMILIES)
def test_a_detached_head_leaves_the_trunk_the_next_token_gradient_and_keeps_the_joint_head_gradient(name, loss_type):
    rows = padded_rows()
    ntp_model, _, _ = _arm(name, rows)
    joint_model, joint_head, _ = _arm(name, rows, weight=_WEIGHT, loss_type=loss_type)
    model, head, _ = _arm(name, rows, weight=_WEIGHT, loss_type=loss_type, scope="detached")
    head_names, trunk_names = _head_and_trunk(model, head)
    assert head_names and trunk_names
    ntp, detached, joint = _grads(ntp_model), _grads(model), _grads(joint_model)
    assert _worst(detached, ntp, trunk_names) == 0.0
    assert _worst(detached, joint, head_names) == 0.0
    # The joint term reaches the trunk, so the equality above is not everyone's.
    assert _worst(joint, ntp, trunk_names) > 1e-3


@pytest.mark.parametrize("loss_type", ["hard", "soft"])
@pytest.mark.parametrize("name", _MOE_TEXT_FAMILIES)
def test_head_only_moves_nothing_outside_the_head_and_matches_the_detached_head(name, loss_type):
    rows = padded_rows()
    detached_model, detached_head, _ = _arm(name, rows, weight=_WEIGHT, loss_type=loss_type, scope="detached")
    model, head = tiny_model_with_head(name)
    freeze_trunk_for_head(model, head)
    install_term(model, head, weight=_WEIGHT, loss_type=loss_type, scope="head_only")
    trunk = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    grad_mode = []
    layer = decoder_layers(backbone_with_layers(model))[0]
    layer.register_forward_pre_hook(lambda _module, _args: grad_mode.append(torch.is_grad_enabled()))
    model(**rows).loss.backward()
    assert grad_mode == [False], "the frozen trunk built a graph"
    assert _worst(_grads(head), _grads(detached_head)) == 0.0
    head_names, trunk_names = _head_and_trunk(model, head)
    assert head_names and not trunk_names, "a frozen trunk took a gradient"
    before = {n: p.detach().clone() for n, p in head.named_parameters()}
    optimizer.step()
    for n, p in model.named_parameters():
        if n in trunk:
            assert torch.equal(p, trunk[n]), n
    assert all(not torch.equal(p, before[n]) for n, p in head.named_parameters() if p.grad.abs().sum() > 0)


@pytest.mark.parametrize("use_reentrant", [True, False], ids=["reentrant", "non_reentrant"])
@pytest.mark.parametrize("scope", ["joint", "detached"])
@pytest.mark.parametrize("name", _FAMILIES)
def test_checkpointing_changes_no_gradient(name, scope, use_reentrant):
    rows = padded_rows()
    plain_model, _, _ = _arm(name, rows, weight=_WEIGHT, loss_type="soft", scope=scope)
    model, head = tiny_model_with_head(name)
    install_term(model, head, weight=_WEIGHT, loss_type="soft", scope=scope)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": use_reentrant})
    model.config.use_cache = False
    model(**rows).loss.backward()
    assert _grads(head), "the head took no gradient, so the comparison holds for the trunk alone"
    assert _worst(_grads(model), _grads(plain_model)) < 1e-9


@pytest.mark.parametrize("scope", ["joint", "detached", "head_only"])
@pytest.mark.parametrize("loss_type", ["hard", "soft"])
@pytest.mark.parametrize("name", _MOE_TEXT_FAMILIES)
def test_a_row_past_the_chunk_size_keeps_every_gradient(name, loss_type, scope, monkeypatch):
    """The fallback scores a long row in checkpointed chunks, so no ``[T, V]`` plane is held through backward."""
    rows = padded_rows()

    def run():
        model, head = tiny_model_with_head(name)
        if scope == "head_only":
            freeze_trunk_for_head(model, head)
        install_term(model, head, weight=_WEIGHT, loss_type=loss_type, scope=scope)
        loss = model(**rows).loss
        loss.backward()
        return loss.detach(), _grads(model)

    whole_loss, whole = run()
    chunks = []
    real = cross_entropy.chunked_token_sum

    def counted(fn, rows_plane, labels, chunk_rows):
        chunks.append(rows_plane.size(0) // chunk_rows)
        return real(fn, rows_plane, labels, chunk_rows)

    monkeypatch.setattr(cross_entropy, "logit_chunk_rows", lambda _vocab: 5)
    monkeypatch.setattr(cross_entropy, "chunked_token_sum", counted)
    chunked_loss, chunked = run()
    assert chunks and chunks[0] > 1, "the row fit one chunk, so nothing was split"
    assert torch.isclose(chunked_loss, whole_loss, rtol=1e-9)
    assert _worst(chunked, whole) < 1e-9


@pytest.mark.parametrize("name", _FAMILIES)
def test_the_split_reports_each_cross_entropy_over_its_own_targets(name):
    rows = padded_rows()
    # A target at position 1, which the next-token loss scores and the head, predicting two ahead, cannot.
    rows["labels"] = rows["input_ids"].masked_fill(rows["labels"][:, -1:] == LABEL_IGNORE, LABEL_IGNORE)
    rows["labels"][:, 0] = LABEL_IGNORE
    plain, _ = tiny_model_with_head(name)
    next_token_mean = plain(**rows).loss.detach()
    model, head = tiny_model_with_head(name)
    install_term(model, head, weight=_WEIGHT)
    draft_seen = []
    head.register_forward_hook(lambda _module, _args, output: draft_seen.append(output))
    output = model(**rows)
    split = output.mtp_loss_split
    labels = rows["labels"]
    weight = model.get_output_embeddings().weight
    head_sum = F.cross_entropy(
        F.linear(draft_seen[0][:, :-2], weight).reshape(-1, weight.size(0)),
        labels[:, 2:].reshape(-1),
        ignore_index=LABEL_IGNORE,
        reduction="sum",
    )
    assert split.next_token_targets == (labels[:, 1:] != LABEL_IGNORE).sum()
    assert split.head_targets == (labels[:, 2:] != LABEL_IGNORE).sum()
    assert split.head_targets != split.next_token_targets, "one count for both would hide a wrong denominator"
    # HF's loss reduces in fp32 on this fp64 model.
    assert torch.isclose(split.next_token_sum / split.next_token_targets, next_token_mean, rtol=1e-6)
    assert torch.isclose(split.head_sum, head_sum, rtol=1e-9)
    assert not split.next_token_sum.requires_grad and not split.head_sum.requires_grad


def test_the_split_stays_per_target_under_a_shared_normalizer():
    rows = padded_rows()
    name = "qwen3_5_moe_text"
    own, head = tiny_model_with_head(name)
    install_term(own, head, weight=_WEIGHT)
    expected = own(**rows).mtp_loss_split
    model, head = tiny_model_with_head(name)
    install_term(model, head, weight=_WEIGHT)
    given = torch.tensor(1000)
    split = model(**rows, num_items_in_batch=given).mtp_loss_split
    assert torch.isclose(split.next_token_sum, expected.next_token_sum, rtol=1e-6)
    assert torch.isclose(split.head_sum, expected.head_sum, rtol=1e-9)


def test_the_router_aux_term_stays_out_of_the_next_token_cross_entropy():
    """The forward folds it into ``loss``, and ``ntp_ce`` is what a detached run compares against a run with no head."""
    rows = {**padded_rows(), "output_router_logits": True}
    name = "qwen3_5_moe_text"
    plain, _ = tiny_model_with_head(name)
    ce_alone = plain(**padded_rows()).loss.detach()
    model, head = tiny_model_with_head(name)
    model.config.router_aux_loss_coef = 0.5
    model.router_aux_loss_coef = 0.5
    install_term(model, head, weight=_WEIGHT)
    output = model(**rows)
    assert output.aux_loss is not None and output.aux_loss > 0, "the row took no aux term, so nothing was subtracted"
    split = output.mtp_loss_split
    assert torch.isclose(split.next_token_sum / split.next_token_targets, ce_alone, rtol=1e-6)
    # The head's gate runs past the backbone forward its recorder wraps, so the term stays the trunk's.
    assert torch.isclose(output.aux_loss, plain(**rows).aux_loss, rtol=1e-9)


def test_one_rounded_weight_scales_the_term_and_the_route():
    """bf16 rounds 0.3 up by 0.26%. The route takes the draft's dtype, so a term at the unrounded value would leave the
    output embedding's cancellation a residue."""
    rows = padded_rows()
    model, head = tiny_model_with_head("qwen3_5_moe_text", dtype=torch.bfloat16)
    install_term(model, head, weight=0.3, loss_type="soft")
    output = model(**rows)
    split = output.mtp_loss_split
    rounded = torch.tensor(0.3, dtype=torch.bfloat16).to(torch.float32)
    assert rounded != torch.tensor(0.3, dtype=torch.float32), "bf16 keeps this weight exactly, so nothing is pinned"
    next_token = split.next_token_sum / split.next_token_targets
    assert torch.isclose(output.loss, next_token + rounded * split.head_sum / split.next_token_targets, rtol=1e-6)


def test_only_a_frozen_trunk_is_named_for_the_wrap_to_leave_out():
    model, head = tiny_model_with_head("glm4_moe_lite")
    assert frozen_trunk_params(model) == []
    freeze_trunk_for_head(model, head)
    head_ids = {id(param) for param in head.parameters()}
    frozen = frozen_trunk_params(model)
    assert frozen and not any(id(param) in head_ids for param in frozen)
    assert {id(param) for param in frozen} | head_ids == {id(param) for param in model.parameters()}
    bare, _ = tiny_model_with_head("glm4_moe_lite")
    decoder_layers(backbone_with_layers(bare)).pop(-1)
    assert frozen_trunk_params(bare) == []


@pytest.mark.parametrize("name", _MOE_TEXT_FAMILIES)
def test_the_head_at_t_reads_token_t_plus_one_and_not_its_own_target(name):
    """The input-copy trap: a head reading the embedding of the token it must predict fits to top-1 1.0 on that target
    and to chance on the true one, at a falling loss."""
    model, head = tiny_model_with_head(name)
    install_term(model, head, weight=0.1, scope="detached")
    rows = padded_rows()
    rows = {"input_ids": rows["input_ids"][:1], "labels": rows["input_ids"][:1].clone(), "use_cache": False}
    t = 9
    base = _draft_of(model, head, rows)

    def changed(index: int) -> torch.Tensor:
        ids = rows["input_ids"].clone()
        ids[0, index] = (ids[0, index] + 1) % 90 + 4
        return _draft_of(model, head, {**rows, "input_ids": ids})

    assert (changed(t + 1)[0, t] - base[0, t]).abs().max() > 1e-3, "the head ignores the next token's embedding"
    assert (changed(t + 2)[0, t] - base[0, t]).abs().max() < 1e-9, "the head reads its own target"


def test_the_glm_head_zeroes_the_embedding_at_a_document_start_as_its_engine_does():
    """vLLM's drafter masks position 0, so training with the token there fits an input serving never sends."""
    model, head = tiny_model_with_head("glm4_moe_lite")
    install_term(model, head, weight=0.1, scope="detached")
    rows = packed_rows()
    base = _draft_of(model, head, rows)
    for start in (0, 12):
        ids = rows["input_ids"].clone()
        ids[0, start + 1] = (ids[0, start + 1] + 1) % 90 + 4
        moved = _draft_of(model, head, {**rows, "input_ids": ids})
        assert (moved[0, start] - base[0, start]).abs().max() < 1e-9, f"index {start} read the next token"
        assert (moved[0, start + 1] - base[0, start + 1]).abs().max() > 1e-3, "the change reached nothing"


@pytest.mark.parametrize("name", ["qwen3_5_moe_text", "nemotron_h"])
def test_a_root_head_keeps_the_embedding_at_a_document_start(name):
    """Neither engine's Nemotron-H drafter zeroes position 0, where both GLM-4's and DeepSeek's do."""
    model, head = tiny_model_with_head(name)
    install_term(model, head, weight=0.1, scope="detached")
    rows = packed_rows()
    base = _draft_of(model, head, rows)
    ids = rows["input_ids"].clone()
    ids[0, 1] = (ids[0, 1] + 1) % 90 + 4
    assert (_draft_of(model, head, {**rows, "input_ids": ids})[0, 0] - base[0, 0]).abs().max() > 1e-3


@pytest.mark.parametrize("name", _MOE_TEXT_FAMILIES)
def test_no_draft_scores_or_routes_across_a_document_boundary(name):
    rows = packed_rows()
    soft_model, soft_head, _ = _arm(name, rows, weight=_WEIGHT, loss_type="soft", scope="detached")

    model, head = tiny_model_with_head(name)
    install_term(model, head, weight=0.0, scope="detached")
    hidden_seen, draft_seen = [], []
    handles = (
        head.register_forward_pre_hook(lambda _module, args: hidden_seen.append(args[1].detach())),
        head.register_forward_hook(lambda _module, _args, output: draft_seen.append(output)),
    )
    model(**rows)
    for handle in handles:
        handle.remove()
    model.zero_grad()
    weight = model.get_output_embeddings().weight.detach()
    labels, positions = rows["labels"], rows["position_ids"]
    keep = (labels[:, 2:] != LABEL_IGNORE) & (positions[:, 1:-1] != 0)
    teacher = F.softmax(F.linear(hidden_seen[0][:, 1:-1], weight), dim=-1)
    student = F.log_softmax(F.linear(draft_seen[0][:, :-2], weight), dim=-1)
    (_WEIGHT * (-(teacher * student).sum(-1) * keep).sum() / (labels[:, 1:] != LABEL_IGNORE).sum()).backward()
    assert _worst(_grads(soft_head), _grads(head)) < 1e-5


@pytest.mark.parametrize("length", [2, 1])
@pytest.mark.parametrize("loss_type", ["hard", "soft"])
def test_a_row_too_short_for_a_target_adds_nothing_and_still_runs_the_head(loss_type, length):
    """An expert-parallel group's backward needs the head's dispatch on every rank, target or not."""
    ids = padded_rows()["input_ids"][:, :length]
    plain, _ = tiny_model_with_head("qwen3_5_moe_text")
    expected = plain(input_ids=ids, labels=ids, use_cache=False).loss
    model, head = tiny_model_with_head("qwen3_5_moe_text")
    install_term(model, head, weight=_WEIGHT, loss_type=loss_type)
    seen = []
    head.register_forward_hook(lambda _module, _args, output: seen.append(output.shape))
    loss = model(input_ids=ids, labels=ids, use_cache=False).loss
    assert seen and seen[0][1] == length
    assert (loss == expected) or (loss.isnan() and expected.isnan())


@pytest.mark.parametrize("name", _FAMILIES)
def test_the_chain_is_the_one_a_server_drafts_with(name):
    """Both engines norm the embedding and the hidden, concatenate them in that order, project, run the layer and norm.
    A swapped pair trains on tiny weights, and a release's own weights read as no acceptance through it."""
    model, head = tiny_model_with_head(name)
    install_term(model, head, weight=_WEIGHT, scope="detached")
    spec = head._mtp_spec
    seen = {}

    def record(key: str):
        return lambda _module, args, output: seen.update({key: (args[0], output)})

    for key, path in (
        ("embedding", spec.embedding_norm),
        ("hidden", spec.hidden_norm),
        ("projection", spec.projection),
        ("output", spec.output_norm),
    ):
        head.get_submodule(path).register_forward_hook(record(key))
    head.register_forward_hook(lambda _module, args, output: seen.update(chain=(args, output)))
    model(**padded_rows())
    (embedding_input, hidden_input), draft = seen["chain"]
    assert torch.equal(seen["embedding"][0], embedding_input)
    assert torch.equal(seen["hidden"][0], hidden_input)
    assert not torch.equal(embedding_input, hidden_input), (
        "the two inputs are one tensor, so their order proves nothing"
    )
    assert torch.equal(seen["projection"][0], torch.cat([seen["embedding"][1], seen["hidden"][1]], dim=-1))
    assert torch.equal(seen["output"][1], draft)
    assert not torch.equal(seen["output"][0], seen["projection"][1]), (
        "the layer ran nothing between projection and norm"
    )


@pytest.mark.parametrize("name", _NEMOTRON_ROUTES)
def test_the_chain_is_the_arithmetic_both_engines_draft_with(name):
    """Recomputed by hand off the head's own modules: the projection opens the attention block, each block's residual
    folds before the next one's norm, no norm separates them, and the final norm is the last thing before the shared
    output embedding. A block run in the other order, or a dropped residual, trains and reads as no acceptance."""
    model, head = tiny_model_with_head(name)
    install_term(model, head, weight=_WEIGHT, scope="detached")
    seen = {}
    head.register_forward_hook(lambda _module, args, output: seen.update(args=args, draft=output))
    head.layers[0].mixer.register_forward_pre_hook(
        lambda _module, _args, kwargs: seen.update(attention_kwargs=kwargs), with_kwargs=True
    )
    model(**padded_rows())
    embedding, hidden = seen["args"]
    attention, moe = head.layers
    x0 = attention.eh_proj(torch.cat([attention.enorm(embedding), attention.hnorm(hidden)], dim=-1))
    call = {key: value for key, value in seen["attention_kwargs"].items() if key != "hidden_states"}
    x1 = x0 + attention.mixer(hidden_states=attention.norm(x0), **call)[0]
    x2 = x1 + moe.mixer(moe.norm(x1))
    assert torch.equal(moe.final_layernorm(x2), seen["draft"])
    assert not torch.equal(x1, x0) and not torch.equal(x2, x1), (
        "a block contributed nothing, so its order proves nothing"
    )
    assert not hasattr(attention, "final_layernorm") and not hasattr(moe, "eh_proj")


@pytest.mark.parametrize("name", _NEMOTRON_ROUTES)
def test_the_head_shares_the_trunks_output_embedding_past_its_final_norm(name):
    """Both engines project a draft through the target's lm_head, and the term scores through the one the trunk trains."""
    model, head = tiny_model_with_head(name)
    assert not [key for key in head.state_dict() if "lm_head" in key or "embeddings" in key]
    install_term(model, head, weight=_WEIGHT, scope="detached")
    draft_seen = []
    head.register_forward_hook(lambda _module, _args, output: draft_seen.append(output))
    loss = model(**padded_rows()).loss
    weight = model.get_output_embeddings().weight
    labels = padded_rows()["labels"]
    expected = F.cross_entropy(
        F.linear(draft_seen[0][:, :-2], weight).reshape(-1, weight.size(0)),
        labels[:, 2:].reshape(-1),
        ignore_index=LABEL_IGNORE,
        reduction="sum",
    )
    plain, _ = tiny_model_with_head(name)
    assert torch.isclose(
        (loss - plain(**padded_rows()).loss) * (labels[:, 1:] != LABEL_IGNORE).sum(), _WEIGHT * expected
    )


@pytest.mark.parametrize("name", _NEMOTRON_ROUTES)
def test_the_moe_block_takes_a_call_it_ignores_as_the_trunks_does(name):
    """The attention layer's mask and markers reach the attention block; a MoE mixer runs on the hidden state alone."""
    model, head = tiny_model_with_head(name)
    install_term(model, head, weight=_WEIGHT)
    seen = {}
    head.layers[0].mixer.register_forward_pre_hook(
        lambda _m, _a, kwargs: seen.update(attention=kwargs), with_kwargs=True
    )
    head.layers[1].mixer.register_forward_pre_hook(
        lambda _m, args, kwargs: seen.update(moe=(args, kwargs)), with_kwargs=True
    )
    rows = padded_rows()
    model(**rows)
    assert seen["attention"]["attention_mask"] is not None, "the attention block lost the trunk's mask"
    assert len(seen["moe"][0]) == 1 and not seen["moe"][1]


@pytest.mark.parametrize("name", _FAMILIES)
def test_the_head_takes_the_call_of_the_trunk_layer_running_its_attention(name):
    """A linear-attention layer's mask and position arguments are not a full-attention layer's."""
    model, head = tiny_model_with_head(name)
    install_term(model, head, weight=_WEIGHT)
    layer = decoder_layers(backbone_with_layers(model))[mtp_draft_layer_index(text_config(model.config))]
    assert getattr(layer, "self_attn", None) is not None or layer.block_type == "full_attention", (
        "the draft layer's kind has no full attention"
    )
    seen = {}
    layer.register_forward_pre_hook(lambda _m, args, kwargs: seen.update(trunk=(args[1:], kwargs)), with_kwargs=True)
    head.register_forward_pre_hook(lambda _m, args, kwargs: seen.update(head=(args[2:], kwargs)), with_kwargs=True)
    model(**padded_rows())
    trunk_args, trunk_kwargs = seen["trunk"]
    head_args, head_kwargs = seen["head"]
    assert all(a is b for a, b in zip(trunk_args, head_args, strict=True))
    assert trunk_kwargs.keys() == head_kwargs.keys() and all(
        trunk_kwargs[key] is head_kwargs[key] for key in trunk_kwargs
    )


@pytest.mark.parametrize("name", _FAMILIES)
def test_eval_scores_the_next_token_loss_alone(name):
    rows = padded_rows()
    plain, _ = tiny_model_with_head(name)
    model, head = tiny_model_with_head(name)
    install_term(model, head, weight=_WEIGHT, loss_type="soft")
    with torch.no_grad():
        assert model.eval()(**rows).loss == plain.eval()(**rows).loss


def test_a_forward_with_no_labels_runs_no_head():
    model, head = tiny_model_with_head("qwen3_5_moe_text")
    term = install_term(model, head, weight=_WEIGHT, loss_type="soft")
    calls = []
    head.register_forward_hook(lambda *_args: calls.append(1))
    model(input_ids=padded_rows()["input_ids"], use_cache=False)
    assert not calls
    assert term._embeddings is None and term._layer_args is None and term._layer_kwargs is None, (
        "a forward the head never runs for held the trunk layer's call until the next one"
    )


def test_the_hooks_read_the_trunk_layer_the_head_was_built_from_on_a_trunk_ending_in_another_kind():
    """A 6-layer Qwen3-Next ends in linear attention. Its head takes the class and call of layer 3, whose full attention
    takes arguments a linear layer's does not."""
    kinds = ["linear_attention"] * 3 + ["full_attention"] + ["linear_attention"] * 2
    model, head = tiny_model_with_head("qwen3_next", overrides={"num_hidden_layers": 6, "layer_types": kinds})
    install_term(model, head, weight=_WEIGHT)
    layers = decoder_layers(backbone_with_layers(model))
    assert getattr(layers[5], "self_attn", None) is None, "this trunk no longer ends in another kind"
    assert getattr(head.layers[0], "self_attn", None) is not None
    seen = {}
    layers[3].register_forward_pre_hook(lambda _m, args, kwargs: seen.update(trunk=kwargs), with_kwargs=True)
    head.register_forward_pre_hook(lambda _m, args, kwargs: seen.update(head=kwargs), with_kwargs=True)
    model(**padded_rows()).loss.backward()
    assert seen["trunk"] and all(seen["head"][key] is seen["trunk"][key] for key in seen["trunk"])
    assert next(head.parameters()).grad is not None


def _release_head_shapes(release: dict, spec) -> dict[str, list[int]]:
    """The release's tail as the live head holds it: per-expert keys fused, the trunk's copies left to the trunk."""
    config = release["config"]
    num_experts = config.get("num_experts") or config.get("n_routed_experts")
    copies = tuple(key for key, _embedding in spec.trunk_copies)
    expected = {}
    for key, shape in release["tail"].items():
        if key.endswith(copies):
            continue
        prefix, separator, name = key.partition(".experts.N.")
        if not separator:
            expected[key] = shape
        elif name == "gate_proj.weight":
            expected[f"{prefix}.experts.gate_up_proj"] = [num_experts, 2 * shape[0], shape[1]]
        elif name == "down_proj.weight":
            expected[f"{prefix}.experts.down_proj"] = [num_experts, *shape]
        elif f"{prefix}.experts.N.gate_proj.weight" not in release["tail"]:
            # A gateless bank stores ``up_proj`` apart.
            expected[f"{prefix}.experts.up_proj"] = [num_experts, *shape]
    return expected


@pytest.mark.parametrize("repo", sorted(_RELEASE_TAILS))
def test_the_head_holds_a_release_tail_key_for_key_and_shape_for_shape(repo):
    release = _RELEASE_TAILS[repo]
    config = CONFIG_MAPPING[release["config"]["model_type"]](**release["config"])
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    spec = require_mtp_head_spec(config)
    path = mtp_head_path(model, spec)
    live = {f"{path}.{key}": list(tensor.shape) for key, tensor in build_mtp_head(model, spec).state_dict().items()}
    release = {**release, "tail": {key.removeprefix(_SUPER_VL_TAIL_LEVEL): v for key, v in release["tail"].items()}}
    assert live == _release_head_shapes(release, spec)


@pytest.mark.parametrize(("repo", "keys"), [(_LIGHTNING, 270), (_SUPER_VL, 1040)])
def test_the_pinned_nemotron_tails_expand_to_the_release_key_counts(repo, keys):
    """Read off each release's safetensors headers, 16 and 18 spellings, so a dropped row is not a shorter pass."""
    release = _RELEASE_TAILS[repo]
    experts = release["config"]["n_routed_experts"]
    per_expert = [key for key in release["tail"] if ".experts.N." in key]
    assert len(per_expert) == 2
    assert len(release["tail"]) - len(per_expert) + len(per_expert) * experts == keys


def test_the_pinned_super_vl_tail_is_the_digest_pinned_release_index():
    expected = {re.sub(r"\.experts\.\d+\.", ".experts.N.", key) for key in nemotron_h_hub_keys() if ".mtp." in key}
    assert set(_RELEASE_TAILS[_SUPER_VL]["tail"]) == expected


def test_super_vls_composite_holds_its_tail_at_the_release_path():
    release = _RELEASE_TAILS[_SUPER_VL]
    config = NemotronH_Omni_Reasoning_V3_Config(
        text_config=NEMOTRON_H_RELEASE_TEXT_CONFIG,
        vision_config=NEMOTRON_H_RELEASE_VISION_CONFIG,
        vision_hidden_size=NEMOTRON_H_RELEASE_VISION_CONFIG["hidden_size"],
        projector_hidden_size=20480,
        image_token_id=18,
    )
    with init_empty_weights():
        model = NemotronH_Omni_Reasoning_V3(config)
    spec = require_mtp_head_spec(config)
    path = mtp_head_path(model, spec)
    assert path == "language_model.mtp"
    live = {f"{path}.{key}": list(tensor.shape) for key, tensor in build_mtp_head(model, spec).state_dict().items()}
    assert live == _release_head_shapes(release, spec)


def test_the_head_is_found_on_both_placements_and_absent_elsewhere():
    for name in ("qwen3_5_moe_text", "glm4_moe_lite"):
        model, head = tiny_model_with_head(name)
        assert find_mtp_head(model) is head
    bare = MTP_TINY_FAMILIES["qwen3_next"].build({})
    assert find_mtp_head(bare) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
