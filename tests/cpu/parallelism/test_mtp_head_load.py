#!/usr/bin/env python
"""The tail streams through the lazy planner onto a head-only shell: a release's bytes land at the release's keys under
every expert layout, a tail that is not this head is refused, and the EP patch leaves one capacity generation."""

import json
import re
import weakref
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import load_file
from transformers import Qwen3Config

from halo.checkpoint.parallel.ep import save_ep_gathered_checkpoint
from halo.checkpoint.parallel.sidecars import restore_balancing_biases, save_balancing_biases
from halo.distributed.expert_parallel.deepep import capacity as ep_capacity
from halo.distributed.expert_parallel.deepep import dispatcher as ep_dispatcher
from halo.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from halo.model_loading.parallel import loader
from halo.model_loading.parallel import mtp_head as mtp_head_module
from halo.model_loading.parallel.model_source import ModelSource
from halo.model_loading.parallel.mtp_head import attach_mtp_head
from halo.models.moe.balancing import BALANCING_BIASES_ATTR, iter_balancing_routers
from halo.models.mtp.head import find_mtp_head, mtp_head_path, mtp_trunk_copy_keys
from halo.models.mtp.spec import MtpHeadRequest, MtpObjective, require_mtp_head_spec, stored_tail_keys
from tests.common.ep_stubs import LoopbackBackend, no_ep_dispatcher
from tests.common.mtp_heads import MTP_TINY_FAMILIES, padded_rows, write_checkpoint_with_tail
from tests.common.parallelism import make_parallelism_config, single_process_ep_config, stubbed_rank_ep_config
from tests.common.tiny_models import tiny_family_model

_PER_EXPERT_KEY = re.compile(r"(?P<bank>.*\.experts\.)(?P<index>\d+)\.(?P<name>\w+)\.weight")
_EXPERTS = 8
_BANK_PATHS = {"qwen3_5_moe_text": ".layers.0.mlp.experts.", "glm4_moe_lite": ".mlp.experts."}
_ROUTED = ("qwen3_5_moe_text", "qwen3_5_moe", "qwen3_next", "glm4_moe_lite")


def _pc(ep_config=None, *, keep_fp32: bool = False) -> SimpleNamespace:
    return SimpleNamespace(fp32_non_ep_params=keep_fp32, ep_config=lambda: ep_config)


def _source(path) -> ModelSource:
    return ModelSource(str(path), None, str(path))


def _trunk(name: str, *, ep_config=None):
    torch.manual_seed(1)
    model = tiny_family_model(MTP_TINY_FAMILIES[name]).to(torch.bfloat16)
    if ep_config is not None:
        apply_ep_layers(model, ep_config)
    return model


def _attach(model, path, *, ep_config=None, objective: MtpObjective | None = None, source=None, **kwargs):
    objective = objective or MtpObjective(weight=1.0)
    return attach_mtp_head(
        model,
        MtpHeadRequest(objective, source),
        pc=_pc(ep_config, keep_fp32=kwargs.pop("keep_fp32", False)),
        source=_source(path),
        dtype=torch.bfloat16,
        wraps_experts=ep_config is not None,
        init_from_scratch=kwargs.pop("init_from_scratch", False),
        preserve_checkpoint_precision=kwargs.pop("preserve_checkpoint_precision", False),
    )


def _assert_head_holds(model, tail: dict[str, torch.Tensor]) -> None:
    state = model.state_dict()
    for key, expected in tail.items():
        assert key in state, key
        assert torch.equal(state[key].float(), expected.float()), key
    assert not [key for key, value in state.items() if value.is_meta]


def _gathered_tail(model, target) -> dict[str, torch.Tensor]:
    """What a gathered EP save writes for the head, expert tensors fused back: the release's keys and bytes are the claim
    an engine reads, and an EP layer holds its experts under names of its own."""
    context = SimpleNamespace(
        model=model,
        parallelism_config=SimpleNamespace(is_cp_mode=False),
        is_save_rank=True,
        training_checkpoint=False,
        merges_expert_lora=False,
        max_shard_size="4GB",
    )
    save_ep_gathered_checkpoint(context, str(target))
    path = mtp_head_path(model, require_mtp_head_spec(model.config))
    fused: dict[str, torch.Tensor] = {}
    banks: dict[str, dict[str, dict[int, torch.Tensor]]] = {}
    for key, value in load_file(target / "model.safetensors").items():
        if key.startswith(f"{path}."):
            per_expert = _PER_EXPERT_KEY.fullmatch(key)
            if per_expert is None:
                fused[key] = value
            else:
                bank = banks.setdefault(per_expert["bank"], {}).setdefault(per_expert["name"], {})
                bank[int(per_expert["index"])] = value
    for bank, names in banks.items():
        gate, up, down = (names[name] for name in ("gate_proj", "up_proj", "down_proj"))
        indices = sorted(gate)
        fused[f"{bank}gate_up_proj"] = torch.stack([torch.cat([gate[i], up[i]]) for i in indices])
        fused[f"{bank}down_proj"] = torch.stack([down[i] for i in indices])
    return fused


@pytest.mark.parametrize("per_expert", [True, False], ids=["per_expert", "fused"])
@pytest.mark.parametrize("name", _ROUTED)
def test_a_release_tail_lands_byte_equal_under_ep_layers(name, per_expert, tmp_path):
    ep_config = single_process_ep_config()
    tail = write_checkpoint_with_tail(name, tmp_path / "ckpt", per_expert=per_expert)
    model = _trunk(name, ep_config=ep_config)
    trunk_layers = len(named_ep_layers(model))
    head = _attach(model, tmp_path / "ckpt", ep_config=ep_config)
    assert find_mtp_head(model) is head
    assert len(named_ep_layers(model)) == trunk_layers + 1, "the EP patch did not claim the head's MoE block"
    gathered = _gathered_tail(model, tmp_path / "export")
    # The trunk copies a release stores are the trained embeddings' to write, not the tail's.
    copies = {copied for keys in mtp_trunk_copy_keys(model).values() for copied in keys}
    assert set(gathered) == set(tail) | copies
    for key, expected in tail.items():
        assert torch.equal(gathered[key].float(), expected.float()), key


@pytest.mark.parametrize("per_expert", [True, False], ids=["per_expert", "fused"])
@pytest.mark.parametrize("name", sorted(MTP_TINY_FAMILIES))
def test_a_release_tail_lands_byte_equal_with_no_ep_layer(name, per_expert, tmp_path):
    tail = write_checkpoint_with_tail(name, tmp_path / "ckpt", per_expert=per_expert)
    model = _trunk(name)
    _attach(model, tmp_path / "ckpt")
    _assert_head_holds(model, tail)
    assert not named_ep_layers(model)


_EP_WORLD = 2


@pytest.mark.parametrize("per_expert", [True, False], ids=["per_expert", "fused"])
@pytest.mark.parametrize("rank", range(_EP_WORLD))
@pytest.mark.parametrize("name", ["qwen3_5_moe_text", "glm4_moe_lite"])
def test_each_rank_holds_its_own_slice_of_the_tails_expert_bank(name, rank, per_expert, tmp_path):
    tail = write_checkpoint_with_tail(name, tmp_path / "ckpt", per_expert=per_expert)
    ep_config = stubbed_rank_ep_config(_EP_WORLD, _EP_WORLD, rank, _EXPERTS)
    model = _trunk(name)
    with no_ep_dispatcher():
        head = _attach(model, tmp_path / "ckpt", ep_config=ep_config)
    start, end = ep_config.expert_start_idx, ep_config.expert_end_idx
    assert end - start == _EXPERTS // _EP_WORLD and start == rank * (end - start)
    path = mtp_head_path(model, require_mtp_head_spec(model.config))
    [(_, layer)] = named_ep_layers(head)
    held = dict(layer.expert_named_params())
    for name_in_tail in ("gate_up_proj", "down_proj"):
        full = tail[f"{path}{_BANK_PATHS[name]}{name_in_tail}"]
        assert full.shape[0] == _EXPERTS
        # The layer keeps a bank in its contraction order, which the gather turns back.
        slice_held, slice_stored = held[name_in_tail].float(), full[start:end].float()
        assert torch.equal(slice_held, slice_stored) or torch.equal(slice_held.mT, slice_stored), name_in_tail


def test_the_head_trains_at_the_run_dtype_with_fp32_masters_kept_where_asked(tmp_path):
    write_checkpoint_with_tail("qwen3_5_moe_text", tmp_path / "ckpt", per_expert=True)
    bf16 = find_mtp_head(_with_attached("qwen3_5_moe_text", tmp_path / "ckpt"))
    assert {param.dtype for param in bf16.parameters()} == {torch.bfloat16}
    fp32 = find_mtp_head(_with_attached("qwen3_5_moe_text", tmp_path / "ckpt", keep_fp32=True))
    assert fp32.fc.weight.dtype == torch.float32, "fp32_non_ep_params must reach the head's masters"


def _with_attached(name, path, **kwargs):
    model = _trunk(name)
    _attach(model, path, **kwargs)
    return model


def _drop_one(stored: dict) -> None:
    del stored[next(key for key in stored if key.endswith("norm.weight"))]


def _add_one(stored: dict) -> None:
    stored[next(iter(stored)).rsplit(".", 1)[0] + ".stray.weight"] = torch.ones(2)


def _drop_one_expert(stored: dict) -> None:
    del stored[next(key for key in stored if ".experts.3.gate_proj" in key)]


@pytest.mark.parametrize(
    ("edit", "per_expert", "match"),
    [
        (_drop_one, True, "lacks 1 tensor"),
        (_add_one, True, "adds 1"),
        (_drop_one_expert, True, "its two GLU halves|absent from it"),
        (lambda stored: stored.clear(), True, "stores no MTP head"),
    ],
    ids=["partial", "extra", "expert_half_missing", "absent"],
)
@pytest.mark.parametrize("name", ["qwen3_5_moe_text", "glm4_moe_lite"])
def test_a_tail_that_is_not_this_head_is_refused_by_name(name, edit, per_expert, match, tmp_path):
    write_checkpoint_with_tail(name, tmp_path / "ckpt", per_expert=per_expert, edit=edit)
    model = _trunk(name)
    with pytest.raises(RuntimeError, match=match):
        _attach(model, tmp_path / "ckpt")
    assert find_mtp_head(model) is None


def test_a_refusal_names_the_field_that_fixes_it(tmp_path):
    write_checkpoint_with_tail(
        "qwen3_5_moe_text", tmp_path / "ckpt", per_expert=True, edit=lambda stored: stored.clear()
    )
    with pytest.raises(RuntimeError, match="mtp_head_source"):
        _attach(_trunk("qwen3_5_moe_text"), tmp_path / "ckpt")


def test_a_quantized_source_is_refused(tmp_path):
    write_checkpoint_with_tail("qwen3_5_moe_text", tmp_path / "ckpt", per_expert=True)
    config = tmp_path / "ckpt" / "config.json"
    config.write_text(json.dumps({**json.loads(config.read_text()), "quantization_config": {"quant_method": "any"}}))
    with pytest.raises(RuntimeError, match="stores quantized weights"):
        _attach(_trunk("qwen3_5_moe_text"), tmp_path / "ckpt")


def test_a_wrong_width_source_is_refused_by_the_shape_check(tmp_path):
    def narrower(stored):
        key = next(key for key in stored if key.endswith("fc.weight"))
        stored[key] = stored[key][:-1]

    write_checkpoint_with_tail("qwen3_5_moe_text", tmp_path / "ckpt", per_expert=True, edit=narrower)
    with pytest.raises(RuntimeError, match="has shape"):
        _attach(_trunk("qwen3_5_moe_text"), tmp_path / "ckpt")


def test_mtp_head_source_wins_over_the_models_own_tail_and_agrees_through_the_source_resolver(tmp_path, monkeypatch):
    write_checkpoint_with_tail("qwen3_5_moe_text", tmp_path / "own", per_expert=True, seed=1)
    tail = write_checkpoint_with_tail("qwen3_5_moe_text", tmp_path / "other", per_expert=True, seed=2)
    resolved = []

    def resolve(name, revision, *, tag):
        resolved.append((name, tag))
        return _source(name)

    monkeypatch.setattr(mtp_head_module, "resolve_model_source", resolve)
    model = _trunk("qwen3_5_moe_text")
    _attach(model, tmp_path / "own", source=str(tmp_path / "other"))
    _assert_head_holds(model, tail)
    assert resolved == [(str(tmp_path / "other"), "mtp_head")]


def test_a_resume_from_a_checkpoint_keeps_its_trained_head_and_says_the_source_is_ignored(
    tmp_path, monkeypatch, caplog
):
    tail = write_checkpoint_with_tail("qwen3_5_moe_text", tmp_path / "ckpt", per_expert=True, seed=1)
    write_checkpoint_with_tail("qwen3_5_moe_text", tmp_path / "other", per_expert=True, seed=2)
    monkeypatch.setattr(mtp_head_module, "resolve_model_source", lambda *_args, **_kwargs: pytest.fail("not resolved"))
    model = _trunk("qwen3_5_moe_text")
    _attach(model, tmp_path / "ckpt", source=str(tmp_path / "other"), preserve_checkpoint_precision=True)
    _assert_head_holds(model, tail)
    assert any("is ignored on this resume" in record.getMessage() for record in caplog.records)


def _from_scratch(name, path, *, ep_config=None):
    torch.manual_seed(1)
    model = tiny_family_model(MTP_TINY_FAMILIES[name]).float()
    if ep_config is not None:
        apply_ep_layers(model, ep_config)
    head = attach_mtp_head(
        model,
        MtpHeadRequest(MtpObjective(weight=1.0)),
        pc=_pc(ep_config),
        source=_source(path),
        dtype=torch.float32,
        wraps_experts=ep_config is not None,
        init_from_scratch=True,
        preserve_checkpoint_precision=False,
    )
    return model, head


def _assert_every_parameter_trains(model, head) -> None:
    model.train()
    model(**padded_rows()).loss.backward()
    untrained = [n for n, p in head.named_parameters() if p.grad is None or p.grad.abs().sum() == 0]
    assert not untrained, untrained
    assert all(torch.isfinite(param).all() for param in head.parameters())


@pytest.mark.parametrize("name", sorted(MTP_TINY_FAMILIES))
def test_a_from_scratch_head_draws_every_parameter_with_the_familys_own_init(name, tmp_path):
    """A directly built head leaves the gate and both expert tensors at exactly zero gradient."""
    _assert_every_parameter_trains(*_from_scratch(name, tmp_path))


@pytest.mark.parametrize("name", _ROUTED)
def test_a_from_scratch_head_trains_every_parameter_through_its_ep_layer(name, tmp_path):
    model, head = _from_scratch(name, tmp_path, ep_config=single_process_ep_config())
    assert named_ep_layers(head), "the EP patch did not claim the head's MoE block"
    _assert_every_parameter_trains(model, head)


def test_a_from_scratch_head_is_the_same_on_every_rank_given_one_stream(tmp_path):
    first = _from_scratch("qwen3_5_moe_text", tmp_path)[1].state_dict()
    second = _from_scratch("qwen3_5_moe_text", tmp_path)[1].state_dict()
    for key, value in first.items():
        assert torch.equal(value, second[key]), key


def _loopback(model, log) -> None:
    for _, layer in named_ep_layers(model):
        layer.dispatcher._noop = False
        layer.dispatcher.backend = LoopbackBackend(layer.dispatcher, log)


@pytest.fixture
def loopback_transport(monkeypatch):
    monkeypatch.setattr(ep_dispatcher, "deep_ep", lambda: SimpleNamespace(topk_idx_t=torch.int64))
    monkeypatch.setattr(ep_dispatcher, "_live_dispatchers", weakref.WeakValueDictionary())


def _head_ops(model, log) -> list[str]:
    head_dispatchers = {layer.dispatcher for _path, layer in named_ep_layers(find_mtp_head(model))}
    assert head_dispatchers, "the head holds no EP layer, so its ops would count nothing"
    return [op for op, dispatcher, _handle in log if dispatcher in head_dispatchers]


@pytest.mark.parametrize("name", _ROUTED)
def test_a_called_head_dispatches_once_and_keeps_one_capacity_generation(name, tmp_path, loopback_transport):
    """A second generation would size the head's arena apart from the first MoE layer's all-reduced capacity."""
    ep_config = single_process_ep_config()
    write_checkpoint_with_tail(name, tmp_path / "ckpt", per_expert=True)
    model = _trunk(name, ep_config=ep_config).float()
    _attach(model, tmp_path / "ckpt", ep_config=ep_config)
    model.float().train()
    log: list = []
    _loopback(model, log)
    before = ep_capacity.capacity_scope.generation
    model(**padded_rows()).loss.backward()
    assert ep_capacity.capacity_scope.generation - before == 1
    assert _head_ops(model, log).count("dispatch_forward") == 1
    assert _head_ops(model, log).count("dispatch_backward") == 1


@pytest.mark.parametrize("loss_type", ["hard", "soft"])
def test_head_only_freezes_the_trunk_and_only_the_head_reverses_its_dispatch(loss_type, tmp_path, loopback_transport):
    ep_config = single_process_ep_config()
    name = "qwen3_5_moe_text"
    write_checkpoint_with_tail(name, tmp_path / "ckpt", per_expert=True)
    model = _trunk(name, ep_config=ep_config).float()
    head = _attach(
        model,
        tmp_path / "ckpt",
        ep_config=ep_config,
        objective=MtpObjective(1.0, loss_type=loss_type, scope="head_only"),
    )
    model.float().train()
    log: list = []
    _loopback(model, log)
    head_ids = {id(param) for param in head.parameters()}
    assert all(param.requires_grad == (id(param) in head_ids) for param in model.parameters())
    model(**padded_rows()).loss.backward()
    assert not [n for n, p in model.named_parameters() if id(p) not in head_ids and p.grad is not None]
    reversing = {dispatcher for op, dispatcher, _handle in log if op == "dispatch_backward"}
    assert len(reversing) == 1
    assert _head_ops(model, log).count("dispatch_backward") == 1


def _load_recording_order(monkeypatch, request_value) -> list[str]:
    """The real ``load_distributed_model`` over a stopped per-mode load, recording what touches the loaded model."""
    order: list[str] = []
    model = SimpleNamespace(config=SimpleNamespace())
    pc = make_parallelism_config(world_size=1, gpus_per_node=1)
    monkeypatch.setattr(
        loader, "resolve_model_source", lambda path, revision, **kwargs: ModelSource(path, revision, path)
    )
    monkeypatch.setattr(loader.AutoConfig, "from_pretrained", lambda *args, **kwargs: Qwen3Config())
    monkeypatch.setattr(loader.AutoTokenizer, "from_pretrained", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(loader, "reject_unsupported_structure", lambda *args, **kwargs: None)
    monkeypatch.setattr(loader, "_dispatch_model_loading", lambda *args, **kwargs: order.append("trunk") or model)
    monkeypatch.setattr(loader, "attach_mtp_head", lambda *args, **kwargs: order.append("head"))
    monkeypatch.setattr(loader, "apply_run_model_policies", lambda *args, **kwargs: order.append("policies"))
    monkeypatch.setattr(loader, "warm_attention_kernels", lambda *args, **kwargs: order.append("warmup"))
    loader.load_distributed_model(
        "org/base", pc, attn_implementation="eager", trust_remote_code=False, mtp_head=request_value
    )
    return order


def test_no_request_never_reaches_the_seam(monkeypatch):
    """At weight 0 the trunk's load runs its own path and nothing of the head's is called."""
    assert _load_recording_order(monkeypatch, None) == ["trunk", "policies", "warmup"]


def test_a_request_attaches_right_after_the_trunk_and_before_the_walks_over_its_modules(monkeypatch):
    request = MtpHeadRequest(MtpObjective(weight=1.0))
    assert _load_recording_order(monkeypatch, request) == ["trunk", "head", "policies", "warmup"]


@pytest.mark.parametrize("name", sorted(MTP_TINY_FAMILIES))
def test_a_trunk_with_no_head_has_no_key_or_hook_of_the_heads(name):
    torch.manual_seed(1)
    model = tiny_family_model(MTP_TINY_FAMILIES[name])
    assert find_mtp_head(model) is None
    assert not model._forward_hooks and not model._forward_pre_hooks
    assert not stored_tail_keys(model.state_dict(), getattr(type(model), "_keys_to_ignore_on_load_unexpected", ()))


def test_bias_balancing_takes_the_heads_gate_and_the_sidecar_restores_it_by_name(tmp_path):
    """The head's slot stays the release's key, so an export carries it where its engine reads it."""
    ep_config = single_process_ep_config()
    write_checkpoint_with_tail("glm4_moe_lite", tmp_path / "ckpt", per_expert=True)
    model = _trunk("glm4_moe_lite", ep_config=ep_config)
    head = _attach(model, tmp_path / "ckpt", ep_config=ep_config)
    keys = set(model.state_dict())
    trunk_layers = len(named_ep_layers(model)) - 1
    for module in model.modules():
        if callable(getattr(module, "enable_bias_balancing", None)):
            module.enable_bias_balancing()
    assert set(model.state_dict()) == keys
    assert len(list(iter_balancing_routers(model))) == trunk_layers + 1
    router = next(iter(iter_balancing_routers(head)))
    bias = getattr(router, BALANCING_BIASES_ATTR)
    with torch.no_grad():
        bias.fill_(0.25)
    save_balancing_biases(model, str(tmp_path), is_pp_mode=False)
    with torch.no_grad():
        bias.zero_()
    restore_balancing_biases(model, str(tmp_path), is_pp_mode=False)
    assert torch.all(getattr(router, BALANCING_BIASES_ATTR) == 0.25)


def test_the_trunk_copies_are_read_from_the_trunk_and_not_loaded_into_the_head(tmp_path):
    tail = write_checkpoint_with_tail("glm4_moe_lite", tmp_path / "ckpt", per_expert=True)
    model = _trunk("glm4_moe_lite")
    head = _attach(model, tmp_path / "ckpt")
    spec = require_mtp_head_spec(model.config)
    assert spec.trunk_copies, "the row this test pins went away"
    assert not any(key.endswith(copy) for key in head.state_dict() for copy, _ in spec.trunk_copies)
    path = mtp_head_path(model, spec)
    stored = load_file(tmp_path / "ckpt" / "model.safetensors")
    assert {f"{path}.{copy}" for copy, _ in spec.trunk_copies} <= set(stored)
    _assert_head_holds(model, tail)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
