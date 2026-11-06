#!/usr/bin/env python
"""Configuration knobs reach the code they claim to control: each test asserts an observable difference between the
knob on and off, or that an unsupported knob raises."""

import ast
import dataclasses
import importlib
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from datasets import Dataset, DatasetDict
from trl import SFTConfig

from halo.configs.distillation import PRIVILEGED_HINT_TEMPLATE, SDPGArguments, SelfDistillScriptArguments
from halo.configs.grpo import RLRRConfig
from halo.configs.grpo_environmental import AsyncRolloutConfig, ISRMaskConfig
from halo.configs.grpo_online import OnlineGRPOScriptArguments
from halo.configs.parallelism_arguments import ParallelismArguments, parallelism_config_from_args
from halo.entrypoints.data_setup import reject_trl_dataset_prep_args
from halo.entrypoints.script import distributed_trainer_kwargs
from halo.trainers.distillation.divergences import generalized_jsd_loss
from halo.trainers.distillation.sdpg import DistributedSDPGTrainer
from halo.trainers.distillation.self_distillation import DistributedSelfDistillationTrainer
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from halo.trainers.grpo.online import DistributedOnlineGRPOTrainer
from halo.trainers.optimizer_selection import resolve_bf16_optimizer
from halo.trainers.preference.trl_preference import TRLPreferenceMixin
from halo.trainers.sft import DistributedSFTTrainer
from halo.trainers.trainer_rules import reject_stock_adamw_on_mixed_params
from tests.common.repo import REPO_ROOT, load_script_module


@pytest.mark.parametrize("requested", [None, True, False])
def test_bf16_optimizer_reaches_parallelism_config(requested):
    """ParallelismConfig is the only object every entry script threads into every trainer, so a drop here makes the
    knob unreachable."""
    config = parallelism_config_from_args(
        ParallelismArguments(bf16_optimizer=requested), trainer_cls=DistributedTrainerMixin
    )
    assert config.bf16_optimizer is requested


def _resolve_bf16_optimizer(*, config_value, bf16=True, optim="adamw_torch_fused", ctor_value=None):
    config = SimpleNamespace(bf16_optimizer=config_value)
    return resolve_bf16_optimizer(ctor_value, config, SimpleNamespace(bf16=bf16, optim=optim), replicated_ddp=False)


def test_bf16_optimizer_config_true_forces_on_where_auto_would_disable():
    """optim != the default AdamW makes the auto path pick False; an explicit config True must win."""
    assert _resolve_bf16_optimizer(config_value=None, optim="adamw_bnb_8bit") is False
    assert _resolve_bf16_optimizer(config_value=True, optim="adamw_bnb_8bit") is True


def test_bf16_optimizer_config_false_forces_off_where_auto_would_enable():
    assert _resolve_bf16_optimizer(config_value=None) is True
    assert _resolve_bf16_optimizer(config_value=False) is False


def test_bf16_optimizer_ctor_kwarg_wins_over_config():
    """Tests/benchmarks construct trainers directly; their explicit kwarg must beat the YAML value."""
    assert _resolve_bf16_optimizer(config_value=True, ctor_value=False) is False
    assert _resolve_bf16_optimizer(config_value=False, ctor_value=True) is True


def _mixed_precision_config(*, ep_group_size: int, bf16_optimizer=False, fp32_non_ep_params: bool = False):
    return SimpleNamespace(
        bf16_optimizer=bf16_optimizer,
        ep_group_size=ep_group_size,
        needs_ep_layers=True,
        experts_fsdp_managed=False,
        fp32_non_ep_params=fp32_non_ep_params,
    )


def _optimizer_build_stub(
    *, ep_group_size: int, experts_fsdp_managed: bool = False, moe: bool = True, model_type: str = "qwen3_moe"
):
    """Stub carrying exactly what the optimizer-build refusal reads: the parallelism shape and the
    model config it resolves the family's EP layer class from."""
    return SimpleNamespace(
        parallelism_config=SimpleNamespace(
            ep_group_size=ep_group_size,
            needs_ep_layers=True,
            experts_fsdp_managed=experts_fsdp_managed,
            fp32_non_ep_params=False,
        ),
        model=SimpleNamespace(
            config=SimpleNamespace(num_local_experts=8, model_type=model_type)
            if moe
            else SimpleNamespace(hidden_size=8, model_type="llama")
        ),
    )


def test_stock_adamw_refused_where_plain_experts_meet_dtensor_peers():
    """``aten._fused_adamw_`` raises 'mixed torch.Tensor and DTensor' over that set, so the build refuses first,
    when EP distributes the experts and when a grouped-GEMM ep1 run leaves them FSDP-unmanaged."""
    for stub in (
        _optimizer_build_stub(ep_group_size=8),
        _optimizer_build_stub(ep_group_size=1, experts_fsdp_managed=False),
    ):
        for optim in ("adamw_torch_fused", "adamw_torch"):
            with pytest.raises(ValueError, match="mixed torch.Tensor and DTensor"):
                reject_stock_adamw_on_mixed_params(optim, stub.parallelism_config, stub.model.config)


def test_stock_adamw_allowed_where_every_param_is_a_dtensor():
    """The refusal is about a plain/DTensor mix. ``fsdp_shard_ep1_experts`` shards the experts at ``ep_group_size ==
    1``, a dense model has none, and a MoE with no registered EP layer class (Mixtral) keeps ordinary FSDP2
    DTensors."""
    for stub in (
        _optimizer_build_stub(ep_group_size=1, experts_fsdp_managed=True),
        _optimizer_build_stub(ep_group_size=1, moe=False),
        _optimizer_build_stub(ep_group_size=1, model_type="mixtral"),
    ):
        reject_stock_adamw_on_mixed_params("adamw_torch_fused", stub.parallelism_config, stub.model.config)


def test_named_optimizers_are_not_refused_by_the_stock_adamw_guard():
    """Muon/FlashAdamW/bnb are per-parameter and reach their own builders, which handle the mixed set."""
    stub = _optimizer_build_stub(ep_group_size=8)
    for optim in ("muon", "flash_adamw", "adamw_bnb_8bit"):
        reject_stock_adamw_on_mixed_params(optim, stub.parallelism_config, stub.model.config)


def test_fp32_on_a_moe_builds_the_trainer_and_only_refuses_at_the_optimizer():
    """Forward, backward and the EP-aware clip are correct over the mixed set; only building the stock optimizer is
    the defect."""
    config = _mixed_precision_config(ep_group_size=8, bf16_optimizer=None)
    args = SimpleNamespace(bf16=False, optim="adamw_torch_fused")
    assert resolve_bf16_optimizer(None, config, args, replicated_ddp=False) is False


def test_create_optimizer_wires_the_refusal_before_building_the_stock_optimizer():
    """The probe is a real subclass instance (``__new__``, no ``__init__``) so the method's ``super()`` resolves;
    reaching it raises ``AttributeError``, which is how the dense case proves the guard let it through."""

    class _Probe(DistributedTrainerMixin):
        def _ep_layers(self):  # shadows the base walk over self.model
            return []

    def _build(*, moe: bool):
        trainer = _Probe.__new__(_Probe)
        stub = _optimizer_build_stub(ep_group_size=8 if moe else 1, moe=moe)
        trainer.optimizer = None
        trainer._bf16_optimizer = False
        trainer.args = SimpleNamespace(optim="adamw_torch_fused")
        trainer.get_decay_parameter_names = lambda model: []
        trainer.get_optimizer_cls_and_kwargs = lambda *args: None
        trainer.model = stub.model
        trainer.model.modules = lambda: []  # the FSDP2 re-registration seam walks modules; the stub is no nn.Module
        trainer.parallelism_config = stub.parallelism_config
        return trainer

    with pytest.raises(ValueError, match="mixed torch.Tensor and DTensor"):
        _build(moe=True).create_optimizer()
    with pytest.raises(AttributeError):  # guard passed: the probe has no real Trainer above it
        _build(moe=False).create_optimizer()


def test_explicit_bf16_optimizer_false_selects_the_stock_path_without_raising_at_construction():
    """The unsupported combination is caught by the build guard above, not at construction."""
    config = _mixed_precision_config(ep_group_size=8, bf16_optimizer=False)
    args = SimpleNamespace(bf16=True, optim="adamw_torch_fused")
    assert resolve_bf16_optimizer(None, config, args, replicated_ddp=False) is False


def test_fp32_non_ep_params_is_the_remedy_the_message_names_and_is_not_itself_refused():
    """The refusal names ``fp32_non_ep_params``, which routes ``create_optimizer`` to the tensor-type-grouped
    builder; construction must not block the fix it names."""
    config = _mixed_precision_config(ep_group_size=8, fp32_non_ep_params=True, bf16_optimizer=None)
    args = SimpleNamespace(bf16=False, optim="adamw_torch_fused")
    assert resolve_bf16_optimizer(None, config, args, replicated_ddp=False) is False


def _sft_config(**overrides) -> SFTConfig:
    return SFTConfig(output_dir="unused", bf16=False, use_cpu=True, **overrides)


@pytest.mark.parametrize(
    ("overrides", "knob"),
    [
        # Tri-state: an explicit False is as unhonorable a request as True.
        ({"completion_only_loss": False}, "completion_only_loss"),
        ({"assistant_only_loss": True}, "assistant_only_loss"),
        ({"dataset_text_field": "messages"}, "dataset_text_field"),
        ({"dataset_kwargs": {"add_special_tokens": False}}, "dataset_kwargs"),
        ({"eval_packing": True}, "eval_packing"),
        ({"chat_template_path": "template.jinja"}, "chat_template_path"),
    ],
)
def test_trl_dataset_prep_knobs_are_refused(overrides, knob):
    config = _sft_config(**overrides)
    with pytest.raises(ValueError, match=rf"does not support these config fields.*{knob}"):
        reject_trl_dataset_prep_args("Some script", config, "eval_packing")


def test_trl_dataset_prep_defaults_pass():
    reject_trl_dataset_prep_args("Some script", _sft_config(), "eval_packing")


def test_each_trl_dataset_prep_refusal_names_its_own_way_out():
    """No refusal message carries another's advice."""
    with pytest.raises(ValueError) as masked:
        reject_trl_dataset_prep_args("Some script", _sft_config(assistant_only_loss=True), render_remedy="REMEDY")
    assert "train_on_completions_only" in str(masked.value) and "REMEDY" not in str(masked.value)

    with pytest.raises(ValueError) as rendered:
        reject_trl_dataset_prep_args("Some script", _sft_config(dataset_text_field="messages"), render_remedy="REMEDY")
    assert "REMEDY" in str(rendered.value) and "train_on_completions_only" not in str(rendered.value)

    with pytest.raises(ValueError) as templated:
        reject_trl_dataset_prep_args(
            "Some script", _sft_config(chat_template_path="template.jinja"), render_remedy="REMEDY"
        )
    message = str(templated.value)
    assert "force_chat_template: true" in message and "REMEDY" not in message
    assert "train_on_completions_only" not in message


def _rejected_knobs(script_path: str) -> set[str]:
    """Knob names passed to ``reject_unsupported_args`` with a value read off a parsed config. The guard filters on
    truthiness, so a literal (``assistant_only_loss=None``) keeps the keyword and makes it unfirable."""
    tree = ast.parse((REPO_ROOT / script_path).read_text())
    rejected: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "reject_unsupported_args":
            rejected.update(
                kw.arg
                for kw in node.keywords
                if kw.arg and any(isinstance(sub, ast.Attribute) for sub in ast.walk(kw.value))
            )
    return rejected


@pytest.mark.parametrize(
    ("script", "fields"),
    [
        ("scripts/training/preference/kto.py", {"tools_field", "log_decoded_samples"}),
        ("scripts/training/reward/reward_model.py", {"log_decoded_samples", "text_only_model", "tools_field"}),
        ("scripts/training/preference/dpo.py", {"tools_field"}),
        ("scripts/training/preference/smpo.py", {"tools_field"}),
        ("scripts/training/classification.py", {"text_only_model"}),
        ("scripts/training/grpo/environmental.py", {"tools_field", "text_only_model"}),
        ("scripts/training/grpo/online.py", {"text_only_model"}),
        (
            "scripts/training/embedding.py",
            {
                "pad_token",
                "chat_template",
                "freeze_layers_patterns",
                "tools_field",
                "log_decoded_samples",
                "text_only_model",
            },
        ),
    ],
)
def test_script_rejects_knobs_it_cannot_honor(script, fields):
    """These knobs parse on the listed script but nothing consumes them."""
    assert fields <= _rejected_knobs(script), f"{script} no longer rejects {sorted(fields - _rejected_knobs(script))}"


def _run_script_main(script: str, config_body: str, tmp_path: Path) -> None:
    """Drive ``scripts/training/<script>:main()`` from a minimal YAML, with nothing past the parse stubbed."""
    module = load_script_module(f"scripts/training/{script}", "halo_test_knob_wiring_" + script.replace("/", "_"))
    config = tmp_path / "config.yaml"
    config.write_text(
        f"model_name_or_path: stub/qwen3-4b\ndataset:\n- dummy/dataset\noutput_dir: {tmp_path / 'out'}\n"
        f"bf16: false\nuse_cpu: true\n{config_body}"
    )
    with (
        mock.patch("halo.entrypoints.script.install_log_tee"),
        mock.patch.object(sys, "argv", ["prog", str(config)]),
    ):
        module.main()


@pytest.mark.parametrize(
    ("script", "config_body", "knob"),
    [
        # The text-only CausalLM spells its decoder model.layers.* while the served multimodal
        # checkpoint spells model.language_model.layers.*; the sync forwards names verbatim.
        ("grpo/online.py", "text_only_model: true\n", "text_only_model"),
        ("grpo/environmental.py", "text_only_model: true\nenvironment_type: react_math\n", "text_only_model"),
        # TRL's own dataset prep and default collator, the only consumers, are both replaced.
        ("sft.py", "assistant_only_loss: true\n", "assistant_only_loss"),
        ("distillation/self_distill.py", "completion_only_loss: false\n", "completion_only_loss"),
        # TRL's trainer re-templates the processing class after the script rendered every row, so the
        # checkpoint would export a template the model never trained on.
        ("sft.py", "chat_template_path: template.jinja\n", "chat_template_path"),
        ("distillation/self_distill.py", "chat_template_path: template.jinja\n", "chat_template_path"),
        # Both only feed the reference term, and no reference model is loaded at this coefficient.
        (
            "distillation/self_distill.py",
            "reference_kl_coef: 0.0\nreference_model_name_or_path: stub/reference\n",
            "reference_model_name_or_path",
        ),
        (
            "distillation/self_distill.py",
            "reference_kl_coef: 0.0\nreference_kl_loss: reverse_kl\n",
            "reference_kl_loss",
        ),
        # Both only shape the confidence weights, which no row carries without the column.
        ("distillation/self_distill.py", "confidence_power: 2.0\n", "confidence_power"),
        ("distillation/self_distill.py", "confidence_weight_opd: false\n", "confidence_weight_opd"),
        # SentenceTransformer tokenizes, so another text→ids backend would be read by nothing.
        ("embedding.py", "tokenizer_backend: gigatoken\n", "tokenizer_backend"),
    ],
)
def test_script_main_refuses_the_knob_before_any_load(script, config_body, knob, tmp_path):
    """A rejection firing after a multi-node load has already spent the cluster's time on a run it refuses."""
    with pytest.raises(ValueError, match=rf"does not support these config fields.*{knob}"):
        _run_script_main(script, config_body, tmp_path)


# The helper the scripts splat their distributed knobs through, and the knobs it owes them.
_BUNDLE_HELPER = "distributed_trainer_kwargs"
_BUNDLED_KNOBS = frozenset(
    {
        "parallelism_config",
        "moe_balancing",
        "save_sharded_ep",
        "save_max_shard_size",
        "allow_optimizer_warm_restart",
        "max_consecutive_nonfinite_steps",
        "dataset_presharded",
    }
)


def _splats_the_bundle(call: ast.Call) -> bool:
    return any(
        kw.arg is None and isinstance(kw.value, ast.Call) and getattr(kw.value.func, "id", None) == _BUNDLE_HELPER
        for kw in call.keywords
    )


def _trainer_call(script: Path) -> ast.Call | None:
    """The script's trainer construction, or ``None``; identified by the distributed bundle splat or a hand-spelled
    ``parallelism_config=``."""
    for node in ast.walk(ast.parse(script.read_text())):
        if not isinstance(node, ast.Call):
            continue
        if _splats_the_bundle(node) or any(kw.arg == "parallelism_config" for kw in node.keywords):
            return node
    return None


def _forwards_moe_balancing(script: Path) -> bool:
    """Whether the trainer receives the parsed ``moe_balancing``; ``moe_balancing=None`` keeps the keyword present
    and restores the silent downgrade."""
    call = _trainer_call(script)
    if call is None:
        return False
    for kw in call.keywords:
        if kw.arg == "moe_balancing":
            return isinstance(kw.value, ast.Attribute)
    return _splats_the_bundle(call)


_TRAINER_SCRIPTS = sorted(
    path for path in (REPO_ROOT / "scripts" / "training").rglob("*.py") if _trainer_call(path) is not None
)


def test_every_training_script_forwards_moe_balancing_to_its_trainer():
    """``_disable_inert_router_logits`` decides raise-vs-warn from ``self._moe_balancing``, which defaults to
    ``"auto"``, so a script omitting the kwarg downgrades that rejection to a warning: ``aux_loss`` pays the
    router-logit cost and balances nothing."""
    assert _TRAINER_SCRIPTS, "no training script matched the trainer-construction scan"
    missing = [
        str(script.relative_to(REPO_ROOT)) for script in _TRAINER_SCRIPTS if not _forwards_moe_balancing(script)
    ]
    assert not missing, f"these scripts do not forward the parsed moe_balancing to their trainer: {missing}"


def _constructed_trainer_class(script: Path, call: ast.Call) -> type | None:
    """The class ``call`` constructs, resolved through the script's own ``from ... import`` of it."""
    name = getattr(call.func, "id", None)
    for node in ast.walk(ast.parse(script.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module and any(alias.name == name for alias in node.names):
            return getattr(importlib.import_module(node.module), name)
    return None


def test_precompute_trainers_receive_the_resume_context():
    """TRL runs the reference precompute inside ``__init__``, before ``train()`` sees the resume checkpoint. Without
    ``resume_checkpoint`` and ``policy_from_checkpoint``, a Path-B resume sweeps the trained policy as its own
    reference."""
    wired: dict[str, bool] = {}
    for script in _TRAINER_SCRIPTS:
        call = _trainer_call(script)
        trainer_cls = _constructed_trainer_class(script, call)
        if trainer_cls is None or not issubclass(trainer_cls, TRLPreferenceMixin):
            continue
        passed = {kw.arg: kw.value for kw in call.keywords if kw.arg}
        wired[str(script.relative_to(REPO_ROOT))] = all(
            isinstance(passed.get(field), ast.Attribute)
            and passed[field].attr == field
            and getattr(passed[field].value, "id", None) == "context"
            for field in ("resume_checkpoint", "policy_from_checkpoint")
        )
    assert len(wired) >= 2, f"the scan found {sorted(wired)}, not the DPO and KTO scripts"
    unwired = sorted(script for script, ok in wired.items() if not ok)
    assert not unwired, (
        f"these scripts do not hand their trainer context.resume_checkpoint/policy_from_checkpoint: {unwired}"
    )


def test_the_shared_bundle_carries_every_distributed_knob():
    """A knob dropped from the helper is dropped from every trainer while the AST scan still sees a well-formed
    splat, so each value is checked against its own source."""
    args = SimpleNamespace(
        moe_balancing="aux_loss",
        save_sharded_ep=True,
        save_max_shard_size="1GB",
        allow_optimizer_warm_restart=True,
        max_consecutive_nonfinite_steps=7,
    )
    parallelism_config = object()

    bundle = distributed_trainer_kwargs(args, parallelism_config, dataset_presharded=True)

    assert set(bundle) == _BUNDLED_KNOBS, f"the shared bundle no longer carries {_BUNDLED_KNOBS ^ set(bundle)}"
    assert bundle["parallelism_config"] is parallelism_config
    assert bundle["moe_balancing"] == "aux_loss"
    assert bundle["save_sharded_ep"] is True
    assert bundle["save_max_shard_size"] == "1GB"
    assert bundle["allow_optimizer_warm_restart"] is True
    assert bundle["max_consecutive_nonfinite_steps"] == 7
    assert bundle["dataset_presharded"] is True


def test_rlrr_script_args_thread_every_tunable_into_the_shaping_config():
    """``build_rlrr_config`` is the only path from the CLI/YAML to :class:`RLRRConfig`. Asserted against non-default
    values, so a dropped field cannot pass by matching its default."""
    args = OnlineGRPOScriptArguments(
        use_rlrr=True,
        rlrr_mode="prr",
        rlrr_tau=0.25,
        rlrr_lambda=512.0,
        rlrr_std_normalize=True,
        rlrr_length_rerank=False,
        rlrr_correctness_clip=True,
        rlrr_correctness_threshold=0.75,
        rlrr_xi_pos=0.05,
        rlrr_xi_neg=-0.02,
    )

    config = args.build_rlrr_config()

    assert config == RLRRConfig(
        mode="prr",
        tau=0.25,
        lam=512.0,
        std_normalize=True,
        length_rerank=False,
        correctness_clip=True,
        correctness_threshold=0.75,
        xi_pos=0.05,
        xi_neg=-0.02,
    )
    # A hyperparameter without an ``rlrr_``-prefixed arg is unreachable from a config file.
    unreachable = [
        f.name
        for f in dataclasses.fields(RLRRConfig)
        if not hasattr(args, f"rlrr_{'lambda' if f.name == 'lam' else f.name}")
    ]
    assert not unreachable, f"RLRRConfig fields with no script arg: {unreachable}"


def test_rlrr_rejects_an_inverted_clip_band_from_the_script_args():
    """The script args build the config eagerly, so it fails at parse time. An inverted band floors correct
    responses above the cap on incorrect ones, inverting the Eq. 5 clip."""
    for use_rlrr in (True, False):
        with pytest.raises(ValueError, match="rlrr_xi_neg <= rlrr_xi_pos"):
            OnlineGRPOScriptArguments(use_rlrr=use_rlrr, rlrr_xi_pos=-1.0, rlrr_xi_neg=1.0)


# Every SDPGArguments field with a non-default value, so a field the builder drops cannot pass by matching what the
# trainer would have used.
_SDPG_TUNABLES = {
    "sdpg_hint_template": "the answer is {answer}",
    "sdpg_loss": "jensen_shannon",
    "sdpg_jsd_beta": 0.3,
    "sdpg_temperature": 0.5,
    "sdpg_beta_base": 0.25,
    "sdpg_beta_warmup_steps": 7,
    "sdpg_beta_decay_steps": 9,
}
# What the RLVR builder passes beside the tunables, which the trainer requires rather than defaults.
_RLVR_TRAINER_FIELDS = {"sdpg_answer_field": "answer", "opd_positive_advantage_only": True}


def _sdpg_params(trainer_cls) -> set[str]:
    """The ``sdpg_``-prefixed keyword arguments a trainer spells in its own signature."""
    return {name for name in inspect.signature(trainer_cls.__init__).parameters if name.startswith("sdpg_")}


def _construct_sdpg_shell(**kwargs) -> tuple[DistributedSDPGTrainer, dict]:
    """Run ``DistributedSDPGTrainer.__init__`` with the GRPO parent stubbed; returns the trainer and the kwargs the
    parent received, which TRL's explicit signature would reject."""
    leaked = {}

    def parent_init(self, *args, **kw):
        leaked.update(kw)
        self.train_dataset = None

    with mock.patch.object(DistributedOnlineGRPOTrainer, "__init__", parent_init):
        return DistributedSDPGTrainer(**{**_RLVR_TRAINER_FIELDS, **kwargs}), leaked


def test_sdpg_tunables_cover_the_shared_mixin():
    """A field added to SDPGArguments needs a non-default value here, or the wiring tests keep passing without
    covering it."""
    assert set(_SDPG_TUNABLES) == {f.name for f in dataclasses.fields(SDPGArguments)}


def test_rlvr_script_args_thread_every_sdpg_tunable_into_the_trainer_kwargs():
    """``build_sdpg_kwargs`` is the only path from the RLVR CLI/YAML to :class:`DistributedSDPGTrainer`."""
    args = OnlineGRPOScriptArguments(use_sdpg=True, opd_positive_advantage_only=False, **_SDPG_TUNABLES)

    kwargs = args.build_sdpg_kwargs()

    assert kwargs == {
        **_SDPG_TUNABLES,
        # Fixed, not forwarded: process_for_rlvr has already normalized the answer column.
        "sdpg_answer_field": "answer",
        "opd_positive_advantage_only": False,
    }
    _, leaked = _construct_sdpg_shell(**kwargs)
    assert leaked == {}, (
        f"build_sdpg_kwargs passes {sorted(leaked)}, which DistributedSDPGTrainer does not take: "
        "a TypeError at construction, after the model and the rollout servers are already up."
    )


def test_sdpg_kwargs_are_empty_while_the_gate_is_off():
    """The RLVR script splats the result unconditionally, so an off run must contribute no trainer kwargs."""
    assert OnlineGRPOScriptArguments(**_SDPG_TUNABLES).build_sdpg_kwargs() == {}


def test_sdpg_trainer_adopts_every_tunable_from_the_dataclass():
    """Through the real ctor: each tunable lands in ``sdpg`` under its own name, none leaks to the parent, and an
    omitted one takes the dataclass's default."""
    trainer, leaked = _construct_sdpg_shell(**_SDPG_TUNABLES)
    assert {name: getattr(trainer.sdpg, name) for name in _SDPG_TUNABLES} == _SDPG_TUNABLES
    assert leaked == {}

    defaulted, _ = _construct_sdpg_shell()
    declared = {f.name: f.default for f in dataclasses.fields(SDPGArguments)}
    assert {name: getattr(defaulted.sdpg, name) for name in declared} == declared


def test_sdpg_trainer_has_no_spelling_of_the_tunables_of_its_own():
    """A tunable back in the ctor signature can drift from SDPGArguments, and the pop is by declared field, so a
    misspelt kwarg reaches TRL's explicit signature."""
    restated = _sdpg_params(DistributedSDPGTrainer) & set(_SDPG_TUNABLES)
    assert restated == set(), f"DistributedSDPGTrainer restates {sorted(restated)} instead of reading SDPGArguments"

    _, leaked = _construct_sdpg_shell(sdpg_temprature=0.5)
    assert leaked == {"sdpg_temprature": 0.5}


def _self_distill_script_splats_the_builder() -> bool:
    tree = ast.parse((REPO_ROOT / "scripts/training/distillation/self_distill.py").read_text())
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "DistributedSelfDistillationTrainer"):
            continue
        for keyword in node.keywords:
            if keyword.arg is None and ast.unparse(keyword.value) == "args.build_sdpg_kwargs()":
                return True
    return False


# The trainer's own knobs, which the script passes explicitly (required, so none restates a default).
_SELF_DISTILL_TRAINER_KNOBS = {
    "reference_kl_coef": 0.0,
    "reference_kl_loss": "unnormalized_kl",
    "confidence_weight_opd": True,
    "opd_exclude_eos": True,
}


def _construct_self_distill_shell(**kwargs) -> tuple[DistributedSelfDistillationTrainer, dict]:
    """Run ``DistributedSelfDistillationTrainer.__init__`` with the SFT parent and the stop-id resolver stubbed;
    returns the trainer and the kwargs the parent received."""
    leaked = {}

    def parent_init(self, *args, **kw):
        leaked.update(kw)
        self.data_collator = SimpleNamespace(builds_teacher_branch=True)

    with (
        mock.patch.object(DistributedSFTTrainer, "__init__", parent_init),
        mock.patch.object(DistributedSelfDistillationTrainer, "_resolve_stop_token_ids", lambda self: None),
    ):
        return DistributedSelfDistillationTrainer(**_SELF_DISTILL_TRAINER_KNOBS, **kwargs), leaked


def test_self_distill_trainer_adopts_every_forwarded_sdpg_field():
    """Through the real ctor: each forwarded tunable lands in ``sdpg``, an omitted one takes the dataclass's
    default, and a dataset-side field reaches the parent's explicit signature instead of silently never hinting
    the teacher."""
    dataset_side = SelfDistillScriptArguments.DATASET_SIDE_SDPG_FIELDS
    assert _self_distill_script_splats_the_builder(), (
        "self_distill.py must splat args.build_sdpg_kwargs() into its trainer call — the trainer pops "
        "exactly the complement of DATASET_SIDE_SDPG_FIELDS that it builds"
    )
    forwarded = SelfDistillScriptArguments(**_SDPG_TUNABLES).build_sdpg_kwargs()
    assert forwarded == {name: value for name, value in _SDPG_TUNABLES.items() if name not in dataset_side}
    assert forwarded and set(_SDPG_TUNABLES) - set(forwarded) == dataset_side

    trainer, leaked = _construct_self_distill_shell(**forwarded)
    assert {name: getattr(trainer.sdpg, name) for name in forwarded} == forwarded
    assert leaked == {}

    defaulted, _ = _construct_self_distill_shell()
    declared = {f.name: f.default for f in dataclasses.fields(SDPGArguments) if f.name not in dataset_side}
    assert {name: getattr(defaulted.sdpg, name) for name in declared} == declared

    _, leaked = _construct_self_distill_shell(sdpg_hint_template="the answer is {answer}")
    assert leaked == {"sdpg_hint_template": "the answer is {answer}"}


def _answer_column_args(**overrides):
    defaults = {
        "sdpg_beta_base": 1.0,
        "sdpg_answer_field": "answer",
        "privileged_solution_field": "solution",
        "sdpg_hint_template": PRIVILEGED_HINT_TEMPLATE,
        "confidence_field": None,
        "dataset": "dummy/dataset",
    }
    return SimpleNamespace(**{**defaults, **overrides})


def _dataset_with(columns: dict[str, list]) -> DatasetDict:
    return DatasetDict({"train": Dataset.from_dict(columns), "test": Dataset.from_dict(columns)})


def test_self_distill_requires_the_answer_column_while_the_opd_term_carries_weight():
    """The collators read the answer with ``.get``, so a missing column hints an empty answer as fact and the OPD
    term distils toward a misled teacher."""
    module = load_script_module("scripts/training/distillation/self_distill.py", "halo_test_self_distill_answer")
    without = _dataset_with({"prompt": [[{"role": "user", "content": "q"}]]})
    with pytest.raises(ValueError, match="sdpg_answer_field='answer' names a column"):
        module._require_privileged_columns(without, _answer_column_args())

    carried = _dataset_with({"prompt": [[{"role": "user", "content": "q"}]], "answer": ["42"]})
    module._require_privileged_columns(carried, _answer_column_args())


def test_self_distill_answer_column_gate_stands_down_with_the_opd_term_off():
    """beta 0 drops the OPD term and with it the teacher branch, so no hint may demand the column."""
    module = load_script_module("scripts/training/distillation/self_distill.py", "halo_test_self_distill_answer_off")
    without = _dataset_with({"prompt": [[{"role": "user", "content": "q"}]]})
    module._require_privileged_columns(without, _answer_column_args(sdpg_beta_base=0.0))


def test_self_distill_requires_the_confidence_column_it_weights_by():
    """The weights read the column per row; a typo'd one would fail inside the first batch."""
    module = load_script_module("scripts/training/distillation/self_distill.py", "halo_test_self_distill_conf")
    without = _dataset_with({"prompt": [[{"role": "user", "content": "q"}]], "answer": ["42"]})
    with pytest.raises(ValueError, match="confidence_field='conf' names a column"):
        module._require_privileged_columns(without, _answer_column_args(sdpg_beta_base=0.0, confidence_field="conf"))


def test_self_distill_requires_the_solution_column_only_where_the_template_names_it():
    """A template naming ``{solution}`` over a dataset without that column hands the teacher a blank solution
    through the collators' ``.get``."""
    module = load_script_module("scripts/training/distillation/self_distill.py", "halo_test_self_distill_solution")
    answer_only = _dataset_with({"prompt": [[{"role": "user", "content": "q"}]], "answer": ["42"]})
    module._require_privileged_columns(answer_only, _answer_column_args())

    names_solution = _answer_column_args(sdpg_hint_template="answer {answer}, worked: {solution}")
    with pytest.raises(ValueError, match="privileged_solution_field='solution' names a column"):
        module._require_privileged_columns(answer_only, names_solution)

    carried = _dataset_with({"prompt": [[{"role": "user", "content": "q"}]], "answer": ["42"], "solution": ["6 * 7"]})
    module._require_privileged_columns(carried, names_solution)
    # A slot the template does not name demands no column: a solution-only hint takes no answer column.
    solution_only = _dataset_with({"prompt": [[{"role": "user", "content": "q"}]], "solution": ["6 * 7"]})
    module._require_privileged_columns(solution_only, _answer_column_args(sdpg_hint_template="worked: {solution}"))


@pytest.mark.parametrize(
    "construct", [_construct_sdpg_shell, _construct_self_distill_shell], ids=["sdpg", "self_distill"]
)
def test_sdpg_trainers_score_the_opd_term_at_the_configured_jsd_beta(construct):
    """The β set in the config is the β the OPD loss runs at, not the dataclass default."""
    student, teacher = torch.randn(2, 2, 3, 16, generator=torch.Generator().manual_seed(0))
    trainer, _ = construct(sdpg_loss="jensen_shannon", sdpg_jsd_beta=0.3)
    assert torch.equal(
        trainer.sdpg_loss_fn(student, teacher, 1.0), generalized_jsd_loss(student, teacher, 1.0, jsd_beta=0.3)
    )


def test_self_distill_trainer_has_no_spelling_of_the_tunables_of_its_own():
    """A tunable back in the ctor signature can drift from SDPGArguments, and a misspelt kwarg must reach the
    parent."""
    restated = _sdpg_params(DistributedSelfDistillationTrainer) & set(_SDPG_TUNABLES)
    assert restated == set(), f"DistributedSelfDistillationTrainer restates {sorted(restated)}"

    _, leaked = _construct_self_distill_shell(sdpg_temprature=0.5)
    assert leaked == {"sdpg_temprature": 0.5}


# Searched for the construction, so the trainer may move it into a helper without this test going quiet.
_ISR_MASK_CONSUMER_ROOT = "halo/trainers/grpo"

# ISRMaskConfig field -> the ``isr_``-prefixed AsyncRolloutConfig knob supplying it; derived, so a new stage is covered.
_ISR_MASK_SOURCES = {f.name: f"isr_{f.name}" for f in dataclasses.fields(ISRMaskConfig)}

# Not an ISRMaskConfig field: the circuit breaker reads the stages' output, so the trainer reads it off the config,
# unprefixed.
_MASK_FRACTION_KNOB = "skip_update_masked_frac"


def _grpo_trainer_trees() -> list[ast.Module]:
    return [ast.parse(path.read_text()) for path in sorted((REPO_ROOT / _ISR_MASK_CONSUMER_ROOT).rglob("*.py"))]


def test_every_isr_mask_knob_reaches_the_mask_config():
    """A dropped keyword leaves the stage off while the YAML says on, and the run keeps training on the drifted
    tokens the band exists to mask."""
    stages = {"geo_band_min": 0.98, "geo_band_max": 1.02, "veto_min": 1e-4, "opsm_delta": 0.3}
    assert set(stages) == set(_ISR_MASK_SOURCES), "a mask stage this test does not set"
    config = AsyncRolloutConfig(**{_ISR_MASK_SOURCES[name]: value for name, value in stages.items()})
    assert dataclasses.asdict(config.build_isr_mask_config()) == stages


def test_the_trainer_reads_the_configs_mask_config():
    """The trainer's stages are the config's, not a second construction that could drop a keyword."""
    builds = [
        node
        for tree in _grpo_trainer_trees()
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "build_isr_mask_config"
    ]
    constructions = [
        node
        for tree in _grpo_trainer_trees()
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "ISRMaskConfig"
    ]
    assert len(builds) == 1 and not constructions, (len(builds), len(constructions))


def test_every_isr_mask_stage_has_a_config_knob():
    """Reverse direction: a mask stage with no ``isr_`` knob is unreachable from a config file."""
    declared = {f.name for f in dataclasses.fields(AsyncRolloutConfig)}
    unreachable = sorted(knob for knob in _ISR_MASK_SOURCES.values() if knob not in declared)
    assert unreachable == [], f"ISRMaskConfig stages with no AsyncRolloutConfig knob: {unreachable}"


def test_the_masked_fraction_circuit_breaker_is_read_from_the_config():
    """The breaker zeroes a whole step's policy gradient; unread, the run trains on the selection-biased survivors
    the knob was set to refuse."""
    assert _MASK_FRACTION_KNOB in {f.name for f in dataclasses.fields(AsyncRolloutConfig)}
    reads = {node.attr for tree in _grpo_trainer_trees() for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert _MASK_FRACTION_KNOB in reads, (
        f"nothing under {_ISR_MASK_CONSUMER_ROOT} reads config.{_MASK_FRACTION_KNOB} any more"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
