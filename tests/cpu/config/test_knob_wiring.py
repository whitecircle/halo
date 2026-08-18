#!/usr/bin/env python
"""Tests that configuration knobs reach the code they claim to control.

The failure mode these pin down is a knob that parses from YAML and then does nothing: the run
looks configured and trains something else. Each test asserts an OBSERVABLE difference between the
knob on and off, or that an unsupported knob raises instead of being dropped.

Run: pytest tests/cpu/config/test_knob_wiring.py
"""

import ast
import dataclasses
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from src.args.distributed_args import DistributedArguments
from src.args.mixins import RLRRConfig, SDPGArguments
from src.args.rlvr_online_grpo_args import RLVROnlineGRPOScriptArguments
from src.args.self_distill_args import SelfDistillationArguments
from src.configs.async_training_config import AsyncTrainingConfig
from src.trainers.distillation.sdpg import DistributedSDPGTrainer
from src.trainers.distillation.self_distillation import DistributedSelfDistillationTrainer
from src.trainers.grpo.objective.logratio import ISMaskConfig
from src.trainers.grpo.online import DistributedGRPOTrainer
from src.trainers.mixins.base import DistributedTrainerMixin
from src.trainers.sft import DistributedSFTTrainer
from src.training.parallelism_args import parallelism_config_from_args
from src.training.script_runner import (
    distributed_trainer_kwargs,
    reject_non_default_args,
    reject_unsupported_args,
)
from tests.common.utils import load_script_module

PROJECT_ROOT = Path(__file__).resolve().parents[3]

# bf16_optimizer: YAML/CLI reachable, tri-state, and honored by the mixin


@pytest.mark.parametrize("requested", [None, True, False])
def test_bf16_optimizer_reaches_parallelism_config(requested):
    """The DistributedArguments knob must survive into ParallelismConfig — that object is the only
    thing every entry script threads into every trainer, so a drop here makes the knob unreachable."""
    config = parallelism_config_from_args(DistributedArguments(bf16_optimizer=requested))
    assert config.bf16_optimizer is requested


def _resolve_bf16_optimizer(*, config_value, bf16=True, optim="adamw_torch_fused", ctor_value=None):
    """Run the mixin's resolution with a stub trainer; returns the resolved ``_bf16_optimizer``."""
    stub = SimpleNamespace(
        parallelism_config=SimpleNamespace(
            bf16_optimizer=config_value,
            fp32_grad_reduce=False,
            ep_group_size=1,
            needs_ep_wrappers=False,
            experts_fsdp_managed=False,
            fp32_non_ep_params=False,
        ),
        _should_accelerate_manage_ddp=lambda: False,
    )
    training_args = SimpleNamespace(bf16=bf16, optim=optim, fp32_grad_reduce=None)
    kwargs = {} if ctor_value is None else {"bf16_optimizer": ctor_value}
    DistributedTrainerMixin._configure_mixed_precision(stub, kwargs, training_args)
    return stub._bf16_optimizer


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


def _mixed_precision_stub(
    *,
    ep_group_size: int,
    experts_fsdp_managed: bool = False,
    bf16_optimizer=False,
    fp32_non_ep_params: bool = False,
):
    return SimpleNamespace(
        parallelism_config=SimpleNamespace(
            bf16_optimizer=bf16_optimizer,
            fp32_grad_reduce=False,
            ep_group_size=ep_group_size,
            needs_ep_wrappers=True,
            experts_fsdp_managed=experts_fsdp_managed,
            fp32_non_ep_params=fp32_non_ep_params,
        ),
        _should_accelerate_manage_ddp=lambda: False,
    )


def _moe_model_kwargs():
    """Ctor kwargs whose model reads as a REGISTERED MoE family. The refusal resolves the family
    through the EP-wrapper registry (``ep_wraps_experts``), so a config without a registered
    ``model_type`` never has plain expert tensors and would pin nothing."""
    return {"model": SimpleNamespace(config=SimpleNamespace(num_local_experts=8, model_type="qwen3_moe"))}


def _optimizer_build_stub(
    *, ep_group_size: int, experts_fsdp_managed: bool = False, moe: bool = True, model_type: str = "qwen3_moe"
):
    """Stub carrying exactly what the optimizer-build refusal reads: the parallelism shape and the
    model config it resolves the family's EP wrapper class from."""
    return SimpleNamespace(
        parallelism_config=SimpleNamespace(
            ep_group_size=ep_group_size,
            needs_ep_wrappers=True,
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
    """``aten._fused_adamw_`` raises 'mixed torch.Tensor and DTensor' over that parameter set, so the
    build must refuse first — both when EP distributes the experts and when a grouped-GEMM ep1 run
    leaves them FSDP-unmanaged."""
    for stub in (
        _optimizer_build_stub(ep_group_size=8),
        _optimizer_build_stub(ep_group_size=1, experts_fsdp_managed=False),
    ):
        for optim in ("adamw_torch_fused", "adamw_torch"):
            with pytest.raises(ValueError, match="mixed torch.Tensor and DTensor"):
                DistributedTrainerMixin._refuse_stock_optimizer_on_mixed_params(stub, optim)


def test_stock_adamw_allowed_where_every_param_is_a_dtensor():
    """The refusal is about a plain/DTensor MIX, not about the model being MoE. At ``ep_group_size
    == 1`` with ``fsdp_shard_ep1_experts`` FSDP2 shards the experts too; a dense model has no expert
    tensors at all; and a MoE family with NO registered EP wrapper class (qwen3_next) never gets
    wrapped, so its experts are ordinary FSDP2 DTensors. Refusing any of the three would block a
    shape that steps fine — which is what makes ``experts_fsdp_managed`` and the registry resolution
    load-bearing rather than decorative."""
    DistributedTrainerMixin._refuse_stock_optimizer_on_mixed_params(
        _optimizer_build_stub(ep_group_size=1, experts_fsdp_managed=True), "adamw_torch_fused"
    )
    DistributedTrainerMixin._refuse_stock_optimizer_on_mixed_params(
        _optimizer_build_stub(ep_group_size=1, moe=False), "adamw_torch_fused"
    )
    DistributedTrainerMixin._refuse_stock_optimizer_on_mixed_params(
        _optimizer_build_stub(ep_group_size=1, model_type="qwen3_next"), "adamw_torch_fused"
    )


def test_named_optimizers_are_not_refused_by_the_stock_adamw_guard():
    """Muon/FlashAdamW/bnb are per-parameter and reach their own builders — refusing them here would
    block optimizers that handle the mixed set."""
    stub = _optimizer_build_stub(ep_group_size=8)
    for optim in ("muon", "flash_adamw", "adamw_bnb_8bit"):
        DistributedTrainerMixin._refuse_stock_optimizer_on_mixed_params(stub, optim)


def test_fp32_on_a_moe_builds_the_trainer_and_only_refuses_at_the_optimizer():
    """Constructing a trainer with ``bf16: false`` on a MoE must NOT raise: forward, backward and the
    EP-aware clip are correct over the mixed parameter set (the fp32 PP+ETP equivalence gate runs
    exactly that, with no optimizer in play). Only building the stock optimizer is the defect."""
    stub = _mixed_precision_stub(ep_group_size=8, bf16_optimizer=None)
    DistributedTrainerMixin._configure_mixed_precision(
        stub, _moe_model_kwargs(), SimpleNamespace(bf16=False, optim="adamw_torch_fused", fp32_grad_reduce=None)
    )
    assert stub._bf16_optimizer is False


def test_create_optimizer_wires_the_refusal_before_building_the_stock_optimizer():
    """The guard only matters if ``create_optimizer``'s stock branch actually calls it: a green
    helper with severed wiring lets the run die at step 1 with the raw DTensor error. The probe is a
    REAL subclass instance (``__new__``, no ``__init__``) so the method's zero-arg ``super()``
    resolves — reaching it raises ``AttributeError`` (nothing above the mixin defines
    ``create_optimizer``), which is how the dense case proves the guard let it through."""

    class _Probe(DistributedTrainerMixin):
        _has_ep_layers = False  # shadows the base property (it walks self.model.modules())

    def _build(*, moe: bool):
        trainer = _Probe.__new__(_Probe)
        stub = _optimizer_build_stub(ep_group_size=8 if moe else 1, moe=moe)
        trainer.optimizer = None
        trainer._bf16_optimizer = False
        trainer.args = SimpleNamespace(optim="adamw_torch_fused")
        trainer.get_decay_parameter_names = lambda model: []
        trainer.model = stub.model
        trainer.model.modules = lambda: []  # the FSDP2 re-registration seam walks modules; the stub is no nn.Module
        trainer.parallelism_config = stub.parallelism_config
        return trainer

    with pytest.raises(ValueError, match="mixed torch.Tensor and DTensor"):
        _build(moe=True).create_optimizer()
    with pytest.raises(AttributeError):  # guard passed: the probe has no real Trainer above it
        _build(moe=False).create_optimizer()


def test_explicit_bf16_optimizer_false_selects_the_stock_path_without_raising_at_construction():
    """``bf16_optimizer=False`` selects the stock optimizer rather than raising at construction —
    the unsupported combination is caught by the build guard above, not here."""
    stub = _mixed_precision_stub(ep_group_size=8, bf16_optimizer=False)
    DistributedTrainerMixin._configure_mixed_precision(
        stub, _moe_model_kwargs(), SimpleNamespace(bf16=True, optim="adamw_torch_fused", fp32_grad_reduce=None)
    )
    assert stub._bf16_optimizer is False


def test_fp32_non_ep_params_is_the_remedy_the_message_names_and_is_not_itself_refused():
    """The refusal tells the user to set ``fp32_non_ep_params``, which routes ``create_optimizer`` to
    the tensor-type-grouped builder before the stock branch is reached — the one path that handles a
    mixed plain/DTensor set. Construction must not refuse it either, or it blocks the fix it names."""
    stub = _mixed_precision_stub(ep_group_size=8, fp32_non_ep_params=True, bf16_optimizer=None)
    DistributedTrainerMixin._configure_mixed_precision(
        stub, {}, SimpleNamespace(bf16=False, optim="adamw_torch_fused", fp32_grad_reduce=None)
    )
    assert stub._bf16_optimizer is False


# reject_unsupported_args: the shared "would be silently ignored" gate


def test_reject_unsupported_args_passes_when_nothing_set():
    reject_unsupported_args("Some script", tools_field=None, images_field="", train_flag=False)


def test_reject_unsupported_args_names_every_set_field():
    with pytest.raises(ValueError) as excinfo:
        reject_unsupported_args("Some script", tools_field="tools", images_field=None, train_flag=True)
    message = str(excinfo.value)
    assert "Some script" in message
    assert "tools_field" in message
    assert "train_flag" in message
    assert "images_field" not in message  # unset fields are not a request


# reject_non_default_args: the same gate for knobs whose own default is truthy


def test_reject_non_default_args_passes_at_the_declared_defaults():
    """The whole point: ``num_eval_examples`` defaults to 50 — truthy — so the truthiness gate could
    never guard it without rejecting every run."""
    reject_non_default_args("Some script", SelfDistillationArguments(), "generate_eval_examples", "num_eval_examples")


def test_reject_non_default_args_rejects_a_changed_truthy_default():
    args = SelfDistillationArguments()
    args.num_eval_examples = 8
    with pytest.raises(ValueError, match="num_eval_examples"):
        reject_non_default_args("Self-distillation", args, "generate_eval_examples", "num_eval_examples")


def test_reject_non_default_args_still_rejects_a_flipped_flag():
    args = SelfDistillationArguments()
    args.generate_eval_examples = True
    with pytest.raises(ValueError, match="generate_eval_examples"):
        reject_non_default_args("Self-distillation", args, "generate_eval_examples", "num_eval_examples")


def test_reject_non_default_args_refuses_a_field_the_dataclass_does_not_declare():
    """A typo'd name is a guard that can never fire — the silent no-op this family exists to stop."""
    with pytest.raises(ValueError, match="num_gen_exmaples"):
        reject_non_default_args("Some script", SelfDistillationArguments(), "num_gen_exmaples")


def test_self_distill_guards_both_generation_eval_knobs():
    """Self-distillation keeps its dataset raw, so GenerateExamplesCallback is unreachable there and
    BOTH knobs are inert. Reading the call keeps the script's guard from losing one silently."""
    source = (PROJECT_ROOT / "scripts/training/distillation/self_distill.py").read_text()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "reject_non_default_args":
            guarded = {arg.value for arg in node.args if isinstance(arg, ast.Constant)}
            assert {"generate_eval_examples", "num_eval_examples"} <= guarded
            return
    raise AssertionError("self_distill.py no longer calls reject_non_default_args")


# Per-script rejection of knobs the script cannot honor


def _rejected_knobs(script_path: str) -> set[str]:
    """Knob names passed to ``reject_unsupported_args`` with a value read off a parsed config.

    The value matters as much as the keyword: ``reject_unsupported_args`` filters on truthiness, so
    a literal (``assistant_only_loss=None``) keeps the keyword present while making the guard
    unfirable — restoring exactly the silent no-op it exists to stop. Requiring an attribute load
    somewhere in the value subtree covers the plain reads and the ``x if y else None`` forms alike.
    """
    tree = ast.parse((PROJECT_ROOT / script_path).read_text())
    rejected: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "reject_unsupported_args":
            rejected.update(
                kw.arg
                for kw in node.keywords
                if kw.arg and any(isinstance(sub, ast.Attribute) for sub in ast.walk(kw.value))
            )
    return rejected


# Consumed only by TRL's own dataset prep + default collator, which these scripts replace.
_TRL_SFT_MASK_KNOBS = {"assistant_only_loss", "completion_only_loss"}


@pytest.mark.parametrize(
    ("script", "fields"),
    [
        ("scripts/training/preference/kto.py", {"tools_field", "log_decoded_samples"}),
        ("scripts/training/preference/rewards.py", {"log_decoded_samples", "text_only_model", "tools_field"}),
        ("scripts/training/preference/dpo.py", {"tools_field"}),
        ("scripts/training/preference/smpo.py", {"tools_field"}),
        ("scripts/training/classification.py", {"text_only_model"}),
        ("scripts/training/environmental_grpo.py", {"tools_field", "text_only_model"}),
        ("scripts/training/online_grpo/rlvr.py", {"text_only_model"}),
        ("scripts/training/distillation/self_distill.py", _TRL_SFT_MASK_KNOBS),
        ("scripts/training/sft.py", _TRL_SFT_MASK_KNOBS),
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
    """These knobs parse on the listed script but nothing consumes them. Dropping a guard without
    wiring the knob restores the silent no-op, which this catches."""
    assert fields <= _rejected_knobs(script), f"{script} no longer rejects {sorted(fields - _rejected_knobs(script))}"


def _run_script_main(script: str, config_body: str, tmp_path: Path) -> None:
    """Drive ``scripts/training/<script>:main()`` from a minimal YAML, with nothing past the parse stubbed.

    A guard that only fires after the distributed init or the model load fails here on that step
    instead of raising the expected rejection.
    """
    module = load_script_module(f"scripts/training/{script}", "halo_test_knob_wiring_" + script.replace("/", "_"))
    config = tmp_path / "config.yaml"
    config.write_text(
        f"model_name_or_path: stub/qwen3-4b\ndataset:\n- dummy/dataset\noutput_dir: {tmp_path / 'out'}\n"
        f"bf16: false\nuse_cpu: true\n{config_body}"
    )
    with (
        mock.patch("src.training.parser.install_log_tee"),
        mock.patch.object(sys, "argv", ["prog", str(config)]),
    ):
        module.main()


@pytest.mark.parametrize(
    ("script", "config_body", "knob"),
    [
        # The text-only CausalLM spells its decoder model.layers.* while the served multimodal
        # checkpoint spells model.language_model.layers.*; the sync forwards names verbatim.
        ("online_grpo/rlvr.py", "text_only_model: true\n", "text_only_model"),
        ("environmental_grpo.py", "text_only_model: true\nenvironment_type: react_math\n", "text_only_model"),
        # TRL's own dataset prep and default collator — the only consumers — are both replaced.
        ("sft.py", "assistant_only_loss: true\n", "assistant_only_loss"),
        ("distillation/self_distill.py", "completion_only_loss: false\n", "completion_only_loss"),
    ],
)
def test_script_main_refuses_the_knob_before_any_load(script, config_body, knob, tmp_path):
    """The guard must sit ahead of the distributed init and the model load: a rejection that only
    fires after a multi-node load has already spent the cluster's time on a run it then refuses."""
    with pytest.raises(ValueError, match=rf"does not support these config fields.*{knob}"):
        _run_script_main(script, config_body, tmp_path)


# moe_balancing: the trainer needs the value, not just the callback builder


# The helper the scripts splat their distributed knobs through, and the knobs it owes them.
_BUNDLE_HELPER = "distributed_trainer_kwargs"
_BUNDLED_KNOBS = frozenset({"parallelism_config", "moe_balancing", "save_sharded_ep", "dataset_presharded"})


def _splats_the_bundle(call: ast.Call) -> bool:
    """Whether ``call`` carries ``**distributed_trainer_kwargs(...)``."""
    return any(
        kw.arg is None and isinstance(kw.value, ast.Call) and getattr(kw.value.func, "id", None) == _BUNDLE_HELPER
        for kw in call.keywords
    )


def _trainer_call(script: Path) -> ast.Call | None:
    """The script's trainer construction, or ``None`` when it builds none (helpers, ``__init__``).

    Identified by the distributed bundle it carries: the shared ``**distributed_trainer_kwargs(...)``
    splat, or a hand-spelled ``parallelism_config=`` for a call that still lists the knobs itself.
    """
    for node in ast.walk(ast.parse(script.read_text())):
        if not isinstance(node, ast.Call):
            continue
        if _splats_the_bundle(node) or any(kw.arg == "parallelism_config" for kw in node.keywords):
            return node
    return None


def _forwards_moe_balancing(script: Path) -> bool:
    """Whether the script's trainer receives the PARSED ``moe_balancing``.

    Checking the value, not just the keyword: ``moe_balancing=None`` keeps the keyword present while
    restoring exactly the silent downgrade this file exists to catch. The shared bundle counts —
    :func:`test_the_shared_bundle_carries_every_distributed_knob` is what holds it to its contents.
    """
    call = _trainer_call(script)
    if call is None:
        return False
    for kw in call.keywords:
        if kw.arg == "moe_balancing":
            return isinstance(kw.value, ast.Attribute)
    return _splats_the_bundle(call)


_TRAINER_SCRIPTS = sorted(
    path for path in (PROJECT_ROOT / "scripts" / "training").rglob("*.py") if _trainer_call(path) is not None
)


def test_every_training_script_forwards_moe_balancing_to_its_trainer():
    """``build_perf_callbacks`` reads ``moe_balancing`` off the script args to pick the strategy, but the
    trainer needs its own copy: ``_validate_router_aux_loss_consumable`` decides raise-vs-warn from
    ``self._moe_balancing``, which ``_init_distributed_config`` defaults to ``"auto"``. A script
    that omits the kwarg silently downgrades that rejection to a warning, so a run asking for
    ``aux_loss`` on a trainer whose objective never adds it pays the router-logit cost and balances
    nothing — the exact case the guard exists to refuse.
    """
    assert _TRAINER_SCRIPTS, "no training script matched the trainer-construction scan"
    missing = [
        str(script.relative_to(PROJECT_ROOT)) for script in _TRAINER_SCRIPTS if not _forwards_moe_balancing(script)
    ]
    assert not missing, f"these scripts do not forward the parsed moe_balancing to their trainer: {missing}"


def test_the_shared_bundle_carries_every_distributed_knob():
    """The other half of the scan above: the scripts splat one helper, so a knob dropped from IT is
    dropped from every trainer at once — and the AST scan would still see a well-formed splat.

    Each value is checked against its own source, not merely present: wiring ``moe_balancing`` to the
    distributed args (where the field does not live) would hand every trainer ``None`` and restore
    the same silent downgrade.
    """
    args = SimpleNamespace(moe_balancing="aux_loss")
    dist_args = SimpleNamespace(save_sharded_ep=True)
    parallelism_config = object()

    bundle = distributed_trainer_kwargs(args, dist_args, parallelism_config, dataset_presharded=True)

    assert set(bundle) == _BUNDLED_KNOBS, f"the shared bundle no longer carries {_BUNDLED_KNOBS ^ set(bundle)}"
    assert bundle["parallelism_config"] is parallelism_config
    assert bundle["moe_balancing"] == "aux_loss"
    assert bundle["save_sharded_ep"] is True
    assert bundle["dataset_presharded"] is True


def test_rlrr_script_args_thread_every_tunable_into_the_shaping_config():
    """``build_rlrr_config`` is the only path from the CLI/YAML to :class:`RLRRConfig`, so a field it
    forgets is a knob the user can set and the shaping never sees — the run trains with the paper's
    default clip while the config says otherwise. Asserted field-by-field against non-default values
    so a dropped or mis-wired field cannot pass by matching its own default.
    """
    args = RLVROnlineGRPOScriptArguments(
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
    # Every RLRRConfig field must have an ``rlrr_``-prefixed arg: a new hyperparameter added to the
    # shaping config without one is unreachable from a config file.
    unreachable = [
        f.name
        for f in dataclasses.fields(RLRRConfig)
        if not hasattr(args, f"rlrr_{'lambda' if f.name == 'lam' else f.name}")
    ]
    assert not unreachable, f"RLRRConfig fields with no script arg: {unreachable}"


def test_rlrr_rejects_an_inverted_clip_band_from_the_script_args():
    """``xi_neg <= xi_pos`` is RLRRConfig's own invariant, and the script args must carry it there.

    An inverted band floors correct responses above the cap on incorrect ones, inverting the Eq. 5
    clip the config exists to apply — silently, since both values parse.
    """
    args = RLVROnlineGRPOScriptArguments(use_rlrr=True, rlrr_xi_pos=-1.0, rlrr_xi_neg=1.0)

    with pytest.raises(ValueError, match="xi_neg <= xi_pos"):
        args.build_rlrr_config()


# SDPG: the shared argument mixin and the two arms that hand it to a trainer

# Every SDPGArguments field with a value that is NOT its default, so a field the builder drops cannot
# pass by matching what the trainer would have used anyway.
_SDPG_TUNABLES = {
    "sdpg_hint_template": "the answer is {answer}",
    "sdpg_loss": "forward_kl",
    "sdpg_temperature": 0.5,
    "sdpg_beta_base": 0.25,
    "sdpg_beta_warmup_steps": 7,
    "sdpg_beta_decay_steps": 9,
}


def _sdpg_params(trainer_cls) -> set[str]:
    """The ``sdpg_``-prefixed keyword arguments a trainer spells in its own signature."""
    return {name for name in inspect.signature(trainer_cls.__init__).parameters if name.startswith("sdpg_")}


def _construct_sdpg_shell(**kwargs) -> tuple[DistributedSDPGTrainer, dict]:
    """Run ``DistributedSDPGTrainer.__init__`` for real with the GRPO parent stubbed out.

    Returns the trainer and the kwargs the parent received — whatever the SDPG ctor did not consume,
    which the real parent chain hands to TRL's explicit signature to reject.
    """
    leaked = {}

    def parent_init(self, *args, **kw):
        leaked.update(kw)
        self.train_dataset = None

    with mock.patch.object(DistributedGRPOTrainer, "__init__", parent_init):
        return DistributedSDPGTrainer(**kwargs), leaked


def test_sdpg_tunables_cover_the_shared_mixin():
    """Pins the table above against the dataclass: a field added to SDPGArguments must be given a
    non-default value here, or the wiring tests below would keep passing without covering it."""
    assert set(_SDPG_TUNABLES) == {f.name for f in dataclasses.fields(SDPGArguments)}


def test_rlvr_script_args_thread_every_sdpg_tunable_into_the_trainer_kwargs():
    """``build_sdpg_kwargs`` is the only path from the RLVR CLI/YAML to :class:`DistributedSDPGTrainer`,
    so a field it forgets is a knob the user sets and the OPD term never sees — the run distils on the
    paper's defaults while the config file says otherwise.
    """
    args = RLVROnlineGRPOScriptArguments(use_sdpg=True, opd_positive_advantage_only=False, **_SDPG_TUNABLES)

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
    """The gate is the builder's whole remaining job — the RLVR script splats the result
    unconditionally, so an off run must contribute no trainer kwargs at all."""
    assert RLVROnlineGRPOScriptArguments(**_SDPG_TUNABLES).build_sdpg_kwargs() == {}


def test_sdpg_trainer_adopts_every_tunable_from_the_dataclass():
    """Driven through the real ctor: each tunable lands on the trainer under its own name, none leaks
    past it to the parent, and an omitted one takes the dataclass's default — so a directly built
    trainer (the documented pattern for tests and notebooks) runs the OPD schedule the identical YAML
    run would."""
    trainer, leaked = _construct_sdpg_shell(**_SDPG_TUNABLES)
    assert {name: getattr(trainer, name) for name in _SDPG_TUNABLES} == _SDPG_TUNABLES
    assert leaked == {}

    defaulted, _ = _construct_sdpg_shell()
    declared = {f.name: f.default for f in dataclasses.fields(SDPGArguments)}
    assert {name: getattr(defaulted, name) for name in declared} == declared


def test_sdpg_trainer_has_no_spelling_of_the_tunables_of_its_own():
    """The copy this guards against: a tunable back in the ctor signature is a name and a default that
    can drift from SDPGArguments again. And the pop is by declared field, not by prefix — a misspelt
    kwarg must reach the parent, where TRL's explicit signature rejects it, rather than vanish."""
    restated = _sdpg_params(DistributedSDPGTrainer) & set(_SDPG_TUNABLES)
    assert restated == set(), f"DistributedSDPGTrainer restates {sorted(restated)} instead of reading SDPGArguments"

    _, leaked = _construct_sdpg_shell(sdpg_temprature=0.5)
    assert leaked == {"sdpg_temprature": 0.5}


def _self_distill_script_splats_the_complement() -> bool:
    """Whether the script's trainer call splats a dict comprehension filtered on the args class's
    ``DATASET_SIDE_SDPG_FIELDS`` — the same declaration the trainer pops against."""
    tree = ast.parse((PROJECT_ROOT / "scripts/training/distillation/self_distill.py").read_text())
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and getattr(node.func, "id", None) == "DistributedSelfDistillationTrainer"):
            continue
        for keyword in node.keywords:
            if keyword.arg is None and isinstance(keyword.value, ast.DictComp):
                for generator in keyword.value.generators:
                    for test in generator.ifs:
                        for sub in ast.walk(test):
                            if (
                                isinstance(sub, ast.Attribute)
                                and sub.attr == "DATASET_SIDE_SDPG_FIELDS"
                                and getattr(sub.value, "id", None) == "SelfDistillationArguments"
                            ):
                                return True
    return False


def _construct_self_distill_shell(**kwargs) -> tuple[DistributedSelfDistillationTrainer, dict]:
    """Run ``DistributedSelfDistillationTrainer.__init__`` for real with the SFT parent stubbed out
    (and the tokenizer-reading stop-id resolver, which needs a processing class). Returns the trainer
    and the kwargs the parent received — whatever the ctor did not consume, which the real parent
    chain hands to TRL's explicit signature to reject."""
    leaked = {}

    def parent_init(self, *args, **kw):
        leaked.update(kw)

    with (
        mock.patch.object(DistributedSFTTrainer, "__init__", parent_init),
        mock.patch.object(DistributedSelfDistillationTrainer, "_resolve_stop_token_ids", lambda self: None),
    ):
        return DistributedSelfDistillationTrainer(**kwargs), leaked


def test_self_distill_trainer_adopts_every_forwarded_sdpg_field():
    """The offline arm forwards every SDPG field but the dataset-side ones, which the script applies
    while building the teacher prompts. Driven through the real ctor: each forwarded tunable lands on
    the trainer under its own name, none leaks to the parent, an omitted one takes the dataclass's
    default, and a dataset-side field handed to the trainer is NOT swallowed — it reaches the parent,
    whose explicit signature rejects it, instead of silently never hinting the teacher."""
    dataset_side = SelfDistillationArguments.DATASET_SIDE_SDPG_FIELDS
    assert _self_distill_script_splats_the_complement(), (
        "self_distill.py must splat fields(SDPGArguments) minus SelfDistillationArguments.DATASET_SIDE_SDPG_FIELDS "
        "into its trainer call — the trainer pops exactly that complement"
    )
    forwarded = {name: value for name, value in _SDPG_TUNABLES.items() if name not in dataset_side}
    assert forwarded and set(_SDPG_TUNABLES) - set(forwarded) == dataset_side

    trainer, leaked = _construct_self_distill_shell(**forwarded)
    assert {name: getattr(trainer, name) for name in forwarded} == forwarded
    assert leaked == {}

    defaulted, _ = _construct_self_distill_shell()
    declared = {f.name: f.default for f in dataclasses.fields(SDPGArguments) if f.name not in dataset_side}
    assert {name: getattr(defaulted, name) for name in declared} == declared

    _, leaked = _construct_self_distill_shell(sdpg_hint_template="the answer is {answer}")
    assert leaked == {"sdpg_hint_template": "the answer is {answer}"}


def test_self_distill_trainer_has_no_spelling_of_the_tunables_of_its_own():
    """A tunable back in the ctor signature is a name and a default that can drift from
    SDPGArguments again; and the pop is by declared field, so a misspelt kwarg must reach the parent."""
    restated = _sdpg_params(DistributedSelfDistillationTrainer) & set(_SDPG_TUNABLES)
    assert restated == set(), f"DistributedSelfDistillationTrainer restates {sorted(restated)}"

    _, leaked = _construct_self_distill_shell(sdpg_temprature=0.5)
    assert leaked == {"sdpg_temprature": 0.5}


# Importance-sampling mask knobs: the AsyncTrainingConfig -> ISMaskConfig contract

# The package that consumes the knobs. Searched for the construction rather than pinned to a file, so
# the trainer may move it into a helper or a mixin without this test going quiet.
_IS_MASK_CONSUMER_ROOT = "src/trainers/grpo"

# ISMaskConfig field -> the AsyncTrainingConfig knob that must supply it, under the ``isr_`` prefix
# the YAML surface spells them with. Derived, so a stage added to the mask config is covered here.
_IS_MASK_SOURCES = {f.name: f"isr_{f.name}" for f in dataclasses.fields(ISMaskConfig)}

# Not an ISMaskConfig field: the circuit breaker reads the mask stages' OUTPUT, so the trainer holds
# it directly. Named unprefixed, which is only how the config read is spelled (the trainer's own
# attribute is ``_skip_update_masked_frac``).
_MASK_FRACTION_KNOB = "skip_update_masked_frac"


def _grpo_trainer_trees() -> list[ast.Module]:
    return [ast.parse(path.read_text()) for path in sorted((PROJECT_ROOT / _IS_MASK_CONSUMER_ROOT).rglob("*.py"))]


def _is_mask_construction() -> dict[str, str | None]:
    """The ``ISMaskConfig(...)`` keywords, each mapped to the attribute name its value reads.

    ``None`` for a value that is not a plain attribute read: a literal there is a stage pinned off
    whatever the config says, which is the same silent no-op as a missing keyword.
    """
    calls = [
        node
        for tree in _grpo_trainer_trees()
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "ISMaskConfig"
    ]
    assert len(calls) == 1, (
        f"expected exactly one ISMaskConfig construction under {_IS_MASK_CONSUMER_ROOT}, found {len(calls)}"
    )
    return {kw.arg: getattr(kw.value, "attr", None) for kw in calls[0].keywords if kw.arg}


def test_every_is_mask_knob_reaches_the_mask_config():
    """Fourteen shipped env-GRPO configs set these, and this construction is the only thing that
    reads them. A dropped keyword leaves the stage off while the YAML says it is on, and the run
    keeps training on exactly the drifted tokens the band exists to mask — with nothing in the logs."""
    assert _is_mask_construction() == _IS_MASK_SOURCES


def test_every_is_mask_stage_has_a_config_knob():
    """Reverse direction: a mask stage with no ``isr_`` knob is unreachable from a config file."""
    declared = {f.name for f in dataclasses.fields(AsyncTrainingConfig)}
    unreachable = sorted(knob for knob in _IS_MASK_SOURCES.values() if knob not in declared)
    assert unreachable == [], f"ISMaskConfig stages with no AsyncTrainingConfig knob: {unreachable}"


def test_the_masked_fraction_circuit_breaker_is_read_from_the_config():
    """``skip_update_masked_frac`` zeroes a whole step's policy gradient. Unread, it is a trust region
    that never fires: the run trains on the selection-biased survivors the knob was set to refuse."""
    assert _MASK_FRACTION_KNOB in {f.name for f in dataclasses.fields(AsyncTrainingConfig)}
    reads = {node.attr for tree in _grpo_trainer_trees() for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    assert _MASK_FRACTION_KNOB in reads, (
        f"nothing under {_IS_MASK_CONSUMER_ROOT} reads config.{_MASK_FRACTION_KNOB} any more"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
