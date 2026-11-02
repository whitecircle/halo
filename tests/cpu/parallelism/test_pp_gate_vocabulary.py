#!/usr/bin/env python
"""The shared PP rejection vocabulary (``halo.trainers.parallelism.pipeline_rules``) and its call sites.

``test_pp_trainer_gates.py`` covers *which trainers* may run PP (``PP_REFUSAL``). This file covers
what happens once a supported trainer is under PP and carries an option the pipeline cannot honor.
Every helper here guards a **silent** failure — PEFT injection resolving module names against a tree
no stage holds, a full reference model resident beside one stage, TRL's ``activation_offloading``
wrapper that the schedule-driven step bypasses so it never engages — so a deleted call site does not
crash, it trains something else.

Two layers, because each catches a different regression:

* the helpers themselves — inert on the allowed value, and each raise names its mechanism;
* the **wiring**, which drives the REAL ``_reject_unsupported_pp_options`` / ``_split_pipeline_stage`` of
  every trainer that calls them. Deleting ``reject_pp_peft(...)`` from any single call site fails
  exactly one parametrized case here; helper-only tests would stay green.

Run: python tests/cpu/parallelism/test_pp_gate_vocabulary.py  (or pytest -m cpu)
"""

from __future__ import annotations

import inspect
from types import SimpleNamespace

import pytest
import torch
from peft import LoraConfig, PromptTuningConfig, get_peft_model, inject_adapter_in_model
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PretrainedConfig, PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from halo.data.collators.packing import DataCollatorWithFlattening, DataCollatorWithPacking
from halo.distributed.pipeline_parallel.stage import PipelineStageModule
from halo.trainers.ctor_slots import resolve_ctor_positions
from halo.trainers.grpo.offline.offline_rules import reject_offline_pp_options
from halo.trainers.parallelism.pipeline_rules import (
    reject_missing_precomputed_columns,
    reject_missing_precomputed_reference,
    reject_positional_model_or_args,
    reject_pp_collator,
    reject_pp_compute_metrics,
    reject_pp_peft,
    reject_pp_ref_model,
    reject_unsupported_pp_training_args,
    require_pad_token_id,
    require_pooling_pad_id,
)
from halo.trainers.parallelism.pipeline_trainer import PipelineTrainerMixin
from halo.trainers.preference.dpo import DistributedDPOTrainer
from halo.trainers.preference.kto import DistributedKTOTrainer
from halo.trainers.preference.preference_rules import reject_smpo_pp_options
from tests.common.models import TINY_QWEN3_CONFIG
from tests.common.parallelism import make_parallelism_config
from tests.common.rosters import distributed_trainer_classes

PEFT_SENTINEL = object()
"""Stand-in for a ``peft_config``: the gates test identity against ``None``, never structure."""


class _Dataset:
    """Minimal stand-in for the ``column_names`` protocol the precompute gates read."""

    def __init__(self, *columns: str):
        self.column_names = list(columns)


def _training_args(**overrides):
    """A PP-legal training-args namespace; overrides inject the one option under test.

    Every field is one a validator on the path to a ``pipeline_rules`` call reads, set to the value that
    lets the walk reach it — so a failure here is always the gate under test, never a neighbour.
    """
    base = {
        "max_length": 128,
        "gradient_checkpointing": False,
        "gradient_checkpointing_kwargs": None,
        "eval_strategy": "no",
        "eval_on_start": False,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "torch_compile": False,
        "activation_offloading": False,
        "precompute_ref_log_probs": True,
        "ld_alpha": None,
        "use_weighting": False,
        "f_divergence_type": "reverse_kl",
        "use_chunked_grpo_logprobs": False,
        "kl_beta": 0.0,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _tiny_lm() -> Qwen3ForCausalLM:
    torch.manual_seed(0)
    return Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG))


def _dpo_args(**overrides):
    """DPO's PP-legal args: a per-pair loss the PP last-stage loss implements."""
    return _training_args(**{"loss_type": "sigmoid", **overrides})


def _kto_args(**overrides):
    """KTO's PP-legal args: the KL-free ``apo_zero_unpaired`` loss, the only decomposable one."""
    return _training_args(**{"loss_type": "apo_zero_unpaired", **overrides})


# ── The helpers: inert on the allowed value ─────────────────────────────────────
#
# Anti-vacuity for everything below: if a helper raised unconditionally, the wiring cases would pass
# for the wrong reason.


def test_every_helper_is_inert_on_the_value_pp_accepts():
    reject_pp_peft(None)
    reject_pp_peft(None, explicit_param_trainer="SMPO")
    reject_pp_ref_model(None, "the ref_logps column")
    reject_pp_compute_metrics(None, "mechanism")
    reject_unsupported_pp_training_args(_training_args(), save_sharded_ep=False, evaluates=False)
    # The flag absent entirely (plain TrainingArguments carry no activation_offloading).
    reject_unsupported_pp_training_args(
        SimpleNamespace(max_length=8, gradient_checkpointing=False, torch_compile=False),
        save_sharded_ep=False,
        evaluates=False,
    )
    reject_positional_model_or_args({"model": object(), "args": object()})
    reject_missing_precomputed_columns(_Dataset("ref_logps", "prompt"), ("ref_logps",))
    reject_missing_precomputed_reference(
        "DPO",
        _training_args(),
        _Dataset("ref_chosen_logps", "ref_rejected_logps"),
        None,
        ("ref_chosen_logps", "ref_rejected_logps"),
    )


# ── The helpers: each raise names its mechanism ─────────────────────────────────


def test_peft_rejection_names_the_stage_unaware_adapter_checkpoint_mechanism():
    """The blocker is the adapter save/resume path, not injection (which works on a stage): the
    raise must name the stage-local layer indices the adapter file would carry and the resume gap."""
    with pytest.raises(ValueError) as err:
        reject_pp_peft(PEFT_SENTINEL)
    text = str(err.value)
    assert "re-based layer indices" in text and "adapter restore" in text
    assert "get_submodule" not in text, "injection is not the mechanism; naming it sends the reader to the wrong fix"


def test_peft_rejection_names_the_trainer_whose_ctor_hides_the_config():
    """SMPO and offline GRPO take ``peft_config`` as an explicit ctor parameter, so the mixin's
    kwargs-based rejection cannot see it — the message must say so, or the next reader adds a
    duplicate gate in the mixin that still cannot reach it."""
    with pytest.raises(ValueError) as err:
        reject_pp_peft(PEFT_SENTINEL, explicit_param_trainer="SMPO")
    assert "SMPO takes peft_config as an explicit parameter" in str(err.value)


def test_ref_model_rejection_names_the_column_to_ship_instead():
    with pytest.raises(ValueError) as err:
        reject_pp_ref_model(object(), "the ref_logps dataset column")
    text = str(err.value)
    assert "precompute_ref_log_probs=True" in text and "the ref_logps dataset column" in text


def test_compute_metrics_rejection_carries_the_callers_mechanism():
    with pytest.raises(ValueError) as err:
        reject_pp_compute_metrics(lambda *_: {}, "this trainer returns no predictions.")
    assert "this trainer returns no predictions." in str(err.value)


def test_activation_offloading_rejection_names_the_bypassed_wrapper():
    with pytest.raises(ValueError) as err:
        reject_unsupported_pp_training_args(
            _training_args(activation_offloading=True), save_sharded_ep=False, evaluates=False
        )
    assert "training_step" in str(err.value) and "never engage" in str(err.value)


def _right_padded_tokenizer(pad_token: str | None = "<pad>") -> PreTrainedTokenizerFast:
    backend = Tokenizer(WordLevel({"<unk>": 0, "<pad>": 1, "a": 2}, unk_token="<unk>"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>", pad_token=pad_token)
    tokenizer.padding_side = "right"
    return tokenizer


def _pooling_config(pad_token_id: int | None) -> PretrainedConfig:
    return PretrainedConfig(pad_token_id=pad_token_id)


def test_the_fixed_shape_helpers_accept_what_the_pipeline_can_take():
    """Anti-vacuity for the shape refusals below: each one passes the input a pipeline does take."""
    tokenizer = _right_padded_tokenizer()
    reject_pp_collator(
        DataCollatorWithPacking(tokenizer), per_device_train_batch_size=2, n_microbatches=2, model_config=None
    )
    reject_pp_collator(object(), per_device_train_batch_size=4, n_microbatches=2, model_config=None)
    assert require_pad_token_id(tokenizer, "why") == tokenizer.pad_token_id
    assert require_pooling_pad_id(_pooling_config(1), tokenizer_pad_id=1) == 1
    assert require_pooling_pad_id(_pooling_config(1), tokenizer_pad_id=None) == 1


@pytest.mark.parametrize(
    ("refusal", "match"),
    [
        (
            lambda: reject_unsupported_pp_training_args(
                _training_args(max_length=None), save_sharded_ep=False, evaluates=False
            ),
            r"requires a fixed sequence length .* Set max_length",
        ),
        (
            lambda: reject_unsupported_pp_training_args(
                _training_args(torch_compile=True), save_sharded_ep=False, evaluates=False
            ),
            r"torch_compile .* the schedule captures the stage module at setup",
        ),
        (
            lambda: reject_pp_collator(None, per_device_train_batch_size=1, n_microbatches=1, model_config=None),
            r"requires a data collator; the trainer has none",
        ),
        (
            lambda: reject_pp_collator(
                DataCollatorWithFlattening(tokenizer=_right_padded_tokenizer()),
                per_device_train_batch_size=1,
                n_microbatches=1,
                model_config=None,
            ),
            r"DataCollatorWithFlattening \(padding-free\) .* summed document length",
        ),
        (
            lambda: reject_pp_collator(
                DataCollatorWithPacking(_right_padded_tokenizer()),
                per_device_train_batch_size=2,
                n_microbatches=1,
                model_config=None,
            ),
            r"one packed row per microbatch, .* leaks attention across documents",
        ),
        (
            lambda: require_pad_token_id(_right_padded_tokenizer(pad_token=None), "PP pads every batch"),
            r"^PP pads every batch and needs the tokenizer's pad_token_id",
        ),
        (
            lambda: require_pooling_pad_id(_pooling_config(0), tokenizer_pad_id=1),
            r"config.pad_token_id \(0\) disagrees with the tokenizer's pad_token_id \(1\)",
        ),
    ],
    ids=[
        "no-max-length",
        "torch-compile",
        "no-collator",
        "padding-free-collator",
        "packing-rows-per-microbatch",
        "no-pad-token",
        "pooling-pad-disagrees",
    ],
)
def test_each_fixed_shape_refusal_names_its_mechanism(refusal, match):
    with pytest.raises(ValueError, match=match):
        refusal()


@pytest.mark.parametrize("missing", ["model", "args"])
def test_positional_model_or_args_is_rejected(missing):
    kwargs = {"model": object(), "args": object()}
    del kwargs[missing]
    with pytest.raises(ValueError, match="passed as keywords"):
        reject_positional_model_or_args(kwargs)


def test_a_missing_reference_column_names_the_column_and_the_split():
    with pytest.raises(ValueError) as err:
        reject_missing_precomputed_columns(_Dataset("prompt"), ("ref_logps",), role="eval")
    text = str(err.value)
    assert "'ref_logps'" in text and "eval dataset" in text


def test_a_partially_precomputed_pair_is_still_refused():
    """``all(...)`` not ``any(...)``: DPO needs BOTH columns, and a half-precomputed dataset would
    otherwise slip through to a TRL sweep that no stage can run."""
    with pytest.raises(ValueError, match="ref_chosen_logps"):
        reject_missing_precomputed_columns(_Dataset("ref_chosen_logps"), ("ref_chosen_logps", "ref_rejected_logps"))


def test_precompute_flag_off_is_refused_before_the_columns_are_even_checked():
    with pytest.raises(ValueError, match="precompute-only"):
        reject_missing_precomputed_reference(
            "KTO", _training_args(precompute_ref_log_probs=False), _Dataset("ref_logps"), None, ("ref_logps",)
        )


def test_the_eval_dataset_is_checked_too_including_a_dict_of_splits():
    """TRL's in-init sweep runs over the eval dataset as well, so a train-only check would let the
    sweep start on a rank holding one bare stage."""
    train = _Dataset("ref_logps")
    with pytest.raises(ValueError) as err:
        reject_missing_precomputed_reference("KTO", _training_args(), train, _Dataset("prompt"), ("ref_logps",))
    assert "eval dataset" in str(err.value)

    with pytest.raises(ValueError, match="eval dataset"):
        reject_missing_precomputed_reference(
            "KTO", _training_args(), train, {"a": _Dataset("ref_logps"), "b": _Dataset("prompt")}, ("ref_logps",)
        )
    # A dict whose every split carries the column is accepted (anti-vacuity for the loop above).
    reject_missing_precomputed_reference(
        "KTO", _training_args(), train, {"a": _Dataset("ref_logps"), "b": _Dataset("ref_logps")}, ("ref_logps",)
    )


# ── The wiring: the real validators must still call the vocabulary ──────────────


def _pp_config(is_pp: bool = True):
    return SimpleNamespace(is_pp_mode=is_pp)


def _drive_gates(trainer_cls, **kwargs):
    """The REAL construction-time walk: ``_split_pipeline_stage`` plus the trainer's own
    ``_reject_unsupported_pp_options`` hook, which it drives once the PP early-out has passed.

    ``object.__new__`` rather than a namespace: the hook is dispatched through ``self``, so the
    walk has to run on the trainer class itself. The two attributes below are all it reads before
    the gates under test.
    """
    stub = object.__new__(trainer_cls)
    stub.parallelism_config = kwargs.pop("parallelism_config")
    stub.save_sharded_ep = False
    PipelineTrainerMixin._split_pipeline_stage(stub, kwargs, kwargs["args"])


def _drive_dpo(**overrides):
    kwargs = {
        "model": object(),
        "args": _dpo_args(),
        "parallelism_config": _pp_config(),
        "train_dataset": _Dataset("ref_chosen_logps", "ref_rejected_logps"),
    }
    kwargs.update(overrides)
    _drive_gates(DistributedDPOTrainer, **kwargs)


def _drive_kto(**overrides):
    kwargs = {
        "model": object(),
        "args": _kto_args(),
        "parallelism_config": _pp_config(),
        "train_dataset": _Dataset("ref_logps"),
    }
    kwargs.update(overrides)
    _drive_gates(DistributedKTOTrainer, **kwargs)


def _drive_smpo(peft_config=None):
    """SMPO gates its explicit ctor parameters itself — the mixin's kwargs hook never sees them."""
    args = SimpleNamespace(padding_free=False, lower_clip_percentile=None, upper_clip_percentile=None)
    reject_smpo_pp_options(args, _pp_config(), peft_config, is_vlm=False)


def _drive_offline_grpo(peft_config=None, compute_metrics=None):
    """Offline GRPO likewise: ``peft_config`` and ``compute_metrics`` are explicit ctor parameters."""
    reject_offline_pp_options(
        _training_args(max_prompt_length=64, max_completion_length=64), _pp_config(), peft_config, compute_metrics
    )


def _drive_pipeline_mixin(**overrides):
    """``PipelineTrainerMixin._split_pipeline_stage`` — the shared gate SFT/reward ride."""
    kwargs = {
        "model": _tiny_lm(),
        "args": _training_args(),
        "parallelism_config": _pp_config(),
    }
    kwargs.update(overrides)
    _drive_gates(PipelineTrainerMixin, **kwargs)


_PEFT_CALL_SITES = {
    "dpo": lambda: _drive_dpo(peft_config=PEFT_SENTINEL),
    "kto": lambda: _drive_kto(peft_config=PEFT_SENTINEL),
    "smpo": lambda: _drive_smpo(peft_config=PEFT_SENTINEL),
    "offline_grpo": lambda: _drive_offline_grpo(peft_config=PEFT_SENTINEL),
    "pipeline_mixin": lambda: _drive_pipeline_mixin(peft_config=PEFT_SENTINEL),
}


@pytest.mark.parametrize("site", sorted(_PEFT_CALL_SITES))
def test_every_pp_trainer_still_rejects_peft(site):
    """Deleting ``reject_pp_peft(...)`` from one call site fails exactly this case.

    SMPO and offline GRPO are the ones worth the parametrize: their ``peft_config`` is an explicit
    ctor parameter, invisible to the mixin's kwargs-based gate, so nothing upstream would catch it.
    """
    with pytest.raises(ValueError, match="PEFT/LoRA is not supported under pipeline parallelism"):
        _PEFT_CALL_SITES[site]()


@pytest.mark.parametrize("site", sorted(_PEFT_CALL_SITES))
def test_no_pp_trainer_rejects_an_absent_peft_config(site):
    """Anti-vacuity: with ``peft_config=None`` the PEFT message must not appear.

    The pipeline mixin continues past its gate into checks this stub cannot satisfy, so only the
    message is asserted — a gate that raised unconditionally would still be caught.
    """
    sites = {
        "dpo": _drive_dpo,
        "kto": _drive_kto,
        "smpo": _drive_smpo,
        "offline_grpo": _drive_offline_grpo,
        "pipeline_mixin": _drive_pipeline_mixin,
    }
    try:
        sites[site]()
    except Exception as err:  # noqa: BLE001 — a later gate tripping on the stub is fine; a PEFT one is not
        assert "PEFT/LoRA" not in str(err), f"{site} rejects PEFT with no peft_config: {err}"


_PEFT_WRAPS = {
    "peft_model": lambda model: get_peft_model(model, LoraConfig(r=4, target_modules=["q_proj", "v_proj"])),
    "injected_tuner_layers": lambda model: inject_adapter_in_model(LoraConfig(r=4, target_modules=["q_proj"]), model),
    "prompt_learning_peft_model": lambda model: get_peft_model(
        model, PromptTuningConfig(task_type="CAUSAL_LM", num_virtual_tokens=4)
    ),
}


def _split(model, *, pp_size: int = 2) -> dict:
    """The REAL ``_split_pipeline_stage`` on ``model``, through to ``build_pipeline_stage``, as rank 0 of a
    pp2 x dp2 topology (``pp_size=1``: plain dp4)."""
    stub = object.__new__(PipelineTrainerMixin)
    stub.parallelism_config = make_parallelism_config(world_size=4, gpus_per_node=2, pp_size=pp_size)
    stub.save_sharded_ep = False
    stub._moe_balancing = "none"
    args = _training_args(per_device_train_batch_size=2, per_device_eval_batch_size=2, gradient_accumulation_steps=2)
    return PipelineTrainerMixin._split_pipeline_stage(stub, {"model": model, "args": args}, args)


@pytest.mark.parametrize("wrap", sorted(_PEFT_WRAPS))
def test_a_model_wrapped_before_the_trainer_is_refused_at_the_split(wrap):
    """A model the caller PEFT-wrapped itself arrives with ``peft_config=None``, past ``reject_pp_peft``.

    Unrefused, the stage builds with the tuner layers inside it and no ``PeftModel`` above, so the
    saver ladder never finds the adapter and the PP save writes ``base_model.model.*.base_layer.*``
    keys ``from_pretrained`` cannot load. Each wrap is a distinct detection path: the ``PeftModel``
    type (prompt learning carries no tuner layer), and tuner layers injected in place (no ``PeftModel``).
    """
    with pytest.raises(ValueError, match="already PEFT-wrapped") as err:
        _split(_PEFT_WRAPS[wrap](_tiny_lm()))
    assert "fully fine-tune under PP" in str(err.value)


@pytest.mark.parametrize("driver", [_drive_dpo, _drive_kto], ids=["dpo", "kto"])
def test_dpo_and_kto_reach_the_wrapped_model_refusal_past_their_own_hooks(driver):
    """DPO and KTO run their own PP hook ahead of the shared model gate; a wrapped model passes the hook
    and must still meet the gate."""
    with pytest.raises(ValueError, match="already PEFT-wrapped"):
        driver(model=_PEFT_WRAPS["peft_model"](_tiny_lm()))


def test_a_plain_model_passes_the_peft_gate_and_splits():
    """Anti-vacuity: the gate refuses the wrap, not every model the split takes."""
    assert isinstance(_split(_tiny_lm())["model"], PipelineStageModule)


def test_a_peft_model_without_pp_is_untouched():
    """The refusal sits behind ``is_pp_mode``: a non-PP run keeps the caller's ``PeftModel`` as given."""
    model = _PEFT_WRAPS["peft_model"](_tiny_lm())
    assert _split(model, pp_size=1)["model"] is model


def _positional_trl_bases() -> dict[str, type]:
    """Trainer name → TRL base, for each PP-enabled trainer whose ``__init__`` forwards ``*args`` to that base;
    one naming ``peft_config`` binds it by name, which ``reject_pp_peft`` reads."""
    bases = {}
    for cls in distributed_trainer_classes():
        params = inspect.signature(cls.__init__).parameters
        forwards = any(param.kind is param.VAR_POSITIONAL for param in params.values())
        if cls.PP_REFUSAL is None and forwards and "peft_config" not in params:
            bases[cls.__name__] = next(base for base in cls.__mro__ if base.__module__.startswith("trl."))
    return bases


def test_positional_trainers_are_found():
    """Anti-vacuity for the parametrize below."""
    assert _positional_trl_bases()


@pytest.mark.parametrize("trl_base", list(_positional_trl_bases().values()), ids=list(_positional_trl_bases()))
def test_a_positional_peft_config_implies_a_positional_model(trl_base):
    """No gate reads a positional ``peft_config``; none needs to while the TRL base takes ``model`` first, since
    filling the slot by position then leaves no ``model`` keyword, which PP refuses. Read off the installed
    signature, so a TRL release that moved ``peft_config`` ahead of ``model`` fails here."""
    slots = resolve_ctor_positions(trl_base, "model", "peft_config")
    assert slots["model"] < slots["peft_config"], f"{trl_base.__name__} takes peft_config before model"


@pytest.mark.parametrize(
    ("site", "driver"),
    [
        ("dpo", lambda: _drive_dpo(args=_dpo_args(activation_offloading=True))),
        ("kto", lambda: _drive_kto(args=_kto_args(activation_offloading=True))),
        ("pipeline_mixin", lambda: _drive_pipeline_mixin(args=_training_args(activation_offloading=True))),
    ],
)
def test_activation_offloading_is_rejected_at_every_site_that_gates_it(site, driver):
    with pytest.raises(ValueError, match="activation_offloading is not supported"):
        driver()


@pytest.mark.parametrize(
    ("site", "driver"),
    [
        ("dpo", lambda: _drive_dpo(ref_model=object())),
        ("kto", lambda: _drive_kto(ref_model=object())),
    ],
)
def test_an_explicit_reference_model_is_rejected_under_pp(site, driver):
    with pytest.raises(ValueError, match="An explicit ref_model is not supported"):
        driver()


@pytest.mark.parametrize(
    ("site", "driver"),
    [
        ("dpo", lambda: _drive_dpo(compute_metrics=lambda *_: {})),
        ("kto", lambda: _drive_kto(compute_metrics=lambda *_: {})),
        ("offline_grpo", lambda: _drive_offline_grpo(compute_metrics=lambda *_: {})),
    ],
)
def test_compute_metrics_is_rejected_at_every_site_that_gates_it(site, driver):
    with pytest.raises(ValueError, match="compute_metrics is not supported"):
        driver()


_REF_TRAINERS = [(_drive_dpo, _dpo_args), (_drive_kto, _kto_args)]


@pytest.mark.parametrize(("driver", "make_args"), _REF_TRAINERS, ids=["dpo", "kto"])
def test_the_precompute_contract_is_enforced_from_the_real_validator(driver, make_args):
    """The reference columns must already be in the dataset — the gate the DPO/KTO validators reach
    through ``reject_missing_precomputed_reference``, not just the helper in isolation."""
    with pytest.raises(ValueError, match="precompute-only"):
        driver(args=make_args(precompute_ref_log_probs=False))
    with pytest.raises(ValueError, match="ALREADY"):
        driver(train_dataset=_Dataset("prompt"))


@pytest.mark.parametrize(("driver", "make_args"), _REF_TRAINERS, ids=["dpo", "kto"])
def test_the_whole_vocabulary_is_inert_when_pp_is_off(driver, make_args):
    """Every gate above sits behind ``is_pp_mode``: a non-PP run must keep PEFT, a ref model,
    compute_metrics and activation_offloading."""
    driver(
        parallelism_config=_pp_config(is_pp=False),
        peft_config=PEFT_SENTINEL,
        ref_model=object(),
        compute_metrics=lambda *_: {},
        args=make_args(activation_offloading=True, precompute_ref_log_probs=False),
        train_dataset=_Dataset("prompt"),
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
