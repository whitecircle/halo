#!/usr/bin/env python
"""The shared PP rejection vocabulary (``src/trainers/mixins/pp_gates.py``) and its call sites.

``test_pp_trainer_gates.py`` covers *which trainers* may run PP (``_supports_pp``). This file covers
what happens once a supported trainer is under PP and carries an option the pipeline cannot honor.
Every helper here guards a **silent** failure — PEFT injection resolving module names against a tree
no stage holds, a full reference model resident beside one stage, TRL's ``activation_offloading``
wrapper that the schedule-driven step bypasses so it never engages — so a deleted call site does not
crash, it trains something else.

Two layers, because each catches a different regression:

* the helpers themselves — inert on the allowed value, and each raise names its mechanism;
* the **wiring**, which drives the REAL ``_validate_pp_mode`` / ``_maybe_prepare_pipeline_model`` of
  every trainer that calls them. Deleting ``reject_pp_peft(...)`` from any single call site fails
  exactly one parametrized case here; helper-only tests would stay green.

Run: python tests/cpu/parallelism/test_pp_gate_vocabulary.py  (or pytest -m cpu)
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.data.spans import LABEL_IGNORE_INDEX
from src.trainers.mixins.pp_gates import (
    reject_pp_activation_offloading,
    reject_pp_compute_metrics,
    reject_pp_peft,
    reject_pp_ref_model,
    require_model_and_args_kwargs,
    require_precomputed_columns,
    require_precomputed_reference,
)

PEFT_SENTINEL = object()
"""Stand-in for a ``peft_config``: the gates test identity against ``None``, never structure."""


class _Dataset:
    """Minimal stand-in for the ``column_names`` protocol the precompute gates read."""

    def __init__(self, *columns: str):
        self.column_names = list(columns)


def _training_args(**overrides):
    """A PP-legal training-args namespace; overrides inject the one option under test.

    Every field is one a validator on the path to a ``pp_gates`` call reads, set to the value that
    lets the walk reach it — so a failure here is always the gate under test, never a neighbour.
    """
    base = {
        "max_length": 128,
        "gradient_checkpointing": False,
        "gradient_checkpointing_kwargs": None,
        "eval_strategy": "no",
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
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
    reject_pp_activation_offloading(_training_args())
    reject_pp_activation_offloading(SimpleNamespace())  # flag absent entirely (getattr default)
    require_model_and_args_kwargs({"model": object(), "args": object()})
    require_precomputed_columns(_Dataset("ref_logps", "prompt"), ("ref_logps",))
    require_precomputed_reference(
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
        reject_pp_activation_offloading(_training_args(activation_offloading=True))
    assert "training_step" in str(err.value) and "never engage" in str(err.value)


@pytest.mark.parametrize("missing", ["model", "args"])
def test_positional_model_or_args_is_rejected(missing):
    kwargs = {"model": object(), "args": object()}
    del kwargs[missing]
    with pytest.raises(ValueError, match="passed as keywords"):
        require_model_and_args_kwargs(kwargs)


def test_a_missing_reference_column_names_the_column_and_the_split():
    with pytest.raises(ValueError) as err:
        require_precomputed_columns(_Dataset("prompt"), ("ref_logps",), role="eval")
    text = str(err.value)
    assert "'ref_logps'" in text and "eval dataset" in text


def test_a_partially_precomputed_pair_is_still_refused():
    """``all(...)`` not ``any(...)``: DPO needs BOTH columns, and a half-precomputed dataset would
    otherwise slip through to a TRL sweep that no stage can run."""
    with pytest.raises(ValueError, match="ref_chosen_logps"):
        require_precomputed_columns(_Dataset("ref_chosen_logps"), ("ref_chosen_logps", "ref_rejected_logps"))


def test_precompute_flag_off_is_refused_before_the_columns_are_even_checked():
    with pytest.raises(ValueError, match="precompute-only"):
        require_precomputed_reference(
            "KTO", _training_args(precompute_ref_log_probs=False), _Dataset("ref_logps"), None, ("ref_logps",)
        )


def test_the_eval_dataset_is_checked_too_including_a_dict_of_splits():
    """TRL's in-init sweep runs over the eval dataset as well, so a train-only check would let the
    sweep start on a rank holding one bare stage."""
    train = _Dataset("ref_logps")
    with pytest.raises(ValueError) as err:
        require_precomputed_reference("KTO", _training_args(), train, _Dataset("prompt"), ("ref_logps",))
    assert "eval dataset" in str(err.value)

    with pytest.raises(ValueError, match="eval dataset"):
        require_precomputed_reference(
            "KTO", _training_args(), train, {"a": _Dataset("ref_logps"), "b": _Dataset("prompt")}, ("ref_logps",)
        )
    # A dict whose every split carries the column is accepted (anti-vacuity for the loop above).
    require_precomputed_reference(
        "KTO", _training_args(), train, {"a": _Dataset("ref_logps"), "b": _Dataset("ref_logps")}, ("ref_logps",)
    )


# ── The wiring: the real validators must still call the vocabulary ──────────────


def _pp_config(is_pp: bool = True):
    return SimpleNamespace(is_pp_mode=is_pp)


def _drive_gates(trainer_cls, **kwargs):
    """The REAL construction-time walk: ``_maybe_prepare_pipeline_model`` plus the trainer's own
    ``_validate_pp_mode`` hook, which it drives once the PP early-out has passed.

    ``object.__new__`` rather than a namespace: the hook is dispatched through ``self``, so the
    walk has to run on the trainer class itself. The two attributes below are all it reads before
    the gates under test.
    """
    from src.trainers.mixins.pipeline import PipelineTrainerMixin

    stub = object.__new__(trainer_cls)
    stub.parallelism_config = kwargs.pop("parallelism_config")
    stub.save_sharded_ep = False
    PipelineTrainerMixin._maybe_prepare_pipeline_model(stub, kwargs, kwargs["args"])


def _drive_dpo(**overrides):
    from src.trainers.preference.dpo import DistributedDPOTrainer

    kwargs = {
        "model": object(),
        "args": _dpo_args(),
        "parallelism_config": _pp_config(),
        "train_dataset": _Dataset("ref_chosen_logps", "ref_rejected_logps"),
    }
    kwargs.update(overrides)
    _drive_gates(DistributedDPOTrainer, **kwargs)


def _drive_kto(**overrides):
    from src.trainers.preference.kto import DistributedKTOTrainer

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
    from src.trainers.preference.smpo import SmoothMarginPOTrainer

    trainer = SimpleNamespace(
        is_vlm=False,
        padding_free=False,
        lower_clip_percentile=None,
        upper_clip_percentile=None,
        label_pad_token_id=LABEL_IGNORE_INDEX,
    )
    SmoothMarginPOTrainer._reject_pp_explicit_options(trainer, _pp_config(), peft_config)


def _drive_offline_grpo(peft_config=None, compute_metrics=None):
    """Offline GRPO likewise: ``peft_config`` and ``compute_metrics`` are explicit ctor parameters."""
    from src.trainers.grpo.offline import OfflineGRPOTrainer

    trainer = SimpleNamespace(
        loss_type="grpo", policy_gradient_formulation="prob_weighted", max_prompt_length=64, max_completion_length=64
    )
    OfflineGRPOTrainer._reject_pp_explicit_options(
        trainer, _training_args(), _pp_config(), peft_config, compute_metrics
    )


def _drive_pipeline_mixin(**overrides):
    """``PipelineTrainerMixin._maybe_prepare_pipeline_model`` — the shared gate SFT/reward ride."""
    from src.trainers.mixins.pipeline import PipelineTrainerMixin

    kwargs = {
        "model": SimpleNamespace(config=SimpleNamespace()),
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
    through ``require_precomputed_reference``, not just the helper in isolation."""
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
