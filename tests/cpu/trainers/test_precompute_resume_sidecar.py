#!/usr/bin/env python
"""Precomputed reference log-probs ride every checkpoint and replace the sweep on resume.

TRL runs ``precompute_ref_log_probs`` inside ``__init__`` over ``self.ref_model or self.model``. A
Path-B resume (EP, CP, TP and the default ``use_grouped_gemm``) builds the policy from the
checkpoint before the trainer exists, so with no separate reference that sweep scores the TRAINED
weights: the reference equals the policy and every log-ratio is zero. The mixin persists the swept
columns into each checkpoint (``reference_logps.pt``) and, handed the resume checkpoint at
construction, attaches them in place of the sweep once the saved split's row count, token digests
and reference settings match — or refuses where a sweep would be wrong.

The trainers are the real ``DistributedDPOTrainer`` / ``DistributedKTOTrainer`` over a stub reference
forward whose values say which weights produced them: ``BASE`` on the saving run, ``TRAINED`` on the
resumed one (``tests/common/preference_precompute.py``).

    python tests/cpu/trainers/test_precompute_resume_sidecar.py
"""

import contextlib
import datetime
import errno
import hashlib
import os
import re
import stat
from types import SimpleNamespace

import pyarrow as pa
import pytest
import torch
from accelerate import PartialState
from datasets import Dataset, Features, List, Value, concatenate_datasets
from trl import DPOConfig, KTOConfig, ModelConfig

import src.checkpoint.atomic as atomic_mod
import src.trainers.mixins.reference_logps as precompute_mod
from src.args.distributed_args import DistributedArguments
from src.args.dpo_args import DPOScriptArguments
from src.args.kto_args import KTOScriptArguments
from src.checkpoint.format import REFERENCE_LOGPS_FILE
from src.distributed import runtime
from src.trainers.mixins.checkpointing import CheckpointingMixin
from src.trainers.mixins.reference_logps import token_digest as _token_digest
from src.trainers.preference.precompute import PrecomputeRefLogpsRankConsistentMixin
from src.training.environment import detect_resume_checkpoint
from src.training.parser import H4ArgumentParser
from src.training.script_runner import ScriptRuntime
from tests.common.gloo import run_gloo_ranks
from tests.common.preference_precompute import (
    BASE,
    MAX_LENGTH,
    N_ROWS,
    REFERENCE_COLUMNS,
    SWEEP_BATCH_SIZE,
    TOKEN_COLUMNS,
    TRAINED,
    TRAINERS,
    column,
    precompute_trainer,
    token_rows,
)

PartialState()

# The dataclasses scripts/training/preference/{dpo,kto}.py parse their config into.
SCRIPT_CONFIGS = {
    "dpo": (DPOScriptArguments, DPOConfig, ModelConfig, DistributedArguments),
    "kto": (KTOScriptArguments, KTOConfig, ModelConfig, DistributedArguments),
}

_REFERENCE_WORLD_SIZE = 2
_REFERENCE_PG_TIMEOUT = datetime.timedelta(seconds=120)
_LEGACY_TOKEN_DIGESTS = {
    "dpo": {
        "prompt_ids": "22b7c1ed76f6e7e90ec08c0f1cb911415391eb6198dd44efdaf5050ea7f8e28d",
        "chosen_ids": "b0e3d982120f58bbb6a11b088d85ea429a06549b261deaf1d3e32f7a17539ed4",
        "rejected_ids": "fa3ce5253465f0980bc3fcf4458b8ae0030b43548e3801b059c5c8e0ecf63f7d",
    },
    "kto": {
        "prompt_ids": "22b7c1ed76f6e7e90ec08c0f1cb911415391eb6198dd44efdaf5050ea7f8e28d",
        "completion_ids": "195ce9761f8eba384acd45fc00e455fc8f48768b55e405d95ff225330b3d7f5d",
        "KL_completion_ids": "ca6906fa4c2e8562486a60f017f4ed6371c5baf7201b7f4421a6e26c948f615e",
        "label": "40bc541aead3c7f633f8a38e94e1b5adc2b00db3a5249d46284114ca7abf4706",
    },
}


@pytest.fixture(params=sorted(TRAINERS))
def kind(request):
    return request.param


def _save_base_run(kind, checkpoint_dir, splits, **trainer_kwargs) -> dict[str, Dataset]:
    """A fresh run's precompute over ``splits`` (name → dataset), persisted into ``checkpoint_dir``."""
    trainer = precompute_trainer(kind, weights=BASE, **trainer_kwargs)
    prepared = {
        name: trainer._precompute_ref_logps(dataset, name, SWEEP_BATCH_SIZE) for name, dataset in splits.items()
    }
    trainer._persist_trainer_sidecars(str(checkpoint_dir))
    return prepared


def _resumed(kind, checkpoint_dir, **trainer_kwargs):
    """A trainer resuming the production way, whose own sweep would score the trained weights."""
    return precompute_trainer(
        kind, weights=TRAINED, resume_checkpoint=str(checkpoint_dir), policy_from_checkpoint=True, **trainer_kwargs
    )


def test_the_swept_columns_are_persisted_as_training_reads_them(kind, tmp_path):
    prepared = _save_base_run(kind, tmp_path, {"train": token_rows(kind)})["train"]

    saved = torch.load(tmp_path / REFERENCE_LOGPS_FILE, weights_only=True)
    assert set(saved) == {"train"}
    entry = saved["train"]
    assert entry["num_rows"] == N_ROWS
    assert set(entry["columns"]) == set(REFERENCE_COLUMNS[kind])
    assert set(entry["token_digests"]) == set(TOKEN_COLUMNS[kind])
    for name in REFERENCE_COLUMNS[kind]:
        assert prepared.features[name].dtype == "float32"
        assert torch.equal(entry["columns"][name], column(prepared, name))


@pytest.mark.parametrize("umask", [0o002, 0o022, 0o077])
def test_a_fresh_reference_sidecar_takes_the_umask_mode(kind, tmp_path, umask):
    previous = os.umask(umask)
    try:
        _save_base_run(kind, tmp_path, {"train": token_rows(kind)})
    finally:
        os.umask(previous)

    path = tmp_path / REFERENCE_LOGPS_FILE
    assert stat.S_IMODE(path.stat().st_mode) == 0o666 & ~umask
    assert torch.load(path, weights_only=True)["train"]["num_rows"] == N_ROWS


def test_a_resume_attaches_the_saved_columns_instead_of_sweeping_the_trained_policy(kind, tmp_path):
    base = _save_base_run(kind, tmp_path, {"train": token_rows(kind)})["train"]
    resumed = _resumed(kind, tmp_path)

    prepared = resumed._precompute_ref_logps(token_rows(kind), "train", SWEEP_BATCH_SIZE)

    assert resumed.compute_ref_log_probs.batches == 0, "the resume swept the policy, which holds the trained weights"
    for name in REFERENCE_COLUMNS[kind]:
        assert torch.equal(column(prepared, name), column(base, name))
    # The resumed run's own checkpoints carry the same values on, so a second resume restores them.
    resumed._persist_trainer_sidecars(str(tmp_path / "next"))
    again = torch.load(tmp_path / "next" / REFERENCE_LOGPS_FILE, weights_only=True)["train"]
    first = torch.load(tmp_path / REFERENCE_LOGPS_FILE, weights_only=True)["train"]
    assert again["token_digests"] == first["token_digests"]
    assert again["settings"] == first["settings"]
    for name in REFERENCE_COLUMNS[kind]:
        assert torch.equal(again["columns"][name], first["columns"][name])


def test_pre_extraction_reference_schema_resumes_without_rewriting_its_identity(kind, tmp_path):
    """The fixed fixture pins the existing sidecar schema and token digest, not the new writer."""
    columns = (
        {
            "ref_chosen_logps": torch.tensor([-6.0, -12.0, -18.0, -24.0]),
            "ref_rejected_logps": torch.tensor([-106.0, -112.0, -118.0, -124.0]),
        }
        if kind == "dpo"
        else {
            "ref_logps": torch.tensor([-9.0, -9.0, -11.0, -11.0]),
            "ref_KL_logps": torch.tensor([-109.0, -109.0, -111.0, -111.0]),
        }
    )
    settings = {"max_length": MAX_LENGTH, "logprob_precision": "float32"}
    if kind == "dpo":
        settings.update(truncation_mode="keep_start", ld_alpha=None)
    legacy = {
        "num_rows": N_ROWS,
        "token_digests": _LEGACY_TOKEN_DIGESTS[kind],
        "settings": settings,
        "columns": columns,
    }
    torch.save({"train": legacy}, tmp_path / REFERENCE_LOGPS_FILE)

    trainer = _resumed(kind, tmp_path)
    restored = trainer._precompute_ref_logps(token_rows(kind), "train", SWEEP_BATCH_SIZE)
    assert trainer.compute_ref_log_probs.batches == 0
    for name, expected in columns.items():
        assert torch.equal(column(restored, name), expected)

    trainer._persist_trainer_sidecars(str(tmp_path / "next"))
    saved = torch.load(tmp_path / "next" / REFERENCE_LOGPS_FILE, weights_only=True)["train"]
    assert saved.keys() == legacy.keys()
    for name in ("num_rows", "token_digests", "settings"):
        assert saved[name] == legacy[name]
    for name, expected in columns.items():
        assert torch.equal(saved["columns"][name], expected)


def test_a_resumed_dataset_with_a_new_fingerprint_still_restores(kind, tmp_path):
    """The identity is the token content: a fingerprint re-drawn by an unhashable map must not refuse."""
    _save_base_run(kind, tmp_path, {"train": token_rows(kind)})
    dataset = token_rows(kind)
    dataset._fingerprint = "redrawn-per-process"
    resumed = _resumed(kind, tmp_path)

    resumed._precompute_ref_logps(dataset, "train", SWEEP_BATCH_SIZE)

    assert resumed.compute_ref_log_probs.batches == 0


def test_a_policy_built_from_the_checkpoint_without_saved_columns_refuses(kind, tmp_path):
    trainer = _resumed(kind, tmp_path)

    with pytest.raises(RuntimeError, match="TRAINED weights as the reference") as raised:
        trainer._precompute_ref_logps(token_rows(kind), "train", SWEEP_BATCH_SIZE)
    assert "this resume built the policy from the checkpoint, so the sweep would score the TRAINED weights" in str(
        raised.value
    ), "the refusal must explain why this resume cannot sweep the policy"
    assert "--max_steps=1" in str(raised.value), "the refusal must name a way to recover"
    assert trainer.compute_ref_log_probs.batches == 0


def test_the_named_recovery_parses_into_a_fresh_one_step_save(kind, tmp_path):
    """The flags the refusal prints go through the entry script's own parser onto the config that
    resumed: they must start a fresh run from the base that saves checkpoint-1 and stops, outside
    the run's own output_dir, whose rotation could otherwise delete the checkpoint being recovered.
    Anything less and the recovery the refusal names cannot produce the sidecar it asks for."""
    run_dir = tmp_path / "run"
    resumed_from = run_dir / "checkpoint-7"
    resumed_from.mkdir(parents=True)
    (resumed_from / "trainer_state.json").write_text("{}")
    trainer = _resumed(kind, resumed_from)
    with pytest.raises(RuntimeError, match="To recover") as raised:
        trainer._precompute_ref_logps(token_rows(kind), "train", SWEEP_BATCH_SIZE)
    assert "on every node" in str(raised.value), "per-node storage needs a copy on each node"
    recovery = re.findall(r"--\w+=[^\s)]+", str(raised.value))
    assert recovery, "the refusal names no flags to recover with"

    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"model_name_or_path: base/model\noutput_dir: {run_dir}\nbf16: false\nuse_cpu: true\n"
        f"precompute_ref_log_probs: true\nsave_total_limit: 1\nresume_from_checkpoint: {resumed_from}\n"
    )
    as_resumed = H4ArgumentParser(SCRIPT_CONFIGS[kind]).parse_yaml_and_args(str(config_path), [])[1]
    assert detect_resume_checkpoint(as_resumed) == str(resumed_from), "premise: the config resumes"

    config = H4ArgumentParser(SCRIPT_CONFIGS[kind]).parse_yaml_and_args(str(config_path), recovery)[1]

    assert detect_resume_checkpoint(config) is None, "the recovery run must start from the base"
    assert (config.max_steps, config.save_strategy, config.save_steps) == (1, "steps", 1)
    assert config.save_only_model is True
    assert os.path.commonpath([os.path.abspath(config.output_dir), str(run_dir)]) != str(run_dir), (
        f"the recovery writes into the resumed run's output_dir ({config.output_dir})"
    )


@pytest.mark.parametrize(
    ("policy_from_checkpoint", "ref_model"),
    [(False, None), (True, torch.nn.Linear(1, 1))],
    ids=["policy_from_base", "separate_reference"],
)
def test_a_resume_sweeps_where_the_reference_weights_are_untrained(kind, tmp_path, policy_from_checkpoint, ref_model):
    """An adapter checkpoint (the policy resumes the base) or a separate reference model: the sweep
    scores untrained weights, so a checkpoint without the sidecar still resumes."""
    trainer = precompute_trainer(
        kind, resume_checkpoint=str(tmp_path), policy_from_checkpoint=policy_from_checkpoint, ref_model=ref_model
    )

    trainer._precompute_ref_logps(token_rows(kind), "train", SWEEP_BATCH_SIZE)

    assert trainer.compute_ref_log_probs.batches > 0


@pytest.mark.parametrize(
    ("resumed_rows", "reason"),
    [
        (lambda kind: token_rows(kind, N_ROWS - 1), f"saved for {N_ROWS} rows"),
        (lambda kind: token_rows(kind).select(list(reversed(range(N_ROWS)))), "differ from the saved run's"),
    ],
    ids=["fewer_rows", "reordered_rows"],
)
def test_saved_columns_for_other_rows_refuse(kind, tmp_path, resumed_rows, reason):
    _save_base_run(kind, tmp_path, {"train": token_rows(kind)})
    trainer = _resumed(kind, tmp_path)

    with pytest.raises(ValueError, match=reason):
        trainer._precompute_ref_logps(resumed_rows(kind), "train", SWEEP_BATCH_SIZE)
    assert trainer.compute_ref_log_probs.batches == 0


@pytest.mark.parametrize(
    ("kind", "bumped"),
    [("dpo", name) for name in TOKEN_COLUMNS["dpo"]] + [("kto", name) for name in TOKEN_COLUMNS["kto"]],
)
def test_every_column_the_reference_reads_is_in_the_digest(tmp_path, kind, bumped):
    """One changed value in any token-id or label column must refuse the saved split."""
    _save_base_run(kind, tmp_path, {"train": token_rows(kind)})
    trainer = _resumed(kind, tmp_path)

    with pytest.raises(ValueError, match=f"'{bumped}'") as raised:
        trainer._precompute_ref_logps(token_rows(kind, bump=bumped), "train", SWEEP_BATCH_SIZE)
    if bumped == "KL_completion_ids":
        # TRL pairs the KL completions within map batches of the per-device size, across num_proc shards.
        assert "per_device_train_batch_size" in str(raised.value)
        assert "dataset_num_proc" in str(raised.value)


@pytest.mark.parametrize(
    ("kind", "changed"),
    [
        ("dpo", {"max_length": 8}),
        ("dpo", {"truncation_mode": "keep_end"}),
        ("dpo", {"ld_alpha": 0.5}),
        ("kto", {"max_length": 8}),
    ],
    ids=["dpo_max_length", "dpo_truncation_mode", "dpo_ld_alpha", "kto_max_length"],
)
def test_changed_reference_settings_refuse(tmp_path, kind, changed):
    """The collator truncates at batch time and ``ld_alpha`` reshapes DPO's sums, so a resume that
    changes either reads other values off the same tokens."""
    _save_base_run(kind, tmp_path, {"train": token_rows(kind)})
    trainer = _resumed(kind, tmp_path)
    for name, value in changed.items():
        if name == "ld_alpha":
            trainer.ld_alpha = value
        else:
            setattr(trainer.args, name, value)

    with pytest.raises(ValueError, match="computed under"):
        trainer._precompute_ref_logps(token_rows(kind), "train", SWEEP_BATCH_SIZE)


@pytest.mark.parametrize("changed", ["tokens", "settings"])
def test_a_reference_mismatch_names_the_data_and_settings_to_resume_with(kind, tmp_path, changed):
    _save_base_run(kind, tmp_path, {"train": token_rows(kind)})
    trainer = _resumed(kind, tmp_path)
    dataset = token_rows(kind, bump="prompt_ids") if changed == "tokens" else token_rows(kind)
    if changed == "settings":
        trainer.args.max_length = MAX_LENGTH // 2

    with pytest.raises(ValueError, match="does not belong to this 'train' dataset") as raised:
        trainer._precompute_ref_logps(dataset, "train", SWEEP_BATCH_SIZE)

    message = str(raised.value)
    assert "Resume with the data and reference settings the checkpoint was written with" in message
    assert "row's reference" in message
    assert "--max_steps=1" in message, "the refusal must also name how to regenerate the file"
    assert trainer.compute_ref_log_probs.batches == 0


def test_a_split_saved_at_the_runs_precision_is_restored(kind, tmp_path):
    """Every saved split records the precision its log-probs were summed in, and a resume summing at
    that precision attaches it."""
    _save_base_run(kind, tmp_path, {"train": token_rows(kind)})
    saved = torch.load(tmp_path / REFERENCE_LOGPS_FILE, weights_only=True)["train"]
    assert saved["settings"]["logprob_precision"] == TRAINERS[kind][0].logprob_precision

    resumed = _resumed(kind, tmp_path)
    resumed._precompute_ref_logps(token_rows(kind), "train", SWEEP_BATCH_SIZE)

    assert resumed.compute_ref_log_probs.batches == 0


@pytest.mark.parametrize("saved_precision", [None, "bfloat16"], ids=["unrecorded", "bfloat16"])
def test_a_split_summed_at_another_precision_is_not_reused(kind, tmp_path, saved_precision):
    """A split whose settings record another log-prob precision, or none, carries that precision's
    rounding: a resume whose sweep would score the trained weights refuses it and names how to
    regenerate the file, and any other resume re-sweeps."""
    _save_base_run(kind, tmp_path, {"train": token_rows(kind)})
    saved = torch.load(tmp_path / REFERENCE_LOGPS_FILE, weights_only=True)
    settings = saved["train"]["settings"]
    if saved_precision is None:
        settings.pop("logprob_precision", None)
    else:
        settings["logprob_precision"] = saved_precision
    torch.save(saved, tmp_path / REFERENCE_LOGPS_FILE)
    precision = TRAINERS[kind][0].logprob_precision

    resumed = _resumed(kind, tmp_path)
    with pytest.raises(ValueError, match=f"^Regenerate .* summed at .* this run sums them in {precision}") as raised:
        resumed._precompute_ref_logps(token_rows(kind), "train", SWEEP_BATCH_SIZE)
    assert "--max_steps=1" in str(raised.value), "the refusal must name how to regenerate the file"
    assert "settings the checkpoint was written with" not in str(raised.value), "no setting selects the precision"
    assert resumed.compute_ref_log_probs.batches == 0
    from_base = precompute_trainer(kind, resume_checkpoint=str(tmp_path), policy_from_checkpoint=False)
    from_base._precompute_ref_logps(token_rows(kind), "train", SWEEP_BATCH_SIZE)
    assert from_base.compute_ref_log_probs.batches > 0, "the old-precision split was reused, not re-swept"


def test_a_trainer_without_a_declared_precision_refuses_before_sweeping(kind):
    """The precision is every saved split's identity; a trainer that does not declare it would save
    splits a resume could not tell apart from another precision's."""
    trainer = precompute_trainer(kind)
    trainer.logprob_precision = None

    with pytest.raises(NotImplementedError, match="logprob_precision"):
        trainer._precompute_ref_logps(token_rows(kind), "train", SWEEP_BATCH_SIZE)
    assert trainer.compute_ref_log_probs.batches == 0


@pytest.mark.parametrize(
    ("policy_from_checkpoint", "ref_model"),
    [(False, None), (True, torch.nn.Linear(1, 1))],
    ids=["policy_from_base", "separate_reference"],
)
@pytest.mark.parametrize("changed", ["rows", "max_length"])
def test_a_mismatched_split_is_swept_where_the_reference_weights_are_untrained(
    kind, tmp_path, policy_from_checkpoint, ref_model, changed
):
    """Where the sweep scores untrained weights (an adapter or merged-adapter resume builds the
    policy from the base; a separate reference model), a saved split that does not match the
    resumed rows or settings is swept afresh: only a sweep over trained weights must refuse."""
    _save_base_run(kind, tmp_path, {"train": token_rows(kind)})
    trainer = precompute_trainer(
        kind,
        weights=BASE,
        resume_checkpoint=str(tmp_path),
        policy_from_checkpoint=policy_from_checkpoint,
        ref_model=ref_model,
        max_length=8 if changed == "max_length" else MAX_LENGTH,
    )
    rows = token_rows(kind, N_ROWS - 1) if changed == "rows" else token_rows(kind)

    prepared = trainer._precompute_ref_logps(rows, "train", SWEEP_BATCH_SIZE)

    assert trainer.compute_ref_log_probs.batches > 0, "the mismatched split was neither refused nor swept"
    assert len(prepared) == len(rows)
    trainer._persist_trainer_sidecars(str(tmp_path / "next"))
    swept = torch.load(tmp_path / "next" / REFERENCE_LOGPS_FILE, weights_only=True)["train"]
    assert swept["num_rows"] == len(rows)
    assert swept["settings"]["max_length"] == trainer.args.max_length


@pytest.mark.parametrize(
    "corrupt",
    [
        lambda entry: entry.pop("token_digests"),
        lambda entry: entry.update(num_rows=str(N_ROWS)),
        lambda entry: entry["columns"].update({name: values[:-1] for name, values in entry["columns"].items()}),
    ],
    ids=["missing_key", "mistyped_rows", "short_columns"],
)
def test_a_malformed_saved_split_is_a_mismatch_not_a_crash(kind, tmp_path, corrupt):
    """A malformed entry must reach the verdict every rank joins: raised on one node's bad copy
    alone, it would leave the others waiting in the collective that follows."""
    _save_base_run(kind, tmp_path, {"train": token_rows(kind)})
    saved = torch.load(tmp_path / REFERENCE_LOGPS_FILE, weights_only=True)
    corrupt(saved["train"])
    torch.save(saved, tmp_path / REFERENCE_LOGPS_FILE)

    with pytest.raises(ValueError, match="does not belong to this 'train' dataset"):
        _resumed(kind, tmp_path)._precompute_ref_logps(token_rows(kind), "train", SWEEP_BATCH_SIZE)
    from_base = precompute_trainer(kind, resume_checkpoint=str(tmp_path), policy_from_checkpoint=False)
    from_base._precompute_ref_logps(token_rows(kind), "train", SWEEP_BATCH_SIZE)
    assert from_base.compute_ref_log_probs.batches > 0


def test_a_split_the_resumed_run_skips_rides_into_its_checkpoints(kind, tmp_path):
    """A resume with its eval split switched off still writes that split's saved reference into the
    checkpoints it makes, so a later resume that turns eval back on restores it rather than refusing."""
    base = _save_base_run(kind, tmp_path / "a", {"train": token_rows(kind), "eval": token_rows(kind, 3)})
    train_only = _resumed(kind, tmp_path / "a")
    train_only._precompute_ref_logps(token_rows(kind), "train", SWEEP_BATCH_SIZE)
    train_only._persist_trainer_sidecars(str(tmp_path / "b"))

    with_eval = _resumed(kind, tmp_path / "b")
    prepared = with_eval._precompute_ref_logps(token_rows(kind, 3), "eval", SWEEP_BATCH_SIZE)

    assert with_eval.compute_ref_log_probs.batches == 0
    for name in REFERENCE_COLUMNS[kind]:
        assert torch.equal(column(prepared, name), column(base["eval"], name))


def test_a_kto_resume_that_needs_the_kl_column_the_save_lacks_refuses(tmp_path):
    """The needed set is derived per run (``calculate_KL``), so a resume under a KL loss cannot be
    served by a save written under a KL-free one."""
    kl_free = token_rows("kto").remove_columns(["KL_completion_ids"])
    _save_base_run("kto", tmp_path, {"train": kl_free}, calculate_kl=False)
    trainer = _resumed("kto", tmp_path, calculate_kl=True)

    with pytest.raises(ValueError, match="ref_KL_logps"):
        trainer._precompute_ref_logps(token_rows("kto"), "train", SWEEP_BATCH_SIZE)


def test_a_kto_resume_dropping_the_kl_term_restores_ref_logps(tmp_path):
    """The reverse direction is served: the saved ``ref_logps`` still belongs to these rows, and the
    KL completions the save digested are no longer read."""
    base = _save_base_run("kto", tmp_path, {"train": token_rows("kto")}, calculate_kl=True)["train"]
    trainer = _resumed("kto", tmp_path, calculate_kl=False)

    prepared = trainer._precompute_ref_logps(
        token_rows("kto").remove_columns(["KL_completion_ids"]), "train", SWEEP_BATCH_SIZE
    )

    assert trainer.compute_ref_log_probs.batches == 0
    assert torch.equal(column(prepared, "ref_logps"), column(base, "ref_logps"))
    assert "ref_KL_logps" not in prepared.column_names


def test_every_split_is_restored_under_its_own_name(kind, tmp_path):
    """TRL precomputes the train split and each eval dataset by name; a split the save never held
    refuses under a policy built from the checkpoint."""
    splits = {"train": token_rows(kind), "eval_a": token_rows(kind, 2), "eval_b": token_rows(kind, 3)}
    base = _save_base_run(kind, tmp_path, splits)
    resumed = _resumed(kind, tmp_path)

    for name, dataset in splits.items():
        prepared = resumed._precompute_ref_logps(dataset, name, SWEEP_BATCH_SIZE)
        for reference in REFERENCE_COLUMNS[kind]:
            assert torch.equal(column(prepared, reference), column(base[name], reference)), name
    assert resumed.compute_ref_log_probs.batches == 0
    with pytest.raises(RuntimeError, match="lacks that split"):
        resumed._precompute_ref_logps(token_rows(kind, 2), "eval_c", SWEEP_BATCH_SIZE)


def test_a_split_saved_on_some_ranks_only_refuses_without_sweeping(kind, tmp_path, monkeypatch):
    """A node whose copy differs must stop the whole world before any rank enters the sweep's
    collectives; the partial verdict comes from the presence consensus."""
    _save_base_run(kind, tmp_path, {"train": token_rows(kind)})
    monkeypatch.setattr(precompute_mod, "rank_consensus", lambda local_ok: (False, True))
    trainer = _resumed(kind, tmp_path)

    with pytest.raises(RuntimeError, match="on some ranks only"):
        trainer._precompute_ref_logps(token_rows(kind), "train", SWEEP_BATCH_SIZE)
    assert trainer.compute_ref_log_probs.batches == 0


def test_the_sidecar_is_written_by_the_save_rank_inside_the_fence(kind, tmp_path, monkeypatch):
    trainer = precompute_trainer(kind)
    trainer._precompute_ref_logps(token_rows(kind), "train", SWEEP_BATCH_SIZE)
    path = tmp_path / REFERENCE_LOGPS_FILE
    fence: list[tuple[str, bool]] = []

    @contextlib.contextmanager
    def recording_fence():
        fence.append(("enter", path.exists()))
        try:
            yield
        finally:
            fence.append(("exit", path.exists()))

    monkeypatch.setattr(precompute_mod, "barrier_on_exit", recording_fence)
    monkeypatch.setattr(precompute_mod, "fs_aware_save_rank", lambda: False)
    trainer._persist_trainer_sidecars(str(tmp_path))
    assert not path.exists(), "a rank that is not its node's save rank wrote the sidecar"
    assert fence == [("enter", False), ("exit", False)], "every rank must enter the fence"

    fence.clear()
    monkeypatch.setattr(precompute_mod, "fs_aware_save_rank", lambda: True)
    trainer._persist_trainer_sidecars(str(tmp_path))
    assert fence == [("enter", False), ("exit", True)], "the save rank must write inside the fence"


def test_dataset_supplied_columns_are_not_persisted(kind, tmp_path):
    """Columns the dataset already carries (the PP contract) ship with it again on resume."""
    trainer = precompute_trainer(kind)
    supplied = {name: [-1.0] * N_ROWS for name in REFERENCE_COLUMNS[kind]}
    dataset = concatenate_datasets([token_rows(kind), Dataset.from_dict(supplied)], axis=1)

    assert trainer._precompute_ref_logps(dataset, "train", SWEEP_BATCH_SIZE) is dataset
    trainer._persist_trainer_sidecars(str(tmp_path))

    assert not (tmp_path / REFERENCE_LOGPS_FILE).exists()


def test_partial_reference_write_does_not_replace_the_complete_sidecar(kind, tmp_path, monkeypatch):
    _save_base_run(kind, tmp_path, {"train": token_rows(kind)})
    path = tmp_path / REFERENCE_LOGPS_FILE
    complete = path.read_bytes()
    trainer = precompute_trainer(kind, weights=TRAINED)
    trainer._precompute_ref_logps(token_rows(kind), "train", SWEEP_BATCH_SIZE)

    def interrupted_save(payload, destination):
        with open(destination, "wb") as incomplete:
            incomplete.write(b"partial sidecar")
        raise OSError("interrupted reference write")

    monkeypatch.setattr(precompute_mod.torch, "save", interrupted_save)
    with pytest.raises(RuntimeError, match="interrupted reference write"):
        trainer._persist_trainer_sidecars(str(tmp_path))
    assert path.read_bytes() == complete
    assert not list(tmp_path.glob(f".{REFERENCE_LOGPS_FILE}.*"))


def _atomic_save_inputs(tmp_path, monkeypatch, reuse):
    previous = tmp_path / "previous.pt"
    payload = {"values": torch.tensor([-1.0, -2.0])}
    torch.save(payload, previous)
    calls = []

    def build_payload():
        calls.append("serialize")
        return payload

    def cross_filesystem(*args, **kwargs):
        raise OSError(errno.EXDEV, "separate mounts")

    if reuse == "copy":
        monkeypatch.setattr(atomic_mod.os, "link", cross_filesystem)
    return previous, payload, build_payload, calls


@pytest.mark.parametrize("reuse", ["serialize", "hardlink", "copy"])
def test_reference_save_syncs_file_then_rename_then_directory(tmp_path, monkeypatch, reuse):
    previous, payload, build_payload, calls = _atomic_save_inputs(tmp_path, monkeypatch, reuse)
    path = tmp_path / "checkpoint" / REFERENCE_LOGPS_FILE
    events = []
    original_sync, original_replace = os.fsync, os.replace

    def recording_sync(descriptor):
        events.append("directory_sync" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file_sync")
        original_sync(descriptor)

    def recording_replace(source, destination):
        original_replace(source, destination)
        events.append("rename")

    monkeypatch.setattr(atomic_mod.os, "fsync", recording_sync)
    monkeypatch.setattr(atomic_mod.os, "replace", recording_replace)
    atomic_mod.atomic_torch_save(str(path), build_payload, None if reuse == "serialize" else str(previous))

    assert events == ["file_sync", "rename", "directory_sync"]
    assert calls == (["serialize"] if reuse == "serialize" else [])
    torch.testing.assert_close(torch.load(path, weights_only=True)["values"], payload["values"])


@pytest.mark.parametrize("reuse", ["serialize", "hardlink", "copy"])
def test_reference_directory_sync_failure_propagates_and_closes_descriptor(tmp_path, monkeypatch, reuse):
    previous, _, build_payload, _ = _atomic_save_inputs(tmp_path, monkeypatch, reuse)
    path = tmp_path / "checkpoint" / REFERENCE_LOGPS_FILE
    original_sync = os.fsync
    directory_descriptors = []

    def failed_directory_sync(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory_descriptors.append(descriptor)
            raise OSError(errno.EIO, "reference directory sync failed")
        original_sync(descriptor)

    monkeypatch.setattr(atomic_mod.os, "fsync", failed_directory_sync)
    with pytest.raises(OSError, match="reference directory sync failed"):
        atomic_mod.atomic_torch_save(str(path), build_payload, None if reuse == "serialize" else str(previous))
    assert len(directory_descriptors) == 1
    with pytest.raises(OSError) as closed:
        os.fstat(directory_descriptors[0])
    assert closed.value.errno == errno.EBADF
    assert previous.exists()
    assert path.exists(), "the failure must occur after the rename"
    assert not list(path.parent.glob(f".{REFERENCE_LOGPS_FILE}.*"))


def test_directory_sync_failure_keeps_the_previous_reference_checkpoint(kind, tmp_path, monkeypatch):
    trainer = precompute_trainer(kind)
    trainer._precompute_ref_logps(token_rows(kind), "train", SWEEP_BATCH_SIZE)
    first, second = tmp_path / "first", tmp_path / "second"
    trainer._persist_trainer_sidecars(str(first))
    original_sync = os.fsync

    def failed_directory_sync(descriptor):
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError(errno.EIO, "reference directory sync failed")
        original_sync(descriptor)

    monkeypatch.setattr(atomic_mod.os, "fsync", failed_directory_sync)
    with pytest.raises(RuntimeError, match="reference directory sync failed"):
        trainer._persist_trainer_sidecars(str(second))
    assert trainer._reference_immutable_path == str(first / REFERENCE_LOGPS_FILE)
    assert torch.load(first / REFERENCE_LOGPS_FILE, weights_only=True)["train"]["num_rows"] == N_ROWS


def _reference_writer_failure_worker(rank: int, root: str, shared: bool) -> None:
    runtime.resolve_shared_filesystem_consensus()
    try:
        assert runtime.is_output_shared_filesystem() is shared
        assert runtime.fs_aware_save_rank() is (rank == 0 if shared else True)
        failing_rank = 0 if shared else 1
        node = "shared" if shared else f"node_{rank}"
        for kind in TRAINERS:
            first = os.path.join(root, node, kind, "checkpoint-1")
            second = os.path.join(root, node, kind, "checkpoint-2")
            trainer = precompute_trainer(kind)
            prepared = trainer._precompute_ref_logps(token_rows(kind), "train", SWEEP_BATCH_SIZE)
            trainer._persist_trainer_sidecars(first)
            previous = trainer._reference_immutable_path
            saved_generation = trainer._reference_saved_generation
            trainer._precompute_ref_logps(token_rows(kind, 2), "eval", SWEEP_BATCH_SIZE)

            def fail_publish(source, destination):
                raise OSError(errno.ENOSPC, "reference writer ran out of space")

            with pytest.MonkeyPatch.context() as patch:
                if rank == failing_rank:
                    patch.setattr(atomic_mod.os, "replace", fail_publish)
                with pytest.raises(RuntimeError, match="reference writer ran out of space") as raised:
                    trainer._persist_trainer_sidecars(second)
            assert f"1 of {_REFERENCE_WORLD_SIZE} rank(s)" in str(raised.value)
            assert f"rank {failing_rank}" in str(raised.value)
            assert trainer._reference_immutable_path == previous
            assert trainer._reference_saved_generation == saved_generation

            resumed = _resumed(kind, first)
            restored = resumed._precompute_ref_logps(token_rows(kind), "train", SWEEP_BATCH_SIZE)
            assert resumed.compute_ref_log_probs.batches == 0
            for name in REFERENCE_COLUMNS[kind]:
                assert torch.equal(column(restored, name), column(prepared, name))
    finally:
        runtime.reset_shared_filesystem_consensus()


@pytest.mark.parametrize("shared", [True, False], ids=["shared_fs", "node_local_fs"])
def test_a_reference_writer_failure_rejects_every_rank_before_advancing_checkpoint_state(tmp_path, shared):
    env = {"DIST_OUTPUT_SHARED_FILESYSTEM": "1" if shared else "0"}
    if not shared:
        env.update(LOCAL_RANK="0", LOCAL_WORLD_SIZE="1")
    run_gloo_ranks(
        _reference_writer_failure_worker,
        _REFERENCE_WORLD_SIZE,
        str(tmp_path),
        shared,
        pg_timeout=_REFERENCE_PG_TIMEOUT,
        env=env,
    )


@pytest.mark.parametrize("copy_fallback", [False, True])
def test_unchanged_reference_checkpoints_do_not_serialize_the_token_table_again(
    kind, tmp_path, monkeypatch, copy_fallback
):
    trainer = precompute_trainer(kind)
    trainer._precompute_ref_logps(token_rows(kind), "train", SWEEP_BATCH_SIZE)
    first, second = tmp_path / "first", tmp_path / "second"
    trainer._persist_trainer_sidecars(str(first))

    def refuse_save(*args, **kwargs):
        raise AssertionError("unchanged reference serialized again")

    def cross_filesystem(*args, **kwargs):
        raise OSError(errno.EXDEV, "separate mounts")

    monkeypatch.setattr(precompute_mod.torch, "save", refuse_save)
    if copy_fallback:
        monkeypatch.setattr(atomic_mod.os, "link", cross_filesystem)
    trainer._persist_trainer_sidecars(str(second))
    assert (first / REFERENCE_LOGPS_FILE).read_bytes() == (second / REFERENCE_LOGPS_FILE).read_bytes()
    (first / REFERENCE_LOGPS_FILE).unlink()
    assert torch.load(second / REFERENCE_LOGPS_FILE, weights_only=True)["train"]["num_rows"] == N_ROWS


def test_adding_a_reference_split_invalidates_the_immutable_checkpoint_copy(kind, tmp_path):
    trainer = precompute_trainer(kind)
    trainer._precompute_ref_logps(token_rows(kind), "train", SWEEP_BATCH_SIZE)
    first, second = tmp_path / "first", tmp_path / "second"
    trainer._persist_trainer_sidecars(str(first))
    trainer._precompute_ref_logps(token_rows(kind, 2), "eval", SWEEP_BATCH_SIZE)
    trainer._persist_trainer_sidecars(str(second))
    assert set(torch.load(first / REFERENCE_LOGPS_FILE, weights_only=True)) == {"train"}
    assert set(torch.load(second / REFERENCE_LOGPS_FILE, weights_only=True)) == {"train", "eval"}


def test_two_splits_sharing_a_name_are_refused(kind):
    trainer = precompute_trainer(kind)
    trainer._precompute_ref_logps(token_rows(kind), "train", SWEEP_BATCH_SIZE)

    with pytest.raises(ValueError, match="share the name 'train'"):
        trainer._precompute_ref_logps(token_rows(kind, 2), "train", SWEEP_BATCH_SIZE)


def test_a_resume_request_without_the_resume_context_refuses(kind):
    """A hand-built trainer resumed through ``args`` would sweep before ``train()`` sees the
    checkpoint; the scripts always pass the context, ``None`` included."""
    trainer = precompute_trainer(kind, resume_from_checkpoint="out/checkpoint-7")
    with pytest.raises(ValueError, match="built without resume_checkpoint"):
        trainer._precompute_ref_logps(token_rows(kind), "train", SWEEP_BATCH_SIZE)
    assert trainer.compute_ref_log_probs.batches == 0

    fresh_script_run = precompute_trainer(kind, resume_from_checkpoint=True, resume_checkpoint=None)
    fresh_script_run._precompute_ref_logps(token_rows(kind), "train", SWEEP_BATCH_SIZE)
    assert fresh_script_run.compute_ref_log_probs.batches > 0


def test_the_token_digest_reads_content_in_row_order():
    """Layout-blind (int width, an indices mapping, batch boundaries) but order- and boundary-sensitive."""
    ids = [[5, 6], [7], [8, 9, 10]]
    digest = _token_digest(Dataset.from_dict({"ids": ids}), "ids")

    narrowed = Dataset.from_dict({"ids": ids}, features=Features({"ids": List(Value("int32"))}))
    assert narrowed.features.arrow_schema.field("ids").type == pa.list_(pa.int32())
    assert _token_digest(narrowed, "ids") == digest
    remapped = Dataset.from_dict({"ids": list(reversed(ids))}).select([2, 1, 0])
    assert _token_digest(remapped, "ids") == digest
    original_batch_rows = precompute_mod._DIGEST_BATCH_ROWS
    try:
        precompute_mod._DIGEST_BATCH_ROWS = 2
        assert _token_digest(Dataset.from_dict({"ids": ids}), "ids") == digest
    finally:
        precompute_mod._DIGEST_BATCH_ROWS = original_batch_rows

    assert _token_digest(Dataset.from_dict({"ids": list(reversed(ids))}), "ids") != digest
    assert _token_digest(Dataset.from_dict({"ids": [[5], [6, 7], [8, 9, 10]]}), "ids") != digest


class _RecordingHash:
    """``hashlib.sha256`` that records the size of every update it takes."""

    def __init__(self, sizes: list[int], data: bytes = b""):
        self._hash = hashlib.sha256(data)
        self._sizes = sizes

    def update(self, data: bytes) -> None:
        self._sizes.append(len(data))
        self._hash.update(data)

    def digest(self) -> bytes:
        return self._hash.digest()

    def hexdigest(self) -> str:
        return self._hash.hexdigest()


def test_the_token_digest_hashes_in_bounded_chunks(monkeypatch):
    """Long rows must not widen a whole batch to int64 at once (32k-token rows by the thousand are
    gigabytes per rank), and the chunking must not change the digest a checkpoint already holds."""
    ids = [list(range(50)), list(range(50, 100))]
    digest = _token_digest(Dataset.from_dict({"ids": ids}), "ids")
    sizes: list[int] = []
    monkeypatch.setattr(precompute_mod, "_DIGEST_CHUNK_VALUES", 3)
    monkeypatch.setattr(
        precompute_mod, "hashlib", SimpleNamespace(sha256=lambda data=b"": _RecordingHash(sizes, data))
    )

    assert _token_digest(Dataset.from_dict({"ids": ids}), "ids") == digest
    # The two rows' lengths are one 16-byte update; every value update is at most three int64s.
    assert sizes and max(sizes) <= 3 * 8, f"a hash update took {max(sizes)} bytes"


def test_the_trainers_checkpoint_hook_is_the_precompute_mixins(kind):
    """The mixin must precede ``DistributedTrainerMixin`` in the bases, or ``CheckpointingMixin``'s
    empty default shadows the write and no checkpoint carries the columns."""
    trainer_cls, _ = TRAINERS[kind]
    assert trainer_cls._persist_trainer_sidecars is PrecomputeRefLogpsRankConsistentMixin._persist_trainer_sidecars
    mro = trainer_cls.__mro__
    assert mro.index(PrecomputeRefLogpsRankConsistentMixin) < mro.index(CheckpointingMixin)


@pytest.mark.parametrize(
    ("resume_checkpoint", "model_source", "expected"),
    [
        (None, "base/model", False),
        ("out/checkpoint-7", "base/model", False),
        ("out/checkpoint-7", "out/checkpoint-7", True),
    ],
    ids=["fresh", "path_a", "path_b"],
)
def test_the_runtime_reports_a_policy_built_from_the_checkpoint(resume_checkpoint, model_source, expected):
    runtime = ScriptRuntime(None, "ep2", 0, resume_checkpoint, model_source)
    assert runtime.policy_from_checkpoint is expected


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
