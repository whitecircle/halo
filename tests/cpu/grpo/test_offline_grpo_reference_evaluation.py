"""Dynamic KL evaluation never turns a trained policy into its own reference."""

from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
from torch import nn

from scripts.training.offline_grpo import _requested_attention
from src.data.collators.offline_grpo import REF_PER_TOKEN_LOGPS_COLUMN
from src.distributed.runtime import DeferredRankFailure
from src.trainers.grpo.reference_lifecycle import OfflineGRPOReferenceLifecycleMixin
from tests.common.gloo import run_gloo_ranks
from tests.common.offline_grpo_reference import mapped_scores

_SETTINGS = {"model_type": "fixture", "temperature": 1.0}


def _dataset():
    return Dataset.from_dict(
        {"prompt_input_ids": [[1], [2], [3]], "completion_input_ids": [[4, 5], [6], []], "group_id": [0, 1, 2]}
    )


class _Base:
    def evaluate(self, dataset=None, *, ignore_keys=None, metric_key_prefix="eval"):
        self.evaluated = dataset
        self.eval_options = (ignore_keys, metric_key_prefix)
        if isinstance(dataset, dict):
            return {
                f"{metric_key_prefix}_{name}": self.evaluate(data, metric_key_prefix=f"{metric_key_prefix}_{name}")
                for name, data in dataset.items()
            }
        return {f"{metric_key_prefix}_loss": 42.0}


class _Trainer(OfflineGRPOReferenceLifecycleMixin, _Base):
    def __init__(self, output_dir):
        self.args = SimpleNamespace(output_dir=output_dir)
        self._init_reference_logps(resume_checkpoint=None)
        self._precompute_reference = True
        self.model = nn.Linear(1, 1)
        self.ref_model = None
        self.parallelism_config = SimpleNamespace(is_cp_mode=False)
        self._pp_runtime = None
        self._reference_settings = lambda: dict(_SETTINGS)
        dataset = _dataset()
        self.train_dataset = self._attach_scored_reference_logps(
            dataset,
            "train",
            mapped_scores(output_dir, dataset, [[-0.25, -1.5], [-0.75], []]),
            identity=self._reference_split_identity(dataset, "train", _SETTINGS),
        )


def test_reordered_subset_and_duplicate_rows_reuse_exact_original_scores(tmp_path):
    trainer = _Trainer(tmp_path)
    copied = _dataset().select([1, 0, 1, 2])
    trainer._sweep_reference_logps = lambda *args: pytest.fail("reuse must not sweep the trained model")
    result = trainer.evaluate(copied, ignore_keys=["hidden_states"], metric_key_prefix="heldout")
    assert result == {"heldout_loss": 42.0}
    assert trainer.eval_options == (["hidden_states"], "heldout")
    assert trainer.evaluated[REF_PER_TOKEN_LOGPS_COLUMN] == [[-0.75], [-0.25, -1.5], [-0.75], []]
    assert not trainer._reference_evaluation_datasets


def test_named_eval_splits_preserve_standard_recursive_evaluation(tmp_path):
    trainer = _Trainer(tmp_path)
    result = trainer.evaluate({"first": _dataset().select([0]), "second": _dataset().select([1])})
    assert result == {"eval_first": {"eval_first_loss": 42.0}, "eval_second": {"eval_second_loss": 42.0}}


def test_later_named_split_failure_releases_earlier_temporary_reference_cache(tmp_path):
    trainer = _Trainer(tmp_path)
    unseen = _dataset().select([0]).remove_columns("prompt_input_ids").add_column("prompt_input_ids", [[99]])
    with pytest.raises(ValueError, match="unseen token rows"):
        trainer.evaluate({"known": _dataset().select([0]), "unseen": unseen})
    assert not trainer._reference_evaluation_datasets
    assert trainer._reference_evaluation_depth == 0
    assert not list((tmp_path / "_reference_cache").iterdir()), "failed evaluation left a temporary cache"


def test_unseen_rows_fail_before_any_live_policy_forward_with_a_recovery_path(tmp_path):
    trainer = _Trainer(tmp_path)
    unseen = _dataset().select([0]).remove_columns("prompt_input_ids").add_column("prompt_input_ids", [[99]])
    trainer._sweep_reference_logps = lambda *args: pytest.fail("trained policy cannot recover unseen scores")
    with pytest.raises(ValueError, match="original_reference_model=original_frozen_policy"):
        trainer.evaluate(unseen)


def test_changed_reference_settings_do_not_reuse_old_token_scores(tmp_path):
    trainer = _Trainer(tmp_path)
    trainer._reference_settings = lambda: {**_SETTINGS, "temperature": 2.0}
    with pytest.raises(ValueError, match="unseen token rows"):
        trainer.evaluate(trainer.train_dataset)


@pytest.mark.parametrize("kind", ["live", "trainable", "training", "cp", "pp"])
def test_invalid_original_reference_is_rejected_before_scoring(tmp_path, kind):
    trainer = _Trainer(tmp_path)
    reference = nn.Linear(1, 1).requires_grad_(False).eval()
    if kind == "live":
        reference = trainer.model
    elif kind == "trainable":
        reference.requires_grad_(True)
    elif kind == "training":
        reference.train()
    elif kind == "cp":
        trainer.parallelism_config.is_cp_mode = True
    else:
        trainer._pp_runtime = object()
    with pytest.raises(ValueError, match="trained/live|frozen|CP/PP"):
        trainer.evaluate(_dataset(), original_reference_model=reference)


@pytest.mark.parametrize("failure", [False, True])
def test_explicit_original_reference_scores_unseen_rows_and_restores_ownership(tmp_path, monkeypatch, failure):
    trainer = _Trainer(tmp_path)
    reference = nn.Linear(1, 1).requires_grad_(False).eval()
    original_weights = reference.weight.clone()
    previous_reference = trainer.ref_model = object()
    device = reference.weight.device
    moves = []
    scoring_failed = False
    original_reject = DeferredRankFailure.reject

    def reject_before_forward_failure(guard):
        assert not scoring_failed, "caller-reference cleanup entered a collective after its forward failed"
        return original_reject(guard)

    monkeypatch.setattr(DeferredRankFailure, "reject", reject_before_forward_failure)
    reference_to = reference.to

    def record_to(target):
        moves.append(target)
        return reference_to(target)

    reference.to = record_to
    unseen = Dataset.from_dict({"prompt_input_ids": [[99]], "completion_input_ids": [[7, 8]], "group_id": [0]})

    def sweep(dataset, split):
        nonlocal scoring_failed
        assert trainer.ref_model is reference and not reference.training
        assert dataset["prompt_input_ids"] == [[99]]
        if failure:
            scoring_failed = True
            raise RuntimeError("scoring failed")
        return mapped_scores(tmp_path, dataset, [[-4.25, -5.75]])

    trainer._sweep_reference_logps = sweep
    if failure:
        with pytest.raises(RuntimeError, match="scoring failed"):
            trainer.evaluate(unseen, original_reference_model=reference)
    else:
        assert trainer.evaluate(unseen, original_reference_model=reference) == {"eval_loss": 42.0}
        assert trainer.evaluated[REF_PER_TOKEN_LOGPS_COLUMN] == [[-4.25, -5.75]]
    assert trainer.ref_model is previous_reference
    assert reference.weight.device == device
    assert moves == [trainer.model.weight.device, device]
    torch.testing.assert_close(reference.weight, original_weights, rtol=0, atol=0)
    assert not reference.training and not reference.weight.requires_grad


def _ranked_object_identity(rank, root):
    trainer = _Trainer(root)
    dataset = trainer.train_dataset if rank == 0 else _dataset()
    trainer.evaluate(dataset)
    assert trainer.evaluated[REF_PER_TOKEN_LOGPS_COLUMN] == [[-0.25, -1.5], [-0.75], []]


def test_original_reference_restore_failure_does_not_replace_the_forward_failure(tmp_path):
    trainer = _Trainer(tmp_path)
    reference = nn.Linear(1, 1).requires_grad_(False).eval()
    unseen = Dataset.from_dict({"prompt_input_ids": [[99]], "completion_input_ids": [[7]], "group_id": [0]})
    original_to = reference.to
    calls = 0

    def fail_restore(device):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("original device restoration failed")
        return original_to(device)

    def fail_forward(dataset, split):
        raise RuntimeError("original reference forward OOM")

    reference.to = fail_restore
    trainer._sweep_reference_logps = fail_forward
    with pytest.raises(RuntimeError, match="original reference forward OOM") as error:
        trainer.evaluate(unseen, original_reference_model=reference)
    assert trainer.ref_model is None
    assert error.value.__notes__ == [
        "Original evaluation reference device restoration also failed: original device restoration failed"
    ]


def test_rank_local_object_identity_does_not_skip_collectives(tmp_path):
    run_gloo_ranks(_ranked_object_identity, 2, str(tmp_path))


@pytest.mark.parametrize("cp", [False, True])
@pytest.mark.parametrize("explicit", [None, "eager", "flash_attention_2"])
def test_script_attention_preserves_explicit_choice_and_lets_cp_select_flash(cp, explicit):
    result = _requested_attention(
        SimpleNamespace(attn_implementation=explicit), SimpleNamespace(is_cp_mode=cp), sinks_reset=True
    )
    assert result == (explicit if explicit else None if cp else "sdpa")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
