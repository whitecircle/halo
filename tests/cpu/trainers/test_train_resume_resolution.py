#!/usr/bin/env python
"""``trainer.train(resume_from_checkpoint=True)`` resumes the checkpoint the entry scripts would.

The base ``Trainer.train`` resolves ``True`` with ``get_last_checkpoint`` — the highest-numbered
``checkpoint-<N>``, complete or not — then reads that directory's ``trainer_state.json``, which a save
stopped partway never published. The mixin resolves ``True`` through the scripts' detection instead:
the newest checkpoint whose trainer state is present, the incomplete ones moved out of the namespace.

``_HFTrainBase`` is HF's resume preamble line for line over the real ``get_last_checkpoint`` and
``TrainerState``, so the base's own failure is what the tests would see without the mixin.

    python tests/cpu/trainers/test_train_resume_resolution.py
"""

import json
import os
from types import SimpleNamespace

import pytest
from accelerate import PartialState
from transformers.trainer import TRAINER_STATE_NAME
from transformers.trainer_callback import TrainerState
from transformers.trainer_utils import get_last_checkpoint

from src.trainers.mixins.base import DistributedTrainerMixin

PartialState(cpu=True)


class _HFTrainBase:
    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None):
        if resume_from_checkpoint is False:
            resume_from_checkpoint = None
        if isinstance(resume_from_checkpoint, bool) and resume_from_checkpoint:
            resume_from_checkpoint = get_last_checkpoint(self.args.output_dir)
            if resume_from_checkpoint is None:
                raise ValueError(f"No valid checkpoint found in output directory ({self.args.output_dir})")
        if resume_from_checkpoint is not None:
            self.resumed_state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
        self.resumed_from = resume_from_checkpoint
        return "trained"


class _Trainer(DistributedTrainerMixin, _HFTrainBase):
    def __init__(self, output_dir):
        self.args = SimpleNamespace(output_dir=str(output_dir))


def _checkpoint(output_dir, step, *, complete=True):
    path = output_dir / f"checkpoint-{step}"
    path.mkdir(parents=True)
    (path / "model.safetensors").write_bytes(b"weights")
    if complete:
        (path / TRAINER_STATE_NAME).write_text(json.dumps({"global_step": step}))
    return path


def test_true_resumes_the_newest_complete_checkpoint_and_sets_the_stopped_one_aside(tmp_path):
    complete = _checkpoint(tmp_path, 5)
    _checkpoint(tmp_path, 10, complete=False)
    trainer = _Trainer(tmp_path)

    assert trainer.train(resume_from_checkpoint=True) == "trained"

    assert trainer.resumed_from == str(complete)
    assert trainer.resumed_state.global_step == 5
    assert get_last_checkpoint(str(tmp_path)) == str(complete), "the stopped save still outranks the complete one"


def test_true_with_only_an_incomplete_checkpoint_raises_naming_it(tmp_path):
    _checkpoint(tmp_path, 10, complete=False)
    with pytest.raises(RuntimeError, match="checkpoint-10.*incomplete"):
        _Trainer(tmp_path).train(resume_from_checkpoint=True)


def test_true_with_no_checkpoint_raises_the_base_error(tmp_path):
    with pytest.raises(ValueError, match="No valid checkpoint found"):
        _Trainer(tmp_path).train(resume_from_checkpoint=True)


@pytest.mark.parametrize("resume", [None, False], ids=["unset", "false"])
def test_no_resume_and_an_explicit_path_pass_through(tmp_path, resume):
    explicit = _checkpoint(tmp_path, 5)
    _checkpoint(tmp_path, 10, complete=False)
    trainer = _Trainer(tmp_path)

    trainer.train(resume_from_checkpoint=resume)
    assert trainer.resumed_from is None
    trainer.train(resume_from_checkpoint=str(explicit))
    assert trainer.resumed_from == str(explicit)
    assert (tmp_path / "checkpoint-10").is_dir(), "only a True resume resolves through the detection"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
