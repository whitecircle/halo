"""A stand-in for HF ``Trainer._save_checkpoint`` that writes the way the real one does, since the mixin reaches into
two of its calls."""

import os

from transformers.trainer import TRAINER_STATE_NAME
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, rotate_checkpoints


class BaseTrainerSave:
    """HF's ``_save_checkpoint`` as far as files go, in order: weights, optimizer and scheduler, RNG, trainer state,
    then push and rotation. The trainer over it sets ``run_dir``, ``args`` and ``state`` (a real ``TrainerState``) and
    overrides ``save_model``."""

    def _get_output_dir(self, trial=None):
        return self.run_dir

    def _save_checkpoint(self, model, trial):
        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}")
        if self.args.should_save:
            os.makedirs(output_dir, exist_ok=True)
        self.save_model(output_dir, _internal_call=True)
        if not self.args.save_only_model:
            self._save_optimizer_and_scheduler(output_dir)
            self._save_rng_state(output_dir)
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))
        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)
        if self.args.should_save:
            rotate_checkpoints(
                output_dir=run_dir,
                save_total_limit=self.args.save_total_limit,
                best_model_checkpoint=self.state.best_model_checkpoint,
                use_mtime=True,
            )

    def _save_optimizer_and_scheduler(self, output_dir):
        """Nothing by default: a trainer whose test reads the base's rank-0 optimizer view writes it."""

    def _save_rng_state(self, output_dir):
        """Nothing by default: a trainer whose test reads the per-rank RNG files binds HF's own writer."""
