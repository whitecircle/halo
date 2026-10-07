"""A stand-in for HF ``Trainer._save_checkpoint`` that writes the way the real one does.

The mixin's ``_save_checkpoint`` wraps the base save and reaches into it at two calls: the weights go
through ``save_model`` (the toolkit's own, whose gathers are world collectives), and the trainer state
through ``self.state.save_to_json``, which the mixin redirects to the withheld name until every
sidecar is on disk. A stand-in that writes ``trainer_state.json`` any other way, or whose ``state``
is not a ``TrainerState``, tests a base the mixin never runs over.
"""

import os

from transformers.trainer import TRAINER_STATE_NAME
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, rotate_checkpoints


class BaseTrainerSave:
    """HF's ``_save_checkpoint`` as far as files go, in its order: the weights through ``save_model``;
    unless ``save_only_model``, the optimizer and scheduler files (``_save_optimizer_and_scheduler``);
    the trainer state through ``self.state.save_to_json`` on the ``should_save`` rank(s); then the
    checkpoint push and the rotation, which the mixin neutralizes for this call.

    The trainer over it sets ``run_dir``, ``args`` and ``state`` (a ``transformers.TrainerState``) and
    overrides ``save_model``, which the mixin's own would otherwise shadow. The checkpoint directory
    is created on the ``should_save`` rank(s), where ``save_model``'s writer creates it.
    """

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
