"""A construction-free ``SmoothMarginPOTrainer`` for the CPU tests that drive its forward and loss paths."""

from types import SimpleNamespace

from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.preference.smpo import SmoothMarginPOTrainer
from src.trainers.preference.smpo_objective import SMPOObjective

# The SmoothMarginPOConfig fields those paths read, at values a test overrides per case.
_ARGS = {
    "beta": 2.0,
    "loss_type": "sigmoid",
    "chosen_sft_ratio": 0.7,
    "lower_clip_percentile": None,
    "upper_clip_percentile": None,
    "min_log_prob": None,
    "target_margin": 0.3,
    "use_margin_schedule": False,
    "padding_free": False,
    "max_length": 24,
    "truncation_mode": "keep_start",
    "dataset_num_proc": None,
}


def smpo_trainer(*, parallelism_config=None, cp_config=None, pad_token_id: int = 0, **args) -> SmoothMarginPOTrainer:
    """The trainer as ``__init__`` leaves what those paths read, its config fields overridden by ``args``."""
    trainer = object.__new__(SmoothMarginPOTrainer)
    trainer.args = SimpleNamespace(**{**_ARGS, **args})
    trainer.objective = SMPOObjective.from_args(trainer.args)
    trainer.parallelism_config = ParallelismConfig() if parallelism_config is None else parallelism_config
    trainer.cp_config = cp_config
    trainer.pad_token_id = pad_token_id
    return trainer
