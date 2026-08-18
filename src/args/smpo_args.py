"""
Script arguments for SMPO training.
"""

from dataclasses import dataclass

from src.args.common_script_args import CommonScriptArguments
from src.args.mixins import GenerationEvalArguments


@dataclass
class SMPOScriptArguments(GenerationEvalArguments, CommonScriptArguments):
    """Script-level arguments for SMPO training (not training hyperparameters)."""

    def __post_init__(self):
        self._apply_default_project_name("smpo-tuning")
        self._validate_ranges()
