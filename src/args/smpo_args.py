"""
Script arguments for SMPO training.
"""

from dataclasses import dataclass
from typing import ClassVar

from src.args.common_script_args import CommonScriptArguments
from src.args.mixins import GenerationEvalArguments


@dataclass
class SMPOScriptArguments(GenerationEvalArguments, CommonScriptArguments):
    """Script-level arguments for SMPO training (not training hyperparameters)."""

    PROJECT_NAME: ClassVar[str] = "smpo-tuning"
