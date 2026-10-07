"""Script arguments for offline GRPO training."""

from dataclasses import dataclass
from typing import ClassVar

from src.args.common_script_args import CommonScriptArguments
from src.args.mixins import GenerationEvalArguments


@dataclass
class OfflineGRPOScriptArguments(GenerationEvalArguments, CommonScriptArguments):
    PROJECT_NAME: ClassVar[str] = "offline-grpo-tuning"
