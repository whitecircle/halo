"""Script arguments for offline GRPO training."""

from dataclasses import dataclass
from typing import ClassVar

from src.configs.script_arguments import CommonScriptArguments, GenerationEvalArguments


@dataclass
class OfflineGRPOScriptArguments(GenerationEvalArguments, CommonScriptArguments):
    PROJECT_NAME: ClassVar[str] = "offline-grpo-tuning"
