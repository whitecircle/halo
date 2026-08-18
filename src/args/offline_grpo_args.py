"""Script arguments for offline GRPO training."""

from dataclasses import dataclass

from src.args.common_script_args import CommonScriptArguments
from src.args.mixins import GenerationEvalArguments


@dataclass
class OfflineGRPOScriptArguments(GenerationEvalArguments, CommonScriptArguments):
    def __post_init__(self):
        self._apply_default_project_name("offline-grpo-tuning")
        self._validate_ranges()
