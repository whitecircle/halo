"""Script arguments for embedding training."""

from dataclasses import dataclass

from src.args.common_script_args import CommonScriptArguments


@dataclass
class EmbeddingScriptArguments(CommonScriptArguments):
    def __post_init__(self):
        self._apply_default_project_name("embedding")
        self._validate_ranges()
