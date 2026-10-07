"""Script arguments for embedding training."""

from dataclasses import dataclass
from typing import ClassVar

from src.args.common_script_args import CommonScriptArguments


@dataclass
class EmbeddingScriptArguments(CommonScriptArguments):
    PROJECT_NAME: ClassVar[str] = "embedding"
