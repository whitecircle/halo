"""Environment selection and parameters for Environmental GRPO training; the async infrastructure
knobs are in AsyncTrainingConfig."""

from dataclasses import dataclass, field
from typing import Any

from src.args.validation import RangeValidatedConfig


@dataclass
class EnvironmentConfig(RangeValidatedConfig):
    """Environment selection and reward parameters, parsed from YAML and resolved through
    src/environments/registry.py, which lists the available ``environment_type`` values."""

    environment_type: str = field(
        default="react_math",
        metadata={
            "help": "Environment registry name, resolved through src/environments/registry.py "
            "(get_registered_environments() lists the live set, which custom register_environment() "
            "calls extend; an unknown name raises with the available names)."
        },
    )

    success_reward: float = field(default=1.0, metadata={"help": "Reward for correct final answer."})

    failure_reward: float = field(default=0.0, metadata={"help": "Reward for incorrect final answer."})

    # No partial-credit reward: `compute_answer_reward` grades all-or-nothing, since a similarity
    # threshold would reward near-miss wrong answers.

    max_turns: int | None = field(
        default=None,
        metadata={
            "help": "Maximum environment turns per episode. None (default) keeps the environment "
            "class's own default: code_contests 15, codeforces 15, swe 20, exam_qa 8; "
            "every other environment 10."
        },
    )

    environment_kwargs: dict[str, Any] = field(
        default_factory=dict,
        metadata={
            "help": "Env-specific kwargs passed through to the registry factory. "
            "Examples: {search_backend: duckduckgo}, {open_book: true}, "
            "{timeout_per_test: 10}, {mcp_server: filesystem}."
        },
    )

    def __post_init__(self):
        self._validate_ranges()

    def _validate_ranges(self) -> None:
        super()._validate_ranges()
        # ``BaseEnvironment`` re-checks this, but that construction happens inside a Ray actor, after
        # the cluster and the vLLM servers are up. Check at parse time instead.
        if self.max_turns is not None and self.max_turns < 1:
            raise ValueError(
                f"max_turns must be >= 1 (null keeps the environment class default), got {self.max_turns}"
            )

    def to_env_config(self) -> dict[str, Any]:
        """Merge the core reward/turn settings with ``environment_kwargs`` into the dict passed to
        ``resolve_environment(environment_type, config)``."""
        config: dict[str, Any] = {
            "success_reward": self.success_reward,
            "failure_reward": self.failure_reward,
        }
        # Only an explicitly configured turn cap is forwarded; injecting the dataclass default would
        # override the per-class defaults (listed in the ``max_turns`` help).
        if self.max_turns is not None:
            config["max_turns"] = self.max_turns
        config.update(self.environment_kwargs)
        return config
