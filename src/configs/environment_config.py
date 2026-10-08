"""Environment selection and parameters for Environmental GRPO training; the async infrastructure
knobs are in AsyncTrainingConfig."""

from dataclasses import dataclass, field
from typing import Any

from src.args.validation import RangeValidatedConfig
from src.rewards.terms import ENVIRONMENT_REWARD_SOURCES, RewardTerm, parse_reward_terms


@dataclass
class EnvironmentConfig(RangeValidatedConfig):
    """Environment selection, reward terms and turn budget, parsed from YAML and resolved through
    src/environments/registry.py, which lists the available ``environment_type`` values."""

    environment_type: str = field(
        default="react_math",
        metadata={
            "help": "Environment registry name, resolved through src/environments/registry.py "
            "(get_registered_environments() lists the live set, which custom register_environment() "
            "calls extend; an unknown name raises with the available names)."
        },
    )

    rewards: list[dict[str, Any]] = field(
        default_factory=lambda: [{"source": "environment"}],
        metadata={
            "help": "Reward terms, each {source, name (required for judge/reward_model), weight?, exponent?, ...}, "
            "summed into the episode reward as weight x score^exponent over a score in [0, 1]: 'environment' (the "
            "environment's own grade, logged as reward/objective), 'judge' (a generative judge; model, requirements "
            "or checks, view, ...) and 'reward_model' (a served Bradley-Terry model; url, model, backend, ...). The "
            "environment's per-turn and shaping terms add on top. YAML-only; see "
            "agent-docs/training-methods/grpo/rewards.md."
        },
    )

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
        self._normalize_environment_type()
        self._validate_ranges()

    def _normalize_environment_type(self) -> None:
        """Spell ``environment_type`` as the registry keys it (lowercased), so every consumer comparing
        the name — the run name, the coding eval's env check — reads the type the run resolves."""
        if not isinstance(self.environment_type, str):
            raise ValueError(f"environment_type must be a registry name, got {self.environment_type!r}")
        self.environment_type = self.environment_type.lower()

    def _validate_ranges(self) -> None:
        super()._validate_ranges()
        # ``BaseEnvironment`` re-checks both, but that construction happens inside a Ray actor, after
        # the cluster and the vLLM servers are up. Check at parse time instead.
        turns = self.max_turns
        if turns is not None and (isinstance(turns, bool) or not isinstance(turns, int) or turns < 1):
            raise ValueError(
                f"max_turns must be an int >= 1 (null keeps the environment class default), got {turns!r}"
            )
        self.reward_terms  # noqa: B018  parse at config time so a bad or empty list fails here
        # Merged last in to_env_config, a key it writes from a top-level field would override that field
        # past the validation above.
        shadowing = sorted(set(self.environment_kwargs) & set(self._core_env_config()))
        if shadowing:
            raise ValueError(
                f"environment_kwargs must not carry {shadowing}: the config writes them from its own validated "
                "top-level fields (rewards for reward_terms, max_turns). Set those instead."
            )

    @property
    def reward_terms(self) -> tuple[RewardTerm, ...]:
        """The typed reward terms of ``rewards``, validated."""
        return parse_reward_terms(self.rewards, ENVIRONMENT_REWARD_SOURCES)

    def _core_env_config(self) -> dict[str, Any]:
        """The environment-config keys the top-level fields write, ``None`` where a field is unset."""
        return {"reward_terms": [dict(term) for term in self.rewards], "max_turns": self.max_turns}

    def to_env_config(self) -> dict[str, Any]:
        """Merge the reward terms and turn budget with ``environment_kwargs`` into the dict passed to
        ``resolve_environment(environment_type, config)``."""
        # Only an explicitly configured turn cap is forwarded; injecting the dataclass default would
        # override the per-class defaults (listed in the ``max_turns`` help).
        config = {key: value for key, value in self._core_env_config().items() if value is not None}
        config.update(self.environment_kwargs)
        return config
