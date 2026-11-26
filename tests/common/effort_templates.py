"""Each effort level's line as the effort templates state it (the level and what it asks for, never a token count),
and the repo's Qwen3.6 effort template."""

from tests.common.repo import REPO_ROOT

QWEN3_6_EFFORT_TEMPLATE = REPO_ROOT / "jinja-templates" / "qwen3" / "qwen3.6-reasoning-effort.jinja"

LEVEL_LINES = {
    "low": "Reasoning effort: low. Keep your reasoning short: take the most direct approach and act on it.",
    "medium": "Reasoning effort: medium. Reason as far as the problem needs, without exhaustive exploration.",
    "high": "Reasoning effort: high. Reason thoroughly: weigh alternatives and check your work before you act.",
}
