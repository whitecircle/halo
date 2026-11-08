"""The line each effort level's template states: the level and what it asks for, never a token count. One
copy for the Qwen3.6 and Gemma 4 template tests, which pin the same wording."""

LEVEL_LINES = {
    "low": "Reasoning effort: low. Keep your reasoning short: take the most direct approach and act on it.",
    "medium": "Reasoning effort: medium. Reason as far as the problem needs, without exhaustive exploration.",
    "high": "Reasoning effort: high. Reason thoroughly: weigh alternatives and check your work before you act.",
}
