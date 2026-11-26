"""GLM chat templates that end an assistant turn only at the next role marker, which leaves a row's last assistant
turn open, loaded on GLM-4.7-Flash's cached tokenizer."""

from tests.common.models import GLM4_FLASH
from tests.common.repo import REPO_ROOT
from tests.common.tokenizers import load_cached_config, load_cached_tokenizer

# The tokenizer's own template, the bundled copy and GLM-5.3-Flash's, which shares the role tokens.
OPEN_FINAL_TURN_TEMPLATES = {
    "hub": None,
    "glm-native": "jinja-templates/glm/glm-native.jinja",
    "glm-5.3-flash-hub": "tests/common/hub_templates/glm-5.3-flash.jinja",
}
GLM_ASSISTANT_MARKER = "<|assistant|>"
# The first assistant turn ends at the second user turn's role marker; nothing ends the second.
TWO_EXCHANGES = [
    {"role": "user", "content": "Question alpha numero uno"},
    {"role": "assistant", "content": "Response bravo the first answer"},
    {"role": "user", "content": "Question charlie numero dos"},
    {"role": "assistant", "content": "Response delta the second answer"},
]


def load_open_turn_tokenizer(name: str):
    """``(tokenizer, model config)`` under the template ``OPEN_FINAL_TURN_TEMPLATES`` names ``name``; skips where
    GLM-4.7-Flash is not cached."""
    tokenizer = load_cached_tokenizer(GLM4_FLASH, trust_remote_code=True)
    path = OPEN_FINAL_TURN_TEMPLATES[name]
    if path:
        tokenizer.chat_template = (REPO_ROOT / path).read_text()
    return tokenizer, load_cached_config(GLM4_FLASH, trust_remote_code=True)
