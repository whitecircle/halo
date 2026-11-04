"""The grown-vocabulary setup shared by tokenizer seam, adapter save and merge tests: tiny Qwen3 padded to 128 rows,
grown to 132 by added tokens."""

from transformers import PreTrainedModel, PreTrainedTokenizerBase

from halo.configs.script_arguments import CommonScriptArguments
from halo.model_loading.tokenizer_setup import configure_run_tokenizer
from tests.common.tiny_models import TINY_DENSE_FAMILY, build_tiny_family_checkpoint, word_level_tokenizer

WORDS = [f"w{i}" for i in range(100)]
GROWING_TOKENS = [f"<added_{i}>" for i in range(30)]
FITTING_TOKENS = GROWING_TOKENS[:10]


def build_word_level_checkpoint(directory: str, *, tied: bool) -> None:
    """The tiny Qwen3 with a :data:`WORDS` tokenizer saved to ``directory``, its head tied or not."""
    build_tiny_family_checkpoint(
        TINY_DENSE_FAMILY, directory, word_level_tokenizer(WORDS), overrides={"tie_word_embeddings": tied}
    )


def grow_vocabulary(
    model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase, added: list[str] | None
) -> PreTrainedTokenizerBase:
    """``model`` and ``tokenizer`` through the scripts' seam with ``added`` special tokens, embeddings unsharded;
    returns the run's tokenizer."""
    return configure_run_tokenizer(
        CommonScriptArguments(added_special_tokens=added), model, tokenizer, embeddings_sharded=lambda _model: False
    )
