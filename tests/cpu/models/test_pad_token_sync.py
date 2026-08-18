"""The pad-token id a pooled head reads back: tokenizer → model config → pooled position.

``GenericForSequenceClassification.forward`` — which every reward / classification head in the roster
inherits — pools each row at the last token whose id differs from
``config.get_text_config().pad_token_id``, and refuses any batch larger than one while that id is
``None``. Two properties of the setup seam follow, and the tests here pin both — plus the cost that
comes with them, that the recorded id is serialized and binds ``nn.Embedding.padding_idx`` on the
next load:

  * the id the collator pads with must be RECORDED on the model whatever settled it, not only when
    something changed the tokenizer's pad token. A checkpoint that ships no ``pad_token_id`` paired
    with a tokenizer that already has a pad token (Qwen3 / Qwen3.5 reward and classification bases,
    with no ``pad_token:`` in the YAML) changes nothing at all, so a change-driven sync writes
    nothing and the head raises at the first training step. Including a pad token that IS eos
    (DeepSeek-V4): withholding that one would not avert the ``padding_idx`` binding it costs, because
    ``Trainer.train``'s ``align_special_tokens`` records it with no pad-vs-eos test of its own — it
    would only delay the agreement past trainer construction.
  * these heads carry NO ``generation_config``: transformers attaches one only where
    ``can_generate()`` is true, so on a pooled head a bare attribute read raises ``AttributeError``
    out of ``nn.Module.__getattr__``.

    python tests/cpu/models/test_pad_token_sync.py
"""

import logging

import pytest
import torch
from accelerate import PartialState
from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3ForSequenceClassification
from transformers.trainer_utils import align_special_tokens

from src.args.common_script_args import CommonScriptArguments
from src.models.loading import model_preparation
from src.models.loading.tokenizer_setup import setup_model_and_tokenizer, sync_special_token_id
from src.trainers.reward.pooling import sequence_classification_pad_id

PartialState()  # the loader's logger needs accelerate state initialized
from tests.common.models import TINY_QWEN3_CONFIG

MAX_LENGTH = 128
# Arbitrary but distinct in-vocab ids for the stub tokenizer's special tokens.
SPECIAL_TOKEN_IDS = {"<pad>": 3, "<eos>": 2, "<other_pad>": 5}


class _Tokenizer:
    """A tokenizer stand-in for the setup seam: assigning a special token moves its id with it, and a
    token has an id only once it is IN the vocabulary — which is what makes the seam's ordering
    observable, since every id below is read back rather than assigned."""

    def __init__(self, pad_token: str | None = "<pad>"):
        self.model_max_length = int(1e30)
        self.extra_special_tokens: list[str] = []
        self.vocab = dict(SPECIAL_TOKEN_IDS)
        self.eos_token = "<eos>"
        self.bos_token = None
        self.chat_template = "{}"
        self.pad_token = pad_token

    def __len__(self) -> int:
        return max(self.vocab.values()) + 1

    def add_special_tokens(self, tokens: dict[str, list[str]], replace_extra_special_tokens: bool = True) -> None:
        """Mirrors transformers, whose default REPLACES the extra-special list."""
        requested = tokens["additional_special_tokens"]
        for token in requested:
            self.vocab.setdefault(token, len(self))
        if replace_extra_special_tokens:
            self.extra_special_tokens = list(requested)
        else:
            self.extra_special_tokens += [t for t in requested if t not in self.extra_special_tokens]

    @property
    def pad_token_id(self) -> int | None:
        return self.vocab.get(self.pad_token)

    @property
    def eos_token_id(self) -> int | None:
        return self.vocab.get(self.eos_token)

    @property
    def bos_token_id(self) -> int | None:
        return self.vocab.get(self.bos_token)


def _seq_cls_model(num_labels: int = 2) -> Qwen3ForSequenceClassification:
    """A real pooled head, built from a config so nothing is downloaded.

    ``TINY_QWEN3_CONFIG`` declares no ``pad_token_id`` — the checkpoint shape this file is about.
    """
    torch.manual_seed(0)
    return Qwen3ForSequenceClassification(Qwen3Config(**TINY_QWEN3_CONFIG, num_labels=num_labels))


# sync_special_token_id against a head that cannot generate


def test_a_pooled_head_carries_no_generation_config():
    """The premise: if transformers ever attached one, the guard below would stop meaning anything."""
    model = _seq_cls_model()
    assert model.can_generate() is False
    assert not hasattr(model, "generation_config")


def test_sync_records_the_id_on_a_head_without_a_generation_config():
    model = _seq_cls_model()
    sync_special_token_id(model, "pad_token_id", SPECIAL_TOKEN_IDS["<pad>"])
    assert model.config.get_text_config().pad_token_id == SPECIAL_TOKEN_IDS["<pad>"]


def test_sync_still_writes_the_generation_config_when_the_head_has_one():
    """The guard may not cost the causal path its write — generation pads with this id."""
    model = Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG))
    sync_special_token_id(model, "pad_token_id", SPECIAL_TOKEN_IDS["<pad>"])
    assert model.generation_config.pad_token_id == SPECIAL_TOKEN_IDS["<pad>"]


# setup_model_and_tokenizer — the production entry point


def test_setup_records_the_pad_id_the_tokenizer_already_had():
    """Nothing changes here, and the id still has to land: this is the shipped reward / classification
    configs' exact case."""
    model = _seq_cls_model()
    tokenizer = setup_model_and_tokenizer(CommonScriptArguments(), model, _Tokenizer(), MAX_LENGTH)
    assert model.config.get_text_config().pad_token_id == tokenizer.pad_token_id


def test_setup_records_a_pad_token_the_args_changed():
    model = _seq_cls_model()
    tokenizer = setup_model_and_tokenizer(
        CommonScriptArguments(pad_token="<other_pad>"), model, _Tokenizer(), MAX_LENGTH
    )
    assert tokenizer.pad_token == "<other_pad>"
    assert model.config.get_text_config().pad_token_id == SPECIAL_TOKEN_IDS["<other_pad>"]


def test_a_pad_token_the_args_ADD_is_recorded_too():
    """The way out the pooled-head raise names has to actually work.

    ``--added_special_tokens`` grows the vocabulary and ``--pad_token`` names one of the tokens in it,
    so the addition has to happen first: read the other way round, the id does not exist yet, the
    tokenizer ends up padding with one the model config never recorded, and the head raises again —
    naming the recipe that just failed.
    """
    model = _seq_cls_model()
    args = CommonScriptArguments(pad_token="<new_pad>", added_special_tokens=["<new_pad>"])
    tokenizer = setup_model_and_tokenizer(args, model, _Tokenizer(), MAX_LENGTH)

    assert tokenizer.pad_token_id is not None, "the added token never reached the vocabulary"
    assert tokenizer.pad_token_id != tokenizer.eos_token_id
    assert model.config.get_text_config().pad_token_id == tokenizer.pad_token_id


def test_an_eos_token_the_args_ADD_is_not_recorded_as_None():
    """The same ordering, with a sharper failure than a missing write.

    ``eos_token_id`` is recorded unconditionally on change, so an id read before the token exists
    writes ``None`` OVER the model's real eos — leaving generation with no terminator at all.
    """
    model = Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG))
    args = CommonScriptArguments(eos_token="<new_eos>", added_special_tokens=["<new_eos>"])
    tokenizer = setup_model_and_tokenizer(args, model, _Tokenizer(), MAX_LENGTH)

    assert tokenizer.eos_token_id is not None
    assert model.config.get_text_config().eos_token_id == tokenizer.eos_token_id
    assert model.generation_config.eos_token_id == tokenizer.eos_token_id


def test_the_second_setup_call_records_the_ids_on_its_own_model():
    """A preference or distillation script runs this seam once per model against ONE tokenizer.

    Guarding the eos/bos writes on the tokenizer having CHANGED breaks that: the first call (the
    reference) moves the token and records it, and the second (the policy) finds the token already
    equal to the request and skips the model write entirely. The policy then trains and exports with
    the checkpoint's eos while the reference it is scored against carries the run's — a terminator
    mismatch between the two legs of every logratio, and in the exported config.
    """
    tokenizer = _Tokenizer()
    args = CommonScriptArguments(eos_token="<other_pad>", bos_token="<pad>")
    reference = Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG))
    policy = Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG))

    setup_model_and_tokenizer(args, reference, tokenizer, MAX_LENGTH)
    assert reference.config.get_text_config().eos_token_id == SPECIAL_TOKEN_IDS["<other_pad>"]

    setup_model_and_tokenizer(args, policy, tokenizer, MAX_LENGTH)
    assert policy.config.get_text_config().eos_token_id == SPECIAL_TOKEN_IDS["<other_pad>"]
    assert policy.config.get_text_config().bos_token_id == SPECIAL_TOKEN_IDS["<pad>"]
    assert policy.generation_config.eos_token_id == SPECIAL_TOKEN_IDS["<other_pad>"]


def test_setup_records_the_eos_fallback_too():
    """A tokenizer that pads with eos still has its id recorded, because declining averts nothing.

    ``Trainer.train`` calls transformers' ``align_special_tokens``, which writes the tokenizer's pad
    id onto the config with no pad-vs-eos test, so the id — and the ``padding_idx`` binding it costs
    on the next load — reaches the checkpoint either way. Skipping it here would only leave the
    pooled heads refusing a batch until the first ``train()`` call.
    """
    model = _seq_cls_model()
    tokenizer = setup_model_and_tokenizer(CommonScriptArguments(), model, _Tokenizer(pad_token=None), MAX_LENGTH)
    assert tokenizer.pad_token == "<eos>"
    assert model.config.get_text_config().pad_token_id == SPECIAL_TOKEN_IDS["<eos>"]


def test_transformers_own_alignment_records_an_eos_pad_regardless():
    """The premise of the test above, pinned upstream: were this to gain a pad-vs-eos gate, declining
    to record the id here would become a real choice again rather than a delay."""
    model = _seq_cls_model()
    tokenizer = _Tokenizer(pad_token=None)
    tokenizer.pad_token = tokenizer.eos_token

    align_special_tokens(model, tokenizer)

    assert model.config.pad_token_id == SPECIAL_TOKEN_IDS["<eos>"]


def test_an_eos_pad_states_the_cost_it_carries(caplog):
    """Recording is right, but silence is not: from the next load on, the eos row's input-embedding
    gradient is masked, so the line has to name both the cost and the way out."""
    model = _seq_cls_model()
    with caplog.at_level(logging.INFO, logger=model_preparation.logger.name):
        setup_model_and_tokenizer(CommonScriptArguments(), model, _Tokenizer(pad_token=None), MAX_LENGTH)

    lines = [record.getMessage() for record in caplog.records if "pad token IS eos" in record.getMessage()]
    assert len(lines) == 1, f"expected exactly one eos-pad line, got {lines}"
    assert str(SPECIAL_TOKEN_IDS["<eos>"]) in lines[0], lines[0]
    assert "padding_idx" in lines[0] and "--pad_token" in lines[0], lines[0]


def test_a_real_pad_token_says_nothing_about_eos(caplog):
    """Anti-noise, and the control for the test above: the ordinary case must not emit that line."""
    model = _seq_cls_model()
    with caplog.at_level(logging.INFO, logger=model_preparation.logger.name):
        setup_model_and_tokenizer(CommonScriptArguments(), model, _Tokenizer(), MAX_LENGTH)

    assert not [record for record in caplog.records if "pad token IS eos" in record.getMessage()]


def test_a_borrowed_eos_still_satisfies_the_pp_pooling_seam():
    """Pipeline parallelism pools in the loss rather than the model, so it reads the id back from the
    config; the two must agree, and an eos-padding tokenizer is no exception — with nothing recorded
    the seam would refuse a run that pools perfectly well on the id it pads with."""
    model = _seq_cls_model()
    tokenizer = setup_model_and_tokenizer(CommonScriptArguments(), model, _Tokenizer(pad_token=None), MAX_LENGTH)
    assert sequence_classification_pad_id(model.config, tokenizer.pad_token_id) == SPECIAL_TOKEN_IDS["<eos>"]


def test_a_config_with_no_pad_id_at_all_still_raises_at_the_pp_seam():
    """The raise the seam cannot pre-empt: nothing settled a pad token, so pooling would score every
    row at the last filler token at a finite loss. The message has to name the way out."""
    model = _seq_cls_model()
    with pytest.raises(ValueError, match="--added_special_tokens"):
        sequence_classification_pad_id(model.config, None)


def test_setup_is_a_no_op_where_the_ids_already_agree():
    model = _seq_cls_model()
    model.config.pad_token_id = SPECIAL_TOKEN_IDS["<pad>"]
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}

    setup_model_and_tokenizer(CommonScriptArguments(), model, _Tokenizer(), MAX_LENGTH)

    assert model.config.pad_token_id == SPECIAL_TOKEN_IDS["<pad>"]
    assert all(torch.equal(before[name], tensor) for name, tensor in model.state_dict().items())


def test_recording_the_id_cannot_retro_affect_the_embedding():
    """``nn.Embedding``'s ``padding_idx`` is bound in ``__init__`` from the config's value at LOAD
    time, so a post-load write moves the pooling rule and nothing else *for this run*. Were it
    re-read per forward, an id recorded here would start zeroing a real token's embedding gradient."""
    model = _seq_cls_model()
    assert model.model.embed_tokens.padding_idx is None

    setup_model_and_tokenizer(CommonScriptArguments(), model, _Tokenizer(), MAX_LENGTH)

    assert model.config.pad_token_id == SPECIAL_TOKEN_IDS["<pad>"]
    assert model.model.embed_tokens.padding_idx is None


def test_the_recorded_id_binds_padding_idx_on_the_next_load(tmp_path):
    """The durable half, pinned because it is this seam's real cost rather than a side note.

    The id is serialized, so a checkpoint exported from a base that shipped none while its tokenizer
    carries a pad token of its own — the whole Qwen3 family — comes back with ``padding_idx`` bound,
    and from then on that row's INPUT-embedding gradient is masked. The weights themselves are
    untouched and the output side still trains, which is exactly what ``padding_idx`` means; it is
    pinned here so the cost is a chosen one, and it is why eos borrowed as pad is left unrecorded.
    """
    model = Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG))
    setup_model_and_tokenizer(CommonScriptArguments(), model, _Tokenizer(), MAX_LENGTH)
    model.save_pretrained(tmp_path)
    reloaded = Qwen3ForCausalLM.from_pretrained(tmp_path)

    pad_id = SPECIAL_TOKEN_IDS["<pad>"]
    assert reloaded.model.embed_tokens.padding_idx == pad_id
    assert torch.equal(model.model.embed_tokens.weight[pad_id], reloaded.model.embed_tokens.weight[pad_id])

    reloaded.train()
    ids = torch.tensor([[pad_id, 5, 7]])
    reloaded(input_ids=ids, labels=ids).loss.backward()
    grad = reloaded.model.embed_tokens.weight.grad
    assert grad[pad_id].abs().sum() == 0, "the pad row's input-embedding gradient is masked"
    assert grad[5].abs().sum() > 0, "a content row's gradient must be unaffected"


# What the id is FOR: the pooled position of a padded batch


def test_padded_batch_pools_at_the_last_content_token():
    """End to end, at the batch size the head refuses without the id.

    The padded row's pooled logits must equal the same row forwarded alone, and must NOT equal the
    logits at the final (padded) column — the control, without which a pin that pooled on padding
    could still pass.
    """
    model = _seq_cls_model().eval()
    tokenizer = setup_model_and_tokenizer(CommonScriptArguments(), model, _Tokenizer(), MAX_LENGTH)
    pad_id = tokenizer.pad_token_id

    generator = torch.Generator().manual_seed(7)
    content = torch.randint(10, 1000, (2, 6), generator=generator)
    lengths = [6, 3]
    input_ids = content.clone()
    attention_mask = torch.ones_like(input_ids)
    for row, length in enumerate(lengths):
        input_ids[row, length:] = pad_id
        attention_mask[row, length:] = 0

    with torch.no_grad():
        pooled = model(input_ids=input_ids, attention_mask=attention_mask).logits
        alone = torch.cat(
            [
                model(
                    input_ids=content[row : row + 1, :length],
                    attention_mask=torch.ones(1, length, dtype=torch.long),
                ).logits
                for row, length in enumerate(lengths)
            ]
        )
        hidden = model.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        at_pad = model.score(hidden)[:, -1]

    assert torch.allclose(pooled, alone, atol=1e-4)
    assert not torch.allclose(pooled[1], at_pad[1], atol=1e-3), "the padded row's control is vacuous"


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
