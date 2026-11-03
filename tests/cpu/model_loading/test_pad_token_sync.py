"""The pad-token id a pooled head reads back. ``GenericForSequenceClassification.forward`` pools at the last non-pad
token and refuses a batch over one while the id is ``None``."""

import logging

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, Qwen3ForSequenceClassification, Qwen4ExpForCausalLM
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.trainer_utils import align_special_tokens

from halo.configs.script_arguments import CommonScriptArguments
from halo.distributed.tensor_parallel.hf_plan import is_input_embedding_tp_sharded
from halo.model_loading.tokenizer_setup import configure_run_tokenizer, sync_special_token_id
from halo.trainers.parallelism.pipeline_rules import require_pooling_pad_id
from tests.common.models import TINY_QWEN3_CONFIG, TINY_QWEN4_EXP_CONFIG

MAX_LENGTH = 128
SPECIAL_TOKEN_IDS = {"<pad>": 3, "<eos>": 2, "<other_pad>": 5}


class _Tokenizer:
    """A token has an id only once it is in the vocabulary, which makes the seam's ordering observable."""

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
        """Mirrors transformers, whose default replaces the extra-special list."""
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
    """``TINY_QWEN3_CONFIG`` declares no ``pad_token_id``, the checkpoint shape this file is about."""
    torch.manual_seed(0)
    return Qwen3ForSequenceClassification(Qwen3Config(**TINY_QWEN3_CONFIG, num_labels=num_labels))


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
    """Generation pads with this id."""
    model = Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG))
    sync_special_token_id(model, "pad_token_id", SPECIAL_TOKEN_IDS["<pad>"])
    assert model.generation_config.pad_token_id == SPECIAL_TOKEN_IDS["<pad>"]


def test_setup_records_the_pad_id_the_tokenizer_already_had():
    """Nothing changes, and the id still has to land: the shipped reward and classification configs' case."""
    model = _seq_cls_model()
    tokenizer = configure_run_tokenizer(
        CommonScriptArguments(), model, _Tokenizer(), MAX_LENGTH, embeddings_sharded=is_input_embedding_tp_sharded
    )
    assert model.config.get_text_config().pad_token_id == tokenizer.pad_token_id


def test_setup_records_a_pad_token_the_args_changed():
    model = _seq_cls_model()
    tokenizer = configure_run_tokenizer(
        CommonScriptArguments(pad_token="<other_pad>"),
        model,
        _Tokenizer(),
        MAX_LENGTH,
        embeddings_sharded=is_input_embedding_tp_sharded,
    )
    assert tokenizer.pad_token == "<other_pad>"
    assert model.config.get_text_config().pad_token_id == SPECIAL_TOKEN_IDS["<other_pad>"]


def test_a_pad_token_the_args_ADD_is_recorded_too():
    """The addition has to happen first: read the other way, the model config never records the id and the head
    raises again."""
    model = _seq_cls_model()
    args = CommonScriptArguments(pad_token="<new_pad>", added_special_tokens=["<new_pad>"])
    tokenizer = configure_run_tokenizer(
        args, model, _Tokenizer(), MAX_LENGTH, embeddings_sharded=is_input_embedding_tp_sharded
    )

    assert tokenizer.pad_token_id is not None, "the added token never reached the vocabulary"
    assert tokenizer.pad_token_id != tokenizer.eos_token_id
    assert model.config.get_text_config().pad_token_id == tokenizer.pad_token_id


def test_an_eos_token_the_args_ADD_is_not_recorded_as_None():
    """Read before the token exists, the id writes ``None`` over the model's real eos, leaving generation with no
    terminator."""
    model = Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG))
    args = CommonScriptArguments(eos_token="<new_eos>", added_special_tokens=["<new_eos>"])
    tokenizer = configure_run_tokenizer(
        args, model, _Tokenizer(), MAX_LENGTH, embeddings_sharded=is_input_embedding_tp_sharded
    )

    assert tokenizer.eos_token_id is not None
    assert model.config.get_text_config().eos_token_id == tokenizer.eos_token_id
    assert model.generation_config.eos_token_id == tokenizer.eos_token_id


def test_the_second_setup_call_records_the_ids_on_its_own_model():
    """Guarding on the tokenizer having changed skips the second model's write, a terminator mismatch across every
    logratio and the export."""
    tokenizer = _Tokenizer()
    args = CommonScriptArguments(eos_token="<other_pad>", bos_token="<pad>")
    reference = Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG))
    policy = Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG))

    configure_run_tokenizer(args, reference, tokenizer, MAX_LENGTH, embeddings_sharded=is_input_embedding_tp_sharded)
    assert reference.config.get_text_config().eos_token_id == SPECIAL_TOKEN_IDS["<other_pad>"]

    configure_run_tokenizer(args, policy, tokenizer, MAX_LENGTH, embeddings_sharded=is_input_embedding_tp_sharded)
    assert policy.config.get_text_config().eos_token_id == SPECIAL_TOKEN_IDS["<other_pad>"]
    assert policy.config.get_text_config().bos_token_id == SPECIAL_TOKEN_IDS["<pad>"]
    assert policy.generation_config.eos_token_id == SPECIAL_TOKEN_IDS["<other_pad>"]


def test_an_eos_change_a_built_module_cannot_follow_is_refused():
    """Qwen3.8-Flash-Next's n-gram embedding copies ``eos_token_id`` at build, so a served copy would restart at
    another token."""
    flash_next = Qwen4ExpForCausalLM(Qwen4ExpTextConfig(**TINY_QWEN4_EXP_CONFIG))
    copied = TINY_QWEN4_EXP_CONFIG["eos_token_id"]
    with pytest.raises(ValueError, match="Qwen4ExpTextNGramEmbedding"):
        configure_run_tokenizer(
            CommonScriptArguments(eos_token="<eos>"),
            flash_next,
            _Tokenizer(),
            MAX_LENGTH,
            embeddings_sharded=is_input_embedding_tp_sharded,
        )
    assert flash_next.config.eos_token_id == copied

    tokenizer = _Tokenizer()
    tokenizer.vocab["<endoftext>"] = copied
    configure_run_tokenizer(
        CommonScriptArguments(eos_token="<endoftext>"),
        flash_next,
        tokenizer,
        MAX_LENGTH,
        embeddings_sharded=is_input_embedding_tp_sharded,
    )
    assert flash_next.config.eos_token_id == copied

    qwen3 = Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG))
    configure_run_tokenizer(
        CommonScriptArguments(eos_token="<eos>"),
        qwen3,
        _Tokenizer(),
        MAX_LENGTH,
        embeddings_sharded=is_input_embedding_tp_sharded,
    )
    assert qwen3.config.eos_token_id == SPECIAL_TOKEN_IDS["<eos>"]


def test_setup_records_the_eos_fallback_too():
    """``Trainer.train``'s ``align_special_tokens`` writes the pad id anyway, so skipping here would only delay
    agreement to ``train()``."""
    model = _seq_cls_model()
    tokenizer = configure_run_tokenizer(
        CommonScriptArguments(),
        model,
        _Tokenizer(pad_token=None),
        MAX_LENGTH,
        embeddings_sharded=is_input_embedding_tp_sharded,
    )
    assert tokenizer.pad_token == "<eos>"
    assert model.config.get_text_config().pad_token_id == SPECIAL_TOKEN_IDS["<eos>"]


def test_transformers_own_alignment_records_an_eos_pad_regardless():
    """The premise above, pinned upstream: a pad-vs-eos gate there would make declining a real choice."""
    model = _seq_cls_model()
    tokenizer = _Tokenizer(pad_token=None)
    tokenizer.pad_token = tokenizer.eos_token

    align_special_tokens(model, tokenizer)

    assert model.config.pad_token_id == SPECIAL_TOKEN_IDS["<eos>"]


def test_an_eos_pad_states_the_cost_it_carries(caplog):
    """From the next load on, the eos row's input-embedding gradient is masked, so the line names the cost and the
    way out."""
    model = _seq_cls_model()
    with caplog.at_level(logging.INFO, logger="halo.model_loading.tokenizer_setup"):
        configure_run_tokenizer(
            CommonScriptArguments(),
            model,
            _Tokenizer(pad_token=None),
            MAX_LENGTH,
            embeddings_sharded=is_input_embedding_tp_sharded,
        )

    lines = [record.getMessage() for record in caplog.records if "pad token IS eos" in record.getMessage()]
    assert len(lines) == 1, f"expected exactly one eos-pad line, got {lines}"
    assert str(SPECIAL_TOKEN_IDS["<eos>"]) in lines[0], lines[0]
    assert "padding_idx" in lines[0] and "--pad_token" in lines[0], lines[0]


def test_a_real_pad_token_says_nothing_about_eos(caplog):
    """Control for the test above."""
    model = _seq_cls_model()
    with caplog.at_level(logging.INFO, logger="halo.model_loading.tokenizer_setup"):
        configure_run_tokenizer(
            CommonScriptArguments(), model, _Tokenizer(), MAX_LENGTH, embeddings_sharded=is_input_embedding_tp_sharded
        )

    assert not [record for record in caplog.records if "pad token IS eos" in record.getMessage()]


def test_a_borrowed_eos_still_satisfies_the_pp_pooling_seam():
    """PP pools in the loss and reads the id from the config, so an eos-padding tokenizer must still satisfy the
    seam."""
    model = _seq_cls_model()
    tokenizer = configure_run_tokenizer(
        CommonScriptArguments(),
        model,
        _Tokenizer(pad_token=None),
        MAX_LENGTH,
        embeddings_sharded=is_input_embedding_tp_sharded,
    )
    assert require_pooling_pad_id(model.config, tokenizer.pad_token_id) == SPECIAL_TOKEN_IDS["<eos>"]


def test_a_config_with_no_pad_id_at_all_still_raises_at_the_pp_seam():
    """Pooling would score every row at the last filler token at a finite loss."""
    model = _seq_cls_model()
    with pytest.raises(ValueError, match="--added_special_tokens"):
        require_pooling_pad_id(model.config, None)


def test_setup_is_a_no_op_where_the_ids_already_agree():
    model = _seq_cls_model()
    model.config.pad_token_id = SPECIAL_TOKEN_IDS["<pad>"]
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}

    configure_run_tokenizer(
        CommonScriptArguments(), model, _Tokenizer(), MAX_LENGTH, embeddings_sharded=is_input_embedding_tp_sharded
    )

    assert model.config.pad_token_id == SPECIAL_TOKEN_IDS["<pad>"]
    assert all(torch.equal(before[name], tensor) for name, tensor in model.state_dict().items())


def test_recording_the_id_cannot_retro_affect_the_embedding():
    """``padding_idx`` binds in ``__init__`` at load, so a post-load write moves only the pooling rule for this run."""
    model = _seq_cls_model()
    assert model.model.embed_tokens.padding_idx is None

    configure_run_tokenizer(
        CommonScriptArguments(), model, _Tokenizer(), MAX_LENGTH, embeddings_sharded=is_input_embedding_tp_sharded
    )

    assert model.config.pad_token_id == SPECIAL_TOKEN_IDS["<pad>"]
    assert model.model.embed_tokens.padding_idx is None


def test_the_recorded_id_binds_padding_idx_on_the_next_load(tmp_path):
    """The durable cost: the id is serialized, and that row's input-embedding gradient is masked from then on. The
    output side still trains."""
    model = Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG))
    configure_run_tokenizer(
        CommonScriptArguments(), model, _Tokenizer(), MAX_LENGTH, embeddings_sharded=is_input_embedding_tp_sharded
    )
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


def test_padded_batch_pools_at_the_last_content_token():
    """Compared to the same row alone and to the padded column's logits, the control for a pin that pooled on
    padding."""
    model = _seq_cls_model().eval()
    tokenizer = configure_run_tokenizer(
        CommonScriptArguments(), model, _Tokenizer(), MAX_LENGTH, embeddings_sharded=is_input_embedding_tp_sharded
    )
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
    raise SystemExit(pytest.main([__file__, "-v"]))
