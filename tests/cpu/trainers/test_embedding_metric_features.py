#!/usr/bin/env python
"""The embedding metrics read the sentence embeddings the loss forward already computed.

``embed/*`` (logged before each log step and averaged over every eval batch) used to re-encode each text group
under ``no_grad``, a second forward of every eval batch. The trainer now records the pipeline's own forwards
inside the loss: sentence-transformers runs the anchor column alone, the candidate columns merged column-major,
a cached loss chunk by chunk and Matryoshka once at full width, so the rows lie group after group. Driven
through the real loss call of each pattern, against an independent per-group encode, counting the backbone's
forwards.

Run: pytest tests/cpu/trainers/test_embedding_metric_features.py
"""

import pytest
import torch
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.models import Pooling
from transformers import Qwen3Config, Qwen3Model

from halo.trainers.embedding.preloaded_transformer import PreloadedTransformer
from halo.trainers.embedding.trainer import _METRIC_MAX_SAMPLES, EmbeddingTrainer, _group_heads, _group_samples
from tests.common.models import TINY_QWEN3_CONFIG
from tests.common.tiny_models import word_level_tokenizer

_WORDS = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel")
_ROWS = 4
_COLUMNS = {
    "anchor": ["alpha bravo", "charlie", "delta echo foxtrot", "golf"],
    "positive": ["bravo alpha", "charlie delta", "foxtrot", "golf hotel alpha"],
    "negative": ["hotel", "echo echo", "alpha golf bravo charlie", "delta"],
}


def _pipeline() -> SentenceTransformer:
    torch.manual_seed(0)
    backbone = PreloadedTransformer(
        Qwen3Model(Qwen3Config(**TINY_QWEN3_CONFIG, attn_implementation="eager")), word_level_tokenizer(_WORDS)
    )
    model = SentenceTransformer(modules=[backbone, Pooling(backbone.get_embedding_dimension())], device="cpu")
    return model.eval()


def _inputs(model: SentenceTransformer, columns: tuple[str, ...]) -> dict:
    """A collated batch as sentence-transformers hands it to ``compute_loss``: ``{column}_{feature}`` keys."""
    inputs = {}
    for column in columns:
        for key, value in model[0].preprocess(_COLUMNS[column]).items():
            inputs[f"{column}_{key}"] = value
    return inputs


def _eval_trainer(model: SentenceTransformer, loss) -> EmbeddingTrainer:
    trainer = object.__new__(EmbeddingTrainer)
    trainer.model, trainer.model_wrapped, trainer.loss = model, None, loss
    trainer._in_eval_loop, trainer._eval_embedding_accum = True, {}
    return trainer


def _backbone_forwards(model: SentenceTransformer) -> list[int]:
    calls = []
    model[0].auto_model.register_forward_hook(lambda *_: calls.append(1))
    return calls


def _reference_metrics(model: SentenceTransformer, inputs: dict, columns: tuple[str, ...]) -> dict[str, float]:
    """The metrics over an independent encode of each text group alone."""
    with torch.no_grad():
        embeddings = [
            model({key[len(column) + 1 :]: value for key, value in inputs.items() if key.startswith(f"{column}_")})[
                "sentence_embedding"
            ]
            for column in columns
        ]
    return EmbeddingTrainer._compute_embedding_metrics(embeddings)


_LOSSES = {
    "mnrl-merged": (lambda model: losses.MultipleNegativesRankingLoss(model), ("anchor", "positive", "negative")),
    "cached-chunks": (
        lambda model: losses.CachedMultipleNegativesRankingLoss(model, mini_batch_size=3),
        ("anchor", "positive", "negative"),
    ),
    "matryoshka": (
        lambda model: losses.MatryoshkaLoss(
            model, losses.MultipleNegativesRankingLoss(model), matryoshka_dims=[64, 32]
        ),
        ("anchor", "positive"),
    ),
    "triplet": (lambda model: losses.TripletLoss(model), ("anchor", "positive", "negative")),
}


@pytest.mark.parametrize("loss_name", sorted(_LOSSES))
def test_eval_metrics_come_from_the_loss_forward_without_a_second_one(loss_name):
    build_loss, columns = _LOSSES[loss_name]
    model = _pipeline()
    loss = build_loss(model)
    inputs = _inputs(model, columns)
    forwards = _backbone_forwards(model)
    with torch.no_grad():
        loss(*_eval_trainer(model, loss).collect_features(dict(inputs)))
    loss_forwards = len(forwards)

    trainer = _eval_trainer(model, loss)
    with torch.no_grad():
        trainer.compute_loss(model, dict(inputs), return_outputs=True)

    assert len(forwards) == 2 * loss_forwards, "the metrics must add no forward of their own"
    expected = _reference_metrics(model, inputs, columns)
    assert trainer._eval_embedding_accum.keys() == expected.keys()
    for key, value in expected.items():
        assert trainer._eval_embedding_accum[key] == [pytest.approx(value, rel=1e-4, abs=1e-5)], key


def test_group_heads_take_each_groups_leading_rows_across_chunks():
    """Groups of five, four and six rows over chunks that straddle the group boundaries, column-major."""
    rows = torch.arange(15.0).unsqueeze(1)
    heads = _group_heads(list(rows.split([4, 7, 2, 2])), [5, 4, 6])
    assert [head.squeeze(1).tolist() for head in heads] == [[0, 1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12, 13, 14]]


def test_group_heads_cap_every_group_and_refuse_rows_the_groups_do_not_account_for():
    rows = torch.arange(2.0 * (_METRIC_MAX_SAMPLES + 3)).unsqueeze(1)
    heads = _group_heads(list(rows.split(97)), [_METRIC_MAX_SAMPLES + 3] * 2)
    assert [head.size(0) for head in heads] == [_METRIC_MAX_SAMPLES, _METRIC_MAX_SAMPLES]
    assert heads[1][0].item() == _METRIC_MAX_SAMPLES + 3, "the second group starts past the first's uncapped rows"
    # A loss that forwards its first column alone (the batch triplet losses) over a two-column batch.
    assert _group_heads([torch.zeros(4, 1)], [4, 4]) is None
    assert _group_heads([], [4]) is None


def test_group_samples_count_padded_rows_and_packed_sequences():
    inputs = {
        "anchor_input_ids": torch.ones(3, 7, dtype=torch.long),
        "anchor_attention_mask": torch.ones(3, 7, dtype=torch.long),
        "positive_input_ids": torch.ones(1, 20, dtype=torch.long),
        "positive_cu_seq_lens_q": torch.tensor([0, 5, 12, 20], dtype=torch.int32),
        "label": torch.zeros(3),
    }
    assert _group_samples(inputs) == [3, 3]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
