#!/usr/bin/env python
"""Qwen3.8-Flash-Next's document forwards (``patch_qwen4_exp_document_forwards``) through the whole tiny model, as a
run loads it: the family patches on its config, the n-gram table attached in host memory."""

from pathlib import Path

import pytest
import torch
import yaml
from peft import LoraConfig, get_peft_model

import halo.models.qwen4_exp.forwards as qwen4_exp_forwards
from halo.models.gradient_checkpointing import install_recompute_scopes
from halo.models.moe.expert_lora_spec import expert_target_projections
from tests.common.models import TINY_QWEN4_EXP_CONFIG
from tests.common.qwen4_exp import (
    LargestTensor,
    build_tiny_qwen4_exp_checkpoint,
    load_host_table_model,
    packed_markers,
    packed_positions,
    transformers_forwards,
)
from tests.common.repo import REPO_ROOT

TOLERANCE = 1e-6
# Below this the stock leak would not tell a leaking forward from an isolated one.
LEAK_FLOOR = 1e-3
# One document inside the indexer budget and the PLE conv's reach, one across a query tile, one past the budget.
DOCUMENTS = [5, 150, 61]
ALLOCATION_ROW = 2048
LORA_RECIPE = REPO_ROOT / "examples/sft/qwen4_exp/qwen3.8-flash-next-ep8-lora.yaml"


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory) -> str:
    directory = str(tmp_path_factory.mktemp("tiny") / "qwen4_exp")
    build_tiny_qwen4_exp_checkpoint(directory)
    return directory


def _load(checkpoint: str) -> torch.nn.Module:
    """The run's load with random norm weights, which the stock zero init (``1 + weight``) would leave inert."""
    model = load_host_table_model(checkpoint)
    torch.manual_seed(0)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "norm" in name:
                parameter.normal_(std=0.3)
    return model


@pytest.fixture
def model(checkpoint) -> torch.nn.Module:
    return _load(checkpoint).eval()


def _ids(length: int, batch: int = 1, seed: int = 1) -> torch.Tensor:
    """Random ids with a mid-row eos, where the n-gram history restarts on its own."""
    ids = torch.randint(
        2, TINY_QWEN4_EXP_CONFIG["vocab_size"], (batch, length), generator=torch.Generator().manual_seed(seed)
    )
    ids[:, length // 3] = TINY_QWEN4_EXP_CONFIG["eos_token_id"]
    return ids


def _packed(model) -> tuple[torch.Tensor, dict]:
    positions = packed_positions([DOCUMENTS])
    return _ids(sum(DOCUMENTS)), {"position_ids": positions, **packed_markers(model.config, positions)}


def _document_slices() -> list[slice]:
    bounds = torch.tensor([0, *DOCUMENTS]).cumsum(0).tolist()
    return [slice(lo, hi) for lo, hi in zip(bounds[:-1], bounds[1:], strict=True)]


def _logits(model, ids: torch.Tensor, **kwargs) -> torch.Tensor:
    with torch.no_grad():
        return model(input_ids=ids, use_cache=False, **kwargs).logits


def test_each_packed_document_computes_its_logits_alone(model):
    ids, markers = _packed(model)
    packed = _logits(model, ids, **markers)
    for document in _document_slices():
        alone = _logits(model, ids[:, document])
        assert (packed[:, document] - alone).abs().max() <= TOLERANCE, document
    with transformers_forwards():
        leaked = _logits(model, ids, **markers)
    assert (leaked - packed).abs().max() > LEAK_FLOOR, "transformers' forwards no longer leak across this row"


def _padded(length: int, pad: int, side: str) -> torch.Tensor:
    mask = torch.ones(2, length, dtype=torch.long)
    if side == "right":
        mask[0, length - pad :] = 0
    else:
        mask[0, :pad] = 0
    return mask


@pytest.mark.parametrize("rows", ["one-document", "right-padded", "left-padded", "left-padded-positions"])
def test_an_unpacked_row_computes_transformers_logits(model, rows):
    """Under a padding mask, position restarts are padding and mark no documents."""
    ids = _ids(160, batch=2)
    kwargs = {}
    if rows != "one-document":
        kwargs["attention_mask"] = _padded(160, 40, rows.split("-")[0])
    if rows == "left-padded-positions":
        kwargs["position_ids"] = (kwargs["attention_mask"].cumsum(-1) - 1).clamp_min(0)
    real = kwargs.get("attention_mask", torch.ones_like(ids)).bool()
    patched = _logits(model, ids, **kwargs)
    with transformers_forwards():
        stock = _logits(model, ids, **kwargs)
    assert (patched - stock)[real].abs().max() <= TOLERANCE


def test_a_packed_row_without_seq_idx_is_refused(model):
    ids, markers = _packed(model)
    with pytest.raises(ValueError, match="without seq_idx"):
        _logits(model, ids, position_ids=markers["position_ids"])


def test_no_tensor_grows_with_the_square_of_the_row(model):
    ids = _ids(ALLOCATION_ROW)
    with LargestTensor() as tiled:
        model.train()(input_ids=ids, labels=ids, use_cache=False).loss.backward()
    assert tiled.numel < ALLOCATION_ROW * ALLOCATION_ROW, f"the forward allocated {tiled.shape}"

    # The probe sees transformers' own [B, 1, T, T] masks.
    with transformers_forwards(), LargestTensor() as dense:
        _logits(model.eval(), ids)
    assert dense.numel >= ALLOCATION_ROW * ALLOCATION_ROW


def test_a_cached_call_keeps_transformers_forwards(model):
    ids = _ids(40)

    def generate_step():
        with torch.no_grad():
            cache = model(input_ids=ids[:, :-1], use_cache=True).past_key_values
            return model(input_ids=ids[:, -1:], past_key_values=cache, use_cache=True).logits

    patched = generate_step()
    with transformers_forwards():
        stock = generate_step()
    assert torch.equal(patched, stock)


def _packed_step_gradients(model) -> dict[str, torch.Tensor]:
    ids, markers = _packed(model)
    model.train()(input_ids=ids, labels=ids, use_cache=False, **markers).loss.backward()
    return {name: parameter.grad for name, parameter in model.named_parameters() if parameter.grad is not None}


def test_a_checkpointed_packed_step_matches_the_plain_step(checkpoint):
    """The reentrant recompute replays ``seq_idx`` and the first pass's QSA selection and n-gram rows."""
    plain = _packed_step_gradients(_load(checkpoint))
    checkpointed_model = _load(checkpoint)
    checkpointed_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": True})
    assert install_recompute_scopes(checkpointed_model) > 0
    checkpointed = _packed_step_gradients(checkpointed_model)
    assert checkpointed.keys() == plain.keys()
    for name, gradient in plain.items():
        scale = gradient.abs().max().clamp_min(1.0)
        assert ((checkpointed[name] - gradient).abs().max() / scale) <= TOLERANCE, name


def test_the_lora_recipe_targets_all_train_on_a_packed_row(model):
    """No target reaches the indexer, whose selection passes no gradient."""
    recipe = yaml.safe_load(Path(LORA_RECIPE).read_text())
    targets = [target for target in recipe["lora_target_modules"] if not expert_target_projections(target)]
    peft_model = get_peft_model(model, LoraConfig(r=4, lora_alpha=8, target_modules=targets))
    adapted = {name.rsplit(".", 1)[-1] for name, module in peft_model.named_modules() if hasattr(module, "lora_A")}
    assert adapted == set(targets)
    assert not [name for name, _ in peft_model.named_modules() if "indexer" in name and "lora" in name]

    gradients = _packed_step_gradients(peft_model)
    untrained = [name for name, _ in peft_model.named_parameters() if "lora_B" in name and not gradients[name].any()]
    assert not untrained, untrained


class _RenamedDecoderLayer(torch.nn.Module):
    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        conv_mask=None,
        cache_params=None,
        ple_input_ids=None,
        **kwargs,
    ):
        """The forward of a release that renamed its cache argument."""


class _UnmaskedLinearModel(torch.nn.Module):
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        ple_input_ids=None,
        **kwargs,
    ):
        """The forward of a release whose mask mapping has no linear-attention entry."""
        return {"indexed_attention": attention_mask}


class _RenamedConvPLELayer(qwen4_exp_forwards.qwen4_exp_modeling.Qwen4ExpTextPLELayer):
    def _short_conv(self, hidden_states, cache_params):
        """The PLE conv of a release that renamed its cache argument; the document conv re-derives it."""


class _StreamsGatedResidual(qwen4_exp_forwards.qwen4_exp_modeling.Qwen4ExpTextGatedResidual):
    def forward(self, hyper_input, streams):
        """The hyper-connection of a release that takes another input; the document layer calls it."""


def _renamed_recurrent_mask(config, input_embeds, attention_mask, past_key_values=None, **kwargs):
    """The padding-mask builder of a release that renamed its row argument; the document model calls it."""


@pytest.mark.parametrize(
    ("module", "attribute", "replacement", "message"),
    [
        (
            qwen4_exp_forwards.qwen4_exp_modeling,
            "Qwen4ExpTextDecoderLayer",
            _RenamedDecoderLayer,
            "re-derive it on this release's forward",
        ),
        (
            qwen4_exp_forwards.qwen4_exp_modeling,
            "Qwen4ExpTextModel",
            _UnmaskedLinearModel,
            r"does not read the \['linear_attention'\] entries",
        ),
        (
            qwen4_exp_forwards.qwen4_exp_modeling,
            "Qwen4ExpTextPLELayer",
            _RenamedConvPLELayer,
            "_RenamedConvPLELayer._short_conv takes",
        ),
        (
            qwen4_exp_forwards.qwen4_exp_modeling,
            "Qwen4ExpTextGatedResidual",
            _StreamsGatedResidual,
            "_StreamsGatedResidual.forward takes",
        ),
        (
            qwen4_exp_forwards,
            "create_recurrent_attention_mask",
            _renamed_recurrent_mask,
            "_renamed_recurrent_mask takes",
        ),
    ],
    ids=[
        "renamed-argument",
        "dropped-mask-kind",
        "renamed-ple-conv",
        "changed-hyper-connection",
        "renamed-mask-builder",
    ],
)
def test_an_upstream_change_fails_the_document_patch(monkeypatch, module, attribute, replacement, message):
    with transformers_forwards():
        monkeypatch.setattr(module, attribute, replacement)
        with pytest.raises(RuntimeError, match=message):
            qwen4_exp_forwards.patch_qwen4_exp_document_forwards()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
