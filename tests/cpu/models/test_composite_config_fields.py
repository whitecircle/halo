"""Reading and writing config fields on composite model configs, whose decoder lives under ``sub_configs``.
``PreTrainedConfig`` has no ``__getattr__``, so a decoder field does not exist at the top level."""

from pathlib import Path

import pytest
import torch
from transformers import CONFIG_MAPPING, Qwen3_5MoeForCausalLM, Qwen3_5MoeForConditionalGeneration
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTopKRouter

from halo.callbacks.balancing import apply_balancing_mode
from halo.distributed.expert_parallel.families.qwen3_5 import EPQwen3_5MoELayer
from halo.distributed.expert_parallel.moe_layer.layer import EPMoELayerBase
from halo.model_loading.tokenizer_setup import sync_special_token_id
from halo.models.attention.layout import resolve_head_dim, resolve_num_key_value_heads
from halo.models.introspection import config_access
from halo.models.introspection.config_access import (
    apply_config_overrides,
    config_sources,
    configs_declaring,
    set_config_field,
    ties_word_embeddings,
)
from halo.models.moe import balancing
from halo.models.moe.config_fields import has_routed_experts
from halo.models.moe.router_logits import honors_router_logits_flag

# Families whose decoder lives on a nested ``text_config``; the plain ones are the control group.
COMPOSITE_MODEL_TYPES = ("qwen3_5_moe", "qwen3_5", "gemma4")
PLAIN_MODEL_TYPES = ("qwen3_moe", "gpt_oss", "llama")


def _config(model_type: str):
    return CONFIG_MAPPING[model_type]()


class _FakeModel:
    def __init__(self, config):
        self.config = config
        self.generation_config = type("GenerationConfig", (), {})()


@pytest.mark.parametrize("model_type", COMPOSITE_MODEL_TYPES)
def test_composite_config_really_hides_decoder_fields(model_type):
    """If transformers ever flattens these, the tests below stop meaning anything."""
    config = _config(model_type)
    text_config = config.get_text_config()
    assert text_config is not config
    assert not hasattr(config, "num_hidden_layers")
    assert hasattr(text_config, "num_hidden_layers")


@pytest.mark.parametrize("model_type", PLAIN_MODEL_TYPES)
def test_plain_config_declares_its_own_fields_once(model_type):
    config = _config(model_type)
    assert config.get_text_config() is config
    assert configs_declaring(config, "num_hidden_layers") == [config]
    assert configs_declaring(config, "not_a_field_on_any_config") == []


def test_set_config_field_reports_an_undeclared_field():
    assert set_config_field(_config("qwen3_moe"), "not_a_field_on_any_config", 1, only_declared=True) is False


def test_set_config_field_plants_a_toolkit_flag_on_every_level():
    """A level not yet declaring the flag must still answer for it, or wrapper and decoder disagree on balancing."""
    config = _config("qwen3_5_moe")
    assert len(config_sources(config)) == 2

    assert set_config_field(config, "_toolkit_only_flag", True, only_declared=True) is False
    assert not any(hasattr(level, "_toolkit_only_flag") for level in config_sources(config))

    assert set_config_field(config, "_toolkit_only_flag", True, only_declared=False) is True
    assert all(level._toolkit_only_flag is True for level in config_sources(config))


def test_config_access_stays_the_leaf_both_writers_folded_into():
    """The leaf must not reach back into the MoE module (a cycle), and no second unconditional writer may reappear."""
    assert "models.moe" not in Path(config_access.__file__).read_text()
    assert not hasattr(balancing, "set_router_field")


def test_override_reaches_the_nested_decoder_config():
    """The shipped Qwen3.6 GRPO configs set these; a top-level-only write rejects them."""
    config = _config("qwen3_5_moe")
    apply_config_overrides(config, {"output_router_logits": True, "router_aux_loss_coef": 0.001})

    text_config = config.get_text_config()
    assert text_config.output_router_logits is True
    assert text_config.router_aux_loss_coef == 0.001


def test_override_reaches_a_nested_non_moe_field():
    config = _config("gemma4")
    apply_config_overrides(config, {"sliding_window": 512})
    assert config.get_text_config().sliding_window == 512


@pytest.mark.parametrize("model_type", PLAIN_MODEL_TYPES + COMPOSITE_MODEL_TYPES)
def test_override_still_rejects_a_key_no_level_declares(model_type):
    """A typo may not train the stock value."""
    with pytest.raises(ValueError, match="does not exist"):
        apply_config_overrides(_config(model_type), {"not_a_field_on_any_config": 1})


def test_override_on_a_plain_config_lands_on_the_config_itself():
    config = _config("qwen3_moe")
    apply_config_overrides(config, {"router_aux_loss_coef": 0.002})
    assert config.router_aux_loss_coef == 0.002


@pytest.mark.parametrize("model_type", COMPOSITE_MODEL_TYPES + PLAIN_MODEL_TYPES)
@pytest.mark.parametrize("field", ["pad_token_id", "eos_token_id", "bos_token_id"])
def test_special_token_id_lands_where_the_model_reads_it(model_type, field):
    """Seq-cls pooling reads ``config.get_text_config().pad_token_id``; generation reads the generation config."""
    config = _config(model_type)
    model = _FakeModel(config)

    sync_special_token_id(model, field, 4242)

    text_config = config.get_text_config()
    assert getattr(text_config, field) == 4242, f"{model_type}: {field} never reached the decoder config"
    assert getattr(model.generation_config, field) == 4242
    # A half-written id is the silent-pooling failure.
    for level in configs_declaring(config, field):
        assert getattr(level, field) == 4242


def test_special_token_id_does_not_invent_a_field():
    config = _config("qwen3_moe")
    model = _FakeModel(config)
    sync_special_token_id(model, "not_a_field_on_any_config", 7)
    assert not hasattr(config, "not_a_field_on_any_config")
    assert model.generation_config.not_a_field_on_any_config == 7


def test_sync_tolerates_no_model():
    sync_special_token_id(None, "pad_token_id", 1)  # scripts call it with model=None


@pytest.mark.parametrize(
    ("model_type", "is_moe"),
    [
        ("qwen3_5_moe", True),  # experts only on text_config
        ("qwen3_moe", True),
        ("gpt_oss", True),
        ("deepseek_v4", True),
        ("qwen3_5", False),  # composite dense
        ("llama", False),
    ],
)
def test_has_routed_experts_sees_both_levels(model_type, is_moe):
    assert has_routed_experts(_config(model_type)) is is_moe


def test_has_routed_experts_tolerates_none():
    assert has_routed_experts(None) is False


def _forward_declaring_the_flag(self, input_ids=None, output_router_logits=None, **kwargs):
    raise NotImplementedError


def _forward_reading_the_flag_from_kwargs(self, input_ids=None, **kwargs):
    raise NotImplementedError


def test_a_forward_reading_the_flag_from_kwargs_does_not_honor_the_config_flag():
    """HF resolves a config-backed flag on an explicit parameter only, so a forward reading ``kwargs`` never sees it.
    Balancing there records router logits, yet no aux loss enters the loss and the router trains unbalanced."""

    class _KwargsOnly(torch.nn.Module):
        forward = _forward_reading_the_flag_from_kwargs

    class _Declared(torch.nn.Module):
        forward = _forward_declaring_the_flag

    assert honors_router_logits_flag(_Declared()) is True
    assert honors_router_logits_flag(_KwargsOnly()) is False
    for head in (Qwen3_5MoeForConditionalGeneration, Qwen3_5MoeForCausalLM):

        class _Head(torch.nn.Module):
            forward = head.forward

        assert honors_router_logits_flag(_Head()) is True, head.__name__


def test_aux_loss_balancing_refuses_a_model_it_cannot_reach():
    """An inert balancing mode is invisible at runtime."""
    config = _config("qwen3_5_moe")
    set_config_field(config, "router_aux_loss_coef", 0.001, only_declared=True)

    class _Model(torch.nn.Module):
        forward = _forward_reading_the_flag_from_kwargs

        def __init__(self, cfg):
            super().__init__()
            self.config = cfg

        def modules(self):
            return iter(())

    with pytest.raises(ValueError, match="does not take output_router_logits"):
        apply_balancing_mode(_Model(config), "aux_loss", is_moe=True)


def test_qwen3_5_ep_layer_declares_the_bias_update_contract():
    """``bias_update`` is the only balancing a pipelined Qwen3.5 can get.
    Flipping ``SEVERS_AUX_LOSS`` also resolves ``auto`` to it on non-PP runs, whose HF aux-loss recorder is live."""
    assert EPQwen3_5MoELayer.SUPPORTS_BIAS_BALANCING is True
    assert EPQwen3_5MoELayer.SEVERS_AUX_LOSS is False


def test_zero_bias_routing_matches_the_stock_qwen3_5_router():
    """``bias_update`` re-selects from the raw logits instead of taking the router's output.
    If the algebra differs, enabling balancing silently changes which experts every token visits."""
    config = CONFIG_MAPPING["qwen3_5_moe"]().get_text_config()
    config.num_experts, config.num_experts_per_tok, config.hidden_size = 16, 4, 32
    torch.manual_seed(0)
    router = Qwen3_5MoeTopKRouter(config)

    class _Wrapper:
        """What Qwen3.5's ``_biased_route`` reads at zero bias. It must track the base methods it borrows by name."""

        top_k = config.num_experts_per_tok
        balancing_biases = torch.zeros(config.num_experts)
        _biased_route = EPQwen3_5MoELayer._biased_route
        _biased_topk = EPMoELayerBase._biased_topk
        _transient_balancing_bias = EPMoELayerBase._transient_balancing_bias

        def _apply_routing_replay(self, indices):
            return indices

    logits, router_weights, router_indices = router(torch.randn(64, config.hidden_size))
    indices, weights = _Wrapper()._biased_route(logits)

    assert torch.equal(indices, router_indices), "a zero bias must select exactly the router's experts"
    assert torch.allclose(weights, router_weights.float(), atol=1e-6), "and gate them the same way"


def _declared_head_dim(decoder):
    """Per-layer-aware: 5.20 heterogeneity raises on the global attribute. The max mirrors ``resolve_head_dim``."""
    try:
        return getattr(decoder, "head_dim", None)
    except RuntimeError:
        values = {getattr(lc, "head_dim", None) for lc in decoder.per_layer_config} - {None}
        return max(values) if values else None


@pytest.mark.parametrize("model_type", COMPOSITE_MODEL_TYPES + PLAIN_MODEL_TYPES)
def test_resolve_head_dim_matches_the_declared_value(model_type):
    """They are not interchangeable: Gemma 4 declares a head_dim its hidden/heads ratio does not reproduce.
    The RoPE inv_freq rebuild, the flash-attention warm-up and the pipeline-split cost model all size off it."""
    config = _config(model_type)
    decoder = config.get_text_config()
    expected = _declared_head_dim(decoder) or decoder.hidden_size // decoder.num_attention_heads
    assert resolve_head_dim(config) == expected


def test_resolve_head_dim_reads_through_a_composite_wrapper():
    """A wrapper read of ``head_dim`` gives ``None``, and the caller falls through to ``config.hidden_size``.
    That raises, or it is the vision tower's and silently sizes a decoder table from an unrelated dimension."""
    config = _config("gemma4")
    assert getattr(config, "head_dim", None) is None, "premise: the wrapper hides it"
    # 512, not 256: per-layer head_dims (sliding 256 / global 512), and the helper sizes for the largest.
    assert resolve_head_dim(config) == _declared_head_dim(config.get_text_config()) == 512


@pytest.mark.parametrize(("wrapper", "decoder"), [(True, False), (False, True), (True, True), (False, False)])
def test_the_tie_verdict_reads_both_levels_of_a_composite_wrapper(wrapper, decoder):
    """transformers resolves the tie per submodel, so either level's declaration ties the pair.
    False on a tied model lets the TP plan slice ``lm_head`` and halves the tied gradient."""
    config = _config("gemma4")
    config.tie_word_embeddings = wrapper
    config.get_text_config().tie_word_embeddings = decoder
    assert ties_word_embeddings(config) is (wrapper or decoder)


def test_the_tie_read_survives_a_config_without_the_composite_accessor():
    """A remote-code config with no ``get_text_config`` is read at its own level rather than aborting the load."""

    class _RemoteCodeConfig:
        tie_word_embeddings = True

    assert not hasattr(_RemoteCodeConfig(), "get_text_config"), "premise: no composite accessor"
    assert ties_word_embeddings(_RemoteCodeConfig()) is True
    assert ties_word_embeddings(None) is False


def test_resolve_num_key_value_heads_reduces_per_layer_heterogeneity():
    """Hub Gemma 4 registers it per layer ([2, 8]). The global read raises, and the helper sizes for the largest."""
    base = _config("gemma4").get_text_config()
    overrides = {i: {"num_key_value_heads": 2 if i % 2 else 8} for i in range(base.num_hidden_layers)}
    config = type(base)(**{**base.to_dict(), "per_layer_config": overrides})
    with pytest.raises(RuntimeError):
        _ = config.num_key_value_heads
    assert resolve_num_key_value_heads(config) == 8


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
