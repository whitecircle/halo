#!/usr/bin/env python
"""GptOss sinks policy must reach the parameter's OWNER through the CP (Ulysses) wrapper.

The trap this pins: a CP wrapper that aliases ``sinks`` at construction puts the FA2 branch of
``_reset_gpt_oss_sinks`` (a rebind, ``attn.sinks = None``) on the wrapper while
``original_attention`` keeps the live pretrained tensor — which the save paths surface under the
clean HF key after ``strip_cp_attention_prefix``. A model trained with no sinks then exports live
pretrained sinks, shifting every served logprob, while the wrapper-side guard reads its own None and
passes. The in-place branches (eager/flex ``fill_``) are visible through both references and cannot
fail that way — the matrix below pins all of it.

    python tests/cpu/parallelism/test_gptoss_cp_sinks_export.py
"""

import pytest
import torch
import torch.nn as nn
from accelerate import PartialState
from transformers import CONFIG_MAPPING
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssAttention

PartialState()  # the sinks policy logs through accelerate's logger, which requires live state

from src.distributed.context_parallel.key_mapping import strip_cp_attention_prefix
from src.distributed.context_parallel.layers.gpt_oss import GptOssUlyssesAttention
from src.models.patches.gpt_oss_sinks import _reset_gpt_oss_sinks, neutralized_gpt_oss_sinks

LIVE_SINK_VALUE = 0.75


def _tiny_config():
    config = CONFIG_MAPPING["gpt_oss"]()
    config.hidden_size = 32
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.head_dim = 8
    config.num_hidden_layers = 2
    return config


class _Layer(nn.Module):
    def __init__(self, attn: nn.Module):
        super().__init__()
        self.self_attn = attn


class _Backbone(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)


class _Model(nn.Module):
    def __init__(self, config, cp: bool):
        super().__init__()
        attns = []
        for idx in range(config.num_hidden_layers):
            attn = GptOssAttention(config, layer_idx=idx)
            with torch.no_grad():
                attn.sinks.fill_(LIVE_SINK_VALUE)
            if cp:
                attn = GptOssUlyssesAttention(attn, cp_group=None, cp_size=2)
            attns.append(attn)
        self.model = _Backbone([_Layer(a) for a in attns])
        self.config = config


def _exported_sinks(model) -> dict:
    """State dict as every CP-aware serializer writes it (prefix-strip applied)."""
    return {strip_cp_attention_prefix(k): v for k, v in model.state_dict().items() if "sinks" in k}


@pytest.mark.parametrize("cp", [False, True], ids=["plain", "cp"])
@pytest.mark.parametrize("impl", ["flash_attention_2", "eager"])
def test_reset_sinks_reaches_the_owner(cp, impl):
    config = _tiny_config()
    model = _Model(config, cp=cp)

    _reset_gpt_oss_sinks(model, config, attn_implementation=impl)

    exported = _exported_sinks(model)
    if impl == "flash_attention_2":
        # The rebind must land on the owner: no live sinks key may survive into the export; the
        # save paths re-emit the neutralized tensors via neutralized_gpt_oss_sinks.
        assert exported == {}, f"live sinks leaked into the export: {list(exported)}"
        rebuilt = neutralized_gpt_oss_sinks(model)
        assert set(rebuilt) == {f"model.layers.{i}.self_attn.sinks" for i in range(config.num_hidden_layers)}
        for tensor in rebuilt.values():
            assert torch.all(tensor == torch.finfo(tensor.dtype).min)
    else:
        assert set(exported) == {f"model.layers.{i}.self_attn.sinks" for i in range(config.num_hidden_layers)}
        for tensor in exported.values():
            assert torch.all(tensor == torch.finfo(tensor.dtype).min), "eager fill must neutralize in place"

    if cp:
        for layer in model.model.layers:
            layer.self_attn._check_sinks_neutralized()  # guard reads the owner and must pass


def test_cp_guard_fires_on_live_sinks():
    config = _tiny_config()
    model = _Model(config, cp=True)
    with pytest.raises(ValueError, match="reset sinks"):
        model.model.layers[0].self_attn._check_sinks_neutralized()


def test_cp_wrapper_holds_no_second_binding():
    config = _tiny_config()
    model = _Model(config, cp=True)
    wrapper = model.model.layers[0].self_attn
    assert "sinks" not in dict(wrapper.named_parameters(recurse=False))
    assert wrapper.sinks is wrapper.original_attention.sinks
    with pytest.raises(AttributeError):
        wrapper.sinks = None  # read-only: the policy must rebind through the owner


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
