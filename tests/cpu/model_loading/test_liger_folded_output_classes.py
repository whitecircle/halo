"""The output classes ``supply_folded_output_classes`` binds for Liger's multimodal fused forwards, which construct them
with the model output's ``rope_deltas``."""

import pytest
import torch
from liger_kernel.transformers.model import output_classes
from transformers.modeling_outputs import BaseModelOutputWithPast, MoeModelOutputWithPast

from halo.model_loading.liger.upstream_outputs import _FOLDED_OUTPUT_CLASSES, supply_folded_output_classes


@pytest.mark.parametrize("name", sorted(_FOLDED_OUTPUT_CLASSES))
def test_a_folded_output_takes_the_rope_deltas_its_fused_forward_passes(name):
    supply_folded_output_classes()
    deltas = torch.tensor([3, -1])
    output = getattr(output_classes, name)(loss=torch.tensor(1.0), logits=None, rope_deltas=deltas)
    assert output.rope_deltas is deltas
    assert output.loss.item() == 1.0


@pytest.mark.parametrize("model_output", [BaseModelOutputWithPast, MoeModelOutputWithPast])
def test_a_folded_family_decoder_output_answers_none_for_its_cached_rope_deltas(model_output):
    supply_folded_output_classes()
    output = model_output(last_hidden_state=torch.zeros(1, 2, 4))
    assert output.rope_deltas is None
    assert "rope_deltas" not in output


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
