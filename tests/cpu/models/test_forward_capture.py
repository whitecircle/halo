#!/usr/bin/env python
"""``capturing_forward_outputs``: a module's forward outputs, projected, for the length of one block only.
A hook outliving its block would keep appending and holding graphs, so it must go on exit, raised or not."""

import pytest
import torch
from torch import nn

from halo.models.forward_capture import capturing_forward_outputs


def test_every_forward_in_the_block_is_captured_in_call_order_and_none_after():
    layer = nn.Linear(2, 2)
    inputs = [torch.randn(1, 2) for _ in range(3)]
    with capturing_forward_outputs(layer, lambda output: output.detach() * 2) as captured:
        outputs = [layer(x) for x in inputs[:2]]
    layer(inputs[2])
    assert len(captured) == 2
    for got, output in zip(captured, outputs, strict=True):
        torch.testing.assert_close(got, output.detach() * 2)


def test_the_hook_goes_when_the_block_raises():
    layer = nn.Linear(2, 2)
    with pytest.raises(RuntimeError, match="loss failed"), capturing_forward_outputs(layer, lambda o: o) as captured:
        layer(torch.randn(1, 2))
        raise RuntimeError("loss failed")
    layer(torch.randn(1, 2))
    assert len(captured) == 1 and not layer._forward_hooks


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
