#!/usr/bin/env python
"""``resolve_retained`` holds one handed-in item at a time.

A gathered save hands it tensors derived lazily inside the loop (an injected-LoRA fold per weight), so an
item still referenced while the next one is made doubles the transient peak: adjacent fold targets such as
one layer's q/k/v would each live beside their successor.

Run: python tests/cpu/runtime/test_resolve_retained.py
"""

import weakref

import pytest
import torch

from src.runtime.dtensor import resolve_retained

ITEMS = 4


@pytest.mark.parametrize("retain", [True, False], ids=["writer", "non-writer"])
def test_each_item_is_released_before_the_next_is_made(retain):
    live = {"now": 0, "peak": 0}

    def released():
        live["now"] -= 1

    def made(index: int) -> torch.Tensor:
        tensor = torch.full((8,), float(index))
        weakref.finalize(tensor, released)
        live["now"] += 1
        live["peak"] = max(live["peak"], live["now"])
        return tensor

    resolved = resolve_retained(((f"w{index}", made(index)) for index in range(ITEMS)), retain=retain)

    assert live["peak"] == 1
    expected = {f"w{index}": torch.full((8,), float(index)) for index in range(ITEMS)} if retain else {}
    assert resolved.keys() == expected.keys()
    assert all(torch.equal(resolved[name], tensor) for name, tensor in expected.items())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
