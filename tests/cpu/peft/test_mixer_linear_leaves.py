"""The tiny-checkpoint target list drops a leaf name a convolution also ends in, since PEFT matches by suffix."""

import pytest
from torch import nn

from tests.common.adapters import mixer_linear_leaves


class _Attention(nn.Module):
    def __init__(self, proj: nn.Module):
        super().__init__()
        self.proj = proj
        self.q_proj = nn.Linear(4, 4)


class _Tower(nn.Module):
    def __init__(self, proj: nn.Module):
        super().__init__()
        self.attn = _Attention(proj)


def test_a_leaf_shared_with_a_patch_embedding_convolution_is_dropped():
    model = nn.ModuleDict(
        {"vision": _Tower(nn.Linear(4, 4)), "patch_embed": nn.ModuleDict({"proj": nn.Conv3d(3, 4, 2)})}
    )
    assert mixer_linear_leaves(model) == ["q_proj"]


def test_a_linear_leaf_no_convolution_shares_is_kept():
    model = nn.ModuleDict(
        {"vision": _Tower(nn.Linear(4, 4)), "patch_embed": nn.ModuleDict({"embed": nn.Conv3d(3, 4, 2)})}
    )
    assert mixer_linear_leaves(model) == ["proj", "q_proj"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
