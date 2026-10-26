"""``assert_rank_uniform`` on a real 2-rank gloo group, and the digest its ranks gather.

Agreeing ranks pass. A divergent value raises on every rank under ``strict`` or, by default,
``HALO_TP_CONSISTENCY_CHECK``, and otherwise only warns; either names the probe's label. The digest
must be dtype-agnostic: a ``TypeError`` on one rank (``.numpy()`` on bf16/fp8) inside the
``all_gather_object`` would hang its peers.

Run: python tests/cpu/runtime/test_assert_rank_uniform.py
"""

import datetime
import json
import logging
import os

import pytest
import torch
import torch.distributed as dist

from halo.runtime.consensus import _value_digest, assert_rank_uniform
from tests.common.gloo import run_gloo_ranks

WORLD_SIZE = 2
PG_TIMEOUT = datetime.timedelta(seconds=90)
_DIVERGENCE = "divergent values across ranks"


class _Recorder(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.WARNING)
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


def _outcome(recorder: _Recorder, value, label: str, **kwargs) -> dict[str, object]:
    """Whether one check raised, with its error text, and the warnings it logged."""
    recorder.messages.clear()
    try:
        assert_rank_uniform(value, group=dist.group.WORLD, label=label, **kwargs)
        error = None
    except RuntimeError as exc:
        error = str(exc)
    return {"error": error, "warnings": list(recorder.messages)}


def _worker(rank: int, tmp_dir: str) -> None:
    recorder = _Recorder()
    logging.getLogger("halo.runtime.consensus").addHandler(recorder)
    divergent_shape = tuple(torch.zeros(2, 4 + rank).shape)
    os.environ.pop("HALO_TP_CONSISTENCY_CHECK", None)
    outcomes = {
        label: _outcome(recorder, value, label, **kwargs)
        for label, value, kwargs in (
            ("agree", {"step": 3}, {"strict": True}),
            ("same_shape", tuple(torch.zeros(2, 4).shape), {"strict": True}),
            ("strict", divergent_shape, {"strict": True}),
            ("warned", divergent_shape, {"strict": False}),
            ("env_unset", divergent_shape, {}),
        )
    }
    os.environ["HALO_TP_CONSISTENCY_CHECK"] = "1"
    outcomes["env_set"] = _outcome(recorder, divergent_shape, "env_set")
    with open(os.path.join(tmp_dir, f"rank{rank}.json"), "w") as handle:
        json.dump(outcomes, handle)


def test_divergence_raises_or_warns_on_every_rank_and_agreement_passes(tmp_path):
    run_gloo_ranks(_worker, WORLD_SIZE, str(tmp_path), pg_timeout=PG_TIMEOUT)
    for rank in range(WORLD_SIZE):
        outcomes = json.loads((tmp_path / f"rank{rank}.json").read_text())
        for label in ("agree", "same_shape"):
            assert outcomes[label] == {"error": None, "warnings": []}, f"rank {rank} {label}: {outcomes[label]}"
        for label in ("strict", "env_set"):
            expected = f"[assert_rank_uniform:{label}] {_DIVERGENCE}"
            assert expected in (outcomes[label]["error"] or ""), f"rank {rank} {label}: {outcomes[label]}"
        for label in ("warned", "env_unset"):
            expected = f"[assert_rank_uniform:{label}] {_DIVERGENCE}"
            warnings = outcomes[label]["warnings"]
            assert outcomes[label]["error"] is None, f"rank {rank} {label}: {outcomes[label]}"
            assert len(warnings) == 1 and expected in warnings[0], f"rank {rank} {label}: {outcomes[label]}"


def test_no_group_is_a_no_op(monkeypatch):
    monkeypatch.setattr(dist, "get_world_size", lambda *args, **kwargs: pytest.fail("group=None queried a group"))
    assert_rank_uniform(object(), group=None, label="unset", strict=True)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float8_e4m3fn, torch.float32, torch.int64, torch.bool])
def test_equal_tensors_hash_equal_and_different_differ(dtype):
    base = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    a = base.to(dtype)
    b = base.clone().to(dtype)
    assert _value_digest(a) == _value_digest(b)

    changed = base.clone()
    changed[0, 0] = 5.0
    assert _value_digest(a) != _value_digest(changed.to(dtype))


def test_dtype_and_shape_distinguish():
    z = torch.zeros(4, dtype=torch.bfloat16)
    assert _value_digest(z) != _value_digest(torch.zeros(4, dtype=torch.float16))  # same bytes, other dtype
    assert _value_digest(z) != _value_digest(torch.zeros(2, 2, dtype=torch.bfloat16))  # same bytes, other shape


def test_zero_dim_and_empty_bf16():
    assert _value_digest(torch.tensor(1.5, dtype=torch.bfloat16)) == _value_digest(
        torch.tensor(1.5, dtype=torch.bfloat16)
    )
    assert _value_digest(torch.empty(0, dtype=torch.bfloat16)) == _value_digest(torch.empty(0, dtype=torch.bfloat16))


def test_noncontiguous_matches_contiguous():
    t = torch.arange(16, dtype=torch.bfloat16).reshape(4, 4)
    assert _value_digest(t.t()) == _value_digest(t.t().contiguous())


def test_mixed_containers_hash_element_wise():
    t = torch.ones(3, dtype=torch.bfloat16)
    # Non-tensor first element must not divert the container into the pickle branch.
    assert _value_digest([5, t]) == _value_digest([5, t.clone()])
    assert _value_digest([5, t]) != _value_digest([5, t * 2])
    assert _value_digest([5, t]) != _value_digest([6, t])
    # Tensor-first mixed list (bf16) must not crash and must track the non-tensor tail.
    assert _value_digest([t, "x"]) == _value_digest([t.clone(), "x"])
    assert _value_digest([t, "x"]) != _value_digest([t, "y"])
    # Nesting is distinguished by the type/length prefix.
    assert _value_digest([t]) != _value_digest([[t]])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
