"""The CP equivalence harness: its roster covers every Ulysses wrapper class, its CPU stand-ins for the flash
kernel and the list all-to-all keep their contracts, the GPU tier keeps CP's own kernel choice, two launches of
today's CP code match bit for bit, the envelope rule judges a base that does not repeat, each seeded change to the
CP path shows up in exactly the records it touches, and a negative control that raises fails the verdict.

Run: python tests/cpu/parallelism/test_cp_equivalence_harness.py
"""

import contextlib
from pathlib import Path
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from halo.distributed.context_parallel.families import (
    ULYSSES_ATTENTION_MAP,
    GptOssUlyssesAttention,
    Mistral4UlyssesAttention,
)
from halo.distributed.context_parallel.ulysses_attention import UlyssesAttention
from halo.distributed.context_parallel.ulysses_model import UlyssesModel
from tests.common.equivalence import cp_equivalence
from tests.common.equivalence.cp_equivalence import (
    CP_SIZE,
    FAMILIES,
    LEVELS,
    assess,
    deterministic_kernels,
    gloo_all_to_all,
    judge,
    produce,
    record_rank,
    reference_flash_attn,
    tier_controls,
)
from tests.common.equivalence.revisions import run_in_tree
from tests.common.equivalence.tensor_records import (
    GRADIENT_PREFIX,
    case_of,
    compare_records,
    load_record,
    raised_records,
    record_case,
    record_paths,
)
from tests.common.gloo import run_gloo_ranks
from tests.common.repo import REPO_ROOT
from tests.common.tiny_models import tiny_family_model

MODULE = produce.__module__
# A windowed family per wrapper path that windows (GPT-OSS on the optimized path with its own rotary, Cohere2 MoE
# on its interleaved one) and an unwindowed one.
SAMPLE_FAMILIES = ("gpt_oss", "cohere2_moe", "qwen3")
ATTEND = UlyssesAttention._attend
CP_LOSS = UlyssesModel._compute_cp_loss
APPLY_ROTARY = Mistral4UlyssesAttention._apply_rotary


def _unwindowed(self, query, key, value):
    window, self.sliding_window = self.sliding_window, None
    try:
        return ATTEND(self, query, key, value)
    finally:
        self.sliding_window = window


def _swapped_rotary(x, cos, sin):
    first, second = x.chunk(2, dim=-1)
    return torch.cat((second * cos + first * sin, first * cos - second * sin), dim=-1)


def _halved_cp_loss(self, logits, labels):
    return CP_LOSS(self, logits, labels) * 0.5


def _chunk_local_llama4_scale(self, q, k, cos, sin, position_ids=None):
    """The llama-4 query scale read off positions restarting at each chunk instead of the sequence's."""
    return APPLY_ROTARY(self, q, k, cos, sin, position_ids % position_ids.shape[-1])


def _raising_nudge(attention):
    raise RuntimeError("the negative control found no weight to move")


MUTATIONS = {
    "unwindowed": (UlyssesAttention, "_attend", _unwindowed),
    "swapped_rotary": (GptOssUlyssesAttention, "_rotate_split_halves", staticmethod(_swapped_rotary)),
    "halved_cp_loss": (UlyssesModel, "_compute_cp_loss", _halved_cp_loss),
    "chunk_local_llama4_scale": (Mistral4UlyssesAttention, "_apply_rotary", _chunk_local_llama4_scale),
    "raising_nudge": (cp_equivalence, "_nudge", _raising_nudge),
}


def _record_worker(rank: int, output: str, families, mutation: str | None, perturb: bool) -> None:
    patched = patch.object(*MUTATIONS[mutation]) if mutation else contextlib.nullcontext()
    with patched, tier_controls("cpu"):
        record_rank(Path(output), families, LEVELS, tier="cpu", device=torch.device("cpu"), perturb=perturb)


def _recorded(output: Path, mutation: str | None = None, *, families=SAMPLE_FAMILIES, perturb: bool = False) -> Path:
    run_gloo_ranks(_record_worker, CP_SIZE, str(output), families, mutation, perturb)
    return output


def _changed_cases(baseline: Path, changed: Path) -> set[str]:
    deltas = compare_records(baseline, changed)
    assert all(delta["delta"] == "differs" for delta in deltas)
    return {case_of(delta["path"]) for delta in deltas}


def _exchange_worker(rank: int) -> None:
    """Peer ``j`` receives ``j + 1`` values from each rank, so the split sizes differ per peer."""
    inputs = [torch.full((peer + 1,), float(10 * rank + peer)) for peer in range(CP_SIZE)]
    outputs = [torch.empty(rank + 1) for _ in range(CP_SIZE)]
    gloo_all_to_all(outputs, inputs)
    assert [output.tolist() for output in outputs] == [
        [float(10 * peer + rank)] * (rank + 1) for peer in range(CP_SIZE)
    ]


@pytest.fixture(scope="module")
def baseline(tmp_path_factory):
    return _recorded(tmp_path_factory.mktemp("baseline"))


def test_the_roster_runs_every_ulysses_wrapper_class_once():
    wrapped = {}
    for name, family in FAMILIES.items():
        model = tiny_family_model(family.tiny, overrides=family.overrides)
        wrapped[name] = {
            ULYSSES_ATTENTION_MAP[type(m).__name__]
            for m in model.modules()
            if type(m).__name__ in ULYSSES_ATTENTION_MAP
        }
    assert all(len(classes) == 1 for classes in wrapped.values()), wrapped
    assert set().union(*wrapped.values()) == set(ULYSSES_ATTENTION_MAP.values())
    assert len(FAMILIES) == len(set(ULYSSES_ATTENTION_MAP.values()))


@pytest.mark.parametrize("window", [None, 5])
def test_the_reference_attention_is_causal_gqa_attention(window):
    generator = torch.Generator().manual_seed(0)
    q = torch.randn(2, 16, 4, 8, generator=generator)
    k, v = (torch.randn(2, 16, 2, 8, generator=generator) for _ in range(2))
    rows, cols = torch.arange(16)[:, None], torch.arange(16)[None, :]
    mask = (cols <= rows) & (cols >= rows - window if window else True)

    expected = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), attn_mask=mask, scale=0.3, enable_gqa=True
    ).transpose(1, 2)
    got = reference_flash_attn(
        q, k, v, softmax_scale=0.3, causal=True, window_size=(window or -1, 0 if window else -1)
    )

    torch.testing.assert_close(got, expected)


def test_the_reference_attention_refuses_a_keyword_it_does_not_implement():
    q = torch.randn(1, 4, 2, 8)
    with pytest.raises(TypeError, match="softcap"):
        reference_flash_attn(q, q, q, causal=True, softcap=30.0)


def test_the_gpu_tier_keeps_the_kernel_cp_picks_and_makes_it_deterministic():
    picked = []

    def probe(allow_fa4):
        picked.append(allow_fa4)
        return lambda *args, **kwargs: kwargs

    _flash_attn_func = deterministic_kernels(probe)

    assert _flash_attn_func(False)(causal=True) == {"causal": True, "deterministic": True}
    assert _flash_attn_func()() == {"deterministic": True}
    assert picked == [False, True]


def test_the_gloo_all_to_all_delivers_what_the_list_collective_does():
    run_gloo_ranks(_exchange_worker, CP_SIZE)


def test_two_launches_of_every_family_match_bit_for_bit(tmp_path):
    """Base against base: each run is its own ``torchrun`` launch, as ``make verify-cp`` runs them."""
    outputs = [tmp_path / "first", tmp_path / "second"]
    for output in outputs:
        run_in_tree(REPO_ROOT, MODULE, "--produce", str(output), "--tier", "cpu", nproc=CP_SIZE)

    records = record_paths(outputs[0])
    assert len(records) == len(FAMILIES) * len(LEVELS) * CP_SIZE
    assert judge(outputs[0], outputs[1], outputs[1]) == ([], [])
    assert raised_records(outputs[0]) == {}
    for record, path in records.items():
        tensors = load_record(path)["tensors"]
        assert any(name.startswith(GRADIENT_PREFIX) and tensor.any() for name, tensor in tensors.items()), record


def test_a_dropped_sliding_window_changes_only_the_windowed_families(baseline, tmp_path):
    changed = _changed_cases(baseline, _recorded(tmp_path / "mutated", "unwindowed"))
    assert changed == {f"{family}/{level}" for family in ("gpt_oss", "cohere2_moe") for level in LEVELS}


def test_a_wrong_rotary_changes_its_family_at_both_levels(baseline, tmp_path):
    changed = _changed_cases(baseline, _recorded(tmp_path / "mutated", "swapped_rotary"))
    assert changed == {f"gpt_oss/{level}" for level in LEVELS}


def test_a_rescaled_cp_loss_changes_the_model_level_only(baseline, tmp_path):
    changed = _changed_cases(baseline, _recorded(tmp_path / "mutated", "halved_cp_loss"))
    assert changed == {f"{family}/model" for family in SAMPLE_FAMILIES}


def test_a_llama4_scale_at_chunk_positions_changes_mistral4_at_both_levels(tmp_path):
    families = ("mistral4", "glm4_moe_lite")
    baseline = _recorded(tmp_path / "baseline", families=families)

    changed = _changed_cases(baseline, _recorded(tmp_path / "mutated", "chunk_local_llama4_scale", families=families))

    assert changed == {f"mistral4/{level}" for level in LEVELS}


def test_the_negative_control_changes_every_record(baseline, tmp_path):
    perturbed = _recorded(tmp_path / "perturbed", perturb=True)

    changed = _changed_cases(baseline, perturbed)

    assert changed == {f"{family}/{level}" for family in SAMPLE_FAMILIES for level in LEVELS}
    assert assess(baseline, baseline, baseline, perturbed)[0] == "pass"


def test_a_negative_control_that_raises_fails_the_verdict(baseline, tmp_path):
    control = _recorded(tmp_path / "control", "raising_nudge", families=SAMPLE_FAMILIES[:1], perturb=True)

    verdict, findings = assess(baseline, baseline, baseline, control)

    assert verdict == "fail"
    assert findings["negative_control"]["unseen"] == sorted(
        f"{SAMPLE_FAMILIES[0]}/{level}/rank{rank}" for level in LEVELS for rank in range(CP_SIZE)
    )


def _write(root: Path, **tensors: torch.Tensor) -> Path:
    record_case(root, "family/model", 0, lambda: (tensors, {}))
    return root


def test_a_repeating_base_is_judged_bit_for_bit(tmp_path):
    base = _write(tmp_path / "base", loss=torch.tensor(1.0), **{"grad/w": torch.tensor([1.0, 2.0])})
    head = _write(tmp_path / "head", loss=torch.tensor(1.0), **{"grad/w": torch.tensor([1.0, 2.0 + 2**-20])})

    deltas, enveloped = judge(base, base, head)

    assert enveloped == []
    assert deltas == [
        {
            "path": "family/model/rank0",
            "delta": "differs",
            "entries": ["tensors/grad/w: bits differ, max |delta| 9.537e-07"],
        }
    ]


def test_a_base_that_does_not_repeat_takes_the_gradient_envelope(tmp_path):
    base = _write(tmp_path / "base", loss=torch.tensor(1.0), **{"grad/w": torch.tensor([1.0, 2.0])})
    repeat = _write(tmp_path / "repeat", loss=torch.tensor(1.0), **{"grad/w": torch.tensor([1.0, 2.5])})
    inside = _write(tmp_path / "inside", loss=torch.tensor(1.0), **{"grad/w": torch.tensor([1.25, 2.0])})
    outside = _write(tmp_path / "outside", loss=torch.tensor(1.0), **{"grad/w": torch.tensor([1.0, 3.0])})
    moved_loss = _write(tmp_path / "loss", loss=torch.tensor(1.5), **{"grad/w": torch.tensor([1.0, 2.0])})

    judged = {
        label: judge(base, repeat, head)
        for label, head in (("inside", inside), ("outside", outside), ("loss", moved_loss))
    }

    assert {label: enveloped for label, (_, enveloped) in judged.items()} == dict.fromkeys(
        judged, ["family/model/rank0"]
    )
    assert judged["inside"][0] == []
    assert judged["outside"][0][0]["entries"] == [
        "tensors/grad/w: max |delta| 1.000e+00 outside the base envelope 5.000e-01"
    ]
    assert judged["loss"][0][0]["entries"] == ["tensors/loss: bits differ, max |delta| 5.000e-01"]


def test_a_record_missing_from_the_head_is_a_delta(tmp_path):
    base = _write(tmp_path / "base", loss=torch.tensor(1.0))
    (tmp_path / "empty").mkdir()

    deltas, _ = judge(base, base, tmp_path / "empty")

    assert deltas == [{"path": "family/model/rank0", "delta": "only in base"}]


def test_a_base_whose_two_runs_record_different_records_is_refused(tmp_path):
    base = _write(tmp_path / "base", loss=torch.tensor(1.0))
    (tmp_path / "empty").mkdir()

    with pytest.raises(RuntimeError, match="family/model/rank0"):
        judge(base, tmp_path / "empty", base)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
