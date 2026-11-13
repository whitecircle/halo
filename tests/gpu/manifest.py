"""Launch spec per GPU test script, keyed by path under ``tests/gpu/``; ``conftest.py`` makes one pytest node per row.

A script missing here fails collection. A spec's markers are the union over its rows, so ``-m "gpu and cp"`` also
selects its non-CP rows. Marker meanings and server-tier setup: ``agent-docs/contributing/README.md``. A trainer and
server on different NCCL transports hang to the 120 s formation deadline per test; restart a server whose trainer was
killed while attached.
"""

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Literal

GPU_DIR = Path(__file__).parent
# Pytest collects these directly; everything else under tests/gpu/ is a torchrun script.
LAUNCHER_ENTRYPOINTS = frozenset({"test_suite.py", "test_launcher_contract.py"})


@dataclass(frozen=True)
class Row:
    """An ``args_matrix`` entry whose set launch fields replace the spec's, so one script runs at several widths."""

    args: str
    nproc: int | None = None
    markers: tuple | None = None
    timeout: int | None = None
    flaky: bool | None = None


@dataclass(frozen=True)
class TestSpec:
    """One script's launch. ``gpu_test_main(exact_world_size=N)`` in the script owns world-size strictness. ``timeout``
    hard-kills the process group, since NCCL and FA hangs do not return. ``launcher="python"`` runs one process that
    starts its own ``nproc`` ranks, handed the count in ``HALO_TEST_NPROC``."""

    # Keeps pytest from collecting this as a test class.
    __test__ = False

    nproc: int
    markers: tuple = ()
    args_matrix: tuple = ("",)
    timeout: int = 1200
    flaky: bool = False
    launcher: Literal["torchrun", "python"] = "torchrun"

    def launches(self) -> list[tuple[str, "TestSpec"]]:
        """``(args, spec)`` per ``args_matrix`` entry, the spec carrying that entry's own launch values."""
        launches = []
        for entry in self.args_matrix:
            row = Row(entry) if isinstance(entry, str) else entry
            own = {f.name: getattr(row, f.name) for f in fields(Row)[1:] if getattr(row, f.name) is not None}
            launches.append((row.args, replace(self, args_matrix=(row.args,), **own)))
        return launches


def _rows(args_matrix, **launch) -> tuple[Row, ...]:
    return tuple(Row(args, **launch) for args in args_matrix)


# ``--family`` names of TINY_MOE_FAMILIES and their markers; tests/cpu/conventions/test_tiny_family_roster.py holds the
# names and the rows below to the registry.
_TINY_MOE_FAMILY_MARKERS = {
    "bailing_moe": "bailing",
    "cohere2_moe": "cohere2_moe",
    "deepseek_v4": "deepseek_v4",
    "gemma4_text": "gemma4",
    "glm4_moe_lite": "glm4",
    "glm5_next": "glm5",
    "gpt_oss": "gptoss",
    "inkling_text": "inkling",
    "laguna": "laguna",
    "lfm2_moe": "lfm2",
    "mistral4": "mistral4",
    "nemotron_h": "nemotron",
    "qwen3_5_moe_text": "qwen3",
    "qwen3_moe": "qwen3",
    "qwen3_next": "qwen3",
    "qwen4_exp_text": "qwen3",
    "step3p7": "step3p7",
    "zaya": "zaya",
}
_TINY_MOE_FAMILIES = tuple(_TINY_MOE_FAMILY_MARKERS)
# The families test_family_packing_isolation.py holds on the kernel a packed run resolves, past its original rows.
_PACKED_DEFAULT_FAMILIES = (
    "cohere2_moe",
    "gemma4_text",
    "glm4_moe_lite",
    "gpt_oss",
    "laguna",
    "lfm2_moe",
    "mistral4",
    "nemotron_h",
    "qwen3_5_moe_text",
    "qwen3_moe",
    "qwen3_next",
    "step3p7",
)
_PACKED_EP_FAMILIES = ("gpt_oss", "qwen3_5_moe_text", "qwen3_moe", "zaya")
# The other EP layers a packed recipe runs.
_PACKED_EP_FULL_FAMILIES = (
    "bailing_moe",
    "cohere2_moe",
    "gemma4_text",
    "glm4_moe_lite",
    "laguna",
    "lfm2_moe",
    "mistral4",
    "nemotron_h",
    "qwen3_next",
    "step3p7",
)
# One family per expert layout the reentrant-recompute suite crosses; every other family runs only its full fine-tune.
_REENTRANT_FAMILIES = ("qwen3_moe", "gpt_oss", "gemma4_text")
_REENTRANT_SWEEP_FAMILIES = tuple(family for family in _TINY_MOE_FAMILIES if family not in _REENTRANT_FAMILIES)


def _family_markers(families) -> tuple[str, ...]:
    return tuple(sorted({_TINY_MOE_FAMILY_MARKERS[family] for family in families}))


_MERGED_RESUME_CORE_ROWS = (
    "--family qwen3_moe --adapters expert",
    "--family qwen3_moe --adapters mixed",
    "--family gpt_oss --adapters expert",
    "--family gpt_oss --adapters mixed",
    "--family qwen3_moe --adapters expert --ep-size 1",
    "--family gpt_oss --adapters mixed --ep-size 1",
    "--family qwen3_moe --adapters mixed --cp-size 2",
    "--family qwen3_moe --adapters expert --fp32-masters",
    "--family qwen3_moe --adapters mixed --fp32-masters",
)
_MERGED_RESUME_FAMILY_ROWS = tuple(
    row
    for family in _TINY_MOE_FAMILIES
    for adapters in ("expert", "mixed")
    for layout in ("", " --ep-size 1", " --cp-size 2")
    if (row := f"--family {family} --adapters {adapters}{layout}") not in _MERGED_RESUME_CORE_ROWS
)
_PRECOMPUTE_CORE_ROWS = (
    "--trainer dpo --family qwen3_moe",
    "--trainer kto --family qwen3_moe",
    "--trainer kto --family qwen3_moe --kto-loss apo_zero_unpaired",
    "--trainer dpo --family qwen3_moe --mode ep1",
    "--trainer dpo --family qwen3_moe --mode etp2",
    "--trainer dpo --family qwen3_moe --mode tp2",
    "--trainer dpo --family qwen3_moe --mode ep2tp2",
    "--trainer dpo --family dense --mode dp2",
    "--trainer dpo --family dense --mode tp2",
    "--trainer kto --family dense --mode tp2",
    "--trainer dpo --family dense --mode dp2 --peft",
    "--trainer kto --family dense --mode dp2 --peft --kto-loss apo_zero_unpaired",
)
# The core rows cover Qwen3-MoE.
_PRECOMPUTE_SWEEP_FAMILIES = tuple(family for family in _TINY_MOE_FAMILIES if family != "qwen3_moe")
# The CP roster (tests/common/equivalence/cp_equivalence.py); test_tiny_family_roster.py holds the rows to it.
_CP_MOE_FAMILIES = (
    "bailing_moe",
    "cohere2_moe",
    "glm4_moe_lite",
    "gpt_oss",
    "mistral4",
    "qwen3_5_moe_text",
    "qwen3_moe",
)
_CP_ROSTER_ROWS = (
    "--family qwen3",
    Row("--family qwen3_vl", markers=("gpu", "core", "2gpu", "cp", "vlm", "qwen3")),
    *(
        Row(f"--family {family}", markers=("gpu", "core", "2gpu", "cp", "moe", _TINY_MOE_FAMILY_MARKERS[family]))
        for family in _CP_MOE_FAMILIES
    ),
)
# Qwen3.5-MoE is left out: the CP suites run it with its gated DeltaNet layers swapped for attention.
_ULYSSES_MOE_FAMILIES = tuple(family for family in _CP_MOE_FAMILIES if family != "qwen3_5_moe_text")
# GPT-OSS has its own PP suites.
_PP_MOE_FAMILIES = (
    "deepseek_v4",
    "glm4_moe_lite",
    "glm5_next",
    "inkling_text",
    "laguna",
    "nemotron_h",
    "qwen3_5_moe_text",
    "qwen3_moe",
    "step3p7",
)
# test_tiny_family_roster.py holds this to halo/distributed/tensor_parallel/parallelize_attention.py.
_TP_FAMILIES = ("cohere2_moe", "glm4_moe_lite", "gpt_oss", "lfm2_moe", "mistral4", "qwen3_5_moe_text", "qwen3_moe")
_PRECOMPUTE_FAMILY_ROWS = tuple(
    f"--trainer {trainer} --family {family}{mode}"
    for family in _PRECOMPUTE_SWEEP_FAMILIES
    for trainer, mode in (
        ("dpo", ""),
        ("dpo", " --mode ep1"),
        ("kto", ""),
        ("dpo", " --mode etp2"),
        *((("dpo", " --mode tp2"), ("dpo", " --mode ep2tp2")) if family in _TP_FAMILIES else ()),
    )
)
_DETERMINISM_CORE_ROWS = (
    *(f"--family gpt_oss --mode {mode}" for mode in ("ep2", "ep2_top1", "ep2_loop", "ep2_legacy", "ep1", "etp2")),
    "--family qwen3_moe --mode ep2",
)
_DETERMINISM_SWEEP_FAMILIES = tuple(family for family in _TINY_MOE_FAMILIES if family not in ("gpt_oss", "qwen3_moe"))
# Every syncable MoE family but Qwen3-MoE; the roster test holds this to the families an engine can update online.
_SYNC_EXACTNESS_SWEEP_FAMILIES = (
    "bailing_moe",
    "gemma4_text",
    "glm4_moe_lite",
    "glm5_next",
    "gpt_oss",
    "laguna",
    "lfm2_moe",
    "qwen3_5_moe_text",
    "step3p7",
)

# Row family → its marker, on both engines but for the two SGLang refuses; each needs a fresh server on the family's
# checkpoint (tests/common/weight_push_exact.py), so a tier pass on another checkpoint skips them.
# Qwen3.6's text tower is past the tiny roster, on Qwen3.5's layer.
_PUSH_EXACT_MOE_FAMILY_MARKERS = {
    family: {**_TINY_MOE_FAMILY_MARKERS, "qwen3_6_moe_text": "qwen3"}[family]
    for family in (
        "bailing_moe",
        "gemma4_text",
        "glm4_moe_lite",
        "glm5_next",
        "gpt_oss",
        "laguna",
        "lfm2_moe",
        "qwen3_5_moe_text",
        "qwen3_6_moe_text",
        "qwen3_moe",
        "step3p7",
    )
}
_PUSH_EXACT_VLLM_ONLY = ("laguna", "step3p7")
_PUSH_EXACT_MOE_ROWS = tuple(
    Row(
        f"--family {family} --backend {backend} --ep-size {ep_size}",
        markers=("gpu", "full", "2gpu", "ep", "moe", marker, f"{backend}_server"),
    )
    for family, marker in _PUSH_EXACT_MOE_FAMILY_MARKERS.items()
    for backend in ("vllm", "sglang")
    if backend == "vllm" or family not in _PUSH_EXACT_VLLM_ONLY
    for ep_size in (1, 2)
)
# One push per forwarded name, on the families whose real-step bar is too gross to catch a name the engine drops.
_PUSH_EXACT_REACH_ROWS = tuple(
    Row(
        f"--family {family} --backend {backend} --ep-size 2 --reach",
        markers=("gpu", "full", "2gpu", "ep", "moe", _PUSH_EXACT_MOE_FAMILY_MARKERS[family], f"{backend}_server"),
        timeout=3600,
    )
    for family, backend in (("laguna", "vllm"), ("gemma4_text", "vllm"), ("gemma4_text", "sglang"))
)
_PUSH_EXACT_DENSE_ROWS = tuple(
    Row(
        f"--family qwen3 --backend {backend}{layout}",
        markers=("gpu", "full", "2gpu", *axes, "qwen3", f"{backend}_server"),
    )
    for backend in ("vllm", "sglang")
    for layout, axes in (("", ()), (" --tp-size 2", ("tp",)))
)

_ENV_GRPO_4GPU_ROWS = (
    "--ep-size 2 --etp-size 2",
    "--ep-size 2 --tp-size 2",
    "--ep-size 2 --etp-size 2 --peft lora",
    "--ep-size 4",
    "--ep-size 4 --peft expert_lora",
)

_LORA_SFT_ROWS = {
    "gpt_oss_20b": (
        "ep --convergence --save-reload",
        "ep --attn flash_attention_2",
        "ep_cp --save-reload",
        "etp --save-reload",
        "dp_stock_experts --qlora --save-reload",
    ),
    "qwen3_0_6b": (
        "fsdp --save-reload",
        "fsdp --qlora --save-reload",
        "cp --save-reload --resume",
        "cp --qlora --save-reload",
    ),
}

# Family x --lora (no flag: attention adapters; off: full fine-tune) x run shape. --head adds a projection head after
# the pooling, refused under FSDP2 and TP, accepted under DDP.
_EMBEDDING_FAMILIES = ("bert", "qwen3", "qwen3_5", "gemma4", "gpt_oss")
_EMBEDDING_LORA_FLAGS = {
    lora: "" if lora == "attention" else f" --lora {lora}"
    for lora in ("attention", "mixed", "embedding", "dora", "off")
}
_EMBEDDING_TP_FAMILIES = ("qwen3", "qwen3_5", "gpt_oss")
_EMBEDDING_CORE_ROWS = (
    "--family bert --mode fsdp",
    "--family bert --mode ddp",
    "--family bert --mode presharded",
    "--family bert --mode ddp --lora mixed",
    "--family qwen3 --mode fsdp",
    "--family qwen3 --mode tp",
    "--family qwen3 --mode fsdp --lora mixed",
    "--family qwen3 --mode fsdp --lora embedding",
    "--family qwen3 --mode fsdp --lora dora",
    "--family qwen3 --mode fsdp --lora off",
    "--family qwen3 --mode presharded --lora off",
    "--family qwen3 --mode tp --lora off",
    "--family bert --mode fsdp --head",
    "--family bert --mode fsdp --lora off --head",
    "--family bert --mode ddp --head",
    "--family qwen3 --mode tp --lora off --head",
)
_EMBEDDING_CORE_1GPU_ROWS = (
    "--family bert --mode single",
    "--family qwen3 --mode single",
    "--family qwen3 --mode single --lora embedding",
    "--family qwen3 --mode single --lora dora",
    "--family qwen3 --mode single --lora off",
)

MANIFEST: dict[str, TestSpec] = {
    "data/test_cache_isolation.py": TestSpec(nproc=2, markers=("gpu", "core", "2gpu"), timeout=600),
    "data/test_coordinated_processing.py": TestSpec(nproc=2, markers=("gpu", "core", "2gpu"), timeout=600),
    "data/test_padding_free_sft.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "tp", "ep", "moe", "qwen3"),
        args_matrix=(
            Row("--mode dense", markers=("gpu", "full", "2gpu", "qwen3")),
            Row("--mode tp2", markers=("gpu", "full", "2gpu", "tp", "qwen3")),
            Row("--mode ep2", markers=("gpu", "full", "2gpu", "ep", "moe", "qwen3")),
        ),
        timeout=900,
    ),
    "data/test_mrope_padding_free.py": TestSpec(
        nproc=1,
        markers=("gpu", "full", "1gpu", "qwen3"),
        args_matrix=(
            "--family qwen3_5 --attn flash_attention_2",
            "--family qwen3_5 --attn flash_attention_4",
            "--family qwen3_vl --attn flash_attention_2",
            "--family qwen3_vl --attn flash_attention_4",
        ),
        timeout=900,
    ),
    "data/test_packing_isolation.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=600),
    "data/test_family_packing_isolation.py": TestSpec(
        nproc=1,
        markers=("gpu", "core", "1gpu", "moe"),
        args_matrix=(
            *(
                Row(f"--family {family}{attn}", markers=("gpu", "core", "1gpu", "moe", marker))
                for family, marker, attn in (
                    ("bailing_moe", "bailing", ""),
                    ("bailing_moe", "bailing", " --attn flash_attention_2"),
                    ("bailing_moe_linear", "bailing", ""),
                    ("bailing_moe_linear", "bailing", " --attn flash_attention_2"),
                    ("zaya", "zaya", ""),
                    ("zaya", "zaya", " --attn flash_attention_2"),
                    ("zaya", "zaya", " --attn sdpa"),
                    ("qwen4_exp_text", "qwen3", " --attn sdpa"),
                    ("nemotron_h", "nemotron", " --attn sdpa"),
                    ("nemotron_h", "nemotron", " --attn flash_attention_2"),
                    ("nemotron_h", "nemotron", " --attn flash_attention_4"),
                )
            ),
            *(
                Row(f"--family {family}", markers=("gpu", "core", "1gpu", "moe", _TINY_MOE_FAMILY_MARKERS[family]))
                for family in _PACKED_DEFAULT_FAMILIES
            ),
            Row("--family qwen3", markers=("gpu", "core", "1gpu", "qwen3")),
            Row("--family qwen3_5", markers=("gpu", "core", "1gpu", "qwen3")),
            Row("--family qwen3_5_moe", markers=("gpu", "core", "1gpu", "moe", "qwen3")),
            Row("--family gemma4", markers=("gpu", "core", "1gpu", "moe", "gemma4")),
            Row("--family nemotron_h_omni", markers=("gpu", "core", "1gpu", "moe", "nemotron")),
            *(
                Row(
                    f"--family {family} --ep 2",
                    nproc=2,
                    markers=("gpu", "core", "2gpu", "ep", "moe", _TINY_MOE_FAMILY_MARKERS[family]),
                )
                for family in _PACKED_EP_FAMILIES
            ),
            *(
                Row(
                    f"--family {family} --attn {attn}",
                    markers=("gpu", "full", "1gpu", "moe", _TINY_MOE_FAMILY_MARKERS[family]),
                )
                for family, attn in (
                    ("cohere2_moe", "flash_attention_2"),
                    ("gpt_oss", "flash_attention_2"),
                    ("laguna", "flash_attention_2"),
                    ("lfm2_moe", "flash_attention_2"),
                    ("mistral4", "flash_attention_2"),
                    ("qwen3_moe", "flash_attention_2"),
                    ("qwen3_5_moe_text", "flash_attention_4"),
                    ("qwen3_next", "flash_attention_4"),
                )
            ),
            Row("--family qwen3 --attn flash_attention_2", markers=("gpu", "full", "1gpu", "qwen3")),
            Row("--family qwen3_5 --attn flash_attention_4", markers=("gpu", "full", "1gpu", "qwen3")),
            *(
                Row(f"--family {family} --ep 2", nproc=2, markers=("gpu", "full", "2gpu", "ep", "moe", marker))
                for family, marker in (
                    *((family, _TINY_MOE_FAMILY_MARKERS[family]) for family in _PACKED_EP_FULL_FAMILIES),
                    ("laguna_remote", "laguna"),
                    ("gemma4", "gemma4"),
                    ("nemotron_h_omni", "nemotron"),
                    ("qwen3_5_moe", "qwen3"),
                )
            ),
            Row(
                "--family qwen4_exp_text --attn sdpa --ep 2",
                nproc=2,
                markers=("gpu", "full", "2gpu", "ep", "moe", "qwen3"),
            ),
            *(Row(f"--family {family}", markers=("gpu", "full", "1gpu")) for family in ("llama", "mistral", "qwen2")),
            Row("--family laguna_remote", markers=("gpu", "full", "1gpu", "moe", "laguna")),
            Row("--family laguna_remote --liger", markers=("gpu", "full", "1gpu", "moe", "laguna")),
            *(
                Row(
                    f"--family {family} --liger",
                    markers=("gpu", "full", "1gpu", "moe", _TINY_MOE_FAMILY_MARKERS[family]),
                )
                for family in ("bailing_moe", "zaya", *_PACKED_DEFAULT_FAMILIES)
            ),
            Row("--family qwen3 --liger", markers=("gpu", "full", "1gpu", "qwen3")),
            Row("--family qwen3_5_moe --liger", markers=("gpu", "full", "1gpu", "moe", "qwen3")),
            Row("--family gemma4 --liger", markers=("gpu", "full", "1gpu", "moe", "gemma4")),
            Row("--family nemotron_h_omni --liger", markers=("gpu", "full", "1gpu", "moe", "nemotron")),
            Row("--family qwen3_5 --liger", markers=("gpu", "full", "1gpu", "qwen3")),
            *(
                Row(
                    f"--family {family} --ep 2 --liger",
                    nproc=2,
                    markers=("gpu", "full", "2gpu", "ep", "moe", _TINY_MOE_FAMILY_MARKERS[family]),
                )
                for family in _PACKED_EP_FAMILIES
            ),
        ),
        timeout=900,
    ),
    # 1800s: the FA4 varlen backward JITs per segment shape, and packed batch>1 rows never converge to a cached set.
    "data/test_packing_batch_gt1.py": TestSpec(nproc=2, markers=("gpu", "core", "2gpu"), timeout=1800),
    "data/test_sft_caching_e2e.py": TestSpec(nproc=2, markers=("gpu", "core", "2gpu"), timeout=600),
    "data/test_sharded_distributed_load.py": TestSpec(nproc=2, markers=("gpu", "core", "2gpu"), timeout=600),
    # Two `halo launch sft` runs plus the CPU row-count probe, with room for the FA4 split-form warm-up.
    "data/test_tree_packing_e2e.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "qwen3"), timeout=1200, launcher="python"
    ),
    "kernels/test_chunked_logprob_precision.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=600),
    "kernels/test_deepgemm.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=600),
    "kernels/test_fa4_trainable_sink_rescale.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=600),
    "kernels/test_nemotron_mamba_mixer.py": TestSpec(
        nproc=1, markers=("gpu", "core", "1gpu", "nemotron"), timeout=900
    ),
    # 900s: FA4 JIT-compiles the 16:1 and head_dim-80 kernels per form, forward and backward, causal and flipped.
    "kernels/test_nemotron_attention_kernels.py": TestSpec(
        nproc=1, markers=("gpu", "core", "1gpu", "nemotron"), timeout=900
    ),
    # 1800s: FA4 JIT-compiles the head_dim-256 forward and backward per shape and layout, dense and varlen.
    "kernels/test_fa4_hd256_strided_inputs.py": TestSpec(
        nproc=1, markers=("gpu", "core", "1gpu", "qwen3", "glm4"), timeout=1800
    ),
    # One launch per family and emission: an illegal address poisons the process. The markers row of each family is
    # the emission the packed GRPO forward uses; position_ids and flash_varlen are negative controls that must leak.
    "kernels/test_packed_rl_forward_mrope.py": TestSpec(
        nproc=1,
        markers=("gpu", "core", "1gpu", "qwen3"),
        args_matrix=(
            "--family qwen3_5 --emission markers",
            "--family qwen3_5 --emission position_ids",
            "--family qwen3_5 --emission flash_varlen",
            "--family qwen3_5 --emission mrope_plane",
            Row("--family qwen3_5_moe_text --emission markers", markers=("gpu", "core", "1gpu", "moe", "qwen3")),
            Row("--family qwen3_5_composite --emission markers", markers=("gpu", "core", "1gpu", "qwen3")),
            Row("--family qwen3_5_moe_composite --emission markers", markers=("gpu", "core", "1gpu", "moe", "qwen3")),
            Row("--family qwen3_next --emission markers", markers=("gpu", "core", "1gpu", "moe", "qwen3")),
            Row("--family qwen3_next --emission flash_varlen", markers=("gpu", "full", "1gpu", "moe", "qwen3")),
            Row("--family qwen3_moe --emission markers", markers=("gpu", "core", "1gpu", "moe", "qwen3")),
            Row("--family llama --emission markers", markers=("gpu", "core", "1gpu")),
            Row("--family nemotron_h --emission markers", markers=("gpu", "full", "1gpu", "moe", "nemotron")),
            Row("--family lfm2_moe --emission markers", markers=("gpu", "full", "1gpu", "moe", "lfm2")),
            Row("--family gpt_oss --emission markers", markers=("gpu", "full", "1gpu", "moe", "gptoss")),
        ),
    ),
    "kernels/test_fa4_hd256_family_parity.py": TestSpec(
        nproc=1,
        markers=("gpu", "core", "1gpu", "qwen3"),
        args_matrix=(
            "--family qwen3_5",
            Row("--family qwen3_5_moe_text", markers=("gpu", "core", "1gpu", "moe", "qwen3")),
            Row("--family qwen3_next", markers=("gpu", "core", "1gpu", "moe", "qwen3")),
            Row("--family glm4_moe_lite", markers=("gpu", "core", "1gpu", "moe", "glm4")),
            Row("--family cohere2_moe", markers=("gpu", "full", "1gpu", "moe", "cohere2_moe")),
            Row("--family laguna", markers=("gpu", "full", "1gpu", "moe", "laguna")),
            Row("--family lfm2_moe", markers=("gpu", "full", "1gpu", "moe", "lfm2")),
            Row("--family mistral4", markers=("gpu", "full", "1gpu", "moe", "mistral4")),
            Row("--family qwen2", markers=("gpu", "full", "1gpu")),
            Row("--family gemma", markers=("gpu", "full", "1gpu")),
            Row("--family llama", markers=("gpu", "full", "1gpu")),
            Row("--family mistral", markers=("gpu", "full", "1gpu")),
            Row("--family gemma3_text_hd128", markers=("gpu", "full", "1gpu")),
        ),
        timeout=1200,
    ),
    "kernels/test_fa4_tree_attention.py": TestSpec(
        nproc=1, markers=("gpu", "core", "1gpu", "moe", "gptoss", "qwen3"), timeout=1500
    ),
    # 900s: FA4 JIT-compiles forward and backward per form.
    "kernels/test_fa4_gqa_ratio.py": TestSpec(
        nproc=1,
        markers=("gpu", "core", "1gpu"),
        args_matrix=(
            Row("--heads 32 --kv-heads 2", markers=("gpu", "core", "1gpu", "nemotron")),
            "--heads 64 --kv-heads 4",
            "--heads 16 --kv-heads 1",
        ),
        timeout=900,
    ),
    "kernels/test_tree_sparse_attention.py": TestSpec(
        nproc=1, markers=("gpu", "core", "1gpu", "moe", "glm5"), timeout=900
    ),
    "kernels/test_masked_matmul_attention.py": TestSpec(
        nproc=1, markers=("gpu", "core", "1gpu", "moe", "glm5"), timeout=900
    ),
    "kernels/test_sdpa_flex_sliding.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=900),
    "kernels/test_reward_scorer_flex_sliding.py": TestSpec(
        nproc=1, markers=("gpu", "core", "1gpu", "gemma4"), timeout=900
    ),
    "kernels/test_softcap_window_attention.py": TestSpec(
        nproc=1,
        markers=("gpu", "core", "1gpu"),
        args_matrix=("--family gemma2", "--family gemma2_hd128", "--family gemma3_text"),
        timeout=600,
    ),
    # 900s: every warmed geometry JITs its FA4 forward and backward and its FlexAttention graphs once.
    "kernels/test_attention_warmup_per_layer_heads.py": TestSpec(
        nproc=1, markers=("gpu", "core", "1gpu", "moe", "step3p7", "laguna"), timeout=900
    ),
    "kernels/test_flex_masked_layers.py": TestSpec(
        nproc=1, markers=("gpu", "core", "1gpu", "moe", "step3p7", "gemma4"), timeout=900
    ),
    "kernels/test_fused_glu.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=600),
    "kernels/test_moe_permute.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu", "moe"), timeout=600),
    "kernels/test_hyper_connection_kernels.py": TestSpec(
        nproc=1, markers=("gpu", "core", "1gpu", "moe", "deepseek_v4", "glm5"), timeout=900
    ),
    "kernels/test_family_kernel_stack_numerics.py": TestSpec(
        nproc=1,
        markers=("gpu", "full", "1gpu", "moe", *_family_markers(_TINY_MOE_FAMILIES)),
        args_matrix=tuple(f"--family {family}" for family in _TINY_MOE_FAMILIES),
        timeout=900,
    ),
    "kernels/test_expert_gemm.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=600),
    "kernels/test_grouped_mm_empty_groups.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu", "moe"), timeout=600),
    "kernels/test_liger_family_kernels.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=900),
    "kernels/test_liger_routed_experts.py": TestSpec(
        nproc=1, markers=("gpu", "core", "1gpu", "moe", "qwen3"), timeout=600
    ),
    "kernels/test_liger_rope_repeated_calls.py": TestSpec(
        nproc=1, markers=("gpu", "core", "1gpu", "qwen3"), timeout=300
    ),
    "kernels/test_fla_blackwell_autotune.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu", "qwen3"), timeout=600),
    "kernels/test_fla_simple_gla.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu", "bailing"), timeout=600),
    "kernels/test_fake_quant_triton.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=600),
    "kernels/test_deepgemm_cast_triton.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=600),
    "kernels/test_lowp_expert_lora.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu", "lora", "moe"), timeout=600),
    "kernels/test_packed_broadcast_memory.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=600),
    "kernels/test_lowp_ep_trainer.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "ep", "moe", "qwen3"),
        args_matrix=(
            "--precision fp8",
            "--precision fp4",
            "--precision mxfp4",
            "--precision fp8 --native",
            "--precision mxfp4 --native",
            "--precision fp8 --native --backward fp8",
            "--precision mxfp4 --native --backward fp8",
        ),
        timeout=900,
    ),
    "kernels/test_lowp_fsdp2_weight_cache.py": TestSpec(nproc=2, markers=("gpu", "core", "2gpu"), timeout=600),
    "kernels/test_lowp_production.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=600),
    "kernels/test_weight_sync_param_buffer_cuda.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=300),
    "kernels/test_weight_sync_stage_once.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=300),
    "kernels/test_weight_sync_stream_safety.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=300),
    # Skips without the local Hub snapshot. The cut model holds ~20 GB of experts; each export or resume checkpoint is
    # ~125 GB, ~100 GB of it the n-gram table, written once and hard-linked after.
    "models/test_qwen4_exp_real_checkpoint.py": TestSpec(
        nproc=8,
        markers=("gpu", "full", "8gpu", "ep", "moe", "qwen3"),
        args_matrix=("--mode layers", "--mode export", "--mode resume"),
        timeout=5400,
    ),
    # The cut's weights, or 241 GB per export at --layers 88 (Super), where rank 0's CPU reload outlasts the NCCL watchdog.
    "models/test_nemotron_h_real_checkpoint.py": TestSpec(
        nproc=8,
        markers=("gpu", "full", "8gpu", "ep", "moe", "nemotron"),
        args_matrix=(
            "--mode layers",
            "--mode export",
            "--mode export --route composite",
            "--mode resume",
            "--mode profile --seq 8192",
            "--release lightning --mode layers",
            "--release lightning --mode export",
            "--release lightning --mode resume",
            "--release lightning --mode profile --seq 8192",
        ),
        timeout=5400,
    ),
    "optimizers/test_adamw_bf16.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=600),
    # Pure FSDP2, so no parallelism-axis marker.
    "optimizers/test_adamw_bf16_fused.py": TestSpec(nproc=2, markers=("gpu", "core", "2gpu"), timeout=600),
    "optimizers/test_bf16_optimizer_ep.py": TestSpec(
        nproc=8, markers=("gpu", "full", "8gpu", "ep", "moe", "gptoss"), timeout=1000
    ),
    "optimizers/test_flash_adamw.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=600),
    "optimizers/test_lowp_master_feasibility.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=600),
    "optimizers/test_muon_parallel.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "ep", "tp", "etp", "moe", "qwen3"),
        args_matrix=("--mode ep2", "--mode tp2", "--mode etp2"),
        timeout=900,
    ),
    "optimizers/test_flash_adamw_parallel.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "ep", "moe", "qwen3"),
        args_matrix=("--mode ep2", "--mode fsdp", "--mode uneven"),
        timeout=900,
    ),
    "optimizers/test_muon.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=600),
    "optimizers/test_muon_fsdp.py": TestSpec(nproc=2, markers=("gpu", "full", "2gpu"), timeout=1000),
    "optimizers/test_muon_fsdp_qwen35.py": TestSpec(nproc=2, markers=("gpu", "full", "2gpu", "qwen3"), timeout=1000),
    "parallelism/combined/test_ep_cp_correctness.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "cp", "moe", "gptoss"), timeout=600
    ),
    "parallelism/combined/test_ep_cp_save_reload_roundtrip.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "ep", "cp", "moe", "gptoss"), timeout=1500
    ),
    "parallelism/combined/test_ep_cp_train_correctness.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "ep", "cp", "moe", "gptoss"), timeout=1000
    ),
    "parallelism/combined/test_qwen3_moe_cp_grpo_scoring.py": TestSpec(
        nproc=8, markers=("gpu", "full", "8gpu", "ep", "cp", "moe", "qwen3"), timeout=1200
    ),
    "parallelism/combined/test_combined_ref_correctness.py": TestSpec(
        nproc=4,
        markers=("gpu", "core", "4gpu", "ep", "tp", "etp", "moe", "mistral4"),
        # EP/TP/ETP feed the full sequence, so these need no CP aggregation against the single-GPU reference.
        args_matrix=(
            "--mode ep --ep 4",
            "--mode ep --ep 2",
            "--mode tp --tp 4",
            "--mode tp --tp 2",
            "--mode etp --etp 4",
            "--mode etp --etp 2",
            "--mode ep_tp --ep 2 --tp 2",
            "--mode ep_etp --ep 2 --etp 2",
        ),
        timeout=1200,
    ),
    "parallelism/combined/test_ep_sharded_save_tp_sinks.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "tp", "moe", "gptoss"), timeout=600
    ),
    "parallelism/combined/test_ep_etp_combo_correctness.py": TestSpec(
        nproc=4, markers=("gpu", "full", "4gpu", "ep", "etp", "moe", "gptoss"), timeout=1000
    ),
    "parallelism/combined/test_gptoss_etp_grouped_gemm.py": TestSpec(
        nproc=8, markers=("gpu", "full", "8gpu", "ep", "etp", "moe", "gptoss"), timeout=1500
    ),
    "parallelism/combined/test_ep_etp_fused_glu_correctness.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "etp", "mistral4"), timeout=600
    ),
    "parallelism/combined/test_ep_tp_replicated_grad_sync.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "ep", "tp", "moe", "gptoss"), timeout=900
    ),
    "parallelism/cp/test_cp_correctness.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "cp", "qwen3"), timeout=600
    ),
    "parallelism/cp/test_cp_grpo_scoring.py": TestSpec(
        nproc=4, markers=("gpu", "full", "4gpu", "cp", "qwen3"), timeout=900
    ),
    "parallelism/cp/test_cp_smpo_logprobs.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "cp", "qwen3"), timeout=600
    ),
    "parallelism/cp/test_cp_train_correctness.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "cp", "qwen3"), timeout=1000
    ),
    "parallelism/cp/test_cp_vlm.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "vlm", "cp", "qwen3"), timeout=1000
    ),
    "parallelism/cp/test_cp_family_vs_reference.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "cp", "qwen3"), args_matrix=_CP_ROSTER_ROWS, timeout=900
    ),
    "parallelism/cp/test_cp_fused_head_loss.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "cp", "qwen3"), args_matrix=_CP_ROSTER_ROWS, timeout=900
    ),
    "parallelism/ep/test_ep1_fsdp_shard_experts.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "moe", "gptoss"), timeout=1500
    ),
    "parallelism/ep/test_ep1_knob_weight_sync.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "moe", "gptoss"), timeout=1500
    ),
    "parallelism/ep/test_ep1_weight_sync_names.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "moe", "gptoss"), timeout=1500
    ),
    "parallelism/ep/test_ep_expert_gather_to_forwarder.py": TestSpec(
        nproc=4, markers=("gpu", "core", "4gpu", "ep", "moe"), timeout=600
    ),
    "parallelism/ep/test_ep_buffer_backends.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "moe", "gptoss"), timeout=1500
    ),
    "parallelism/ep/test_ep_shared_arena.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "moe", "qwen3"), timeout=1800
    ),
    "parallelism/ep/test_ep_buffer_gc_safety.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "moe"), timeout=300
    ),
    # Eleven fresh-interpreter lazy-load probes on top of the in-process checks.
    "parallelism/ep/test_ep_sort_sync_free.py": TestSpec(
        nproc=1,
        markers=("gpu", "core", "1gpu", "ep", "moe", *_family_markers(("gpt_oss", "qwen3_5_moe_text", "qwen3_moe"))),
        timeout=1500,
    ),
    "parallelism/ep/test_ep_receive_counts.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "ep", "moe", *_family_markers(_TINY_MOE_FAMILIES)),
        args_matrix=(
            "--backend elastic",
            "--backend legacy",
            "--backend elastic --no-grouped-gemm --expect-path loop",
            Row(
                "--backend elastic --family gpt_oss --expert-tp-size 2",
                nproc=4,
                markers=("gpu", "core", "4gpu", "ep", "etp", "moe", "gptoss"),
                timeout=600,
            ),
            Row(
                "--backend elastic --ep-size 8",
                nproc=8,
                markers=("gpu", "full", "8gpu", "ep", "moe", *_family_markers(_TINY_MOE_FAMILIES)),
            ),
        ),
        timeout=1200,
    ),
    "parallelism/ep/test_deepep_config_ranks_drift.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "moe"), timeout=300
    ),
    "parallelism/ep/test_qwen3_moe_bias_balancing.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "moe", "qwen3"), timeout=1800
    ),
    "parallelism/ep/test_ep_long_context.py": TestSpec(
        nproc=8, markers=("gpu", "full", "8gpu", "ep", "moe", "gptoss"), timeout=1200
    ),
    # Each row loads its checkpoint twice: the rank-0 reference, then the parallel model.
    "parallelism/ep/test_ep_modes_vs_reference.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "ep", "tp", "etp", "moe", "gptoss", "qwen3"),
        args_matrix=(
            "--model gpt_oss_20b --mode ep",
            "--model gpt_oss_20b --mode ep_tp",
            "--model gpt_oss_20b --mode etp",
            "--model qwen3_30b_a3b --mode ep",
            Row(
                "--model gpt_oss_20b --mode etp_dp2", nproc=4, markers=("gpu", "full", "4gpu", "etp", "moe", "gptoss")
            ),
        ),
        timeout=1200,
    ),
    "parallelism/ep/test_ep2_weight_sync_values.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "moe", "gptoss"), timeout=1200
    ),
    "parallelism/ep/test_etp_weight_sync.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "etp", "moe", "gptoss"), timeout=600
    ),
    # The failure mode is a hang (a rank fails reading the main process's cache file while its peer waits in a
    # collective), so the timeout is the assertion; 420s covers the sweep.
    "parallelism/ep/test_ep_preference_precompute.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "ep", "moe", "qwen3"),
        args_matrix=("--trainer dpo", "--trainer kto"),
        timeout=420,
    ),
    # The refusal phase raises on every rank together, so a rank-local raise surfaces as a hang against the timeout.
    "parallelism/ep/test_ep_preference_precompute_resume.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "ep", "etp", "tp", "lora", "moe", "qwen3"),
        args_matrix=(
            *_PRECOMPUTE_CORE_ROWS,
            *_rows(
                _PRECOMPUTE_FAMILY_ROWS,
                markers=(
                    "gpu",
                    "full",
                    "2gpu",
                    "ep",
                    "etp",
                    "tp",
                    "moe",
                    *_family_markers(_PRECOMPUTE_SWEEP_FAMILIES),
                ),
            ),
            *_rows(
                (f"--trainer dpo --family {family} --mode ep2etp2" for family in _TINY_MOE_FAMILIES),
                nproc=4,
                markers=("gpu", "full", "4gpu", "ep", "etp", "moe", *_family_markers(_TINY_MOE_FAMILIES)),
            ),
            *(
                Row(f"--trainer {trainer} --family {family} --mode {mode}", nproc=nproc, markers=markers)
                for trainer in ("dpo", "kto")
                for family, mode, nproc, markers in (
                    ("dense", "dp2_zero3", 2, ("gpu", "full", "2gpu", "qwen3")),
                    ("dense", "hsdp", 4, ("gpu", "full", "4gpu", "hsdp", "qwen3")),
                    ("dense", "tp2dp2", 4, ("gpu", "full", "4gpu", "tp", "qwen3")),
                    ("qwen3_moe", "ep1_zero3", 2, ("gpu", "full", "2gpu", "ep", "moe", "qwen3")),
                    ("qwen3_moe", "ep1_unsharded", 2, ("gpu", "full", "2gpu", "ep", "moe", "qwen3")),
                    ("qwen3_moe", "ep2dp2", 4, ("gpu", "full", "4gpu", "ep", "moe", "qwen3")),
                    ("qwen3_moe", "ep2x2dom", 4, ("gpu", "full", "4gpu", "ep", "moe", "qwen3")),
                    ("qwen3_moe", "tp2dp2", 4, ("gpu", "full", "4gpu", "tp", "moe", "qwen3")),
                )
            ),
            # Attention LoRA under expert TP on the families whose projections the index cannot name.
            *(
                Row(f"--trainer dpo --family {family} --mode {mode} --peft", nproc=nproc, markers=markers)
                for family, mode, nproc, markers in (
                    ("deepseek_v4", "etp2", 2, ("gpu", "full", "2gpu", "etp", "lora", "moe", "deepseek_v4")),
                    ("zaya", "etp2", 2, ("gpu", "full", "2gpu", "etp", "lora", "moe", "zaya")),
                    ("bailing_moe", "ep2etp2", 4, ("gpu", "full", "4gpu", "ep", "etp", "lora", "moe", "bailing")),
                )
            ),
        ),
        timeout=900,
    ),
    "parallelism/ep/test_ep_pooled_head_trainers.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "ep", "moe", "qwen3"),
        args_matrix=(
            "--trainer reward",
            "--trainer classification",
            *(
                Row(
                    f"--trainer {trainer} --mode {mode}",
                    nproc=nproc,
                    markers=("gpu", "full", f"{nproc}gpu", *axes, "moe", "qwen3"),
                )
                for mode, nproc, axes in (
                    ("tp2", 2, ("tp",)),
                    ("etp2", 2, ("etp",)),
                    ("ep1", 2, ("ep",)),
                    ("ep1_zero3", 2, ("ep",)),
                    ("ep1_unsharded", 2, ("ep",)),
                    ("ep2tp2", 2, ("ep", "tp")),
                    ("ep2etp2", 4, ("ep", "etp")),
                    ("tp2dp2", 4, ("tp",)),
                    ("ep2dp2", 4, ("ep",)),
                    ("ep2x2dom", 4, ("ep",)),
                )
                for trainer in ("reward", "classification")
            ),
        ),
        timeout=900,
    ),
    # The ep1 row's FSDP2 DTensor experts are a different shard layout from the ep row's plain FSDP-ignored tensors.
    "parallelism/ep/test_ep_optimizer_resume.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "ep", "cp", "tp", "moe", "qwen3"),
        args_matrix=(
            "--mode ep",
            "--mode ep1",
            "--mode cp",
            "--mode ep --fp32-masters",
            "--mode ep_cp --fp32-masters",
            "--mode ep --fp32-masters --eager-loading",
            "--mode ep1 --fp32-masters --unsharded-ep1-experts",
            "--mode cp --fp32-masters",
            "--mode fsdp --fp32-masters",
            "--mode tp --fp32-masters",
            "--mode fsdp --fp32-masters --fused-adamw",
            "--mode ep --fp32-masters --fused-adamw",
        ),
        timeout=1200,
    ),
    "parallelism/ep/test_ep_replay_cache.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "moe", "gptoss"), timeout=1200
    ),
    "parallelism/ep/test_ep_save_reload_roundtrip.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "ep", "moe"), timeout=1500
    ),
    "parallelism/ep/test_ep_sharded_merge_roundtrip.py": TestSpec(
        nproc=2,
        markers=(
            "gpu",
            "core",
            "2gpu",
            "ep",
            "moe",
            *_family_markers(
                ("gpt_oss", "qwen3_moe", "qwen3_5_moe_text", "laguna", "cohere2_moe", "deepseek_v4", "glm5_next")
            ),
        ),
        timeout=900,
    ),
    "parallelism/ep/test_ep_striped_save.py": TestSpec(
        nproc=2,
        markers=(
            "gpu",
            "core",
            "2gpu",
            "ep",
            "moe",
            *_family_markers(
                ("gpt_oss", "qwen3_moe", "qwen3_5_moe_text", "laguna", "cohere2_moe", "deepseek_v4", "glm5_next")
            ),
        ),
        timeout=900,
    ),
    "parallelism/ep/test_routing_replay.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "moe", "gptoss"), timeout=900
    ),
    "parallelism/ep/test_ep_deterministic_expert_grads.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "ep", "etp", "moe", "gptoss", "qwen3"),
        args_matrix=(
            *_DETERMINISM_CORE_ROWS,
            *_rows(
                (f"--family {family}" for family in _DETERMINISM_SWEEP_FAMILIES),
                markers=("gpu", "full", "2gpu", "ep", "moe", *_family_markers(_DETERMINISM_SWEEP_FAMILIES)),
            ),
        ),
        timeout=600,
    ),
    "parallelism/ep/test_ep_expert_divide_zero_token.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "moe", "qwen3"), timeout=900
    ),
    "parallelism/ep/test_ep_expert_only_rank_uniform_graph.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "ep", "moe", "lora", "gptoss", "qwen3", "zaya"),
        timeout=600,
        args_matrix=("--family qwen3_moe", "--family gpt_oss", "--family zaya"),
    ),
    # The --expert-tp-size 2 rows are EP+ETP on four ranks; launched by hand on two they run pure ETP.
    "parallelism/ep/test_ep_family_vs_reference.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "ep", "moe", *_family_markers(_TINY_MOE_FAMILIES)),
        args_matrix=(
            *(f"--family {family}" for family in _TINY_MOE_FAMILIES),
            *(
                Row(
                    f"--family {family} --expert-tp-size 2",
                    nproc=4,
                    markers=("gpu", "full", "4gpu", "ep", "etp", "moe", _TINY_MOE_FAMILY_MARKERS[family]),
                    timeout=900,
                )
                for family in ("gpt_oss", "inkling_text", "nemotron_h", "qwen4_exp_text")
            ),
            # No latent, the shape of Nemotron-H's Lightning release, on the wire and through the expert split.
            Row("--family nemotron_h --no-latent", markers=("gpu", "core", "2gpu", "ep", "moe", "nemotron")),
            Row(
                "--family nemotron_h --no-latent --expert-tp-size 2",
                nproc=4,
                markers=("gpu", "full", "4gpu", "ep", "etp", "moe", "nemotron"),
                timeout=900,
            ),
        ),
        timeout=1500,
    ),
    "parallelism/ep/test_ep_chunked_head_path.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "ep", "moe", "nemotron"),
        timeout=600,
        args_matrix=("--family nemotron_h", "--family nemotron_h_omni"),
    ),
    "parallelism/ep/test_ep_vlm_composites.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "ep", "vlm", "moe", "qwen3", "gemma4", "nemotron"),
        args_matrix=(
            Row("--family qwen3_5_moe", markers=("gpu", "full", "2gpu", "ep", "vlm", "moe", "qwen3")),
            Row("--family gemma4", markers=("gpu", "full", "2gpu", "ep", "vlm", "moe", "gemma4")),
            Row("--family nemotron_h_omni", markers=("gpu", "full", "2gpu", "ep", "vlm", "moe", "nemotron")),
        ),
        timeout=900,
    ),
    "parallelism/ep/test_ep_vlm_inkling.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "ep", "moe", "vlm", "inkling"), timeout=900
    ),
    "parallelism/ep/test_lazy_load_inkling.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "ep", "moe", "inkling"), timeout=900
    ),
    # A family joins this matrix when transformers' conversion is its own to replay: ``hub_conversion_keys``, or
    # Nemotron-H's level drop and per-expert banks.
    "parallelism/ep/test_lazy_load_converted.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "ep", "moe", "glm5", "nemotron", "step3p7"),
        timeout=900,
        args_matrix=("--family glm5_next", "--family nemotron_h_omni", "--family step3p7"),
    ),
    # Three loads of gpt-oss-20b: the rank-0 reference, the no-EP load on every rank, the EP model.
    "parallelism/ep/test_mok_backend.py": TestSpec(
        nproc=4,
        markers=("gpu", "core", "4gpu", "ep", "moe", "qwen3"),
        args_matrix=("--family qwen3", "--family qwen3_5", "--family qwen3 --mxfp8", "--family qwen3_5 --mxfp8"),
        timeout=900,
    ),
    "parallelism/ep/test_ep_vs_no_ep.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "ep", "moe", "gptoss"), timeout=1500
    ),
    "parallelism/ep/test_gptoss_bias_balancing.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "moe", "gptoss"), timeout=1500
    ),
    "parallelism/ep/test_gptoss_expert_bias_grad.py": TestSpec(
        nproc=1, markers=("gpu", "core", "1gpu", "ep", "moe", "gptoss"), timeout=300
    ),
    "parallelism/ep/test_ep_gc_bias_balancing.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "moe", "gptoss"), timeout=900
    ),
    "parallelism/ep/test_ep_gc_router_aux_loss.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "ep", "moe", "gptoss", "qwen3"),
        timeout=600,
        args_matrix=("--family gpt_oss", "--family qwen3_moe"),
    ),
    "parallelism/ep/test_grouped_mm_b300.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=600),
    "parallelism/ep/test_zaya_ep.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "ep", "moe", "zaya"), timeout=1500
    ),
    "parallelism/ep/test_zaya_ep_save_roundtrip.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "moe", "zaya"), timeout=1500
    ),
    "parallelism/ep/test_zaya_ep_discard_dispatch.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "moe", "zaya"), timeout=600
    ),
    "parallelism/pp/test_pp_sft_trainer_e2e.py": TestSpec(
        nproc=4, markers=("gpu", "core", "4gpu", "pp", "qwen3"), timeout=1200
    ),
    "parallelism/pp/test_pp_packing_pad_tail.py": TestSpec(
        nproc=4, markers=("gpu", "core", "4gpu", "pp", "qwen3"), timeout=1200
    ),
    # FA2 except the two FA4 rows, which skip off Blackwell.
    "parallelism/pp/test_pp_packed_flash_varlen.py": TestSpec(
        nproc=4,
        markers=("gpu", "core", "4gpu", "pp", "qwen3"),
        timeout=1500,
        args_matrix=(
            "",
            "--microbatches 4 --gc --fused",
            "--schedule gpipe --microbatches 4",
            "--attn flash_attention_4",
            "--trainer --gc",
            Row("--trainer --ep 2 --gc", markers=("gpu", "full", "4gpu", "pp", "ep", "moe", "gptoss")),
            Row(
                "--trainer --ep 2 --gc --attn flash_attention_4",
                markers=("gpu", "full", "4gpu", "pp", "ep", "moe", "gptoss"),
            ),
        ),
    ),
    "parallelism/pp/test_pp_vs_single_gpu.py": TestSpec(
        nproc=4,
        markers=("gpu", "core", "4gpu", "pp", "qwen3"),
        timeout=900,
        args_matrix=("", "--gc", "--schedule gpipe", "--pp_split 5,3"),
    ),
    "parallelism/pp/test_pp_forward_only.py": TestSpec(
        nproc=4, markers=("gpu", "core", "4gpu", "pp", "qwen3"), timeout=900
    ),
    "parallelism/pp/test_pp_fused_head_loss.py": TestSpec(
        nproc=4, markers=("gpu", "core", "4gpu", "pp", "qwen3"), timeout=1200, args_matrix=("", "--gc")
    ),
    # Both ranks belong to one stage (the EP group under --ep 2); each walks both pp_ranks of a virtual pp2 pipeline.
    "parallelism/pp/test_pp_lazy_load_equivalence.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "pp", "ep", "moe", "qwen3", "gptoss"),
        timeout=1200,
        args_matrix=(
            "",
            "--pp_split 5,3",
            "--ep 2",
            "--head seq_cls",
            "--ep 2 --moe_family qwen3_moe",
            "--ep 2 --fp32_masters",
            "--ep 2 --moe_family qwen3_moe --fp32_masters",
        ),
    ),
    # Each node drives a different mask path, so the matrix must run whole. The FA4 row skips off Blackwell.
    "parallelism/pp/test_pp_ragged_mask_vs_single_gpu.py": TestSpec(
        nproc=4,
        markers=("gpu", "core", "4gpu", "pp", "qwen3"),
        timeout=900,
        args_matrix=("", "--attn sdpa", "--positions", "--attn flash_attention_4", "--family qwen3_5"),
    ),
    "parallelism/pp/test_pp_ep_vs_unsplit.py": TestSpec(
        nproc=4,
        markers=("gpu", "full", "4gpu", "pp", "ep", "moe", "gptoss"),
        timeout=1500,
        args_matrix=("", "--gc"),
    ),
    # World 8 so the deferred replica sweep engages (two ETP groups per stage, dp2). fp32: a bf16 twin is not a valid
    # ETP gate.
    "parallelism/pp/test_pp_etp_vs_unsplit.py": TestSpec(
        nproc=8, markers=("gpu", "full", "8gpu", "pp", "etp", "moe", "gptoss"), timeout=1500
    ),
    "parallelism/pp/test_pp_family_vs_reference.py": TestSpec(
        nproc=4,
        markers=("gpu", "full", "4gpu", "pp", "ep", "etp", "moe", *_family_markers(_PP_MOE_FAMILIES)),
        timeout=1500,
        args_matrix=(
            Row("--family qwen3_5_moe_text --ep-size 1", markers=("gpu", "full", "4gpu", "pp", "moe", "qwen3")),
            *(
                Row(
                    f"--family {family} --ep-size 2",
                    markers=("gpu", "full", "4gpu", "pp", "ep", "moe", _TINY_MOE_FAMILY_MARKERS[family]),
                )
                for family in _PP_MOE_FAMILIES
            ),
            *(
                Row(
                    f"--family {family} --ep-size 1 --etp-size 2",
                    markers=("gpu", "full", "4gpu", "pp", "etp", "moe", _TINY_MOE_FAMILY_MARKERS[family]),
                )
                for family in _PP_MOE_FAMILIES
            ),
            # A composite release read from disk through its text class, which no row above builds from a checkpoint.
            Row(
                "--family qwen3_5_moe --text-only --ep-size 1",
                markers=("gpu", "full", "4gpu", "pp", "moe", "vlm", "qwen3"),
            ),
            Row(
                "--family qwen3_5_moe --text-only --ep-size 2",
                markers=("gpu", "full", "4gpu", "pp", "ep", "moe", "vlm", "qwen3"),
            ),
            Row(
                "--family qwen3_5_moe --text-only --ep-size 1 --etp-size 2",
                markers=("gpu", "full", "4gpu", "pp", "etp", "moe", "vlm", "qwen3"),
            ),
            Row(
                "--family qwen4_exp --text-only --ep-size 2",
                markers=("gpu", "full", "4gpu", "pp", "ep", "moe", "vlm", "qwen3"),
            ),
            Row(
                "--family nemotron_h_omni --text-only --ep-size 2",
                markers=("gpu", "full", "4gpu", "pp", "ep", "moe", "vlm", "nemotron"),
            ),
        ),
    ),
    "parallelism/pp/test_pp_ep_gc_equivalence.py": TestSpec(
        nproc=4, markers=("gpu", "core", "4gpu", "pp", "ep", "moe", "gptoss"), timeout=1800
    ),
    "parallelism/pp/test_pp_ep1_gmm_vs_unsplit.py": TestSpec(
        nproc=4, markers=("gpu", "full", "4gpu", "pp", "ep", "moe", "gptoss"), timeout=1500
    ),
    "parallelism/pp/test_pp_ep_sft_trainer_e2e.py": TestSpec(
        nproc=4, markers=("gpu", "full", "4gpu", "pp", "ep", "moe", "gptoss"), timeout=1500
    ),
    "parallelism/pp/test_pp_smpo.py": TestSpec(nproc=4, markers=("gpu", "full", "4gpu", "pp", "qwen3"), timeout=1500),
    "parallelism/pp/test_pp_reward_bt.py": TestSpec(
        nproc=4, markers=("gpu", "full", "4gpu", "pp", "qwen3"), timeout=1200
    ),
    "parallelism/pp/test_pp_classification.py": TestSpec(
        nproc=4, markers=("gpu", "full", "4gpu", "pp", "qwen3"), timeout=1500
    ),
    "parallelism/pp/test_pp_eval_metrics.py": TestSpec(
        nproc=4, markers=("gpu", "core", "4gpu", "pp", "qwen3"), timeout=900
    ),
    "parallelism/pp/test_pp_gptoss_fa2_sinks_save.py": TestSpec(
        nproc=4, markers=("gpu", "full", "4gpu", "pp", "moe", "gptoss"), timeout=1200
    ),
    "parallelism/pp/test_pp_resume.py": TestSpec(
        nproc=4, markers=("gpu", "full", "4gpu", "pp", "qwen3"), timeout=1500
    ),
    "parallelism/pp/test_pp_ep_resume.py": TestSpec(
        nproc=4,
        markers=("gpu", "full", "4gpu", "pp", "ep", "etp", "moe", "gptoss"),
        timeout=1500,
        args_matrix=("--mode ep2", "--mode etp2"),
    ),
    "parallelism/pp/test_pp_trainer_checkpoints.py": TestSpec(
        nproc=4,
        markers=("gpu", "full", "4gpu", "pp", "qwen3"),
        timeout=1200,
        args_matrix=tuple(
            f"--trainer {trainer}"
            for trainer in ("smpo", "dpo", "kto", "reward", "classification", "offline_grpo", "offline_grpo_kl")
        ),
    ),
    "parallelism/pp/test_pp_offline_grpo.py": TestSpec(
        nproc=4, markers=("gpu", "full", "4gpu", "pp", "qwen3"), timeout=1500
    ),
    "parallelism/pp/test_pp_offline_grpo_kl.py": TestSpec(
        nproc=4, markers=("gpu", "full", "4gpu", "pp", "qwen3"), timeout=1500
    ),
    "parallelism/pp/test_pp_text_only_vlm.py": TestSpec(
        nproc=4,
        markers=("gpu", "core", "4gpu", "pp", "moe", "glm5", "step3p7", "vlm"),
        timeout=1200,
        args_matrix=("--family glm5_next", "--family step3p7"),
    ),
    "parallelism/pp/test_pp_preference_precompute.py": TestSpec(
        nproc=4,
        markers=("gpu", "full", "4gpu", "pp", "qwen3"),
        timeout=1500,
        args_matrix=("--trainer dpo", "--trainer kto"),
    ),
    "parallelism/pp/test_pp_combos_8gpu.py": TestSpec(
        nproc=8,
        markers=("gpu", "full", "8gpu", "pp", "ep", "moe", "gptoss", "qwen3"),
        args_matrix=(
            "--mode dense_pp2_dp4",
            "--mode ep4",
            "--mode pp4_ep2",
            "--mode ep2_dp2",
            "--mode ep1_dp4",
        ),
        timeout=1500,
    ),
    "parallelism/pp/test_pp_inkling_text_only.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "pp", "ep", "moe", "inkling"), timeout=900
    ),
    "parallelism/pp/test_pp_qwen4_exp_vs_reference.py": TestSpec(
        nproc=4,
        markers=("gpu", "full", "4gpu", "pp", "ep", "etp", "moe", "qwen3"),
        timeout=1500,
        args_matrix=(
            "--mode ep2",
            "--mode etp2",
            "--mode packed",
            "--mode gc",
            "--family qwen3_5_moe_text --mode packed",
        ),
    ),
    "parallelism/pp/test_pp_qwen4_exp_resume.py": TestSpec(
        nproc=4,
        markers=("gpu", "full", "4gpu", "pp", "ep", "etp", "moe", "qwen3"),
        timeout=1500,
        args_matrix=("--mode ep2", "--mode etp2"),
    ),
    "parallelism/pp/test_pp_hc_vs_single_gpu.py": TestSpec(
        nproc=4,
        markers=("gpu", "core", "4gpu", "pp", "moe", "deepseek_v4", "glm5"),
        timeout=900,
        args_matrix=("--family deepseek_v4", "--family glm5_next"),
    ),
    "parallelism/test_fsdp_tied_embeddings.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "qwen3"), timeout=600
    ),
    "parallelism/test_fsdp2_reentrant_recompute.py": TestSpec(
        nproc=2,
        markers=(
            "gpu",
            "full",
            "2gpu",
            "ep",
            "etp",
            "tp",
            "hsdp",
            "moe",
            "lora",
            *_family_markers(_TINY_MOE_FAMILIES),
        ),
        timeout=900,
        args_matrix=(
            *(
                Row(
                    f"--family {family} --layout ep1 --adapter {adapter}{zero3}",
                    markers=(
                        "gpu",
                        "core" if (family, adapter, zero3) == ("qwen3_moe", "full", "") else "full",
                        "2gpu",
                        "ep",
                        "moe",
                        "lora",
                        _TINY_MOE_FAMILY_MARKERS[family],
                    ),
                )
                for family in _REENTRANT_FAMILIES
                for adapter in ("full", "lora")
                for zero3 in ("", " --reshard-after-forward")
            ),
            Row(
                "--family qwen3_moe --layout ep1 --fp32-router",
                markers=("gpu", "full", "2gpu", "ep", "moe", "qwen3"),
            ),
            *(
                Row(
                    f"--family {family} --layout {layout} --adapter {adapter}",
                    nproc=4,
                    markers=("gpu", "full", "4gpu", "ep", "moe", "lora", _TINY_MOE_FAMILY_MARKERS[family]),
                )
                for family in _REENTRANT_FAMILIES
                for layout in ("ep_single", "ep_multi")
                for adapter in ("full", "lora")
            ),
            *(
                Row(
                    f"--family {family} --layout {layout} --adapter full",
                    nproc=nproc,
                    markers=("gpu", "full", f"{nproc}gpu", "ep", "moe", _TINY_MOE_FAMILY_MARKERS[family]),
                )
                for family in _REENTRANT_SWEEP_FAMILIES
                for layout, nproc in (("ep1", 2), ("ep_multi", 4))
            ),
            *(
                Row(
                    f"--family {family} --layout {layout}",
                    nproc=nproc,
                    markers=("gpu", "full", f"{nproc}gpu", *axes, "moe", _TINY_MOE_FAMILY_MARKERS[family]),
                )
                for family in ("qwen3_moe", "gpt_oss")
                for layout, nproc, axes in (
                    ("etp", 2, ("etp",)),
                    ("ep_etp", 4, ("ep", "etp")),
                    ("ep_tp", 4, ("ep", "tp")),
                    ("tp_dp", 4, ("tp",)),
                    ("hsdp", 4, ("hsdp",)),
                )
            ),
        ),
    ),
    # EP+TP needs a single dispatch group: ep4+tp2 on 8 forms two 4-rank groups, which ParallelismConfig rejects.
    # Node-local EP+CP pins ep_size to the 8-GPU domain; Mistral4 runs EP+CP at ep2+cp2 in the merged-resume family
    # rows.
    "parallelism/test_moe_all_parallelism.py": TestSpec(
        nproc=8,
        markers=("gpu", "full", "8gpu", "ep", "cp", "tp", "etp", "moe", "cohere2_moe", "mistral4"),
        args_matrix=(
            "--family mistral4 --mode ep --ep 8 --liger",
            "--family mistral4 --mode cp --cp 8",
            "--family mistral4 --mode tp --tp 8",
            "--family mistral4 --mode ep_tp --ep 8 --tp 2",
            "--family mistral4 --mode ep_etp --ep 2 --etp 4",
            "--family cohere2_moe --mode ep --ep 8",
            "--family cohere2_moe --mode cp --cp 8",
            "--family cohere2_moe --mode tp --tp 8",
            "--family cohere2_moe --mode ep_cp --ep 8 --cp 2",
            "--family cohere2_moe --mode ep_tp --ep 8 --tp 2",
            "--family cohere2_moe --mode etp --etp 8",
            "--family cohere2_moe --mode ep_etp --ep 2 --etp 4",
        ),
        timeout=2100,
    ),
    "parallelism/tp/test_replay_mask_tp_broadcast.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "tp"), timeout=300
    ),
    # nproc=2 is pure TP. At nproc=4 (TP+DP) FSDP2 hides the plain-slice/replica distinction the TP grad-norm
    # classification depends on.
    "parallelism/tp/test_tp_family_vs_reference.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "tp"),
        args_matrix=(
            *(
                f"--family {family} --mode tp2"
                for family in ("cohere2", "gemma2", "gemma3_text", "llama", "mistral", "olmo3", "qwen2")
            ),
            Row("--family qwen3_5 --mode tp2", markers=("gpu", "full", "2gpu", "tp", "qwen3")),
            *(
                Row(
                    f"--family {family} --mode {mode}",
                    markers=(
                        "gpu",
                        "full",
                        "2gpu",
                        "tp",
                        *(("ep",) if mode == "ep2tp2" else ()),
                        "moe",
                        _TINY_MOE_FAMILY_MARKERS[family],
                    ),
                )
                for family in _TP_FAMILIES
                for mode in ("tp2", "ep2tp2")
            ),
        ),
        timeout=900,
    ),
    "parallelism/tp/test_tp_correctness.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "tp", "qwen3"), timeout=600
    ),
    "parallelism/tp/test_tp_attention_norm_grad.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "tp", "moe", "qwen3"), timeout=600
    ),
    "parallelism/tp/test_tp_dp_correctness.py": TestSpec(
        nproc=4, markers=("gpu", "core", "4gpu", "tp", "qwen3"), timeout=900
    ),
    "parallelism/tp/test_tp_gathered_save_sinks.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "tp", "gptoss"), timeout=600
    ),
    "trainers/grpo/test_environmental_grpo_benchmarks.py": TestSpec(
        nproc=1, markers=("gpu", "full", "1gpu", "qwen3", "vllm_server"), timeout=600
    ),
    # The packed window against the per-row one on FA2; the --moe row adds DeepEP dispatch per bin and R2/R3 replay.
    "trainers/grpo/test_env_packed_microbatches.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "qwen3"),
        args_matrix=("", Row("--moe", markers=("gpu", "core", "2gpu", "ep", "moe", "qwen3"))),
        timeout=900,
    ),
    # Online GRPO: TRL's whole-round old-policy pass planned and scored bin by bin, then the packed window against the
    # per-row one on FA2.
    "trainers/grpo/test_online_packed_microbatches.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "qwen3"), timeout=900
    ),
    # Online GRPO rank balance: the deal over NCCL evens the ranks' tokens and the dealt window trains as the undealt
    # one does.
    "trainers/grpo/test_online_rank_balance.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "qwen3"), timeout=900
    ),
    # The IS weight and KL clamp from the loss forward against the no-grad pass's window; --packed trains packed bins and
    # the --moe row adds DeepEP dispatch and R3 replay with no pass to arm the masks.
    "trainers/grpo/test_loss_forward_weights_modes.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "qwen3"),
        args_matrix=(
            "",
            "--packed",
            Row("--moe", markers=("gpu", "core", "2gpu", "ep", "moe", "qwen3")),
        ),
        timeout=900,
    ),
    "trainers/grpo/test_env_length_ordered_rows.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "qwen3"),
        args_matrix=(
            "",
            Row("--timed", markers=("gpu", "full", "2gpu", "qwen3")),
            Row("--moe", markers=("gpu", "core", "2gpu", "ep", "moe", "qwen3")),
        ),
        timeout=900,
    ),
    "trainers/grpo/test_row_spans_sync_free.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=300),
    "trainers/grpo/test_sglang_weight_sync_e2e.py": TestSpec(
        nproc=1, markers=("gpu", "full", "1gpu", "qwen3", "sglang_server"), timeout=900
    ),
    # Serves its own tiny hub-layout checkpoint (``--write-checkpoint``, HALO_TEST_STEP3P7_MODEL), not either
    # SERVER_TIER model; the script docstring has the server launch.
    "trainers/grpo/test_step3p7_vllm_weight_sync_e2e.py": TestSpec(
        nproc=1, markers=("gpu", "full", "1gpu", "ep", "moe", "step3p7", "vllm_server"), timeout=1200
    ),
    "trainers/grpo/test_weight_push_exact.py": TestSpec(
        nproc=2,
        markers=(
            "gpu",
            "full",
            "2gpu",
            "ep",
            "tp",
            "moe",
            *dict.fromkeys(_PUSH_EXACT_MOE_FAMILY_MARKERS.values()),
            "vllm_server",
            "sglang_server",
        ),
        args_matrix=(*_PUSH_EXACT_MOE_ROWS, *_PUSH_EXACT_REACH_ROWS, *_PUSH_EXACT_DENSE_ROWS),
        timeout=1800,
    ),
    # Every row asserts the served policy changed, so the server must run the row's checkpoint: on vLLM Qwen3-30B on
    # two ranks and gpt-oss on four (SERVER_TIER='moe and not gptoss' / 'moe and gptoss'), on SGLang gpt-oss.
    # HALO_TEST_ENV_GRPO_MODEL / HALO_TEST_ENV_GRPO_SGLANG_MODEL point the two-rank rows at another family; a family
    # pass runs -k "not peft and not resume and not routing and not prefetch and not zero3 and not unsharded and not
    # world and not servers". --thinking-budget is not a row: that gpt-oss shape needs the image's reasoning plugin
    # (agent-docs/models/gpt-oss.md#serving-for-grpo-vllm). --routing-replay rollout rows need the server on VLLM_ENABLE_R3=1,
    # or SGLANG_ENABLE_R3=1 with SGLANG_MOE_RUNNER_BACKEND=triton, since SGLang's fused runners bypass the capture
    # hook; the flag is additive, so one server carrying it runs every row. The SGLang --tp-size 2 and --resume R3 rows
    # produce runaway completions after the sync, the zero-gradient batch the replay gate exempts.
    "trainers/grpo/test_env_grpo_e2e.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "ep", "etp", "tp", "lora", "moe", "qwen3", "vllm_server"),
        args_matrix=(
            *(
                f"--backend vllm {args}"
                for args in (
                    "--ep-size 1",
                    "--ep-size 2",
                    "--tp-size 2",
                    "--ep-size 1 --peft lora",
                    "--ep-size 2 --peft lora",
                    "--ep-size 2 --peft expert_lora",
                    "--etp-size 2 --peft lora",
                    "--tp-size 2 --peft lora",
                    "--ep-size 1 --peft lora --resume",
                    "--ep-size 2 --peft expert_lora --resume",
                    "--ep-size 2 --resume",
                    "--ep-size 1 --routing-replay rollout",
                    "--ep-size 2 --routing-replay rollout",
                    "--ep-size 1 --routing-replay recompute",
                    "--ep-size 2 --routing-replay recompute",
                    "--ep-size 2 --prefetch-depth 1",
                    "--ep-size 2 --prefetch-depth 2",
                    "--tp-size 2 --prefetch-depth 2",
                    "--etp-size 2 --prefetch-depth 2",
                    "--ep-size 2 --routing-replay rollout --prefetch-depth 1",
                    "--ep-size 1 --routing-replay rollout --chunked",
                    "--ep-size 1 --routing-replay rollout --chunked --length-ordered false",
                    "--ep-size 2 --routing-replay rollout --chunked",
                    "--ep-size 2 --routing-replay rollout --chunked --length-ordered false",
                    "--ep-size 2 --chunked --phase-times",
                    "--tp-size 2 --chunked --phase-times",
                    "--ep-size 1 --zero3",
                    "--ep-size 1 --zero3 --routing-replay rollout --chunked",
                    "--ep-size 1 --unsharded-ep1-experts",
                )
            ),
            # Two vLLM servers of the row's checkpoint in HALO_TEST_VLLM_SERVER_URLS; skips with fewer.
            Row(
                "--backend vllm --ep-size 2 --servers 2 --prefetch-depth 1",
                markers=("gpu", "full", "2gpu", "ep", "moe", "qwen3", "vllm_server"),
            ),
            *_rows(
                (f"--backend vllm {args}" for args in _ENV_GRPO_4GPU_ROWS),
                nproc=4,
                markers=("gpu", "full", "4gpu", "ep", "etp", "tp", "lora", "moe", "gptoss", "vllm_server"),
                timeout=1800,
            ),
            *_rows(
                (
                    f"--backend vllm {args}"
                    for args in (
                        "--tp-size 2 --world 4",
                        "--ep-size 2 --world 4",
                        "--ep-size 2 --world 4 --gpus-per-node 2",
                        "--world 4 --hsdp",
                        "--ep-size 4 --chunked --prefetch-depth 1 --fp32-router --fp32-experts",
                    )
                ),
                nproc=4,
                markers=("gpu", "full", "4gpu", "ep", "tp", "hsdp", "moe", "gptoss", "vllm_server"),
                timeout=1800,
            ),
            *_rows(
                (
                    f"--backend sglang {args}"
                    for args in (
                        "--ep-size 1",
                        "--ep-size 2",
                        "--tp-size 2",
                        "--ep-size 1 --peft lora",
                        "--ep-size 1 --resume",
                        "--ep-size 1 --peft lora --resume",
                        "--ep-size 1 --routing-replay rollout",
                        "--ep-size 1 --routing-replay recompute",
                        "--ep-size 1 --peft lora --routing-replay rollout",
                        "--tp-size 2 --routing-replay rollout",
                        "--ep-size 1 --resume --routing-replay rollout",
                        "--ep-size 2 --peft expert_lora",
                        "--ep-size 2 --peft expert_lora --resume",
                        "--ep-size 1 --prefetch-depth 1",
                        "--ep-size 1 --routing-replay rollout --prefetch-depth 1",
                        "--ep-size 1 --routing-replay rollout --chunked",
                        "--ep-size 1 --routing-replay rollout --chunked --length-ordered false",
                        "--ep-size 1 --zero3",
                    )
                ),
                markers=("gpu", "full", "2gpu", "ep", "tp", "lora", "moe", "gptoss", "sglang_server"),
                flaky=True,
            ),
            *_rows(
                (f"--backend sglang {args}" for args in _ENV_GRPO_4GPU_ROWS),
                nproc=4,
                markers=("gpu", "full", "4gpu", "ep", "etp", "tp", "lora", "moe", "gptoss", "sglang_server"),
                timeout=1800,
            ),
        ),
        timeout=2400,
    ),
    "trainers/grpo/test_offline_grpo.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "qwen3"),
        args_matrix=("", Row("--hsdp", nproc=4, markers=("gpu", "full", "4gpu", "hsdp", "qwen3"))),
        timeout=600,
    ),
    "trainers/grpo/test_offline_grpo_bnpo.py": TestSpec(
        # FSDP then TP=2 in one process; 900s covers the one-time FA2 compile + both modes + evals.
        nproc=2,
        markers=("gpu", "core", "2gpu", "tp", "qwen3"),
        timeout=900,
    ),
    "trainers/grpo/test_offline_grpo_bs4.py": TestSpec(nproc=2, markers=("gpu", "core", "2gpu", "qwen3"), timeout=600),
    "trainers/grpo/test_offline_grpo_chunked.py": TestSpec(
        # FA4 dense + varlen kernels JIT once each before the parity check; 900s covers both compiles.
        nproc=2,
        markers=("gpu", "core", "2gpu", "qwen3"),
        timeout=900,
    ),
    "trainers/grpo/test_offline_grpo_cp_resume.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "cp", "ep", "moe", "qwen3", *_family_markers(_ULYSSES_MOE_FAMILIES)),
        args_matrix=(
            "--cp-size 1",
            "--cp-size 2",
            *(f"--cp-size 2 --family {family}" for family in _ULYSSES_MOE_FAMILIES),
            *(f"--cp-size 2 --ep-size 2 --family {family} --fp32-masters" for family in _ULYSSES_MOE_FAMILIES),
            "--cp-size 1 --zero3",
            "--cp-size 1 --family qwen3_moe --zero3",
            "--cp-size 1 --family qwen3_moe --unsharded-ep1-experts",
            *_rows(
                (
                    "--cp-size 2 --world 4",
                    "--cp-size 1 --family qwen3_moe --ep-size 2 --world 4",
                    "--cp-size 1 --family qwen3_moe --ep-size 2 --world 4 --gpus-per-node 2",
                ),
                nproc=4,
                markers=("gpu", "full", "4gpu", "cp", "ep", "moe", "qwen3"),
            ),
        ),
        timeout=2400,
    ),
    "trainers/grpo/test_offline_grpo_ep_cp.py": TestSpec(
        nproc=8,
        markers=("gpu", "full", "8gpu", "ep", "cp", "moe", "qwen3"),
        args_matrix=("--ep-loading lazy", "--ep-loading eager"),
        timeout=1800,
    ),
    "trainers/grpo/test_offline_grpo_expert_lora_kl.py": TestSpec(
        nproc=8, markers=("gpu", "full", "8gpu", "ep", "moe", "lora", "qwen3"), timeout=1800
    ),
    "trainers/grpo/test_offline_grpo_reference.py": TestSpec(
        nproc=8,
        markers=("gpu", "full", "8gpu", "ep", "moe", "qwen3"),
        args_matrix=(
            "--mode ep8 --ep-loading lazy",
            "--mode ep8 --ep-loading eager",
            *_rows(
                ("--mode tp", "--mode etp", "--mode ep_tp", "--mode ep_etp"),
                nproc=4,
                markers=("gpu", "full", "4gpu", "ep", "tp", "etp", "moe", "qwen3"),
            ),
        ),
        timeout=1800,
    ),
    "trainers/grpo/test_offline_grpo_row_positions.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "ep", "moe"),
        timeout=600,
        args_matrix=(
            "--family deepseek_v4 --gc off",
            "--family deepseek_v4 --gc reentrant",
            "--family mistral4 --gc off",
            "--family mistral4 --gc reentrant",
        ),
    ),
    "trainers/grpo/test_offline_grpo_tp_resume.py": TestSpec(
        # 2400s: the FA4 backward JITs per shape on Blackwell, and this resume trains 6 steps across a reload.
        nproc=2,
        markers=("gpu", "core", "2gpu", "tp", "qwen3"),
        timeout=2400,
    ),
    # Each row asserts logprobs only its own checkpoint's server can produce: VLLM_SERVER_URL serves
    # Qwen/Qwen3-30B-A3B-Instruct-2507 (MoE rows), HALO_TEST_VLLM_DENSE_SERVER_URL Qwen/Qwen3-0.6B (dense and
    # one-process rows). --servers 2 rows skip without a second server in HALO_TEST_VLLM_SERVER_URLS. --resume rows run
    # two trainers in one process and load the policy twice, hence the longest timeouts of their policy.
    "trainers/grpo/test_online_grpo_vllm_e2e.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "lora", "ep", "etp", "moe", "qwen3", "vllm_server"),
        args_matrix=(
            "--trainer online --mode full_ep2",
            "--trainer sdpg --mode full_ep2",
            "--trainer online --mode lora_ep2",
            "--trainer sdpg --mode lora_ep2",
            "--trainer online --mode expert_lora_ep2",
            "--trainer sdpg --mode expert_lora_ep2",
            "--trainer online --mode lora_etp2",
            "--trainer sdpg --mode lora_etp2",
            "--trainer online --mode expert_lora_ep2 --resume",
            "--trainer sdpg --mode expert_lora_ep2 --resume",
            "--trainer sdpg --mode full_ep2 --resume",
            *(
                f"--trainer {trainer} --mode {mode}"
                for mode in ("full_ep1", "full_ep1_unsharded")
                for trainer in ("online", "sdpg")
            ),
            *_rows(
                ("--trainer online --mode full_ep2_tp2", "--trainer sdpg --mode full_ep2_tp2"),
                markers=("gpu", "full", "2gpu", "ep", "tp", "moe", "qwen3", "vllm_server"),
            ),
            *_rows(
                (
                    f"--trainer {trainer} --mode {mode}"
                    for mode in ("full_ep2_dp2", "full_ep2x2dom", "full_ep2_etp2")
                    for trainer in ("online", "sdpg")
                ),
                nproc=4,
                markers=("gpu", "full", "4gpu", "ep", "etp", "moe", "qwen3", "vllm_server"),
            ),
            *_rows(
                (
                    f"--trainer {trainer} --mode {mode}"
                    for mode in ("full_tp2_dp2", "full_hsdp")
                    for trainer in ("online", "sdpg")
                ),
                nproc=4,
                markers=("gpu", "full", "4gpu", "tp", "hsdp", "qwen3", "vllm_server"),
                timeout=900,
            ),
            *_rows(
                (
                    "--trainer online --mode full_tp2",
                    "--trainer sdpg --mode full_tp2",
                    "--trainer online --mode lora_fsdp",
                    "--trainer sdpg --mode lora_fsdp",
                    "--trainer online --mode lora_tp2_rejected",
                    "--trainer sdpg --mode lora_tp2_rejected",
                    "--trainer online --mode full_fsdp_kl",
                    "--trainer online --mode full_fsdp_zero3",
                    "--trainer sdpg --mode full_fsdp_zero3",
                    "--trainer online --mode full_tp2 --resume",
                    "--trainer sdpg --mode full_tp2 --resume",
                    "--trainer online --mode lora_fsdp --resume",
                    "--trainer sdpg --mode lora_fsdp --resume",
                ),
                markers=("gpu", "full", "2gpu", "lora", "tp", "qwen3", "vllm_server"),
                timeout=900,
            ),
            *_rows(
                (
                    "--trainer online --mode full_world1",
                    "--trainer sdpg --mode full_world1",
                    "--trainer online --mode lora_world1",
                    "--trainer environmental --mode full_world1",
                    "--trainer environmental --mode lora_world1",
                    "--trainer environmental --mode full_world1 --servers 2",
                    "--trainer environmental --mode full_world1 --servers 2 --prefetch-depth 1",
                    "--trainer environmental --mode full_world1 --servers 2 --prefetch-depth 2",
                ),
                nproc=1,
                markers=("gpu", "full", "1gpu", "lora", "qwen3", "vllm_server"),
                timeout=3000,
                flaky=True,
            ),
        ),
        timeout=1800,
    ),
    # A dozen connect/sync/disconnect cycles on one group port, the reuse that lets the rows above share one server.
    "trainers/grpo/test_weight_transfer_reinit.py": TestSpec(
        nproc=1,
        markers=("gpu", "full", "1gpu", "qwen3", "vllm_server"),
        args_matrix=(
            "--backend vllm",
            Row("--backend sglang", markers=("gpu", "full", "1gpu", "qwen3", "sglang_server")),
        ),
        timeout=600,
    ),
    # SGLang's single_group topology ships only if the empty-declare probe tells a registered group from an unregistered one.
    "trainers/grpo/test_sglang_membership_probe.py": TestSpec(
        nproc=1, markers=("gpu", "full", "1gpu", "qwen3", "sglang_server"), timeout=300
    ),
    # One group over 1, 2, 4 and 8 servers per engine (the URL lists are HALO_TEST_{VLLM,SGLANG}_SERVER_URLS); the time
    # bar compares the N-server push with a one-server group.
    "trainers/grpo/test_single_group_sync.py": TestSpec(
        nproc=1,
        markers=("gpu", "full", "1gpu", "qwen3", "vllm_server"),
        args_matrix=(
            "--backend vllm --servers 1",
            "--backend vllm --servers 2",
            "--backend vllm --servers 4",
            "--backend vllm --servers 8",
            Row("--backend sglang --servers 1", markers=("gpu", "full", "1gpu", "qwen3", "sglang_server")),
            Row("--backend sglang --servers 2", markers=("gpu", "full", "1gpu", "qwen3", "sglang_server")),
            Row("--backend sglang --servers 4", markers=("gpu", "full", "1gpu", "qwen3", "sglang_server")),
            Row("--backend sglang --servers 8", markers=("gpu", "full", "1gpu", "qwen3", "sglang_server")),
        ),
        timeout=1800,
    ),
    # rollout_adapter_sync: the base once, the adapter alone after training, engine log-probs against the trainer's and a
    # merged push's. The server runs Qwen3-0.6B with SGLANG_LORA_MAX_RANK=16.
    "trainers/grpo/test_adapter_sync_e2e.py": TestSpec(
        nproc=1, markers=("gpu", "full", "1gpu", "qwen3", "sglang_server"), timeout=900
    ),
    # auto resolves to one group for two or more identical servers and to per-server groups for one.
    "trainers/grpo/test_sync_topology_auto.py": TestSpec(
        nproc=1,
        markers=("gpu", "full", "1gpu", "qwen3", "vllm_server"),
        args_matrix=(
            "--backend vllm --servers 1",
            "--backend vllm --servers 2",
            "--backend vllm --servers 4",
            "--backend vllm --servers 8",
            Row("--backend sglang --servers 1", markers=("gpu", "full", "1gpu", "qwen3", "sglang_server")),
            Row("--backend sglang --servers 2", markers=("gpu", "full", "1gpu", "qwen3", "sglang_server")),
            Row("--backend sglang --servers 4", markers=("gpu", "full", "1gpu", "qwen3", "sglang_server")),
            Row("--backend sglang --servers 8", markers=("gpu", "full", "1gpu", "qwen3", "sglang_server")),
        ),
        timeout=1800,
    ),
    # Restarts and kills servers through HALO_TEST_SYNC_SERVER_CONTROL; skips without it.
    "trainers/grpo/test_single_group_failure.py": TestSpec(
        nproc=1,
        markers=("gpu", "full", "1gpu", "qwen3", "vllm_server"),
        args_matrix=(
            "--backend vllm --servers 2",
            Row("--backend sglang --servers 2", markers=("gpu", "full", "1gpu", "qwen3", "sglang_server")),
        ),
        # Two server restarts and a kill dominate; a wedged teardown runs past this instead of parking the harness.
        timeout=1800,
    ),
    "trainers/grpo/test_weight_sync_pool_reconnect.py": TestSpec(
        nproc=1, markers=("gpu", "full", "1gpu", "qwen3", "vllm_server"), timeout=600
    ),
    "trainers/lora/test_embedding_lora_resume.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "lora", "tp", "qwen3"),
        args_matrix=(
            *_EMBEDDING_CORE_ROWS,
            *_rows(_EMBEDDING_CORE_1GPU_ROWS, nproc=1, markers=("gpu", "core", "1gpu", "lora", "qwen3"), timeout=600),
            *_rows(
                (f"--family {family} --mode tpdp --lora off" for family in _EMBEDDING_TP_FAMILIES),
                nproc=4,
                markers=("gpu", "full", "4gpu", "tp", "moe", "qwen3", "gptoss"),
            ),
            *_rows(
                (
                    "--family qwen3 --mode zero3",
                    "--family qwen3 --mode zero3 --lora off",
                    "--family gpt_oss --mode zero3 --lora off",
                    "--family gpt_oss --mode ep1",
                    "--family gpt_oss --mode ep1 --lora off",
                    "--family gpt_oss --mode ep1 --lora off --gradient-checkpointing",
                    "--family gemma4 --mode ep1 --lora off",
                    "--family gpt_oss --mode ep1_unsharded",
                    "--family gpt_oss --mode ep1_unsharded --lora off",
                    "--family gpt_oss --mode etp",
                    "--family gpt_oss --mode etp --lora off",
                    "--family gpt_oss --mode eptp --lora off",
                ),
                markers=("gpu", "full", "2gpu", "lora", "ep", "etp", "tp", "moe", "qwen3", "gemma4", "gptoss"),
                timeout=1200,
            ),
            *_rows(
                ("--family gpt_oss --mode ep1_single --lora off",),
                nproc=1,
                markers=("gpu", "full", "1gpu", "lora", "moe", "gptoss"),
                timeout=1200,
            ),
            *_rows(
                (
                    "--family qwen3 --mode hsdp",
                    "--family qwen3 --mode hsdp --lora off",
                    "--family gpt_oss --mode hsdp",
                    "--family gpt_oss --mode epetp --lora off",
                    "--family gpt_oss --mode ep2dp2 --lora off",
                    "--family gpt_oss --mode ep2x2dom --lora off",
                ),
                nproc=4,
                markers=("gpu", "full", "4gpu", "lora", "ep", "etp", "hsdp", "moe", "qwen3", "gptoss"),
                timeout=1200,
            ),
            *_rows(
                (
                    row
                    for lora, flag in _EMBEDDING_LORA_FLAGS.items()
                    for family in _EMBEDDING_FAMILIES
                    for mode in (
                        "fsdp",
                        "ddp",
                        "presharded",
                        *(("tp",) if lora == "off" and family in _EMBEDDING_TP_FAMILIES else ()),
                        *(
                            ("ep",)
                            if lora in ("attention", "embedding", "off") and family in ("gemma4", "gpt_oss")
                            else ()
                        ),
                    )
                    if (row := f"--family {family} --mode {mode}{flag}") not in _EMBEDDING_CORE_ROWS
                ),
                markers=("gpu", "full", "2gpu", "lora", "tp", "ep", "moe", "qwen3", "gemma4", "gptoss"),
                timeout=1200,
            ),
            *_rows(
                (
                    row
                    for flag in _EMBEDDING_LORA_FLAGS.values()
                    for family in _EMBEDDING_FAMILIES
                    if (row := f"--family {family} --mode single{flag}") not in _EMBEDDING_CORE_1GPU_ROWS
                ),
                nproc=1,
                markers=("gpu", "full", "1gpu", "lora", "moe", "qwen3", "gemma4", "gptoss"),
            ),
        ),
        timeout=900,
    ),
    "trainers/lora/test_lora_ep_experts.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "lora", "ep", "moe", "gptoss"), timeout=1200
    ),
    "trainers/lora/test_lora_ep_experts_idle_rank.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "lora", "ep", "moe", "zaya"),
        timeout=600,
        args_matrix=("--idle all", "--idle first"),
    ),
    "trainers/lora/test_lora_ep_router_modules_to_save.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "lora", "ep", "moe", "gptoss"), timeout=1200
    ),
    "trainers/lora/test_lora_ep_experts_resume.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "lora", "ep", "moe", "gptoss"), timeout=1800
    ),
    "trainers/lora/test_lora_ep1_fsdp_experts_resume.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "lora", "ep", "moe", "gptoss"), timeout=1800
    ),
    # `full`: writes a whole merged checkpoint (40+ GB at the default GptOss-20B).
    "trainers/lora/test_lora_mixed_merged_save.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "lora", "ep", "moe", "gptoss"), timeout=2400
    ),
    "trainers/lora/test_lora_merged_save_resume.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "lora", "ep", "cp", "moe", "qwen3", "gptoss"),
        args_matrix=(
            *_MERGED_RESUME_CORE_ROWS,
            *_rows(
                _MERGED_RESUME_FAMILY_ROWS,
                markers=("gpu", "full", "2gpu", "lora", "ep", "cp", "moe", *_family_markers(_TINY_MOE_FAMILIES)),
            ),
        ),
        timeout=1200,
    ),
    # tests/cpu/conventions/test_tiny_family_roster.py holds the rows to the syncable tiny-family roster.
    "trainers/lora/test_lora_weight_sync_exact.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "lora", "ep", "etp", "moe", "qwen3"),
        args_matrix=(
            "--family qwen3 --mode fsdp",
            "--family qwen3_moe --mode ep1 --adapters peft",
            "--family qwen3_moe --mode ep1 --adapters mixed",
            "--family qwen3_moe --mode ep2 --adapters peft",
            "--family qwen3_moe --mode ep2 --adapters mixed",
            "--family qwen3_moe --mode etp2 --adapters peft",
            *_rows(
                (
                    "--family qwen3_5 --mode fsdp",
                    *(
                        f"--family {family} --mode {shape}"
                        for family in _SYNC_EXACTNESS_SWEEP_FAMILIES
                        for shape in (
                            "ep1 --adapters peft",
                            "ep1 --adapters mixed",
                            "ep2 --adapters peft",
                            "ep2 --adapters mixed",
                            "etp2 --adapters peft",
                        )
                    ),
                ),
                # qwen3 also marks the dense Qwen3.5 row.
                markers=(
                    "gpu",
                    "full",
                    "2gpu",
                    "lora",
                    "ep",
                    "etp",
                    "moe",
                    *_family_markers(_SYNC_EXACTNESS_SWEEP_FAMILIES),
                ),
            ),
        ),
        timeout=900,
    ),
    # Real checkpoints, so `full`.
    "trainers/lora/test_lora_sft_modes.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "lora", "ep", "cp", "etp", "moe", "gptoss", "qwen3"),
        args_matrix=tuple(f"--model {model} --mode {row}" for model, rows in _LORA_SFT_ROWS.items() for row in rows),
        timeout=1500,
    ),
    "trainers/lora/test_lora_sft_bailing.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "lora", "ep", "etp", "moe", "bailing"),
        args_matrix=("--mode ep", "--mode etp"),
        timeout=1500,
    ),
    "trainers/lora/test_lora_offline_grpo.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "lora", "ep", "etp", "moe", "gptoss", "qwen3"),
        args_matrix=(
            "--mode lora",
            "--mode qlora",
            "--mode lora_ep",
            "--mode expert_lora",
            "--mode lora_etp",
        ),
        timeout=1500,
    ),
    # 1800s: two trainers per row (three on expert_lora), a cold checkpoint read and the first FA4/flex_attention JIT.
    "trainers/lora/test_lora_offline_grpo_resume.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "lora", "ep", "etp", "moe", "gptoss", "qwen3"),
        args_matrix=("--mode lora", "--mode lora_ep", "--mode expert_lora", "--mode lora_etp"),
        timeout=1800,
    ),
    "trainers/lora/test_lora_teacher_distill.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "lora", "ep", "moe", "gptoss", "qwen3"),
        args_matrix=("--mode lora", "--mode qlora", "--mode lora_ep", "--mode expert_lora"),
        timeout=1500,
    ),
    "trainers/lora/test_qlora_grad_sweep.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "lora", "qwen3"),
        args_matrix=(
            "--model qwen3_0_6b",
            Row("--model gpt_oss_20b", markers=("gpu", "core", "2gpu", "lora", "moe", "gptoss")),
            Row("--model mixtral_tiny", markers=("gpu", "core", "2gpu", "lora", "moe")),
        ),
        timeout=900,
    ),
    "trainers/lora/test_lora_reference_pass_fsdp2.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "lora", "qwen3"), timeout=600
    ),
    "trainers/lora/test_lora_self_distill.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "lora", "ep", "etp", "tp", "moe", "gptoss", "qwen3"),
        args_matrix=(
            *(
                f"--mode {mode}"
                for mode in (
                    "lora",
                    "qlora",
                    "lora_ep",
                    "expert_lora",
                    "lora_etp",
                    "full_ep1",
                    "ep1_unsharded",
                    "full_ep2_tp2",
                    "full_tp2",
                    "zero3",
                )
            ),
            *_rows(
                (
                    "--mode lora_ep_etp",
                    "--mode hsdp",
                    "--mode expert_lora_ep1_hsdp --per-rank-adapters",
                    "--mode expert_lora_ep1_unsharded_hsdp --per-rank-adapters",
                    "--mode full_tp2 --world 4",
                    "--mode lora_ep --world 4",
                    "--mode lora_ep --world 4 --gpus-per-node 2",
                ),
                nproc=4,
                markers=("gpu", "full", "4gpu", "lora", "ep", "etp", "tp", "hsdp", "moe", "gptoss", "qwen3"),
            ),
        ),
        timeout=1500,
    ),
    "trainers/lora/test_lora_tied_vocab_resume.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "lora", "qwen3"),
        args_matrix=(
            Row("--mode single", nproc=1, markers=("gpu", "full", "1gpu", "lora", "qwen3")),
            "--mode fsdp",
            Row("--mode cp2", markers=("gpu", "full", "2gpu", "cp", "lora", "qwen3")),
            "--mode qlora",
            Row("--mode ep2", markers=("gpu", "full", "2gpu", "ep", "moe", "lora", "qwen3"), timeout=2400),
        ),
        timeout=900,
    ),
    "trainers/lora/test_lora_pref_heads.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "lora", "qwen3"),
        args_matrix=(
            "--trainer dpo --mode lora",
            "--trainer dpo --mode qlora",
            "--trainer kto --mode lora",
            "--trainer kto --mode qlora",
            "--trainer smpo --mode lora",
            "--trainer smpo --mode qlora",
            "--trainer reward --mode lora",
            "--trainer reward --mode qlora",
            "--trainer classification --mode lora",
            "--trainer classification --mode qlora",
        ),
        timeout=900,
    ),
    "trainers/other/test_accelerate_full_shard_loading.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "qwen3", "lora"),
        args_matrix=(
            "--case dpo",
            "--case dpo_peft_precompute",
            "--case kto",
            "--case offline_grpo_kl",
            "--case teacher_distill",
        ),
        timeout=900,
        launcher="python",
    ),
    "trainers/other/test_checkpoint_roundtrip.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "ep", "etp", "tp", "cp", "moe", "qwen3", "gptoss"),
        timeout=1500,
        # Every implemented mode gets a row; an unregistered one would report coverage never run.
        args_matrix=(
            "--mode fsdp",
            "--mode tp2",
            "--mode cp2",
            "--mode ep2",
            "--mode ep2_tp2",
            "--mode ep2_no_gmm",
            "--mode etp",
        ),
    ),
    "trainers/other/test_classification.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "qwen3"),
        args_matrix=(
            "",
            Row("--world 1", nproc=1, markers=("gpu", "full", "1gpu", "qwen3")),
            Row("--hsdp", nproc=4, markers=("gpu", "full", "4gpu", "hsdp", "qwen3")),
        ),
        timeout=600,
    ),
    "trainers/other/test_distillation.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "qwen3"),
        args_matrix=(
            "",
            Row("--world 1", nproc=1, markers=("gpu", "full", "1gpu", "qwen3")),
            Row("--zero3", markers=("gpu", "full", "2gpu", "qwen3")),
            Row("--hsdp", nproc=4, markers=("gpu", "full", "4gpu", "hsdp", "qwen3")),
        ),
        timeout=600,
    ),
    "trainers/other/test_distillation_oss20b.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "ep", "tp", "etp", "moe", "gptoss"),
        args_matrix=(
            *(f"--mode {mode}" for mode in ("fsdp", "fsdp_unsharded", "ep", "tp", "ep_tp", "etp")),
            *_rows(
                ("--mode ep_etp", "--mode ep_dp2", "--mode ep_dp2 --gpus-per-node 2"),
                nproc=4,
                markers=("gpu", "full", "4gpu", "ep", "etp", "moe", "gptoss"),
            ),
            Row("--mode tp_dp2", nproc=4, markers=("gpu", "full", "4gpu", "tp", "moe", "gptoss")),
        ),
        timeout=1500,
    ),
    "trainers/other/test_embedding.py": TestSpec(nproc=2, markers=("gpu", "core", "2gpu"), timeout=600),
    # One row per method under scripts/training, which tests/cpu/config/test_entry_script_recipes.py holds them to. The
    # padded-batch methods without an engine also run with the recipe's attention unset, where Blackwell's default is
    # FA4.
    "trainers/other/test_entry_scripts.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "qwen3"),
        args_matrix=(
            "--method classification",
            "--method distillation/self-distill",
            "--method distillation/teacher-distill",
            "--method embedding",
            Row("--method embedding --family gpt_oss", markers=("gpu", "full", "2gpu", "moe", "gptoss")),
            "--method grpo/offline",
            "--method preference/dpo",
            "--method preference/kto",
            "--method preference/smpo",
            "--method reward/reward-model",
            "--method sft",
            *_rows(
                ("--method grpo/environmental", "--method grpo/online"),
                markers=("gpu", "full", "2gpu", "vllm_server", "qwen3"),
            ),
            *_rows(
                tuple(
                    f"--method {method} --attn unset"
                    for method in (
                        "classification",
                        "distillation/teacher-distill",
                        "grpo/offline",
                        "preference/dpo",
                        "preference/kto",
                        "preference/smpo",
                        "reward/reward-model",
                    )
                ),
                markers=("gpu", "full", "2gpu", "qwen3"),
            ),
        ),
        timeout=1500,
        launcher="python",
    ),
    "trainers/other/test_reward.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "qwen3"),
        args_matrix=(
            "",
            Row("--world 1", nproc=1, markers=("gpu", "full", "1gpu", "qwen3")),
            Row("--hsdp", nproc=4, markers=("gpu", "full", "4gpu", "hsdp", "qwen3")),
        ),
        timeout=600,
    ),
    "trainers/other/test_reward_vlm_e2e.py": TestSpec(nproc=2, markers=("gpu", "core", "2gpu", "vlm"), timeout=1200),
    "trainers/other/test_step_guard_device_verdict.py": TestSpec(
        nproc=1, markers=("gpu", "core", "1gpu"), timeout=300
    ),
    "trainers/preference/test_dpo.py": TestSpec(nproc=2, markers=("gpu", "core", "2gpu", "qwen3"), timeout=600),
    "trainers/preference/test_dpo_vlm.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu", "vlm"), timeout=900),
    "trainers/preference/test_smpo_vlm.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu", "vlm"), timeout=900),
    "trainers/preference/test_smpo_text_on_vlm.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "vlm"), timeout=1200
    ),
    "trainers/preference/test_kto.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu", "qwen3"), timeout=600),
    "trainers/preference/test_kto_fsdp_multi_gpu.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "qwen3"), timeout=600
    ),
    "trainers/preference/test_pref_ep_expert_lora_reference.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "lora", "ep", "moe", "gptoss"), timeout=1200
    ),
    "trainers/preference/test_smpo_cp.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "cp", "qwen3"), timeout=600
    ),
    "trainers/preference/test_smpo_ep.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "ep", "moe", "gptoss"), timeout=1000
    ),
    "trainers/preference/test_smpo_ep_experts.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "lora", "ep", "moe", "gptoss"), timeout=1200
    ),
    "trainers/preference/test_smpo_ep_cp.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "ep", "cp", "moe", "gptoss"), timeout=1000
    ),
    "trainers/preference/test_smpo_fsdp.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "qwen3"),
        args_matrix=("--mode fsdp", "--mode padding_free"),
        timeout=600,
    ),
    "trainers/preference/test_smpo_layouts.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "ep", "moe", "qwen3"),
        args_matrix=tuple(
            Row(f"--mode {mode}", nproc=nproc, markers=("gpu", "full", f"{nproc}gpu", *axes, "qwen3"))
            for mode, nproc, axes in (
                ("ep2", 2, ("ep", "moe")),
                ("ep1", 2, ("ep", "moe")),
                ("ep1_zero3", 2, ("ep", "moe")),
                ("ep1_unsharded", 2, ("ep", "moe")),
                ("etp2", 2, ("etp", "moe")),
                ("ep2tp2", 2, ("ep", "tp", "moe")),
                ("ep2etp2", 4, ("ep", "etp", "moe")),
                ("ep2dp2", 4, ("ep", "moe")),
                ("ep2x2dom", 4, ("ep", "moe")),
                ("zero3", 2, ()),
                ("hsdp", 4, ("hsdp",)),
            )
        ),
        timeout=900,
    ),
    "trainers/preference/test_smpo_padding_free_segments.py": TestSpec(
        nproc=1, markers=("gpu", "core", "1gpu", "moe", "lfm2", "qwen3"), timeout=600
    ),
    "trainers/preference/test_smpo_tp.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "tp", "qwen3"), timeout=600
    ),
    "trainers/preference/test_smpo_tp_resume.py": TestSpec(
        # 2400s: the FA4 backward JITs per shape over the 6 steps of this two-phase resume.
        nproc=2,
        markers=("gpu", "core", "2gpu", "tp", "qwen3"),
        timeout=2400,
    ),
    "trainers/sft/test_optimizer_shard_save_after_eval.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "ep", "moe", "vlm", "qwen3"),
        # fsdp pins the family-agnostic FSDP2 mechanism, ep the composite-VLM + plain-expert + AdamWBF16 shape.
        args_matrix=("--mode fsdp", "--mode ep"),
        timeout=900,
    ),
    "trainers/sft/test_sft_accelerate_modes.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "qwen3"), timeout=600
    ),
    # The MoE suites are `full`; an fsdp-only launch would leave the EP lazy-load path untested (on LFM2 the
    # non-persistent expert_bias buffers land on meta).
    "trainers/sft/test_sft_families.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "tp", "cp", "qwen3"),
        args_matrix=(
            "--suite qwen3_dense --mode fsdp",
            "--suite qwen3_dense --mode tp",
            "--suite qwen3_dense --mode cp",
            Row("--suite qwen3_5_dense", markers=("gpu", "core", "2gpu", "tp", "qwen3"), timeout=600),
            Row("--suite qwen3_moe", markers=("gpu", "full", "2gpu", "ep", "moe", "qwen3"), timeout=1000),
            *_rows(
                (
                    "--suite qwen3_5_moe --mode ep",
                    "--suite qwen3_5_moe --mode ep_no_gmm",
                    "--suite qwen3_5_moe --mode etp",
                ),
                markers=("gpu", "full", "2gpu", "ep", "etp", "moe", "qwen3"),
                timeout=1500,
            ),
            Row("--suite qwen3_5_ep", nproc=4, markers=("gpu", "full", "4gpu", "ep", "moe", "qwen3"), timeout=1500),
            Row("--suite glm4_moe", markers=("gpu", "full", "2gpu", "ep", "moe", "glm4"), timeout=1500),
            Row("--suite gemma4_moe", markers=("gpu", "full", "2gpu", "ep", "moe", "gemma4"), timeout=1500),
            *_rows(
                ("--suite lfm2_moe --mode fsdp", "--suite lfm2_moe --mode ep"),
                markers=("gpu", "full", "2gpu", "ep", "moe", "lfm2"),
                timeout=1500,
            ),
            *_rows(
                (
                    "--suite bailing_moe --mode fsdp",
                    "--suite bailing_moe --mode ep",
                    "--suite bailing_moe --mode ep_no_gmm",
                ),
                markers=("gpu", "full", "2gpu", "ep", "moe", "bailing"),
                timeout=1500,
            ),
        ),
        timeout=900,
    ),
    "trainers/sft/test_sft_checkpoint_resume.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "cp", "tp", "ep", "moe", "qwen3", "gptoss"),
        args_matrix=("--mode fsdp", "--mode fsdp_base", "--mode cp", "--mode tp", "--mode ep"),
        # 2400s: the cp row's 4096-token FA4 JITs per shape, and every mode runs a 6-step two-phase resume plus two
        # host copies of its optimizer state.
        timeout=2400,
    ),
    "trainers/sft/test_sft_ep_roundtrip.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "ep", "moe"),
        args_matrix=tuple(
            Row(f"--family {family}", markers=("gpu", "core", "2gpu", "ep", "moe", _TINY_MOE_FAMILY_MARKERS[family]))
            for family in (
                "cohere2_moe",
                "deepseek_v4",
                "glm5_next",
                "nemotron_h",
                "qwen3_next",
                "qwen4_exp_text",
                "step3p7",
            )
        )
        + (Row("--family nemotron_h_omni", markers=("gpu", "core", "2gpu", "ep", "moe", "vlm", "nemotron")),),
        timeout=1200,
    ),
    # No `etp` marker: EP+TP+ETP is not a supported axis set.
    "trainers/sft/test_sft_ep_tp_flex.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "ep", "tp", "moe", "gptoss"),
        timeout=1000,
    ),
    # Every scope on one family per placement, joint soft on the rest, over each shard layout. A row at `--attn auto`
    # takes the backend the family's production load selects.
    "trainers/sft/test_sft_mtp_head.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "1gpu", "2gpu", "4gpu", "ep", "etp", "moe", "qwen3", "glm4", "nemotron"),
        args_matrix=(
            *_rows(
                (
                    "--family nemotron_h --mode single",
                    "--family nemotron_h_no_latent --mode single --arms soft",
                    "--family nemotron_h_omni --mode single --arms soft --kernel chunked",
                ),
                nproc=1,
                markers=("gpu", "full", "1gpu", "moe", "nemotron"),
            ),
            *_rows(
                tuple(
                    f"--family nemotron_h --mode {mode}{gc}"
                    for mode in ("fsdp", "fsdp_zero3", "ep", "ep1", "ep1_ignored")
                    for gc in ("", " --gc")
                )
                + (
                    "--family nemotron_h_no_latent --mode ep --arms soft --gc",
                    "--family nemotron_h_no_latent --mode ep --arms soft --attn flash_attention_4",
                    "--family nemotron_h_omni --mode ep --arms soft --kernel chunked --gc",
                    "--family nemotron_h_omni --mode fsdp --arms soft --kernel chunked",
                    "--family nemotron_h --mode ep --kernel chunked",
                    "--family nemotron_h --mode ep --starve",
                    "--family nemotron_h --mode ep1_fp32_router --bf16",
                    "--family nemotron_h --mode ep_fp32_masters --arms trunk_trains --gc",
                    "--family nemotron_h --mode ep --attn auto --gc",
                    "--family nemotron_h --mode ep --attn flash_attention_2 --packed",
                    "--family nemotron_h --mode ep --attn flash_attention_4 --packed --gc",
                    "--family nemotron_h --mode etp --gc",
                    "--family nemotron_h --mode ep --arms soft --accumulation 2 --gc",
                ),
                markers=("gpu", "full", "2gpu", "ep", "moe", "nemotron"),
            ),
            *_rows(
                ("--family nemotron_h --mode ep_etp --gc", "--family nemotron_h --mode ep_etp"),
                nproc=4,
                markers=("gpu", "full", "4gpu", "ep", "etp", "moe", "nemotron"),
            ),
            *_rows(
                tuple(
                    f"--family {family} --mode single{arms}"
                    for family, arms in (
                        ("qwen3_5_moe_text", ""),
                        ("glm4_moe_lite", ""),
                        ("qwen3_5", " --arms soft"),
                        ("qwen3_5_moe", " --arms soft"),
                        ("qwen3_next", " --arms soft"),
                    )
                )
                + ("--family qwen3_5_moe_text --mode single --bf16",),
                nproc=1,
                markers=("gpu", "full", "1gpu", "moe", "qwen3", "glm4"),
            ),
            *_rows(
                tuple(
                    f"--family {family} --mode {mode}{gc}"
                    for family in ("qwen3_5_moe_text", "glm4_moe_lite")
                    for mode in ("fsdp", "fsdp_zero3", "ep", "ep1", "ep1_ignored")
                    for gc in ("", " --gc")
                )
                + (
                    "--family qwen3_5_moe_text --mode ep1_fp32_router --bf16 --gc",
                    "--family glm4_moe_lite --mode ep1_fp32_router --bf16",
                    "--family qwen3_5 --mode fsdp_zero3 --arms soft --gc",
                    "--family qwen3_next --mode ep --arms soft --gc",
                    "--family qwen3_5_moe --mode ep --arms soft --gc",
                    "--family qwen3_5_moe_text --mode ep --kernel chunked --gc",
                    "--family glm4_moe_lite --mode ep --kernel chunked",
                    "--family qwen3_5_moe_text --mode ep --starve",
                    "--family glm4_moe_lite --mode ep --starve --gc",
                    "--family qwen3_5_moe_text --mode ep_fp32_masters --arms trunk_trains --gc",
                    "--family glm4_moe_lite --mode ep_fp32_masters --arms trunk_trains",
                    "--family qwen3_5_moe_text --mode ep --attn auto --gc",
                    "--family glm4_moe_lite --mode ep --attn auto",
                    "--family qwen3_5_moe_text --mode ep --attn flash_attention_2 --packed",
                    "--family qwen3_5_moe_text --mode ep --attn flash_attention_4 --packed --gc",
                    "--family glm4_moe_lite --mode ep --attn flash_attention_2 --packed --gc",
                    "--family qwen3_5_moe_text --mode etp --gc",
                    "--family glm4_moe_lite --mode etp",
                    "--family qwen3_5_moe_text --mode ep --arms soft --accumulation 2 --gc",
                    "--family glm4_moe_lite --mode ep1 --arms trunk_trains --accumulation 2",
                    # The joint recipe's own shape.
                    "--family qwen3_5_moe_text --mode ep_fp32_masters --arms trunk_trains --attn flash_attention_2 "
                    "--packed --accumulation 2 --gc",
                ),
                markers=("gpu", "full", "2gpu", "ep", "moe", "qwen3", "glm4"),
            ),
            *_rows(
                (
                    "--family qwen3_5_moe_text --mode ep_etp --gc",
                    "--family qwen3_5_moe_text --mode ep_etp",
                    "--family glm4_moe_lite --mode ep_etp --gc",
                ),
                nproc=4,
                markers=("gpu", "full", "4gpu", "ep", "etp", "moe", "qwen3", "glm4"),
            ),
        ),
        timeout=900,
    ),
    # An exact resume, its export, the shard merge, the scope flip and the gate. The `multi_node` rows hold two EP groups
    # as replicas over two NVLink domains simulated on one machine; the script's docstring has the launch on two.
    "trainers/sft/test_sft_mtp_head_checkpoint.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "1gpu", "2gpu", "4gpu", "ep", "etp", "moe", "qwen3", "glm4", "nemotron"),
        args_matrix=(
            *_rows(
                ("--family nemotron_h --mode single",), nproc=1, markers=("gpu", "full", "1gpu", "moe", "nemotron")
            ),
            *_rows(
                (
                    "--family nemotron_h --mode ep --flip --gate",
                    "--family nemotron_h --mode ep --gc",
                    "--family nemotron_h_no_latent --mode ep --flip",
                    "--family nemotron_h_omni --mode ep",
                    "--family nemotron_h --mode ep_fp32_masters --gc",
                    "--family nemotron_h --mode etp",
                    "--family nemotron_h --mode ep1",
                    "--family nemotron_h --mode ep1_ignored",
                    "--family nemotron_h --mode fsdp --flip --gate",
                ),
                markers=("gpu", "full", "2gpu", "ep", "moe", "nemotron"),
            ),
            *_rows(
                (
                    "--family nemotron_h --mode ep_etp",
                    "--family nemotron_h --mode multi_node --ep_size 2 --simulated_domain 2",
                    "--family nemotron_h --mode multi_node --simulated_domain 2",
                ),
                nproc=4,
                markers=("gpu", "full", "4gpu", "ep", "etp", "moe", "nemotron"),
            ),
            *_rows(
                ("--family qwen3_5_moe_text --mode single", "--family glm4_moe_lite --mode single"),
                nproc=1,
                markers=("gpu", "full", "1gpu", "moe", "qwen3", "glm4"),
            ),
            *_rows(
                (
                    "--family qwen3_5_moe_text --mode ep --flip --gate --fused_gate",
                    "--family glm4_moe_lite --mode ep --flip --gate",
                    "--family qwen3_5_moe_text --mode ep --gc",
                    "--family glm4_moe_lite --mode ep --gc",
                    "--family qwen3_5_moe_text --mode ep --sharded",
                    "--family qwen3_next --mode ep --sharded",
                    "--family qwen3_5_moe --mode ep",
                    "--family qwen3_5_moe_text --mode ep_fp32_masters --gc",
                    "--family qwen3_5_moe_text --mode etp",
                    "--family glm4_moe_lite --mode ep1",
                    "--family qwen3_5_moe_text --mode ep1_ignored",
                    "--family qwen3_5 --mode fsdp --flip --gate",
                    "--family qwen3_5_moe_text --mode fsdp --flip",
                    "--family qwen3_5 --mode fsdp_zero3 --gc",
                ),
                markers=("gpu", "full", "2gpu", "ep", "moe", "qwen3", "glm4"),
            ),
            *_rows(
                (
                    "--family qwen3_5_moe_text --mode ep_etp",
                    "--family glm4_moe_lite --mode ep_etp --gc",
                    "--family qwen3_5_moe_text --mode multi_node --ep_size 2 --simulated_domain 2",
                    "--family glm4_moe_lite --mode multi_node --ep_size 2 --simulated_domain 2 --gc",
                    "--family qwen3_5_moe_text --mode multi_node --simulated_domain 2",
                ),
                nproc=4,
                markers=("gpu", "full", "4gpu", "ep", "etp", "moe", "qwen3", "glm4"),
            ),
        ),
        timeout=1200,
    ),
    # A few minutes a row on Qwen3.6-35B. Each records its acceptance curve; a bar holds at the default weights only.
    "trainers/sft/test_sft_mtp_head_real.py": TestSpec(
        nproc=8,
        markers=("gpu", "full", "8gpu", "ep", "moe", "qwen3", "nemotron"),
        args_matrix=(
            *_rows(
                (
                    "--release lightning --row untrained_export --attn flash_attention_4",
                    "--release super --row untrained_export --attn flash_attention_4 --text_only",
                    "--release super --row untrained_export --attn flash_attention_4 --kernel chunked",
                    "--release lightning --row own_soft --attn flash_attention_4",
                    "--release lightning --row own_hard --attn flash_attention_4",
                    "--release super --row own_soft --attn flash_attention_4 --text_only",
                    "--release super --row own_soft --attn flash_attention_4 --kernel chunked",
                ),
                markers=("gpu", "full", "8gpu", "ep", "moe", "nemotron"),
            ),
            "--row stale_soft",
            "--row stale_hard",
            "--row own_soft",
            "--row own_hard",
            "--row stale_soft_fused",
            "--row stale_soft_fa4",
            "--row own_samples_soft",
            "--row oversized_lr",
            "--row off_by_one",
            "--row detached",
        ),
        timeout=3600,
    ),
    # Eight arms of about 15 minutes in one launch; `--text_only` takes the fused kernel a composite forces off.
    "trainers/sft/test_sft_mtp_head_cost.py": TestSpec(
        nproc=8,
        markers=("gpu", "full", "8gpu", "ep", "moe", "qwen3"),
        args_matrix=("", "--text_only"),
        timeout=10800,
    ),
    # Skips on a server not drafting, so a tier's server with no speculative config passes over it. The script's docstring
    # has the launch.
    "trainers/sft/test_sft_mtp_head_serving.py": TestSpec(
        nproc=1,
        markers=("gpu", "full", "1gpu", "qwen3", "nemotron", "vllm_server"),
        args_matrix=(
            Row("--engine vllm --at_least 0.7"),
            Row("--engine sglang --at_least 0.7", markers=("gpu", "full", "1gpu", "qwen3", "sglang_server")),
            Row("--engine vllm --release lightning", markers=("gpu", "full", "1gpu", "nemotron", "vllm_server")),
            Row("--engine sglang --release lightning", markers=("gpu", "full", "1gpu", "nemotron", "sglang_server")),
            Row("--engine vllm --release super", markers=("gpu", "full", "1gpu", "nemotron", "vllm_server")),
            Row("--engine sglang --release super", markers=("gpu", "full", "1gpu", "nemotron", "sglang_server")),
        ),
        timeout=3600,
    ),
    "trainers/sft/test_sft_ling_mtp_tail.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "4gpu", "ep", "etp", "moe", "bailing"),
        args_matrix=(
            *_rows(
                ("--mode ep", "--mode ep1", "--mode ep1_ignored"),
                markers=("gpu", "full", "2gpu", "ep", "moe", "bailing"),
            ),
            *_rows(("--mode ep_etp",), nproc=4, markers=("gpu", "full", "4gpu", "ep", "etp", "moe", "bailing")),
        ),
        timeout=900,
    ),
    "trainers/sft/test_sft_eval_flce.py": TestSpec(
        nproc=1, markers=("gpu", "core", "1gpu", "glm4", "moe"), timeout=600
    ),
    "trainers/sft/test_sft_fsdp_reshard.py": TestSpec(nproc=4, markers=("gpu", "core", "4gpu", "qwen3"), timeout=1200),
    # 600s: two 4-step Qwen3-0.6B arms plus two model loads, with headroom for a cold HF cache and the FA4 JIT.
    "trainers/sft/test_sft_fsdp_backward_reshard.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "qwen3"), timeout=600
    ),
    "trainers/sft/test_sft_fsdp_defer_grad_sync.py": TestSpec(
        nproc=4,
        markers=("gpu", "full", "4gpu", "hsdp", "tp", "cp", "ep", "moe", "qwen3"),
        args_matrix=(
            "--mode dp",
            "--mode hsdp",
            "--mode tp",
            "--mode cp",
            "--mode ep1",
            "--mode ep1_fp32_router",
            "--mode ep",
            "--mode ep2",
        ),
        timeout=900,
    ),
    "trainers/other/test_self_distillation_vlm.py": TestSpec(
        nproc=1, markers=("gpu", "core", "1gpu", "vlm"), timeout=900
    ),
    "trainers/other/test_self_distillation_text.py": TestSpec(
        nproc=1, markers=("gpu", "core", "1gpu", "qwen3"), timeout=600
    ),
    "trainers/sft/test_sft_hsdp.py": TestSpec(
        nproc=4,
        markers=("gpu", "core", "4gpu", "hsdp", "qwen3"),
        args_matrix=(
            "",
            Row("--lora", markers=("gpu", "full", "4gpu", "hsdp", "lora", "qwen3")),
            Row("--lora --cp-size 2", markers=("gpu", "full", "4gpu", "hsdp", "cp", "lora", "qwen3")),
        ),
        timeout=900,
    ),
    "trainers/sft/test_sft_ep_multinode_sim.py": TestSpec(
        nproc=4,
        markers=("gpu", "full", "4gpu", "ep", "moe", "gptoss"),
        args_matrix=(
            "--mode full_ft",
            Row("--mode expert_lora_resume", markers=("gpu", "full", "4gpu", "ep", "lora", "moe", "qwen3")),
        ),
        timeout=1200,
    ),
    "trainers/sft/test_fsdp2_pretrain_eval_then_train.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "qwen3"), timeout=900
    ),
    "trainers/sft/test_sft_gemma4_vlm.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "ep", "vlm", "moe", "gemma4"), timeout=1500
    ),
    # The CP, TP-MoE and EP+TP sequential sites are checked statically on CPU.
    "trainers/sft/test_sft_fp32_pinned_params.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "ep", "lora", "moe", "deepseek_v4", "glm5", "inkling"),
        args_matrix=(
            "--family deepseek_v4 --mode full --ep 1",
            "--family glm5_next --mode full --ep 1",
            "--family inkling_text --mode full --ep 1",
            "--family inkling_text --mode full --ep 1 --no-grouped-gemm",
            "--family inkling_text --mode full --ep 1 --no-grouped-gemm --fp32-masters",
            "--family inkling_text --mode full --ep 1 --fp32-masters",
            "--family deepseek_v4 --mode full --ep 1 --fp32-masters",
            "--family inkling_text --mode full --ep 2 --fp32-masters",
            "--family glm5_next --mode full --ep 2 --fp32-masters",
            "--family glm5_next --mode full --ep 2 --dsa-kernel",
            "--family deepseek_v4 --mode expert_lora --ep 1",
            "--family glm5_next --mode mixed --ep 1",
            "--family deepseek_v4 --mode expert_lora --ep 2",
            "--family glm5_next --mode mixed --ep 2",
            "--family deepseek_v4 --mode full --ep 2",
            "--family inkling_text --mode full --ep 2",
            Row(
                "--family deepseek_v4 --mode full --ep 1 --world 1",
                nproc=1,
                markers=("gpu", "core", "1gpu", "moe", "deepseek_v4"),
                timeout=600,
            ),
        ),
        timeout=900,
    ),
    "trainers/sft/test_sft_fsdp_excluded_params.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "lora", "moe", "glm4", "glm5"),
        args_matrix=("--case router_only", "--case glm5_next_mixed_uncast"),
        timeout=900,
    ),
    "trainers/sft/test_sft_fp32_router_ep1.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "ep", "lora", "moe", "qwen3", "gptoss", "zaya", "lfm2"),
        args_matrix=(
            "--family qwen3_moe",
            "--family gpt_oss",
            "--family zaya",
            "--family lfm2_moe",
            "--family qwen3_moe --lora",
        ),
        timeout=900,
    ),
    # `full`: plain SFT smokes on the 20B checkpoint. The core tier keeps the gpt-oss correctness gates
    # (`parallelism/ep/test_ep_modes_vs_reference.py`).
    "trainers/sft/test_sft_gptoss_modes.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "ep", "cp", "tp", "etp", "moe", "gptoss"),
        args_matrix=(
            "--mode fsdp",
            "--mode fsdp_fa2",
            "--mode ep",
            "--mode ep_fa2",
            "--mode cp",
            "--mode tp",
            "--mode ep_cp",
            "--mode ep_tp",
            "--mode ep_etp",
        ),
        timeout=1500,
    ),
    "trainers/sft/test_sft_gptoss_trainable_sinks.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "tp", "ep", "moe", "gptoss"),
        args_matrix=("--mode fsdp", "--mode tp", "--mode ep"),
        timeout=1500,
    ),
    "trainers/sft/test_sft_nonfinite_step_guard.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "qwen3"),
        args_matrix=(
            Row("--mode ep1", markers=("gpu", "core", "2gpu", "ep", "moe", "qwen3")),
            Row("--mode ep2", markers=("gpu", "core", "2gpu", "ep", "moe", "qwen3")),
            Row("--mode ep2_dp2", nproc=4, markers=("gpu", "core", "4gpu", "ep", "moe", "qwen3")),
            Row("--mode ep2 --fp32-masters", markers=("gpu", "core", "2gpu", "ep", "moe", "qwen3")),
            Row("--mode ep2_dp2 --fp32-masters", nproc=4, markers=("gpu", "core", "4gpu", "ep", "moe", "qwen3")),
            Row("--mode etp2", markers=("gpu", "core", "2gpu", "etp", "moe", "qwen3")),
            "--mode dense",
            Row("--mode tp", markers=("gpu", "core", "2gpu", "tp", "qwen3")),
            Row("--mode cp", markers=("gpu", "core", "2gpu", "cp", "qwen3")),
            Row("--mode qlora", markers=("gpu", "core", "2gpu", "lora", "qwen3")),
            Row("--mode pp", nproc=4, markers=("gpu", "core", "4gpu", "pp", "qwen3")),
            Row("--mode ep2_tp2", nproc=4, markers=("gpu", "full", "4gpu", "ep", "tp", "moe", "qwen3")),
            Row("--mode ep2_cp2", markers=("gpu", "full", "2gpu", "ep", "cp", "moe", "qwen3")),
            Row("--mode ep2_etp2", nproc=4, markers=("gpu", "full", "4gpu", "ep", "etp", "moe", "qwen3")),
            Row("--mode pp_ep", nproc=4, markers=("gpu", "full", "4gpu", "pp", "ep", "moe", "gptoss")),
            Row("--mode ep2_dp2 --max-grad-norm 0", nproc=4, markers=("gpu", "full", "4gpu", "ep", "moe", "qwen3")),
            Row("--mode tp --max-grad-norm 0", markers=("gpu", "full", "2gpu", "tp", "qwen3")),
            Row(
                "--mode ep2_tp2 --max-grad-norm 0",
                nproc=4,
                markers=("gpu", "full", "4gpu", "ep", "tp", "moe", "qwen3"),
            ),
        ),
        timeout=1200,
    ),
    "trainers/sft/test_sft_prefix_lm.py": TestSpec(
        nproc=1,
        markers=("gpu", "core", "1gpu", "vlm"),
        args_matrix=(
            "--family paligemma",
            "--family paligemma2",
            Row("--family paligemma --layout fsdp-gc", nproc=2, markers=("gpu", "core", "2gpu", "vlm")),
            Row("--family paligemma2 --layout fsdp-gc", nproc=2, markers=("gpu", "core", "2gpu", "vlm")),
        ),
        timeout=900,
    ),
    "trainers/sft/test_sft_tp_packed_varlen.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "tp", "qwen3"), timeout=900
    ),
    "trainers/sft/test_sft_mixtral_fsdp2.py": TestSpec(nproc=2, markers=("gpu", "core", "2gpu", "moe"), timeout=600),
    "trainers/sft/test_sft_qwen3_modes.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "tp", "cp", "qwen3"), timeout=600
    ),
    "trainers/sft/test_sft_vlm_images.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "vlm"),
        timeout=900,
        args_matrix=(
            "--family qwen3_vl --data all",
            "--family qwen3_5 --data all",
            "--family qwen3_5_moe --data all",
            "--family gemma4 --data all",
            "--family qwen3_vl --data mixed --freeze-tower",
            "--family qwen3_5 --data mixed --freeze-tower",
            "--family qwen3_5_moe --data mixed --freeze-tower",
            "--family gemma4 --data mixed --freeze-tower",
            "--family qwen3_vl --data mixed",
            "--family qwen3_5 --data mixed",
            "--family qwen3_5_moe --data mixed",
            "--family gemma4 --data mixed",
            "--family qwen3_5 --data mixed --zero3",
            "--family qwen3_5 --data mixed --freeze-embedding",
            "--family qwen3_vl --data mixed --tower-adapters",
            "--family qwen3_5_moe --data mixed --tower-adapters",
            "--family qwen3_5 --data mixed --tower-adapters --zero3",
            "--family gemma4 --data mixed --tower-adapters",
            Row("--family nemotron_h_omni --data all", markers=("gpu", "core", "2gpu", "vlm", "nemotron")),
            Row("--family nemotron_h_omni --data mixed", markers=("gpu", "core", "2gpu", "vlm", "nemotron")),
            Row(
                "--family nemotron_h_omni --data mixed --freeze-embedding",
                markers=("gpu", "core", "2gpu", "vlm", "nemotron"),
            ),
            Row(
                "--family nemotron_h_omni --data mixed --freeze-tower",
                markers=("gpu", "core", "2gpu", "vlm", "nemotron"),
            ),
            Row(
                "--family nemotron_h_omni --data mixed --tower-adapters",
                markers=("gpu", "core", "2gpu", "vlm", "nemotron"),
            ),
        ),
    ),
    "trainers/sft/test_sft_vlm.py": TestSpec(nproc=2, markers=("gpu", "core", "2gpu", "vlm", "qwen3"), timeout=600),
    "trainers/sft/test_sft_composite_text_rows_qwen3_5.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "qwen3"), timeout=1000
    ),
    "trainers/sft/test_zaya_fsdp.py": TestSpec(nproc=2, markers=("gpu", "full", "2gpu", "zaya"), timeout=1500),
    "trainers/sft/test_zaya_load_forward_backward.py": TestSpec(
        nproc=1, markers=("gpu", "core", "1gpu", "zaya"), timeout=1500
    ),
}

# Registered in `tests/conftest.py::pytest_configure`; `tests/cpu/conventions/test_gpu_harness_conventions.py` holds
# every spec marker to these.
ALL_MARKERS = (
    "gpu",
    "core",
    "full",
    "1gpu",
    "2gpu",
    "4gpu",
    "8gpu",
    "ep",
    "cp",
    "tp",
    "etp",
    "pp",
    "hsdp",
    "vlm",
    "vllm_server",
    "sglang_server",
    "lora",
    "moe",
    "gptoss",
    "qwen3",
    "glm4",
    "gemma4",
    "mistral4",
    "nemotron",
    "bailing",
    "lfm2",
    "laguna",
    "zaya",
    "deepseek_v4",
    "inkling",
    "cohere2_moe",
    "glm5",
    "step3p7",
)


def script_path(rel: str) -> Path:
    return GPU_DIR / rel


# Driven by hand from a docs recipe or by `make bench`, never by the pytest launcher. Listing them is what marks an
# unlisted `bench*.py` as an orphan.
_UNMANIFESTED_BENCHMARKS = {
    "optimizers/bench_adamw_bf16.py",
    "optimizers/bench_muon.py",
    "optimizers/bench_muon_qwen3_5.py",
    "profiling/bench_ep_arena_headroom.py",
    "profiling/bench_ep_buffer_backends.py",
    "profiling/benchmark_attention_implementations.py",
    "profiling/benchmark_collators.py",
    "profiling/benchmark_convergence.py",
    "profiling/benchmark_gemma4_attention.py",
    "profiling/benchmark_grouped_mm.py",
    "profiling/benchmark_head_logprobs.py",
    "profiling/benchmark_hyper_connection.py",
    "profiling/benchmark_lowp_kernels.py",
    "profiling/benchmark_moe_block.py",
    "profiling/benchmark_roofline.py",
    "profiling/benchmark_sft_dense.py",
    "profiling/benchmark_sft_ep.py",
    "profiling/benchmark_smpo_ep.py",
    "profiling/benchmark_torch_compile.py",
    "profiling/benchmark_trl_baseline.py",
}


def unregistered_scripts() -> list[str]:
    """Scripts under ``tests/gpu/`` with no launch spec; the conftest fails collection on any."""
    on_disk = {
        str(p.relative_to(GPU_DIR))
        for pattern in ("test_*.py", "bench*.py")
        for p in GPU_DIR.rglob(pattern)
        if "__pycache__" not in p.parts
    }
    return sorted(on_disk - set(MANIFEST) - LAUNCHER_ENTRYPOINTS - _UNMANIFESTED_BENCHMARKS)


def stale_entries() -> list[str]:
    """Listed scripts that no longer exist on disk."""
    listed = (*MANIFEST, *LAUNCHER_ENTRYPOINTS, *_UNMANIFESTED_BENCHMARKS)
    return sorted(rel for rel in listed if not script_path(rel).exists())
