"""Launch specs for the GPU test suite.

GPU tests are external ``torchrun`` scripts, each run through its ``gpu_test_main`` entry except the
few whose lifecycle the harness cannot express, which keep their own (``_OWN_LIFECYCLE`` in
``tests/cpu/conventions/test_gpu_harness_conventions.py`` names each with its reason). Pytest cannot
read ``nproc`` / markers / timeout from inside them, so this manifest maps
each script (path relative to ``tests/gpu/``) to its launch spec; ``tests/gpu/conftest.py``
reads it and generates one pytest node per ``(script, args)`` with the right markers,
process count and timeout. The launcher shells out ``torchrun --nproc_per_node=<nproc>``
and asserts the exit code, parsing the structured result line when the script uses
``tests.common.harness.gpu_test_main``.

To add a test: drop the script under ``tests/gpu/`` and add one ``TestSpec`` line here.
A script present on disk but missing from the manifest is reported by
:func:`unregistered_scripts`, and the conftest fails collection on that drift. The
:data:`LAUNCHER_ENTRYPOINTS` are the only modules there pytest collects itself.

Markers (selection):
    gpu                        — every entry (the suite tier).
    core | full                — ``core`` is the intended PR gate: small, fast, <=2 GPU, tiny
                                 model. The registered tier is wider than that intent; see
                                 ``agent-docs/contributing/README.md`` for the measured budget and the
                                 entries that exceed it. ``full`` = large model or many-GPU.
                                 CI runs ``-m "gpu and core"``; ``-m gpu`` is hand-run.
    1gpu | 2gpu | 4gpu | 8gpu  — required GPU count, matches the row's ``nproc``.
    ep cp tp etp pp hsdp       — parallelism axis under test.
    vlm lora moe               — capability under test. ``vlm`` is a vision-language model;
                                 a test needing a live vLLM server is ``vllm_server``, and a
                                 live SGLang server ``sglang_server``.
    vllm_server                — needs the vLLM container from ``docker-compose.vllm.yml`` already
                                 serving (``VLLM_SERVER_URL``, default localhost:8000). External
                                 infrastructure, so these are ``full``, never a PR gate: deselect
                                 with ``-m "gpu and not vllm_server"`` when no server is up.
                                 Two launch requirements this tier cannot check (``make
                                 test-gpu-vllm`` sets both up):
                                   * the server must own a GPU the trainer does not use, since weight
                                     sync is an NCCL broadcast and a rank cannot broadcast to itself
                                     (drive the trainer with ``CUDA_VISIBLE_DEVICES`` excluding it);
                                   * the trainer and the server must agree on the NCCL transport:
                                     the socket recipe both compose bases default to (``make``
                                     passes it), or EFA on both ends (``EFA=1`` plus the compose
                                     overlay). A mismatch hangs the cross-container group instead of
                                     failing: both GPUs spin until the 120 s formation deadline, per
                                     test.
                                 A trainer killed while attached to the weight-transfer engine leaves
                                 the server's scheduler stuck while ``/health`` still answers 200;
                                 restart the container before re-running.
                                 Both server tiers split by ``moe``: these tests broadcast the
                                 trainer's own weights into the served model and assert the served
                                 policy changed, so the server must serve the checkpoint being
                                 trained: dense (Qwen3-0.6B) for the ``not moe`` half, Qwen3-30B-A3B
                                 for the ``moe`` half. No single server satisfies both; run two passes
                                 (``make test-gpu-vllm`` then ``... SERVER_TIER=moe``).
    <model family>             — gptoss / qwen3 / glm4 / glm5 / gemma4 / mistral4 /
                                 bailing / lfm2 / zaya / deepseek_v4 / inkling / cohere2_moe /
                                 step3p7 / laguna.
"""

from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Literal

GPU_DIR = Path(__file__).parent
# The pytest modules under tests/gpu/, relative to it like the MANIFEST keys: the manifest launcher
# and its contract tests. Pytest collects these directly; everything else there is a torchrun script.
LAUNCHER_ENTRYPOINTS = frozenset({"test_suite.py", "test_launcher_contract.py"})


@dataclass(frozen=True)
class Row:
    """An ``args_matrix`` entry that launches unlike the rest of its spec: each launch field set here (every
    field after ``args``) replaces the spec's, so one script runs at several widths, tiers or engines."""

    args: str
    nproc: int | None = None
    markers: tuple | None = None
    timeout: int | None = None
    flaky: bool | None = None


@dataclass(frozen=True)
class TestSpec:
    """How to launch one GPU test script.

    Attributes:
        nproc: GPUs ``torchrun`` must launch (``--nproc_per_node``).
        markers: pytest markers for selection (always includes ``"gpu"``).
        args_matrix: CLI arg-strings, or :class:`Row` entries carrying their own launch values; the
            conftest generates one node per entry, so a multi-mode script (``--mode fsdp`` /
            ``--mode ep``) becomes several nodes. Default ``("",)`` = a single node launched with no
            extra args.
        timeout: seconds before the launcher kills the process group (a hard kill, since
            NCCL / FA hangs do not return).
        flaky: known-transient; the conftest applies scoped reruns.
        launcher: ``"torchrun"`` launches ``nproc`` ranks of the script; ``"python"`` runs the script as
            one process that starts its own ``nproc`` ranks (a ``halo launch`` driver), handed the count
            in ``HALO_TEST_NPROC``.

    World-size strictness is not declared here: each script owns it via
    ``gpu_test_main(exact_world_size=N)``, which is authoritative and more precise than a
    manifest bool; a script whose rows run at several widths derives it from the row's arguments.
    The launcher always launches exactly ``nproc``.
    """

    # pytest would otherwise try to collect this as a test class (leading "Test").
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
    """``args_matrix`` as :class:`Row` entries that all launch with ``launch`` over their spec's values."""
    return tuple(Row(args, **launch) for args in args_matrix)


# One per EP MoE family, the ``--family`` names of tests/common/tiny_models.py's TINY_MOE_FAMILIES,
# each with the model-family marker its rows carry; tests/cpu/conventions/test_tiny_family_roster.py
# holds the names and the rows below to the registry.
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
    "qwen3_5_moe_text": "qwen3",
    "qwen3_moe": "qwen3",
    "step3p7": "step3p7",
    "zaya": "zaya",
}
_TINY_MOE_FAMILIES = tuple(_TINY_MOE_FAMILY_MARKERS)
# The families the reentrant-recompute suite scores row by row: one per expert layout it crosses.
_REENTRANT_FAMILIES = ("qwen3_moe", "gpt_oss", "gemma4_text")


def _family_markers(families) -> tuple[str, ...]:
    """The model-family markers of a sweep over ``families`` (``_TINY_MOE_FAMILIES`` names)."""
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
# The core rows cover Qwen3-MoE; the family sweep runs every other family.
_PRECOMPUTE_SWEEP_FAMILIES = tuple(family for family in _TINY_MOE_FAMILIES if family != "qwen3_moe")
# The MoE families of the CP roster, one tiny model per Ulysses wrapper class beside the dense qwen3;
# tests/cpu/conventions/test_tiny_family_roster.py holds them to tests/common/equivalence/cp_equivalence.py.
_CP_MOE_FAMILIES = ("bailing_moe", "cohere2_moe", "glm4_moe_lite", "gpt_oss", "mistral4", "qwen3_5_moe_text")
# The pipeline-registered MoE families test_pp_family_vs_reference.py runs (GPT-OSS has its own PP suites).
_PP_MOE_FAMILIES = ("deepseek_v4", "glm5_next", "qwen3_5_moe_text", "step3p7")
# The families whose attention the selective-TP path shards; tests/cpu/conventions/test_tiny_family_roster.py
# holds this to halo/distributed/tensor_parallel/parallelize_attention.py.
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
# GPT-OSS on every two-rank layout of the full_determinism backward replay, Qwen3-MoE on ep2; the sweep
# runs ep2 on every other family. tests/cpu/conventions/test_tiny_family_roster.py holds both to the
# replay's layouts and to the roster.
_DETERMINISM_CORE_ROWS = (
    *(f"--family gpt_oss --mode {mode}" for mode in ("ep2", "ep2_top1", "ep2_loop", "ep2_legacy", "ep1", "etp2")),
    "--family qwen3_moe --mode ep2",
)
_DETERMINISM_SWEEP_FAMILIES = tuple(family for family in _TINY_MOE_FAMILIES if family not in ("gpt_oss", "qwen3_moe"))
# Every syncable MoE family beyond the representative Qwen3-MoE; the roster test holds this to the
# families some rollout engine takes an online update for.
_SYNC_EXACTNESS_SWEEP_FAMILIES = (
    "bailing_moe",
    "gemma4_text",
    "glm4_moe_lite",
    "gpt_oss",
    "laguna",
    "lfm2_moe",
    "qwen3_5_moe_text",
    "step3p7",
)

# The four-rank Environmental-GRPO shapes, the same on both engines.
_ENV_GRPO_4GPU_ROWS = (
    "--ep-size 2 --etp-size 2",
    "--ep-size 2 --tp-size 2",
    "--ep-size 2 --etp-size 2 --peft lora",
    "--ep-size 4",
    "--ep-size 4 --peft expert_lora",
)

# test_lora_sft_modes.py's rows, `--mode` and its flags per `--model`: one per distinct assertion set.
_LORA_SFT_ROWS = {
    "gpt_oss_20b": (
        "ep --convergence --save-reload",
        "ep --attn flash_attention_2",
        "ep_cp --save-reload",
        "etp --save-reload",
    ),
    "qwen3_0_6b": (
        "fsdp --save-reload",
        "fsdp --qlora --save-reload",
        "cp --save-reload --resume",
        "cp --qlora --save-reload",
    ),
}

# Embedding resume, family x --lora (attention adapters = no flag; mixed / embedding adapters, DoRA on the
# attention, off = full fine-tune) x run shape. Core: every data-parallel shape on the ST encoder, FSDP2 and
# the TP refusal on a decoder, the input-embedding targets, the full fine-tune's FSDP2 / pre-sharded / TP
# reloads, and --head (a projection head after the pooling: refused under FSDP2 and TP, accepted under DDP).
# The full rows run the rest: full fine-tunes under TP too, EP's refusal of attention and embedding adapters
# beside its full fine-tune, and TP over FSDP2 on four GPUs on the TP families.
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
    # ── data ──
    "data/test_cache_isolation.py": TestSpec(nproc=2, markers=("gpu", "core", "2gpu"), timeout=600),
    "data/test_coordinated_processing.py": TestSpec(nproc=2, markers=("gpu", "core", "2gpu"), timeout=600),
    "data/test_packing_isolation.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=600),
    # 1800s: the FA4 varlen backward JITs per segment shape, and a packed batch>1 row spans many that
    # never converge to a cached set, so the budget has to cover repeated JIT rather than a warm run.
    "data/test_packing_batch_gt1.py": TestSpec(nproc=2, markers=("gpu", "core", "2gpu"), timeout=1800),
    "data/test_sft_caching_e2e.py": TestSpec(nproc=2, markers=("gpu", "core", "2gpu"), timeout=600),
    "data/test_sharded_distributed_load.py": TestSpec(nproc=2, markers=("gpu", "core", "2gpu"), timeout=600),
    # ── kernels ──
    "kernels/test_chunked_logprob_precision.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=600),
    "kernels/test_deepgemm.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=600),
    "kernels/test_fa4_trainable_sink_rescale.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=600),
    "kernels/test_fa4_tree_attention.py": TestSpec(
        nproc=1, markers=("gpu", "core", "1gpu", "moe", "gptoss", "qwen3"), timeout=900
    ),
    "kernels/test_sdpa_flex_sliding.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=900),
    "kernels/test_fused_glu.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=600),
    "kernels/test_moe_permute.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu", "moe"), timeout=600),
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
    "kernels/test_lowp_expert_lora.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu", "lora", "moe"), timeout=600),
    "kernels/test_packed_broadcast_memory.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=600),
    "kernels/test_lowp_fsdp2_weight_cache.py": TestSpec(nproc=2, markers=("gpu", "core", "2gpu"), timeout=600),
    "kernels/test_lowp_production.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=600),
    "kernels/test_weight_sync_param_buffer_cuda.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=300),
    "kernels/test_weight_sync_stream_safety.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=300),
    # ── optimizers ──
    "optimizers/test_adamw_bf16.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=600),
    "optimizers/test_bf16_optimizer_ep.py": TestSpec(
        nproc=8, markers=("gpu", "full", "8gpu", "ep", "moe", "gptoss"), timeout=1000
    ),
    "optimizers/test_flash_adamw.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=600),
    "optimizers/test_lowp_master_feasibility.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=600),
    "optimizers/test_muon.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=600),
    # Pure FSDP2 (no TP mesh anywhere), so no parallelism-axis marker, like the plain-FSDP entries.
    "optimizers/test_muon_fsdp.py": TestSpec(nproc=2, markers=("gpu", "full", "2gpu"), timeout=1000),
    "optimizers/test_muon_fsdp_qwen35.py": TestSpec(nproc=2, markers=("gpu", "full", "2gpu", "qwen3"), timeout=1000),
    # ── parallelism ──
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
        # Every shape is valid at world_size=4 and must match the single-GPU reference (EP/TP/ETP
        # feed the full sequence, so no CP aggregation).
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
    # Tiny models, one per Ulysses wrapper class, each row carrying only its own family marker.
    "parallelism/cp/test_cp_family_vs_reference.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "cp", "qwen3"),
        args_matrix=(
            "--family qwen3",
            *(
                Row(
                    f"--family {family}",
                    markers=("gpu", "core", "2gpu", "cp", "moe", _TINY_MOE_FAMILY_MARKERS[family]),
                )
                for family in _CP_MOE_FAMILIES
            ),
        ),
        timeout=900,
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
    "parallelism/ep/test_ep_buffer_backends.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "moe", "gptoss"), timeout=1500
    ),
    "parallelism/ep/test_ep_shared_arena.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "moe", "qwen3"), timeout=1800
    ),
    # No model: a bare dispatcher pair is enough to pin buffer teardown.
    "parallelism/ep/test_ep_buffer_gc_safety.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "moe"), timeout=300
    ),
    "parallelism/ep/test_ep_sort_sync_free.py": TestSpec(
        nproc=1, markers=("gpu", "core", "1gpu", "ep", "moe"), timeout=300
    ),
    # Table arithmetic against the installed extension: no model, no buffer, no collective.
    "parallelism/ep/test_deepep_config_ranks_drift.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "moe"), timeout=300
    ),
    "parallelism/ep/test_qwen3_moe_bias_balancing.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "moe", "qwen3"), timeout=1800
    ),
    "parallelism/ep/test_ep_long_context.py": TestSpec(
        nproc=8, markers=("gpu", "full", "8gpu", "ep", "moe", "gptoss"), timeout=1200
    ),
    # Each row loads its checkpoint twice: the undistributed reference on rank 0, then the parallel model.
    "parallelism/ep/test_ep_modes_vs_reference.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "ep", "tp", "etp", "moe", "gptoss", "qwen3"),
        args_matrix=(
            "--model gpt_oss_20b --mode ep",
            "--model gpt_oss_20b --mode ep_tp",
            "--model gpt_oss_20b --mode etp",
            "--model qwen3_30b_a3b --mode ep",
        ),
        timeout=1200,
    ),
    "parallelism/ep/test_ep2_weight_sync_values.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "moe", "gptoss"), timeout=1200
    ),
    "parallelism/ep/test_etp_weight_sync.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "etp", "moe", "gptoss"), timeout=600
    ),
    # The failure mode is a hang (a rank that cannot read the main process's cache file fails while
    # its peer waits in the next collective), so the timeout is the assertion: 420s covers the sweep.
    "parallelism/ep/test_ep_preference_precompute.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "ep", "moe", "qwen3"),
        args_matrix=("--trainer dpo", "--trainer kto"),
        timeout=420,
    ),
    # Four tiny-model builds and one checkpoint round-trip per row; the refusal phase raises on every
    # rank together, so a rank-local raise would surface as a hang against the timeout. The core MoE rows
    # run each Path-B layout, the dense ones FSDP2 DP, TP and a LoRA resume from the base. The full rows:
    # every other MoE family under DPO at ep2, ep1 and etp2, plus tp2 and ep2tp2 where its attention has
    # a TP plan, and KTO at ep2; then DPO at ep2etp2 on every MoE family, whose expert group needs four.
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
        ),
        timeout=900,
    ),
    # The two pooled-head trainers on a tiny Qwen3-MoE under ep2 (the bare rows), tp2 and etp2: the step-1 loss
    # against the dense objective, then save, exact resume and export.
    "parallelism/ep/test_ep_pooled_head_trainers.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "ep", "moe", "qwen3"),
        args_matrix=(
            "--trainer reward",
            "--trainer classification",
            *(
                Row(f"--trainer {trainer} --mode {mode}", markers=("gpu", "full", "2gpu", axis, "moe", "qwen3"))
                for mode, axis in (("tp2", "tp"), ("etp2", "etp"))
                for trainer in ("reward", "classification")
            ),
        ),
        timeout=900,
    ),
    # The ep1 row is the default MoE shape at ep_size=1 (fsdp_shard_ep1_experts → FSDP2 DTensor
    # experts), a different shard layout from the ep row's plain FSDP-ignored expert tensors.
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
        # Hermetic tiny models, no DeepEP dispatch: merged-from-sharded == gathered for six families,
        # and the sharded save refused for three whose names the merge cannot respell.
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
    # The Inkling row at --expert-tp-size 2 is EP+ETP on four ranks; launched by hand on two it runs pure ETP.
    "parallelism/ep/test_ep_family_vs_reference.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "ep", "moe", *_family_markers(_TINY_MOE_FAMILIES)),
        args_matrix=(
            *(f"--family {family}" for family in _TINY_MOE_FAMILIES),
            Row(
                "--family inkling_text --expert-tp-size 2",
                nproc=4,
                markers=("gpu", "full", "4gpu", "ep", "etp", "moe", "inkling"),
                timeout=900,
            ),
        ),
        timeout=1500,
    ),
    "parallelism/ep/test_ep_vlm_inkling.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "ep", "moe", "vlm", "inkling"), timeout=900
    ),
    "parallelism/ep/test_lazy_load_inkling.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "ep", "moe", "inkling"), timeout=900
    ),
    # Hub-layout composites the lazy loaders convert per key (fan-in Concatenate, scoped vision
    # tower). Markers are the union across the matrix; a family joins it when it declares
    # ``hub_conversion_keys`` and drops ``supports_lazy_loading=False``.
    "parallelism/ep/test_lazy_load_converted.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "ep", "moe", "glm5", "step3p7"),
        timeout=900,
        args_matrix=("--family glm5_next", "--family step3p7"),
    ),
    # Three loads of gpt-oss-20b: the rank-0 reference, the no-EP load on every rank, the EP model.
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
    # The two ranks are one stage's ranks (the EP group under --ep 2); each walks both pp_ranks of a
    # virtual pp2 pipeline.
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
    # Each node drives a different mask path (eager / SDPA / position_ids / FA4 varlen / Qwen3.5),
    # so the matrix must run whole. The FA4 row skips off Blackwell.
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
    # pp2 × etp2 at world 8 (two ETP groups per stage, dp2 — the deferred replica sweep engages); the
    # file also runs at 4 ranks (dp1) but the manifest gates the harder shape. fp32 throughout: a
    # bf16 twin is not a valid ETP gate.
    "parallelism/pp/test_pp_etp_vs_unsplit.py": TestSpec(
        nproc=8, markers=("gpu", "full", "8gpu", "pp", "etp", "moe", "gptoss"), timeout=1500
    ),
    # The pipeline-registered MoE families tiny_family_model builds, through the real trainer against their unsplit
    # ep1 twin: Qwen3.5-MoE on the ep1 grouped-GEMM EP layers, and each family with its experts distributed.
    "parallelism/pp/test_pp_family_vs_reference.py": TestSpec(
        nproc=4,
        markers=("gpu", "full", "4gpu", "pp", "ep", "moe", *_family_markers(_PP_MOE_FAMILIES)),
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
    # pp2 × ep2 and pp2 × etp2: save, export reload, exact resume. Markers are the union across the matrix.
    "parallelism/pp/test_pp_ep_resume.py": TestSpec(
        nproc=4,
        markers=("gpu", "full", "4gpu", "pp", "ep", "etp", "moe", "gptoss"),
        timeout=1500,
        args_matrix=("--mode ep2", "--mode etp2"),
    ),
    # Save, exact resume and export of each PP trainer beyond SFT: one row per trainer, two for offline GRPO
    # (without and with the KL reference).
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
    # A text-only run of a multimodal wrapper (tiny GLM-5 Next and Step-3.7 composites) through the
    # real SFT trainer: the image-bearing refusal, the vision tower dropped from every stage, and the
    # save reloading as the complete wrapper class.
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
        # Markers are the union across the matrix.
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
    # Hyper-connection boundary families: the [B, S, hc_mult, D] stream crosses the real P2P
    # schedule (pp2 x dp2) and both must reproduce single-GPU gradients. Markers are the union across
    # the matrix.
    "parallelism/pp/test_pp_hc_vs_single_gpu.py": TestSpec(
        nproc=4,
        markers=("gpu", "core", "4gpu", "pp", "moe", "deepseek_v4", "glm5"),
        timeout=900,
        args_matrix=("--family deepseek_v4", "--family glm5_next"),
    ),
    "parallelism/test_fsdp_tied_embeddings.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "qwen3"), timeout=600
    ),
    # Per-row scoring under FSDP2 and reentrant checkpointing: ep1 (ZeRO-2 and ZeRO-3, a nested fp32-router group)
    # on two ranks, one EP group of four and two EP groups of two on four, each full and LoRA.
    "parallelism/test_fsdp2_reentrant_recompute.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "ep", "moe", "lora", *_family_markers(_REENTRANT_FAMILIES)),
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
        ),
    ),
    # One node per family and mode; the markers are the union over the rows. EP+TP needs a single dispatch group
    # (ep_size == world): ep4+tp2 on 8 forms two 4-rank groups, the topology ParallelismConfig rejects.
    # Node-local EP+CP pins ep_size to the 8-GPU domain, cp dividing it (EP is orthogonal to DP); Mistral4 runs
    # EP+CP at ep2+cp2 in the merged-resume family rows.
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
    # nproc=2 is pure TP; also valid at nproc=4 (TP+DP), where FSDP2 hides the plain-slice/replica
    # distinction the TP grad-norm classification depends on.
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
    # ── trainers ──
    "trainers/grpo/test_environmental_grpo_benchmarks.py": TestSpec(
        nproc=1, markers=("gpu", "full", "1gpu", "qwen3", "vllm_server"), timeout=600
    ),
    "trainers/grpo/test_sglang_weight_sync_e2e.py": TestSpec(
        nproc=1, markers=("gpu", "full", "1gpu", "qwen3", "sglang_server"), timeout=900
    ),
    # Serves its own tiny hub-layout checkpoint (``--write-checkpoint``, HALO_TEST_STEP3P7_MODEL),
    # not either SERVER_TIER model — see the script header for the server launch.
    "trainers/grpo/test_step3p7_vllm_weight_sync_e2e.py": TestSpec(
        nproc=1, markers=("gpu", "full", "1gpu", "ep", "moe", "step3p7", "vllm_server"), timeout=1200
    ),
    # Every row asserts the served policy changed, so the server must run the checkpoint the row trains:
    # on vLLM Qwen3-30B on two ranks and gpt-oss on four, so those are separate marker passes
    # (SERVER_TIER='moe and not gptoss' / 'moe and gptoss'), on SGLang gpt-oss; HALO_TEST_ENV_GRPO_MODEL /
    # HALO_TEST_ENV_GRPO_SGLANG_MODEL point the two-rank rows at another family for a per-family pass.
    # --ep-size 1 gathers DTensor experts out of the FSDP2 shard, --ep-size 2 gathers FSDP-ignored plain
    # tensors; both must land in the engine's loader. The three bare vLLM rows are the per-family server
    # arms (a family pass runs them with -k "not peft and not resume and not routing"); the --peft /
    # --resume rows are the Qwen3-30B pass. --thinking-budget is not a row: it is a gpt-oss shape that
    # needs that image's reasoning plugin (agent-docs/models/gpt-oss.md#serving-for-grpo-vllm). The
    # --routing-replay rows need the server on VLLM_ENABLE_R3=1, or SGLANG_ENABLE_R3=1 with
    # SGLANG_MOE_RUNNER_BACKEND=triton, since SGLang's fused runners bypass the capture hook and return no
    # ids; the flag is additive, so one server carrying it runs every row. The SGLang --tp-size 2 and
    # --resume R3 rows are the ones whose post-sync policy produces runaway completions, the zero-gradient
    # batch the replay gate exempts. The resume rows set the two-rank budget (two model builds and a
    # checkpoint round-trip), EP+TP the four-rank one (a DTensor attention plan on top of the EP
    # layers), each with room for the FA4-JIT and checkpoint-download cold paths.
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
                )
            ),
            *_rows(
                (f"--backend vllm {args}" for args in _ENV_GRPO_4GPU_ROWS),
                nproc=4,
                markers=("gpu", "full", "4gpu", "ep", "etp", "tp", "lora", "moe", "gptoss", "vllm_server"),
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
                        "--ep-size 1 --peft lora --routing-replay rollout",
                        "--tp-size 2 --routing-replay rollout",
                        "--ep-size 1 --resume --routing-replay rollout",
                        "--ep-size 2 --peft expert_lora",
                        "--ep-size 2 --peft expert_lora --resume",
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
    "trainers/grpo/test_offline_grpo.py": TestSpec(nproc=2, markers=("gpu", "core", "2gpu", "qwen3"), timeout=600),
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
        markers=("gpu", "full", "2gpu", "cp", "ep", "moe", "qwen3", "gptoss", "cohere2_moe"),
        args_matrix=(
            "--cp-size 1",
            "--cp-size 2",
            "--cp-size 2 --ep-size 2 --family gpt_oss --fp32-masters",
            "--cp-size 2 --ep-size 2 --family cohere2_moe --fp32-masters",
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
            *_rows(("--mode tp", "--mode etp"), nproc=4, markers=("gpu", "full", "4gpu", "tp", "etp", "moe", "qwen3")),
        ),
        timeout=1800,
    ),
    "trainers/grpo/test_offline_grpo_tp_resume.py": TestSpec(
        # FA4 backward JITs per shape on Blackwell and this resume test trains 6 steps across a
        # reload, so 2400s has to cover a compile per step in both phases, not a warm run.
        nproc=2,
        markers=("gpu", "core", "2gpu", "tp", "qwen3"),
        timeout=2400,
    ),
    # The PEFT x parallelism x resume matrix for both on-policy trainers, asserted on the served policy, plus
    # the single-process rows of the online and environmental bodies. Each policy has its own server, and
    # each row asserts on logprobs only its own checkpoint's server can produce: VLLM_SERVER_URL serves
    # Qwen/Qwen3-30B-A3B-Instruct-2507 (the MoE rows), HALO_TEST_VLLM_DENSE_SERVER_URL Qwen/Qwen3-0.6B (the
    # dense and one-process rows). The --servers 2 row needs a second server of the same checkpoint in
    # HALO_TEST_VLLM_SERVER_URLS and skips without one. --resume rows run two trainers in one process (phase 2
    # is what restores TRL's -1 sync sentinel), so they load the policy twice and are the longest rows of their
    # policy: the MoE resume loads 30B twice plus the gathered checkpoint, so its rows take 1800s with room for
    # a cold cache, the dense rows 900s, the one-process rows 3000s.
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
            *_rows(
                (
                    "--trainer online --mode full_tp2",
                    "--trainer sdpg --mode full_tp2",
                    "--trainer online --mode lora_fsdp",
                    "--trainer sdpg --mode lora_fsdp",
                    "--trainer online --mode lora_tp2_rejected",
                    "--trainer sdpg --mode lora_tp2_rejected",
                    "--trainer online --mode full_fsdp_kl",
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
                ),
                nproc=1,
                markers=("gpu", "full", "1gpu", "lora", "qwen3", "vllm_server"),
                timeout=3000,
                flaky=True,
            ),
        ),
        timeout=1800,
    ),
    # Runs the dense server through a dozen trainer connect/sync/disconnect cycles on one group port (and,
    # on SGLang, one group name), which is what lets the rows above share one long-lived server. 600s
    # covers the cycles and the policy load, with room for a cold cache.
    "trainers/grpo/test_weight_transfer_reinit.py": TestSpec(
        nproc=1,
        markers=("gpu", "full", "1gpu", "qwen3", "vllm_server"),
        args_matrix=(
            "--backend vllm",
            Row("--backend sglang", markers=("gpu", "full", "1gpu", "qwen3", "sglang_server")),
        ),
        timeout=600,
    ),
    # A pool client whose first update open fails is rebuilt once on its port and re-flushed, against the
    # dense server.
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
    # `full` rather than `core`: writes a whole merged checkpoint (40+ GB at the default
    # GptOss-20B).
    "trainers/lora/test_lora_mixed_merged_save.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "lora", "ep", "moe", "gptoss"), timeout=2400
    ),
    # Tiny random-init MoE, hence `core`: both adapter shapes on the per-expert (Qwen3) and interleaved
    # (GptOss) layouts at ep2, one row each at ep1's DTensor experts, one EP+CP row, and both adapter
    # shapes trained as fp32 masters. The full rows are the rest of family x adapter shape x layout; tiny
    # models, but ~80 rows.
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
    # Tiny random-init models, no server: one dense and one MoE family under every sharding PEFT LoRA
    # syncs in, x adapter shape; the full rows run the same shapes for every other family served.
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
    # Real checkpoints, so `full`; one row per distinct assertion set (`_LORA_SFT_ROWS`).
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
        # dense LoRA/QLoRA (Qwen3-0.6B FSDP2) + attention/native expert-LoRA (GptOss-20B EP=2) +
        # attention LoRA under pure ETP (GptOss-20B, ep_size=1 + expert_tp_size=2).
        args_matrix=(
            "--mode lora",
            "--mode qlora",
            "--mode lora_ep",
            "--mode expert_lora",
            "--mode lora_etp",
        ),
        timeout=1500,
    ),
    # Two trainers per row (three on expert_lora, which also resumes the ep2 checkpoint at ep1).
    # 1800s matches the sibling adapter-resume entries, covering a cold checkpoint read and the
    # first-use FA4/flex_attention JIT.
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
    "trainers/lora/test_lora_reference_pass_fsdp2.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "lora", "qwen3"), timeout=600
    ),
    "trainers/lora/test_lora_self_distill.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "lora", "ep", "moe", "gptoss", "qwen3"),
        args_matrix=("--mode lora", "--mode qlora", "--mode lora_ep", "--mode expert_lora"),
        timeout=1500,
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
    # Frozen models and the policy under accelerate's rank-0-only loading; the row runs the file as one process and
    # `halo launch --accelerate` starts the ranks.
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
    "trainers/other/test_classification.py": TestSpec(nproc=2, markers=("gpu", "core", "2gpu", "qwen3"), timeout=600),
    "trainers/other/test_distillation.py": TestSpec(nproc=2, markers=("gpu", "core", "2gpu", "qwen3"), timeout=600),
    "trainers/other/test_distillation_oss20b.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "ep", "tp", "moe", "gptoss"), timeout=1500
    ),
    "trainers/other/test_embedding.py": TestSpec(nproc=2, markers=("gpu", "core", "2gpu"), timeout=600),
    # One row per method under scripts/training, which tests/cpu/config/test_entry_script_recipes.py holds them
    # to: `halo launch` on two ranks, a save per step, a resume and the export read back. The script starts
    # the ranks, so each row runs it as one process. The two engine-driven methods train the dense vLLM tier's
    # served checkpoint against it.
    "trainers/other/test_entry_scripts.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "qwen3"),
        args_matrix=(
            "--method classification",
            "--method distillation/self-distill",
            "--method distillation/teacher-distill",
            "--method embedding",
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
        ),
        timeout=1500,
        launcher="python",
    ),
    "trainers/other/test_reward.py": TestSpec(nproc=2, markers=("gpu", "core", "2gpu", "qwen3"), timeout=600),
    "trainers/other/test_reward_vlm_e2e.py": TestSpec(nproc=2, markers=("gpu", "core", "2gpu", "vlm"), timeout=1200),
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
    "trainers/preference/test_smpo_padding_free_segments.py": TestSpec(
        nproc=1, markers=("gpu", "core", "1gpu", "moe", "lfm2", "qwen3"), timeout=600
    ),
    "trainers/preference/test_smpo_tp.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "tp", "qwen3"), timeout=600
    ),
    "trainers/preference/test_smpo_tp_resume.py": TestSpec(
        # 2400s: FA4 backward JITs per shape over the 6 steps of this two-phase resume, so the budget
        # has to cover a compile per step rather than a warm run.
        nproc=2,
        markers=("gpu", "core", "2gpu", "tp", "qwen3"),
        timeout=2400,
    ),
    "trainers/sft/test_optimizer_shard_save_after_eval.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "ep", "moe", "vlm", "qwen3"),
        # Both modes: fsdp pins the family-agnostic FSDP2 mechanism, ep the composite-VLM +
        # plain-expert + AdamWBF16 shape.
        args_matrix=("--mode fsdp", "--mode ep"),
        timeout=900,
    ),
    "trainers/sft/test_sft_accelerate_modes.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "qwen3"), timeout=600
    ),
    # One real checkpoint per family; one node per mode, so a regression names the mode. The dense suites are
    # `core`; the MoE ones `full`, where an fsdp-only launch would leave the EP lazy-load path untested (on
    # LFM2 the non-persistent expert_bias buffers land on meta) and ep_no_gmm runs EP on the per-expert loop.
    # The markers are the union over a suite's rows, so `-m "gpu and cp"` selects the qwen3_dense fsdp and tp
    # rows too.
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
        # Every mode the script implements gets a row; an unregistered one would report coverage
        # never run. All are 2-GPU (cp_size=2 / tp_size=2 / ep_size=2 on world 2).
        nproc=2,
        # The markers are the union over the rows, so `-m "gpu and cp"` selects the fsdp and tp rows
        # too.
        markers=("gpu", "core", "2gpu", "cp", "tp", "qwen3", "gptoss"),
        args_matrix=("--mode fsdp", "--mode fsdp_base", "--mode cp", "--mode tp", "--mode ep"),
        # 2400s: the cp row trains at a 4096-token cap on auto-selected FA4, which JITs per shape, and
        # every mode runs the same 6-step two-phase resume plus two host copies of its optimizer state.
        timeout=2400,
    ),
    # Tiny random-init models, one row per family, each carrying only its own family marker.
    "trainers/sft/test_sft_ep_roundtrip.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "ep", "moe"),
        args_matrix=tuple(
            Row(f"--family {family}", markers=("gpu", "core", "2gpu", "ep", "moe", _TINY_MOE_FAMILY_MARKERS[family]))
            for family in ("cohere2_moe", "deepseek_v4", "glm5_next", "step3p7")
        ),
        timeout=1200,
    ),
    # EP+TP only (EP+TP+ETP is not a supported axis set, so this carries no `etp` marker).
    "trainers/sft/test_sft_ep_tp_flex.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "ep", "tp", "moe", "gptoss"),
        timeout=1000,
    ),
    "trainers/sft/test_sft_eval_flce.py": TestSpec(
        nproc=1, markers=("gpu", "core", "1gpu", "glm4", "moe"), timeout=600
    ),
    "trainers/sft/test_sft_fsdp_reshard.py": TestSpec(nproc=4, markers=("gpu", "core", "4gpu", "qwen3"), timeout=1200),
    # Two 4-step Qwen3-0.6B arms plus two model loads; 600s leaves headroom for a cold HF cache and
    # the FA4 kernel JIT.
    "trainers/sft/test_sft_fsdp_backward_reshard.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "qwen3"), timeout=600
    ),
    # One row per FSDP2 wrap the knob defers: dense DP / HSDP / TP+DP / CP+DP, and the three MoE
    # expert-sync regimes (FSDP-sharded ep1 experts, in-backward EP hooks, the deferred EP sweep).
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
    "trainers/sft/test_sft_hsdp.py": TestSpec(nproc=4, markers=("gpu", "core", "4gpu", "hsdp", "qwen3"), timeout=900),
    "trainers/sft/test_sft_ep_multinode_sim.py": TestSpec(
        nproc=4, markers=("gpu", "full", "4gpu", "ep", "moe", "gptoss"), timeout=1200
    ),
    "trainers/sft/test_fsdp2_pretrain_eval_then_train.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "qwen3"), timeout=900
    ),
    "trainers/sft/test_sft_gemma4_vlm.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "ep", "vlm", "moe", "gemma4"), timeout=1500
    ),
    # The families transformers pins parameters of in fp32, loaded through the single-GPU, ep1 grouped-GEMM,
    # plain FSDP2 and EP lazy loaders (the CP, TP-MoE and EP+TP sequential sites are checked statically on
    # CPU), with ep2 as the dtype control and fp32-masters rows whose pins must keep their stored values.
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
    # At ep1, trainable parameters beside a frozen parameter of another dtype stay in their shard group.
    "trainers/sft/test_sft_fsdp_excluded_params.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "lora", "moe", "glm4", "glm5"),
        args_matrix=("--case router_only", "--case glm5_next_mixed_uncast"),
        timeout=900,
    ),
    # fp32 routers at ep1 under fsdp_shard_ep1_experts, each in a nested shard group: a router the EP
    # forward calls, one with a bias, one owning no parameter itself, one read without being called,
    # and a modules_to_save copy beside bf16 adapters. Two tiny-model builds and a resume per row.
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
    # `full` rather than `core`: plain SFT smokes on the 20B checkpoint. The core tier keeps the
    # gpt-oss correctness gates (`parallelism/ep/test_ep_modes_vs_reference.py`), not smokes. One row per
    # parallel shape and attention kernel; the markers are the union over the rows, so `-m "gpu and etp"`
    # selects the other shapes too.
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
        markers=("gpu", "full", "2gpu", "moe", "gptoss"),
        args_matrix=("--mode fsdp", "--mode tp", "--mode ep"),
        timeout=1500,
    ),
    "trainers/sft/test_sft_qwen3_modes.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "tp", "cp", "qwen3"), timeout=600
    ),
    "trainers/sft/test_sft_vlm.py": TestSpec(nproc=2, markers=("gpu", "core", "2gpu", "vlm", "qwen3"), timeout=600),
    "trainers/sft/test_sft_vlm_qwen3_5.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "vlm", "qwen3"), timeout=1000
    ),
    "trainers/sft/test_zaya_fsdp.py": TestSpec(nproc=2, markers=("gpu", "full", "2gpu", "zaya"), timeout=1500),
    "trainers/sft/test_zaya_load_forward_backward.py": TestSpec(
        nproc=1, markers=("gpu", "core", "1gpu", "zaya"), timeout=1500
    ),
}

# Registered by `tests/conftest.py::pytest_configure`; every spec marker must be one of these
# (`tests/cpu/conventions/test_gpu_harness_conventions.py`), so a typo cannot drop a row from a selection.
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
    """Absolute path to a script under ``tests/gpu/``, given relative to it."""
    return GPU_DIR / rel


# Measurement entry points driven by hand from a docs recipe or by `make bench`, never by the pytest
# launcher. Listing them here is what marks an unlisted `bench*.py` as an orphan, since nothing else
# globs those files.
_UNMANIFESTED_BENCHMARKS = {
    "optimizers/bench_adamw_bf16.py",
    "optimizers/bench_muon.py",
    "optimizers/bench_muon_qwen3_5.py",
    "profiling/bench_ep_buffer_backends.py",
    "profiling/benchmark_attention_implementations.py",
    "profiling/benchmark_collators.py",
    "profiling/benchmark_convergence.py",
    "profiling/benchmark_gemma4_attention.py",
    "profiling/benchmark_grouped_mm.py",
    "profiling/benchmark_moe_block.py",
    "profiling/benchmark_roofline.py",
    "profiling/benchmark_sft_dense.py",
    "profiling/benchmark_sft_ep.py",
    "profiling/benchmark_smpo_ep.py",
    "profiling/benchmark_torch_compile.py",
    "profiling/benchmark_trl_baseline.py",
}


def unregistered_scripts() -> list[str]:
    """Executable scripts under ``tests/gpu/`` that no launch spec accounts for.

    The conftest fails collection if this is non-empty, so a new test cannot be added
    without a launch spec. The :data:`LAUNCHER_ENTRYPOINTS` are excluded, since pytest
    collects them directly rather than launching them. ``bench*.py`` files are globbed
    too, against :data:`_UNMANIFESTED_BENCHMARKS`.
    """
    on_disk = {
        str(p.relative_to(GPU_DIR))
        for pattern in ("test_*.py", "bench*.py")
        for p in GPU_DIR.rglob(pattern)
        if "__pycache__" not in p.parts
    }
    return sorted(on_disk - set(MANIFEST) - LAUNCHER_ENTRYPOINTS - _UNMANIFESTED_BENCHMARKS)


def stale_entries() -> list[str]:
    """Listed scripts that no longer exist on disk: manifest entries, launcher entry points and
    benchmarks alike."""
    listed = (*MANIFEST, *LAUNCHER_ENTRYPOINTS, *_UNMANIFESTED_BENCHMARKS)
    return sorted(rel for rel in listed if not script_path(rel).exists())
