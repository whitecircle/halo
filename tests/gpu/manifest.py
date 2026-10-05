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
    1gpu | 2gpu | 4gpu | 8gpu  — required GPU count, matches ``nproc``.
    ep cp tp etp hsdp          — parallelism axis under test.
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

from dataclasses import dataclass
from pathlib import Path

GPU_DIR = Path(__file__).parent
# The pytest modules under tests/gpu/, relative to it like the MANIFEST keys: the manifest launcher
# and its contract tests. Pytest collects these directly; everything else there is a torchrun script.
LAUNCHER_ENTRYPOINTS = frozenset({"test_suite.py", "test_launcher_contract.py"})


@dataclass(frozen=True)
class TestSpec:
    """How to launch one GPU test script.

    Attributes:
        nproc: GPUs ``torchrun`` must launch (``--nproc_per_node``).
        markers: pytest markers for selection (always includes ``"gpu"``).
        args_matrix: CLI arg-strings; the conftest generates one node per entry, so a
            multi-mode script (``--mode fsdp`` / ``--mode ep``) becomes several nodes.
            Default ``("",)`` = a single node launched with no extra args.
        timeout: seconds before the launcher kills the process group (a hard kill, since
            NCCL / FA hangs do not return).
        flaky: known-transient; the conftest applies scoped reruns.

    World-size strictness is not declared here: each script owns it via
    ``gpu_test_main(exact_world_size=N)``, which is authoritative and more precise than a
    manifest bool. The launcher always launches exactly ``nproc``.
    """

    # pytest would otherwise try to collect this as a test class (leading "Test").
    __test__ = False

    nproc: int
    markers: tuple = ()
    args_matrix: tuple = ("",)
    timeout: int = 1200
    flaky: bool = False


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
    "--trainer dpo --family dense --mode dp2",
    "--trainer dpo --family dense --mode tp2",
    "--trainer kto --family dense --mode tp2",
    "--trainer dpo --family dense --mode dp2 --peft",
    "--trainer kto --family dense --mode dp2 --peft --kto-loss apo_zero_unpaired",
)
# The core rows cover Qwen3-MoE; the family sweep runs every other family.
_PRECOMPUTE_SWEEP_FAMILIES = tuple(family for family in _TINY_MOE_FAMILIES if family != "qwen3_moe")
_PRECOMPUTE_FAMILY_ROWS = tuple(
    f"--trainer {trainer} --family {family}{mode}"
    for family in _PRECOMPUTE_SWEEP_FAMILIES
    for trainer, mode in (("dpo", ""), ("dpo", " --mode ep1"), ("kto", ""))
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
    "kernels/test_flex_sliding_attention.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=900),
    "kernels/test_fused_glu.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=600),
    "kernels/test_moe_permute.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu", "moe"), timeout=600),
    "kernels/test_family_kernel_stack_numerics.py": TestSpec(
        nproc=1,
        markers=("gpu", "full", "1gpu", "moe", *_family_markers(_TINY_MOE_FAMILIES)),
        args_matrix=tuple(f"--family {family}" for family in _TINY_MOE_FAMILIES),
        timeout=900,
    ),
    "kernels/test_grouped_gemm.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=600),
    "kernels/test_grouped_mm_empty_groups.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu", "moe"), timeout=600),
    "kernels/test_liger_family_kernels.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=900),
    "kernels/test_liger_routed_experts.py": TestSpec(
        nproc=1, markers=("gpu", "core", "1gpu", "moe", "qwen3"), timeout=600
    ),
    "kernels/test_lowp_expert_lora.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu", "lora", "moe"), timeout=600),
    "kernels/test_packed_broadcast_memory.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=600),
    "kernels/test_lowp_fsdp2_weight_cache.py": TestSpec(nproc=2, markers=("gpu", "core", "2gpu"), timeout=600),
    "kernels/test_lowp_production.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=600),
    "kernels/test_weight_sync_param_buffer_cuda.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=300),
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
    "parallelism/combined/test_ep_etp_correctness.py": TestSpec(
        # Loads gpt-oss-20b twice (undistributed reference on rank 0, then the ETP model).
        nproc=2,
        markers=("gpu", "core", "2gpu", "ep", "etp", "moe", "gptoss"),
        timeout=1200,
    ),
    "parallelism/combined/test_ep_etp_fused_glu_correctness.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "etp", "mistral4"), timeout=600
    ),
    "parallelism/combined/test_ep_tp_correctness.py": TestSpec(
        # Loads gpt-oss-20b twice (undistributed reference on rank 0, then the EP+TP model).
        nproc=2,
        markers=("gpu", "core", "2gpu", "ep", "tp", "moe", "gptoss"),
        timeout=1200,
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
    "parallelism/cp/test_qwen3_5_cp_correctness.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "cp", "qwen3"), timeout=600
    ),
    "parallelism/cp/test_glm4_cp_correctness.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "cp", "moe", "glm4"), timeout=900
    ),
    "parallelism/cp/test_bailing_cp_correctness.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "cp", "moe", "bailing"), timeout=900
    ),
    "parallelism/ep/test_ep_vs_reference_bailing_v3.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "moe", "bailing"), timeout=900
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
    "parallelism/ep/test_ep_correctness.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "moe", "gptoss"), timeout=1200
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
    # rank together, so a rank-local raise would surface as a hang against the timeout. The MoE rows
    # run each Path-B layout, the dense ones FSDP2 DP, TP and a LoRA resume from the base.
    "parallelism/ep/test_ep_preference_precompute_resume.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "ep", "etp", "tp", "lora", "moe", "qwen3"),
        args_matrix=_PRECOMPUTE_CORE_ROWS,
        timeout=900,
    ),
    "parallelism/ep/test_ep_pooled_head_trainers.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "ep", "moe", "qwen3"),
        args_matrix=("--trainer reward", "--trainer classification"),
        timeout=420,
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
        # Hermetic tiny models, no DeepEP dispatch: merged-from-sharded == gathered for six families.
        nproc=2,
        markers=("gpu", "core", "2gpu", "ep", "moe", "gptoss", "qwen3"),
        timeout=900,
    ),
    "parallelism/ep/test_routing_replay.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "moe", "gptoss"), timeout=900
    ),
    "parallelism/ep/test_ep_deterministic_expert_grads.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "ep", "etp", "moe", "gptoss", "qwen3"),
        args_matrix=_DETERMINISM_CORE_ROWS,
        timeout=600,
    ),
    "parallelism/ep/test_ep_deterministic_expert_grads_families.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "ep", "moe", *_family_markers(_DETERMINISM_SWEEP_FAMILIES)),
        args_matrix=tuple(f"--family {family}" for family in _DETERMINISM_SWEEP_FAMILIES),
        timeout=600,
    ),
    "parallelism/ep/test_ep_hook_divide_zero_token.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "moe", "qwen3"), timeout=900
    ),
    "parallelism/ep/test_ep_expert_only_rank_uniform_graph.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "ep", "moe", "lora", "gptoss", "qwen3", "zaya"),
        timeout=600,
        args_matrix=("--family qwen3_moe", "--family gpt_oss", "--family zaya"),
    ),
    "parallelism/ep/test_ep_vs_fsdp_deepseek_v4.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "moe", "deepseek_v4"), timeout=900
    ),
    "parallelism/ep/test_ep_vs_fsdp_cohere2_moe.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "moe", "cohere2_moe"), timeout=900
    ),
    "parallelism/ep/test_ep_vs_fsdp_glm5_next.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "moe", "glm5"), timeout=900
    ),
    "parallelism/ep/test_ep_vs_fsdp_step3p7.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "moe", "step3p7"), timeout=900
    ),
    "parallelism/ep/test_ep_vs_reference_inkling.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "ep", "moe", "inkling"), timeout=900
    ),
    "parallelism/combined/test_ep_etp_inkling.py": TestSpec(
        nproc=4, markers=("gpu", "full", "4gpu", "ep", "etp", "moe", "inkling"), timeout=900
    ),
    "parallelism/ep/test_ep_vlm_inkling.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "ep", "moe", "vlm", "inkling"), timeout=900
    ),
    "parallelism/ep/test_lazy_load_inkling.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "ep", "moe", "inkling"), timeout=900
    ),
    # Hub-layout composites the lazy loaders convert per key (fan-in Concatenate, scoped vision
    # tower). Markers are the union across the matrix; a family joins it when it declares
    # ``_HUB_CONVERSION_KEYS`` and drops ``_supports_lazy_loading = False``.
    "parallelism/ep/test_lazy_load_converted.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "ep", "moe", "glm5", "step3p7"),
        timeout=900,
        args_matrix=("--family glm5_next", "--family step3p7"),
    ),
    "parallelism/ep/test_ep_vs_fsdp_glm4_moe.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "moe", "glm4"), timeout=1500
    ),
    "parallelism/ep/test_ep_vs_reference_qwen3_moe.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "moe", "qwen3"), timeout=1200
    ),
    "parallelism/ep/test_ep_vs_no_ep.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "ep", "moe", "gptoss"), timeout=1000
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
    "parallelism/test_fsdp_tied_embeddings.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "qwen3"), timeout=600
    ),
    "parallelism/test_mistral4_all_parallelism.py": TestSpec(
        nproc=8,
        markers=("gpu", "full", "8gpu", "ep", "cp", "tp", "etp", "moe", "mistral4"),
        # One node per parallelism mode; --mode is required. EP+CP (ep8+cp2 is a valid single-node
        # shape — EP is orthogonal to DP) is no row here: the tiny Mistral4 runs it at ep2+cp2 in the
        # merged-resume family rows, and the cohere2_moe matrix below carries the 8-GPU ep_cp row.
        args_matrix=(
            "--mode ep --ep 8 --liger",
            "--mode cp --cp 8",
            "--mode tp --tp 8",
            # EP+TP needs a single dispatch group (ep_size == world): ep4+tp2 on 8 forms two 4-rank
            # groups, the topology ParallelismConfig rejects (combine-vs-DP-NCCL race).
            "--mode ep_tp --ep 8 --tp 2",
            # 4-way expert sharding via ETP with a benign 2-rank dispatch group. The sibling ep4+etp2
            # is a valid shape too (ETP raises ep_group_size to the domain), absent only because it
            # has not been validated on this model.
            "--mode ep_etp --ep 2 --etp 4",
        ),
        timeout=1500,
    ),
    "parallelism/test_cohere2_moe_all_parallelism.py": TestSpec(
        nproc=8,
        markers=("gpu", "full", "8gpu", "ep", "cp", "tp", "etp", "moe", "cohere2_moe"),
        # One node per parallelism mode; --mode is required.
        args_matrix=(
            "--mode ep --ep 8",
            "--mode cp --cp 8",
            "--mode tp --tp 8",
            # Node-local EP+CP pins ep_size to the 8-GPU domain; cp divides it (EP is orthogonal to DP).
            "--mode ep_cp --ep 8 --cp 2",
            # EP+TP needs a single dispatch group (ep_size == world): ep4+tp2 on 8 forms two 4-rank
            # groups, the topology ParallelismConfig rejects (combine-vs-DP-NCCL race).
            "--mode ep_tp --ep 8 --tp 2",
            # Pure ETP (ep_size=1, experts replicated, FFN sharded 8-way) and EP+ETP with a benign
            # 2-rank dispatch group and one domain-wide EP group.
            "--mode etp --etp 8",
            "--mode ep_etp --ep 2 --etp 4",
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
    "parallelism/tp/test_vlm_parallelism.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "vlm", "tp", "cp", "qwen3"), timeout=1000
    ),
    # ── trainers ──
    "trainers/grpo/test_environmental_grpo_benchmarks.py": TestSpec(
        nproc=1, markers=("gpu", "full", "1gpu", "qwen3", "vllm_server"), timeout=600
    ),
    "trainers/grpo/test_sglang_weight_sync_e2e.py": TestSpec(
        nproc=1, markers=("gpu", "full", "1gpu", "qwen3", "sglang_server"), timeout=900
    ),
    # The entries below assert the served policy changed, so the server must run the same checkpoint
    # the test trains; the defaults differ per engine (VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507,
    # SGLANG_MODEL=unsloth/gpt-oss-20b-BF16), and HALO_TEST_ENV_GRPO_MODEL / HALO_TEST_ENV_GRPO_SGLANG_MODEL
    # point a wrapper at another family for a per-family pass.
    # Serves its own tiny hub-layout checkpoint (``--write-checkpoint``, HALO_TEST_STEP3P7_MODEL),
    # not either SERVER_TIER model — see the script header for the server launch.
    "trainers/grpo/test_step3p7_vllm_weight_sync_e2e.py": TestSpec(
        nproc=1, markers=("gpu", "full", "1gpu", "ep", "moe", "step3p7", "vllm_server"), timeout=1200
    ),
    # --ep-size 1 gathers DTensor experts out of the FSDP2 shard, --ep-size 2 gathers FSDP-ignored
    # plain tensors; both must land in the engine's loader. The three bare rows are the per-family
    # server arms (a family pass runs them with -k "not peft and not resume and not routing"); the
    # --peft / --resume rows are the Qwen3-30B pass. --thinking-budget is not a row: it is a gpt-oss
    # shape that needs that image's reasoning plugin (agent-docs/models/gpt-oss.md#serving-for-grpo-vllm).
    # The --routing-replay rows need the server on VLLM_ENABLE_R3=1; the flag is additive.
    "trainers/grpo/test_env_grpo_vllm_e2e.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "ep", "etp", "tp", "lora", "moe", "qwen3", "vllm_server"),
        args_matrix=(
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
        ),
        # The full-finetune resume sets the budget: two model builds plus a checkpoint round-trip.
        # 2400s covers the FA4-JIT and checkpoint-download cold paths on top of it.
        timeout=2400,
    ),
    # Two axes at once: the shapes two ranks cannot form. Same URL and server GPU as the 2-GPU entry
    # above but a different checkpoint, so the two are separate marker passes:
    # SERVER_TIER='moe and not gptoss' against Qwen3-30B, 'moe and gptoss' against gpt-oss.
    "trainers/grpo/test_env_grpo_vllm_4gpu_e2e.py": TestSpec(
        nproc=4,
        markers=("gpu", "full", "4gpu", "ep", "etp", "tp", "lora", "moe", "gptoss", "vllm_server"),
        args_matrix=(
            "--ep-size 2 --etp-size 2",
            "--ep-size 2 --tp-size 2",
            "--ep-size 2 --etp-size 2 --peft lora",
            "--ep-size 4",
            "--ep-size 4 --peft expert_lora",
        ),
        # EP+TP sets the budget: a DTensor attention plan on top of the EP wrappers. 1800s covers the
        # FA4-JIT and checkpoint-download cold paths on top of it.
        timeout=1800,
    ),
    # The SGLang counterpart of the 4-GPU entry above: the same two-axis shapes into the engine
    # whose loader assembles nothing itself. Same server as the 2-GPU SGLang entry.
    "trainers/grpo/test_env_grpo_sglang_4gpu_e2e.py": TestSpec(
        nproc=4,
        markers=("gpu", "full", "4gpu", "ep", "etp", "tp", "lora", "moe", "gptoss", "sglang_server"),
        args_matrix=(
            "--ep-size 2 --etp-size 2",
            "--ep-size 2 --tp-size 2",
            "--ep-size 2 --etp-size 2 --peft lora",
            "--ep-size 4",
            "--ep-size 4 --peft expert_lora",
        ),
        timeout=1800,
    ),
    # --ep-size 2 gathers FSDP-ignored plain experts through DeepEP with the SGLang group in the same
    # process; the server's cuMem parity (docker-compose.sglang.yml) is what lets the two coexist.
    "trainers/grpo/test_env_grpo_sglang_e2e.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "ep", "tp", "lora", "moe", "gptoss", "sglang_server"),
        args_matrix=(
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
        ),
        # The --routing-replay rows need SGLANG_ENABLE_R3=1 with SGLANG_MOE_RUNNER_BACKEND=triton,
        # since the fused runners bypass the capture hook and return no ids; the flag is additive, so
        # one server carrying it runs the whole entry. The --tp-size 2 and --resume R3 rows are the ones
        # whose post-sync policy produces runaway completions, the zero-gradient batch the replay gate
        # exempts. The resume rows set the budget: two model builds and a checkpoint round-trip.
        timeout=2400,
        flaky=True,
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
    "trainers/grpo/test_offline_grpo_tp_resume.py": TestSpec(
        # FA4 backward JITs per shape on Blackwell and this resume test trains 6 steps across a
        # reload, so 2400s has to cover a compile per step in both phases, not a warm run.
        nproc=2,
        markers=("gpu", "core", "2gpu", "tp", "qwen3"),
        timeout=2400,
    ),
    # One node per leg: each leg holds its trainer-side weight-transfer port for the life of the
    # process (only close_communicator frees it, which a leg never calls), and the environmental legs
    # additionally stand up Ray actors.
    "trainers/grpo/test_online_grpo_vllm_e2e.py": TestSpec(
        nproc=1,
        markers=("gpu", "full", "1gpu", "lora", "qwen3", "vllm_server"),
        args_matrix=(
            "--mode online",
            "--mode sdpg",
            "--mode online_lora",
            "--mode environmental",
            "--mode environmental_lora",
        ),
        timeout=3000,
        flaky=True,
    ),
    # The PEFT x parallelism x resume pair for both on-policy trainers, asserted on the served policy.
    # Two servers: VLLM_SERVER_URL must serve Qwen/Qwen3-30B-A3B-Instruct-2507 and
    # HALO_TEST_VLLM_DENSE_SERVER_URL Qwen/Qwen3-0.6B — each file asserts on logprobs only its own
    # checkpoint's server can produce.
    # --resume rows run two trainers in one process (phase 2 is what restores TRL's -1 sync sentinel),
    # so they load the policy twice and are the longest rows in each file.
    "trainers/grpo/test_online_grpo_vllm_moe_e2e.py": TestSpec(
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
        ),
        # The resume row sets the budget: it loads 30B twice plus the gathered checkpoint. 1800s
        # leaves room for a cold cache on top of that.
        timeout=1800,
    ),
    "trainers/grpo/test_online_grpo_vllm_dense_e2e.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "lora", "tp", "qwen3", "vllm_server"),
        args_matrix=(
            "--trainer online --mode full_tp2",
            "--trainer sdpg --mode full_tp2",
            "--trainer online --mode lora_fsdp",
            "--trainer sdpg --mode lora_fsdp",
            "--trainer online --mode lora_tp2_rejected",
            "--trainer sdpg --mode lora_tp2_rejected",
            "--trainer online --mode full_tp2 --resume",
            "--trainer sdpg --mode full_tp2 --resume",
            "--trainer online --mode lora_fsdp --resume",
            "--trainer sdpg --mode lora_fsdp --resume",
        ),
        # 900s leaves room for a cold cache on top of the short dense-policy rows.
        timeout=900,
    ),
    # Runs the dense server through a dozen trainer connect/sync/disconnect cycles, which is what
    # lets the rows above share one long-lived server.
    "trainers/grpo/test_vllm_weight_transfer_reinit.py": TestSpec(
        nproc=1,
        markers=("gpu", "full", "1gpu", "qwen3", "vllm_server"),
        # 600s covers the cycles and the policy load, with room for a cold cache.
        timeout=600,
    ),
    # Embedding resume, family x run shape x --lora (attention / mixed / embedding adapters, DoRA on the
    # attention, off = full fine-tune). Core: every data-parallel shape on the ST encoder, FSDP2 and the TP refusal on a
    # decoder, the input-embedding targets and the full fine-tune's FSDP2 / pre-sharded / TP reloads,
    # and --head: a projection head after the pooling, refused under FSDP2 and TP, accepted under DDP;
    # the roster scripts carry the other families and the EP rows.
    "trainers/lora/test_embedding_lora_resume.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "lora", "tp", "qwen3"),
        args_matrix=(
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
        ),
        timeout=900,
    ),
    "trainers/lora/test_embedding_lora_resume_1gpu.py": TestSpec(
        nproc=1,
        markers=("gpu", "core", "1gpu", "lora", "qwen3"),
        args_matrix=(
            "--family bert",
            "--family qwen3",
            "--family qwen3 --lora embedding",
            "--family qwen3 --lora dora",
            "--family qwen3 --lora off",
        ),
        timeout=600,
    ),
    "trainers/lora/test_embedding_lora_resume_4gpu.py": TestSpec(
        nproc=4,
        markers=("gpu", "full", "4gpu", "tp", "moe", "qwen3", "gptoss"),
        args_matrix=("--family qwen3", "--family qwen3_5", "--family gpt_oss"),
        timeout=900,
    ),
    "trainers/lora/test_embedding_lora_resume_roster.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "lora", "tp", "ep", "moe", "qwen3", "gemma4", "gptoss"),
        args_matrix=(
            "--family qwen3 --mode ddp",
            "--family qwen3 --mode presharded",
            "--family qwen3_5 --mode fsdp",
            "--family qwen3_5 --mode ddp",
            "--family qwen3_5 --mode presharded",
            "--family gemma4 --mode fsdp",
            "--family gemma4 --mode ddp",
            "--family gemma4 --mode presharded",
            "--family gemma4 --mode ep",
            "--family gpt_oss --mode fsdp",
            "--family gpt_oss --mode ddp",
            "--family gpt_oss --mode presharded",
            "--family gpt_oss --mode ep",
            "--family bert --mode fsdp --lora mixed",
            "--family bert --mode presharded --lora mixed",
            "--family qwen3 --mode ddp --lora mixed",
            "--family qwen3 --mode presharded --lora mixed",
            "--family qwen3_5 --mode fsdp --lora mixed",
            "--family qwen3_5 --mode ddp --lora mixed",
            "--family qwen3_5 --mode presharded --lora mixed",
            "--family gemma4 --mode fsdp --lora mixed",
            "--family gemma4 --mode ddp --lora mixed",
            "--family gemma4 --mode presharded --lora mixed",
            "--family gpt_oss --mode fsdp --lora mixed",
            "--family gpt_oss --mode ddp --lora mixed",
            "--family gpt_oss --mode presharded --lora mixed",
            "--family bert --mode fsdp --lora embedding",
            "--family bert --mode ddp --lora embedding",
            "--family bert --mode presharded --lora embedding",
            "--family qwen3 --mode ddp --lora embedding",
            "--family qwen3 --mode presharded --lora embedding",
            "--family qwen3_5 --mode fsdp --lora embedding",
            "--family qwen3_5 --mode ddp --lora embedding",
            "--family qwen3_5 --mode presharded --lora embedding",
            "--family gemma4 --mode fsdp --lora embedding",
            "--family gemma4 --mode ddp --lora embedding",
            "--family gemma4 --mode presharded --lora embedding",
            "--family gemma4 --mode ep --lora embedding",
            "--family gpt_oss --mode fsdp --lora embedding",
            "--family gpt_oss --mode ddp --lora embedding",
            "--family gpt_oss --mode presharded --lora embedding",
            "--family gpt_oss --mode ep --lora embedding",
            "--family bert --mode fsdp --lora dora",
            "--family bert --mode ddp --lora dora",
            "--family bert --mode presharded --lora dora",
            "--family qwen3 --mode ddp --lora dora",
            "--family qwen3 --mode presharded --lora dora",
            "--family qwen3_5 --mode fsdp --lora dora",
            "--family qwen3_5 --mode ddp --lora dora",
            "--family qwen3_5 --mode presharded --lora dora",
            "--family gemma4 --mode fsdp --lora dora",
            "--family gemma4 --mode ddp --lora dora",
            "--family gemma4 --mode presharded --lora dora",
            "--family gpt_oss --mode fsdp --lora dora",
            "--family gpt_oss --mode ddp --lora dora",
            "--family gpt_oss --mode presharded --lora dora",
            "--family bert --mode fsdp --lora off",
            "--family bert --mode ddp --lora off",
            "--family bert --mode presharded --lora off",
            "--family qwen3 --mode ddp --lora off",
            "--family qwen3_5 --mode fsdp --lora off",
            "--family qwen3_5 --mode ddp --lora off",
            "--family qwen3_5 --mode presharded --lora off",
            "--family qwen3_5 --mode tp --lora off",
            "--family gemma4 --mode fsdp --lora off",
            "--family gemma4 --mode ddp --lora off",
            "--family gemma4 --mode presharded --lora off",
            "--family gemma4 --mode ep --lora off",
            "--family gpt_oss --mode fsdp --lora off",
            "--family gpt_oss --mode ddp --lora off",
            "--family gpt_oss --mode presharded --lora off",
            "--family gpt_oss --mode tp --lora off",
            "--family gpt_oss --mode ep --lora off",
        ),
        timeout=1200,
    ),
    "trainers/lora/test_embedding_lora_resume_roster_1gpu.py": TestSpec(
        nproc=1,
        markers=("gpu", "full", "1gpu", "lora", "moe", "qwen3", "gemma4", "gptoss"),
        args_matrix=(
            "--family qwen3_5",
            "--family gemma4",
            "--family gpt_oss",
            "--family bert --lora mixed",
            "--family qwen3 --lora mixed",
            "--family qwen3_5 --lora mixed",
            "--family gemma4 --lora mixed",
            "--family gpt_oss --lora mixed",
            "--family bert --lora embedding",
            "--family qwen3_5 --lora embedding",
            "--family gemma4 --lora embedding",
            "--family gpt_oss --lora embedding",
            "--family bert --lora dora",
            "--family qwen3_5 --lora dora",
            "--family gemma4 --lora dora",
            "--family gpt_oss --lora dora",
            "--family bert --lora off",
            "--family qwen3_5 --lora off",
            "--family gemma4 --lora off",
            "--family gpt_oss --lora off",
        ),
        timeout=900,
    ),
    "trainers/lora/test_lora_bailing_moe.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "lora", "moe", "etp", "bailing"), timeout=1500
    ),
    "trainers/lora/test_lora_cp_dense.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "lora", "cp", "qwen3"), timeout=1300
    ),
    "trainers/lora/test_lora_cp_tp.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "lora", "cp", "tp", "qwen3"), timeout=1000
    ),
    "trainers/lora/test_lora_ep.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "lora", "ep", "moe", "gptoss"), timeout=1000
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
    # shapes trained as fp32 masters.
    "trainers/lora/test_lora_merged_save_resume.py": TestSpec(
        nproc=2,
        markers=("gpu", "core", "2gpu", "lora", "ep", "cp", "moe", "qwen3", "gptoss"),
        args_matrix=_MERGED_RESUME_CORE_ROWS,
        timeout=1200,
    ),
    # The rest of family x adapter shape x layout; tiny models, but ~80 rows.
    "trainers/lora/test_lora_merged_save_resume_families.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "lora", "ep", "cp", "moe", *_family_markers(_TINY_MOE_FAMILIES)),
        args_matrix=_MERGED_RESUME_FAMILY_ROWS,
        timeout=1200,
    ),
    # Tiny random-init models, no server: one dense and one MoE family under every sharding PEFT LoRA
    # syncs in, x adapter shape; the sweep below runs the same rows for every other family served.
    # tests/cpu/conventions/test_tiny_family_roster.py holds both to the syncable tiny-family roster.
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
        ),
        timeout=900,
    ),
    "trainers/lora/test_lora_weight_sync_exact_families.py": TestSpec(
        nproc=2,
        # qwen3 also marks the dense Qwen3.5 row.
        markers=("gpu", "full", "2gpu", "lora", "ep", "etp", "moe", *_family_markers(_SYNC_EXACTNESS_SWEEP_FAMILIES)),
        args_matrix=(
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
        timeout=900,
    ),
    "trainers/lora/test_lora_ep_convergence.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "lora", "ep", "moe", "gptoss"), timeout=1000
    ),
    "trainers/lora/test_lora_ep_cp_etp.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "lora", "ep", "cp", "etp", "moe", "gptoss"), timeout=1000
    ),
    "trainers/lora/test_lora_tp_save_load.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "lora", "tp", "ep", "moe", "qwen3", "gptoss"),
        timeout=1000,
    ),
    "trainers/lora/test_sft_oss20b_ep_lora.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "lora", "ep", "moe", "gptoss"), timeout=1500
    ),
    "trainers/lora/test_sft_qwen3_4b_lora.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "lora", "tp", "qwen3"), timeout=1000
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
    "trainers/other/test_checkpoint_roundtrip_gptoss_20b.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "ep", "tp", "moe", "gptoss"),
        timeout=1500,
        # Every implemented mode gets a row; an unregistered one would report coverage never run.
        args_matrix=("--mode ep2", "--mode ep2_tp2", "--mode ep2_no_gmm", "--mode etp"),
    ),
    "trainers/other/test_checkpoint_roundtrip_qwen3_8b.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "tp", "cp", "qwen3"),
        timeout=1500,
        args_matrix=("--mode fsdp", "--mode tp2", "--mode cp2"),
    ),
    "trainers/other/test_checkpoint_save_load.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "cp", "tp", "qwen3"), timeout=1500
    ),
    "trainers/other/test_classification.py": TestSpec(nproc=2, markers=("gpu", "core", "2gpu", "qwen3"), timeout=600),
    "trainers/other/test_distillation.py": TestSpec(nproc=2, markers=("gpu", "core", "2gpu", "qwen3"), timeout=600),
    "trainers/other/test_distillation_oss20b.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "ep", "tp", "moe", "gptoss"), timeout=1500
    ),
    "trainers/other/test_embedding.py": TestSpec(nproc=2, markers=("gpu", "core", "2gpu"), timeout=600),
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
    # Every other MoE family: DPO at ep2 and ep1, KTO at ep2.
    "trainers/preference/test_preference_precompute_resume_families.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "ep", "moe", *_family_markers(_PRECOMPUTE_SWEEP_FAMILIES)),
        args_matrix=_PRECOMPUTE_FAMILY_ROWS,
        timeout=900,
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
    "trainers/preference/test_smpo_fsdp.py": TestSpec(nproc=2, markers=("gpu", "core", "2gpu", "qwen3"), timeout=600),
    "trainers/preference/test_smpo_padding_free.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "qwen3"), timeout=600
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
        # Both modes: fsdp pins the family-agnostic FSDP2 mechanism, ep the reported
        # composite-VLM + plain-expert + AdamWBF16 shape.
        args_matrix=("--mode fsdp", "--mode ep"),
        timeout=900,
    ),
    "trainers/sft/test_sft_accelerate_modes.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "qwen3"), timeout=600
    ),
    "trainers/sft/test_sft_bailing_moe.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "ep", "moe", "bailing"),
        # Both modes: an fsdp-only launch leaves the EP lazy-load path untested.
        args_matrix=("--mode fsdp", "--mode ep"),
        timeout=1500,
    ),
    "trainers/sft/test_sft_checkpoint_resume.py": TestSpec(
        # Every mode the script implements gets a row; an unregistered one would report coverage
        # never run. All four are 2-GPU (cp_size=2 / tp_size=2 / ep_size=2 on world 2).
        nproc=2,
        # The markers are the union over the rows, so `-m "gpu and cp"` selects the fsdp and tp rows
        # too.
        markers=("gpu", "core", "2gpu", "cp", "tp", "qwen3", "gptoss"),
        args_matrix=("--mode fsdp", "--mode cp", "--mode tp", "--mode ep"),
        # 2400s: the cp row trains at MAX_SEQ_LENGTH_CP=4096 on auto-selected FA4, which JITs per
        # shape, and every mode runs the same 6-step two-phase resume.
        timeout=2400,
    ),
    "trainers/sft/test_sft_ep_fa2_modes.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "ep", "moe", "gptoss"),
        args_matrix=("--mode full", "--mode lora", "--mode qlora"),
        timeout=1000,
    ),
    "trainers/sft/test_sft_ep_flex_vs_fa2.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "ep", "moe", "gptoss"),
        # FA2 needs --reset_sinks: validate_attn_implementation rejects FA2 while sinks are live, so
        # without the flag the test fails on its own premise.
        args_matrix=("--mode flex", "--mode fa2 --reset_sinks"),
        timeout=1000,
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
    "trainers/sft/test_sft_fsdp_resume.py": TestSpec(nproc=2, markers=("gpu", "core", "2gpu", "qwen3"), timeout=1500),
    "trainers/sft/test_fsdp2_pretrain_eval_then_train.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "qwen3"), timeout=900
    ),
    "trainers/sft/test_sft_gemma4_moe.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "ep", "moe", "gemma4"), timeout=1500
    ),
    "trainers/sft/test_sft_gemma4_vlm.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "ep", "vlm", "moe", "gemma4"), timeout=1500
    ),
    "trainers/sft/test_sft_deepseek_v4_moe.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "moe", "deepseek_v4"), timeout=1200
    ),
    "trainers/sft/test_sft_cohere2_moe.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "moe", "cohere2_moe"), timeout=1200
    ),
    "trainers/sft/test_sft_glm5_next.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "moe", "glm5"), timeout=1200
    ),
    # The families transformers pins parameters of in fp32, loaded through the ep1 grouped-GEMM, plain
    # FSDP2 and EP lazy loaders (the CP, TP-MoE and EP+TP sequential sites are checked statically on CPU),
    # with ep2 as the dtype control and fp32-masters rows whose pins must keep their stored values.
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
    "trainers/sft/test_sft_fp32_pinned_params_single_gpu.py": TestSpec(
        nproc=1,
        markers=("gpu", "core", "1gpu", "moe", "deepseek_v4"),
        args_matrix=("--family deepseek_v4",),
        timeout=600,
    ),
    "trainers/sft/test_sft_step3p7.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "moe", "step3p7"), timeout=1200
    ),
    "trainers/sft/test_sft_glm4_moe.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "ep", "moe", "glm4"), timeout=1500
    ),
    "trainers/sft/test_sft_lfm2_moe.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "ep", "moe", "lfm2"),
        # Both modes: an fsdp-only launch leaves the EP lazy-load path untested (non-persistent
        # expert_bias buffers land on meta).
        args_matrix=("--mode fsdp", "--mode ep"),
        timeout=1500,
    ),
    # `full` rather than `core`: plain SFT smokes on the 20B checkpoint. The core tier keeps the
    # gpt-oss correctness gates (`parallelism/ep/test_ep_correctness.py`), not smokes.
    "trainers/sft/test_sft_oss20b_default.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "gptoss"), timeout=1500
    ),
    # One row per parallel shape. The markers are the union over the rows, so `-m "gpu and etp"` selects
    # the other shapes too.
    "trainers/sft/test_sft_gptoss_modes.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "ep", "cp", "tp", "etp", "moe", "gptoss"),
        args_matrix=(
            "--mode fsdp",
            "--mode ep",
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
    "trainers/sft/test_sft_qwen3_5_dense.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "tp", "qwen3"), timeout=600
    ),
    "trainers/sft/test_sft_qwen3_5_ep.py": TestSpec(
        nproc=4, markers=("gpu", "full", "4gpu", "ep", "moe", "qwen3"), timeout=1500
    ),
    "trainers/sft/test_sft_qwen3_5_moe.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "ep", "tp", "etp", "moe", "qwen3"), timeout=1500
    ),
    "trainers/sft/test_sft_qwen3_dense.py": TestSpec(
        nproc=2,
        # One node per mode, so a regression names the mode instead of collapsing three verdicts into
        # one. The markers are the union over the rows, and every row is the same dense model, so
        # `-m "gpu and cp"` selects the fsdp and tp rows too.
        markers=("gpu", "core", "2gpu", "tp", "cp", "qwen3"),
        args_matrix=("--mode fsdp", "--mode tp", "--mode cp"),
        timeout=900,
    ),
    "trainers/sft/test_sft_qwen3_modes.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "tp", "cp", "qwen3"), timeout=600
    ),
    "trainers/sft/test_sft_qwen3_moe.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "ep", "moe", "qwen3"), timeout=1000
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

# Registered by `tests/conftest.py::pytest_configure` so `--strict-markers` rejects typos.
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


# Measurement entry points driven by hand from a docs recipe or a `tests/gpu/profiling/run_*.sh`
# runner, never by the pytest launcher. Listing them here is what marks an unlisted `bench*.py` as an
# orphan, since nothing else globs those files.
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
    "profiling/benchmark_offline_grpo_ep.py",
    "profiling/benchmark_roofline.py",
    "profiling/benchmark_sft_dense.py",
    "profiling/benchmark_sft_ep.py",
    "profiling/benchmark_sft_ep_cp.py",
    "profiling/benchmark_sft_ep_tp.py",
    "profiling/benchmark_smpo_ep.py",
    "profiling/benchmark_smpo_ep_cp.py",
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
