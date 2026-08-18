"""Launch specs for the GPU test suite.

GPU tests are external ``torchrun`` scripts (each ends in ``sys.exit(main())``), so
pytest cannot read ``nproc`` / markers / timeout from inside them. This manifest maps
each script (path relative to ``tests/gpu/``) to its launch spec; ``tests/gpu/conftest.py``
reads it and generates one pytest node per ``(script, args)`` with the right markers,
process count and timeout. The launcher shells out ``torchrun --nproc_per_node=<nproc>``
and asserts the exit code, parsing the structured result line when the script uses
``tests.common.harness.gpu_test_main``.

To add a test: drop the script under ``tests/gpu/`` and add one ``TestSpec`` line here.
A script present on disk but missing from the manifest is reported by
:func:`unregistered_scripts`, and the conftest fails collection on that drift.

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
                                   * on a host without InfiniBand, launch with ``NCCL_IB_DISABLE=1
                                     NCCL_NET=Socket``. The image's OFI/Gin defaults hang the
                                     cross-container group instead of failing: both GPUs spin until
                                     the 120 s formation deadline, per test.
                                 A trainer killed while attached to the weight-transfer engine leaves
                                 the server's scheduler stuck while ``/health`` still answers 200;
                                 restart the container before re-running.
                                 Both server tiers split by ``moe``: these tests broadcast the
                                 trainer's own weights into the served model and assert the served
                                 policy changed, so the server must serve the checkpoint being
                                 trained: dense (Qwen3-0.6B) for the ``not moe`` half, Qwen3-30B-A3B
                                 for the ``moe`` half. No single server satisfies both; run two passes
                                 (``make test-gpu-vllm`` then ``... SERVER_TIER=moe``).
    <model family>             — gptoss / qwen3 / glm4 / glm5 / gemma4 / mistral4 / mistral3 /
                                 bailing / lfm2 / zaya / deepseek_v4 / inkling / cohere2_moe /
                                 step3p7.
"""

from dataclasses import dataclass
from pathlib import Path

_GPU_DIR = Path(__file__).parent


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
    "kernels/test_deepgemm.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=600),
    "kernels/test_fa4_trainable_sink_rescale.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=600),
    "kernels/test_fused_glu.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=600),
    "kernels/test_grouped_gemm.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=600),
    "kernels/test_grouped_mm_empty_groups.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu", "moe"), timeout=600),
    "kernels/test_liger_family_kernels.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=900),
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
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "tp", "gptoss"), timeout=600
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
    "parallelism/cp/test_cp_rejection.py": TestSpec(nproc=2, markers=("gpu", "core", "2gpu", "cp"), timeout=600),
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
    "parallelism/ep/test_ep_gradient_checkpointing.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "moe", "gptoss"), timeout=1500
    ),
    "parallelism/ep/test_ep2_weight_sync_values.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "moe", "gptoss"), timeout=1200
    ),
    "parallelism/ep/test_etp_weight_sync.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "etp", "moe", "gptoss"), timeout=600
    ),
    # The failure mode is a hang (a rank-divergent cache path leaves non-main ranks short of the
    # sweep's collectives), so the timeout is the assertion: 420s covers the sweep and fails fast.
    "parallelism/ep/test_ep_preference_precompute.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "ep", "moe", "qwen3"),
        args_matrix=("--trainer dpo", "--trainer kto"),
        timeout=420,
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
        markers=("gpu", "core", "2gpu", "ep", "cp", "moe", "qwen3"),
        args_matrix=("--mode ep", "--mode ep1", "--mode cp"),
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
    "parallelism/ep/test_ep_hook_divide_zero_token.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "moe", "qwen3"), timeout=900
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
    "parallelism/ep/test_grouped_mm_b300.py": TestSpec(nproc=1, markers=("gpu", "core", "1gpu"), timeout=600),
    "parallelism/ep/test_zaya_ep.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "ep", "moe", "zaya"), timeout=1500
    ),
    "parallelism/ep/test_zaya_ep_save_roundtrip.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "ep", "moe", "zaya"), timeout=1500
    ),
    "parallelism/test_fsdp_tied_embeddings.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "qwen3"), timeout=600
    ),
    "parallelism/test_mistral3_vision_smoke.py": TestSpec(
        nproc=1, markers=("gpu", "core", "1gpu", "vlm", "mistral3"), timeout=600
    ),
    "parallelism/test_mistral4_all_parallelism.py": TestSpec(
        nproc=8,
        markers=("gpu", "full", "8gpu", "ep", "cp", "tp", "etp", "moe", "mistral4"),
        # One node per parallelism mode; --mode is required. EP+CP (ep8+cp2 is a valid single-node
        # shape — EP is orthogonal to DP) has not been run for this model; the cohere2_moe matrix
        # below carries the single-node ep_cp coverage.
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
    "parallelism/test_parallelism_config.py": TestSpec(nproc=2, markers=("gpu", "core", "2gpu"), timeout=600),
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
    # the test trains, and the engines differ: VLLM_MODEL=Qwen/Qwen3-30B-A3B-Instruct-2507,
    # SGLANG_MODEL=unsloth/gpt-oss-20b-BF16 (the only family whose EP layer gathers SGLang's fused
    # layout; every other MoE family is refused for that backend at construction).
    # Serves its own tiny hub-layout checkpoint (``--write-checkpoint``, HALO_TEST_STEP3P7_MODEL),
    # not either SERVER_TIER model — see the script header for the server launch.
    "trainers/grpo/test_step3p7_vllm_weight_sync_e2e.py": TestSpec(
        nproc=1, markers=("gpu", "full", "1gpu", "ep", "moe", "step3p7", "vllm_server"), timeout=1200
    ),
    # --ep-size 1 gathers DTensor experts out of the FSDP2 shard, --ep-size 2 gathers FSDP-ignored
    # plain tensors; both must land in the engine's loader.
    # The three bare rows are the per-family server arms (a family pass runs them with
    # -k "not peft and not resume and not thinking"); the --peft / --resume rows are the Qwen3-30B pass.
    # --thinking-budget is not a row: the budget is a gpt-oss shape (its arming marker lives in the
    # server image's reasoning plugin, agent-docs/models/gpt-oss.md#serving-for-grpo-vllm) while this
    # file's server runs Qwen3-30B. It needs HALO_TEST_ENV_GRPO_MODEL and VLLM_MODEL both pointed at
    # a gpt-oss checkpoint, against a server carrying that plugin.
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
    # No --ep-size 2 row: SGLang weight sync and DeepEP need opposite process-global NCCL transport
    # settings, so the trainer refuses that pairing at construction (validate_backend_parallelism).
    "trainers/grpo/test_env_grpo_sglang_e2e.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "tp", "lora", "moe", "gptoss", "sglang_server"),
        args_matrix=(
            "--ep-size 1",
            "--tp-size 2",
            "--ep-size 1 --peft lora",
            "--ep-size 1 --resume",
            "--ep-size 1 --routing-replay rollout",
            "--ep-size 1 --peft lora --routing-replay rollout",
            "--tp-size 2 --routing-replay rollout",
            "--ep-size 1 --resume --routing-replay rollout",
        ),
        # The --routing-replay rows need more of the server than the others: SGLANG_ENABLE_R3=1
        # (--enable-return-routed-experts) with SGLANG_MOE_RUNNER_BACKEND=triton, since the fused
        # runners bypass the capture hook and return no ids. The flag is additive, so one server
        # carrying it runs the whole entry. The last two are the R3 rows whose post-sync policy
        # produces runaway completions: every turn is cut at the token cap and excluded, giving the
        # zero-gradient batch the replay gate exempts (``_assemble_rollout_routing``).
        # The resume row sets the budget: two model builds plus a checkpoint round-trip, on an engine
        # that quiesces for every sync.
        timeout=2400,
        flaky=True,
    ),
    "trainers/grpo/test_environmental_grpo_mock.py": TestSpec(nproc=2, markers=("gpu", "core", "2gpu"), timeout=600),
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
    # No family marker: nothing here loads a checkpoint (online GRPO refuses to construct without a
    # vLLM server), so the node is config/class-surface plus reward extraction.
    "trainers/grpo/test_online_grpo_mock.py": TestSpec(nproc=2, markers=("gpu", "core", "2gpu"), timeout=600),
    # One node per leg: each leg holds its trainer-side weight-transfer port for the life of the
    # process (only close_communicator frees it, which a leg never calls), and the environmental legs
    # additionally stand up Ray actors.
    "trainers/grpo/test_online_grpo_vllm_e2e.py": TestSpec(
        nproc=1,
        markers=("gpu", "full", "1gpu", "qwen3", "vllm_server"),
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
    "trainers/lora/test_lora_ep_convergence.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "lora", "ep", "moe", "gptoss"), timeout=1000
    ),
    "trainers/lora/test_lora_ep_cp_etp.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "lora", "ep", "cp", "etp", "moe", "gptoss"), timeout=1000
    ),
    "trainers/lora/test_lora_tp_save_load.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "lora", "tp", "ep", "qwen3", "gptoss"),
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
    "trainers/preference/test_smpo_cp.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "cp", "qwen3"), timeout=600
    ),
    "trainers/preference/test_smpo_ep.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "ep", "gptoss"), timeout=1000
    ),
    "trainers/preference/test_smpo_ep_experts.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "lora", "ep", "moe", "gptoss"), timeout=1200
    ),
    "trainers/preference/test_smpo_ep_cp.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "ep", "cp", "gptoss"), timeout=1000
    ),
    "trainers/preference/test_smpo_fsdp.py": TestSpec(nproc=2, markers=("gpu", "core", "2gpu", "qwen3"), timeout=600),
    "trainers/preference/test_smpo_padding_free.py": TestSpec(
        nproc=2, markers=("gpu", "core", "2gpu", "qwen3"), timeout=600
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
    "trainers/sft/test_sft_ep.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "ep", "moe", "gptoss"), timeout=1000
    ),
    "trainers/sft/test_sft_ep_cp.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "ep", "cp", "moe", "gptoss"),
        timeout=1000,
    ),
    "trainers/sft/test_sft_ep_etp.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "ep", "etp", "moe", "gptoss"), timeout=1000
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
    # EP+TP only (EP+TP+ETP is not a supported axis set, so these carry no `etp` marker).
    "trainers/sft/test_sft_ep_tp.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "ep", "tp", "moe", "gptoss"),
        timeout=1000,
    ),
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
        nproc=2, markers=("gpu", "full", "2gpu", "ep", "vlm", "gemma4"), timeout=1500
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
    # `full` rather than `core`: a plain SFT smoke on the 20B checkpoint, like every other `oss20b_*`
    # entry. The core tier keeps the gpt-oss correctness gates
    # (`parallelism/ep/test_ep_correctness.py`), not smokes.
    "trainers/sft/test_sft_oss20b_default.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "gptoss"), timeout=1500
    ),
    "trainers/sft/test_sft_oss20b_tp.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "tp", "moe", "gptoss"), timeout=1500
    ),
    "trainers/sft/test_sft_oss20b_fsdp.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "moe", "gptoss"), timeout=1500
    ),
    "trainers/sft/test_sft_gptoss_trainable_sinks.py": TestSpec(
        nproc=2,
        markers=("gpu", "full", "2gpu", "moe", "gptoss"),
        args_matrix=("--mode fsdp", "--mode tp", "--mode ep"),
        timeout=1500,
    ),
    "trainers/sft/test_sft_oss20b_cp.py": TestSpec(
        nproc=2, markers=("gpu", "full", "2gpu", "cp", "moe", "gptoss"), timeout=1500
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
    "mistral3",
    "bailing",
    "lfm2",
    "zaya",
    "deepseek_v4",
    "inkling",
    "cohere2_moe",
    "glm5",
    "step3p7",
)


def script_path(rel: str) -> Path:
    """Absolute path to a manifest script."""
    return _GPU_DIR / rel


# Pytest-native modules collected directly, not torchrun scripts.
_NOT_MANIFEST_SCRIPTS = {"test_suite.py", "test_launcher_contract.py"}

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
    "profiling/benchmark_grouped_mm.py",
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
    without a launch spec. The launcher entrypoint (``test_suite.py``) is excluded, since
    pytest collects it directly rather than launching it. ``bench*.py`` files are globbed
    too, against :data:`_UNMANIFESTED_BENCHMARKS`.
    """
    on_disk = {
        str(p.relative_to(_GPU_DIR))
        for pattern in ("test_*.py", "bench*.py")
        for p in _GPU_DIR.rglob(pattern)
        if "__pycache__" not in p.parts and p.name not in _NOT_MANIFEST_SCRIPTS
    }
    return sorted(on_disk - set(MANIFEST) - _UNMANIFESTED_BENCHMARKS)


def stale_entries() -> list[str]:
    """Registered scripts, manifest entries and benchmarks alike, that no longer exist on disk."""
    return sorted(rel for rel in (*MANIFEST, *_UNMANIFESTED_BENCHMARKS) if not script_path(rel).exists())
