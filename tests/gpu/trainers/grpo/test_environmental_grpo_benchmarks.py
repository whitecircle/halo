#!/usr/bin/env python
"""
End-to-end Environmental GRPO training with real benchmark datasets.

Tests 4 environment types against actual evaluation datasets:
1. ReAct Math        → GSM8K (grade school math)
2. SearchQA          → SimpleQA (factual QA)
3. ExamQA            → MMLU-Pro (hard multiple choice)
4. CodeContests      → Codeforces (competitive programming)

Each test runs a short training loop (3-5 steps) with a live vLLM server, sequentially in one
process: every benchmark forms its own weight-transfer group on the same ``HALO_TEST_VLLM_GROUP_PORT``, so it
also covers the client releasing that listener on close.

Prerequisites:
    # vLLM server on its OWN GPU (weight sync is NCCL — a server sharing the trainer's GPU cannot
    # broadcast to itself). ``VLLM_MODEL`` must be the model trained here: the trainer broadcasts its
    # own parameters into the served model. Neither server setting below is optional — code_contests
    # binds a per-effort CoT budget that reaches the server as ``thinking_token_budget``, which draws a
    # 400 on EVERY rollout without a reasoning parser and again under Model Runner V2, which does not
    # implement the field; the benchmark then "runs" with zero generated tokens and loss=0 / reward=0.
    VLLM_CUDA_DEVICES=7 VLLM_REASONING_PARSER=qwen3 VLLM_USE_V2_MODEL_RUNNER=0 \
        docker compose -f docker-compose.vllm.yml up -d vllm-server

Usage:
    # Trainer on a different GPU than the server. On a host WITHOUT InfiniBand the image's OFI/Gin
    # NCCL defaults wedge the cross-container group — override them:
    CUDA_VISIBLE_DEVICES=0 NCCL_IB_DISABLE=1 NCCL_NET=Socket python \
        tests/gpu/trainers/grpo/test_environmental_grpo_benchmarks.py --test react_math

    # All four environments:
    CUDA_VISIBLE_DEVICES=0 NCCL_IB_DISABLE=1 NCCL_NET=Socket python \
        tests/gpu/trainers/grpo/test_environmental_grpo_benchmarks.py --test all

    # With torchrun:
    CUDA_VISIBLE_DEVICES=0 NCCL_IB_DISABLE=1 NCCL_NET=Socket torchrun --nproc_per_node=1 \
        tests/gpu/trainers/grpo/test_environmental_grpo_benchmarks.py --test react_math
"""

import argparse
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
import traceback
from typing import Any

import torch
import torch.distributed as dist
from datasets import Dataset, load_dataset

from src.env import env_int, env_str
from tests.common.models import QWEN3_0_6B
from tests.common.utils import cleanup_memory
from tests.common.utils import log as rank0_log

# Same knob as docker-compose.vllm.yml: the trainer broadcasts its own weights into the served
# model, so the two must be the same checkpoint.
MODEL_NAME = env_str("VLLM_MODEL", QWEN3_0_6B)
# ``or`` (not an env_str default): an exported-but-empty VLLM_SERVER_URL passes the conftest gate,
# which reads it the same way, so the client must fall back to the same URL rather than to "".
VLLM_SERVER_URL = env_str("VLLM_SERVER_URL") or "http://localhost:8000"
HALO_TEST_VLLM_GROUP_PORT = env_int("HALO_TEST_VLLM_GROUP_PORT", 51216)
# The search benchmark keeps the offline fabricated-results backend instead of rate-limited live
# search; naming it is refused at env construction without this opt-in. Module scope, because Ray
# actors snapshot the driver environment at the FIRST ray.init — an earlier benchmark in the same
# process starts Ray before test_search_qa runs.
os.environ["HALO_ALLOW_MOCK_SEARCH"] = "1"
MAX_STEPS = env_int("HALO_TEST_MAX_STEPS", 3)
SEED = 42

# Sized for ONE vLLM server: batch × grad_accum × num_generations = 16 multi-turn rollouts per step,
# which is about what a single engine serves concurrently. Scale the workers with the engine count.
DEFAULT_BATCH_SIZE = env_int("HALO_TEST_BATCH_SIZE", 2)
DEFAULT_GRAD_ACCUM = env_int("HALO_TEST_GRAD_ACCUM", 2)
DEFAULT_NUM_GENERATIONS = env_int("HALO_TEST_NUM_GENERATIONS", 4)
DEFAULT_NUM_WORKERS = env_int("HALO_TEST_NUM_WORKERS", 8)
DEFAULT_MAX_CONCURRENT = env_int("HALO_TEST_MAX_CONCURRENT", 8)
DEFAULT_ROLLOUT_MAX_TOKENS = env_int("HALO_TEST_ROLLOUT_MAX_TOKENS", 2048)
DEFAULT_MAX_COMPLETION = env_int("HALO_TEST_MAX_COMPLETION", 2048)


def log(msg: str) -> None:
    """A benchmark line, from the main process only."""
    rank0_log(f"[benchmark] {msg}")


def check_vllm_server() -> bool:
    """Check if vLLM server is healthy."""
    import urllib.request

    try:
        req = urllib.request.Request(f"{VLLM_SERVER_URL}/health/")
        with urllib.request.urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except Exception:
        return False


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{seconds / 60:.1f}m"


def prepare_gsm8k_dataset(num_samples: int = 200) -> Dataset:
    """Load GSM8K and format for Environmental GRPO (ReAct math)."""
    log(f"Loading GSM8K dataset ({num_samples} samples)...")
    ds = load_dataset("openai/gsm8k", "main", split=f"train[:{num_samples}]")

    data = []
    for row in ds:
        answer_text = row["answer"]
        match = re.search(r"####\s*(.+)", answer_text)
        answer = match.group(1).strip() if match else answer_text.strip()

        data.append(
            {
                "prompt": [{"role": "user", "content": row["question"]}],
                "answer": answer,
            }
        )

    dataset = Dataset.from_list(data)
    log(f"  GSM8K: {len(dataset)} samples, example answer: '{data[0]['answer']}'")
    return dataset


def prepare_simpleqa_dataset(num_samples: int = 200) -> Dataset:
    """Load SimpleQA and format for SearchQA environment."""
    log(f"Loading SimpleQA dataset ({num_samples} samples)...")
    ds = load_dataset("basicv8vc/SimpleQA", split=f"test[:{num_samples}]")

    data = []
    for row in ds:
        data.append(
            {
                "prompt": [{"role": "user", "content": row["problem"]}],
                "answer": row["answer"],
            }
        )

    dataset = Dataset.from_list(data)
    log(f"  SimpleQA: {len(dataset)} samples, example: '{data[0]['answer']}'")
    return dataset


def prepare_mmlu_pro_dataset(num_samples: int = 200) -> Dataset:
    """Load MMLU-Pro and format for ExamQA environment (multiple choice)."""
    log(f"Loading MMLU-Pro dataset ({num_samples} samples)...")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split=f"test[:{num_samples}]")

    data = []
    choice_letters = "ABCDEFGHIJ"

    for row in ds:
        options = row["options"]
        choices = [f"{choice_letters[i]}: {opt}" for i, opt in enumerate(options)]
        answer_letter = choice_letters[row["answer_index"]]

        question = row["question"] + "\n\n" + "\n".join(choices)

        data.append(
            {
                "prompt": [{"role": "user", "content": question}],
                "answer": answer_letter,
                "choices": choices,
            }
        )

    dataset = Dataset.from_list(data)
    log(f"  MMLU-Pro: {len(dataset)} samples, categories: {set(ds['category'][:10])}")
    return dataset


def prepare_code_contests_dataset(num_samples: int = 100) -> Dataset:
    """Load Codeforces problems from deepmind/code_contests for CodeContests env."""
    log(f"Loading Codeforces dataset ({num_samples} samples)...")
    ds = load_dataset("deepmind/code_contests", split="test")

    data = []
    for row in ds:
        test_cases = []
        for inp, out in zip(row["public_tests"]["input"], row["public_tests"]["output"], strict=False):
            test_cases.append({"input": inp.strip(), "output": out.strip()})

        if not test_cases:
            continue

        if len(test_cases) > 20:
            test_cases = test_cases[:20]

        data.append(
            {
                "prompt": [{"role": "user", "content": row["description"]}],
                "answer": json.dumps({"test_cases": test_cases}),
            }
        )

        if len(data) >= num_samples:
            break

    dataset = Dataset.from_list(data)
    log(f"  Codeforces: {len(dataset)} problems with test cases")
    return dataset


def run_env_grpo_training(
    env_name: str,
    dataset: Dataset,
    environment_cls,
    environment_kwargs: dict[str, Any],
    max_steps: int | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    grad_accum: int = DEFAULT_GRAD_ACCUM,
    num_generations: int = DEFAULT_NUM_GENERATIONS,
    num_workers: int = DEFAULT_NUM_WORKERS,
    max_concurrent: int = DEFAULT_MAX_CONCURRENT,
    rollout_max_tokens: int = DEFAULT_ROLLOUT_MAX_TOKENS,
    max_completion_length: int = DEFAULT_MAX_COMPLETION,
    extra_grpo_kwargs: dict | None = None,
) -> dict[str, Any]:
    """Run a single Environmental GRPO training session.

    Returns dict with metrics and status.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig

    from src.configs.async_training_config import AsyncTrainingConfig
    from src.distributed.parallelism_config import ParallelismConfig
    from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer

    # Resolved here, not as a def-time default: ``--max-steps`` rebinds the module global in main(),
    # which a default bound at import time would never see.
    max_steps = MAX_STEPS if max_steps is None else max_steps

    output_dir = tempfile.mkdtemp(prefix=f"test_env_grpo_{env_name}_")
    result = {"env": env_name, "success": False, "steps": 0, "error": None}

    try:
        log(f"  Loading model: {MODEL_NAME}")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
        )

        grpo_kwargs = {
            "output_dir": output_dir,
            "max_steps": max_steps,
            "per_device_train_batch_size": batch_size,
            "gradient_accumulation_steps": grad_accum,
            "learning_rate": 5e-6,
            "bf16": True,
            "logging_steps": 1,
            "logging_first_step": True,
            "save_strategy": "no",
            "report_to": "none",
            "num_generations": num_generations,
            "max_completion_length": max_completion_length,
            "beta": 0.01,
            "generation_kwargs": {"temperature": 0.7},
            "fsdp": "",
            "reward_weights": [1.0],
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "optim": "adamw_torch_fused",
            "vllm_group_port": HALO_TEST_VLLM_GROUP_PORT,
            "remove_unused_columns": False,
        }
        if extra_grpo_kwargs:
            grpo_kwargs.update(extra_grpo_kwargs)

        grpo_config = GRPOConfig(**grpo_kwargs)

        async_config = AsyncTrainingConfig(
            rollout_server_url=VLLM_SERVER_URL,
            rollout_connection_timeout=120.0,
            num_rollout_workers=num_workers,
            max_concurrent_rollouts=max_concurrent,
            rollout_temperature=0.7,
            rollout_top_p=0.95,
            rollout_max_tokens=rollout_max_tokens,
            enable_prefetch=False,
            sync_weights_every_n_steps=1,
            model_name=MODEL_NAME,
        )

        log(
            f"  Config: batch={batch_size}, grad_accum={grad_accum}, "
            f"num_gen={num_generations}, workers={num_workers}, "
            f"max_concurrent={max_concurrent}"
        )
        log(
            f"  Effective batch: {batch_size * grad_accum} prompts "
            f"× {num_generations} gen = {batch_size * grad_accum * num_generations} sequences/step"
        )

        # environment_cls takes real BaseEnvironment subclasses only; factory-built envs (qa_search)
        # arrive as registry names and must go through EnvironmentConfig instead.
        log("  Creating DistributedAsyncEnvironmentalGRPOTrainer...")
        if isinstance(environment_cls, str):
            from src.configs.environment_config import EnvironmentConfig

            env_selector = {
                "environment_config": EnvironmentConfig(
                    environment_type=environment_cls, environment_kwargs=environment_kwargs
                )
            }
        else:
            env_selector = {"environment_cls": environment_cls, "environment_kwargs": environment_kwargs}
        trainer = DistributedAsyncEnvironmentalGRPOTrainer(
            model=model,
            args=grpo_config,
            train_dataset=dataset,
            processing_class=tokenizer,
            async_config=async_config,
            parallelism_config=ParallelismConfig(),
            **env_selector,
        )

        log(f"  Starting training for {max_steps} steps...")
        start_time = time.time()
        trainer.train()
        elapsed = time.time() - start_time

        result["steps"] = max_steps
        result["elapsed"] = elapsed

        losses: list[float] = []
        rewards: list[float] = []
        if hasattr(trainer, "state") and trainer.state.log_history:
            for entry in trainer.state.log_history:
                if "loss" in entry:
                    result["last_loss"] = entry["loss"]
                    losses.append(float(entry["loss"]))
                if "reward" in entry:
                    result["last_reward"] = entry["reward"]
                    rewards.append(float(entry["reward"]))

        # Graded against what the SERVED model can deliver: react_math is solvable by the default
        # 0.6B so a stuck verifier (all-equal reward → zero advantage) must show as reward == 0,
        # while the other three score 0 at that size no matter how healthy the stack is and are
        # instead pinned on episodes having run at all — a dead server or rejected request shows
        # zero generated tokens.
        gen_tokens = [
            float(entry["async/total_generation_tokens"])
            for entry in (trainer.state.log_history if hasattr(trainer, "state") else [])
            if "async/total_generation_tokens" in entry
        ]
        ran_healthy = (
            len(losses) == max_steps
            and all(math.isfinite(v) for v in losses)
            and bool(rewards)
            and all(math.isfinite(v) for v in rewards)
            and bool(gen_tokens)
            and gen_tokens[-1] > 0
        )
        if env_name == "react_math":
            result["success"] = ran_healthy and any(v > 0.0 for v in rewards)
            if not result["success"]:
                result["error"] = (
                    f"losses={losses} rewards={rewards} (expected {max_steps} finite losses, a reward > 0)"
                )
        else:
            result["success"] = ran_healthy
            if not result["success"]:
                result["error"] = (
                    f"losses={losses} rewards={rewards} gen_tokens={gen_tokens} (expected "
                    f"{max_steps} finite losses and generated tokens > 0 — episodes never ran)"
                )

        log(f"  Training completed in {format_duration(elapsed)}")
        if "last_loss" in result:
            log(f"  Final loss: {result['last_loss']:.4f}")
        if "last_reward" in result:
            log(f"  Final reward: {result['last_reward']:.4f}")

        if torch.cuda.is_available():
            mem_allocated = torch.cuda.max_memory_allocated() / 1e9
            mem_reserved = torch.cuda.max_memory_reserved() / 1e9
            result["peak_mem_gb"] = mem_allocated
            log(f"  Peak GPU memory: {mem_allocated:.1f}GB allocated, {mem_reserved:.1f}GB reserved")

    except Exception as e:
        result["error"] = str(e)
        log(f"  FAILED: {e}")
        traceback.print_exc()

    finally:
        cleanup_memory()
        shutil.rmtree(output_dir, ignore_errors=True)

    return result


def test_react_math():
    """ReAct Math environment with GSM8K dataset."""
    from src.environments.envs.protocols.react import ReActEnvironment
    from src.environments.tools.factories import create_native_math_tools, create_native_python_tools

    log("=" * 70)
    log("TEST: ReAct Math Environment + GSM8K")
    log("=" * 70)

    dataset = prepare_gsm8k_dataset(num_samples=200)

    registry = create_native_math_tools()
    for tool in create_native_python_tools().list_tools():
        registry.register(tool)

    return run_env_grpo_training(
        env_name="react_math",
        dataset=dataset,
        environment_cls=ReActEnvironment,
        environment_kwargs={
            "tool_registry": registry,
            "max_turns": 8,
            "success_reward": 1.0,
        },
        rollout_max_tokens=1024,
        max_completion_length=1024,
    )


def test_search_qa():
    """SearchQA environment with SimpleQA dataset."""

    log("=" * 70)
    log("TEST: SearchQA Environment + SimpleQA")
    log("=" * 70)

    dataset = prepare_simpleqa_dataset(num_samples=200)

    return run_env_grpo_training(
        env_name="search_qa",
        dataset=dataset,
        environment_cls="qa_search",  # factory-built env → registry name, not a class
        environment_kwargs={
            "max_turns": 5,
            "search_backend": "mock",
            "include_python_tools": False,
            "success_reward": 1.0,
        },
        rollout_max_tokens=1024,
        max_completion_length=1024,
    )


def test_exam_qa():
    """ExamQA environment with MMLU-Pro dataset."""
    from src.environments.envs.tasks.qa import ExamQAEnvironment

    log("=" * 70)
    log("TEST: ExamQA Environment + MMLU-Pro")
    log("=" * 70)

    dataset = prepare_mmlu_pro_dataset(num_samples=200)

    return run_env_grpo_training(
        env_name="exam_qa",
        dataset=dataset,
        environment_cls=ExamQAEnvironment,
        environment_kwargs={
            "max_turns": 3,
            "open_book": False,
            "success_reward": 1.0,
        },
        rollout_max_tokens=512,
        max_completion_length=512,
        num_generations=DEFAULT_NUM_GENERATIONS,
    )


def test_code_contests():
    """CodeContests environment with Codeforces dataset."""
    from src.environments.envs.tasks.coding.code_contests import CodeContestsEnvironment

    log("=" * 70)
    log("TEST: CodeContests Environment + Codeforces")
    log("=" * 70)

    dataset = prepare_code_contests_dataset(num_samples=100)

    return run_env_grpo_training(
        env_name="code_contests",
        dataset=dataset,
        environment_cls=CodeContestsEnvironment,
        environment_kwargs={
            "max_turns": 4,
            "timeout_per_test": 5,
            "success_reward": 1.0,
        },
        num_generations=4,
        batch_size=4,  # must be divisible by num_generations
        grad_accum=1,
        rollout_max_tokens=2048,
        max_completion_length=2048,
        extra_grpo_kwargs={"learning_rate": 1e-5},
    )


TEST_MAP = {
    "react_math": test_react_math,
    "search_qa": test_search_qa,
    "exam_qa": test_exam_qa,
    "code_contests": test_code_contests,
}


def main():
    parser = argparse.ArgumentParser(description="Environmental GRPO Benchmark Tests")
    parser.add_argument(
        "--test", type=str, default="all", choices=list(TEST_MAP.keys()) + ["all"], help="Which test to run"
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Override HALO_TEST_MAX_STEPS")
    args = parser.parse_args()

    if args.max_steps:
        global MAX_STEPS
        MAX_STEPS = args.max_steps

    if "RANK" in os.environ:
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
    elif torch.cuda.is_available():
        torch.cuda.set_device(0)

    log(f"\n{'=' * 70}")
    log("  Environmental GRPO Benchmark Training Tests")
    log(f"  Model: {MODEL_NAME}")
    log(f"  vLLM: {VLLM_SERVER_URL}")
    log(f"  GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    log(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    log(f"  Max steps per test: {MAX_STEPS}")
    log(f"{'=' * 70}")

    if not check_vllm_server():
        log("ERROR: vLLM server not reachable at " + VLLM_SERVER_URL)
        log("Start it with: docker run -d --name vllm-qwen3-4b ...")
        sys.exit(1)
    log("vLLM server is healthy")

    test_names = list(TEST_MAP.keys()) if args.test == "all" else [args.test]

    # No server restart between benchmarks — each forms a fresh weight-transfer group on the same
    # port. Re-checking health first reports a dead server as such, not as N downstream failures.
    results = []
    for name in test_names:
        if not check_vllm_server():
            log(f"ERROR: vLLM server at {VLLM_SERVER_URL} stopped answering /health before {name}")
            results.append({"env": name, "success": False, "error": "vLLM server unreachable"})
            continue

        result = TEST_MAP[name]()
        results.append(result)

    log(f"\n{'=' * 70}")
    log("  RESULTS SUMMARY")
    log(f"{'=' * 70}")

    all_passed = True
    for r in results:
        status = "PASS" if r["success"] else "FAIL"
        if not r["success"]:
            all_passed = False

        extras = []
        if "elapsed" in r:
            extras.append(f"time={format_duration(r['elapsed'])}")
        if "last_loss" in r:
            extras.append(f"loss={r['last_loss']:.4f}")
        if "last_reward" in r:
            extras.append(f"reward={r['last_reward']:.4f}")
        if "peak_mem_gb" in r:
            extras.append(f"peak_mem={r['peak_mem_gb']:.1f}GB")
        if r.get("error"):
            extras.append(f"error={r['error'][:80]}")

        extra_str = f" ({', '.join(extras)})" if extras else ""
        log(f"  [{status}] {r['env']}{extra_str}")

    log(f"{'=' * 70}")
    log(f"  {sum(1 for r in results if r['success'])}/{len(results)} tests passed")
    log(f"{'=' * 70}")

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
