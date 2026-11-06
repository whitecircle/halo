#!/usr/bin/env python
"""Environmental GRPO on real benchmark datasets against a live vLLM server, in sequence on one group port.

The server needs its own GPU, the trained model (``VLLM_MODEL``) and ``VLLM_REASONING_PARSER=qwen3``: code_contests'
per-effort CoT budget draws a 400 on every rollout without one, giving zero tokens and loss=0. Without InfiniBand, set
``NCCL_IB_DISABLE=1 NCCL_NET=Socket`` or the image's OFI/Gin defaults wedge the cross-container group.
"""

import argparse
import json
import math
import os
import re
import shutil
import tempfile
import time
import traceback
import urllib.request
from typing import Any

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig

from halo.configs.grpo_environmental import AsyncRolloutConfig, EnvironmentConfig
from halo.distributed.parallelism_config import ParallelismConfig
from halo.env import env_int, env_str
from halo.environments.protocols.react import ReActEnvironment
from halo.environments.tasks.coding.code_contests import CodeContestsEnvironment
from halo.environments.tasks.qa import ExamQAEnvironment
from halo.environments.tools.toolsets import build_math_tools, build_python_tools
from halo.model_loading.parallel.frozen_models import load_frozen_auxiliary_model
from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer
from tests.common.harness import gpu_test_main, record_check
from tests.common.memory import cleanup_memory
from tests.common.models import QWEN3_0_6B
from tests.common.rank_log import log as rank0_log
from tests.common.training_checks import step_losses
from tests.common.weight_sync import weight_transfer_port

# The trainer broadcasts its own weights into the served model, so the two must be the same checkpoint.
MODEL_NAME = env_str("VLLM_MODEL", QWEN3_0_6B)
# ``or``: an exported-but-empty VLLM_SERVER_URL passes the conftest gate, which reads it the same way.
VLLM_SERVER_URL = env_str("VLLM_SERVER_URL") or "http://localhost:8000"
# Drawn once for the process (one rank): every benchmark rebinds it.
GROUP_PORT = weight_transfer_port("HALO_TEST_VLLM_GROUP_PORT")
# The offline fabricated-results backend needs this opt-in at env construction. Module scope: Ray actors snapshot the
# driver environment at the first ray.init, which an earlier benchmark in this process reaches first.
os.environ["HALO_ALLOW_MOCK_SEARCH"] = "1"
MAX_STEPS = env_int("HALO_TEST_MAX_STEPS", 3)
# The harness gives every run an empty HF_DATASETS_CACHE, so an offline host names a prepared cache here.
DATASETS_CACHE = env_str("HALO_TEST_BENCHMARK_DATASETS_CACHE")

# Sized for one vLLM server: batch x grad_accum x num_generations = 16 multi-turn rollouts per step.
DEFAULT_BATCH_SIZE = env_int("HALO_TEST_BATCH_SIZE", 2)
DEFAULT_GRAD_ACCUM = env_int("HALO_TEST_GRAD_ACCUM", 2)
DEFAULT_NUM_GENERATIONS = env_int("HALO_TEST_NUM_GENERATIONS", 4)
DEFAULT_NUM_WORKERS = env_int("HALO_TEST_NUM_WORKERS", 8)
DEFAULT_MAX_CONCURRENT = env_int("HALO_TEST_MAX_CONCURRENT", 8)
DEFAULT_ROLLOUT_MAX_TOKENS = env_int("HALO_TEST_ROLLOUT_MAX_TOKENS", 2048)
DEFAULT_MAX_COMPLETION = env_int("HALO_TEST_MAX_COMPLETION", 2048)


def log(msg: str) -> None:
    rank0_log(f"[benchmark] {msg}")


def check_vllm_server() -> bool:
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
    log(f"Loading GSM8K dataset ({num_samples} samples)...")
    ds = load_dataset("openai/gsm8k", "main", split=f"train[:{num_samples}]", cache_dir=DATASETS_CACHE)

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
    log(f"Loading SimpleQA dataset ({num_samples} samples)...")
    ds = load_dataset("basicv8vc/SimpleQA", split=f"test[:{num_samples}]", cache_dir=DATASETS_CACHE)

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
    log(f"Loading MMLU-Pro dataset ({num_samples} samples)...")
    ds = load_dataset("TIGER-Lab/MMLU-Pro", split=f"test[:{num_samples}]", cache_dir=DATASETS_CACHE)

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
    log(f"Loading Codeforces dataset ({num_samples} samples)...")
    ds = load_dataset("deepmind/code_contests", split="test", cache_dir=DATASETS_CACHE)

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
    # Resolved here: ``--max-steps`` rebinds the module global in run(), which a def-time default would never see.
    max_steps = MAX_STEPS if max_steps is None else max_steps

    output_dir = tempfile.mkdtemp(prefix=f"test_env_grpo_{env_name}_")
    result = {"success": False, "error": None}

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
            "logging_nan_inf_filter": False,
            "num_generations": num_generations,
            "max_completion_length": max_completion_length,
            "beta": 0.01,
            "reward_weights": [1.0],
            "gradient_checkpointing": True,
            "gradient_checkpointing_kwargs": {"use_reentrant": False},
            "optim": "adamw_torch_fused",
            "remove_unused_columns": False,
        }
        if extra_grpo_kwargs:
            grpo_kwargs.update(extra_grpo_kwargs)

        grpo_config = GRPOConfig(**grpo_kwargs)

        async_config = AsyncRolloutConfig(
            rollout_server_url=VLLM_SERVER_URL,
            rollout_group_port=GROUP_PORT,
            rollout_connection_timeout=120.0,
            rollout_actor_budget=num_workers,
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

        # environment_cls takes real EnvironmentBase subclasses only; factory-built envs (qa_search) arrive as registry
        # names through EnvironmentConfig, whose turn cap is its own field.
        log("  Creating DistributedAsyncEnvironmentalGRPOTrainer...")
        if isinstance(environment_cls, str):
            env_kwargs = dict(environment_kwargs)
            env_selector = {
                "environment_config": EnvironmentConfig(
                    environment_type=environment_cls,
                    max_turns=env_kwargs.pop("max_turns", None),
                    environment_kwargs=env_kwargs,
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
            # beta != 0 on a full fine-tune holds a frozen reference, loaded as the policy is.
            ref_model=load_frozen_auxiliary_model(
                MODEL_NAME,
                dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation="flash_attention_2",
                download_tag=None,
            ),
            **env_selector,
        )

        log(f"  Starting training for {max_steps} steps...")
        start_time = time.time()
        trainer.train()
        elapsed = time.time() - start_time

        losses = step_losses(trainer)
        rewards = [float(entry["reward"]) for entry in trainer.state.log_history if "reward" in entry]

        # Graded against what the served model can deliver: react_math is solvable at the default 0.6B, so a stuck
        # verifier must show as reward == 0. The other three score 0 at that size and are pinned on episodes having run
        # at all.
        gen_tokens = [
            float(entry["async/total_generation_tokens"])
            for entry in trainer.state.log_history
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
        if losses:
            log(f"  Final loss: {losses[-1]:.4f}")
        if rewards:
            log(f"  Final reward: {rewards[-1]:.4f}")

        if torch.cuda.is_available():
            mem_allocated = torch.cuda.max_memory_allocated() / 1e9
            mem_reserved = torch.cuda.max_memory_reserved() / 1e9
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
    log("=" * 70)
    log("TEST: ReAct Math Environment + GSM8K")
    log("=" * 70)

    dataset = prepare_gsm8k_dataset(num_samples=200)

    registry = build_math_tools()
    for tool in build_python_tools().list_tools():
        registry.register(tool)

    return run_env_grpo_training(
        env_name="react_math",
        dataset=dataset,
        environment_cls=ReActEnvironment,
        environment_kwargs={"tool_registry": registry, "max_turns": 8},
        rollout_max_tokens=1024,
        max_completion_length=1024,
    )


def test_search_qa():
    log("=" * 70)
    log("TEST: SearchQA Environment + SimpleQA")
    log("=" * 70)

    dataset = prepare_simpleqa_dataset(num_samples=200)

    return run_env_grpo_training(
        env_name="search_qa",
        dataset=dataset,
        environment_cls="qa_search",  # Factory-built env: a registry name, not a class.
        environment_kwargs={"max_turns": 5, "search_backend": "mock", "include_python_tools": False},
        rollout_max_tokens=1024,
        max_completion_length=1024,
    )


def test_exam_qa():
    log("=" * 70)
    log("TEST: ExamQA Environment + MMLU-Pro")
    log("=" * 70)

    dataset = prepare_mmlu_pro_dataset(num_samples=200)

    return run_env_grpo_training(
        env_name="exam_qa",
        dataset=dataset,
        environment_cls=ExamQAEnvironment,
        environment_kwargs={"max_turns": 3, "open_book": False},
        rollout_max_tokens=512,
        max_completion_length=512,
        num_generations=DEFAULT_NUM_GENERATIONS,
    )


def test_code_contests():
    log("=" * 70)
    log("TEST: CodeContests Environment + Codeforces")
    log("=" * 70)

    dataset = prepare_code_contests_dataset(num_samples=100)

    return run_env_grpo_training(
        env_name="code_contests",
        dataset=dataset,
        environment_cls=CodeContestsEnvironment,
        # The class default level's cap (medium, 8192) would fill the 2048-token turn, which the run refuses.
        environment_kwargs={
            "max_turns": 4,
            "timeout_per_test": 5,
            "reasoning_effort_profiles": {"medium": {"thinking_tokens": 1024}},
        },
        num_generations=4,
        batch_size=4,  # Must be divisible by num_generations.
        grad_accum=1,
        rollout_max_tokens=2048,
        max_completion_length=2048,
        extra_grpo_kwargs={"learning_rate": 1e-5},
    )


BENCHMARKS = ("react_math", "search_qa", "exam_qa", "code_contests")


def _run_benchmark(name: str, benchmark) -> None:
    """Health is re-checked first, so a server that died mid-suite reports as such over every later benchmark's
    failures."""
    if not check_vllm_server():
        raise RuntimeError(f"vLLM server at {VLLM_SERVER_URL} stopped answering /health before {name}")
    result = benchmark()
    if not result["success"]:
        raise AssertionError(result.get("error") or f"{name} did not run healthy")


@gpu_test_main(exact_world_size=1, prefix="env_grpo_benchmarks")
def run(ctx) -> dict:
    parser = argparse.ArgumentParser(description="Environmental GRPO Benchmark Tests")
    parser.add_argument("--test", type=str, default="all", choices=[*BENCHMARKS, "all"], help="Which test to run")
    parser.add_argument("--max-steps", type=int, default=None, help="Override HALO_TEST_MAX_STEPS")
    args, _ = parser.parse_known_args()

    if args.max_steps:
        global MAX_STEPS
        MAX_STEPS = args.max_steps

    log(f"\n{'=' * 70}")
    log("  Environmental GRPO Benchmark Training Tests")
    log(f"  Model: {MODEL_NAME}")
    log(f"  vLLM: {VLLM_SERVER_URL}")
    log(f"  GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    log(f"  Max steps per test: {MAX_STEPS}")
    log(f"{'=' * 70}")

    if not check_vllm_server():
        raise RuntimeError(f"vLLM server not reachable at {VLLM_SERVER_URL}")
    log("vLLM server is healthy")

    selected = BENCHMARKS if args.test == "all" else (args.test,)
    checks: dict[str, bool] = {}
    if "react_math" in selected:
        record_check(checks, "react_math", lambda: _run_benchmark("react_math", test_react_math))
    if "search_qa" in selected:
        record_check(checks, "search_qa", lambda: _run_benchmark("search_qa", test_search_qa))
    if "exam_qa" in selected:
        record_check(checks, "exam_qa", lambda: _run_benchmark("exam_qa", test_exam_qa))
    if "code_contests" in selected:
        record_check(checks, "code_contests", lambda: _run_benchmark("code_contests", test_code_contests))
    return {"checks": checks}


if __name__ == "__main__":
    run()
