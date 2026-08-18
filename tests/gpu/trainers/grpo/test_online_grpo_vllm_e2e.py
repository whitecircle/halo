#!/usr/bin/env python
"""
End-to-end test for Online GRPO with a live vLLM server (Docker).

Tests the full training loop with DistributedGRPOTrainer against a real
vLLM server with native NCCL weight transfer. Validates:

1. vLLM server health and weight transfer endpoints
2. vLLM-powered generation during training
3. Packed NCCL weight synchronization back to vLLM after gradient steps
4. Full training loop completes without errors

Prerequisites:
    # Start vLLM server in Docker (--gpus all required for NCCL P2P):
    docker run --gpus all --network=host --ipc=host -e VLLM_USE_V2_MODEL_RUNNER=0 \
        vllm-server:0.26.0 Qwen/Qwen3-0.6B --port 8000 --dtype bfloat16 \
        --max-model-len 512 --enforce-eager --reasoning-parser qwen3 \
        --weight-transfer-config '{"backend": "nccl"}'

One leg per invocation, and ``--mode`` is required: each leg binds its own trainer-side
weight-transfer port and holds it for the life of the process — only ``close_communicator`` frees it
(``src/distributed/nccl/clients/vllm.py``), and a leg that trains to completion never calls it — and
the environmental legs additionally stand up Ray actors. The manifest's args_matrix gives each leg
its own invocation; there is no mode that runs several. Every leg first checks server connectivity.

Usage:
    # Run on GPU 1 (different from vLLM):
    CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 \
        tests/gpu/trainers/grpo/test_online_grpo_vllm_e2e.py --mode online

    # Rolling multi-server sync: one server per URL, each serving the same model on its own GPU.
    HALO_TEST_VLLM_SERVER_URLS=http://localhost:8000,http://localhost:8001 \
        CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=1 \
        tests/gpu/trainers/grpo/test_online_grpo_vllm_e2e.py --mode environmental
"""

import argparse
import math
import os
import random
import shutil
import tempfile

import torch
from datasets import Dataset

from src.env import env_str
from src.environments.rewards import extract_last_boxed
from tests.common.harness import gpu_test_main, record_check
from tests.common.models import QWEN3_0_6B
from tests.common.on_policy_e2e import probe_top_logprobs
from tests.common.ports import free_port
from tests.common.utils import cleanup_memory, log

MODEL_NAME = QWEN3_0_6B
VLLM_SERVER_URL = env_str("VLLM_SERVER_URL") or "http://localhost:8000"
# Rollout endpoints the environmental legs drive, comma-separated. Two or more put the weight sync on
# the ROLLING path (an InferenceClientManager pushing to one server at a time while the rest keep
# serving) — a shape a single URL never reaches. Every server must serve MODEL_NAME.
VLLM_SERVER_URLS = [
    url.strip() for url in (env_str("HALO_TEST_VLLM_SERVER_URLS") or VLLM_SERVER_URL).split(",") if url.strip()
]
NUM_TRAIN_SAMPLES = 16
MAX_STEPS = 3
SEED = 42


def rollout_server_configs() -> list[dict]:
    """The leg's ``rollout_server_configs``: one entry per configured server, each on its own
    :func:`free_port` (never re-issued within the process, so the ports cannot overlap)."""
    return [{"url": url, "group_port": free_port()} for url in VLLM_SERVER_URLS]


def accuracy_reward(completions, answer, **kwargs):
    """Graded accuracy: exact answers score 1, near-misses decay with relative error.

    The gradation is what keeps this e2e trainable: GRPO's advantage is the within-group reward
    spread, and a binary exact-match reward ties every group the moment the task is uniformly easy
    (all 1) or uniformly hard (all 0) for the served model — three steps of zero loss and the
    zero-gradient assert below fires on a healthy stack. Two temperature-0.7 samples almost never
    produce the same wrong product, so graded credit keeps the spread alive at any difficulty."""
    rewards = []
    for completion, gt in zip(completions, answer, strict=False):
        content = (completion[-1]["content"] if completion else "") if isinstance(completion, list) else completion
        extracted = (extract_last_boxed(content) or "").strip().replace(",", "")
        gt_int = int(str(gt).strip())
        try:
            value = int(extracted)
        except ValueError:
            rewards.append(0.0)
            continue
        relative_error = abs(value - gt_int) / max(1, abs(gt_int))
        rewards.append(1.0 if value == gt_int else max(0.0, 0.8 - relative_error))
    return rewards


def create_grpo_dataset(num_samples: int, seed: int = SEED) -> Dataset:
    """Create synthetic math dataset for GRPO."""
    rng = random.Random(seed)
    data = []
    for _ in range(num_samples):
        # Three-digit products: easier arithmetic ties every group, leaving zero advantage/gradient.
        a = rng.randint(101, 999)
        b = rng.randint(101, 999)
        answer = a * b

        data.append(
            {
                "prompt": [
                    {"role": "user", "content": f"What is {a} * {b}? Put your answer in \\boxed{{}}. /no_think"}
                ],
                "answer": str(answer),
            }
        )
    return Dataset.from_list(data)


def test_vllm_server_reachable():
    """Verify vLLM server is running and healthy with weight transfer endpoints."""
    import json
    import urllib.request

    url = f"{VLLM_SERVER_URL}/health"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=10) as resp:
        assert resp.status == 200, f"Health check failed: {resp.status}"

    url = f"{VLLM_SERVER_URL}/openapi.json"
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req, timeout=10) as resp:
        spec = json.loads(resp.read())
        paths = list(spec.get("paths", {}).keys())
        assert "/init_weight_transfer_engine" in paths, f"Missing /init_weight_transfer_engine. Paths: {paths}"
        assert "/update_weights" in paths, "Missing /update_weights endpoint"
        assert "/get_world_size" in paths, "Missing /get_world_size endpoint"

    log(f"  vLLM server at {VLLM_SERVER_URL} is healthy with native weight transfer endpoints")


def test_vllm_generation():
    """Test that vLLM can generate text."""
    import json
    import urllib.request

    url = f"{VLLM_SERVER_URL}/v1/chat/completions"
    payload = json.dumps(
        {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "What is 1+1?"}],
            "max_tokens": 64,
            "temperature": 0.7,
        }
    ).encode()

    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
        assert "choices" in result, f"No choices in response: {result}"
        message = result["choices"][0]["message"]
        # The required VLLM_REASONING_PARSER routes plain replies into ``reasoning`` (0.26.0
        # spelling; older builds ``reasoning_content``), leaving ``content`` null — mirror the
        # toolkit client's ``_get_reasoning`` fallback instead of assuming the unparsed shape.
        content = message.get("content") or message.get("reasoning") or message.get("reasoning_content") or ""
        assert len(content) > 0, f"Empty generation: {message}"

    log(f"  vLLM generation works: '{content[:80]}...'")


def test_online_grpo_e2e():
    """End-to-end online GRPO training with live vLLM server.

    This is the main test: instantiates DistributedGRPOTrainer with
    use_vllm=True pointing to the Docker vLLM server, runs a few
    training steps, and verifies completion.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig

    from src.distributed.parallelism_config import ParallelismConfig
    from src.trainers.grpo.online import DistributedGRPOTrainer

    output_dir = tempfile.mkdtemp(prefix="test_grpo_vllm_e2e_")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        from urllib.parse import urlparse

        parsed = urlparse(VLLM_SERVER_URL)
        vllm_host = parsed.hostname or "localhost"
        vllm_port = parsed.port or 8000

        config = GRPOConfig(
            output_dir=output_dir,
            max_steps=MAX_STEPS,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=5e-6,
            bf16=True,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            use_vllm=True,
            vllm_mode="server",
            vllm_server_host=vllm_host,
            vllm_server_port=vllm_port,
            vllm_server_timeout=60.0,
            # Allocated, never a literal: a fixed port collides with foreign holders on a shared
            # host (concurrent suites share the 512xx block).
            vllm_group_port=free_port(),
            num_generations=2,
            max_completion_length=256,
            beta=0.0,
            generation_kwargs={"temperature": 0.7},
            fsdp="",
        )

        dataset = create_grpo_dataset(NUM_TRAIN_SAMPLES, seed=SEED)
        parallelism_config = ParallelismConfig()

        log(f"  Creating DistributedGRPOTrainer with vLLM at {vllm_host}:{vllm_port}...")

        trainer = DistributedGRPOTrainer(
            model=model,
            reward_funcs=[accuracy_reward],
            args=config,
            train_dataset=dataset,
            processing_class=tokenizer,
            parallelism_config=parallelism_config,
        )

        log(f"  Trainer created. Starting training for {MAX_STEPS} steps...")

        trainer.train()

        log("  Training completed successfully!")

        # log_history[-1] is HF's end-of-run summary (train_loss, no 'loss' key) — scan for the
        # per-step entries or every assertion below is unreachable.
        losses = [float(e["loss"]) for e in trainer.state.log_history if "loss" in e]
        log(f"  Per-step losses: {[f'{v:.4f}' for v in losses]}")
        assert losses, f"no per-step loss was logged in {MAX_STEPS} steps: {trainer.state.log_history}"
        assert all(math.isfinite(v) for v in losses), f"non-finite loss: {losses}"
        assert any(v != 0.0 for v in losses), "every step logged a zero loss — no gradient signal reached the policy"

    finally:
        cleanup_memory()
        shutil.rmtree(output_dir, ignore_errors=True)


def test_online_sdpg_e2e():
    """End-to-end online SDPG (DistributedSDPGTrainer) with live vLLM server.

    SDPG = online GRPO + a privileged-teacher reverse-KL OPD term on positive-advantage rollouts
    (the faithful arXiv:2606.04036 method, run via rlvr_online_grpo.py --use_sdpg). This validates
    the full path: vLLM rollouts → verifier advantages → privileged-teacher forward → OPD term.
    It asserts the OPD term actually fired (the ``opd_loss``/``opd_beta`` metrics are recorded), so a
    regression that silently drops OPD (e.g. the fused-Liger loss bypass) fails here.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig

    from src.distributed.parallelism_config import ParallelismConfig
    from src.trainers.distillation.sdpg import DistributedSDPGTrainer

    output_dir = tempfile.mkdtemp(prefix="test_sdpg_vllm_e2e_")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, trust_remote_code=True)

        from urllib.parse import urlparse

        parsed = urlparse(VLLM_SERVER_URL)
        vllm_host = parsed.hostname or "localhost"
        vllm_port = parsed.port or 8000

        config = GRPOConfig(
            output_dir=output_dir,
            max_steps=MAX_STEPS,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=5e-6,
            bf16=True,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            use_vllm=True,
            vllm_mode="server",
            vllm_server_host=vllm_host,
            vllm_server_port=vllm_port,
            vllm_server_timeout=60.0,
            # Allocated per leg (see the fixed-port note above).
            vllm_group_port=free_port(),
            num_generations=2,
            max_completion_length=256,
            beta=0.0,
            generation_kwargs={"temperature": 0.7},
            fsdp="",
        )

        dataset = create_grpo_dataset(NUM_TRAIN_SAMPLES, seed=SEED)

        log(f"  Creating DistributedSDPGTrainer with vLLM at {vllm_host}:{vllm_port}...")

        trainer = DistributedSDPGTrainer(
            model=model,
            reward_funcs=[accuracy_reward],
            args=config,
            train_dataset=dataset,
            processing_class=tokenizer,
            parallelism_config=ParallelismConfig(),
            sdpg_answer_field="answer",
            sdpg_loss="reverse_kl",
            sdpg_beta_base=1.0,
        )

        log(f"  Trainer created. Starting SDPG training for {MAX_STEPS} steps...")
        trainer.train()
        log("  SDPG training completed successfully!")

        # opd_loss is legitimately 0.0 when no rollout earns a positive advantage (the >0 gate masks
        # every token), so pin presence + finiteness, not >0.
        opd_steps = [h for h in trainer.state.log_history if "opd_beta" in h]
        assert opd_steps, "SDPG OPD term never fired (no opd_beta in log_history)"
        opd_losses = [float(h["opd_loss"]) for h in opd_steps if "opd_loss" in h]
        assert opd_losses, "opd_beta logged but opd_loss missing"
        assert all(torch.isfinite(torch.tensor(v)) for v in opd_losses), f"Non-finite opd_loss: {opd_losses}"
        log(
            f"  OPD term fired across {len(opd_steps)} step(s); opd_loss={opd_losses}, "
            f"opd_beta={[h['opd_beta'] for h in opd_steps]}"
        )

        last_loss = next((h["loss"] for h in reversed(trainer.state.log_history) if "loss" in h), None)
        if last_loss is not None:
            assert not torch.isnan(torch.tensor(float(last_loss))), "Loss is NaN"
            assert not torch.isinf(torch.tensor(float(last_loss))), "Loss is Inf"
            log(f"  Final loss: {float(last_loss):.4f}")

    finally:
        cleanup_memory()
        shutil.rmtree(output_dir, ignore_errors=True)


def test_environmental_grpo_e2e():
    """End-to-end environmental GRPO with live vLLM server.

    Tests DistributedAsyncEnvironmentalGRPOTrainer with a ReAct math environment,
    using the Docker vLLM server for generation and weight sync.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig

    from src.configs.async_training_config import AsyncTrainingConfig
    from src.distributed.parallelism_config import ParallelismConfig
    from src.environments.envs.protocols.react import ReActEnvironment
    from src.environments.tools.factories import create_native_math_tools, create_native_python_tools
    from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer

    output_dir = tempfile.mkdtemp(prefix="test_env_grpo_vllm_e2e_")

    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        )

        grpo_config = GRPOConfig(
            output_dir=output_dir,
            max_steps=2,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=5e-6,
            bf16=True,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            num_generations=2,
            max_completion_length=256,
            beta=0.01,
            generation_kwargs={"temperature": 0.7},
            fsdp="",
            reward_weights=[1.0],
        )

        async_config = AsyncTrainingConfig(
            rollout_server_configs=rollout_server_configs(),
            rollout_connection_timeout=60.0,
            num_rollout_workers=2,
            max_concurrent_rollouts=4,
            rollout_temperature=0.7,
            rollout_max_tokens=128,
            enable_prefetch=False,
            sync_weights_every_n_steps=1,
            model_name=MODEL_NAME,
        )

        registry = create_native_math_tools()
        for tool in create_native_python_tools().list_tools():
            registry.register(tool)

        environment_kwargs = {
            "tool_registry": registry,
            "max_turns": 3,
            "success_reward": 1.0,
        }

        dataset = Dataset.from_list(
            [
                {"prompt": [{"role": "user", "content": f"What is {a} + {b}?"}], "answer": str(a + b)}
                for a, b in [(2, 3), (5, 7), (10, 15), (3, 8), (6, 4), (9, 1), (12, 8), (7, 13)]
            ]
        )

        log(f"  Creating DistributedAsyncEnvironmentalGRPOTrainer with vLLM at {VLLM_SERVER_URL}...")

        trainer = DistributedAsyncEnvironmentalGRPOTrainer(
            model=model,
            args=grpo_config,
            train_dataset=dataset,
            processing_class=tokenizer,
            async_config=async_config,
            environment_cls=ReActEnvironment,
            environment_kwargs=environment_kwargs,
            parallelism_config=ParallelismConfig(),
        )

        log("  Trainer created. Starting training for 2 steps...")

        trainer.train()

        log("  Environmental GRPO training completed successfully!")

        # Without these, "train() did not raise" is the whole assertion — a weight sync that never
        # landed (stale served policy, zero advantage) still produces a clean run.
        losses = [float(e["loss"]) for e in trainer.state.log_history if "loss" in e]
        log(f"  Per-step losses: {[f'{v:.4f}' for v in losses]}")
        assert losses, f"no per-step loss was logged: {trainer.state.log_history}"
        assert all(math.isfinite(v) for v in losses), f"non-finite loss: {losses}"

        # The sync has to reach EVERY configured server. With more than one it goes through the
        # rolling path (one server at a time, the rest kept serving), where a fan-out that stops
        # after the first leaves the others generating from a stale policy with no error at all.
        # Perturbed deliberately: a GRPO group whose samples tie has zero advantage and leaves the
        # weights bit-identical, and "the logprobs did not change" would then prove nothing.
        before = {url: probe_top_logprobs(url, MODEL_NAME) for url in VLLM_SERVER_URLS}
        with torch.no_grad():
            for name, param in trainer.model.named_parameters():
                if "layers.0" in name and name.endswith(".weight") and param.dtype.is_floating_point:
                    param.mul_(1.05)
        assert trainer._sync_weights_to_engine(force=True), "the forced weight sync pushed nothing"
        assert trainer._multi_server_mode and trainer._weight_sync_client.num_servers == len(VLLM_SERVER_URLS), (
            f"the leg drove {getattr(trainer._weight_sync_client, 'num_servers', None)} client(s) for "
            f"{len(VLLM_SERVER_URLS)} configured server(s)"
        )
        for url in VLLM_SERVER_URLS:
            after = probe_top_logprobs(url, MODEL_NAME)
            assert after != before[url], f"{url} still serves the pre-sync policy — the sync never reached it"
        log(f"  forced sync moved the served policy on all {len(VLLM_SERVER_URLS)} server(s)")

    finally:
        cleanup_memory()
        shutil.rmtree(output_dir, ignore_errors=True)


def test_online_grpo_lora_e2e():
    """Online GRPO + LoRA against the live vLLM server — validates the PEFT-aware NCCL weight sync.

    With a peft_config the trainer must merge the adapter into the base before each sync and forward
    the merged weights under plain (non-PEFT) names. Forwarding ``base_model.*`` / ``lora_*`` names
    instead makes the vendored client reject unknown params (and vLLM then generates from the
    un-adapted base — broken on-policy RL). A clean multi-step run with finite loss,
    on a confirmed PeftModel, exercises the merge→strip→unmerge sync path end-to-end.
    """
    from urllib.parse import urlparse

    from accelerate.utils import is_peft_model
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig

    from src.distributed.parallelism_config import ParallelismConfig
    from src.trainers.grpo.online import DistributedGRPOTrainer

    output_dir = tempfile.mkdtemp(prefix="test_grpo_lora_vllm_e2e_")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, trust_remote_code=True)

        parsed = urlparse(VLLM_SERVER_URL)
        config = GRPOConfig(
            output_dir=output_dir,
            max_steps=MAX_STEPS,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=5e-6,
            bf16=True,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            use_vllm=True,
            vllm_mode="server",
            vllm_server_host=parsed.hostname or "localhost",
            vllm_server_port=parsed.port or 8000,
            vllm_server_timeout=60.0,
            # Allocated per leg (see the fixed-port note above).
            vllm_group_port=free_port(),
            num_generations=2,
            max_completion_length=256,
            beta=0.0,
            generation_kwargs={"temperature": 0.7},
            fsdp="",
        )
        trainer = DistributedGRPOTrainer(
            model=model,
            reward_funcs=[accuracy_reward],
            args=config,
            train_dataset=create_grpo_dataset(NUM_TRAIN_SAMPLES, seed=SEED),
            processing_class=tokenizer,
            parallelism_config=ParallelismConfig(),
            peft_config=LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
        )
        assert is_peft_model(trainer.accelerator.unwrap_model(trainer.model)), (
            "peft_config did not produce a PeftModel"
        )
        log("  Trainer created (PeftModel). Training + syncing merged LoRA weights to vLLM...")
        trainer.train()
        log("  Online GRPO+LoRA training completed (PEFT weight sync OK)!")

        last = next((h["loss"] for h in reversed(trainer.state.log_history) if "loss" in h), None)
        if last is not None:
            assert torch.isfinite(torch.tensor(float(last))), f"Non-finite loss {last}"
            log(f"  Final loss: {float(last):.4f}")
    finally:
        cleanup_memory()
        shutil.rmtree(output_dir, ignore_errors=True)


def test_environmental_grpo_lora_e2e():
    """Environmental GRPO + LoRA against the live vLLM server — PEFT weight sync via the env path."""
    from accelerate.utils import is_peft_model
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig

    from src.configs.async_training_config import AsyncTrainingConfig
    from src.distributed.parallelism_config import ParallelismConfig
    from src.environments.envs.protocols.react import ReActEnvironment
    from src.environments.tools.factories import create_native_math_tools, create_native_python_tools
    from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer

    output_dir = tempfile.mkdtemp(prefix="test_env_grpo_lora_vllm_e2e_")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.bfloat16, trust_remote_code=True)

        grpo_config = GRPOConfig(
            output_dir=output_dir,
            max_steps=2,
            per_device_train_batch_size=2,
            gradient_accumulation_steps=1,
            learning_rate=5e-6,
            bf16=True,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            num_generations=2,
            max_completion_length=256,
            beta=0.01,
            generation_kwargs={"temperature": 0.7},
            fsdp="",
            reward_weights=[1.0],
        )
        async_config = AsyncTrainingConfig(
            rollout_server_configs=rollout_server_configs(),
            rollout_connection_timeout=60.0,
            num_rollout_workers=2,
            max_concurrent_rollouts=4,
            rollout_temperature=0.7,
            rollout_max_tokens=128,
            enable_prefetch=False,
            sync_weights_every_n_steps=1,
            model_name=MODEL_NAME,
        )
        registry = create_native_math_tools()
        for tool in create_native_python_tools().list_tools():
            registry.register(tool)
        dataset = Dataset.from_list(
            [
                {"prompt": [{"role": "user", "content": f"What is {a} + {b}?"}], "answer": str(a + b)}
                for a, b in [(2, 3), (5, 7), (10, 15), (3, 8), (6, 4), (9, 1), (12, 8), (7, 13)]
            ]
        )
        trainer = DistributedAsyncEnvironmentalGRPOTrainer(
            model=model,
            args=grpo_config,
            train_dataset=dataset,
            processing_class=tokenizer,
            async_config=async_config,
            environment_cls=ReActEnvironment,
            environment_kwargs={"tool_registry": registry, "max_turns": 3, "success_reward": 1.0},
            parallelism_config=ParallelismConfig(),
            peft_config=LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"),
        )
        assert is_peft_model(trainer.accelerator.unwrap_model(trainer.model)), (
            "peft_config did not produce a PeftModel"
        )
        log("  Trainer created (PeftModel). Training + syncing merged LoRA weights to vLLM...")
        trainer.train()
        log("  Environmental GRPO+LoRA training completed (PEFT weight sync OK)!")
    finally:
        cleanup_memory()
        shutil.rmtree(output_dir, ignore_errors=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        required=True,
        choices=("online", "sdpg", "online_lora", "environmental", "environmental_lora"),
        help="which leg to run; exactly one per invocation (the server's NCCL state is not reusable)",
    )
    return parser.parse_args()


def run(ctx) -> dict:
    """Drive the leg named by ``--mode`` against the live vLLM server."""
    log(f"\n{'=' * 70}")
    log("  Online GRPO End-to-End Test (Live vLLM Server)")
    log(f"  Model: {MODEL_NAME}")
    log(f"  vLLM Server: {VLLM_SERVER_URL}")
    log(f"  GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    log(f"{'=' * 70}")

    checks: dict[str, bool] = {}
    test_mode = _parse_args().mode

    log("\n--- vLLM Server Connectivity ---")
    record_check(checks, "vLLM server reachable", test_vllm_server_reachable)
    record_check(checks, "vLLM generation works", test_vllm_generation)

    if test_mode == "online":
        log("\n--- Online GRPO E2E (DistributedGRPOTrainer + vLLM) ---")
        record_check(checks, "Online GRPO training (3 steps)", test_online_grpo_e2e)

    if test_mode == "sdpg":
        log("\n--- Online SDPG E2E (DistributedSDPGTrainer + vLLM) ---")
        record_check(checks, "Online SDPG training (3 steps)", test_online_sdpg_e2e)

    if test_mode == "online_lora":
        log("\n--- Online GRPO + LoRA E2E (DistributedGRPOTrainer + PEFT + vLLM) ---")
        record_check(checks, "Online GRPO+LoRA training (3 steps)", test_online_grpo_lora_e2e)

    if test_mode == "environmental_lora":
        log("\n--- Environmental GRPO + LoRA E2E (DistributedAsyncEnvironmentalGRPOTrainer + PEFT + vLLM) ---")
        record_check(checks, "Environmental GRPO+LoRA training (2 steps)", test_environmental_grpo_lora_e2e)

    if test_mode == "environmental":
        log("\n--- Environmental GRPO E2E (DistributedAsyncEnvironmentalGRPOTrainer + vLLM) ---")
        record_check(checks, "Environmental GRPO training (2 steps)", test_environmental_grpo_e2e)

    return {"checks": checks}


main = gpu_test_main(exact_world_size=1, prefix="online_grpo_vllm_e2e", partial_state=False)(run)

if __name__ == "__main__":
    main()
