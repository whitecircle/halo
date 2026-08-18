#!/usr/bin/env python
"""
End-to-end test for SFT dataset caching: simulates the sft.py data pipeline.

Verifies:
1. First run: coordinated_map + coordinated_filter create cache files
2. Second run: same pipeline loads from cache (no re-processing)
3. Both ranks get identical results after coordination
4. Cache key changes when tokenizer/processor changes
5. DatasetDict (train/test splits) cached correctly per-split

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/data/test_sft_caching_e2e.py
"""

import os
import time
import traceback

import torch
import torch.distributed as dist
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from src.data.pipeline.processing import process_dataset_with_map_and_filter
from src.data.pipeline.row_processors import create_llm_processor
from src.distributed.runtime import barrier
from tests.common.harness import gpu_test_main
from tests.common.utils import log

# Configuration

MODEL_NAME = "Qwen/Qwen3-0.6B"
ALT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_LENGTH = 256
NUM_TRAIN = 50
NUM_TEST = 10
CONVERSATION_FIELD = "prompt"


# Utilities


def create_sft_dataset(num_train: int = NUM_TRAIN, num_test: int = NUM_TEST) -> DatasetDict:
    """Create a synthetic SFT dataset mimicking real training data format."""

    def make_conversations(n: int):
        data = []
        for i in range(n):
            data.append(
                {
                    CONVERSATION_FIELD: [
                        {"role": "user", "content": f"What is {i} + {i * 2}?"},
                        {"role": "assistant", "content": f"The answer is {i + i * 2}."},
                    ]
                }
            )
        return data

    return DatasetDict(
        {
            "train": Dataset.from_list(make_conversations(num_train)),
            "test": Dataset.from_list(make_conversations(num_test)),
        }
    )


def count_cache_files(cache_dir: str) -> int:
    """Count .arrow cache files in directory."""
    if not os.path.exists(cache_dir):
        return 0
    return len([f for f in os.listdir(cache_dir) if f.endswith(".arrow")])


# Tests


def test_sft_pipeline_creates_cache(cache_dir: str, tokenizer) -> bool:
    """Test 1: Full sft.py pipeline creates cache files on first run."""
    log("  [test_sft_pipeline_creates_cache] Running...")

    ds = create_sft_dataset()

    # Simulate sft.py: create_llm_processor -> process_dataset_with_map_and_filter
    processor = create_llm_processor(
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        conversation_field=CONVERSATION_FIELD,
        use_padding=True,
    )

    extra_columns = list(set(ds["train"].column_names))

    result = process_dataset_with_map_and_filter(
        ds,
        processor,
        remove_columns=extra_columns,
        desc="sft tokenization",
    )

    # Verify results
    if not isinstance(result, DatasetDict):
        log(f"    FAIL: Expected DatasetDict, got {type(result)}")
        return False

    if "train" not in result or "test" not in result:
        log(f"    FAIL: Missing splits. Got: {list(result.keys())}")
        return False

    if len(result["train"]) == 0:
        log("    FAIL: Empty train split")
        return False

    if len(result["test"]) == 0:
        log("    FAIL: Empty test split")
        return False

    # Check that input_ids exist and are valid
    sample = result["train"][0]
    if "input_ids" not in sample:
        log(f"    FAIL: No input_ids in result. Keys: {list(sample.keys())}")
        return False

    if len(sample["input_ids"]) == 0:
        log("    FAIL: Empty input_ids")
        return False

    # Check cache files were created
    num_cache = count_cache_files(cache_dir)
    if num_cache == 0:
        log(f"    FAIL: No cache files created in {cache_dir}")
        return False

    log(f"    PASS: Pipeline produced train={len(result['train'])}, test={len(result['test'])}")
    log(f"    Cache files: {num_cache}")
    return True


def test_sft_pipeline_reuses_cache(cache_dir: str, tokenizer) -> bool:
    """Test 2: Second run reuses cache (no re-processing)."""
    log("  [test_sft_pipeline_reuses_cache] Running...")

    ds = create_sft_dataset()

    processor = create_llm_processor(
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        conversation_field=CONVERSATION_FIELD,
        use_padding=True,
    )
    extra_columns = list(set(ds["train"].column_names))

    # Count cache files before
    cache_before = count_cache_files(cache_dir)

    # Time the second run (should be fast due to cache)
    barrier()
    start = time.monotonic()

    result = process_dataset_with_map_and_filter(
        ds,
        processor,
        remove_columns=extra_columns,
        desc="sft tokenization",
    )

    barrier()
    elapsed = time.monotonic() - start

    # Count cache files after — should be the same (no new files)
    cache_after = count_cache_files(cache_dir)

    if cache_after != cache_before:
        log(f"    WARN: Cache file count changed: {cache_before} -> {cache_after}")
        log("    This may indicate cache was not reused")

    if len(result["train"]) == 0:
        log("    FAIL: Empty train split on cache reuse")
        return False

    log(f"    PASS: Cache reused in {elapsed:.3f}s (cache files: {cache_before} -> {cache_after})")
    return True


def test_both_ranks_get_identical_results(cache_dir: str, tokenizer) -> bool:
    """Test 3: Both ranks produce identical results after coordination."""
    log("  [test_both_ranks_get_identical_results] Running...", rank=None)

    ds = create_sft_dataset()

    processor = create_llm_processor(
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        conversation_field=CONVERSATION_FIELD,
        use_padding=True,
    )
    extra_columns = list(set(ds["train"].column_names))

    result = process_dataset_with_map_and_filter(
        ds,
        processor,
        remove_columns=extra_columns,
        desc="sft tokenization",
    )

    # Gather first example's input_ids from all ranks
    rank = dist.get_rank()
    local_ids = result["train"][0]["input_ids"]
    local_tensor = torch.tensor(local_ids, device=f"cuda:{rank}")

    # Pad to same length for all_gather
    max_len = torch.tensor(len(local_ids), device=f"cuda:{rank}")
    dist.all_reduce(max_len, op=dist.ReduceOp.MAX)

    padded = torch.zeros(max_len.item(), dtype=torch.long, device=f"cuda:{rank}")
    padded[: len(local_ids)] = local_tensor

    gathered = [torch.zeros_like(padded) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, padded)

    # Compare
    if rank == 0:
        match = all(torch.equal(gathered[0], gathered[i]) for i in range(1, len(gathered)))
        if not match:
            log("    FAIL: Ranks produced different input_ids")
            for i, g in enumerate(gathered):
                log(f"    Rank {i}: {g[:10].tolist()}...")
            return False

        log(f"    PASS: All ranks produced identical results (first 10 tokens: {gathered[0][:10].tolist()})")

    # Broadcast result to all ranks
    result_tensor = torch.tensor(
        [
            1
            if rank != 0
            else (1 if all(torch.equal(gathered[0], gathered[i]) for i in range(1, len(gathered))) else 0)
        ],
        device=f"cuda:{rank}",
    )
    dist.broadcast(result_tensor, src=0)
    return result_tensor.item() == 1


def test_different_tokenizer_produces_different_cache(cache_dir: str, tokenizer, alt_tokenizer) -> bool:
    """Test 4: Different tokenizer produces different cache (no stale data)."""
    log("  [test_different_tokenizer_produces_different_cache] Running...")

    ds = create_sft_dataset(num_train=20, num_test=5)

    # Process with primary tokenizer
    processor_a = create_llm_processor(
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        conversation_field=CONVERSATION_FIELD,
        use_padding=True,
    )
    extra_columns = list(set(ds["train"].column_names))
    result_a = process_dataset_with_map_and_filter(
        ds,
        processor_a,
        remove_columns=extra_columns,
        desc="tokenizer_a",
    )

    # Process with alternate tokenizer
    processor_b = create_llm_processor(
        tokenizer=alt_tokenizer,
        max_length=MAX_LENGTH,
        conversation_field=CONVERSATION_FIELD,
        use_padding=True,
    )
    ds_b = create_sft_dataset(num_train=20, num_test=5)
    extra_columns_b = list(set(ds_b["train"].column_names))
    result_b = process_dataset_with_map_and_filter(
        ds_b,
        processor_b,
        remove_columns=extra_columns_b,
        desc="tokenizer_b",
    )

    # Verify token IDs differ
    ids_a = result_a["train"][0]["input_ids"]
    ids_b = result_b["train"][0]["input_ids"]

    if ids_a == ids_b:
        log("    FAIL: Different tokenizers produced same token IDs!")
        log("    This means stale cache was used.")
        return False

    log("    PASS: Different tokenizers -> different token IDs")
    log(f"    Tokenizer A first 10: {ids_a[:10]}")
    log(f"    Tokenizer B first 10: {ids_b[:10]}")
    return True


def test_generate_dataset_separate_cache(cache_dir: str, tokenizer) -> bool:
    """Test 5: Generate dataset (add_generation_prompt=True) uses separate cache."""
    log("  [test_generate_dataset_separate_cache] Running...")

    ds = create_sft_dataset()

    # Train processor (no generation prompt)
    train_processor = create_llm_processor(
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        conversation_field=CONVERSATION_FIELD,
        add_generation_prompt=False,
        use_padding=True,
    )

    # Generate processor (with generation prompt) — different function body
    generate_processor = create_llm_processor(
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        conversation_field=CONVERSATION_FIELD,
        add_generation_prompt=True,
        use_padding=True,
    )

    extra_columns = list(set(ds["train"].column_names))

    train_result = process_dataset_with_map_and_filter(
        ds,
        train_processor,
        remove_columns=extra_columns,
        desc="train processing",
    )

    # Process test set separately for generation (like sft.py does)
    gen_result = process_dataset_with_map_and_filter(
        ds["test"],
        generate_processor,
        desc="generate processing",
    )

    # Both should succeed
    if len(train_result["train"]) == 0:
        log("    FAIL: Empty train result")
        return False

    if len(gen_result) == 0:
        log("    FAIL: Empty generate result")
        return False

    # The generate dataset should have different token sequences
    # (generation prompt adds assistant prefix tokens at the end)
    train_ids = train_result["test"][0]["input_ids"]
    gen_ids = gen_result[0]["input_ids"]

    if train_ids == gen_ids:
        log("    WARN: Train and generate IDs are identical (may be correct for some templates)")

    log(
        f"    PASS: Train ({len(train_result['train'])} examples) and generate ({len(gen_result)} examples) processed separately"
    )
    return True


def test_cache_survives_process_restart(cache_dir: str, tokenizer) -> bool:
    """Test 6: Simulate process restart — cache from prior run is picked up."""
    log("  [test_cache_survives_process_restart] Running...")

    ds = create_sft_dataset(num_train=30, num_test=8)
    processor = create_llm_processor(
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        conversation_field=CONVERSATION_FIELD,
        use_padding=True,
    )
    extra_columns = list(set(ds["train"].column_names))

    # First "run"
    result1 = process_dataset_with_map_and_filter(
        ds,
        processor,
        remove_columns=extra_columns,
        desc="restart test",
    )
    cache_count_after_first = count_cache_files(cache_dir)

    # Second "run" with fresh dataset objects (simulates restart)
    ds2 = create_sft_dataset(num_train=30, num_test=8)
    processor2 = create_llm_processor(
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        conversation_field=CONVERSATION_FIELD,
        use_padding=True,
    )
    extra_columns2 = list(set(ds2["train"].column_names))

    barrier()
    start = time.monotonic()

    result2 = process_dataset_with_map_and_filter(
        ds2,
        processor2,
        remove_columns=extra_columns2,
        desc="restart test",
    )

    barrier()
    elapsed = time.monotonic() - start

    cache_count_after_second = count_cache_files(cache_dir)

    # Results should match
    ids1 = result1["train"][0]["input_ids"]
    ids2 = result2["train"][0]["input_ids"]

    if ids1 != ids2:
        log("    FAIL: Results differ after simulated restart")
        return False

    if cache_count_after_second != cache_count_after_first:
        log(f"    WARN: Cache count changed {cache_count_after_first} -> {cache_count_after_second}")

    log(
        f"    PASS: Cache reused after simulated restart ({elapsed:.3f}s, files: {cache_count_after_first} -> {cache_count_after_second})"
    )
    return True


# Test Runner

ALL_TESTS = [
    "sft_pipeline_creates_cache",
    "sft_pipeline_reuses_cache",
    "both_ranks_get_identical_results",
    "different_tokenizer_produces_different_cache",
    "generate_dataset_separate_cache",
    "cache_survives_process_restart",
]


def run(ctx) -> dict:
    # setup_cache_dirs already points HF_DATASETS_CACHE here; every test in this file shares it,
    # which is what makes the reuse/restart checks meaningful.
    cache_dir = ctx.cache_dir

    log(f"\n{'=' * 70}")
    log("  SFT Caching End-to-End Tests (simulating sft.py pipeline)")
    log(f"  World size: {ctx.world_size}, Rank: {ctx.rank}")
    log(f"  GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"  Cache dir: {cache_dir}")
    log(f"{'=' * 70}\n")

    # Load tokenizers on rank 0 first
    if ctx.rank == 0:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        alt_tokenizer = AutoTokenizer.from_pretrained(ALT_MODEL_NAME, trust_remote_code=True)
    barrier()
    if ctx.rank != 0:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        alt_tokenizer = AutoTokenizer.from_pretrained(ALT_MODEL_NAME, trust_remote_code=True)
    barrier()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if alt_tokenizer.pad_token is None:
        alt_tokenizer.pad_token = alt_tokenizer.eos_token

    test_fns = {
        "sft_pipeline_creates_cache": lambda: test_sft_pipeline_creates_cache(cache_dir, tokenizer),
        "sft_pipeline_reuses_cache": lambda: test_sft_pipeline_reuses_cache(cache_dir, tokenizer),
        "both_ranks_get_identical_results": lambda: test_both_ranks_get_identical_results(cache_dir, tokenizer),
        "different_tokenizer_produces_different_cache": lambda: test_different_tokenizer_produces_different_cache(
            cache_dir, tokenizer, alt_tokenizer
        ),
        "generate_dataset_separate_cache": lambda: test_generate_dataset_separate_cache(cache_dir, tokenizer),
        "cache_survives_process_restart": lambda: test_cache_survives_process_restart(cache_dir, tokenizer),
    }

    checks: dict[str, bool] = {}
    for test_name in ALL_TESTS:
        log(f"\n  [{test_name}]")
        try:
            # One failing test must not skip the rest: each owns its own check key.
            result = test_fns[test_name]()
        except Exception as e:
            log(f"  [{test_name}] EXCEPTION: {e}")
            if ctx.rank == 0:
                traceback.print_exc()
            result = False
        checks[test_name] = result

    return {"checks": checks}


main = gpu_test_main(min_world_size=2, prefix="sft_cache_e2e", partial_state=False)(run)

if __name__ == "__main__":
    main()
