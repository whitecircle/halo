#!/usr/bin/env python
"""
End-to-end test for SFT dataset caching: simulates the sft.py data pipeline.

Verifies:
1. First run: coordinated_map + coordinated_filter create cache files
2. Second run: same pipeline loads from cache (no new cache files)
3. Both ranks get identical results after coordination
4. Cache key changes when tokenizer/processor changes
5. DatasetDict (train/test splits) cached correctly per-split

Every test raises on failure and is recorded as its own check; each ends its collectives on every
rank before it can raise, so a failure never strands a peer inside one.

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/data/test_sft_caching_e2e.py
"""

import os
import time

import torch
import torch.distributed as dist
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from src.data.pipeline.processing import process_dataset_with_map_and_filter
from src.data.pipeline.row_processors import create_llm_processor
from src.distributed.runtime import barrier
from tests.common.harness import gpu_test_main, record_check
from tests.common.models import QWEN3_0_6B
from tests.common.utils import log

# Configuration

MODEL_NAME = QWEN3_0_6B
ALT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_LENGTH = 256
NUM_TRAIN = 50
NUM_TEST = 10
CONVERSATION_FIELD = "prompt"


# Utilities


def create_conversation_splits(num_train: int = NUM_TRAIN, num_test: int = NUM_TEST) -> DatasetDict:
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


def test_sft_pipeline_creates_cache(cache_dir: str, tokenizer) -> None:
    """Test 1: Full sft.py pipeline creates cache files on first run."""
    ds = create_conversation_splits()

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

    assert isinstance(result, DatasetDict), f"Expected DatasetDict, got {type(result)}"
    assert "train" in result and "test" in result, f"Missing splits. Got: {list(result.keys())}"
    assert len(result["train"]) > 0, "Empty train split"
    assert len(result["test"]) > 0, "Empty test split"

    sample = result["train"][0]
    assert "input_ids" in sample, f"No input_ids in result. Keys: {list(sample.keys())}"
    assert len(sample["input_ids"]) > 0, "Empty input_ids"

    num_cache = count_cache_files(cache_dir)
    assert num_cache > 0, f"No cache files created in {cache_dir}"

    log(f"    Pipeline produced train={len(result['train'])}, test={len(result['test'])}; cache files: {num_cache}")


def test_sft_pipeline_reuses_cache(cache_dir: str, tokenizer) -> None:
    """Test 2: Second run reuses cache: it writes no new cache file."""
    ds = create_conversation_splits()

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

    # A reused cache is loaded, not rewritten: a new file means the cache key changed between runs.
    cache_after = count_cache_files(cache_dir)
    assert cache_after == cache_before, f"cache not reused: .arrow files {cache_before} -> {cache_after}"
    assert len(result["train"]) > 0, "Empty train split on cache reuse"

    log(f"    Cache reused in {elapsed:.3f}s (cache files: {cache_before} -> {cache_after})")


def test_both_ranks_get_identical_results(cache_dir: str, tokenizer) -> None:
    """Test 3: Both ranks produce identical results after coordination."""
    ds = create_conversation_splits()

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
    device = torch.device("cuda", torch.cuda.current_device())
    local_ids = result["train"][0]["input_ids"]

    # Pad to same length for all_gather. -1 is no token id, so rows of different lengths never
    # compare equal through the padding.
    max_len = torch.tensor(len(local_ids), device=device)
    dist.all_reduce(max_len, op=dist.ReduceOp.MAX)

    padded = torch.full((int(max_len.item()),), -1, dtype=torch.long, device=device)
    padded[: len(local_ids)] = torch.tensor(local_ids, device=device)

    gathered = [torch.zeros_like(padded) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, padded)

    # Every rank holds every row after the gather, so every rank reaches the same verdict.
    mismatched = [i for i, g in enumerate(gathered) if not torch.equal(gathered[0], g)]
    assert not mismatched, f"ranks {mismatched} produced input_ids differing from rank 0's: " + "; ".join(
        f"rank {i}: {g[:10].tolist()}..." for i, g in enumerate(gathered)
    )
    log(f"    All ranks produced identical results (first 10 tokens: {gathered[0][:10].tolist()})")


def test_different_tokenizer_produces_different_cache(cache_dir: str, tokenizer, alt_tokenizer) -> None:
    """Test 4: Different tokenizer produces different cache (no stale data)."""
    ds = create_conversation_splits(num_train=20, num_test=5)

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
    ds_b = create_conversation_splits(num_train=20, num_test=5)
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

    assert ids_a != ids_b, "Different tokenizers produced the same token IDs: a stale cache was used"

    log(f"    Different tokenizers -> different token IDs (A first 10: {ids_a[:10]}, B first 10: {ids_b[:10]})")


def test_generate_dataset_separate_cache(cache_dir: str, tokenizer) -> None:
    """Test 5: Generate dataset (add_generation_prompt=True) uses separate cache."""
    ds = create_conversation_splits()

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

    assert len(train_result["train"]) > 0, "Empty train result"
    assert len(gen_result) > 0, "Empty generate result"

    # The generation render drops the final assistant turn and appends the generation prompt, so the
    # same row can only tokenize identically if the generate pass was served the training rows.
    train_ids = train_result["test"][0]["input_ids"]
    gen_ids = gen_result[0]["input_ids"]
    assert train_ids != gen_ids, "the generate pass returned the training rows' token IDs"

    log(
        f"    Train ({len(train_result['train'])} examples) and generate ({len(gen_result)} examples) processed separately"
    )


def test_cache_survives_process_restart(cache_dir: str, tokenizer) -> None:
    """Test 6: Simulate process restart — cache from prior run is picked up."""
    ds = create_conversation_splits(num_train=30, num_test=8)
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
    ds2 = create_conversation_splits(num_train=30, num_test=8)
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

    assert ids1 == ids2, "Results differ after simulated restart"
    assert cache_count_after_second == cache_count_after_first, (
        f"cache not reused after the simulated restart: .arrow files "
        f"{cache_count_after_first} -> {cache_count_after_second}"
    )

    log(
        f"    Cache reused after simulated restart ({elapsed:.3f}s, files: {cache_count_after_first} -> {cache_count_after_second})"
    )


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

    # Order matters: the reuse checks read the cache the creation check wrote.
    checks: dict[str, bool] = {}
    record_check(checks, "sft_pipeline_creates_cache", lambda: test_sft_pipeline_creates_cache(cache_dir, tokenizer))
    record_check(checks, "sft_pipeline_reuses_cache", lambda: test_sft_pipeline_reuses_cache(cache_dir, tokenizer))
    record_check(
        checks,
        "both_ranks_get_identical_results",
        lambda: test_both_ranks_get_identical_results(cache_dir, tokenizer),
    )
    record_check(
        checks,
        "different_tokenizer_produces_different_cache",
        lambda: test_different_tokenizer_produces_different_cache(cache_dir, tokenizer, alt_tokenizer),
    )
    record_check(
        checks, "generate_dataset_separate_cache", lambda: test_generate_dataset_separate_cache(cache_dir, tokenizer)
    )
    record_check(
        checks, "cache_survives_process_restart", lambda: test_cache_survives_process_restart(cache_dir, tokenizer)
    )
    return {"checks": checks}


main = gpu_test_main(min_world_size=2, prefix="sft_cache_e2e", partial_state=False)(run)

if __name__ == "__main__":
    main()
