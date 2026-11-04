#!/usr/bin/env python
"""The sft.py data pipeline's cache: created once, reused on rerun, per-tokenizer, per-split and across a restart.

Each test ends its collectives on every rank before it can raise, so a failure never strands a peer.
"""

import os
import time

import torch
import torch.distributed as dist
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

from halo.data.pipeline.coordinated_map import coordinated_map_and_filter
from halo.data.pipeline.text_rows import build_chat_row_processor
from halo.runtime.collectives import barrier
from tests.common.harness import gpu_test_main, record_check
from tests.common.models import QWEN3_0_6B
from tests.common.rank_log import log

MODEL_NAME = QWEN3_0_6B
ALT_MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_LENGTH = 256
NUM_TRAIN = 50
NUM_TEST = 10
CONVERSATION_FIELD = "prompt"


def create_conversation_splits(num_train: int = NUM_TRAIN, num_test: int = NUM_TEST) -> DatasetDict:
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
    if not os.path.exists(cache_dir):
        return 0
    return len([f for f in os.listdir(cache_dir) if f.endswith(".arrow")])


def test_sft_pipeline_creates_cache(cache_dir: str, tokenizer) -> None:
    ds = create_conversation_splits()

    processor = build_chat_row_processor(
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        conversation_field=CONVERSATION_FIELD,
    )

    extra_columns = list(set(ds["train"].column_names))

    result = coordinated_map_and_filter(
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
    """A reused cache writes no new file."""
    ds = create_conversation_splits()

    processor = build_chat_row_processor(
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        conversation_field=CONVERSATION_FIELD,
    )
    extra_columns = list(set(ds["train"].column_names))

    cache_before = count_cache_files(cache_dir)

    barrier()
    start = time.monotonic()

    result = coordinated_map_and_filter(
        ds,
        processor,
        remove_columns=extra_columns,
        desc="sft tokenization",
    )

    barrier()
    elapsed = time.monotonic() - start

    # A new file means the cache key changed between runs.
    cache_after = count_cache_files(cache_dir)
    assert cache_after == cache_before, f"cache not reused: .arrow files {cache_before} -> {cache_after}"
    assert len(result["train"]) > 0, "Empty train split on cache reuse"

    log(f"    Cache reused in {elapsed:.3f}s (cache files: {cache_before} -> {cache_after})")


def test_both_ranks_get_identical_results(cache_dir: str, tokenizer) -> None:
    ds = create_conversation_splits()

    processor = build_chat_row_processor(
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        conversation_field=CONVERSATION_FIELD,
    )
    extra_columns = list(set(ds["train"].column_names))

    result = coordinated_map_and_filter(
        ds,
        processor,
        remove_columns=extra_columns,
        desc="sft tokenization",
    )

    device = torch.device("cuda", torch.cuda.current_device())
    local_ids = result["train"][0]["input_ids"]

    # -1 is no token id, so rows of different lengths never compare equal through the padding.
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
    ds = create_conversation_splits(num_train=20, num_test=5)

    processor_a = build_chat_row_processor(
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        conversation_field=CONVERSATION_FIELD,
    )
    extra_columns = list(set(ds["train"].column_names))
    result_a = coordinated_map_and_filter(
        ds,
        processor_a,
        remove_columns=extra_columns,
        desc="tokenizer_a",
    )

    processor_b = build_chat_row_processor(
        tokenizer=alt_tokenizer,
        max_length=MAX_LENGTH,
        conversation_field=CONVERSATION_FIELD,
    )
    ds_b = create_conversation_splits(num_train=20, num_test=5)
    extra_columns_b = list(set(ds_b["train"].column_names))
    result_b = coordinated_map_and_filter(
        ds_b,
        processor_b,
        remove_columns=extra_columns_b,
        desc="tokenizer_b",
    )

    ids_a = result_a["train"][0]["input_ids"]
    ids_b = result_b["train"][0]["input_ids"]

    assert ids_a != ids_b, "Different tokenizers produced the same token IDs: a stale cache was used"

    log(f"    Different tokenizers -> different token IDs (A first 10: {ids_a[:10]}, B first 10: {ids_b[:10]})")


def test_generate_dataset_separate_cache(cache_dir: str, tokenizer) -> None:
    ds = create_conversation_splits()

    train_processor = build_chat_row_processor(
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        conversation_field=CONVERSATION_FIELD,
        add_generation_prompt=False,
    )

    # Different function body.
    generate_processor = build_chat_row_processor(
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        conversation_field=CONVERSATION_FIELD,
        add_generation_prompt=True,
    )

    extra_columns = list(set(ds["train"].column_names))

    train_result = coordinated_map_and_filter(
        ds,
        train_processor,
        remove_columns=extra_columns,
        desc="train processing",
    )

    # Separately for generation, as sft.py does.
    gen_result = coordinated_map_and_filter(
        ds["test"],
        generate_processor,
        desc="generate processing",
    )

    assert len(train_result["train"]) > 0, "Empty train result"
    assert len(gen_result) > 0, "Empty generate result"

    # The generation render drops the final assistant turn, so identical tokens mean it was served the training rows.
    train_ids = train_result["test"][0]["input_ids"]
    gen_ids = gen_result[0]["input_ids"]
    assert train_ids != gen_ids, "the generate pass returned the training rows' token IDs"

    log(
        f"    Train ({len(train_result['train'])} examples) and generate ({len(gen_result)} examples) processed separately"
    )


def test_cache_survives_process_restart(cache_dir: str, tokenizer) -> None:
    ds = create_conversation_splits(num_train=30, num_test=8)
    processor = build_chat_row_processor(
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        conversation_field=CONVERSATION_FIELD,
    )
    extra_columns = list(set(ds["train"].column_names))

    result1 = coordinated_map_and_filter(
        ds,
        processor,
        remove_columns=extra_columns,
        desc="restart test",
    )
    cache_count_after_first = count_cache_files(cache_dir)

    # Fresh dataset objects simulate a restart.
    ds2 = create_conversation_splits(num_train=30, num_test=8)
    processor2 = build_chat_row_processor(
        tokenizer=tokenizer,
        max_length=MAX_LENGTH,
        conversation_field=CONVERSATION_FIELD,
    )
    extra_columns2 = list(set(ds2["train"].column_names))

    barrier()
    start = time.monotonic()

    result2 = coordinated_map_and_filter(
        ds2,
        processor2,
        remove_columns=extra_columns2,
        desc="restart test",
    )

    barrier()
    elapsed = time.monotonic() - start

    cache_count_after_second = count_cache_files(cache_dir)

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
    # Every test shares setup_cache_dirs's HF_DATASETS_CACHE, which makes the reuse and restart checks meaningful.
    cache_dir = ctx.cache_dir

    log(f"\n{'=' * 70}")
    log("  SFT Caching End-to-End Tests (simulating sft.py pipeline)")
    log(f"  World size: {ctx.world_size}, Rank: {ctx.rank}")
    log(f"  GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log(f"  Cache dir: {cache_dir}")
    log(f"{'=' * 70}\n")

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
