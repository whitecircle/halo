"""Reference storage remains bounded, mapped and filesystem-aware."""

import datetime
import errno
import os
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from datasets import Dataset

import src.trainers.grpo.reference_cache as cache_module
from src.checkpoint.format import REFERENCE_CACHE_DIR_NAME
from src.data.collators.offline_grpo import REF_PER_TOKEN_LOGPS_COLUMN
from src.trainers.grpo.reference_cache import ReferenceScoreCache, reference_cache_writers
from src.training.environment import _validate_output_dir
from tests.common.distributed import shared_output_dir
from tests.common.gloo import run_gloo_ranks
from tests.common.offline_grpo_reference import (
    SETTINGS,
    ReferenceStorageTrainer,
    attach_reference,
    mapped_scores,
    reference_dataset,
    reference_rows,
    restore_reference,
)


@pytest.mark.parametrize("owners", [(1, 0, 0, 0), (0, 1, 0, 1), (1, 1, 1, 1)])
def test_cache_writer_election_gathers_the_checkpoint_owner_predicate(monkeypatch, owners):
    monkeypatch.setattr(cache_module, "get_global_world_size", lambda: len(owners))
    monkeypatch.setattr(cache_module, "fs_aware_save_rank", lambda: bool(owners[1]))

    def gather(output, flag):
        assert flag.item() == owners[1]
        for tensor, owner in zip(output, owners, strict=True):
            tensor.fill_(owner)

    monkeypatch.setattr(dist, "all_gather", gather)
    assert reference_cache_writers() == tuple(rank for rank, owner in enumerate(owners) if owner)


def _map_shared_cache_from_rank_local_scratch(rank: int, root: str, broadcast_output: bool) -> None:
    rank_directory = os.path.join(root, f"rank-{rank}")
    os.makedirs(rank_directory)
    context = SimpleNamespace(rank=rank, output_dir=rank_directory)
    output_dir = shared_output_dir(context) if broadcast_output else context.output_dir
    dataset = Dataset.from_dict({"completion_input_ids": [[1], [2, 3]]})
    scores = [torch.tensor([-0.25]), torch.tensor([-0.75, -1.5])]
    cache = ReferenceScoreCache(output_dir, dp_size=2)
    cache.collect_batch([scores[rank]], {0: 0, 1: 1})
    mapped = cache.finish(dataset)
    assert mapped.column().to_pylist() == [[-0.25], [-0.75, -1.5]]
    assert mapped.lengths.tolist() == [1, 2]


def test_shared_reference_cache_maps_from_distinct_rank_local_scratch_directories(tmp_path):
    run_gloo_ranks(
        _map_shared_cache_from_rank_local_scratch,
        2,
        str(tmp_path),
        True,
        env={"DIST_OUTPUT_SHARED_FILESYSTEM": "1"},
        pg_timeout=datetime.timedelta(seconds=30),
    )


def test_rank_local_output_negative_control_cannot_map_the_shared_reference_cache(tmp_path):
    with pytest.raises(torch.multiprocessing.ProcessRaisedException, match="No such file or directory.*merged"):
        run_gloo_ranks(
            _map_shared_cache_from_rank_local_scratch,
            2,
            str(tmp_path),
            False,
            env={"DIST_OUTPUT_SHARED_FILESYSTEM": "1"},
            pg_timeout=datetime.timedelta(seconds=30),
        )


def _failed_reader_mapping(rank: int, root: str, shared_output: bool) -> None:
    if not shared_output:
        os.environ["LOCAL_RANK"] = "0"
        os.environ["LOCAL_WORLD_SIZE"] = "1"
    output = root if shared_output else os.path.join(root, f"node-{rank}")
    dataset = Dataset.from_dict({"completion_input_ids": [[1], [2, 3]]})
    rows = [torch.tensor([-0.25]), torch.tensor([-0.75, -1.5])]
    cache = ReferenceScoreCache(output, dp_size=2)
    cache.collect_batch([rows[rank]], {0: 0, 1: 1})
    patch = pytest.MonkeyPatch()
    original_map = cache._map
    map_calls = 0

    def fail_reader(shard):
        nonlocal map_calls
        map_calls += 1
        # A filesystem writer first maps to validate its merge, then maps as a reader.
        if rank == 1 and map_calls == (1 if shared_output else 2):
            raise OSError("rank-1 reference mapping denied")
        return original_map(shard)

    patch.setattr(cache, "_map", fail_reader)
    try:
        cache.finish(dataset)
        outcome = "NO RAISE"
    except Exception as exc:
        outcome = f"{type(exc).__name__}: {exc}"
    finally:
        patch.undo()
    with open(os.path.join(root, f"mapping-outcome-{rank}.txt"), "w") as result:
        result.write(outcome)


@pytest.mark.parametrize("shared_output", [True, False])
def test_one_reader_mapping_failure_is_joined_before_scratch_is_removed(tmp_path, shared_output):
    run_gloo_ranks(
        _failed_reader_mapping,
        2,
        str(tmp_path),
        shared_output,
        env={"DIST_OUTPUT_SHARED_FILESYSTEM": "1" if shared_output else "0"},
        pg_timeout=datetime.timedelta(seconds=10),
    )
    outcomes = [(tmp_path / f"mapping-outcome-{rank}.txt").read_text() for rank in range(2)]
    assert outcomes[0] == outcomes[1]
    assert outcomes[0] == (
        "ValueError: Mapping the offline GRPO reference cache failed on 1 of 2 rank(s) [1]. "
        "First (rank 1): OSError: rank-1 reference mapping denied"
    )
    assert not list(tmp_path.rglob("*.lengths"))
    assert not list(tmp_path.rglob("*.values"))


def test_ephemeral_cache_needs_no_durable_publication(tmp_path, monkeypatch):
    dataset = Dataset.from_dict({"completion_input_ids": [[1], [2, 3]]})
    cache = ReferenceScoreCache(tmp_path, dp_size=1)
    cache.append_rows(0, [torch.tensor([-0.25]), torch.tensor([-0.75, -1.5])])

    def unavailable(*args):
        raise OSError("durable publication unavailable on scratch storage")

    monkeypatch.setattr(os, "fsync", unavailable)
    monkeypatch.setattr(os, "replace", unavailable)
    mapped = cache.finish(dataset)
    assert mapped.column().to_pylist() == [[-0.25], [-0.75, -1.5]]
    assert not list((tmp_path / REFERENCE_CACHE_DIR_NAME).iterdir())
    _validate_output_dir(str(tmp_path))


def test_reference_cache_maps_one_arrow_token_buffer_and_serializes_it_without_repacking(tmp_path, monkeypatch):
    trainer = ReferenceStorageTrainer(tmp_path)
    allocated = []
    original_empty = torch.empty

    def record_empty(*args, **kwargs):
        if kwargs.get("dtype") is torch.float32:
            allocated.append(args)
        return original_empty(*args, **kwargs)

    scores = mapped_scores(tmp_path, reference_dataset(), reference_rows())
    monkeypatch.setattr(torch, "empty", record_empty)
    dataset = reference_dataset()
    attached = trainer._attach_scored_reference_logps(
        dataset, "train", scores, identity=trainer._reference_split_identity(dataset, "train", SETTINGS)
    )
    owner = trainer._reference_storage_by_split["train"]
    arrow = attached.data.column(REF_PER_TOKEN_LOGPS_COLUMN).chunk(0)
    assert arrow.values.buffers()[1].address == owner.values.data_ptr()
    payload = trainer._reference_checkpoint_payload()["train"]
    assert payload["values"].data_ptr() == owner.values.data_ptr()
    assert not allocated, "attachment or save repacked the whole completion-token table"
    assert not list((tmp_path / REFERENCE_CACHE_DIR_NAME).iterdir()), "mapped scratch files were retained"
    owner.values[0] = -9.25
    assert attached[REF_PER_TOKEN_LOGPS_COLUMN][0][0] == -9.25
    trainer.save_checkpoint()
    saved = torch.load(tmp_path / "checkpoint-1" / "reference_logps.pt", weights_only=True)
    assert saved["train"]["values"][0] == -9.25


@pytest.mark.parametrize("completions", [[], [[], []], [[], [1, 2]]])
def test_empty_reference_buffers_skip_file_mapping_and_preserve_ragged_rows(tmp_path, monkeypatch, completions):
    dataset = Dataset.from_dict({"completion_input_ids": completions})
    cache = ReferenceScoreCache(tmp_path, dp_size=1)
    cache.append_rows(0, [torch.full((len(row),), -0.5, dtype=torch.float32) for row in completions])
    mapped_files = []
    original_from_file = torch.from_file

    def map_nonempty(filename, **kwargs):
        assert os.path.getsize(filename) > 0, "an empty reference buffer reached mmap"
        assert kwargs["size"] > 0
        mapped_files.append(os.path.basename(filename))
        return original_from_file(filename, **kwargs)

    monkeypatch.setattr(torch, "from_file", map_nonempty)
    mapped = cache.finish(dataset)
    lengths = [len(row) for row in completions]
    assert mapped.lengths.dtype == torch.int64
    assert mapped.values.dtype == torch.float32
    assert mapped.lengths.tolist() == lengths
    assert mapped.offsets.tolist() == [0, *torch.tensor(lengths, dtype=torch.int64).cumsum(0).tolist()]
    assert mapped.column().to_pylist() == [[-0.5] * length for length in lengths]
    expected_files = (["merged.lengths"] if lengths else []) + (["merged.values"] if any(lengths) else [])
    assert mapped_files == expected_files * 2
    assert not list((tmp_path / REFERENCE_CACHE_DIR_NAME).iterdir())


@pytest.mark.parametrize(
    "damage", ["missing_rows", "missing_lengths", "missing_values", "wrong_lengths", "nan", "truncated"]
)
def test_incomplete_or_corrupt_cache_is_rejected_and_removed(tmp_path, damage):
    dataset = Dataset.from_dict({"completion_input_ids": [[1, 2], [3]]})
    cache = ReferenceScoreCache(tmp_path, dp_size=1)
    if damage == "missing_rows":
        cache.append_rows(0, [torch.tensor([-1.0, -2.0])])
    elif damage == "wrong_lengths":
        cache.append_rows(0, [torch.tensor([-1.0]), torch.tensor([-2.0, -3.0])])
    else:
        cache.append_rows(0, [torch.tensor([-1.0, -2.0]), torch.tensor([-3.0])])
        if damage.startswith("missing_"):
            os.unlink(cache._path(0, damage.removeprefix("missing_")))
        else:
            with open(cache._path(0, "values"), "ab" if damage == "truncated" else "r+b") as output:
                output.write(b"!" if damage == "truncated" else torch.tensor([float("nan")]).numpy().tobytes())
    with pytest.raises(ValueError, match="Incomplete|truncated"):
        cache.finish(dataset)
    assert not list((tmp_path / REFERENCE_CACHE_DIR_NAME).iterdir())


def test_buffered_validation_detects_corruption_after_the_first_chunk(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_module, "REFERENCE_BUFFER_VALUES", 2)
    dataset = Dataset.from_dict({"completion_input_ids": [[1, 2, 3, 4]]})
    cache = ReferenceScoreCache(tmp_path, dp_size=1)
    cache.append_rows(0, [torch.tensor([-1.0, -2.0, -3.0, -4.0])])
    with open(cache._path(0, "values"), "r+b") as output:
        output.seek(3 * 4)
        output.write(torch.tensor([float("nan")]).numpy().tobytes())
    with pytest.raises(ValueError, match="non-finite"):
        cache.finish(dataset)
    assert not list(tmp_path.rglob("*.values")), "failed validation left an unreachable token cache"


def test_failed_cache_write_is_cleaned_up_before_checkpointing(tmp_path, monkeypatch):
    trainer = ReferenceStorageTrainer(tmp_path)

    def fail_write(*args):
        raise OSError("reference cache disk full")

    monkeypatch.setattr(ReferenceScoreCache, "_append", fail_write)
    with pytest.raises(ValueError, match="reference cache disk full"):
        attach_reference(trainer, reference_dataset(), "train", reference_rows(), settings=SETTINGS)
    assert not list((tmp_path / REFERENCE_CACHE_DIR_NAME).iterdir())
    assert not trainer._reference_logps_by_split


def test_resume_reads_mmap_and_reuses_the_mapped_checkpoint_storage(tmp_path, monkeypatch):
    first = ReferenceStorageTrainer(tmp_path)
    attach_reference(first, reference_dataset(), "train", reference_rows(), settings=SETTINGS)
    first.save_checkpoint()
    resumed = ReferenceStorageTrainer(tmp_path, checkpoint=str(tmp_path / "checkpoint-1"))
    calls = []
    original_load = torch.load

    def record_load(*args, **kwargs):
        calls.append(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(torch, "load", record_load)
    attached = restore_reference(resumed, reference_dataset(), "train")
    assert calls == [{"map_location": "cpu", "weights_only": True, "mmap": True}]
    owner = resumed._reference_storage_by_split["train"]
    assert (
        attached.data.column(REF_PER_TOKEN_LOGPS_COLUMN).chunk(0).values.buffers()[1].address
        == owner.values.data_ptr()
    )
    assert resumed._reference_checkpoint_payload()["train"]["values"].data_ptr() == owner.values.data_ptr()


def _node_local_cache(rank, root, damage):
    os.environ["LOCAL_RANK"] = "0"
    os.environ["LOCAL_WORLD_SIZE"] = "1"
    output = os.path.join(root, f"node-{rank}")
    dataset = Dataset.from_dict({"prompt_input_ids": [[7], [8]], "completion_input_ids": [[1, 2, 3], [4, 5]]})
    expected = [[-0.25, -0.5, -0.75], [-1.25, -1.5]]
    patch = pytest.MonkeyPatch()
    if damage == "missing_writer":
        patch.setattr(cache_module, "reference_cache_writers", lambda: (0,))
    cache = ReferenceScoreCache(output, dp_size=2)
    if damage == "finite_corruption" and rank == 1:
        original_append = cache._append

        def corrupt(shard, kind, values):
            if shard == 0 and kind == "values":
                values = values.clone()
                values[0] -= 0.125
            original_append(shard, kind, values)

        patch.setattr(cache, "_append", corrupt)
    try:
        cache.collect_batch([torch.tensor(expected[rank])], {0: 0, 1: 1})
        mapped = cache.finish(dataset)
        trainer = ReferenceStorageTrainer(output)
        attached = trainer._attach_scored_reference_logps(
            dataset, "train", mapped, identity=trainer._reference_split_identity(dataset, "train", SETTINGS)
        )
        assert attached[REF_PER_TOKEN_LOGPS_COLUMN] == expected
        assert not os.listdir(os.path.join(output, REFERENCE_CACHE_DIR_NAME))
        trainer.save_checkpoint()
        restored = ReferenceStorageTrainer(output, checkpoint=os.path.join(output, "checkpoint-1"))
        assert restore_reference(restored, dataset, "train")[REF_PER_TOKEN_LOGPS_COLUMN] == expected
        outcome = "NO RAISE"
    except Exception as exc:
        outcome = f"{type(exc).__name__}: {exc}"
    finally:
        patch.undo()
        cache.discard()
    with open(os.path.join(root, f"node-outcome-{rank}.txt"), "w") as result:
        result.write(outcome)


@pytest.mark.parametrize("damage", ["none", "missing_writer", "finite_corruption"])
def test_node_local_cache_is_complete_replicated_and_removed(tmp_path, damage):
    run_gloo_ranks(
        _node_local_cache,
        2,
        str(tmp_path),
        damage,
        env={"DIST_OUTPUT_SHARED_FILESYSTEM": "0"},
        pg_timeout=datetime.timedelta(seconds=20),
    )
    for rank in range(2):
        outcome = (tmp_path / f"node-outcome-{rank}.txt").read_text()
        if damage == "none":
            assert outcome == "NO RAISE"
        else:
            assert (
                "Incomplete offline GRPO reference cache" in outcome
                if damage == "missing_writer"
                else ("reference values differs across ranks" in outcome)
            )
        assert "timed out" not in outcome


def _bounded_batch(rank, root, failure, dp_size, writers, buffer_values):
    os.environ["LOCAL_RANK"] = "0"
    os.environ["LOCAL_WORLD_SIZE"] = "1"
    patch = pytest.MonkeyPatch()
    patch.setattr(cache_module, "REFERENCE_BUFFER_VALUES", buffer_values)
    cache = ReferenceScoreCache(os.path.join(root, f"writer-{rank}"), dp_size=dp_size)
    cache.writers = writers
    rejects = []
    metadata = []
    original_gather = dist.all_gather_object
    original_metadata_gather = dist.all_gather_into_tensor

    def record_reject(output, value):
        rejects.append(value)
        return original_gather(output, value)

    def record_metadata(output, value):
        assert value.dtype == torch.int64 and value.shape == (2,)
        assert output.dtype == torch.int64 and output.shape == (4,)
        metadata.append(value.cpu().tolist())
        return original_metadata_gather(output, value)

    def unexpected_broadcast(*args, **kwargs):
        pytest.fail("batch metadata used per-shard broadcasts")

    patch.setattr(dist, "all_gather_object", record_reject)
    patch.setattr(dist, "all_gather_into_tensor", record_metadata)
    patch.setattr(dist, "broadcast", unexpected_broadcast)
    if failure == "write" and rank == writers[-1]:

        def fail_append(*args):
            raise OSError("node-local disk full")

        patch.setattr(cache, "_append", fail_append)
    rows = [torch.arange(7, dtype=torch.float32).neg() - rank] if rank < dp_size else None
    if failure == "pack" and rank == dp_size - 1:
        rows[0][0] = float("nan")
    try:
        cache.collect_batch(rows, {shard: shard for shard in range(dp_size)})
        outcome = "NO RAISE"
    except Exception as exc:
        outcome = f"{type(exc).__name__}: {exc}"
    finally:
        patch.undo()
        cache.discard()
    assert len(rejects) == 1, "failure gathers grew with chunks, DP shards or filesystem writers"
    assert metadata == [[0, 0] if rows is None or (failure == "pack" and rank == dp_size - 1) else [1, 7]], (
        "metadata gathers grew with chunks, DP shards or filesystem writers"
    )
    with open(os.path.join(root, f"bounded-outcome-{rank}.txt"), "w") as result:
        result.write(outcome)


@pytest.mark.parametrize("failure", ["none", "pack", "write"])
@pytest.mark.parametrize("dp_size,writers,buffer_values", [(1, (0,), 32), (2, (0, 1), 2)])
def test_bounded_transfers_have_one_metadata_gather_and_failure_join(
    tmp_path, failure, dp_size, writers, buffer_values
):
    run_gloo_ranks(
        _bounded_batch,
        2,
        str(tmp_path),
        failure,
        dp_size,
        writers,
        buffer_values,
        env={"DIST_OUTPUT_SHARED_FILESYSTEM": "0"},
        pg_timeout=datetime.timedelta(seconds=20),
    )
    for rank in range(2):
        outcome = (tmp_path / f"bounded-outcome-{rank}.txt").read_text()
        assert (
            outcome == "NO RAISE"
            if failure == "none"
            else ("non-finite" in outcome if failure == "pack" else "node-local disk full" in outcome)
        )
        assert "timed out" not in outcome


def test_cache_length_validation_uses_arrow_not_python_token_rows(tmp_path, monkeypatch):
    dataset = Dataset.from_dict({"completion_input_ids": [[1, 2], [], [3]]})
    cache = ReferenceScoreCache(tmp_path, dp_size=1)
    cache.append_rows(0, [torch.tensor([-1.0, -2.0]), torch.empty(0), torch.tensor([-3.0])])
    original_iter = Dataset.iter

    def arrow_only(self, *args, **kwargs):
        assert self.format["type"] == "arrow", "length validation materialized Python token rows"
        return original_iter(self, *args, **kwargs)

    monkeypatch.setattr(Dataset, "iter", arrow_only)
    mapped = cache.finish(dataset)
    assert mapped.lengths.tolist() == [2, 0, 1]
    assert mapped.values.tolist() == [-1.0, -2.0, -3.0]


def test_nfs_live_mapping_remnants_are_not_hub_upload_candidates(tmp_path, monkeypatch):
    cache = ReferenceScoreCache(tmp_path, dp_size=1)
    remnant = os.path.join(cache.directory, ".nfs-live")
    with open(remnant, "wb") as output:
        output.write(b"live mmap inode")
    cache.discard()
    assert os.path.isfile(remnant)
    assert os.path.relpath(cache.directory, tmp_path).startswith("_")
    _validate_output_dir(str(tmp_path))
    with open(os.path.join(cache.directory, "unexpected"), "wb") as output:
        output.write(b"unexpected")
    original_rmdir = os.rmdir

    def refuse(directory):
        if os.fspath(directory) == cache.directory:
            raise OSError(errno.EACCES, "cleanup permission denied")
        return original_rmdir(directory)

    monkeypatch.setattr(os, "rmdir", refuse)
    with pytest.raises(OSError, match="permission denied"):
        cache.discard()
    monkeypatch.undo()
    os.unlink(remnant)
    cache.discard()


def test_secondary_cleanup_failure_preserves_the_original_mapping_failure(tmp_path, monkeypatch):
    cache = ReferenceScoreCache(tmp_path, dp_size=1)
    original = RuntimeError("original reference mapping failed")

    def fail_mapping(dataset):
        raise original

    def fail_cleanup():
        raise OSError("cache cleanup permission denied")

    monkeypatch.setattr(cache, "_finish", fail_mapping)
    monkeypatch.setattr(cache, "discard", fail_cleanup)
    with pytest.raises(RuntimeError, match="original reference mapping failed") as caught:
        cache.finish(Dataset.from_dict({"completion_input_ids": [[1]]}))
    assert caught.value is original
    assert caught.value.__notes__ == ["Reference scratch cleanup also failed: cache cleanup permission denied"]


def test_transport_failure_is_not_deferred_to_a_world_failure_join(tmp_path, monkeypatch):
    cache = ReferenceScoreCache(tmp_path, dp_size=1)
    cache.writers = (1,)
    monkeypatch.setattr(cache_module, "get_global_world_size", lambda: 2)

    def gather_metadata(output, value):
        output[:2].copy_(value)
        output[2:].zero_()

    monkeypatch.setattr(dist, "all_gather_into_tensor", gather_metadata)
    original = OSError("original reference transport failed")

    def fail_send(*args, **kwargs):
        raise original

    def unexpected_reject(self):
        pytest.fail("collective transport error was swallowed before a world failure join")

    monkeypatch.setattr(dist, "send", fail_send)
    monkeypatch.setattr(cache_module.DeferredRankFailure, "reject", unexpected_reject)
    with pytest.raises(OSError, match="original reference transport failed") as caught:
        cache.collect_batch([torch.tensor([-1.0])], {0: 0})
    assert caught.value is original


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
