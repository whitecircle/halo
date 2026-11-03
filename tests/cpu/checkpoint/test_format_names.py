"""The on-disk names in ``halo.checkpoint.format`` are what every saved checkpoint carries, so a changed spelling
orphans them. Adapter and trainer names are spelled here, held equal to the libraries' own."""

import peft.utils
import pytest
import transformers.trainer

from halo.checkpoint import format as checkpoint_format

GOLDEN = {
    "DEFAULT_MAX_SHARD_SIZE": "5GB",
    "SAFETENSORS_INDEX_FILE": "model.safetensors.index.json",
    "SAFETENSORS_METADATA": {"format": "pt"},
    "EP_SHARDED_FORMAT": "ep_sharded",
    "EP_SHARD_KEY_INFIX": ".shard_",
    "SAFETENSORS_SHARD_PATTERN": "model{suffix}.safetensors",
    "SAFETENSORS_WEIGHTS_FILE": "model.safetensors",
    "SAFETENSORS_FAMILY_GLOB": "model*.safetensors",
    "STREAM_PART_PREFIX": "model-streaming",
    "LEGACY_WEIGHTS_FILE": "pytorch_model.bin",
    "TRAINER_STATE_FILE": "trainer_state.json",
    "SCHEDULER_STATE_FILE": "scheduler.pt",
    "OPTIMIZER_STATE_FILES": ("optimizer.pt", "optimizer.bin"),
    "OPTIMIZER_META_FILE": "optimizer_meta.pt",
    "ROUTER_BALANCING_BIASES_FILE": "router_balancing_biases.pt",
    "REFERENCE_LOGPS_FILE": "reference_logps.pt",
    "REFERENCE_CACHE_DIR_NAME": "_reference_cache",
    "INCOMPLETE_CHECKPOINTS_DIR_NAME": "_incomplete_checkpoints",
    "PREFETCH_PENDING_PREFIX": "prefetch_pending",
    "ADAPTER_SAFETENSORS_FILE": "adapter_model.safetensors",
    "ADAPTER_BIN_FILE": "adapter_model.bin",
    "ADAPTER_CONFIG_FILE": "adapter_config.json",
    "MERGED_ADAPTER_CONFIG_DIR": "original_adapter_config",
    "TRAINING_PROVENANCE_FILE": "training_provenance.json",
    "EXPORT_SIDECAR_FILES": ("router_balancing_biases.pt", "training_provenance.json"),
    "PROCESSOR_FILES": ("processor_config.json", "preprocessor_config.json", "video_preprocessor_config.json"),
    "RESUME_ADAPTER_DIR": "resume_adapter",
    "RESUME_ADAPTER_MARKER_FILE": "resume_adapter.json",
    "UNSUFFIXED_RESUME_ENTRIES": ("trainer_state.json", "resume_adapter", "resume_adapter.json"),
    "RESUME_SIDECAR_FILES": ("scheduler.pt", "router_balancing_biases.pt", "reference_logps.pt"),
    "RESUME_SIDECAR_PREFIXES": ("rng_state", "prefetch_pending"),
    "WEIGHT_DUMP_DIRS": ("original", "consolidated", "metal"),
}


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_an_on_disk_name_keeps_its_spelling(name):
    assert getattr(checkpoint_format, name) == GOLDEN[name]


def test_the_hand_spelled_names_match_the_libraries_that_write_them():
    assert checkpoint_format.ADAPTER_CONFIG_FILE == peft.utils.CONFIG_NAME
    assert checkpoint_format.ADAPTER_SAFETENSORS_FILE == peft.utils.SAFETENSORS_WEIGHTS_NAME
    assert checkpoint_format.ADAPTER_BIN_FILE == peft.utils.WEIGHTS_NAME
    assert checkpoint_format.TRAINER_STATE_FILE == transformers.trainer.TRAINER_STATE_NAME
    assert checkpoint_format.SCHEDULER_STATE_FILE == transformers.trainer.SCHEDULER_NAME


def test_the_name_builders_spell_their_files():
    assert checkpoint_format.ep_shard_filename(0, 4) == "model-00000-of-00004.safetensors"
    assert checkpoint_format.hf_shard_filename(1, 1) == "model.safetensors"
    assert checkpoint_format.hf_shard_filename(2, 3) == "model-00002-of-00003.safetensors"
    assert checkpoint_format.prefetch_pending_filename(1, 8) == "prefetch_pending-00001-of-00008.pt"
    stream_part = checkpoint_format.stream_part_filename(checkpoint_format.STREAM_PART_PREFIX, 2)
    assert stream_part == "model-streaming-00002.safetensors"


@pytest.mark.parametrize(
    "name",
    [
        checkpoint_format.optimizer_shard_filename(3),
        checkpoint_format.prefetch_pending_filename(1, 8),
        "rng_state_3.pth",
        "rng_state.pth",
    ],
)
def test_a_ranks_own_resume_file_is_per_rank(name):
    assert checkpoint_format.is_per_rank_resume_file(name)


@pytest.mark.parametrize(
    "name",
    [
        checkpoint_format.OPTIMIZER_META_FILE,
        *checkpoint_format.OPTIMIZER_STATE_FILES,
        checkpoint_format.SCHEDULER_STATE_FILE,
        checkpoint_format.TRAINER_STATE_FILE,
        checkpoint_format.REFERENCE_LOGPS_FILE,
        checkpoint_format.ROUTER_BALANCING_BIASES_FILE,
        checkpoint_format.ep_shard_filename(0, 2),
    ],
)
def test_a_file_of_the_whole_save_is_not_per_rank(name):
    """None is a rank's resume state."""
    assert not checkpoint_format.is_per_rank_resume_file(name)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
