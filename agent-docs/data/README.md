# Data

Datasets load from the HuggingFace Hub, local paths, or S3 (optional; bring your own bucket), with offline pre-processing, packing, and sharding for large-scale training.

- [Dataset formats](dataset-formats.md): required columns and conversation structure for SFT, DPO, SMPO, KTO, offline GRPO, classification and reward modeling. Every method's columns are in [Methods at a glance](../training-methods/README.md#methods-at-a-glance).
- [SFT dataset pre-processing](dataset-preparation.md): offline tokenize, pack and Megatron-style shard, so training skips tokenization.
- [Collators](collators.md): padding, packing, and padding-free modes, with completion-only masking and an auto-selecting factory.
- [Filesystem handling](filesystem-handling.md): shared and local filesystem coordination for multi-node runs, through the read and write `DIST_*_SHARED_FILESYSTEM` knobs and `fs_aware_main_first()`.
- [S3 utilities](s3-utilities.md): upload, download, and `s3://` URIs for datasets and arbitrary folders.
