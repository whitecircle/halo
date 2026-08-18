# Data

Datasets load from the HuggingFace Hub, local paths, or S3 (optional — bring your own bucket), with offline pre-processing, packing, and sharding for large-scale training.

---

- **[Dataset formats](dataset-formats.md)** — Required columns and conversation structure for every method — SFT, DPO/SMPO, GRPO, classification, reward, distillation.
- **[SFT dataset pre-processing](dataset-preparation.md)** — Offline tokenize, pack, and Megatron-style shard — skip tokenization at training time.
- **[Collators](collators.md)** — Padding, packing, and padding-free modes, with completion-only masking and an auto-selecting factory.
- **[Filesystem handling](filesystem-handling.md)** — Shared vs. local filesystem coordination for multi-node — the read/write `DIST_*_SHARED_FILESYSTEM` knobs and `fs_aware_main_first()`.
- **[S3 utilities](s3-utilities.md)** — Upload, download, and `s3://` URIs for datasets and arbitrary folders.
