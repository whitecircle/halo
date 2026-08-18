# Data

Datasets load from the HuggingFace Hub, local paths, or S3 (optional — bring your own bucket), with offline pre-processing, packing, and sharding for large-scale training.

---

<!-- markdownlint-disable MD030 -- mkdocs-material grid cards require the 4-space content indent -->

<div class="grid cards" markdown>

-   :material-table:{ .lg .middle } **Dataset formats**

    ---

    Required columns and conversation structure for every method — SFT, DPO/SMPO, GRPO, classification, reward, distillation.

    [:octicons-arrow-right-24: Dataset formats](dataset-formats.md)

-   :material-cog-transfer:{ .lg .middle } **SFT dataset pre-processing**

    ---

    Offline tokenize, pack, and Megatron-style shard — skip tokenization at training time.

    [:octicons-arrow-right-24: SFT pre-processing](dataset-preparation.md)

-   :material-format-align-left:{ .lg .middle } **Collators**

    ---

    Padding, packing, and padding-free modes, with completion-only masking and an auto-selecting factory.

    [:octicons-arrow-right-24: Collators](collators.md)

-   :material-folder-network:{ .lg .middle } **Filesystem handling**

    ---

    Shared vs. local filesystem coordination for multi-node — the read/write `DIST_*_SHARED_FILESYSTEM` knobs and `fs_aware_main_first()`.

    [:octicons-arrow-right-24: Filesystem handling](filesystem-handling.md)

-   :material-cloud-upload:{ .lg .middle } **S3 utilities**

    ---

    Upload, download, and `s3://` URIs for datasets and arbitrary folders.

    [:octicons-arrow-right-24: S3 utilities](s3-utilities.md)

</div>
