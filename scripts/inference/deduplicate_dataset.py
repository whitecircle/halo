#!/usr/bin/env python
"""Semantic deduplication of a large dataset via sentence embeddings + FAISS similarity search.

Embeds ``--text_field``, then drops every row whose cosine similarity to an earlier row exceeds
``--similarity_threshold``.

Usage:
    python scripts/inference/deduplicate_dataset.py \\
        --dataset dataset.jsonl \\
        --output_path deduplicated_dataset.jsonl \\
        --text_field "text" \\
        --model_id "sentence-transformers/all-MiniLM-L6-v2" \\
        --similarity_threshold 0.95 \\
        --batch_size 32
"""

import argparse
import logging
import sys
from pathlib import Path

from halo.data.deduplication import OUTPUT_WRITERS, run_deduplication
from halo.data.sources.paths import DATA_FILE_BUILDERS
from halo.log import configure_cli_logging
from scripts.cli_arguments import add_hub_source_args, add_trust_remote_code_arg


def _add_dataset_args(parser: argparse.ArgumentParser) -> None:
    """The input dataset, the field deduplicated on, the kept fields and the output file."""
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help=f"Local data file ({', '.join(DATA_FILE_BUILDERS)}) or HuggingFace dataset id 'org/name'",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path of the deduplicated dataset file (its log lands beside it).",
    )
    parser.add_argument(
        "--output_format",
        type=str,
        choices=list(OUTPUT_WRITERS),
        default="jsonl",
        help="Output format for the deduplicated dataset (default: %(default)s).",
    )
    parser.add_argument(
        "--text_field", type=str, required=True, help="Name of the field containing text to deduplicate on"
    )
    parser.add_argument(
        "--additional_fields",
        type=str,
        nargs="+",
        default=None,
        help="Keep only these fields in the output (whitelist; if not specified, all fields are kept). "
        "nargs=+ so a bare --additional_fields is rejected rather than whitelisting nothing and "
        "stripping every column.",
    )
    parser.add_argument("--config", type=str, default=None, help="Hub dataset config name (default: none).")
    parser.add_argument("--split", type=str, default="train", help="Hub dataset split (default: %(default)s).")


def _add_embedding_args(parser: argparse.ArgumentParser) -> None:
    """The sentence-embedding model, its tokenization length, batch and device."""
    # No --revision: the embedding load threads none.
    add_hub_source_args(
        parser, source="The sentence-embedding model", default="sentence-transformers/all-MiniLM-L6-v2", revision=False
    )
    add_trust_remote_code_arg(parser, default=False)
    parser.add_argument(
        "--model_max_length",
        type=int,
        default=512,
        help="Maximum sequence length for tokenization (default: %(default)s).",
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for embedding computation (default: %(default)s)."
    )
    parser.add_argument(
        "--device", type=str, default="auto", help="Device to compute on: auto, cpu or cuda (default: %(default)s)."
    )


def _add_search_args(parser: argparse.ArgumentParser) -> None:
    """The similarity threshold and the batched FAISS search passes."""
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0.95,
        help="Cosine similarity above which two rows are duplicates (default: %(default)s).",
    )
    parser.add_argument(
        "--dedup_batch_size", type=int, default=100000, help="Rows per similarity-search batch (default: %(default)s)."
    )
    parser.add_argument(
        "--dedup_steps",
        type=int,
        default=3,
        help="Number of deduplication passes (default: %(default)s).",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Worker threads for the similarity search (default: the CPU count).",
    )


def _add_run_args(parser: argparse.ArgumentParser) -> None:
    """Embedding reuse, the dry run, the row sample and verbosity."""
    parser.add_argument("--save_embeddings", action="store_true", help="Save computed embeddings to disk for reuse")
    parser.add_argument("--load_embeddings", type=str, default=None, help="Path to precomputed embeddings file (.npy)")
    parser.add_argument(
        "--embeddings_output_path",
        type=str,
        default=None,
        help="Where to write the computed embeddings; only used together with --save_embeddings "
        "(default: <output_path>.embeddings.npy)",
    )
    parser.add_argument("--dry_run", action="store_true", help="Perform a dry run without saving the output")
    parser.add_argument("--sample_size", type=int, default=None, help="Process only the first N rows (default: all).")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Deduplicate large datasets using semantic embeddings")
    _add_dataset_args(parser)
    _add_embedding_args(parser)
    _add_search_args(parser)
    _add_run_args(parser)
    return parser.parse_args()


def main() -> int:
    """Run the deduplication pipeline, returning the process exit code."""
    args = parse_arguments()
    configure_cli_logging(verbose=args.verbose)

    # Attached once the output path is known, so the log lands next to the dataset it describes
    # rather than in whatever directory the script was launched from.
    log_path = Path(args.output_path).with_suffix(".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.getLogger().addHandler(logging.FileHandler(log_path))
    return run_deduplication(args)


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
