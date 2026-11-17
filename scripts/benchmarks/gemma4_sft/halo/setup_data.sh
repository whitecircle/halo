#!/usr/bin/env bash
# Build Halo's packed copy of the 512 rows (no GPU); build_canonical.py takes its shard index and metadata as the
# template for the canonical pre-sharded dataset. Runs inside the Halo image (prepare_data.sh). Halo's chat-mode
# tokenizer DROPS rows longer
# than max_length (every protocol row renders to >= 2300 tokens), so the rows are rendered with the
# Gemma 4 chat template to text and baked with Halo's own offline preprocessor in text mode: bfd packing
# keeps one document per 2048-token pack and discards the overflow (= truncation to 2048).
# Labels = all tokens (text mode has no completion-only masking).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../paths.env"
set -euo pipefail
HERE=$BUNDLE/gemma4_sft/halo
H=${BENCH_ROOT}/halo
python $HERE/render_chat_text.py ${BENCH_ROOT}/data/ultrachat_gemma4_2k${BENCH_ROWS_SUFFIX}.jsonl $H/data/ultrachat_gemma4_2k_text.jsonl google/gemma-4-26B-A4B-it
cd "$HALO_TREE"
python scripts/before_training/prepare_dataset.py \
  --input $H/data/ultrachat_gemma4_2k_text.jsonl --output $H/data/ultrachat_gemma4_2k_halo2048 \
  --model-name google/gemma-4-26B-A4B-it --mode text --text-field text --no-append-eos \
  --max-length 2048 --pack-sequences --packing-strategy bfd --num-shards 2 --test-size 0.01 --pad-token "<pad>" --overwrite
# -> 506 train rows (253 per DP shard), each exactly 2048 tokens, 6 test rows (unused).
