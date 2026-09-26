#!/bin/bash
# Build every dataset the protocol uses, inside the Halo image (no host Python needed):
#   $BENCH_ROOT/data/ultrachat_gemma4_2k[_<ROWS>].jsonl   BENCH_ROWS UltraChat rows (512)  (sha256-checked)
#   $BENCH_ROOT/data/{messages,canonical_tokens}.jsonl   the shared rows of PROTOCOL.txt  (sha256-checked)
#   $BENCH_ROOT/halo/data/ultrachat_gemma4_2k_halo2048/  Halo's packed copy (index and metadata template)
#   $BENCH_ROOT/data/{messages,canonical_tokens}_<SEQ>.jsonl  the same rows packed to BENCH_SEQ tokens (BENCH_SEQ != 2048)
#   $BENCH_ROOT/halo/data/{canonical,alltokens}[_<SEQ>]/  Halo's pre-sharded copy of the canonical rows
# Needs HF_TOKEN (Gemma 4 is gated). Paths and GPU flags: see paths.env. Framework-specific copies are built by
# the framework's own scripts (e.g. axolotl/stage.sh).
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../paths.env"
set -euo pipefail
IMAGE=${IMAGE:-public.ecr.aws/whitecircle/halo:blackwell}
# RUNNER=docker (default) runs each step in the Halo image; RUNNER=direct runs it here (already inside that image).
in_image() {
  if [ "${RUNNER:-docker}" = direct ]; then (cd "$HALO_TREE" && PYTHONPATH="$HALO_TREE" "$@")
  else docker run --rm "${BENCH_DOCKER_MOUNTS[@]}" -w "$HALO_TREE" -e PYTHONPATH="$HALO_TREE" "$IMAGE" "$@"; fi
}
check() {  # check <dir> <recorded .sha256 file>
  (cd "$1" && sha256sum -c "$2") || { echo "sha256 mismatch in $1 against $2" >&2; exit 1; }
}
mkdir -p "$BENCH_ROOT/data" "$BENCH_ROOT/halo/data"

in_image python "$BUNDLE/gemma4_sft/data/build_dataset.py"
check "$BENCH_ROOT" "$BUNDLE/gemma4_sft/data/ultrachat_gemma4_2k$BENCH_ROWS_SUFFIX.jsonl.sha256"

# build_canonical.py writes fresh .sha256 files next to its outputs; the recorded ones are the reference.
in_image python "$BUNDLE/gemma4_sft/data/build_canonical.py"
for f in messages canonical_tokens; do check "$BENCH_ROOT/data" "$BUNDLE/gemma4_sft/data/$f$BENCH_ROWS_SUFFIX.jsonl.sha256"; done

if [ "$BENCH_SEQ" != 2048 ]; then  # BENCH_SEQ-token rows packed from the protocol rows (paths.env)
  in_image python "$BUNDLE/gemma4_sft/data/pack_canonical.py"
  for f in messages canonical_tokens; do check "$BENCH_ROOT/data" "$BUNDLE/gemma4_sft/data/${f}_$BENCH_SEQ.jsonl.sha256"; done
fi

in_image bash "$BUNDLE/gemma4_sft/halo/setup_data.sh"
in_image python "$BUNDLE/gemma4_sft/halo/build_canonical.py"
echo "datasets ready under $BENCH_ROOT"
