#!/usr/bin/env bash
# One protocol measurement of Halo SFT (25 steps, bench_sft.py wrapper) in the Halo image, under the GPU lock.
# Usage: run.sh <config.yaml> <out.json> [extra halo --args...]
#   HALO_TREE    Halo checkout to run (paths.env); mounted at /workspace.
#   IMAGE        Halo image (public.ecr.aws/whitecircle/halo:blackwell); GPUs attach through $GPU_FLAGS (paths.env).
#   NPROC (2), MASTER_PORT (29641), RUN_TIMEOUT (1800 s, counted after the lock is acquired).
# The config is rendered with render_config. The log goes to <out.json%.json>.log; out paths live under $BENCH_ROOT.
# (--ulimit stack is omitted: runc cannot raise it past the host's hard stack limit.)
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../../paths.env"
set -euo pipefail
[ $# -ge 2 ] || { echo "usage: $0 <config.yaml> <out.json> [extra args]"; exit 1; }
CFG=$(render_config "$(readlink -f "$1")"); OUT=$(readlink -m "$2"); shift 2
IMAGE=${IMAGE:-public.ecr.aws/whitecircle/halo:blackwell}
LOG="${OUT%.json}.log"; mkdir -p "$(dirname "$OUT")"
NAME="halo-bench-$$"
DC=$BENCH_ROOT/halo/docker_cache; mkdir -p "$DC"
trap 'docker rm -f "$NAME" >/dev/null 2>&1 || true' EXIT
echo "[run.sh] tree=$HALO_TREE ($(git -C "$HALO_TREE" rev-parse --short HEAD 2>/dev/null || echo ?)) image=$IMAGE cfg=$CFG out=$OUT log=$LOG"
docker image inspect --format '[run.sh] image digest: {{index .RepoDigests 0}}' "$IMAGE" | tee "${OUT%.json}.image.txt"
flock "$GPU_LOCK" timeout "${RUN_TIMEOUT:-1800}" \
  docker run --rm --name "$NAME" $GPU_FLAGS \
    --ipc=host --ulimit memlock=-1 --shm-size=128g --network host \
    "${BENCH_DOCKER_MOUNTS[@]}" -v "$HALO_TREE":/workspace -w /workspace \
    -e HALO_TREE=/workspace -e PYTHONPATH=/workspace -e BENCH_OUT="$OUT" \
    -e HALO_DATA_ROOT="$BENCH_ROOT/halo/data_root" \
    -e TRITON_CACHE_DIR="$DC/triton" -e EP_JIT_CACHE_DIR="$DC/ep_jit" \
    -e WANDB_DISABLED=true -e WANDB_MODE=disabled -e HALO_FLEX_SLIDING \
    "$IMAGE" python -m torch.distributed.run --nproc_per_node="${NPROC:-2}" --master_port="${MASTER_PORT:-29641}" \
      "$BUNDLE/gemma4_sft/halo/bench_sft.py" "$CFG" "$@" > "$LOG" 2>&1 \
  || { echo "[run.sh] FAILED, see $LOG"; tail -30 "$LOG"; exit 1; }
grep '\[bench\] RESULT' "$LOG" || true
