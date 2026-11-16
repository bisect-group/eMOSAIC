#!/usr/bin/env bash
# eMOSAIC-T Ki paper runs: bf16 (--amp). train also evaluates. Paper used 3 runs per GPU on an 80 GB GPU.
# Usage: emulator_bench/run.sh [--dry-run] [cache|train|aggregate|all]
# Env: PY GPUS SEEDS RUNS_PER_GPU DATA_ROOT OUT_ROOT EMULATOR_ROOT DRY_RUN=1
# shellcheck disable=SC2086  # PY, GPUS, SEEDS and the task lists split into words on purpose
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
EMULATOR_ROOT=${EMULATOR_ROOT:-$(realpath ../..)}
DATA_ROOT=${DATA_ROOT:-$EMULATOR_ROOT/data/processed/datasets}
OUT_ROOT=${OUT_ROOT:-$EMULATOR_ROOT/outputs/baselines/eMOSAIC}
PY=${PY:-python}
GPUS=${GPUS:-0}
RUNS_PER_GPU=${RUNS_PER_GPU:-1}
SEEDS=${SEEDS:-"666 777 888"}
STEP=all
for a in "$@"; do
  case $a in
    --dry-run) DRY_RUN=1 ;;
    cache|train|aggregate|all) STEP=$a ;;
    *) echo "usage: $0 [--dry-run] [cache|train|aggregate|all]" >&2; exit 2 ;;
  esac
done
run() { echo "+ $*"; [[ -n ${DRY_RUN:-} ]] || "$@"; }
want() { [[ $STEP == all || $STEP == "$1" ]]; }

KI=$DATA_ROOT/kinetic_dataset/ki

want cache && run $PY emulator_bench/cache_embeddings.py --base_dir "$KI" \
  --embeddings_dir "$OUT_ROOT/embeddings" --gpus $GPUS --max_seq_len 700 --chunk_size 32
want train && run $PY emulator_bench/run_split_benchmarks.py --skip_cache --amp --seeds $SEEDS \
  --base_dir "$KI" --embeddings_dir "$OUT_ROOT/embeddings" --output_dir "$OUT_ROOT/ki" \
  --gpus $GPUS --runs_per_gpu "$RUNS_PER_GPU"
want aggregate && run $PY emulator_bench/aggregate_results.py --results_dir "$OUT_ROOT/ki"
exit 0
