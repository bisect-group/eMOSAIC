#!/usr/bin/env bash
# Checks that the emulator_bench wrapper reproduces the original eMOSAIC code path.
# Usage: verify_equivalence.sh [--gpu | --negative-control]
# Expected env: a conda env with torch, torch_geometric, rdkit, scikit-learn, transformers and pytest
# (EMULaToR name: mldb). This script does not create envs.
set -euo pipefail
cd "$(dirname "$0")/.."
echo "Expected env: mldb (torch, torch_geometric, rdkit, scikit-learn, pytest)"

if [[ ${1:-} == --negative-control ]]; then
    # Multiplies each candidate weight by 1 + size * randn (size 0.20, the smallest error the bf16 test detects):
    # every gpu test must fail. A passing gpu test has a vacuous bound.
    export EQUIV_NEGATIVE_CONTROL=1 EQUIV_NEGATIVE_CONTROL_SIZE=${EQUIV_NEGATIVE_CONTROL_SIZE:-0.20}
    status=0
    out=$(python -m pytest tests -q -rs -p no:cacheprovider --color=no -m gpu -s 2>&1) || status=$?
    echo "$out"
    if [[ $status -ne 1 ]] || grep -Eq "[0-9]+ (passed|skipped|errors?)" <<<"$out"; then
        echo "FAIL: a gpu test did not fail under the negative control"
        exit 1
    fi
    echo "OK: every gpu test failed under the negative control"
    exit 0
fi
markers=(-m "not gpu")
extra=()
if [[ "${1:-}" == "--gpu" ]]; then
    markers=()
    extra=(-s)  # shows the printed gap and floor
    echo "GPU mode: also runs the bf16 tolerance test. It skips when no CUDA device is visible."
else
    export CUDA_VISIBLE_DEVICES=""
fi

out=$(python -m pytest tests -q -rs -p no:cacheprovider --color=no "${markers[@]}" "${extra[@]}" 2>&1) || { echo "$out"; exit 1; }
echo "$out"
if grep -Eq "[0-9]+ skipped" <<<"$out"; then
    if [[ ${#markers[@]} -eq 0 ]] && ! python -c "import torch; raise SystemExit(not torch.cuda.is_available())"; then
        echo "No CUDA device is visible: the GPU tests were skipped."
    else
        echo "FAIL: tests were skipped"
        exit 1
    fi
fi
echo "OK: wrapper matches the original code path"
