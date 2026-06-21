#!/usr/bin/env bash
# Batch generate folding dataset for all garment categories.
# Each category runs in a separate process (PyFlex limitation).
set -euo pipefail

CONDA_ENV="foldnet"
OUT_DIR="${1:-/tmp/so100_fold_dataset}"
NUM_TRAJS="${2:-5}"
VARIANTS="${3:-0 1 2}"
CATEGORIES="${4:-tshirt_sp tshirt trousers vest vest_close shirt shirt_close hooded hooded_close}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}/../.."

mkdir -p "$OUT_DIR"

for cat in $CATEGORIES; do
    for var in $VARIANTS; do
        cloth="foldnet_garments/${cat}_${var}/mesh.obj"
        if [ ! -f "$cloth" ]; then
            echo "SKIP ${cat}_${var}: cloth not found"
            continue
        fi

        echo ""
        echo "========================================================================"
        echo "  ${cat}_${var}  (${NUM_TRAJS} trajs → ${OUT_DIR}/${cat}_${var})"
        echo "========================================================================"

        conda run -n "$CONDA_ENV" bash -c "
            export CUDA_VISIBLE_DEVICES=0
            export PYFLEX_PATH=${PWD}/FoldNet_code/src/pyflex/PyFlex
            export LD_LIBRARY_PATH=${PWD}/FoldNet_code/src/pyflex/libs:/usr/local/cuda/lib64:\$LD_LIBRARY_PATH
            export FOLDNET_BASE_DIR=${PWD}/FoldNet_code
            python isaac_sim/scripts/generate_fold_dataset.py \
                --category $cat --variant $var \
                --num_trajs $NUM_TRAJS --out $OUT_DIR
        " 2>&1 | tail -n +2

        echo "  DONE: ${cat}_${var}"
    done
done

echo ""
echo "========================================================================"
echo "  All done! Dataset in: ${OUT_DIR}"
echo "========================================================================"
du -sh "$OUT_DIR"
