#!/bin/bash
# Link weights/ into logs/ layout expected by code/test.py
# Covers: Synapse baseline (3 folds), Synapse FusedProxy (fold1), AMOS FusedProxy (fold1)
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "=== Synapse 20% DHC Baseline ==="
for fold in 1 2 3; do
  src="weights/synapse_baseline/fold${fold}/ckpts/best_model.pth"
  dst="logs/Task_synapse_20p/dhc/fold${fold}/ckpts/best_model.pth"
  [ ! -f "$src" ] && echo "Missing $src — run: git lfs pull" && exit 1
  mkdir -p "$(dirname "$dst")"
  [ ! -f "$dst" ] && ln -sf "${ROOT}/$src" "$dst" && echo "Linked fold${fold}"
done

echo "=== Synapse 20% DHC+FusedProxy (fold1 only) ==="
src="weights/synapse_fused/fold1/ckpts/best_model.pth"
dst="logs/Task_synapse_20p/dhc_fused/fold1/ckpts/best_model.pth"
[ ! -f "$src" ] && echo "Missing $src — run: git lfs pull" && exit 1
mkdir -p "$(dirname "$dst")"
[ ! -f "$dst" ] && ln -sf "${ROOT}/$src" "$dst" && echo "Linked synapse_fused/fold1"

echo "=== AMOS 5% DHC+FusedProxy (fold1 only) ==="
src="weights/amos_fused/fold1/ckpts/best_model.pth"
dst="logs/Task_amos_5p/dhc/fold1/ckpts/best_model.pth"
[ ! -f "$src" ] && echo "Missing $src — run: git lfs pull" && exit 1
mkdir -p "$(dirname "$dst")"
[ ! -f "$dst" ] && ln -sf "${ROOT}/$src" "$dst" && echo "Linked amos_fused/fold1"

echo ""
echo "All weights linked. Example inference commands:"
echo "  python code/test.py --task synapse --exp Task_synapse_20p/dhc/fold1       -g 0 --cps AB"
echo "  python code/test.py --task synapse --exp Task_synapse_20p/dhc_fused/fold1 -g 0 --cps AB"
echo "  python code/test.py --task amos   --exp Task_amos_5p/dhc/fold1            -g 0 --cps AB"
