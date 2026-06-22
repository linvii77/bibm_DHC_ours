#!/bin/bash
# Wait for fold1 training to finish, then test + evaluate (single fold only).
set -e
cd "$(dirname "$0")"
source env.sh

echo "Waiting for fold1 training to finish..."
while pgrep -f "train_dhc.py.*Task_amos_5p/dhc/fold1" > /dev/null; do
  sleep 30
done

echo "===== fold1 test ====="
python code/test.py --task amos --exp Task_amos_5p/dhc/fold1 -g 0 --cps AB

echo "===== fold1 evaluate ====="
python code/evaluate_Ntimes_npy.py --task amos --exp Task_amos_5p/dhc --folds 1 --cps AB

echo "===== DONE (fold1 only) ====="
