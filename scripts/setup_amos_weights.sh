#!/bin/bash
# Link weights/amos_5p into logs/ layout expected by code/test.py
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

src="weights/amos_5p/fold1/ckpts/best_model.pth"
dst="logs/Task_amos_5p/dhc/fold1/ckpts/best_model.pth"
if [ ! -f "$src" ]; then
  echo "Missing $src — run: git lfs pull"
  exit 1
fi
mkdir -p "$(dirname "$dst")"
if [ ! -f "$dst" ]; then
  ln -sf "$(pwd)/$src" "$dst"
  echo "Linked $dst -> $src"
fi

echo "Ready for: python code/test.py --task amos --exp Task_amos_5p/dhc/fold1 -g 0 --cps AB"
