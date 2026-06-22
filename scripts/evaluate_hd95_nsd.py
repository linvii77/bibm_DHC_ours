#!/usr/bin/env python3
"""Compute Dice, HD95, NSD, ASD from saved predictions (nii.gz)."""
import argparse
import os
import sys

import numpy as np
from medpy import metric
from surface_distance import compute_surface_distances, compute_surface_dice_at_tolerance
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "code"))
from utils import read_list, read_nifti  # noqa: E402
from utils.config import Config  # noqa: E402

SYNAPSE_CLASSES = [
    "spleen", "R.kidney", "L.kidney", "gallbladder", "esophagus", "liver",
    "stomach", "aorta", "IVC", "PSV", "pancreas", "R.adrenal", "L.adrenal",
]
AMOS_CLASSES = [
    "spleen", "R.kidney", "L.kidney", "gallbladder", "esophagus", "liver",
    "stomach", "aorta", "IVC", "pancreas", "R.adrenal", "L.adrenal",
    "duodenum", "bladder", "prostate/uterus",
]


def cal_metric(pred, gt):
  if pred.sum() > 0 and gt.sum() > 0:
    dice = metric.binary.dc(pred, gt) * 100
    hd95 = metric.binary.hd95(pred, gt)
    asd = metric.binary.asd(pred, gt)
    sf = compute_surface_distances(gt.astype(bool), pred.astype(bool), spacing_mm=(1.0, 1.0, 1.0))
    nsd = compute_surface_dice_at_tolerance(sf, tolerance_mm=1.0) * 100
  elif pred.sum() == 0 and gt.sum() > 0:
    dice, hd95, asd, nsd = 0.0, 128.0, 128.0, 0.0
  elif pred.sum() == 0 and gt.sum() == 0:
    dice, hd95, asd, nsd = 100.0, 0.0, 0.0, 100.0
  else:
    dice, hd95, asd, nsd = 0.0, 128.0, 128.0, 0.0
  return np.array([dice, hd95, nsd, asd])


def evaluate_task(task, exp, folds, cps):
  config = Config(task)
  class_names = SYNAPSE_CLASSES if task == "synapse" else AMOS_CLASSES
  ids_list = read_list("test", task=task)
  test_cls = list(range(1, config.num_cls))
  n_cls = len(test_cls)

  fold_metrics = []
  for fold in range(1, folds + 1):
    values = np.zeros((len(ids_list), n_cls, 4))
    pred_dir = os.path.join(ROOT, "logs", exp, f"fold{fold}", f"predictions_{cps}")
    for idx, data_id in enumerate(tqdm(ids_list, desc=f"{task} fold{fold}")):
      pred = read_nifti(os.path.join(pred_dir, f"{data_id}.nii.gz"))
      label = np.load(os.path.join(config.save_dir, "npy", f"{data_id}_label.npy")).astype(np.int8)
      for i in test_cls:
        values[idx][i - 1] = cal_metric(pred == i, label == i)
    fold_metrics.append(values)

  arr = np.array(fold_metrics)
  per_class = arr.mean(axis=0).mean(axis=0)
  fold_means = arr.mean(axis=1).mean(axis=1)
  overall = per_class.mean(axis=0)

  return {
    "task": task,
    "class_names": class_names,
    "per_class": per_class,
    "fold_means": fold_means,
    "overall": overall,
    "fold_std": fold_means.std(axis=0) if folds > 1 else np.zeros(4),
  }


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--out", default=os.path.join(ROOT, "results", "metrics_hd95_nsd_summary.txt"))
  args = parser.parse_args()

  jobs = [
    ("synapse", "Task_synapse_20p/dhc", 3, "AB"),
    ("amos", "Task_amos_5p/dhc", 1, "AB"),
  ]
  results = [evaluate_task(*j) for j in jobs]

  lines = []
  header = f"{'Organ':<18} {'Dice%':>8} {'HD95':>8} {'NSD%':>8} {'ASD':>8}"
  sep = "-" * len(header)

  for r in results:
    task = r["task"]
    folds = len(r["fold_means"]) if r["task"] == "synapse" else 1
    lines.append(f"\n{'='*60}")
    lines.append(f"{task.upper()} ({'3-fold mean' if folds == 3 else 'fold1'})")
    lines.append(f"{'='*60}")
    lines.append(header)
    lines.append(sep)
    for name, m in zip(r["class_names"], r["per_class"]):
      lines.append(f"{name:<18} {m[0]:8.1f} {m[1]:8.1f} {m[2]:8.1f} {m[3]:8.1f}")
    lines.append(sep)
    o = r["overall"]
    if folds > 1:
      s = r["fold_std"]
      lines.append(
        f"{'Mean ± std':<18} {o[0]:6.2f}±{s[0]:4.2f} {o[1]:6.2f}±{s[1]:4.2f} "
        f"{o[2]:6.2f}±{s[2]:4.2f} {o[3]:6.2f}±{s[3]:4.2f}"
      )
    else:
      lines.append(f"{'Mean':<18} {o[0]:8.2f} {o[1]:8.2f} {o[2]:8.2f} {o[3]:8.2f}")

  text = "\n".join(lines) + "\n"
  print(text)
  os.makedirs(os.path.dirname(args.out), exist_ok=True)
  with open(args.out, "w", encoding="utf-8") as f:
    f.write(text)
  print(f"Saved: {args.out}")


if __name__ == "__main__":
  main()
