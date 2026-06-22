#!/usr/bin/env python3
"""Watch DHC training log and show epoch progress bar + ETA."""
import argparse
import os
import re
import sys
import time
from datetime import datetime, timedelta

EPOCH_RE = re.compile(r"epoch (\d+) : loss : ([-\d.]+)")
BEST_RE = re.compile(r"best eval dice is ([\d.]+) in epoch (\d+)")
MAX_EPOCH_RE = re.compile(r"max_epoch=(\d+)")


def parse_log(path):
    epoch, loss, best_dice, best_epoch, max_epoch = 0, None, None, None, 300
    epoch_times = []
    last_ts = None

    if not os.path.exists(path):
        return epoch, loss, best_dice, best_epoch, max_epoch, epoch_times

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if "max_epoch" in line and max_epoch == 300:
                m = MAX_EPOCH_RE.search(line)
                if m:
                    max_epoch = int(m.group(1))

            m = EPOCH_RE.search(line)
            if m:
                ts_match = re.match(r"\[(\d{2}):(\d{2}):(\d{2})", line)
                if ts_match:
                    h, mi, s = map(int, ts_match.groups())
                    cur_ts = h * 3600 + mi * 60 + s
                    if last_ts is not None and cur_ts >= last_ts:
                        epoch_times.append(cur_ts - last_ts)
                    last_ts = cur_ts
                epoch = int(m.group(1))
                loss = float(m.group(2))

            m = BEST_RE.search(line)
            if m:
                best_dice = float(m.group(1))
                best_epoch = int(m.group(2))

    return epoch, loss, best_dice, best_epoch, max_epoch, epoch_times[-20:]


def bar(pct, width=40):
    filled = int(width * pct)
    return "█" * filled + "░" * (width - filled)


def main():
    parser = argparse.ArgumentParser(description="Watch training progress with progress bar")
    parser.add_argument(
        "--log",
        default="./logs/Task_amos_5p/dhc/fold1/train.log",
        help="path to train.log",
    )
    parser.add_argument("--interval", type=float, default=5.0, help="refresh seconds")
    args = parser.parse_args()
    log_path = os.path.abspath(args.log)

    print(f"Watching: {log_path}")
    print("Press Ctrl+C to exit\n")

    try:
        while True:
            epoch, loss, best_dice, best_epoch, max_epoch, recent = parse_log(log_path)
            # training logs epoch N after finishing epoch N; show N+1 as in-progress when process alive
            done = min(epoch + 1, max_epoch)
            pct = done / max_epoch if max_epoch else 0.0

            if recent:
                avg_sec = sum(recent) / len(recent)
                remain_epochs = max(max_epoch - done, 0)
                eta = timedelta(seconds=int(avg_sec * remain_epochs))
                eta_str = str(eta)
                speed_str = f"{avg_sec:.0f}s/epoch"
            else:
                eta_str = "calculating..."
                speed_str = "-"

            loss_str = f"{loss:.4f}" if loss is not None else "-"
            best_str = (
                f"{best_dice * 100:.2f}% @ ep{best_epoch}"
                if best_dice is not None
                else "-"
            )

            line = (
                f"\r[{bar(pct)}] {pct * 100:5.1f}%  "
                f"epoch {done}/{max_epoch}  loss={loss_str}  best_dice={best_str}  "
                f"{speed_str}  ETA {eta_str}   "
            )
            sys.stdout.write(line)
            sys.stdout.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped watching.")


if __name__ == "__main__":
    main()
