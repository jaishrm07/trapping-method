"""Stage 1 — compute RFD between two saved probe trajectories.

Each `results.json` produced by run_baseline_probe.py / future immunization
runs contains an `epoch_accs` list. This script loads two such files and
prints their pairwise RFD.

Usage:
    python experiments/run_rfd.py \
        --baseline results/baseline_probe_resnet18_cars/results.json \
        --immunized results/<immunized_run_name>/results.json

For Stage 1 sanity-checking before any immunization run exists, you can pass
the baseline twice and confirm RFD=0:

    python experiments/run_rfd.py \
        --baseline results/baseline_probe_resnet18_cars/results.json \
        --immunized results/baseline_probe_resnet18_cars/results.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import relative_fine_tuning_deviation


def load_trajectory(path: str | Path) -> list[float]:
    with open(path, "r") as f:
        data = json.load(f)
    if "epoch_accs" not in data:
        raise KeyError(f"{path} missing 'epoch_accs'")
    return [float(v) for v in data["epoch_accs"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, help="Path to baseline run's results.json")
    parser.add_argument("--immunized", required=True, help="Path to immunized run's results.json")
    args = parser.parse_args()

    base = load_trajectory(args.baseline)
    imm = load_trajectory(args.immunized)
    rfd = relative_fine_tuning_deviation(base, imm)

    print(f"baseline epochs: {len(base)} | final acc: {base[-1] * 100:.2f}%")
    print(f"immunized epochs: {len(imm)} | final acc: {imm[-1] * 100:.2f}%")
    print(f"\nRFD = {rfd:.2f}%")


if __name__ == "__main__":
    main()
