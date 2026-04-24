"""
AUDIT-1 patch demonstration — threshold-plateau-width diagnostics.

NOT APPLIED TO PRODUCTION. This script demonstrates what Fix H1 (elevate
Lead_Months_Fixed + emit threshold plateau width) would look like if it were
wired into the production training path. It reads the current
threshold_sweep.csv and run_manifest.json and emits the plateau-width
metric + flags the run if the plateau is flat.

Usage:
    python3 scripts/audit_patches/threshold_plateau_diagnostics.py \
        --models-dir data/models \
        --plateau-f1-tol 0.005

Behavior:
    - Loads threshold_sweep.csv (post-C5 only).
    - Computes plateau width: number of thresholds whose F1 is within
      `--plateau-f1-tol` of the winner's F1, and the range of those thresholds.
    - Prints a summary with a WARNING if plateau width is "flat" (>=5 thresholds
      within tol, or range spans >=0.10 threshold units).
    - Would emit `threshold_plateau` block into run_manifest.json when wired
      into the training pipeline — DOES NOT WRITE in this demo script.

Rationale (from AUDIT-1): the drift between baseline_efb307e (thr=0.41) and
baseline_0411da9 (thr=0.30) was caused by F1 tie-break flipping across a
10-wide plateau (0.30–0.41) where every threshold was within 0.01 F1 of the
best. Emitting plateau width makes this failure mode visible in the manifest.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


def analyze_plateau(sweep: pd.DataFrame, plateau_tol: float = 0.005) -> dict:
    """Return a dict describing the plateau width around the winning threshold."""
    # Winning threshold (project tie-break policy: f1, precision, recall,
    # specificity, -threshold)
    winner = sweep.sort_values(
        ["f1", "precision", "recall", "specificity"],
        ascending=[False, False, False, False],
    ).iloc[0]
    best_f1 = float(winner["f1"])
    winner_thr = float(winner["threshold"])

    plateau_mask = sweep["f1"] >= best_f1 - plateau_tol
    plateau = sweep[plateau_mask]

    plateau_width_count = int(plateau_mask.sum())
    plateau_thr_min = float(plateau["threshold"].min())
    plateau_thr_max = float(plateau["threshold"].max())
    plateau_thr_range = round(plateau_thr_max - plateau_thr_min, 4)

    # Also compute 0.01-F1 band (more lenient tolerance) for reporting
    lenient_mask = sweep["f1"] >= best_f1 - 0.01
    lenient_count = int(lenient_mask.sum())

    is_flat = plateau_width_count >= 5 or plateau_thr_range >= 0.10
    return {
        "winner_threshold": winner_thr,
        "best_f1": round(best_f1, 6),
        "plateau_f1_tolerance": plateau_tol,
        "plateau_width_count": plateau_width_count,
        "plateau_threshold_min": plateau_thr_min,
        "plateau_threshold_max": plateau_thr_max,
        "plateau_threshold_range": plateau_thr_range,
        "lenient_001_f1_count": lenient_count,
        "is_flat_plateau_warning": is_flat,
        "flat_plateau_criteria": "plateau_width_count >= 5 OR threshold_range >= 0.10",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Threshold-plateau diagnostics demo")
    parser.add_argument(
        "--models-dir",
        type=str,
        default="data/models",
        help="Directory containing threshold_sweep.csv",
    )
    parser.add_argument(
        "--plateau-f1-tol",
        type=float,
        default=0.005,
        help="F1 tolerance defining plateau membership (default 0.005)",
    )
    args = parser.parse_args()

    models_dir = Path(args.models_dir)
    sweep_path = models_dir / "threshold_sweep.csv"
    if not sweep_path.exists():
        print(f"ERROR: {sweep_path} not found (C5 post-training artifact required).",
              file=sys.stderr)
        return 2

    sweep = pd.read_csv(sweep_path)
    if "f1" not in sweep.columns:
        print(f"ERROR: {sweep_path} missing 'f1' column.", file=sys.stderr)
        return 2

    result = analyze_plateau(sweep, plateau_tol=args.plateau_f1_tol)
    print(json.dumps(result, indent=2))

    # Warning behavior demo
    if result["is_flat_plateau_warning"]:
        print(
            "\nWARNING: threshold plateau is FLAT — the F1 optimum is unstable to "
            "tiny input perturbations. Challenger vs baseline comparisons should use "
            "a fixed reference threshold (see C5 Lead_Months_Fixed). "
            "Recommended reference: 0.41 (from data/models/baseline_efb307e).",
            file=sys.stderr,
        )
        return 1

    print("\nPlateau is NARROW — decision threshold is stable.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
