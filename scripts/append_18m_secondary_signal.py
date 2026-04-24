"""Append the 18-month horizon recession-probability signal to
``data/reports/executive_report.txt`` as a "secondary signal" section.

Reads the 18M artifacts produced by
    python3 -m scheduler.update_job --horizon 18 \
        --variant-output-dir data/models/horizon_18m/

and writes a short section documenting the 18M probability, threshold,
and decision alongside the primary 6M signal. Idempotent: if the
section already exists, it is replaced.

Intended to be run immediately after the 18M update_job call in the
weekly scheduler. See A3 ledger row for context.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
EXECUTIVE_REPORT = ROOT / "data" / "reports" / "executive_report.txt"
HORIZON_18M_DIR = ROOT / "data" / "models" / "horizon_18m"
MANIFEST_PATH = HORIZON_18M_DIR / "run_manifest.json"
PREDICTIONS_PATH = HORIZON_18M_DIR / "predictions.csv"

SECTION_HEADER = "SECONDARY SIGNAL — 18 MONTHS FORWARD (A3 horizon)"
SECTION_SENTINEL = "SECONDARY SIGNAL — 18 MONTHS FORWARD"
TRAILING_BANNER = "=" * 80


def _load_latest_18m():
    """Return (latest_date, prob_ensemble, threshold, active_models)."""
    if not MANIFEST_PATH.exists() or not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            "18M artifacts missing. Run:\n"
            "  python3 -m scheduler.update_job --horizon 18 "
            "--variant-output-dir data/models/horizon_18m/"
        )

    manifest = json.loads(MANIFEST_PATH.read_text())
    predictions = pd.read_csv(PREDICTIONS_PATH)
    latest = predictions.iloc[-1]

    return {
        "latest_date": latest["Date"],
        "prob_ensemble": float(latest["Prob_Ensemble"]),
        "threshold": float(manifest["decision_threshold_used"]),
        "active_models": manifest["active_models"],
        "ensemble_weights": manifest["ensemble_weights"],
        "git_sha": manifest["git_sha"][:8],
        "horizon_months": manifest["horizon_months"],
        "cv_auc": manifest["ensemble_metrics"]["auc"],
        "cv_pr_auc": manifest["ensemble_metrics"]["pr_auc"],
    }


def _format_section(data: dict) -> str:
    prob = data["prob_ensemble"]
    threshold = data["threshold"]
    crossed = prob >= threshold
    signal = "ELEVATED (crosses 18M threshold)" if crossed else "CALM (below 18M threshold)"
    gap = prob - threshold

    weights_line = ", ".join(
        f"{m}={w:.2f}" for m, w in data["ensemble_weights"].items()
    )

    lines = [
        "",
        TRAILING_BANNER,
        SECTION_HEADER,
        TRAILING_BANNER,
        f"As of: {data['latest_date']}",
        f"Ensemble probability (18M forward): {prob * 100:.1f}%",
        f"Decision threshold (18M):          {threshold * 100:.1f}%",
        f"Gap vs threshold:                  {gap * 100:+.1f}pp",
        f"Signal:                            {signal}",
        "",
        f"Active models: {', '.join(data['active_models'])}",
        f"Weights:       {weights_line}",
        f"Training CV:   AUC {data['cv_auc']:.3f}, PR-AUC {data['cv_pr_auc']:.3f}",
        "",
        "The 18M horizon is a secondary signal added post-A3. It asks:",
        "\"Is a recession likely to start within the next 18 months?\"",
        "A3 established (historical holdout) that 18M forward-window",
        "predictions surface stronger peaks for Dot-com / S&L / GFC",
        "than 6M does — useful for early-stage monitoring. The 6M",
        "signal above remains the primary, more time-specific forecast.",
        "",
        f"Model artifacts: data/models/horizon_18m/ (git SHA {data['git_sha']})",
        TRAILING_BANNER,
    ]
    return "\n".join(lines) + "\n"


def _replace_or_append(report_text: str, new_section: str) -> str:
    """If an 18M section already exists, replace it; else append at end."""
    start_marker = SECTION_SENTINEL
    if start_marker not in report_text:
        # Append to end, ensuring trailing newline from existing content
        if not report_text.endswith("\n"):
            report_text += "\n"
        return report_text + new_section

    # Replace existing section up to the next standalone "=" banner after it.
    start_idx = report_text.find(start_marker)
    # Back up to include the opening banner line
    banner_before = report_text.rfind(TRAILING_BANNER, 0, start_idx)
    if banner_before == -1:
        banner_before = start_idx
    # Find end: the trailing banner that closes the section
    closing_idx = report_text.find(TRAILING_BANNER, start_idx)
    if closing_idx == -1:
        return report_text[:banner_before] + new_section
    end_of_section = closing_idx + len(TRAILING_BANNER)
    # Consume trailing newline
    if end_of_section < len(report_text) and report_text[end_of_section] == "\n":
        end_of_section += 1
    return report_text[:banner_before] + new_section + report_text[end_of_section:]


def main() -> int:
    data = _load_latest_18m()
    section = _format_section(data)

    if not EXECUTIVE_REPORT.exists():
        print(f"ERROR: {EXECUTIVE_REPORT} does not exist", file=sys.stderr)
        return 1

    current = EXECUTIVE_REPORT.read_text()
    updated = _replace_or_append(current, section)
    EXECUTIVE_REPORT.write_text(updated)

    print(f"✓ Updated {EXECUTIVE_REPORT}")
    print(f"  18M probability: {data['prob_ensemble']*100:.1f}% (threshold: {data['threshold']*100:.1f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
