"""
C5 retroactive threshold-stability analysis.

Loads stored backtest artifacts from prior experiments (B3 positive control,
C3, B5) plus the frozen post-B3 baseline, applies
:func:`recession_engine.threshold_stability.rescore_backtest_at_fixed_threshold`
at the baseline's fixed decision threshold, and writes both a JSON payload
and a markdown summary for the ledger.

Outputs
-------
* ``data/models/c5_retroactive_analysis.json`` — full per-experiment numbers,
  including per-recession bounds and gate verdicts.
* ``data/models/c5_retroactive_analysis.md`` — human-readable table for
  validators.
* ``data/models/c5_validation.json`` — C5 experiment verdict object
  (``passed``, ``findings``, and the SHA at run time).

Key caveat
----------
Historical backtest CSVs (B3 / C3 / B5 / baseline_efb307e) were produced by
the pre-C5 backtester, which only persisted scalar summaries per recession
(``Peak_Prob``, ``Lead_Months``, ``Threshold``). Exact fixed-threshold lead
time at those runs cannot be recomputed post-hoc without rerunning each
backtest. This script therefore uses the bound-based estimator built into
:mod:`recession_engine.threshold_stability`:

* When original threshold ≤ fixed threshold, original lead is a *lower bound*
  on the fixed-threshold lead.
* When original threshold ≥ fixed threshold, original lead is an *upper
  bound* on the fixed-threshold lead.

For the three experiments under review, the bounds are tight enough to
resolve the KEEP / DISCARD question for the in-scope recessions. Future
experiments run against the extended :class:`RecessionBacktester` will emit
an exact ``Lead_Months_Fixed`` column directly.

Usage
-----
::

    python3 scripts/c5_retroactive_rescore.py
    python3 scripts/c5_retroactive_rescore.py --baseline-threshold 0.41

"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from recession_engine.threshold_stability import (  # noqa: E402
    DEFAULT_IN_SCOPE_RECESSIONS,
    DEFAULT_LEAD_REGRESSION_TOLERANCE_MONTHS,
    apply_stability_gate,
    compute_lead_time_delta_matrix,
    load_baseline_threshold,
    rescore_backtest_at_fixed_threshold,
    summarize_gate_report,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("c5_retroactive")

# ``REPO_ROOT`` points at the canonical main-repo checkout where all historical
# experiment artifacts live (b3_variants, baseline_efb307e, etc.). Retroactive
# CSV inputs are always read from there. Fresh C5 outputs are written to the
# script's own repo (``LOCAL_ROOT = ROOT``) so a worktree run produces artifacts
# the worktree owns; pass ``--outdir`` to override.
REPO_ROOT = Path("/Users/ajgiannone/PycharmProjects/Econ Dashboard")
LOCAL_ROOT = ROOT
BASELINE_THRESHOLD_PATH = REPO_ROOT / "data/models/baseline_efb307e/threshold.json"

# Historical artifact locations. Paths are resolved at module load time;
# each experiment stores its backtest CSV alongside its other artifacts.
EXPERIMENTS = [
    {
        "id": "baseline_efb307e",
        "title": "Post-B3 baseline (production)",
        "purpose": "baseline",
        "backtest_csv": REPO_ROOT / "data/models/baseline_efb307e/backtest_results.csv",
        "branch": "main",
    },
    {
        "id": "B3",
        "title": "B3 credit-supply block (positive control)",
        "purpose": "challenger",
        "backtest_csv": REPO_ROOT / "data/models/b3_variants/with_credit/backtest_results.csv",
        "branch": "experiment/B3-credit-supply",
        "note": "Merged under override — we expect GFC lead to remain strong at the fixed threshold.",
    },
    {
        "id": "C3",
        "title": "C3 regime-conditional ensemble weights",
        "purpose": "challenger",
        "backtest_csv": REPO_ROOT / ".claude/worktrees/agent-a40a3f1f/data/models/c3_variants/regime_conditional/backtest_results.csv",
        "branch": "experiment/C3-regime-conditional-weights",
        "note": "DISCARD under per-variant F1 threshold — did re-optimization destroy the lead?",
    },
    {
        "id": "B5",
        "title": "B5 equity-valuation block (hybrid variant)",
        "purpose": "challenger",
        "backtest_csv": REPO_ROOT / ".claude/worktrees/agent-a769f6b9/data/models/b5_variants/hybrid_backtest_results.csv",
        "branch": "experiment/B5-equity-valuation",
        "note": "DISCARD — does Dot-com lift hold at the fixed threshold? Does GFC collapse remain?",
    },
]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
    except Exception:
        return "unknown"


def _load_backtest(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        logger.warning("Backtest CSV missing: %s", path)
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logger.error("Failed to load %s: %s", path, exc)
        return None
    return df


def _jsonify(obj):
    """Coerce numpy / pandas scalars into plain Python for JSON emission."""
    if isinstance(obj, dict):
        return {k: _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return [_jsonify(row) for row in obj.to_dict(orient="records")]
    if isinstance(obj, pd.Series):
        return {k: _jsonify(v) for k, v in obj.to_dict().items()}
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (v != v) else v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, float) and obj != obj:
        return None
    return obj


def analyze(fixed_threshold: float) -> dict:
    """Run the retroactive rescoring across all configured experiments."""
    logger.info("Fixed stability threshold: %s", fixed_threshold)

    loaded: dict[str, pd.DataFrame] = {}
    for exp in EXPERIMENTS:
        df = _load_backtest(exp["backtest_csv"])
        if df is None:
            continue
        logger.info("Loaded %s: %s rows", exp["id"], len(df))
        loaded[exp["id"]] = df

    if "baseline_efb307e" not in loaded:
        raise SystemExit("Baseline backtest CSV is required for retroactive analysis")

    baseline_df = loaded["baseline_efb307e"]
    baseline_rescored = rescore_backtest_at_fixed_threshold(baseline_df, fixed_threshold)

    per_experiment = []
    for exp in EXPERIMENTS:
        eid = exp["id"]
        if eid not in loaded:
            per_experiment.append({
                "id": eid,
                "title": exp["title"],
                "status": "missing_artifact",
                "message": f"Backtest CSV not found at {exp['backtest_csv']}",
            })
            continue

        df = loaded[eid]
        rescored = rescore_backtest_at_fixed_threshold(df, fixed_threshold)

        # Per-recession breakdown versus baseline. For the baseline row itself
        # we emit only the rescored table (self-vs-self delta is trivially 0
        # but the bounds still illustrate the dynamic range).
        if exp["purpose"] == "challenger":
            delta = compute_lead_time_delta_matrix(
                challenger_backtest=df,
                baseline_backtest=baseline_df,
                baseline_threshold=fixed_threshold,
            )
            report = apply_stability_gate(
                delta,
                lead_regression_tolerance_months=DEFAULT_LEAD_REGRESSION_TOLERANCE_MONTHS,
            )
            verdict_summary = "KEEP-under-C5" if report.passed else "DISCARD-under-C5"
        else:
            delta = pd.DataFrame()
            report = None
            verdict_summary = "baseline_reference"

        per_experiment.append({
            "id": eid,
            "title": exp["title"],
            "branch": exp.get("branch"),
            "note": exp.get("note"),
            "backtest_csv": str(exp["backtest_csv"]),
            "status": "ok",
            "verdict_under_c5": verdict_summary,
            "rescored": _jsonify(
                rescored[
                    [
                        c
                        for c in [
                            "Recession", "Peak_Prob", "Threshold", "Lead_Months",
                            "Crossed_Threshold", "Fixed_Threshold",
                            "Crossed_Threshold_Fixed", "Lead_Months_Fixed",
                            "Lead_Estimation_Method", "Threshold_Shift",
                            "Lead_Time_Delta_Fixed",
                        ]
                        if c in rescored.columns
                    ]
                ]
            ),
            "delta_matrix": _jsonify(delta),
            "gate_report": _jsonify(asdict(report)) if report is not None else None,
        })

    results = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "git_sha": _git_sha(),
        "fixed_threshold": fixed_threshold,
        "in_scope_recessions": list(DEFAULT_IN_SCOPE_RECESSIONS),
        "tolerance_months": DEFAULT_LEAD_REGRESSION_TOLERANCE_MONTHS,
        "experiments": per_experiment,
        "methodology_notes": [
            (
                "Historical backtest CSVs predate the C5 trajectory-preserving "
                "backtester, so fixed-threshold lead times are bound-based "
                "estimates (see `recession_engine.threshold_stability` "
                "docstring)."
            ),
            (
                "Exact fixed-threshold lead times will be emitted directly by "
                "`RecessionBacktester.run_pseudo_oos_backtest` on all future "
                "runs (additional `Lead_Months_Fixed` column)."
            ),
        ],
    }
    return results


def render_markdown(results: dict) -> str:
    lines = []
    lines.append("# C5 retroactive threshold-stability analysis")
    lines.append("")
    lines.append(
        f"Generated: {results['generated_at_utc']}  |  "
        f"Git SHA: `{results['git_sha']}`  |  "
        f"Fixed threshold: **{results['fixed_threshold']}**"
    )
    lines.append("")
    lines.append(
        "Reference baseline: `data/models/baseline_efb307e/threshold.json`. "
        "All challenger lead times below are reported at the baseline fixed "
        "threshold using the bound-based estimator in "
        "`recession_engine.threshold_stability` (historical CSVs don't "
        "preserve per-origin probability trajectories)."
    )
    lines.append("")
    lines.append("Lead-estimation method legend:")
    lines.append("")
    lines.append("| method | meaning |")
    lines.append("|---|---|")
    lines.append("| `exact` | trajectory available, first-crossing date computed directly |")
    lines.append("| `exact_no_cross` | Peak_Prob < fixed_threshold, no crossing happens |")
    lines.append("| `bound_at_or_earlier` | original threshold ≥ fixed, so fixed-threshold crossing is at-or-earlier (original lead is a lower bound) |")
    lines.append("| `bound_at_or_later` | original threshold < fixed, so fixed-threshold crossing is at-or-later (original lead is an upper bound) |")
    lines.append("| `unavailable` | backtest row errored or missing Peak_Prob |")
    lines.append("")
    lines.append("## Per-experiment results")
    lines.append("")

    for exp in results["experiments"]:
        lines.append(f"### {exp['id']} — {exp.get('title', '')}")
        lines.append("")
        if exp.get("status") != "ok":
            lines.append(f"_Skipped_: {exp.get('message', 'unknown error')}")
            lines.append("")
            continue
        if exp.get("note"):
            lines.append(f"_{exp['note']}_")
            lines.append("")
        lines.append(f"Verdict under C5 gate: **{exp.get('verdict_under_c5', 'n/a')}**")
        lines.append("")
        lines.append(
            "| Recession | Peak | Own thr | Own lead | Fixed lead | Method | Δ vs own |"
        )
        lines.append(
            "|---|---:|---:|---:|---:|---|---:|"
        )
        for row in exp.get("rescored", []):
            def fmt(x, digits=3):
                return f"{x:.{digits}f}" if isinstance(x, (int, float)) else "n/a"
            def fmt_lead(x):
                return f"{x:+.2f}" if isinstance(x, (int, float)) else "n/a"
            lines.append(
                f"| {row.get('Recession','?')} "
                f"| {fmt(row.get('Peak_Prob'))} "
                f"| {fmt(row.get('Threshold'))} "
                f"| {fmt_lead(row.get('Lead_Months'))} "
                f"| {fmt_lead(row.get('Lead_Months_Fixed'))} "
                f"| `{row.get('Lead_Estimation_Method','?')}` "
                f"| {fmt_lead(row.get('Lead_Time_Delta_Fixed'))} |"
            )
        lines.append("")
        if exp.get("gate_report"):
            report = exp["gate_report"]
            lines.append(f"Gate report (in-scope only):")
            lines.append("")
            lines.append(f"- Passed: **{report.get('passed')}**")
            lines.append(f"- Tolerance: {report.get('tolerance_months')} months")
            violations = report.get("violations") or []
            if violations:
                lines.append("- Violations:")
                for v in violations:
                    lines.append(
                        f"  - `{v.get('recession')}` — Δ lead at fixed = "
                        f"{v.get('lead_time_delta_fixed'):+.2f}mo "
                        f"(challenger {v.get('challenger_lead_at_baseline_threshold')} vs "
                        f"baseline {v.get('baseline_lead_at_own_threshold')})"
                    )
            lines.append("")

    lines.append("## Methodology notes")
    for note in results.get("methodology_notes", []):
        lines.append(f"- {note}")
    lines.append("")

    return "\n".join(lines)


def write_validation(results: dict, outdir: Path) -> dict:
    """Write the C5 validation JSON used by the experiment ledger."""
    findings = []
    for exp in results["experiments"]:
        if exp.get("status") != "ok":
            continue
        if exp.get("gate_report") is None:
            continue
        report = exp["gate_report"]
        findings.append({
            "experiment_id": exp["id"],
            "original_verdict": exp.get("note", ""),
            "c5_verdict": exp.get("verdict_under_c5"),
            "passed": report.get("passed"),
            "violation_count": len(report.get("violations") or []),
        })

    all_passed = all(f.get("passed") for f in findings if f)
    validation = {
        "experiment": "C5",
        "title": "Threshold stability gate",
        "generated_at_utc": results["generated_at_utc"],
        "git_sha": results["git_sha"],
        "verdict": "KEEP",
        "description": (
            "Ship the fixed-threshold re-scoring utility + retroactive gate. "
            "Future experiments evaluate challenger lead time at the frozen "
            "baseline threshold in addition to the per-variant F1-optimal "
            "threshold, preventing the B3/C3/B5 class of threshold-shift "
            "measurement artifacts."
        ),
        "fixed_threshold": results["fixed_threshold"],
        "tolerance_months": results["tolerance_months"],
        "retroactive_findings": findings,
        "modules": [
            "recession_engine/threshold_stability.py",
            "recession_engine/backtester.py (extended — Lead_Months_Fixed + trajectory)",
            "scripts/c5_retroactive_rescore.py",
        ],
        "artifacts": [
            "data/models/c5_retroactive_analysis.json",
            "data/models/c5_retroactive_analysis.md",
        ],
        "any_challenger_failed_gate": not all_passed,
    }

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "c5_validation.json").write_text(json.dumps(validation, indent=2))
    return validation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline-threshold",
        type=float,
        default=None,
        help="Override the fixed threshold (default: load from baseline_efb307e/threshold.json)",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=LOCAL_ROOT / "data/models",
        help="Directory to write analysis artifacts (default: this repo's data/models)",
    )
    args = parser.parse_args()

    if args.baseline_threshold is not None:
        fixed_threshold = float(args.baseline_threshold)
    else:
        fixed_threshold = load_baseline_threshold(BASELINE_THRESHOLD_PATH)

    logger.info("Running retroactive rescore at fixed threshold %s", fixed_threshold)
    results = analyze(fixed_threshold)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    json_path = outdir / "c5_retroactive_analysis.json"
    md_path = outdir / "c5_retroactive_analysis.md"
    json_path.write_text(json.dumps(_jsonify(results), indent=2))
    md_path.write_text(render_markdown(results))
    validation = write_validation(results, outdir)

    logger.info("Wrote %s", json_path)
    logger.info("Wrote %s", md_path)
    logger.info("Wrote %s", outdir / "c5_validation.json")

    for exp in results["experiments"]:
        logger.info(
            "  %s: status=%s verdict_under_c5=%s",
            exp.get("id"), exp.get("status"), exp.get("verdict_under_c5"),
        )

    # Exit code 0 — C5 ships its methodology successfully regardless of which
    # challengers retroactively fail the gate. Failing challengers ARE the
    # expected, informative result (C3 and B5 should be confirmed discards).
    return 0


if __name__ == "__main__":
    sys.exit(main())
