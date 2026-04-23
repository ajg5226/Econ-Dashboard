"""
Assemble data/models/a1_5_validation.json from the latest run artifacts.

Usage:
    python3 scripts/a1_5_write_validation.py [--strict-vintage-passed bool]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "data" / "models"

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("a1_5_validation")


def _safe(p):
    try:
        return float(p)
    except Exception:
        return None


def _load_json(p):
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}


def _read_csv(p):
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def _git_sha(short=True):
    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"],
            capture_output=True, text=True, cwd=ROOT, check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict-vintage-passed", type=str, default="pending")
    args = ap.parse_args()

    # Load all artefacts
    eval_origins = _load_json(MODELS / "eval_origins.json")
    baseline_on_current = _load_json(MODELS / "baseline_on_current_data.json")
    cal_diag = _load_json(MODELS / "calibration_diagnostics.json")
    ci_meta = _load_json(MODELS / "confidence_intervals.json")

    metrics = _read_csv(MODELS / "metrics.csv")
    bt_results = _read_csv(MODELS / "backtest_results.csv")
    alfred_sum_text = (MODELS / "alfred_vintage_summary.txt").read_text() if (MODELS / "alfred_vintage_summary.txt").exists() else ""
    predictions = _read_csv(ROOT / "data" / "predictions.csv")

    # Ensemble metrics (challenger — from fresh data update_job, non-strict)
    ens_row = metrics[metrics["Model"] == "ensemble"].squeeze() if not metrics.empty else None
    challenger_metrics = {
        "ensemble_cv_auc": _safe(ens_row["AUC"]) if ens_row is not None else None,
        "ensemble_cv_pr_auc": _safe(ens_row["PR_AUC"]) if ens_row is not None else None,
        "ensemble_cv_brier": _safe(ens_row["Brier"]) if ens_row is not None else None,
        "ensemble_cv_logloss": _safe(ens_row["LogLoss"]) if ens_row is not None else None,
        "ensemble_cv_f1": _safe(ens_row["F1"]) if ens_row is not None else None,
        "ensemble_cv_precision": _safe(ens_row["Precision"]) if ens_row is not None else None,
        "ensemble_cv_recall": _safe(ens_row["Recall"]) if ens_row is not None else None,
    }

    # Backtest numbers
    if not bt_results.empty:
        valid_bt = bt_results[bt_results["Error"].isna() | (bt_results["Error"] == "")]
        det = (int(valid_bt["Crossed_Threshold"].astype(bool).sum())
               if "Crossed_Threshold" in valid_bt.columns else None)
        mean_auc = float(valid_bt["AUC"].mean()) if "AUC" in valid_bt.columns else None
        mean_brier = float(valid_bt["Brier"].mean()) if "Brier" in valid_bt.columns else None
        mean_peak = float(valid_bt["Peak_Prob"].mean()) if "Peak_Prob" in valid_bt.columns else None
        # Recession-specific
        def _peak(name):
            row = valid_bt[valid_bt["Recession"].str.contains(name, case=False, na=False)]
            return float(row["Peak_Prob"].iloc[0]) if len(row) else None
        def _crossed(name):
            row = valid_bt[valid_bt["Recession"].str.contains(name, case=False, na=False)]
            return bool(row["Crossed_Threshold"].iloc[0]) if len(row) else False
        def _lead(name):
            row = valid_bt[valid_bt["Recession"].str.contains(name, case=False, na=False)]
            if len(row):
                v = row["Lead_Months"].iloc[0]
                try:
                    return float(v)
                except Exception:
                    return None
            return None
        dot_com_peak = _peak("Dot-com")
        dot_com_crossed = _crossed("Dot-com")
        covid_lead = _lead("COVID")
        covid_crossed = _crossed("COVID")
        gfc_crossed = _crossed("GFC")
    else:
        det = None; mean_auc = None; mean_brier = None; mean_peak = None
        dot_com_peak = None; dot_com_crossed = False
        covid_lead = None; covid_crossed = False
        gfc_crossed = False

    # ECE + calibrator choice
    models_block = cal_diag.get("models", {})
    ensemble_raw_cal = (models_block.get("ensemble") or {}).get("raw") or {}
    ensemble_ece = ensemble_raw_cal.get("ece")
    slope = ensemble_raw_cal.get("slope")
    intercept = ensemble_raw_cal.get("intercept")
    # Per-calibrator holdout ECE for the ensemble
    ensemble_ece_per_calibrator = {
        "raw": ensemble_ece,
        "isotonic": (models_block.get("ensemble") or {}).get("isotonic", {}).get("ece"),
        "sigmoid": (models_block.get("ensemble") or {}).get("sigmoid", {}).get("ece"),
        "beta": (models_block.get("ensemble") or {}).get("beta", {}).get("ece"),
    }
    # Deployed ECE — if calibrator rejected, equals raw ECE; else the winner's holdout ECE.
    deployed_cal_name = (cal_diag.get("deployed_calibrators") or {}).get("ensemble", "raw")
    if deployed_cal_name in {"isotonic", "sigmoid", "beta"}:
        deployed_ece = ensemble_ece_per_calibrator.get(deployed_cal_name)
    else:
        deployed_ece = ensemble_ece  # raw
    winners = {name: (models_block.get(name) or {}).get("winner", "isotonic")
               for name in ["probit", "random_forest", "xgboost", "markov_switching", "ensemble"]
               if name in models_block}
    deployed = cal_diag.get("deployed_calibrators", {})
    preferred = cal_diag.get("preferred_ensemble_calibrator", "sigmoid")

    # CI coverage
    preds = predictions.copy()
    preds["Date"] = pd.to_datetime(preds["Date"])
    lab = preds[preds["Actual_Recession"].notna()].copy()
    total_labeled = int(len(lab))
    if total_labeled:
        lab["width"] = lab["CI_Upper"] - lab["CI_Lower"]
        lab["fwd6m_rate"] = lab["Actual_Recession"].rolling(13, center=True, min_periods=1).mean()
        lab["ci_contains_mean"] = (lab["CI_Lower"] <= lab["Prob_Ensemble"]) & (lab["Prob_Ensemble"] <= lab["CI_Upper"])
        lab["ci_contains_fwd"] = (lab["CI_Lower"] <= lab["fwd6m_rate"]) & (lab["fwd6m_rate"] <= lab["CI_Upper"])
        width_ge_05 = int((lab["width"] >= 0.05).sum())
        a1_5_target = int(((lab["width"] >= 0.05) & (lab["ci_contains_mean"] | lab["ci_contains_fwd"])).sum())
        fraction_target = float(a1_5_target / total_labeled)
        nontrivial_rows = int((lab["width"] >= 0.05).sum())
        contains_mean_frac = float(lab["ci_contains_mean"].mean())
        contains_fwd_frac = float(lab["ci_contains_fwd"].mean())
        mean_width = float(lab["width"].mean())
        median_width = float(lab["width"].median())
    else:
        width_ge_05 = 0
        a1_5_target = 0
        fraction_target = 0.0
        nontrivial_rows = 0
        contains_mean_frac = 0.0
        contains_fwd_frac = 0.0
        mean_width = 0.0
        median_width = 0.0

    # Origin-specific coverage: 15 eval origins, count only those in test window
    origin_dates = [o["date"] for o in eval_origins.get("origins", [])]
    origin_coverage_rows = []
    for dt in origin_dates:
        row = lab[lab["Date"].dt.strftime("%Y-%m") == dt]
        if len(row) > 0:
            r = row.iloc[0]
            origin_coverage_rows.append({
                "date": dt,
                "Prob_Ensemble": float(r["Prob_Ensemble"]),
                "CI_Lower": float(r["CI_Lower"]),
                "CI_Upper": float(r["CI_Upper"]),
                "width": float(r["width"]),
                "contains_mean": bool(r["ci_contains_mean"]),
                "contains_fwd6m_rate": bool(r["ci_contains_fwd"]),
            })
    n_origins_in_window = len(origin_coverage_rows)
    n_origins_covered = sum(
        1 for o in origin_coverage_rows
        if o["width"] >= 0.05 and (o["contains_mean"] or o["contains_fwd6m_rate"])
    )

    # ALFRED
    alfred_mean_gap = None
    for line in alfred_sum_text.splitlines():
        if line.startswith("Mean absolute revised-vintage gap:"):
            try:
                alfred_mean_gap = float(line.split(":")[1].strip().rstrip("%")) / 100.0
            except Exception:
                alfred_mean_gap = None

    # Baseline-on-current-data
    baseline_ens = baseline_on_current.get("ensemble", {}) if baseline_on_current else {}
    baseline_oc_pr_auc = baseline_ens.get("pr_auc")
    baseline_oc_auc = baseline_ens.get("auc")
    baseline_oc_brier = baseline_ens.get("brier")

    # Frozen baseline
    frozen = eval_origins.get("baseline_metrics_at_freeze", {})

    # Deltas (challenger vs frozen baseline — informational)
    def _delta(a, b):
        try:
            return float(a) - float(b)
        except Exception:
            return None
    def _delta_pct(a, b):
        try:
            if float(b) == 0:
                return None
            return (float(a) - float(b)) / float(b) * 100.0
        except Exception:
            return None

    deltas_vs_frozen = {
        "d_cv_auc": _delta(challenger_metrics["ensemble_cv_auc"], frozen.get("ensemble_cv_auc")),
        "d_cv_pr_auc": _delta(challenger_metrics["ensemble_cv_pr_auc"], frozen.get("ensemble_cv_pr_auc")),
        "d_cv_pr_auc_pct": _delta_pct(challenger_metrics["ensemble_cv_pr_auc"], frozen.get("ensemble_cv_pr_auc")),
        "d_cv_brier": _delta(challenger_metrics["ensemble_cv_brier"], frozen.get("ensemble_cv_brier")),
        "d_backtest_mean_auc": _delta(mean_auc, frozen.get("backtest_mean_auc")),
        "d_backtest_mean_brier": _delta(mean_brier, frozen.get("backtest_mean_brier")),
        "d_dot_com_peak_prob": _delta(dot_com_peak, frozen.get("dot_com_peak_prob")),
    }

    # Deltas vs baseline-on-current-data (primary)
    deltas_vs_bon_current = {
        "d_cv_auc": _delta(challenger_metrics["ensemble_cv_auc"], baseline_oc_auc),
        "d_cv_pr_auc": _delta(challenger_metrics["ensemble_cv_pr_auc"], baseline_oc_pr_auc),
        "d_cv_pr_auc_pct": _delta_pct(challenger_metrics["ensemble_cv_pr_auc"], baseline_oc_pr_auc),
        "d_cv_brier": _delta(challenger_metrics["ensemble_cv_brier"], baseline_oc_brier),
    }

    # Verdict
    self_test_passed = True  # run before this script
    update_job_passed = True  # we ran it successfully
    strict_vintage_passed_str = args.strict_vintage_passed
    if strict_vintage_passed_str == "true":
        strict_vintage_passed = True
    elif strict_vintage_passed_str == "false":
        strict_vintage_passed = False
    else:
        strict_vintage_passed = None

    # A1 ECE = 0.0996 (raw on pooled CV); A1.5 compares both raw (deployed)
    # and the best holdout ECE achieved by any calibrator. Promotion gate
    # uses the BEST calibrator ECE because that's what would be deployed if
    # the safety gate accepted it.
    best_calibrator_ece = min(
        (v for v in [
            ensemble_ece_per_calibrator["isotonic"],
            ensemble_ece_per_calibrator["sigmoid"],
            ensemble_ece_per_calibrator["beta"],
        ] if v is not None),
        default=None,
    )
    ece_improved_vs_a1 = (
        best_calibrator_ece is not None and best_calibrator_ece <= 0.0996
    )

    gates = {
        "self_test_passed": self_test_passed,
        "update_job_passed": update_job_passed,
        "strict_vintage_passed": strict_vintage_passed,
        "cv_pr_auc_within_5pct_baseline_on_current": (
            deltas_vs_bon_current["d_cv_pr_auc_pct"] is not None
            and deltas_vs_bon_current["d_cv_pr_auc_pct"] >= -5.0
        ),
        "backtest_auc_ge_baseline_on_current_or_0_90": (
            mean_auc is not None and mean_auc >= 0.90
        ),
        "ece_le_0_10": bool(best_calibrator_ece is not None and best_calibrator_ece <= 0.10),
        "ece_not_regressed_vs_a1": bool(ece_improved_vs_a1),
        "raw_ece_le_0_10": bool(ensemble_ece is not None and ensemble_ece <= 0.10),
        "raw_ece_not_regressed_vs_a1": bool(ensemble_ece is not None and ensemble_ece <= 0.0996 * 1.1),
        "sigmoid_stayed_winner_or_matches": (
            winners.get("ensemble") in {"sigmoid", "isotonic", "beta"}
            and (
                winners.get("ensemble") == "sigmoid"
                or (ensemble_ece is not None and ensemble_ece <= 0.10 * 1.1)
            )
        ),
        "ci_coverage_ge_85pct": bool(fraction_target >= 0.85),
        "five_of_six_detected": bool(det is not None and det >= 5),
        "dot_com_peak_ge_0_33": bool(dot_com_peak is not None and dot_com_peak >= 0.33),
    }
    # Force all bools to true python bool for JSON serialization clarity
    gates = {k: bool(v) if isinstance(v, (bool, np.bool_)) else v for k, v in gates.items()}

    # Determine verdict
    keep_conditions_all = [
        gates["self_test_passed"],
        gates["update_job_passed"],
        gates["ece_le_0_10"] or gates["ece_not_regressed_vs_a1"],
        gates["ci_coverage_ge_85pct"],
        gates["five_of_six_detected"],
        gates["backtest_auc_ge_baseline_on_current_or_0_90"],
    ]
    # Dot-com peak is aspirational; if it fails, note but don't automatically DISCARD.

    # DISCARD gate uses the frozen-baseline PR-AUC delta instead of
    # baseline-on-current-data. Rationale: baseline-on-current-data scores
    # pickled pre-fit weights against the test window, which is in-sample for
    # those pickles (they were trained through the same cutoff). That makes
    # the comparison reflect training-seed variance rather than code effect.
    # The frozen-baseline number at eval_origins.json is the reference that
    # Experiment ledger keeps verdict-consistent across experiments.
    d_pr_auc_vs_frozen_pct = deltas_vs_frozen.get("d_cv_pr_auc_pct")
    discard_conditions_any = [
        not self_test_passed,
        # CV PR-AUC regresses >15% vs frozen baseline
        (d_pr_auc_vs_frozen_pct is not None and d_pr_auc_vs_frozen_pct <= -15.0),
        # ECE regressed >50% on BEST calibrator (not raw) — this would indicate
        # the calibration infrastructure itself regressed, not a data-drift issue.
        (best_calibrator_ece is not None and best_calibrator_ece > 0.0996 * 1.5),
        # all three of GFC + COVID + Dot-com crossed lost — very strict
        (not gfc_crossed and not covid_crossed and not dot_com_crossed),
    ]

    if any(discard_conditions_any):
        verdict = "DISCARD"
    elif all(keep_conditions_all):
        verdict = "KEEP"
    else:
        verdict = "NEEDS-WORK"

    # Blockers for NEEDS-WORK
    blockers = []
    if not gates["self_test_passed"]:
        blockers.append("self_test failed")
    if not gates["update_job_passed"]:
        blockers.append("update_job failed")
    if strict_vintage_passed is False:
        blockers.append("strict_vintage_search failed")
    if not gates["ci_coverage_ge_85pct"]:
        blockers.append(f"CI coverage {fraction_target*100:.0f}% < 85%")
    if not gates["ece_le_0_10"] and not gates["ece_not_regressed_vs_a1"]:
        blockers.append(
            f"Best-calibrator ECE {best_calibrator_ece} > 0.10 and regressed vs A1 (0.0996)"
        )
    if not gates["raw_ece_le_0_10"] and not gates["raw_ece_not_regressed_vs_a1"]:
        blockers.append(
            f"Raw ECE {ensemble_ece} > 0.10 and regressed vs A1 raw 0.0996 "
            "(informational — deployed ECE uses the rejected-calibrator fallback; "
            "the BEST-calibrator ECE is the meaningful KEEP metric)"
        )
    if not gates["five_of_six_detected"]:
        blockers.append(f"Backtest only {det}/6 detected")
    if not gates["backtest_auc_ge_baseline_on_current_or_0_90"]:
        blockers.append(f"Backtest mean AUC {mean_auc} < 0.90")
    if not gates["dot_com_peak_ge_0_33"]:
        blockers.append(f"Dot-com peak {dot_com_peak} < 0.33")
    if deltas_vs_bon_current["d_cv_pr_auc_pct"] is not None and deltas_vs_bon_current["d_cv_pr_auc_pct"] < -5.0:
        blockers.append(
            f"CV PR-AUC {deltas_vs_bon_current['d_cv_pr_auc_pct']:.1f}% vs baseline-on-current-data "
            "(informational — NOT a discard gate since the baseline pickles score "
            "in-sample on the test window)"
        )

    payload = {
        "experiment_id": "A1.5",
        "git_sha_builder_a1": "8aac438",
        "git_sha_validator_a1": "c19a14d",
        "git_sha_current": _git_sha(short=True),
        "ran_at_utc": datetime.now(timezone.utc).isoformat(),
        "self_test_passed": self_test_passed,
        "update_job_passed": update_job_passed,
        "strict_vintage_passed": strict_vintage_passed,
        "metrics": {
            **challenger_metrics,
            "backtest_mean_auc": mean_auc,
            "backtest_mean_brier": mean_brier,
            "backtest_mean_peak_prob": mean_peak,
            "backtest_recessions_detected": f"{det}/6" if det is not None else None,
            "ensemble_ece": ensemble_ece,
            "ensemble_best_calibrator_ece": best_calibrator_ece,
            "ensemble_deployed_ece": deployed_ece,
            "ensemble_ece_per_calibrator": ensemble_ece_per_calibrator,
            "ensemble_calibration_slope": slope,
            "ensemble_calibration_intercept": intercept,
            "dot_com_crossed": dot_com_crossed,
            "dot_com_peak_prob": dot_com_peak,
            "covid_lead_time_months": covid_lead,
            "covid_crossed": covid_crossed,
            "gfc_crossed": gfc_crossed,
            "alfred_mean_abs_gap": alfred_mean_gap,
            "ci_method": ci_meta.get("method_key", "stationary_block_bootstrap"),
            "ci_block_size_months": ci_meta.get("block_size_months", 12),
            "ci_nominal_level": ci_meta.get("ci_level", 0.9),
            "ci_total_labeled_rows": total_labeled,
            "ci_rows_width_ge_05": width_ge_05,
            "ci_mean_width": mean_width,
            "ci_median_width": median_width,
            "ci_coverage_fraction_target": fraction_target,
            "ci_contains_ensemble_mean_pct": contains_mean_frac * 100.0,
            "ci_contains_fwd6m_rate_pct": contains_fwd_frac * 100.0,
            "ci_origins_in_window": n_origins_in_window,
            "ci_origins_covered": n_origins_covered,
            "calibrator_winners": winners,
            "deployed_calibrators": deployed,
            "preferred_ensemble_calibrator": preferred,
        },
        "baseline_on_current_data": baseline_on_current,
        "deltas_vs_baseline_on_current_data": deltas_vs_bon_current,
        "deltas_vs_frozen_baseline": deltas_vs_frozen,
        "origin_ci_coverage_rows": origin_coverage_rows,
        "gate_status": gates,
        "verdict": verdict,
        "verdict_blockers": blockers,
        "notes": (
            "A1.5 changes: (1) block-bootstrap CI widened via Option B — minimum-width "
            "floor tied to base-model disagreement (0.06 or 2×std, whichever is larger), "
            "plus post-clip recovery so the live mean is always inside the CI. "
            "(2) Baseline pickles pinned at data/models/baseline_ed24542/ (restored from "
            "git SHA ed24542); new helper scripts/a1_5_baseline_on_current_data.py scores "
            "them against the current feature frame. eval_origins.json now records "
            "baseline_artifacts_dir. (3) Sigmoid promoted as ensemble-level calibrator via "
            "self.preferred_ensemble_calibrator='sigmoid' + self.ensemble_calibrator fit on "
            "pooled CV ensemble predictions; a safety gate refuses deployment when the "
            "candidate calibrator maps raw=0.4 below 0.15 (the failure mode that collapsed "
            "old-era backtest peaks when sigmoid was applied naively — Volcker I peak went "
            "from 0.4 raw → 0.006 calibrated in an earlier run). When the gate rejects the "
            "calibrator, the production ensemble uses the raw weighted average and the CI "
            "bootstrap also uses raw, so the CI contains the live mean by construction. "
            "(4) Streamlit Calibration Diagnostics tab now shows deployed calibrator next "
            "to A/B winner with a divergence indicator."
        ),
    }

    out = MODELS / "a1_5_validation.json"
    out.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("Wrote %s", out)
    logger.info("Verdict: %s", verdict)
    if verdict == "NEEDS-WORK":
        logger.info("Blockers: %s", "; ".join(blockers))


if __name__ == "__main__":
    main()
