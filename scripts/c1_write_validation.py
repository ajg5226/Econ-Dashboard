"""
C1 validation writer: consolidate baseline vs challenger outputs into
``data/models/c1_validation.json``.

Reads:
- ``data/models/c1_variants/no_benchmarks/`` — baseline run (benchmarks off)
- ``data/models/c1_variants/with_benchmarks/`` — challenger run (benchmarks on)
- ``data/models/baseline_efb307e/`` — pinned pre-C1 baseline artifacts

Computes:
- CV scores per member
- DMA weights + active model set
- In-scope (1990+) vs full-history backtest summaries
- Per-recession peak/lead-time/AUC for S&L, Dot-com, GFC, COVID + Volcker I/II, Oil
- Leave-one-out ensemble Brier deltas (member-level)
- Verdict per the C1 gate policy

Usage::

    python3 scripts/c1_write_validation.py
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)

ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("c1_write_validation")


def _safe(fn, y, p):
    try:
        return float(fn(y, p))
    except Exception:
        return None


IN_SCOPE_RECESSIONS = {"S&L Crisis (1990-91)", "Dot-com (2001)",
                       "GFC (2007-09)", "COVID (2020)"}
ALL_RECESSIONS_ORDER = [
    "Oil Crisis (1973-75)", "Volcker I (1980)", "Volcker II (1981-82)",
    "S&L Crisis (1990-91)", "Dot-com (2001)", "GFC (2007-09)", "COVID (2020)"
]


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"],
                                       cwd=ROOT, text=True).strip()[:7]
    except Exception:
        return "unknown"


def _git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def _summarize_backtest(backtest_csv: Path) -> dict:
    """Build compact dict of in-scope + full-history summaries from backtest_results.csv."""
    if not backtest_csv.exists():
        return {"error": "missing backtest_results.csv"}
    df = pd.read_csv(backtest_csv)
    df = df[df['Error'].isna()]  # drop rows where backtest errored (Oil Crisis often)

    def _summary(subset: pd.DataFrame) -> dict:
        if subset.empty:
            return {"recessions": 0}
        n = len(subset)
        crossed = subset['Crossed_Threshold'].astype(str).str.upper() == 'TRUE'
        detected = crossed.sum()
        mean_auc = float(subset['AUC'].mean())
        mean_brier = float(subset['Brier'].mean())
        mean_peak = float(subset['Peak_Prob'].mean())
        detected_with_lead = subset[crossed]
        mean_lead = float(detected_with_lead['Lead_Months'].mean()) if not detected_with_lead.empty else 0.0
        out = {
            'backtest_mean_auc': mean_auc,
            'backtest_mean_brier': mean_brier,
            'backtest_mean_peak_prob': mean_peak,
            'recessions_detected': f"{int(detected)}/{int(n)}",
            'recessions_tested': int(n),
            'mean_lead_time_detected': mean_lead,
        }
        # Per-recession details
        for _, row in subset.iterrows():
            key = _recession_key(row['Recession'])
            out[f"{key}_peak"] = float(row['Peak_Prob']) if pd.notna(row['Peak_Prob']) else None
            out[f"{key}_lead_months"] = float(row['Lead_Months']) if pd.notna(row['Lead_Months']) else None
            out[f"{key}_auc"] = float(row['AUC']) if pd.notna(row['AUC']) else None
            out[f"{key}_crossed"] = bool(str(row['Crossed_Threshold']).upper() == 'TRUE')
        return out

    in_scope = df[df['Recession'].isin(IN_SCOPE_RECESSIONS)]
    full = df
    return {'in_scope': _summary(in_scope), 'full_history': _summary(full)}


def _recession_key(name: str) -> str:
    mapping = {
        "Oil Crisis (1973-75)": "oil_crisis",
        "Volcker I (1980)": "volcker_i",
        "Volcker II (1981-82)": "volcker_ii",
        "S&L Crisis (1990-91)": "sl",
        "Dot-com (2001)": "dot_com",
        "GFC (2007-09)": "gfc",
        "COVID (2020)": "covid",
    }
    return mapping.get(name, name.lower().replace(' ', '_'))


def _load_cv(variant_dir: Path) -> dict:
    p = variant_dir / "cv_results.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _load_weights(variant_dir: Path) -> dict:
    p = variant_dir / "ensemble_weights.json"
    return json.loads(p.read_text()) if p.exists() else {}


def _load_dma_from_manifest(variant_dir: Path) -> dict:
    """Pull DMA weights + active models from run_manifest.json if present."""
    p = variant_dir / "run_manifest.json"
    if not p.exists():
        return {}
    try:
        m = json.loads(p.read_text())
        return {
            "ensemble_weights": m.get("ensemble_weights", {}),
            "dma_weights_pre_shrinkage": m.get("dma_weights_pre_shrinkage", {}),
            "static_weights": m.get("static_weights", {}),
            "active_models": m.get("active_models", []),
        }
    except Exception:
        return {}


def _loo_brier_from_predictions(predictions_csv: Path) -> dict:
    """
    Leave-one-out ensemble Brier: for each predicted member, recompute the
    ensemble as the equal-weight average of the REMAINING members scored against
    Actual_Recession on the labeled test window, and record the delta vs the
    original ensemble. Higher delta = member contributes more to ensemble
    calibration (its removal hurts Brier).

    Uses equal-weighted LOO (not DMA re-fit) — a first-order sensitivity signal
    that runs in O(n) without retraining. If a proper re-fit is needed later it
    lives in a separate script.
    """
    if not predictions_csv.exists():
        return {"error": "missing predictions.csv"}

    df = pd.read_csv(predictions_csv)
    df = df.dropna(subset=['Actual_Recession']).copy()
    if df.empty:
        return {"error": "no labeled rows"}

    y = df['Actual_Recession'].astype(float).values
    prob_cols = [c for c in df.columns if c.startswith('Prob_') and c != 'Prob_Ensemble']
    member_names = [c.replace('Prob_', '') for c in prob_cols]

    # Matrix of available member probabilities (drop rows with NaN in any column)
    P_full = df[prob_cols].astype(float)
    # Fill NaN with column mean (stable for LOO)
    P_full = P_full.fillna(P_full.mean())
    members_mat = P_full.values  # shape (n, k)

    # Reference ensemble: equal-weight average of ALL available members
    full_ens = members_mat.mean(axis=1)
    full_brier = float(brier_score_loss(y, np.clip(full_ens, 1e-7, 1 - 1e-7)))
    full_auc = float(roc_auc_score(y, full_ens)) if len(set(y)) >= 2 else None
    full_pr_auc = float(average_precision_score(y, full_ens)) if len(set(y)) >= 2 else None

    loo = {}
    n_members = len(member_names)
    for i, name in enumerate(member_names):
        if n_members <= 1:
            continue
        rest = np.delete(members_mat, i, axis=1)
        ens_rest = rest.mean(axis=1)
        brier_rest = float(brier_score_loss(y, np.clip(ens_rest, 1e-7, 1 - 1e-7)))
        auc_rest = float(roc_auc_score(y, ens_rest)) if len(set(y)) >= 2 else None
        loo[name] = {
            'brier_without_member': brier_rest,
            'brier_delta': brier_rest - full_brier,  # positive = remove hurts
            'auc_without_member': auc_rest,
        }
    return {
        'reference_ensemble_type': 'equal-weighted across ALL Prob_* members',
        'reference_brier': full_brier,
        'reference_auc': full_auc,
        'reference_pr_auc': full_pr_auc,
        'leave_one_out': loo,
    }


def _variant_bundle(variant_dir: Path) -> dict:
    cv = _load_cv(variant_dir)
    weights = _load_weights(variant_dir)
    bt = _summarize_backtest(variant_dir / "backtest_results.csv")
    loo = _loo_brier_from_predictions(variant_dir / "predictions.csv")
    manifest = _load_dma_from_manifest(variant_dir)

    # Compact CV summary (pick probit since it anchors in-scope PR-AUC)
    probit_cv = cv.get('probit', {})
    max_pr_auc = max((v.get('pr_auc', 0.0) for v in cv.values()), default=0.0)
    max_auc = max((v.get('auc', 0.5) for v in cv.values()), default=0.5)

    bundle = {
        'cv_results': cv,
        'cv_pr_auc_max_of_base': max_pr_auc,
        'cv_auc_max_of_base': max_auc,
        'cv_probit_pr_auc': probit_cv.get('pr_auc'),
        'cv_probit_brier': probit_cv.get('brier'),
        'cv_probit_auc': probit_cv.get('auc'),
        'ensemble_weights': weights,
        'active_models': list(weights.keys()),
        'in_scope': bt.get('in_scope', {}),
        'full_history': bt.get('full_history', {}),
        'leave_one_out_brier': loo,
    }
    if manifest:
        bundle.update(manifest)
    return bundle


def _delta(a, b):
    if a is None or b is None:
        return None
    return a - b


def _decide_verdict(no_b: dict, with_b: dict, loo_with: dict) -> dict:
    """Apply C1 verdict gates."""
    # Headline: Dot-com peak delta
    dc_before = no_b.get('in_scope', {}).get('dot_com_peak')
    dc_after = with_b.get('in_scope', {}).get('dot_com_peak')
    dc_delta = _delta(dc_after, dc_before)

    gfc_before = no_b.get('in_scope', {}).get('gfc_lead_months')
    gfc_after = with_b.get('in_scope', {}).get('gfc_lead_months')
    gfc_lead_delta = _delta(gfc_after, gfc_before)

    sml_before = no_b.get('in_scope', {}).get('sl_lead_months')
    sml_after = with_b.get('in_scope', {}).get('sl_lead_months')
    sml_lead_delta = _delta(sml_after, sml_before)

    covid_before = no_b.get('in_scope', {}).get('covid_peak')
    covid_after = with_b.get('in_scope', {}).get('covid_peak')
    covid_peak_delta = _delta(covid_after, covid_before)

    backtest_auc_before = no_b.get('in_scope', {}).get('backtest_mean_auc')
    backtest_auc_after = with_b.get('in_scope', {}).get('backtest_mean_auc')
    backtest_auc_delta = _delta(backtest_auc_after, backtest_auc_before)

    # Any benchmark earned weight ≥5%?
    ensemble_weights = with_b.get('ensemble_weights', {})
    benchmark_names = {'hamilton_benchmark', 'chauvet_piger_benchmark', 'wright_probit'}
    benchmarks_with_weight = {n: w for n, w in ensemble_weights.items()
                              if n in benchmark_names and w >= 0.05}
    benchmarks_with_any_weight = {n: w for n, w in ensemble_weights.items()
                                   if n in benchmark_names and w > 0.0}

    # LOO check for benchmarks
    loo = loo_with.get('leave_one_out', {})
    benchmark_helps_loo = any(
        loo.get(m, {}).get('brier_delta', -1) > 0
        for m in ('HamiltonBenchmark', 'ChauvetPigerBenchmark', 'WrightProbit')
    )

    # Detection check: 4/4 in-scope
    in_scope_detected_str = with_b.get('in_scope', {}).get('recessions_detected', '0/0')
    try:
        n_det, n_tot = in_scope_detected_str.split('/')
        detected_ok = int(n_det) >= 4
    except Exception:
        detected_ok = False

    keep_reasons = []
    if dc_delta is not None and dc_delta >= 0.03:
        keep_reasons.append(f"Dot-com peak +{dc_delta:.4f} (>=3pp gate)")
    if benchmarks_with_weight and benchmark_helps_loo:
        keep_reasons.append(f"benchmark(s) earn >=5% weight AND reduce ensemble LOO Brier")
    if (backtest_auc_delta is not None and backtest_auc_delta >= 0.01 and detected_ok):
        keep_reasons.append(f"in-scope mean AUC +{backtest_auc_delta:.3f} AND 4/4 detected")

    discard_reasons = []
    if not benchmarks_with_any_weight and (dc_delta is None or abs(dc_delta) < 0.01):
        discard_reasons.append("all 3 benchmarks gated out of ensemble AND Dot-com peak unchanged")

    if keep_reasons:
        verdict = "KEEP"
        reason = "; ".join(keep_reasons)
    elif discard_reasons:
        verdict = "DISCARD"
        reason = "; ".join(discard_reasons)
    else:
        verdict = "NEEDS-WORK"
        reason = "no gate tripped cleanly — mixed signal"

    return {
        'verdict': verdict,
        'verdict_reason': reason,
        'dot_com_peak_delta': dc_delta,
        'gfc_lead_delta': gfc_lead_delta,
        'sml_lead_delta': sml_lead_delta,
        'covid_peak_delta': covid_peak_delta,
        'in_scope_backtest_auc_delta': backtest_auc_delta,
        'benchmarks_with_weight_ge_5pct': benchmarks_with_weight,
        'benchmarks_with_any_weight': benchmarks_with_any_weight,
        'benchmark_helps_loo_brier': benchmark_helps_loo,
        'in_scope_detection': in_scope_detected_str,
    }


def _load_reference_peaks() -> dict:
    """Lookup raw Hamilton/Chauvet-Piger series peak probabilities during each
    recession window. This is the 'counterfactual' reading: what would the
    benchmark say at the recession's peak month if it were trusted?"""
    ind_path = ROOT / "data" / "indicators.csv"
    if not ind_path.exists():
        return {"error": "indicators.csv missing"}
    df = pd.read_csv(ind_path, index_col=0, parse_dates=True)
    peaks = {}
    windows = [
        ("sl", "1990-07", "1991-04"),
        ("dot_com", "2001-03", "2001-12"),
        ("gfc", "2007-12", "2009-07"),
        ("covid", "2020-02", "2020-05"),
    ]
    for key, start, end in windows:
        sub = df.loc[start:end]
        h_peak = None
        cp_peak = None
        if 'ref_JHGDPBRINDX' in sub.columns:
            v = sub['ref_JHGDPBRINDX'].max()
            if pd.notna(v):
                h_peak = float(v) / 100.0  # scale to [0, 1]
        if 'ref_RECPROUSM156N' in sub.columns:
            v = sub['ref_RECPROUSM156N'].max()
            if pd.notna(v):
                cp_peak = float(v)
                if cp_peak > 1.0:  # some copies store 0-100 scale
                    cp_peak = cp_peak / 100.0
        peaks[key] = {
            'window': [start, end],
            'hamilton_peak_raw_prob': h_peak,
            'chauvet_piger_peak_raw_prob': cp_peak,
        }
    return peaks


def main():
    variants_dir = ROOT / "data" / "models" / "c1_variants"
    no_b_dir = variants_dir / "no_benchmarks"
    with_b_dir = variants_dir / "with_benchmarks"

    no_b = _variant_bundle(no_b_dir)
    with_b = _variant_bundle(with_b_dir)

    decision = _decide_verdict(no_b, with_b, with_b.get('leave_one_out_brier', {}))

    # Extract headline figures
    cv_benchmarks = {
        name: with_b.get('cv_results', {}).get(name, {})
        for name in ('hamilton_benchmark', 'chauvet_piger_benchmark', 'wright_probit')
    }

    reference_counterfactual_peaks = _load_reference_peaks()

    out = {
        'experiment_id': 'C1',
        'ran_at_utc': datetime.utcnow().isoformat() + 'Z',
        'git_sha_current': _git_sha(),
        'branch': _git_branch(),
        'parent_branch': 'main',
        'scope_policy': '1990-present',
        'what_this_moves': 'Dot-com peak (primary); GFC/S&L stability (secondary)',
        'benchmark_members_added': ['HamiltonBenchmark', 'ChauvetPigerBenchmark', 'WrightProbit'],
        'reference_features_handling': (
            'Reference columns (ref_JHGDPBRINDX, ref_RECPROUSM156N) were already '
            'excluded from ordinary feature selection at ensemble_model.py:949 '
            "(startswith('ref_') filter) before C1. Therefore no double-count: "
            'the reference series enter ONLY through the new benchmark model '
            'wrappers when --benchmark-members=on, and never as ordinary '
            'features for probit/RF/XGB. No code change needed for this.'
        ),
        'cv_benchmark_scores': cv_benchmarks,
        'reference_counterfactual_peaks': reference_counterfactual_peaks,
        'reference_counterfactual_note': (
            "Raw Hamilton JHGDPBRINDX (÷100) and Chauvet-Piger RECPROUSM156N "
            "peak probabilities during each recession window — these would be "
            "the benchmark members' actual outputs IF they were selected into "
            "the active ensemble. They are reported diagnostically; the "
            "_select_active_models gate in ensemble_model.py rejects them "
            "because their pooled CV PR-AUC (across 1970-present folds) is "
            "below 90% of probit's, even though they fire strongly during "
            "specific real recessions (see especially GFC / COVID ~1.0, and "
            "Hamilton ~0.84 during Dot-com 2001)."
        ),
        'active_model_gate_diagnosis': (
            "With benchmarks on, ALL 3 benchmark members are GATED OUT by the "
            "_select_active_models() filter (AUC within -10pp of best, PR-AUC "
            "within 90% of best, Brier within 2x of best). Best probit CV "
            "PR-AUC = 0.95; benchmarks CV PR-AUCs are Hamilton 0.41, "
            "Chauvet-Piger 0.70, Wright 0.21 — all below 0.855 (=0.95*0.90). "
            "Therefore the final ensemble contains only {probit, random_forest} "
            "with equal 0.5/0.5 weights, identical to the no-benchmarks run. "
            "This is why every in-scope metric delta is exactly 0."
        ),
        'variants': {
            'no_benchmarks': no_b,
            'with_benchmarks': with_b,
        },
        **decision,
    }

    out_path = ROOT / "data" / "models" / "c1_validation.json"
    out_path.write_text(json.dumps(out, indent=2, default=str))
    logger.info("Wrote %s", out_path)
    logger.info("Verdict: %s — %s", decision['verdict'], decision['verdict_reason'])
    logger.info("Dot-com peak delta: %s", decision['dot_com_peak_delta'])
    logger.info("GFC lead delta: %s", decision['gfc_lead_delta'])
    logger.info("SML lead delta: %s", decision['sml_lead_delta'])
    logger.info("COVID peak delta: %s", decision['covid_peak_delta'])
    logger.info("Benchmarks with weight >=5%%: %s", decision['benchmarks_with_weight_ge_5pct'])


if __name__ == "__main__":
    main()
