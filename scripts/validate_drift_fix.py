"""
Cross-validate the drift/calibration fix without retraining the model.

Phase 1: Re-run data acquisition with the fix and confirm the PRFI tail no
         longer carries forward indefinitely.
Phase 2: Run ModelMonitor's drift check against the freshly-acquired indicators
         (using existing predictions.csv for the calibration leg) and compare
         alert counts / top-drifted PSI values to the pre-fix monitor_report.json.

Pre-fix expectations (so we can grade the result):
  * PRFI tail: zero "flat carry" beyond Q4-published value + 2-month ffill grace.
  * Top drifted features RESIDENTIAL_INV_Z / RESIDENTIAL_INV_YOY / leading_PRFI_MoM:
    PSI should drop dramatically (10.92 -> << 1).
  * Binary features (e.g. GOODS_DECLINE_FLAG, HOUSE_PRICE_DECLINING) should
    be excluded with status "skipped-binary".
  * AT_RISK_DIFFUSION_WEIGHTED is multi-valued real signal -> should remain
    in the drift set with a moderately reduced PSI.
  * Total drifted feature count: meaningfully below 36.
"""
import json
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")
FRED_API_KEY = os.environ.get("FRED_API_KEY")
if not FRED_API_KEY:
    raise SystemExit("FRED_API_KEY not set in environment / .env")

from recession_engine.data_acquisition import RecessionDataAcquisition
from recession_engine.model_monitor import ModelMonitor


def main():
    pre = json.loads((ROOT / "data/models/monitor_report.json").read_text())

    # ---- Snapshot pre-fix state -------------------------------------------------
    print("=" * 64)
    print("PRE-FIX (data/models/monitor_report.json)")
    print("=" * 64)
    fd_pre = pre["checks"]["feature_drift"]["details"]
    print(f"  features_checked: {fd_pre['features_checked']}")
    print(f"  features_drifted: {fd_pre['features_drifted']}")
    print(f"  mean_psi: {fd_pre['mean_psi']}, max_psi: {fd_pre['max_psi']}")
    print(f"  top_drifted: {fd_pre.get('top_drifted')}")
    print()

    # ---- Phase 1: re-acquire data with fix --------------------------------------
    print("=" * 64)
    print("Phase 1 — Re-acquiring FRED data with fix applied")
    print("=" * 64)
    acq = RecessionDataAcquisition(fred_api_key=FRED_API_KEY)
    raw = acq.fetch_data()
    eng = acq.engineer_features(raw)
    print(f"  indicators rows: {len(eng)}, columns: {len(eng.columns)}")
    print(f"  date range: {eng.index.min().date()} -> {eng.index.max().date()}")

    print()
    print("  leading_PRFI tail-10 (post-fix):")
    prfi_tail = eng["leading_PRFI"].tail(10)
    print(prfi_tail.to_string().replace("\n", "\n    "))
    flat_run = (prfi_tail.diff() == 0).sum()
    nan_count = prfi_tail.isna().sum()
    print(f"  zero-change months in tail-10: {flat_run}  (pre-fix was 5)")
    print(f"  NaN months in tail-10: {nan_count}  (pre-fix was 2)")

    # ---- Phase 2: run drift / calibration monitor on new indicators -------------
    print()
    print("=" * 64)
    print("Phase 2 — Run ModelMonitor on new indicators")
    print("=" * 64)
    features = (ROOT / "data/models/features.txt").read_text().strip().splitlines()
    pre_pred = pd.read_csv(ROOT / "data/predictions.csv")
    pre_pred["Date"] = pd.to_datetime(pre_pred["Date"])
    pre_pred = pre_pred.set_index("Date", drop=False)

    mon = ModelMonitor()
    report = mon.run_all_checks(
        predictions_df=pre_pred,
        indicators_df=eng,
        feature_cols=features,
    )

    fd_post = report["checks"]["feature_drift"]["details"]
    print(f"  features_checked: {fd_post['features_checked']}")
    print(f"  features_drifted: {fd_post['features_drifted']}")
    print(f"  features_skipped_binary: {fd_post.get('features_skipped_binary', 'n/a')}")
    print(f"  mean_psi: {fd_post['mean_psi']}, max_psi: {fd_post['max_psi']}")
    print(f"  top_drifted: {fd_post.get('top_drifted')}")
    print()
    print("  Alerts:")
    for a in report["alerts"]:
        print(f"    [{a['level']}] {a['check']}: {a['message']}")

    # ---- Cross-validation: per-feature PSI before/after -------------------------
    print()
    print("=" * 64)
    print("Per-feature PSI: pre-fix vs post-fix (manual recompute)")
    print("=" * 64)
    suspects = [
        "RESIDENTIAL_INV_Z", "RESIDENTIAL_INV_YOY", "leading_PRFI_MoM",
        "GOODS_DECLINE_FLAG", "AT_RISK_DIFFUSION_WEIGHTED",
        "RESIDENTIAL_INV_DECLINING", "HOUSE_PRICE_DECLINING",
    ]
    pre_top = fd_pre.get("top_drifted", {})
    for f in suspects:
        before = pre_top.get(f, "(<top-5 in pre>)")
        if f not in eng.columns:
            after = "missing"
        else:
            ref = eng[f].iloc[-36:-12].dropna().values
            rec = eng[f].iloc[-12:].dropna().values
            if len(np.unique(ref)) <= 2:
                after = "skipped-binary"
            elif len(ref) >= 10 and len(rec) >= 5:
                after = round(mon._compute_psi(ref, rec), 3)
            else:
                after = "n/a"
        print(f"  {f}: {before} -> {after}")

    # ---- Summary delta ----------------------------------------------------------
    print()
    print("=" * 64)
    print("DELTA SUMMARY")
    print("=" * 64)
    print(f"  features_drifted: {fd_pre['features_drifted']} -> {fd_post['features_drifted']} "
          f"(delta {fd_post['features_drifted'] - fd_pre['features_drifted']:+d})")
    print(f"  mean_psi: {fd_pre['mean_psi']} -> {fd_post['mean_psi']}")
    print(f"  max_psi:  {fd_pre['max_psi']} -> {fd_post['max_psi']}")
    print(f"  alert_count: {pre['alert_count']} -> {report['alert_count']}")


if __name__ == "__main__":
    main()
