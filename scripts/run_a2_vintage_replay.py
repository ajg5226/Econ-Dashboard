"""
A2 — Full ALFRED Vintage Replay
===============================

Extends the existing 5-origin ALFRED audit (``data/models/alfred_vintage_results.csv``)
to >=20 forecast origins. For each origin we compute three predicted recession
probabilities at ``origin + 6 months`` horizon:

* ``prob_vintage``   — core series replaced with their ALFRED as-of snapshots.
* ``prob_simulated`` — core series left at their revised values but the last
  ``PUBLICATION_LAGS`` rows nulled (production default).
* ``prob_revised``   — full revised data, no masking (upper-bound diagnostic).

Training labels for all three pipelines are only observable through
``origin - horizon_months`` so the comparison is apples-to-apples.

Outputs
-------
* ``data/models/a2_vintage_replay.csv``          — per-origin probabilities + gaps
* ``data/models/a2_vintage_replay_summary.md``   — aggregate stats + recommendation
* ``data/models/a2_validation.json``             — machine-readable verdict
* ``data/alfred_cache/*.parquet``                — raw ALFRED vintage snapshots

The cache is populated on first run and re-used on every subsequent call, so
repeat executions of this experiment are nearly free.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# When executing from a git worktree the ``.env`` lives only at the main repo
# root — fall back to that location so FRED_API_KEY resolves reliably.
_env_candidates = [REPO_ROOT / ".env"]
for _p in list(REPO_ROOT.parents):
    _env_candidates.append(_p / ".env")
for _env in _env_candidates:
    if _env.exists():
        load_dotenv(_env)
        break

if not os.environ.get("FRED_API_KEY"):
    try:
        import streamlit as st  # type: ignore
        os.environ["FRED_API_KEY"] = st.secrets.get("FRED_API_KEY", "")
    except Exception:
        pass

from recession_engine.backtester import PUBLICATION_LAGS, RecessionBacktester  # noqa: E402
from recession_engine.data_acquisition import RecessionDataAcquisition  # noqa: E402
from recession_engine.ensemble_model import RecessionEnsembleModel  # noqa: E402
from recession_engine.vintage_cache import (  # noqa: E402
    get_vintage,
    vintage_series_to_monthly,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("a2_vintage_replay")
logging.getLogger("recession_engine.ensemble_model").setLevel(logging.WARNING)
logging.getLogger("recession_engine.data_acquisition").setLevel(logging.WARNING)

# Core ALFRED-backed series called out in the A2 brief.
CORE_SERIES = [
    "coincident_PAYEMS",
    "coincident_UNRATE",
    "coincident_INDPRO",
    "coincident_PI",
    "coincident_RSXFS",
    "coincident_CMRMTSPL",
    "lagging_UEMPMEAN",  # UEMPMEAN lives under lagging/ in data_acquisition
    "leading_PERMIT",
    "leading_HOUST",
    # Optional: UNEMPLOY (not currently a feature column — tracked for coverage)
]

HORIZON_MONTHS = 6
MIN_TRAIN_MONTHS = 180
MAX_FEATURES = 50
N_CV_SPLITS = 5

# Output artifacts
A2_ORIGINS_PATH = REPO_ROOT / "data/models/a2_vintage_origins.json"
REPLAY_CSV = REPO_ROOT / "data/models/a2_vintage_replay.csv"
REPLAY_MD = REPO_ROOT / "data/models/a2_vintage_replay_summary.md"
VALIDATION_JSON = REPO_ROOT / "data/models/a2_validation.json"
LEDGER_PATH = REPO_ROOT / "data/reports/experiment_ledger.md"


def _month_end(ts) -> pd.Timestamp:
    return pd.Timestamp(ts).to_period("M").to_timestamp("M")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown"


def _load_origins() -> list[dict]:
    if not A2_ORIGINS_PATH.exists():
        raise FileNotFoundError(
            f"Missing A2 origin list at {A2_ORIGINS_PATH}; run the setup step first."
        )
    with A2_ORIGINS_PATH.open() as f:
        payload = json.load(f)
    return payload.get("origins", [])


def _fetch_or_load_raw(acq: RecessionDataAcquisition, cache_path: Path) -> pd.DataFrame:
    """Load cached raw frame when available, otherwise fetch fresh."""
    if cache_path.exists():
        df_raw = pd.read_parquet(cache_path)
        logger.info(
            "Loaded cached raw frame %s (%d rows x %d cols)",
            cache_path.name,
            df_raw.shape[0],
            df_raw.shape[1],
        )
        return df_raw

    logger.info("Fetching raw indicator frame from FRED...")
    df_raw = acq.fetch_data(start_date="1970-01-01")
    df_raw.to_parquet(cache_path)
    logger.info(
        "Cached raw frame to %s (%d rows x %d cols)",
        cache_path,
        df_raw.shape[0],
        df_raw.shape[1],
    )
    return df_raw


def _apply_simulated_lags(df_raw: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """Trim to ``asof`` and null the most recent N rows per PUBLICATION_LAGS.

    This mirrors ``RecessionBacktester._build_realtime_feature_frame`` so the
    simulated scenario matches production's default behaviour: the last
    ``lag`` monthly rows of each column are unknown at the forecast origin.
    """
    df_scoped = df_raw[df_raw.index <= asof].copy()
    for col in df_scoped.columns:
        lag = PUBLICATION_LAGS.get(col, 1)
        if lag > 0 and col != "RECESSION":
            df_scoped.iloc[-lag:, df_scoped.columns.get_loc(col)] = np.nan
    return df_scoped


def _substitute_alfred_vintages(
    df_raw: pd.DataFrame,
    asof: pd.Timestamp,
    api_key: str,
    core_series: list[str],
) -> tuple[pd.DataFrame, dict]:
    """Swap core series with their ALFRED vintage and trim to ``asof``.

    The returned frame contains:
    * observations only through ``asof``
    * core series replaced with the ALFRED as-of vintage snapshot
      (where the series exists in ALFRED)
    * non-core columns left at revised values *except* for their last
      ``PUBLICATION_LAGS`` rows, which are nulled to match production
      information-set assumptions.
    """
    df_vintage = df_raw[df_raw.index <= asof].copy()
    coverage: dict[str, str] = {}
    asof_str = asof.strftime("%Y-%m-%d")

    for col in core_series:
        if col not in df_vintage.columns:
            coverage[col] = "missing_in_frame"
            continue

        vintage_raw = get_vintage(col, asof_str, api_key)
        if vintage_raw is None or vintage_raw.empty:
            # Fall back to simulated lag for this column.
            lag = PUBLICATION_LAGS.get(col, 1)
            if lag > 0:
                df_vintage.iloc[-lag:, df_vintage.columns.get_loc(col)] = np.nan
            coverage[col] = "alfred_unavailable"
            continue

        monthly = vintage_series_to_monthly(vintage_raw)
        aligned = monthly.reindex(df_vintage.index)
        if aligned.notna().any():
            df_vintage[col] = aligned.values
            coverage[col] = "alfred_ok"
        else:
            lag = PUBLICATION_LAGS.get(col, 1)
            if lag > 0:
                df_vintage.iloc[-lag:, df_vintage.columns.get_loc(col)] = np.nan
            coverage[col] = "alfred_empty_after_align"

    # For non-core columns apply the simulated lag so we don't leak revised
    # data from an unrelated series (e.g. ICSA, NFCI) into the vintage run.
    for col in df_vintage.columns:
        if col in core_series or col == "RECESSION":
            continue
        lag = PUBLICATION_LAGS.get(col, 1)
        if lag > 0:
            df_vintage.iloc[-lag:, df_vintage.columns.get_loc(col)] = np.nan

    return df_vintage, coverage


def _run_single_pipeline(
    bt: RecessionBacktester,
    df_raw_scenario: pd.DataFrame,
    df_target_full: pd.DataFrame,
    origin: pd.Timestamp,
    *,
    label: str,
) -> dict:
    """Fit + predict once for the given scenario frame."""
    target_col = f"RECESSION_FORWARD_{HORIZON_MONTHS}M"
    label_cutoff = _month_end(origin - pd.DateOffset(months=HORIZON_MONTHS))

    # Rebuild engineered features on the scenario raw frame.
    df_features = bt.acq.engineer_features(df_raw_scenario.copy())
    df_features_with_target = df_features.copy()
    df_features_with_target[target_col] = df_target_full[target_col].reindex(
        df_features.index
    )

    train_df = df_features_with_target[
        (df_features_with_target.index <= label_cutoff)
        & df_features_with_target[target_col].notna()
    ].copy()

    if len(train_df) < MIN_TRAIN_MONTHS:
        return {
            "pipeline": label,
            "status": "insufficient_train_rows",
            "train_rows": int(len(train_df)),
            "prob": np.nan,
            "threshold": np.nan,
        }

    if train_df[target_col].nunique() < 2:
        return {
            "pipeline": label,
            "status": "single_class_train",
            "train_rows": int(len(train_df)),
            "prob": np.nan,
            "threshold": np.nan,
        }

    if origin not in df_features_with_target.index:
        return {
            "pipeline": label,
            "status": "origin_missing_from_frame",
            "train_rows": int(len(train_df)),
            "prob": np.nan,
            "threshold": np.nan,
        }

    pred_df = df_features_with_target.loc[[origin]].copy()

    try:
        model = bt._instantiate_model(n_cv_splits=N_CV_SPLITS, model_config={})
        model.fit(train_df, max_features=MAX_FEATURES)
        preds = model.predict(pred_df)
        prob = bt._coerce_probability(preds["ensemble"])
        threshold = float(model.decision_threshold)
        actual = pred_df[target_col].iloc[-1]
    except Exception as exc:
        logger.exception("Pipeline %s failed at origin %s: %s", label, origin, exc)
        return {
            "pipeline": label,
            "status": f"error:{type(exc).__name__}",
            "train_rows": int(len(train_df)),
            "prob": np.nan,
            "threshold": np.nan,
        }

    return {
        "pipeline": label,
        "status": "ok",
        "train_rows": int(len(train_df)),
        "prob": float(prob),
        "threshold": threshold,
        "actual": float(actual) if pd.notna(actual) else np.nan,
    }


def run_replay(origins: list[dict]) -> pd.DataFrame:
    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key:
        raise RuntimeError("FRED_API_KEY is required for A2 vintage replay")

    acq = RecessionDataAcquisition(api_key)
    bt = RecessionBacktester(acq, RecessionEnsembleModel, target_horizon=HORIZON_MONTHS)

    df_raw = _fetch_or_load_raw(acq, Path("/tmp/a2_df_raw.parquet"))

    # Shared target frame (same for every scenario — label leakage is controlled
    # via the per-origin ``label_cutoff`` inside ``_run_single_pipeline``).
    df_features_full = acq.engineer_features(df_raw.copy())
    df_target_full = acq.create_forecast_target(df_features_full, HORIZON_MONTHS)

    rows: list[dict] = []
    coverage_summary: dict[str, set[str]] = {}

    t_start = time.time()
    for idx, origin_spec in enumerate(origins, 1):
        origin = _month_end(origin_spec["origin_date"])
        label = origin_spec["label"]
        category = origin_spec.get("category", "unknown")
        logger.info(
            "[%d/%d] %s (%s) — %s",
            idx,
            len(origins),
            origin_spec["origin_date"],
            category,
            label,
        )

        # ── Revised pipeline (no masking, but trimmed to ``asof``) ───────
        df_revised_scoped = df_raw[df_raw.index <= origin].copy()
        revised = _run_single_pipeline(
            bt, df_revised_scoped, df_target_full, origin, label="revised",
        )

        # ── Simulated-lag pipeline ───────────────────────────────────────
        df_simulated = _apply_simulated_lags(df_raw, origin)
        simulated = _run_single_pipeline(
            bt, df_simulated, df_target_full, origin, label="simulated",
        )

        # ── ALFRED vintage pipeline ──────────────────────────────────────
        df_vintage, coverage = _substitute_alfred_vintages(
            df_raw, origin, api_key, CORE_SERIES,
        )
        vintage = _run_single_pipeline(
            bt, df_vintage, df_target_full, origin, label="vintage",
        )

        coverage_summary[origin_spec["origin_date"]] = coverage

        prob_vintage = vintage.get("prob", np.nan)
        prob_simulated = simulated.get("prob", np.nan)
        prob_revised = revised.get("prob", np.nan)

        def _gap(a, b):
            if pd.isna(a) or pd.isna(b):
                return np.nan
            return float(a - b)

        actual = (
            revised.get("actual")
            if revised.get("actual") is not None
            else simulated.get("actual", np.nan)
        )

        alfred_ok_cnt = sum(1 for v in coverage.values() if v == "alfred_ok")

        rows.append(
            {
                "origin_date": origin_spec["origin_date"],
                "label": label,
                "category": category,
                "actual_forward": actual,
                "prob_vintage": prob_vintage,
                "prob_simulated": prob_simulated,
                "prob_revised": prob_revised,
                "gap_vintage_vs_simulated": _gap(prob_vintage, prob_simulated),
                "gap_vintage_vs_revised": _gap(prob_vintage, prob_revised),
                "gap_simulated_vs_revised": _gap(prob_simulated, prob_revised),
                "threshold_vintage": vintage.get("threshold", np.nan),
                "threshold_simulated": simulated.get("threshold", np.nan),
                "threshold_revised": revised.get("threshold", np.nan),
                "signal_vintage": (
                    int(prob_vintage >= vintage.get("threshold", np.nan))
                    if pd.notna(prob_vintage) and pd.notna(vintage.get("threshold"))
                    else np.nan
                ),
                "signal_simulated": (
                    int(prob_simulated >= simulated.get("threshold", np.nan))
                    if pd.notna(prob_simulated) and pd.notna(simulated.get("threshold"))
                    else np.nan
                ),
                "signal_revised": (
                    int(prob_revised >= revised.get("threshold", np.nan))
                    if pd.notna(prob_revised) and pd.notna(revised.get("threshold"))
                    else np.nan
                ),
                "alfred_ok_count": alfred_ok_cnt,
                "alfred_coverage": json.dumps(coverage),
                "train_rows": vintage.get("train_rows", np.nan),
                "status_vintage": vintage.get("status"),
                "status_simulated": simulated.get("status"),
                "status_revised": revised.get("status"),
            }
        )

        elapsed = time.time() - t_start
        logger.info(
            "    vintage=%.3f simulated=%.3f revised=%.3f | alfred_ok=%d | elapsed=%.1fs",
            prob_vintage if pd.notna(prob_vintage) else float("nan"),
            prob_simulated if pd.notna(prob_simulated) else float("nan"),
            prob_revised if pd.notna(prob_revised) else float("nan"),
            alfred_ok_cnt,
            elapsed,
        )

    df_results = pd.DataFrame(rows)
    return df_results


def _aggregate_stats(df: pd.DataFrame) -> dict:
    """Compute headline diagnostics on the replay frame."""
    valid = df.dropna(subset=["prob_vintage", "prob_simulated"], how="any").copy()
    if valid.empty:
        return {
            "origins": 0,
            "mean_abs_gap_vintage_vs_simulated": None,
            "mean_signed_gap_vintage_vs_simulated": None,
            "max_abs_gap_vintage_vs_simulated": None,
            "max_abs_gap_origin": None,
            "stddev_gap_vintage_vs_simulated": None,
            "mean_abs_gap_vintage_vs_revised": None,
            "mean_abs_gap_simulated_vs_revised": None,
        }

    vs_sim_gap = valid["gap_vintage_vs_simulated"].astype(float)
    vs_rev_gap = valid["gap_vintage_vs_revised"].astype(float)
    sim_vs_rev_gap = valid["gap_simulated_vs_revised"].astype(float)

    max_idx = vs_sim_gap.abs().idxmax()
    return {
        "origins": int(len(valid)),
        "mean_abs_gap_vintage_vs_simulated": float(vs_sim_gap.abs().mean()),
        "mean_signed_gap_vintage_vs_simulated": float(vs_sim_gap.mean()),
        "max_abs_gap_vintage_vs_simulated": float(vs_sim_gap.abs().max()),
        "max_abs_gap_origin": str(valid.loc[max_idx, "origin_date"]),
        "stddev_gap_vintage_vs_simulated": float(vs_sim_gap.std(ddof=0)),
        "mean_abs_gap_vintage_vs_revised": float(vs_rev_gap.abs().mean()),
        "mean_abs_gap_simulated_vs_revised": float(sim_vs_rev_gap.abs().mean()),
    }


def _per_recession_behavior(df: pd.DataFrame) -> dict:
    """Extract per-recession divergence: detection flips and magnitudes."""
    positives = df[df["category"] == "recession_peak"].copy()
    out: dict[str, dict] = {}
    for _, row in positives.iterrows():
        origin = row["origin_date"]
        prob_v = row.get("prob_vintage")
        prob_s = row.get("prob_simulated")
        sig_v = row.get("signal_vintage")
        sig_s = row.get("signal_simulated")
        out[origin] = {
            "label": row.get("label"),
            "prob_vintage": (
                float(prob_v) if pd.notna(prob_v) else None
            ),
            "prob_simulated": (
                float(prob_s) if pd.notna(prob_s) else None
            ),
            "prob_revised": (
                float(row.get("prob_revised"))
                if pd.notna(row.get("prob_revised"))
                else None
            ),
            "gap_vintage_vs_simulated": (
                float(row.get("gap_vintage_vs_simulated"))
                if pd.notna(row.get("gap_vintage_vs_simulated"))
                else None
            ),
            "signal_vintage": (
                int(sig_v) if pd.notna(sig_v) else None
            ),
            "signal_simulated": (
                int(sig_s) if pd.notna(sig_s) else None
            ),
            "detection_changes": (
                bool(int(sig_v) != int(sig_s))
                if pd.notna(sig_v) and pd.notna(sig_s)
                else False
            ),
            "alfred_ok_count": (
                int(row["alfred_ok_count"])
                if pd.notna(row["alfred_ok_count"])
                else None
            ),
        }
    return out


def _coverage_summary(df: pd.DataFrame, core_series: list[str]) -> dict:
    """Aggregate per-series ALFRED availability across all origins."""
    per_series_ok: dict[str, int] = {c: 0 for c in core_series}
    per_series_total: dict[str, int] = {c: 0 for c in core_series}
    for _, row in df.iterrows():
        coverage = json.loads(row.get("alfred_coverage", "{}") or "{}")
        for col, status in coverage.items():
            per_series_total[col] = per_series_total.get(col, 0) + 1
            if status == "alfred_ok":
                per_series_ok[col] = per_series_ok.get(col, 0) + 1
    with_alfred = [c for c in core_series if per_series_ok.get(c, 0) > 0]
    without_alfred = [c for c in core_series if per_series_ok.get(c, 0) == 0]
    return {
        "per_series_hits": per_series_ok,
        "per_series_attempts": per_series_total,
        "with_alfred": with_alfred,
        "without_alfred": without_alfred,
    }


def _decide_verdict(stats: dict, per_recession: dict) -> tuple[str, str, str]:
    """Gate A2 against the verdict criteria in the experiment brief."""
    mean_abs = stats.get("mean_abs_gap_vintage_vs_simulated")
    max_abs = stats.get("max_abs_gap_vintage_vs_simulated")

    detection_flips = sum(
        1
        for v in per_recession.values()
        if v.get("detection_changes")
    )

    if mean_abs is None:
        return "NEEDS-WORK", "No valid vintage-vs-simulated comparisons produced.", "mixed"

    # Production recommendation
    if mean_abs < 0.015:
        recommendation = "keep_simulation"
    elif mean_abs > 0.03 or (max_abs is not None and max_abs > 0.10):
        recommendation = "switch_to_vintage_for_strict_search"
    else:
        recommendation = "mixed"

    # Verdict gates
    keep_reasons = []
    if mean_abs > 0.03:
        keep_reasons.append(f"mean abs gap {mean_abs*100:.1f}pp > 3pp")
    if max_abs is not None and max_abs > 0.10:
        keep_reasons.append(f"max abs gap {max_abs*100:.1f}pp > 10pp")
    if detection_flips > 0:
        keep_reasons.append(
            f"{detection_flips} in-scope recession(s) show detection changes"
        )
    if recommendation == "switch_to_vintage_for_strict_search":
        keep_reasons.append("production recommendation is switch_to_vintage")

    if keep_reasons:
        return (
            "KEEP",
            "Simulation is materially off: " + "; ".join(keep_reasons),
            recommendation,
        )

    if mean_abs < 0.015 and detection_flips == 0:
        return (
            "DISCARD",
            (
                f"Simulation matches vintage within {mean_abs*100:.2f}pp mean gap "
                "and no detection changes — simulated lags are an accurate proxy."
            ),
            recommendation,
        )

    return (
        "NEEDS-WORK",
        (
            f"Mean abs gap {mean_abs*100:.2f}pp with 0 detection flips but "
            "above the DISCARD threshold — continue monitoring."
        ),
        recommendation,
    )


def _render_markdown_summary(
    df: pd.DataFrame,
    stats: dict,
    per_recession: dict,
    coverage: dict,
    verdict: str,
    verdict_reason: str,
    recommendation: str,
    core_series: list[str],
    origins_count: int,
) -> str:
    lines: list[str] = []
    lines.append("# A2 — Full ALFRED Vintage Replay Summary")
    lines.append("")
    lines.append(
        f"Ran at: `{datetime.now(timezone.utc).isoformat(timespec='seconds')}`"
    )
    lines.append(f"Origins attempted: **{origins_count}**  ")
    lines.append(
        f"Origins with valid vintage/simulated comparisons: **{stats.get('origins', 0)}**"
    )
    lines.append(f"Core series considered: {len(core_series)}")
    lines.append("")

    lines.append("## Verdict")
    lines.append(f"**{verdict}** — {verdict_reason}")
    lines.append("")
    lines.append(f"Production recommendation: `{recommendation}`")
    lines.append("")

    lines.append("## Aggregate gap (vintage vs simulated)")
    mean_abs = stats.get("mean_abs_gap_vintage_vs_simulated") or 0.0
    mean_signed = stats.get("mean_signed_gap_vintage_vs_simulated") or 0.0
    max_abs = stats.get("max_abs_gap_vintage_vs_simulated") or 0.0
    std_gap = stats.get("stddev_gap_vintage_vs_simulated") or 0.0
    mean_abs_rev = stats.get("mean_abs_gap_vintage_vs_revised") or 0.0
    mean_abs_sim_rev = stats.get("mean_abs_gap_simulated_vs_revised") or 0.0
    lines.append(f"- Mean |gap|: **{mean_abs*100:.2f} pp**")
    lines.append(f"- Mean signed gap (vintage − simulated): {mean_signed*100:+.2f} pp")
    lines.append(
        f"- Max |gap|: {max_abs*100:.2f} pp (origin {stats.get('max_abs_gap_origin')})"
    )
    lines.append(f"- Std of gap: {std_gap*100:.2f} pp")
    lines.append("")
    lines.append(f"Adjacent diagnostics:")
    lines.append(f"- Mean |vintage − revised|: {mean_abs_rev*100:.2f} pp")
    lines.append(f"- Mean |simulated − revised|: {mean_abs_sim_rev*100:.2f} pp")
    lines.append("")

    lines.append("## Per-recession behaviour")
    if not per_recession:
        lines.append("(no in-scope recession origins in the sample)")
    else:
        lines.append(
            "| Origin | Label | Prob vintage | Prob simulated | Prob revised | Gap | "
            "Signal vintage | Signal simulated | Detection flip |"
        )
        lines.append(
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"
        )
        for origin, rec in per_recession.items():
            lines.append(
                "| {origin} | {label} | {pv} | {ps} | {pr} | {gap} | {sv} | {ss} | {flip} |".format(
                    origin=origin,
                    label=rec.get("label", ""),
                    pv=(f"{rec['prob_vintage']*100:.1f}%" if rec.get("prob_vintage") is not None else "n/a"),
                    ps=(f"{rec['prob_simulated']*100:.1f}%" if rec.get("prob_simulated") is not None else "n/a"),
                    pr=(f"{rec['prob_revised']*100:.1f}%" if rec.get("prob_revised") is not None else "n/a"),
                    gap=(
                        f"{rec['gap_vintage_vs_simulated']*100:+.1f} pp"
                        if rec.get("gap_vintage_vs_simulated") is not None
                        else "n/a"
                    ),
                    sv=rec.get("signal_vintage"),
                    ss=rec.get("signal_simulated"),
                    flip="YES" if rec.get("detection_changes") else "no",
                )
            )
    lines.append("")

    lines.append("## Five origins with largest |vintage − simulated| gap")
    df_sorted = (
        df.dropna(subset=["gap_vintage_vs_simulated"])
        .assign(abs_gap=lambda d: d["gap_vintage_vs_simulated"].abs())
        .sort_values("abs_gap", ascending=False)
    )
    if df_sorted.empty:
        lines.append("(no valid comparisons)")
    else:
        lines.append(
            "| Origin | Category | Label | Prob vintage | Prob simulated | Gap |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for _, row in df_sorted.head(5).iterrows():
            lines.append(
                "| {origin} | {cat} | {label} | {pv:.1%} | {ps:.1%} | {gap:+.1%} |".format(
                    origin=row["origin_date"],
                    cat=row["category"],
                    label=row["label"],
                    pv=row["prob_vintage"],
                    ps=row["prob_simulated"],
                    gap=row["gap_vintage_vs_simulated"],
                )
            )
    lines.append("")

    lines.append("## ALFRED coverage by series")
    lines.append("| Series | Hits | Attempts |")
    lines.append("| --- | --- | --- |")
    for col in core_series:
        hits = coverage.get("per_series_hits", {}).get(col, 0)
        attempts = coverage.get("per_series_attempts", {}).get(col, 0)
        lines.append(f"| `{col}` | {hits} | {attempts} |")
    lines.append("")
    lines.append(f"Series with ALFRED coverage: {', '.join(coverage.get('with_alfred', [])) or 'none'}")
    lines.append("")
    lines.append(f"Series without ALFRED coverage: {', '.join(coverage.get('without_alfred', [])) or 'none'}")
    lines.append("")

    lines.append("## Production guidance")
    if recommendation == "keep_simulation":
        lines.append(
            "- Keep the publication-lag simulation as the default backtest and "
            "production feature-engineering path."
        )
        lines.append(
            "- Retain the ALFRED cache as a verification-only tool (e.g. spot "
            "checks before promoting a model)."
        )
    elif recommendation == "switch_to_vintage_for_strict_search":
        lines.append(
            "- Simulation systematically diverges from ALFRED vintage data by "
            ">3pp on average or >10pp on a peak origin. Treat the current "
            "strict_vintage_search as optimistic."
        )
        lines.append(
            "- Switch strict_vintage_search to use ALFRED vintages for core "
            "series; leave the live production pipeline on simulated lag but "
            "record the bias in the executive report."
        )
    else:
        lines.append(
            "- Results are mixed — mean gap is moderate and no detection flips."
            " Continue monitoring on refresh cycles."
        )
    lines.append("")

    return "\n".join(lines) + "\n"


def _append_ledger(verdict: str, stats: dict, per_recession: dict, origins_count: int,
                   series_count: int, cached_parquets: int, git_sha: str, recommendation: str) -> None:
    LEDGER_PATH.parent.mkdir(parents=True, exist_ok=True)
    lead_note = "n/a"
    detection_flips = sum(
        1 for v in per_recession.values() if v.get("detection_changes")
    )
    n_positives = len(per_recession)
    mean_abs = stats.get("mean_abs_gap_vintage_vs_simulated") or 0.0
    max_abs = stats.get("max_abs_gap_vintage_vs_simulated") or 0.0
    origin_of_max = stats.get("max_abs_gap_origin", "n/a")

    # Lead-time change tracking is not part of a single-origin evaluation; we
    # record whether detection flipped on any positive origin instead.
    lead_time_note = (
        "detection flips" if detection_flips > 0 else "no detection flips"
    )

    row = (
        f"| A2 | Vintage pipeline formalization (full ALFRED replay) | DONE | "
        f"{verdict} | n/a | n/a | n/a | {lead_time_note} | "
        f"Branch `experiment/A2-full-alfred-replay` SHA `{git_sha}`. Extended "
        f"ALFRED replay from 5 → {origins_count} origins across {series_count} "
        f"core series. Mean abs gap: {mean_abs*100:.2f}pp (prev 19.19pp on 5 "
        f"points). Max gap: {max_abs*100:.2f}pp at {origin_of_max}. Detection "
        f"changes: {detection_flips} of {n_positives} in-scope recessions. "
        f"Production recommendation: {recommendation}. Cached {cached_parquets} "
        f"vintage parquets at `data/alfred_cache/`. See "
        f"`data/models/a2_vintage_replay_summary.md`. |\n"
    )

    # Create ledger if absent; otherwise append.
    if not LEDGER_PATH.exists():
        header = (
            "# Experiment ledger\n\n"
            "| ID | Title | Status | Verdict | AUC delta | Brier delta | PR-AUC delta | Lead-time note | Notes |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
        )
        LEDGER_PATH.write_text(header + row)
    else:
        with LEDGER_PATH.open("a") as f:
            f.write(row)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run A2 full ALFRED vintage replay")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional cap on number of origins (for smoke tests)",
    )
    args = parser.parse_args()

    origins = _load_origins()
    if args.limit is not None:
        origins = origins[: args.limit]

    t0 = time.time()
    df = run_replay(origins)
    elapsed = time.time() - t0

    REPLAY_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(REPLAY_CSV, index=False)
    logger.info("Wrote %s (%d rows)", REPLAY_CSV, len(df))

    stats = _aggregate_stats(df)
    per_recession = _per_recession_behavior(df)
    coverage = _coverage_summary(df, CORE_SERIES)
    verdict, verdict_reason, recommendation = _decide_verdict(stats, per_recession)

    md = _render_markdown_summary(
        df, stats, per_recession, coverage, verdict, verdict_reason,
        recommendation, CORE_SERIES, origins_count=len(origins),
    )
    REPLAY_MD.write_text(md)
    logger.info("Wrote %s", REPLAY_MD)

    cached_parquets = len(list((REPO_ROOT / "data/alfred_cache").glob("*.parquet")))

    validation = {
        "experiment_id": "A2",
        "ran_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha_current": _git_sha(),
        "origins_attempted": len(origins),
        "origins_with_full_alfred_coverage": int(
            (df["alfred_ok_count"] == len(coverage.get("with_alfred", []))).sum()
        ),
        "origins_with_partial_coverage": int(
            ((df["alfred_ok_count"] > 0) & (df["alfred_ok_count"] < len(CORE_SERIES))).sum()
        ),
        "core_series_attempted": CORE_SERIES,
        "core_series_with_alfred": coverage.get("with_alfred", []),
        "core_series_without_alfred": coverage.get("without_alfred", []),
        "aggregate_stats": stats,
        "per_recession_behavior": per_recession,
        "production_recommendation": recommendation,
        "verdict": verdict,
        "verdict_reason": verdict_reason,
        "replay_runtime_seconds": round(elapsed, 2),
        "cached_alfred_parquets": cached_parquets,
    }
    VALIDATION_JSON.write_text(json.dumps(validation, indent=2, default=str))
    logger.info("Wrote %s", VALIDATION_JSON)

    _append_ledger(
        verdict=verdict,
        stats=stats,
        per_recession=per_recession,
        origins_count=len(origins),
        series_count=len(CORE_SERIES),
        cached_parquets=cached_parquets,
        git_sha=_git_sha(),
        recommendation=recommendation,
    )

    logger.info("A2 replay complete in %.1fs — verdict %s", elapsed, verdict)
    return 0


if __name__ == "__main__":
    sys.exit(main())
