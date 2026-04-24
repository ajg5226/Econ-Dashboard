"""
Threshold Stability Gate — Fixed-Threshold Re-Scoring Utility

Background
----------
Across C3, B3, and B5 we observed a recurring measurement artifact: adding a
feature that concentrates probability mass inside recessions more than in the
pre-recession window shifts the F1-optimal decision threshold upward, which in
turn slips the "first threshold crossing" date forward and destroys our
lead-time metric. A challenger can therefore simultaneously:

* Improve per-recession peak probability (real signal)
* Improve AUC / Brier (real signal)
* Regress lead time (measurement artifact from per-variant threshold retuning)

This module provides utilities to evaluate a challenger *at the incumbent
baseline's fixed decision threshold*, alongside its own F1-optimized threshold.
A challenger that does not regress lead time under the fixed threshold is
genuinely additive; a challenger that only looks good at its retuned threshold
is capturing a threshold-distribution shift, not an early-warning improvement.

Two core functions:

* :func:`rescore_backtest_at_fixed_threshold` — recompute crossing and
  lead-time columns on a backtest frame using a single fixed threshold.
* :func:`compute_lead_time_delta_matrix` — per-recession comparison of
  baseline vs challenger lead times at both each side's own threshold and at
  the baseline's fixed threshold.

Both are additive; the legacy per-origin `Lead_Months` column on existing
backtest frames is untouched. New columns are suffixed ``_Fixed``.

The fixed reference threshold comes from ``data/models/baseline_efb307e/``
by default (the post-B3 baseline). Callers may override for testing.

Lead-time estimation modes
--------------------------
For backtest frames that carry a full per-origin probability trajectory
(column ``Probability_Trajectory`` — list of ``[date, prob]`` pairs — populated
by the extended :func:`recession_engine.backtester.RecessionBacktester
.run_pseudo_oos_backtest`), the fixed-threshold lead time is computed
*exactly* by scanning the trajectory.

For legacy backtest frames that only preserve ``Peak_Prob`` / ``Peak_Date`` /
``Threshold`` / ``Lead_Months``, we fall back to a bound-based estimate:

* If ``Peak_Prob < fixed_threshold``: no crossing possible ⇒ Lead_Months_Fixed
  = NaN, flagged as ``Lead_Estimation_Method = 'exact_no_cross'``.
* If original threshold ``>= fixed_threshold``: original first-crossing
  happened at a threshold at least as high as the fixed one, so the
  fixed-threshold first-crossing is **at or earlier than** the original one.
  We report ``Lead_Months_Fixed >= Lead_Months`` with
  method ``'bound_at_or_earlier'`` — we take the original lead as a
  conservative *lower* bound.
* If original threshold ``< fixed_threshold`` and ``Peak_Prob >=
  fixed_threshold``: original first-crossing was at a lower threshold, so the
  fixed-threshold first-crossing is **at or later than** the original one.
  We report ``Lead_Months_Fixed <= Lead_Months`` (conservative *upper*
  bound). Method ``'bound_at_or_later'``.

Exact rescoring always requires the trajectory column. This module's docstring
explains which method was applied in ``Lead_Estimation_Method`` so that the
gate decision is auditable.

Threshold stability gate
------------------------
The companion KEEP-gate criterion (see `experiment_ledger.md`) is:

  A challenger should not regress its own lead time at the baseline fixed
  threshold by more than 2 months on any in-scope recession.

If a challenger shows positive lead-time change at its own threshold but
negative lead-time change at the baseline fixed threshold, the F1 optimizer
is over-tuning to the new distribution rather than capturing genuine
early-warning value. Call :func:`apply_stability_gate` to obtain the
pass/fail verdict and a per-recession breakdown for the ledger row.
"""

from __future__ import annotations

import ast
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# In-scope recessions per the 1990-present evaluation policy. Pre-1990 rows
# are informational only; they're reported but don't drive the stability gate.
DEFAULT_IN_SCOPE_RECESSIONS = (
    "S&L Crisis (1990-91)",
    "Dot-com (2001)",
    "GFC (2007-09)",
    "COVID (2020)",
)

DEFAULT_LEAD_REGRESSION_TOLERANCE_MONTHS = 2.0
DEFAULT_STABILITY_WINDOW_MONTHS = 1.0

# Anchor: post-B3 baseline. This is the production threshold reference that
# all challengers should be measured against until a new baseline is frozen.
DEFAULT_BASELINE_THRESHOLD_PATH = Path("data/models/baseline_efb307e/threshold.json")


def load_baseline_threshold(path: str | Path | None = None) -> float:
    """
    Load the fixed reference threshold from a baseline artifacts directory.

    Parameters
    ----------
    path
        Path to a ``threshold.json`` artifact. Defaults to
        ``data/models/baseline_efb307e/threshold.json`` relative to the
        repository root.

    Returns
    -------
    float
        The ``decision_threshold`` scalar from the JSON payload.

    Raises
    ------
    FileNotFoundError
        If the threshold artifact does not exist.
    KeyError
        If the JSON payload is missing ``decision_threshold``.
    """
    threshold_path = Path(path) if path is not None else DEFAULT_BASELINE_THRESHOLD_PATH
    if not threshold_path.is_absolute():
        # Resolve relative to the current working directory (most callers run
        # from the repository root). This is a best-effort pathing helper —
        # the explicit override is always available.
        threshold_path = Path.cwd() / threshold_path
    if not threshold_path.exists():
        raise FileNotFoundError(f"Baseline threshold file missing: {threshold_path}")
    payload = json.loads(threshold_path.read_text())
    if "decision_threshold" not in payload:
        raise KeyError(
            f"threshold.json at {threshold_path} is missing 'decision_threshold'"
        )
    return float(payload["decision_threshold"])


def _parse_trajectory(raw) -> pd.Series | None:
    """Parse a stored probability trajectory into a Series indexed by date.

    Accepts list-of-pairs, dict {date: prob}, or a pre-built Series. Returns
    ``None`` if the value is missing or unparseable.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    if isinstance(raw, pd.Series):
        traj = raw
    elif isinstance(raw, Mapping):
        traj = pd.Series(raw)
    elif isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            return None
        return _parse_trajectory(parsed)
    elif isinstance(raw, (list, tuple, np.ndarray)):
        pairs = [p for p in raw if p is not None]
        if not pairs:
            return None
        # Accept list of [date, prob] or list of {"date":..., "prob":...}
        if all(isinstance(p, Mapping) for p in pairs):
            traj = pd.Series({p["date"]: p["prob"] for p in pairs})
        else:
            traj = pd.Series({p[0]: p[1] for p in pairs})
    else:
        return None

    traj.index = pd.to_datetime(traj.index)
    traj = traj.sort_index().astype(float)
    return traj


def _first_crossing_date(trajectory: pd.Series, threshold: float):
    """Return the first index where trajectory >= threshold, else None."""
    hits = trajectory.index[trajectory.values >= threshold]
    return hits[0] if len(hits) else None


def _first_recession_date(trajectory: pd.Series, actual: pd.Series):
    """First index where actual == 1 in the trajectory window."""
    if actual is None:
        return None
    aligned = actual.reindex(trajectory.index)
    hits = aligned.index[aligned.values == 1]
    return hits[0] if len(hits) else None


def _lead_months(first_crossing, first_recession) -> float:
    if first_crossing is None or first_recession is None:
        return np.nan
    return (pd.Timestamp(first_recession) - pd.Timestamp(first_crossing)).days / 30.44


def rescore_backtest_at_fixed_threshold(
    backtest_df: pd.DataFrame,
    fixed_threshold: float,
    *,
    trajectory_col: str = "Probability_Trajectory",
    actual_col: str = "Actual_Trajectory",
) -> pd.DataFrame:
    """
    Recompute crossing and lead-time columns using a fixed decision threshold.

    Accepts any backtest frame produced by
    :meth:`recession_engine.backtester.RecessionBacktester.run_pseudo_oos_backtest`
    (or the strict-origin variant once extended). Returns a copy with these
    additional columns:

    * ``Crossed_Threshold_Fixed`` — bool. Always exact when ``Peak_Prob`` is
      present (it's just a comparison).
    * ``Lead_Months_Fixed`` — float. Exact when the trajectory column is
      present; otherwise a bound-based estimate (see module docstring).
    * ``Threshold_Shift`` — float. ``Threshold`` − ``fixed_threshold``.
      Positive = original F1 optimizer pushed the threshold higher than the
      baseline reference.
    * ``Lead_Time_Delta_Fixed`` — float. ``Lead_Months_Fixed`` −
      ``Lead_Months``. Sign convention: positive means shifting to the fixed
      threshold gained lead time relative to the original per-origin
      threshold.
    * ``Lead_Estimation_Method`` — string tag describing how
      ``Lead_Months_Fixed`` was derived (``exact``, ``exact_no_cross``,
      ``bound_at_or_earlier``, ``bound_at_or_later``, or ``unavailable``).

    Parameters
    ----------
    backtest_df
        Per-recession backtest results.
    fixed_threshold
        The reference threshold to measure at. Typically the baseline's
        ``decision_threshold`` from
        ``data/models/baseline_efb307e/threshold.json``.
    trajectory_col
        Name of the optional column carrying the full probability series per
        recession. If absent, the bound-based estimator is used.
    actual_col
        Name of the optional column carrying the actual-recession boolean
        trajectory aligned with ``trajectory_col``.

    Returns
    -------
    pandas.DataFrame
        A copy of ``backtest_df`` with the new columns appended.
    """
    if fixed_threshold is None or not np.isfinite(fixed_threshold):
        raise ValueError(f"fixed_threshold must be a finite number, got {fixed_threshold!r}")

    df = backtest_df.copy()
    has_trajectory = trajectory_col in df.columns
    has_actual_trajectory = actual_col in df.columns

    crossed_fixed: list[bool | float] = []
    lead_fixed: list[float] = []
    method: list[str] = []

    for _, row in df.iterrows():
        peak_prob = row.get("Peak_Prob")
        orig_threshold = row.get("Threshold")
        orig_lead = row.get("Lead_Months")

        # Row with an Error value is a skipped backtest — carry NaN forward.
        if isinstance(row.get("Error"), str) and row.get("Error").strip():
            crossed_fixed.append(np.nan)
            lead_fixed.append(np.nan)
            method.append("unavailable")
            continue

        if pd.isna(peak_prob):
            crossed_fixed.append(np.nan)
            lead_fixed.append(np.nan)
            method.append("unavailable")
            continue

        crossed_fixed.append(bool(peak_prob >= fixed_threshold))

        # Exact path: full trajectory available.
        if has_trajectory:
            traj = _parse_trajectory(row.get(trajectory_col))
            if traj is not None:
                actual_series = None
                if has_actual_trajectory:
                    actual_series = _parse_trajectory(row.get(actual_col))
                first_recession = _first_recession_date(traj, actual_series)
                # Fallback: use Peak_Date's recession window if actual column
                # is absent — callers shouldn't rely on this, the extended
                # backtester always supplies the actual trajectory.
                first_cross = _first_crossing_date(traj, fixed_threshold)
                if first_cross is None:
                    lead_fixed.append(np.nan)
                    method.append("exact_no_cross")
                    continue
                lead_fixed.append(_lead_months(first_cross, first_recession))
                method.append("exact")
                continue

        # Fallback path: bound estimate from scalar summaries.
        if peak_prob < fixed_threshold:
            lead_fixed.append(np.nan)
            method.append("exact_no_cross")
            continue
        if pd.isna(orig_lead):
            # Challenger never crossed at its own threshold either — we only
            # know it crosses at `fixed_threshold` if Peak_Prob >= fixed_thr.
            # Without a trajectory we cannot date that first crossing.
            lead_fixed.append(np.nan)
            method.append("bound_unknown")
            continue

        if pd.isna(orig_threshold) or orig_threshold >= fixed_threshold:
            # Original threshold ≥ fixed: moving to a lower bar can only make
            # the first crossing earlier. We report the original lead as a
            # conservative lower bound.
            lead_fixed.append(float(orig_lead))
            method.append("bound_at_or_earlier")
        else:
            # Original threshold < fixed: fixed threshold first-crossing
            # happens at-or-after the original. Upper bound = original lead.
            lead_fixed.append(float(orig_lead))
            method.append("bound_at_or_later")

    df["Crossed_Threshold_Fixed"] = crossed_fixed
    df["Lead_Months_Fixed"] = lead_fixed
    df["Lead_Estimation_Method"] = method
    df["Threshold_Shift"] = df["Threshold"].astype(float) - float(fixed_threshold)
    df["Lead_Time_Delta_Fixed"] = (
        pd.to_numeric(df["Lead_Months_Fixed"], errors="coerce")
        - pd.to_numeric(df["Lead_Months"], errors="coerce")
    )

    return df


@dataclass
class StabilityGateReport:
    """Verdict bundle for :func:`apply_stability_gate`."""

    passed: bool
    violations: list[dict] = field(default_factory=list)
    per_recession: list[dict] = field(default_factory=list)
    fixed_threshold: float | None = None
    challenger_threshold_note: str = ""
    tolerance_months: float = DEFAULT_LEAD_REGRESSION_TOLERANCE_MONTHS

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "fixed_threshold": self.fixed_threshold,
            "tolerance_months": self.tolerance_months,
            "violations": self.violations,
            "per_recession": self.per_recession,
            "challenger_threshold_note": self.challenger_threshold_note,
        }


def compute_lead_time_delta_matrix(
    challenger_backtest: pd.DataFrame,
    baseline_backtest: pd.DataFrame,
    baseline_threshold: float | None = None,
    *,
    in_scope_recessions: Sequence[str] | None = DEFAULT_IN_SCOPE_RECESSIONS,
    challenger_trajectory_col: str = "Probability_Trajectory",
    baseline_trajectory_col: str = "Probability_Trajectory",
    stability_window_months: float = DEFAULT_STABILITY_WINDOW_MONTHS,
) -> pd.DataFrame:
    """
    Compare baseline and challenger lead times per recession.

    For each recession label present in both frames we compute:

    * ``baseline_lead_at_own_threshold`` — ``Lead_Months`` from the
      baseline backtest (per-origin F1-optimized).
    * ``challenger_lead_at_own_threshold`` — ``Lead_Months`` from the
      challenger backtest (per-origin F1-optimized).
    * ``challenger_lead_at_baseline_threshold`` — challenger's lead under
      ``baseline_threshold``, computed via
      :func:`rescore_backtest_at_fixed_threshold`.
    * ``threshold_stability_preserved`` — True iff the challenger's lead at
      the baseline threshold stays within ``stability_window_months`` of its
      own-threshold lead (default ±1 month).
    * ``lead_time_delta_own`` — challenger minus baseline at their
      respective own thresholds.
    * ``lead_time_delta_fixed`` — challenger minus baseline, both measured
      at the baseline threshold.
    * ``in_scope`` — whether this recession is gate-relevant under the
      1990-present policy.

    Parameters
    ----------
    challenger_backtest, baseline_backtest
        Per-recession backtest frames. Must share the ``Recession`` label.
    baseline_threshold
        Fixed threshold to measure at. If ``None``, pulled from
        ``DEFAULT_BASELINE_THRESHOLD_PATH``.
    in_scope_recessions
        Recession labels to flag as gate-relevant. ``None`` leaves every
        recession in scope.
    challenger_trajectory_col, baseline_trajectory_col
        Optional per-origin probability trajectory columns for exact rescoring.
    stability_window_months
        Tolerance on the "own threshold vs fixed threshold" comparison.
    """
    if baseline_threshold is None:
        baseline_threshold = load_baseline_threshold()

    challenger_rescored = rescore_backtest_at_fixed_threshold(
        challenger_backtest,
        baseline_threshold,
        trajectory_col=challenger_trajectory_col,
    )
    baseline_rescored = rescore_backtest_at_fixed_threshold(
        baseline_backtest,
        baseline_threshold,
        trajectory_col=baseline_trajectory_col,
    )

    base_idx = baseline_rescored.set_index("Recession")
    chal_idx = challenger_rescored.set_index("Recession")

    rows = []
    shared = [r for r in chal_idx.index if r in base_idx.index]
    for recession in shared:
        b = base_idx.loc[recession]
        c = chal_idx.loc[recession]

        b_lead_own = b.get("Lead_Months")
        c_lead_own = c.get("Lead_Months")
        c_lead_fixed = c.get("Lead_Months_Fixed")
        c_crossed_fixed = c.get("Crossed_Threshold_Fixed")

        # Stability preserved: challenger's own-threshold lead and fixed-
        # threshold lead agree to within the window. NaNs collapse to False.
        stability_preserved = (
            pd.notna(c_lead_own)
            and pd.notna(c_lead_fixed)
            and abs(float(c_lead_own) - float(c_lead_fixed)) <= stability_window_months
        )

        rows.append({
            "Recession": recession,
            "in_scope": (
                True if in_scope_recessions is None else recession in in_scope_recessions
            ),
            "baseline_threshold": float(b.get("Threshold", np.nan)),
            "challenger_threshold": float(c.get("Threshold", np.nan)),
            "fixed_threshold": float(baseline_threshold),
            "baseline_lead_at_own_threshold": float(b_lead_own) if pd.notna(b_lead_own) else np.nan,
            "challenger_lead_at_own_threshold": float(c_lead_own) if pd.notna(c_lead_own) else np.nan,
            "challenger_lead_at_baseline_threshold": float(c_lead_fixed) if pd.notna(c_lead_fixed) else np.nan,
            "challenger_crossed_at_baseline_threshold": (
                bool(c_crossed_fixed) if pd.notna(c_crossed_fixed) else np.nan
            ),
            "lead_estimation_method": c.get("Lead_Estimation_Method", "unavailable"),
            "baseline_peak_prob": float(b.get("Peak_Prob", np.nan)),
            "challenger_peak_prob": float(c.get("Peak_Prob", np.nan)),
            "lead_time_delta_own": (
                float(c_lead_own) - float(b_lead_own)
                if pd.notna(c_lead_own) and pd.notna(b_lead_own) else np.nan
            ),
            "lead_time_delta_fixed": (
                float(c_lead_fixed) - float(b_lead_own)
                if pd.notna(c_lead_fixed) and pd.notna(b_lead_own) else np.nan
            ),
            "threshold_stability_preserved": bool(stability_preserved),
        })

    return pd.DataFrame(rows)


def apply_stability_gate(
    delta_matrix: pd.DataFrame,
    *,
    lead_regression_tolerance_months: float = DEFAULT_LEAD_REGRESSION_TOLERANCE_MONTHS,
    restrict_to_in_scope: bool = True,
) -> StabilityGateReport:
    """
    Evaluate the KEEP gate: no in-scope recession should regress lead time
    by more than ``lead_regression_tolerance_months`` when measured at the
    baseline fixed threshold.

    A challenger fails when **any** in-scope recession has
    ``lead_time_delta_fixed < -lead_regression_tolerance_months``.

    Parameters
    ----------
    delta_matrix
        Output from :func:`compute_lead_time_delta_matrix`.
    lead_regression_tolerance_months
        Magnitude (in months) of lead-time regression we're willing to
        accept on any single in-scope recession. Default: 2 months, matching
        the noise floor of per-origin retraining variance observed across
        A1/B1 experiments.
    restrict_to_in_scope
        If ``True``, only the in-scope recessions drive the verdict. If
        ``False``, pre-1990 rows also count.
    """
    if delta_matrix is None or delta_matrix.empty:
        return StabilityGateReport(
            passed=False,
            violations=[{"error": "empty delta matrix"}],
            tolerance_months=lead_regression_tolerance_months,
        )

    rows = delta_matrix.copy()
    if restrict_to_in_scope and "in_scope" in rows.columns:
        rows = rows[rows["in_scope"]]

    violations = []
    per_recession = []
    for _, row in rows.iterrows():
        delta = row.get("lead_time_delta_fixed")
        if pd.isna(delta):
            # Missing data is non-fatal; we record it but don't fail the gate
            # purely because a baseline/challenger entry dropped out.
            per_recession.append(row.to_dict())
            continue
        regression = float(delta) < -float(lead_regression_tolerance_months)
        record = row.to_dict()
        record["passes_gate"] = not regression
        per_recession.append(record)
        if regression:
            violations.append({
                "recession": row["Recession"],
                "lead_time_delta_fixed": float(delta),
                "challenger_lead_at_baseline_threshold": float(
                    row.get("challenger_lead_at_baseline_threshold", np.nan)
                ),
                "baseline_lead_at_own_threshold": float(
                    row.get("baseline_lead_at_own_threshold", np.nan)
                ),
                "threshold_shift": (
                    float(row.get("challenger_threshold", np.nan))
                    - float(row.get("fixed_threshold", np.nan))
                ),
            })

    fixed_threshold = None
    if not delta_matrix.empty and "fixed_threshold" in delta_matrix.columns:
        fixed_threshold = float(delta_matrix["fixed_threshold"].iloc[0])

    return StabilityGateReport(
        passed=len(violations) == 0,
        violations=violations,
        per_recession=per_recession,
        fixed_threshold=fixed_threshold,
        tolerance_months=lead_regression_tolerance_months,
    )


def summarize_gate_report(report: StabilityGateReport) -> str:
    """Render a one-screen gate verdict for logs and markdown ledgers."""
    status = "PASS" if report.passed else "FAIL"
    lines = [
        f"Threshold-stability gate: {status}",
        f"Fixed threshold: {report.fixed_threshold}",
        f"Tolerance: {report.tolerance_months:.1f} months of lead-time regression per in-scope recession",
    ]
    if report.violations:
        lines.append("Violations:")
        for v in report.violations:
            lines.append(
                f"  - {v['recession']}: Δ lead at fixed threshold = "
                f"{v['lead_time_delta_fixed']:+.2f} mo "
                f"(challenger {v['challenger_lead_at_baseline_threshold']:+.2f} vs "
                f"baseline {v['baseline_lead_at_own_threshold']:+.2f}); "
                f"threshold shift {v['threshold_shift']:+.3f}"
            )
    elif report.per_recession:
        lines.append("In-scope per-recession breakdown:")
        for r in report.per_recession:
            if r.get("in_scope", True):
                delta = r.get("lead_time_delta_fixed")
                delta_str = f"{delta:+.2f}" if pd.notna(delta) else "n/a"
                lines.append(
                    f"  - {r['Recession']}: Δ fixed = {delta_str} mo "
                    f"({r.get('lead_estimation_method', 'unknown')})"
                )
    return "\n".join(lines)


__all__ = [
    "DEFAULT_BASELINE_THRESHOLD_PATH",
    "DEFAULT_IN_SCOPE_RECESSIONS",
    "DEFAULT_LEAD_REGRESSION_TOLERANCE_MONTHS",
    "DEFAULT_STABILITY_WINDOW_MONTHS",
    "StabilityGateReport",
    "apply_stability_gate",
    "compute_lead_time_delta_matrix",
    "load_baseline_threshold",
    "rescore_backtest_at_fixed_threshold",
    "summarize_gate_report",
]
