# C5 retroactive threshold-stability analysis

Generated: 2026-04-24T15:20:26.475983Z  |  Git SHA: `51a7b6a948317da8ebc16f484e26cb67196bd894`  |  Fixed threshold: **0.41**

Reference baseline: `data/models/baseline_efb307e/threshold.json`. All challenger lead times below are reported at the baseline fixed threshold using the bound-based estimator in `recession_engine.threshold_stability` (historical CSVs don't preserve per-origin probability trajectories).

Lead-estimation method legend:

| method | meaning |
|---|---|
| `exact` | trajectory available, first-crossing date computed directly |
| `exact_no_cross` | Peak_Prob < fixed_threshold, no crossing happens |
| `bound_at_or_earlier` | original threshold ≥ fixed, so fixed-threshold crossing is at-or-earlier (original lead is a lower bound) |
| `bound_at_or_later` | original threshold < fixed, so fixed-threshold crossing is at-or-later (original lead is an upper bound) |
| `unavailable` | backtest row errored or missing Peak_Prob |

## Per-experiment results

### baseline_efb307e — Post-B3 baseline (production)

Verdict under C5 gate: **baseline_reference**

| Recession | Peak | Own thr | Own lead | Fixed lead | Method | Δ vs own |
|---|---:|---:|---:|---:|---|---:|
| Oil Crisis (1973-75) | n/a | n/a | n/a | n/a | `unavailable` | n/a |
| Volcker I (1980) | 0.357 | 0.160 | +6.04 | n/a | `exact_no_cross` | n/a |
| Volcker II (1981-82) | 0.499 | 0.410 | -6.04 | -6.04 | `bound_at_or_earlier` | +0.00 |
| S&L Crisis (1990-91) | 0.727 | 0.310 | +5.95 | +5.95 | `bound_at_or_later` | +0.00 |
| Dot-com (2001) | 0.230 | 0.280 | n/a | n/a | `exact_no_cross` | n/a |
| GFC (2007-09) | 0.671 | 0.110 | +6.96 | +6.96 | `bound_at_or_later` | +0.00 |
| COVID (2020) | 0.661 | 0.350 | -6.01 | -6.01 | `bound_at_or_later` | +0.00 |

### B3 — B3 credit-supply block (positive control)

_Merged under override — we expect GFC lead to remain strong at the fixed threshold._

Verdict under C5 gate: **KEEP-under-C5**

| Recession | Peak | Own thr | Own lead | Fixed lead | Method | Δ vs own |
|---|---:|---:|---:|---:|---|---:|
| Oil Crisis (1973-75) | n/a | n/a | n/a | n/a | `unavailable` | n/a |
| Volcker I (1980) | 0.357 | 0.160 | +6.04 | n/a | `exact_no_cross` | n/a |
| Volcker II (1981-82) | 0.499 | 0.410 | -6.04 | -6.04 | `bound_at_or_earlier` | +0.00 |
| S&L Crisis (1990-91) | 0.727 | 0.310 | +5.95 | +5.95 | `bound_at_or_later` | +0.00 |
| Dot-com (2001) | 0.230 | 0.280 | n/a | n/a | `exact_no_cross` | n/a |
| GFC (2007-09) | 0.671 | 0.110 | +6.96 | +6.96 | `bound_at_or_later` | +0.00 |
| COVID (2020) | 0.661 | 0.350 | -6.01 | -6.01 | `bound_at_or_later` | +0.00 |

Gate report (in-scope only):

- Passed: **True**
- Tolerance: 2.0 months

### C3 — C3 regime-conditional ensemble weights

_DISCARD under per-variant F1 threshold — did re-optimization destroy the lead?_

Verdict under C5 gate: **DISCARD-under-C5**

| Recession | Peak | Own thr | Own lead | Fixed lead | Method | Δ vs own |
|---|---:|---:|---:|---:|---|---:|
| Oil Crisis (1973-75) | n/a | n/a | n/a | n/a | `unavailable` | n/a |
| Volcker I (1980) | 0.366 | 0.190 | +4.04 | n/a | `exact_no_cross` | n/a |
| Volcker II (1981-82) | 0.516 | 0.420 | -6.04 | -6.04 | `bound_at_or_earlier` | +0.00 |
| S&L Crisis (1990-91) | 0.723 | 0.130 | +6.96 | +6.96 | `bound_at_or_later` | +0.00 |
| Dot-com (2001) | 0.247 | 0.270 | n/a | n/a | `exact_no_cross` | n/a |
| GFC (2007-09) | 0.676 | 0.230 | +1.02 | +1.02 | `bound_at_or_later` | +0.00 |
| COVID (2020) | 0.696 | 0.590 | -6.01 | -6.01 | `bound_at_or_earlier` | +0.00 |

Gate report (in-scope only):

- Passed: **False**
- Tolerance: 2.0 months
- Violations:
  - `GFC (2007-09)` — Δ lead at fixed = -5.95mo (challenger 1.0183968462549278 vs baseline 6.964520367936925)

### B5 — B5 equity-valuation block (hybrid variant)

_DISCARD — does Dot-com lift hold at the fixed threshold? Does GFC collapse remain?_

Verdict under C5 gate: **DISCARD-under-C5**

| Recession | Peak | Own thr | Own lead | Fixed lead | Method | Δ vs own |
|---|---:|---:|---:|---:|---|---:|
| Oil Crisis (1973-75) | n/a | n/a | n/a | n/a | `unavailable` | n/a |
| Volcker I (1980) | 0.400 | 0.170 | +6.04 | n/a | `exact_no_cross` | n/a |
| Volcker II (1981-82) | 0.475 | 0.320 | -6.04 | -6.04 | `bound_at_or_later` | +0.00 |
| S&L Crisis (1990-91) | 0.826 | 0.390 | +4.96 | +4.96 | `bound_at_or_later` | +0.00 |
| Dot-com (2001) | 0.248 | 0.390 | n/a | n/a | `exact_no_cross` | n/a |
| GFC (2007-09) | 0.830 | 0.470 | +0.00 | +0.00 | `bound_at_or_earlier` | +0.00 |
| COVID (2020) | 0.800 | 0.430 | -6.01 | -6.01 | `bound_at_or_earlier` | +0.00 |

Gate report (in-scope only):

- Passed: **False**
- Tolerance: 2.0 months
- Violations:
  - `GFC (2007-09)` — Δ lead at fixed = -6.96mo (challenger 0.0 vs baseline 6.964520367936925)

## Methodology notes
- Historical backtest CSVs predate the C5 trajectory-preserving backtester, so fixed-threshold lead times are bound-based estimates (see `recession_engine.threshold_stability` docstring).
- Exact fixed-threshold lead times will be emitted directly by `RecessionBacktester.run_pseudo_oos_backtest` on all future runs (additional `Lead_Months_Fixed` column).
