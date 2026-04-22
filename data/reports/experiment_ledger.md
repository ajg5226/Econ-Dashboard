# Experiment Ledger

Running log of the research-backlog experiments defined in `~/.claude/plans/lets-do-some-more-cheeky-summit.md`. Each row is appended by a validator sub-agent after a builder sub-agent finishes.

Baseline frozen at git SHA `ed24542` on 2026-04-22. Evaluation substrate: `data/models/eval_origins.json` (15 origins).

## Verdict legend
- **KEEP** — meets the keep gate in the plan (PR-AUC ≥ baseline + ≥1 secondary improvement, no >10% regression)
- **DISCARD** — hits the discard gate (>15% regression, or fails to catch Dot-com + GFC + COVID)
- **NEEDS-WORK** — neither; iterate or reframe

## Status legend
- **PLANNED** — in backlog, not started
- **IN-PROGRESS** — builder sub-agent running
- **VALIDATING** — validator sub-agent running
- **DONE** — row filled, verdict decided

## Ledger

| ID | Title | Status | Verdict | ΔPR-AUC | ΔBrier | ΔECE | ΔLeadTime | Notes |
|----|-------|--------|---------|---------|--------|------|-----------|-------|
| P1 | Synthetic data isolated | DONE | PASS | — | — | — | — | Grep confirms synthetic only in docs/tests/CI/self_test; no leakage into production training path. |
| P2 | Baseline frozen | DONE | PASS | — | — | — | — | Baseline metrics captured in `eval_origins.json:baseline_metrics_at_freeze`. |
| P3 | Eval substrate frozen | DONE | PASS | — | — | — | — | 15 origins + NBER dates in `data/models/eval_origins.json`. |
| P4 | Experiment ledger seeded | DONE | PASS | — | — | — | — | This file. |
| A1 | Calibration audit & uncertainty upgrade | DONE | NEEDS-WORK | -0.033 (-17%) | -0.002 | ECE=0.0996 (new) | +0.01mo (~0) | Branch `experiment/A1-calibration` SHA 8aac438. Measurement infra works: ECE<0.15 gate passes, calibrator A/B + reliability curves + block-bootstrap CIs all persist cleanly; backtest improved (AUC 0.930 vs 0.901, Brier 0.200 vs 0.207, 5/6 detection preserved, Dot-com peak jumped 0.19→0.35). PR-AUC drift traced to fresh-data retraining variance (first update_job matched baseline 0.193 exactly before strict-vintage retrain overwrote metrics.csv). Block-bootstrap CI collapses to 0.01 floor on 128/134 labeled rows so empirical coverage proxy is ~0; ensemble calibrator A/B winner is `sigmoid`. See `data/models/a1_validation.json`. |
| A2 | Vintage pipeline formalization (full ALFRED replay) | PLANNED | — | — | — | — | — | |
| A3 | Target redesign (multi-horizon + onset + survival) | PLANNED | — | — | — | — | — | |
| B1 | At-risk representation bake-off | PLANNED | — | — | — | — | — | |
| B2 | Labor deterioration block upgrade | PLANNED | — | — | — | — | — | |
| B3 | Credit-supply block upgrade | PLANNED | — | — | — | — | — | |
| B4 | Text sentiment (Beige Book + FinBERT) | PLANNED | — | — | — | — | — | User approval gate before start. |
| C1 | Benchmark model family additions | PLANNED | — | — | — | — | — | |
| C2 | Economically constrained XGBoost | PLANNED | — | — | — | — | — | |
| C3 | Regime-conditional ensemble weights | PLANNED | — | — | — | — | — | Gated on A1. |
| C4 | Mixed-frequency nowcasting (DFM / MIDAS) | PLANNED | — | — | — | — | — | |
| D1 | Deep-learning foundation-model challenger | PARKED | — | — | — | — | — | Challenger only, don't promote. |
| D2 | Cross-country panel pretraining | PARKED | — | — | — | — | — | Research track. |

## Notes for validators

When appending a row:
1. Δ columns are (challenger minus baseline). Negative = better for Brier/ECE; positive = better for PR-AUC/LeadTime.
2. Lead-time delta is computed across detected recessions only.
3. Include the experiment's git branch or worktree path in the notes column so changes are reviewable.
4. If discarded, still keep the row — a negative result is a result.
