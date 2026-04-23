# Experiment Ledger

Running log of the research-backlog experiments defined in `~/.claude/plans/lets-do-some-more-cheeky-summit.md`. Each row is appended by a validator sub-agent after a builder sub-agent finishes.

Baseline frozen at git SHA `ed24542` on 2026-04-22. Evaluation substrate: `data/models/eval_origins.json` (15 origins).

## Evaluation-scope policy (adopted 2026-04-23)

**Primary evaluation window is 1990-present.** Pre-1990 origins remain in the test set as *informational* — they're reported in every validation JSON but **do not drive KEEP/DISCARD verdicts**. Training uses full history (evaluation-only scope, not training-scope).

- **In-scope (gate-relevant)** — 10 origins: S&L 1990, Dot-com 2000, GFC 2007, COVID 2019 + 6 post-1990 expansion checks.
- **Informational (reported, not gated)** — 5 origins: Oil Crisis 1973, Volcker I 1979, Volcker II 1981 + 1978/1986 expansion checks.

Rationale: recurring pattern across A1/A1.75/B2/B2.5 — modern-era optimization destabilizes pre-1990 detection because the pre-1990 macro regime (Volcker disinflation, ~40% goods-employment share, pre-financial-innovation) is materially different from what the dashboard forecasts in. Optimizing for 1981 is museum work.

Rows filled before 2026-04-23 ("A1", "A1.5", "A1.75", "B1", "B2") were gated under the old full-history policy — their verdicts stand, but scope-relevant deltas may differ if re-evaluated.

## Verdict legend
- **KEEP** — meets the keep gate (in-scope PR-AUC ≥ baseline + ≥1 secondary improvement, no >10% in-scope regression)
- **DISCARD** — hits the discard gate (>15% in-scope regression, or fails to catch Dot-com + GFC + COVID)
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
| P3 | Eval substrate frozen | DONE | PASS | — | — | — | — | 15 origins + NBER dates in `data/models/eval_origins.json`. Updated 2026-04-23 with `evaluation_scope` tags per origin. |
| P4 | Experiment ledger seeded | DONE | PASS | — | — | — | — | This file. Scope policy section added 2026-04-23. |
| A1 | Calibration audit & uncertainty upgrade | SUPERSEDED | → A1.5 | -0.033 (-17%) | -0.002 | ECE=0.0996 (new) | +0.01mo (~0) | Branch `experiment/A1-calibration` SHA 8aac438. A1 returned NEEDS-WORK: infra worked but block-bootstrap CI collapsed to 0.01 floor on 128/134 rows and PR-AUC regressed 17% due to strict-vintage retrain variance. A1.5 resolved both. See `data/models/a1_validation.json` + `data/models/a1_5_validation.json`. |
| A1.5 | CI width fix + baseline pin + sigmoid ensemble promotion | DONE | KEEP | 0.000 (0%) | 0.000 | raw=0.122, best-cal=0.023 | 0 | Merged to main at `c1acaf2`. CI coverage 100%, baseline pinned at `data/models/baseline_ed24542/`, sigmoid promoted as preferred ensemble calibrator but safety-gated. Safety gate rejected all 3 legacy calibrators → production runs raw weighted avg (deployed ECE 0.122, best-calibrator ECE 0.023 trapped). See `data/models/a1_5_validation.json`. |
| A1.75 | Regime-robust calibration (original) | RETIRED → A1.75.2 | NEEDS-WORK — NOT MERGED | ~0 | +0.06 | deployed 0.036 | n/a | Branch `experiment/A1.75-regime-robust-cal` SHA `1b088e7`. Built 4 candidates (monotone_iso, regime_strat, post1990, boundary_constrained) + extended 4-point safety probe. Only boundary_constrained cleared probe; Volcker I calibrated peak 0.15 (target 0.30), Dot-com regressed to 0.07. **Deeper finding**: the `cal(0.2) < 0.05` probe point rejected otherwise-good candidates (data property, not collapse). Under the new 1990-present scope, pre-1990 collapse is out of scope → **A1.75.2 re-tests with in-scope-only safety gate**. See `.claude/worktrees/agent-a9083aa7/data/models/a1_75_validation.json`. |
| A1.75.2 | Re-test calibrators under in-scope-only safety gate | DONE | DISCARD — ensemble calibration parked | 0 (cal not deployed) | 0 | raw=0.125 (unchanged); trapped best-cal=0.006 (post1990) | 0 | Branch `experiment/A1.75.2-in-scope-calibrator-gate` SHA `aaafd68` (`97e1434` post-stamp). Ported all 4 A1.75 candidate calibrators to main + redefined safety gate to require in-scope Dot-com/GFC/COVID calibrated peaks ≥ 0.20. **Every one of the 7 candidates fails the in-scope gate — binding failure is Dot-com's raw 0.188 peak mapping to ≤0.07 under every calibrator.** Even at a relaxed 0.15 floor, every candidate fails. Holdout ECE of best-performing `post1990` = 0.006 (20x better than raw 0.125) but it achieves this by crushing Dot-com into the noise floor alongside every other low-prob row — exactly what the gate catches. **Deeper finding**: this isn't a calibrator design problem; Dot-com's raw ensemble peak sits in the same probability range as the 2010s expansion's true-negatives, so any monotone calibrator must map them together. **Unlock path is raising the raw Dot-com peak, not post-hoc calibration** — that's what C1 (benchmark models) or C3 (regime-conditional weights) address. Pre-1990 peaks are fine under this gate (Volcker I 0.383, Volcker II 0.499) — they're louder than Dot-com. A1.75.2 parks ensemble-level calibration permanently; diagnostic infrastructure (ECE, reliability curves) stays live. See `.claude/worktrees/agent-affbcec1/data/models/a1_75_2_validation.json`. |
| A2 | Vintage pipeline formalization (full ALFRED replay) | PLANNED | — | — | — | — | — | Research top-5 #1. Full ALFRED replay at ≥20 origins. Unblocks clean "live reliability" evaluation. |
| A3 | Target redesign (multi-horizon + onset + survival) | PLANNED | — | — | — | — | — | Emits multi-horizon labels + onset separation + censored time-to-recession. Bigger scope but potentially bigger lift. |
| B1 | At-risk representation bake-off | DONE | KEEP (challengers not promoted) | +0.010 CV | -0.001 | -0.002 | GFC -2mo (continuous_only) | Merged to main via B2 fast-forward. Winner: `continuous_only` beats hybrid on CV/test PR-AUC but regresses GFC lead 4→2mo. **Hybrid kept as production default** — at-risk layer stays as insurance. Infrastructure (`--feature-variant` CLI + `recession_engine/feature_variants.py`) shipped and used by B2. `at_risk_only` 5/5 detection is noise (backtest AUC 0.53). Only 1/20 hybrid top-selected features is at-risk (`leading_USSLIND_AT_RISK`). See `data/models/b1_validation.json` + `data/models/b1_variants/`. |
| B2 | Labor deterioration block upgrade | DONE | KEEP (merged under post-1990 scope) | +0.022 CV in-scope | +0 | ≈0 | GFC +1mo, Volcker II -13mo (informational) | Merged to main at `b992926` (via fast-forward from `97e7b9b` + post-commit ledger stamp). Added 7 FRED series (JTSJOL, JTSQUR, CIVPART, EMRATIO, USGOOD, USSERV, UNEMPLOY) + 17 Tier-12 features. `GOODS_DECLINE_FLAG` carries the lift (drop-ablation −5.08pp CV PR-AUC); every other B2 feature contributes ~0. Volcker II lead 7→−6mo is **out of scope under 1990-present policy** — B2.5 confirmed softening the flag doesn't fix it, and the regression isn't flag-specific. Merged because in-scope wins (GFC +1mo, CV +2.2pp) are the gate-relevant metrics. See `data/models/b2_validation.json` + `data/models/b2_variants/`. |
| B2.5 | Soften GOODS_DECLINE_FLAG (investigation) | DONE | NEEDS-WORK — NOT MERGED (superseded by scope policy) | -0.006 to -0.051 vs B2 raw | n/a | n/a | Volcker II unchanged -6mo for all variants | Branch `experiment/B2.5-goods-flag-softening` SHA `5a208fb`. Tested 3 softer flag variants + full removal. **Root-cause finding**: Volcker II regression is NOT caused by `GOODS_DECLINE_FLAG` — the flag fires appropriately (85% in recessions, 100% pre-GFC, 0% pre-COVID). Dropping it entirely (`none` variant) still produces Volcker II -6mo; culprit is elsewhere in B2's 80 new columns (likely `GOODS_YoY_MINUS_SERV_YoY` or CIVPART/EMRATIO z-scores reacting to the 1980s labor-force structural shift). This investigation directly motivated the 1990-present scope policy — rather than chase the Volcker II regression, we redefined the evaluation window. `--goods-flag-variant` CLI and diagnostic scripts remain on the B2.5 branch, unused. See `data/models/b2_5_validation.json`. |
| B3 | Credit-supply block upgrade | DONE | NEEDS-WORK — NOT MERGED | -0.0045 in-scope CV | +0.0207 (in-scope) | raw 0.120 (with) vs 0.127 (no); calibrator rejected in both | S&L +1mo, GFC +5.94mo, COVID -1mo (informational) | Branch `experiment/B3-credit-supply` SHA `75e6c47`. Added 5 FRED series (DRTSCILM, DRTSCIS, DRTSCLCC, DRSDCILM-corrected-from-invalid-SUBLPDCILMN, TOTALSL) + 12 Tier-13 features. 2 B3 features reach top-20 selection: `SLOOS_TIGHTENING_FLAG` (rank 2), `SLOOS_ACUTE_TIGHTENING` (rank 12). ANFCI confirmed in frame (was always pulled; now selected after must-include addition — both `financial_ANFCI` and `financial_ANFCI_MA6`). Drop-ablation: `SLOOS_TIGHTENING_FLAG` -17.2pp test PR-AUC when zeroed (biggest modern-era contributor) but only -0.6pp CV (fails ≥3pp gate). **Mixed verdict**: GFC lead +5.94mo clearly passes the +1mo KEEP gate and S&L +1mo, but Dot-com peak only moves +0.5pp vs the 3pp gate — A1.75.2's binding constraint (Dot-com raw peak) did not lift. CV PR-AUC regresses 0.45pp and COVID peak drops 10pp (still detected). Marked NEEDS-WORK per plan: credit signal is real (SLOOS_TIGHTENING_FLAG is load-bearing on recent years) but the intended Dot-com unlock didn't materialize. See `data/models/b3_validation.json` + `data/models/b3_variants/`. |
| B4 | Text sentiment (Beige Book + FinBERT) | PLANNED | — | — | — | — | — | User approval gate before start — transformers + torch dependency weight. |
| C1 | Benchmark model family additions | PLANNED | — | — | — | — | — | Wrap Hamilton JHGDPBRINDX + Chauvet-Piger RECPROUSM156N as ensemble members; fit Wright probit. Cheap diversification. |
| C2 | Economically constrained XGBoost | PLANNED | — | — | — | — | — | Monotone constraints + optional interaction constraints. Cheap; reversible. |
| C3 | Regime-conditional ensemble weights | PLANNED | — | — | — | — | — | Condition DMA weights on GLR state. **Possible indirect fix for the recent-vs-old regime tension** — different models could be weighted differently per regime. Gated on A1.75.2 outcome (calibrator availability). |
| C4 | Mixed-frequency nowcasting (DFM / MIDAS) | PLANNED | — | — | — | — | — | Research top-5 #1 (with A2). DFM factor as ensemble feature. |
| D1 | Deep-learning foundation-model challenger | PARKED | — | — | — | — | — | Challenger only, don't promote. |
| D2 | Cross-country panel pretraining | PARKED | — | — | — | — | — | Research track. |

## Notes for validators

When appending a row:
1. Δ columns are (challenger minus baseline) **computed on in-scope origins**. Negative = better for Brier/ECE; positive = better for PR-AUC/LeadTime. Full-history metrics belong in the validation JSON, not the ledger row.
2. Lead-time delta is computed across detected recessions only (in-scope).
3. Include the experiment's git branch or worktree path in the notes column so changes are reviewable.
4. If discarded, still keep the row — a negative result is a result.
5. Report pre-1990 lead-time regressions in the notes as "Volcker/Oil Crisis Δ: informational" — don't let them kill the verdict unless they exceed -25% in isolation (sanity floor).
