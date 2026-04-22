# ATLAS IP Transfer Summary

## Purpose

This document summarizes the core intellectual property in the Econ Dashboard recession stack so it can be ported into the ATLAS investment engine. The useful IP is not the Streamlit app itself. It is the econometric workflow: indicator selection, literature-backed feature engineering, real-time validation discipline, probability calibration, decision-threshold design, and the operational fixes that made the pipeline robust enough to run repeatedly.

## Stated Objective

The project objective is to estimate the probability of a U.S. recession within the next 6 months using public macro and market data, then convert that probability into an interpretable risk signal suitable for portfolio, risk-management, and monitoring workflows.

For ATLAS, the direct translation is:

- Build a macro regime engine that turns public economic time series into calibrated recession/regime probabilities.
- Use those probabilities as upstream state variables for allocation, risk throttling, exposure adjustment, and defensive rotation logic.
- Preserve real-time integrity so signals are useful in live decision-making, not just in revised-history backtests.

## What Actually Gives The Stack "Edge"

The edge comes from combining five things rather than from any single model.

### 1. Literature-backed feature design rather than generic macro transforms

The stack does not stop at raw levels and percent changes. It embeds specific recession mechanisms from the literature:

- Yield-curve inversion depth, duration, momentum, and deviation from rolling norms.
- Near-term forward spread dynamics.
- Monetary stance via fed funds interactions with term spreads.
- Credit stress via Baa spreads, TED spread, NFCI, and an excess-bond-premium proxy.
- Labor deterioration via both Sahm-style and SOS-style unemployment triggers.
- "At-risk" percentile transforms that convert many indicators into weak-state flags.
- Housing confirmation via house-price decline logic.
- Residential investment and consumer durables as rate-sensitive early-warning sectors.
- Sectoral divergence features that detect capex strength coexisting with weak employment.

This is the core transferable IP: the system encodes recession causality as a set of structured state variables instead of relying on a black-box model to discover everything from raw data.

### 2. Ensemble design optimized for rare-event usefulness

The stack uses multiple model types because recession prediction is a rare-event problem with unstable relationships:

- L1 logistic/probit-style model for sparse, explainable macro signals.
- Random forest for nonlinear interactions.
- XGBoost for higher-accuracy supervised learning.
- Optional Markov-switching and LSTM components.

The ensemble is not a naive average. It uses:

- Time-series cross-validation.
- Inverse-Brier weighting.
- Dynamic model averaging with exponential forgetting.
- Gating of materially weaker models before combining.
- Weight caps and floors to prevent one model from dominating or collapsing the ensemble.

The practical insight is that ensemble governance matters at least as much as model selection.

### 3. Probability calibration instead of raw classifier scores

A major source of edge is that the system treats raw model scores as untrustworthy until calibrated:

- Base models are calibrated with isotonic regression.
- Ensemble evaluation emphasizes Brier score and log loss, not only AUC.
- Threshold optimization is done on deployed probability outputs, not on arbitrary raw scores.

For ATLAS this matters because regime probabilities will feed downstream sizing and risk logic. Miscalibrated probabilities are worse than slightly less accurate but well-calibrated ones.

### 4. Real-time validation discipline

This is one of the most important lessons in the repo.

The codebase explicitly addresses the gap between revised-history performance and live usability:

- Publication lags are simulated per series.
- Forecast-origin training labels are cut off at `origin - horizon` to avoid label leakage.
- Strict real-time origin testing is used to compare candidate configurations.
- ALFRED vintage checks are used to quantify revised-vs-vintage drift.

This is likely the most valuable IP for ATLAS. Many econometric systems look good on revised data and fail in production because they ignore release lags, revisions, and unknowable future labels.

### 5. Signal design around actionable thresholds

The system does not default to a `0.50` classification threshold. It searches thresholds from `0.10` to `0.60` and selects using a precision-aware F1 policy with tie-breaks on precision, recall, specificity, and then slightly lower thresholds for earlier warning.

That is useful for ATLAS because macro signals should be tuned for action quality:

- Too high a threshold delays defensiveness.
- Too low a threshold creates whipsaw and over-hedging.
- The correct threshold depends on the downstream portfolio cost function, not classification convention.

## Inputs

The actual code uses a curated public-data macro set, not just a generic "45+ indicators" bucket. The live stack pulls from FRED and related public series grouped into:

- Leading: yield spreads, treasury bills, building permits, housing starts, claims, sentiment, new orders, residential fixed investment, nonresidential fixed investment, consumer durables.
- Coincident: payrolls, unemployment, industrial production, income, retail sales, manufacturing sales, insured unemployment.
- Lagging: unemployment duration, CPI, inventory/sales.
- Monetary/credit: fed funds, Baa-Treasury spread, TED spread.
- Financial conditions: NFCI, adjusted NFCI, high-yield OAS, corporate OAS.
- Housing/term-premium: Case-Shiller, Kim-Wright term premium.
- Reference series for benchmarking: NY Fed and Hamilton-style recession references.
- NBER recession indicator as the target source.

The target is a forward recession label:

- `RECESSION_FORWARD_6M = max(RECESSION over next 6 months)`

That target construction is directly portable into ATLAS for regime-state forecasting.

## Process

The core process is:

1. Acquire monthly macro and market series.
2. Resample to month-end and interpolate quarterly BEA series to monthly.
3. Engineer a large feature set:
   - Standard transforms: MoM, 3M, 6M, YoY, moving averages, rolling volatility.
   - Structural transforms: inversion flags, inversion duration, momentum, z-scores, interaction terms.
   - Binary recession-state transforms: at-risk flags, Sahm/SOS triggers, confirmation composites.
4. Create the forward target.
5. Train on an expanding time-series basis.
6. Run time-series cross-validation.
7. Weight and gate ensemble members using calibration-aware metrics.
8. Calibrate probabilities.
9. Optimize the decision threshold.
10. Evaluate on pseudo-OOS and strict real-time origin tests.
11. Save artifacts for monitoring: metrics, threshold sweep, rolling metrics, backtests, vintage summaries, model files, and probability outputs.

## Outputs

The stack produces outputs at three levels.

### 1. Model outputs

- Calibrated recession probabilities by model and ensemble.
- Decision threshold and threshold sweep diagnostics.
- Ensemble weights and active-model set.

### 2. Validation outputs

- Pseudo out-of-sample recession backtests.
- Strict real-time origin test results.
- ALFRED vintage gap diagnostics.
- Rolling metrics.

### 3. Operational outputs

- Dashboard-ready CSVs and charts.
- Executive summary / report text.
- Serialized models and feature lists.

For ATLAS, the essential output is a monthly calibrated macro-regime probability series plus supporting diagnostics explaining whether the signal is trustworthy.

## What The Evidence Says

The documentation contains both promotional metrics and stricter validation artifacts. For transfer purposes, the stricter numbers are more credible.

### Headline/demo claims in top-level docs

- README / executive docs cite out-of-sample AUC around `0.994` and near-perfect accuracy.
- Those docs also discuss synthetic-data workflows and aspirational business framing.

### More useful validation artifacts saved in `data/models`

- Pseudo-OOS backtest summary:
  - Recessions tested: `6`
  - Mean AUC: `0.927`
  - Mean Brier: `0.1974`
  - Recessions detected: `5/6`
  - Mean lead time: `3 months`
- Strict vintage candidate search:
  - Best candidate: `conservative_40`
  - PR-AUC: `0.692`
  - Brier: `0.1938`
  - F1: `0.667`
  - AUC: `0.786`
- ALFRED vintage audit:
  - Mean absolute revised-vintage gap: `12.52%`
  - Mean signed gap: `4.82%`
  - Max absolute gap: `20.27%`

The lesson is clear: the system has real value, but the real edge is the validation discipline that narrows the gap between idealized historical performance and live decision quality.

## Lessons Learned

These are the main lessons that should move with the IP into ATLAS.

### 1. Do not trust revised-history performance without real-time simulation

The repo added publication-lag simulation, vintage checks, and strict origin tests because standard train/test evaluation overstated live performance.

### 2. Calibration matters as much as discrimination

AUC alone is not enough for portfolio decisions. Brier score, log loss, and calibration-aware ensembling materially improve usefulness.

### 3. Rare-event forecasting needs precision-aware thresholding

Using a default `0.50` threshold is lazy. Recession signals should be optimized around the warning tradeoff that matters operationally.

### 4. Curated macro structure beats indiscriminate feature expansion

The strongest features are not just more transforms. They are economically motivated transforms tied to policy stance, yield-curve regime, credit stress, labor deterioration, housing confirmation, and rate-sensitive sectors.

### 5. Simpler, more conservative model configurations can win live-style tests

The saved search results favor `conservative_40`, not the richest feature set. That implies stability and reduced overfitting beat maximal complexity in stricter validation.

### 6. Robustness engineering is part of the IP

A macro engine only compounds value if it runs reliably every refresh cycle. Defensive data handling and artifact persistence are not peripheral; they are part of production edge.

## Fixes Implemented That Matter

The bug-fix log is mostly UI/operational, but several fixes are important to preserve because they prevent silent signal corruption or system failure:

- Empty DataFrame and missing-column handling across dashboard, metrics, and loaders.
- Date coercion and invalid-date cleanup for CSV-driven workflows.
- Safer index and array access for last-value logic and metrics rendering.
- AUC handling when only one class is present.
- Plot length validation and fallback plotting paths.
- File-access and subprocess error handling in update flows.
- Authentication/config validation for admin-triggered refresh workflows.
- Import/caching fixes to avoid circular-dependency failures.

At the modeling layer, the more important "fixes" are architectural:

- Quarterly series are interpolated to monthly before feature generation.
- Infinite values from percent changes are replaced with `NaN`.
- Expanding windows are used for percentile and z-score calculations to avoid look-ahead bias.
- Decision thresholds are optimized from calibrated deployed probabilities.
- Weaker models are gated out before ensembling.

## What To Port Into ATLAS First

If the goal is to accelerate ATLAS's econometrics stack, the first-pass port should include only the high-value core:

1. Data acquisition schema and monthly alignment logic.
2. Forward-target construction.
3. Literature-backed macro feature library.
4. Time-series CV, calibration, threshold optimization, and ensemble weighting.
5. Strict real-time validation framework with publication lags and vintage auditing.
6. Artifact outputs needed for monitoring and research comparison.

## What Not To Port First

These are useful but not core to the econometric IP transfer:

- Streamlit UI pages.
- Authentication layer.
- Cloud deployment wrappers.
- Scheduler shell scripts unless ATLAS needs the same runtime model.

## Recommended ATLAS Integration Pattern

The cleanest integration is to treat this stack as a macro regime module inside ATLAS:

- Inputs: public macro/market time series snapshot as of date `T`.
- Engine: feature generation -> calibrated ensemble -> regime probabilities.
- Outputs to ATLAS:
  - recession probability
  - macro stress sub-scores
  - trigger states
  - confidence / calibration diagnostics
  - versioned metadata describing training window, active models, and threshold

ATLAS can then consume those outputs for:

- portfolio risk throttles
- defensive overlays
- regime-aware expected return adjustments
- econometric conditioning of other signals

## Bottom Line

The project's durable edge is not "we used XGBoost on FRED data." The edge is a disciplined macro forecasting stack that:

- encodes recession mechanisms from the literature,
- converts them into structured state features,
- calibrates and governs model probabilities carefully,
- validates under realistic real-time constraints,
- and persists diagnostics so the signal can be trusted operationally.

That is the IP worth transferring into ATLAS.
