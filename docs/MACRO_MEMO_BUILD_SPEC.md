# Macro Monitor Weekly Memo — Build Spec / Outline

**Status:** design spec for a layer that does not yet exist.
**Audience:** whoever builds the client-memo reporting layer in ATLAS (or back-ports it into Econ-Dashboard).
**Scope:** everything from "models have been retrained" to "a reviewed HTML client memo lands in the repo." It does **not** cover the recession/regime engine itself — that IP already transferred (see `ATLAS_IP_TRANSFER_SUMMARY.md`).

---

## 0. Why this document exists

The weekly Claude routine that authors the "Bluestone Capital — Macro Monitor" memo is a **consumer**. It reads a deterministic data bundle + a governing spec, writes one `narrative.json`, renders HTML, and commits it. That consumer contract is fully specified (in the routine prompt). The **producer** half — the bundle builder, the spec file, the template, the chart module, the render CLI, and the CI wiring that emits the bundle — was never implemented. This document specifies the producer half and the contract surface between the two halves.

**Design invariant:** facts are deterministic and version-pinned; prose is Claude's job. The two never mix. The bundle is the only channel of facts into the memo; the spec is the only channel of rules. A fresh, zero-memory Claude session must be able to produce a correct memo from `bundle + spec` alone.

---

## 1. Producer / consumer architecture

```
STAGE 1 — PRODUCER (deterministic · CI, no LLM)
  retrain models ─► build_memo_bundle.py ─► memo_bundle_latest.json   (+ rotate prior)
                                            ▲
                                            │ schema_version (frozen contract)
                                            ▼
STAGE 2 — CONSUMER (judgment · weekly Claude session)
  read bundle + memo_spec.md ─► narrative.json ─► render_memo.py ─► memo.html ─► commit
```

Two runtimes, one contract (`schema_version`). Stage 1 must run and commit the bundle **before** Stage 2 fires. If Stage 1 didn't produce a bundle, Stage 2 aborts loudly — it never fabricates facts.

---

## 2. Files to create

| Path | Owner | Purpose | Mutable per week? |
|---|---|---|---|
| `templates/memo_spec.md` | human | Single governing rule set (palette, sections, voice, slots, edge cases) | No — versioned contract |
| `templates/memo.html.j2` | human | Jinja2 template, `StrictUndefined` | No |
| `scripts/build_memo_bundle.py` | human | Aggregate model outputs → `memo_bundle_latest.json` | No |
| `scripts/memo_charts.py` | human | Build the 4 chart payloads/divs | No |
| `scripts/render_memo.py` | human | CLI: bundle + narrative + template → HTML | No |
| `.claude/commands/weekly-memo.md` | human | Slash-command: condensed routine + most-violated voice rules | No |
| `data/reports/memo_bundle_latest.json` | CI (build script) | The week's facts | Regenerated weekly |
| `data/reports/memo_bundle_prior.json` | CI (build script) | Last week's facts, for Δ | Rotated weekly |
| `data/reports/memos/<date>/narrative.json` | Claude (Stage 2) | The week's prose, 9 slots | Yes — the only weekly authored artifact |
| `data/reports/memos/<date>/memo.html` | render script | The deliverable | Yes |

**Hard rule for Stage 2:** the consumer may only ever write `narrative.json` (and trigger the render). It must never edit any "No / human" row.

---

## 3. THE CONTRACT — `memo_bundle` JSON schema (most important section)

This is the load-bearing artifact. Freeze it, version it, test it. `schema_version` is an integer; bump it only with a coordinated spec change. Stage 2 asserts `schema_version == 1` and aborts if higher.

```jsonc
{
  "schema_version": 1,

  "headline": {
    "report_date": "2026-06-14",          // ISO; drives data/reports/memos/<report_date>/
    "prior_report_date": "2026-06-07",    // null if no prior bundle
    "data_as_of": "2026-06-30",           // last obs date in the series
    "generated_utc": "2026-06-14T07:35:00Z",
    "prob_6m_pct": 0.25,                  // Prob_Ensemble (6M) * 100, 2dp
    "prob_18m_pct": 21.6,                 // Prob_Ensemble (18M) * 100, 2dp
    "prob_6m_ci": [0.0, 6.46],            // CI_Lower/Upper * 100
    "prob_18m_ci": [/* ... */],
    "risk_label_6m": "LOW",               // bucketed vs thresholds, see §3.1
    "risk_label_18m": "ELEVATED",
    "threshold_6m_pct": 10.0,             // decision threshold * 100 (data/models/threshold.json)
    "threshold_18m_pct": 21.0,
    "delta_18m_pp": 6.4,                  // prob_18m_pct - prior; null if no prior
    "calibration_step_18m": true          // |delta_18m_pp| > 5 → narrative must note bin transition
  },

  "glr": {                                 // Growth / Liquidity / Risk-appetite composites
    "growth":          { "z": -0.31, "state": "neutral", "delta_z": -0.12, "top_contributors": [ /* see §3.2 */ ] },
    "liquidity":       { "z":  0.84, "state": "strong",  "delta_z":  0.05, "top_contributors": [ ... ] },
    "risk_appetite":   { "z":  0.42, "state": "neutral", "delta_z": -0.20, "top_contributors": [ ... ] },
    "cross_current":   "growth soft, liquidity supportive, risk-appetite mixed"  // optional precomputed summary string
  },

  "movers": [                              // sorted by |delta_z| desc; Stage 2 leads "what_changed" with [0]
    { "id": "ICSA", "label": "Initial Claims", "z": -0.57, "delta_z": 0.34,
      "level": 214000, "unit": "claims", "direction": "improving" },
    /* ... 6–12 entries ... */
  ],

  "spotlight": {                           // EXACTLY these 6 FRED IDs, all present
    "T10Y3M":       { "label": "10Y–3M Treasury Spread", "z": -1.2, "level": -0.18, "unit": "pp",
                      "delta_z": 0.10, "series": [ /* {date, value} for chart */ ] },
    "ICSA":         { ... },
    "PAYEMS":       { ... },
    "PERMIT":       { ... },
    "BAMLH0A0HYM2": { ... },              // HY OAS
    "VIXCLS":       { ... }
  },

  "charts": {                              // payloads from memo_charts.py (see §5)
    "prob_history":   { /* x, y, ci bands, threshold line, recession shading */ },
    "glr_radar":      { /* 3 composite z-scores */ },
    "spotlight_grid": { /* small-multiples of the 6 spotlight series */ },
    "contrib_bar":    { /* GLR contributor decomposition */ }
  },

  "model_meta": {                          // for the Methodology block ONLY (never in Bottom Line prose)
    "horizon_6m": { "auc": 0.927, "brier": 0.197, "pr_auc": 0.692, "threshold": 0.10,
                    "active_models": ["probit","random_forest","xgboost"], "train_end": "2026-05-31" },
    "horizon_18m": { ..., "calibrator": "sigmoid" }
  }
}
```

### 3.1 Risk-label bucketing (deterministic — done in build script, not by Claude)
Buckets are relative to the per-horizon decision threshold. Use the same logic for 6M (threshold ~10%) and 18M (threshold ~21%):

| Label | Rule (prob vs threshold `t`) |
|---|---|
| `LOW` | `prob < 0.5·t` |
| `MODERATE` | `0.5·t ≤ prob < t` |
| `ELEVATED` | `t ≤ prob < 2·t` |
| `HIGH` | `prob ≥ 2·t` |

Emojis are assigned in the template, not the bundle (presentation concern): 🟢/🟡/🟠/🔴.

### 3.2 `top_contributors` shape
```jsonc
{ "feature": "growth_ICSA_neg", "contribution_z": 0.41, "level": 214000, "unit": "claims" }
```
Derived from `glr_components.csv` columns grouped by prefix (`growth_*`, `liquidity_*`, `risk_*`). Rank by |contribution|.

---

## 4. `build_memo_bundle.py` — the aggregator (CI, deterministic)

**Inputs (Econ-Dashboard names; repoint for ATLAS):**
- `data/predictions.csv` — `Prob_Ensemble`, `CI_Lower`, `CI_Upper` filtered to last row per `Forecast_Horizon ∈ {6,18}`. (18M lives under `data/models/horizon_18m/predictions.csv`.)
- `data/glr_components.csv` — raw sub-components grouped by `growth_/liquidity_/risk_` prefix → composite z + state + contributors. **Note:** today these columns are largely NaN at the latest date; the build script owns NaN handling (skip-NaN mean over available members, record coverage).
- `data/models/threshold.json` + `horizon_18m/threshold.json` — decision thresholds.
- `data/models/metrics.csv`, `calibration_diagnostics.json`, `run_manifest.json` (+ 18M) — `model_meta`.
- A spotlight source for the 6 FRED IDs with **z-score AND level** + a short history slice for charts. (Pull from `data/indicators.csv` if the raw levels live there; otherwise add a small spotlight extractor.)
- The existing `data/reports/memo_bundle_latest.json` (to rotate → prior, and compute deltas).

**Procedure:**
1. Load current model outputs.
2. **Rotate:** if `memo_bundle_latest.json` exists, copy it to `memo_bundle_prior.json` before overwriting.
3. Compute `headline` (probs, CIs, risk labels via §3.1, deltas, `calibration_step_18m`).
4. Compute `glr` composites + states + contributors.
5. Build `movers` (per-indicator z + Δz vs prior; sort by |Δz|).
6. Build `spotlight` for the six fixed IDs (assert all six present, each with z **and** level).
7. Build `charts` via `memo_charts.py`.
8. Assemble `model_meta`.
9. Stamp `schema_version`, `headline.report_date`, `generated_utc`.
10. Validate against a JSON Schema (ship `templates/memo_bundle.schema.json`); fail CI on violation.
11. Write `memo_bundle_latest.json`.

**State thresholds (composite z → state):** `z ≥ +0.5` → `strong`; `−0.5 < z < +0.5` → `neutral`; `z ≤ −0.5` → `weak`. (Tune; keep in the spec so Stage 2 can describe them.)

---

## 5. `memo_charts.py` — exactly 4 charts

The template references four `<div id="chart_*">`. Stage 2's self-check requires all four present.

1. `chart_prob_history` — 6M (and 18M) calibrated probability over time, with CI band, decision-threshold line, and NBER recession shading.
2. `chart_glr_radar` — the three composite z-scores (Growth / Liquidity / Risk-appetite).
3. `chart_spotlight_grid` — small multiples of the six spotlight series.
4. `chart_contrib_bar` — GLR contributor decomposition (top movers within each composite).

Self-contained payloads (inline Plotly JSON or pre-rendered SVG) so the HTML is portable and offline-openable. No external CDN dependency at view time if memos are emailed.

---

## 6. `memo.html.j2` + `render_memo.py`

**Template rules:**
- `Environment(undefined=StrictUndefined)` — a missing bundle key raises `UndefinedError` rather than rendering blank. This is intentional: a missing fact must break the build, not ship silently.
- **8 H2 sections in locked order** (define exact titles in spec §B.3). Suggested order:
  1. Executive Summary
  2. Headline Chart
  3. Worth Flagging (callout)
  4. What Changed
  5. Attribution
  6. Spotlight (the 6 FRED IDs)
  7. Bottom Line
  8. Methodology
- Palette + typography frozen in spec §A. Emojis/risk colors assigned here from `risk_label_*`.
- Every indicator reference renders **z-score AND level/unit together** (template helper enforces it).

**`render_memo.py` CLI:**
```
python3 scripts/render_memo.py \
  --bundle data/reports/memo_bundle_latest.json \
  --narrative data/reports/memos/<report_date>/narrative.json \
  [--out data/reports/memos/<report_date>/memo.html]
```
Loads both JSONs, merges (`bundle.*` facts + `narrative.*` prose) into the template context, renders, writes HTML. Output dir derives from `bundle.headline.report_date`.

---

## 7. `narrative.json` — the 9 consumer slots (Stage 2 output)

Exact slot names (spec §E). This is the **only** thing Claude writes each week.

| Slot | Type | Rule |
|---|---|---|
| `executive_summary` | string | Lead with GLR composites (Growth → Liquidity → Risk-appetite → cross-current), then recession probs as corroboration. **Never** lead with the recession probability. |
| `headline_chart_caption` | string | If `calibration_step_18m`, add one sentence on the bin-transition mechanism. |
| `what_changed` | string[4–6] | Lead with `movers[0]` (largest \|Δz\|). |
| `attribution_paragraph_1` | string | Three numbered forces driving the 6M. |
| `attribution_paragraph_2` | string | Two areas to watch. |
| `callout` | string | "Worth Flagging" prose. |
| `spotlight_callouts` | object | Keyed by the 6 FRED IDs; **all 6 keys present**. |
| `bottom_line_summary` | string | **No model jargon** (no AUC/PR-AUC/Brier/isotonic/calibration). |
| `bottom_line_watch` | string | "Three things are worth tracking" pattern. |

---

## 8. Voice rules (spec §F — applied to every slot)

1. Numbers first, opinions second.
2. No predictions — "the model places" / "the regime read is", never "we predict/forecast/expect".
3. Cite z-score **and** level/unit together for every indicator (e.g., "claims at 214,000 — at −0.57σ").
4. Bold sparingly; ≤ 6 bold spans per paragraph.
5. Bottom Line forbids model jargon (AUC, PR-AUC, Brier, isotonic, calibration) — that lives only in Methodology.
6. Italics reserved for *rate* vs level distinctions.
7. (room to extend to 10 in spec — keep them numbered and stable so the slash-command can cite "rules 1–10".)

The `.claude/commands/weekly-memo.md` slash-command restates the most-violated rules (lead-with-GLR, no "we predict", z+level pairing, Bottom-Line jargon ban, step-function note) up front.

---

## 9. CI wiring (Stage 1)

Add to the weekly workflow (`scheduler.yml` equivalent), **after** the model retrain/commit:

```yaml
  - name: Build memo bundle
    run: |
      python scripts/build_memo_bundle.py \
        --out data/reports/memo_bundle_latest.json \
        --rotate-prior data/reports/memo_bundle_prior.json
  - name: Validate bundle schema
    run: python scripts/validate_bundle.py data/reports/memo_bundle_latest.json
  - name: Commit bundle
    run: |
      git add data/reports/memo_bundle_latest.json data/reports/memo_bundle_prior.json
      git diff --staged --quiet || git commit -m "Memo bundle $(date +'%Y-%m-%d')"
      git push
```

Stage 2 (the Claude routine) runs on a later schedule, pulls main, and finds the bundle waiting. If the build/validate step fails, **no bundle is committed** and Stage 2 aborts cleanly — exactly the behavior we want.

---

## 10. Acceptance tests / self-checks (spec §I.3)

Stage 2 verifies after render:
- `memo.html` exists at `data/reports/memos/<report_date>/memo.html`.
- Size 80 KB–500 KB.
- `grep -c '{{' memo.html` == 0 (no unfilled Jinja slots).
- All 8 H2 sections present in locked order.
- All four `<div id="chart_*">` present.
- Spot-check 3 numbers against the bundle: `prob_6m_pct`, GLR Growth z, top mover.

Producer-side CI tests (build once, assert):
- Bundle validates against `memo_bundle.schema.json`.
- All 6 spotlight IDs present, each with non-null `z` and `level`.
- `risk_label_*` matches §3.1 bucketing for the computed prob.
- Prior rotation happened (or `prior_report_date == null` on first run).
- `calibration_step_18m == (abs(delta_18m_pp) > 5)`.

---

## 11. Build sequence (recommended order)

1. **`memo_bundle.schema.json` + `memo_spec.md`** — lock the contract and rules first. Nothing else can be correct until these exist.
2. **`build_memo_bundle.py`** against real model outputs → produces a valid `memo_bundle_latest.json`. Run it twice to exercise rotation/deltas.
3. **`memo_charts.py`** → chart payloads.
4. **`memo.html.j2` + `render_memo.py`** → render from a hand-written sample `narrative.json`. Iterate on layout/palette.
5. **`.claude/commands/weekly-memo.md`** + the scheduled routine → end-to-end dry run.
6. **CI wiring** → bundle emitted automatically each week.
7. Turn on the weekly Claude routine.

Ship 1–2 before anything else: with a frozen schema and a real bundle, Stage 1 and Stage 2 can be built in parallel.

---

## 12. Open decisions to resolve for ATLAS

- **Spotlight level source.** Confirm where raw FRED levels for the 6 IDs live in ATLAS (Econ-Dashboard scatters them across `indicators.csv` / `glr_components.csv`). The bundle requires z **and** level for each.
- **GLR composite definition.** Today `glr_components.csv` holds raw sub-members, often NaN at the latest date. Decide the exact composite formula, coverage threshold, and state cutoffs — and write them into the spec so the prose can describe them honestly.
- **Chart runtime.** Inline Plotly vs. pre-rendered SVG — depends on whether memos get emailed (offline) or only viewed in-repo.
- **Distribution.** Spec mandates no auto-send; AJ reviews first. Keep that invariant in ATLAS.
- **Schema governance.** Decide who can bump `schema_version` and the spec-change checklist that must accompany it.
```
