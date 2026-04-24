# A2 — Full ALFRED Vintage Replay Summary

Ran at: `2026-04-24T01:48:20+00:00`
Origins attempted: **21**  
Origins with valid vintage/simulated comparisons: **21**
Core series considered: 9

## Verdict
**KEEP** — Simulation is materially off: mean abs gap 6.6pp > 3pp; max abs gap 20.0pp > 10pp; 1 in-scope recession(s) show detection changes; production recommendation is switch_to_vintage

Production recommendation: `switch_to_vintage_for_strict_search`

## Aggregate gap (vintage vs simulated)
- Mean |gap|: **6.65 pp**
- Mean signed gap (vintage − simulated): -0.23 pp
- Max |gap|: 19.96 pp (origin 2010-06)
- Std of gap: 9.21 pp

Adjacent diagnostics:
- Mean |vintage − revised|: 14.21 pp
- Mean |simulated − revised|: 14.34 pp

## Per-recession behaviour
| Origin | Label | Prob vintage | Prob simulated | Prob revised | Gap | Signal vintage | Signal simulated | Detection flip |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1990-02 | S&L (positive window) | 47.6% | 47.7% | 36.8% | -0.1 pp | 1 | 1 | no |
| 2000-10 | Dot-com (positive window) | 30.0% | 46.4% | 36.7% | -16.4 pp | 0 | 1 | YES |
| 2007-07 | GFC (positive window) | 83.5% | 84.2% | 74.0% | -0.7 pp | 1 | 1 | no |
| 2007-12 | GFC peak month | 87.8% | 99.1% | 97.4% | -11.3 pp | 1 | 1 | no |
| 2019-09 | COVID (positive window) | 21.5% | 24.8% | 8.9% | -3.3 pp | 0 | 0 | no |

## Five origins with largest |vintage − simulated| gap
| Origin | Category | Label | Prob vintage | Prob simulated | Gap |
| --- | --- | --- | --- | --- | --- |
| 2010-06 | expansion | Expansion check (early 2010s) | 45.0% | 25.0% | +20.0% |
| 2023-06 | expansion | Expansion check (recent cycle) | 67.9% | 84.9% | -17.1% |
| 2022-06 | expansion | Expansion check (recent tightening) | 26.4% | 9.7% | +16.7% |
| 2000-10 | recession_peak | Dot-com (positive window) | 30.0% | 46.4% | -16.4% |
| 1988-06 | expansion | Expansion check (late 1980s) | 58.6% | 45.5% | +13.1% |

## ALFRED coverage by series
| Series | Hits | Attempts |
| --- | --- | --- |
| `coincident_PAYEMS` | 20 | 21 |
| `coincident_UNRATE` | 21 | 21 |
| `coincident_INDPRO` | 21 | 21 |
| `coincident_PI` | 21 | 21 |
| `coincident_RSXFS` | 14 | 21 |
| `coincident_CMRMTSPL` | 8 | 21 |
| `lagging_UEMPMEAN` | 21 | 21 |
| `leading_PERMIT` | 15 | 21 |
| `leading_HOUST` | 21 | 21 |

Series with ALFRED coverage: coincident_PAYEMS, coincident_UNRATE, coincident_INDPRO, coincident_PI, coincident_RSXFS, coincident_CMRMTSPL, lagging_UEMPMEAN, leading_PERMIT, leading_HOUST

Series without ALFRED coverage: none

## Production guidance
- Simulation systematically diverges from ALFRED vintage data by >3pp on average or >10pp on a peak origin. Treat the current strict_vintage_search as optimistic.
- Switch strict_vintage_search to use ALFRED vintages for core series; leave the live production pipeline on simulated lag but record the bias in the executive report.

