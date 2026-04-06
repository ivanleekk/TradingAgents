# LLM Rank IC and Sub-Asset Attribution

## 1. Cross-Sectional Information Coefficient (Rank IC)
Evaluates the LLM's raw predictive power by ranking its 30-day Target Returns against actual realized 1-month forward returns for the 14 funds.

| Model | Mean Rank IC | IC > 0 (Hit Rate) | T-Stat | p-value |
|-------|--------------|-------------------|--------|---------|
| LLM-BL-A | 0.003 | 51.6% | 0.08 | 0.9383 |
| LLM-BL-B | 0.041 | 59.4% | 1.27 | 0.2076 |
| LLM-BL-C | 0.021 | 46.9% | 0.61 | 0.5423 |
| LLM-BL-D | -0.006 | 53.1% | -0.15 | 0.8837 |

## 2. Sub-Asset Attribution (LLM-BL-A-Monthly-Banded)
Breaks down the Intra-Class Selection effect into individual ticker contributions by normalizing the sub-asset weights strictly within their asset class buckets (Equity/FI) and sizing by the benchmark class weight. This isolates *pure* asset selection independent of the portfolio's top-level macro allocation drift.

**Total Equity Selection Contribution:** -3.67%

| Equity Asset | Avg Active Bucket Weight | Selection Contribution |
|--------------|--------------------------|------------------------|
| 0P0001EF2T.SI | +6.25% | +0.43% |
| EIMI.L | +7.36% | +0.28% |
| DE000SLA4YD9.SG | -0.09% | +0.26% |
| 0P0001AF7U.SI | -14.76% | -0.60% |
| SPY | -0.97% | -0.60% |
| 0P0001AF7Z.SI | +4.99% | -1.17% |
| ^990100-USD-STRD | -2.79% | -2.28% |

**Total FI Selection Contribution:** 1.01%

| Fixed Income Asset | Avg Active Bucket Weight | Selection Contribution |
|--------------------|--------------------------|------------------------|
| 0P0001EQUE.SI | +2.57% | +0.65% |
| PEBIX | +0.66% | +0.64% |
| AGGG.L | -6.61% | +0.53% |
| 0P0001CC3M | +13.15% | +0.32% |
| 0P0001DWI0.SI | +7.34% | +0.18% |
| 0P0000KYEE.SI | -12.78% | -0.54% |
| IE0002461055.IR | -4.33% | -0.78% |
