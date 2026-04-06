# Statistical Significance of Performance Tests

## 1. Rolling Window "Hit Rate" (Binomial Test)
We broke the equity curves into rolling periods to see how consistently the LLM beats the benchmark, performing a Binomial Test ($H_0: p = 0.5$).

- **3-Month Rolling Windows**: 
  - Hit Rate: 45.9% (28 out of 61 windows)
  - Binomial Test $p$-value: 0.7787
- **6-Month Rolling Windows**: 
  - Hit Rate: 46.6% (27 out of 58 windows)
  - Binomial Test $p$-value: 0.7441

## 2. Paired t-Test for Excess Returns (Newey-West)
Testing if the active returns of the system (LLM - Benchmark) are statistically greater than 0, adjusting for autocorrelation.

- **Mean Monthly Active Return**: 0.081%
- **Newey-West t-statistic** (lags=3): 0.5590
- **Newey-West $p$-value**: 0.5762

## 3. Jobson-Korkie Test for Sharpe Ratio Significance (Memmel correction)
Comparing the Sharpe Ratio of the LLM vs the Benchmark accounting for the high correlation between the portfolios.

- **Annualized Sharpe Ratios (Weekly Data)**:
  - LLM ($SR_1$): 0.414
  - Benchmark ($SR_2$): 0.372
- **Test Statistic ($Z$)**: 0.0751
- **Jobson-Korkie $p$-value** (two-tailed): 0.9401
