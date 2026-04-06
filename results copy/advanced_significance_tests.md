# Advanced Statistical Significance Tests

This report runs four rigorous statistical significance tests across the LLM portfolios to evaluate the validity of the 'Crisis Alpha' thesis, downside protection, and sleeve selection capabilities.

## 1. Market Timing Significance (Henriksson-Merton Test)
Regression with a dummy indicating a down-market for the benchmark.
$R_p - R_f = \alpha + \beta_1(R_m - R_f) + \beta_2\max(0, R_f - R_m) + \epsilon$

If $\beta_2 > 0$ and $p < 0.05$, the model statistically reduces downside beta during market shocks.

| Portfolio               |   HM Beta2 |   HM p-val |
|:------------------------|-----------:|-----------:|
| LLM-BL-A                |      0.037 |     0.8436 |
| LLM-BL-B                |     -0.082 |     0.6457 |
| LLM-BL-C                |     -0.008 |     0.9586 |
| LLM-BL-D                |     -0.002 |     0.9917 |
| LLM-BL-A-Monthly        |      0.292 |     0.1042 |
| LLM-BL-B-Monthly        |      0.131 |     0.1828 |
| LLM-BL-C-Monthly        |      0.072 |     0.4642 |
| LLM-BL-D-Monthly        |      0.047 |     0.7128 |
| LLM-BL-A-Monthly-Banded |      0.382 |     0.0036 |
| LLM-BL-B-Monthly-Banded |      0.121 |     0.26   |
| LLM-BL-C-Monthly-Banded |      0.113 |     0.4617 |
| LLM-BL-D-Monthly-Banded |     -0.062 |     0.7011 |
| LLM-BL-A-Banded         |      0.388 |     0.1484 |
| LLM-BL-B-Banded         |     -0.016 |     0.9593 |
| LLM-BL-C-Banded         |      0.121 |     0.615  |
| LLM-BL-D-Banded         |     -0.05  |     0.8218 |

## 2. Tail-Risk Reduction (95% CVaR Bootstrap Difference Test)
Comparing the 95% Conditional Value at Risk (average return of the worst 5% of months). We run 10,000 bootstrap resamples on paired differences. The `Prob(LLM > Bench)` indicates the percentage of simulated paths where the LLM had a smaller expected shortfall (i.e., empirical p-value for the difference).

| Portfolio               | CVaR(P)   | CVaR(B)   | CVaR P(P>B)   |
|:------------------------|:----------|:----------|:--------------|
| LLM-BL-A                | -6.75%    | -6.39%    | 26.9%         |
| LLM-BL-B                | -7.21%    | -6.39%    | 21.7%         |
| LLM-BL-C                | -7.04%    | -6.39%    | 11.5%         |
| LLM-BL-D                | -7.15%    | -6.39%    | 4.8%          |
| LLM-BL-A-Monthly        | -7.09%    | -6.39%    | 10.0%         |
| LLM-BL-B-Monthly        | -6.98%    | -6.39%    | 9.6%          |
| LLM-BL-C-Monthly        | -6.98%    | -6.39%    | 15.3%         |
| LLM-BL-D-Monthly        | -6.96%    | -6.39%    | 15.3%         |
| LLM-BL-A-Monthly-Banded | -6.36%    | -6.39%    | 73.5%         |
| LLM-BL-B-Monthly-Banded | -6.51%    | -6.39%    | 29.9%         |
| LLM-BL-C-Monthly-Banded | -6.68%    | -6.39%    | 29.0%         |
| LLM-BL-D-Monthly-Banded | -6.38%    | -6.39%    | 52.4%         |
| LLM-BL-A-Banded         | -5.95%    | -6.39%    | 82.9%         |
| LLM-BL-B-Banded         | -6.82%    | -6.39%    | 36.4%         |
| LLM-BL-C-Banded         | -6.88%    | -6.39%    | 23.1%         |
| LLM-BL-D-Banded         | -7.39%    | -6.39%    | 3.5%          |

## 3. Sleeve-Level Alpha (Equity & Fixed Income Paired t-Tests)
Isolating the active return strictly from instrument selection by stripping out the macro asset-class allocation drift. A significant negative Equity t-stat confirms the Contrarian Bias, while a significant positive FI t-stat confirms yield-curve optimization skill.

| Portfolio               | EQ AR Mean   |   EQ NW t-stat |   EQ NW p-val | FI AR Mean   |   FI NW t-stat |   FI NW p-val |
|:------------------------|:-------------|---------------:|--------------:|:-------------|---------------:|--------------:|
| LLM-BL-A                | -0.091%      |         -1.496 |        0.1347 | +0.038%      |          1.044 |        0.2964 |
| LLM-BL-B                | -0.125%      |         -1.624 |        0.1044 | +0.092%      |          2.604 |        0.0092 |
| LLM-BL-C                | +0.062%      |          0.552 |        0.5808 | +0.018%      |          0.627 |        0.5304 |
| LLM-BL-D                | -0.100%      |         -1.794 |        0.0729 | +0.034%      |          1.097 |        0.2727 |
| LLM-BL-A-Monthly        | -0.097%      |         -1.525 |        0.1273 | +0.040%      |          1.099 |        0.272  |
| LLM-BL-B-Monthly        | -0.146%      |         -1.942 |        0.0521 | +0.087%      |          2.562 |        0.0104 |
| LLM-BL-C-Monthly        | -0.013%      |         -0.151 |        0.88   | +0.009%      |          0.336 |        0.7372 |
| LLM-BL-D-Monthly        | -0.087%      |         -1.416 |        0.1569 | +0.035%      |          1.115 |        0.2649 |
| LLM-BL-A-Monthly-Banded | -0.097%      |         -1.525 |        0.1273 | +0.040%      |          1.099 |        0.272  |
| LLM-BL-B-Monthly-Banded | -0.146%      |         -1.942 |        0.0521 | +0.087%      |          2.562 |        0.0104 |
| LLM-BL-C-Monthly-Banded | -0.013%      |         -0.151 |        0.88   | +0.009%      |          0.336 |        0.7372 |
| LLM-BL-D-Monthly-Banded | -0.087%      |         -1.416 |        0.1569 | +0.035%      |          1.115 |        0.2649 |
| LLM-BL-A-Banded         | -0.091%      |         -1.496 |        0.1347 | +0.038%      |          1.044 |        0.2964 |
| LLM-BL-B-Banded         | -0.125%      |         -1.624 |        0.1044 | +0.092%      |          2.604 |        0.0092 |
| LLM-BL-C-Banded         | +0.062%      |          0.552 |        0.5808 | +0.018%      |          0.627 |        0.5304 |
| LLM-BL-D-Banded         | -0.100%      |         -1.794 |        0.0729 | +0.034%      |          1.097 |        0.2727 |

## 4. Event-Study Cumulative Abnormal Return (CAR)
Evaluating the 6 specific crisis epochs. Is the Cumulative Abnormal Return across the 6 events statistically greater than zero? (1-sample, 1-tailed t-test).

| Portfolio               | Crisis Mean CAR   |   Crisis CAR p-val |
|:------------------------|:------------------|-------------------:|
| LLM-BL-A                | -0.74%            |             0.701  |
| LLM-BL-B                | -0.44%            |             0.6143 |
| LLM-BL-C                | -0.73%            |             0.7409 |
| LLM-BL-D                | -0.51%            |             0.6893 |
| LLM-BL-A-Monthly        | +0.76%            |             0.2587 |
| LLM-BL-B-Monthly        | -0.36%            |             0.7298 |
| LLM-BL-C-Monthly        | +0.29%            |             0.302  |
| LLM-BL-D-Monthly        | -0.64%            |             0.8056 |
| LLM-BL-A-Monthly-Banded | +0.66%            |             0.286  |
| LLM-BL-B-Monthly-Banded | -1.11%            |             0.9245 |
| LLM-BL-C-Monthly-Banded | +0.30%            |             0.3254 |
| LLM-BL-D-Monthly-Banded | -1.27%            |             0.893  |
| LLM-BL-A-Banded         | -0.22%            |             0.5634 |
| LLM-BL-B-Banded         | -0.34%            |             0.5798 |
| LLM-BL-C-Banded         | -1.07%            |             0.8191 |
| LLM-BL-D-Banded         | -1.39%            |             0.8556 |

