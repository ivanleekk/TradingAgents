# Systematic BL Hyperparameter Grid Search Report

Generated at: 2026-03-13T00:21:17.365667

## Grid Configuration

- Optimizers: max-sharpe, max-quadratic-utility
- Risk aversion values: 1.0, 2.0, 5.0
- L2 gamma values: 0.1, 0.3, 0.5, 0.7, 0.9, 1.0
- Total runs attempted: 36
- Successful runs: 36
- Failed runs: 0

## Outputs

- run_status.csv
- strategy_metrics_long.csv
- best_by_strategy.csv
- per-run artifacts in runs/<run_tag>/

## Best Config by Strategy (Sharpe)

| Strategy                           | optimizer             |   risk_aversion |   l2_gamma |   Sharpe Ratio | Ann. Return   | Max Drawdown   |   Calmar Ratio | run_tag                         |
|:-----------------------------------|:----------------------|----------------:|-----------:|---------------:|:--------------|:---------------|---------------:|:--------------------------------|
| Endowus-60/40                      | max-sharpe            |               2 |        0.5 |          0.452 | 7.17%         | -21.53%        |          0.333 | max-sharpe_ra2_l20p5            |
| Endowus-60/40-BuyHold              | max-sharpe            |               2 |        0.3 |          0.445 | 7.34%         | -22.29%        |          0.329 | max-sharpe_ra2_l20p3            |
| Endowus-60/40-NoRebalanceIfMissing | max-quadratic-utility |               1 |        0.1 |          0.452 | 7.18%         | -21.53%        |          0.334 | max-quadratic-utility_ra1_l20p1 |
| Endowus_Actual_0/100               | max-sharpe            |               1 |        0.1 |         -0.321 | 0.48%         | -14.46%        |          0.033 | max-sharpe_ra1_l20p1            |
| Endowus_Actual_100/0               | max-sharpe            |               1 |        0.1 |          0.628 | 10.74%        | -20.00%        |          0.537 | max-sharpe_ra1_l20p1            |
| Endowus_Actual_20/80               | max-sharpe            |               1 |        0.1 |          0.062 | 2.61%         | -15.01%        |          0.174 | max-sharpe_ra1_l20p1            |
| Endowus_Actual_40/60               | max-sharpe            |               1 |        0.1 |          0.304 | 4.67%         | -16.01%        |          0.292 | max-sharpe_ra1_l20p1            |
| Endowus_Actual_60/40               | max-sharpe            |               1 |        0.1 |          0.455 | 6.72%         | -17.40%        |          0.386 | max-sharpe_ra1_l20p1            |
| Endowus_Actual_80/20               | max-sharpe            |               1 |        0.1 |          0.556 | 8.73%         | -18.67%        |          0.468 | max-sharpe_ra1_l20p1            |
| LLM-BL-A                           | max-quadratic-utility |               1 |        1   |          0.361 | 6.00%         | -21.44%        |          0.28  | max-quadratic-utility_ra1_l21   |
| LLM-BL-A-Banded                    | max-quadratic-utility |               1 |        1   |          0.339 | 5.63%         | -20.51%        |          0.274 | max-quadratic-utility_ra1_l21   |
| LLM-BL-A-Monthly                   | max-sharpe            |               1 |        0.9 |          0.44  | 7.15%         | -20.08%        |          0.356 | max-sharpe_ra1_l20p9            |
| LLM-BL-A-Monthly-Banded            | max-sharpe            |               1 |        0.9 |          0.467 | 7.37%         | -19.72%        |          0.374 | max-sharpe_ra1_l20p9            |
| LLM-BL-B                           | max-quadratic-utility |               5 |        1   |          0.367 | 6.07%         | -21.43%        |          0.283 | max-quadratic-utility_ra5_l21   |
| LLM-BL-B-Banded                    | max-quadratic-utility |               1 |        1   |          0.354 | 5.80%         | -20.73%        |          0.28  | max-quadratic-utility_ra1_l21   |
| LLM-BL-B-Monthly                   | max-sharpe            |               2 |        0.7 |          0.4   | 6.49%         | -20.35%        |          0.319 | max-sharpe_ra2_l20p7            |
| LLM-BL-B-Monthly-Banded            | max-sharpe            |               1 |        0.7 |          0.404 | 6.49%         | -20.55%        |          0.316 | max-sharpe_ra1_l20p7            |
| LLM-BL-C                           | max-quadratic-utility |               2 |        1   |          0.35  | 5.86%         | -21.48%        |          0.273 | max-quadratic-utility_ra2_l21   |
| LLM-BL-C-Banded                    | max-quadratic-utility |               5 |        1   |          0.314 | 5.29%         | -20.86%        |          0.253 | max-quadratic-utility_ra5_l21   |
| LLM-BL-C-Monthly                   | max-quadratic-utility |               2 |        0.9 |          0.376 | 6.12%         | -20.10%        |          0.305 | max-quadratic-utility_ra2_l20p9 |
| LLM-BL-C-Monthly-Banded            | max-sharpe            |               1 |        1   |          0.363 | 5.92%         | -19.22%        |          0.308 | max-sharpe_ra1_l21              |
| LLM-BL-D                           | max-quadratic-utility |               1 |        1   |          0.358 | 5.96%         | -21.47%        |          0.277 | max-quadratic-utility_ra1_l21   |
| LLM-BL-D-Banded                    | max-quadratic-utility |               1 |        1   |          0.311 | 5.31%         | -21.75%        |          0.244 | max-quadratic-utility_ra1_l21   |
| LLM-BL-D-Monthly                   | max-quadratic-utility |               2 |        1   |          0.379 | 6.14%         | -20.26%        |          0.303 | max-quadratic-utility_ra2_l21   |
| LLM-BL-D-Monthly-Banded            | max-quadratic-utility |               2 |        1   |          0.336 | 5.53%         | -19.90%        |          0.278 | max-quadratic-utility_ra2_l21   |
| Markowitz                          | max-quadratic-utility |               2 |        0.5 |          0.384 | 6.50%         | -22.52%        |          0.289 | max-quadratic-utility_ra2_l20p5 |
| Systematic-BL                      | max-quadratic-utility |               1 |        0.1 |          0.367 | 6.33%         | -21.83%        |          0.29  | max-quadratic-utility_ra1_l20p1 |
| Systematic-BL-Tactical-Monthly     | max-quadratic-utility |               1 |        0.1 |          0.454 | 7.38%         | -21.90%        |          0.337 | max-quadratic-utility_ra1_l20p1 |

## Top 10 Systematic-BL Configs by Sharpe

| optimizer             |   risk_aversion |   l2_gamma |   Sharpe Ratio | Ann. Return   | Max Drawdown   |   Calmar Ratio | run_tag                         |
|:----------------------|----------------:|-----------:|---------------:|:--------------|:---------------|---------------:|:--------------------------------|
| max-quadratic-utility |               1 |        0.1 |          0.367 | 6.33%         | -21.83%        |          0.29  | max-quadratic-utility_ra1_l20p1 |
| max-quadratic-utility |               2 |        1   |          0.364 | 6.41%         | -21.54%        |          0.297 | max-quadratic-utility_ra2_l21   |
| max-quadratic-utility |               2 |        0.9 |          0.363 | 6.40%         | -21.53%        |          0.297 | max-quadratic-utility_ra2_l20p9 |
| max-quadratic-utility |               1 |        1   |          0.363 | 6.37%         | -21.56%        |          0.296 | max-quadratic-utility_ra1_l21   |
| max-quadratic-utility |               1 |        0.9 |          0.363 | 6.38%         | -21.56%        |          0.296 | max-quadratic-utility_ra1_l20p9 |
| max-quadratic-utility |               2 |        0.7 |          0.36  | 6.37%         | -21.53%        |          0.296 | max-quadratic-utility_ra2_l20p7 |
| max-quadratic-utility |               1 |        0.7 |          0.358 | 6.31%         | -21.57%        |          0.293 | max-quadratic-utility_ra1_l20p7 |
| max-quadratic-utility |               2 |        0.5 |          0.357 | 6.36%         | -21.55%        |          0.295 | max-quadratic-utility_ra2_l20p5 |
| max-quadratic-utility |               5 |        0.9 |          0.348 | 6.29%         | -21.46%        |          0.293 | max-quadratic-utility_ra5_l20p9 |
| max-quadratic-utility |               5 |        1   |          0.343 | 6.19%         | -21.47%        |          0.288 | max-quadratic-utility_ra5_l21   |

