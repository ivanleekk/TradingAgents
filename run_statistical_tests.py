import pandas as pd
import numpy as np
import scipy.stats
import statsmodels.api as sm
import portfolio_common

def calculate_tests():
    # 1. Load Data
    df = pd.read_csv("results/endowus_systematic_equity_curves.csv")
    df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'])
    df.set_index('Unnamed: 0', inplace=True)
    
    llm_col = 'LLM-BL-A-Monthly-Banded'
    bench_col_actual = 'Endowus_Actual_60/40'
    bench_col_sim = 'Endowus-60/40'
    
    # Check if the columns exist
    if llm_col not in df.columns or bench_col_actual not in df.columns or bench_col_sim not in df.columns:
        print("Columns not found!")
        return

    # Extract price series
    llm_cum = df[llm_col]
    bench_cum_actual = df[bench_col_actual]
    bench_cum_sim = df[bench_col_sim]
    
    # Calculate weekly returns first to see what the frequency is
    # The data involves roughly weekly data points.
    
    # Convert to monthly frequency for standard financial metrics (Hit rates, Excess Returns)
    df_monthly = df.resample('ME').last()
    llm_cum_monthly = df_monthly[llm_col]
    bench_cum_monthly = df_monthly[bench_col_actual]
    
    # Monthly returns
    llm_ret_m = llm_cum_monthly.pct_change().dropna()
    bench_ret_m = bench_cum_monthly.pct_change().dropna()
    
    print("--- 1. Rolling Window Hit Rate ---")
    ret_3m_llm = llm_cum_monthly.pct_change(3).dropna()
    ret_3m_bench = bench_cum_monthly.pct_change(3).dropna()
    
    ret_6m_llm = llm_cum_monthly.pct_change(6).dropna()
    ret_6m_bench = bench_cum_monthly.pct_change(6).dropna()
    
    # 3-month metrics
    hits_3m = (ret_3m_llm > ret_3m_bench).sum()
    n_3m = len(ret_3m_llm)
    hit_rate_3m = hits_3m / n_3m
    binom_3m = scipy.stats.binomtest(k=hits_3m, n=n_3m, p=0.5, alternative='greater')
    
    print(f"3-Month Rolling Hit Rate: {hit_rate_3m*100:.1f}% ({hits_3m}/{n_3m})")
    print(f"Binomial Test p-value: {binom_3m.pvalue:.4f}")
    
    # 6-month metrics
    hits_6m = (ret_6m_llm > ret_6m_bench).sum()
    n_6m = len(ret_6m_llm)
    hit_rate_6m = hits_6m / n_6m
    binom_6m = scipy.stats.binomtest(k=hits_6m, n=n_6m, p=0.5, alternative='greater')
    
    print(f"6-Month Rolling Hit Rate: {hit_rate_6m*100:.1f}% ({hits_6m}/{n_6m})")
    print(f"Binomial Test p-value: {binom_6m.pvalue:.4f}")
    
    print("\n--- 2. Paired t-Test for Excess Returns (Newey-West) ---")
    active_return = llm_ret_m - bench_ret_m
    
    # 1-sample t-test is equivalent to regression on constant
    X = sm.add_constant(np.ones(len(active_return)))
    model = sm.OLS(active_return, X)
    # Using Newey-West standard errors. Maxlags rule of thumb for monthly is often 3 to 6
    max_lags = 3
    results = model.fit(cov_type='HAC', cov_kwds={'maxlags': max_lags})
    
    t_stat_nw = results.tvalues[0]
    p_value_nw = results.pvalues[0]
    mean_active_return = active_return.mean()
    
    print(f"Mean Monthly Active Return: {mean_active_return*100:.3f}%")
    print(f"Newey-West t-statistic: {t_stat_nw:.4f}")
    print(f"Newey-West p-value: {p_value_nw:.4f}")

    print("\n--- 3. Jobson-Korkie Test for Sharpe Ratio Significance (Memmel correction) ---")
    # Weekly data was originally provided, let's use weekly returns to calculate Sharpe if the report used weekly.
    # The prompt explicitly asked to compare with 0.451 which is Endowus-60/40
    llm_ret_w = llm_cum.pct_change().dropna()
    bench_ret_w = bench_cum_sim.pct_change().dropna()
    
    # Calculate excess returns using historical risk free rate like the baseline
    rf_series = portfolio_common.load_us_3m_tbill_series(start="2019-09-01", end="2024-12-31")
    rf_rate_period = portfolio_common.get_period_risk_free_returns(
        llm_ret_w.index, 52, rf_series, fallback_rate=0.04
    )
    
    llm_excess_w = llm_ret_w.sub(rf_rate_period, axis=0).dropna()
    bench_excess_w = bench_ret_w.sub(rf_rate_period, axis=0).dropna()
    
    sh_w_llm = llm_excess_w.mean() / llm_excess_w.std(ddof=1) * np.sqrt(52)
    sh_w_bench = bench_excess_w.mean() / bench_excess_w.std(ddof=1) * np.sqrt(52)
    print(f"Weekly-based Annualized Sharpe - LLM: {sh_w_llm:.3f}, Bench: {sh_w_bench:.3f}")
    
    # Let's perform Jobson-Korkie on weekly data
    mu1 = llm_excess_w.mean()
    mu2 = bench_excess_w.mean()
    sigma1 = llm_excess_w.std(ddof=1)
    sigma2 = bench_excess_w.std(ddof=1)
    
    common_idx = llm_excess_w.index.intersection(bench_excess_w.index)
    cov_matrix = np.cov(llm_excess_w.loc[common_idx], bench_excess_w.loc[common_idx], ddof=1)
    sigma12 = cov_matrix[0, 1]
    rho = sigma12 / (sigma1 * sigma2)
    
    sh1 = mu1 / sigma1
    sh2 = mu2 / sigma2
    
    # Memmel (2003) Correction for Jobson-Korkie test
    # theta = V_asymptotic(diff in Sharpes)
    # The statistic for Memmel's unannualized Sharpe is Z = (sh1 - sh2) / sqrt(theta)
    T = len(llm_ret_w)
    theta = (1 / T) * (2 * (1 - rho) + 0.5 * (sh1**2 + sh2**2 - 2 * sh1 * sh2 * rho**2))
    
    z_stat = (sh1 - sh2) / np.sqrt(theta)
    # One-sided test (alternative='greater') p-value or two-sided?
    # Usually it's a two-sided test but outperformance implies one-sided
    p_value_jk = 2 * (1 - scipy.stats.norm.cdf(abs(z_stat)))
    
    print(f"Jobson-Korkie Memmel Test Z-statistic: {z_stat:.4f}")
    print(f"Jobson-Korkie Memmel Test p-value: {p_value_jk:.4f}")
    
    # Also write a clean markdown string for output
    report = f"""# Statistical Significance of Performance Tests

## 1. Rolling Window "Hit Rate" (Binomial Test)
We broke the equity curves into rolling periods to see how consistently the LLM beats the benchmark, performing a Binomial Test ($H_0: p = 0.5$).

- **3-Month Rolling Windows**: 
  - Hit Rate: {hit_rate_3m*100:.1f}% ({hits_3m} out of {n_3m} windows)
  - Binomial Test $p$-value: {binom_3m.pvalue:.4f}
- **6-Month Rolling Windows**: 
  - Hit Rate: {hit_rate_6m*100:.1f}% ({hits_6m} out of {n_6m} windows)
  - Binomial Test $p$-value: {binom_6m.pvalue:.4f}

## 2. Paired t-Test for Excess Returns (Newey-West)
Testing if the active returns of the system (LLM - Benchmark) are statistically greater than 0, adjusting for autocorrelation.

- **Mean Monthly Active Return**: {mean_active_return*100:.3f}%
- **Newey-West t-statistic** (lags={max_lags}): {t_stat_nw:.4f}
- **Newey-West $p$-value**: {p_value_nw:.4f}

## 3. Jobson-Korkie Test for Sharpe Ratio Significance (Memmel correction)
Comparing the Sharpe Ratio of the LLM vs the Benchmark accounting for the high correlation between the portfolios.

- **Annualized Sharpe Ratios (Weekly Data)**:
  - LLM ($SR_1$): {sh_w_llm:.3f}
  - Benchmark ($SR_2$): {sh_w_bench:.3f}
- **Test Statistic ($Z$)**: {z_stat:.4f}
- **Jobson-Korkie $p$-value** (two-tailed): {p_value_jk:.4f}
"""

    with open("results/significance_tests_report.md", "w") as f:
        f.write(report)

if __name__ == "__main__":
    calculate_tests()
