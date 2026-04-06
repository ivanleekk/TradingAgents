import pandas as pd
import numpy as np
import scipy.stats
import statsmodels.api as sm
import os

EVENTS = {
    "COVID-19 Crash & Rebound": ("2020-02-01", "2020-05-31"),
    "The Emerging Market Divergence": ("2021-08-01", "2021-12-31"),
    "The Inflation Print Shock": ("2021-11-01", "2022-01-31"),
    "Russia-Ukraine Invasion": ("2022-02-01", "2022-04-30"),
    "The BOJ Yield Curve Surprise": ("2022-12-01", "2023-01-31"),
    "Regional Banking Crisis": ("2023-03-01", "2023-05-31"),
}

def get_risk_free_rate(idx):
    import portfolio_common
    rf_series = portfolio_common.load_us_3m_tbill_series(start="2019-09-01", end="2025-01-01")
    rf_rate_period = portfolio_common.get_period_risk_free_returns(
        idx, 12, rf_series, fallback_rate=0.04
    )
    return rf_rate_period

def run_hm_test(r_p, r_m, r_f):
    # Align and drop any missing data rows to prevent statsmodels crashes
    df_hm = pd.DataFrame({'rp': r_p, 'rm': r_m, 'rf': r_f}).dropna()
    if len(df_hm) < 5:
        return None, None
        
    y = df_hm['rp'] - df_hm['rf']
    x1 = df_hm['rm'] - df_hm['rf']
    x2 = np.maximum(0, df_hm['rf'] - df_hm['rm'])
    
    X = sm.add_constant(pd.DataFrame({'Mkt_Ex': x1, 'Down_Mkt': x2}))
    
    lags = min(3, len(y) - 1)
    model = sm.OLS(y, X).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
    
    b2 = model.params['Down_Mkt']
    p_val = model.pvalues['Down_Mkt']
    return b2, p_val

def run_cvar_bootstrap(r_p, r_m, n_samples=10000, alpha=0.05):
    def calc_cvar(series):
        q = np.percentile(series, alpha * 100)
        return series[series <= q].mean()
    
    df_cvar = pd.DataFrame({'p': r_p, 'm': r_m}).dropna()
    if len(df_cvar) < 10:
        return None, None, None
        
    arr_p = df_cvar['p'].values
    arr_m = df_cvar['m'].values
    
    cvar_p = calc_cvar(arr_p)
    cvar_m = calc_cvar(arr_m)
    
    count_better = 0
    n = len(arr_p)
    
    np.random.seed(42)
    for _ in range(n_samples):
        indices = np.random.randint(0, n, n)
        s_p = arr_p[indices]
        s_m = arr_m[indices]
        
        # In returns, bigger algebraic number is "better" (e.g. -5% > -10%)
        if calc_cvar(s_p) > calc_cvar(s_m):
            count_better += 1
            
    prob_better = count_better / n_samples
    
    return cvar_p, cvar_m, prob_better

def run_sleeve_test(portfolio_name, df_ret_monthly_all_assets, sleeve_assets):
    weights_path = f"results/weights_raw/{portfolio_name}_raw_weights.csv"
    if not os.path.exists(weights_path):
        return None, None, None
        
    w_df = pd.read_csv(weights_path)
    w_df['Date'] = pd.to_datetime(w_df['Date'])
    w_df.set_index('Date', inplace=True)
    w_monthly = w_df.resample('ME').last().ffill()
    
    common_idx = df_ret_monthly_all_assets.index.intersection(w_monthly.index)
    w_monthly = w_monthly.loc[common_idx]
    df_ret = df_ret_monthly_all_assets.loc[common_idx]
    
    try:
        from systematic_bl_baseline import ENDOWUS_WEIGHTS
        w_b = pd.Series(ENDOWUS_WEIGHTS)
    except:
        return None, None, None
        
    total_w_b = sum(w_b.get(a, 0) for a in sleeve_assets)
    if total_w_b == 0:
        return None, None, None
        
    llm_ret = pd.Series(0.0, index=common_idx)
    bench_ret = pd.Series(0.0, index=common_idx)
    
    total_w_p = w_monthly[sleeve_assets].sum(axis=1)
    
    for asset in sleeve_assets:
        if asset not in w_monthly.columns or asset not in df_ret.columns: continue
        # Normalize weights strictly within the bucket
        w_p_norm = w_monthly[asset] / total_w_p
        w_b_norm = w_b.get(asset, 0) / total_w_b
        
        llm_ret += w_p_norm * df_ret[asset]
        bench_ret += w_b_norm * df_ret[asset]
        
    active_ret = llm_ret - bench_ret
    active_ret = active_ret.dropna()
    
    if len(active_ret) < 3: return None, None, None
    
    X = sm.add_constant(np.ones(len(active_ret)))
    model = sm.OLS(active_ret, X).fit(cov_type='HAC', cov_kwds={'maxlags': min(3, len(active_ret)-1)})
    
    ar_mean = active_ret.mean()
    t_stat = model.tvalues[0]
    p_val = model.pvalues[0]
    
    return ar_mean, t_stat, p_val

def run_car_test(df_daily, port_name, bench_name="Endowus-60/40"):
    cars = []
    for event, (start_str, end_str) in EVENTS.items():
        start_dt = pd.to_datetime(start_str)
        end_dt = pd.to_datetime(end_str)
        
        mask = (df_daily.index >= start_dt) & (df_daily.index <= end_dt)
        window_df = df_daily.loc[mask]
        
        if len(window_df) == 0:
            continue
            
        p_val = window_df[port_name]
        b_val = window_df[bench_name]
        
        p_ret = (p_val.iloc[-1] / p_val.iloc[0]) - 1
        b_ret = (b_val.iloc[-1] / b_val.iloc[0]) - 1
        
        car = p_ret - b_ret
        cars.append(car)
        
    if not cars:
        return None, None
        
    cars_arr = np.array(cars)
    mean_car = np.mean(cars_arr)
    res = scipy.stats.ttest_1samp(cars_arr, 0, alternative='greater')
    
    return mean_car, res.pvalue

def main():
    print("Loading data...")
    df = pd.read_csv("results/endowus_systematic_equity_curves.csv")
    df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'])
    df.set_index('Unnamed: 0', inplace=True)
    
    df_m = df.resample('ME').last()
    ret_m = df_m.pct_change().dropna()
    
    try:
        from systematic_bl_baseline import get_yfinance_data, EQUITY_ASSETS, FIXED_INCOME_ASSETS
        all_assets = EQUITY_ASSETS + FIXED_INCOME_ASSETS
        df_prices, _ = get_yfinance_data(all_assets, "2019-08-01", "2025-01-01")
        df_ret_monthly_all = df_prices.resample('ME').last().pct_change().dropna()
    except Exception as e:
        print(f"Error loading individual asset data: {e}")
        df_ret_monthly_all = pd.DataFrame()
        EQUITY_ASSETS, FIXED_INCOME_ASSETS = [], []

    r_f = get_risk_free_rate(ret_m.index)
    bench_name = "Endowus_Actual_60/40" if "Endowus_Actual_60/40" in ret_m.columns else "Endowus-60/40"
    r_m = ret_m[bench_name]
    
    portfolios = [c for c in df.columns if c.startswith('LLM-BL-')]
    
    results = []
    
    print("Running advanced tests for each portfolio...")
    for p in portfolios:
        print(f" > Processing {p}...")
        r_p = ret_m[p]
        
        # 1. Market Timing Significance (HM)
        common_idx = r_p.index.intersection(r_m.index).intersection(r_f.index)
        b2, hm_pval = run_hm_test(r_p.loc[common_idx], r_m.loc[common_idx], r_f.loc[common_idx])
        
        # 2. Tail-Risk Reduction (CVaR Bootstrap)
        cvar_p, cvar_m, prob_better = run_cvar_bootstrap(r_p.loc[common_idx], r_m.loc[common_idx])
        
        # 3. Sleeve-Level Alpha (Equity & Fixed Income paired t-tests)
        if not df_ret_monthly_all.empty:
            eq_mean, eq_t, eq_pval = run_sleeve_test(p, df_ret_monthly_all, EQUITY_ASSETS)
            fi_mean, fi_t, fi_pval = run_sleeve_test(p, df_ret_monthly_all, FIXED_INCOME_ASSETS)
        else:
            eq_mean, eq_t, eq_pval = None, None, None
            fi_mean, fi_t, fi_pval = None, None, None
            
        # 4. Event-Study CAR Significance
        mean_car, car_pval = run_car_test(df, p, bench_name=bench_name)
        
        results.append({
            'Portfolio': p,
            'HM Beta2': b2,
            'HM p-val': hm_pval,
            'CVaR(P)': cvar_p,
            'CVaR(B)': cvar_m,
            'CVaR P(P>B)': prob_better,
            'EQ AR Mean': eq_mean,
            'EQ NW t-stat': eq_t,
            'EQ NW p-val': eq_pval,
            'FI AR Mean': fi_mean,
            'FI NW t-stat': fi_t,
            'FI NW p-val': fi_pval,
            'Crisis Mean CAR': mean_car,
            'Crisis CAR p-val': car_pval
        })
        
    df_res = pd.DataFrame(results)
    
    # Generate Markdown Report
    report = "# Advanced Statistical Significance Tests\n\n"
    report += "This report runs four rigorous statistical significance tests across the LLM portfolios to evaluate the validity of the 'Crisis Alpha' thesis, downside protection, and sleeve selection capabilities.\n\n"
    
    # Table 1: Market Timing
    report += "## 1. Market Timing Significance (Henriksson-Merton Test)\n"
    report += "Regression with a dummy indicating a down-market for the benchmark.\n"
    report += "$R_p - R_f = \\alpha + \\beta_1(R_m - R_f) + \\beta_2\\max(0, R_f - R_m) + \\epsilon$\n\n"
    report += "If $\\beta_2 > 0$ and $p < 0.05$, the model statistically reduces downside beta during market shocks.\n\n"
    
    tbl1 = df_res[['Portfolio', 'HM Beta2', 'HM p-val']].copy()
    tbl1['HM Beta2'] = tbl1['HM Beta2'].map('{:+.3f}'.format, na_action='ignore')
    tbl1['HM p-val'] = tbl1['HM p-val'].map('{:.4f}'.format, na_action='ignore')
    report += tbl1.to_markdown(index=False) + "\n\n"
    
    # Table 2: CVaR
    report += "## 2. Tail-Risk Reduction (95% CVaR Bootstrap Difference Test)\n"
    report += "Comparing the 95% Conditional Value at Risk (average return of the worst 5% of months). We run 10,000 bootstrap resamples on paired differences. The `Prob(LLM > Bench)` indicates the percentage of simulated paths where the LLM had a smaller expected shortfall (i.e., empirical p-value for the difference).\n\n"
    
    tbl2 = df_res[['Portfolio', 'CVaR(P)', 'CVaR(B)', 'CVaR P(P>B)']].copy()
    tbl2['CVaR(P)'] = tbl2['CVaR(P)'].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
    tbl2['CVaR(B)'] = tbl2['CVaR(B)'].apply(lambda x: f"{x*100:.2f}%" if pd.notna(x) else "N/A")
    tbl2['CVaR P(P>B)'] = tbl2['CVaR P(P>B)'].apply(lambda x: f"{x*100:.1f}%" if pd.notna(x) else "N/A")
    report += tbl2.to_markdown(index=False) + "\n\n"
    
    # Table 3: Sleeve Alpha
    report += "## 3. Sleeve-Level Alpha (Equity & Fixed Income Paired t-Tests)\n"
    report += "Isolating the active return strictly from instrument selection by stripping out the macro asset-class allocation drift. A significant negative Equity t-stat confirms the Contrarian Bias, while a significant positive FI t-stat confirms yield-curve optimization skill.\n\n"
    
    tbl3 = df_res[['Portfolio', 'EQ AR Mean', 'EQ NW t-stat', 'EQ NW p-val', 'FI AR Mean', 'FI NW t-stat', 'FI NW p-val']].copy()
    
    for c in ['EQ AR Mean', 'FI AR Mean']:
        tbl3[c] = tbl3[c].apply(lambda x: f"{x*100:+.3f}%" if pd.notna(x) else "N/A")
    for c in ['EQ NW t-stat', 'FI NW t-stat']:
        tbl3[c] = tbl3[c].apply(lambda x: f"{x:.3f}" if pd.notna(x) else "N/A")
    for c in ['EQ NW p-val', 'FI NW p-val']:
        tbl3[c] = tbl3[c].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
        
    report += tbl3.to_markdown(index=False) + "\n\n"
    
    # Table 4: Event
    report += "## 4. Event-Study Cumulative Abnormal Return (CAR)\n"
    report += "Evaluating the 6 specific crisis epochs. Is the Cumulative Abnormal Return across the 6 events statistically greater than zero? (1-sample, 1-tailed t-test).\n\n"
    
    tbl4 = df_res[['Portfolio', 'Crisis Mean CAR', 'Crisis CAR p-val']].copy()
    tbl4['Crisis Mean CAR'] = tbl4['Crisis Mean CAR'].apply(lambda x: f"{x*100:+.2f}%" if pd.notna(x) else "N/A")
    tbl4['Crisis CAR p-val'] = tbl4['Crisis CAR p-val'].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    report += tbl4.to_markdown(index=False) + "\n\n"
    
    with open("results/advanced_significance_tests.md", "w") as f:
        f.write(report)
    print("Done! Check results/advanced_significance_tests.md")

if __name__ == "__main__":
    main()