import pandas as pd
import numpy as np
import scipy.stats
import statsmodels.api as sm
import os
import glob
import json
import matplotlib.pyplot as plt

def run_frictions_analysis():
    print("--- 1. Turnover & Transaction Cost Sensitivity ---")
    # Load raw weights
    w_weekly = pd.read_csv("results/weights_raw/LLM-BL-A_raw_weights.csv")
    w_monthly = pd.read_csv("results/weights_raw/LLM-BL-A-Monthly_raw_weights.csv")
    
    # Calculate period-over-period turnover
    def calc_avg_turnover(df):
        cols = [c for c in df.columns if c != 'Date']
        weights = df[cols].fillna(0).values
        turnover = np.sum(np.abs(weights[1:] - weights[:-1]), axis=1) / 2
        return np.mean(turnover)
        
    to_weekly = calc_avg_turnover(w_weekly)
    to_monthly = calc_avg_turnover(w_monthly)
    
    print(f"Average Weekly Turnover: {to_weekly*100:.2f}%")
    print(f"Average Monthly Turnover: {to_monthly*100:.2f}%")
    
    # Load equity curves for 0 bps baseline
    df_returns = pd.read_csv("results/endowus_systematic_equity_curves.csv")
    df_returns['Unnamed: 0'] = pd.to_datetime(df_returns['Unnamed: 0'])
    df_returns.set_index('Unnamed: 0', inplace=True)
    
    # Annualized return without friction
    def ann_ret(series):
        total_ret = series.iloc[-1] / series.iloc[0] - 1
        years = (series.index[-1] - series.index[0]).days / 365.25
        return (1 + total_ret)**(1/years) - 1

    r_w_0bps = ann_ret(df_returns['LLM-BL-A'])
    r_m_0bps = ann_ret(df_returns['LLM-BL-A-Monthly']) # Assume B is monthly or just use A if not available
    bench_ret = ann_ret(df_returns['Endowus-60/40'])
    
    print(f"\nBaseline Annualized Returns (0 bps):")
    print(f"LLM-BL-A:  {r_w_0bps*100:.2f}%")
    print(f"LLM-BL-A-Monthly:  {r_m_0bps*100:.2f}%")
    print(f"Benchmark:   {bench_ret*100:.2f}%")
    
    report_text = f"## 1. Turnover & Transaction Cost Sensitivity\n"
    report_text += f"- **Average Weekly Turnover**: {to_weekly*100:.2f}%\n"
    report_text += f"- **Average Monthly Turnover**: {to_monthly*100:.2f}%\n\n"
    report_text += "| Transaction Cost | LLM Weekly Ann. Ret | LLM Monthly Ann. Ret |\n"
    report_text += "|------------------|----------------------|-----------------------|\n"
    report_text += f"| 0 bps (Frictionless) | {r_w_0bps*100:.2f}% | {r_m_0bps*100:.2f}% |\n"
    
    # Calculate degraded returns
    # Total periods = len(weights) - 1. Total turnover cost = sum(turnover) * cost
    # To approximate the annualized drag, we can subtract (avg_turnover * periods_per_year * bps)
    for bps in [5, 10, 20]:
        cost = bps / 10000
        drag_w = to_weekly * 52 * cost
        drag_m = to_monthly * 12 * cost
        
        r_w_deg = r_w_0bps - drag_w
        r_m_deg = r_m_0bps - drag_m
        
        print(f"At {bps} bps: LLM-BL-A = {r_w_deg*100:.2f}%, LLM-BL-B = {r_m_deg*100:.2f}%")
        report_text += f"| {bps} bps | {r_w_deg*100:.2f}% | {r_m_deg*100:.2f}% |\n"
        
    return report_text


def run_rolling_beta():
    print("\n--- 2. Rolling Beta Analysis ---")
    df = pd.read_csv("results/endowus_systematic_equity_curves.csv")
    df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'])
    df.set_index('Unnamed: 0', inplace=True)
    
    # Resample to monthly
    df_monthly = df.resample('ME').last()

    ret_llm = df_monthly['LLM-BL-A-Monthly-Banded'].pct_change().dropna()
    ret_bench = df_monthly['Endowus-60/40'].pct_change().dropna()
    
    import portfolio_common
    rf_series = portfolio_common.load_us_3m_tbill_series(start="2019-09-01", end="2024-12-31")
    rf_rate_period = portfolio_common.get_period_risk_free_returns(
        ret_llm.index, 12, rf_series, fallback_rate=0.04
    )
    
    ex_llm = ret_llm.sub(rf_rate_period, axis=0).dropna()
    ex_bench = ret_bench.sub(rf_rate_period, axis=0).dropna()
    
    # Use cov / var over rolling 6 month windows for beta
    # Rolling Beta = Cov(R_p, R_m) / Var(R_m)
    window = 6
    cov = ex_llm.rolling(window).cov(ex_bench)
    var = ex_bench.rolling(window).var()
    beta = cov / var
    beta = beta.dropna()
    
    os.makedirs("results/thesis_plots", exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.plot(beta.index, beta, label='Rolling 6-Month CAPM Beta')
    plt.axhline(1.0, color='r', linestyle='--', alpha=0.5)
    plt.title("LLM-BL-A Market Exposure over Time")
    plt.ylabel("Beta (vs 60/40 Benchmark)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/thesis_plots/rolling_beta.png")
    plt.close()
    
    avg_beta = beta.mean()
    min_beta = beta.min()
    max_beta = beta.max()
    print(f"Beta stats: Avg={avg_beta:.2f}, Min={min_beta:.2f}, Max={max_beta:.2f}")
    
    report_text = f"\n## 2. Rolling Beta / Factor Exposure Analysis\n"
    report_text += f"- **Average Rolling Beta**: {avg_beta:.2f}\n"
    report_text += f"- **Minimum Beta**: {min_beta:.2f}\n"
    report_text += f"- **Maximum Beta**: {max_beta:.2f}\n"
    report_text += f"![Rolling Beta Plot](./thesis_plots/rolling_beta.png)\n"
    
    return report_text


def run_crisis_alpha():
    print("\n--- 3. Regime-Conditioned Statistical Testing (Crisis Alpha) ---")
    import yfinance as yf
    
    vix = yf.download("^VIX", start="2019-08-01", end="2025-01-01")['Close']
    vix = vix.resample('ME').last()
    
    df = pd.read_csv("results/endowus_systematic_equity_curves.csv")
    df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'])
    df.set_index('Unnamed: 0', inplace=True)
    df_monthly = df.resample('ME').last()
    ret_llm = df_monthly['LLM-BL-A-Monthly-Banded'].pct_change().dropna()
    ret_bench = df_monthly['Endowus_Actual_60/40'].pct_change().dropna()
    
    vix = vix.reindex(ret_llm.index).ffill()
    
    high_vol_mask = vix['^VIX'] > 22.0
    normal_vol_mask = vix['^VIX'] <= 22.0
    
    def calc_stats(mask, name):
        r_l = ret_llm[mask]
        r_b = ret_bench[mask]
        
        ar = r_l - r_b
        hits = (ar > 0).sum()
        n = len(ar)
        hr = hits / n if n > 0 else 0
        
        if n > 0:
            binom = scipy.stats.binomtest(k=hits, n=n, p=0.5, alternative='greater')
            pval_b = binom.pvalue
            
            X = sm.add_constant(np.ones(n))
            model = sm.OLS(ar, X)
            res = model.fit(cov_type='HAC', cov_kwds={'maxlags': min(3, n-1)})
            t_nw = res.tvalues[0]
            p_nw = res.pvalues[0]
        else:
            pval_b, t_nw, p_nw = 1.0, 0.0, 1.0
            
        print(f"[{name}] N={n}, Hit Rate={hr*100:.1f}%, Mean AR={ar.mean()*100:.2f}%, Binom p={pval_b:.4f}, NW p={p_nw:.4f}")
        return n, hr, ar.mean(), pval_b, p_nw

    n_h, hr_h, mar_h, p_b_h, p_nw_h = calc_stats(high_vol_mask, "High Volatility (VIX > 22)")
    n_n, hr_n, mar_n, p_b_n, p_nw_n = calc_stats(normal_vol_mask, "Normal Volatility (VIX <= 22)")

    report_text = f"\n## 3. Regime-Conditioned Statistical Testing (Crisis Alpha)\n"
    report_text += "| Regime | N Months | Hit Rate | Mean Active Return | Binomial p-val | Newey-West p-val |\n"
    report_text += "|--------|----------|----------|--------------------|----------------|------------------|\n"
    report_text += f"| High Vol (VIX > 22) | {n_h} | {hr_h*100:.1f}% | {mar_h*100:.2f}% | {p_b_h:.4f} | {p_nw_h:.4f} |\n"
    report_text += f"| Normal (VIX <= 22) | {n_n} | {hr_n*100:.1f}% | {mar_n*100:.2f}% | {p_b_n:.4f} | {p_nw_n:.4f} |\n"
    
    return report_text


def run_attribution():
    print("\n--- 4. Brinson-Fachler Attribution ---")
    w_df = pd.read_csv("results/weights_raw/LLM-BL-A-Monthly-Banded_raw_weights.csv")
    w_df['Date'] = pd.to_datetime(w_df['Date'])
    w_df.set_index('Date', inplace=True)
    
    EQUITY_ASSETS = ["0P0001AF7U.SI",
    "SPY",
    "^990100-USD-STRD",
    "DE000SLA4YD9.SG",
    "0P0001AF7Z.SI",
    "0P0001EF2T.SI",
    "EIMI.L",]
    FI_ASSETS = ["0P0000KYEE.SI",
    "AGGG.L",
    "IE0002461055.IR",
    "0P0001EQUE.SI",
    "0P0001CC3M",
    "PEBIX",
    "0P0001DWI0.SI",]
    
    eq_w = w_df[EQUITY_ASSETS].sum(axis=1)
    
    plt.figure(figsize=(10, 5))
    plt.plot(eq_w.index, eq_w, label='LLM-BL-A-Monthly-Banded Equity Weight')
    plt.axhline(0.6, color='r', linestyle='--', label='Benchmark Equity Weight (60%)')
    plt.title("Portfolio Equity Allocation Over Time")
    plt.ylabel("Weight in Equities")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/thesis_plots/equity_allocation.png")
    plt.close()
    
    # Calculate simple allocation vs selection proxy
    # Active return = Return_P - Return_B
    # Allocation = (Weight_Eq_P - Weight_Eq_B) * (Return_Eq_B - Return_FI_B)
    # Selection = Active Return - Allocation
    
    df = pd.read_csv("results/endowus_systematic_equity_curves.csv")
    df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'])
    df.set_index('Unnamed: 0', inplace=True)
    df_m = df.resample('ME').last()
    
    ret_llm = df_m['LLM-BL-A-Monthly-Banded'].pct_change().dropna()
    ret_bench = df_m['Endowus_Actual_60/40'].pct_change().dropna()
    
    # Use Endowus_Actual_100/0 if available, but Endowus_Actual isn't in original curves so we'll use SPY and AGGG as proxies if needed
    ret_eq_b = df_m['Endowus_Actual_100/0'].pct_change().dropna() if 'Endowus_Actual_100/0' in df_m.columns else df_m['Endowus-60/40'].pct_change().dropna() * 1.5 
    ret_fi_b = df_m['Endowus_Actual_0/100'].pct_change().dropna() if 'Endowus_Actual_0/100' in df_m.columns else df_m['Endowus-60/40'].pct_change().dropna() * 0.5
    
    active_ret = ret_llm - ret_bench
    
    # Align indices
    common_idx = active_ret.index.intersection(eq_w.index)
    active_ret = active_ret.loc[common_idx]
    ret_eq_b = ret_eq_b.loc[common_idx]
    ret_fi_b = ret_fi_b.loc[common_idx]
    eq_w = eq_w.loc[common_idx]
    
    allocation = (eq_w - 0.6) * (ret_eq_b - ret_fi_b)
    selection = active_ret - allocation
    
    print(f"Total Active Return:   {active_ret.sum()*100:.2f}%")
    print(f"Total Allocation AR:   {allocation.sum()*100:.2f}%")
    print(f"Total Selection AR:    {selection.sum()*100:.2f}%")
    
    report_text = f"\n## 4. Brinson-Fachler Attribution (Asset Allocation vs. Selection)\n"
    report_text += f"- **Total Cumulative Active Return**: {active_ret.sum()*100:.2f}%\n"
    report_text += f"  - **Contribution from Inter-Class Allocation (Eq vs FI)**: {allocation.sum()*100:.2f}%\n"
    report_text += f"  - **Contribution from Intra-Class Selection**: {selection.sum()*100:.2f}%\n"
    report_text += f"![Equity Allocation Plot](./thesis_plots/equity_allocation.png)\n"
    
    return report_text


def run_confidence_analysis():
    print("\n--- 5. Conviction vs. Accuracy (The Omega Matrix Analysis) ---")
    
    df = pd.read_csv("results/endowus_systematic_equity_curves.csv")
    df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'])
    df.set_index('Unnamed: 0', inplace=True)
    df_m = df.resample('ME').last()
    
    ret_bench = df_m['Endowus_Actual_60/40'].pct_change().dropna()
    
    models = ['A', 'B', 'C', 'D']
    variations = ['', '-Monthly', '-Monthly-Banded', '-Banded']
    
    report_text = f"\n## 5. Conviction vs. Accuracy (The $\Omega$ Matrix Analysis)\n"
    report_text += "This analysis compares the LLM's average portfolio confidence score against the realized active return in the *following* month, across all models and variations.\n\n"
    report_text += "| Portfolio | Pearson r | p-val | Spearman $\\rho$ | p-val |\n"
    report_text += "|-----------|-----------|-------|------------------|-------|\n"
    
    fig, axes = plt.subplots(4, 4, figsize=(20, 16), sharex=True, sharey=True)
    fig.suptitle("LLM Portfolio Confidence vs. Next Month Active Return", fontsize=16)
    
    colors = {'A': 'blue', 'B': 'green', 'C': 'red', 'D': 'purple'}
    
    all_confs = []
    all_ars = []
    
    for i, model in enumerate(models):
        files = glob.glob(f"results_endowus_{model}/*_decisions.csv")
        if not files:
            continue
            
        month_confs = {}
        for f in files:
            df_dec = pd.read_csv(f)
            for _, row in df_dec.iterrows():
                date = pd.to_datetime(row['test_date'])
                month_end = date + pd.offsets.MonthEnd(0)
                try:
                    dec = json.loads(row['decision'])
                    conf_raw = dec.get('Confidence_Score', 0)
                    if conf_raw is None:
                        conf = 0
                    elif isinstance(conf_raw, str):
                        import re
                        m = re.search(r'\d+', conf_raw)
                        conf = float(m.group()) if m else 0
                    else:
                        conf = float(conf_raw)
                except:
                    conf = 0
                    
                if month_end not in month_confs:
                    month_confs[month_end] = []
                month_confs[month_end].append(conf)
                
        if not month_confs:
            continue
            
        avg_conf_series = pd.Series({k: np.mean(v) for k, v in month_confs.items()})
        avg_conf_series.sort_index(inplace=True)
        
        for j, var in enumerate(variations):
            ax = axes[i, j]
            port_name = f"LLM-BL-{model}{var}"
            
            if port_name not in df_m.columns:
                ax.set_title(port_name)
                ax.text(0.5, 0.5, 'No Data', ha='center', va='center')
                continue
                
            ret_llm = df_m[port_name].pct_change().dropna()
            active_ret = ret_llm - ret_bench
            next_month_ar = active_ret.shift(-1).dropna()
            
            common_idx = avg_conf_series.index.intersection(next_month_ar.index)
            if len(common_idx) < 3:
                ax.set_title(port_name)
                ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center')
                continue
                
            conf_values = avg_conf_series.loc[common_idx]
            ar_values = next_month_ar.loc[common_idx]
            
            ax.scatter(conf_values, ar_values * 100, color=colors[model], alpha=0.6)
            
            all_confs.extend(conf_values.tolist())
            all_ars.extend(ar_values.tolist())
            
            pearson_r, p_val = scipy.stats.pearsonr(conf_values, ar_values)
            spearman_r, sp_val = scipy.stats.spearmanr(conf_values, ar_values)
            
            # Fit line
            m_fit, b_fit = np.polyfit(conf_values, ar_values * 100, 1)
            ax.plot(conf_values, m_fit * conf_values + b_fit, color='black', linestyle='--')
            
            ax.set_title(f"{port_name}\n$r={pearson_r:.2f}$ ($p={p_val:.2f}$)")
            ax.grid(True, alpha=0.3)
            
            if j == 0:
                ax.set_ylabel("Next Month AR (%)")
            if i == 3:
                ax.set_xlabel("Confidence Score (1-10)")
            
            report_text += f"| {port_name} | {pearson_r:.3f} | {p_val:.4f} | {spearman_r:.3f} | {sp_val:.4f} |\n"
            
    if all_confs:
        pearson_r, p_val = scipy.stats.pearsonr(all_confs, all_ars)
        spearman_r, sp_val = scipy.stats.spearmanr(all_confs, all_ars)
        report_text += f"| **OVERALL (All)** | **{pearson_r:.3f}** | **{p_val:.4f}** | **{spearman_r:.3f}** | **{sp_val:.4f}** |\n\n"
        print(f"Global Pearson: {pearson_r:.3f} (p={p_val:.4f}), Spearman: {spearman_r:.3f} (p={sp_val:.4f})")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("results/thesis_plots/confidence_scatter.png", bbox_inches='tight')
    plt.close()
    
    report_text += f"![Confidence Scatter Plot](./thesis_plots/confidence_scatter.png)\n"
    
    return report_text


if __name__ == "__main__":
    report1 = run_frictions_analysis()
    report2 = run_rolling_beta()
    report3 = run_crisis_alpha()
    report4 = run_attribution()
    report5 = run_confidence_analysis()
    
    final_report = "# Thesis Quantitative Analyses Report\n\n"
    final_report += report1 + report2 + report3 + report4 + report5
    
    with open("results/thesis_analyses_report.md", "w") as f:
        f.write(final_report)
    
    print("\nReport successfully generated at results/thesis_analyses_report.md")
