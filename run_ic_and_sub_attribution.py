import pandas as pd
import numpy as np
import scipy.stats
import glob
import json
import os
import matplotlib.pyplot as plt

def run_rank_ic():
    print("--- 1. Cross-Sectional Information Coefficient (Rank IC) ---")
    
    models = ['A', 'B', 'C', 'D']
    try:
        from systematic_bl_baseline import get_yfinance_data, EQUITY_ASSETS, FIXED_INCOME_ASSETS
        all_assets = EQUITY_ASSETS + FIXED_INCOME_ASSETS
        df_prices_close, _ = get_yfinance_data(all_assets, "2019-08-01", "2025-01-01")
    except Exception as e:
        print(f"Error loading prices: {e}")
        return ""
        
    df_monthly_prices = df_prices_close.resample('ME').last()
    df_fwd_returns = df_monthly_prices.pct_change().shift(-1)
    
    report_text = "\n## 1. Cross-Sectional Information Coefficient (Rank IC)\n"
    report_text += "Evaluates the LLM's raw predictive power by ranking its 30-day Target Returns against actual realized 1-month forward returns for the 14 funds.\n\n"
    report_text += "| Model | Mean Rank IC | IC > 0 (Hit Rate) | T-Stat | p-value |\n"
    report_text += "|-------|--------------|-------------------|--------|---------|\n"

    for model in models:
        files = glob.glob(f"results_endowus_{model}/*_decisions.csv")
        if not files:
            continue
            
        expected_returns = {}
        
        for f in files:
            ticker = os.path.basename(f).split('_')[0]
            if ticker not in all_assets:
                continue
            df_dec = pd.read_csv(f)
            
            for _, row in df_dec.iterrows():
                date = pd.to_datetime(row['test_date'])
                month_end = date + pd.offsets.MonthEnd(0)
                
                try:
                    dec = json.loads(row['decision'])
                    tr_str = dec.get('Target_Return_30d', '0%')
                    if isinstance(tr_str, str):
                        tr = float(tr_str.replace('%', '')) / 100.0
                    else:
                        tr = float(tr_str)
                except:
                    tr = 0.0
                    
                if month_end not in expected_returns:
                    expected_returns[month_end] = {}
                expected_returns[month_end][ticker] = tr
                
        ic_series = []
        
        for me in sorted(expected_returns.keys()):
            preds = expected_returns[me]
            
            if me not in df_fwd_returns.index:
                continue
            actuals = df_fwd_returns.loc[me]
            
            valid_tickers = []
            valid_preds = []
            valid_acts = []
            
            for t in all_assets:
                if t in preds and pd.notna(actuals.get(t)):
                    valid_tickers.append(t)
                    valid_preds.append(preds[t])
                    valid_acts.append(actuals[t])
                    
            if len(valid_tickers) > 5:
                # Add tiny random noise to breaking ties in expected returns if they output 0% for everything
                noise = np.random.normal(0, 1e-8, len(valid_preds))
                valid_preds = np.array(valid_preds) + noise
                
                r, _ = scipy.stats.spearmanr(valid_preds, valid_acts)
                if pd.notna(r):
                    ic_series.append(r)
                    
        if ic_series:
            ic_array = np.array(ic_series)
            mean_ic = np.mean(ic_array)
            hit_rate = np.mean(ic_array > 0)
            
            t_stat, p_val = scipy.stats.ttest_1samp(ic_array, 0)
            
            print(f"Model {model} - Mean IC: {mean_ic:.3f}, Hit Rate: {hit_rate:.1%}, T: {t_stat:.2f} (p={p_val:.3f})")
            report_text += f"| LLM-BL-{model} | {mean_ic:.3f} | {hit_rate:.1%} | {t_stat:.2f} | {p_val:.4f} |\n"
            
    return report_text

def run_sub_asset_attribution():
    print("\n--- 2. Sub-Asset Selection Attribution ---")
    
    try:
        from systematic_bl_baseline import ENDOWUS_WEIGHTS, EQUITY_ASSETS, FIXED_INCOME_ASSETS
    except ImportError:
        print("Cannot import ENDOWUS_WEIGHTS")
        return ""
        
    portfolio_name = "LLM-BL-A-Monthly-Banded"
    weights_path = f"results/weights_raw/{portfolio_name}_raw_weights.csv"
    
    if not os.path.exists(weights_path):
        print(f"Weights file not found: {weights_path}")
        return ""
        
    w_df = pd.read_csv(weights_path)
    w_df['Date'] = pd.to_datetime(w_df['Date'])
    w_df.set_index('Date', inplace=True)
    
    w_monthly = w_df.resample('ME').last().ffill()
    
    from systematic_bl_baseline import get_yfinance_data
    all_assets = EQUITY_ASSETS + FIXED_INCOME_ASSETS
    df_prices, _ = get_yfinance_data(all_assets, "2019-08-01", "2025-01-01")
    df_p_monthly = df_prices.resample('ME').last()
    df_ret = df_p_monthly.pct_change().dropna()
    
    w_b = pd.Series(ENDOWUS_WEIGHTS)
    
    common_idx = df_ret.index.intersection(w_monthly.index)
    w_monthly = w_monthly.loc[common_idx]
    df_ret = df_ret.loc[common_idx]
    
    df_curves = pd.read_csv("results/endowus_systematic_equity_curves.csv")
    df_curves['Unnamed: 0'] = pd.to_datetime(df_curves['Unnamed: 0'])
    df_curves.set_index('Unnamed: 0', inplace=True)
    df_curves_m = df_curves.resample('ME').last()
    
    if 'Endowus_Actual_100/0' in df_curves_m.columns:
        ret_bench_eq = df_curves_m['Endowus_Actual_100/0'].pct_change().dropna()
        ret_bench_fi = df_curves_m['Endowus_Actual_0/100'].pct_change().dropna()
    else:
        print("Endowus_Actual benchmarks not found in curves!")
        return ""
        
    ret_bench_eq = ret_bench_eq.reindex(common_idx).ffill()
    ret_bench_fi = ret_bench_fi.reindex(common_idx).ffill()
    
    total_selection_equity = 0
    total_selection_fi = 0
    
    eq_summary = []
    total_eq_w_p = w_monthly[EQUITY_ASSETS].sum(axis=1)
    total_eq_w_b = sum(w_b.get(a, 0) for a in EQUITY_ASSETS) # Should be 0.60
    
    for asset in EQUITY_ASSETS:
        if asset not in w_monthly.columns or asset not in df_ret.columns: continue
        # Normalize weights inside the bucket
        w_p_norm = w_monthly[asset] / total_eq_w_p
        w_b_norm = w_b.get(asset, 0) / total_eq_w_b
        
        # Pure intra-class active weight
        active_weight_norm = w_p_norm - w_b_norm
        
        # Contribution = Benchmark Class Weight (0.60) * (Active Normalized Weight * Asset Return)
        contrib = total_eq_w_b * (active_weight_norm * df_ret[asset])
        
        total_contrib = contrib.sum()
        avg_act_weight = active_weight_norm.mean()
        
        eq_summary.append({
            'Asset': asset,
            'Avg Active Bucket Weight': avg_act_weight,
            'Selection Contribution': total_contrib
        })
        total_selection_equity += total_contrib
        
    fi_summary = []
    total_fi_w_p = w_monthly[FIXED_INCOME_ASSETS].sum(axis=1)
    total_fi_w_b = sum(w_b.get(a, 0) for a in FIXED_INCOME_ASSETS) # Should be 0.40
    
    for asset in FIXED_INCOME_ASSETS:
        if asset not in w_monthly.columns or asset not in df_ret.columns: continue
        w_p_norm = w_monthly[asset] / total_fi_w_p
        w_b_norm = w_b.get(asset, 0) / total_fi_w_b
        
        active_weight_norm = w_p_norm - w_b_norm
        
        # Contribution = Benchmark Class Weight (0.40) * (Active Normalized Weight * Asset Return)
        contrib = total_fi_w_b * (active_weight_norm * df_ret[asset])
        
        total_contrib = contrib.sum()
        avg_act_weight = active_weight_norm.mean()
        
        fi_summary.append({
            'Asset': asset,
            'Avg Active Bucket Weight': avg_act_weight,
            'Selection Contribution': total_contrib
        })
        total_selection_fi += total_contrib
        
    eq_df = pd.DataFrame(eq_summary).sort_values('Selection Contribution', ascending=False)
    fi_df = pd.DataFrame(fi_summary).sort_values('Selection Contribution', ascending=False)
    
    report = f"\n## 2. Sub-Asset Attribution ({portfolio_name})\n"
    report += "Breaks down the Intra-Class Selection effect into individual ticker contributions by normalizing the sub-asset weights strictly within their asset class buckets (Equity/FI) and sizing by the benchmark class weight. This isolates *pure* asset selection independent of the portfolio's top-level macro allocation drift.\n\n"
    report += f"**Total Equity Selection Contribution:** {total_selection_equity*100:.2f}%\n\n"
    report += "| Equity Asset | Avg Active Bucket Weight | Selection Contribution |\n"
    report += "|--------------|--------------------------|------------------------|\n"
    for _, row in eq_df.iterrows():
        report += f"| {row['Asset']} | {row['Avg Active Bucket Weight']*100:+.2f}% | {row['Selection Contribution']*100:+.2f}% |\n"
        
    report += f"\n**Total FI Selection Contribution:** {total_selection_fi*100:.2f}%\n\n"
    report += "| Fixed Income Asset | Avg Active Bucket Weight | Selection Contribution |\n"
    report += "|--------------------|--------------------------|------------------------|\n"
    for _, row in fi_df.iterrows():
        report += f"| {row['Asset']} | {row['Avg Active Bucket Weight']*100:+.2f}% | {row['Selection Contribution']*100:+.2f}% |\n"
        
    return report

if __name__ == '__main__':
    r1 = run_rank_ic()
    r2 = run_sub_asset_attribution()
    
    res = "# LLM Rank IC and Sub-Asset Attribution\n" + r1 + r2
    with open("results/ic_and_sub_attribution_report.md", "w") as f:
        f.write(res)
    print("\nReport written to results/ic_and_sub_attribution_report.md")
