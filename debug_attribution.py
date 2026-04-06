import pandas as pd
import numpy as np

def debug_attribution(model_name, weight_file):
    print(f"\n--- Brinson-Fachler Attribution for {model_name} ---")
    w_df = pd.read_csv(weight_file)
    w_df['Date'] = pd.to_datetime(w_df['Date'])
    w_df.set_index('Date', inplace=True)
    
    EQUITY_ASSETS = ['0P0001AF7U.SI', '0P0000KYEE.SI', 'SPY', '^990100-USD-STRD', 'DE000SLA4YD9.SG']
    # Check which assets are actually in the columns
    actual_eq_assets = [c for c in EQUITY_ASSETS if c in w_df.columns]
    eq_w = w_df[actual_eq_assets].sum(axis=1)
    
    df = pd.read_csv("results/endowus_systematic_equity_curves.csv")
    df['Unnamed: 0'] = pd.to_datetime(df['Unnamed: 0'])
    df.set_index('Unnamed: 0', inplace=True)
    df_m = df.resample('ME').last()
    
    ret_llm = df_m[model_name].pct_change().dropna()
    ret_bench = df_m['Endowus-60/40'].pct_change().dropna()
    
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

if __name__ == "__main__":
    debug_attribution('LLM-BL-A', 'results/weights_raw/LLM-BL-A_raw_weights.csv')
    debug_attribution('LLM-BL-A-Monthly', 'results/weights_raw/LLM-BL-A-Monthly_raw_weights.csv')
    debug_attribution('LLM-BL-A-Monthly-Banded', 'results/weights_raw/LLM-BL-A-Monthly-Banded_raw_weights.csv')

