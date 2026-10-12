#!/usr/bin/env python3
"""Camera-ready (reviewer 4, item 4): attribution with correctly lagged weights.

The weights logged by `systematic_bl_baseline.py` at date t are the targets SET
at t (executed at the next open) and held into the following month. The
published attribution (`run_thesis_analyses.py` section 4 and
`run_ic_and_sub_attribution.py`) paired the weights logged at month-end t with
the return realised over the month ENDING at t -- a one-month look-ahead. It also
dropped December 2024 (no weight row is logged after the last rebalance),
measured against the live product's track record (`Endowus_Actual_60/40`,
Sharpe 0.455) rather than the simulated passive the paper reports, and
reported the arithmetic SUM of monthly active returns as a "cumulative" figure.

This script recomputes everything with weights set at t-1 applied to the return
over month t, over all 63 months, and reports both the arithmetic sum (which the
attribution terms add up to) and the compounded terminal-wealth gap.

Replay-only: reads stored curves, weights and the cached price file.
"""
import ast

import pandas as pd


def _engine_constants(*names):
    # Parse the constants rather than import the engine, which pulls in cvxpy.
    tree = ast.parse(open("systematic_bl_baseline.py").read())
    found = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in names:
                found[node.targets[0].id] = ast.literal_eval(node.value)
    return [found[n] for n in names]


ENDOWUS_WEIGHTS, EQUITY_ASSETS, FIXED_INCOME_ASSETS = _engine_constants(
    "ENDOWUS_WEIGHTS", "EQUITY_ASSETS", "FIXED_INCOME_ASSETS")

CFG = "LLM-BL-A-Monthly-Banded"
BENCH = "Endowus-60/40"  # the simulated 14-line passive the paper reports (Sharpe 0.411)
PRICES = "data/yfinance_close_0P0001AF7U.SI_0P0000KYEE.SI_SP.csv"
RUNS = {"GLM": "results_v2", "GPT": "results_gpt54_v2"}
COVID = ("2020-02-01", "2020-05-31")


def attribution(results_dir: str, px: pd.DataFrame) -> dict:
    curves = pd.read_csv(f"{results_dir}/endowus_systematic_equity_curves.csv",
                         index_col=0, parse_dates=True)
    w = pd.read_csv(f"{results_dir}/weights_raw/{CFG}_raw_weights.csv",
                    parse_dates=["Date"]).set_index("Date")
    cm = curves.resample("ME").last()
    r = cm.pct_change().dropna()
    active = r[CFG] - r[BENCH]
    idx = active.index

    # Weights set at the end of month t-1 are held over month t.
    w_held = w.resample("ME").last().shift(1).reindex(idx).ffill()
    eq_w = w_held[EQUITY_ASSETS].sum(axis=1)
    fi_w = w_held[FIXED_INCOME_ASSETS].sum(axis=1)

    # Benchmark sleeve returns from the same 14 lines at benchmark weights
    # within each sleeve, so allocation is measured against the passive the
    # paper reports rather than the live product's track record.
    ra = px.ffill().resample("ME").last().pct_change().reindex(idx)
    wb = pd.Series(ENDOWUS_WEIGHTS)
    r_eq = (ra[EQUITY_ASSETS] * (wb[EQUITY_ASSETS] / wb[EQUITY_ASSETS].sum())).sum(axis=1)
    r_fi = (ra[FIXED_INCOME_ASSETS] * (wb[FIXED_INCOME_ASSETS] / wb[FIXED_INCOME_ASSETS].sum())).sum(axis=1)
    alloc = (eq_w - 0.6) * (r_eq - r_fi)
    resid = active - alloc

    # Within-sleeve selection, normalised inside each sleeve and sized by the
    # benchmark sleeve weight (excludes allocation drift by construction).
    sel = {}
    for name, assets, sleeve_w in [("eq", EQUITY_ASSETS, eq_w), ("fi", FIXED_INCOME_ASSETS, fi_w)]:
        wb_sleeve = wb[assets].sum()
        rel = w_held[assets].div(sleeve_w, axis=0) - wb[assets] / wb_sleeve
        sel[name] = float((wb_sleeve * (rel * ra[assets]).sum(axis=1)).sum())

    s, b = curves[CFG], curves[BENCH]
    covid = (idx >= COVID[0]) & (idx <= COVID[1])
    return {
        "months": len(idx),
        "sum_active": active.sum(),
        "alloc": alloc.sum(),
        "resid": resid.sum(),
        "sel_eq": sel["eq"],
        "sel_fi": sel["fi"],
        "wealth_gap": s.iloc[-1] / s.iloc[0] - b.iloc[-1] / b.iloc[0],
        "mean_eq_w": eq_w.mean(),
        "covid_active": active[covid].sum(),
        "rest_active": active[~covid].sum(),
        "covid_months": int(covid.sum()),
    }


def main():
    px = pd.read_csv(PRICES, index_col=0, parse_dates=True)
    out = {k: attribution(v, px) for k, v in RUNS.items()}
    for k, a in out.items():
        print(f"{k} ({CFG}, {a['months']} months, weights lagged one month)")
        print(f"  sum of monthly active   {a['sum_active']*100:+.2f}%")
        print(f"    allocation (eq vs FI) {a['alloc']*100:+.2f}%")
        print(f"    residual              {a['resid']*100:+.2f}%")
        print(f"  sleeve selection eq/fi  {a['sel_eq']*100:+.2f}% / {a['sel_fi']*100:+.2f}%")
        print(f"  terminal-wealth gap     {a['wealth_gap']*100:+.2f}pp (compounded)")
        print(f"  mean held equity weight {a['mean_eq_w']:.3f}")
        print(f"  Feb-May 2020 active sum {a['covid_active']*100:+.2f}pp over {a['covid_months']} months;"
              f" other months {a['rest_active']*100:+.2f}pp")
    g, f = out["GLM"], out["GPT"]
    print("\nCross-stack (GLM minus GPT), arithmetic sums")
    print(f"  total gap  {(g['sum_active']-f['sum_active'])*100:+.2f}pp")
    print(f"  allocation {(g['alloc']-f['alloc'])*100:+.2f}pp")
    print(f"  residual   {(g['resid']-f['resid'])*100:+.2f}pp")


if __name__ == "__main__":
    main()
