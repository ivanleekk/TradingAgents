#!/usr/bin/env python3
"""Stationary block bootstrap for the cross-stack Sharpe difference.

Replay-only: reads the stored equity curves for both stacks and the cached
risk-free series. No LLM calls, no network.

The Jobson--Korkie test reported in the paper is asymptotic and assumes iid
returns. This script relaxes both assumptions: a stationary bootstrap
(Politis & Romano 1994) resamples geometric-length blocks, preserving serial
dependence, and the two stacks are resampled with the *same* block indices so
their cross-correlation (rho ~ 0.87) is retained. The resulting interval is on
the Sharpe *difference*, which is the quantity the paper's central claim rests
on and which is far less sensitive to the risk-free convention than the levels.

Usage:
    python run_sharpe_bootstrap.py
    python run_sharpe_bootstrap.py --reps 20000 --strategy LLM-BL-A-Monthly-Banded
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

GLM_CURVES = "results_v2/endowus_systematic_equity_curves.csv"
GPT_CURVES = "results_gpt54_v2/endowus_systematic_equity_curves.csv"
RF_PANEL = "data/yfinance_close_0P0001AF7U.SI_0P0000KYEE.SI_SP.csv"
RF_COL = "^IRX"
BENCHMARK = "Endowus-60/40"
PERIODS_PER_YEAR = 12  # monthly, matching the paper's active-return statistics


def load_monthly_excess(strategy: str) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Monthly excess returns for (GLM strategy, GPT strategy, benchmark)."""
    glm = pd.read_csv(GLM_CURVES, index_col=0, parse_dates=True).resample("ME").last()
    gpt = pd.read_csv(GPT_CURVES, index_col=0, parse_dates=True).resample("ME").last()
    rf_annual = (
        pd.read_csv(RF_PANEL, index_col=0, parse_dates=True)[RF_COL].dropna().sort_index()
        / 100.0
    )

    def excess(curve: pd.Series) -> pd.Series:
        r = curve.pct_change().dropna()
        aligned = rf_annual.reindex(r.index, method="ffill").fillna(0.04)
        return r - (np.power(1.0 + aligned, 1.0 / PERIODS_PER_YEAR) - 1.0)

    return excess(glm[strategy]), excess(gpt[strategy]), excess(glm[BENCHMARK])


def sharpe(x: np.ndarray) -> float:
    sd = x.std(ddof=1)
    return float(x.mean() / sd * np.sqrt(PERIODS_PER_YEAR)) if sd > 0 else 0.0


def stationary_indices(n: int, expected_block: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap index draw (circular, geometric blocks)."""
    if expected_block <= 1:
        return rng.integers(0, n, size=n)
    p = 1.0 / expected_block
    idx = np.empty(n, dtype=np.int64)
    cur = rng.integers(0, n)
    for t in range(n):
        if t > 0 and rng.random() < p:
            cur = rng.integers(0, n)  # start a new block
        else:
            cur = cur if t == 0 else (cur + 1) % n  # extend current block
        idx[t] = cur
    return idx


def bootstrap(a: np.ndarray, b: np.ndarray, reps: int, expected_block: float,
              seed: int) -> dict:
    """Paired stationary bootstrap of sharpe(a) - sharpe(b)."""
    rng = np.random.default_rng(seed)
    n = len(a)
    diffs = np.empty(reps)
    sa = np.empty(reps)
    sb = np.empty(reps)
    for i in range(reps):
        idx = stationary_indices(n, expected_block, rng)  # SAME indices for both
        sa[i] = sharpe(a[idx])
        sb[i] = sharpe(b[idx])
        diffs[i] = sa[i] - sb[i]
    point = sharpe(a) - sharpe(b)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    # two-sided bootstrap p-value for H0: difference = 0
    p_val = 2.0 * min((diffs <= 0).mean(), (diffs >= 0).mean())
    return {
        "point": point,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "p": float(min(p_val, 1.0)),
        "se": float(diffs.std(ddof=1)),
        "sa_ci": tuple(np.percentile(sa, [2.5, 97.5])),
        "sb_ci": tuple(np.percentile(sb, [2.5, 97.5])),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strategy", default="LLM-BL-A-Monthly-Banded")
    ap.add_argument("--reps", type=int, default=10000,
                    help="bootstrap replicates (paper's CVaR bootstrap uses 10,000)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    glm, gpt, bench = load_monthly_excess(args.strategy)
    a, b, m = glm.to_numpy(), gpt.to_numpy(), bench.to_numpy()
    n = len(a)
    rho = float(np.corrcoef(a, b)[0, 1])

    print(f"Strategy: {args.strategy}")
    print(f"Monthly observations: n = {n}   corr(GLM, GPT) = {rho:.3f}")
    print(f"Sharpe  GLM {sharpe(a):.3f}   GPT {sharpe(b):.3f}   "
          f"benchmark {sharpe(m):.3f}   difference {sharpe(a) - sharpe(b):.3f}")
    print(f"\nStationary bootstrap, {args.reps:,} replicates, paired resampling\n")

    header = f"{'E[block]':>9s} {'diff':>7s} {'95% CI':>18s} {'boot SE':>8s} {'p':>7s}"
    print("GLM vs GPT (same configuration, both stacks)")
    print(header)
    results = {}
    for L in (1, 3, 6, 12):
        r = bootstrap(a, b, args.reps, L, args.seed)
        results[L] = r
        print(f"{L:>9d} {r['point']:>7.3f}  [{r['ci_lo']:>6.3f}, {r['ci_hi']:>6.3f}] "
              f"{r['se']:>8.3f} {r['p']:>7.3f}")

    print("\nGLM vs passive benchmark (for reference)")
    print(header)
    for L in (1, 6):
        r = bootstrap(a, m, args.reps, L, args.seed)
        print(f"{L:>9d} {r['point']:>7.3f}  [{r['ci_lo']:>6.3f}, {r['ci_hi']:>6.3f}] "
              f"{r['se']:>8.3f} {r['p']:>7.3f}")

    head = results[6]
    print(f"\nPer-stack Sharpe 95% CIs (E[block]=6): "
          f"GLM [{head['sa_ci'][0]:.3f}, {head['sa_ci'][1]:.3f}]  "
          f"GPT [{head['sb_ci'][0]:.3f}, {head['sb_ci'][1]:.3f}]")


if __name__ == "__main__":
    main()
