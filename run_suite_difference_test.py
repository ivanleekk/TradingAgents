#!/usr/bin/env python3
"""M1: paired test of the cross-stack difference in suite active return.

The paper's central claim is that substitution moved the 16-variant suite from
parity with passive to significant underperformance. Comparing a significant
p-value against a non-significant one is not a test of that difference. This
script forms the monthly paired series

    d_t = (GPT suite active return)_t - (GLM suite active return)_t

which is paired by month and by configuration, and tests it with the same
machinery Section 4.1 uses for the Sharpe: Newey-West HAC t-statistic and a
stationary block bootstrap.

Replay-only: reads stored equity curves. No LLM calls, no network.
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from scipy import stats

GLM_CURVES = "results_v2/endowus_systematic_equity_curves.csv"
GPT_CURVES = "results_gpt54_v2/endowus_systematic_equity_curves.csv"
BENCHMARK = "Endowus-60/40"


def suite_active(path: str) -> tuple[pd.Series, pd.DataFrame]:
    """Monthly suite-mean active return, and the per-variant active matrix."""
    curves = pd.read_csv(path, index_col=0, parse_dates=True).resample("ME").last()
    rets = curves.pct_change().dropna()
    variants = [c for c in rets.columns if c.startswith("LLM-BL-")]
    active = rets[variants].sub(rets[BENCHMARK], axis=0)
    return active.mean(axis=1), active


def newey_west_t(x: np.ndarray, lags: int | None = None) -> tuple[float, float, int]:
    """HAC t-statistic for H0: mean(x) = 0, Bartlett kernel."""
    n = len(x)
    if lags is None:  # Newey-West (1994) automatic rule
        lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    e = x - x.mean()
    gamma0 = float(e @ e / n)
    var = gamma0
    for L in range(1, lags + 1):
        cov = float(e[L:] @ e[:-L] / n)
        var += 2.0 * (1.0 - L / (lags + 1.0)) * cov
    se = np.sqrt(max(var, 1e-24) / n)
    t = x.mean() / se
    p = 2.0 * (1.0 - stats.t.cdf(abs(t), df=n - 1))
    return float(t), float(p), lags


def stationary_indices(n: int, expected_block: float, rng: np.random.Generator) -> np.ndarray:
    if expected_block <= 1:
        return rng.integers(0, n, size=n)
    p = 1.0 / expected_block
    idx = np.empty(n, dtype=np.int64)
    cur = rng.integers(0, n)
    for t in range(n):
        if t > 0 and rng.random() < p:
            cur = rng.integers(0, n)
        elif t > 0:
            cur = (cur + 1) % n
        idx[t] = cur
    return idx


def bootstrap_mean(x: np.ndarray, reps: int, block: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(x)
    means = np.empty(reps)
    for i in range(reps):
        means[i] = x[stationary_indices(n, block, rng)].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2.0 * min((means <= 0).mean(), (means >= 0).mean())
    return {"lo": float(lo), "hi": float(hi), "p": float(min(p, 1.0))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    glm_suite, glm_mat = suite_active(GLM_CURVES)
    gpt_suite, gpt_mat = suite_active(GPT_CURVES)
    idx = glm_suite.index.intersection(gpt_suite.index)
    glm_suite, gpt_suite = glm_suite.loc[idx], gpt_suite.loc[idx]
    n = len(idx)

    print("=" * 72)
    print("LEVELS (reproducing the paper's reported suite active returns)")
    print("=" * 72)
    for tag, s in [("GLM suite", glm_suite), ("GPT suite", gpt_suite)]:
        t, p, L = newey_west_t(s.to_numpy())
        print(f"  {tag}: mean {s.mean()*100:+.3f}%/mo   NW t={t:+.2f}  p={p:.3f}  (lags={L}, n={n})")

    print("\n" + "=" * 72)
    print("M1: PAIRED DIFFERENCE  (GPT suite - GLM suite), monthly")
    print("=" * 72)
    d = (gpt_suite - glm_suite).to_numpy()
    rho = float(np.corrcoef(glm_suite, gpt_suite)[0, 1])
    t, p, L = newey_west_t(d)
    print(f"  mean difference : {d.mean()*100:+.3f}%/mo")
    print(f"  corr(GLM, GPT)  : {rho:.3f}   (pairing is what shrinks the SE)")
    print(f"  Newey-West      : t={t:+.3f}  p={p:.4f}  (auto lags={L}, n={n})")
    for L2 in (3, 6, 12):
        t2, p2, _ = newey_west_t(d, lags=L2)
        print(f"      lags={L2:<3d}      t={t2:+.3f}  p={p2:.4f}")
    print("  stationary block bootstrap:")
    for block in (1, 3, 6, 12):
        r = bootstrap_mean(d, args.reps, block, args.seed)
        print(f"      E[block]={block:<3d} 95% CI [{r['lo']*100:+.3f}%, {r['hi']*100:+.3f}%]  p={r['p']:.4f}")

    # Per-configuration paired differences: all 16 matched configurations.
    print("\n" + "=" * 72)
    print("PER-CONFIGURATION PAIRED DIFFERENCE (GPT - GLM), monthly active return")
    print("=" * 72)
    common = [c for c in glm_mat.columns if c in gpt_mat.columns]
    diffs = (gpt_mat[common].loc[idx] - glm_mat[common].loc[idx])
    neg = 0
    for c in common:
        x = diffs[c].to_numpy()
        t, p, _ = newey_west_t(x)
        neg += x.mean() < 0
        print(f"  {c:26s} {x.mean()*100:+.3f}%/mo  t={t:+.2f}  p={p:.3f}")
    print(f"\n  configurations where GPT is worse: {neg}/{len(common)}")


if __name__ == "__main__":
    main()
