#!/usr/bin/env python3
"""Test the cross-stack difference in the raw view distribution.

This is the one quantity in the paper estimated from many samples (n ~ 15,800
views per stack) rather than from a single portfolio path, and it is measured
at the model's output layer before any optimization.

Views within a decision date are not independent (the agents share a macro
context), so a naive n = 15,800 t-test would overstate precision by a wide
margin. Every test here is clustered or blocked BY DATE, which is the level at
which the draws are plausibly independent. The unclustered figure is printed
alongside only to show how much the clustering costs.

Replay-only: reads stored agent decisions. No LLM calls, no network.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

GLM_DIRS = ["results_endowus_A", "results_endowus_B", "results_endowus_C", "results_endowus_D"]
GPT_DIRS = ["gpt54_endowus_A", "gpt54_endowus_B", "gpt54_endowus_C", "gpt54_endowus_D"]


def load(dirs: list[str]) -> pd.DataFrame:
    rows = []
    for d in dirs:
        var = d[-1]
        for f in glob.glob(os.path.join(d, "*_decisions.csv")):
            ticker = os.path.basename(f).replace("_decisions.csv", "")
            for _, r in pd.read_csv(f).iterrows():
                try:
                    j = json.loads(r["decision"])
                except Exception:
                    continue
                v = j.get("Target_Return_30d")
                c = j.get("Confidence_Score")
                if v is None:
                    continue
                if isinstance(v, str):
                    v = v.strip().rstrip("%")
                try:
                    v = float(v)
                except ValueError:
                    continue
                rows.append({"date": pd.to_datetime(r["test_date"]), "ticker": ticker,
                             "variation": var, "view": v,
                             "conf": float(c) if c is not None else np.nan})
    return pd.DataFrame(rows)


def cluster_by_date(df: pd.DataFrame, col: str) -> pd.Series:
    return df.groupby("date")[col].mean()


def date_block_bootstrap(a: pd.Series, b: pd.Series, reps: int, block: int, seed: int) -> dict:
    """Paired stationary bootstrap over the shared date index."""
    idx = a.index.intersection(b.index)
    x = (a.loc[idx] - b.loc[idx]).to_numpy()
    n = len(x)
    rng = np.random.default_rng(seed)
    p_ = 1.0 / block
    means = np.empty(reps)
    for i in range(reps):
        out = np.empty(n, dtype=np.int64)
        cur = rng.integers(0, n)
        for t in range(n):
            if t > 0 and rng.random() < p_:
                cur = rng.integers(0, n)
            elif t > 0:
                cur = (cur + 1) % n
            out[t] = cur
        means[i] = x[out].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    pv = 2.0 * min((means <= 0).mean(), (means >= 0).mean())
    return {"diff": float(x.mean()), "lo": float(lo), "hi": float(hi),
            "p": float(min(pv, 1.0)), "n_dates": n}


def main() -> None:
    glm, gpt = load(GLM_DIRS), load(GPT_DIRS)
    print(f"views loaded: GLM n={len(glm):,}  GPT n={len(gpt):,}")
    print(f"dates: GLM {glm.date.nunique()}  GPT {gpt.date.nunique()}   "
          f"tickers {glm.ticker.nunique()}   variations {glm.variation.nunique()}\n")

    print("=" * 72)
    print("MEAN 30-DAY VIEW  (percent, as emitted)")
    print("=" * 72)
    print(f"  GLM {glm.view.mean():+.3f}%   GPT {gpt.view.mean():+.3f}%   "
          f"difference {gpt.view.mean() - glm.view.mean():+.3f}pp")
    print(f"  share negative: GLM {(glm.view < 0).mean()*100:.1f}%   GPT {(gpt.view < 0).mean()*100:.1f}%")

    t_naive, p_naive = stats.ttest_ind(gpt.view, glm.view, equal_var=False)
    print(f"\n  [NOT USED] unclustered Welch t: t={t_naive:.1f}  p={p_naive:.2e}"
          f"   <- overstates precision, views correlate within date")

    ga, pa = cluster_by_date(glm, "view"), cluster_by_date(gpt, "view")
    idx = ga.index.intersection(pa.index)
    d = (pa.loc[idx] - ga.loc[idx])
    t_cl, p_cl = stats.ttest_1samp(d, 0.0)
    print(f"\n  date-clustered paired t (n={len(d)} dates): "
          f"mean {d.mean():+.3f}pp  t={t_cl:.2f}  p={p_cl:.3e}")
    for block in (1, 3, 6, 12):
        r = date_block_bootstrap(pa, ga, 10000, block, 0)
        print(f"      block bootstrap E[block]={block:<3d} diff {r['diff']:+.3f}pp "
              f"95% CI [{r['lo']:+.3f}, {r['hi']:+.3f}]  p={r['p']:.4f}")

    print("\n" + "=" * 72)
    print("CONFIDENCE DISPERSION  (per-date cross-sectional s.d.)")
    print("=" * 72)
    gsd = glm.groupby("date")["conf"].std()
    psd = gpt.groupby("date")["conf"].std()
    idx2 = gsd.index.intersection(psd.index)
    dd = (psd.loc[idx2] - gsd.loc[idx2]).dropna()
    t2, p2 = stats.ttest_1samp(dd, 0.0)
    print(f"  GLM mean within-date s.d. {gsd.mean():.3f}   GPT {psd.mean():.3f}")
    print(f"  date-clustered paired t (n={len(dd)}): mean {dd.mean():+.3f}  t={t2:.2f}  p={p2:.3e}")

    print("\n" + "=" * 72)
    print("DISTRIBUTIONAL EQUALITY (per-date, then aggregated)")
    print("=" * 72)
    ks = [stats.ks_2samp(glm[glm.date == dt].view, gpt[gpt.date == dt].view).pvalue
          for dt in idx]
    ks = np.array(ks)
    print(f"  per-date KS: median p={np.median(ks):.4f}   "
          f"share of dates rejecting at 5%: {(ks < 0.05).mean()*100:.1f}%  ({len(ks)} dates)")


if __name__ == "__main__":
    main()
