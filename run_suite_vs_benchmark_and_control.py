#!/usr/bin/env python3
"""M3 and M4: suite active return against the benchmark, and against the
non-LLM heuristic control.

M3: is the 16-variant suite mean active return significantly different from
    zero, separately for each stack? (The paper reports -0.297%/mo as an
    intermediate quantity and never tests it.)
M4: the same paired machinery applied to LLM suite vs Systematic-BL-Tactical,
    which is the comparison a practitioner actually faces.

Replay-only: reads stored equity curves.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

CURVES = {"GLM": "results_v2/endowus_systematic_equity_curves.csv",
          "GPT": "results_gpt54_v2/endowus_systematic_equity_curves.csv"}
BM = "Endowus-60/40"
CONTROL = "Systematic-BL-Tactical-Monthly"


def newey_west_t(x, lags=None):
    n = len(x)
    if lags is None:
        lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    e = x - x.mean()
    var = float(e @ e / n)
    for L in range(1, lags + 1):
        var += 2.0 * (1.0 - L / (lags + 1.0)) * float(e[L:] @ e[:-L] / n)
    se = np.sqrt(max(var, 1e-24) / n)
    t = x.mean() / se
    return float(t), float(2 * (1 - stats.t.cdf(abs(t), df=n - 1))), lags, float(se)


def stat_idx(n, L, rng):
    p = 1.0 / L
    idx = np.empty(n, dtype=np.int64)
    cur = rng.integers(0, n)
    for t in range(n):
        if t > 0 and rng.random() < p:
            cur = rng.integers(0, n)
        elif t > 0:
            cur = (cur + 1) % n
        idx[t] = cur
    return idx


def boot(x, reps=10000, L=6, seed=0):
    rng = np.random.default_rng(seed)
    m = np.array([x[stat_idx(len(x), L, rng)].mean() for _ in range(reps)])
    lo, hi = np.percentile(m, [2.5, 97.5])
    return float(lo), float(hi), float(2 * min((m <= 0).mean(), (m >= 0).mean()))


def monthly(path):
    c = pd.read_csv(path, index_col=0, parse_dates=True).resample("ME").last()
    return c.pct_change().dropna()


def main():
    data = {k: monthly(v) for k, v in CURVES.items()}
    idx = data["GLM"].index.intersection(data["GPT"].index)
    data = {k: v.loc[idx] for k, v in data.items()}
    n = len(idx)

    print("=" * 74)
    print(f"M3: SUITE ACTIVE RETURN vs PASSIVE BENCHMARK   (n = {n} months)")
    print("=" * 74)
    suites = {}
    for st, d in data.items():
        lv = [c for c in d.columns if c.startswith("LLM-BL-")]
        s = d[lv].sub(d[BM], axis=0).mean(axis=1)
        suites[st] = s
        x = s.to_numpy()
        t, p, L, se = newey_west_t(x)
        lo, hi, pb = boot(x)
        ann = ((1 + x.mean()) ** 12 - 1) * 100
        print(f"  {st} suite: {x.mean()*100:+.3f}%/mo  ({ann:+.2f}%/yr)   "
              f"NW t={t:+.2f} p={p:.4f} (lags={L})")
        print(f"           bootstrap 95% CI [{lo*100:+.3f}%, {hi*100:+.3f}%]  p={pb:.4f}")

    print("\n" + "=" * 74)
    print("M4: LLM SUITE vs SYSTEMATIC-BL-TACTICAL (non-LLM heuristic control)")
    print("=" * 74)
    ctrl_active = (data["GLM"][CONTROL] - data["GLM"][BM])
    tc, pc, Lc, _ = newey_west_t(ctrl_active.to_numpy())
    annc = ((1 + ctrl_active.mean()) ** 12 - 1) * 100
    print(f"  control active vs benchmark: {ctrl_active.mean()*100:+.3f}%/mo "
          f"({annc:+.2f}%/yr)  NW t={tc:+.2f} p={pc:.4f}")
    # control is stack-invariant; confirm
    ctrl_gpt = (data["GPT"][CONTROL] - data["GPT"][BM])
    print(f"  (stack-invariance check: max |GLM-GPT| control diff = "
          f"{float((ctrl_active-ctrl_gpt).abs().max()):.2e})")
    for st in data:
        d = (suites[st] - ctrl_active).to_numpy()
        t, p, L, _ = newey_west_t(d)
        lo, hi, pb = boot(d)
        ann = ((1 + d.mean()) ** 12 - 1) * 100
        print(f"\n  {st} suite - control: {d.mean()*100:+.3f}%/mo ({ann:+.2f}%/yr)"
              f"   NW t={t:+.2f} p={p:.4f} (lags={L})")
        print(f"      bootstrap 95% CI [{lo*100:+.3f}%, {hi*100:+.3f}%]  p={pb:.4f}")

    print("\n" + "=" * 74)
    print("S1: POWER / MINIMUM DETECTABLE EFFECT")
    print("=" * 74)
    # SE of the paired cross-stack difference for the matched configuration
    S = "LLM-BL-A-Monthly-Banded"
    dpair = ((data["GPT"][S] - data["GPT"][BM]) - (data["GLM"][S] - data["GLM"][BM])).to_numpy()
    t, p, L, se = newey_west_t(dpair)
    mde = (1.96 + 0.84) * se           # 80% power, two-sided alpha=0.05
    obs = abs(dpair.mean())
    n_req = n * (mde / obs) ** 2       # SE scales as 1/sqrt(n)
    print(f"  observed paired difference : {dpair.mean()*100:+.3f}%/mo (t={t:+.2f})")
    print(f"  Newey-West SE              : {se*100:.3f}%/mo")
    print(f"  MDE at 80% power, a=0.05   : {mde*100:.3f}%/mo  "
          f"({((1+mde)**12-1)*100:.2f}%/yr)")
    print(f"  months needed to detect the observed effect: {n_req:.0f} "
          f"({n_req/12:.1f} years) vs {n} ({n/12:.2f} years) available")


if __name__ == "__main__":
    main()
