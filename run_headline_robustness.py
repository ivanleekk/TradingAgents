#!/usr/bin/env python3
"""Referee round 4, M1/M2/M4: robustness of the "both stacks lose" result.

Section 4.1 reports the suite-vs-passive and suite-vs-control levels at 10bps
on the March-2026 snapshot universe, but never states the Newey-West lag and
never carries the levels through the cost sweep or the ex-ante-proxy universe
that the cross-stack difference is subjected to. This reports all three.

Replay-only: consumes equity curves produced by systematic_bl_baseline.py at
the relevant --transaction-cost / --universe settings. No LLM output changes.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from scipy import stats

BM = "Endowus-60/40"
CONTROL = "Systematic-BL-Tactical-Monthly"

# (label, GLM curves, GPT curves) -- baselines are the published results dirs,
# the stress arms are replays written to /tmp by the sweep driver.
ARMS = [
    ("10bps, snapshot (published)",
     "results_v2", "results_gpt54_v2"),
    ("25bps, snapshot",
     "/tmp/costsweep/glm4_c25", "/tmp/costsweep/gpt54_c25"),
    ("50bps, snapshot",
     "/tmp/costsweep/glm4_c50", "/tmp/costsweep/gpt54_c50"),
    ("10bps, ex-ante proxy",
     "/tmp/costsweep/glm4_proxy", "/tmp/costsweep/gpt54_proxy"),
]


def newey_west_t(x, lags=None):
    """HAC t-statistic for H0: mean(x) = 0, Bartlett kernel.

    Default lag is the same automatic rule used in
    run_suite_vs_benchmark_and_control.py, which gives 3 at n = 63.
    """
    n = len(x)
    if lags is None:
        lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    e = x - x.mean()
    var = float(e @ e / n)
    for L in range(1, lags + 1):
        var += 2.0 * (1.0 - L / (lags + 1.0)) * float(e[L:] @ e[:-L] / n)
    se = np.sqrt(max(var, 1e-24) / n)
    t = x.mean() / se
    return float(t), float(2 * (1 - stats.t.cdf(abs(t), df=n - 1))), lags


def monthly(results_dir):
    path = os.path.join(results_dir, "endowus_systematic_equity_curves.csv")
    c = pd.read_csv(path, index_col=0, parse_dates=True).resample("ME").last()
    return c.pct_change().dropna()


def ann(x):
    return ((1 + x.mean()) ** 12 - 1) * 100


def arm_stats(glm_dir, gpt_dir):
    d = {"GLM": monthly(glm_dir), "GPT": monthly(gpt_dir)}
    idx = d["GLM"].index.intersection(d["GPT"].index)
    d = {k: v.loc[idx] for k, v in d.items()}

    suites = {}
    for st, v in d.items():
        lv = [c for c in v.columns if c.startswith("LLM-BL-")]
        suites[st] = v[lv].sub(v[BM], axis=0).mean(axis=1)
    ctrl = d["GLM"][CONTROL] - d["GLM"][BM]

    out = {"n": len(idx), "ctrl_ann": ann(ctrl),
           "ctrl_p": newey_west_t(ctrl.to_numpy())[1]}
    for st in ("GLM", "GPT"):
        x = suites[st].to_numpy()
        t, p, L = newey_west_t(x)
        out[st] = {"mo": x.mean() * 100, "ann": ann(x), "t": t, "p": p, "lag": L}
        dv = (suites[st] - ctrl).to_numpy()
        t2, p2, _ = newey_west_t(dv)
        out[st + "_ctrl"] = {"ann": ann(dv), "t": t2, "p": p2}
    return out


def main():
    print("=" * 78)
    print("HEADLINE ROBUSTNESS: does 'both stacks lose' survive costs and universe?")
    print("=" * 78)

    rows = []
    for label, g, p in ARMS:
        if not os.path.exists(os.path.join(g, "endowus_systematic_equity_curves.csv")):
            print(f"  [skip] {label}: {g} not found")
            continue
        rows.append((label, arm_stats(g, p)))

    print(f"\n{'arm':<28} {'stack':<5} {'vs passive':>22} {'vs control':>22}")
    print("-" * 78)
    for label, s in rows:
        for st in ("GLM", "GPT"):
            a, c = s[st], s[st + "_ctrl"]
            print(f"{label if st == 'GLM' else '':<28} {st:<5} "
                  f"{a['ann']:+6.2f}%/yr t={a['t']:+5.2f} p={a['p']:.4f} "
                  f"{c['ann']:+7.2f}%/yr t={c['t']:+5.2f} p={c['p']:.4f}")
    print("-" * 78)
    print(f"Newey-West lag = {rows[0][1]['GLM']['lag']} "
          f"(automatic, floor(4(n/100)^(2/9)) at n = {rows[0][1]['n']})")

    print("\nControl's own active return vs passive (should stay insignificant):")
    for label, s in rows:
        print(f"  {label:<28} {s['ctrl_ann']:+.2f}%/yr  p={s['ctrl_p']:.3f}")

    print("\nLag sensitivity of the published arm (M1):")
    label, gdir, pdir = ARMS[0]
    d = {"GLM": monthly(gdir), "GPT": monthly(pdir)}
    idx = d["GLM"].index.intersection(d["GPT"].index)
    d = {k: v.loc[idx] for k, v in d.items()}
    suites = {st: v[[c for c in v.columns if c.startswith("LLM-BL-")]]
              .sub(v[BM], axis=0).mean(axis=1) for st, v in d.items()}
    ctrl = d["GLM"][CONTROL] - d["GLM"][BM]
    print(f"  {'lag':>4} {'GLM vs pass':>14} {'GPT vs pass':>14} "
          f"{'GLM vs ctrl':>14} {'GPT vs ctrl':>14}")
    for L in range(3, 13):
        cells = []
        for st in ("GLM", "GPT"):
            t, p, _ = newey_west_t(suites[st].to_numpy(), L)
            cells.append(f"t={t:+.2f}")
        for st in ("GLM", "GPT"):
            t, p, _ = newey_west_t((suites[st] - ctrl).to_numpy(), L)
            cells.append(f"p={p:.4f}")
        print(f"  {L:>4} {cells[0]:>14} {cells[1]:>14} {cells[2]:>14} {cells[3]:>14}")


if __name__ == "__main__":
    main()
