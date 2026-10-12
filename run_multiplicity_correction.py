#!/usr/bin/env python3
"""M3: multiplicity corrections for the HM and Rank-IC families.

Reads the per-stack advanced-significance tables and applies Bonferroni and
Benjamini-Hochberg within each family, then reports a deflated Sharpe ratio for
the best-of-16 statistic that Table 2 bolds.

Replay-only: reads generated reports. No LLM calls, no network.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy import stats

HM_REPORTS = {"GLM": "results_v2/advanced_significance_tests.md",
              "GPT": "results_gpt54_v2/advanced_significance_tests.md"}
IC_REPORTS = {"GLM": "results_v2/ic_and_sub_attribution_report.md",
              "GPT": "results_gpt54_v2/ic_and_sub_attribution_report.md"}


def parse_hm(path: str) -> pd.DataFrame:
    rows, inside = [], False
    for line in open(path):
        if "HM Beta2" in line:
            inside = True
            continue
        if not inside:
            continue
        if line.startswith("##"):
            break
        m = re.match(r"\|\s*(LLM-BL-[\w\-]+)\s*\|\s*(-?[\d.]+)\s*\|\s*([\d.]+)\s*\|", line)
        if m:
            rows.append({"portfolio": m.group(1), "beta2": float(m.group(2)),
                         "p": float(m.group(3))})
    return pd.DataFrame(rows)


def parse_ic(path: str) -> pd.DataFrame:
    rows = []
    for line in open(path):
        m = re.match(r"\|\s*(LLM-BL-[ABCD])\s*\|\s*(-?[\d.]+)\s*\|[^|]*\|[^|]*\|\s*([\d.]+)\s*\|", line)
        if m:
            rows.append({"channel": m.group(1), "ic": float(m.group(2)), "p": float(m.group(3))})
    return pd.DataFrame(rows)


def benjamini_hochberg(p: np.ndarray, q: float = 0.05):
    n = len(p)
    order = np.argsort(p)
    thresh = q * np.arange(1, n + 1) / n
    passed = p[order] <= thresh
    k = int(np.max(np.where(passed)[0]) + 1) if passed.any() else 0
    crit = thresh[k - 1] if k else 0.0
    keep = np.zeros(n, dtype=bool)
    if k:
        keep[order[:k]] = True
    adj, prev = np.empty(n), 1.0
    for i in range(n - 1, -1, -1):
        prev = min(prev, p[order[i]] * n / (i + 1))
        adj[order[i]] = min(prev, 1.0)
    return keep, crit, adj


def report_family(name: str, labels, pvals, extra=None):
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    bonf = np.minimum(p * n, 1.0)
    keep, crit, bh = benjamini_hochberg(p)
    print(f"\n{name}   family size m = {n}")
    print(f"  Bonferroni threshold {0.05/n:.5f}    BH critical value {crit:.5f}")
    print(f"  {'test':32s} {'raw p':>9s} {'Bonf p':>9s} {'BH p':>9s}   verdict")
    for i in np.argsort(p)[:6]:
        lab = labels[i] + (f" {extra[i]}" if extra is not None else "")
        verdict = "survives BH" if keep[i] else ("survives Bonf" if bonf[i] < 0.05 else "-")
        print(f"  {lab:32s} {p[i]:9.4f} {bonf[i]:9.4f} {bh[i]:9.4f}   {verdict}")
    print(f"  significant at raw 0.05: {(p<0.05).sum()}/{n}"
          f"   Bonferroni: {(bonf<0.05).sum()}/{n}   BH: {keep.sum()}/{n}")


def deflated_sharpe_p(sr_max: float, sr_all: np.ndarray, n_obs: int) -> float:
    """Bailey & Lopez de Prado: p-value for H0 true SR <= 0, given best of N trials."""
    N = len(sr_all)
    v = np.var(sr_all, ddof=1)
    e = 0.5772156649
    sr0 = np.sqrt(v) * ((1 - e) * stats.norm.ppf(1 - 1.0 / N)
                        + e * stats.norm.ppf(1 - 1.0 / (N * np.e)))
    z = (sr_max - sr0) * np.sqrt(n_obs - 1)
    return float(1.0 - stats.norm.cdf(z))


def main() -> None:
    print("=" * 76)
    print("M3: MULTIPLICITY CORRECTION")
    print("=" * 76)

    hm = pd.concat([parse_hm(p).assign(stk=k) for k, p in HM_REPORTS.items()], ignore_index=True)
    report_family("HENRIKSSON-MERTON beta2", (hm["stk"] + " " + hm["portfolio"]).tolist(),
                  hm["p"].to_numpy(), extra=[f"b2={b:+.3f}" for b in hm["beta2"]])

    ic = pd.concat([parse_ic(p).assign(stk=k) for k, p in IC_REPORTS.items()], ignore_index=True)
    report_family("CROSS-SECTIONAL RANK IC", (ic["stk"] + " " + ic["channel"]).tolist(),
                  ic["p"].to_numpy(), extra=[f"IC={v:+.3f}" for v in ic["ic"]])

    print("\n" + "=" * 76)
    print("DEFLATED SHARPE for the best-of-16 statistic bolded in Table 2")
    print("=" * 76)
    for tag, d in [("GLM", "results_v2"), ("GPT", "results_gpt54_v2")]:
        s = pd.read_csv(f"{d}/endowus_systematic_performance_summary.csv")
        k = "Strategy" if "Strategy" in s.columns else s.columns[0]
        s = s.set_index(k)
        llm = [i for i in s.index if str(i).startswith("LLM-BL-")]
        sr = pd.to_numeric(s.loc[llm, "Sharpe Ratio"], errors="coerce").dropna().to_numpy()
        pv = deflated_sharpe_p(sr.max(), sr, 63)
        print(f"  {tag}: best SR {sr.max():.3f} of {len(sr)} trials (s.d. {sr.std(ddof=1):.3f})"
              f"   deflated p = {pv:.3f}   {'survives' if pv < 0.05 else 'DOES NOT survive'}")


if __name__ == "__main__":
    main()
