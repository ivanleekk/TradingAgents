#!/usr/bin/env python3
"""M2: permutation test for the confidence-to-Omega map.

Section 4.4 rests its harmfulness claim on published-vs-shuffled, the one
weight-matched contrast. A single shuffle is one draw from the null. This
script builds the null distribution properly: N independent within-date
permutations of confidence across instruments, each replayed through the full
optimizer, and reports where the published map's suite Sharpe falls.

Replay-only: no LLM calls, no network. ~45s per permutation per stack.

Usage:
    python run_omega_permutation_test.py --n 200 --workers 6
    python run_omega_permutation_test.py --report-only
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
STACKS = {"GLM": ("glm4", "results_v2"), "GPT": ("gpt54", "results_gpt54_v2")}
OUT = os.path.join(BASE, "perm_test_results.json")


def suite_sharpe(results_dir: str) -> float:
    """Mean Sharpe over the 16 LLM variants, as Table 3 reports it."""
    f = os.path.join(results_dir, "endowus_systematic_performance_summary.csv")
    s = pd.read_csv(f)
    k = "Strategy" if "Strategy" in s.columns else s.columns[0]
    s = s.set_index(k)
    llm = [i for i in s.index if str(i).startswith("LLM-BL-")]
    return float(pd.to_numeric(s.loc[llm, "Sharpe Ratio"], errors="coerce").dropna().mean())


def one_perm(args) -> tuple[str, int, float]:
    stack, model, seed = args
    with tempfile.TemporaryDirectory(prefix=f"perm_{stack}_{seed}_") as td:
        cmd = [os.path.join(BASE, ".venv/bin/python"),
               os.path.join(BASE, "systematic_bl_baseline.py"),
               "--model", model, "--omega-mode", "shuffled",
               "--omega-seed", str(seed), "--results-dir", td]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=BASE)
        if r.returncode != 0:
            return stack, seed, float("nan")
        try:
            return stack, seed, suite_sharpe(td)
        except Exception:
            return stack, seed, float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    if not args.report_only:
        jobs = [(st, model, seed)
                for st, (model, _) in STACKS.items()
                for seed in range(1, args.n + 1)]
        results: dict[str, list[float]] = {st: [] for st in STACKS}
        done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(one_perm, j) for j in jobs]
            for f in as_completed(futs):
                st, seed, sh = f.result()
                if not np.isnan(sh):
                    results[st].append(sh)
                done += 1
                if done % 20 == 0:
                    print(f"  {done}/{len(jobs)} permutations complete", flush=True)
        json.dump(results, open(OUT, "w"))
        print(f"saved -> {OUT}")
    else:
        results = json.load(open(OUT))

    print("\n" + "=" * 72)
    print("M2: PERMUTATION TEST OF THE CONFIDENCE-TO-OMEGA MAP")
    print("=" * 72)
    published = {}
    for st, (_, d) in STACKS.items():
        published[st] = suite_sharpe(os.path.join(BASE, d))
    for st in STACKS:
        x = np.array(sorted(results[st]))
        pub = published[st]
        pct = float((x <= pub).mean())
        print(f"\n{st}:  published suite Sharpe = {pub:.4f}   (n = {len(x)} permutations)")
        print(f"  null distribution: mean {x.mean():.4f}  sd {x.std(ddof=1):.4f}  "
              f"min {x.min():.4f}  max {x.max():.4f}")
        print(f"  2.5th/50th/97.5th pct: {np.percentile(x,2.5):.4f} / "
              f"{np.percentile(x,50):.4f} / {np.percentile(x,97.5):.4f}")
        print(f"  published falls at percentile {pct*100:.1f} of the null")
        print(f"  one-sided p (published <= shuffled): {(x <= pub).mean():.4f}"
              f"   [{(x <= pub).sum()} of {len(x)}]")


if __name__ == "__main__":
    main()
