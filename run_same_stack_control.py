#!/usr/bin/env python3
"""Referee rounds 3-5, W1/Q1: the same-stack control.

The paper's leading positive claim is that substituting the backbone shifts the
output distribution (mean 30-day view -1.33pp, confidence dispersion -0.32,
both date-clustered). Every quantity in that comparison is a BETWEEN-DEPLOYMENT
difference measured with one deployment per arm, so date clustering bounds
within-deployment dependence but says nothing about between-deployment variance.

This script draws a SECOND unseeded sample from one stack and asks how large the
within-stack difference is. If it comes back an order of magnitude below 1.33pp,
the claim strengthens from "these two runs differ" to "stacks differ more than
deployments of one stack do" -- the claim the title makes. If it comes back
comparable, that is the more important finding.

Nothing here is seeded on purpose: trading_graph.py builds ChatOpenAI without
`temperature` or `seed`, so the published runs were already unseeded draws and a
re-run is a genuine second draw from the same sampling distribution.

SAFETY: this is the only script in the repo that spends money. It is dry-run by
default and refuses to call an API without --execute.

  # 1. inspect the plan, verify keys and config match the published run (free)
  uv run python run_same_stack_control.py --stack glm

  # 2. measure real cost and latency on a few tasks (spends a little)
  uv run python run_same_stack_control.py --stack glm --probe 5 --execute

  # 3. the full second draw
  uv run python run_same_stack_control.py --stack glm --execute

  # 4. compare within-stack against cross-stack (free, replay-only)
  uv run python run_same_stack_control.py --analyze
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import glob
import json
import os
import sys
import time
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration mirrored from the published runners. These MUST match or the
# control measures a config change instead of decoding variance.
#   GLM  -> run_endowus_ablations.py:main()
#   GPT  -> run_gpt54_endowus_ablations.py:main()
# ---------------------------------------------------------------------------
STACKS = {
    "glm": {
        "provider": "openrouter",
        "backend_url": "https://openrouter.ai/api/v1",
        "deep": "z-ai/glm-4-32b",
        "quick": "z-ai/glm-4-32b",
        "published_dir": "results_endowus_{var}",
        "draw2_dir": "results_endowus_{var}_draw2",
    },
    "gpt": {
        "provider": "openai",
        "backend_url": "https://api.openai.com/v1",
        "deep": "gpt-5.4",
        "quick": "gpt-5-nano-2025-08-07",
        "published_dir": "gpt54_endowus_{var}",
        "draw2_dir": "gpt54_endowus_{var}_draw2",
    },
}

PROVIDER_BASE_URL = {
    "openrouter": "https://openrouter.ai/api/v1",
    "openai": "https://api.openai.com/v1",
}
PROVIDER_KEY_ENV = {
    "openrouter": "OPENROUTER_API_KEY",
    "openai": "OPENAI_API_KEY",
}

# Embeddings always go to OpenAI: memory.py builds OpenAI(base_url=...) with no
# api_key argument, so OPENAI_API_KEY is required even on the OpenRouter path.
# Five agents call get_memories() per graph run.
EMBEDDING_BACKEND_URL = "https://api.openai.com/v1"
EMBEDDING_MODEL = "text-embedding-3-small"

VARIATIONS = {
    "A": ["fundamentals", "market"],
    "B": ["news"],
    "C": ["market"],
    "D": ["market", "news", "fundamentals"],
}


def load_universe_and_dates():
    """Reuse the published runner's universe and date grid verbatim."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("_abl", "run_endowus_ablations.py")
    mod = importlib.util.module_from_spec(spec)
    argv = sys.argv
    sys.argv = ["run_endowus_ablations.py"]
    try:
        spec.loader.exec_module(mod)
    except SystemExit:
        pass
    finally:
        sys.argv = argv
    return list(mod.ETFS), mod.get_trading_dates()


# ---------------------------------------------------------------------------
# Draw generation
# ---------------------------------------------------------------------------
def build_config(args):
    from tradingagents.default_config import DEFAULT_CONFIG

    preset = STACKS[args.stack]
    provider = args.provider or preset["provider"]
    cfg = DEFAULT_CONFIG.copy()
    cfg["llm_provider"] = provider
    cfg["backend_url"] = args.backend_url or PROVIDER_BASE_URL[provider]
    cfg["embedding_backend_url"] = EMBEDDING_BACKEND_URL
    cfg["deep_think_llm"] = args.deep_model or preset["deep"]
    cfg["quick_think_llm"] = args.quick_model or preset["quick"]
    cfg["max_debate_rounds"] = 1
    cfg["online_tools"] = True
    cfg["data_dir"] = "./data_dir"
    return cfg


def check_keys(cfg):
    """Return (ok, messages). Never prints or logs key material."""
    msgs, ok = [], True
    env = PROVIDER_KEY_ENV[cfg["llm_provider"]]
    if os.getenv(env):
        msgs.append(f"  [ok]      {env} is set (chat completions)")
    else:
        msgs.append(f"  [MISSING] {env} -- required for {cfg['llm_provider']} chat")
        ok = False
    if os.getenv("OPENAI_API_KEY"):
        msgs.append(f"  [ok]      OPENAI_API_KEY is set ({EMBEDDING_MODEL} embeddings)")
    else:
        msgs.append("  [MISSING] OPENAI_API_KEY -- memory.py embeds via OpenAI even on "
                    "the OpenRouter path; 5 agents call get_memories() per run")
        ok = False
    return ok, msgs


CHECKPOINT_DB = "data_dir/checkpoints/checkpoints.db"


def check_thread_namespace(prefix, variation):
    """Refuse to reuse a thread_id namespace that already has checkpoints.

    LangGraph resumes from a checkpoint when the thread_id matches, so running
    the control under the published run's namespace would silently replay draw 1
    instead of sampling a new one -- and would look like near-zero within-stack
    variance, i.e. a false confirmation of the paper's claim.
    """
    import sqlite3

    if not os.path.exists(CHECKPOINT_DB):
        return True, "  [ok]      no checkpoint DB; nothing to collide with"
    pat = f"{prefix}_{variation}_%"
    try:
        conn = sqlite3.connect(f"file:{CHECKPOINT_DB}?mode=ro", uri=True, timeout=60)
        try:
            n = conn.execute(
                "SELECT COUNT(*) FROM (SELECT 1 FROM checkpoints "
                "WHERE thread_id LIKE ? LIMIT 1)", (pat,)).fetchone()[0]
        finally:
            conn.close()
    except Exception as exc:                       # unreadable DB -> don't block
        return True, f"  [warn]    could not read checkpoint DB ({exc})"
    size_gb = os.path.getsize(CHECKPOINT_DB) / 1e9
    if n:
        return False, (f"  [COLLISION] thread_ids matching '{pat}' already exist in "
                       f"{CHECKPOINT_DB} ({size_gb:.1f} GB).\n"
                       "              LangGraph would RESUME those checkpoints and "
                       "return the previous draw's\n"
                       "              outputs. Pass a fresh --thread-prefix.")
    return True, (f"  [ok]      thread namespace '{pat}' is unused "
                  f"(DB {size_gb:.1f} GB) -- no resume risk")


def completed_tasks(out_dir, tickers):
    done = set()
    for t in tickers:
        f = os.path.join(out_dir, f"{t}_decisions.csv")
        if os.path.isfile(f):
            try:
                df = pd.read_csv(f)
            except pd.errors.EmptyDataError:
                continue
            if "test_date" in df.columns:
                for d in df["test_date"].astype(str):
                    done.add((t, d))
    return done


def select_dates(dates, args):
    if args.sample_dates:
        rng = np.random.default_rng(args.sample_seed)
        idx = sorted(rng.choice(len(dates), size=min(args.sample_dates, len(dates)),
                                replace=False))
        return [dates[i] for i in idx]
    if args.dates:
        return dates[: args.dates]
    return dates


def run_draw(args, cfg, tickers, dates):
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from tenacity import retry, stop_after_attempt, wait_exponential

    analysts = VARIATIONS[args.variation]
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    @retry(stop=stop_after_attempt(5),
           wait=wait_exponential(multiplier=2, min=2, max=60), reraise=True)
    def evaluate(ticker, dt):
        t0 = time.time()
        ta = TradingAgentsGraph(selected_analysts=analysts, config=cfg)
        # The graph is compiled with a persistent SQLite checkpointer
        # (setup.py:213) but propagation.get_graph_args() supplies no thread_id,
        # so propagate() raises. We inject one -- and it MUST live in a fresh
        # namespace: the published run used "A_{ticker}_{date}", and reusing
        # that would make LangGraph RESUME draw 1 and hand back its outputs,
        # yielding a fake second draw with near-zero variance.
        tid = f"{args.thread_prefix}_{args.variation}_{ticker}_{dt}"
        base = ta.propagator.get_graph_args()
        ta.propagator.get_graph_args = lambda: {
            **base,
            "config": {**base.get("config", {}), "configurable": {"thread_id": tid}},
        }
        _state, decision = ta.propagate(company_name=ticker, trade_date=dt)
        return {"ticker": ticker, "test_date": dt, "decision": decision,
                "elapsed": time.time() - t0}

    done = completed_tasks(out_dir, tickers)
    tasks = [(t, d) for t in tickers for d in dates if (t, d) not in done]
    if args.probe:
        # Spread the probe across the grid. Tasks are ticker-major, so the first
        # N would all be one instrument -- and cost per run varies by instrument
        # (news coverage differs sharply across these 14 lines).
        step = max(1, len(tasks) // args.probe)
        tasks = tasks[::step][: args.probe]
    if not tasks:
        print("Nothing pending -- this draw is already complete.")
        return

    for t in tickers:
        f = os.path.join(out_dir, f"{t}_decisions.csv")
        if not os.path.isfile(f):
            with open(f, "w", newline="") as fh:
                csv.writer(fh).writerow(["test_date", "decision"])

    print(f"Running {len(tasks)} graph runs -> {out_dir}/ "
          f"({args.max_workers} workers)\n", flush=True)
    t_start, times, n_done, n_err = time.time(), [], 0, 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {ex.submit(evaluate, t, d): (t, d) for t, d in tasks}
        for fut in concurrent.futures.as_completed(futs):
            t, d = futs[fut]
            try:
                r = fut.result()
            except Exception as exc:
                n_err += 1
                print(f"  [error] {t} {d}: {exc}", flush=True)
                continue
            with open(os.path.join(out_dir, f"{r['ticker']}_decisions.csv"),
                      "a", newline="") as fh:
                csv.writer(fh).writerow([r["test_date"], r["decision"]])
            times.append(r["elapsed"])
            n_done += 1
            eta = (sum(times) / len(times) * (len(tasks) - n_done)) / args.max_workers
            print(f"  [{n_done}/{len(tasks)}] {t} {d}  {r['elapsed']:.1f}s  "
                  f"ETA {timedelta(seconds=int(eta))}", flush=True)

    wall = time.time() - t_start
    print(f"\nDone: {n_done} succeeded, {n_err} failed, wall {timedelta(seconds=int(wall))}")
    if times:
        print(f"Mean {np.mean(times):.1f}s/run.")
        if args.probe:
            full = len(tickers) * 283
            proj = (np.mean(times) * full) / args.max_workers
            print(f"\nPROBE PROJECTION for the full {full} runs at "
                  f"{args.max_workers} workers: {timedelta(seconds=int(proj))} wall.")
            print("Multiply your provider's per-run cost by "
                  f"{full} for the money estimate; the probe above is "
                  f"{len(times)} runs of real spend to calibrate it.")


# ---------------------------------------------------------------------------
# Analysis: within-stack vs cross-stack, matched methodology
# ---------------------------------------------------------------------------
def load_views(dirs):
    """Same parsing as run_view_distribution_test.py so figures are comparable."""
    rows = []
    for d in dirs:
        if not os.path.isdir(d):
            continue
        for f in glob.glob(os.path.join(d, "*_decisions.csv")):
            ticker = os.path.basename(f).replace("_decisions.csv", "")
            try:
                df = pd.read_csv(f)
            except pd.errors.EmptyDataError:
                continue
            for _, r in df.iterrows():
                try:
                    j = json.loads(r["decision"])
                except Exception:
                    continue
                v, c = j.get("Target_Return_30d"), j.get("Confidence_Score")
                if v is None:
                    continue
                if isinstance(v, str):
                    v = v.strip().rstrip("%")
                try:
                    v = float(v)
                except (TypeError, ValueError):
                    continue
                rows.append({"date": pd.to_datetime(r["test_date"]), "ticker": ticker,
                             "view": v, "conf": float(c) if c is not None else np.nan})
    return pd.DataFrame(rows)


def block_bootstrap(x, reps=10000, block=6, seed=0):
    n = len(x)
    rng = np.random.default_rng(seed)
    p = 1.0 / block
    means = np.empty(reps)
    for i in range(reps):
        out = np.empty(n, dtype=np.int64)
        cur = rng.integers(0, n)
        for t in range(n):
            if t > 0 and rng.random() < p:
                cur = rng.integers(0, n)
            elif t > 0:
                cur = (cur + 1) % n
            out[t] = cur
        means[i] = x[out].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def paired_by_date(a, b, col, agg):
    sa = a.groupby("date")[col].agg(agg)
    sb = b.groupby("date")[col].agg(agg)
    idx = sa.index.intersection(sb.index)
    return (sb.loc[idx] - sa.loc[idx]).dropna()


def contrast(label, a, b):
    if a.empty or b.empty:
        print(f"  {label:<34} (missing data -- draw not run yet)")
        return None
    dv = paired_by_date(a, b, "view", "mean")
    dc = paired_by_date(a, b, "conf", "std")
    if len(dv) < 3:
        print(f"  {label:<34} (only {len(dv)} shared dates)")
        return None
    tv, pv = stats.ttest_1samp(dv, 0.0)
    tc, pc = stats.ttest_1samp(dc, 0.0)
    lo, hi = block_bootstrap(dv.to_numpy())
    print(f"  {label:<34} n={len(dv):>3} dates")
    print(f"      mean 30d view diff   {dv.mean():+7.3f}pp  t={tv:+7.2f}  p={pv:.3e}"
          f"   boot95 [{lo:+.3f}, {hi:+.3f}]")
    print(f"      conf dispersion diff {dc.mean():+7.3f}    t={tc:+7.2f}  p={pc:.3e}")
    return {"view": dv.mean(), "conf": dc.mean(), "n": len(dv)}


def analyze(args):
    var = args.variation
    glm_pub = [STACKS["glm"]["published_dir"].format(var=var)]
    gpt_pub = [STACKS["gpt"]["published_dir"].format(var=var)]
    glm_d2 = [STACKS["glm"]["draw2_dir"].format(var=var)]
    gpt_d2 = [STACKS["gpt"]["draw2_dir"].format(var=var)]

    g1, p1 = load_views(glm_pub), load_views(gpt_pub)
    g2, p2 = load_views(glm_d2), load_views(gpt_d2)

    print("=" * 78)
    print(f"SAME-STACK CONTROL -- variation {var}")
    print("=" * 78)
    for nm, df in (("GLM draw 1 (published)", g1), ("GPT draw 1 (published)", p1),
                   ("GLM draw 2 (control)", g2), ("GPT draw 2 (control)", p2)):
        print(f"  {nm:<26} {len(df):>6,} views   "
              f"{df.date.nunique() if not df.empty else 0:>3} dates")

    print("\nCROSS-STACK (the paper's claim, restricted to this variation)")
    cross = contrast("GPT-1 vs GLM-1", g1, p1)
    print("\nWITHIN-STACK (the control)")
    w_glm = contrast("GLM-2 vs GLM-1", g1, g2)
    w_gpt = contrast("GPT-2 vs GPT-1", p1, p2)

    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    if not cross:
        print("  Cross-stack contrast unavailable.")
        return
    withins = [w for w in (w_glm, w_gpt) if w]
    if not withins:
        print("  No second draw found yet. Run with --execute first.")
        print(f"  Cross-stack reference: view {cross['view']:+.3f}pp, "
              f"conf {cross['conf']:+.3f}")
        return
    for nm, w in zip(("GLM", "GPT"), withins):
        rv = abs(cross["view"]) / abs(w["view"]) if w["view"] else float("inf")
        rc = abs(cross["conf"]) / abs(w["conf"]) if w["conf"] else float("inf")
        print(f"  {nm}: cross-stack view difference is {rv:.1f}x the within-stack "
              f"difference ({cross['view']:+.3f} vs {w['view']:+.3f} pp)")
        print(f"       confidence dispersion {rc:.1f}x "
              f"({cross['conf']:+.3f} vs {w['conf']:+.3f})")
    print("\n  Read it this way: a large ratio supports the title's claim (stacks")
    print("  differ more than deployments of one stack do). A ratio near 1 means")
    print("  the substitution literature is measuring decoding noise.")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Same-stack control: a second unseeded draw per stack.")
    ap.add_argument("--stack", choices=sorted(STACKS), help="which backbone to redraw")
    ap.add_argument("--variation", default="A", choices=sorted(VARIATIONS))
    ap.add_argument("--provider", choices=sorted(PROVIDER_BASE_URL),
                    help="override the stack's provider (e.g. run gpt via openrouter)")
    ap.add_argument("--backend-url", help="override the provider base URL")
    ap.add_argument("--deep-model", help="override the deep-thinking model id")
    ap.add_argument("--quick-model", help="override the quick-thinking model id")
    ap.add_argument("--out-dir", help="output dir (default: <published>_draw2)")
    ap.add_argument("--dates", type=int, help="use only the first N dates")
    ap.add_argument("--sample-dates", type=int,
                    help="use a random subset of N dates (better than --dates "
                         "for a pilot: keeps the regime mix)")
    ap.add_argument("--sample-seed", type=int, default=0)
    ap.add_argument("--max-workers", type=int, default=16)
    ap.add_argument("--thread-prefix", default="draw2",
                    help="checkpointer thread_id namespace; must not collide "
                         "with the published run (which used '<variation>_...')")
    ap.add_argument("--probe", type=int, metavar="N",
                    help="run only N tasks to calibrate cost and latency")
    ap.add_argument("--execute", action="store_true",
                    help="actually call the API. Without this the script is inert.")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    ap.add_argument("--analyze", action="store_true",
                    help="replay-only comparison; makes no API calls")
    args = ap.parse_args()

    if args.analyze:
        analyze(args)
        return
    if not args.stack:
        ap.error("--stack is required (or use --analyze)")

    tickers, all_dates = load_universe_and_dates()
    dates = select_dates(all_dates, args)
    cfg = build_config(args)
    preset = STACKS[args.stack]
    if not args.out_dir:
        args.out_dir = preset["draw2_dir"].format(var=args.variation)

    pub_dir = preset["published_dir"].format(var=args.variation)
    done = completed_tasks(args.out_dir, tickers)
    pending = [(t, d) for t in tickers for d in dates if (t, d) not in done]
    n_plan = min(args.probe, len(pending)) if args.probe else len(pending)

    print("=" * 78)
    print(f"SAME-STACK CONTROL -- {args.stack.upper()} draw 2, variation {args.variation}")
    print("=" * 78)
    print(f"  provider        {cfg['llm_provider']}  ({cfg['backend_url']})")
    print(f"  deep model      {cfg['deep_think_llm']}")
    print(f"  quick model     {cfg['quick_think_llm']}")
    print(f"  embeddings      {EMBEDDING_MODEL} @ {EMBEDDING_BACKEND_URL}")
    print(f"  analysts        {VARIATIONS[args.variation]}")
    print(f"  sampling        unseeded (no temperature/seed set -- matches published)")
    print(f"  published draw  {pub_dir}/")
    print(f"  output          {args.out_dir}/")
    print(f"  universe        {len(tickers)} instruments")
    print(f"  dates           {len(dates)} of {len(all_dates)}")
    print(f"  already done    {len(done)}")
    print(f"  to run now      {n_plan}" + ("  (probe)" if args.probe else ""))

    print("\nCredentials")
    ok, msgs = check_keys(cfg)
    print("\n".join(msgs))

    print("\nCheckpoint namespace")
    ns_ok, ns_msg = check_thread_namespace(args.thread_prefix, args.variation)
    print(ns_msg)
    ok = ok and ns_ok

    # Same model routed through a different provider carries a vendor prefix
    # (openai/gpt-5.4 on OpenRouter == gpt-5.4 on OpenAI), and a dated snapshot
    # is the same model as its alias. Neither is a config change.
    def same_model(a, b):
        a, b = a.split("/")[-1], b.split("/")[-1]
        return a == b or a.startswith(b) or b.startswith(a)

    if not (same_model(cfg["deep_think_llm"], preset["deep"])
            and same_model(cfg["quick_think_llm"], preset["quick"])):
        print("\n  [WARNING] models differ from the published run "
              f"({preset['deep']} / {preset['quick']}). The control would then")
        print("            measure a config change, not decoding variance.")

    if not args.execute:
        print("\nDRY RUN -- no API calls were made. Add --execute to spend.")
        print("Suggested order:")
        print(f"  1. --stack {args.stack} --probe 5 --execute      "
              f"# calibrate cost/latency")
        print(f"  2. --stack {args.stack} --execute                # full draw")
        print("  3. --analyze                                    # free comparison")
        return

    if not ok:
        why = []
        if not check_keys(cfg)[0]:
            why.append("credentials missing")
        if not ns_ok:
            why.append("checkpoint namespace collision (pass a fresh --thread-prefix)")
        print(f"\nRefusing to run: {'; '.join(why)}.")
        sys.exit(1)
    if not args.yes:
        print(f"\nAbout to make live API calls for {n_plan} graph runs.")
        if input("Type 'yes' to proceed: ").strip().lower() != "yes":
            print("Aborted.")
            return
    run_draw(args, cfg, tickers, dates)


if __name__ == "__main__":
    main()
