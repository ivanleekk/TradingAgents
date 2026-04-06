import argparse
import itertools
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


BASELINE_SCRIPT = "systematic_bl_baseline.py"
BASELINE_RESULTS_DIR = "results"
SUMMARY_FILE = "endowus_systematic_performance_summary.csv"
FULL_REPORT_FILE = "systematic_bl_full_period_report.md"
EQUITY_CURVES_FILE = "endowus_systematic_equity_curves.csv"

ALLOWED_OPTIMIZERS = ("max-sharpe", "max-quadratic-utility")


def parse_float_list(raw: str) -> list[float]:
    values: list[float] = []
    for token in raw.replace(" ", "").split(","):
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("Expected at least one numeric value.")
    return values


def parse_optimizer_list(raw: str) -> list[str]:
    values = [x.strip() for x in raw.split(",") if x.strip()]
    if not values:
        raise ValueError("Expected at least one optimizer value.")
    invalid = [x for x in values if x not in ALLOWED_OPTIMIZERS]
    if invalid:
        raise ValueError(
            f"Invalid optimizer(s): {invalid}. Allowed: {ALLOWED_OPTIMIZERS}."
        )
    return values


def sanitize_float_for_tag(value: float) -> str:
    text = f"{value:g}"
    return text.replace("-", "m").replace(".", "p")


def metric_to_float(x) -> float:
    if pd.isna(x):
        return np.nan
    text = str(x).strip()
    if not text or text.upper() == "N/A":
        return np.nan
    if text.endswith("%"):
        try:
            return float(text[:-1]) / 100.0
        except ValueError:
            return np.nan
    try:
        return float(text)
    except ValueError:
        return np.nan


def copy_if_exists(src: Path, dst: Path):
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def run_single_combination(
    repo_root: Path,
    output_dir: Path,
    python_exec: str,
    optimizer: str,
    risk_aversion: float,
    l2_gamma: float,
    parallel_workers: int,
) -> dict:
    run_tag = (
        f"{optimizer}_ra{sanitize_float_for_tag(risk_aversion)}"
        f"_l2{sanitize_float_for_tag(l2_gamma)}"
    )
    run_dir = output_dir / "runs" / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        python_exec,
        BASELINE_SCRIPT,
        "--optimizer",
        optimizer,
        "--risk-aversion",
        str(risk_aversion),
        "--l2-gamma",
        str(l2_gamma),
        "--parallel-workers",
        str(parallel_workers),
    ]

    start_ts = datetime.now(timezone.utc)
    start_perf = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=str(repo_root),
        text=True,
        capture_output=True,
    )
    elapsed_sec = time.perf_counter() - start_perf
    end_ts = datetime.now(timezone.utc)

    log_path = run_dir / "run.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("COMMAND: " + " ".join(cmd) + "\n\n")
        f.write("STDOUT:\n")
        f.write(proc.stdout or "")
        f.write("\n\nSTDERR:\n")
        f.write(proc.stderr or "")

    baseline_results = repo_root / BASELINE_RESULTS_DIR
    summary_src = baseline_results / SUMMARY_FILE
    report_src = baseline_results / FULL_REPORT_FILE
    curves_src = baseline_results / EQUITY_CURVES_FILE

    summary_dst = run_dir / SUMMARY_FILE
    report_dst = run_dir / FULL_REPORT_FILE
    curves_dst = run_dir / EQUITY_CURVES_FILE

    copy_if_exists(summary_src, summary_dst)
    copy_if_exists(report_src, report_dst)
    copy_if_exists(curves_src, curves_dst)

    return {
        "run_tag": run_tag,
        "optimizer": optimizer,
        "risk_aversion": float(risk_aversion),
        "l2_gamma": float(l2_gamma),
        "started_at_utc": start_ts.isoformat(),
        "ended_at_utc": end_ts.isoformat(),
        "elapsed_seconds": float(elapsed_sec),
        "return_code": int(proc.returncode),
        "status": "success" if proc.returncode == 0 else "failed",
        "run_dir": str(run_dir),
        "summary_path": str(summary_dst) if summary_dst.exists() else "",
        "report_path": str(report_dst) if report_dst.exists() else "",
        "log_path": str(log_path),
    }


def build_markdown_report(
    output_dir: Path,
    run_status_df: pd.DataFrame,
    long_metrics_df: pd.DataFrame,
    optimizers: list[str],
    risk_values: list[float],
    l2_values: list[float],
) -> str:
    report_path = output_dir / "grid_search_report.md"

    total_runs = len(run_status_df)
    success_runs = int((run_status_df["status"] == "success").sum())
    failed_runs = total_runs - success_runs

    lines = []
    lines.append("# Systematic BL Hyperparameter Grid Search Report")
    lines.append("")
    lines.append(f"Generated at: {datetime.now().isoformat()}")
    lines.append("")
    lines.append("## Grid Configuration")
    lines.append("")
    lines.append(f"- Optimizers: {', '.join(optimizers)}")
    lines.append(f"- Risk aversion values: {', '.join(str(v) for v in risk_values)}")
    lines.append(f"- L2 gamma values: {', '.join(str(v) for v in l2_values)}")
    lines.append(f"- Total runs attempted: {total_runs}")
    lines.append(f"- Successful runs: {success_runs}")
    lines.append(f"- Failed runs: {failed_runs}")
    lines.append("")
    lines.append("## Outputs")
    lines.append("")
    lines.append("- run_status.csv")
    lines.append("- strategy_metrics_long.csv")
    lines.append("- best_by_strategy.csv")
    lines.append("- per-run artifacts in runs/<run_tag>/")
    lines.append("")

    if not long_metrics_df.empty:
        for metric_col in ["Sharpe Ratio", "Ann. Return", "Calmar Ratio"]:
            if metric_col in long_metrics_df.columns:
                long_metrics_df[f"{metric_col}_num"] = long_metrics_df[metric_col].map(
                    metric_to_float
                )

        strategy_col = "Strategy" if "Strategy" in long_metrics_df.columns else None

        if strategy_col and "Sharpe Ratio_num" in long_metrics_df.columns:
            lines.append("## Best Config by Strategy (Sharpe)")
            lines.append("")
            best_rows = (
                long_metrics_df.dropna(subset=["Sharpe Ratio_num"])
                .sort_values(
                    [strategy_col, "Sharpe Ratio_num"], ascending=[True, False]
                )
                .groupby(strategy_col, as_index=False)
                .head(1)
            )
            if not best_rows.empty:
                cols = [
                    strategy_col,
                    "optimizer",
                    "risk_aversion",
                    "l2_gamma",
                    "Sharpe Ratio",
                    "Ann. Return",
                    "Max Drawdown",
                    "Calmar Ratio",
                    "run_tag",
                ]
                cols = [c for c in cols if c in best_rows.columns]
                lines.append(best_rows[cols].to_markdown(index=False))
                lines.append("")

        if "Strategy" in long_metrics_df.columns:
            sys_rows = long_metrics_df[long_metrics_df["Strategy"] == "Systematic-BL"]
            if not sys_rows.empty and "Sharpe Ratio_num" in sys_rows.columns:
                top = sys_rows.dropna(subset=["Sharpe Ratio_num"]).sort_values(
                    "Sharpe Ratio_num", ascending=False
                )
                top = top.head(10)
                if not top.empty:
                    lines.append("## Top 10 Systematic-BL Configs by Sharpe")
                    lines.append("")
                    cols = [
                        "optimizer",
                        "risk_aversion",
                        "l2_gamma",
                        "Sharpe Ratio",
                        "Ann. Return",
                        "Max Drawdown",
                        "Calmar Ratio",
                        "run_tag",
                    ]
                    cols = [c for c in cols if c in top.columns]
                    lines.append(top[cols].to_markdown(index=False))
                    lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    return str(report_path)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Grid search over optimizer/risk-aversion/L2-gamma by invoking "
            "systematic_bl_baseline.py and aggregating results."
        )
    )
    parser.add_argument(
        "--optimizers",
        type=str,
        default="max-sharpe,max-quadratic-utility",
        help="Comma-separated optimizer list.",
    )
    parser.add_argument(
        "--risk-aversion-values",
        type=str,
        required=True,
        help="Comma-separated risk aversion values.",
    )
    parser.add_argument(
        "--l2-gamma-values",
        type=str,
        required=True,
        help="Comma-separated L2 gamma values.",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=1,
        help="parallel-workers passed through to systematic_bl_baseline.py",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/hyperparam_grid",
        help="Directory to store aggregated grid search outputs.",
    )
    parser.add_argument(
        "--python-exec",
        type=str,
        default=sys.executable,
        help="Python executable used to invoke systematic_bl_baseline.py",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="If > 0, only run the first N combinations (useful for smoke tests).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip combinations where per-run summary already exists in output dir.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    baseline_path = repo_root / BASELINE_SCRIPT
    if not baseline_path.exists():
        raise FileNotFoundError(f"Could not find {BASELINE_SCRIPT} at {baseline_path}")

    optimizers = parse_optimizer_list(args.optimizers)
    risk_values = parse_float_list(args.risk_aversion_values)
    l2_values = parse_float_list(args.l2_gamma_values)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_combos = list(itertools.product(optimizers, risk_values, l2_values))
    if args.max_runs > 0:
        all_combos = all_combos[: args.max_runs]

    run_status_rows: list[dict] = []
    long_metric_rows: list[dict] = []

    print(
        f"Starting grid search with {len(all_combos)} combinations "
        f"({len(optimizers)} optimizer(s) x {len(risk_values)} risk values x {len(l2_values)} l2 values)."
    )

    for idx, (optimizer, risk_aversion, l2_gamma) in enumerate(all_combos, start=1):
        run_tag = (
            f"{optimizer}_ra{sanitize_float_for_tag(risk_aversion)}"
            f"_l2{sanitize_float_for_tag(l2_gamma)}"
        )
        run_summary_path = output_dir / "runs" / run_tag / SUMMARY_FILE

        if args.skip_existing and run_summary_path.exists():
            print(f"[{idx}/{len(all_combos)}] Skipping existing run: {run_tag}")
            run_status_rows.append(
                {
                    "run_tag": run_tag,
                    "optimizer": optimizer,
                    "risk_aversion": float(risk_aversion),
                    "l2_gamma": float(l2_gamma),
                    "started_at_utc": "",
                    "ended_at_utc": "",
                    "elapsed_seconds": 0.0,
                    "return_code": 0,
                    "status": "skipped_existing",
                    "run_dir": str(output_dir / "runs" / run_tag),
                    "summary_path": str(run_summary_path),
                    "report_path": str(
                        output_dir / "runs" / run_tag / FULL_REPORT_FILE
                    ),
                    "log_path": str(output_dir / "runs" / run_tag / "run.log"),
                }
            )
            continue

        print(
            f"[{idx}/{len(all_combos)}] Running {run_tag} "
            f"(optimizer={optimizer}, risk_aversion={risk_aversion}, l2_gamma={l2_gamma})"
        )

        status = run_single_combination(
            repo_root=repo_root,
            output_dir=output_dir,
            python_exec=args.python_exec,
            optimizer=optimizer,
            risk_aversion=risk_aversion,
            l2_gamma=l2_gamma,
            parallel_workers=args.parallel_workers,
        )
        run_status_rows.append(status)

        if status["status"] != "success":
            print(f"  -> FAILED (rc={status['return_code']}). See {status['log_path']}")
            continue

        print(f"  -> SUCCESS in {status['elapsed_seconds']:.1f}s")

        summary_path = Path(status["summary_path"])
        if summary_path.exists():
            try:
                summary_df = pd.read_csv(summary_path)
                if not summary_df.empty:
                    for _, row in summary_df.iterrows():
                        row_dict = row.to_dict()
                        row_dict.update(
                            {
                                "run_tag": status["run_tag"],
                                "optimizer": optimizer,
                                "risk_aversion": float(risk_aversion),
                                "l2_gamma": float(l2_gamma),
                                "elapsed_seconds": status["elapsed_seconds"],
                            }
                        )
                        long_metric_rows.append(row_dict)
            except Exception as exc:
                print(
                    f"  -> WARNING: failed to parse summary csv ({summary_path}): {exc}"
                )

    run_status_df = pd.DataFrame(run_status_rows)
    long_metrics_df = pd.DataFrame(long_metric_rows)

    run_status_csv = output_dir / "run_status.csv"
    long_metrics_csv = output_dir / "strategy_metrics_long.csv"
    best_by_strategy_csv = output_dir / "best_by_strategy.csv"

    run_status_df.to_csv(run_status_csv, index=False)
    long_metrics_df.to_csv(long_metrics_csv, index=False)

    if not long_metrics_df.empty and "Strategy" in long_metrics_df.columns:
        work = long_metrics_df.copy()
        if "Sharpe Ratio" in work.columns:
            work["Sharpe Ratio_num"] = work["Sharpe Ratio"].map(metric_to_float)
            best = (
                work.dropna(subset=["Sharpe Ratio_num"])
                .sort_values(["Strategy", "Sharpe Ratio_num"], ascending=[True, False])
                .groupby("Strategy", as_index=False)
                .head(1)
            )
        else:
            best = work.groupby("Strategy", as_index=False).head(1)
        best.to_csv(best_by_strategy_csv, index=False)
    else:
        pd.DataFrame().to_csv(best_by_strategy_csv, index=False)

    report_path = build_markdown_report(
        output_dir=output_dir,
        run_status_df=run_status_df,
        long_metrics_df=long_metrics_df,
        optimizers=optimizers,
        risk_values=risk_values,
        l2_values=l2_values,
    )

    print("\nGrid search complete.")
    print(f"- Run status: {run_status_csv}")
    print(f"- Long metrics: {long_metrics_csv}")
    print(f"- Best by strategy: {best_by_strategy_csv}")
    print(f"- Report: {report_path}")


if __name__ == "__main__":
    main()
