"""Generate the cross-model figures used in the ICAIF paper.

Previously these two PDFs were produced ad hoc and had no generator in the repo,
so they could not be regenerated after a backtest change. This script rebuilds
`cross_model_equity.pdf` from the stored equity curves.

Figures are authored at final print size (single acmart sigconf column, ~3.33in)
so LaTeX includes them at scale 1 and the type stays at the intended size.

Usage:
    python make_thesis_figures.py \
        --glm-dir results_v2 --gpt-dir results_gpt54_v2 \
        --out-dir "/path/to/ACM_ICAIF/results"
"""

from __future__ import annotations

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

# Categorical hues validated for CVD separation (deuteran/protan/tritan dE >= 19.7,
# normal-vision dE >= 21.9, all >= 3:1 contrast on a white surface).
BLUE = "#1F6FB2"
RED = "#C4342B"
PURPLE = "#7B4FA8"
GREY = "#6E6E6E"

INK = "#1A1A1A"
MUTED = "#5A5A5A"
GRID = "#D8D8D8"


def _load(results_dir: str) -> pd.DataFrame:
    path = os.path.join(results_dir, "endowus_systematic_equity_curves.csv")
    return pd.read_csv(path, index_col=0, parse_dates=True)


def plot_cross_model_equity(glm_dir: str, gpt_dir: str, out_path: str) -> None:
    glm = _load(glm_dir)
    gpt = _load(gpt_dir)

    bench = glm["Endowus-60/40"]
    other_bench = gpt["Endowus-60/40"]
    max_gap = float((bench - other_bench).abs().max())
    if max_gap > 1e-6:
        raise SystemExit(
            f"Benchmark differs across runs by {max_gap:.6f}; it should be identical. "
            "Re-run the backtests before regenerating this figure."
        )

    series = [
        ("Passive 60/40", bench, GREY, "-", 1.4),
        ("GLM-4-32B, A-Monthly-Banded", glm["LLM-BL-A-Monthly-Banded"], BLUE, "-", 1.8),
        ("GPT-5.4/nano, A-Monthly-Banded", gpt["LLM-BL-A-Monthly-Banded"], RED, "-", 1.8),
        ("GPT-5.4/nano, C-Monthly (best)", gpt["LLM-BL-C-Monthly"], PURPLE, (0, (4, 2)), 1.5),
    ]

    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.2,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(3.34, 2.35))

    for label, s, color, ls, lw in series:
        ax.plot(s.index, s.values, label=label, color=color, linestyle=ls, linewidth=lw,
                solid_capstyle="round")

    # Recessive grid and axes; no top/right spines.
    ax.grid(True, which="major", color=GRID, linewidth=0.5, alpha=0.9)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
        ax.spines[side].set_linewidth(0.7)
    ax.tick_params(colors=MUTED, length=2, width=0.6)

    ax.set_ylabel("Value of \\$10,000", color=INK)
    ax.axhline(10000, color=GRID, linewidth=0.7, zorder=0)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1000:,.0f}k")
    )

    # Legend below the axes: with four entries there is no in-plot whitespace
    # large enough to avoid colliding with the series.
    leg = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        frameon=False,
        handlelength=1.9,
        borderaxespad=0.0,
        labelspacing=0.3,
        columnspacing=1.2,
        handletextpad=0.5,
    )
    for text in leg.get_texts():
        text.set_color(INK)

    fig.tight_layout(pad=0.3)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    finals = {label: float(s.iloc[-1]) for label, s, *_ in series}
    print(f"Wrote {out_path}")
    for label, v in finals.items():
        print(f"   final value  {label:34s} ${v:,.0f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--glm-dir", default="results_v2")
    parser.add_argument("--gpt-dir", default="results_gpt54_v2")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    plot_cross_model_equity(
        args.glm_dir, args.gpt_dir, os.path.join(args.out_dir, "cross_model_equity.pdf")
    )


if __name__ == "__main__":
    main()
