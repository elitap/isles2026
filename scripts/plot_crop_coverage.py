#!/usr/bin/env python3
"""
plot_crop_coverage.py — Histogram of per-case GT-lesion coverage inside the
Stage-1-prediction-derived bounding box (see crop_coverage.py). Writes both a
linear-count and a log-count version — the distribution is a huge spike at
100% (~100/131 cases) plus a thin scatter of small bins, so the linear y-axis
makes everything below ~10 cases hard to read; the log version trades that
for legibility of the small bins at the cost of visually compressing the
spike.

Usage:
    python isles/scripts/plot_crop_coverage.py \
        --csv isles/evaluation/paper_fold0/crop_coverage/crop_coverage.csv \
        --output isles/evaluation/paper_fold0/crop_coverage
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, LogLocator, FuncFormatter
import numpy as np

BLUE = "#2a78d6"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--csv", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--label", default="fold 0 validation",
        help="Describes the case subset in the title/footnote, e.g. "
             "'folds 0-2 pooled (partial CV)' (default: 'fold 0 validation')",
    )
    return p.parse_args()


def build_histogram(
    values: list[float], mean_v: float, median_v: float, n_gt_empty: int, log_y: bool, label: str,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.5, 5))
    bins = np.arange(0, 105, 5)
    counts, bin_edges, _ = ax.hist(values, bins=bins, color=BLUE, alpha=0.88, edgecolor="white", linewidth=0.6)

    ax.set_xlabel("% of GT lesion voxels inside the Stage-1-prediction bbox crop (margin = 15 vx)", fontsize=10)
    ax.set_ylabel("Number of cases", fontsize=10)
    title_suffix = " — log y-axis" if log_y else ""
    ax.set_title(
        f"ISLES — GT coverage of the Stage-2 crop, {label} (n={len(values)}){title_suffix}",
        fontsize=12, fontweight="bold",
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(-2, 102)

    if log_y:
        ax.set_yscale("log")
        ax.set_ylim(0.8, max(counts) * 2.2)  # headroom for the bin-count labels
        ax.yaxis.set_major_locator(LogLocator(base=10))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))  # "1, 10, 100", not "10^0"
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=range(2, 10)))
        ax.grid(axis="y", which="major", linestyle="--", alpha=0.35)
        ax.grid(axis="y", which="minor", linestyle="--", alpha=0.15)

        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        for x, count in zip(bin_centers, counts):
            if count > 0:
                ax.text(x, count * 1.08, f"{int(count)}", ha="center", va="bottom", fontsize=8, color="#1b2330")
    else:
        ax.yaxis.set_major_locator(MultipleLocator(10))  # a gridline every 10 cases
        ax.grid(axis="y", linestyle="--", alpha=0.35)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.08)

    ax.axvline(mean_v, color="#333230", linestyle="--", linewidth=1.3)
    ax.axvline(median_v, color=BLUE, linestyle=":", linewidth=1.3)

    # Stats box, anchored top-left, well clear of the tall bar at x=100.
    stats_text = (
        f"mean    {mean_v:.1f}%\n"
        f"median  {median_v:.1f}%\n"
        f"min–max {np.min(values):.0f}–{np.max(values):.0f}%"
    )
    ax.text(
        0.03, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
        va="top", ha="left", family="monospace", color="#1b2330",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#dfe3e8"),
    )

    note = f"{n_gt_empty} GT-empty case(s) excluded from this histogram." if n_gt_empty else \
           f"All {len(values)} {label} cases have a non-empty GT lesion mask."
    fig.text(0.5, -0.02, note, ha="center", fontsize=8.5, color="#5c6674")

    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    rows = list(csv.DictReader(args.csv.open()))

    values = [float(r["coverage_pct"]) for r in rows if r["gt_empty"] != "True" and r["coverage_pct"] != ""]
    n_gt_empty = sum(1 for r in rows if r["gt_empty"] == "True")

    mean_v = float(np.mean(values))
    median_v = float(np.median(values))

    for log_y, suffix in [(False, ""), (True, "_log")]:
        fig = build_histogram(values, mean_v, median_v, n_gt_empty, log_y, args.label)
        png_path = args.output / f"crop_coverage_hist{suffix}.png"
        fig.savefig(png_path, dpi=150, bbox_inches="tight")
        fig.savefig(png_path.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {png_path} (+ .pdf)")


if __name__ == "__main__":
    main()
