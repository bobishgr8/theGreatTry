"""Renders val-RMSE heatmaps (Experiments 2/3/6) and r-/sigma-sweep line
plots (Experiments 1/7) from experiments2/results/*_grid.csv - the full
per-config grids each run_experiment*.py script writes, not just the
winning row. UAIMC/IGMC (Experiments 5/8) aren't covered - they use
coordinate-descent local search rather than a brute grid, so the points
they visit are sparse and path-dependent, not a rectangle a heatmap can
meaningfully show.

  python generate_heatmaps.py                    # reads results/*.csv, writes results/plots/*.png
  python generate_heatmaps.py --results-dir X --out-dir Y
  python generate_heatmaps.py --suffix _final     # reads experiment1_plain_grid_final.csv etc.

Safe to run anytime mid-sweep: a dataset that hasn't finished a given
experiment yet is just skipped (with a note printed), matching
experiments/generate_table1.py's philosophy of reading whatever's already
on disk rather than requiring a finished run.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATASET_ORDER = ["ml-100k", "ml-1m", "douban", "amazon-games"]


def _read_grid(path: Path, value_cols: list[str]) -> dict:
    """dataset -> list of row dicts (parsed floats for value_cols)."""
    by_dataset = defaultdict(list)
    if not path.exists():
        print(f"  (missing: {path.name})")
        return by_dataset
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            parsed = {k: (float(row[k]) if k in row else None) for k in value_cols}
            parsed["dataset"] = row["dataset"]
            by_dataset[row["dataset"]].append(parsed)
    return by_dataset


def plot_line_sweep(results_dir: Path, csv_name: str, x_col: str, title: str, out_name: str, out_dir: Path,
                     xlabel: str, xscale: str = "log"):
    print(f"{title}:")
    by_dataset = _read_grid(results_dir / csv_name, [x_col, "val_rmse"])
    if not by_dataset:
        return

    plt.figure(figsize=(8, 5))
    for dataset in DATASET_ORDER:
        rows = by_dataset.get(dataset)
        if not rows:
            print(f"  skipping {dataset} (not run yet)")
            continue
        rows = sorted(rows, key=lambda r: r[x_col])
        plt.plot([r[x_col] for r in rows], [r["val_rmse"] for r in rows], marker="o", label=dataset)
        print(f"  {dataset}: {len(rows)} {x_col}-values plotted")

    plt.xlabel(xlabel)
    plt.ylabel("Validation RMSE")
    plt.title(title)
    if xscale:
        plt.xscale(xscale)
    plt.legend()
    plt.grid(alpha=0.3)
    out_path = out_dir / out_name
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  -> {out_path}")


def plot_heatmap_grid(results_dir: Path, csv_name: str, title: str, out_name: str, out_dir: Path,
                       y_col: str = "r", x_col: str = "lam", y_label: str = "r", x_label: str = "lambda",
                       slice_col: str | None = None):
    """slice_col: an extra hyperparameter axis present in the CSV (e.g. AutoRec's weight_decay)
    that isn't one of the two plotted axes - fixed per-dataset at whichever value gave that
    dataset's best val_rmse, so the 3-D grid collapses to the 2-D slice through its own optimum
    rather than an arbitrary one."""
    print(f"{title}:")
    cols = [y_col, x_col, "val_rmse"] + ([slice_col] if slice_col else [])
    by_dataset = _read_grid(results_dir / csv_name, cols)
    if not by_dataset:
        return

    available = [d for d in DATASET_ORDER if by_dataset.get(d)]
    for d in DATASET_ORDER:
        if d not in available:
            print(f"  skipping {d} (not run yet)")
    if not available:
        return

    fig, axes = plt.subplots(1, len(available), figsize=(6 * len(available), 5.5), squeeze=False)
    axes = axes[0]

    for ax, dataset in zip(axes, available):
        rows = by_dataset[dataset]
        if slice_col:
            best_slice_value = min(rows, key=lambda r: r["val_rmse"])[slice_col]
            rows = [r for r in rows if r[slice_col] == best_slice_value]
        y_values = sorted({r[y_col] for r in rows})
        x_values = sorted({r[x_col] for r in rows})
        grid = np.full((len(y_values), len(x_values)), np.nan)
        for row in rows:
            i = y_values.index(row[y_col])
            j = x_values.index(row[x_col])
            grid[i, j] = row["val_rmse"]

        im = ax.imshow(grid, cmap="viridis_r", aspect="auto")
        ax.set_xticks(range(len(x_values)))
        ax.set_xticklabels([f"{v:g}" for v in x_values], rotation=45, fontsize=7)
        ax.set_yticks(range(len(y_values)))
        ax.set_yticklabels([f"{v:g}" for v in y_values], fontsize=8)
        ax.set_xlabel(x_label)
        if ax is axes[0]:
            ax.set_ylabel(y_label)
        title_suffix = f" ({slice_col}={best_slice_value:g})" if slice_col else ""
        ax.set_title(f"{dataset}{title_suffix}", fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for i in range(len(y_values)):
            for j in range(len(x_values)):
                if not np.isnan(grid[i, j]):
                    ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center",
                            color="white", fontsize=5.5)
        print(f"  {dataset}: {len(y_values)} {y_label} x {len(x_values)} {x_label} plotted"
              f"{f' (sliced at {slice_col}={best_slice_value:g})' if slice_col else ''}")

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "experiments2" / "results")
    parser.add_argument("--out-dir", type=Path, default=None, help="default: <results-dir>/plots")
    parser.add_argument("--suffix", type=str, default="",
                         help="appended before .csv on every grid filename read, e.g. _final to read "
                              "experiment1_plain_grid_final.csv instead of experiment1_plain_grid.csv")
    args = parser.parse_args()

    out_dir = args.out_dir or (args.results_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    s = args.suffix

    plot_line_sweep(args.results_dir, f"experiment1_plain_grid{s}.csv", "r",
                     "Experiment 1 (Plain MF): validation RMSE vs r, by dataset",
                     f"experiment1_r_sweep{s}.png", out_dir, xlabel="r")
    plot_heatmap_grid(args.results_dir, f"experiment2_lasso_grid{s}.csv",
                       "Experiment 2 (Lasso MF): validation RMSE over (r, lambda)",
                       f"experiment2_lasso_heatmaps{s}.png", out_dir)
    plot_heatmap_grid(args.results_dir, f"experiment3_ours_grid{s}.csv",
                       "Experiment 3 (Ours): validation RMSE over (r, lambda)",
                       f"experiment3_ours_heatmaps{s}.png", out_dir)
    plot_heatmap_grid(args.results_dir, f"experiment6_autorec_grid{s}.csv",
                       "Experiment 6 (AutoRec): validation RMSE over (hidden, lr)",
                       f"experiment6_autorec_heatmaps{s}.png", out_dir,
                       y_col="hidden", x_col="lr", y_label="hidden", x_label="lr", slice_col="weight_decay")
    plot_line_sweep(args.results_dir, f"experiment7_softimpute_grid{s}.csv", "sigma",
                     "Experiment 7 (SoftImpute): validation RMSE vs sigma, by dataset",
                     f"experiment7_sigma_sweep{s}.png", out_dir, xlabel="sigma")

    print(f"\nall plots written to {out_dir}")
    print("(UAIMC/IGMC - Experiments 5/8 - use coordinate-descent local search, not a brute grid, "
          "so they're not plotted here; see their *_best*.csv for the winning configs instead.)")


if __name__ == "__main__":
    main()
