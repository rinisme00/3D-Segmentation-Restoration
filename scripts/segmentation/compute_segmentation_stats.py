"""
Purpose:
    Compute global and per-object statistics across the entire generated 3D segmentation dataset.
    It scans the output directory containing `.pts` and `.seg` files, counts the number of points
    belonging to each semantic class (Intact, Crack, Fragment), and computes the ratio of each 
    class relative to the total points for every object.

Usage:
    python scripts/compute_segmentation_stats.py \
        --pts-dir data/processed/points \
        --seg-dir data/processed/seg \
        --output-csv data/processed/stats_summary.csv \
        --barplot-png data/processed/global_class_counts.png \
        --hist-png data/processed/class_ratio_histograms.png

Core Logic:
    1. Iterate through every `.pts` file in the given `--pts-dir`.
    2. Attempt to find the matching `.seg` file in `--seg-dir`.
    3. Load the coordinates and labels, verifying that their lengths match.
    4. Count the frequencies of 0 (Intact), 1 (Crack), and 2 (Fragment) labels.
    5. Aggregate global dataset counts and per-object distribution ratios.
    6. Generate a summary CSV and visual distribution plots (bar plots and histograms).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict

import numpy as np 
import pandas as pd 

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils import (
    INTACT_LABEL,
    CRACK_LABEL,
    FRAGMENT_LABEL,
)
LABEL_NAMES = {
    INTACT_LABEL: "intact",
    CRACK_LABEL: "crack",
    FRAGMENT_LABEL: "fragment",
}


def load_pts_seg(pts_path: Path, seg_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a 3D point cloud and its corresponding label mask from text files.
    Throws a ValueError if the points are missing XYZ dimensions or if the 
    number of labels doesn't exactly match the number of vertices.
    """
    points = np.loadtxt(pts_path, dtype=np.float64)
    labels = np.loadtxt(seg_path, dtype=np.int64)

    if points.ndim == 1:
        points = points.reshape(1, -1)
    if labels.ndim == 0:
        labels = labels.reshape(1)

    if points.shape[1] != 3:
        raise ValueError(f"{pts_path} does not have 3 columns for XYZ")
    if len(points) != len(labels):
        raise ValueError(f"Mismatch: {pts_path.name} has {len(points)} points but {seg_path.name} has {len(labels)} labels")

    return points, labels


def compute_object_stats(object_id: str, pts_path: Path, seg_path: Path) -> Dict[str, object]:
    """
    Extract the total point counts for each semantic class (0, 1, 2) on a single object.
    Computes the mathematical ratio of each class relative to the total geometry density.
    Returns a dictionary row ready for insertion into a Pandas DataFrame.
    """
    points, labels = load_pts_seg(pts_path, seg_path)

    unique, counts = np.unique(labels, return_counts=True)
    count_map = {int(k): int(v) for k, v in zip(unique, counts)}

    n_intact = count_map.get(INTACT_LABEL, 0)
    n_crack = count_map.get(CRACK_LABEL, 0)
    n_fragment = count_map.get(FRAGMENT_LABEL, 0)
    n_total = len(labels)

    return {
        "object_id": object_id,
        "pts_path": str(pts_path.resolve()),
        "seg_path": str(seg_path.resolve()),
        "n_total": n_total,
        "n_intact": n_intact,
        "n_crack": n_crack,
        "n_fragment": n_fragment,
        "ratio_intact": n_intact / n_total if n_total else 0.0,
        "ratio_crack": n_crack / n_total if n_total else 0.0,
        "ratio_fragment": n_fragment / n_total if n_total else 0.0,
    }


def save_global_barplot(global_counts: Dict[int, int], out_png: Path) -> None:
    """
    Render and save a Matplotlib bar chart illustrating the total volume
    of points categorized globally across the entire dataset.
    """
    labels = [LABEL_NAMES[k] for k in [INTACT_LABEL, CRACK_LABEL, FRAGMENT_LABEL]]
    values = [global_counts.get(k, 0) for k in [INTACT_LABEL, CRACK_LABEL, FRAGMENT_LABEL]]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(labels, values)
    ax.set_title("Global Class Counts")
    ax.set_ylabel("Number of points")
    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def save_ratio_histograms(df: pd.DataFrame, out_png: Path) -> None:
    """
    Render and save a 3-pane Matplotlib figure showing the distribution frequency 
    histograms for the Intact, Crack, and Fragment spatial point ratios.
    Useful for spotting dataset imbalance or anomalies across objects.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].hist(df["ratio_intact"], bins=30)
    axes[0].set_title("Intact ratio per object")
    axes[0].set_xlabel("ratio")

    axes[1].hist(df["ratio_crack"], bins=30)
    axes[1].set_title("Crack ratio per object")
    axes[1].set_xlabel("ratio")

    axes[2].hist(df["ratio_fragment"], bins=30)
    axes[2].set_title("Fragment ratio per object")
    axes[2].set_xlabel("ratio")

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """
    Entry point: Parse directories from command-line arguments, execute the
    verification loops for all `.pts`/`.seg` pairs, perform the aggregations,
    and output the CSV/PNG analytical results to disk.
    """
    parser = argparse.ArgumentParser(description="Compute statistics for processed segmentation dataset.")
    parser.add_argument("--pts-dir", type=Path, required=True)
    parser.add_argument("--seg-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--barplot-png", type=Path, required=True)
    parser.add_argument("--hist-png", type=Path, required=True)
    args = parser.parse_args()

    pts_files = sorted(args.pts_dir.glob("*.pts"))
    if not pts_files:
        raise ValueError(f"No .pts files found in {args.pts_dir}")

    rows: List[Dict[str, object]] = []
    global_counts = {
        INTACT_LABEL: 0,
        CRACK_LABEL: 0,
        FRAGMENT_LABEL: 0,
    }

    for pts_path in pts_files:
        object_id = pts_path.stem
        seg_path = args.seg_dir / f"{object_id}.seg"
        if not seg_path.exists():
            print(f"[WARN] Missing seg file for {object_id}, skipping.")
            continue

        row = compute_object_stats(object_id, pts_path, seg_path)
        rows.append(row)

        global_counts[INTACT_LABEL] += int(row["n_intact"])
        global_counts[CRACK_LABEL] += int(row["n_crack"])
        global_counts[FRAGMENT_LABEL] += int(row["n_fragment"])

    df = pd.DataFrame(rows)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    save_global_barplot(global_counts, args.barplot_png)
    save_ratio_histograms(df, args.hist_png)

    total_points = sum(global_counts.values())
    print(f"Objects analyzed: {len(df)}")
    print(f"Total points: {total_points}")
    print("Global class counts:")
    for k in [INTACT_LABEL, CRACK_LABEL, FRAGMENT_LABEL]:
        count = global_counts[k]
        ratio = count / total_points if total_points else 0.0
        print(f"  {LABEL_NAMES[k]}: {count} ({ratio:.4f})")

    print(f"Saved per-object stats CSV to: {args.output_csv}")
    print(f"Saved global barplot to: {args.barplot_png}")
    print(f"Saved ratio histograms to: {args.hist_png}")


if __name__ == "__main__":
    main()

