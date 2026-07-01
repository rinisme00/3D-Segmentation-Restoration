"""Build object-disjoint split CSVs for restoration completion pairs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-index", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--prefix", default="completion_pairs_3d")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def read_rows(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[:limit] if limit is not None else rows


def infer_group(row: dict[str, str]) -> tuple[str, str, str]:
    partial = Path(row["partial_path"])
    parts = partial.parts
    if "Fantastic_Breaks_v1" in parts:
        idx = parts.index("Fantastic_Breaks_v1")
        category = parts[idx + 1] if idx + 1 < len(parts) else "unknown"
        object_id = parts[idx + 2] if idx + 2 < len(parts) else partial.parent.name
        return ("FantasticBreaks", category, object_id)
    if "BreakingBad" in parts:
        idx = parts.index("BreakingBad")
        subset = parts[idx + 1] if idx + 1 < len(parts) else "unknown"
        if subset == "artifact":
            object_id = parts[idx + 2] if idx + 2 < len(parts) else partial.name
            return ("BreakingBad", subset, object_id)
        category = parts[idx + 2] if idx + 2 < len(parts) else "unknown"
        object_id = parts[idx + 3] if idx + 3 < len(parts) else partial.name
        return ("BreakingBad", category, object_id)
    return ("unknown", partial.parent.name, partial.name)


def assign_group_splits(groups: list[str], train_ratio: float, val_ratio: float, seed: int) -> dict[str, str]:
    import random

    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    train_cut = int(round(total * train_ratio))
    val_cut = train_cut + int(round(total * val_ratio))
    train_cut = min(max(train_cut, 0), total)
    val_cut = min(max(val_cut, train_cut), total)
    assignments: dict[str, str] = {}
    for index, group in enumerate(shuffled):
        if index < train_cut:
            split = "train"
        elif index < val_cut:
            split = "val"
        else:
            split = "test"
        assignments[group] = split
    return assignments


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"split ratios must sum to 1.0, got {total_ratio}")

    rows = read_rows(args.sample_index, limit=args.limit)
    enriched: list[dict[str, str]] = []
    groups_by_name: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        dataset, category, object_id = infer_group(row)
        split_group = f"{dataset}::{category}::{object_id}"
        out = dict(row)
        out["dataset_name"] = dataset
        out["category"] = category
        out["object_id"] = object_id
        out["split_group"] = split_group
        enriched.append(out)
        groups_by_name[split_group].append(out)

    assignments = assign_group_splits(sorted(groups_by_name), args.train_ratio, args.val_ratio, args.seed)
    split_rows = {"train": [], "val": [], "test": []}
    for row in enriched:
        split = assignments[row["split_group"]]
        out = dict(row)
        out["split"] = split
        split_rows[split].append(out)

    fieldnames = list(split_rows["train"][0].keys() if split_rows["train"] else enriched[0].keys())
    for split, split_data in split_rows.items():
        write_csv(args.output_dir / f"{args.prefix}_{split}.csv", fieldnames, split_data)

    summary = {
        "sample_index": str(args.sample_index),
        "prefix": args.prefix,
        "total_rows": len(enriched),
        "total_groups": len(groups_by_name),
        "ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
        "seed": args.seed,
        "splits": {
            split: {
                "rows": len(split_data),
                "groups": len({row["split_group"] for row in split_data}),
            }
            for split, split_data in split_rows.items()
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / f"{args.prefix}_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
