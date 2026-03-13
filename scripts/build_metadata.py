from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import pandas as pd

PATTERNS = {
    "npz": re.compile(r"Fantastic_Breaks_v1/\d{2}/(?P<object_id>\d+)/meta_.*\.npz"),
    "b": re.compile(r"Fantastic_Breaks_v1/\d{2}/(?P<object_id>\d+)/model_b_.*\.ply"),
    "c": re.compile(r"Fantastic_Breaks_v1/\d{2}/(?P<object_id>\d+)/model_c\.ply"),
    "r": re.compile(r"Fantastic_Breaks_v1/\d{2}/(?P<object_id>\d+)/model_r_.*\.ply"),
}

def match_file(path: Path) -> Optional[Tuple[str, str]]:
    """
    Match a file path to its kind and object ID.
    """
    for kind, pattern in PATTERNS.items():
        m = pattern.search(path.as_posix())
        if m:
            return kind, m.group("object_id")
    return None

def infer_split(path: Path) -> str:
    """
    Infer the split (train/val/test) from the path.
    """
    lowered = [p.lower() for p in path.parts]
    for split in ("train", "val", "test"):
        if split in lowered:
            return split
    return "unknown"

def collect_metadata(raw_root: Path) -> pd.DataFrame:
    records: Dict[str, Dict[str, str]] = {}

    for path in raw_root.rglob("*"):
        if not path.is_file():
            continue
        matched = match_file(path)
        if matched is None:
            continue

        kind, object_id = matched
        rec = records.setdefault(
            object_id,
            {
                "object_id": object_id,
                "split": infer_split(path),
            }
        )
        rec[f"path_{kind}"] = str(path.resolve())
    
    rows = []
    for object_id, rec in sorted(records.items()):
        for kind in ("npz", "b", "c", "r"):
            rec.setdefault(f"path_{kind}", "")
        rec["has_all_files"] = all(rec[f"path_{kind}"] for kind in ("npz", "b", "c", "r"))
        rows.append(rec)
    
    return pd.DataFrame(rows)

def main() -> None:
    parser = argparse.ArgumentParser(description="Build metadata CSV for Fantastic Breaks-like dataset.")
    parser.add_argument("--raw-root", type=Path, required=True, help="Raw dataset root")
    parser.add_argument("--output-csv", type=Path, required=True, help="Output metadata CSV")
    args = parser.parse_args()

    df = collect_metadata(args.raw_root)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_csv, index=False)

    total = len(df)
    complete = int(df["has_all_files"].sum()) if total > 0 else 0

    print(f"Saved metadata to: {args.output_csv}")
    print(f"Total objects: {total}")
    print(f"Complete objects: {complete}")
    print(f"Incomplete objects: {total - complete}")

    if total > 0:
        print("\nPreview:")
        print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()