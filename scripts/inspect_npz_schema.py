"""
Purpose:
This script inspects the schema of .npz metadata files listed in a metadata CSV.
It reads the arrays contained in each .npz, extracting their keys, shapes, and datatypes.
Furthermore, it applies heuristics to identify keys that might represent
3D transformation matrices or boolean masks.

Usage:
    python scripts/inspect_npz_schema.py \\
        --metadata-csv /path/to/metadata.csv \\
        --output-json /path/to/output_summary.json \\
        --output-csv /path/to/output_summary.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

def is_transform_candidate(arr: np.ndarray) -> bool:
    """
    Check if a numpy array's shape suggests it might be a 3D transformation matrix.
    
    Transforms are typically 4x4 or 3x4 matrices. They can also be stacked
    in an array (e.g., N x 4 x 4).
    
    Args:
        arr (np.ndarray): The array to evaluate.
        
    Returns:
        bool: True if it matches typical transform shapes, False otherwise.
    """
    return (
        (arr.ndim == 2 and arr.shape in {(4, 4), (3, 4)})
        # The trailing dimensions should match typical matrix shapes
        or (arr.ndim == 3 and arr.shape[-2:] in {(4, 4), (3, 4)})
    )
    
def describe_npz(npz_path: Path) -> Dict[str, Any]:
    """
    Load an .npz file and describe the schema of arrays it contains.
    
    Args:
        npz_path (Path): Path to the .npz file.
        
    Returns:
        Dict[str, Any]: A dictionary containing the file path, detailed info for each
                        array key, and lists of candidate keys for masks and transforms.
    """
    # Load the compressed numpy archive
    data = np.load(npz_path, allow_pickle=True)

    entries: List[Dict[str, Any]] = []
    transform_candidates: List[str] = []
    mask_candidates: List[str] = []

    # Iterate over every array stored in the npz file
    for key in data.files:
        value = data[key]
        arr = np.asarray(value)

        # Record basic schema information about the array
        entries.append(
            {
                "key": key,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "ndim": int(arr.ndim),
            }
        )

        key_lower = key.lower()
        
        # Check if the array is a transformation candidate based on its shape or key name
        if is_transform_candidate(arr) or "transform" in key_lower or key_lower in {"t", "rt", "pose", "matrix"}:
            transform_candidates.append(key)

        # Check if the array is a mask candidate based on keywords in its name
        if "mask" in key_lower or "fracture" in key_lower or "crack" in key_lower:
            mask_candidates.append(key)

    return {
        "npz_path": str(npz_path.resolve()),
        "keys": entries,
        "transform_candidates": transform_candidates,
        "mask_candidates": mask_candidates,
    }


def main() -> None:
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(description="Inspect .npz schema and candidate keys.")
    parser.add_argument("--metadata-csv", type=Path, required=True, help="Input metadata CSV built by build_metadata.py")
    parser.add_argument("--output-json", type=Path, required=True, help="Path for the output structured JSON summary")
    parser.add_argument("--output-csv", type=Path, required=True, help="Path for the output flat CSV summary")
    args = parser.parse_args()

    # Read the dataset metadata
    df = pd.read_csv(args.metadata_csv)
    # Filter for objects that actually have an .npz file mapped
    df = df[df["path_npz"].astype(str).str.len() > 0].copy()

    summaries = []
    flat_rows = []

    # Process each npz file
    for npz_path_str in df["path_npz"]:
        npz_path = Path(npz_path_str)
        summary = describe_npz(npz_path)
        summaries.append(summary)

        # Flatten the schema info for tabular CSV export
        for item in summary["keys"]:
            flat_rows.append(
                {
                    "npz_path": summary["npz_path"],
                    "key": item["key"],
                    "shape": item["shape"],
                    "dtype": item["dtype"],
                    "ndim": item["ndim"],
                    "is_transform_candidate": item["key"] in summary["transform_candidates"],
                    "is_mask_candidate": item["key"] in summary["mask_candidates"],
                }
            )

    # Ensure the parent directories exist for outputs
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Dump the structured hierarchical data to JSON
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    # Dump the flattened array metadata to CSV
    pd.DataFrame(flat_rows).to_csv(args.output_csv, index=False)

    print(f"Saved JSON summary to: {args.output_json}")
    print(f"Saved flat CSV summary to: {args.output_csv}")

    # Display some summary statistics about the arrays discovered
    all_keys = sorted({row['key'] for row in flat_rows})
    print("\nUnique keys found:")
    for key in all_keys:
        print(f" - {key}")

    print("\nExample transform candidates:")
    for summary in summaries[:5]:
        print(Path(summary["npz_path"]).name, "->", summary["transform_candidates"])

    print("\nExample mask candidates:")
    for summary in summaries[:5]:
        print(Path(summary["npz_path"]).name, "->", summary["mask_candidates"])


if __name__ == "__main__":
    main()