"""Generate fb_splits.json for the new NPZ-based segmentation dataset.

Reads train/val/test object IDs from the canonical FB segmentation HDF5
(data/fb_segmentation/fb_segmentation_official_mask_9d.h5) and writes a JSON
file that the updated FantasticBreaksSeg NPZ loader reads.

Usage:
    python -m src.segmentation.datasets.generate_splits \\
        --h5-path ../data/fb_segmentation/fb_segmentation_official_mask_9d.h5 \\
        --output-json ../preprocessed/segmentation/pointnext_9d_labeled/fb_splits.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate fb_splits.json from HDF5 splits.")
    parser.add_argument("--h5-path", type=Path, required=True,
                        help="Path to fb_segmentation_official_mask_9d.h5.")
    parser.add_argument("--output-json", type=Path, required=True,
                        help="Output path for fb_splits.json.")
    args = parser.parse_args()

    if not args.h5_path.exists():
        print(f"ERROR: {args.h5_path} not found", file=sys.stderr)
        sys.exit(1)

    import h5py
    with h5py.File(str(args.h5_path), "r") as f:
        splits = {
            split: [
                s.decode() if isinstance(s, bytes) else s
                for s in f["splits"][split][:]
            ]
            for split in f["splits"].keys()
        }

    # HDF5 keys look like '00002__b_0'; extract just the object_id part
    splits_clean = {
        split: [k.replace("__b_0", "").strip() for k in keys]
        for split, keys in splits.items()
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(splits_clean, indent=2))
    print(f"Wrote {args.output_json}")
    for split, ids in splits_clean.items():
        print(f"  {split}: {len(ids)} objects")


if __name__ == "__main__":
    main()
