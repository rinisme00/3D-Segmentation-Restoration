"""Build the balanced 9D segmentation manifest.

Composition (all rows feature_mode=9d, stage=segmentation):

  * LABELED Breaking Bad  : every valid manually annotated piece (from
    breakingbad_manual_annotations.csv). Per-piece rows pointing at the RAW mesh
    plus the converted per-vertex labels sidecar. annotation_source_status="converted",
    metadata.label_status="labeled". Splits reused from the annotation manifest.
  * UNLABELED Breaking Bad: a balanced sample of UNannotated pieces for coverage / SSL /
    inference — ~N pieces per everyday category plus a comparable artifact budget.
    mode_0 (completed, no fracture surface) is excluded. Objects already in the labeled
    set are excluded so splits stay object-disjoint. metadata.label_status="unlabeled".
  * FantasticBreaks       : carried through unchanged (model_b_0 has mask labels =>
    label_status="labeled"; model_r_0 / model_c => "unlabeled").

Object-disjoint splits: all variants/pieces of one (subset, object_id) land in one split.
The output marks metadata.label_status in {labeled, unlabeled, review, invalid} per row.
Keep-and-top-up is handled downstream by the builder's --resume flag.
"""

from __future__ import annotations

import argparse
import csv
import pathlib
import random
import sys
from collections import Counter, defaultdict

# Allow importing the manifest schema from the repo's src/ package.
_REPO = pathlib.Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.common.manifests.schema import (  # noqa: E402
    ManifestRow,
    read_manifest_csv,
    validate_manifest_rows,
    write_manifest_csv,
)

csv.field_size_limit(10**7)


# ---------------------------------------------------------------------------
# Object-disjoint split assignment
# ---------------------------------------------------------------------------

def assign_object_splits(object_keys, seed, train_frac=0.7, val_frac=0.15):
    """Return {(subset, object_id): split} with object-disjoint train/val/test."""
    keys = sorted(set(object_keys))
    rng = random.Random(seed)
    rng.shuffle(keys)
    n = len(keys)
    n_train = max(1, int(round(n * train_frac))) if n else 0
    n_val = max(1, int(round(n * val_frac))) if n > 1 else 0
    out = {}
    for i, k in enumerate(keys):
        out[k] = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
    return out


# ---------------------------------------------------------------------------
# Labeled Breaking Bad rows (from the annotation manifest)
# ---------------------------------------------------------------------------

def build_labeled_rows(annotations_manifest: pathlib.Path):
    rows = []
    labeled_objects = set()
    for r in csv.DictReader(open(annotations_manifest)):
        subset, category, object_id = r["subset"], r["category"], r["object_id"]
        variant, piece = r["variant"], r["piece_name"]
        labeled_objects.add((subset, object_id))
        rows.append(ManifestRow(
            dataset_name="BreakingBad",
            stage="segmentation",
            object_id=object_id,
            category=category,
            sample_id=f"{subset}_{category}_{object_id}_{variant}_{piece}_seg_9d",
            source_paths={"mesh": r["raw_obj_path"]},
            label_paths={"labels": r["labels_path"]},
            class_label=None,
            point_labels_available=True,
            feature_mode="9d",
            split=r["split"],
            metadata={
                "annotation_source_status": "converted",
                "label_status": "labeled",
                "subset": subset,
                "variant": variant,
                "piece_id": piece,
                "num_vertices": int(r["num_vertices"]),
                "fracture_fraction": float(r["fracture_fraction"]),
            },
        ))
    return rows, labeled_objects


# ---------------------------------------------------------------------------
# Unlabeled Breaking Bad rows (balanced sample of 'none' variant rows)
# ---------------------------------------------------------------------------

def build_unlabeled_rows(all_rows, labeled_objects, per_category, artifact_budget, seed):
    rng = random.Random(seed)

    # Candidate unlabeled BB 9d variant rows, excluding mode_0 and labeled objects.
    candidates = defaultdict(list)  # bucket -> list[(object_id, ManifestRow)]
    for row in all_rows:
        if row.dataset_name != "BreakingBad" or row.feature_mode != "9d":
            continue
        if str(row.metadata.get("annotation_source_status", "none")) != "none":
            continue
        variant = str(row.metadata.get("variant", ""))
        if variant.startswith("mode_0"):
            continue
        subset = str(row.metadata.get("subset", "artifact" if row.category == "artifact" else "everyday"))
        if (subset, row.object_id) in labeled_objects:
            continue
        bucket = "artifact" if subset == "artifact" else row.category
        candidates[bucket].append((row.object_id, row))

    selected = []
    selected_objects = set()
    coverage = {}
    for bucket, items in candidates.items():
        target = artifact_budget if bucket == "artifact" else per_category
        # Select at the VARIANT level (greedy to ~target pieces). Object-disjointness is
        # a split constraint handled later, so a partial set of an object's variants is fine.
        rng.shuffle(items)
        picked_pieces = 0
        for obj_id, row in items:
            if picked_pieces >= target:
                break
            pc = int(row.metadata.get("piece_count", 0) or 0)
            selected.append(row)
            selected_objects.add((str(row.metadata.get("subset")), obj_id))
            picked_pieces += pc if pc > 0 else 1
        coverage[bucket] = picked_pieces

    return selected, selected_objects, coverage


# ---------------------------------------------------------------------------
# FantasticBreaks passthrough
# ---------------------------------------------------------------------------

def build_fb_rows(all_rows):
    rows = []
    for row in all_rows:
        if row.dataset_name != "FantasticBreaks" or row.feature_mode != "9d":
            continue
        mesh_role = str(row.metadata.get("mesh_role", ""))
        md = dict(row.metadata)
        md["label_status"] = "labeled" if mesh_role == "model_b_0" else "unlabeled"
        row.metadata = md
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build(all_manifest, annotations_manifest, output, per_category, artifact_budget, seed):
    all_rows = read_manifest_csv(all_manifest)

    labeled, labeled_objects = build_labeled_rows(annotations_manifest)
    unlabeled, unlabeled_objects, coverage = build_unlabeled_rows(
        all_rows, labeled_objects, per_category, artifact_budget, seed)
    fb = build_fb_rows(all_rows)

    # Object-disjoint splits for the unlabeled BB objects (labeled splits already set).
    split_map = assign_object_splits(unlabeled_objects, seed=seed)
    for row in unlabeled:
        subset = str(row.metadata.get("subset"))
        row.split = split_map.get((subset, row.object_id), "train")
        md = dict(row.metadata)
        md["label_status"] = "unlabeled"
        row.metadata = md

    out_rows = labeled + unlabeled + fb
    validate_manifest_rows(out_rows, stage="segmentation")  # raises on any invalid row
    write_manifest_csv(output, out_rows)

    # ---- report ----
    everyday_labeled = Counter(r.category for r in labeled if r.metadata.get("subset") == "everyday")
    print(f"Balanced 9D segmentation manifest -> {output}")
    print(f"  LABELED BB pieces: {len(labeled)} "
          f"(artifact={sum(1 for r in labeled if r.metadata.get('subset')=='artifact')}, "
          f"everyday={sum(1 for r in labeled if r.metadata.get('subset')=='everyday')})")
    print(f"  everyday categories with labeled pieces: {len(everyday_labeled)}/20")
    print(f"  UNLABELED BB variant-rows: {len(unlabeled)}  "
          f"(approx pieces per bucket: {dict(sorted(coverage.items()))})")
    print(f"  FantasticBreaks rows: {len(fb)}")
    print(f"  TOTAL rows: {len(out_rows)}")
    lab_splits = Counter(r.split for r in labeled)
    unl_splits = Counter(r.split for r in unlabeled)
    print(f"  labeled splits:   {dict(lab_splits)}")
    print(f"  unlabeled splits: {dict(unl_splits)}")
    # coverage: every everyday category represented across labeled+unlabeled
    everyday_unlabeled = set(b for b in coverage if b != "artifact")
    covered = set(everyday_labeled) | everyday_unlabeled
    print(f"  everyday categories represented (labeled OR unlabeled): {len(covered)}/20")


def parse_args():
    p = argparse.ArgumentParser(description="Build balanced 9D segmentation manifest.")
    p.add_argument("--project-root", required=True)
    p.add_argument("--all-manifest", default=None)
    p.add_argument("--annotations-manifest", default=None)
    p.add_argument("--output", default=None)
    p.add_argument("--unlabeled-per-category", type=int, default=50,
                   help="Approx unannotated pieces per everyday category (default 50).")
    p.add_argument("--artifact-budget", type=int, default=50,
                   help="Approx unannotated artifact pieces (default 50).")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    root = pathlib.Path(args.project_root)
    seg = root / "manifests" / "segmentation"
    build(
        all_manifest=pathlib.Path(args.all_manifest) if args.all_manifest else seg / "all_segmentation_manifest.csv",
        annotations_manifest=pathlib.Path(args.annotations_manifest) if args.annotations_manifest else seg / "breakingbad_manual_annotations.csv",
        output=pathlib.Path(args.output) if args.output else seg / "balanced_segmentation_9d_manifest.csv",
        per_category=args.unlabeled_per_category,
        artifact_budget=args.artifact_budget,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
