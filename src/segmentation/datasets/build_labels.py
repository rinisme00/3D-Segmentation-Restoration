"""Add per-point fracture labels to segmentation NPZ feature files.

For FantasticBreaks:
  - Reads existing NPZ files (points + features, no labels) from preprocessed dir.
  - Loads meta_0.npz (per-vertex mask on model_b_0.ply in FB-normalized space).
  - KNN from sampled 8192 points -> model_b_0 vertices -> labels (0=intact, 1=fracture).
  - Writes labeled NPZ to output_dir (same structure).

Only FantasticBreaks/model_b_0 is handled:
  - meta_0.npz mask is per-vertex on model_b_0.ply only; model_r_0 (fragment) is excluded.
  - BB Blender label parsing is a separate task.

Usage (smoke test):
    python -m src.segmentation.datasets.build_labels \\
        --preprocessed-dir ../preprocessed/segmentation/pointnext_9d \\
        --data-root ../data \\
        --output-dir /tmp/seg_labeled_test \\
        --dataset FantasticBreaks \\
        --limit 4 --debug

Usage (full run, foreground):
    python -m src.segmentation.datasets.build_labels \\
        --preprocessed-dir ../preprocessed/segmentation/pointnext_9d \\
        --data-root ../data \\
        --output-dir ../preprocessed/segmentation/pointnext_9d_labeled \\
        --dataset FantasticBreaks
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


import re as _re

_PIECE_IDX_SUFFIX_RE = _re.compile(r"_\d{4}$")


def _bb_piece_name(npz_stem: str) -> str:
    """``piece_0_0000`` -> ``piece_0`` (strip the builder's mesh-index suffix)."""
    return _PIECE_IDX_SUFFIX_RE.sub("", npz_stem)


def _bb_paths(preprocessed_dir: Path, npz_path: Path, data_root: Path,
              annotations_root: Path) -> tuple[Path, Path]:
    """Return (raw_obj, labels_npy) for a Breaking Bad sample NPZ.

    NPZ path structure (from the shared builder):
      <preprocessed_dir>/samples/BreakingBad/<subset>/[<category>/]<obj_id>/<variant>/<mode>/<stem>.npz
        - artifact: samples/BreakingBad/artifact/<obj_id>/<variant>/<mode>/<stem>.npz
        - everyday: samples/BreakingBad/everyday/<category>/<obj_id>/<variant>/<mode>/<stem>.npz
    Raw mesh:
      <data_root>/BreakingBad/<subset>/[<category>/]<obj_id>/<variant>/<piece>.obj
    Per-vertex labels (from convert_breakingbad_blender):
      <annotations_root>/labels/<subset>/[<category>/]<obj_id>/<variant>/<piece>.labels.npy
    """
    rel = npz_path.relative_to(preprocessed_dir / "samples" / "BreakingBad")
    parts = rel.parts
    subset = parts[0]
    # parts: subset, [category,] obj_id, variant, mode, filename
    inner = parts[1:-3]  # ([category,] obj_id) — variant/mode/file are the last 3
    variant = parts[-3]
    piece = _bb_piece_name(npz_path.stem)
    rel_obj_dir = Path(subset).joinpath(*inner, variant)
    raw_obj = data_root / "BreakingBad" / rel_obj_dir / f"{piece}.obj"
    labels_npy = annotations_root / "labels" / rel_obj_dir / f"{piece}.labels.npy"
    return raw_obj, labels_npy


def _process_one_bb(npz_path: Path, preprocessed_dir: Path, data_root: Path,
                    annotations_root: Path, output_dir: Path, debug: bool) -> dict[str, object]:
    """Process one BB NPZ: attach per-point labels via KNN to the raw mesh vertices."""
    import trimesh

    raw_obj, labels_npy = _bb_paths(preprocessed_dir, npz_path, data_root, annotations_root)
    if not labels_npy.exists():
        return {"status": "skip_no_label", "path": str(npz_path)}
    if not raw_obj.exists():
        return {"status": "skip_no_mesh", "path": str(npz_path)}

    sample = np.load(str(npz_path), allow_pickle=True)
    meta = json.loads(str(sample["metadata"]))
    centroid = np.array(meta["normalization"]["centroid"], dtype=np.float32)
    scale = float(meta["normalization"]["scale"])
    points_norm = sample["points"]

    # Raw mesh vertices in original OBJ order (matches the .labels.npy index order).
    mesh = trimesh.load(str(raw_obj), process=False)
    mesh_verts = np.asarray(mesh.vertices, dtype=np.float32)
    vertex_labels = np.load(str(labels_npy))  # (V,) uint8

    if len(vertex_labels) != len(mesh_verts):
        return {
            "status": "skip_vertex_mismatch",
            "path": str(npz_path),
            "detail": f"labels={len(vertex_labels)} raw_verts={len(mesh_verts)}",
        }

    labels = _assign_labels(points_norm, centroid, scale, mesh_verts, vertex_labels)
    frac = labels.mean()
    if debug:
        print(f"  {npz_path.name}: fracture={frac:.3%} intact={(1-frac):.3%}")

    out_path = output_dir / npz_path.relative_to(preprocessed_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        points=sample["points"],
        features=sample["features"],
        labels=labels,
        metadata=sample["metadata"],
    )
    return {"status": "ok", "fracture_frac": float(frac), "path": str(out_path)}


def _fb_paths(preprocessed_dir: Path, npz_path: Path, data_root: Path) -> tuple[Path, Path]:
    """
    Return (model_b0_ply, meta_npz) for a given FB sample NPZ.

    Expected path structure:
      <preprocessed_dir>/samples/FantasticBreaks/<cat>/<obj_id>/<variant>/<mode>/<stem>.npz
    Corresponding data:
      <data_root>/Fantastic_Breaks_v1/<cat>/<obj_id>/model_b_0.ply
      <data_root>/Fantastic_Breaks_v1/<cat>/<obj_id>/meta_0.npz
    """
    # strip preprocessed_dir/samples/FantasticBreaks
    rel = npz_path.relative_to(preprocessed_dir / "samples" / "FantasticBreaks")
    parts = rel.parts  # cat, obj_id, variant, mode, filename
    cat, obj_id = parts[0], parts[1]
    fb_dir = data_root / "Fantastic_Breaks_v1" / cat / obj_id
    return fb_dir / "model_b_0.ply", fb_dir / "meta_0.npz"


def _load_mesh_vertices(ply_path: Path) -> np.ndarray:
    """Load PLY vertices without processing (preserves FB-normalized coordinates)."""
    import trimesh
    mesh = trimesh.load(str(ply_path), process=False)
    return np.asarray(mesh.vertices, dtype=np.float32)


def _assign_labels(
    points_norm: np.ndarray,
    centroid: np.ndarray,
    scale: float,
    mesh_verts: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    KNN label assignment.

    Points are in NPZ-normalized space. Mesh vertices are in FB-normalized space.
    Un-normalize points to FB space, then query KD-tree on mesh vertices.

    Returns: labels (N,) uint8, 0=intact, 1=fracture.
    """
    pts_orig = points_norm * scale + centroid      # (N, 3) in FB-normalized space
    tree = cKDTree(mesh_verts)
    _, nn_idx = tree.query(pts_orig, k=1, workers=1)
    return mask[nn_idx].astype(np.uint8)


def _process_one(
    npz_path: Path,
    preprocessed_dir: Path,
    data_root: Path,
    output_dir: Path,
    debug: bool,
) -> dict[str, object]:
    """Process one FB NPZ: add labels, write to output_dir. Returns stats dict."""
    mesh_path, meta_path = _fb_paths(preprocessed_dir, npz_path, data_root)

    if not mesh_path.exists():
        return {"status": "skip_no_mesh", "path": str(npz_path)}
    if not meta_path.exists():
        return {"status": "skip_no_mask", "path": str(npz_path)}

    sample = np.load(str(npz_path), allow_pickle=True)
    meta = json.loads(str(sample["metadata"]))

    centroid = np.array(meta["normalization"]["centroid"], dtype=np.float32)
    scale = float(meta["normalization"]["scale"])
    points_norm = sample["points"]              # (8192, 3)

    mesh_verts = _load_mesh_vertices(mesh_path)  # (M, 3)
    mask_data = np.load(str(meta_path))
    mask = mask_data["mask"]                    # (M,) bool

    labels = _assign_labels(points_norm, centroid, scale, mesh_verts, mask)

    frac = labels.mean()
    if debug:
        print(f"  {npz_path.name}: fracture={frac:.3%} intact={(1-frac):.3%}")

    # Write labeled NPZ preserving all original keys
    out_path = output_dir / npz_path.relative_to(preprocessed_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        points=sample["points"],
        features=sample["features"],
        labels=labels,
        metadata=sample["metadata"],
    )

    return {"status": "ok", "fracture_frac": float(frac), "path": str(out_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build per-point labels for segmentation NPZ files.")
    parser.add_argument("--preprocessed-dir", type=Path, required=True,
                        help="Root of preprocessed segmentation NPZ tree.")
    parser.add_argument("--data-root", type=Path, required=True,
                        help="Root data directory (contains Fantastic_Breaks_v1/, etc.).")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Output directory for labeled NPZ files (same tree structure).")
    parser.add_argument("--dataset", choices=["FantasticBreaks", "BreakingBad"], default="FantasticBreaks",
                        help="Dataset to process.")
    parser.add_argument("--annotations-root", type=Path, default=None,
                        help="BreakingBad annotations root (contains labels/); required for --dataset BreakingBad.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of samples to process (for smoke tests).")
    parser.add_argument("--debug", action="store_true",
                        help="Print per-sample stats.")
    args = parser.parse_args()

    if args.dataset == "BreakingBad" and args.annotations_root is None:
        print("ERROR: --annotations-root is required for --dataset BreakingBad", file=sys.stderr)
        sys.exit(1)

    src_dir = args.preprocessed_dir / "samples" / args.dataset
    if not src_dir.exists():
        print(f"ERROR: {src_dir} does not exist", file=sys.stderr)
        sys.exit(1)

    # For FB: only model_b_0 has official mask labels. model_r_0 is the detached
    # fragment whose mask cannot be derived from meta_0.npz via KNN.
    npz_files = sorted(
        p for p in src_dir.rglob("*.npz")
        if args.dataset != "FantasticBreaks" or "/model_b_0/" in str(p)
    )
    if not npz_files:
        print(f"ERROR: no .npz files found under {src_dir}", file=sys.stderr)
        sys.exit(1)

    if args.limit:
        npz_files = npz_files[: args.limit]

    print(f"Processing {len(npz_files)} samples from {src_dir}")

    results = []
    for npz_path in npz_files:
        if args.debug:
            print(f"Processing {npz_path.relative_to(args.preprocessed_dir)}")
        if args.dataset == "BreakingBad":
            result = _process_one_bb(
                npz_path,
                args.preprocessed_dir,
                args.data_root,
                args.annotations_root,
                args.output_dir,
                args.debug,
            )
        else:
            result = _process_one(
                npz_path,
                args.preprocessed_dir,
                args.data_root,
                args.output_dir,
                args.debug,
            )
        results.append(result)

    ok = [r for r in results if r["status"] == "ok"]
    skipped = [r for r in results if r["status"] != "ok"]
    frac_vals = [r["fracture_frac"] for r in ok]
    print(f"\nDone: {len(ok)} labeled, {len(skipped)} skipped")
    if frac_vals:
        print(f"Fracture fraction: min={min(frac_vals):.3%} mean={np.mean(frac_vals):.3%} max={max(frac_vals):.3%}")
    for r in skipped:
        print(f"  SKIP ({r['status']}): {r['path']}")


if __name__ == "__main__":
    main()
