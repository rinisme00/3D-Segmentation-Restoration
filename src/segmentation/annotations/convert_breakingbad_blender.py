"""Convert Blender-annotated Breaking Bad OBJ files to per-vertex label arrays.

Each piece_N_annotated.obj has faces grouped under either:
  usemtl fracture_surface   -> label 1
  usemtl intact_surface     -> label 0

A vertex is fracture (1) if it belongs to at least one fracture_surface face.
Output: piece_N.labels.npy  shape (V,) dtype uint8  alongside a manifest CSV.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Iterator

import numpy as np


# ---------------------------------------------------------------------------
# OBJ parsing
# ---------------------------------------------------------------------------

FRACTURE_MATERIAL = "fracture_surface"
INTACT_MATERIAL = "intact_surface"
KNOWN_MATERIALS = {FRACTURE_MATERIAL, INTACT_MATERIAL}
FRACTURE_ALIASES = {FRACTURE_MATERIAL}

# Known Blender annotation material-name typos.
# Code-only tolerance: alias the typo to the canonical name, never edit raw files.
MATERIAL_ALIASES = {
    "fractured_surface": FRACTURE_MATERIAL,  # 'fractured_surface' -> 'fracture_surface'
}

import re as _re
_MATERIAL_SUFFIX_RE = _re.compile(r"\.\d+$")


def _normalize_material(name: str, typo_log: set | None = None) -> str:
    """Normalize a material name for label assignment.

    Strips the Blender deduplication suffix (.001, .002, …) and applies the known
    typo alias map (AGENTS 13.6). When ``typo_log`` is given, every alias that fires
    is recorded as ``("material_alias", observed, canonical)`` for reporting.
    """
    base = _MATERIAL_SUFFIX_RE.sub("", name.strip())
    canonical = MATERIAL_ALIASES.get(base)
    if canonical is not None:
        if typo_log is not None:
            typo_log.add(("material_alias", base, canonical))
        return canonical
    return base


def _piece_name_from_annotated(obj_file: Path) -> str:
    """Derive the raw piece name from an annotated OBJ filename.

    Tolerant of the double-extension typo (AGENTS 13.6): both
    ``piece_N_annotated.obj`` and ``piece_N_annotated.mtl.obj`` resolve to ``piece_N``.
    """
    name = obj_file.name
    for suffix in (".mtl.obj", ".obj"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
            break
    return name.replace("_annotated", "")


def _iter_annotated_obj_files(var_dir: Path):
    """Yield annotated OBJ files in a variant dir, tolerant of double extensions."""
    seen: set[str] = set()
    files = list(var_dir.glob("piece_*_annotated.obj")) + list(var_dir.glob("piece_*_annotated.mtl.obj"))
    for obj_file in sorted(files):
        if obj_file.name in seen:
            continue
        seen.add(obj_file.name)
        yield obj_file


def _parse_vertex_index(token: str) -> int:
    """Extract 1-based vertex index from OBJ face token (v, v/t, v//n, v/t/n)."""
    return int(token.split("/")[0])


def parse_annotated_obj(path: Path, typo_log: set | None = None) -> np.ndarray:
    """Return per-vertex label array (uint8, 0=intact 1=fracture) for an annotated OBJ.

    Raises ValueError if unknown material names are found, or if faces reference
    out-of-range vertex indices. Known material-name typos (AGENTS 13.6) are aliased
    via ``_normalize_material`` and recorded in ``typo_log`` rather than rejected.
    """
    vertex_count = 0
    # vertex_fracture[i] = True if vertex i+1 (1-based) is fracture
    vertex_fracture: list[bool] = []
    current_material: str | None = None
    unknown_materials: set[str] = set()

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("v "):
                vertex_count += 1
                vertex_fracture.append(False)

            elif line.startswith("usemtl "):
                mat = _normalize_material(line[7:].strip(), typo_log=typo_log)
                if mat not in KNOWN_MATERIALS:
                    unknown_materials.add(mat)
                current_material = mat

            elif line.startswith("f "):
                if current_material is None:
                    continue
                is_frac = current_material in FRACTURE_ALIASES
                tokens = line[2:].split()
                for tok in tokens:
                    vi = _parse_vertex_index(tok) - 1  # 0-based
                    if vi < 0 or vi >= vertex_count:
                        raise ValueError(
                            f"{path}: face vertex index {vi + 1} out of range "
                            f"(vertex_count={vertex_count})"
                        )
                    if is_frac:
                        vertex_fracture[vi] = True

    if unknown_materials:
        raise ValueError(
            f"{path}: unknown material(s): {unknown_materials}. "
            f"Expected {KNOWN_MATERIALS}"
        )

    if vertex_count == 0:
        raise ValueError(f"{path}: no vertices found")

    return np.array(vertex_fracture, dtype=np.uint8)


def count_vertices_obj(path: Path) -> int:
    """Fast vertex count for a raw OBJ (no full parse needed)."""
    count = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if line.startswith("v "):
                count += 1
    return count


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _iter_annotated_objs(annotations_root: Path) -> Iterator[dict]:
    """Yield metadata dicts for every piece_N_annotated.obj under annotations_root."""
    for subset in ("artifact", "everyday"):
        subset_dir = annotations_root / subset
        if not subset_dir.is_dir():
            continue

        for obj_dir in sorted(subset_dir.iterdir()):
            if not obj_dir.is_dir():
                continue

            if subset == "artifact":
                object_id = obj_dir.name
                category = "artifact"
                variant_dirs = sorted(obj_dir.iterdir())
            else:
                # everyday: obj_dir is the category, next level is object_id
                category = obj_dir.name
                variant_dirs_list = []
                for oid_dir in sorted(obj_dir.iterdir()):
                    if not oid_dir.is_dir():
                        continue
                    object_id = oid_dir.name
                    for var_dir in sorted(oid_dir.iterdir()):
                        if var_dir.is_dir():
                            variant_dirs_list.append((object_id, var_dir))
                for object_id, var_dir in variant_dirs_list:
                    for obj_file in _iter_annotated_obj_files(var_dir):
                        yield {
                            "subset": subset,
                            "category": category,
                            "object_id": object_id,
                            "variant": var_dir.name,
                            "piece_name": _piece_name_from_annotated(obj_file),
                            "annotated_obj": obj_file,
                            "filename_typo": obj_file.name.endswith(".mtl.obj"),
                        }
                continue  # already yielded above

            for var_dir in variant_dirs:
                if not var_dir.is_dir():
                    continue
                for obj_file in _iter_annotated_obj_files(var_dir):
                    yield {
                        "subset": subset,
                        "category": category,
                        "object_id": object_id,
                        "variant": var_dir.name,
                        "piece_name": _piece_name_from_annotated(obj_file),
                        "annotated_obj": obj_file,
                        "filename_typo": obj_file.name.endswith(".mtl.obj"),
                    }


# ---------------------------------------------------------------------------
# Split assignment (object-disjoint)
# ---------------------------------------------------------------------------

def assign_splits(
    records: list[dict],
    seed: int,
    train_frac: float = 0.7,
    val_frac: float = 0.15,
) -> None:
    """Mutates each record in-place to add a 'split' key (train/val/test).

    Split is assigned at the (subset, object_id) level so all variants and
    pieces of the same object land in the same split.
    """
    unique_ids = sorted({(r["subset"], r["object_id"]) for r in records})
    rng = random.Random(seed)
    rng.shuffle(unique_ids)

    n = len(unique_ids)
    n_train = max(1, int(round(n * train_frac)))
    n_val = max(1, int(round(n * val_frac)))

    split_map: dict[tuple, str] = {}
    for i, key in enumerate(unique_ids):
        if i < n_train:
            split_map[key] = "train"
        elif i < n_train + n_val:
            split_map[key] = "val"
        else:
            split_map[key] = "test"

    for r in records:
        r["split"] = split_map[(r["subset"], r["object_id"])]


# ---------------------------------------------------------------------------
# Main converter
# ---------------------------------------------------------------------------

def convert(
    annotations_root: Path,
    raw_root: Path,
    output_manifest: Path,
    write_labels_to: Path | None,
    split_seed: int,
    limit: int | None,
    debug: bool,
) -> None:
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    if write_labels_to is not None:
        write_labels_to.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    errors: list[str] = []
    typo_resolutions: list[str] = []  # AGENTS 13.6 typo handling log
    processed = 0

    for meta in _iter_annotated_objs(annotations_root):
        if limit is not None and processed >= limit:
            break

        ann_path: Path = meta["annotated_obj"]
        # piece_name is already tolerant of the double-extension typo (.mtl.obj).
        raw_stem = meta["piece_name"]

        # Log the double-extension filename typo (resolved by tolerant discovery).
        if meta.get("filename_typo"):
            typo_resolutions.append(
                f"filename_double_ext: {ann_path} -> resolved piece '{raw_stem}' "
                f"(recognized *.mtl.obj as annotated OBJ; raw files unchanged)"
            )

        # Locate raw counterpart
        rel = ann_path.relative_to(annotations_root)
        raw_path = raw_root / rel.parent / (raw_stem + ".obj")

        if not raw_path.exists():
            msg = f"MISSING raw: {raw_path}"
            errors.append(msg)
            if debug:
                print(f"[SKIP] {msg}", file=sys.stderr)
            continue

        # Parse annotated OBJ (material typos aliased + logged, not rejected)
        material_typos: set = set()
        try:
            labels = parse_annotated_obj(ann_path, typo_log=material_typos)
        except (ValueError, OSError) as exc:
            errors.append(f"PARSE ERROR {ann_path}: {exc}")
            if debug:
                print(f"[ERROR] {exc}", file=sys.stderr)
            continue
        for _kind, observed, canonical in sorted(material_typos):
            typo_resolutions.append(
                f"material_alias: {ann_path} -> '{observed}' treated as '{canonical}'"
            )

        # Validate vertex count against raw
        raw_vcount = count_vertices_obj(raw_path)
        if raw_vcount != len(labels):
            msg = (
                f"VERTEX MISMATCH {ann_path.name}: "
                f"annotated={len(labels)} raw={raw_vcount}"
            )
            errors.append(msg)
            if debug:
                print(f"[ERROR] {msg}", file=sys.stderr)
            continue

        # Save labels sidecar
        labels_path: Path | None = None
        if write_labels_to is not None:
            rel_dir = ann_path.relative_to(annotations_root).parent
            labels_dir = write_labels_to / rel_dir
            labels_dir.mkdir(parents=True, exist_ok=True)
            labels_path = labels_dir / (raw_stem + ".labels.npy")
            np.save(labels_path, labels)

        fracture_frac = float(labels.mean())
        record = {
            "sample_id": f"{meta['subset']}__{meta['object_id']}__{meta['variant']}__{meta['piece_name']}",
            "subset": meta["subset"],
            "category": meta["category"],
            "object_id": meta["object_id"],
            "variant": meta["variant"],
            "piece_name": meta["piece_name"],
            "annotated_obj_path": str(ann_path),
            "raw_obj_path": str(raw_path),
            "labels_path": str(labels_path) if labels_path else "",
            "num_vertices": len(labels),
            "fracture_fraction": f"{fracture_frac:.6f}",
            "split": "",  # filled by assign_splits
        }
        records.append(record)
        processed += 1

        if debug:
            print(
                f"[OK] {meta['subset']}/{meta['object_id']}/{meta['variant']}/{meta['piece_name']} "
                f"V={len(labels)} frac={fracture_frac:.3f}"
            )

    # Assign object-disjoint splits
    if records:
        assign_splits(records, seed=split_seed)

    # Write manifest CSV
    fieldnames = [
        "sample_id", "subset", "category", "object_id", "variant", "piece_name",
        "split", "num_vertices", "fracture_fraction",
        "annotated_obj_path", "raw_obj_path", "labels_path",
    ]
    with open(output_manifest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)

    # Summary
    splits = {}
    for r in records:
        splits[r["split"]] = splits.get(r["split"], 0) + 1
    fracs = [float(r["fracture_fraction"]) for r in records]
    mean_frac = sum(fracs) / len(fracs) if fracs else 0.0

    print(f"Converted: {len(records)} pieces  Errors: {len(errors)}")
    print(f"Splits: { {k: splits.get(k,0) for k in ('train','val','test')} }")
    print(f"Mean fracture fraction: {mean_frac:.4f}")
    print(f"Typo resolutions (AGENTS 13.6): {len(typo_resolutions)}")
    for t in typo_resolutions:
        print(f"  {t}")
    if errors:
        print(f"\nErrors ({len(errors)}):")
        for e in errors[:10]:
            print(f"  {e}")
    print(f"Manifest: {output_manifest}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Convert Breaking Bad Blender OBJ annotations to per-vertex label arrays."
    )
    p.add_argument("--annotations-root", required=True,
                   help="Root of annotations tree (contains artifact/ everyday/)")
    p.add_argument("--raw-root", required=True,
                   help="Root of raw Breaking Bad OBJ data (contains artifact/ everyday/)")
    p.add_argument("--output-manifest", required=True,
                   help="Path to write manifest CSV")
    p.add_argument("--write-labels-to", default=None,
                   help="Directory to write .labels.npy sidecars (optional)")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--limit", type=int, default=None,
                   help="Process at most N pieces (for smoke testing)")
    p.add_argument("--debug", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    convert(
        annotations_root=Path(args.annotations_root),
        raw_root=Path(args.raw_root),
        output_manifest=Path(args.output_manifest),
        write_labels_to=Path(args.write_labels_to) if args.write_labels_to else None,
        split_seed=args.split_seed,
        limit=args.limit,
        debug=args.debug,
    )


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(1)
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(2)
