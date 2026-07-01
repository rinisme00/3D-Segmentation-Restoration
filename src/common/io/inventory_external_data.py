"""Bounded external dataset inventory utility.

This module intentionally inspects only filesystem metadata. It does not open
HDF5/NPZ/PLY/OBJ contents and must not modify raw datasets.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


DEFAULT_DATASET_SPECS = [
    {
        "name": "breaking_bad",
        "root": "${DATA_ROOT}/BreakingBad",
        "expected_paths": [
            "artifact",
            "everyday",
            "artifact_compressed.zip",
            "everyday_compressed.zip",
        ],
        "notes": [
            "Raw root for Breaking Bad. Inventory must use artifact and everyday categories.",
            "Do not restrict to old 4/20 subsets.",
        ],
    },
    {
        "name": "fantastic_breaks",
        "root": "${DATA_ROOT}/Fantastic_Breaks_v1",
        "expected_names": ["model_c.ply", "model_b_0.ply", "model_r_0.ply"],
        "expected_extensions": [".ply", ".npz", ".npy"],
        "notes": [
            "Raw root for Fantastic Breaks v1. Full dataset should be used later.",
            "Restoration mapping: model_c.ply complete target; model_b_0.ply and model_r_0.ply partial inputs.",
        ],
    },
    {
        "name": "garf",
        "root": "${DATA_ROOT}/garf",
        "expected_paths": ["bone_synthetic.hdf5", "bone_real.hdf5", "fractura_real.hdf5"],
        "expected_extensions": [".hdf5"],
        "notes": [
            "GARF HDF5 root. Schema inspection is deferred until label semantics are confirmed.",
        ],
    },
]

DERIVED_TOP_LEVEL_NAMES = {
    "bb_3d",
    "bb_segmentation",
    "fb_segmentation",
    "external",
    "manifests",
}


@dataclass(frozen=True)
class ScanLimits:
    max_depth: int = 2
    max_files_per_dataset: int = 200
    sample_per_extension: int = 5


def expand_path(value: str, env: dict[str, str]) -> Path:
    expanded = os.path.expandvars(value)
    for key, replacement in env.items():
        expanded = expanded.replace("${" + key + "}", replacement)
    return Path(expanded).expanduser()


def load_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if path.suffix.lower() == ".json":
        return json.loads(path.read_text(encoding="utf-8"))
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "YAML config support requires PyYAML. Use the pointnext env or provide JSON."
        ) from exc
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must contain a mapping: {path}")
    return data


def relative_depth(root: Path, path: Path) -> int:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return 0
    if rel == Path("."):
        return 0
    return len(rel.parts)


def iter_files_bounded(root: Path, max_depth: int, max_files: int) -> Iterable[Path]:
    if not root.exists() or not root.is_dir() or max_files <= 0:
        return
    count = 0
    stack = [root]
    while stack:
        current = stack.pop()
        if relative_depth(root, current) > max_depth:
            continue
        try:
            entries = sorted(current.iterdir(), key=lambda p: p.name)
        except OSError:
            continue
        dirs: list[Path] = []
        for entry in entries:
            if entry.is_file():
                yield entry
                count += 1
                if count >= max_files:
                    return
            elif entry.is_dir() and relative_depth(root, entry) < max_depth:
                dirs.append(entry)
        stack.extend(reversed(dirs))


def list_top_level_folders(root: Path) -> list[str]:
    if not root.exists() or not root.is_dir():
        return []
    try:
        return sorted(path.name for path in root.iterdir() if path.is_dir())
    except OSError:
        return []


def sample_files_by_extension(
    files: Iterable[Path], root: Path, sample_per_extension: int
) -> dict[str, list[str]]:
    samples: dict[str, list[str]] = defaultdict(list)
    for path in files:
        ext = path.suffix.lower() or "<no_ext>"
        if len(samples[ext]) >= sample_per_extension:
            continue
        try:
            samples[ext].append(path.relative_to(root).as_posix())
        except ValueError:
            samples[ext].append(path.as_posix())
    return dict(sorted(samples.items()))


def missing_expected_paths(root: Path, spec: dict[str, Any], files: list[Path]) -> list[str]:
    missing: list[str] = []
    for rel in spec.get("expected_paths", []) or []:
        if not (root / rel).exists():
            missing.append(rel)

    names = set(spec.get("expected_names", []) or [])
    if names:
        seen_names = {path.name for path in files}
        missing.extend(sorted(name for name in names if name not in seen_names))

    extensions = set(spec.get("expected_extensions", []) or [])
    if extensions:
        seen_exts = {path.suffix.lower() for path in files}
        missing.extend(sorted(ext for ext in extensions if ext not in seen_exts))
    return missing


def inventory_dataset(
    spec: dict[str, Any],
    env: dict[str, str],
    limits: ScanLimits,
) -> dict[str, Any]:
    root = expand_path(str(spec["root"]), env)
    files = list(iter_files_bounded(root, limits.max_depth, limits.max_files_per_dataset))
    extensions = sorted({path.suffix.lower() or "<no_ext>" for path in files})
    record_notes = list(spec.get("notes", []) or [])
    if len(files) >= limits.max_files_per_dataset:
        record_notes.append("File scan reached configured max_files_per_dataset limit.")

    return {
        "dataset_name": spec["name"],
        "configured_root": str(root),
        "exists": root.exists(),
        "top_level_folders": list_top_level_folders(root),
        "sample_file_count": len(files),
        "detected_file_types": extensions,
        "sample_files_by_extension": sample_files_by_extension(
            files, root, limits.sample_per_extension
        ),
        "missing_expected_files": missing_expected_paths(root, spec, files),
        "scan_limits": {
            "max_depth": limits.max_depth,
            "max_files_per_dataset": limits.max_files_per_dataset,
            "sample_per_extension": limits.sample_per_extension,
        },
        "notes": record_notes,
    }


def inventory_derived_artifacts(data_root: Path) -> dict[str, Any]:
    present: list[str] = []
    missing: list[str] = []
    for name in sorted(DERIVED_TOP_LEVEL_NAMES):
        if (data_root / name).exists():
            present.append(name)
        else:
            missing.append(name)
    return {
        "data_root": str(data_root),
        "present_top_level_artifacts": present,
        "missing_expected_prior_artifact_names": missing,
        "notes": [
            "These are existing derived/prior-run artifacts and are not produced by this inventory command.",
            "Do not treat these as trusted fresh manifests until later validation tasks.",
        ],
    }


def build_inventory(
    config: dict[str, Any],
    data_root: Path,
    project_root: Path,
    limits: ScanLimits,
) -> dict[str, Any]:
    env = {
        **os.environ,
        "DATA_ROOT": str(data_root),
        "PROJECT_ROOT": str(project_root),
    }
    dataset_specs = config.get("datasets") or DEFAULT_DATASET_SPECS
    return {
        "project_root": str(project_root),
        "data_root": str(data_root),
        "datasets": [
            inventory_dataset(spec, env, limits) for spec in dataset_specs
        ],
        "derived_artifacts": inventory_derived_artifacts(data_root),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a bounded external data inventory.")
    parser.add_argument("--config", type=Path, default=None, help="YAML or JSON config path.")
    parser.add_argument("--data-root", type=Path, default=None, help="Override DATA_ROOT.")
    parser.add_argument("--project-root", type=Path, default=None, help="Override PROJECT_ROOT.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON output path.")
    parser.add_argument("--max-depth", type=int, default=None)
    parser.add_argument("--max-files-per-dataset", type=int, default=None)
    parser.add_argument("--sample-per-extension", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    data_root = args.data_root or Path(os.environ.get("DATA_ROOT", "data"))
    project_root = args.project_root or Path(os.environ.get("PROJECT_ROOT", "."))
    defaults = config.get("scan_defaults") or {}
    limits = ScanLimits(
        max_depth=int(args.max_depth if args.max_depth is not None else defaults.get("max_depth", 2)),
        max_files_per_dataset=int(
            args.max_files_per_dataset
            if args.max_files_per_dataset is not None
            else defaults.get("max_files_per_dataset", 200)
        ),
        sample_per_extension=int(
            args.sample_per_extension
            if args.sample_per_extension is not None
            else defaults.get("sample_per_extension", 5)
        ),
    )
    inventory = build_inventory(config, data_root.resolve(), project_root.resolve(), limits)
    text = json.dumps(inventory, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote inventory JSON: {args.output_json}")
    print(text)


if __name__ == "__main__":
    main()
