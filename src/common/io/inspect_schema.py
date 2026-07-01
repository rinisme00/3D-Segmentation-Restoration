"""Safe schema inspection for point-cloud and array files.

The inspector is intentionally conservative: it reports metadata, shapes,
dtypes, tiny samples, and simple geometry counts without writing or
preprocessing source data.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
from pathlib import Path
from typing import Any

import numpy as np


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return str(value)
    return value


def safe_attrs(attrs: Any, max_array_values: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in attrs.items():
        arr = np.asarray(value)
        entry: dict[str, Any] = {
            "dtype": str(arr.dtype),
            "shape": list(arr.shape),
        }
        if arr.size <= max_array_values:
            entry["value"] = json_safe(arr)
        else:
            entry["sample"] = json_safe(arr.reshape(-1)[:max_array_values])
            entry["truncated"] = True
        result[str(key)] = entry
    return result


def candidate_roles(name: str, shape: list[int], dtype: str) -> list[str]:
    lowered = name.strip("/").split("/")[-1].lower()
    roles: list[str] = []
    is_face_index = lowered in {"faces", "face", "shared_faces"}
    if lowered in {"faces", "face"}:
        roles.append("faces_candidate")
    if "shared_faces" in lowered:
        roles.append("shared_faces_candidate")
    if any(token in lowered for token in ("mask", "crack")):
        roles.append("mask_candidate")
    if any(token in lowered for token in ("label", "seg", "class")):
        roles.append("label_candidate")
    if lowered in {"rt", "transform", "matrix", "pose", "extrinsic", "intrinsic"}:
        roles.append("transform_candidate")
    if any(token in lowered for token in ("point", "coord", "xyz", "vertices")):
        roles.append("points_candidate")
    if "normal" in lowered:
        roles.append("normals_candidate")
    if not is_face_index and len(shape) >= 2 and shape[-1] in (3, 6, 9):
        roles.append(f"feature_{shape[-1]}d_candidate")
    if dtype == "bool" and len(shape) >= 1:
        roles.append("binary_mask_candidate")
    return sorted(set(roles))


def small_array_sample(arr: np.ndarray, max_array_values: int) -> dict[str, Any]:
    flat = arr.reshape(-1) if arr.shape else arr.reshape(1)
    sample_size = min(int(flat.size), max_array_values)
    result: dict[str, Any] = {
        "sample_value_count": sample_size,
    }
    if sample_size > 0:
        result["sample_values"] = json_safe(flat[:sample_size])
    if flat.size > sample_size:
        result["truncated"] = True
    return result


def inspect_hdf5(
    path: Path, max_array_values: int, include_samples: bool, max_hdf5_nodes: int
) -> dict[str, Any]:
    import h5py

    nodes: list[dict[str, Any]] = []
    truncated = False

    def visit(name: str, obj: Any) -> str | None:
        nonlocal truncated
        if len(nodes) >= max_hdf5_nodes:
            truncated = True
            return "STOP"
        node: dict[str, Any] = {
            "name": name or "/",
            "type": "group" if isinstance(obj, h5py.Group) else "dataset",
            "attrs": safe_attrs(obj.attrs, max_array_values),
        }
        if isinstance(obj, h5py.Dataset):
            shape = list(obj.shape)
            dtype = str(obj.dtype)
            node.update(
                {
                    "shape": shape,
                    "dtype": dtype,
                    "ndim": int(obj.ndim),
                    "size": int(obj.size),
                    "chunks": list(obj.chunks) if obj.chunks else None,
                    "compression": obj.compression,
                    "candidate_roles": candidate_roles(name, shape, dtype),
                }
            )
            if include_samples and obj.size <= max_array_values:
                try:
                    node["sample"] = json_safe(obj[()])
                except Exception as exc:
                    node["sample_error"] = str(exc)
            elif include_samples and obj.shape:
                try:
                    slices = tuple(slice(0, min(dim, 1)) for dim in obj.shape)
                    node["sample"] = json_safe(np.asarray(obj[slices]).reshape(-1)[:max_array_values])
                    node["sample_truncated"] = True
                except Exception as exc:
                    node["sample_error"] = str(exc)
        nodes.append(node)
        return None

    with h5py.File(path, "r") as handle:
        visit("/", handle)
        handle.visititems(visit)
    return {
        "format": "hdf5",
        "nodes": nodes,
        "node_count_reported": len(nodes),
        "truncated_nodes": truncated,
        "max_hdf5_nodes": max_hdf5_nodes,
    }


def inspect_npz(path: Path, max_array_values: int, include_samples: bool) -> dict[str, Any]:
    entries: list[dict[str, Any]] = []
    with np.load(path, allow_pickle=False) as data:
        for key in data.files:
            arr = np.asarray(data[key])
            entry: dict[str, Any] = {
                "key": key,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "ndim": int(arr.ndim),
                "size": int(arr.size),
                "candidate_roles": candidate_roles(key, list(arr.shape), str(arr.dtype)),
            }
            if include_samples and arr.size <= max_array_values:
                entry["sample"] = json_safe(arr)
            elif include_samples:
                entry.update(small_array_sample(arr, max_array_values))
            entries.append(entry)
    return {"format": "npz", "arrays": entries}


def inspect_npy(path: Path, max_array_values: int, include_samples: bool) -> dict[str, Any]:
    arr = np.load(path, allow_pickle=False, mmap_mode="r")
    result: dict[str, Any] = {
        "format": "npy",
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "ndim": int(arr.ndim),
        "size": int(arr.size),
        "candidate_roles": candidate_roles(path.name, list(arr.shape), str(arr.dtype)),
    }
    if include_samples:
        result.update(small_array_sample(np.asarray(arr), max_array_values))
    return result


def parse_ply_header(path: Path) -> tuple[dict[str, Any], int]:
    header: dict[str, Any] = {"elements": [], "format": None}
    current: dict[str, Any] | None = None
    offset = 0
    with path.open("rb") as f:
        while True:
            line_bytes = f.readline()
            if not line_bytes:
                raise ValueError("PLY header ended unexpectedly")
            offset += len(line_bytes)
            line = line_bytes.decode("utf-8", errors="replace").strip()
            if line.startswith("format "):
                header["format"] = line.split()[1]
            elif line.startswith("element "):
                parts = line.split()
                current = {"name": parts[1], "count": int(parts[2]), "properties": []}
                header["elements"].append(current)
            elif line.startswith("property ") and current is not None:
                current["properties"].append(line)
            elif line == "end_header":
                break
    return header, offset


PLY_SCALAR_FORMATS = {
    "char": ("b", 1),
    "uchar": ("B", 1),
    "int8": ("b", 1),
    "uint8": ("B", 1),
    "short": ("h", 2),
    "ushort": ("H", 2),
    "int16": ("h", 2),
    "uint16": ("H", 2),
    "int": ("i", 4),
    "uint": ("I", 4),
    "int32": ("i", 4),
    "uint32": ("I", 4),
    "float": ("f", 4),
    "float32": ("f", 4),
    "double": ("d", 8),
    "float64": ("d", 8),
}


def inspect_ply_ascii_vertices(path: Path, vertex_count: int, header_offset: int) -> dict[str, Any]:
    bbox_min = [float("inf"), float("inf"), float("inf")]
    bbox_max = [float("-inf"), float("-inf"), float("-inf")]
    seen = 0
    with path.open("rb") as f:
        f.seek(header_offset)
        for _ in range(vertex_count):
            line = f.readline().decode("utf-8", errors="replace").strip()
            if not line:
                break
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                xyz = [float(parts[0]), float(parts[1]), float(parts[2])]
            except ValueError:
                continue
            for idx in range(3):
                bbox_min[idx] = min(bbox_min[idx], xyz[idx])
                bbox_max[idx] = max(bbox_max[idx], xyz[idx])
            seen += 1
    return {"bbox_min": bbox_min, "bbox_max": bbox_max, "vertices_read_for_bbox": seen}


def inspect_ply_binary_vertices(
    path: Path, vertex_count: int, header: dict[str, Any], header_offset: int
) -> dict[str, Any]:
    vertex_element = next((e for e in header["elements"] if e["name"] == "vertex"), None)
    if not vertex_element:
        return {}
    scalar_props: list[tuple[str, str]] = []
    for prop in vertex_element["properties"]:
        parts = prop.split()
        if len(parts) == 3 and parts[1] in PLY_SCALAR_FORMATS:
            scalar_props.append((parts[2], parts[1]))
        else:
            return {"bbox_note": "binary PLY vertex properties include unsupported list fields"}
    endian = "<" if header["format"] == "binary_little_endian" else ">"
    fmt = endian + "".join(PLY_SCALAR_FORMATS[type_name][0] for _, type_name in scalar_props)
    stride = struct.calcsize(fmt)
    prop_names = [name for name, _ in scalar_props]
    if not all(axis in prop_names for axis in ("x", "y", "z")):
        return {"bbox_note": "binary PLY has no x/y/z scalar properties"}
    xyz_idx = [prop_names.index(axis) for axis in ("x", "y", "z")]
    bbox_min = [float("inf"), float("inf"), float("inf")]
    bbox_max = [float("-inf"), float("-inf"), float("-inf")]
    seen = 0
    with path.open("rb") as f:
        f.seek(header_offset)
        for _ in range(vertex_count):
            chunk = f.read(stride)
            if len(chunk) < stride:
                break
            values = struct.unpack(fmt, chunk)
            xyz = [float(values[i]) for i in xyz_idx]
            for idx in range(3):
                bbox_min[idx] = min(bbox_min[idx], xyz[idx])
                bbox_max[idx] = max(bbox_max[idx], xyz[idx])
            seen += 1
    return {"bbox_min": bbox_min, "bbox_max": bbox_max, "vertices_read_for_bbox": seen}


def inspect_ply(path: Path) -> dict[str, Any]:
    header, header_offset = parse_ply_header(path)
    vertex = next((e for e in header["elements"] if e["name"] == "vertex"), None)
    face = next((e for e in header["elements"] if e["name"] == "face"), None)
    vertex_props = vertex["properties"] if vertex else []
    prop_text = " ".join(vertex_props).lower()
    result: dict[str, Any] = {
        "format": "ply",
        "ply_encoding": header["format"],
        "vertex_count": int(vertex["count"]) if vertex else 0,
        "face_count": int(face["count"]) if face else 0,
        "vertex_properties": vertex_props,
        "has_normals": all(token in prop_text for token in (" nx", " ny", " nz")),
        "has_colors": any(token in prop_text for token in (" red", " green", " blue", " diffuse_red")),
    }
    if vertex and header["format"] == "ascii":
        result.update(inspect_ply_ascii_vertices(path, int(vertex["count"]), header_offset))
    elif vertex and header["format"] in {"binary_little_endian", "binary_big_endian"}:
        result.update(inspect_ply_binary_vertices(path, int(vertex["count"]), header, header_offset))
    return result


def inspect_obj(path: Path, max_text_lines: int) -> dict[str, Any]:
    counts = {"vertices": 0, "faces": 0, "normals": 0, "texcoords": 0}
    bbox_min = [float("inf"), float("inf"), float("inf")]
    bbox_max = [float("-inf"), float("-inf"), float("-inf")]
    lines_read = 0
    truncated = False
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            lines_read += 1
            if lines_read > max_text_lines:
                truncated = True
                break
            if line.startswith("v "):
                counts["vertices"] += 1
                parts = line.split()
                if len(parts) >= 4:
                    try:
                        xyz = [float(parts[1]), float(parts[2]), float(parts[3])]
                    except ValueError:
                        continue
                    for idx in range(3):
                        bbox_min[idx] = min(bbox_min[idx], xyz[idx])
                        bbox_max[idx] = max(bbox_max[idx], xyz[idx])
            elif line.startswith("f "):
                counts["faces"] += 1
            elif line.startswith("vn "):
                counts["normals"] += 1
            elif line.startswith("vt "):
                counts["texcoords"] += 1
    result: dict[str, Any] = {
        "format": "obj",
        "vertex_count": counts["vertices"],
        "face_count": counts["faces"],
        "normal_count": counts["normals"],
        "texcoord_count": counts["texcoords"],
        "has_normals": counts["normals"] > 0,
        "has_texcoords": counts["texcoords"] > 0,
        "lines_read": lines_read,
        "truncated": truncated,
    }
    if counts["vertices"] > 0:
        result["bbox_min"] = bbox_min
        result["bbox_max"] = bbox_max
    return result


def inspect_file(
    path: Path,
    max_array_values: int,
    max_text_lines: int,
    max_hdf5_nodes: int,
    include_samples: bool,
    fail_on_unsupported: bool,
) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if not path.exists():
        raise FileNotFoundError(path)
    base: dict[str, Any] = {
        "path": str(path.resolve()),
        "file_name": path.name,
        "extension": suffix,
        "size_bytes": path.stat().st_size,
    }
    if suffix in {".h5", ".hdf5"}:
        details = inspect_hdf5(path, max_array_values, include_samples, max_hdf5_nodes)
    elif suffix == ".npz":
        details = inspect_npz(path, max_array_values, include_samples)
    elif suffix == ".npy":
        details = inspect_npy(path, max_array_values, include_samples)
    elif suffix == ".ply":
        details = inspect_ply(path)
    elif suffix == ".obj":
        details = inspect_obj(path, max_text_lines)
    else:
        if fail_on_unsupported:
            raise ValueError(f"Unsupported file extension: {suffix}")
        details = {"format": "unsupported", "supported": False}
    base.update(details)
    return json_safe(base)


def summarize_result(result: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": result.get("path"),
        "format": result.get("format"),
        "size_bytes": result.get("size_bytes"),
    }
    if result.get("format") == "hdf5":
        summary.update(
            {
                "node_count_reported": result.get("node_count_reported"),
                "truncated_nodes": result.get("truncated_nodes"),
                "max_hdf5_nodes": result.get("max_hdf5_nodes"),
            }
        )
    elif result.get("format") == "npz":
        arrays = result.get("arrays", [])
        summary["array_count"] = len(arrays)
        summary["keys"] = [entry.get("key") for entry in arrays]
    elif result.get("format") == "npy":
        summary["shape"] = result.get("shape")
        summary["dtype"] = result.get("dtype")
    elif result.get("format") in {"ply", "obj"}:
        summary.update(
            {
                "vertex_count": result.get("vertex_count"),
                "face_count": result.get("face_count"),
                "has_normals": result.get("has_normals"),
                "has_colors": result.get("has_colors"),
                "bbox_min": result.get("bbox_min"),
                "bbox_max": result.get("bbox_max"),
            }
        )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Safely inspect one file schema.")
    parser.add_argument("--file", type=Path, required=True, help="File to inspect.")
    parser.add_argument("--output-json", type=Path, default=None, help="Optional JSON output.")
    parser.add_argument("--max-array-values", type=int, default=16)
    parser.add_argument("--max-text-lines", type=int, default=500000)
    parser.add_argument("--max-hdf5-nodes", type=int, default=500)
    parser.add_argument("--no-samples", action="store_true", help="Do not include tiny data samples.")
    parser.add_argument("--fail-on-unsupported", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = inspect_file(
        args.file,
        max_array_values=args.max_array_values,
        max_text_lines=args.max_text_lines,
        max_hdf5_nodes=args.max_hdf5_nodes,
        include_samples=not args.no_samples,
        fail_on_unsupported=args.fail_on_unsupported,
    )
    text = json.dumps(result, indent=2, sort_keys=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(text + "\n", encoding="utf-8")
        print(f"Wrote schema JSON: {args.output_json}")
        print(json.dumps(summarize_result(result), indent=2, sort_keys=True))
    else:
        print(text)


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(1)
