"""Reusable point-cloud IO and feature helpers.

The shared 9D feature convention is:

    [x, y, z, nx, ny, nz, local_density, surface_variation, eigenentropy]

These utilities are intentionally lightweight and deterministic. They do not
write to source datasets and avoid optional dependencies such as Open3D.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

FEATURE_9D_NAMES = [
    "x",
    "y",
    "z",
    "nx",
    "ny",
    "nz",
    "local_density",
    "surface_variation",
    "eigenentropy",
]


@dataclass(frozen=True)
class MeshData:
    """Mesh vertices/faces and optional vertex colors."""

    vertices: np.ndarray
    faces: np.ndarray | None = None
    colors: np.ndarray | None = None
    path: Path | None = None


@dataclass(frozen=True)
class PointCloudData:
    """Loaded or sampled point-cloud arrays."""

    points: np.ndarray
    normals: np.ndarray | None = None
    colors: np.ndarray | None = None
    labels: np.ndarray | None = None
    source_path: Path | None = None
    metadata: dict[str, Any] | None = None


def _as_float_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"points must have shape [N, 3], got {points.shape}")
    return points


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return str(value)
    return value


def load_mesh(path: Path, process: bool = False) -> MeshData:
    """Load a PLY/OBJ mesh through trimesh and return vertices/faces."""

    import trimesh

    loaded = trimesh.load(path, force="mesh", process=process)
    if loaded.is_empty or len(loaded.vertices) == 0:
        raise ValueError(f"empty mesh or no vertices: {path}")

    vertices = np.asarray(loaded.vertices, dtype=np.float32)
    faces = np.asarray(loaded.faces, dtype=np.int64) if len(loaded.faces) else None
    colors = None
    visual = getattr(loaded, "visual", None)
    vertex_colors = getattr(visual, "vertex_colors", None)
    if vertex_colors is not None and len(vertex_colors) == len(vertices):
        colors = np.asarray(vertex_colors[:, :3], dtype=np.uint8)
    return MeshData(vertices=vertices, faces=faces, colors=colors, path=path)


def load_point_cloud(path: str | Path, sample_points: int | None = None, seed: int = 0) -> PointCloudData:
    """Load points from PLY/OBJ/NPY/NPZ.

    Mesh files return vertices by default. If ``sample_points`` is provided and
    faces exist, the function samples mesh surfaces deterministically.
    """

    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in {".ply", ".obj"}:
        mesh = load_mesh(path)
        if sample_points is not None:
            return sample_mesh_surface(mesh, sample_points, seed=seed)
        return PointCloudData(
            points=_as_float_points(mesh.vertices),
            colors=mesh.colors,
            source_path=path,
            metadata={"source_format": suffix.lstrip("."), "loaded_as": "mesh_vertices"},
        )
    if suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
        return PointCloudData(
            points=_as_float_points(arr[:, :3] if arr.shape[1] >= 3 else arr),
            source_path=path,
            metadata={"source_format": "npy"},
        )
    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            key = _select_point_array_key(data)
            arr = np.asarray(data[key])
        return PointCloudData(
            points=_as_float_points(arr[:, :3] if arr.shape[1] >= 3 else arr),
            source_path=path,
            metadata={"source_format": "npz", "point_key": key},
        )
    raise ValueError(f"unsupported point-cloud extension: {suffix}")


def _select_point_array_key(data: np.lib.npyio.NpzFile) -> str:
    preferred = ("points", "xyz", "vertices", "data")
    for key in preferred:
        if key in data.files:
            arr = np.asarray(data[key])
            if arr.ndim == 2 and arr.shape[1] >= 3:
                return key
    for key in data.files:
        arr = np.asarray(data[key])
        if arr.ndim == 2 and arr.shape[1] >= 3:
            return key
    raise ValueError(f"no [N, >=3] point array found in NPZ keys: {data.files}")


def sample_mesh_surface(mesh: MeshData, num_points: int, seed: int = 0) -> PointCloudData:
    """Sample a mesh surface with deterministic trimesh sampling."""

    if mesh.faces is None or len(mesh.faces) == 0:
        sampled = sample_or_resample_points(mesh.vertices, num_points, seed=seed)
        return PointCloudData(
            points=sampled,
            colors=None,
            source_path=mesh.path,
            metadata={"sample_method": "vertex_resample", "num_points": num_points},
        )

    import trimesh

    tm = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, process=False)
    points, face_index = trimesh.sample.sample_surface(tm, num_points, seed=seed)
    normals = np.asarray(tm.face_normals[face_index], dtype=np.float32)
    return PointCloudData(
        points=np.asarray(points, dtype=np.float32),
        normals=normalize_vectors(normals),
        source_path=mesh.path,
        metadata={"sample_method": "surface", "num_points": num_points},
    )


def sample_or_resample_points(points: np.ndarray, num_points: int, seed: int = 0) -> np.ndarray:
    """Deterministically downsample or upsample points to ``num_points``."""

    points = _as_float_points(points)
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    if len(points) == 0:
        raise ValueError("cannot sample an empty point cloud")
    if len(points) == num_points:
        return points.copy()
    rng = np.random.default_rng(seed)
    replace = len(points) < num_points
    idx = rng.choice(len(points), size=num_points, replace=replace)
    return points[idx].astype(np.float32, copy=False)


def normalize_points(points: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
    """Center points at the centroid and scale to unit sphere."""

    points = _as_float_points(points)
    centroid = points.mean(axis=0, keepdims=True)
    centered = points - centroid
    scale = float(np.max(np.linalg.norm(centered, axis=1))) if len(centered) else 0.0
    if scale > 0:
        normalized = centered / scale
    else:
        normalized = centered
    return normalized.astype(np.float32), {
        "centroid": centroid.reshape(3).astype(float).tolist(),
        "scale": scale,
    }


def normalize_vectors(vectors: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.maximum(norms, eps)


def build_features(
    points: np.ndarray,
    feature_mode: str,
    normals: np.ndarray | None = None,
    k_neighbors: int = 16,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build 3D or 9D features from points."""

    points = _as_float_points(points)
    if feature_mode == "3d":
        return points.astype(np.float32), {"feature_mode": "3d", "feature_names": FEATURE_9D_NAMES[:3]}
    if feature_mode != "9d":
        raise ValueError("feature_mode must be '3d' or '9d'")

    local = compute_local_geometry_features(points, k_neighbors=k_neighbors)
    if normals is None:
        normals_ = local["normals"]
        normal_source = "local_pca"
    else:
        normals_ = normalize_vectors(normals)
        normal_source = "provided"
    features = np.column_stack(
        [
            points,
            normals_,
            local["local_density"],
            local["surface_variation"],
            local["eigenentropy"],
        ]
    ).astype(np.float32)
    return features, {
        "feature_mode": "9d",
        "feature_names": FEATURE_9D_NAMES,
        "normal_source": normal_source,
        "k_neighbors": int(k_neighbors),
    }


def compute_local_geometry_features(points: np.ndarray, k_neighbors: int = 16) -> dict[str, np.ndarray]:
    """Compute local PCA descriptors for each point.

    ``local_density`` is defined as ``1 / mean_knn_distance``. ``surface_variation``
    is ``lambda_min / sum(lambda)``. ``eigenentropy`` is ``-sum(p * log(p))`` for
    normalized covariance eigenvalues.
    """

    points = _as_float_points(points)
    n = len(points)
    if n == 0:
        raise ValueError("cannot compute local features for empty point cloud")
    k = min(max(1, int(k_neighbors)), max(1, n - 1))

    diff = points[:, None, :] - points[None, :, :]
    dist2 = np.einsum("ijk,ijk->ij", diff, diff)
    order = np.argsort(dist2, axis=1)
    neighbor_idx = order[:, 1 : k + 1] if n > 1 else order[:, :1]

    normals = np.zeros((n, 3), dtype=np.float32)
    local_density = np.zeros(n, dtype=np.float32)
    surface_variation = np.zeros(n, dtype=np.float32)
    eigenentropy = np.zeros(n, dtype=np.float32)
    eps = 1e-12

    for i in range(n):
        neigh = points[neighbor_idx[i]]
        distances = np.sqrt(np.maximum(dist2[i, neighbor_idx[i]], 0.0))
        local_density[i] = float(1.0 / (distances.mean() + eps))
        centered = neigh - neigh.mean(axis=0, keepdims=True)
        cov = (centered.T @ centered) / max(len(neigh), 1)
        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.maximum(eigvals, 0.0)
        normals[i] = eigvecs[:, int(np.argmin(eigvals))]
        denom = float(eigvals.sum())
        if denom > eps:
            probs = eigvals / denom
            surface_variation[i] = float(eigvals.min() / denom)
            eigenentropy[i] = float(-np.sum(probs * np.log(probs + eps)))

    return {
        "normals": normalize_vectors(normals),
        "local_density": local_density,
        "surface_variation": surface_variation,
        "eigenentropy": eigenentropy,
    }


def pointcloud_summary(points: np.ndarray, features: np.ndarray | None = None) -> dict[str, Any]:
    points = _as_float_points(points)
    summary: dict[str, Any] = {
        "num_points": int(len(points)),
        "bbox_min": points.min(axis=0).astype(float).tolist() if len(points) else None,
        "bbox_max": points.max(axis=0).astype(float).tolist() if len(points) else None,
        "centroid": points.mean(axis=0).astype(float).tolist() if len(points) else None,
    }
    if features is not None:
        summary["feature_shape"] = list(np.asarray(features).shape)
        summary["feature_dtype"] = str(np.asarray(features).dtype)
    return summary


def save_json_summary(path: str | Path, summary: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(summary), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_ply(
    path: str | Path,
    points: np.ndarray,
    labels: np.ndarray | None = None,
    colors: np.ndarray | None = None,
) -> None:
    """Write an ASCII vertex-only PLY with optional labels/colors."""

    path = Path(path)
    points = _as_float_points(points)
    if labels is not None:
        labels = np.asarray(labels).reshape(-1)
        if len(labels) != len(points):
            raise ValueError(f"labels length {len(labels)} != points length {len(points)}")
    if colors is not None:
        colors = np.asarray(colors)
        if colors.shape != (len(points), 3):
            raise ValueError(f"colors must have shape [N, 3], got {colors.shape}")
        colors = np.clip(colors, 0, 255).astype(np.uint8)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        if colors is not None:
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        if labels is not None:
            f.write("property int label\n")
        f.write("end_header\n")
        for i, xyz in enumerate(points):
            fields: list[str] = [f"{float(xyz[0]):.8f}", f"{float(xyz[1]):.8f}", f"{float(xyz[2]):.8f}"]
            if colors is not None:
                fields.extend(str(int(v)) for v in colors[i])
            if labels is not None:
                fields.append(str(int(labels[i])))
            f.write(" ".join(fields) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Point-cloud utility smoke runner.")
    parser.add_argument("--file", type=Path, required=True)
    parser.add_argument("--num-points", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--feature-mode", choices=["3d", "9d"], default="3d")
    parser.add_argument("--k-neighbors", type=int, default=16)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--output-ply", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cloud = load_point_cloud(args.file, sample_points=args.num_points, seed=args.seed)
    normalized, norm_meta = normalize_points(cloud.points)
    features, feature_meta = build_features(
        normalized,
        feature_mode=args.feature_mode,
        normals=cloud.normals,
        k_neighbors=args.k_neighbors,
    )
    summary = {
        "source_path": str(args.file.resolve()),
        "load_metadata": cloud.metadata or {},
        "normalization": norm_meta,
        "features": feature_meta,
        "pointcloud": pointcloud_summary(normalized, features),
    }
    if args.output_json:
        save_json_summary(args.output_json, summary)
        print(f"Wrote summary JSON: {args.output_json}")
    if args.output_ply:
        write_ply(args.output_ply, normalized)
        print(f"Wrote PLY: {args.output_ply}")
    print(json.dumps(_json_safe(summary), indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except BrokenPipeError:
        sys.exit(1)
