"""Normal-based extrapolation utilities for zero-shot shape completion.

Estimates surface normals at fracture boundary points, then extrapolates
dummy centers outward along estimated normals. Falls back to Gaussian
noise when Open3D is unavailable or normal estimation fails.
"""

from __future__ import annotations

import numpy as np

# Try importing open3d; set a flag if unavailable
try:
    import open3d as o3d
    _HAS_OPEN3D = True
except ImportError:
    _HAS_OPEN3D = False


def estimate_normals_o3d(
    points: np.ndarray,
    knn: int = 30,
    radius: float = 0.0,
) -> np.ndarray:
    """Estimate normals using Open3D's KNN-based normal estimation.

    Args:
        points: [N, 3] float32 array.
        knn: number of nearest neighbours for normal estimation.
        radius: search radius (0 = KNN only, >0 = hybrid search).

    Returns:
        normals: [N, 3] float32 array of estimated normals.

    Raises:
        ImportError: if Open3D is not installed.
    """
    if not _HAS_OPEN3D:
        raise ImportError("open3d is required for normal estimation but is not installed.")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    if radius > 0:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=radius, max_nn=knn)
        )
    else:
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn)
        )

    normals = np.asarray(pcd.normals, dtype=np.float32)
    return normals


def orient_normals_outward(
    points: np.ndarray,
    normals: np.ndarray,
    reference_centroid: np.ndarray | None = None,
) -> np.ndarray:
    """Orient normals to point away from a reference centroid.

    If ``reference_centroid`` is None, uses the mean of ``points``.
    Flips normals that point toward the centroid so they point outward.

    Args:
        points: [N, 3] point positions.
        normals: [N, 3] estimated normals.
        reference_centroid: [3] centroid to orient away from.

    Returns:
        oriented_normals: [N, 3] consistently outward-facing normals.
    """
    if reference_centroid is None:
        reference_centroid = points.mean(axis=0)
    reference_centroid = np.asarray(reference_centroid, dtype=np.float32).reshape(3)

    # Direction from centroid to each point
    outward_dir = points - reference_centroid[None, :]
    # Flip normals that point inward (dot product < 0)
    dots = np.sum(normals * outward_dir, axis=1)
    flip_mask = dots < 0
    oriented = normals.copy()
    oriented[flip_mask] *= -1.0
    return oriented


def extrapolate_along_normals(
    fracture_points: np.ndarray,
    normals: np.ndarray,
    num_points: int,
    distance: float = 0.05,
    jitter: float = 0.01,
    seed: int = 42,
) -> np.ndarray:
    """Generate dummy centers by extrapolating along surface normals.

    Samples base points from ``fracture_points``, shifts them outward
    along estimated normals by ``distance``, and adds Gaussian jitter.

    Args:
        fracture_points: [K, 3] points identified as fracture surface.
        normals: [K, 3] outward-facing normals at fracture points.
        num_points: number of dummy centers to generate.
        distance: base extrapolation distance along normal.
        jitter: standard deviation of Gaussian noise added for variation.
        seed: random seed for reproducibility.

    Returns:
        dummy_centers: [num_points, 3] extrapolated coordinates.
    """
    rng = np.random.default_rng(seed)
    K = len(fracture_points)
    if K == 0:
        raise ValueError("No fracture points provided for extrapolation.")

    # Sample base point indices (with replacement if needed)
    idx = rng.choice(K, size=num_points, replace=True)
    base_pts = fracture_points[idx]  # [num_points, 3]
    base_normals = normals[idx]  # [num_points, 3]

    # Extrapolate along normals with random distance variation
    dist_variation = rng.uniform(0.5, 1.5, size=(num_points, 1)).astype(np.float32)
    shifted = base_pts + base_normals * distance * dist_variation

    # Add small Gaussian jitter for variation
    noise = rng.normal(0, jitter, size=(num_points, 3)).astype(np.float32)
    dummy_centers = shifted + noise

    return dummy_centers.astype(np.float32)


def gaussian_extrapolation(
    fracture_points: np.ndarray,
    num_points: int,
    scale: float = 0.1,
    seed: int = 42,
) -> np.ndarray:
    """Fallback: generate dummy centers with Gaussian noise around fracture points.

    Args:
        fracture_points: [K, 3] fracture surface points.
        num_points: number of dummy centers to generate.
        scale: noise standard deviation.
        seed: random seed.

    Returns:
        dummy_centers: [num_points, 3].
    """
    rng = np.random.default_rng(seed)
    K = len(fracture_points)
    if K == 0:
        raise ValueError("No fracture points provided for extrapolation.")

    idx = rng.choice(K, size=num_points, replace=True)
    base_pts = fracture_points[idx]
    noise = rng.normal(0, scale, size=(num_points, 3)).astype(np.float32)
    return (base_pts + noise).astype(np.float32)


def uniform_random_extrapolation(
    partial_points: np.ndarray,
    num_points: int,
    expansion: float = 0.3,
    seed: int = 42,
) -> np.ndarray:
    """Generate dummy centers uniformly outside the partial cloud's bounding box.

    Used for the ``no_mask`` ablation mode where no segmentation knowledge is used.

    Args:
        partial_points: [N, 3] the partial point cloud.
        num_points: number of dummy centers to generate.
        expansion: fraction of bbox extent to expand outward.
        seed: random seed.

    Returns:
        dummy_centers: [num_points, 3].
    """
    rng = np.random.default_rng(seed)
    bbox_min = partial_points.min(axis=0)
    bbox_max = partial_points.max(axis=0)
    extent = bbox_max - bbox_min
    # Expand bounding box
    expanded_min = bbox_min - extent * expansion
    expanded_max = bbox_max + extent * expansion
    # Uniform random in expanded box
    dummy = rng.uniform(
        expanded_min, expanded_max, size=(num_points, 3)
    ).astype(np.float32)
    return dummy


def generate_extrapolated_centers(
    partial_points: np.ndarray,
    fracture_mask: np.ndarray | None,
    num_dummy: int,
    method: str = "normal",
    distance_scale: float = 0.1,
    jitter_scale: float = 0.02,
    normal_knn: int = 30,
    fracture_threshold: float = 0.3,
    seed: int = 42,
) -> np.ndarray:
    """High-level entry point for generating dummy extrapolation centers.

    Args:
        partial_points: [N, 3] the partial input point cloud.
        fracture_mask: [N] fracture probability or binary mask. None for no_mask mode.
        num_dummy: number of dummy centers to generate.
        method: "normal" (normal-based) or "gaussian" (random noise) or "uniform" (no_mask).
        distance_scale: extrapolation distance as fraction of bbox extent.
        jitter_scale: jitter std as fraction of bbox extent.
        normal_knn: KNN for normal estimation.
        fracture_threshold: threshold for identifying fracture points from probabilities.
        seed: random seed.

    Returns:
        dummy_centers: [num_dummy, 3].
    """
    if num_dummy <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    bbox_extent = (partial_points.max(axis=0) - partial_points.min(axis=0)).max()
    abs_distance = distance_scale * bbox_extent
    abs_jitter = jitter_scale * bbox_extent

    # no_mask mode: uniform random, no fracture knowledge
    if method == "uniform" or fracture_mask is None:
        return uniform_random_extrapolation(
            partial_points, num_dummy, expansion=distance_scale * 3, seed=seed
        )

    # Identify fracture boundary points
    if fracture_mask.dtype == bool:
        frac_indices = np.where(fracture_mask)[0]
    else:
        frac_indices = np.where(fracture_mask > fracture_threshold)[0]

    if len(frac_indices) == 0:
        # No fracture points found — fall back to uniform
        return uniform_random_extrapolation(
            partial_points, num_dummy, expansion=distance_scale * 3, seed=seed
        )

    fracture_pts = partial_points[frac_indices]

    if method == "normal":
        try:
            # Estimate normals on the full partial cloud
            all_normals = estimate_normals_o3d(partial_points, knn=normal_knn)
            frac_normals = all_normals[frac_indices]

            # Orient normals outward (away from intact centroid)
            intact_indices = np.where(
                fracture_mask <= fracture_threshold
                if not fracture_mask.dtype == bool
                else ~fracture_mask
            )[0]
            if len(intact_indices) > 0:
                intact_centroid = partial_points[intact_indices].mean(axis=0)
            else:
                intact_centroid = partial_points.mean(axis=0)

            frac_normals = orient_normals_outward(
                fracture_pts, frac_normals, reference_centroid=intact_centroid
            )

            return extrapolate_along_normals(
                fracture_pts, frac_normals, num_dummy,
                distance=abs_distance, jitter=abs_jitter, seed=seed
            )
        except (ImportError, RuntimeError, Exception) as exc:
            # Fall back to Gaussian if Open3D or normal estimation fails
            import warnings
            warnings.warn(
                f"Normal-based extrapolation failed ({exc}), "
                f"falling back to Gaussian noise."
            )
            return gaussian_extrapolation(
                fracture_pts, num_dummy, scale=abs_distance, seed=seed
            )

    elif method == "gaussian":
        return gaussian_extrapolation(
            fracture_pts, num_dummy, scale=abs_distance, seed=seed
        )

    else:
        raise ValueError(f"Unknown extrapolation method: {method}")
