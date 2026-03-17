"""
Purpose:
    Prepare HDF5 classification data from Fantastic Breaks meshes.
    Each object folder contains:
        - model_c.ply   → label 0 (complete)
        - model_b_0.ply → label 1 (broken)
    
    We sample N points from each mesh, normalize, and save as HDF5
    files compatible with PointNet's provider.py (data: [B,N,3], label: [B,1]).

Usage:
    source /storage/student6/anaconda3/bin/activate pointnet
    python scripts/prepare_classification_data.py \
        --data_root data/Fantastic_Breaks_v1 \
        --output_dir data/classification \
        --num_points 2048 \
        --test_ratio 0.2 \
        --seed 42
"""
from __future__ import annotations

import argparse
import os
import glob
from pathlib import Path

import numpy as np
import h5py
import trimesh


def sample_points_from_mesh(mesh_path: str, num_points: int, seed: int) -> np.ndarray:
    """
    Load a PLY mesh and uniformly sample points from its surface.
    Returns [N, 3] float32 array.
    """
    mesh = trimesh.load(mesh_path, force='mesh')
    points, _ = trimesh.sample.sample_surface(mesh, num_points, seed=seed)
    return points.astype(np.float32)


def normalize_point_cloud(points: np.ndarray) -> np.ndarray:
    """Center at origin and scale to unit sphere."""
    centroid = points.mean(axis=0, keepdims=True)
    points = points - centroid
    scale = np.max(np.linalg.norm(points, axis=1))
    if scale > 0:
        points = points / scale
    return points


def discover_samples(data_root: str) -> list[dict]:
    """
    Walk Fantastic_Breaks_v1 directory to find complete and broken meshes.
    Returns list of dicts with keys: 'path', 'label', 'object_id'.
    """
    samples = []
    for category_dir in sorted(glob.glob(os.path.join(data_root, '*'))):
        if not os.path.isdir(category_dir):
            continue
        for obj_dir in sorted(glob.glob(os.path.join(category_dir, '*'))):
            if not os.path.isdir(obj_dir):
                continue
            obj_id = os.path.basename(obj_dir)

            # Complete mesh
            complete_path = os.path.join(obj_dir, 'model_c.ply')
            if os.path.exists(complete_path):
                samples.append({
                    'path': complete_path,
                    'label': 0,  # complete
                    'object_id': f"{obj_id}_c",
                })

            # Broken mesh
            broken_path = os.path.join(obj_dir, 'model_b_0.ply')
            if os.path.exists(broken_path):
                samples.append({
                    'path': broken_path,
                    'label': 1,  # broken
                    'object_id': f"{obj_id}_b",
                })

    return samples


def save_h5(filepath: str, data: np.ndarray, labels: np.ndarray):
    """Save data and labels in HDF5 format compatible with PointNet provider."""
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('data', data=data, dtype='float32', compression='gzip')
        f.create_dataset('label', data=labels, dtype='int64', compression='gzip')
    print(f"  Saved {filepath}: data={data.shape}, label={labels.shape}")


def main():
    parser = argparse.ArgumentParser(description='Prepare classification HDF5 data')
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to Fantastic_Breaks_v1 directory')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for HDF5 files')
    parser.add_argument('--num_points', type=int, default=2048,
                        help='Number of points to sample per mesh')
    parser.add_argument('--test_ratio', type=float, default=0.2,
                        help='Fraction of data for test set')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Discover all samples
    print("Discovering samples...")
    samples = discover_samples(args.data_root)
    print(f"Found {len(samples)} samples "
          f"({sum(1 for s in samples if s['label']==0)} complete, "
          f"{sum(1 for s in samples if s['label']==1)} broken)")

    # Sample point clouds
    print(f"\nSampling {args.num_points} points from each mesh...")
    all_points = []
    all_labels = []
    all_ids = []
    for i, sample in enumerate(samples):
        if (i + 1) % 1 == 0 or i == 0:
            print(f"  [{i+1}/{len(samples)}] {sample['object_id']}...")
        try:
            pts = sample_points_from_mesh(sample['path'], args.num_points, args.seed)
            pts = normalize_point_cloud(pts)
            all_points.append(pts)
            all_labels.append(sample['label'])
            all_ids.append(sample['object_id'])
        except Exception as e:
            print(f"  WARNING: Failed to process {sample['path']}: {e}")

    all_points = np.array(all_points, dtype=np.float32)   # [B, N, 3]
    all_labels = np.array(all_labels, dtype=np.int64).reshape(-1, 1)  # [B, 1]
    print(f"\nTotal: {all_points.shape[0]} samples, shape={all_points.shape}")

    # Stratified train/test split
    rng = np.random.default_rng(args.seed)
    indices = np.arange(len(all_points))
    rng.shuffle(indices)

    n_test = int(len(indices) * args.test_ratio)
    # Ensure balanced split
    label_flat = all_labels.flatten()
    complete_idx = indices[label_flat[indices] == 0]
    broken_idx = indices[label_flat[indices] == 1]

    n_test_c = int(len(complete_idx) * args.test_ratio)
    n_test_b = int(len(broken_idx) * args.test_ratio)

    test_idx = np.concatenate([complete_idx[:n_test_c], broken_idx[:n_test_b]])
    train_idx = np.concatenate([complete_idx[n_test_c:], broken_idx[n_test_b:]])

    rng.shuffle(train_idx)
    rng.shuffle(test_idx)

    print(f"\nTrain: {len(train_idx)} samples, Test: {len(test_idx)} samples")
    print(f"  Train — complete: {(label_flat[train_idx]==0).sum()}, broken: {(label_flat[train_idx]==1).sum()}")
    print(f"  Test  — complete: {(label_flat[test_idx]==0).sum()}, broken: {(label_flat[test_idx]==1).sum()}")

    # Save HDF5 files
    print("\nSaving HDF5 files...")
    save_h5(os.path.join(args.output_dir, 'train_data.h5'),
            all_points[train_idx], all_labels[train_idx])
    save_h5(os.path.join(args.output_dir, 'test_data.h5'),
            all_points[test_idx], all_labels[test_idx])

    # Save file lists (PointNet expects these)
    with open(os.path.join(args.output_dir, 'train_files.txt'), 'w') as f:
        f.write(os.path.join(args.output_dir, 'train_data.h5') + '\n')
    with open(os.path.join(args.output_dir, 'test_files.txt'), 'w') as f:
        f.write(os.path.join(args.output_dir, 'test_data.h5') + '\n')

    # Save object ID mapping for reference
    with open(os.path.join(args.output_dir, 'object_ids.txt'), 'w') as f:
        for oid in all_ids:
            f.write(oid + '\n')

    print("\nDone! Files saved to:", args.output_dir)
    print("\nTo train PointNet classification:")
    print(f"  python src/pointnet-master/train_cls.py \\")
    print(f"    --data_dir {args.output_dir} \\")
    print(f"    --num_point {args.num_points} --num_classes 2")


if __name__ == '__main__':
    main()
