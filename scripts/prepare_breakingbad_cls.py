"""
Prepare Breaking Bad Dataset for PointNeXt Binary Classification
================================================================

Converts .obj meshes from the Breaking Bad dataset into HDF5 point clouds
compatible with the existing FantasticBreaksCls dataset adapter.

Dataset structure (Breaking Bad):
    <object_id>/mode_0/piece_0.obj        → complete object (label 0)
    <object_id>/fractured_0/piece_*.obj   → broken fragments (label 1)

Annotation strategy:
    - Complete (class 0): mode_0/piece_0.obj  (1 per object)
    - Broken (class 1):   Each piece_*.obj in fractured_0/  (multiple per object)
    - To handle class imbalance, we use one of these strategies:
        --balance=none       → use all pieces (imbalanced)
        --balance=undersample → randomly sample N broken pieces to match complete count
        --balance=one_per_obj → only 1 random broken piece per object (matched count)

Output format (HDF5, same as FantasticBreaks):
    - data:  float32  [B, N, 3]   — normalized point cloud (XYZ)
    - label: int64    [B, 1]      — class index: 0 = complete, 1 = broken

Usage:
    source /storage/student6/anaconda3/bin/activate pointnet
    python scripts/prepare_breakingbad_cls.py \\
        --data_root data/BreakingBad \\
        --split_dir data/BreakingBad/data_split \\
        --output_dir data/breakingbad_classification \\
        --subsets artifact everyday/Vase \\
        --num_points 8192 \\
        --balance undersample \\
        --seed 42
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import json
import glob
from pathlib import Path
from collections import defaultdict

import numpy as np
import h5py
import trimesh


# ──────────────────────────────────────────────────────────────────────
# Point cloud processing
# ──────────────────────────────────────────────────────────────────────

def sample_points_from_obj(mesh_path: str, num_points: int, seed: int) -> np.ndarray:
    """
    Load a .obj mesh and uniformly sample points from its surface.
    Returns [N, 3] float32 array.
    """
    mesh = trimesh.load(mesh_path, force='mesh', process=False)

    if mesh.vertices.shape[0] == 0 or mesh.faces.shape[0] == 0:
        raise ValueError(f"Empty mesh: {mesh_path}")

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


# ──────────────────────────────────────────────────────────────────────
# Split file handling
# ──────────────────────────────────────────────────────────────────────

def load_split_entries(split_dir: str, subset: str, split: str) -> list[str]:
    """
    Load object entries for a given subset and split.

    Args:
        split_dir: Directory containing the .txt split files.
        subset:    'artifact' or 'everyday/Vase'.
        split:     'train' or 'val'.

    Returns:
        list of object paths relative to data_root, e.g. 'artifact/73400_sf'
    """
    prefix = subset.split('/')[0]  # 'artifact' or 'everyday'
    filename = f"{prefix}.{split}.txt"
    filepath = os.path.join(split_dir, filename)

    if not os.path.exists(filepath):
        print(f"  WARNING: Split file not found: {filepath}")
        return []

    with open(filepath, 'r') as f:
        entries = [line.strip() for line in f if line.strip()]

    # Filter for specific category if needed (e.g. everyday/Vase)
    if '/' in subset:
        entries = [e for e in entries if e.startswith(subset + '/')]

    return entries


# ──────────────────────────────────────────────────────────────────────
# Sample discovery
# ──────────────────────────────────────────────────────────────────────

def discover_samples_for_object(data_root: str, obj_entry: str) -> dict:
    """
    For a single object entry, discover the complete and broken mesh paths.

    Returns:
        {
            'object_id': str,
            'complete_path': str or None,    # mode_0/piece_0.obj
            'broken_paths': list[str],       # fractured_0/piece_*.obj
        }
    """
    obj_dir = os.path.join(data_root, obj_entry)

    result = {
        'object_id': obj_entry,
        'complete_path': None,
        'broken_paths': [],
    }

    # Complete object: mode_0/piece_0.obj
    complete_path = os.path.join(obj_dir, 'mode_0', 'piece_0.obj')
    if os.path.exists(complete_path):
        result['complete_path'] = complete_path

    # Broken fragments: fractured_0/piece_*.obj
    fractured_dir = os.path.join(obj_dir, 'fractured_0')
    if os.path.isdir(fractured_dir):
        broken_files = sorted([
            os.path.join(fractured_dir, f)
            for f in os.listdir(fractured_dir)
            if f.endswith('.obj')
        ])
        result['broken_paths'] = broken_files

    return result


def discover_all_samples(data_root: str, split_dir: str,
                         subsets: list[str], split: str) -> list[dict]:
    """
    Discover all classification samples for a given split across subsets.
    
    Returns list of {'path': str, 'label': int, 'object_id': str, 'source': str}.
    """
    samples = []

    for subset in subsets:
        entries = load_split_entries(split_dir, subset, split)
        print(f"  {subset} ({split}): {len(entries)} objects")

        for i, entry in enumerate(entries):
            obj_info = discover_samples_for_object(data_root, entry)

            if (i + 1) % 20 == 0 or i == 0 or i == len(entries) - 1:
                n_broken = len(obj_info['broken_paths'])
                has_complete = obj_info['complete_path'] is not None
                print(f"    [{i+1}/{len(entries)}] {entry}: "
                      f"complete={'✓' if has_complete else '✗'}, "
                      f"broken_pieces={n_broken}")

            # Add complete sample
            if obj_info['complete_path'] is not None:
                samples.append({
                    'path': obj_info['complete_path'],
                    'label': 0,  # complete
                    'object_id': f"{entry}/mode_0",
                    'source': subset,
                })

            # Add broken samples
            for bp in obj_info['broken_paths']:
                piece_name = os.path.basename(bp)
                samples.append({
                    'path': bp,
                    'label': 1,  # broken
                    'object_id': f"{entry}/fractured_0/{piece_name}",
                    'source': subset,
                })

    return samples


# ──────────────────────────────────────────────────────────────────────
# Class balancing
# ──────────────────────────────────────────────────────────────────────

def balance_samples(samples: list[dict], strategy: str,
                    seed: int) -> list[dict]:
    """
    Balance classes according to the chosen strategy.

    Strategies:
        'none':        Keep all samples as-is (may be imbalanced).
        'undersample': Randomly undersample the majority class.
        'one_per_obj': Keep only 1 random broken piece per object
                       (each object contributes 1 complete + 1 broken).
    """
    if strategy == 'none':
        return samples

    rng = np.random.default_rng(seed)

    complete = [s for s in samples if s['label'] == 0]
    broken = [s for s in samples if s['label'] == 1]

    if strategy == 'one_per_obj':
        # Group broken pieces by parent object
        broken_by_obj = defaultdict(list)
        for s in broken:
            # Extract parent: e.g. "artifact/73400_sf/fractured_0/piece_2.obj"
            # → parent = "artifact/73400_sf"
            parts = s['object_id'].split('/')
            parent = '/'.join(parts[:-2])  # drop "fractured_0/piece_X.obj"
            broken_by_obj[parent].append(s)

        # Pick 1 random piece per object
        balanced_broken = []
        for parent, pieces in broken_by_obj.items():
            idx = rng.integers(0, len(pieces))
            balanced_broken.append(pieces[idx])

        balanced = complete + balanced_broken

    elif strategy == 'undersample':
        n_complete = len(complete)
        n_broken = len(broken)

        if n_broken > n_complete:
            # Undersample broken
            indices = rng.choice(n_broken, n_complete, replace=False)
            balanced_broken = [broken[i] for i in indices]
            balanced = complete + balanced_broken
        elif n_complete > n_broken:
            # Undersample complete
            indices = rng.choice(n_complete, n_broken, replace=False)
            balanced_complete = [complete[i] for i in indices]
            balanced = balanced_complete + broken
        else:
            balanced = samples
    else:
        raise ValueError(f"Unknown balance strategy: {strategy}")

    rng.shuffle(balanced)
    return balanced


# ──────────────────────────────────────────────────────────────────────
# HDF5 saving
# ──────────────────────────────────────────────────────────────────────

def save_h5(filepath: str, data: np.ndarray, labels: np.ndarray):
    """Save data and labels in HDF5 format compatible with FantasticBreaksCls."""
    with h5py.File(filepath, 'w') as f:
        f.create_dataset('data', data=data, dtype='float32', compression='gzip')
        f.create_dataset('label', data=labels, dtype='int64', compression='gzip')
    print(f"  Saved {filepath}: data={data.shape}, label={labels.shape}")


# ──────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────

def process_samples(samples: list[dict], num_points: int,
                    seed: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Process a list of samples: load .obj, sample points, normalize.

    Returns:
        points: float32 [N, num_points, 3]
        labels: int64   [N, 1]
        ids:    list of object IDs (for reference)
    """
    all_points = []
    all_labels = []
    all_ids = []
    failed = []

    for i, sample in enumerate(samples):
        if (i + 1) % 10 == 0 or i == 0 or i == len(samples) - 1:
            label_str = "complete" if sample['label'] == 0 else "broken"
            print(f"    [{i+1}/{len(samples)}] {label_str}: "
                  f"{os.path.basename(sample['path'])} ...", end='', flush=True)

        try:
            pts = sample_points_from_obj(sample['path'], num_points, seed)
            pts = normalize_point_cloud(pts)
            all_points.append(pts)
            all_labels.append(sample['label'])
            all_ids.append(sample['object_id'])

            if (i + 1) % 10 == 0 or i == 0 or i == len(samples) - 1:
                print(" ✓")
        except Exception as e:
            failed.append({'id': sample['object_id'], 'error': str(e)})
            if (i + 1) % 10 == 0 or i == 0 or i == len(samples) - 1:
                print(f" ✗ ({e})")

    if failed:
        print(f"\n  WARNING: {len(failed)} samples failed to process:")
        for f in failed[:5]:
            print(f"    {f['id']}: {f['error']}")
        if len(failed) > 5:
            print(f"    ... and {len(failed) - 5} more")

    points_arr = np.array(all_points, dtype=np.float32)
    labels_arr = np.array(all_labels, dtype=np.int64).reshape(-1, 1)

    return points_arr, labels_arr, all_ids


def main():
    parser = argparse.ArgumentParser(
        description='Prepare Breaking Bad HDF5 classification data for PointNeXt',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--data_root', type=str, default='data/BreakingBad',
                        help='Root of Breaking Bad dataset')
    parser.add_argument('--split_dir', type=str, default='data/BreakingBad/data_split',
                        help='Directory with train/val split .txt files')
    parser.add_argument('--output_dir', type=str, default='data/breakingbad_classification',
                        help='Output directory for HDF5 files')
    parser.add_argument('--subsets', nargs='+', default=['artifact', 'everyday/Vase'],
                        help='Subsets to include')
    parser.add_argument('--num_points', type=int, default=8192,
                        help='Number of points to sample per mesh')
    parser.add_argument('--balance', type=str, default='undersample',
                        choices=['none', 'undersample', 'one_per_obj'],
                        help='Class balance strategy')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    args = parser.parse_args()

    # Resolve relative paths
    if not os.path.isabs(args.data_root):
        args.data_root = os.path.join(os.getcwd(), args.data_root)
    if not os.path.isabs(args.split_dir):
        args.split_dir = os.path.join(os.getcwd(), args.split_dir)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(os.getcwd(), args.output_dir)

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 72)
    print("  Breaking Bad → PointNeXt Classification Preprocessor")
    print("=" * 72)
    print(f"  Data root:    {args.data_root}")
    print(f"  Split dir:    {args.split_dir}")
    print(f"  Output dir:   {args.output_dir}")
    print(f"  Subsets:      {args.subsets}")
    print(f"  Num points:   {args.num_points}")
    print(f"  Balance:      {args.balance}")
    print(f"  Seed:         {args.seed}")

    metadata = {
        'created': time.strftime('%Y-%m-%d %H:%M:%S'),
        'args': vars(args),
        'splits': {},
    }

    for split in ['train', 'val']:
        print(f"\n{'─'*72}")
        print(f"  Discovering {split} samples...")
        print(f"{'─'*72}")

        # 1. Discover samples from split files
        samples = discover_all_samples(
            args.data_root, args.split_dir, args.subsets, split
        )

        n_complete = sum(1 for s in samples if s['label'] == 0)
        n_broken = sum(1 for s in samples if s['label'] == 1)
        print(f"\n  Raw counts — complete: {n_complete}, broken: {n_broken}, "
              f"total: {len(samples)}")
        print(f"  Imbalance ratio: {n_broken / max(n_complete, 1):.2f}:1")

        # 2. Balance classes
        if args.balance != 'none':
            print(f"\n  Applying balance strategy: {args.balance}")
            samples = balance_samples(samples, args.balance, args.seed)
            n_complete = sum(1 for s in samples if s['label'] == 0)
            n_broken = sum(1 for s in samples if s['label'] == 1)
            print(f"  After balancing — complete: {n_complete}, broken: {n_broken}, "
                  f"total: {len(samples)}")

        # 3. Process meshes → point clouds
        print(f"\n  Sampling {args.num_points} points from each .obj mesh...")
        t0 = time.time()
        points, labels, ids = process_samples(
            samples, args.num_points, args.seed
        )
        elapsed = time.time() - t0
        print(f"\n  Processed {len(ids)} samples in {elapsed:.1f}s")
        print(f"  Points shape: {points.shape}, Labels shape: {labels.shape}")

        # 4. Save HDF5
        # Use same naming as FantasticBreaks: train_data.h5 / test_data.h5
        # FantasticBreaks adapter expects 'train' and 'test' splits
        h5_split = 'train' if split == 'train' else 'test'
        h5_path = os.path.join(args.output_dir, f'{h5_split}_data.h5')
        save_h5(h5_path, points, labels)

        # 5. Save file lists
        filelist_path = os.path.join(args.output_dir, f'{h5_split}_files.txt')
        with open(filelist_path, 'w') as f:
            f.write(h5_path + '\n')

        # Save object IDs for reference
        ids_path = os.path.join(args.output_dir, f'{h5_split}_object_ids.txt')
        with open(ids_path, 'w') as f:
            for oid in ids:
                f.write(oid + '\n')

        metadata['splits'][split] = {
            'num_complete': int(n_complete),
            'num_broken': int(n_broken),
            'total': int(n_complete + n_broken),
            'h5_file': h5_path,
            'points_shape': list(points.shape),
        }

    # Save metadata
    meta_path = os.path.join(args.output_dir, 'metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  Metadata saved to: {meta_path}")

    print(f"\n{'='*72}")
    print(f"  DONE! Files saved to: {args.output_dir}")
    print(f"{'='*72}")
    print(f"\n  To train PointNeXt classification:")
    print(f"    CUDA_VISIBLE_DEVICES=3 python examples/classification/main.py \\")
    print(f"      --cfg cfgs/breakingbad/pointnext-b.yaml")


if __name__ == '__main__':
    main()
