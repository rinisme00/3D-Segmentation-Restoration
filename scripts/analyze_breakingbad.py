"""
Breaking Bad Dataset — Analysis & Metadata Generation
======================================================

Optimized for slow network filesystems: reads the official split files for
object enumeration and only samples a few objects for deep inspection,
rather than crawling every nested directory.

Outputs:
    1. Console report: counts, class balance, mesh statistics
    2. JSON metadata:  data/BreakingBad/bb_classification_metadata.json

Usage:
    python3 scripts/analyze_breakingbad.py \
        --data_root data/BreakingBad \
        --split_dir data/BreakingBad/data_split \
        --subsets artifact everyday/Vase
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

# Optional trimesh for mesh inspection
try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False


# ──────────────────────────────────────────────────────────────────────
# Split file loader
# ──────────────────────────────────────────────────────────────────────

def load_split_file(path: str) -> list[str]:
    """Read a split .txt file and return non-empty, stripped lines."""
    if not os.path.exists(path):
        return []
    with open(path, 'r') as f:
        return [l.strip() for l in f if l.strip()]


def get_objects_from_splits(split_dir: str, subset: str) -> dict:
    """
    Get train/val object lists from official split files.
    
    For 'artifact':      reads artifact.train.txt / artifact.val.txt
    For 'everyday/Vase': reads everyday.train.txt / everyday.val.txt,
                         filters lines starting with 'everyday/Vase/'
    """
    prefix = subset.split('/')[0]  # 'artifact' or 'everyday'
    
    train_entries = load_split_file(os.path.join(split_dir, f"{prefix}.train.txt"))
    val_entries = load_split_file(os.path.join(split_dir, f"{prefix}.val.txt"))
    
    # Filter for specific subset if needed (e.g. everyday/Vase)
    if '/' in subset:
        train_entries = [e for e in train_entries if e.startswith(subset + '/')]
        val_entries = [e for e in val_entries if e.startswith(subset + '/')]
    
    return {'train': train_entries, 'val': val_entries}


# ──────────────────────────────────────────────────────────────────────
# Object inspector (deep inspection of a few samples)
# ──────────────────────────────────────────────────────────────────────

def inspect_object(object_dir: str, use_trimesh: bool = False) -> dict:
    """
    Deeply inspect one object directory structure.
    
    Returns counts of fractured_N dirs, mode_N dirs, pieces per each, 
    and optionally mesh stats.
    """
    result = {
        'exists': os.path.isdir(object_dir),
        'fractured_dirs': [],
        'mode_dirs': [],
        'total_broken_pieces': 0,
        'total_complete_pieces': 0,
    }
    
    if not result['exists']:
        return result
    
    try:
        entries = sorted(os.listdir(object_dir))
    except OSError as e:
        result['error'] = str(e)
        return result
    
    for entry in entries:
        entry_path = os.path.join(object_dir, entry)
        if not os.path.isdir(entry_path):
            continue
            
        if entry.startswith('fractured_'):
            try:
                pieces = [f for f in os.listdir(entry_path) if f.endswith('.obj')]
            except OSError:
                pieces = []
            
            frac_info = {'name': entry, 'num_pieces': len(pieces), 'pieces': sorted(pieces)}
            
            # Optional mesh stats for first piece
            if use_trimesh and HAS_TRIMESH and pieces and not result['fractured_dirs']:
                try:
                    mesh_path = os.path.join(entry_path, sorted(pieces)[0])
                    mesh = trimesh.load(mesh_path, force='mesh', process=False)
                    frac_info['sample_mesh'] = {
                        'file': sorted(pieces)[0],
                        'vertices': int(mesh.vertices.shape[0]),
                        'faces': int(mesh.faces.shape[0]),
                        'is_watertight': bool(mesh.is_watertight),
                    }
                except Exception as e:
                    frac_info['sample_mesh'] = {'error': str(e)}
            
            result['fractured_dirs'].append(frac_info)
            result['total_broken_pieces'] += len(pieces)
        
        elif entry.startswith('mode_'):
            try:
                pieces = [f for f in os.listdir(entry_path) if f.endswith('.obj')]
            except OSError:
                pieces = []
            
            mode_info = {'name': entry, 'num_pieces': len(pieces), 'pieces': sorted(pieces)}
            
            if use_trimesh and HAS_TRIMESH and pieces and not result['mode_dirs']:
                try:
                    mesh_path = os.path.join(entry_path, sorted(pieces)[0])
                    mesh = trimesh.load(mesh_path, force='mesh', process=False)
                    mode_info['sample_mesh'] = {
                        'file': sorted(pieces)[0],
                        'vertices': int(mesh.vertices.shape[0]),
                        'faces': int(mesh.faces.shape[0]),
                        'is_watertight': bool(mesh.is_watertight),
                    }
                except Exception as e:
                    mode_info['sample_mesh'] = {'error': str(e)}
            
            result['mode_dirs'].append(mode_info)
            result['total_complete_pieces'] += len(pieces)
    
    result['num_fractured_variations'] = len(result['fractured_dirs'])
    result['num_mode_variations'] = len(result['mode_dirs'])
    
    return result


# ──────────────────────────────────────────────────────────────────────
# Classification strategy analysis
# ──────────────────────────────────────────────────────────────────────

def analyze_classification_strategies(sample_inspections: list[dict], 
                                       total_objects: int) -> dict:
    """
    Based on a sample of deeply inspected objects, extrapolate
    classification statistics for the full dataset.
    """
    if not sample_inspections:
        return {}
    
    # Collect per-sample stats
    frac_variations_per_obj = []
    mode_variations_per_obj = []
    pieces_per_frac = []
    pieces_per_mode = []
    total_broken = 0
    total_complete = 0
    
    for s in sample_inspections:
        frac_variations_per_obj.append(s.get('num_fractured_variations', 0))
        mode_variations_per_obj.append(s.get('num_mode_variations', 0))
        total_broken += s.get('total_broken_pieces', 0)
        total_complete += s.get('total_complete_pieces', 0)
        
        for fd in s.get('fractured_dirs', []):
            pieces_per_frac.append(fd['num_pieces'])
        for md in s.get('mode_dirs', []):
            pieces_per_mode.append(md['num_pieces'])
    
    n_samples = len(sample_inspections)
    
    # Compute averages from sample
    avg_frac_vars = sum(frac_variations_per_obj) / max(n_samples, 1)
    avg_mode_vars = sum(mode_variations_per_obj) / max(n_samples, 1)
    avg_pieces_per_frac = sum(pieces_per_frac) / max(len(pieces_per_frac), 1)
    avg_pieces_per_mode = sum(pieces_per_mode) / max(len(pieces_per_mode), 1)
    
    # Extrapolated totals
    est_total_frac_dirs = int(avg_frac_vars * total_objects)
    est_total_mode_dirs = int(avg_mode_vars * total_objects)
    est_broken_pieces = int(total_broken / n_samples * total_objects)
    est_complete_pieces = int(total_complete / n_samples * total_objects)
    
    return {
        'samples_inspected': n_samples,
        'total_objects': total_objects,
        'sample_stats': {
            'avg_fractured_variations_per_obj': round(avg_frac_vars, 1),
            'avg_mode_variations_per_obj': round(avg_mode_vars, 1),
            'avg_pieces_per_fractured_dir': round(avg_pieces_per_frac, 1),
            'avg_pieces_per_mode_dir': round(avg_pieces_per_mode, 1),
            'pieces_per_frac_min': min(pieces_per_frac) if pieces_per_frac else 0,
            'pieces_per_frac_max': max(pieces_per_frac) if pieces_per_frac else 0,
            'pieces_per_mode_min': min(pieces_per_mode) if pieces_per_mode else 0,
            'pieces_per_mode_max': max(pieces_per_mode) if pieces_per_mode else 0,
        },
        'strategy_A_all_pieces': {
            'description': 'Every piece_*.obj in fractured_N = broken,'
                          ' every piece_*.obj in mode_N = complete',
            'broken_samples': est_broken_pieces,
            'complete_samples': est_complete_pieces,
            'total': est_broken_pieces + est_complete_pieces,
            'imbalance_ratio': round(est_broken_pieces / max(est_complete_pieces, 1), 2),
        },
        'strategy_B_one_piece_per_fracture': {
            'description': 'Random 1 piece from each fractured_N = broken,'
                          ' 1 piece from each mode_N = complete',
            'broken_samples': est_total_frac_dirs,
            'complete_samples': est_total_mode_dirs,
            'total': est_total_frac_dirs + est_total_mode_dirs,
            'imbalance_ratio': round(est_total_frac_dirs / max(est_total_mode_dirs, 1), 2),
        },
        'strategy_C_per_mode_matched': {
            'description': 'Each fracture_N with mode_N pair: fractured_N→broken,'
                          ' mode_N→complete (1:1 balanced)',
            'broken_samples': est_total_mode_dirs,
            'complete_samples': est_total_mode_dirs,
            'total': est_total_mode_dirs * 2,
            'imbalance_ratio': 1.0,
            'note': 'Uses only N fracture dirs matching mode dirs, '
                    'drops extra fracture variations to achieve balance',
        },
    }


# ──────────────────────────────────────────────────────────────────────
# Pretty printer
# ──────────────────────────────────────────────────────────────────────

def print_subset_report(subset: str, split_info: dict, 
                        inspections: list[dict], cls_analysis: dict):
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  SUBSET: {subset.upper()}")
    print(f"{sep}")
    
    train_n = len(split_info['train'])
    val_n = len(split_info['val'])
    total_n = train_n + val_n
    
    print(f"\n  Objects from splits:")
    print(f"    Train: {train_n}")
    print(f"    Val:   {val_n}")
    print(f"    Total: {total_n}")
    
    # Sample inspection details
    print(f"\n  Deep inspection of {len(inspections)} sample objects:")
    for i, insp in enumerate(inspections):
        obj_id = insp.get('_obj_id', f'sample_{i}')
        if not insp.get('exists', False):
            print(f"    ✗ {obj_id}: NOT FOUND ON DISK")
            continue
        
        n_frac = insp.get('num_fractured_variations', 0)
        n_mode = insp.get('num_mode_variations', 0)
        n_bp = insp.get('total_broken_pieces', 0)
        n_cp = insp.get('total_complete_pieces', 0)
        
        print(f"    {obj_id}:")
        print(f"      fractured dirs: {n_frac} (total pieces: {n_bp})")
        print(f"      mode dirs:      {n_mode} (total pieces: {n_cp})")
        
        # Show piece distribution for first few fractured dirs
        for fd in insp.get('fractured_dirs', [])[:3]:
            print(f"        {fd['name']}: {fd['num_pieces']} pieces "
                  f"({', '.join(fd['pieces'][:4])}{'...' if len(fd['pieces'])>4 else ''})")
        
        # Show mode dirs
        for md in insp.get('mode_dirs', [])[:3]:
            print(f"        {md['name']}: {md['num_pieces']} pieces "
                  f"({', '.join(md['pieces'][:4])}{'...' if len(md['pieces'])>4 else ''})")
        
        # Mesh stats if available
        for fd in insp.get('fractured_dirs', []):
            if 'sample_mesh' in fd and 'error' not in fd['sample_mesh']:
                ms = fd['sample_mesh']
                print(f"      sample broken mesh ({ms['file']}): "
                      f"{ms['vertices']} verts, {ms['faces']} faces, "
                      f"watertight={ms['is_watertight']}")
                break
        for md in insp.get('mode_dirs', []):
            if 'sample_mesh' in md and 'error' not in md['sample_mesh']:
                ms = md['sample_mesh']
                print(f"      sample complete mesh ({ms['file']}): "
                      f"{ms['vertices']} verts, {ms['faces']} faces, "
                      f"watertight={ms['is_watertight']}")
                break
    
    # Classification strategies
    if cls_analysis:
        print(f"\n  {'─'*60}")
        print(f"  CLASSIFICATION STRATEGIES (extrapolated from {cls_analysis['samples_inspected']} samples → {cls_analysis['total_objects']} objects)")
        print(f"  {'─'*60}")
        
        ss = cls_analysis['sample_stats']
        print(f"\n  Averages per object:")
        print(f"    fractured variations: ~{ss['avg_fractured_variations_per_obj']}")
        print(f"    mode variations:      ~{ss['avg_mode_variations_per_obj']}")
        print(f"    pieces per fractured:  {ss['pieces_per_frac_min']}-{ss['pieces_per_frac_max']} (avg {ss['avg_pieces_per_fractured_dir']})")
        print(f"    pieces per mode:       {ss['pieces_per_mode_min']}-{ss['pieces_per_mode_max']} (avg {ss['avg_pieces_per_mode_dir']})")
        
        for name in ['strategy_A_all_pieces', 'strategy_B_one_piece_per_fracture',
                      'strategy_C_per_mode_matched']:
            s = cls_analysis[name]
            print(f"\n  {name}:")
            print(f"    broken={s['broken_samples']}, complete={s['complete_samples']}, "
                  f"total={s['total']}")
            print(f"    imbalance ratio: {s['imbalance_ratio']}:1")
            print(f"    → {s['description']}")
            if 'note' in s:
                print(f"    NOTE: {s['note']}")


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Analyze Breaking Bad dataset for binary classification'
    )
    parser.add_argument('--data_root', type=str, default='data/BreakingBad')
    parser.add_argument('--split_dir', type=str, default='data/BreakingBad/data_split')
    parser.add_argument('--subsets', nargs='+', default=['artifact', 'everyday/Vase'])
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--sample_count', type=int, default=5,
                        help='Number of objects to deeply inspect per subset')
    parser.add_argument('--use-trimesh', action='store_true',
                        help='Load .obj meshes with trimesh for vertex/face stats')
    args = parser.parse_args()

    # Resolve paths
    if not os.path.isabs(args.data_root):
        args.data_root = os.path.join(os.getcwd(), args.data_root)
    if not os.path.isabs(args.split_dir):
        args.split_dir = os.path.join(os.getcwd(), args.split_dir)
    
    output_path = args.output or os.path.join(
        args.data_root, 'bb_classification_metadata.json'
    )
    
    use_mesh = args.use_trimesh and HAS_TRIMESH
    if args.use_trimesh and not HAS_TRIMESH:
        print("WARNING: trimesh not installed. Mesh stats will be skipped.")
    
    all_metadata = {
        'analysis_date': time.strftime('%Y-%m-%d %H:%M:%S'),
        'data_root': args.data_root,
        'subsets': {},
    }
    
    global_inspections = []
    global_total = 0
    
    for subset in args.subsets:
        print(f"\n{'#'*72}")
        print(f"  Processing subset: {subset}")
        print(f"{'#'*72}")
        
        # 1. Get object list from split files (fast — no filesystem crawl)
        split_info = get_objects_from_splits(args.split_dir, subset)
        all_entries = split_info['train'] + split_info['val']
        
        if not all_entries:
            print(f"  No entries found for {subset} in split files.")
            continue
        
        total_objects = len(all_entries)
        global_total += total_objects
        
        # 2. Deep inspect a sample of objects (slow per object on this fs)
        n_sample = min(args.sample_count, total_objects)
        # Pick evenly spaced samples: first, last, and middle
        if n_sample >= total_objects:
            sample_indices = list(range(total_objects))
        else:
            sample_indices = [int(i * (total_objects - 1) / (n_sample - 1)) 
                            for i in range(n_sample)]
        
        inspections = []
        for idx in sample_indices:
            entry = all_entries[idx]
            # Build the full path: data_root/entry
            obj_dir = os.path.join(args.data_root, entry)
            
            print(f"  Inspecting [{len(inspections)+1}/{n_sample}]: {entry} ... ", 
                  end='', flush=True)
            t0 = time.time()
            insp = inspect_object(obj_dir, use_trimesh=use_mesh)
            insp['_obj_id'] = entry
            elapsed = time.time() - t0
            print(f"done ({elapsed:.1f}s)")
            
            inspections.append(insp)
        
        global_inspections.extend(inspections)
        
        # 3. Analyze classification strategies
        valid_inspections = [i for i in inspections if i.get('exists', False)]
        cls_analysis = analyze_classification_strategies(valid_inspections, total_objects)
        
        # 4. Print report
        print_subset_report(subset, split_info, inspections, cls_analysis)
        
        # 5. Check which entries actually exist on disk
        print(f"\n  Validating split entries on disk (checking first 10)...")
        found = 0
        missing = 0
        missing_list = []
        for entry in all_entries[:10]:
            obj_dir = os.path.join(args.data_root, entry)
            if os.path.isdir(obj_dir):
                found += 1
            else:
                missing += 1
                missing_list.append(entry)
        print(f"    Checked: {found + missing}, Found: {found}, Missing: {missing}")
        if missing_list:
            print(f"    Missing examples: {missing_list[:5]}")
        
        # Save to metadata
        all_metadata['subsets'][subset] = {
            'train_count': len(split_info['train']),
            'val_count': len(split_info['val']),
            'total_objects': total_objects,
            'sample_inspections': inspections,
            'classification_analysis': cls_analysis,
            'missing_from_disk_sample': missing_list,
        }
    
    # ── Global summary ───────────────────────────────────────────────
    if global_inspections:
        valid_global = [i for i in global_inspections if i.get('exists', False)]
        global_cls = analyze_classification_strategies(valid_global, global_total)
        all_metadata['global_stats'] = {
            'total_objects_all_subsets': global_total,
            'classification_analysis': global_cls,
        }
        
        print(f"\n{'#'*72}")
        print(f"  GLOBAL SUMMARY (all subsets combined)")
        print(f"{'#'*72}")
        print(f"  Total objects: {global_total}")
        
        if global_cls:
            for name in ['strategy_A_all_pieces', 'strategy_B_one_piece_per_fracture',
                          'strategy_C_per_mode_matched']:
                s = global_cls[name]
                print(f"\n  {name}:")
                print(f"    broken={s['broken_samples']}, complete={s['complete_samples']}, "
                      f"total={s['total']}")
                print(f"    imbalance ratio: {s['imbalance_ratio']}:1")
            
            # Recommendation
            print(f"\n  {'─'*60}")
            print(f"  RECOMMENDATION")
            print(f"  {'─'*60}")
            strat_a = global_cls['strategy_A_all_pieces']
            strat_c = global_cls['strategy_C_per_mode_matched']
            
            if strat_a['imbalance_ratio'] > 3.0:
                print(f"  ⚠ Strategy A is heavily imbalanced ({strat_a['imbalance_ratio']}:1)")
                print(f"  ✓ Use Strategy C for balanced 1:1 training ({strat_c['total']} samples)")
                print(f"    Or use Strategy B + class weights / weighted sampler")
            else:
                print(f"  Strategy A looks manageable ({strat_a['imbalance_ratio']}:1)")
                print(f"  Can also use Strategy C for perfect balance")
            
            print(f"\n  IMPORTANT FINDING:")
            ss = global_cls.get('sample_stats', {})
            if ss.get('pieces_per_mode_max', 0) > 1:
                print(f"  → mode_N dirs contain MULTIPLE pieces ({ss['pieces_per_mode_min']}-{ss['pieces_per_mode_max']})")
                print(f"    Each piece in mode_N is a FRAGMENT of the complete object!")
                print(f"    The complete object = union of all pieces in a mode_N dir.")
                print(f"    For classification: use the REASSEMBLED complete mesh,")
                print(f"    not individual mode pieces.")
            else:
                print(f"  → mode_N dirs contain 1 piece (pre-assembled complete object)")
    
    # ── GPU recommendation ───────────────────────────────────────────
    print(f"\n  {'─'*60}")
    print(f"  POINTNEXT VARIANT RECOMMENDATION")
    print(f"  {'─'*60}")
    print(f"  System: RTX 3090 24GB (GPU 3, index 3)")
    print(f"  CUDA_VISIBLE_DEVICES=3")
    print(f"")
    print(f"  PointNeXt-S:  blocks=[1,1,1,1,1,1], width=32  →  ~1.4M params, ~2-3GB VRAM  ✓")
    print(f"  PointNeXt-B:  blocks=[1,2,3,2,2],    width=32  →  ~4.2M params, ~5-7GB VRAM  ✓ RECOMMENDED")
    print(f"  PointNeXt-L:  blocks=[1,3,5,3,3],    width=32  →  ~7.1M params, ~8-10GB VRAM ✓ Feasible")
    print(f"  PointNeXt-XL: blocks=[1,4,7,4,4],    width=64  → ~41.6M params, ~14-18GB VRAM ✓ Fits 24GB")
    print(f"")
    print(f"  → PointNeXt-B is recommended for pretraining (good capacity/speed tradeoff)")
    print(f"  → PointNeXt-L is also feasible if more capacity is needed")
    print(f"  → PointNeXt-XL fits but will be slower to train")
    
    # ── Save ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(all_metadata, f, indent=2, default=str)
    print(f"\n  Metadata saved to: {output_path}")


if __name__ == '__main__':
    main()
