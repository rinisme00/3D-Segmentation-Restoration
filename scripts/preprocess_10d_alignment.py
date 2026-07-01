import os
import glob
import numpy as np
from multiprocessing import Pool
import argparse
from tqdm import tqdm
import sys

CODE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.dirname(CODE_ROOT))
sys.path.append(os.path.join(CODE_ROOT, "src"))
from restoration.utils.point_alignment import align_fracture_probabilities_scipy

GUIDANCE_KEYS = ("fracture_probability", "fracture_prob", "p_frac", "prediction", "pred")


def load_guidance(pair_path, pairs_dir, pointnext_dir):
    rel_path = os.path.relpath(pair_path, start=pairs_dir)
    guidance_path = os.path.join(pointnext_dir, rel_path)
    if not os.path.exists(guidance_path):
        raise FileNotFoundError(f"Missing PointNeXt guidance file: {guidance_path}")

    guidance = np.load(guidance_path)
    xyz = guidance["points"] if "points" in guidance else guidance["xyz"]
    for key in GUIDANCE_KEYS:
        if key in guidance:
            p_frac = guidance[key]
            break
    else:
        raise KeyError(f"{guidance_path} does not contain one of {GUIDANCE_KEYS}")

    return np.hstack([xyz.astype(np.float32), np.asarray(p_frac, dtype=np.float32).reshape(-1, 1)])


def process_file(args):
    pair_path, pairs_dir, pointnext_dir, out_dir = args
    try:
        data = np.load(pair_path)
        partial_points = data['partial_points']
        partial_features = data['partial_features']
        x_orig_9d = np.hstack([partial_points, partial_features])
        x_pred_4d = load_guidance(pair_path, pairs_dir, pointnext_dir)
        x_10d = align_fracture_probabilities_scipy(x_orig_9d, x_pred_4d)

        rel_path = os.path.relpath(pair_path, start=pairs_dir)
        dest_path = os.path.join(out_dir, rel_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        new_features = x_10d[:, 3:] # 7 dims (6 features + 1 prob)
        np.savez_compressed(
            dest_path,
            partial_points=partial_points,
            partial_features=new_features,
            complete_points=data.get('complete_points', None),
            complete_features=data.get('complete_features', None),
            metadata=data.get('metadata', None)
        )
        return True
    except Exception as e:
        print(f"Error processing {pair_path}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pairs-dir", type=str, default=os.path.join(PROJECT_ROOT, "preprocessed", "restoration", "completion_pairs_9d"))
    parser.add_argument("--pointnext-dir", type=str, default=os.path.join(PROJECT_ROOT, "preprocessed", "restoration", "guidance", "pointnext_c2_9d"))
    parser.add_argument("--out-dir", type=str, default=os.path.join(PROJECT_ROOT, "preprocessed", "restoration", "completion_pairs_10d"))
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    
    npz_files = glob.glob(os.path.join(args.pairs_dir, "**", "*.npz"), recursive=True)
    print(f"Found {len(npz_files)} pairs to process.")
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    task_args = [(f, args.pairs_dir, args.pointnext_dir, args.out_dir) for f in npz_files]
    
    # Process
    success = 0
    with Pool(args.workers) as p:
        for res in tqdm(p.imap_unordered(process_file, task_args), total=len(task_args)):
            if res:
                success += 1
                
    print(f"Completed {success}/{len(task_args)} files.")

if __name__ == "__main__":
    main()
