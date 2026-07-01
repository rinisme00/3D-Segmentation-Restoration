# 3D Segmentation and Restoration

This repository contains a focused 3D computer vision pipeline for fracture-aware representations:

- Fracture-surface segmentation with PointNeXt-9D.
- Segmentation-guided shape-completion restoration with Conditioned-DeepMend.
- Dataset inventory, preprocessing, evaluation, and thesis-ready visual outputs for fractured objects.

The active datasets are Fantastic Breaks and Breaking Bad. The code is optimized and tested for a Blackwell architecture GPU, specifically an RTX Pro 4500 with 32 GB VRAM.

## Installation

Clone the repository with submodules:

```bash
git clone --recurse-submodules <your_repository_url>
cd 3D-Segmentation-Restoration
```

If the repository was cloned without submodules, initialize them afterwards:

```bash
git submodule update --init --recursive
```

Create and activate the Conda environment:

```bash
conda create -n fracture3d python=3.10 -y
conda activate fracture3d
```

Install PyTorch for CUDA 12.8:

```bash
pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 torchaudio==2.11.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

Install the project dependencies:

```bash
pip install -r requirements.txt
```

Build PointNeXt CUDA extensions if your workflow uses the vendored OpenPoints operators:

```bash
cd src/pointnext
pip install -e .
cd ../..
```

## Submodules

This repository already references the required external code through `.gitmodules`:

- `src/pointnext`: PointNeXt fork at `https://github.com/rinisme00/pointnext`
- `src/pointnext/openpoints`: OpenPoints fork referenced by the PointNeXt submodule
- `breaking-bad-dataset`: Breaking Bad dataset helper repository

You do not need to create a new GitHub repository for these existing submodules. Create a new GitHub repository only when adding a new external dependency that should be versioned independently from this project. In that case, add it with `git submodule add <repo_url> <path>` and commit the resulting `.gitmodules` change plus the submodule pointer.

For non-training commands, disable online tracking by default:

```bash
export WANDB_DISABLED=true
```

## Datasets

Prepare Fantastic Breaks and Breaking Bad outside the git repository or under a configurable data root. The expected labels are:

- `0`: intact surface
- `1`: fracture surface

Use Fantastic Breaks masks and validated Breaking Bad sidecar annotations as supervised segmentation labels. Keep raw datasets and manually annotated files immutable.

## Usage

Build or refresh dataset manifests:

```bash
WANDB_DISABLED=true python scripts/data/build_dataset_manifests.py \
  --project-root <your_project_path> \
  --data-root <your_dataset_path> \
  --summary-json <your_output_path>/manifests/manifest_summary.json
```

Generate PointNeXt 9D segmentation samples:

```bash
WANDB_DISABLED=true python scripts/segmentation/generate_full_dataset.py \
  --metadata-csv <your_manifest_path> \
  --aligned-mesh-dir <your_output_path>/segmentation/aligned_meshes \
  --pts-dir <your_output_path>/segmentation/pointnext_9d/pts \
  --seg-dir <your_output_path>/segmentation/pointnext_9d/seg \
  --txt-dir <your_output_path>/segmentation/pointnext_9d/txt \
  --qa-dir <your_output_path>/segmentation/pointnext_9d/qa \
  --summary-csv <your_output_path>/segmentation/pointnext_9d/summary.csv
```

Train PointNeXt-9D segmentation:

```bash
cd src/pointnext
nohup python examples/segmentation/main.py \
  --cfg cfgs/fantasticbreaks_seg/fb_fullshot_9d.yaml \
  datatransforms.kwargs.data_root=<your_dataset_path> \
  log_dir=<your_output_path>/segmentation/pointnext_9d \
  wandb.use_wandb=False \
  > <your_output_path>/segmentation/pointnext_9d/train.log 2>&1 &
cd ../..
```

Export PointNeXt segmentation guidance for restoration:

```bash
WANDB_DISABLED=true python -m src.restoration.guidance.export_segmentation_guidance \
  --seg-checkpoint <your_checkpoint_path> \
  --split-csv <your_restoration_split_csv> \
  --output-root <your_output_path>/restoration/guidance/pointnext_9d \
  --project-root <your_project_path> \
  --code-root <your_code_path> \
  --feature-mode 9d
```

Attach segmentation guidance to DeepMend inputs:

```bash
WANDB_DISABLED=true python scripts/preprocess_10d_alignment.py \
  --pairs-dir <your_dataset_path>/preprocessed/restoration/completion_pairs_9d \
  --pointnext-dir <your_output_path>/restoration/guidance/pointnext_9d \
  --out-dir <your_dataset_path>/preprocessed/restoration/completion_pairs_10d \
  --workers 8
```

Train Conditioned-DeepMend:

```bash
nohup python scripts/train_deepmend_10d.py \
  --csv-path <your_dataset_path>/preprocessed/restoration/completion_pairs_10d/sample_index.csv \
  --output-dir <your_output_path>/restoration/deepmend_conditioned \
  --batch-size 16 \
  --epochs 100 \
  > <your_output_path>/restoration/deepmend_conditioned/train.log 2>&1 &
```

Evaluate Conditioned-DeepMend:

```bash
WANDB_DISABLED=true python scripts/evaluate_deepmend_10d.py \
  --model-path <your_checkpoint_path>/best_model.pth \
  --csv-path <your_dataset_path>/preprocessed/restoration/completion_pairs_10d/sample_index.csv \
  --output-dir <your_output_path>/restoration/deepmend_conditioned/eval \
  --eval-limit 8
```

## Results

Add final thesis metrics and visual examples here after running the retained segmentation and restoration pipelines:

- PointNeXt-9D fracture IoU / F1: `<pending>`
- Conditioned-DeepMend Chamfer distance / fracture-region Chamfer distance: `<pending>`
- Qualitative segmentation and restoration figures: `<pending>`
