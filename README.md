# DINOv3 Lab

A modular repository for DINOv3-based tasks, designed to keep the backbone separate from task heads.

## Structure

- `configs/`: Hydra configurations for backbones, data, tasks, and experiments.
- `src/dinov3_lab/`: Source code.
  - `core/`: Backbone wrapper and core utilities.
  - `tasks/`: Task-specific heads (segmentation, detection, depth, fusion3d).
  - `data/`: Datasets and datamodules.
- `scripts/`: Training and evaluation scripts.
- `notebooks/`: Jupyter notebooks for exploration.

## Quickstart

```bash
# Install dependencies
pip install -e .

# Download backbone (placeholder)
# python -m scripts.download_backbone --model dinov3_vitl16

# Run a segmentation experiment (placeholder)
# python -m scripts.train_segmentation --config configs/experiments/seg_ade20k_vitl16.yaml
```
