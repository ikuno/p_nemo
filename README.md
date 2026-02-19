# GeoTransolver 2D

2D implementation of **GeoTransolver** — a Multi-scale Geometry-Aware Physics Attention Transformer for surrogate modeling of 2D cylinder flow.

Based on:
- [GeoTransolver (arXiv: 2512.20399v2)](https://arxiv.org/abs/2512.20399) — NVIDIA
- [Benchmarking Framework (arXiv: 2507.10747v1)](https://arxiv.org/abs/2507.10747) — NVIDIA PhysicsNeMo-CFD

---

## Overview

This project implements GeoTransolver for **2D incompressible potential flow around a circular cylinder**, predicting velocity (v_x, v_y) and pressure (p) fields from sparse point samples.

### Architecture

```
[Input]
  positions      : (B, N, 2)     domain points
  geometry       : (M_g, 2/2)    cylinder boundary + normals
  global_params  : (B, 1)        freestream velocity U_inf
       |
[ContextProjector]   geometry + global params → context vector C
       |
[Encoder]            [input features | ball-query augmentation] → d_model
       |
[GALE Block × L]
  ├─ Self-Attention  (within physics slices)
  ├─ Cross-Attention (to shared context C)
  └─ Adaptive Gate   (α blends self/cross)
       |
[Output Heads]
  Slice 0 → velocity  (v_x, v_y)
  Slice 1 → pressure  (p)
       |
[Output]  (B, N, 3)
```

---

## Environment Setup

### Requirements

- Python 3.12
- CUDA 12.8+ (driver: 13.0 tested, RTX PRO 6000 Blackwell)
- conda

### Create / Update conda environment

```bash
# Create environment from yml (first time)
conda env create -f environment.yml

# Activate
conda activate myenv

# Update existing environment
conda env update -f environment.yml --prune
```

### Install Python dependencies

```bash
conda activate myenv
pip install -r requirements.txt
```

---

## Usage

All commands are run with `conda activate myenv` active.

### Training

```bash
# Single GPU
conda run -n myenv python cli.py train --epochs 200 --lr 1e-3 --batch-size 16

# All 4 GPUs (DataParallel)
conda run -n myenv python cli.py train \
    --epochs 200 \
    --lr 1e-3 \
    --batch-size 64 \
    --n-samples 200 \
    --n-points 512 \
    --gpus 0,1,2,3

# Specify specific GPUs
conda run -n myenv python cli.py train --gpus 0,1 --epochs 100
```

**Key training options:**

| Option | Default | Description |
|--------|---------|-------------|
| `--epochs` | 200 | Number of training epochs |
| `--lr` | 1e-3 | Learning rate |
| `--batch-size` | 16 | Batch size (scale up with more GPUs) |
| `--n-samples` | 100 | Number of flow samples |
| `--n-points` | 512 | Domain points per sample |
| `--n-geom` | 128 | Cylinder boundary points |
| `--d-model` | 64 | Hidden dimension |
| `--n-heads` | 4 | Attention heads |
| `--n-layers` | 4 | GALE block layers |
| `--gpus` | all | GPU IDs e.g. `"0,1,2,3"` |

Checkpoints are saved to `output/checkpoints/best.pt`.

### Evaluation

```bash
conda run -n myenv python cli.py eval \
    --checkpoint output/checkpoints/best.pt \
    --gpus 0
```

Outputs: MAE, Relative L1, R² (velocity), R² (pressure).

### Visualization

```bash
conda run -n myenv python cli.py visualize \
    --checkpoint output/checkpoints/best.pt \
    --output output/plots \
    --gpus 0
```

Saves to `output/plots/`:
- `velocity.png` — velocity magnitude (Predicted vs. Analytical)
- `pressure.png` — pressure field comparison
- `velocity_vectors.png` — quiver plot of velocity vectors

---

## Multi-GPU Notes

- Training uses `nn.DataParallel` across all visible GPUs.
- Checkpoints always save the unwrapped (non-DataParallel) `state_dict`.
- For 4x RTX PRO 6000 (96 GB VRAM each), `--batch-size 64` or larger is recommended.
- `--gpus` sets `CUDA_VISIBLE_DEVICES` before PyTorch initializes.

---

## Project Structure

```
nvidia_project/
├── src/geotransolver/       # Library (Article I)
│   ├── config.py            # Centralized configuration
│   ├── data.py              # 2D cylinder flow data generation
│   ├── ball_query.py        # Multi-scale ball query (REQ-002)
│   ├── context_projector.py # Geometry context projection (REQ-003)
│   ├── gale_attention.py    # GALE Attention block (REQ-004/005)
│   ├── model.py             # GeoTransolver2D (REQ-005)
│   ├── metrics.py           # MAE, Relative L1, R² (REQ-006)
│   └── visualize.py         # Plot utilities
├── tests/                   # Test suite (pytest)
│   ├── conftest.py
│   ├── test_data.py
│   ├── test_ball_query.py
│   ├── test_context_projector.py
│   ├── test_gale_attention.py
│   ├── test_model.py
│   ├── test_metrics.py
│   └── test_integration.py
├── output/
│   ├── checkpoints/         # Saved model weights
│   └── plots/               # Visualization outputs
├── steering/                # MUSUBIX project memory
├── storage/specs/           # Requirements & design specs
├── papers/                  # Reference paper summaries
├── cli.py                   # CLI entry point (Article II)
├── environment.yml          # conda environment
└── requirements.txt         # pip dependencies
```

---

## Testing

```bash
conda activate myenv

# Run all tests
pytest tests/ -v

# Run integration tests only
pytest tests/test_integration.py -v

# Run with coverage
pytest tests/ --cov=src/geotransolver --cov-report=term-missing
```

---

## Physics Background

**Analytical solution** (inviscid, incompressible potential flow around circular cylinder):

```
v_r     = U_inf * (1 - R²/r²) * cos(θ)
v_θ     = -U_inf * (1 + R²/r²) * sin(θ)
p       = p_inf + 0.5 * ρ * U_inf² * (1 - |v|²/U_inf²)   [Bernoulli]
```

The model learns to approximate this solution from sparse point observations, conditioned on the cylinder geometry and `U_inf`.

---

## References

1. Adams et al. (2024). *GeoTransolver: Learning Physics on Irregular Domains using Multi-scale Geometry Aware Physics Attention Transformer*. arXiv: 2512.20399v2. NVIDIA.
2. NVIDIA PhysicsNeMo-CFD Team (2025). *Benchmarking Framework for AI-based Automotive Aerodynamics*. arXiv: 2507.10747v1.

---

**MUSUBIX** v3.7.3 | Generated: 2026-02-18
