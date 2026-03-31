# Pseudo-3D Reconstruction with Z-Axis Mamba

This project reconstructs dense 3D sandstone CT volumes from sparse serial sections. The input is a partially observed volume where only every *N*th slice is kept, and the model predicts the missing slices to recover a dense volume.

## Overview

The project began with an asymmetric 3D U-Net for sparse-to-dense reconstruction. The encoder uses slice-wise `(1,3,3)` convolutions so each observed slice is encoded independently, while the decoder uses volumetric reasoning to reconstruct missing depth information. The current extension adds a **Z-Axis Mamba decoder**, which models long-range dependencies along the depth axis more effectively than local 3D convolutions alone.

## Main Idea

Observed slices are sparse along the **Z axis**, so the hardest part of reconstruction is reasoning across missing depth positions. A local kernel such as `(2,3,3)` can only propagate depth information gradually. The Z-Axis Mamba module instead treats each spatial location across depth as a sequence and enables long-range interaction across slices, improving cross-slice continuity and reconstruction quality under higher sparsity.

## Current Pipeline

1. Load dense 3D sandstone CT volumes from the DRP-317 dataset.
2. Extract cubic patches from each volume.
3. Simulate sparse serial sections by keeping only every 5th, 10th, or 15th slice.
4. Train the model to reconstruct the full dense patch from the sparse input.
5. Use masked reconstruction so the model learns to predict missing slices rather than simply copying observed ones.

## Model Variants

### 1. Baseline: Asymmetric 3D Reconstruction Network
- **Encoder:** `(1,3,3)` convolutions
- **Decoder:** `(2,3,3)` convolutions
- **Goal:** local cross-slice fusion for missing-slice reconstruction

### 2. Improved Model: Z-Axis Mamba Decoder
- Keeps the same asymmetric encoder
- Replaces selected decoder stages with **hybrid local conv + Z-Mamba blocks**
- Models long-range structure along depth while preserving local spatial refinement

## Dataset

The project currently uses sandstone CT volumes from:

- **DRP-317** sandstone dataset, link - https://digitalporousmedia.org/published-datasets/drp.project.published.DRP-317

Configured sample splits:
- **Train:** Berea, Bandera Brown, Bandera Gray, Bentheimer, Berea Sister Gray, Berea Upper Gray, Buff Berea, Kirby
- **Validation:** Leopard, CastleGate
- **Test:** Parker

## Repository Structure

```text
3D-Reconstruction/
├── checkpoints/
├── data/
│   ├── __init__.py
│   └── sandstone.py
├── dataset/
├── logs/
├── models/
│   ├── __init__.py
│   ├── losses.py
│   ├── metrics.py
│   ├── optimizers.py
│   ├── progress.py
│   ├── segment.py
│   └── unet3d_rec.py
├── results/
├── config.yaml
├── eval.py
├── options.py
├── README.md
├── requirements.txt
└── train.py
```

## Important Files

- `data/sandstone.py` — dataset loader and sparse slice simulation
- `models/unet3d_rec.py` — baseline asymmetric model and Z-Mamba model
- `models/segment.py` — PyTorch Lightning training wrapper
- `train.py` — training entry point
- `config.yaml` — experiment configuration

## Installation

Create and activate your environment, then install dependencies.

```bash
pip install -r requirements.txt
```

For Z-Axis Mamba, install:

```bash
pip install causal-conv1d>=1.4.0 --no-build-isolation
pip install mamba-ssm --no-build-isolation
```

If installation is unstable on Windows, use **WSL2 / Ubuntu**.

## Configuration

Change data directory `config.yaml` settings

## Training

Run baseline:

```bash
python train.py --config config.yaml --network asym_mae
```

Run Z-Axis Mamba model:

```bash
python train.py --config config.yaml --network asym_zmamba --batch_size 4 --valid_batch_size 1
```

## Notes on Checkpoints

Checkpoints are saved under:

```text
checkpoints/<run_name>/
```

If the folder is created but no `.ckpt` files appear, check the following:
- `validate: true`
- `monitor: val_loss`
- validation is actually running
- dataset contains non-zero patches
- training is completing at least one epoch

## Why Z-Axis Mamba?

A `(2,3,3)` decoder kernel only sees **2 slices at a time** along depth. This is local and can struggle when slices are missing at larger intervals such as every 10th or 15th slice. Z-Axis Mamba improves this by modeling each depth line as a sequence, allowing the network to capture longer-range cross-slice structure more efficiently.

## Expected Benefits

- Better interpolation across missing slices
- Stronger depth continuity
- Improved performance at higher sparsity factors
- Cleaner modeling of long-range volumetric structure

## Suggested Experiments

Recommended comparison set:
- `asym_mae`
- `asym_zmamba` with bottleneck-only Mamba
- `asym_zmamba` with bottleneck + upper decoder Mamba
- performance comparison at subsampling factors 5, 10, and 15

## Resume Summary

- Implemented a Z-Axis Mamba–enhanced 3D reconstruction decoder for sparse volumetric data reconstruction.
- Designed a hybrid convolution–state space architecture for long-range depth modeling.
- Developed an asymmetric masked reconstruction framework for sandstone CT volume completion.

## Status

Current work focuses on:
- stabilizing Z-Axis Mamba training
- validating checkpoint generation
- comparing reconstruction quality against the asymmetric convolutional baseline
- analyzing performance under increasing slice sparsity
