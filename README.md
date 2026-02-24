# Dual-Tower Alignment for Neural Responses ↔ Visual Features (DS-Net + Transformer)

> This repository provides an implementation of a **dual-tower model** that aligns **neural responses** with **visual features** extracted from natural images.

---

## 1. Overview

This codebase trains a **dual-tower** architecture:

- **Image tower**: DS-Net backbone (MixVisionTransformer variant) + intermediate feature hook + 1×1 conv projection  
  Implemented in: `img_tower.py`, `models.py`.
- **Neural tower**: attention-based reconstruction from neuron responses (Transformer-style multi-head attention), producing a spatial feature map  
  Implemented in: `neural_tower.py`.
- **Training objective**: CLIP-style contrastive loss between image-tower features and neural-tower features (contrastive only).  
  Implemented in: `ModelTrainer.py`.
- **Entry script**: `Run.py` loads data, trains the towers, saves checkpoints and evaluation artifacts.

This repository is designed to be **reproducible**: seeds are fixed and deterministic CUDA behavior is enabled where possible.

---

## 2. Method Summary

### 2.1 Data
Each sample consists of:
- A natural image stimulus `image` (single-channel in current image)
- A neural response vector `response` (per-neuron response)

The loader reads:
- `images.mat`  → variable `images`
- `responses.mat` → variable `responses`

and returns:
- `target` (images) as tensor shaped roughly `[N, 1, H, W]`
- `s` (responses) normalized with `MinMaxScaler` to `[0, 1]` then tensor shaped `[num_neurons, N]` (or depending on your `.mat` layout)  
See `DataPre.py`.  


### 2.2 Image Tower
- Backbone: `ds_net_small` (MixVisionTransformer)  
- A forward hook extracts intermediate features at `blocks{stage}.{block-1}` (default stage=2, block=4; configurable).
- Feature projection: `Conv2d(128 → 1, kernel_size=1)`
- Upsample to spatial map `(16, 64)` via bilinear interpolation
- Per-sample min-max normalization

See `img_tower.py`, `models.py`.

### 2.3 Neural Tower
- For each neuron, a special linear layer embeds its scalar response to `d_model` dimension.
- Multi-head attention over neuron embeddings.
- A feed-forward MLP compresses into 256 dims, reshaped to `(8, 32)` then upsampled to `(16, 64)`.

See `neural_tower.py`.

### 2.4 Loss
Default training loss uses **CLIP-style symmetric cross-entropy** on similarity matrix:

- Normalize flattened feature maps
- Compute logits = `z_img @ z_neu^T * temperature`
- Compute CE loss in both directions and average

`ModelTrainer.py` includes a combined loss (`w_clip`, `w_rmse`, `w_ssim`), but in training/eval it is currently used as:
- `CLIPLoss(w_clip=1, w_rmse=0, w_ssim=0)` (contrastive only)

---

## 3. Repository Structure

```text
.
├── Run.py                 # Main entry: training and evaluation
├── DataPre.py             # Data loading and normalization (.mat files)
├── img_tower.py           # Image tower (DS-Net backbone + feature hook)
├── neural_tower.py        # Neural response tower (attention-based encoder)
├── models.py              # DS-Net / MixVisionTransformer implementation
├── ModelTrainer.py        # Training loop, loss functions, reproducibility
├── Model/                 # Saved checkpoints
└── M6/
    └── Data/
        ├── images.mat     # Visual stimuli
        └── responses.mat  # Neural responses
``` 

`Run.py` currently uses `sub_dirs = ['M6']` as an example. If we have multiple subjects, add them into that list.

---

## 4. Installation

### 4.1 Requirements

Tested with typical PyTorch + scientific Python stack.

Core dependencies:
- Python >= 3.8
- PyTorch (recommended >= 2.0 for Flash/SDPA attention)
- numpy
- scipy
- scikit-learn
- hdf5storage
- matplotlib

### 4.2 Installing Dependencies

Dependencies can be installed using `pip`:

- Create and activate a virtual environment (optional but recommended)
- Install required packages from `requirements.txt`

Example:
- pip install -r requirements.txt

### 4.3 Running the Code

After installing dependencies and preparing the dataset,
the entire training and evaluation pipeline can be executed
with a single command:

- python Run.py

The script will automatically:
- Load and preprocess image and neural response data
- Initialize the image and neural response towers
- Train the dual-tower model
- Evaluate on a held-out test set
- Save model checkpoints and evaluation results


