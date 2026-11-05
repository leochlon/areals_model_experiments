# GABI: Geometric Autoencoder Priors for Bayesian Inversion

**Learn First, Observe Later**: A geometry-aware framework for Bayesian inference on physical systems with variable geometries.

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2509.19929)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Overview

GABI learns geometry-conditioned generative models of physical responses that serve as informative priors for Bayesian inversion. Following a "learn first, observe later" paradigm, GABI distills information from large datasets of systems with varying geometries into a rich latent prior—without requiring knowledge of governing PDEs, boundary conditions, or observation processes. At inference time, this prior is combined with the likelihood of the specific observation process, yielding a geometry-adapted posterior distribution.

**Key capabilities:**
- Recovers full-field information from sparse, noisy observations
- Works on systems with complicated and variable geometries
- Architecture-agnostic framework (GCN, GEN, Transformer)
- Train-once-use-anywhere foundation model independent of observation process
- Efficient GPU-accelerated inference via Approximate Bayesian Computation (ABC)

**Applications demonstrated:**
- Steady-state heat transfer over rectangular domains
- 2D Reynolds-Averaged Navier-Stokes (RANS) flow around airfoils
- Helmholtz resonance and source localization on 3D car bodies
- 3D RANS airflow over complex terrain

## Installation

### Requirements

```bash
# Core dependencies
torch>=2.0.0
torch-geometric>=2.3.0
h5py>=3.8.0
numpy>=1.24.0
scipy>=1.10.0
pyyaml>=6.0
python-box>=7.0.0
tqdm>=4.65.0

# For inference
pyro-ppl>=1.8.0  # MCMC sampling

# Optional
einops>=0.6.0  # For transformer models
```

## Quick Start

### 1. Training a Model

Train a GEN-based autoencoder on wind terrain data:

```python
from datatools import WindTerrainDataset, compute_dataset_stats
from models import GENGeomAutoencoder
import torch
import yaml
from box import Box

# Load config
config = Box.from_yaml(filename='configs/example_gen_wind.yml')

# Load dataset
dataset = WindTerrainDataset(
    filename=config.io_settings.train_dataset_path,
    mode='train',
    channels=config.data_settings.channels,
    max_cells_above_terrain=config.data_settings.max_cells_above_terrain
)

# Get dimensions and create model
config.data_dims = dataset.get_data_dims_dict()
model = GENGeomAutoencoder(**config.data_dims, **config.hyperparameters, **config.model_settings)

# Train (see training_scripts for full examples)
```

Or use the provided training scripts:

```bash
# Single GPU
python training_scripts/train_single_gpu.py --config configs/example_gen_wind.yml

# Multi-GPU (DDP)
python training_scripts/train_ddp.py --config configs/example_gen_wind.yml --world_size 4
```

### 2. Bayesian Inference

Perform ABC inference with a trained model:

```python
from sampling import abc_inference
import torch

# Load trained model
model = GENGeomAutoencoder(**model_config)
model.load_state_dict(torch.load('path/to/model.pt')['model_state_dict'])
model.eval()

# Prepare observation data (partial observations with NaNs for unobserved nodes)
# data.x should have NaNs for the features of the unobserved nodes 

# Run ABC inference
result = abc_inference(
    model=model,
    data=observation_data,
    n_total_samples=100_000,
    n_accepted_samples=1000,
    sigma_tc=0.01,
    print_progress=True
)

# Access results
posterior_samples = result['z_posterior']  # Latent samples
decoded_fields = result['u_decoded']        # Decoded physical fields
best_fit = result['u_min_norm']            # MAP estimate
```

### 3. Multi-GPU Inference

For faster inference on multiple GPUs:

```python
from sampling import abc_inference_multigpu

result = abc_inference_multigpu(
    model=model,
    data=observation_data,
    n_total_samples=1_000_000,
    n_accepted_samples=5000,
    device_ids=[0, 1, 2, 3],  # Use 4 GPUs
    batch_size_per_gpu=100
)
```

## Package Structure

```
release/
├── datatools/          # Dataset classes
│   ├── dataset_wt.py       # Wind Terrain
│   ├── dataset_hr.py       # Heat Rectangle
│   ├── dataset_hc.py       # Helmholtz Car
│   ├── dataset_af.py       # Airfoil CFD
│   └── compute_ds_stats.py # Normalization utilities
├── models/             # GABI architectures
│   ├── gae_gcn.py         # GCN-based autoencoder
│   ├── gae_gen.py         # GEN-based autoencoder
│   ├── gae_transformer.py # Transformer autoencoder
│   └── stat.py            # MMD loss
├── training_scripts/   # Training utilities
│   ├── train_single_gpu.py
│   └── train_ddp.py
├── sampling/           # Bayesian inference
│   ├── abc_inference.py
│   ├── abc_inference_multigpu.py
│   └── mcmc_inference.py
├── utils/                  # Plotting and visualization
├── configs/                # Example configurations
└── data_preprocessing/     # # Data Preprocessing Scripts
```

## Model Architectures

### GCN (Graph Convolutional Network)
- Simple and efficient
- Good for small to medium-sized geometries
- Fast training and inference

### GENeralized Aggregation Networks
- Better expressiveness by using edge features
- Recommended for most applications

### Transformer
- Attention-based architecture
- Supports classic and Galerkin attention
- Requires batching nodes into dense format

## Configuration

All models can be configured via YAML files. Key parameters:

```yaml
model_settings:
  model_type: 'GEN'  # 'GCN', 'GEN', or 'Transformer'
  latent_dim: 128    # Hidden layer dimension
  z_dim: 16          # Latent space dimension
  n_layers: 6        # Number of layers (GCN/GEN only)
  use_boundary_encoding: True  # Include boundary type encoding
  use_pos: True     # Include node positions as features

  # Transformer-specific
  n_heads: 4         # Number of attention heads
  attn_type: 'classic'  # 'classic' or 'galerkin'

hyperparameters:
  batch_size: 32
  epochs: 500
  start_lr: 1e-3
  lr_decay: 0.9999
  weight_decay: 1e-4
```

##  Obtaining and Generating Data

You can easily download some of the datasets:
- airfoil RANS simulations from [here](https://zenodo.org/records/14629208). We further downsample these using the downsample_airfoil_data.py script
- Flow over terrain RANS simulations from [here](https://projects.asl.ethz.ch/datasets/doku.php?id=nature_2024_windseer)
- Car meshes from [here](https://zenodo.org/records/13737721)

Generate your own training data:

```bash
# Heat rectangle dataset
python data_preprocessing/gen_heat_rect_data.py --output data/heat_rect.pkl --n_samples 1000

# Helmholtz car dataset
python data_preprocessing/gen_car_helmholtz_data.py  --output data/helmholtz.pkl --n_samples 1000
```

## Citation

If you use GABI in your research, please cite our paper:

```bibtex
@article{gabi2025,
  title={Geometric Autoencoder Priors for Bayesian Inversion: Learn First Observe Later},
  author={Vadeboncoeur, Arnaud and Duth{\'e}, Gregory and Girolami, Mark and Chatzi, Eleni},
  journal={arXiv preprint arXiv:2509.19929},
  year={2025}
}

```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)
- MMD loss adapted from [mmd_loss_pytorch](https://github.com/yiftachbeer/mmd_loss_pytorch)
- Using data from the [ETH ASL Windseer project](https://projects.asl.ethz.ch/datasets/doku.php?id=nature_2024_windseer), by Achermann, Florian, et al. "WindSeer: real-time volumetric wind prediction over complex terrain aboard a small uncrewed aerial vehicle." Nature Communications 15.1 (2024): 3507. 

## Contact

For questions or issues, please open an issue on GitHub or contact [dutheg@ethz.ch](mailto:dutheg@ethz.ch).


### Contributing
Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

**Note**: This is research code. While we strive for correctness and usability, please validate results carefully for your specific application.
