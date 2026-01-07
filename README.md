
# SWSS-CD: Stability-Weighted Self-Supervised Change Detection for Hyperspectral Images

This repository provides the **testing code and pre-trained models** for the paper:

> **SWSS-CD: Stability-Weighted Self-Supervised Contrastive Learning for Hyperspectral Change Detection**  
> *Under review at IEEE Transactions on Image Processing (TIP)*

---

## Overview

Hyperspectral change detection (HSI-CD) aims to identify meaningful land-cover changes from bi-temporal hyperspectral imagery without reliable pixel-level annotations.  
SWSS-CD introduces a **stability-weighted self-supervised contrastive framework** that models spectral–temporal consistency as a continuous stability likelihood, enabling robust and label-free change detection across scenes.

**Key characteristics:**
- Fully unsupervised / self-supervised
- No change labels or hard pseudo labels
- No auxiliary reconstruction or segmentation branches
- Classifier-free and lightweight inference
- Strong cross-dataset generalization

---

## Method Highlights

- **Spectral Stability Modeling**  
  Cross-temporal spectral discrepancies are mapped to an exponential stability likelihood, providing a continuous and confidence-aware estimate of temporal stability.

- **Stability-Weighted Contrastive Learning**  
  A pull–push contrastive objective guided by stability likelihoods encourages compact representations for stable pixels and discriminative separation for changed ones.

- **Representation-Driven Inference**  
  Change maps are generated directly from bi-temporal feature discrepancies, without classification heads or post-processing modules.

---

## Repository Structure

```
SWSSCD/
├── models/                 # Model definitions
├── utils/                  # Metrics, visualization, and helper functions
├── test.py                 # Main testing script
└── README.md
```

---

## Installation

```bash
conda create -n swsscd python=3.9
conda activate swsscd
```

---

## Datasets

The framework is evaluated on public hyperspectral change detection benchmarks such as Farmland, Hermiston, and River.
Please follow the original dataset providers’ instructions to download the data and place them in the appropriate directories.
For the Farmland dataset, please download the Farmland.mat file and place it in the data/ directory. Note that the Farmland.mat file used in our experiments has a size of 25.3 MB; please ensure that the downloaded file matches this size, otherwise loading or evaluation errors may occur.

---

## Testing with Pre-trained Models

```bash
python test.py
```

The generated change maps and quantitative metrics will be saved automatically.

---

## Results

Experiments demonstrate that SWSS-CD achieves competitive or superior performance compared with statistical, unsupervised, self-supervised, transformer-based, and untrained-network-based methods, while maintaining strong cross-dataset generalization without fine-tuning.

---

## Citation

```bibtex
@article{SWSSCD2026,
  title   = {SWSS-CD: Stability-Weighted Self-Supervised Contrastive Learning for Hyperspectral Change Detection},
  author  = {Anonymous},
  journal = {IEEE Transactions on Image Processing},
  year    = {2026}
}
```

---

## License

This project is released for **academic research purposes only**.

---

## Contact

For questions or suggestions, please open an issue in this repository.
