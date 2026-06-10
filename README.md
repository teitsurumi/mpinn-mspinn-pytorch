# mpinn-mspinn-pytorch

A PyTorch implementation of mixed PINN (mPINN) and Multi-stage PINN (MsPINN)

# Features

- Dive into individual loss terms to reveal term entangling in tensor-type PDE systems
- Mechanism analysis: A simple Taylor-expansion-mocked PINN
- Development of multi-stage training
- Energy-based formulation for mechanical problems: PDE order reduction and convergence enhancement
- Thermo-mechanical inclusion problems: implementation of both mPINN & MsPINN
- Ablation studies
- Analysis from an optimization landscape perspective

**Key Features & Highlights:**

- **Pure PyTorch Implementation**
- **Fine-grained Loss Visualization:** Component-wise tracking and visualization of individual loss terms in tensor-type PDE systems to explicitly uncover gradient entanglement and conflicts.
- **Multi-stage Training Paradigm**
- Theoretical principles for relaxing higher-order constraints
  - Loss term coupling during training for well-posed BVPs
  - Methodological parallels between mPINN and MsPINN
- Optimization landscape visualization (tracking of singular value spectra and Frobenius norms during training)
- 2D energy computation via Delaunay triangulation integration

> [!WARNING]
> The main branch of this repo is currently being updated. The old version before our first revision can be found on the branch deprecated/main.

**@TODO**
- **Bugfixes & re-conceptualization:** The methodology is being updated based on recent reviews.
- **Software design:** A PySide6-based playground with GUI is being designed for debugging multi-stage training. The general framework is complete, but many details still require polishing.

**Related links**
- mixed PINN
  - [Paper1](https://www.sciencedirect.com/science/article/abs/pii/S0045782522005722)
  - [Paper2](https://onlinelibrary.wiley.com/doi/full/10.1002/nme.7388)
  - [Code](https://github.com/phyml4e/pinns)
- MsPINN
  - [Our paper](https://www.sciencedirect.com/science/article/abs/pii/S0045782525005250)

---

# Quick Start

## Clone

Clone the `main` branch **(code only)**:

```sh
git clone -b main https://github.com/teitsurumi/mpinn-mspinn-pytorch.git
cd mpinn-mspinn-pytorch
```

If the raw dataset is needed, please clone the `data` branch (about 18.1 Mb):

```sh
git fetch origin data
git checkout origin/data -- data_thermoelastic
```

(Another dataset containing 110,323 nodes is too large for this repo; it will be made available upon request.)

After downloading both branches, your project directory should look like this:

```
mpinn-mspinn-pytorch/
├── LICENSE
├── README.md
├── data_thermoelastic/  # Data from the `data` branch
├── utils/
│   └── <other files>
├── ax_xxx.py (Python file: appendix)
├── tx_xxx.py (Python file: test cases)
└── <other files>
```

## Requirements

- CUDA 11.8 & 12.4 ; cudnn
- Python 3.12.8 (conda version)
- torch 2.5.1+cu124
- numpy 1.26.3
- pyDOE3 1.0.4
- matplotlib 3.10.0
- h5py 3.13.0
- pandas 2.2.3

The `requirements.txt` file is not provided here due to varying CUDA support. Please install the appropriate version of PyTorch from https://pytorch.org/ based on your system configuration.

---

# 1. Diving into individual loss terms

@TODO

# 2. Mechanism analysis: A simple Taylor-expansion-mocked PINN

Run the script: `t2_mechanism_analysis_taylor.py`

By comparing the first 1,500 training steps between the `order2` and `order3` configurations, a clear discrepancy emerges. Although a higher-order Taylor expansion mathematically yields a superior approximation over a given interval, the `order3` case proves significantly more difficult to optimize numerically.

Visualizing each individual loss component within the `order3` total loss reveals the underlying dynamics:
- Higher-order derivative terms themselves are harder to converge.
- Higher-order terms disrupt the convergence of other loss components.

This observation highlights a challenge in multi-objective optimization for PINNs: loss term balancing. Actually, [Wang et al., 2022](https://www.sciencedirect.com/science/article/pii/S002199912100663X) analyzed this using the NTK, and [Anagnostopoulos et al., 2024](https://www.sciencedirect.com/science/article/pii/S0045782524000616) proposed Residual-based Attention for pointwise weighting. However, these methods do not decompose the physics loss into individual terms, nor do they account for the differential order of the loss terms.

---

# Acknowledgment

We would like to express our gratitude to the following open-source repositories:
- [PINN](https://github.com/maziarraissi/PINNs) (TF1 version. Currently they seem to have TF2, PyTorch and Jax versions.)
- [mixed PINN](https://github.com/phyml4e/pinns) (SciANN version)
- [DEM](https://github.com/MinhNguyenIKM/dem_hyperelasticity) (PyTorch version)

In this repository, we provide **a PyTorch implementation** and a **Multi-stage optimization** inspired by the above works.