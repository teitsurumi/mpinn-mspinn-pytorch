# mpinn-mspinn-pytorch
A PyTorch implementation of mPINN &amp; MsPINN

- mPINN
  - [Paper1](https://www.sciencedirect.com/science/article/abs/pii/S0045782522005722)
  - [Paper2](https://onlinelibrary.wiley.com/doi/full/10.1002/nme.7388)
  - [Code](https://github.com/phyml4e/pinns)
- MsPINN

# Usage

## Clone

Clone the `main` branch **(code only)**:

```sh
git clone -b main https://github.com/teitsurumi/mpinn-mspinn-pytorch.git
```

If the raw dataset is needed, please clone the `data` branch (about 18.1 M):

```sh
cd mpinn-mspinn-pytorch
git fetch origin data
git checkout origin/data -- data_thermoelastic
```

After downloading both branches, your project directory should look like this:

```
mpinn-mspinn-pytorch/
├── LICENSE
├── README.md
├── data_thermoelastic/  # Data from the `data` branch
├── utils/
│   └── <other files>
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

`requirements.txt` will not be offered. Please install the PyTorch version https://pytorch.org/ accordingly.

---

# Description

## Introduction

Our method leverages incomplete physics information during the initial stages to enhance convergence, particularly in scenarios where physical fields exhibit **local sharp gradients** . In such cases, unsupervised methods often converge to incorrect local minima. By gradually incorporating complete physics information in the final stages, our approach achieves more robust and accurate solutions.

Compared to the mPINN framework, which introduces correction terms (`_O` terms) to approximate higher-order equations and binds them to computed higher-order terms -- thereby **relaxing the constraints of higher-order equations** -- our framework offers improvements in efficiency and stability. Specifically, our method:
- Relaxes constraints using incomplete physics information during the initial stages to avoid numerical instability.
- Eliminates the need for correction terms, reducing the number of terms from 13 $\to$ 4 in 3-dimensional cases.
- Separates the `samplefile` (FEM samples), `trainfile` (collocation points) and `testfile` (test samples).

## Code optimization

We provide a PyTorch implementation with enhanced computational efficiency:
- Reorganized code structure
  - Streamlined configuration, sampling, and dataset handling
- Improved sampling & masking strategy (`test22`)
  - Replaced random sampling with Latin Hypercube Sampling (LHS) for more uniform and representative point distribution
  - Implemented mask generation using logical computations
  - for better readability :)
- Restructured the training pipeline

## Some compromise

Note that `test23` is not fully optimized. For better organized code, please refer to `test22`.

Due to time constraints, the current thermoelastic version does not incorporate more precise numerical integration methods. However, we have implemented advanced integration techniques in other versions of our framework:
- **Elastic version**: Utilized Delaunay integration for improved accuracy.
- <font color="#b2bec3">Upcoming</font> **hp-VPINN PyTorch version**: Developed Gaussian integration to further enhance precision.

We have also successfully optimized their computatin efficiency by reorganizing the code pipeline and data structure for model training.

---

# Acknowledgment

We would like to express our gratitude to the following open-source repositories:
- [PINN](https://github.com/maziarraissi/PINNs) (TF1 version. Currently they seem to have TF2, PyTorch and Jax versions.)
- [mPINN](https://github.com/phyml4e/pinns) (SciANN version)
- [DEM](https://github.com/MinhNguyenIKM/dem_hyperelasticity) (PyTorch version)

In this repository, we provide **a PyTorch implementation** and a **Multi-stage optimization** inspired by the above works.