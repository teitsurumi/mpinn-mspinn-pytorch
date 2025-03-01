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

## Description of the Numerical Experiments

### `test4_basic`

Before conducting this numerical experiment, we developed a PyTorch replication of PINN based on the [PINN (TF1 ver.)](https://github.com/maziarraissi/PINNs). We tested this model on two structures: an elastic circular hole and a square structure. During these tests, we observed that the original PINN struggled to converge under certain conditions:

1. When the Young's modulus ($E$) is very large, e.g., Steel<br> $\Rightarrow$ the order of magnititude of $u$, $\varepsilon$ and $\sigma$ differs greatly $\Rightarrow$ normalization

2. When there is localized high gradients<br> $\Rightarrow$ adjust the NN structure and training logics

<font color="#ff6b81">*Note:*</font> In 03/2024, Prof. Karniadakis' team introduced the Residual-Based Attention (RBA) mechanism to address this issue: [Residual-based attention in physics-informed neural networks](https://www.sciencedirect.com/science/article/pii/S0045782524000616).

Building on these insights, we designed `(test group 3) test4`. We discovered that by carefully grouping the loss terms, we could avoid numerical instability during the early stages of training. This approach ensures that all loss terms converge properly. Without proper grouping, the loss tends to get stuck in a suboptimal local minimum.

The `test4` experiment consists of two main components: `numerical_integration` and `test4`. The `numerical_integration` module prepares the necessary data for Delaunay integration, a key step in the process.

### Key Improvements:
- **Compared to the original PINN:** Our implementation achieves higher computational efficiency and better convergence.
- **Compared to DEM and some weak-form methods:** Our approach captures finer details in the physical fields, particularly for $\sigma_y$ and $\varepsilon_y$.
- Skipping pre-training stages can lead to visually rougher predictions in the final results.



---

# Acknowledgment

We would like to express our gratitude to the following open-source repositories:
- [PINN](https://github.com/maziarraissi/PINNs) (TF1 version. Currently they seem to have TF2, PyTorch and Jax versions.)
- [mPINN](https://github.com/phyml4e/pinns) (SciANN version)
- [DEM](https://github.com/MinhNguyenIKM/dem_hyperelasticity) (PyTorch version)

In this repository, we provide **a PyTorch implementation** and a **Multi-stage optimization** inspired by the above works.