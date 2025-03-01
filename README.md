# mpinn-mspinn-pytorch
A PyTorch implementation of mPINN &amp; MsPINN

- mPINN
  - https://www.sciencedirect.com/science/article/abs/pii/S0045782522005722
  - https://onlinelibrary.wiley.com/doi/full/10.1002/nme.7388
  - https://github.com/phyml4e/pinns
- MsPINN

# Usage

## Clone

Clone the `main` branch **(code only)**:

```sh
git clone -b main https://github.com/teitsurumi/mpinn-mspinn-pytorch.git
```

If the raw dataset is needed, please clone the `data` branch:

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

`requirements.txt` will not be offered. Please install the correct PyTorch version.

---

# Description

A

