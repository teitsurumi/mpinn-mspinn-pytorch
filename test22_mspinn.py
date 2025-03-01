import numpy as np
import torch
from torch import nn
from utils.nn_fixed import Functional
from pyDOE3 import lhs
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd

import itertools
from typing import List, Dict
import pickle
import h5py
import time


############################### INIT STATE ###############################
torch.manual_seed(12345)
np.random.seed(12345)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

############################### UTILS ###############################
def gradients(outputs, inputs):
    return torch.autograd.grad(outputs, inputs, grad_outputs=torch.ones_like(outputs), allow_unused=True, create_graph=True)

def to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or ' \
                        'np.ndarray, but got {}'.format(type(input)))

def to_tensor(input, dtype=torch.float32, requires_grad=False, device=None):
    if isinstance(input, torch.Tensor):
        if device == None:  input = input.to(dtype=dtype)
        else:               input = input.to(device=device, dtype=dtype)
        input.requires_grad_(requires_grad)
        return input
    elif isinstance(input, np.ndarray):
        if device == None:
            return torch.tensor(input, dtype=dtype, requires_grad=requires_grad)
        else:
            return torch.tensor(input, dtype=dtype, requires_grad=requires_grad, device=device)
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or ' \
                        'np.ndarray, but got {}'.format(type(input)))


############################### CONFIG & CONSTANT ###############################
class Config:
    name: str = ""
    idx:  str = str(0).rjust(3, "0")
    samplefile: str  = 'data_thermoelastic/3d,b-i=10-1,03-refined.h5'  # FEM samples
    trainfile: str = 'data_thermoelastic/3d,b-i=10-1,03-interpolated.h5'  # same as samplefile | a new mesh file | a pixel-based grid
    testfile: str  = 'data_thermoelastic/3d,b-i=10-1,03-interpolated.h5'  # same as trainfile (best physical reliability) | a new mesh file | a pixel-based grid
    input_dim: int  = 3
    output_dim: int = 1
    hidden_layers: List[int] = [64, 64, 64, 64, 64]
    activation: str = "sin"
    iters: int = 25000
    loss_terms: List[str] = [
        'Total_loss', 'Work_m', 'BC1_m', 'BC2_m', 'BC3_m', 'BC4_m', 'BC5_m', 'BC6_m', 'BC7_m', 'BC8_m', 'BC9_m', 'BC10_m', 'BC11_m', 'BC12_m', 
        'Work_t', 'BC1_t', 'BC2_t', 'BC3_t', 'BC4_t'
    ]
cfg = Config()

nu = 0.3
alpha = 0.3
T_0 = 0



############################### SAMPLING ###############################
class FESample:
    """
    :param sample_mode: "random": 0~1 | "linear": int | "None": xxx
    """
    def __init__(
        self,
        x_FE: np.ndarray, y_FE: np.ndarray, z_FE: np.ndarray, e_FE: np.ndarray, k_FE: np.ndarray,
        u_x_FE: np.ndarray, u_y_FE: np.ndarray, u_z_FE: np.ndarray,
        sigma_x_FE: np.ndarray, sigma_y_FE: np.ndarray, sigma_z_FE: np.ndarray, sigma_xy_FE: np.ndarray, sigma_yz_FE: np.ndarray, sigma_xz_FE: np.ndarray,
        T_FE: np.ndarray, q_x_FE: np.ndarray, q_y_FE: np.ndarray, q_z_FE: np.ndarray,
        sample_mode={"random": 0.05}, dtype=np.float32
    ):
        self.x_FE = x_FE.astype(dtype=dtype)
        self.y_FE = y_FE.astype(dtype=dtype)
        self.z_FE = z_FE.astype(dtype=dtype)
        self.e_FE = e_FE.astype(dtype=dtype)
        self.k_FE = k_FE.astype(dtype=dtype)
        self.u_x_FE = u_x_FE.astype(dtype=dtype)
        self.u_y_FE = u_y_FE.astype(dtype=dtype)
        self.u_z_FE = u_z_FE.astype(dtype=dtype)
        self.sigma_x_FE = sigma_x_FE.astype(dtype=dtype)
        self.sigma_y_FE = sigma_y_FE.astype(dtype=dtype)
        self.sigma_z_FE = sigma_z_FE.astype(dtype=dtype)
        self.sigma_xy_FE = sigma_xy_FE.astype(dtype=dtype)
        self.sigma_yz_FE = sigma_yz_FE.astype(dtype=dtype)
        self.sigma_xz_FE = sigma_xz_FE.astype(dtype=dtype)
        self.T_FE = T_FE.astype(dtype=dtype)
        self.q_x_FE = q_x_FE.astype(dtype=dtype)
        self.q_y_FE = q_y_FE.astype(dtype=dtype)
        self.q_z_FE = q_z_FE.astype(dtype=dtype)
        if len(sample_mode) != 1:
            raise ValueError("Multiple sample_modes")
        key, val = next(iter(sample_mode.items()))
        if key == "random":
            samples = int(float(val) * x_FE.shape[0])
            self.__random_sampling(samples)
        elif key == "linear":
            step = int(val)
            self.__linear_sampling(step)
        elif key == "None":
            pass
        else:
            raise ValueError(f"{key} unavailable")

    def __random_sampling(self, samples):
        """Random sampling
        
        :param samples: number of samples
        """
        max_samples = self.x_FE.shape[0]
        if samples > max_samples:
            raise ValueError(f"Cannot sample more than the population size ({max_samples})")
        indices = torch.randperm(max_samples)[:samples]
        self.__apply_indices(indices)

    def __linear_sampling(self, step):
        """Linear sampling
        
        :param step: linear sampling step
        """
        indices = torch.arange(0, self.x_FE.shape[0], step)
        self.__apply_indices(indices)

    def __apply_indices(self, indices):
        """
        :param indices: sampled indices
        """
        self.x_FE = self.x_FE[indices]
        self.y_FE = self.y_FE[indices]
        self.e_FE = self.e_FE[indices]
        self.k_FE = self.k_FE[indices]
        self.u_x_FE = self.u_x_FE[indices]
        self.u_y_FE = self.u_y_FE[indices]
        self.sigma_x_FE = self.sigma_x_FE[indices]
        self.sigma_y_FE = self.sigma_y_FE[indices]
        self.sigma_xy_FE = self.sigma_xy_FE[indices]
        self.T_FE = self.T_FE[indices]
        self.q_x_FE = self.q_x_FE[indices]
        self.q_y_FE = self.q_y_FE[indices]

# -------------------------- Apply Sampling --------------------------
def print_datasets(name, obj):
    if isinstance(obj, h5py.Dataset):  # check dataset name
        print(name)
with h5py.File(cfg.samplefile, 'r') as hf:
    # hf.visititems(print_datasets)
    x_FE = hf['X'][:].astype(np.float32)
    y_FE = hf['Y'][:].astype(np.float32)
    z_FE = hf['Z'][:].astype(np.float32)
    e_FE = hf['solid.E (Pa)'][:].astype(np.float32)
    k_FE = hf['ht.kxx (W/(m*K))'][:].astype(np.float32)
    u_x_FE = hf['u (m)'][:].astype(np.float32)
    u_y_FE = hf['v (m)'][:].astype(np.float32)
    u_z_FE = hf['w (m)'][:].astype(np.float32)
    sigma_x_FE = hf['solid.sxx (N/m^2)'][:].astype(np.float32)
    sigma_y_FE = hf['solid.syy (N/m^2)'][:].astype(np.float32)
    sigma_z_FE = hf['solid.szz (N/m^2)'][:].astype(np.float32)
    sigma_xy_FE = hf['solid.sxy (N/m^2)'][:].astype(np.float32)
    sigma_yz_FE = hf['solid.syz (N/m^2)'][:].astype(np.float32)
    sigma_xz_FE = hf['solid.sxz (N/m^2)'][:].astype(np.float32)
    T_FE = hf['T (K)'][:].astype(np.float32)
    q_x_FE = hf['ht.tfluxx (W/m^2)'][:].astype(np.float32)
    q_y_FE = hf['ht.tfluxy (W/m^2)'][:].astype(np.float32)
    q_z_FE = hf['ht.tfluxz (W/m^2)'][:].astype(np.float32)

x_FE = x_FE.reshape((-1, 1))
y_FE = y_FE.reshape((-1, 1))
z_FE = z_FE.reshape((-1, 1))
e_FE = e_FE.reshape((-1, 1))
k_FE = k_FE.reshape((-1, 1))
u_x_FE = u_x_FE.reshape((-1, 1))
u_y_FE = u_y_FE.reshape((-1, 1))
u_z_FE = u_z_FE.reshape((-1, 1))
sigma_x_FE = sigma_x_FE.reshape((-1, 1))
sigma_y_FE = sigma_y_FE.reshape((-1, 1))
sigma_z_FE = sigma_z_FE.reshape((-1, 1))
sigma_xy_FE = sigma_xy_FE.reshape((-1, 1))
sigma_yz_FE = sigma_yz_FE.reshape((-1, 1))
sigma_xz_FE = sigma_xz_FE.reshape((-1, 1))
T_FE = T_FE.reshape((-1, 1))
q_x_FE = q_x_FE.reshape((-1, 1))
q_y_FE = q_y_FE.reshape((-1, 1))
q_z_FE = q_z_FE.reshape((-1, 1))

sample_FE = FESample(
    x_FE, y_FE, z_FE, e_FE, k_FE, u_x_FE, u_y_FE, u_z_FE, sigma_x_FE, sigma_y_FE, sigma_z_FE, sigma_xy_FE, sigma_yz_FE, sigma_xz_FE,
    T_FE, q_x_FE, q_y_FE, q_z_FE,
    sample_mode={"None": 0.9}
)



class TrainingUtils(nn.Module):
    def __init__(
        self, models: Dict[str, Functional],
        x_train: torch.Tensor, y_train: torch.Tensor, z_train: torch.Tensor,
        nb: float, nb_l: float, nb_r: float, nb_f: float, nb_bh: float, nb_t: float, nb_bt: float, nd: float,
        M_domain: torch.Tensor, M_bound_left: torch.Tensor, M_bound_right: torch.Tensor, M_bound_front: torch.Tensor, M_bound_behind: torch.Tensor, M_bound_bottom: torch.Tensor, M_bound_top: torch.Tensor,
        E: torch.Tensor, K: torch.Tensor,
        cfg: Config, sample_FE: FESample
    ):
        super(TrainingUtils, self).__init__()
        self.Tn       = models["Tn"].to(device)
        self.q_x_O    = models["q_x_O"].to(device)
        self.q_y_O    = models["q_y_O"].to(device)
        self.q_z_O    = models["q_z_O"].to(device)
        self.u_xn     = models["u_xn"].to(device)
        self.u_yn     = models["u_yn"].to(device)
        self.u_zn     = models["u_zn"].to(device)
        self.sig_x_O  = models["sig_x_O"].to(device)
        self.sig_y_O  = models["sig_y_O"].to(device)
        self.sig_z_O  = models["sig_z_O"].to(device)
        self.sig_xy_O = models["sig_xy_O"].to(device)
        self.sig_yz_O = models["sig_yz_O"].to(device)
        self.sig_xz_O = models["sig_xz_O"].to(device)
        self.sample_FE = sample_FE
        
        self.x_fem   = torch.tensor(self.sample_FE.x_FE, dtype=torch.float32, device=device, requires_grad=True).reshape((-1, 1))
        self.y_fem   = torch.tensor(self.sample_FE.y_FE, dtype=torch.float32, device=device, requires_grad=True).reshape((-1, 1))
        self.z_fem   = torch.tensor(self.sample_FE.z_FE, dtype=torch.float32, device=device, requires_grad=True).reshape((-1, 1))
        self.x_train = x_train.to(device)
        self.y_train = y_train.to(device)
        self.z_train = z_train.to(device)
        
        self.E = E.to(device)
        self.K = K.to(device)
        
        self.nb = nb; self.nd = nd;
        self.nb_l = nb_l; self.nb_r = nb_r; self.nb_f = nb_f; self.nb_bh = nb_bh
        self.nb_t = nb_t; self.nb_bt = nb_bt
        self.M_domain = M_domain.to(device)
        self.M_bound_left = M_bound_left.to(device);     self.M_bound_right = M_bound_right.to(device)
        self.M_bound_front = M_bound_front.to(device);   self.M_bound_behind = M_bound_behind.to(device)
        self.M_bound_bottom = M_bound_bottom.to(device); self.M_bound_top = M_bound_top.to(device)
        
        self.cfg = cfg
        self.history = torch.zeros((len(cfg.loss_terms), cfg.iters))
        self.mse = nn.MSELoss()
        
        # FE data
        self.E_FE = torch.tensor(self.sample_FE.e_FE, dtype=torch.float32, device=device).reshape((-1, 1))
        self.K_FE = torch.tensor(self.sample_FE.k_FE, dtype=torch.float32, device=device).reshape((-1, 1))
        self.u_x_FE = torch.tensor(self.sample_FE.u_x_FE, dtype=torch.float32, device=device).reshape((-1, 1))
        self.u_y_FE = torch.tensor(self.sample_FE.u_y_FE, dtype=torch.float32, device=device).reshape((-1, 1))
        self.u_z_FE = torch.tensor(self.sample_FE.u_z_FE, dtype=torch.float32, device=device).reshape((-1, 1))
        self.sigma_x_FE = torch.tensor(self.sample_FE.sigma_x_FE, dtype=torch.float32, device=device).reshape((-1, 1))
        self.sigma_y_FE = torch.tensor(self.sample_FE.sigma_y_FE, dtype=torch.float32, device=device).reshape((-1, 1))
        self.sigma_z_FE = torch.tensor(self.sample_FE.sigma_z_FE, dtype=torch.float32, device=device).reshape((-1, 1))
        self.sigma_xy_FE = torch.tensor(self.sample_FE.sigma_xy_FE, dtype=torch.float32, device=device).reshape((-1, 1))
        self.sigma_yz_FE = torch.tensor(self.sample_FE.sigma_yz_FE, dtype=torch.float32, device=device).reshape((-1, 1))
        self.sigma_xz_FE = torch.tensor(self.sample_FE.sigma_xz_FE, dtype=torch.float32, device=device).reshape((-1, 1))
        self.T_FE = torch.tensor(self.sample_FE.T_FE, dtype=torch.float32, device=device).reshape((-1, 1))
        self.q_x_FE = torch.tensor(self.sample_FE.q_x_FE, dtype=torch.float32, device=device).reshape((-1, 1))
        self.q_y_FE = torch.tensor(self.sample_FE.q_y_FE, dtype=torch.float32, device=device).reshape((-1, 1))
        self.q_z_FE = torch.tensor(self.sample_FE.q_z_FE, dtype=torch.float32, device=device).reshape((-1, 1))
    
    def func1(self, x, y, z, E, K):
        X = torch.hstack((x, y, z))
        Tn = self.Tn(X)
        T = Tn * x * (x - 1.0) + 1.0 - x
        gr_T_x = gradients(T, x)[0]
        gr_T_y = gradients(T, y)[0]
        gr_T_z = gradients(T, z)[0]
        q_x = (-K) * gr_T_x
        q_y = (-K) * gr_T_y
        q_z = (-K) * gr_T_z
        
        u_xn = self.u_xn(X)
        u_yn = self.u_yn(X)
        u_zn = self.u_zn(X)
        u_x = u_xn * x * (x - 1.0)
        u_y = u_yn * y * (y - 1.0)
        u_z = u_zn * z * (z - 1.0)
        strain_x  = gradients(u_x, x)[0]
        strain_y  = gradients(u_y, y)[0]
        strain_z  = gradients(u_z, z)[0]
        strain_xy = (gradients(u_x, y)[0] + gradients(u_y, x)[0]) * 0.5
        strain_yz = (gradients(u_y, z)[0] + gradients(u_z, y)[0]) * 0.5
        strain_xz = (gradients(u_x, z)[0] + gradients(u_z, x)[0]) * 0.5
        strain_T_x = (T - T_0) * alpha
        strain_T_y = (T - T_0) * alpha
        strain_T_z = (T - T_0) * alpha
        sigma_x   = (E/((1+nu)*(1-2*nu))) * ((1-nu)*strain_x + nu*(strain_y+strain_z)) - E*strain_T_x/(1-2*nu)
        sigma_y   = (E/((1+nu)*(1-2*nu))) * ((1-nu)*strain_y + nu*(strain_x+strain_z)) - E*strain_T_y/(1-2*nu)
        sigma_z   = (E/((1+nu)*(1-2*nu))) * ((1-nu)*strain_z + nu*(strain_x+strain_y)) - E*strain_T_z/(1-2*nu)
        sigma_xy  = (E/(1+nu)) * strain_xy
        sigma_yz  = (E/(1+nu)) * strain_yz
        sigma_xz  = (E/(1+nu)) * strain_xz
        
        ###################################### CONSTRUCT LOSS TERMS ######################################
        ################################# Mechanical Governing Equation #################################
        Work_int_m = torch.sum(self.M_domain * (sigma_x*strain_x + sigma_y*strain_y + sigma_z*strain_z + 2*sigma_xy*strain_xy + 2*sigma_yz*strain_yz + 2*sigma_xz*strain_xz))/(2*self.nd)
        Work_e_l_m = torch.sum(self.M_bound_left*(sigma_x * u_x + sigma_xy * u_y + sigma_xz * u_z))/self.nb_l
        Work_e_r_m = torch.sum(self.M_bound_right*(-sigma_x * u_x - sigma_xy * u_y - sigma_xz * u_z))/self.nb_r
        Work_e_f_m = torch.sum(self.M_bound_front*(sigma_y * u_y + sigma_xy * u_x + sigma_yz * u_z))/self.nb_f
        Work_e_bh_m = torch.sum(self.M_bound_behind*(sigma_y * u_y + sigma_xy * u_x - sigma_yz * u_z))/self.nb_bh
        Work_e_t_m = torch.sum(self.M_bound_top*(sigma_z * u_z + sigma_xz * u_x + sigma_yz * u_y))/self.nb_t
        Work_e_bt_m = torch.sum(self.M_bound_bottom*(-sigma_z * u_z - sigma_xz * u_x - sigma_yz * u_y))/self.nb_bt
        Work_ext_m = Work_e_l_m + Work_e_r_m + Work_e_f_m + Work_e_bh_m + Work_e_t_m + Work_e_bt_m
        Work_m = torch.square(Work_int_m - Work_ext_m)
        
        BC1_m = self.M_bound_left*sigma_xy; BC2_m = self.M_bound_left*sigma_xz
        BC3_m = self.M_bound_right*sigma_xy; BC4_m = self.M_bound_right*sigma_xz
        BC5_m = self.M_bound_front*sigma_xy; BC6_m = self.M_bound_front*sigma_yz
        BC7_m = self.M_bound_behind*sigma_xy; BC8_m = self.M_bound_behind*sigma_yz
        BC9_m = self.M_bound_bottom*sigma_yz; BC10_m = self.M_bound_bottom*sigma_xz
        BC11_m = self.M_bound_top*sigma_yz; BC12_m = self.M_bound_top*sigma_xz

        ################################## Thermal Governing Equation ##################################
        Work_int_t = torch.sum(self.M_domain * (q_x*gr_T_x + q_y*gr_T_y + q_z*gr_T_z))/self.nd
        Work_e_l_t = torch.sum(self.M_bound_left * (-q_x * T))/self.nb_l
        Work_t = torch.square(Work_int_t - Work_e_l_t)
        
        BC1_t = self.M_bound_front*q_y; BC2_t = self.M_bound_behind*q_y
        BC3_t = self.M_bound_bottom*q_z; BC4_t = self.M_bound_top*q_z
        
        return Work_m, BC1_m, BC2_m, BC3_m, BC4_m, BC5_m, BC6_m, BC7_m, BC8_m, BC9_m, BC10_m, BC11_m, BC12_m, \
            Work_t, BC1_t, BC2_t, BC3_t, BC4_t
    def func1_FE(self, x, y, z, E, K):
        X = torch.hstack((x, y, z))
        Tn = self.Tn(X)
        T = Tn * x * (x - 1.0) + 1.0 - x
        gr_T_x = gradients(T, x)[0]
        gr_T_y = gradients(T, y)[0]
        gr_T_z = gradients(T, z)[0]
        q_x = (-K) * gr_T_x
        q_y = (-K) * gr_T_y
        q_z = (-K) * gr_T_z
        
        u_xn = self.u_xn(X)
        u_yn = self.u_yn(X)
        u_zn = self.u_zn(X)
        u_x = u_xn * x * (x - 1.0)
        u_y = u_yn * y * (y - 1.0)
        u_z = u_zn * z * (z - 1.0)
        strain_x  = gradients(u_x, x)[0]
        strain_y  = gradients(u_y, y)[0]
        strain_z  = gradients(u_z, z)[0]
        strain_xy = (gradients(u_x, y)[0] + gradients(u_y, x)[0]) * 0.5
        strain_yz = (gradients(u_y, z)[0] + gradients(u_z, y)[0]) * 0.5
        strain_xz = (gradients(u_x, z)[0] + gradients(u_z, x)[0]) * 0.5
        strain_T_x = (T - T_0) * alpha
        strain_T_y = (T - T_0) * alpha
        strain_T_z = (T - T_0) * alpha
        sigma_x   = (E/((1+nu)*(1-2*nu))) * ((1-nu)*strain_x + nu*(strain_y+strain_z)) - E*strain_T_x/(1-2*nu)
        sigma_y   = (E/((1+nu)*(1-2*nu))) * ((1-nu)*strain_y + nu*(strain_x+strain_z)) - E*strain_T_y/(1-2*nu)
        sigma_z   = (E/((1+nu)*(1-2*nu))) * ((1-nu)*strain_z + nu*(strain_x+strain_y)) - E*strain_T_z/(1-2*nu)
        sigma_xy  = (E/(1+nu)) * strain_xy
        sigma_yz  = (E/(1+nu)) * strain_yz
        sigma_xz  = (E/(1+nu)) * strain_xz
        return u_x, u_y, u_z, sigma_x, sigma_y, sigma_z, sigma_xy, sigma_yz, sigma_xz, \
            T, q_x, q_y, q_z
    def func1_train(self):
        optimizer = torch.optim.Adam(itertools.chain(self.Tn.parameters(), self.u_xn.parameters(), self.u_yn.parameters(), self.u_zn.parameters()), lr=4e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=2500, gamma=0.5)
        for i in range(self.cfg.iters):
            optimizer.zero_grad()
            Work_m, BC1_m, BC2_m, BC3_m, BC4_m, BC5_m, BC6_m, BC7_m, BC8_m, BC9_m, BC10_m, BC11_m, BC12_m, \
                Work_t, BC1_t, BC2_t, BC3_t, BC4_t = self.func1(self.x_train, self.y_train, self.z_train, self.E, self.K)
            u_x_FE_, u_y_FE_, u_z_FE_, sigma_x_FE_, sigma_y_FE_, sigma_z_FE_, sigma_xy_FE_, sigma_yz_FE_, sigma_xz_FE_, \
                T_FE_, q_x_FE_, q_y_FE_, q_z_FE_ = self.func1_FE(self.x_fem, self.y_fem, self.z_fem, self.E_FE, self.K_FE)
            # FE Loss
            ls_u_x_FE = self.mse(u_x_FE_, self.u_x_FE)
            ls_u_y_FE = self.mse(u_y_FE_, self.u_y_FE)
            ls_u_z_FE = self.mse(u_z_FE_, self.u_z_FE)
            ls_sigma_x_FE = self.mse(sigma_x_FE_, self.sigma_x_FE)
            ls_sigma_y_FE = self.mse(sigma_y_FE_, self.sigma_y_FE)
            ls_sigma_z_FE = self.mse(sigma_z_FE_, self.sigma_z_FE)
            ls_sigma_xy_FE = self.mse(sigma_xy_FE_, self.sigma_xy_FE)
            ls_sigma_yz_FE = self.mse(sigma_yz_FE_, self.sigma_yz_FE)
            ls_sigma_xz_FE = self.mse(sigma_xz_FE_, self.sigma_xz_FE)
            ls_T_FE = self.mse(T_FE_, self.T_FE)
            ls_q_x_FE = self.mse(q_x_FE_, self.q_x_FE)
            ls_q_y_FE = self.mse(q_y_FE_, self.q_y_FE)
            ls_q_z_FE = self.mse(q_z_FE_, self.q_z_FE)
            # Eq Loss
            """loss_terms: List[str] = [
                'Total_loss', 'Work_m', 'BC1_m', 'BC2_m', 'BC3_m', 'BC4_m', 'BC5_m', 'BC6_m', 'BC7_m', 'BC8_m', 'BC9_m', 'BC10_m', 'BC11_m', 'BC12_m', 
                'Work_t', 'BC1_t', 'BC2_t', 'BC3_t', 'BC4_t'
            ]"""
            ls_Work_m = Work_m;                           self.history[1, i]  = ls_Work_m.item()
            ls_BC1_m  = torch.mean(torch.square(BC1_m));  self.history[2, i]  = ls_BC1_m.item()
            ls_BC2_m  = torch.mean(torch.square(BC2_m));  self.history[3, i]  = ls_BC2_m.item()
            ls_BC3_m  = torch.mean(torch.square(BC3_m));  self.history[4, i]  = ls_BC3_m.item()
            ls_BC4_m  = torch.mean(torch.square(BC4_m));  self.history[5, i]  = ls_BC4_m.item()
            ls_BC5_m  = torch.mean(torch.square(BC5_m));  self.history[6, i]  = ls_BC5_m.item()
            ls_BC6_m  = torch.mean(torch.square(BC6_m));  self.history[7, i]  = ls_BC6_m.item()
            ls_BC7_m  = torch.mean(torch.square(BC7_m));  self.history[8, i]  = ls_BC7_m.item()
            ls_BC8_m  = torch.mean(torch.square(BC8_m));  self.history[9, i]  = ls_BC8_m.item()
            ls_BC9_m  = torch.mean(torch.square(BC9_m));  self.history[10, i] = ls_BC9_m.item()
            ls_BC10_m = torch.mean(torch.square(BC10_m)); self.history[11, i] = ls_BC10_m.item()
            ls_BC11_m = torch.mean(torch.square(BC11_m)); self.history[12, i] = ls_BC11_m.item()
            ls_BC12_m = torch.mean(torch.square(BC12_m)); self.history[13, i] = ls_BC12_m.item()
            ls_Work_t = Work_t;                           self.history[14, i] = ls_Work_t.item()
            ls_BC1_t = torch.mean(torch.square(BC1_t));   self.history[15, i] = ls_BC1_t.item()
            ls_BC2_t = torch.mean(torch.square(BC2_t));   self.history[16, i] = ls_BC2_t.item()
            ls_BC3_t = torch.mean(torch.square(BC3_t));   self.history[17, i] = ls_BC3_t.item()
            ls_BC4_t = torch.mean(torch.square(BC4_t));   self.history[18, i] = ls_BC4_t.item()
            
            if i <= 1000:
                Loss = ls_Work_m + ls_BC1_m + ls_BC2_m + ls_BC3_m + ls_BC4_m + ls_BC5_m + ls_BC6_m + ls_BC7_m + ls_BC8_m + ls_BC9_m + ls_BC10_m + ls_BC11_m + ls_BC12_m + \
                    ls_Work_t + ls_BC1_t + ls_BC2_t + ls_BC3_t + ls_BC4_t + \
                    ls_u_x_FE + ls_u_y_FE + ls_u_z_FE + ls_sigma_x_FE + ls_sigma_y_FE + ls_sigma_z_FE + ls_sigma_xy_FE + ls_sigma_yz_FE + ls_sigma_xz_FE + ls_T_FE + ls_q_x_FE + ls_q_y_FE + ls_q_z_FE
            elif i <= 6000:
                Loss = ls_u_x_FE + ls_u_y_FE + ls_u_z_FE + ls_sigma_x_FE + ls_sigma_y_FE + ls_sigma_z_FE + ls_sigma_xy_FE + ls_sigma_yz_FE + ls_sigma_xz_FE + ls_T_FE + ls_q_x_FE + ls_q_y_FE + ls_q_z_FE
            elif i <= 10000:
                Loss = ls_Work_m + ls_BC1_m + ls_BC2_m + ls_BC3_m + ls_BC4_m + ls_BC5_m + ls_BC6_m + ls_BC7_m + ls_BC8_m + ls_BC9_m + ls_BC10_m + ls_BC11_m + ls_BC12_m + \
                    ls_Work_t + ls_BC1_t + ls_BC2_t + ls_BC3_t + ls_BC4_t + \
                    ls_u_x_FE + ls_u_y_FE + ls_u_z_FE + ls_sigma_x_FE + ls_sigma_y_FE + ls_sigma_z_FE + ls_sigma_xy_FE + ls_sigma_yz_FE + ls_sigma_xz_FE + ls_T_FE + ls_q_x_FE + ls_q_y_FE + ls_q_z_FE
            elif i <= 14000:
                if i == 10000+1:
                    new_lr = 5e-4
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                Loss = ls_u_x_FE + ls_u_y_FE + ls_u_z_FE + ls_sigma_x_FE + ls_sigma_y_FE + ls_sigma_z_FE + ls_sigma_xy_FE + ls_sigma_yz_FE + ls_sigma_xz_FE + ls_T_FE + ls_q_x_FE + ls_q_y_FE + ls_q_z_FE
            else:
                if i == 16000+1:
                    new_lr = 6e-4
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                if i == 22000+1:
                    new_lr = 1e-4
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                Loss = ls_Work_m + ls_BC1_m + ls_BC2_m + ls_BC3_m + ls_BC4_m + ls_BC5_m + ls_BC6_m + ls_BC7_m + ls_BC8_m + ls_BC9_m + ls_BC10_m + ls_BC11_m + ls_BC12_m + \
                    ls_Work_t + ls_BC1_t + ls_BC2_t + ls_BC3_t + ls_BC4_t + \
                    ls_u_x_FE + ls_u_y_FE + ls_u_z_FE + ls_sigma_x_FE + ls_sigma_y_FE + ls_sigma_z_FE + ls_sigma_xy_FE + ls_sigma_yz_FE + ls_sigma_xz_FE + ls_T_FE + ls_q_x_FE + ls_q_y_FE + ls_q_z_FE
            Loss.backward()
            optimizer.step()
            # if i >= 4000 and i <= 20000: scheduler.step()
            self.history[0, i] = Loss.item()
            
            if (i + 1) % 100 == 0:
                print("iter:", i+1, " Loss:", Loss)
                print("  ".join("{}: {:.6e}".format(a_elem, b_elem) for a_elem, b_elem in zip(cfg.loss_terms, self.history[:, i].flatten().detach().cpu().tolist())))

################################## SETUP ##################################
# -------------------------- NN --------------------------
T_n   = Functional(cfg.input_dim, cfg.output_dim, cfg.hidden_layers, cfg.activation)
q_x_O = Functional(cfg.input_dim, cfg.output_dim, cfg.hidden_layers, cfg.activation)  # _O unused in ms-pinn
q_y_O = Functional(cfg.input_dim, cfg.output_dim, cfg.hidden_layers, cfg.activation)
q_z_O = Functional(cfg.input_dim, cfg.output_dim, cfg.hidden_layers, cfg.activation)

u_xn  = Functional(cfg.input_dim, cfg.output_dim, cfg.hidden_layers, cfg.activation)
u_yn  = Functional(cfg.input_dim, cfg.output_dim, cfg.hidden_layers, cfg.activation)
u_zn  = Functional(cfg.input_dim, cfg.output_dim, cfg.hidden_layers, cfg.activation)

sig_x_O  = Functional(cfg.input_dim, cfg.output_dim, cfg.hidden_layers, cfg.activation)
sig_y_O  = Functional(cfg.input_dim, cfg.output_dim, cfg.hidden_layers, cfg.activation)
sig_z_O  = Functional(cfg.input_dim, cfg.output_dim, cfg.hidden_layers, cfg.activation)
sig_xy_O = Functional(cfg.input_dim, cfg.output_dim, cfg.hidden_layers, cfg.activation)
sig_yz_O = Functional(cfg.input_dim, cfg.output_dim, cfg.hidden_layers, cfg.activation)
sig_xz_O = Functional(cfg.input_dim, cfg.output_dim, cfg.hidden_layers, cfg.activation)

models = dict()
models["Tn"] = T_n
models["q_x_O"] = q_x_O
models["q_y_O"] = q_y_O
models["q_z_O"] = q_z_O
models["u_xn"] = u_xn
models["u_yn"] = u_yn
models["u_zn"] = u_zn
models["sig_x_O"] = sig_x_O
models["sig_y_O"] = sig_y_O
models["sig_z_O"] = sig_z_O
models["sig_xy_O"] = sig_xy_O
models["sig_yz_O"] = sig_yz_O
models["sig_xz_O"] = sig_xz_O


# -------------------------- Data --------------------------
with h5py.File(cfg.trainfile, 'r') as hf:
    x1 = hf['X'][:].astype(np.float32)
    y1 = hf['Y'][:].astype(np.float32)
    z1 = hf['Z'][:].astype(np.float32)
    e1 = hf['solid.E (Pa)'][:].astype(np.float32)
    k1 = hf['ht.kxx (W/(m*K))'][:].astype(np.float32)
x1 = x1.flatten(); y1 = y1.flatten(); z1 = z1.flatten(); e1 = e1.flatten(); k1 = k1.flatten()
lb = 0; ub = 1
EK_FE_BC = 1

x_data = x1
y_data = y1
z_data = z1
E_data = e1
K_data = k1

# -------------------------- BC Sample & Mask --------------------------
sample_num_trainBC = 400
# X=0, left
x_new = np.zeros((sample_num_trainBC,))
lhs_sample = lb + (ub - lb) * lhs(2, sample_num_trainBC)
y_new = lhs_sample[:, 0].flatten()
z_new = lhs_sample[:, 1].flatten()
e_new = np.ones((sample_num_trainBC,)) * EK_FE_BC
k_new = np.ones((sample_num_trainBC,)) * EK_FE_BC
x1 = np.concatenate((x1, x_new))
y1 = np.concatenate((y1, y_new))
z1 = np.concatenate((z1, z_new))
e1 = np.concatenate((e1, e_new))
k1 = np.concatenate((k1, k_new))
# X=1, right
x_new = np.ones((sample_num_trainBC,))
lhs_sample = lb + (ub - lb) * lhs(2, sample_num_trainBC)
y_new = lhs_sample[:, 0].flatten()
z_new = lhs_sample[:, 1].flatten()
e_new = np.ones((sample_num_trainBC,)) * EK_FE_BC
k_new = np.ones((sample_num_trainBC,)) * EK_FE_BC
x1 = np.concatenate((x1, x_new))
y1 = np.concatenate((y1, y_new))
z1 = np.concatenate((z1, z_new))
e1 = np.concatenate((e1, e_new))
k1 = np.concatenate((k1, k_new))
# Y=0, front
y_new = np.zeros((sample_num_trainBC,))
lhs_sample = lb + (ub - lb) * lhs(2, sample_num_trainBC)
x_new = lhs_sample[:, 0].flatten()
z_new = lhs_sample[:, 1].flatten()
e_new = np.ones((sample_num_trainBC,)) * EK_FE_BC
k_new = np.ones((sample_num_trainBC,)) * EK_FE_BC
x1 = np.concatenate((x1, x_new))
y1 = np.concatenate((y1, y_new))
z1 = np.concatenate((z1, z_new))
e1 = np.concatenate((e1, e_new))
k1 = np.concatenate((k1, k_new))
# Y=1, behind
y_new = np.ones((sample_num_trainBC,))
lhs_sample = lb + (ub - lb) * lhs(2, sample_num_trainBC)
x_new = lhs_sample[:, 0].flatten()
z_new = lhs_sample[:, 1].flatten()
e_new = np.ones((sample_num_trainBC,)) * EK_FE_BC
k_new = np.ones((sample_num_trainBC,)) * EK_FE_BC
x1 = np.concatenate((x1, x_new))
y1 = np.concatenate((y1, y_new))
z1 = np.concatenate((z1, z_new))
e1 = np.concatenate((e1, e_new))
k1 = np.concatenate((k1, k_new))
# Z=0, bottom
z_new = np.zeros((sample_num_trainBC,))
lhs_sample = lb + (ub - lb) * lhs(2, sample_num_trainBC)
x_new = lhs_sample[:, 0].flatten()
y_new = lhs_sample[:, 1].flatten()
e_new = np.ones((sample_num_trainBC,)) * EK_FE_BC
k_new = np.ones((sample_num_trainBC,)) * EK_FE_BC
x1 = np.concatenate((x1, x_new))
y1 = np.concatenate((y1, y_new))
z1 = np.concatenate((z1, z_new))
e1 = np.concatenate((e1, e_new))
k1 = np.concatenate((k1, k_new))
# Z=1, top
z_new = np.ones((sample_num_trainBC,))
lhs_sample = lb + (ub - lb) * lhs(2, sample_num_trainBC)
x_new = lhs_sample[:, 0].flatten()
y_new = lhs_sample[:, 1].flatten()
e_new = np.ones((sample_num_trainBC,)) * EK_FE_BC
k_new = np.ones((sample_num_trainBC,)) * EK_FE_BC
x1 = np.concatenate((x1, x_new))
y1 = np.concatenate((y1, y_new))
z1 = np.concatenate((z1, z_new))
e1 = np.concatenate((e1, e_new))
k1 = np.concatenate((k1, k_new))

# Mask
M_bound = np.logical_or(
    np.logical_or(np.logical_or(x1==0, x1==1), np.logical_or(y1==0, y1==1)), np.logical_or(z1==0, z1==1)
).astype(dtype=np.float32)
M_domain = np.logical_not(np.logical_or(
    np.logical_or(np.logical_or(x1==0, x1==1), np.logical_or(y1==0, y1==1)), np.logical_or(z1==0, z1==1)
)).astype(dtype=np.float32)
M_bound_left   = (x1==0).astype(dtype=np.float32)
M_bound_right  = (x1==1).astype(dtype=np.float32)
M_bound_front  = (y1==0).astype(dtype=np.float32)
M_bound_behind = (y1==1).astype(dtype=np.float32)
M_bound_bottom = (z1==0).astype(dtype=np.float32)
M_bound_top    = (z1==1).astype(dtype=np.float32)

nb = np.sum(M_bound).item()
nb_l = np.sum(M_bound_left).item()
nb_r = np.sum(M_bound_right).item()
nb_f = np.sum(M_bound_front).item()
nb_bh = np.sum(M_bound_behind).item()
nb_t = np.sum(M_bound_top).item()
nb_bt = np.sum(M_bound_bottom).item()
nd = np.sum(M_domain).item()

M_domain = to_tensor(M_domain).reshape((-1, 1))
M_bound_left   = to_tensor(M_bound_left).reshape((-1, 1))
M_bound_right  = to_tensor(M_bound_right).reshape((-1, 1))
M_bound_front  = to_tensor(M_bound_front).reshape((-1, 1))
M_bound_behind = to_tensor(M_bound_behind).reshape((-1, 1))
M_bound_bottom = to_tensor(M_bound_bottom).reshape((-1, 1))
M_bound_top    = to_tensor(M_bound_top).reshape((-1, 1))

"""
self, models: Dict[str, Functional],
x_train: torch.Tensor, y_train: torch.Tensor, z_train: torch.Tensor,
nb: float, nb_l: float, nb_r: float, nb_f: float, nb_bh: float, nb_t: float, nb_bt: float, nd: float,
M_domain: torch.Tensor, M_bound_left: torch.Tensor, M_bound_right: torch.Tensor, M_bound_front: torch.Tensor, M_bound_behind: torch.Tensor, M_bound_bottom: torch.Tensor, M_bound_top: torch.Tensor,
E: torch.Tensor, K: torch.Tensor, cfg: Config, sample_FE: FESample
"""
x1 = to_tensor(x1, requires_grad=True).reshape((-1, 1))
y1 = to_tensor(y1, requires_grad=True).reshape((-1, 1))
z1 = to_tensor(z1, requires_grad=True).reshape((-1, 1))
e1 = to_tensor(e1).reshape((-1, 1))
k1 = to_tensor(k1).reshape((-1, 1))


################################## TRAINING ##################################
training_util = TrainingUtils(
    models, x1, y1, z1,
    nb, nb_l, nb_r, nb_f, nb_bh, nb_t, nb_bt, nd,
    M_domain, M_bound_left, M_bound_right, M_bound_front, M_bound_behind, M_bound_bottom, M_bound_top,
    e1, k1, cfg, sample_FE
)

time1 = time.time()
training_util.func1_train()
print(f"time consumed: {time.time() - time1}")


################################## POSTPROCESS ##################################
# Test dataset
with h5py.File(cfg.testfile, 'r') as hf:
    x_data = hf['X'][:].astype(np.float32)
    y_data = hf['Y'][:].astype(np.float32)
    z_data = hf['Z'][:].astype(np.float32)
    E_data = hf['solid.E (Pa)'][:].astype(np.float32)
    K_data = hf['ht.kxx (W/(m*K))'][:].astype(np.float32)
    u_x_FE = hf['u (m)'][:].astype(np.float32)
    u_y_FE = hf['v (m)'][:].astype(np.float32)
    u_z_FE = hf['w (m)'][:].astype(np.float32)
    sigma_x_FE = hf['solid.sxx (N/m^2)'][:].astype(np.float32)
    sigma_y_FE = hf['solid.syy (N/m^2)'][:].astype(np.float32)
    sigma_z_FE = hf['solid.szz (N/m^2)'][:].astype(np.float32)
    sigma_xy_FE = hf['solid.sxy (N/m^2)'][:].astype(np.float32)
    sigma_yz_FE = hf['solid.syz (N/m^2)'][:].astype(np.float32)
    sigma_xz_FE = hf['solid.sxz (N/m^2)'][:].astype(np.float32)
    T_FE = hf['T (K)'][:].astype(np.float32)
    q_x_FE = hf['ht.tfluxx (W/m^2)'][:].astype(np.float32)
    q_y_FE = hf['ht.tfluxy (W/m^2)'][:].astype(np.float32)
    q_z_FE = hf['ht.tfluxz (W/m^2)'][:].astype(np.float32)
x_data = x_data.reshape((-1, 1))
y_data = y_data.reshape((-1, 1))
z_data = z_data.reshape((-1, 1))
E_data = E_data.reshape((-1, 1))
K_data = K_data.reshape((-1, 1))
u_x_FE = u_x_FE.flatten()
u_y_FE = u_y_FE.flatten()
u_z_FE = u_z_FE.flatten()
sigma_x_FE = sigma_x_FE.flatten()
sigma_y_FE = sigma_y_FE.flatten()
sigma_z_FE = sigma_z_FE.flatten()
sigma_xy_FE = sigma_xy_FE.flatten()
sigma_yz_FE = sigma_yz_FE.flatten()
sigma_xz_FE = sigma_xz_FE.flatten()
T_FE = T_FE.flatten()
q_x_FE = q_x_FE.flatten()
q_y_FE = q_y_FE.flatten()
q_z_FE = q_z_FE.flatten()
history = training_util.history

x_data = to_tensor(x_data, requires_grad=True).reshape((-1, 1)).to(device)
y_data = to_tensor(y_data, requires_grad=True).reshape((-1, 1)).to(device)
z_data = to_tensor(z_data, requires_grad=True).reshape((-1, 1)).to(device)
X_test = torch.hstack((x_data, y_data, z_data))
E_data = to_tensor(E_data, requires_grad=True).reshape((-1, 1)).to(device)
K_data = to_tensor(K_data, requires_grad=True).reshape((-1, 1)).to(device)

# -------------------------- Predict --------------------------
u_xn_pred = training_util.u_xn(X_test)
u_yn_pred = training_util.u_yn(X_test)
u_zn_pred = training_util.u_zn(X_test)
u_x_pred = u_xn_pred * x_data * (x_data - 1.0)
u_y_pred = u_yn_pred * y_data * (y_data - 1.0)
u_z_pred = u_zn_pred * z_data * (z_data - 1.0)
# sig_x_O_pred = training_util.sig_x_O(X_test)
# sig_y_O_pred = training_util.sig_y_O(X_test)
# sig_xy_O_pred = training_util.sig_xy_O(X_test)
Tn_pred = training_util.Tn(X_test)
# q_x_O_pred = training_util.q_x_O(X_test)
# q_y_O_pred = training_util.q_y_O(X_test)
T_pred = Tn_pred * x_data * (x_data - 1.0) + 1.0 - x_data
gr_T_x_pred = gradients(T_pred, x_data)[0]
gr_T_y_pred = gradients(T_pred, y_data)[0]
gr_T_z_pred = gradients(T_pred, z_data)[0]
q_x_pred = (-K_data) * gr_T_x_pred
q_y_pred = (-K_data) * gr_T_y_pred
q_z_pred = (-K_data) * gr_T_z_pred
strain_x_pred  = gradients(u_x_pred, x_data)[0]
strain_y_pred  = gradients(u_y_pred, y_data)[0]
strain_z_pred  = gradients(u_z_pred, z_data)[0]
strain_xy_pred = (gradients(u_x_pred, y_data)[0] + gradients(u_y_pred, x_data)[0]) * 0.5
strain_yz_pred = (gradients(u_y_pred, z_data)[0] + gradients(u_z_pred, y_data)[0]) * 0.5
strain_xz_pred = (gradients(u_x_pred, z_data)[0] + gradients(u_z_pred, x_data)[0]) * 0.5
strain_T_x_pred = (T_pred - T_0) * alpha
strain_T_y_pred = (T_pred - T_0) * alpha
strain_T_z_pred = (T_pred - T_0) * alpha
sigma_x_pred   = (E_data/((1+nu)*(1-2*nu))) * ((1-nu)*strain_x_pred + nu*(strain_y_pred+strain_z_pred)) - E_data*strain_T_x_pred/(1-2*nu)
sigma_y_pred   = (E_data/((1+nu)*(1-2*nu))) * ((1-nu)*strain_y_pred + nu*(strain_x_pred+strain_z_pred)) - E_data*strain_T_y_pred/(1-2*nu)
sigma_z_pred   = (E_data/((1+nu)*(1-2*nu))) * ((1-nu)*strain_z_pred + nu*(strain_x_pred+strain_y_pred)) - E_data*strain_T_z_pred/(1-2*nu)
sigma_xy_pred  = (E_data/(1+nu)) * strain_xy_pred
sigma_yz_pred  = (E_data/(1+nu)) * strain_yz_pred
sigma_xz_pred  = (E_data/(1+nu)) * strain_xz_pred


x_data = to_numpy(x_data).flatten()
y_data = to_numpy(y_data).flatten()
z_data = to_numpy(z_data).flatten()
# Predictions
u_x_pred = to_numpy(u_x_pred).flatten()
u_y_pred = to_numpy(u_y_pred).flatten()
u_z_pred = to_numpy(u_z_pred).flatten()
sigma_x_pred = to_numpy(sigma_x_pred).flatten()
sigma_y_pred = to_numpy(sigma_y_pred).flatten()
sigma_z_pred = to_numpy(sigma_z_pred).flatten()
sigma_xy_pred = to_numpy(sigma_xy_pred).flatten()
sigma_yz_pred = to_numpy(sigma_yz_pred).flatten()
sigma_xz_pred = to_numpy(sigma_xz_pred).flatten()
strain_x_pred = to_numpy(strain_x_pred).flatten()
strain_y_pred = to_numpy(strain_y_pred).flatten()
strain_z_pred = to_numpy(strain_z_pred).flatten()
strain_xy_pred = to_numpy(strain_xy_pred).flatten()
strain_yz_pred = to_numpy(strain_yz_pred).flatten()
strain_xz_pred = to_numpy(strain_xz_pred).flatten()
T_pred = to_numpy(T_pred).flatten()
q_x_pred = to_numpy(q_x_pred).flatten()
q_y_pred = to_numpy(q_y_pred).flatten()
q_z_pred = to_numpy(q_z_pred).flatten()
# Scarlars
u_abs_pred = np.sqrt(np.square(u_x_pred) + np.square(u_y_pred) + np.square(u_z_pred))
u_abs_FE = np.sqrt(np.square(u_x_FE) + np.square(u_y_FE) + np.square(u_z_FE))
sigma_mises_pred = np.sqrt((
    np.square(sigma_x_pred-sigma_y_pred) + np.square(sigma_y_pred-sigma_z_pred) + np.square(sigma_z_pred-sigma_x_pred)
    + 6 * (np.square(sigma_xy_pred) + np.square(sigma_yz_pred) + np.square(sigma_xz_pred))
) / 2)
sigma_mises_FE = np.sqrt((
    np.square(sigma_x_FE-sigma_y_FE) + np.square(sigma_y_FE-sigma_z_FE) + np.square(sigma_z_FE-sigma_x_FE)
    + 6 * (np.square(sigma_xy_FE) + np.square(sigma_yz_FE) + np.square(sigma_xz_FE))
) / 2)
q_abs_pred = np.sqrt(np.square(q_x_pred) + np.square(q_y_pred) + np.square(q_z_pred))
q_abs_FE = np.sqrt(np.square(q_x_FE) + np.square(q_y_FE) + np.square(q_z_FE))
# 误差
u_abs_error = np.abs(u_abs_pred - u_abs_FE)
sigma_mises_error = np.abs(sigma_mises_pred - sigma_mises_FE)
T_error = np.abs(T_pred - T_FE)
q_abs_error = np.abs(q_abs_pred - q_abs_FE)

# Error
u_x_error = np.abs(u_x_FE - u_x_pred)
u_y_error = np.abs(u_y_FE - u_y_pred)
u_z_error = np.abs(u_z_FE - u_z_pred)
sigma_x_error  = np.abs(sigma_x_FE - sigma_x_pred)
sigma_y_error  = np.abs(sigma_y_FE - sigma_y_pred)
sigma_z_error  = np.abs(sigma_z_FE - sigma_z_pred)
sigma_xy_error = np.abs(sigma_xy_FE - sigma_xy_pred)
sigma_yz_error = np.abs(sigma_yz_FE - sigma_yz_pred)
sigma_xz_error = np.abs(sigma_xz_FE - sigma_xz_pred)
T_error   = np.abs(T_pred - T_FE)
q_x_error = np.abs(q_x_pred - q_x_FE)
q_y_error = np.abs(q_y_pred - q_y_FE)
q_z_error = np.abs(q_z_pred - q_z_FE)


# -------------------------- SAVEDATA --------------------------
results = dict()
results["x_data"] = x_data
results["y_data"] = y_data
results["z_data"] = z_data
results["u_x_pred"] = u_x_pred
results["u_y_pred"] = u_y_pred
results["u_z_pred"] = u_z_pred
results["strain_x_pred"] = strain_x_pred
results["strain_y_pred"] = strain_y_pred
results["strain_z_pred"] = strain_z_pred
results["strain_xy_pred"] = strain_xy_pred
results["strain_yz_pred"] = strain_yz_pred
results["strain_xz_pred"] = strain_xz_pred
results["sigma_x_pred"] = sigma_x_pred
results["sigma_y_pred"] = sigma_y_pred
results["sigma_z_pred"] = sigma_z_pred
results["sigma_xy_pred"] = sigma_xy_pred
results["sigma_yz_pred"] = sigma_yz_pred
results["sigma_xz_pred"] = sigma_xz_pred
results["T_pred"] = T_pred
results["q_x_pred"] = q_x_pred
results["q_y_pred"] = q_y_pred
results["q_z_pred"] = q_z_pred
results["u_x_FE"] = u_x_FE
results["u_y_FE"] = u_y_FE
results["u_z_FE"] = u_z_FE
results["sigma_x_FE"] = sigma_x_FE
results["sigma_y_FE"] = sigma_y_FE
results["sigma_z_FE"] = sigma_z_FE
results["sigma_xy_FE"] = sigma_xy_FE
results["sigma_yz_FE"] = sigma_yz_FE
results["sigma_xz_FE"] = sigma_xz_FE
results["T_FE"] = T_FE
results["q_x_FE"] = q_x_FE
results["q_y_FE"] = q_y_FE
results["q_z_FE"] = q_z_FE
results["history"] = dict()
for (i, name) in enumerate(cfg.loss_terms):
    results["history"][name] = history[i].detach().cpu().numpy().tolist()

# with open('results/10-1-data04-3d/data_mspinn2.pkl', 'wb') as f:
#     pickle.dump(results, f)



# -------------------------- PLOT --------------------------
fig = plt.figure(figsize=(12, 7))
itter = 0 
for word, loss in results["history"].items():
    plt.semilogy(np.array(loss), label=cfg.loss_terms[itter])
    itter+=1
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
# plt.savefig('data_thermoelastic/data/losses')

fig = plt.figure(figsize=(12, 7))
plt.semilogy(results["history"]['Total_loss'], label='Total_loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('Total-loss')
# plt.savefig('data_thermoelastic/data/totalloss')



# -------------------------- PLOT 3D --------------------------
# truncated volume
x_data_tr = x_data[x_data<=0.5]
y_data_tr = y_data[x_data<=0.5]
z_data_tr = z_data[x_data<=0.5]

u_x_pred_tr = u_x_pred[x_data<=0.5]
u_y_pred_tr = u_y_pred[x_data<=0.5]
u_z_pred_tr = u_z_pred[x_data<=0.5]
sigma_x_pred_tr = sigma_x_pred[x_data<=0.5]
sigma_y_pred_tr = sigma_y_pred[x_data<=0.5]
sigma_z_pred_tr = sigma_z_pred[x_data<=0.5]
sigma_xy_pred_tr = sigma_xy_pred[x_data<=0.5]
sigma_yz_pred_tr = sigma_yz_pred[x_data<=0.5]
sigma_xz_pred_tr = sigma_xz_pred[x_data<=0.5]
strain_x_pred_tr = strain_x_pred[x_data<=0.5]
strain_y_pred_tr = strain_y_pred[x_data<=0.5]
strain_z_pred_tr = strain_z_pred[x_data<=0.5]
strain_xy_pred_tr = strain_xy_pred[x_data<=0.5]
strain_yz_pred_tr = strain_yz_pred[x_data<=0.5]
strain_xz_pred_tr = strain_xz_pred[x_data<=0.5]
T_pred_tr = T_pred[x_data<=0.5]
q_x_pred_tr = q_x_pred[x_data<=0.5]
q_y_pred_tr = q_y_pred[x_data<=0.5]
q_z_pred_tr = q_z_pred[x_data<=0.5]

u_x_FE_tr = u_x_FE[x_data<=0.5]
u_y_FE_tr = u_y_FE[x_data<=0.5]
u_z_FE_tr = u_z_FE[x_data<=0.5]
sigma_x_FE_tr = sigma_x_FE[x_data<=0.5]
sigma_y_FE_tr = sigma_y_FE[x_data<=0.5]
sigma_z_FE_tr = sigma_z_FE[x_data<=0.5]
sigma_xy_FE_tr = sigma_xy_FE[x_data<=0.5]
sigma_yz_FE_tr = sigma_yz_FE[x_data<=0.5]
sigma_xz_FE_tr = sigma_xz_FE[x_data<=0.5]
T_FE_tr = T_FE[x_data<=0.5]
q_x_FE_tr = q_x_FE[x_data<=0.5]
q_y_FE_tr = q_y_FE[x_data<=0.5]
q_z_FE_tr = q_z_FE[x_data<=0.5]

u_x_error_tr = u_x_error[x_data<=0.5]
u_y_error_tr = u_y_error[x_data<=0.5]
u_z_error_tr = u_z_error[x_data<=0.5]
sigma_x_error_tr = sigma_x_error[x_data<=0.5]
sigma_y_error_tr = sigma_y_error[x_data<=0.5]
sigma_z_error_tr = sigma_z_error[x_data<=0.5]
sigma_xy_error_tr = sigma_xy_error[x_data<=0.5]
sigma_yz_error_tr = sigma_yz_error[x_data<=0.5]
sigma_xz_error_tr = sigma_xz_error[x_data<=0.5]
T_error_tr = T_error[x_data<=0.5]
q_x_error_tr = q_x_error[x_data<=0.5]
q_y_error_tr = q_y_error[x_data<=0.5]
q_z_error_tr = q_z_error[x_data<=0.5]

u_abs_pred_tr = u_abs_pred[x_data<=0.5]
u_abs_FE_tr = u_abs_FE[x_data<=0.5]
sigma_mises_pred_tr = sigma_mises_pred[x_data<=0.5]
sigma_mises_FE_tr = sigma_mises_FE[x_data<=0.5]
q_abs_pred_tr = q_abs_pred[x_data<=0.5]
q_abs_FE_tr = q_abs_FE[x_data<=0.5]

u_abs_error_tr = u_abs_error[x_data<=0.5]
sigma_mises_error_tr = sigma_mises_error[x_data<=0.5]
T_error_tr = T_error[x_data<=0.5]
q_abs_error_tr = q_abs_error[x_data<=0.5]


def plot_rotated_ellipse_on_existing_axes(fig, ax, a_1, b_1, c_1, x_1, y_1, z_1, deg):
    u = np.linspace(0, 2 * np.pi, 100)
    v = np.linspace(0, np.pi, 100)
    x = a_1 * np.outer(np.cos(u), np.sin(v))
    y = b_1 * np.outer(np.sin(u), np.sin(v))
    z = c_1 * np.outer(np.ones_like(u), np.cos(v))
    # rotate deg
    theta = np.radians(deg)
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    # apply the rotation max pointwise
    xyz_rotated = np.einsum('ij,jkl->ikl', rotation_matrix, np.stack([x, y, z]))
    # move ellip center to (x_1, y_1, z_1)
    ax.plot_wireframe(xyz_rotated[0]+x_1, xyz_rotated[1]+y_1, xyz_rotated[2]+z_1, color="black", rcount=10, ccount=10, linewidth=1)

def plot_cube_on_existing_axes(fig, ax, a_2, b_2, c_2, x_2, y_2, z_2):
    # 8 vertices
    r = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0],
                  [0, 0, 1], [0, 1, 1], [1, 1, 1], [1, 0, 1]])
    # resize and move to (x_2, y_2, z_2)
    r = r * np.array([a_2, b_2, c_2])
    r = r + np.array([x_2, y_2, z_2])
    # 6 faces
    faces = [[r[0],r[1],r[2],r[3]], [r[4],r[5],r[6],r[7]], [r[0],r[1],r[5],r[4]],
             [r[2],r[3],r[7],r[6]], [r[1],r[2],r[6],r[5]], [r[4],r[7],r[3],r[0]]]
    cube = Poly3DCollection(faces, facecolors='cyan', linewidths=1, edgecolors='black', alpha=0.0)
    ax.add_collection3d(cube)


# PLOT3D Group 01 ################################################################################################################################
fig = plt.figure(figsize=(7, 6))
ax = plt.axes(projection="3d", proj_type="ortho")
ax.view_init(elev=-26, azim=78, roll=-180)
cax = ax.inset_axes([1.02, 0.04, 0.02, 0.92])
plt.tight_layout()
ax.set_box_aspect([1, 1, 1])
sc = ax.scatter3D(x_data_tr, y_data_tr, z_data_tr, c=sigma_mises_pred_tr, cmap='jet', s=1, alpha=0.7, vmin=np.min(sigma_mises_FE), vmax=np.max(sigma_mises_FE))
cbar = plt.colorbar(sc, ax=ax, cax=cax)
plot_rotated_ellipse_on_existing_axes(fig, ax, 0.2, 0.3, 0.1, 0.5, 0.5, 0.5, 20)
plot_cube_on_existing_axes(fig, ax, 1, 1, 1, 0, 0, 0)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
# plt.savefig('results/10-1-data04-3d/sigma_mises_pred_tr.png', dpi=400, bbox_inches='tight', pad_inches=0.05)

fig = plt.figure(figsize=(7, 6))
ax = plt.axes(projection="3d", proj_type="ortho")
ax.view_init(elev=-26, azim=78, roll=-180)
cax = ax.inset_axes([1.02, 0.04, 0.02, 0.92])
plt.tight_layout()
ax.set_box_aspect([1, 1, 1])
sc = ax.scatter3D(x_data_tr, y_data_tr, z_data_tr, c=sigma_mises_FE_tr, cmap='jet', s=1, alpha=0.8, vmin=np.min(sigma_mises_FE), vmax=np.max(sigma_mises_FE))
cbar = plt.colorbar(sc, ax=ax, cax=cax)
plot_rotated_ellipse_on_existing_axes(fig, ax, 0.2, 0.3, 0.1, 0.5, 0.5, 0.5, 20)
plot_cube_on_existing_axes(fig, ax, 1, 1, 1, 0, 0, 0)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
# plt.savefig('results/10-1-data04-3d/sigma_mises_FE_tr.png', dpi=400, bbox_inches='tight', pad_inches=0.05)

fig = plt.figure(figsize=(7, 6))
ax = plt.axes(projection="3d", proj_type="ortho")
ax.view_init(elev=-26, azim=78, roll=-180)
cax = ax.inset_axes([1.02, 0.04, 0.02, 0.92])
plt.tight_layout()
ax.set_box_aspect([1, 1, 1])
sc = ax.scatter3D(x_data_tr, y_data_tr, z_data_tr, c=sigma_mises_error_tr, cmap='jet', s=1, alpha=0.8)
cbar = plt.colorbar(sc, ax=ax, cax=cax)
plot_rotated_ellipse_on_existing_axes(fig, ax, 0.2, 0.3, 0.1, 0.5, 0.5, 0.5, 20)
plot_cube_on_existing_axes(fig, ax, 1, 1, 1, 0, 0, 0)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
# plt.savefig('results/10-1-data04-3d/sigma_mises_error_tr.png', dpi=400, bbox_inches='tight', pad_inches=0.05)


# PLOT3D Group 02 ################################################################################################################################
fig = plt.figure(figsize=(7, 6))
ax = plt.axes(projection="3d", proj_type="ortho")
ax.view_init(elev=-26, azim=78, roll=-180)
cax = ax.inset_axes([1.02, 0.04, 0.02, 0.92])
plt.tight_layout()
ax.set_box_aspect([1, 1, 1])
sc = ax.scatter3D(x_data_tr, y_data_tr, z_data_tr, c=u_abs_pred_tr, cmap='jet', s=1, alpha=0.7, vmin=np.min(u_abs_FE), vmax=np.max(u_abs_FE))
cbar = plt.colorbar(sc, ax=ax, cax=cax)
plot_rotated_ellipse_on_existing_axes(fig, ax, 0.2, 0.3, 0.1, 0.5, 0.5, 0.5, 20)
plot_cube_on_existing_axes(fig, ax, 1, 1, 1, 0, 0, 0)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
# plt.savefig('results/10-1-data04-3d/u_abs_pred_tr.png', dpi=400, bbox_inches='tight', pad_inches=0.05)

fig = plt.figure(figsize=(7, 6))
ax = plt.axes(projection="3d", proj_type="ortho")
ax.view_init(elev=-26, azim=78, roll=-180)
cax = ax.inset_axes([1.02, 0.04, 0.02, 0.92])
plt.tight_layout()
ax.set_box_aspect([1, 1, 1])
sc = ax.scatter3D(x_data_tr, y_data_tr, z_data_tr, c=u_abs_FE_tr, cmap='jet', s=1, alpha=0.8, vmin=np.min(u_abs_FE), vmax=np.max(u_abs_FE))
cbar = plt.colorbar(sc, ax=ax, cax=cax)
plot_rotated_ellipse_on_existing_axes(fig, ax, 0.2, 0.3, 0.1, 0.5, 0.5, 0.5, 20)
plot_cube_on_existing_axes(fig, ax, 1, 1, 1, 0, 0, 0)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
# plt.savefig('results/10-1-data04-3d/u_abs_FE_tr.png', dpi=400, bbox_inches='tight', pad_inches=0.05)

fig = plt.figure(figsize=(7, 6))
ax = plt.axes(projection="3d", proj_type="ortho")
ax.view_init(elev=-26, azim=78, roll=-180)
cax = ax.inset_axes([1.02, 0.04, 0.02, 0.92])
plt.tight_layout()
ax.set_box_aspect([1, 1, 1])
sc = ax.scatter3D(x_data_tr, y_data_tr, z_data_tr, c=u_abs_error_tr, cmap='jet', s=1, alpha=0.8)
cbar = plt.colorbar(sc, ax=ax, cax=cax)
plot_rotated_ellipse_on_existing_axes(fig, ax, 0.2, 0.3, 0.1, 0.5, 0.5, 0.5, 20)
plot_cube_on_existing_axes(fig, ax, 1, 1, 1, 0, 0, 0)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
# plt.savefig('results/10-1-data04-3d/u_abs_error_tr.png', dpi=400, bbox_inches='tight', pad_inches=0.05)


# PLOT3D Group 03 ################################################################################################################################
fig = plt.figure(figsize=(7, 6))
ax = plt.axes(projection="3d", proj_type="ortho")
ax.view_init(elev=-26, azim=78, roll=-180)
cax = ax.inset_axes([1.02, 0.04, 0.02, 0.92])
plt.tight_layout()
ax.set_box_aspect([1, 1, 1])
sc = ax.scatter3D(x_data_tr, y_data_tr, z_data_tr, c=T_pred_tr, cmap='jet', s=1, alpha=0.7, vmin=np.min(T_FE), vmax=np.max(T_FE))
cbar = plt.colorbar(sc, ax=ax, cax=cax)
plot_rotated_ellipse_on_existing_axes(fig, ax, 0.2, 0.3, 0.1, 0.5, 0.5, 0.5, 20)
plot_cube_on_existing_axes(fig, ax, 1, 1, 1, 0, 0, 0)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
# plt.savefig('results/10-1-data04-3d/T_pred_tr.png', dpi=400, bbox_inches='tight', pad_inches=0.05)

fig = plt.figure(figsize=(7, 6))
ax = plt.axes(projection="3d", proj_type="ortho")
ax.view_init(elev=-26, azim=78, roll=-180)
cax = ax.inset_axes([1.02, 0.04, 0.02, 0.92])
plt.tight_layout()
ax.set_box_aspect([1, 1, 1])
sc = ax.scatter3D(x_data_tr, y_data_tr, z_data_tr, c=T_FE_tr, cmap='jet', s=1, alpha=0.8, vmin=np.min(T_FE), vmax=np.max(T_FE))
cbar = plt.colorbar(sc, ax=ax, cax=cax)
plot_rotated_ellipse_on_existing_axes(fig, ax, 0.2, 0.3, 0.1, 0.5, 0.5, 0.5, 20)
plot_cube_on_existing_axes(fig, ax, 1, 1, 1, 0, 0, 0)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
# plt.savefig('results/10-1-data04-3d/T_FE_tr.png', dpi=400, bbox_inches='tight', pad_inches=0.05)

fig = plt.figure(figsize=(7, 6))
ax = plt.axes(projection="3d", proj_type="ortho")
ax.view_init(elev=-26, azim=78, roll=-180)
cax = ax.inset_axes([1.02, 0.04, 0.02, 0.92])
plt.tight_layout()
ax.set_box_aspect([1, 1, 1])
sc = ax.scatter3D(x_data_tr, y_data_tr, z_data_tr, c=T_error_tr, cmap='jet', s=1, alpha=0.8)
cbar = plt.colorbar(sc, ax=ax, cax=cax)
plot_rotated_ellipse_on_existing_axes(fig, ax, 0.2, 0.3, 0.1, 0.5, 0.5, 0.5, 20)
plot_cube_on_existing_axes(fig, ax, 1, 1, 1, 0, 0, 0)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
# plt.savefig('results/10-1-data04-3d/T_error_tr.png', dpi=400, bbox_inches='tight', pad_inches=0.05)


# PLOT3D Group 04 ################################################################################################################################
fig = plt.figure(figsize=(7, 6))
ax = plt.axes(projection="3d", proj_type="ortho")
ax.view_init(elev=-26, azim=78, roll=-180)
cax = ax.inset_axes([1.02, 0.04, 0.02, 0.92])
plt.tight_layout()
ax.set_box_aspect([1, 1, 1])
sc = ax.scatter3D(x_data_tr, y_data_tr, z_data_tr, c=q_abs_pred_tr, cmap='jet', s=1, alpha=0.7, vmin=np.min(q_abs_FE), vmax=np.max(q_abs_FE))
cbar = plt.colorbar(sc, ax=ax, cax=cax)
plot_rotated_ellipse_on_existing_axes(fig, ax, 0.2, 0.3, 0.1, 0.5, 0.5, 0.5, 20)
plot_cube_on_existing_axes(fig, ax, 1, 1, 1, 0, 0, 0)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
# plt.savefig('results/10-1-data04-3d/q_abs_pred_tr.png', dpi=400, bbox_inches='tight', pad_inches=0.05)

fig = plt.figure(figsize=(7, 6))
ax = plt.axes(projection="3d", proj_type="ortho")
ax.view_init(elev=-26, azim=78, roll=-180)
cax = ax.inset_axes([1.02, 0.04, 0.02, 0.92])
plt.tight_layout()
ax.set_box_aspect([1, 1, 1])
sc = ax.scatter3D(x_data_tr, y_data_tr, z_data_tr, c=q_abs_FE_tr, cmap='jet', s=1, alpha=0.8, vmin=np.min(q_abs_FE), vmax=np.max(q_abs_FE))
cbar = plt.colorbar(sc, ax=ax, cax=cax)
plot_rotated_ellipse_on_existing_axes(fig, ax, 0.2, 0.3, 0.1, 0.5, 0.5, 0.5, 20)
plot_cube_on_existing_axes(fig, ax, 1, 1, 1, 0, 0, 0)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
# plt.savefig('results/10-1-data04-3d/q_abs_FE_tr.png', dpi=400, bbox_inches='tight', pad_inches=0.05)

fig = plt.figure(figsize=(7, 6))
ax = plt.axes(projection="3d", proj_type="ortho")
ax.view_init(elev=-26, azim=78, roll=-180)
cax = ax.inset_axes([1.02, 0.04, 0.02, 0.92])
plt.tight_layout()
ax.set_box_aspect([1, 1, 1])
sc = ax.scatter3D(x_data_tr, y_data_tr, z_data_tr, c=q_abs_error_tr, cmap='jet', s=1, alpha=0.8)
cbar = plt.colorbar(sc, ax=ax, cax=cax)
plot_rotated_ellipse_on_existing_axes(fig, ax, 0.2, 0.3, 0.1, 0.5, 0.5, 0.5, 20)
plot_cube_on_existing_axes(fig, ax, 1, 1, 1, 0, 0, 0)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
# plt.savefig('results/10-1-data04-3d/q_abs_error_tr.png', dpi=400, bbox_inches='tight', pad_inches=0.05)


plt.show()