import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from utils.nn_fixed import Functional
import pandas as pd

import itertools
from typing import List, Dict

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


class Config:
    name: str = ""
    idx:  str = str(0).rjust(3, "0")
    datafile: str = 'data_thermomechanic/b-i=10-1,16-refined.csv'
    input_dim: int = 2
    output_dim: int = 1
    hidden_layers: List[int] = [64, 64, 64, 64, 64]
    activation: str = "sin"
    iters: int = 25000
    loss_terms: List[str] = [
        'Total_loss', 'Work_m', 'BC6_m', 'BC10_m', 'BC14_m', 'BC18_m', 
        'Work_t', 'BC1_t', 'BC2_t', 'BC5_t', 'BC7_t'
    ]
cfg = Config()

nu = 0.3
alpha = 0.3
T_0 = 0

############################### SAMPLING ###############################
class FESample:
    def __init__(
        self,
        x_FE: torch.Tensor, y_FE: torch.Tensor, e_FE: torch.Tensor, k_FE: torch.Tensor,
        u_x_FE: torch.Tensor, u_y_FE: torch.Tensor, sigma_x_FE: torch.Tensor, sigma_y_FE: torch.Tensor, sigma_xy_FE: torch.Tensor,
        T_FE: torch.Tensor, q_x_FE: torch.Tensor, q_y_FE: torch.Tensor,
        sample_mode={"random": 0.05}, dtype=torch.float32
    ):
        self.x_FE = x_FE.clone().detach().to(dtype=dtype)
        self.y_FE = y_FE.clone().detach().to(dtype=dtype)
        self.e_FE = e_FE.clone().detach().to(dtype=dtype)
        self.k_FE = k_FE.clone().detach().to(dtype=dtype)
        self.u_x_FE = u_x_FE.clone().detach().to(dtype=dtype)
        self.u_y_FE = u_y_FE.clone().detach().to(dtype=dtype)
        self.sigma_x_FE = sigma_x_FE.clone().detach().to(dtype=dtype)
        self.sigma_y_FE = sigma_y_FE.clone().detach().to(dtype=dtype)
        self.sigma_xy_FE = sigma_xy_FE.clone().detach().to(dtype=dtype)
        self.T_FE = T_FE.clone().detach().to(dtype=dtype)
        self.q_x_FE = q_x_FE.clone().detach().to(dtype=dtype)
        self.q_y_FE = q_y_FE.clone().detach().to(dtype=dtype)
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
        max_samples = self.x_FE.shape[0]
        if samples > max_samples:
            raise ValueError(f"Cannot sample more than the population size ({max_samples})")
        indices = torch.randperm(max_samples)[:samples]
        self.__apply_indices(indices)

    def __linear_sampling(self, step):
        indices = torch.arange(0, self.x_FE.shape[0], step)
        self.__apply_indices(indices)

    def __apply_indices(self, indices):
        self.x_FE = self.x_FE[indices].detach().requires_grad_(True)
        self.y_FE = self.y_FE[indices].detach().requires_grad_(True)
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
        
result_FE = pd.read_csv(cfg.datafile)
x_FE = pd.DataFrame(result_FE, columns=['X']).to_numpy(dtype=np.float32)
y_FE = pd.DataFrame(result_FE, columns=['Y']).to_numpy(dtype=np.float32)
e_FE = pd.DataFrame(result_FE, columns=['solid.E (Pa)']).to_numpy(dtype=np.float32)
k_FE = pd.DataFrame(result_FE, columns=['ht.kxx (W/(m*K))']).to_numpy(dtype=np.float32)
u_x_FE =  pd.DataFrame(result_FE, columns= ['u (m)']).to_numpy(dtype=np.float32)
u_y_FE =  pd.DataFrame(result_FE, columns= ['v (m)']).to_numpy(dtype=np.float32)
sigma_x_FE =  pd.DataFrame(result_FE, columns= ['solid.sxx (N/m^2)']).to_numpy(dtype=np.float32)
sigma_y_FE =  pd.DataFrame(result_FE, columns= ['solid.syy (N/m^2)']).to_numpy(dtype=np.float32)
sigma_xy_FE =  pd.DataFrame(result_FE, columns= ['solid.sxy (N/m^2)']).to_numpy(dtype=np.float32)
T_FE =  pd.DataFrame(result_FE, columns= ['T (K)']).to_numpy(dtype=np.float32)
q_x_FE =  pd.DataFrame(result_FE, columns= ['ht.tfluxx (W/m^2)']).to_numpy(dtype=np.float32)
q_y_FE =  pd.DataFrame(result_FE, columns= ['ht.tfluxy (W/m^2)']).to_numpy(dtype=np.float32)

u_x_FE = to_tensor(np.array(u_x_FE)).reshape((-1, 1))
u_y_FE = to_tensor(np.array(u_y_FE)).reshape((-1, 1))
sigma_x_FE = to_tensor(np.array(sigma_x_FE)).reshape((-1, 1))
sigma_y_FE = to_tensor(np.array(sigma_y_FE)).reshape((-1, 1))
sigma_xy_FE = to_tensor(np.array(sigma_xy_FE)).reshape((-1, 1))
T_FE = to_tensor(np.array(T_FE)).reshape((-1, 1))
q_x_FE = to_tensor(np.array(q_x_FE)).reshape((-1, 1))
q_y_FE = to_tensor(np.array(q_y_FE)).reshape((-1, 1))

x_FE = to_tensor(x_FE, requires_grad=True).reshape((-1, 1))
y_FE = to_tensor(y_FE, requires_grad=True).reshape((-1, 1))
e_FE = to_tensor(e_FE).reshape((-1, 1))
k_FE = to_tensor(k_FE).reshape((-1, 1))

sample_FE = FESample(
    x_FE, y_FE, e_FE, k_FE, u_x_FE, u_y_FE, sigma_x_FE, sigma_y_FE, sigma_xy_FE,
    T_FE, q_x_FE, q_y_FE,
    sample_mode={"None": 0.9}
)



class TrainingUtils(nn.Module):
    def __init__(
        self, models: Dict[str, Functional],
        x_train: torch.Tensor, y_train: torch.Tensor,
        nb: float, nb_r: float, nb_l: float, nb_t: float, nb_b: float, nd: float,
        M_bound_right: torch.Tensor, M_bound_left: torch.Tensor, M_bound_top: torch.Tensor, M_bound_bottom: torch.Tensor, M_domain: torch.Tensor,
        E: torch.Tensor, K: torch.Tensor,
        cfg: Config, sample_FE: FESample
    ):
        super(TrainingUtils, self).__init__()
        self.Tn       = models["Tn"].to(device)
        self.q_x_O    = models["q_x_O"].to(device)
        self.q_y_O    = models["q_y_O"].to(device)
        self.u_xn     = models["u_xn"].to(device)
        self.u_yn     = models["u_yn"].to(device)
        self.sig_x_O  = models["sig_x_O"].to(device)
        self.sig_y_O  = models["sig_y_O"].to(device)
        self.sig_xy_O = models["sig_xy_O"].to(device)
        self.sample_FE = sample_FE
        
        self.x_fem   = torch.tensor(self.sample_FE.x_FE, dtype=torch.float32, device=device, requires_grad=True).reshape((-1, 1))
        self.y_fem   = torch.tensor(self.sample_FE.y_FE, dtype=torch.float32, device=device, requires_grad=True).reshape((-1, 1))
        self.x_train = x_train.to(device)
        self.y_train = y_train.to(device)
        
        self.E = E.to(device)
        self.K = K.to(device)
        
        self.nb = nb; self.nb_r = nb_r; self.nb_l = nb_l
        self.nb_t = nb_t; self.nb_b = nb_b; self.nd = nd
        self.M_bound_right = M_bound_right.to(device); self.M_bound_left = M_bound_left.to(device)
        self.M_bound_top = M_bound_top.to(device); self.M_bound_bottom = M_bound_bottom.to(device)
        self.M_domain = M_domain.to(device)
        
        self.cfg = cfg
        self.history = dict()
        for loss_name in cfg.loss_terms:
            self.history[loss_name] = []
        self.mse = nn.MSELoss()
        
        # FE data
        self.E_FE = torch.tensor(self.sample_FE.e_FE, dtype=torch.float32, device=device).reshape((-1, 1))
        self.K_FE = torch.tensor(self.sample_FE.k_FE, dtype=torch.float32, device=device).reshape((-1, 1))
        self.u_x_FE = torch.tensor(self.sample_FE.u_x_FE, dtype=torch.float32, device=device).reshape((-1, 1))
        self.u_y_FE = torch.tensor(self.sample_FE.u_y_FE, dtype=torch.float32, device=device).reshape((-1, 1))
        self.sigma_x_FE = torch.tensor(self.sample_FE.sigma_x_FE, dtype=torch.float32, device=device).reshape((-1, 1))
        self.sigma_y_FE = torch.tensor(self.sample_FE.sigma_y_FE, dtype=torch.float32, device=device).reshape((-1, 1))
        self.sigma_xy_FE = torch.tensor(self.sample_FE.sigma_xy_FE, dtype=torch.float32, device=device).reshape((-1, 1))
        self.T_FE = torch.tensor(self.sample_FE.T_FE, dtype=torch.float32, device=device).reshape((-1, 1))
        self.q_x_FE = torch.tensor(self.sample_FE.q_x_FE, dtype=torch.float32, device=device).reshape((-1, 1))
        self.q_y_FE = torch.tensor(self.sample_FE.q_y_FE, dtype=torch.float32, device=device).reshape((-1, 1))
    
    def func1(self, x, y, E, K):
        X = torch.hstack((x, y))
        Tn = self.Tn(X)
        T = Tn * x * (x - 1.0) + 1.0 - x
        gr_T_x = gradients(T, x)[0]
        gr_T_y = gradients(T, y)[0]
        q_x = (-K) * gr_T_x
        q_y = (-K) * gr_T_y
        
        u_xn = self.u_xn(X)
        u_yn = self.u_yn(X)
        u_x = u_xn * x * (x - 1.0)
        u_y = u_yn  * y * (y - 1.0)
        strain_x  = gradients(u_x, x)[0]
        strain_y  = gradients(u_y, y)[0]
        strain_xy = (gradients(u_x, y)[0] + gradients(u_y, x)[0]) * 0.5
        strain_T_x = (T - T_0) * alpha
        strain_T_y = (T - T_0) * alpha
        sigma_x   = (E/((1+nu)*(1-2*nu)))*((1-nu)*strain_x + nu*strain_y) - E*strain_T_x/(1-2*nu)
        sigma_y   = (E/((1+nu)*(1-2*nu)))*((1-nu)*strain_y + nu*strain_x) - E*strain_T_y/(1-2*nu)
        sigma_xy  = (E/((1+nu)*(1-2*nu)))*strain_xy*(1-2*nu)
        
        ##################### CONSTRUCT LOSS TERMS #####################
        ##################### Mechanical Governing Equation #####################
        Work_int_m = torch.sum(self.M_domain*(sigma_x * strain_x + sigma_y * strain_y + sigma_xy * 2* strain_xy))/(2*self.nd)
        Work_e_r_m = torch.sum(self.M_bound_right*(sigma_x * u_x + sigma_xy * u_y))/self.nb_r
        Work_e_l_m = torch.sum(self.M_bound_left*(-sigma_x * u_x - sigma_xy * u_y))/self.nb_l
        Work_e_t_m = torch.sum(self.M_bound_top*(sigma_y * u_y + sigma_xy * u_x))/self.nb_t
        Work_e_b_m = torch.sum(self.M_bound_bottom*(-sigma_y * u_y - sigma_xy * u_x))/self.nb_b
        Work_ext_m = Work_e_r_m + Work_e_l_m + Work_e_t_m + Work_e_b_m
        Work_m = torch.square(Work_int_m - Work_ext_m)
        # Mechanical BCs 
        BC6_m  = (x==0.)*(sigma_xy)
        BC10_m = (x==1.)*(sigma_xy)
        BC14_m  = (y==0.)*(sigma_xy)
        BC18_m  = (y==1.)*(sigma_xy)

        ##################### Thermal Governing Equation #####################
        Work_int_t = torch.sum(self.M_domain*(q_x * gr_T_x + q_y * gr_T_y))/self.nd
        Work_e_l_t = torch.sum(self.M_bound_left*(-q_x * T))/self.nb_l
        Work_ext_t = Work_e_l_t
        Work_t = torch.square(Work_int_t - Work_ext_t)
        # Thermal BCs
        BC1_t  = (x==0.)*(T - 1)
        BC2_t  = (x==1.)*(T)
        BC5_t  = (y==0.)*(q_y)
        BC7_t = (y==1.)*(q_y)
        
        return Work_m, BC6_m, BC10_m, BC14_m, BC18_m, Work_t, BC1_t, BC2_t, BC5_t, BC7_t
    def func1_FE(self, x, y, E, K):
        X = torch.hstack((x, y))
        Tn = self.Tn(X)
        T = Tn * x * (x - 1.0) + 1.0 - x
        gr_T_x = gradients(T, x)[0]
        gr_T_y = gradients(T, y)[0]
        q_x = (-K) * gr_T_x
        q_y = (-K) * gr_T_y
        
        u_xn = self.u_xn(X)
        u_yn = self.u_yn(X)
        u_x = u_xn * x * (x - 1.0)
        u_y = u_yn  * y * (y - 1.0)
        strain_x  = gradients(u_x, x)[0]
        strain_y  = gradients(u_y, y)[0]
        strain_xy = (gradients(u_x, y)[0] + gradients(u_y, x)[0]) * 0.5
        strain_T_x = (T - T_0) * alpha
        strain_T_y = (T - T_0) * alpha
        sigma_x   = (E/((1+nu)*(1-2*nu)))*((1-nu)*strain_x + nu*strain_y) - E*strain_T_x/(1-2*nu)
        sigma_y   = (E/((1+nu)*(1-2*nu)))*((1-nu)*strain_y + nu*strain_x) - E*strain_T_y/(1-2*nu)
        sigma_xy  = (E/((1+nu)*(1-2*nu)))*strain_xy*(1-2*nu)
        return u_x, u_y, sigma_x, sigma_y, sigma_xy, T, q_x, q_y
    def func1_train(self):
        optimizer = torch.optim.Adam(itertools.chain(self.Tn.parameters(), self.q_x_O.parameters(), self.q_y_O.parameters(), self.u_xn.parameters(), self.u_yn.parameters(), self.sig_x_O.parameters(), self.sig_y_O.parameters(), self.sig_xy_O.parameters()), lr=4e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=2500, gamma=0.5)
        for i in range(self.cfg.iters):
            ls_epoch = []
            optimizer.zero_grad()
            Work_m, BC6_m, BC10_m, BC14_m, BC18_m, Work_t, BC1_t, BC2_t, BC5_t, BC7_t = self.func1(self.x_train, self.y_train, self.E, self.K)
            # FE Loss
            u_x_FE_, u_y_FE_, sigma_x_FE_, sigma_y_FE_, sigma_xy_FE_, T_FE_, q_x_FE_, q_y_FE_ = self.func1_FE(self.x_fem, self.y_fem, self.E_FE, self.K_FE)
            ls_u_x_FE = self.mse(u_x_FE_, self.u_x_FE)
            ls_u_y_FE = self.mse(u_y_FE_, self.u_y_FE)
            ls_T_FE = self.mse(T_FE_, self.T_FE)
            ls_q_x_FE = self.mse(q_x_FE_, self.q_x_FE)
            ls_q_y_FE = self.mse(q_y_FE_, self.q_y_FE)
            ls_sigma_x_FE = self.mse(sigma_x_FE_, self.sigma_x_FE)
            ls_sigma_y_FE = self.mse(sigma_y_FE_, self.sigma_y_FE)
            ls_sigma_xy_FE = self.mse(sigma_xy_FE_, self.sigma_xy_FE)
            # Eq Loss
            ls_Work_m = Work_m; ls_epoch.append(ls_Work_m.item())
            ls_BC6_m = torch.mean(torch.square(BC6_m)); ls_epoch.append(ls_BC6_m.item())
            ls_BC10_m = torch.mean(torch.square(BC10_m)); ls_epoch.append(ls_BC10_m.item())
            ls_BC14_m = torch.mean(torch.square(BC14_m)); ls_epoch.append(ls_BC14_m.item())
            ls_BC18_m = torch.mean(torch.square(BC18_m)); ls_epoch.append(ls_BC18_m.item())
            ls_Work_t = Work_t; ls_epoch.append(ls_Work_t.item())
            ls_BC1_t = torch.mean(torch.square(BC1_t)); ls_epoch.append(ls_BC1_t.item())     # unused in Loss
            ls_BC2_t = torch.mean(torch.square(BC2_t)); ls_epoch.append(ls_BC2_t.item())     # unused in Loss
            ls_BC5_t = torch.mean(torch.square(BC5_t)); ls_epoch.append(ls_BC5_t.item())
            ls_BC7_t = torch.mean(torch.square(BC7_t)); ls_epoch.append(ls_BC7_t.item())
            
            if i <= 1000:
                Loss = ls_Work_m + ls_BC6_m + ls_BC10_m + ls_BC14_m + ls_BC18_m + \
                    ls_Work_t + ls_BC5_t + ls_BC7_t + \
                    ls_u_x_FE + ls_u_y_FE + ls_T_FE + ls_q_x_FE + ls_q_y_FE + ls_sigma_x_FE + ls_sigma_y_FE + ls_sigma_xy_FE
            elif i <= 6000:
                Loss = ls_u_x_FE + ls_u_y_FE + ls_T_FE + ls_q_x_FE + ls_q_y_FE + ls_sigma_x_FE + ls_sigma_y_FE + ls_sigma_xy_FE
            elif i <= 10000:
                Loss = ls_Work_m + ls_BC6_m + ls_BC10_m + ls_BC14_m + ls_BC18_m + \
                    ls_Work_t + ls_BC5_t + ls_BC7_t + \
                    ls_u_x_FE + ls_u_y_FE + ls_T_FE + ls_q_x_FE + ls_q_y_FE + ls_sigma_x_FE + ls_sigma_y_FE + ls_sigma_xy_FE
            elif i <= 14000:
                if i == 10000+1:
                    new_lr = 5e-4
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                Loss = ls_u_x_FE + ls_u_y_FE + ls_T_FE + ls_q_x_FE + ls_q_y_FE + ls_sigma_x_FE + ls_sigma_y_FE + ls_sigma_xy_FE
            elif i <= 20000:
                if i == 16000+1:
                    new_lr = 6e-4
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                Loss = ls_Work_m + ls_BC6_m + ls_BC10_m + ls_BC14_m + ls_BC18_m + \
                    ls_Work_t + ls_BC5_t + ls_BC7_t + \
                    ls_u_x_FE + ls_u_y_FE + ls_T_FE + ls_q_x_FE + ls_q_y_FE + ls_sigma_x_FE + ls_sigma_y_FE + ls_sigma_xy_FE
            elif i <= 22000:
                if i == 20000+1:
                    new_lr = 1e-4
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = new_lr
                Loss = ls_BC6_m + ls_BC10_m + ls_BC14_m + ls_BC18_m + \
                    ls_BC5_t + ls_BC7_t + \
                    ls_u_x_FE + ls_u_y_FE + ls_T_FE + ls_q_x_FE + ls_q_y_FE + ls_sigma_x_FE + ls_sigma_y_FE + ls_sigma_xy_FE
            else:
                Loss = ls_Work_m + ls_BC6_m + ls_BC10_m + ls_BC14_m + ls_BC18_m + \
                    ls_Work_t + ls_BC5_t + ls_BC7_t + \
                    ls_u_x_FE + ls_u_y_FE + ls_T_FE + ls_q_x_FE + ls_q_y_FE + ls_sigma_x_FE + ls_sigma_y_FE + ls_sigma_xy_FE
            Loss.backward()
            optimizer.step()
            # if i >= 4000 and i <= 20000: scheduler.step()
            
            self.history[self.cfg.loss_terms[0]].append(Loss.item())
            for loss_idx in range(1, len(self.cfg.loss_terms)):
                self.history[self.cfg.loss_terms[loss_idx]].append(ls_epoch[loss_idx-1])
            if (i + 1) % 50 == 0:
                print("iter:", i+1, " Loss:", Loss)
                print(f"Work_m: {ls_epoch[0]:.6e}", f"BC6_m: {ls_epoch[1]:.6e}", f"BC10_m: {ls_epoch[2]:.6e}", f"BC14_m: {ls_epoch[3]:.6e}", f"BC18_m: {ls_epoch[4]:.6e}")
                print(f"Work_t: {ls_epoch[5]:.6e}", f"BC1_t: {ls_epoch[6]:.6e}", f"BC2_t: {ls_epoch[7]:.6e}", f"BC5_t: {ls_epoch[8]:.6e}", f"BC7_t: {ls_epoch[9]:.6e}")



T_n   = Functional(cfg.input_dim, cfg.output_dim, cfg.hidden_layers, cfg.activation)
q_x_O = Functional(cfg.input_dim, cfg.output_dim, cfg.hidden_layers, cfg.activation)
q_y_O = Functional(cfg.input_dim, cfg.output_dim, cfg.hidden_layers, cfg.activation)

u_xn  = Functional(cfg.input_dim, cfg.output_dim, cfg.hidden_layers, cfg.activation)
u_yn  = Functional(cfg.input_dim, cfg.output_dim, cfg.hidden_layers, cfg.activation)

sig_x_O  = Functional(cfg.input_dim, cfg.output_dim, cfg.hidden_layers, cfg.activation)
sig_y_O  = Functional(cfg.input_dim, cfg.output_dim, cfg.hidden_layers, cfg.activation)
sig_xy_O = Functional(cfg.input_dim, cfg.output_dim, cfg.hidden_layers, cfg.activation)

models = dict()
models["Tn"] = T_n
models["q_x_O"] = q_x_O
models["q_y_O"] = q_y_O
models["u_xn"] = u_xn
models["u_yn"] = u_yn
models["sig_x_O"] = sig_x_O
models["sig_y_O"] = sig_y_O
models["sig_xy_O"] = sig_xy_O

Load_data = pd.read_csv(cfg.datafile)

x1 = pd.DataFrame(Load_data, columns=['X']).to_numpy(dtype=np.float32)
y1 = pd.DataFrame(Load_data, columns=['Y']).to_numpy(dtype=np.float32)
e1 = pd.DataFrame(Load_data, columns=['solid.E (Pa)']).to_numpy(dtype=np.float32)
k1 = pd.DataFrame(Load_data, columns=['ht.kxx (W/(m*K))']).to_numpy(dtype=np.float32)
x_data = x1
y_data = y1
E_data = e1
K_data = k1

### adding collocation point to BC Top
for iii in range(0,100):
    x_new = np.random.random()
    y_new = 1
    e_new = 10  # @TODO: 注意 1:100=1; 100:1=100
    k_new = 1
    x1 = np.append(x1, x_new)
    y1 = np.append(y1, y_new)
    e1 = np.append(e1, e_new)
    k1 = np.append(k1, k_new)
### adding collocation point to BC bottom
for iiii in range(0,100):
    x_new = np.random.random()
    y_new = 0
    e_new = 10
    k_new = 1
    x1 = np.append(x1, x_new)
    y1 = np.append(y1, y_new)
    e1 = np.append(e1, e_new)
    k1 = np.append(k1, k_new)
### adding collocation point to BC right
for iiii in range(0,100):
    x_new = 1
    y_new = np.random.random()
    e_new = 10
    k_new = 1
    x1 = np.append(x1, x_new)
    y1 = np.append(y1, y_new)
    e1 = np.append(e1, e_new)
    k1 = np.append(k1, k_new)
### adding collocation point to BC left
for iiii in range(0,100):
    x_new = 0
    y_new = np.random.random()
    e_new = 10
    k_new = 1
    x1 = np.append(x1, x_new)
    y1 = np.append(y1, y_new)
    e1 = np.append(e1, e_new)
    k1 = np.append(k1, k_new)

x1 = to_tensor(x1, requires_grad=True).reshape((-1, 1))
y1 = to_tensor(y1, requires_grad=True).reshape((-1, 1))
e1 = to_tensor(e1).reshape((-1, 1))
k1 = to_tensor(k1).reshape((-1, 1))

def boundary_value_mask(x,y):
    M = np.zeros(x.shape)
    for i in range(len(x)):
        if x[i]==0 or x[i]==1 or y[i]==0 or y[i]==1:
            M[i] = 1
    return M

def boundary_value_mask_right(x,y):
    M = np.zeros(x.shape)
    for i in range(len(x)):
        if x[i]==1:
            M[i] = 1
    return M

def boundary_value_mask_left(x,y):
    M = np.zeros(x.shape)
    for i in range(len(x)):
        if x[i]==0:
            M[i] = 1
    return M

def boundary_value_mask_top(x,y):
    M = np.zeros(x.shape)
    for i in range(len(x)):
        if y[i]==1:
            M[i] = 1
    return M

def boundary_value_mask_bottom(x,y):
    M = np.zeros(x.shape)
    for i in range(len(x)):
        if y[i]==0:
            M[i] = 1
    return M

M_bound = boundary_value_mask(x1,y1)
M_bound_right = boundary_value_mask_right(x1,y1)
M_bound_left = boundary_value_mask_left(x1,y1)
M_bound_top = boundary_value_mask_top(x1,y1)
M_bound_bottom = boundary_value_mask_bottom(x1,y1)
M_domain = np.array((M_bound==0)*1.0)

nb = np.sum(M_bound).item()
nb_r = np.sum(M_bound_right).item()
nb_l = np.sum(M_bound_left).item()
nb_t = np.sum(M_bound_top).item()
nb_b = np.sum(M_bound_bottom).item()
nd = np.sum(M_domain).item()

M_bound_right = to_tensor(M_bound_right).reshape((-1, 1))
M_bound_left = to_tensor(M_bound_left).reshape((-1, 1))
M_bound_top = to_tensor(M_bound_top).reshape((-1, 1))
M_bound_bottom = to_tensor(M_bound_bottom).reshape((-1, 1))
M_domain = to_tensor(M_domain).reshape((-1, 1))

training_util = TrainingUtils(models, x1, y1, nb, nb_r, nb_l, nb_t, nb_b, nd, M_bound_right, M_bound_left, M_bound_top, M_bound_bottom, M_domain, e1, k1, cfg, sample_FE)

training_util.func1_train()

result_FE = pd.read_csv("data_thermomechanic/b-i=10-1,16-refined.csv")
# result_FE = pd.read_csv(cfg.datafile)

x_data = pd.DataFrame(result_FE, columns=['X']).to_numpy(dtype=np.float32)
y_data = pd.DataFrame(result_FE, columns=['Y']).to_numpy(dtype=np.float32)
E_data = pd.DataFrame(result_FE, columns=['solid.E (Pa)']).to_numpy(dtype=np.float32)
K_data = pd.DataFrame(result_FE, columns=['ht.kxx (W/(m*K))']).to_numpy(dtype=np.float32)
u_x_FE =  pd.DataFrame(result_FE, columns= ['u (m)'])
u_y_FE =  pd.DataFrame(result_FE, columns= ['v (m)'])
sigma_x_FE =  pd.DataFrame(result_FE, columns= ['solid.sxx (N/m^2)'])
sigma_y_FE =  pd.DataFrame(result_FE, columns= ['solid.syy (N/m^2)'])
sigma_xy_FE =  pd.DataFrame(result_FE, columns= ['solid.sxy (N/m^2)'])
T_FE =  pd.DataFrame(result_FE, columns= ['T (K)'])
q_x_FE =  pd.DataFrame(result_FE, columns= ['ht.tfluxx (W/m^2)'])
q_y_FE =  pd.DataFrame(result_FE, columns= ['ht.tfluxy (W/m^2)'])

u_x_FE = np.array(u_x_FE).flatten()
u_y_FE = np.array(u_y_FE).flatten()
sigma_x_FE = np.array(sigma_x_FE).flatten()
sigma_y_FE = np.array(sigma_y_FE).flatten()
sigma_xy_FE = np.array(sigma_xy_FE).flatten()

T_FE = np.array(T_FE).flatten()
q_x_FE = np.array(q_x_FE).flatten()
q_y_FE = np.array(q_y_FE).flatten()

history = training_util.history

x_data = to_tensor(x_data, requires_grad=True).reshape((-1, 1)).to(device)
y_data = to_tensor(y_data, requires_grad=True).reshape((-1, 1)).to(device)
X_test = torch.hstack((x_data, y_data))
E_data = to_tensor(E_data, requires_grad=True).reshape((-1, 1)).to(device)
K_data = to_tensor(K_data, requires_grad=True).reshape((-1, 1)).to(device)

u_xn_pred = training_util.u_xn(X_test)
u_yn_pred = training_util.u_yn(X_test)
u_x_pred = u_xn_pred * x_data * (x_data - 1.0)
u_y_pred = u_yn_pred  * y_data * (y_data - 1.0)

sig_x_O_pred = training_util.sig_x_O(X_test)
sig_y_O_pred = training_util.sig_y_O(X_test)
sig_xy_O_pred = training_util.sig_xy_O(X_test)

Tn_pred = training_util.Tn(X_test)
q_x_O_pred = training_util.q_x_O(X_test)
q_y_O_pred = training_util.q_y_O(X_test)
T_pred = Tn_pred * x_data * (x_data - 1.0) + 1.0 - x_data
gr_T_x_pred = gradients(T_pred, x_data)[0]
gr_T_y_pred = gradients(T_pred, y_data)[0]
q_x_pred = (-K_data) * gr_T_x_pred
q_y_pred = (-K_data) * gr_T_y_pred
strain_x_pred  = gradients(u_x_pred, x_data)[0]
strain_y_pred  = gradients(u_y_pred, y_data)[0]
strain_xy_pred = (gradients(u_x_pred, y_data)[0] + gradients(u_y_pred, x_data)[0]) * 0.5
strain_T_x_pred = (T_pred - T_0) * alpha
strain_T_y_pred = (T_pred - T_0) * alpha
sigma_x_pred   = (E_data/((1+nu)*(1-2*nu)))*((1-nu)*strain_x_pred + nu*strain_y_pred) - E_data*strain_T_x_pred/(1-2*nu)
sigma_y_pred   = (E_data/((1+nu)*(1-2*nu)))*((1-nu)*strain_y_pred + nu*strain_x_pred) - E_data*strain_T_y_pred/(1-2*nu)
sigma_xy_pred  = (E_data/((1+nu)*(1-2*nu)))*strain_xy_pred*(1-2*nu)

x_data = to_numpy(x_data).flatten()
y_data = to_numpy(y_data).flatten()
T_pred = to_numpy(T_pred).flatten()
q_x_pred = to_numpy(q_x_pred).flatten()
q_y_pred = to_numpy(q_y_pred).flatten()
u_x_pred = to_numpy(u_x_pred).flatten()
u_y_pred = to_numpy(u_y_pred).flatten()
sigma_x_pred = to_numpy(sigma_x_pred).flatten()
sigma_y_pred = to_numpy(sigma_y_pred).flatten()
sigma_xy_pred = to_numpy(sigma_xy_pred).flatten()
strain_x_pred = to_numpy(strain_x_pred).flatten()
strain_y_pred = to_numpy(strain_y_pred).flatten()
strain_xy_pred = to_numpy(strain_xy_pred).flatten()
sig_x_O_pred = to_numpy(sig_x_O_pred).flatten()
sig_y_O_pred = to_numpy(sig_y_O_pred).flatten()
sig_xy_O_pred = to_numpy(sig_xy_O_pred).flatten()
q_x_O_pred = to_numpy(q_x_O_pred).flatten()
q_y_O_pred = to_numpy(q_y_O_pred).flatten()

T_error = np.abs(T_pred- T_FE)
q_x_error = np.abs(q_x_pred- q_x_FE)
q_y_error = np.abs(q_y_pred- q_y_FE)

u_x_error = np.abs(u_x_FE - u_x_pred)
u_y_error = np.abs(u_y_FE - u_y_pred)
sigma_x_error   = np.abs(sigma_x_FE - sigma_x_pred)
sigma_y_error   = np.abs(sigma_y_FE - sigma_y_pred)
sigma_xy_error  = np.abs(sigma_xy_FE - sigma_xy_pred)

import pickle

results = dict()
results["x_data"] = x_data
results["y_data"] = y_data
results["u_x_pred"] = u_x_pred
results["u_y_pred"] = u_y_pred
results["strain_x_pred"] = strain_x_pred
results["strain_y_pred"] = strain_y_pred
results["strain_xy_pred"] = strain_xy_pred
results["sigma_x_pred"] = sigma_x_pred
results["sigma_y_pred"] = sigma_y_pred
results["sigma_xy_pred"] = sigma_xy_pred
results["T_pred"] = T_pred
results["q_x_pred"] = q_x_pred
results["q_y_pred"] = q_y_pred
results["u_x_FE"] = u_x_FE
results["u_y_FE"] = u_y_FE
results["sigma_x_FE"] = sigma_x_FE
results["sigma_y_FE"] = sigma_y_FE
results["sigma_xy_FE"] = sigma_xy_FE
results["T_FE"] = T_FE
results["q_x_FE"] = q_x_FE
results["q_y_FE"] = q_y_FE
results["history"] = history

# with open('results/10-1-data01/data_mspinn.pkl', 'wb') as f:
#     pickle.dump(results, f)

fig = plt.figure(figsize=(12, 7))
itter = 0 
for word, loss in history.items():
    plt.semilogy(np.array(loss), label=cfg.loss_terms[itter])
    itter+=1
plt.legend()
plt.xlabel('epochs')
plt.ylabel('loss')
# plt.savefig('data_thermomechanic/data/losses')

fig = plt.figure(figsize=(12, 7))
plt.semilogy(history['Total_loss'], label='Total_loss')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('Total-loss')
# plt.savefig('data_thermomechanic/data/totalloss')


fig,ax = plt.subplots(2,3,figsize=(12,8))
plt.colorbar(ax[0,0].scatter(x_data, y_data, c=u_x_pred, cmap='jet', s=1, vmax=np.max(u_x_FE), vmin=np.min(u_x_FE)),ax=ax[0,0])
plt.colorbar(ax[0,1].scatter(x_data, y_data, c=u_x_FE, cmap='jet', s=1),ax=ax[0,1])
plt.colorbar(ax[0,2].scatter(x_data, y_data, c=u_x_error, cmap='jet', s=1),ax=ax[0,2])
plt.colorbar(ax[1,0].scatter(x_data, y_data, c=u_y_pred, cmap='jet', s=1, vmax=np.max(u_y_FE), vmin=np.min(u_y_FE)),ax=ax[1,0])
plt.colorbar(ax[1,1].scatter(x_data, y_data, c=u_y_FE, cmap='jet', s=1),ax=ax[1,1])
plt.colorbar(ax[1,2].scatter(x_data, y_data, c=u_y_error, cmap='jet', s=1),ax=ax[1,2])
ax[0,0].set_title('u_x_pred')
ax[0,1].set_title('u_x_FE')
ax[0,2].set_title('u_x_error')
ax[1,0].set_title('u_y_pred')
ax[1,1].set_title('u_y_FE')
ax[1,2].set_title('u_y_error')
ax[0,1].set_yticks([])
ax[0,2].set_yticks([])
ax[1,1].set_yticks([])
ax[1,2].set_yticks([])
# print("Mean Abs Error of u_x = " , np.mean(u_x_error))
# print("Mean Abs Error of u_y = " , np.mean(u_y_error))
# plt.savefig('data_thermomechanic/data/disp')

fig,ax = plt.subplots(1,3,figsize=(15,4.5))
plt.colorbar(ax[0].scatter(x_data, y_data, c=strain_x_pred, cmap='jet', s=2),ax=ax[0])
plt.colorbar(ax[1].scatter(x_data, y_data, c=strain_y_pred, cmap='jet', s=2),ax=ax[1])
plt.colorbar(ax[2].scatter(x_data, y_data, c=strain_xy_pred, cmap='jet', s=2),ax=ax[2])
ax[0].set_title('strain_x_pred')
ax[1].set_title('strain_y_pred')
ax[2].set_title('strain_xy_pred')

fig,ax = plt.subplots(3,4,figsize=(16,12))
plt.colorbar(ax[0,0].scatter(x_data, y_data, c=sigma_x_pred, cmap='jet', s=1, vmax=np.max(sigma_x_FE), vmin=np.min(sigma_x_FE)),ax=ax[0,0])
plt.colorbar(ax[0,1].scatter(x_data, y_data, c=sig_x_O_pred, cmap='jet', s=1),ax=ax[0,1])
plt.colorbar(ax[0,2].scatter(x_data, y_data, c=sigma_x_FE, cmap='jet', s=1),ax=ax[0,2])
plt.colorbar(ax[0,3].scatter(x_data, y_data, c=sigma_x_error, cmap='jet', s=1),ax=ax[0,3])
plt.colorbar(ax[1,0].scatter(x_data, y_data, c=sigma_y_pred, cmap='jet', s=1, vmax=np.max(sigma_y_FE), vmin=np.min(sigma_y_FE)),ax=ax[1,0])
plt.colorbar(ax[1,1].scatter(x_data, y_data, c=sig_y_O_pred, cmap='jet', s=1),ax=ax[1,1])
plt.colorbar(ax[1,2].scatter(x_data, y_data, c=sigma_y_FE, cmap='jet', s=1),ax=ax[1,2])
plt.colorbar(ax[1,3].scatter(x_data, y_data, c=sigma_y_error, cmap='jet', s=1),ax=ax[1,3])
plt.colorbar(ax[2,0].scatter(x_data, y_data, c=sigma_xy_pred, cmap='jet', s=1, vmax=np.max(sigma_xy_FE), vmin=np.min(sigma_xy_FE)),ax=ax[2,0])
plt.colorbar(ax[2,1].scatter(x_data, y_data, c=sig_xy_O_pred, cmap='jet', s=1),ax=ax[2,1])
plt.colorbar(ax[2,2].scatter(x_data, y_data, c=sigma_xy_FE, cmap='jet', s=1),ax=ax[2,2])
plt.colorbar(ax[2,3].scatter(x_data, y_data, c=sigma_xy_error, cmap='jet', s=1),ax=ax[2,3])
ax[0,0].set_title('sigma_x_pred')
ax[0,1].set_title('sig_x_O_pred')
ax[0,2].set_title('sigma_x_FE')
ax[0,3].set_title('sigma_x_error')
ax[1,0].set_title('sigma_y_pred')
ax[1,1].set_title('sig_y_O_pred')
ax[1,2].set_title('sigma_y_FE')
ax[1,3].set_title('sigma_y_error')
ax[2,0].set_title('sigma_xy_pred')
ax[2,1].set_title('sig_xy_O_pred')
ax[2,2].set_title('sigma_xy_FE')
ax[2,3].set_title('sigma_xy_error')
ax[0,1].set_yticks([])
ax[0,2].set_yticks([])
ax[0,3].set_yticks([])
ax[1,2].set_yticks([])
ax[1,1].set_yticks([])
ax[1,3].set_yticks([])
ax[2,1].set_yticks([])
ax[2,2].set_yticks([])
ax[2,3].set_yticks([])
# print("Mean Abs Error of sigma_x = " , np.mean(sigma_x_error))
# print("Mean Abs Error of sigma_y = " , np.mean(sigma_y_error))
# print("Mean Abs Error of sigma_xy = " , np.mean(sigma_xy_error))
# plt.savefig('data_thermomechanic/data/stress')

fac = 1
x_new_pred = x_data + u_x_pred*fac
y_new_pred = y_data + u_y_pred*fac
x_new_FE = x_data + u_x_FE*fac
y_new_FE = y_data + u_y_FE*fac

fig,ax = plt.subplots(1,3,figsize=(20,7))
ax[0].plot(x_data, y_data,'b.')
ax[1].plot(x_new_pred, y_new_pred, 'r.')
ax[2].plot(x_new_FE, y_new_FE, 'r.')
ax[0].set_title('Undeformed')
ax[1].set_title('Deformed_pred')
ax[2].set_title('Deformed_FE')
# plt.savefig('data_thermomechanic/data/deform')

fig,ax = plt.subplots(1,3,figsize=(15,4))
plt.colorbar(ax[0].scatter(x_data, y_data, c=T_pred, cmap='jet', s=1),ax=ax[0])
plt.colorbar(ax[1].scatter(x_data, y_data, c=T_FE, cmap='jet', s=1),ax=ax[1])
plt.colorbar(ax[2].scatter(x_data, y_data, c=T_error, cmap='jet', s=1),ax=ax[2])
ax[0].set_title('T_Pred')
ax[1].set_title('T_FE')
ax[2].set_title('T_Error')
# plt.savefig('data_thermomechanic/data/temp')

fig,ax = plt.subplots(2,4,figsize=(18,8))
plt.colorbar(ax[0,0].scatter(x_data, y_data, c=q_x_pred, cmap='jet', s=1, vmax=np.max(q_x_FE), vmin=np.min(q_x_FE)),ax=ax[0,0])
plt.colorbar(ax[0,1].scatter(x_data, y_data, c=q_x_O_pred, cmap='jet', s=1),ax=ax[0,1])
plt.colorbar(ax[0,2].scatter(x_data, y_data, c=q_x_FE, cmap='jet', s=1),ax=ax[0,2])
plt.colorbar(ax[0,3].scatter(x_data, y_data, c=q_x_error, cmap='jet', s=1),ax=ax[0,3])

plt.colorbar(ax[1,0].scatter(x_data, y_data, c=q_y_pred, cmap='jet', s=1, vmax=np.max(q_y_FE), vmin=np.min(q_y_FE)),ax=ax[1,0])
plt.colorbar(ax[1,1].scatter(x_data, y_data, c=q_y_O_pred, cmap='jet', s=1),ax=ax[1,1])
plt.colorbar(ax[1,2].scatter(x_data, y_data, c=q_y_FE, cmap='jet', s=1),ax=ax[1,2])
plt.colorbar(ax[1,3].scatter(x_data, y_data, c=q_y_error, cmap='jet', s=1),ax=ax[1,3])

ax[0,0].set_title('q_x_pred')
ax[0,1].set_title('q_x_O_pred')
ax[0,2].set_title('q_x_FE')
ax[1,0].set_title('q_y_pred')
ax[1,1].set_title('q_y_O_pred')
ax[1,2].set_title('q_y_FE')
ax[0,1].set_yticks([])
ax[0,2].set_yticks([])
ax[1,1].set_yticks([])
ax[1,2].set_yticks([])