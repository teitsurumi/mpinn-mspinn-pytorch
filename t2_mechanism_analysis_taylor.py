import sys
from pathlib import Path
root_dir = Path(__file__).resolve().parent
sys.path.append(str(root_dir))
tmp_results_dir = root_dir / '.tmp_results'
tmp_results_dir.mkdir(parents=True, exist_ok=True)
t2_results_dir = tmp_results_dir / 't2_mechanism_analysis_taylor'
t2_results_dir.mkdir(parents=True, exist_ok=True)

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pickle

plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset":'stix',
    "font.size": 14,
})

np.random.seed(1234)
torch.manual_seed(1234)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

from utils.data_structure import gradients, to_numpy, to_tensor, remove_mask


def training_set_visualization():

    fig = plt.figure(figsize=(7, 1.4))
    ax = fig.subplots(1, 2)
    plt.tight_layout(rect=[-0.036, -0.14, 1.04, 1.12])
    plt.subplots_adjust(wspace=0.4)
    x = np.linspace(0, 5, 200)
    y = 1/np.sqrt(2*np.pi) * np.exp(-(x*5)**2/2) * 2.5
    dy_dx = -31.25 * np.sqrt(2) * x * np.exp(-25*x**2/2) / np.sqrt(np.pi)
    ax[0].plot(x, y, c="#feca57", linewidth=2, label=r"f(x)", zorder=1)
    ax[1].plot(x, dy_dx, c="#1e90ff", linewidth=2, alpha=0.5, label=r"$df/dx$", zorder=1)
    
    x = np.linspace(0, 5, 50)
    y = 1/np.sqrt(2*np.pi) * np.exp(-(x*5)**2/2) * 2.5
    dy_dx = -31.25 * np.sqrt(2) * x * np.exp(-25*x**2/2) / np.sqrt(np.pi)
    x = np.delete(x, [1,2,4,5,7,10,15])
    y = np.delete(y, [1,2,4,5,7,10,15])
    dy_dx = np.delete(dy_dx, [1,2,4,5,7,10,15])
    ax[0].scatter(x, np.zeros_like(x), s=20, alpha=0.7, marker='x', c="#d23918", label="sample points", zorder=1)
    ax[1].scatter(x, np.zeros_like(x), s=20, alpha=0.7, marker='x', c="#d23918", label="sample points", zorder=1)
    for i in range(x.shape[0]):
        ax[0].plot([x[i], x[i]], [0, y[i]], '--', lw=1, c="#d23918", alpha=0.4)
        ax[1].plot([x[i], x[i]], [0, dy_dx[i]], '--', lw=1, c="#d23918", alpha=0.4)
    ax[0].set_xticks([0, 1, 2, 3, 4, 5])
    ax[0].set_yticks([0, 1])
    ax[1].set_xticks([0, 1, 2, 3, 4, 5])
    ax[1].set_yticks([0, -3])
    ax[0].set_ylim([-0.1, 1.1])
    ax[1].set_ylim([-3.3, 0.3])
    plt.savefig(t2_results_dir / 'sample2.png', dpi=300)


class pinn(nn.Module):
    def __init__(
        self, x: np.ndarray, y: np.ndarray, dy_dx: np.ndarray, ddy_dxx: np.ndarray, dddy_dxxx: np.ndarray,
        iters: int, evals: int, closure_choice: int
    ):
        super(pinn, self).__init__()
        device = torch.device("cuda:0")
        self.ls = 0

        self.__ITERS = iters
        self.__EVALS = evals
        self.__CHOICE = closure_choice

        self.net_y = self.__network()

        self.mse = nn.MSELoss().cuda("cuda:0")

        self._num = x.shape[0]
        self.x   = torch.tensor(x, dtype=torch.float32, requires_grad=True).reshape((self._num, 1)).to(device)
        self.y   = torch.tensor(y, dtype=torch.float32, requires_grad=True).reshape((self._num, 1)).to(device)
        self.dy_dx = torch.tensor(dy_dx, dtype=torch.float32).reshape((self._num, 1)).to(device)
        self.ddy_dxx = torch.tensor(ddy_dxx, dtype=torch.float32).reshape((self._num, 1)).to(device)
        self.dddy_dxxx = torch.tensor(dddy_dxxx, dtype=torch.float32).reshape((self._num, 1)).to(device)
        
        # self.optimizer = torch.optim.LBFGS(
        #     self.net_y.parameters(),
        #     lr=0.1,
        #     max_iter=self.__ITERS,
        #     max_eval=self.__EVALS,
        #     tolerance_grad=1e-9,
        #     tolerance_change=1e-11,
        #     line_search_fn="strong_wolfe"
        # )
        
        self.optimizer = torch.optim.Adam(
            self.net_y.parameters(),
            lr=0.001,
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1000, gamma=0.8)
        
        self.iter = 0
        self.history = torch.zeros((5, self.__ITERS), dtype=torch.float32).to(device)

    # ----------------------- BEGIN NETWORKS -----------------------
    def __network(self):
        net = nn.Sequential()
        net.add_module('Linear_layer_1', nn.Linear(1, 15))
        net.add_module('Tanh_layer_1', nn.Tanh())
        for num in range(2, 5):
            net.add_module('Linear_layer_%d' % (num), nn.Linear(15, 15))
            net.add_module('Tanh_layer_%d' % (num), nn.Tanh())
            # net.add_module('Dropout_layer_%d' % (num), nn.Dropout(0.1))
        net.add_module('Linear_layer_final', nn.Linear(15, 1))
        net = net.cuda("cuda:0")
        return net
    # ----------------------- END NETWORKS -----------------------

    # ----------------------- BEGIN FUNCTIONS -----------------------
    def function(self, x: torch.Tensor):
        y = self.net_y(x)
        dy_dx = gradients(y, x)[0]
        ddy_dxx = gradients(dy_dx, x)[0]
        return y, dy_dx, ddy_dxx
    def function2(self, x: torch.Tensor):
        y, dy_dx, ddy_dxx = self.function(x)
        dddy_dxxx = gradients(ddy_dxx, x)[0]
        return y, dy_dx, ddy_dxx, dddy_dxxx
    # ----------------------- END FUNCTIONS -----------------------


    # -------------------------------------- CLOSURE --------------------------------------
    def closure0(self):
        self.optimizer.zero_grad()
        y, _, _ = self.function(self.x)
        self.ls = self.mse(y, self.y)
        self.ls.backward()
        self.iter += 1
        if not self.iter % 50:
            print(self.ls)
        return self.ls
    
    def closure1(self):
        self.optimizer.zero_grad()
        y, dy_dx, _ = self.function(self.x)
        self.ls = self.mse(y, self.y) + self.mse(dy_dx, self.dy_dx)
        self.ls.backward()
        self.iter += 1
        if not self.iter % 50:
            print(self.ls)
        return self.ls
    
    def closure2(self):
        self.optimizer.zero_grad()
        y, dy_dx, ddy_dxx, dddy_dxxx = self.function2(self.x)
        ls0 = self.mse(y, self.y)
        ls1 = self.mse(dy_dx, self.dy_dx)
        ls2 = self.mse(ddy_dxx, self.ddy_dxx)
        ls3 = self.mse(dddy_dxxx, self.dddy_dxxx)
        self.ls = ls0 + ls1 + ls2
        self.ls.backward()
        self.history[0, self.iter] = self.ls.item()
        self.history[1, self.iter] = ls0.item()
        self.history[2, self.iter] = ls1.item()
        self.history[3, self.iter] = ls2.item()
        self.history[4, self.iter] = ls3.item()
        self.iter += 1
        if not self.iter % 50:
            print(self.ls)
        return self.ls

    def closure3(self):
        self.optimizer.zero_grad()
        y, dy_dx, ddy_dxx, dddy_dxxx = self.function2(self.x)
        ls0 = self.mse(y, self.y)
        ls1 = self.mse(dy_dx, self.dy_dx)
        ls2 = self.mse(ddy_dxx, self.ddy_dxx)
        ls3 = self.mse(dddy_dxxx, self.dddy_dxxx)
        self.ls = ls0 + ls1 + ls2 + ls3
        self.ls.backward()
        self.history[0, self.iter] = self.ls.item()
        self.history[1, self.iter] = ls0.item()
        self.history[2, self.iter] = ls1.item()
        self.history[3, self.iter] = ls2.item()
        self.history[4, self.iter] = ls3.item()
        self.iter += 1
        if not self.iter % 50:
            print(self.ls)
        return self.ls
    # -------------------------------------- END CLOSURE --------------------------------------


    def train(self):
        """training"""
        print(f"========================== CLOSURE {self.__CHOICE} ==========================")
        self.net_y.train()
        if self.__CHOICE == 0:
            closure=self.closure0
        elif self.__CHOICE == 1:
            closure=self.closure1
        elif self.__CHOICE == 2:
            closure=self.closure2
        elif self.__CHOICE == 3:
            closure=self.closure3
        for i in range(self.__ITERS):
            self.optimizer.step(closure=closure)
    
    def get_closure_choice(self):
        return self.__CHOICE


def main_procedure0(pinn_model: pinn):
    pinn_model.train()
    
    x_test = np.linspace(0, 5, 500).reshape(500, 1)
    y_test = 1/np.sqrt(2*np.pi) * np.exp(-(x_test*5)**2/2) * 2.5
    dy_dx_test = -31.25 * np.sqrt(2) * x_test * np.exp(-25*x_test**2/2) / np.sqrt(np.pi)
    ddy_dxx_test = 781.25*np.sqrt(2)*x_test**2*np.exp(-25*x_test**2/2)/np.sqrt(np.pi) - 31.25*np.sqrt(2)*np.exp(-25*x_test**2/2)/np.sqrt(np.pi)
    dddy_dxxx_test = -19531.25*np.sqrt(2)*x_test**3*np.exp(-25*x_test**2/2)/np.sqrt(np.pi) + 2343.75*np.sqrt(2)*x_test*np.exp(-25*x_test**2/2)/np.sqrt(np.pi)
    
    x_test = torch.tensor(x_test, dtype=torch.float32, requires_grad=True).reshape((x_test.shape[0], 1)).to(device)
    y_pred, dy_dx_pred, ddy_dxx_pred, dddy_dxxx_pred = pinn_model.function2(x_test)
    y_test = torch.tensor(y_test, dtype=torch.float32, requires_grad=True).reshape((y_test.shape[0], 1)).to(device)
    dy_dx_test = torch.tensor(dy_dx_test, dtype=torch.float32, requires_grad=True).reshape((dy_dx_test.shape[0], 1)).to(device)
    ddy_dxx_test = torch.tensor(ddy_dxx_test, dtype=torch.float32, requires_grad=True).reshape((dy_dx_test.shape[0], 1)).to(device)
    dddy_dxxx_test = torch.tensor(dddy_dxxx_test, dtype=torch.float32, requires_grad=True).reshape((dy_dx_test.shape[0], 1)).to(device)
    
    mse = nn.MSELoss().cuda("cuda:0")
    y_mse = mse(y_test, y_pred)
    dy_dx_mse = mse(dy_dx_test, dy_dx_pred)
    ddy_dxx_mse = mse(ddy_dxx_test, ddy_dxx_pred)
    dddy_dxxx_mse = mse(dddy_dxxx_test, dddy_dxxx_pred)
    
    x_test = to_numpy(x_test)
    y_test = to_numpy(y_test)
    dy_dx_test = to_numpy(dy_dx_test)
    ddy_dxx_test = to_numpy(ddy_dxx_test)
    y_pred = to_numpy(y_pred)
    dy_dx_pred = to_numpy(dy_dx_pred)
    ddy_dxx_pred = to_numpy(ddy_dxx_pred)
    y_mse = to_numpy(y_mse)
    dy_dx_mse = to_numpy(dy_dx_mse)
    ddy_dxx_mse = to_numpy(ddy_dxx_mse)
    y_rmse = np.sqrt(y_mse)
    dy_dx_rmse = np.sqrt(dy_dx_mse)
    ddy_dxx_rmse = np.sqrt(ddy_dxx_mse)

    fig = plt.figure(figsize=(16, 4))
    ax = fig.subplots(1, 4)
    ax[0].scatter(x_test, y_test - y_pred, s=1, label=r"$f(x)$")
    ax[0].scatter(x_test, dy_dx_test - dy_dx_pred, s=1, label=r"$df/dx$")
    ax[0].scatter(x_test, ddy_dxx_test - ddy_dxx_pred, s=1, label=r"$d^2f/dx^2$")
    ax[0].legend()
    ax[1].plot(x_test, y_test, '-', label=r"Ground Truth")
    ax[1].plot(x_test, y_pred, '-', label=r"Prediction")
    ax[1].legend()
    ax[2].plot(x_test, dy_dx_test, '-', label=r"Ground Truth")
    ax[2].plot(x_test, dy_dx_pred, '-', label=r"Prediction")
    ax[2].legend()
    ax[3].plot(x_test, ddy_dxx_test, '-', label=r"Ground Truth")
    ax[3].plot(x_test, ddy_dxx_pred, '-', label=r"Prediction")
    ax[3].legend()
    plt.savefig(t2_results_dir / f'pinn_closure_id{pinn_model.get_closure_choice()}.png', dpi=300)
    plt.close()
    print(y_rmse, dy_dx_rmse, ddy_dxx_rmse)
    history = pinn_model.history
    
    results = dict()
    results["x_test"] = x_test
    results["y_test"] = y_test
    results["y_pred"] = y_pred
    results["dy_dx_test"] = dy_dx_test
    results["dy_dx_pred"] = dy_dx_pred
    results["history"] = to_numpy(history)
    with open(t2_results_dir / f'pinn_closure_id{pinn_model.get_closure_choice()}.pkl', "wb") as f:
        pickle.dump(results, f)
    
    return x_test, y_test, dy_dx_test, y_pred, dy_dx_pred, y_rmse, dy_dx_rmse, history


if __name__ == '__main__':
    training_set_visualization()
    
    x = np.linspace(0, 5, 50)
    y = 1/np.sqrt(2*np.pi) * np.exp(-(x*5)**2/2) * 2.5
    dy_dx = -31.25 * np.sqrt(2) * x * np.exp(-25*x**2/2) / np.sqrt(np.pi)
    d2y_dx2 = 781.25*np.sqrt(2)*x**2*np.exp(-25*x**2/2)/np.sqrt(np.pi) - 31.25*np.sqrt(2)*np.exp(-25*x**2/2)/np.sqrt(np.pi)
    d3y_dx3 = -19531.25*np.sqrt(2)*x**3*np.exp(-25*x**2/2)/np.sqrt(np.pi) + 2343.75*np.sqrt(2)*x*np.exp(-25*x**2/2)/np.sqrt(np.pi)
    
    # Remove the points featuring the largest gradient area
    mask = [1, 2, 4, 5, 7, 10, 15]
    x, y, dy_dx, d2y_dx2, d3y_dx3 = remove_mask(x, y, dy_dx, d2y_dx2, d3y_dx3, mask=mask)

    x_test, y_test, dy_dx_test, y_pred, dy_dx_pred, y_rmse, dy_dx_rmse, history = main_procedure0(
        pinn(x, y, dy_dx, d2y_dx2, d3y_dx3, 1500, 800, 0)
    )
    x_test, y_test, dy_dx_test, y_pred, dy_dx_pred, y_rmse, dy_dx_rmse, history = main_procedure0(
        pinn(x, y, dy_dx, d2y_dx2, d3y_dx3, 1500, 800, 1)
    )
    x_test, y_test, dy_dx_test, y_pred, dy_dx_pred, y_rmse, dy_dx_rmse, history = main_procedure0(
        pinn(x, y, dy_dx, d2y_dx2, d3y_dx3, 1500, 800, 2)
    )
    x_test, y_test, dy_dx_test, y_pred, dy_dx_pred, y_rmse, dy_dx_rmse, history = main_procedure0(
        pinn_model = pinn(x, y, dy_dx, d2y_dx2, d3y_dx3, 1500, 800, 3)
    )
    
    with open(t2_results_dir / 'pinn_closure_id2.pkl', "rb") as f:
        history2 = pickle.load(f)["history"]
    with open(t2_results_dir / 'pinn_closure_id3.pkl', "rb") as f:
        history3 = pickle.load(f)["history"]

    fig = plt.figure(figsize=(8, 5.1), layout="constrained")
    ax = fig.subplots(1)
    ax.semilogy(history2[0, :], '-', label=r"Fitting $f(x) + \mathrm{d} f / \mathrm{d} x + \mathrm{d}^2 f / \mathrm{d} x^2$")
    ax.semilogy(history3[0, :], '-', label=r"Fitting $f(x) + \mathrm{d} f / \mathrm{d} x + \mathrm{d}^2 f / \mathrm{d} x^2 + \mathrm{d}^3 f / \mathrm{d} x^3$")
    ax.set_ylim([3e-4, 1e4])
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Total loss (MSE, $\log$ axis)")

    plt.legend()
    plt.savefig(t2_results_dir / f'comparison_order_2_3.png', dpi=300)
    
    fig = plt.figure(figsize=(8, 5.1), layout="constrained")
    ax = fig.subplots(1)
    ax.semilogy(history3[0, :], '-', label=r"Total Loss")
    ax.semilogy(history3[1, :], '-', label=r"$0^{\mathrm{th}}$-order loss")
    ax.semilogy(history3[2, :], '-', label=r"$1^{\mathrm{st}}$-order loss")
    ax.semilogy(history3[3, :], '-', label=r"$2^{\mathrm{nd}}$-order loss")
    ax.semilogy(history3[4, :], '-', label=r"$3^{\mathrm{rd}}$-order loss")
    ax.set_ylim([1e-4, 1e4])
    ax.set_xlabel("Iteration")
    ax.set_ylabel(r"Loss terms (MSE, $\log$ axis)")

    plt.legend()
    plt.savefig(t2_results_dir / f'comparison_order3_individual_terms.png', dpi=300)
