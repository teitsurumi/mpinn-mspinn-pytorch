import torch
import torch.nn as nn
import numpy as np
import time
# import scipy.io
import itertools

torch.manual_seed(1234)
# torch.autograd.set_detect_anomaly(True)
np.random.seed(1234)

_E = 2.05e11
_nu = 0.28
_mu = _E / (2 * (1 + _nu))
_lambda = _nu * _E / ((1 - 2*_nu) * (1 + _nu))

class dataset:
    def __init__(
        self, x: np.ndarray, y: np.ndarray,
        s_x: np.ndarray, s_y: np.ndarray, s_xy: np.ndarray,
        e_x: np.ndarray, e_y: np.ndarray, e_xy: np.ndarray,
        u: np.ndarray, v: np.ndarray,
        e_z: np.ndarray,
        norm_params: np.ndarray
    ):
        self.x    = x
        self.y    = y
        self.s_x  = s_x
        self.s_y  = s_y
        self.s_xy = s_xy
        self.e_x  = e_x
        self.e_y  = e_y
        self.e_xy = e_xy
        self.u    = u
        self.v    = v
        self.e_z  = e_z
        self.norm_params = norm_params
        
        

class Model(nn.Module):
    def __init__(
        self, in_data: dataset, bd_data: dataset
    ):
        """
        
        """
        super(Model, self).__init__()
        self.iter1 = 0
        self.iter2 = 0
        self.time1 = time.time()
        self.time2 = time.time()
        self.ls = 1
        
        device = torch.device("cuda:0")
        self.net_u_x = self.__network()
        self.net_u_y = self.__network()
        self.net_e_z = self.__network()

        self.mse = nn.MSELoss().cuda("cuda:0")

        self.interior_num = in_data.x.shape[0]
        self.x   = torch.tensor(in_data.x, dtype=torch.float32, requires_grad=True).reshape((self.interior_num, 1)).to(device)
        self.y   = torch.tensor(in_data.y, dtype=torch.float32, requires_grad=True).reshape((self.interior_num, 1)).to(device)
        self.s_x = torch.tensor(in_data.s_x, dtype=torch.float32).reshape((self.interior_num, 1)).to(device)
        self.s_y = torch.tensor(in_data.s_y, dtype=torch.float32).reshape((self.interior_num, 1)).to(device)
        self.s_xy = torch.tensor(in_data.s_xy, dtype=torch.float32).reshape((self.interior_num, 1)).to(device)
        self.e_x = torch.tensor(in_data.e_x, dtype=torch.float32).reshape((self.interior_num, 1)).to(device)
        self.e_y = torch.tensor(in_data.e_y, dtype=torch.float32).reshape((self.interior_num, 1)).to(device)
        self.e_xy = torch.tensor(in_data.e_xy, dtype=torch.float32).reshape((self.interior_num, 1)).to(device)
        self.u = torch.tensor(in_data.u, dtype=torch.float32).reshape((self.interior_num, 1)).to(device)
        self.v = torch.tensor(in_data.v, dtype=torch.float32).reshape((self.interior_num, 1)).to(device)
        self.e_z = torch.tensor(in_data.e_z, dtype=torch.float32).reshape((self.interior_num, 1)).to(device)
        self.norm_params = torch.tensor(in_data.norm_params, dtype=torch.float32).to(device)
        self.zeros_tensor = torch.tensor(np.zeros(self.interior_num).reshape((self.interior_num, 1)), dtype=torch.float32, requires_grad=True).reshape((self.interior_num, 1)).to(device)

        self.optimizer = torch.optim.AdamW(
            itertools.chain(self.net_u_x.parameters(), self.net_u_y.parameters(), self.net_e_z.parameters()),
            lr=0.00015
        )
        
        # Delaunay integral preparation
        # data
        delaunay_data = np.load("./data/shape6_delaunay2.npz")
        delaunay_triangles_filtered = delaunay_data["delaunay_triangles_filtered"]
        delaunay_tri_area = delaunay_data["delaunay_tri_area"]
        # pre-process
        delaunay_triangles_filtered = delaunay_triangles_filtered.reshape((delaunay_triangles_filtered.shape[0] * 3, 2))
        xx = delaunay_triangles_filtered[:, 0].copy()
        yy = delaunay_triangles_filtered[:, 1].copy()
        self.delaunay_vertexes_x = torch.tensor(xx, dtype=torch.float32, requires_grad=True).reshape((xx.shape[0], 1)).to(device)
        self.delaunay_vertexes_y = torch.tensor(yy, dtype=torch.float32, requires_grad=True).reshape((xx.shape[0], 1)).to(device)
        self.delaunay_tri_area = torch.tensor(delaunay_tri_area, dtype=torch.float32).reshape((int(xx.shape[0]/3), 1)).to(device)
        self.delaunay_func_value_mean = torch.tensor(np.zeros(int(xx.shape[0]/3)), dtype=torch.float32).reshape((int(xx.shape[0]/3), 1)).to(device)

        rightbd_x = np.ones((100+1,)) * 0.6
        rightbd_y = np.linspace(0, 0.6, 100+1)
        self.rightbd_x = torch.tensor(rightbd_x, dtype=torch.float32, requires_grad=True).reshape((rightbd_y.shape[0], 1)).to(device)
        self.rightbd_y = torch.tensor(rightbd_y, dtype=torch.float32, requires_grad=True).reshape((rightbd_y.shape[0], 1)).to(device)
        self.rightbd_dy = 0.6 / 100



    # ----------------------- BEGIN NETWORKS -----------------------
    def __network(self):
        net = nn.Sequential()
        net.add_module('Linear_layer_1', nn.Linear(2, 80))
        net.add_module('Tanh_layer_1', nn.Tanh())
        # for num in range(2, 6):
        for num in range(2, 7):
            net.add_module('Linear_layer_%d' % (num), nn.Linear(80, 80))
            net.add_module('Tanh_layer_%d' % (num), nn.Tanh())
        net.add_module('Linear_layer_final', nn.Linear(80, 1))
        net = net.cuda("cuda:0")
        return net
    # ----------------------- END NETWORKS -----------------------
    
    
    # ----------------------- BEGIN FUNCTIONS -----------------------
    def function_full(
        self, x: torch.Tensor, y: torch.Tensor
    ):
        """"""
        u_x = self.net_u_x(torch.hstack((x, y)))
        u_y = self.net_u_y(torch.hstack((x, y)))
        e_z = self.net_e_z(torch.hstack((x, y)))
        u_xx = gradients(u_x, x)[0]
        u_yy = gradients(u_y, y)[0]
        u_xy = gradients(u_x, y)[0]
        u_yx = gradients(u_y, x)[0]
        e_x = (self.norm_params[3][1]/self.norm_params[6][1]) * u_xx - (self.norm_params[3][0] * self.norm_params[3][1])
        e_y = (self.norm_params[4][1]/self.norm_params[7][1]) * u_yy - (self.norm_params[4][0] * self.norm_params[4][1])
        e_xy = self.norm_params[5][1] / 2 * ( u_yx/self.norm_params[7][1] + u_xy/self.norm_params[6][1] ) - self.norm_params[5][0] * self.norm_params[5][1]
        
        s_x = ((_lambda + 2*_mu) * (e_x / self.norm_params[3][1] + self.norm_params[3][0]) + _lambda * (e_y / self.norm_params[4][1] + self.norm_params[4][0] + e_z / self.norm_params[8][1] + self.norm_params[8][0]) - self.norm_params[0][0]) * self.norm_params[0][1]
        s_y = ((_lambda + 2*_mu) * (e_y / self.norm_params[4][1] + self.norm_params[4][0]) + _lambda * (e_x / self.norm_params[3][1] + self.norm_params[3][0] + e_z / self.norm_params[8][1] + self.norm_params[8][0]) - self.norm_params[1][0]) * self.norm_params[1][1]
        s_xy = (2*_mu * (e_xy / self.norm_params[5][1] + self.norm_params[5][0]) - self.norm_params[2][0]) * self.norm_params[2][1]
        return u_x, u_y, e_x, e_y, e_xy, e_z, s_x, s_y, s_xy
    
    def function_extra(
        self, x: torch.Tensor, y: torch.Tensor
    ):
        """"""
        u_x = self.net_u_x(torch.hstack((x, y)))
        u_y = self.net_u_y(torch.hstack((x, y)))
        e_z = self.net_e_z(torch.hstack((x, y)))
        u_xx = gradients(u_x, x)[0]
        u_yy = gradients(u_y, y)[0]
        u_xy = gradients(u_x, y)[0]
        u_yx = gradients(u_y, x)[0]
        e_x = (self.norm_params[3][1]/self.norm_params[6][1]) * u_xx - (self.norm_params[3][0] * self.norm_params[3][1])
        e_y = (self.norm_params[4][1]/self.norm_params[7][1]) * u_yy - (self.norm_params[4][0] * self.norm_params[4][1])
        e_xy = self.norm_params[5][1] / 2 * ( u_yx/self.norm_params[7][1] + u_xy/self.norm_params[6][1] ) - self.norm_params[5][0] * self.norm_params[5][1]
        
        s_x = ((_lambda + 2*_mu) * (e_x / self.norm_params[3][1] + self.norm_params[3][0]) + _lambda * (e_y / self.norm_params[4][1] + self.norm_params[4][0] + e_z / self.norm_params[8][1] + self.norm_params[8][0]) - self.norm_params[0][0]) * self.norm_params[0][1]
        s_y = ((_lambda + 2*_mu) * (e_y / self.norm_params[4][1] + self.norm_params[4][0]) + _lambda * (e_x / self.norm_params[3][1] + self.norm_params[3][0] + e_z / self.norm_params[8][1] + self.norm_params[8][0]) - self.norm_params[1][0]) * self.norm_params[1][1]
        s_xy = (2*_mu * (e_xy / self.norm_params[5][1] + self.norm_params[5][0]) - self.norm_params[2][0]) * self.norm_params[2][1]
        s_x_x = gradients(s_x, x)[0]
        s_y_y = gradients(s_y, y)[0]
        s_xy_x = gradients(s_xy, x)[0]
        s_xy_y = gradients(s_xy, y)[0]
        return u_x, u_y, e_x, e_y, e_xy, e_z, s_x, s_y, s_xy, s_x_x, s_y_y, s_xy_x, s_xy_y
    
    def delaunay_func(self, x, y):
        u_x = self.net_u_x(torch.hstack((x, y)))
        u_y = self.net_u_y(torch.hstack((x, y)))
        e_z = self.net_e_z(torch.hstack((x, y)))
        u_xx = gradients(u_x, x)[0]
        u_yy = gradients(u_y, y)[0]
        u_xy = gradients(u_x, y)[0]
        u_yx = gradients(u_y, x)[0]
        e_x = (self.norm_params[3][1]/self.norm_params[6][1]) * u_xx - (self.norm_params[3][0] * self.norm_params[3][1])
        e_y = (self.norm_params[4][1]/self.norm_params[7][1]) * u_yy - (self.norm_params[4][0] * self.norm_params[4][1])
        e_xy = self.norm_params[5][1] / 2 * ( u_yx/self.norm_params[7][1] + u_xy/self.norm_params[6][1] ) - self.norm_params[5][0] * self.norm_params[5][1]
        s_x = ((_lambda + 2*_mu) * (e_x / self.norm_params[3][1] + self.norm_params[3][0]) + _lambda * (e_y / self.norm_params[4][1] + self.norm_params[4][0] + e_z / self.norm_params[8][1] + self.norm_params[8][0]) - self.norm_params[0][0]) * self.norm_params[0][1]
        s_y = ((_lambda + 2*_mu) * (e_y / self.norm_params[4][1] + self.norm_params[4][0]) + _lambda * (e_x / self.norm_params[3][1] + self.norm_params[3][0] + e_z / self.norm_params[8][1] + self.norm_params[8][0]) - self.norm_params[1][0]) * self.norm_params[1][1]
        s_xy = (2*_mu * (e_xy / self.norm_params[5][1] + self.norm_params[5][0]) - self.norm_params[2][0]) * self.norm_params[2][1]
        # s_x_x = gradients(s_x, x)[0]
        # s_y_y = gradients(s_y, y)[0]
        # s_xy_x = gradients(s_xy, x)[0]
        # s_xy_y = gradients(s_xy, y)[0]
        
        func_value = (s_x/self.norm_params[0][1]+self.norm_params[0][0]) * (e_x/self.norm_params[3][1]+self.norm_params[3][0]) \
            + (s_y/self.norm_params[1][1]+self.norm_params[1][0]) * (e_y/self.norm_params[4][1]+self.norm_params[4][0]) \
                + (s_xy/self.norm_params[2][1]+self.norm_params[2][0]) * (e_xy/self.norm_params[5][1]+self.norm_params[5][0])
        return func_value / 2

    def rbd_func(self, x, y):
        u_x = self.net_u_x(torch.hstack((x, y)))
        func_value = 100 * self.rightbd_dy * (u_x/self.norm_params[6][1]+self.norm_params[6][0])
        return func_value / 2
    # ----------------------- END FUNCTIONS -----------------------

    def closure0(self):
        self.optimizer.zero_grad()
        u_x, u_y, e_x, e_y, e_xy, e_z, s_x, s_y, s_xy = self.function_full(self.x, self.y)
        ls1 = self.mse(u_x, self.u)
        ls2 = self.mse(u_y, self.v)
        ls6 = self.mse(e_z, self.e_z)
        self.ls = ls1 + ls2 + ls6
        self.ls.backward()
        return self.ls

    def closure0_1(self):
        self.optimizer.zero_grad()
        u_x, u_y, e_x, e_y, e_xy, e_z, s_x, s_y, s_xy = self.function_full(self.x, self.y)
        ls3 = self.mse(e_x, self.e_x)
        ls4 = self.mse(e_y, self.e_y)
        ls5 = self.mse(e_xy, self.e_xy)
        ls6 = self.mse(e_z, self.e_z)
        self.ls = ls3 + ls4 + ls5 + ls6
        self.ls.backward()
        return self.ls

    def closure1(self):
        self.optimizer.zero_grad()
        u_x, u_y, e_x, e_y, e_xy, e_z, s_x, s_y, s_xy, s_x_x, s_y_y, s_xy_x, s_xy_y = self.function_extra(self.x, self.y)
        ls1 = self.mse(u_x, self.u)
        ls2 = self.mse(u_y, self.v)
        ls3 = self.mse(e_x, self.e_x)
        ls4 = self.mse(e_y, self.e_y)
        ls5 = self.mse(e_xy, self.e_xy)
        ls6 = self.mse(e_z, self.e_z)
        ls7 = self.mse(s_x, self.s_x)
        ls8 = self.mse(s_y, self.s_y)
        ls9 = self.mse(s_xy, self.s_xy)
        ls10 = self.mse(self.norm_params[2][1] * s_x_x / self.norm_params[0][1] + s_xy_y, self.zeros_tensor)
        ls11 = self.mse(self.norm_params[2][1] * s_y_y / self.norm_params[1][1] + s_xy_x, self.zeros_tensor)
        self.ls = ls1 + ls2 + ls3 + ls4 + ls5 + ls6 + ls7 + ls8 + ls9
        self.ls.backward()
        try:
            self.__history_loss[0][self.iter1] = self.ls.item()
            self.__history_loss[1][self.iter1] = ls1.item()
            self.__history_loss[2][self.iter1] = ls2.item()
            self.__history_loss[3][self.iter1] = ls3.item()
            self.__history_loss[4][self.iter1] = ls4.item()
            self.__history_loss[5][self.iter1] = ls5.item()
            self.__history_loss[6][self.iter1] = ls6.item()
            self.__history_loss[7][self.iter1] = ls7.item()
            self.__history_loss[8][self.iter1] = ls8.item()
            self.__history_loss[9][self.iter1] = ls9.item()
            self.__history_loss[10][self.iter1] = ls10.item()
            self.__history_loss[11][self.iter1] = ls11.item()
        except:
            pass
        if not self.iter1 % 200:
            self.time2 = time.time()
            t = self.time2 - self.time1
            print('Iteration: {:}, Loss: {:0.6f}, Δt={:0.3f}s'.format(self.iter1, self.ls, t))
            print('u:  ', ls1.item(), ls2.item())
            print('epsilon:  ', ls3.item(), ls4.item(), ls5.item(), ls6.item())
            print('sigma:  ', ls7.item(), ls8.item(), ls9.item())
            self.time1 = time.time()
        self.iter1 += 1
        return self.ls
    
    def closure1_e(self):
        self.optimizer.zero_grad()
        u_x, u_y, e_x, e_y, e_xy, e_z, s_x, s_y, s_xy, s_x_x, s_y_y, s_xy_x, s_xy_y = self.function_extra(self.x, self.y)
        delaunay_func_value_mean = self.delaunay_func_value_mean * 0

        ls1 = self.mse(u_x, self.u)
        ls2 = self.mse(u_y, self.v)
        ls3 = self.mse(e_x, self.e_x)
        ls4 = self.mse(e_y, self.e_y)
        ls5 = self.mse(e_xy, self.e_xy)
        ls6 = self.mse(e_z, self.e_z)
        ls7 = self.mse(s_x, self.s_x)
        ls8 = self.mse(s_y, self.s_y)
        ls9 = self.mse(s_xy, self.s_xy)
        ls10 = self.mse(self.norm_params[2][1] * s_x_x / self.norm_params[0][1] + s_xy_y, self.zeros_tensor)
        ls11 = self.mse(self.norm_params[2][1] * s_y_y / self.norm_params[1][1] + s_xy_x, self.zeros_tensor)
        
        # ---------------------------------- strain energy ----------------------------------
        func_value = self.delaunay_func(self.delaunay_vertexes_x, self.delaunay_vertexes_y)
        delaunay_func_value_mean = torch.sum(func_value.view((int(func_value.shape[0]/3), 3)), 1)
        delaunay_func_value_mean = delaunay_func_value_mean.reshape((delaunay_func_value_mean.shape[0], 1))
        ls12 = torch.sum(delaunay_func_value_mean * self.delaunay_tri_area)
        # ---------------------------------- work ----------------------------------
        rbd_work = self.rbd_func(self.rightbd_x, self.rightbd_y)
        ls13 = torch.sum(rbd_work)
        # ---------------------------------------------------------------------------
        
        self.ls = ls1 + ls2 + ls3 + ls4 + ls5 + ls6 + ls7 + ls8 + ls9 + ls12 - ls13
        self.ls.backward()
        
        try:
            self.__history_loss[0][self.iter1] = self.ls.item()
            self.__history_loss[1][self.iter1] = ls1.item()
            self.__history_loss[2][self.iter1] = ls2.item()
            self.__history_loss[3][self.iter1] = ls3.item()
            self.__history_loss[4][self.iter1] = ls4.item()
            self.__history_loss[5][self.iter1] = ls5.item()
            self.__history_loss[6][self.iter1] = ls6.item()
            self.__history_loss[7][self.iter1] = ls7.item()
            self.__history_loss[8][self.iter1] = ls8.item()
            self.__history_loss[9][self.iter1] = ls9.item()
            self.__history_loss[10][self.iter1] = ls10.item()
            self.__history_loss[11][self.iter1] = ls11.item()
            self.__history_loss[12][self.iter1] = ls12.item() - ls13.item()
        except:
            pass
        if not self.iter1 % 200:
            self.time2 = time.time()
            t = self.time2 - self.time1
            print('Iteration: {:}, Loss: {:0.6f}, Δt={:0.3f}s'.format(self.iter1, self.ls, t))
            print('u:  ', ls1.item(), ls2.item())
            print('epsilon:  ', ls3.item(), ls4.item(), ls5.item(), ls6.item())
            print('sigma:  ', ls7.item(), ls8.item(), ls9.item())
            print('stress energy | work | potential energy:   ', ls12.item(), ls13.item(), ls12.item() - ls13.item())
            self.time1 = time.time()
        self.iter1 += 1
        return self.ls


    def closure2(self):
        self.optimizer.zero_grad()
        u_x, u_y, e_x, e_y, e_xy, e_z, s_x, s_y, s_xy = self.function_full(self.x, self.y)
        ls3 = self.mse(e_x, self.e_x)
        ls4 = self.mse(e_y, self.e_y)
        ls5 = self.mse(e_xy, self.e_xy)
        ls6 = self.mse(e_z, self.e_z)
        ls7 = self.mse(s_x, self.s_x)
        ls8 = self.mse(s_y, self.s_y)
        ls9 = self.mse(s_xy, self.s_xy)
        self.ls = ls3 + ls4 + ls5 + ls6 + ls7 + ls8 + ls9
        self.ls.backward()
        if not self.iter2 % 200:
            print('closure2 Iteration: {:}, Loss: {:0.6f}'.format(self.iter2, self.ls))
        self.iter2 += 1
        return self.ls

    def closure3(self):
        self.optimizer.zero_grad()
        u_x, u_y, e_x, e_y, e_xy, e_z, s_x, s_y, s_xy = self.function_full(self.x, self.y)
        ls3 = self.mse(e_x, self.e_x)
        ls4 = self.mse(e_y, self.e_y)
        ls5 = self.mse(e_xy, self.e_xy)
        ls6 = self.mse(e_z, self.e_z)
        ls7 = self.mse(s_x, self.s_x)
        ls8 = self.mse(s_y, self.s_y)
        ls9 = self.mse(s_xy, self.s_xy)
        self.ls = ls7 + ls8 + ls9
        self.ls.backward()
        return self.ls

    def closure_extra(self):
        self.optimizer.zero_grad()
        u_x, u_y, e_x, e_y, e_xy, e_z, s_x, s_y, s_xy, s_x_x, s_y_y, s_xy_x, s_xy_y = self.function_extra(self.x, self.y)
        ls1 = self.mse(u_x, self.u)
        ls2 = self.mse(u_y, self.v)
        ls3 = self.mse(e_x, self.e_x)
        ls4 = self.mse(e_y, self.e_y)
        ls5 = self.mse(e_xy, self.e_xy)
        ls6 = self.mse(e_z, self.e_z)
        ls7 = self.mse(s_x, self.s_x)
        ls8 = self.mse(s_y, self.s_y)
        ls9 = self.mse(s_xy, self.s_xy)
        ls10 = self.mse(s_x_x / self.norm_params[0][1] + s_xy_y / self.norm_params[2][1], self.zeros_tensor)
        ls11 = self.mse(s_y_y / self.norm_params[1][1] + s_xy_x / self.norm_params[2][1], self.zeros_tensor)
        self.ls = ls1 + ls2 + ls3 + ls4 + ls5 + ls6 + ls7 + ls8 + ls9 + ls10 + ls11
        self.ls.backward()
        return self.ls

    # --------------------------------- TRAINING 1 ---------------------------------
    def train(self):
        device = torch.device("cuda:0")
        self.__history_generate_object_number = 12 + 1
        self.__history_loss = torch.zeros((60000) * self.__history_generate_object_number, dtype=torch.float32) \
                            .reshape((self.__history_generate_object_number, 60000)) \
                                .to(device)
        """training"""
        self.net_u_x.train()
        self.net_u_y.train()
        self.net_e_z.train()

        for i in range(5000):
            self.optimizer.step(self.closure1_e)
        # for i in range(5000):
        #     self.optimizer.step(self.closure2)
        # self.optimizer = torch.optim.AdamW(
        #     itertools.chain(self.net_u_x.parameters(), self.net_u_y.parameters(), self.net_e_z.parameters()),
        #     lr=0.00005
        # )
        # for i in range(5000):
        #     self.optimizer.step(self.closure2)
        # self.optimizer = torch.optim.AdamW(
        #     itertools.chain(self.net_u_x.parameters(), self.net_u_y.parameters(), self.net_e_z.parameters()),
        #     lr=0.00001
        # )
        # for i in range(10000):
        #     self.optimizer.step(self.closure2)
        # self.optimizer = torch.optim.AdamW(
        #     itertools.chain(self.net_u_x.parameters(), self.net_u_y.parameters(), self.net_e_z.parameters()),
        #     lr=0.000005
        # )
        # for i in range(5000):
        #     self.optimizer.step(self.closure2)


    # --------------------------------- TRAINING 2 ---------------------------------
    def re_train(self):
        device = torch.device("cuda:0")
        self.__history_generate_object_number = 12 + 1
        self.__history_loss = torch.zeros((20000) * self.__history_generate_object_number, dtype=torch.float32) \
                            .reshape((self.__history_generate_object_number, 20000)) \
                                .to(device)
        """training"""
        self.net_u_x.train()
        self.net_u_y.train()
        self.net_e_z.train()

        self.optimizer = torch.optim.AdamW(
            itertools.chain(self.net_u_x.parameters(), self.net_u_y.parameters(), self.net_e_z.parameters()),
            lr=0.00004
        )
        for i in range(4000):
            self.optimizer.step(self.closure0_1)
        for i in range(2000):
            self.optimizer.step(self.closure1_e)
        self.optimizer = torch.optim.AdamW(
            itertools.chain(self.net_u_x.parameters(), self.net_u_y.parameters(), self.net_e_z.parameters()),
            lr=0.00001
        )
        for i in range(5000):
            self.optimizer.step(self.closure1_e)



    def get_history_loss(self):
        return self.__history_loss


# supporting functions

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



import torch
import torch.nn as nn
import numpy as np
import time

from utils.normalization import norm_array_0_1, norm_array_m1_1
import utils.errors_np as errors_np

if __name__ == "__main__":
    device = torch.device("cuda:0")
    # with open("./data/shape4node4.txt", 'r') as f:
    #     data = np.loadtxt(f)
    with open("./data/shape6.txt", 'r') as f:
        data = np.loadtxt(f)

    # NORMALIZATION
    norm_s_x  = norm_array_0_1(data[:, 3])
    norm_s_y  = norm_array_0_1(data[:, 5])
    norm_s_xy = norm_array_0_1(data[:, 4])
    norm_e_x = norm_array_0_1(data[:, 6])
    norm_e_y = norm_array_0_1(data[:, 8])
    norm_e_xy = norm_array_0_1(data[:, 7])
    norm_u = norm_array_0_1(data[:, 9])
    norm_v = norm_array_0_1(data[:, 10])
    norm_e_z = norm_array_0_1(data[:, 11])
    data[:, 3], aa = norm_s_x.norm_forward()
    data[:, 5], bb = norm_s_y.norm_forward()
    data[:, 4], cc = norm_s_xy.norm_forward()
    data[:, 6], dd = norm_e_x.norm_forward()
    data[:, 8], ee = norm_e_y.norm_forward()
    data[:, 7], ff = norm_e_xy.norm_forward()
    data[:, 9], gg = norm_u.norm_forward()
    data[:, 10], hh = norm_v.norm_forward()
    data[:, 11], ii = norm_e_z.norm_forward()
    norm_params = [aa, bb, cc, dd, ee, ff, gg, hh, ii]
    for i, _ in enumerate(norm_params):
        for j, _ in enumerate(norm_params[i]):
            norm_params[i][j] = float(norm_params[i][j])
    norm_params = np.array(norm_params, dtype=float)

    x    = data[:, 0]
    y    = data[:, 1]
    s_x  = data[:, 3]
    s_y  = data[:, 5]
    s_xy = data[:, 4]
    e_x = data[:, 6]
    e_y = data[:, 8]
    e_xy = data[:, 7]
    u = data[:, 9]
    v = data[:, 10]
    
    e_z = data[:, 11]
    
    in_data = dataset(x,y,s_x,s_y,s_xy,e_x,e_y,e_xy,u,v,e_z,norm_params)
    bd_data = dataset(x,y,s_x,s_y,s_xy,e_x,e_y,e_xy,u,v,e_z,norm_params)
    print(norm_params)
    print(min(u), min(v), min(e_x), min(e_y), min(e_xy))
    print(max(u), max(v), max(e_x), max(e_y), max(e_xy))

    # --------------------------------- MODEL & TRAIN ---------------------------------
    pinn_model = Model(in_data, bd_data)
    pinn_model.train()
    # torch.save(pinn_model.net_u_x, './models_v6/shape6_model1_u_x.pt')
    # torch.save(pinn_model.net_u_y, './models_v6/shape6_model1_u_y.pt')
    # torch.save(pinn_model.net_e_z, './models_v6/shape6_model1_e_z.pt')
    # # record loss
    # history = pinn_model.get_history_loss()
    # history = to_numpy(history)
    # np.save("./models_v6/shape6_history_start2.npy", history)






    # ------------------------------ MODEL RE-GENERATE ------------------------------
    # del pinn_model
    # pinn_model = Model(in_data, bd_data)
    # pinn_model.net_u_x = torch.load('./models_v4/shape6_model9_u_x.pt')
    # pinn_model.net_u_y = torch.load('./models_v4/shape6_model9_u_y.pt')
    # pinn_model.net_e_z = torch.load('./models_v4/shape6_model9_e_z.pt')

    # pinn_model.net_u_x.eval()
    # pinn_model.net_u_y.eval()
    # pinn_model.net_e_z.eval()
    
    # ------------------------------ RE-TRAIN ------------------------------
    # pinn_model.re_train()

    # torch.save(pinn_model.net_u_x, './models_v6/shape6_model1_e6_u_x.pt')
    # torch.save(pinn_model.net_u_y, './models_v6/shape6_model1_e6_u_y.pt')
    # torch.save(pinn_model.net_e_z, './models_v6/shape6_model1_e6_e_z.pt')
    
    # history = pinn_model.get_history_loss()
    # history = to_numpy(history)
    # np.save("./models_v6/shape6_history1_e6.npy", history)

    # -------------------------------------- TEST --------------------------------------
    # with open("./data/shape4node4_med.txt", 'r') as f:
    #     data = np.loadtxt(f)
    with open("./data/shape6_full_full.txt", 'r') as f:
        data = np.loadtxt(f)
    # data = data[data[:, 0] < 0.05, :]
    # data = data[data[:, 1] < 0.3, :]
    # MORMALIZATION
    norm_s_x  = norm_array_0_1(data[:, 3])
    norm_s_y  = norm_array_0_1(data[:, 5])
    norm_s_xy = norm_array_0_1(data[:, 4])
    norm_e_x = norm_array_0_1(data[:, 6])
    norm_e_y = norm_array_0_1(data[:, 8])
    norm_e_xy = norm_array_0_1(data[:, 7])
    norm_u = norm_array_0_1(data[:, 9])
    norm_v = norm_array_0_1(data[:, 10])
    norm_e_z = norm_array_0_1(data[:, 11])
    data[:, 3], aa = norm_s_x.norm_forward()
    data[:, 5], bb = norm_s_y.norm_forward()
    data[:, 4], cc = norm_s_xy.norm_forward()
    data[:, 6], dd = norm_e_x.norm_forward()
    data[:, 8], ee = norm_e_y.norm_forward()
    data[:, 7], ff = norm_e_xy.norm_forward()
    data[:, 9], gg = norm_u.norm_forward()
    data[:, 10], hh = norm_v.norm_forward()
    data[:, 11], ii = norm_e_z.norm_forward()
    # norm_params = [aa, bb, cc, dd, ee, ff, gg, hh]
    # for i, _ in enumerate(norm_params):
    #     for j, _ in enumerate(norm_params[i]):
    #         norm_params[i][j] = float(norm_params[i][j])
    # norm_params = np.array(norm_params, dtype=float)

    x    = data[:, 0]
    y    = data[:, 1]
    s_x  = data[:, 3]
    s_y  = data[:, 5]
    s_xy = data[:, 4]
    e_x = data[:, 6]
    e_y = data[:, 8]
    e_xy = data[:, 7]
    u = data[:, 9]
    v = data[:, 10]
    e_z = data[:, 11]
    x_test = torch.tensor(x, dtype=torch.float32, requires_grad=True).reshape((x.shape[0], 1)).to(device)
    y_test = torch.tensor(y, dtype=torch.float32, requires_grad=True).reshape((y.shape[0], 1)).to(device)

    u_x_pred, u_y_pred, e_x_pred, e_y_pred, e_xy_pred, e_z_pred, s_x_pred, s_y_pred, s_xy_pred = pinn_model.function_full(x_test, y_test)

    s_x_pred  = to_numpy(s_x_pred)
    s_y_pred  = to_numpy(s_y_pred)
    s_xy_pred = to_numpy(s_xy_pred)
    e_x_pred  = to_numpy(e_x_pred)
    e_y_pred  = to_numpy(e_y_pred)
    e_xy_pred = to_numpy(e_xy_pred)
    e_z_pred  = to_numpy(e_z_pred)
    u_x_pred  = to_numpy(u_x_pred)
    u_y_pred  = to_numpy(u_y_pred)
    s_x_pred  = s_x_pred.reshape((s_x_pred.shape[0], ))
    s_y_pred  = s_y_pred.reshape((s_y_pred.shape[0], ))
    s_xy_pred = s_xy_pred.reshape((s_xy_pred.shape[0], ))
    e_x_pred  = e_x_pred.reshape((e_x_pred.shape[0], ))
    e_y_pred  = e_y_pred.reshape((e_y_pred.shape[0], ))
    e_xy_pred = e_xy_pred.reshape((e_xy_pred.shape[0], ))
    e_z_pred  = e_z_pred.reshape((e_z_pred.shape[0], ))
    u_x_pred  = u_x_pred.reshape((u_x_pred.shape[0], ))
    u_y_pred  = u_y_pred.reshape((u_y_pred.shape[0], ))
    
    # norm backward
    s_x = norm_s_x.norm_backward(s_x)
    s_y = norm_s_y.norm_backward(s_y)
    s_xy = norm_s_xy.norm_backward(s_xy)
    s_x_pred = norm_s_x.norm_backward(s_x_pred)
    s_y_pred = norm_s_y.norm_backward(s_y_pred)
    s_xy_pred = norm_s_xy.norm_backward(s_xy_pred)
    e_x = norm_e_x.norm_backward(e_x)
    e_y = norm_e_y.norm_backward(e_y)
    e_xy = norm_e_xy.norm_backward(e_xy)
    e_z = norm_e_z.norm_backward(e_z)
    e_x_pred = norm_e_x.norm_backward(e_x_pred)
    e_y_pred = norm_e_y.norm_backward(e_y_pred)
    e_xy_pred = norm_e_xy.norm_backward(e_xy_pred)
    e_z_pred = norm_e_z.norm_backward(e_z_pred)
    u = norm_u.norm_backward(u)
    v = norm_v.norm_backward(v)
    u_x_pred = norm_u.norm_backward(u_x_pred)
    u_y_pred = norm_v.norm_backward(u_y_pred)
    
    s_x_pred = (_lambda+2*_mu) * e_x_pred + _lambda * (e_y_pred + e_z)
    s_y_pred = (_lambda+2*_mu) * e_y_pred + _lambda * (e_x_pred + e_z)
    s_xy_pred = 2 * _mu * e_xy_pred

    # ERRORS
    print(errors_np.rmse(s_x, s_x_pred))
    print(errors_np.rmse(s_y, s_y_pred))
    print(errors_np.rmse(s_xy, s_xy_pred))
    
    print(errors_np.rmse(e_x, e_x_pred))
    print(errors_np.rmse(e_y, e_y_pred))
    print(errors_np.rmse(e_xy, e_xy_pred))
    
    print(errors_np.rmse(u, u_x_pred))
    print(errors_np.rmse(v, u_y_pred))
    
    
    data = np.ndarray((x.shape[0], 12))
    data[:, 0] = x
    data[:, 1] = y
    data[:, 3] = s_x
    data[:, 5] = s_y
    data[:, 4] = s_xy
    data[:, 6] = e_x
    data[:, 8] = e_y
    data[:, 7] = e_xy
    data[:, 9] = u
    data[:, 10] = v
    data[:, 11] = e_z
    data = data[data[:, 0] < 0.04, :]
    data = data[data[:, 1] < 0.24, :]
    data = data[data[:, 1] > 0.16, :]
    # x    = data[:, 0]
    # y    = data[:, 1]
    s_x  = data[:, 3]
    s_y  = data[:, 5]
    s_xy = data[:, 4]
    e_x = data[:, 6]
    e_y = data[:, 8]
    e_xy = data[:, 7]
    u = data[:, 9]
    v = data[:, 10]
    e_z = data[:, 11]
    data = np.ndarray((x.shape[0], 12))
    data[:, 0] = x
    data[:, 1] = y
    data[:, 3] = s_x_pred
    data[:, 5] = s_y_pred
    data[:, 4] = s_xy_pred
    data[:, 6] = e_x_pred
    data[:, 8] = e_y_pred
    data[:, 7] = e_xy_pred
    data[:, 9] = u_x_pred
    data[:, 10] = u_y_pred
    data[:, 11] = e_z_pred
    data = data[data[:, 0] < 0.04, :]
    data = data[data[:, 1] < 0.24, :]
    data = data[data[:, 1] > 0.16, :]
    x    = data[:, 0]
    y    = data[:, 1]
    s_x_pred  = data[:, 3]
    s_y_pred  = data[:, 5]
    s_xy_pred = data[:, 4]
    e_x_pred = data[:, 6]
    e_y_pred = data[:, 8]
    e_xy_pred = data[:, 7]
    u_x_pred = data[:, 9]
    u_y_pred = data[:, 10]
    e_z_pred = data[:, 11]
    
    
    # Local ERRORS
    print("Local Errors:")
    print(errors_np.rmse(s_x, s_x_pred))
    print(errors_np.rmse(s_y, s_y_pred))
    print(errors_np.rmse(s_xy, s_xy_pred))
    
    print(errors_np.rmse(e_x, e_x_pred))
    print(errors_np.rmse(e_y, e_y_pred))
    print(errors_np.rmse(e_xy, e_xy_pred))
    
    print(errors_np.rmse(u, u_x_pred))
    print(errors_np.rmse(v, u_y_pred))
    
    
    # PLOTS
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    plt.rcParams.update({
        "font.size": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
    
    fig = plt.figure(0, figsize=(12,13))
    ax = fig.subplots(4, 4)
    fig.tight_layout(h_pad=1)
    
    fig.suptitle(r'$\sigma, \varepsilon$ : Real(left); Predict(right)')
    plt.subplots_adjust(top=0.95)
    
    ax[0, 0].scatter(x, y, c=s_x.reshape(x.shape[0]), cmap=plt.cm.jet, vmin=min(s_x), vmax=max(s_x), alpha=0.9, s=3)
    ax[0, 1].scatter(x, y, c=s_x_pred.reshape(x.shape[0]), cmap=plt.cm.jet, vmin=min(s_x), vmax=max(s_x), alpha=0.9, s=3)
    ax[1, 0].scatter(x, y, c=s_y.reshape(x.shape[0]), cmap=plt.cm.jet, vmin=min(s_y), vmax=max(s_y), alpha=0.9, s=3)
    ax[1, 1].scatter(x, y, c=s_y_pred.reshape(x.shape[0]), cmap=plt.cm.jet, vmin=min(s_y), vmax=max(s_y), alpha=0.9, s=3)
    ax[2, 0].scatter(x, y, c=s_xy.reshape(x.shape[0]), cmap=plt.cm.jet, vmin=min(s_xy), vmax=max(s_xy), alpha=0.9, s=3)
    ax[2, 1].scatter(x, y, c=s_xy_pred.reshape(x.shape[0]), cmap=plt.cm.jet, vmin=min(s_xy), vmax=max(s_xy), alpha=0.9, s=3)
    ax[0, 0].set_title(r"$\sigma_x$")
    ax[0, 1].set_title(r"$\sigma_{x,pred}$")
    ax[1, 0].set_title(r"$\sigma_y$")
    ax[1, 1].set_title(r"$\sigma_{y,pred}$")
    ax[2, 0].set_title(r"$\tau_{xy}$")
    ax[2, 1].set_title(r"$\tau_{xy,pred}$")
    ax[0, 2].scatter(x, y, c=e_x.reshape(x.shape[0]), cmap=plt.cm.jet, vmin=min(e_x), vmax=max(e_x), alpha=0.9, s=3)
    ax[0, 3].scatter(x, y, c=e_x_pred.reshape(x.shape[0]), cmap=plt.cm.jet, vmin=min(e_x), vmax=max(e_x), alpha=0.9, s=3)
    ax[1, 2].scatter(x, y, c=e_y.reshape(x.shape[0]), cmap=plt.cm.jet, vmin=min(e_y), vmax=max(e_y), alpha=0.9, s=3)
    ax[1, 3].scatter(x, y, c=e_y_pred.reshape(x.shape[0]), cmap=plt.cm.jet, vmin=min(e_y), vmax=max(e_y), alpha=0.9, s=3)
    ax[2, 2].scatter(x, y, c=e_xy.reshape(x.shape[0]), cmap=plt.cm.jet, vmin=min(e_xy), vmax=max(e_xy), alpha=0.9, s=3)
    ax[2, 3].scatter(x, y, c=e_xy_pred.reshape(x.shape[0]), cmap=plt.cm.jet, vmin=min(e_xy), vmax=max(e_xy), alpha=0.9, s=3)
    ax[0, 2].set_title(r"$\varepsilon_x$")
    ax[0, 3].set_title(r"$\varepsilon_{x,pred}$")
    ax[1, 2].set_title(r"$\varepsilon_y$")
    ax[1, 3].set_title(r"$\varepsilon_{y,pred}$")
    ax[2, 2].set_title(r"$\gamma_{xy}$")
    ax[2, 3].set_title(r"$\gamma_{xy,pred}$")
    ax[3, 0].scatter(x, y, c=u.reshape(x.shape[0]), cmap=plt.cm.jet, vmin=min(u), vmax=max(u), alpha=0.9, s=3)
    ax[3, 1].scatter(x, y, c=u_x_pred.reshape(x.shape[0]), cmap=plt.cm.jet, vmin=min(u), vmax=max(u), alpha=0.9, s=3)
    ax[3, 2].scatter(x, y, c=v.reshape(x.shape[0]), cmap=plt.cm.jet, vmin=min(v), vmax=max(v), alpha=0.9, s=3)
    ax[3, 3].scatter(x, y, c=u_y_pred.reshape(x.shape[0]), cmap=plt.cm.jet, vmin=min(v), vmax=max(v), alpha=0.9, s=3)
    ax[3, 0].set_title(r"$u_x$")
    ax[3, 1].set_title(r"$u_{x,pred}$")
    ax[3, 2].set_title(r"$u_y$")
    ax[3, 3].set_title(r"$u_{y,pred}$")
    
    fig.colorbar(plt.cm.ScalarMappable(mpl.colors.Normalize(vmin=min(s_x), vmax=max(s_x), clip=True), cmap=plt.cm.jet), ax=ax[0, 0])
    fig.colorbar(plt.cm.ScalarMappable(mpl.colors.Normalize(vmin=min(s_x), vmax=max(s_x), clip=True), cmap=plt.cm.jet), ax=ax[0, 1])
    fig.colorbar(plt.cm.ScalarMappable(mpl.colors.Normalize(vmin=min(s_y), vmax=max(s_y), clip=True), cmap=plt.cm.jet), ax=ax[1, 0])
    fig.colorbar(plt.cm.ScalarMappable(mpl.colors.Normalize(vmin=min(s_y), vmax=max(s_y), clip=True), cmap=plt.cm.jet), ax=ax[1, 1])
    fig.colorbar(plt.cm.ScalarMappable(mpl.colors.Normalize(vmin=min(s_xy), vmax=max(s_xy), clip=True), cmap=plt.cm.jet), ax=ax[2, 0])
    fig.colorbar(plt.cm.ScalarMappable(mpl.colors.Normalize(vmin=min(s_xy), vmax=max(s_xy), clip=True), cmap=plt.cm.jet), ax=ax[2, 1])
    fig.colorbar(plt.cm.ScalarMappable(mpl.colors.Normalize(vmin=min(e_x), vmax=max(e_x), clip=True), cmap=plt.cm.jet), ax=ax[0, 2])
    fig.colorbar(plt.cm.ScalarMappable(mpl.colors.Normalize(vmin=min(e_x), vmax=max(e_x), clip=True), cmap=plt.cm.jet), ax=ax[0, 3])
    fig.colorbar(plt.cm.ScalarMappable(mpl.colors.Normalize(vmin=min(e_y), vmax=max(e_y), clip=True), cmap=plt.cm.jet), ax=ax[1, 2])
    fig.colorbar(plt.cm.ScalarMappable(mpl.colors.Normalize(vmin=min(e_y), vmax=max(e_y), clip=True), cmap=plt.cm.jet), ax=ax[1, 3])
    fig.colorbar(plt.cm.ScalarMappable(mpl.colors.Normalize(vmin=min(e_xy), vmax=max(e_xy), clip=True), cmap=plt.cm.jet), ax=ax[2, 2])
    fig.colorbar(plt.cm.ScalarMappable(mpl.colors.Normalize(vmin=min(e_xy), vmax=max(e_xy), clip=True), cmap=plt.cm.jet), ax=ax[2, 3])
    fig.colorbar(plt.cm.ScalarMappable(mpl.colors.Normalize(vmin=min(u), vmax=max(u), clip=True), cmap=plt.cm.jet), ax=ax[3, 0])
    fig.colorbar(plt.cm.ScalarMappable(mpl.colors.Normalize(vmin=min(u), vmax=max(u), clip=True), cmap=plt.cm.jet), ax=ax[3, 1])
    fig.colorbar(plt.cm.ScalarMappable(mpl.colors.Normalize(vmin=min(v), vmax=max(v), clip=True), cmap=plt.cm.jet), ax=ax[3, 2])
    fig.colorbar(plt.cm.ScalarMappable(mpl.colors.Normalize(vmin=min(v), vmax=max(v), clip=True), cmap=plt.cm.jet), ax=ax[3, 3])
    
    plt.show()