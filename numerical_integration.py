"""
@author: tsurumitei
@email: tsurumitei@foxmail.com

Delaunay integration
"""

import torch
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from utils.sequence import ArithmeticSequenceSolverModified as SeqM


def _area(triangle: np.ndarray) -> float:
    """
        ```
        points = np.concatenate((x, y), axis=1)  # n*2
        tri = Delaunay(points)
        triangles = points[tri.simplices]
        triangle = triangle[num]
        ```
    """
    a = triangle[1] - triangle[0]
    b = triangle[2] - triangle[0]
    area_triangle = float(abs(a[0]*b[1] - a[1]*b[0]) / 2)  # |x1y2 - x2y1| / 2
    return area_triangle

def area(triangles: np.ndarray) -> np.ndarray:
    n = triangles.shape[0]
    area_list = np.ndarray((n, 1))
    for i in range(n):
        area_list[i] = _area(triangles[i])
    return area_list


def gen_points():
    splits_angle = 40
    num_ellip = 40
    # Structral node generation
    solver = SeqM()
    # seed on ellipse
    start, stop, num = np.pi/2, 0, splits_angle
    phi = solver.a0_an_n(start, stop, num)
    k = np.tan(phi[1:]).reshape((num-1, 1))
    start, stop, num = 0.02, 0.1*np.sqrt(1.16), num_ellip
    a = solver.a0_an_n(start, stop, num)
    b = 10*a
    ellips = np.ndarray((0, 2))
    _a = np.ndarray((0, 2))
    for i in range(num_ellip):
        x = a[i] * np.sqrt(1 / (1 + np.square(k/10)))
        y = x * k
        coor = np.concatenate((x,y), axis=1)
        ellips = np.concatenate((ellips, coor), axis=0)
    _a = np.concatenate(
        (a[0] * np.sqrt(1 / (1 + np.square(k/10))), a[0] * np.sqrt(1 / (1 + np.square(k/10)))*k),
        axis=1
    )
    _a = np.concatenate(
        (_a, np.array([0, 0.2]).reshape((1, 2))),
        axis=0
    )
    y = np.array(b, dtype=np.float32).reshape((num_ellip, 1))
    x = np.zeros(y.shape)
    ellips = np.concatenate(
        (ellips, np.concatenate((x, y), axis=1)),
        axis=0
    )
    ellips = ellips[ellips[:, 0] < 0.1]
    ellips = ellips[ellips[:, 1] < 0.4]
    
    x = np.linspace(0, 0.6, 41)
    y = np.linspace(0, 0.6, 41)
    x, y = np.meshgrid(x, y)
    x = x.reshape((x.shape[0]*x.shape[1], 1))
    y = y.reshape((y.shape[0]*y.shape[1], 1))
    all_points = np.concatenate((x, y), axis=1)
    outer_points = all_points[np.logical_or(all_points[:,0]>=0.1, all_points[:,1]>=0.4)]
    
    vertex_points = np.concatenate((ellips, outer_points), axis=0)
    
    # # ----------------------------- generate sample points -----------------------------
    # rbd = vertex_points[np.abs(vertex_points[:, 0] - 0.6) <= 1e-10 , :]
    # np.savetxt("./data/shape6_rbd_points2.txt", rbd, delimiter=" ")
    # np.savetxt("./data/shape6_strain_energy_points2.txt", vertex_points, delimiter=" ")
    
    tri = Delaunay(vertex_points)
    delaunay_triangles = vertex_points[tri.simplices]
    
    # Geometric Boolean operation
    delaunay_triangles_filtered = list()
    vertexes_filtered = list()
    for i in range(delaunay_triangles.shape[0]):
        if delaunay_triangles[i, 0, :] in _a and delaunay_triangles[i, 1, :] in _a and delaunay_triangles[i, 2, :] in _a:
            pass
        else:
            delaunay_triangles_filtered.append(delaunay_triangles[i])
            vertexes_filtered.append(tri.simplices[i])

    delaunay_triangles_filtered = np.array(delaunay_triangles_filtered)
    delaunay_tri_area = area(delaunay_triangles_filtered)
    return vertex_points, tri, delaunay_tri_area, delaunay_triangles_filtered, vertexes_filtered

# def iint():
#     torch.



if __name__ == '__main__':
    # def __network():
    #     """test nn"""
    #     net = torch.nn.Sequential()
    #     net.add_module('Linear_layer_1', torch.nn.Linear(2, 64))
    #     net.add_module('Tanh_layer_1', torch.nn.Tanh())
    #     for num in range(2, 6):
    #         net.add_module('Linear_layer_%d' % (num), torch.nn.Linear(64, 64))
    #         net.add_module('Tanh_layer_%d' % (num), torch.nn.Tanh())
    #     net.add_module('Linear_layer_final', torch.nn.Linear(64, 1))
    #     net = net.cuda("cuda:0")
    #     return net
    def __network():
        """mocked output"""
        def net(X: torch.Tensor):
            return torch.ones_like(X[:, 0:1]).to(X.device)
        return net
    def to_numpy(input):
        if isinstance(input, torch.Tensor):
            return input.detach().cpu().numpy()
        elif isinstance(input, np.ndarray):
            return input
        else:
            raise TypeError('Unknown type of input, expected torch.Tensor or ' \
                            'np.ndarray, but got {}'.format(type(input)))

    vertex_points, tri, delaunay_tri_area, delaunay_triangles_filtered, vertexes_filtered = gen_points()
    
    delaunay_triangles_filtered = np.array(delaunay_triangles_filtered)
    delaunay_tri_area = np.array(delaunay_tri_area)
    # ################ SAVE AS VERTICES ################
    # np.savez(
    #     "./data/shape6_delaunay2.npz",
    #     delaunay_triangles_filtered=delaunay_triangles_filtered,
    #     delaunay_tri_area=delaunay_tri_area
    # )

    ################ PREVIEW MESH ################
    plt.figure(0, (6,6))
    plt.triplot(vertex_points[:,0], vertex_points[:,1], vertexes_filtered)
    plt.show()

    delaunay_triangles_filtered = np.array(delaunay_triangles_filtered)
    delaunay_triangles_filtered = delaunay_triangles_filtered.reshape((delaunay_triangles_filtered.shape[0] * 3, 2))

    xx = delaunay_triangles_filtered[:, 0].copy()
    yy = delaunay_triangles_filtered[:, 1].copy()

    device = torch.device("cuda:0")
    vertexes_x = torch.tensor(xx, dtype=torch.float32, requires_grad=True).reshape((xx.shape[0], 1)).to(device)
    vertexes_y = torch.tensor(yy, dtype=torch.float32, requires_grad=True).reshape((xx.shape[0], 1)).to(device)
    delaunay_tri_area = torch.tensor(delaunay_tri_area, dtype=torch.float32, requires_grad=True).reshape((int(xx.shape[0]/3), 1)).to(device)
    func = __network()
    func_value = func(torch.hstack((vertexes_x, vertexes_y))).to(device)
    
    # CAUTION: For computational graphs, assigning values during training is an illegal operation.
    # func_value_mean = torch.tensor(
    #                     np.zeros(int(xx.shape[0]/3)),
    #                     dtype=torch.float32, requires_grad=True
    #                 ).reshape((int(xx.shape[0]/3), 1)).to(device)
    # for i in range(int(xx.shape[0]/3)):
    #     func_value_mean[i] = torch.sum(func_value[3*i:3*(i+1)]) / 3.0
    # This must be alternated by the method below:
    func_value_mean = torch.sum(func_value.view((int(func_value.shape[0]/3), 3)), 1, keepdim=True) / 3
    
    print(to_numpy(torch.sum(func_value_mean * delaunay_tri_area)))
    print(0.6**2 - np.pi * 0.02 * 0.2 * 0.25)
    print(np.abs((0.6**2 - np.pi * 0.02 * 0.2 * 0.25) - to_numpy(torch.sum(func_value_mean * delaunay_tri_area))) / (0.6**2 - np.pi * 0.02 * 0.2 * 0.25) * 100, "%")
