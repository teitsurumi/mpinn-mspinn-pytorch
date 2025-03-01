"""
@author: tsurumitei
@email: tsurumitei@foxmail.com

norm to [0, 1] or [-1, 1] in the current project
"""

import numpy as np

class norm_array_m1_1():
    """1. norm to [-1, 1]
    
    ```python
    a = norm_array(x)
    b = a.norm_forward()
    a.norm_backward(b)
    
    a_predict = network(xxx)
    a_predict.to_numpy()
    a_predict = a.norm_backward(a_predict)
    ```
    """
    def __init__(self, x: np.ndarray) -> None:
        self.arr = x
        self.XMID = 0
        self.XSPAN = 0
        self.param = list()

    def norm_forward(self):
        """forward pass
        """
        XMAX = self.arr.max()
        XMIN = self.arr.min()
        XMID = (XMAX + XMIN) / 2
        XSPAN = (XMAX - XMIN) / 2  # lenth: 2
        self.XMID = XMID
        self.XSPAN = XSPAN
        a = self.arr.copy() - XMID
        a = a / XSPAN
        self.param.append(XMID)
        self.param.append(1/XSPAN)
        return a, self.param

    def norm_backward(self, arr: np.ndarray):
        """backward pass
        
        the `arr` here is the target new array, for example, the predicted value
        """
        a_new = arr * self.XSPAN
        a_new = a_new + self.XMID
        return a_new


class norm_array_0_1():
    """2. norm to [0, 1]
    ```python
    a = norm_array(x)
    b = a.norm_forward()
    a.norm_backward(b)
    
    a_predict = network(xxx)
    a_predict.to_numpy()
    a_predict = a.norm_backward(a_predict)
    ```
    """
    def __init__(self, x: np.ndarray) -> None:
        self.arr = x
        self.XMIN = 0
        self.XSPAN = 0
        self.param = list()

    def norm_forward(self):
        """forward pass
        """
        XMAX = self.arr.max()
        XMIN = self.arr.min()
        XSPAN = XMAX - XMIN
        self.XMIN = XMIN
        self.XSPAN = XSPAN
        if self.XSPAN != 0:
            a = self.arr.copy() - XMIN
            a = a / XSPAN
            self.param.append(XMIN)
            self.param.append(1/XSPAN)
            return a, self.param
        else:
            self.param.append(XMIN)
            self.param.append(0)
            return self.arr, self.param
    
    def re_norm(self, new_arr: np.ndarray):
        """apply normalize to a new sample
        
        to generate the Boundary conditions here
        """
        if self.XSPAN != 0:
            a = new_arr.copy() - self.XMIN
            a = a / self.XSPAN
            return a
        else:
            return new_arr

    def norm_backward(self, arr: np.ndarray):
        """backward pass
        
        the `arr` here is the target new array, for example, the predicted value
        """
        if self.XSPAN != 0:
            a_new = arr * self.XSPAN
            a_new = a_new + self.XMIN
            return a_new
        else:
            return arr