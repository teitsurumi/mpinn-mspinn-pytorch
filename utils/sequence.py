"""
@author: tsurumitei
@email: tsurumitei@foxmail.com

generate the sequence

used to generate structural mesh for the elliptical structure
"""

import numpy as np


class ArithmeticSequenceSolver:
    def __init__(self) -> None:
        pass


class ArithmeticSequenceSolverModified:
    def __init__(self) -> None:
        pass
    
    def a0_an_n(self, start: float, stop: float, n: float) -> np.ndarray:
        """n+1 terms
        
        a_1-a_0=d
        
        ...
        
        a_n-a_{n-1}=nd
        
        => d = \\frac{2(a_n-a_0)}{n(n+1)}
        
        >>> solver = ArithmeticSequenceSolverModified()
        >>> start, stop, num = 1, 10, 50
        >>> ans = solver.a0_an_n(start, stop, num)
        >>> print(ans[1]-ans[0], (ans[2]-ans[1])/2, (ans[3]-ans[2])/3)
        >>> print(ans.shape)
        """
        n = n-1
        d = 2 * (stop - start) / (n * (n+1))
        ans = list()
        ans.append(start)
        a_i = start
        for i in range(1, n+1, 1):
            a_i += i*d
            ans.append(a_i)
        ans = np.array(ans)
        return ans


if __name__ =='__main__':
    solver = ArithmeticSequenceSolverModified()
    start, stop, num = 5, 10, 50
    ans = solver.a0_an_n(start, stop, num)
    print(ans.shape)
    print(ans[1]-ans[0], (ans[2]-ans[1])/2, (ans[3]-ans[2])/3)
    print(ans)
    
    import matplotlib.pyplot as plt
    plt.scatter(ans, ans)
    plt.show()