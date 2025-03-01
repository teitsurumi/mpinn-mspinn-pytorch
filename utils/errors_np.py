import numpy as np
import torch


def mse(a: np.ndarray, b: np.ndarray):
    a = a.flatten()
    b = b.flatten()
    if len(a) != len(b):
        print('\033[31m' + "Error" + '\033[0m' + ": (mse) Length of the input array collapsed.")
        return 0
    _a = torch.tensor(a, dtype=torch.float32)
    _b = torch.tensor(b, dtype=torch.float32)
    mse_loss = torch.nn.MSELoss()
    ans = mse_loss(_a, _b)
    ans = __to_numpy(ans)
    return ans

def rmse(a: np.ndarray, b: np.ndarray):
    a = a.flatten()
    b = b.flatten()
    if len(a) != len(b):
        print('\033[31m' + "Error" + '\033[0m' + ": (rmse) Length of the input array collapsed.")
        return 0
    _a = torch.tensor(a, dtype=torch.float32)
    _b = torch.tensor(b, dtype=torch.float32)
    mse_loss = torch.nn.MSELoss()
    ans = mse_loss(_a, _b)
    ans = __to_numpy(ans)
    ans = np.sqrt(ans)
    return ans



def __to_numpy(input):
    if isinstance(input, torch.Tensor):
        return input.detach().cpu().numpy()
    elif isinstance(input, np.ndarray):
        return input
    else:
        raise TypeError('Unknown type of input, expected torch.Tensor or ' \
                        'np.ndarray, but got {}'.format(type(input)))

# if __name__ == "__main__":
#     a = np.array([[1], [2], [4]])
#     b = np.array([[1], [2], [4], [3]])
#     ans = mse(a, b)
#     print(ans)