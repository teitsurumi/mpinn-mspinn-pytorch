"""Neural Network Templates
    
- `Functional`: Fully connected NN
    - input_dim: int
    - output_dim: int
    - hidden_layers: List[int]
    - activation: str = "tanh" | "relu" | "prelu" | 'sin'
"""
import torch
from torch import nn

torch.random.manual_seed(12345)

class SinActivation(nn.Module):
    """`sin`"""
    def __init__(self):
        super(SinActivation, self).__init__()
    
    def forward(self, x):
        return torch.sin(x)


class Functional(nn.Module):
    """
    :param input_dim: int
    :param output_dim: int
    :param hidden_layers: List[int]
    :param activation: str = "tanh" | "relu" | "prelu" | 'sin'
    """
    def __init__(self, input_dim, output_dim, hidden_layers, activation="tanh"):
        super(Functional, self).__init__()
        self.input_dim     = input_dim
        self.output_dim    = output_dim
        self.hidden_layers = hidden_layers
        
        activations = {
            'relu': nn.ReLU,
            'tanh': nn.Tanh,
            'prelu': nn.PReLU,
            'sin': SinActivation
        }
        if activation not in activations:
            raise ValueError(f"Activation '{activation}' is not supported")
        self.activation = activations[activation]

        self.module_list = nn.ModuleList()
        self.module_list.append(nn.Linear(self.input_dim, self.hidden_layers[0]))
        if len(hidden_layers) == 1:
            pass
        elif len(hidden_layers) > 1:
            for i in range(len(self.hidden_layers)-1):
                self.module_list.append(self.activation())
                self.module_list.append(nn.Linear(self.hidden_layers[i], self.hidden_layers[i+1]))
        self.module_list.append(self.activation())
        self.module_list.append(nn.Linear(self.hidden_layers[-1], self.output_dim))
    
    def forward(self, x):
        for layer in self.module_list:
            x = layer(x)
        return x


if __name__ == "__main__":
    x_test = torch.linspace(0, 1, 20, dtype=torch.float32, requires_grad=True)
    func = lambda x: torch.sin(x)
    f_test = func(x_test)
    model1 = Functional(1, 1, [4, 16, 16, 8], "sin")
    model2 = Functional(1, 1, [4])
    print(model1)
    print(model2)