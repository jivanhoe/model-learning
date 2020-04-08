import torch.nn as nn
import torch.nn.functional as f
import torch

from typing import Optional, Callable, List


class FullyConnectedLayer(nn.Module):

    def __init__(self, in_features: int, out_features: int, activation: Optional[Callable] = None):
        super(FullyConnectedLayer, self).__init__()
        self.weight_matrix = nn.Linear(in_features, out_features, bias=False)
        self.activation = activation

    def forward(self, input: torch.Tensor, activation: Optional[Callable] = None) -> torch.Tensor:
        output = self.weight_matrix(input)
        if self.activation:
            output = self.activation(output)
        return output


class MultiLayerPerceptron(nn.Module):

    def __init__(
            self,
            in_features: int,
            hidden_layer_sizes: List[int],
            activation: Callable = f.relu,
    ):
        super(MultiLayerPerceptron, self).__init__()
        self.num_hidden_layers = len(hidden_layer_sizes)
        self.network = nn.Sequential(
            *[
                FullyConnectedLayer(
                    in_features=hidden_layer_sizes[i - 1] if i > 0 else in_features,
                    out_features=hidden_layer_sizes[i],
                    activation=activation if i < self.num_hidden_layers else None
                )
                for i in range(self.num_hidden_layers)
            ]
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.network(input)
