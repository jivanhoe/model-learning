from typing import List, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as f

from models.mlp import MultiLayerPerceptron


class InverseModel(nn.Module):

    def __init__(
            self,
            num_state_dims: int,
            num_action_dims: int,
            hidden_layer_sizes: List[int],
            activation: Callable = f.selu,
            seed: int = 0
    ):
        torch.manual_seed(seed)
        super(InverseModel, self).__init__()
        self.mlp = MultiLayerPerceptron(
            in_features=num_state_dims * 2,
            hidden_layer_sizes=hidden_layer_sizes + [num_action_dims],
            activation=activation
        )

    def forward(self, from_state: torch.Tensor, to_state: torch.Tensor) -> torch.Tensor:
        input = torch.cat((from_state, to_state), dim=1)
        return self.mlp(input)

