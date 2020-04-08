from typing import List, Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as f

from models.mlp import MultiLayerPerceptron


class ForwardModel(nn.Module):

    def __init__(
            self,
            num_state_dims: int,
            num_action_dims: int,
            hidden_layer_sizes: List[int],
            activation: Callable = f.selu,
            seed: int = 0
    ):
        torch.manual_seed(seed)
        super(ForwardModel, self).__init__()
        self.mlp = MultiLayerPerceptron(
            in_features=num_state_dims + num_action_dims,
            hidden_layer_sizes=hidden_layer_sizes + [num_state_dims],
            activation=activation
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        input = torch.cat((state, action), dim=1)
        return self.mlp(input)


