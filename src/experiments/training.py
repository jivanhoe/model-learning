import logging
from typing import List, Callable, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

from models.inverse_model import InverseModel
from models.forward_model import ForwardModel

# Set-up device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Set-up logging
logger = logging.getLogger(__name__)


def train(
        model: nn.Module,
        train_data: DataLoader,
        criterion: Callable = nn.MSELoss(),
        num_epochs: int = 10,
        learning_rate: float = 1e-3,
        model_path: Optional[str] = None,
        training_loss_path: Optional[str] = None
) -> None:

    # Send model to device and initialize optimizer
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    losses = []
    for i in range(num_epochs):

        epoch_losses = []
        for from_states, to_states, actions in train_data:

            # Send data to device
            from_states = from_states.float().to(DEVICE)
            to_states = to_states.float().to(DEVICE)
            actions = actions.float().to(DEVICE)

            # Compute prediction and loss
            if type(model) == InverseModel:
                preds = model(from_states, to_states).to(DEVICE)
                loss = criterion(preds, actions).to(DEVICE)
            elif type(model) == ForwardModel:
                preds = model(from_states, actions).to(DEVICE)
                loss = criterion(preds, to_states).to(DEVICE)
            else:
                raise NotImplementedError

            # Perform gradient step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Track progress
            epoch_losses.append(loss.cpu().item())

        # Log progress
        losses += epoch_losses
        logger.info(f"epochs completed: \t {i + 1}/{num_epochs}")
        logger.info(f"mean loss: \t {'{0:.2E}'.format(np.mean(epoch_losses))}")
        logger.info("-" * 50)

    if model_path:
        torch.save(model.state_dict(), f"{model_path}.pt")

    if training_loss_path:
        pd.DataFrame(losses).to_csv(f"{training_loss_path}.csv")

