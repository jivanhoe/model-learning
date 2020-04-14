import logging
from typing import Dict, Tuple

import numpy as np
import torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader

from models.forward_model import ForwardModel
from models.inverse_model import InverseModel
from models.training import DEVICE

logger = logging.getLogger(__name__)


def get_data_for_metrics(
        model: nn.Module,
        data: DataLoader
) -> Tuple[np.array, np.array]:
    predictions = []
    targets = []
    for from_states, to_states, actions in data:

        from_states = from_states.float().to(DEVICE)
        to_states = to_states.float().to(DEVICE)
        actions = actions.float().to(DEVICE)

        if type(model) == InverseModel:
            predictions.append(
                model(from_states, to_states).detach().data.cpu().numpy()
            )
            targets.append(actions.data.cpu().numpy())
        elif type(model) == ForwardModel:
            predictions.append(
                model(from_states, actions).detach().data.cpu().numpy()
            )
            targets.append(to_states.data.cpu().numpy())
        else:
            raise NotImplementedError
    return np.concatenate(predictions), np.concatenate(targets)


def calculate_metrics(
        model: nn.Module,
        data: DataLoader,
) -> Dict[str, float]:
    predictions, targets = get_data_for_metrics(model=model, data=data)
    return {
        metric.__name__: metric(targets, predictions)
        for metric in (
                r2_score,
                mean_absolute_error,
                mean_squared_error
        )
    }


def log_metrics(
        model: nn.Module,
        train_data: DataLoader,
        test_data: DataLoader
) -> None:
    for data_name, data in [
        ("training", train_data),
        ("test", test_data)
    ]:
        logger.info(f"calculating {data_name} metrics...")
        for metric_name, metric_value in calculate_metrics(model=model, data=data).items():
            logger.info(f"{metric_name}: \t {'{0:.2E}'.format(metric_value)}")
        logger.info("-" * 50)
