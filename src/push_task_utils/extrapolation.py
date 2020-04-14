from typing import Callable, List, Tuple

import numpy as np
import torch

from cross_entropy.cross_entropy_planner import CrossEntropyPlanner
from models.inverse_model import InverseModel
from push_task_utils.push_env import PushEnv


def extrapolate(
        state: np.ndarray,
        goal_state: np.ndarray,
        infer: Callable,
        env: PushEnv,
        num_steps: int = 2
) -> Tuple[np.ndarray, List[np.ndarray]]:
    env.reset_box()
    actions = []
    for _ in range(num_steps):
        action = infer(state, goal_state)
        actions.append(action)
        _, state = env.execute_push(*action)
    env.reset_box()
    return state, actions


def make_infer_callback_for_inverse_model(inverse_model: InverseModel) -> Callable:
    def infer(
            state: np.ndarray,
            goal_state: np.ndarray
    ) -> np.ndarray:
        state = torch.from_numpy(state).float().unsqueeze(0)
        goal_state = torch.from_numpy(goal_state).float().unsqueeze(0)
        return inverse_model(state, goal_state).data.numpy()[0]
    return infer


def make_infer_callback_for_forward_model(ce_planner: CrossEntropyPlanner) -> Callable:
    def infer(
            state: np.ndarray,
            goal_state: np.ndarray
    ) -> np.ndarray:
        state = torch.tensor(state).float()
        goal_state = torch.tensor(goal_state).float()
        return ce_planner.plan_action(state, goal_state)
    return infer


