import logging

import numpy as np
import torch

from models.forward_model import ForwardModel
from push_task_utils.push_env import PushEnv
from push_task_utils.push_sampler import PushSampler

env = PushEnv(ifRender=False)
logger = logging.getLogger(__name__)


class CrossEntropyPlanner:

    def __init__(
            self,
            forward_model: ForwardModel,
            num_iterations: int = 50,
            batch_size: int = 256,
            elite_action_quantile: float = 0.8,
            sample_std: float = 0.1,
            alpha: float = 1.0
    ):
        self.forward_model = forward_model
        self.num_iterations = num_iterations
        self.elite_actions_per_batch = int(np.round(batch_size * (1 - elite_action_quantile)))
        self.alpha = alpha
        self.sampler = PushSampler(batch_size=batch_size, sample_std=sample_std)

    def evaluate_action(self, action: torch.Tensor, state: torch.Tensor, goal_state: torch.Tensor, ) -> float:
        predicted_state = self.forward_model(action=action, state=state)
        return torch.norm(predicted_state - goal_state, 2).item()

    def update_sampler_params(
            self,
            push_angles: np.ndarray,
            push_lengths: np.ndarray,
            losses: np.ndarray
    ):
        elite_idxs = np.argsort(losses)[:self.elite_actions_per_batch]
        self.sampler.push_angle_mean = self.alpha * np.mean(push_angles[elite_idxs]) + (
                    1 - self.alpha) * self.sampler.push_angle_mean
        self.sampler.push_length_mean = self.alpha * np.mean(push_lengths[elite_idxs]) + (
                    1 - self.alpha) * self.sampler.push_length_mean

    def plan(
            self,
            state: torch.Tensor,
            goal_state: torch.Tensor
    ) -> np.array:
        for i in range(self.num_iterations):
            push_angles, push_lengths, actions = self.sampler.sample_batch(state=state)
            losses = np.array(
                [
                    self.evaluate_action(
                        action=action,
                        state=state,
                        goal_state=goal_state
                    ) for action in actions
                ]
            )
            self.update_sampler_params(
                push_angles=push_angles,
                push_lengths=push_lengths,
                losses=losses
            )
            logger.info(f"epochs completed: \t {i + 1}/{self.num_iterations}")
            logger.info(f"mean loss: \t {'{0:.2E}'.format(np.mean(losses))}")
            logger.info("-" * 50)
        return self.sampler.get_argmax_action(state=state)


