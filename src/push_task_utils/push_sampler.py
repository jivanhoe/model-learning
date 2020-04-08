from typing import List, Tuple

import numpy as np
import torch

from push_task_utils.push_env import PushEnv


class PushSampler:

    def __init__(self, sigma: float, batch_size: int):
        self.environment = PushEnv(ifRender=False)
        self.sigma = sigma
        self.push_angle_mean = 0
        self.push_angle_std = np.pi * self.sigma
        self.push_length_mean = self.environment.push_len_min + self.environment.push_len_range / 2
        self.push_length_std = self.environment.push_len_range * self.sigma / 2
        self.batch_size = batch_size

    def sample_push_angle(self) -> float:
        while True:
            push_angle = self.push_angle_std * np.random.randn() + self.push_angle_mean
            if -np.pi < push_angle < np.pi:
                return push_angle

    def sample_push_length(self) -> float:
        while True:
            push_length = self.push_length_std * np.random.randn() + self.push_length_mean
            if self.environment.push_len_min < push_length < self.environment.push_len_min + self.environment.push_len_range:
                return push_length

    def sample(self, state: torch.Tensor) -> Tuple[float, float, torch.Tensor]:
        while True:
            push_angle = self.sample_push_angle()
            push_length = self.sample_push_length()
            action = self.environment.calculate_push_locations(state.data.numpy()[0], push_angle, push_length)
            if self.environment.push_is_feasible(*action):
                return push_angle, push_length, torch.from_numpy(np.array(action)).float().unsqueeze(0)

    def sample_batch(self, state: torch.Tensor) -> Tuple[np.ndarray, np.ndarray, List[torch.Tensor]]:
        push_angles, push_lengths, actions = zip(*[self.sample(state) for _ in range(self.batch_size)])
        return np.array(push_angles), np.array(push_lengths), actions

    def get_argmax_action(self, state: torch.Tensor) -> np.ndarray:
        return np.array(
            self.environment.calculate_push_locations(state.data.numpy()[0], self.push_angle_mean, self.push_length_mean)
        )

    def reset(self) -> None:
        self.push_angle_mean = 0
        self.push_length_mean = self.environment.push_len_min + self.environment.push_len_range / 2
