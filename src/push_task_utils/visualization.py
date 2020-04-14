import pybullet as p
from push_task_utils.push_env import PushEnv
import numpy as np

from typing import Optional, List, Union


def save_push_video(
        state: np.array,
        actions: List[np.array],
        path: str,
        env: Optional[PushEnv] = None,
        return_state: bool = False
) -> Optional[np.ndarray]:
    if env is None:
        env = PushEnv(ifRender=True)
    p.startStateLogging(p.STATE_LOGGING_VIDEO_MP4, path)
    env.reset_box(pos=[state[0], state[1], env.box_z])
    for action in actions:
        _, state = env.execute_push(*action)
    p.stopStateLogging(p.STATE_LOGGING_VIDEO_MP4)
    if return_state:
        return np.array(state)


