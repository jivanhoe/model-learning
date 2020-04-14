from push_task_utils.visualization import save_push_video
from push_task_utils.push_env import PushEnv
from experiments.push_forward_model_evaluation import FORWARD_MODEL_PATH
from experiments.push_inverse_model_evaluation import INVERSE_MODEL_PATH
import pandas as pd
import numpy as np

env = PushEnv(ifRender=True)

for model_path in (INVERSE_MODEL_PATH, FORWARD_MODEL_PATH):
    for push_type in ("ground_truth", "predicted"):

        # for i, push in pd.read_csv(f"{model_path}/{push_type}_pushes.csv", index_col=0).iterrows():
        #     state = np.array([push["obj_x"], push["obj_y"]])
        #     actions = [np.array([push["start_push_x"], push["start_push_y"], push["end_push_x"], push["end_push_y"]])]
        #     save_push_video(
        #         state=state,
        #         actions=actions,
        #         env=env,
        #         path=f"{model_path}/videos/{push_type}_push_{i}.mp4"
        #     )

        for i, push in pd.read_csv(f"{model_path}/{push_type}_two_step_pushes.csv", index_col=0).iterrows():
            state = np.array([push["obj_x"], push["obj_y"]])
            actions = [
                np.array([push["start_push_x_1"], push["start_push_y_1"], push["end_push_x_1"], push["end_push_y_1"]]),
                np.array([push["start_push_x_2"], push["start_push_y_2"], push["end_push_x_2"], push["end_push_y_2"]])
            ]
            save_push_video(
                state=state,
                actions=actions,
                env=env,
                path=f"{model_path}/videos/{push_type}_two_step_push_{i}.mp4"
            )
