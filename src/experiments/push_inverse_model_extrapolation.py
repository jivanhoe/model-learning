import numpy as np
import pandas as pd
import torch

from push_task_utils.push_env import PushEnv
from push_task_utils.extrapolation import extrapolate, make_infer_callback_for_inverse_model
from experiments.push_model_learning import *

# PATHS
INVERSE_MODEL_PATH = f"{RESULTS_PATH}/inverse_model"

# PARAMS
NUM_EXAMPLES = 10


if __name__ == "__main__":

    inverse_model = InverseModel(
        num_state_dims=NUM_STATE_DIMS,
        num_action_dims=NUM_ACTION_DIMS,
        hidden_layer_sizes=HIDDEN_LAYER_SIZES
    )
    inverse_model.load_state_dict(torch.load(f"{INVERSE_MODEL_PATH}/inverse_model_params.pt"))
    infer = make_infer_callback_for_inverse_model(inverse_model)

    env = PushEnv(ifRender=False)

    errors = []
    ground_truth_pushes = []
    predicted_pushes = []
    for i in range(NUM_EXAMPLES):

        # Sample push
        state, goal_state, (action_1, action_2) = env.sample_multi_step_push(seed=i)

        final_state, (predicted_action_1, predicted_action_2) = extrapolate(
            state=state,
            goal_state=goal_state,
            infer=infer,
            env=env
        )

        # Record push data for videos
        ground_truth_pushes.append(
            dict(
                obj_x=state[0],
                obj_y=state[1],
                start_push_x_1=action_1[0],
                start_push_y_1=action_1[1],
                end_push_x_1=action_1[2],
                end_push_y_1=action_1[3],
                start_push_x_2=action_2[0],
                start_push_y_2=action_2[1],
                end_push_x_2=action_2[2],
                end_push_y_2=action_2[3]
            )
        )
        predicted_pushes.append(
            dict(
                obj_x=state[0],
                obj_y=state[1],
                start_push_x_1=predicted_action_1[0],
                start_push_y_1=predicted_action_1[1],
                end_push_x_1=predicted_action_1[2],
                end_push_y_1=predicted_action_1[3],
                start_push_x_2=predicted_action_2[0],
                start_push_y_2=predicted_action_2[1],
                end_push_x_2=predicted_action_2[2],
                end_push_y_2=predicted_action_2[3]
            )
        )

        # Calculate errors
        errors.append(
            dict(
                action_1_error=np.linalg.norm(action_1 - predicted_action_1, 2),
                action_2_error=np.linalg.norm(action_2 - predicted_action_2, 2),
                state_error=np.linalg.norm(goal_state - final_state, 2)
            )
        )

    pd.DataFrame(errors).to_csv(f"{INVERSE_MODEL_PATH}/inverse_model_extrapolation_errors.csv")
    pd.DataFrame(ground_truth_pushes).to_csv(f"{INVERSE_MODEL_PATH}/ground_truth_two_step_pushes.csv")
    pd.DataFrame(ground_truth_pushes).to_csv(f"{INVERSE_MODEL_PATH}/predicted_two_step_pushes.csv")
