import numpy as np
import pandas as pd
import torch

from push_task_utils.push_env import PushEnv
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

    env = PushEnv(ifRender=False)

    test_data = DataLoader(PushDataset(TEST_DATA_PATH), batch_size=1, shuffle=True)
    errors = []
    ground_truth_pushes = []
    predicted_pushes = []
    for i, (state, goal_state, action) in enumerate(test_data):

        # Get predicted action using inverse model
        state = state.float()
        goal_state = goal_state.float()
        action = action.float()
        predicted_action = inverse_model(state, goal_state)

        # Recast data as numpy arrays
        state = state.data.numpy()[0]
        action = action.data.numpy()[0]
        goal_state = goal_state.data.numpy()[0]
        predicted_action = predicted_action.data.numpy()[0]
        final_state = np.array(env.execute_push(*predicted_action))

        # Calculate errors
        errors.append(
            dict(
                action_error=np.linalg.norm(action - predicted_action, 2),
                state_error=np.linalg.norm(goal_state - final_state, 2)
            )
        )

        # Record push data for videos
        ground_truth_pushes.append(
            dict(
                obj_x=state[0],
                obj_y=state[1],
                start_push_x=action[0],
                start_push_y=action[1],
                end_push_x=action[2],
                end_push_y=action[3]
            )
        )
        predicted_pushes.append(
            dict(
                obj_x=state[0],
                obj_y=state[1],
                start_push_x=predicted_action[0],
                start_push_y=predicted_action[1],
                end_push_x=predicted_action[2],
                end_push_y=predicted_action[3]
            )
        )

        if i > NUM_EXAMPLES - 1:
            break

        pd.DataFrame(errors).to_csv(f"{INVERSE_MODEL_PATH}/inverse_model_errors.csv")
        pd.DataFrame(ground_truth_pushes).to_csv(f"{INVERSE_MODEL_PATH}/ground_truth_pushes.csv")
        pd.DataFrame(ground_truth_pushes).to_csv(f"{INVERSE_MODEL_PATH}/predicted_pushes.csv")
