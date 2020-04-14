import numpy as np
import pandas as pd
import torch

from cross_entropy.cross_entropy_planner import CrossEntropyPlanner
from experiments.push_model_learning import *

# PATHS
FORWARD_MODEL_PATH = f"{RESULTS_PATH}/forward_model"

# CE METHOD PARAMS
CE_BATCH_SIZE = 2048
NUM_ITERATIONS = 200
SIGMA = 1e-2  # exploration param (sigma = 1 -> 1 std = valid sampling range)
ALPHA = 1.0  # update smoothing param (alpha = 1 -> no smoothing)
NUM_EXAMPLES = 10


if __name__ == "__main__":

    forward_model = ForwardModel(
        num_state_dims=NUM_STATE_DIMS,
        num_action_dims=NUM_ACTION_DIMS,
        hidden_layer_sizes=HIDDEN_LAYER_SIZES
    )
    forward_model.load_state_dict(torch.load(f"{FORWARD_MODEL_PATH}/forward_model_params.pt"))

    ce_planner = CrossEntropyPlanner(
        forward_model=forward_model,
        batch_size=CE_BATCH_SIZE,
        num_iterations=NUM_ITERATIONS,
        sigma=SIGMA,
        alpha=ALPHA
    )

    test_data = DataLoader(PushDataset(TEST_DATA_PATH), batch_size=1, shuffle=True)
    errors = []
    ground_truth_pushes = []
    predicted_pushes = []
    num_failures = 0
    for i, (state, goal_state, action) in enumerate(test_data):

        # Get predicted action using CE planner
        state = state.float()[0]
        goal_state = goal_state.float()[0]
        action = action.float()[0]
        try:
            predicted_action = ce_planner.plan_action(state=state, goal_state=goal_state)
        except ValueError:
            predicted_action = None
            num_failures += 1

        if predicted_action is not None:

            # Recast data as numpy arrays
            state = state.data.numpy()
            action = action.data.numpy()
            goal_state = goal_state.data.numpy()
            final_state = np.array(ce_planner.sampler.environment.execute_push(*predicted_action))

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

        if i - num_failures > NUM_EXAMPLES - 1:
            break

        pd.DataFrame(errors).to_csv(f"{FORWARD_MODEL_PATH}/forward_model_errors.csv")
        pd.DataFrame(ground_truth_pushes).to_csv(f"{FORWARD_MODEL_PATH}/ground_truth_pushes.csv")
        pd.DataFrame(ground_truth_pushes).to_csv(f"{FORWARD_MODEL_PATH}/predicted_pushes.csv")
