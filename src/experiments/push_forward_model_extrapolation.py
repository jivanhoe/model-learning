import numpy as np
import pandas as pd
import torch

from push_task_utils.extrapolation import extrapolate, make_infer_callback_for_forward_model
from cross_entropy.cross_entropy_planner import CrossEntropyPlanner
from experiments.push_model_learning import *

# PATHS
FORWARD_MODEL_PATH = f"{RESULTS_PATH}/forward_model"

# CE METHOD PARAMS
CE_BATCH_SIZE = 2048
NUM_ITERATIONS = 150
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
    infer = make_infer_callback_for_forward_model(ce_planner)

    errors = []
    ground_truth_pushes = []
    predicted_pushes = []
    for i in range(NUM_EXAMPLES):

        # Sample push
        state, goal_state, (action_1, action_2) = ce_planner.sampler.environment.sample_multi_step_push(seed=i)

        final_state, (predicted_action_1, predicted_action_2) = extrapolate(
            state=state,
            goal_state=goal_state,
            infer=infer,
            env=ce_planner.sampler.environment
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

        pd.DataFrame(errors).to_csv(f"{FORWARD_MODEL_PATH}/forward_model_extrapolation_errors.csv")
        pd.DataFrame(ground_truth_pushes).to_csv(f"{FORWARD_MODEL_PATH}/ground_truth_two_step_pushes.csv")
        pd.DataFrame(ground_truth_pushes).to_csv(f"{FORWARD_MODEL_PATH}/predicted_two_step_pushes.csv")
