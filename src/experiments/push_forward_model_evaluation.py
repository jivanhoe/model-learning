import torch

from experiments.push_model_learning import *
from cross_entropy.cross_entropy import CrossEntropyPlanner

# PATHS
FORWARD_MODEL_PATH = f"{RESULTS_PATH}/forward_model_params.pt"

# CE METHOD PARAMS
CE_BATCH_SIZE = 2048
SIGMA = 1e-2  # exploration param (sigma = 1 -> 1 std = valid sampling range)
ALPHA = 1.0  # update smoothing param (alpha = 1 -> no smoothing)
NUM_EXAMPLES = 10

if __name__ == "__main__":

    forward_model = ForwardModel(
        num_state_dims=NUM_STATE_DIMS,
        num_action_dims=NUM_ACTION_DIMS,
        hidden_layer_sizes=HIDDEN_LAYER_SIZES
    )
    forward_model.load_state_dict(torch.load(FORWARD_MODEL_PATH))

    ce_planner = CrossEntropyPlanner(
        forward_model=forward_model,
        batch_size=CE_BATCH_SIZE,
        sigma=SIGMA,
        alpha=ALPHA
    )

    test_data = DataLoader(PushDataset(TEST_DATA_PATH), batch_size=1, shuffle=True)
    actions = []
    predicted_actions = []
    for i, (state, goal_state, action) in enumerate(test_data):
        state = state.float()
        goal_state = goal_state.float()
        action = action.float()
        planned_action = ce_planner.plan_action(state=state, goal_state=goal_state)
        torch.norm(action - planned_action, 2).item() ** 2

        if i > NUM_EXAMPLES - 1:
            break
