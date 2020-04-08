import torch

from experiments.push_model_learning import *
from cross_entropy.cross_entropy import CrossEntropyPlanner


FORWARD_MODEL_PATH = f"{RESULTS_PATH}/forward_model_params.pt"

forward_model = ForwardModel(
    num_state_dims=NUM_STATE_DIMS,
    num_action_dims=NUM_ACTION_DIMS,
    hidden_layer_sizes=HIDDEN_LAYER_SIZES
)
forward_model.load_state_dict(torch.load(FORWARD_MODEL_PATH))

planner = CrossEntropyPlanner(forward_model=forward_model)

train_data = DataLoader(PushDataset(TRAINING_DATA_PATH), batch_size=1, shuffle=True)
for data in train_data:
    state, goal_state, action = data
    state = state.float()
    goal_state = goal_state.float()
    action = action.float()
    planned_action = planner.plan_action(state=state.float(), goal_state=goal_state)
    torch.norm(action - planned_action, 2)
