import torch
from cross_entropy.cross_entropy import CrossEntropyPlanner
from experiments.push_model_learning import NUM_STATE_DIMS, NUM_ACTION_DIMS, HIDDEN_LAYER_SIZES, TRAINING_DATA_PATH
from models.forward_model import ForwardModel
from torch.utils.data import DataLoader
from push_task_utils.data_processing import PushDataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FORWARD_MODEL_PATH = "../../data/results/forward_model_params.pt"

forward_model = ForwardModel(
    num_state_dims=NUM_STATE_DIMS,
    num_action_dims=NUM_ACTION_DIMS,
    hidden_layer_sizes=HIDDEN_LAYER_SIZES
).load_state_dict(torch.load(FORWARD_MODEL_PATH))

planner = CrossEntropyPlanner(forward_model=forward_model)

train_data = DataLoader(PushDataset(TRAINING_DATA_PATH), batch_size=1, shuffle=True)
for data in train_data:
    state, goal_state, action = data
    break

planned_action = planner.plan(state=state, goal_state=goal_state)
