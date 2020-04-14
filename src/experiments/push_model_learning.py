import logging

from torch.utils.data import DataLoader

from experiments.metrics import log_metrics
from models.training import train
from models.inverse_model import InverseModel
from models.forward_model import ForwardModel
from push_task_utils.data_processing import PushDataset

# Set-up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
TRAINING_DATA_PATH = "../../data/train"
TEST_DATA_PATH = "../../data/test"
RESULTS_PATH = "../../data/results"

# Environment params
NUM_STATE_DIMS = 2
NUM_ACTION_DIMS = 4

# Model architecture
HIDDEN_LAYER_SIZES = [32, 16]

# Training params
BATCH_SIZE = 512
NUM_EPOCHS = 50
LEARNING_RATE = 3e-4

if __name__ == "__main__":

    train_data = DataLoader(PushDataset(TRAINING_DATA_PATH), batch_size=BATCH_SIZE, shuffle=True)
    test_data = DataLoader(PushDataset(TEST_DATA_PATH), batch_size=BATCH_SIZE, shuffle=True)

    for model_name, model_type in [
        ("inverse", InverseModel),
        ("forward", ForwardModel)
    ]:

        model = model_type(
            num_state_dims=NUM_STATE_DIMS,
            num_action_dims=NUM_ACTION_DIMS,
            hidden_layer_sizes=HIDDEN_LAYER_SIZES
        )

        logger.info(f"training {model_name} model...")
        logger.info("-" * 50)
        train(
            model=model,
            train_data=train_data,
            learning_rate=LEARNING_RATE,
            num_epochs=NUM_EPOCHS,
            model_path=f"{RESULTS_PATH}/{model_name}/model_params",
            training_loss_path=f"{RESULTS_PATH}/{model_name}/training_loss"
        )
        log_metrics(
            model=model,
            train_data=train_data,
            test_data=test_data
        )

