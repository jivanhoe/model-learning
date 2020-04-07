from torch.utils.data import DataLoader
from push_task_utils.push_data_processing import PushDataset

TRAINING_DATA_PATH = "../../data/train"
TEST_DATA_PATH = "../../data/test"
BATCH_SIZE = 64

train_data_loader = DataLoader(PushDataset(TRAINING_DATA_PATH), batch_size=BATCH_SIZE, shuffle=True)
test_data_loader = DataLoader(PushDataset(TEST_DATA_PATH), batch_size=BATCH_SIZE, shuffle=True)
