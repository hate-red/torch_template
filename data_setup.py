from torch.utils.data import DataLoader

from pathlib import Path


root_dir = Path(__file__).parent
datasets_dir = root_dir / ''

train_folder = datasets_dir / ''
test_folder = datasets_dir / ''


train_dataset = 
test_dataset = 

CLASS_NAMES = train_dataset.classes
BATCH_SIZE = 32

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)
