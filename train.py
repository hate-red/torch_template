import torch
import time

from model import model, device
from data_setup import train_loader, test_loader
from engine import train


criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

start_time = time.time()

results = train(
    model=model,
    criterion=criterion,
    optimizer=optimizer,
    train_dataloader=train_loader,
    test_dataloader=test_loader,
    epochs=15,
    device=device
)

finish_time = time.time()

print(f'Train time: {finish_time - start_time :.2f}')