import torch

from tqdm.auto import tqdm
from collections import namedtuple


TrainResults = namedtuple('Losses', ['train_loss', 'train_acc'])
TestResults = namedtuple('Losses', ['test_loss', 'test_acc'])


def train_step(
        model: torch.nn.Module, 
        dataloader: torch.utils.data.DataLoader, 
        criterion: torch.nn.Module, 
        optimizer: torch.optim.Optimizer,
        device: str = 'cpu',
    ) -> TrainResults:
    """
    Trains a PyTorch model for a single epoch.
    """
    model.train()

    train_loss, train_acc = 0, 0

    for X, y in dataloader:
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = criterion(y_pred, y)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)

        train_loss += loss.item() 
        train_acc += torch.eq(y_pred_class, y).sum().item()/len(y_pred)

    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return TrainResults(train_loss, train_acc)


def test_step(
        model: torch.nn.Module, 
        dataloader: torch.utils.data.DataLoader, 
        criterion: torch.nn.Module,
        device: str = 'cpu'
    ) -> TestResults:
    """
    Tests a PyTorch model for a single epoch.
    """
    model.eval() 

    test_loss, test_acc = 0, 0

    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            y_logits = model(X)

            loss = criterion(y_logits, y)
            pred_labels = y_logits.argmax(dim=1)

            test_loss += loss.item()
            test_acc += torch.eq(pred_labels, y).sum().item() / len(pred_labels)

    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)

    return TestResults(test_loss, test_acc)


def train(
        model: torch.nn.Module, 
        train_dataloader: torch.utils.data.DataLoader, 
        test_dataloader: torch.utils.data.DataLoader, 
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        epochs: int = 10,
        device: str  = 'cpu',
    ) -> dict[str, list[float]]:
    """
    Trains and tests a PyTorch model.
    """

    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }

    model.to(device)

    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )
        test_loss, test_acc = test_step(
            model=model,
            dataloader=test_dataloader,
            criterion=criterion,
            device=device
        )

        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results
