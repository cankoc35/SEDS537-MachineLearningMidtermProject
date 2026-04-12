"""Training helpers for Question 5."""

import copy

import pandas as pd
import torch


class EarlyStopping:
    """Minimal early stopping utility for Question 5."""

    def __init__(self, patience: int = 5) -> None:
        self.patience = patience
        self.best_loss = None
        self.counter = 0

    def should_stop(self, validation_loss: float) -> bool:
        """Return True when validation loss has not improved for `patience` checks."""
        if self.best_loss is None or validation_loss < self.best_loss:
            self.best_loss = validation_loss
            self.counter = 0
            return False

        self.counter += 1
        return self.counter >= self.patience


def get_device() -> torch.device:
    """Choose the best available device for training."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_one_epoch(model, data_loader, loss_fn, optimizer, device: torch.device) -> tuple[float, float]:
    """Run one training epoch and return average loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total_examples = 0

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(inputs)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        predictions = logits.argmax(dim=1)
        correct += (predictions == targets).sum().item()
        total_examples += targets.size(0)

    return total_loss / total_examples, correct / total_examples


@torch.no_grad()
def evaluate_one_epoch(model, data_loader, loss_fn, device: torch.device) -> tuple[float, float]:
    """Run one evaluation epoch and return average loss and accuracy."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total_examples = 0

    for inputs, targets in data_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        logits = model(inputs)
        loss = loss_fn(logits, targets)

        total_loss += loss.item() * inputs.size(0)
        predictions = logits.argmax(dim=1)
        correct += (predictions == targets).sum().item()
        total_examples += targets.size(0)

    return total_loss / total_examples, correct / total_examples


def fit_model(
    model,
    train_loader,
    validation_loader,
    loss_fn,
    optimizer,
    device: torch.device,
    num_epochs: int = 15,
    patience: int = 3,
) -> tuple:
    """Train a model with early stopping and return the best model plus history."""
    model.to(device)
    early_stopping = EarlyStopping(patience=patience)
    best_state_dict = copy.deepcopy(model.state_dict())
    history = []

    for epoch in range(1, num_epochs + 1):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, loss_fn, optimizer, device)
        validation_loss, validation_accuracy = evaluate_one_epoch(model, validation_loader, loss_fn, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_loss": validation_loss,
                "validation_accuracy": validation_accuracy,
            }
        )

        if early_stopping.best_loss is None or validation_loss < early_stopping.best_loss:
            best_state_dict = copy.deepcopy(model.state_dict())

        if early_stopping.should_stop(validation_loss):
            break

    model.load_state_dict(best_state_dict)
    history_df = pd.DataFrame(history).round(4)
    return model, history_df
