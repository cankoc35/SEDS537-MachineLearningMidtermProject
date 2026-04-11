"""Training helpers for Question 5."""


class EarlyStopping:
    """Minimal early stopping utility for future Q5 training."""

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
