"""Model definitions for Question 5."""

import torch.nn as nn


INPUT_DIM = 28 * 28
NUM_CLASSES = 10


class MLPClassifier(nn.Module):
    """Simple multilayer perceptron for Fashion-MNIST."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(INPUT_DIM, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, NUM_CLASSES),
        )

    def forward(self, x):
        return self.network(x)


class CNNClassifier(nn.Module):
    """Simple convolutional neural network for Fashion-MNIST."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, NUM_CLASSES),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
