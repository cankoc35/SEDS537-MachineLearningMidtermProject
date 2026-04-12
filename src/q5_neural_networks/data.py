"""Dataset helpers for Question 5."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from src.common.config import DATA_DIR, RANDOM_STATE


FASHION_MNIST_DIR = DATA_DIR / "raw" / "fashion_mnist"
TRAIN_SIZE = 55_000
VALIDATION_SIZE = 5_000
IMAGE_SIZE = 28 * 28
BATCH_SIZE = 128
CLASS_NAMES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]


def load_fashion_mnist_datasets() -> tuple[Dataset, Dataset]:
    """Download and return the Fashion-MNIST train and test datasets."""
    transform = transforms.ToTensor()
    dataset_dir = Path(FASHION_MNIST_DIR)
    dataset_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = datasets.FashionMNIST(
        root=dataset_dir,
        train=True,
        download=True,
        transform=transform,
    )
    test_dataset = datasets.FashionMNIST(
        root=dataset_dir,
        train=False,
        download=True,
        transform=transform,
    )

    return train_dataset, test_dataset


def build_train_validation_split(train_dataset: Dataset) -> tuple[Dataset, Dataset]:
    """Create a fixed train/validation split for fair model comparison."""
    generator = torch.Generator().manual_seed(RANDOM_STATE)
    return random_split(train_dataset, [TRAIN_SIZE, VALIDATION_SIZE], generator=generator)


def build_dataloaders(
    train_subset: Dataset,
    validation_subset: Dataset,
    test_dataset: Dataset,
    batch_size: int = BATCH_SIZE,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders."""
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, validation_loader, test_loader


def inspect_fashion_mnist(train_dataset: Dataset, test_dataset: Dataset) -> None:
    """Print basic dataset information for Question 5."""
    sample_image, sample_label = train_dataset[0]

    print(f"Training set size: {len(train_dataset)}")
    print(f"Test set size: {len(test_dataset)}")
    print(f"Single image tensor shape: {tuple(sample_image.shape)}")
    print(f"Flattened image size for MLP: {IMAGE_SIZE}")
    print(f"Number of classes: {len(CLASS_NAMES)}")
    print(f"Class names: {', '.join(CLASS_NAMES)}")
    print(f"First sample label: {sample_label} ({CLASS_NAMES[sample_label]})")
