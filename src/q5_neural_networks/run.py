"""Entry point for Question 5."""

from src.common.metrics import print_section
from src.q5_neural_networks.data import (
    build_train_validation_split,
    inspect_fashion_mnist,
    load_fashion_mnist_datasets,
)
from src.q5_neural_networks.plots import save_fashion_mnist_sample_grid


def main() -> None:
    print_section("Question 5")

    train_dataset, test_dataset = load_fashion_mnist_datasets()
    inspect_fashion_mnist(train_dataset, test_dataset)

    train_subset, validation_subset = build_train_validation_split(train_dataset)
    print(f"\nTrain subset size: {len(train_subset)}")
    print(f"Validation subset size: {len(validation_subset)}")
    print(f"Test set size: {len(test_dataset)}")

    save_fashion_mnist_sample_grid(train_dataset)
    print("\nSaved Fashion-MNIST sample grid to:")
    print("results/figures/q5/fashion_mnist_sample_grid.png")

    print("\nQ5 starter files are ready for model implementation:")
    print("src/q5_neural_networks/data.py")
    print("src/q5_neural_networks/models.py")
    print("src/q5_neural_networks/training.py")
    print("src/q5_neural_networks/evaluation.py")
    print("src/q5_neural_networks/plots.py")


if __name__ == "__main__":
    main()
