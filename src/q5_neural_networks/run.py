"""Entry point for Question 5."""

import torch

from src.common.config import RESULTS_DIR
from src.common.metrics import print_section
from src.q5_neural_networks.data import (
    CLASS_NAMES,
    build_dataloaders,
    build_train_validation_split,
    inspect_fashion_mnist,
    load_fashion_mnist_datasets,
)
from src.q5_neural_networks.evaluation import (
    classification_metrics,
    collect_misclassified_examples,
    confusion_matrix_frame,
    predict_labels,
)
from src.q5_neural_networks.models import CNNClassifier, MLPClassifier
from src.q5_neural_networks.plots import (
    save_confusion_matrix_heatmap,
    save_fashion_mnist_sample_grid,
    save_misclassified_examples_plot,
    save_training_history_plot,
)
from src.q5_neural_networks.training import fit_model, get_device


MLP_EPOCHS = 15
MLP_PATIENCE = 3
MLP_LEARNING_RATE = 1e-3
CNN_EPOCHS = 15
CNN_PATIENCE = 3
CNN_LEARNING_RATE = 1e-3


def main() -> None:
    print_section("Question 5")

    train_dataset, test_dataset = load_fashion_mnist_datasets()
    inspect_fashion_mnist(train_dataset, test_dataset)

    train_subset, validation_subset = build_train_validation_split(train_dataset)
    print(f"\nTrain subset size: {len(train_subset)}")
    print(f"Validation subset size: {len(validation_subset)}")
    print(f"Test set size: {len(test_dataset)}")

    train_loader, validation_loader, test_loader = build_dataloaders(
        train_subset=train_subset,
        validation_subset=validation_subset,
        test_dataset=test_dataset,
    )
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Validation batches: {len(validation_loader)}")
    print(f"Test batches: {len(test_loader)}")

    save_fashion_mnist_sample_grid(train_dataset)
    print("\nSaved Fashion-MNIST sample grid to:")
    print("results/figures/q5/fashion_mnist_sample_grid.png")

    device = get_device()
    print(f"\nTraining device: {device}")

    mlp_model = MLPClassifier()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=MLP_LEARNING_RATE)

    mlp_model, mlp_history_df = fit_model(
        model=mlp_model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        device=device,
        num_epochs=MLP_EPOCHS,
        patience=MLP_PATIENCE,
    )

    table_dir = RESULTS_DIR / "tables" / "q5"
    table_dir.mkdir(parents=True, exist_ok=True)
    mlp_history_df.to_csv(table_dir / "mlp_training_history.csv", index=False)
    save_training_history_plot(mlp_history_df, model_name="MLP")

    print("\nSaved MLP training history to:")
    print("results/tables/q5/mlp_training_history.csv")
    print("results/figures/q5/mlp_training_curves.png")
    print("\nLast recorded MLP epoch:")
    print(mlp_history_df.tail(1).to_string(index=False))

    cnn_model = CNNClassifier()
    cnn_optimizer = torch.optim.Adam(cnn_model.parameters(), lr=CNN_LEARNING_RATE)

    cnn_model, cnn_history_df = fit_model(
        model=cnn_model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        loss_fn=loss_fn,
        optimizer=cnn_optimizer,
        device=device,
        num_epochs=CNN_EPOCHS,
        patience=CNN_PATIENCE,
    )

    cnn_history_df.to_csv(table_dir / "cnn_training_history.csv", index=False)
    save_training_history_plot(cnn_history_df, model_name="CNN")

    print("\nSaved CNN training history to:")
    print("results/tables/q5/cnn_training_history.csv")
    print("results/figures/q5/cnn_training_curves.png")
    print("\nLast recorded CNN epoch:")
    print(cnn_history_df.tail(1).to_string(index=False))

    mlp_y_true, mlp_y_pred = predict_labels(mlp_model, test_loader, device)
    mlp_metrics_df = classification_metrics(mlp_y_true, mlp_y_pred)
    mlp_confusion_df = confusion_matrix_frame(mlp_y_true, mlp_y_pred, CLASS_NAMES)
    mlp_metrics_df.to_csv(table_dir / "mlp_test_metrics.csv", index=False)
    mlp_confusion_df.to_csv(table_dir / "mlp_confusion_matrix.csv")
    save_confusion_matrix_heatmap(mlp_confusion_df, model_name="MLP")

    print("\nSaved MLP test evaluation to:")
    print("results/tables/q5/mlp_test_metrics.csv")
    print("results/tables/q5/mlp_confusion_matrix.csv")
    print("results/figures/q5/mlp_confusion_matrix.png")
    print("\nMLP test metrics:")
    print(mlp_metrics_df.to_string(index=False))

    mlp_misclassified_examples = collect_misclassified_examples(mlp_model, test_loader, device, max_examples=5)
    save_misclassified_examples_plot(mlp_misclassified_examples, model_name="MLP")
    print("\nSaved MLP misclassified examples to:")
    print("results/figures/q5/mlp_misclassified_examples.png")

    cnn_y_true, cnn_y_pred = predict_labels(cnn_model, test_loader, device)
    cnn_metrics_df = classification_metrics(cnn_y_true, cnn_y_pred)
    cnn_confusion_df = confusion_matrix_frame(cnn_y_true, cnn_y_pred, CLASS_NAMES)
    cnn_metrics_df.to_csv(table_dir / "cnn_test_metrics.csv", index=False)
    cnn_confusion_df.to_csv(table_dir / "cnn_confusion_matrix.csv")
    save_confusion_matrix_heatmap(cnn_confusion_df, model_name="CNN")

    print("\nSaved CNN test evaluation to:")
    print("results/tables/q5/cnn_test_metrics.csv")
    print("results/tables/q5/cnn_confusion_matrix.csv")
    print("results/figures/q5/cnn_confusion_matrix.png")
    print("\nCNN test metrics:")
    print(cnn_metrics_df.to_string(index=False))

    cnn_misclassified_examples = collect_misclassified_examples(cnn_model, test_loader, device, max_examples=5)
    save_misclassified_examples_plot(cnn_misclassified_examples, model_name="CNN")
    print("\nSaved CNN misclassified examples to:")
    print("results/figures/q5/cnn_misclassified_examples.png")


if __name__ == "__main__":
    main()
