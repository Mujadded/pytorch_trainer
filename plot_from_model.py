import torch
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix
import matplotlib.pyplot as plt
from typing import Dict, List

def plot_the_confusion_matrix(class_names: List[str], y_pred: torch.Tensor, test_data: torch.utils.data.Dataset, task_type: str) -> None:
    """
    Plots a confusion matrix using the predictions and targets from a classification model.

    Args:
        class_names (List[str]): List of class names.
        y_pred (torch.Tensor): Model predictions.
        test_data (torch.utils.data.Dataset): Test dataset.
        task_type (str): Type of task (e.g., 'multiclass', 'binary').

    Returns:
        None
    """
    # 2. Setup confusion matrix instance and compare predictions to targets
    confmat = ConfusionMatrix(num_classes=len(class_names), task=task_type)
    confmat_tensor = confmat(preds=y_pred, target=torch.tensor(test_data.targets))

    # 3. Plot the confusion matrix
    fig, ax = plot_confusion_matrix(
        conf_mat=confmat_tensor.numpy(),  # matplotlib likes working with NumPy
        class_names=class_names,  # turn the row and column labels into class names
        figsize=(10, 7)
    )
    return fig

def plot_loss_curves(results: Dict[str, List[float]]) -> None:
    """
    Plots training and validation loss curves along with accuracy curves.

    Args:
        results (Dict[str, List[float]]): Dictionary containing lists of training and validation metrics.

    Returns:
        None
    """
    loss = results["train_loss"]
    test_loss = results["val_loss"]
    accuracy = results["train_acc"]
    test_accuracy = results["val_acc"]
    epochs = range(len(results["train_loss"]))

    figure = plt.figure(figsize=(15, 7))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label="train_loss")
    plt.plot(epochs, test_loss, label="val_loss")
    plt.title("Loss")
    plt.xlabel("Epochs")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label="train_accuracy")
    plt.plot(epochs, test_accuracy, label="val_accuracy")
    plt.title("Accuracy")
    plt.xlabel("Epochs")
    plt.legend()

    return figure