"""
Contains functions for training and validing a PyTorch model.
"""
import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device) -> Tuple[float, float]:
    """Trains a PyTorch model for a single epoch.

    Turns a target PyTorch model to training mode and then
    runs through all of the required training steps (forward
    pass, loss calculation, optimizer step).

    Args:
    model: A PyTorch model to be trained.
    dataloader: A DataLoader instance for the model to be trained on.
    loss_fn: A PyTorch loss function to minimize.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of training loss and training accuracy metrics.
    In the form (train_loss, train_accuracy). For example:

    (0.1112, 0.8743)
    """
    # Put model in train mode
    model.train()

    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0

    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc


def valid_step(model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               device: torch.device) -> Tuple[float, float]:
    """valids a PyTorch model for a single epoch.

    Turns a target PyTorch model to "eval" mode and then performs
    a forward pass on a validing dataset.

    Args:
    model: A PyTorch model to be valided.
    dataloader: A DataLoader instance for the model to be valided on.
    loss_fn: A PyTorch loss function to calculate loss on the valid data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of validing loss and validing accuracy metrics.
    In the form (valid_loss, valid_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval()

    # Setup valid loss and valid accuracy values
    valid_loss, valid_acc = 0, 0

    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            valid_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(valid_pred_logits, y)
            valid_loss += loss.item()

            # Calculate and accumulate accuracy
            valid_pred_labels = valid_pred_logits.argmax(dim=1)
            valid_acc += ((valid_pred_labels == y).sum().item() /
                          len(valid_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch
    valid_loss = valid_loss / len(dataloader)
    valid_acc = valid_acc / len(dataloader)
    return valid_loss, valid_acc


class EarlyStopper:
    """
      An implementation of Earlystopper for Pytorch
    """

    def __init__(self, patience: int = 1, min_delta: int = 0):
        """
        Constructs all the necessary attributes for the EarlyStopper.

        Parameters
        ----------
            paitence : int
                how many epoch should it wait
            min_delta : int
                the difference it should consider

        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_valid_loss = float('inf')

    def early_stop(self, valid_loss):
        """
          Gives signal to stop early if needed

          Parameters
          ----------
          valid_loss : int, required
              valid loss of the step

          Returns
          -------
          None
        """
        if valid_loss < self.min_valid_loss:
            self.min_valid_loss = valid_loss
            self.counter = 0
        elif valid_loss > (self.min_valid_loss + self.min_delta):
            self.counter += 1
            print(
                f"Taking count of no valid loss improvement count: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                print(f" ----------- Stopping Early ----------------")
                return True
        return False


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          valid_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          scheduler: torch.optim.lr_scheduler = None,
          early_stopper_paitence: int = 10,
          early_stopper_min_delta: int = 0,
          device: torch.device = DEVICE
          ) -> Dict[str, List]:
    """Trains and Valids a PyTorch model.

    Passes a target PyTorch models through train_step() and valid_step()
    functions for a number of epochs, training and validing the model
    in the same epoch loop.

    Calculates, prints and stores evaluation metrics throughout.

    Args:
    model: A PyTorch model to be trained and valided.
    train_dataloader: A DataLoader instance for the model to be trained on.
    valid_dataloader: A DataLoader instance for the model to be valided on.
    optimizer: A PyTorch optimizer to help minimize the loss function.
    loss_fn: A PyTorch loss function to calculate loss on both datasets.
    epochs: An integer indicating how many epochs to train for.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A dictionary of training and validing loss as well as training and
    validing accuracy metrics. Each metric has a value in a list for 
    each epoch.
    In the form: {train_loss: [...],
              train_acc: [...],
              valid_loss: [...],
              valid_acc: [...]} 
    For example if training for epochs=2: 
             {train_loss: [2.0616, 1.0537],
              train_acc: [0.3945, 0.3945],
              valid_loss: [1.2641, 1.5706],
              valid_acc: [0.3400, 0.2973]} 
    """
    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "valid_loss": [],
               "valid_acc": []
               }

    # Early Stopper for the model
    early_stopper = EarlyStopper(
        patience=early_stopper_paitence, min_delta=early_stopper_min_delta)

    # Make sure model on target device
    model.to(device)

    # Loop through training and validing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)

        if scheduler is not None:
            # Taking the step in scheduler for learning rate decrease
            scheduler.step()

        valid_loss, valid_acc = valid_step(model=model,
                                           dataloader=valid_dataloader,
                                           loss_fn=loss_fn,
                                           device=device)

        if early_stopper.early_stop(valid_loss):
            break

        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"valid_loss: {valid_loss:.4f} | "
            f"valid_acc: {valid_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["valid_loss"].append(valid_loss)
        results["valid_acc"].append(valid_acc)

    # Return the filled results at the end of the epochs
    return results


def test(model: torch.nn.Module,
         test_dataloader: torch.utils.data.DataLoader,
         loss_fn: torch.nn.Module,
         device: torch.device = DEVICE
         ) -> Dict[str, List]:
    """
    Evaluate a PyTorch model on a test set.

    Args:
        model (torch.nn.Module): The PyTorch model to be evaluated.
        test_dataloader (torch.utils.data.DataLoader): DataLoader for the test set.
        loss_fn (torch.nn.Module): Loss function used for evaluation.
        device (torch.device): Device on which to perform evaluation (default is DEVICE).

    Returns:
        Dict[str, List]: A dictionary containing the accuracy, loss, and predictions on the test set.
    """
    y_preds = []  # List to store predictions
    test_loss = 0   # Variable to accumulate loss
    test_acc = 0    # Variable to accumulate accuracy

    model.to(device)
    model.eval()

    with torch.inference_mode():
        for X, y in tqdm(test_dataloader, desc="Making predictions"):
            # Send data and targets to target device
            X, y = X.to(device), y.to(device)

            # Do the forward pass
            y_logit = model(X)

            # Turn predictions from logits -> prediction probabilities -> predictions labels
            y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)

            # Put predictions on CPU for evaluation
            y_preds.append(y_pred.cpu())

            # Calculate loss
            loss = loss_fn(y_logit, y)
            test_loss += loss.item()

            # Calculate accuracy
            test_acc += (y_pred == y).sum().item() / len(y_pred)

        # Average loss and accuracy over the test set
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)

    # Concatenate list of predictions into a tensor
    y_pred_tensor = torch.cat(y_preds)

    return {"test_accuracy": test_acc, "test_loss": test_loss, "prediction_tensors": y_pred_tensor}
