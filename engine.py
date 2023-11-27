"""
Contains functions for training and validing a PyTorch model.
"""
import torch
import os
from tqdm import tqdm
from typing import Dict, List, Tuple
from pytorch_trainer.logger import GenericLogger, colorstr, LOGGER, set_logging
from torchinfo import summary
from datetime import datetime
from pathlib import Path
from pytorch_trainer.plot_from_model import plot_the_confusion_matrix, plot_loss_curves
from pytorch_trainer.dataloaders import plot_random_images_from_dataloader

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TQDM_BAR_FORMAT = '{l_bar}{bar:10}{r_bar}'


def train_step(epoch: int,
               epochs: int,
               model: torch.nn.Module,
               dataloader: torch.utils.data.DataLoader,
               val_dataloader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               scheduler: torch.optim.lr_scheduler,
               early_stopper,
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
    train_loss, train_acc = 0.0, 0.0
    val_loss, val_acc = 0, 0
    pbar = tqdm(enumerate(dataloader), total=len(
        dataloader), bar_format=TQDM_BAR_FORMAT)

    # Loop through data loader data batches
    for batch, (X, y) in pbar:
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        # train_loss += loss.item()

        train_loss = (train_loss * batch + loss.item()) / (batch + 1)
        mem = '%.3gG' % (torch.cuda.memory_reserved() /
                         1E9 if torch.cuda.is_available() else 0)  # (GB)

        # Calculate and accumulate accuracy metric across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_accuracy = (y_pred_class == y).sum().item()/len(y_pred)
        train_acc = (train_acc * batch + train_accuracy) / (batch + 1)

        pbar.desc = f"{f'{epoch + 1}/{epochs}':>10}{mem:>10}{train_loss:>12.3g}{train_acc*100:>12.3g}%" + ' ' * 36

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        if batch == len(pbar) - 1:
            if scheduler is not None:
                # Taking the step in scheduler for learning rate decrease
                scheduler.step()

            val_loss, val_acc = valid_step(model=model,
                                             dataloader=val_dataloader,
                                             loss_fn=loss_fn,
                                             device=device)

            should_stop, early_stop_count = early_stopper.early_stop(
                val_loss)

            pbar.desc = f'{pbar.desc[:-36]}{val_loss:>11.3g}{val_acc*100:>11.3g}%{early_stop_count:>10}' + "      "

    return train_loss, train_acc, val_loss, val_acc, should_stop


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
    loss_fn: A PyTorch loss function to calculate loss on the val data.
    device: A target device to compute on (e.g. "cuda" or "cpu").

    Returns:
    A tuple of validing loss and validing accuracy metrics.
    In the form (val_loss, val_accuracy). For example:

    (0.0223, 0.8985)
    """
    # Put model in eval mode
    model.eval()

    # Setup val loss and val accuracy values
    val_loss, val_acc = 0, 0

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
            val_loss += loss.item()

            # Calculate and accumulate accuracy
            valid_pred_labels = valid_pred_logits.argmax(dim=1)
            val_acc += ((valid_pred_labels == y).sum().item() /
                          len(valid_pred_labels))

    # Adjust metrics to get average loss and accuracy per batch
    val_loss = val_loss / len(dataloader)
    val_acc = val_acc / len(dataloader)
    return val_loss, val_acc


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
        self.min_val_loss = float('inf')

    def early_stop(self, val_loss):
        """
          Gives signal to stop early if needed

          Parameters
          ----------
          val_loss : int, required
              val loss of the step

          Returns
          -------
          None
        """
        output = f'{self.counter+1}/{self.patience}'

        if val_loss < self.min_val_loss:
            self.min_val_loss = val_loss
            self.counter = 0
        elif val_loss > (self.min_val_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True, output
        if self.counter == 0:
            output = 'Start'
        return False, output


def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          val_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          optimizer: torch.optim.Optimizer,
          loss_fn: torch.nn.Module,
          epochs: int,
          scheduler: torch.optim.lr_scheduler = None,
          early_stopper_paitence: int = 10,
          early_stopper_min_delta: int = 0,
          device: torch.device = DEVICE
          ) -> Dict[str, List]:

    image_size = tuple(train_dataloader.dataset[0][0].shape)
    batch_size = train_dataloader.batch_size
    optimizer_name = type(optimizer).__name__
    lr = optimizer.param_groups[0]['lr']
    betas = optimizer.param_groups[0]['betas']
    weight_decay = optimizer.param_groups[0]['weight_decay']
    loss_fn_name = type(loss_fn).__name__
    scheduler_name = type(scheduler).__name__
    class_names = train_dataloader.dataset.classes
    model_name = type(model).__name__
    dataset_size = len(train_dataloader.dataset)
    save_dir = Path('runs') / datetime.now().strftime("%Y%m%d-%H%M%S")
    nc = len(class_names)
    best_val_acc = 0.0
    best_val_loss = 999

    Path(save_dir / 'models').mkdir(parents=True, exist_ok=True)
    Path(save_dir / 'logs').mkdir(parents=True, exist_ok=True)

    best, last = save_dir / 'models/best.pt', save_dir / 'models/last.pt'

    set_logging(model_name)
    logger = GenericLogger(opt=save_dir / 'logs', console_logger=LOGGER)
    # print(image_size)
    hyperparameters = {
        'epochs': epochs,
        'image_size': image_size,
        'batch_size': batch_size,
        'optimizer': optimizer_name,
        'lr': lr,
        'betas': betas,
        'weight_decay': weight_decay,
        'loss': loss_fn_name,
        'scheduler': scheduler_name,
        'es_paitence': early_stopper_paitence,
        'es_min_delta': early_stopper_min_delta,
        'device': device,
    }

    logger.log_hyperparameters(hyperparameters)

    LOGGER.info(colorstr('Hyperparameters: ') +
                ', '.join(f'{k}={v}' for k, v in hyperparameters.items()))

    # Get a summary of the model (uncomment for full output)
    LOGGER.info(colorstr('Model Structure: '))
    LOGGER.info(summary(model,
                        input_size=(1, *image_size),
                        verbose=0,
                        col_names=["input_size", "output_size",
                                   "num_params", "trainable"],
                        col_width=20,
                        row_settings=["var_names"]
                        ))

    LOGGER.info(f'Image sizes {image_size} train, validation and test\n'
                f"Logging results to {colorstr('bold', save_dir)}\n"
                f'Starting {model_name} training on dataset of {dataset_size} images with {nc} classes for {epochs} epochs...\n\n'
                f"{'Epoch':>10}{'GPU mem':>10}{'Train Loss':>12}{'Train Acc':>12}{'Val loss':>12}{'Val Acc':>12}{'ES Count':>12}")

    # Create empty results dictionary
    results = {"train_loss": [],
               "train_acc": [],
               "val_loss": [],
               "val_acc": []
               }
    
    figure = plot_random_images_from_dataloader(train_dataloader, class_names)
    logger.log_figure(figure, name="Images From Train dataset", epoch=0)

    # Early Stopper for the model
    early_stopper = EarlyStopper(
        patience=early_stopper_paitence, min_delta=early_stopper_min_delta)

    # Make sure model on target device
    model.to(device)

    # Early Stopper for the model
    early_stopper = EarlyStopper(
        patience=early_stopper_paitence, min_delta=early_stopper_min_delta)

    # Make sure model on target device
    model.to(device)

    # Loop through training and validing steps for a number of epochs
    for epoch in range(epochs):
        train_loss, train_acc, val_loss, val_acc, should_stop = train_step(epoch=epoch,
                                                                             epochs=epochs,
                                                                             model=model,
                                                                             dataloader=train_dataloader,
                                                                             val_dataloader=val_dataloader,
                                                                             loss_fn=loss_fn,
                                                                             optimizer=optimizer,
                                                                             scheduler=scheduler,
                                                                             early_stopper=early_stopper,
                                                                             device=device)

        # Print out what's happening
        metrics = {
            'train/loss': train_loss,
            'val/loss': val_loss,
            'train/acc': train_acc,
            'val/acc': val_acc,
            'lr/0': lr
        }  # learning rate

        logger.log_metrics(metrics, epoch)

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["val_loss"].append(val_loss)
        results["val_acc"].append(val_acc)

        # Save model
        best_val_acc = val_acc if val_acc >= best_val_acc else best_val_acc
        best_val_loss = val_loss if val_loss < best_val_loss else best_val_acc

        final_epoch = epoch + 1 == epochs
        ckpt = {
            'epoch': epoch,
            'best_fitness': best_val_acc,
            'model': model_name,
            'optimizer': optimizer_name,
            'hyp': hyperparameters,
            'date': datetime.now().isoformat(),
        }

        # Save last, best and delete
        if best_val_acc == val_acc and best_val_loss == val_loss:
            torch.save(ckpt, best)

        if final_epoch or should_stop:
            torch.save(ckpt, last)

            LOGGER.info('Plotting Graphs for Model')

            plots = plot_loss_curves(results=results)
            logger.log_figure(plots, name="Model Plots", epoch=epoch)

            model.load_state_dict(torch.load(best), strict=False)
            LOGGER.info(colorstr('Test Model: ')+ f'loading best model with val acc of {best_val_acc*100:.2f}%')
            
            test_hash = test(model=model, test_dataloader=test_dataloader,
                             loss_fn=loss_fn, device=device)
            meta = {'epochs': epochs, 'val_acc': f'{best_val_acc*100:.2f}%',
                    'test_acc': f'{test_hash["test_accuracy"]*100:.2f}%',
                    'test_loss': test_hash['test_loss'],
                    'date': datetime.now().isoformat()}

            LOGGER.info(colorstr('Final Result: ') +
                        ', '.join(f'{k}={v}' for k, v in meta.items()))

            LOGGER.info(colorstr('Accuracy: ') +
                        f'{test_hash["test_accuracy"]*100:.2f}%')
            logger.log_graph(model, imgsz=(image_size[1], image_size[2]))
            confusion_matrix_type = 'binary' if len(class_names) <= 2 else 'multiclass'
            figure = plot_the_confusion_matrix(
                class_names=class_names, 
                y_pred=test_hash['prediction_tensors'], 
                test_data=test_dataloader.dataset, 
                task_type=confusion_matrix_type
                )
            logger.log_figure(figure, name="Confusion Matrix", epoch=epoch)
            LOGGER.info("------------------------------------------------------- Finished ------------------------------------------------------------")

            break
        del ckpt

    # Return the filled results at the end of the epochs
    # return results


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
    X_tests = []

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
            if len(X_tests) < 5:
                X_tests.append(X.cpu())

        # Average loss and accuracy over the test set
        test_loss /= len(test_dataloader)
        test_acc /= len(test_dataloader)

    # Concatenate list of predictions into a tensor
    y_pred_tensor = torch.cat(y_preds)

    X_tests_5_tensor = torch.cat(X_tests)

    return {"test_accuracy": test_acc, "test_loss": test_loss, "prediction_tensors": y_pred_tensor, "prediction_5_images": X_tests_5_tensor}
