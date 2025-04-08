import itertools

import torch
import logging
import numpy as np

import yaml
from matplotlib import pyplot as plt
from pathlib import Path

from torch import nn
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, ExponentialLR
from torch.utils.data import DataLoader

from experiments.utils.datasets import MultiTickerDataset
from experiments.utils.metrics import WeightedMAELoss


def train_model(
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        n_epochs: int,
        save_path: Path,
        patience: int,
        scaler: torch.cuda.amp.GradScaler,
        scheduler: torch.optim.lr_scheduler._LRScheduler = None,
        model_name: str = "model"
):
    """
    Trains the model, tracks losses, early stopping, and saves the best model.
    Args:
        model: PyTorch model to be trained.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        criterion: Loss function.
        optimizer: Optimizer for model training.
        device: Device to run training ('cpu' or 'cuda').
        n_epochs: Number of epochs.
        save_path: Directory where the best model and plots will be saved.
        patience: Number of epochs to wait for improvement before early stopping.
    Returns:
        Path to the best model saved.
    """
    # Initialization
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    best_model_path = save_path / f"{model_name}_best_val.pth"

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    epochs_no_improve = 0

    logging.info(f"Starting training for {n_epochs} epochs. Model: {model_name}")
    logging.info(f"Saving best model to: {best_model_path}")

    for epoch in range(1, n_epochs + 1):
        # --- Training Step ---
        train_loss = run_epoch(
            model, train_loader, criterion, device=device, optimizer=optimizer, scaler=scaler, train=True
        )
        train_losses.append(train_loss)

        # --- Validation Step ---
        val_loss = None
        if val_loader:
            val_loss = run_epoch(model, val_loader, criterion, device, train=False)
            val_losses.append(val_loss)
            log_msg = f"Epoch {epoch}/{n_epochs} | Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f}"
        else:
            log_msg = f"Epoch {epoch}/{n_epochs} | Train Loss: {train_loss:.6f} | (No Validation)"
            if train_loss < best_val_loss:
                best_val_loss = train_loss
                torch.save(model.state_dict(), best_model_path)
                log_msg += " | New best model (based on train loss) saved."

        # --- Early Stopping & Best Model Saving (only if val_loader exists) ---
        if val_loader and val_loss is not None:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), best_model_path)
                logging.info(f"   -> New best validation model saved.")
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                logging.info(f"   -> No improvement in validation loss for {epochs_no_improve} epoch(s).")

            if epochs_no_improve >= patience:
                logging.info(f"Early stopping triggered after {patience} epochs with no improvement.")
                break

        current_lr = optimizer.param_groups[0]['lr']
        log_msg += f" | LR: {current_lr:.1e}"
        logging.info(log_msg)

        # --- Scheduler Step ---
        if scheduler is not None:
            if isinstance(scheduler, ReduceLROnPlateau):
                if val_loss is not None:
                    scheduler.step(val_loss)
                else:
                    logging.warning(
                        "ReduceLROnPlateau requires validation loss, but val_loader is None. Scheduler not stepped.")
            else:
                scheduler.step()

    # --- Final Logging & Plotting ---
    if val_loader:
        logging.info(f"Training complete. Best Validation Loss: {best_val_loss:.6f}")
    else:
        logging.info(
            f"Training complete (no validation). Best Training Loss (used for saving): {best_val_loss:.6f}")

    plot_losses(train_losses, val_losses, save_path, model_name)

    return best_model_path, best_val_loss

def run_epoch(model: nn.Module,
              data_loader: DataLoader,
              criterion: nn.Module,
              device: torch.device,
              optimizer: torch.optim.Optimizer = None,
              scaler: torch.cuda.amp.GradScaler = None,
              train: bool = True):
    """
    Runs a single epoch of training or validation/evaluation.

    Args:
        model: Model in pytorch framework.
        data_loader: DataLoader for the current dataset split.
        criterion: The loss function.
        device: The device to run on ('cpu' or 'cuda').
        optimizer: The optimizer (only required if train=True).
        scaler: GradScaler for AMP (only required if train=True and AMP is enabled).
        train: Boolean indicating if this is a training epoch (requires optimizer/scaler).

    Returns:
        float: The average loss for the epoch.
    """
    if train:
        if optimizer is None:
            raise ValueError("Optimizer must be provided for training.")
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    num_samples = 0
    amp_enabled = scaler is not None and scaler.is_enabled()

    data_iterator = tqdm(data_loader, desc=f"{'Train' if train else 'Eval '}", leave=False)

    for batch_sequences, batch_targets in data_iterator:
        batch_sequences = batch_sequences.to(device)
        batch_targets = batch_targets.to(device)
        batch_size = batch_sequences.size(0)

        with torch.set_grad_enabled(train):
            with torch.cuda.amp.autocast(enabled=amp_enabled):  # Forward pass with mixed precision
                outputs = model(batch_sequences)
                loss = criterion(outputs, batch_targets)

            if train:
                optimizer.zero_grad(set_to_none=True)
                if amp_enabled:
                    # Previous TODO
                    # scaler.scale(loss).backward()  # Backward pass with gradient scaling
                    # scaler.step(optimizer)  # Update optimizer with scaled gradients
                    # scaler.update()

                    # New for iTransformer
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)  # Potrzebne przed clip_grad_norm_
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Lub inna wartość max_norm
                    scaler.step(optimizer)
                    scaler.update()

                else:
                    # Previous TODO
                    # loss.backward()
                    # optimizer.step()

                    # TODO for itransformer
                    loss.backward()
                    # Optional: Gradient Clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()

            running_loss += loss.item() * batch_size
            num_samples += batch_size

    if num_samples == 0:
        logging.warning(f"No samples processed in {'training' if train else 'evaluation'} epoch.")
        return 0.0

    return running_loss / num_samples


def evaluate_model(model, test_loader, criterion, device, scaler):
    model.eval()
    test_loss = 0.0
    predictions_list, targets_list = [], []

    with torch.no_grad():
        for batch_sequences, batch_targets in test_loader:
            batch_sequences, batch_targets = batch_sequences.to(
                device
            ), batch_targets.to(device)

            with torch.cuda.amp.autocast():  # Mixed precision during evaluation
                predictions = model(batch_sequences)
                loss = criterion(predictions, batch_targets)

            test_loss += loss.item() * batch_sequences.size(0)
            predictions_list.append(predictions.cpu())
            targets_list.append(batch_targets.cpu())

    test_loss /= len(test_loader.dataset)
    test_predictions = torch.cat(predictions_list).numpy()
    test_targets = torch.cat(targets_list).numpy()

    return test_predictions, test_targets, test_loss

def plot_losses(train_losses, val_losses, save_path: Path, model_name: str):
    """Plots and saves training and validation loss curves."""
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label="Training Loss", marker='o', linestyle='-')
    if val_losses:
        plt.plot(epochs, val_losses, label="Validation Loss", marker='x', linestyle='--')
        skip_epochs = 2
        if len(train_losses) > skip_epochs and len(val_losses) > skip_epochs:
             max_loss = max(max(train_losses[skip_epochs:]), max(val_losses[skip_epochs:]))
        elif len(train_losses) > 0 and len(val_losses) > 0:
             max_loss = max(max(train_losses), max(val_losses))
        elif len(train_losses) > 0:
             max_loss = max(train_losses)
        else:
            max_loss = 1.0
        min_loss = 0
    else:
        skip_epochs = 2
        if len(train_losses) > skip_epochs:
             max_loss = max(train_losses[skip_epochs:])
        elif len(train_losses) > 0:
            max_loss = max(train_losses)
        else:
             max_loss = 1.0
        min_loss = 0

    if max_loss <= min_loss:
        max_loss = min_loss + 0.1

    plt.ylim(min_loss, max_loss * 1.1)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name}: Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plot_filename = save_path / f"{model_name}_loss_curve.png"
    try:
        plt.savefig(plot_filename)
        logging.info(f"Loss curve saved to {plot_filename}")
    except Exception as e:
        logging.error(f"Failed to save loss curve plot: {e}")
    plt.close()


def inverse_transform_predictions(predictions_scaled, targets_scaled, tickers, all_scalers, target_col_name):
    """
    Applies inverse transformation to scaled predictions and targets for each stock,
    using a nested dictionary of scalers.

    Args:
        predictions_scaled (np.ndarray): Scaled predictions array, shape (num_samples, num_tickers).
        targets_scaled (np.ndarray): Scaled true targets array, shape (num_samples, num_tickers).
        tickers (list): List of ticker symbols corresponding to the columns in predictions/targets.
        all_scalers (dict): Nested dictionary of scalers: {ticker: {feature_name: scaler_object}}.
        target_col_name (str): The name of the target column to inverse transform.

    Returns:
        tuple(np.ndarray, np.ndarray): Inverse transformed predictions and targets,
                                       same shape as input arrays. Returns original arrays if inverse
                                       transform cannot be performed.
    """
    if predictions_scaled.shape[1] != len(tickers) or targets_scaled.shape[1] != len(tickers):
        logging.error(f"Shape mismatch: Predictions columns ({predictions_scaled.shape[1]}) or "
                      f"Targets columns ({targets_scaled.shape[1]}) do not match number of tickers ({len(tickers)}).")
        # Zwróć oryginalne dane, aby uniknąć crashu, ale zaloguj błąd
        return predictions_scaled, targets_scaled

    num_samples, num_tickers = predictions_scaled.shape
    predictions_inv = np.zeros_like(predictions_scaled)
    targets_inv = np.zeros_like(targets_scaled)
    transform_successful = True  # Flaga do śledzenia czy transformacja się udała

    for i, ticker in enumerate(tickers):
        if ticker not in all_scalers:
            logging.warning(f"No scalers found for ticker '{ticker}'. Skipping inverse transform for this ticker.")
            predictions_inv[:, i] = predictions_scaled[:, i]  # Kopiuj oryginalne wartości
            targets_inv[:, i] = targets_scaled[:, i]
            transform_successful = False
            continue  # Przejdź do następnego tickera

        ticker_scalers = all_scalers[ticker]
        if target_col_name not in ticker_scalers:
            logging.warning(f"Scaler for target column '{target_col_name}' not found for ticker '{ticker}'. "
                            f"Skipping inverse transform for this ticker.")
            predictions_inv[:, i] = predictions_scaled[:, i]  # Kopiuj oryginalne wartości
            targets_inv[:, i] = targets_scaled[:, i]
            transform_successful = False
            continue  # Przejdź do następnego tickera

        target_scaler = ticker_scalers[target_col_name]
        try:
            # Scalery sklearn oczekują wejścia 2D (n_samples, n_features=1)
            # Trzeba dodać i usunąć wymiar
            preds_col = predictions_scaled[:, i].reshape(-1, 1)
            targets_col = targets_scaled[:, i].reshape(-1, 1)

            # Odwróć transformację
            preds_inv_col = target_scaler.inverse_transform(preds_col)
            targets_inv_col = target_scaler.inverse_transform(targets_col)

            # Zapisz wyniki z powrotem do macierzy wyjściowych
            predictions_inv[:, i] = preds_inv_col.flatten()
            targets_inv[:, i] = targets_inv_col.flatten()

        except Exception as e:
            logging.error(f"Error during inverse transform for ticker '{ticker}', target '{target_col_name}': {e}",
                          exc_info=True)
            # W razie błędu, użyj oryginalnych przeskalowanych wartości dla tego tickera
            predictions_inv[:, i] = predictions_scaled[:, i]
            targets_inv[:, i] = targets_scaled[:, i]
            transform_successful = False

    if not transform_successful:
        logging.warning("Inverse transformation could not be fully completed for all tickers/targets. "
                        "Some values in the returned arrays might still be scaled.")

    return predictions_inv, targets_inv


def plot_predictions(test_predictions, test_targets, tickers, save_path=None, dates=None, model_name=None):
    """
    Plots for tickers with date labels on the X-axis.

    Args:
        test_predictions: Array of predictions.
        test_targets: Array of true values.
        tickers: List of ticker names.
        save_path: Directory to save plots (optional).
        dates: DatetimeIndex or list of date strings for X-axis labels.
    """
    save_path = Path(save_path) / "plots"
    save_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(test_targets, label="True Values", linestyle="dashed")
    plt.plot(test_predictions, label="Predictions")
    plt.legend()
    plt.title("iTransformer Multi-Ticker Predictions on Test Set")
    plt.xlabel("Time Steps")
    plt.ylabel("Values")

    # Set X-axis labels
    if dates is not None:
        date_labels = dates.strftime('%Y-%m-%d') if hasattr(dates, 'strftime') else dates
        plt.xticks(ticks=range(0, len(dates), len(dates) // 10),  # Show ~10 ticks
                   labels=date_labels[::len(dates) // 10],
                   rotation=45)
    else:
        plt.xticks(ticks=range(len(test_targets)))  # Default indexing from 0

    # plt.show()
    if save_path is not None:
        plt.savefig(save_path / f"All_predictions")
        plt.close()

    # Separe plots for each ticker
    for i, ticker in enumerate(tickers):
        plt.figure(figsize=(10, 5))
        if len(tickers) == 1:
            plt.plot(test_targets, label=f"True Values - {ticker}", linestyle="dashed")
            plt.plot(test_predictions, label=f"Predictions - {ticker}")
        else:
            plt.plot(
                test_targets[:, i], label=f"True Values - {ticker}", linestyle="dashed"
            )
            plt.plot(test_predictions[:, i], label=f"Predictions - {ticker}")

        plt.legend()
        plt.title(f"{model_name}: {ticker} Predictions on Test Set")
        plt.xlabel("Time Steps")
        plt.ylabel("Values")

        # Set X-axis labels
        if dates is not None:
            plt.xticks(ticks=range(0, len(dates), len(dates) // 10),
                       labels=date_labels[::len(dates) // 10],
                       rotation=45)
        else:
            plt.xticks(ticks=range(len(test_targets)))  # Default indexing from 0
        # plt.show()

        if save_path is not None:
            plt.savefig(save_path / f"{ticker}_predictions")
            plt.close()
    if save_path is not None:
        print(f"Plots saved for each ticker! {save_path}")