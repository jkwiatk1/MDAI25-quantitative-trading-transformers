import torch
import logging
from matplotlib import pyplot as plt
from pathlib import Path

from models.iTransformer import iTransformerModel


def build_transformer(
    input_dim=1,
    d_model: int = 512,
    nhead: int = 8,
    num_encoder_layers: int = 2,
    dim_feedforward: int = 2048,
    dropout: float = 0.1,
    num_features=1,
    columns_amount=1,
) -> iTransformerModel:
    """
    Args:
        d_model:
        num_encoder_layers: num of encoder block
        nhead: num of heads
        dropout: droput probability
        dim_feedforward: hidden layer [FF] size
        seq_len:
    Returns:

    """
    return iTransformerModel(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        num_features=num_features,
        columns_amount=columns_amount,
    )


def train_model(
        model, train_loader, val_loader, criterion, optimizer, device, n_epochs, save_path, patience, scaler
):
    """
    Train the model, track training/validation loss, and save the best model after training.
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
    best_model_path = save_path / "best_model.pth"

    train_losses, val_losses = [], []
    best_val_loss = float("inf")
    epochs_no_improve = 0

    for epoch in range(1, n_epochs + 1):
        # Training step
        train_loss = run_epoch(model, train_loader, criterion, optimizer, device, scaler, train=True)

        # Validation step
        val_loss = run_epoch(model, val_loader, criterion, optimizer, device, scaler, train=False)

        # Logging losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        logging.info(f"Epoch {epoch}/{n_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"New best model saved at epoch {epoch}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logging.info(f"Early stopping after {epoch} epochs with no improvement.")
            break

    # Final logging
    logging.info(f"Training complete. Best Validation Loss: {best_val_loss:.4f}")

    # Plot and save the training curve
    plot_losses(train_losses, val_losses, save_path)

    return best_model_path


def run_epoch(model, data_loader, criterion, optimizer, device,scaler, train=True):
    """
    Run a single epoch (training or validation).
    """
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    for batch_sequences, batch_targets in data_loader:
        batch_sequences, batch_targets = batch_sequences.to(device), batch_targets.to(device)

        with torch.cuda.amp.autocast():  # Forward pass with mixed precision
            outputs = model(batch_sequences)
            loss = criterion(outputs, batch_targets)

        if train:
            optimizer.zero_grad()
            scaler.scale(loss).backward()  # Backward pass with gradient scaling
            scaler.step(optimizer)  # Update optimizer with scaled gradients
            scaler.update()

        running_loss += loss.item() * batch_sequences.size(0)

    return running_loss / len(data_loader.dataset)


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


def plot_losses(train_losses, val_losses, save_path):
    """
    Plot and save training and validation loss curves.
    """
    if len(train_losses) > 2:
        max_loss = max(max(train_losses[2:]), max(val_losses[2:]))
    else:
        max_loss = max(max(train_losses), max(val_losses))

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.ylim(0, int(1 * max_loss + 0.51))
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path / "training_validation_loss")
    plt.show()


def inverse_transform_predictions(
    predictions, targets, tickers, feat_scalers, preproc_target_col
):
    """
    Apply inverse transform for each ticker's predictions.
    Use the scalers dictionary to inverse transform the target feature for each ticker.
    """
    for i, ticker in enumerate(tickers):
        if len(tickers) == 1:
            predictions = (
                feat_scalers[ticker][preproc_target_col]
                .inverse_transform(predictions.reshape(-1, 1))
                .flatten()
            )
            targets = (
                feat_scalers[ticker][preproc_target_col]
                .inverse_transform(targets.reshape(-1, 1))
                .flatten()
            )
        else:
            predictions[:, i] = (
                feat_scalers[ticker][preproc_target_col]
                .inverse_transform(predictions[:, i].reshape(-1, 1))
                .flatten()
            )
            targets[:, i] = (
                feat_scalers[ticker][preproc_target_col]
                .inverse_transform(targets[:, i].reshape(-1, 1))
                .flatten()
            )
    return predictions, targets


def plot_predictions(test_predictions, test_targets, tickers, save_path=None):
    """
    Plots for tickers
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))
    plt.plot(test_targets, label="True Values", linestyle="dashed")
    plt.plot(test_predictions, label="Predictions")
    plt.legend()
    plt.title("iTransformer Multi-Ticker Predictions on Test Set")
    plt.xlabel("Time Steps")
    plt.ylabel("Values")
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
        plt.title(f"{ticker} Predictions on Test Set")
        plt.xlabel("Time Steps")
        plt.ylabel("Values")
        # plt.show()

        if save_path is not None:
            plt.savefig(save_path / f"{ticker}_predictions")
            plt.close()
    if save_path is not None:
        print("Plots saved for each ticker!")