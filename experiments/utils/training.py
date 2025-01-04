import itertools

import torch
import logging

import yaml
from matplotlib import pyplot as plt
from pathlib import Path

from torch import nn
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, ExponentialLR
from torch.utils.data import DataLoader

from experiments.utils.datasets import MultiTickerDataset
from models.WeightedMAELoss import WeightedMAELoss
from models.iTransformer import iTransformerModel
from models.Transformer import TransformerModel


def build_iTransformer(
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

def build_Transformer(
    input_dim=1,
    d_model: int = 512,
    nhead: int = 8,
    num_encoder_layers: int = 2,
    dim_feedforward: int = 2048,
    dropout: float = 0.1,
    num_features=1,
    columns_amount=1,
    max_seq_len=1000
) -> TransformerModel:
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
    return TransformerModel(
        input_dim=input_dim,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        num_features=num_features,
        columns_amount=columns_amount,
        max_seq_len=max_seq_len,
    )



def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    device,
    n_epochs,
    save_path,
    patience,
    scaler,
    scheduler=None,
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
        train_loss = run_epoch(
            model, train_loader, criterion, optimizer, device, scaler, train=True
        )

        # Validation step
        val_loss = run_epoch(
            model, val_loader, criterion, optimizer, device, scaler, train=False
        )

        # Logging losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        logging.info(
            f"Epoch {epoch}/{n_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
        )

        # Update scheduler
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)  # ReduceLROnPlateau needs metric
            else:
                scheduler.step()  # other schedulers: StepLR, ExponentialLR

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"New best model saved at epoch {epoch}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            logging.info(f"Early stopping after {epochs_no_improve} epochs with no improvement.")
            break

    # Final logging
    logging.info(f"Training complete. Best Validation Loss: {best_val_loss:.4f}")

    # Plot and save the training curve
    plot_losses(train_losses, val_losses, save_path)

    return best_model_path, best_val_loss


def grid_search_train(
        config,
        train_sequences,
        train_targets,
        val_sequences,
        val_targets,
        device,
        save_dir,
        input_dim,
        num_features,
        columns_amount,
        max_seq_len,
):
    """
    Perform grid search for the best hyperparameters.
    Args:
        config: Experiment configuration (YAML format).
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        device: Device to use ('cpu' or 'cuda').
        save_dir: Directory to save the best model and hyperparameter configuration.
    Returns:
        None
    """
    global model
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Unpack hyperparameters
    param_grid = {
        "d_model": config["model"]["d_model"],
        "nhead": config["model"]["nhead"],
        "num_encoder_layers": config["model"]["num_encoder_layers"],
        "dim_feedforward": config["model"]["dim_feedforward"],
        "dropout": config["model"]["dropout"],
        "batch_size": config["training"]["batch_size"],
        "learning_rate": config["training"]["learning_rate"],
    }

    best_val_loss = float("inf")
    best_params = None
    best_global_model_path = None

    # criterion = nn.MSELoss()
    criterion = WeightedMAELoss(num_outputs=num_features)
    logging.info(f"loss function: {criterion.__class__}")
    scaler = torch.cuda.amp.GradScaler()

    # # Generate all combinations of parameters
    # for params in itertools.product(*param_grid.values()):

    # Generate all combinations of parameters
    all_combinations = list(itertools.product(*param_grid.values()))
    # Use tqdm for progress tracking
    for params in tqdm(all_combinations, desc="Grid Search Progress", unit="experiment"):

        # Map parameters to dict
        param_dict = dict(zip(param_grid.keys(), params))
        logging.info(f"Testing parameters: {param_dict}")

        # Create DataLoader with current batch_size
        train_loader = DataLoader(
            MultiTickerDataset(train_sequences, train_targets),
            batch_size=param_dict["batch_size"],
            shuffle=True,
        )
        val_loader = DataLoader(
            MultiTickerDataset(val_sequences, val_targets),
            batch_size=param_dict["batch_size"],
            shuffle=False,
        )

        # Build model with current parameters
        if config['model']['name'] == "iTransformer":
            model = build_iTransformer(
                input_dim=input_dim,
                d_model=param_dict["d_model"],
                nhead=param_dict["nhead"],
                num_encoder_layers=param_dict["num_encoder_layers"],
                dim_feedforward=param_dict["dim_feedforward"],
                dropout=param_dict["dropout"],
                num_features=num_features,
                columns_amount=columns_amount,
            ).to(device)
        elif config['model']['name'] == "Transformer":
            model = build_Transformer(
                input_dim=input_dim,
                d_model=param_dict["d_model"],
                nhead=param_dict["nhead"],
                num_encoder_layers=param_dict["num_encoder_layers"],
                dim_feedforward=param_dict["dim_feedforward"],
                dropout=param_dict["dropout"],
                num_features=num_features,
                columns_amount=columns_amount,
                max_seq_len=max_seq_len,
            ).to(device)

        # Optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=param_dict["learning_rate"])

        if config["training"]["lr_scheduler"]["type"] == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=config["training"]["lr_scheduler"]["mode"],
                factor=config["training"]["lr_scheduler"]["factor"],
                patience=config["training"]["lr_scheduler"]["patience"],
            )
            logging.info(f"lr_scheduler: {config['training']['lr_scheduler']['type']}")
        elif config["training"]["lr_scheduler"]["type"] == "StepLR":
            scheduler = StepLR(
                optimizer,
                step_size=config["training"]["lr_scheduler"]["step_size"],
                gamma=config["training"]["lr_scheduler"]["gamma"],
            )
            logging.info(f"lr_scheduler: {config['training']['lr_scheduler']['type']}")
        elif config["training"]["lr_scheduler"]["type"] == "ExponentialLR":
            scheduler = ExponentialLR(
                optimizer,
                gamma=config["training"]["lr_scheduler"]["gamma"],
            )
            logging.info(f"lr_scheduler: {config['training']['lr_scheduler']['type']}")
        else:
            scheduler = None
            logging.info("No scheduler!")
            logging.info(f"lr_scheduler: {config['training']['lr_scheduler']['type']}")

        tmp_model_path = save_dir / "tmp_model"

        # Train and validate model
        best_local_model_path, val_loss = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            n_epochs=config["training"]["n_epochs"],
            save_path=tmp_model_path,
            patience=config["training"]["patience"],
            scaler=scaler,
            scheduler=scheduler,
        )

        # Update best model if needed
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = param_dict
            best_global_model_path = save_dir / "best_grid_search_model.pth"
            torch.save(model.state_dict(), best_global_model_path)

    # Save the best parameters and results
    best_params_path = save_dir / "best_params.yaml"
    with open(best_params_path, "w") as f:
        yaml.dump(
            {
                "best_val_loss": best_val_loss,
                "best_params": best_params,
            },
            f,
        )
    logging.info(f"Best model saved at: {best_global_model_path}")
    logging.info(f"Best parameters saved at: {best_params_path}")

    # Load best model and evaluate
    model.load_state_dict(torch.load(best_global_model_path))

    return model, criterion, scaler, best_params["batch_size"]


def run_epoch(model, data_loader, criterion, optimizer, device, scaler, train=True):
    """
    Run a single epoch (training or validation).
    """
    if train:
        model.train()
    else:
        model.eval()

    running_loss = 0.0
    for batch_sequences, batch_targets in data_loader:
        batch_sequences, batch_targets = batch_sequences.to(device), batch_targets.to(
            device
        )

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

    if max_loss <= 0:
        max_loss = 0.1

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    # plt.ylim(0, int(1 * max_loss + 0.51))
    plt.ylim(0, max_loss * 1.1)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path / "training_validation_loss")
    # uncomment below to print training & val loss
    # plt.show()


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


def plot_predictions(test_predictions, test_targets, tickers, save_path=None, dates=None):
    """
    Plots for tickers with date labels on the X-axis.

    Args:
        test_predictions: Array of predictions.
        test_targets: Array of true values.
        tickers: List of ticker names.
        save_path: Directory to save plots (optional).
        dates: DatetimeIndex or list of date strings for X-axis labels.
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
        plt.title(f"{ticker} Predictions on Test Set")
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
        print("Plots saved for each ticker!")


def grid_search_train_Transformer_only(
        config,
        train_sequences,
        train_targets,
        val_sequences,
        val_targets,
        device,
        save_dir,
        input_dim,
        num_features,
        columns_amount,
        max_seq_len,
):
    """
    Perform grid search for the best hyperparameters.
    Args:
        config: Experiment configuration (YAML format).
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation data.
        device: Device to use ('cpu' or 'cuda').
        save_dir: Directory to save the best model and hyperparameter configuration.
    Returns:
        None
    """
    global model
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Unpack hyperparameters
    param_grid = {
        "d_model": config["model"]["d_model"],
        "nhead": config["model"]["nhead"],
        "num_encoder_layers": config["model"]["num_encoder_layers"],
        "dim_feedforward": config["model"]["dim_feedforward"],
        "dropout": config["model"]["dropout"],
        "batch_size": config["training"]["batch_size"],
        "learning_rate": config["training"]["learning_rate"],
    }

    best_val_loss = float("inf")
    best_params = None
    best_global_model_path = None

    criterion = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler()

    # Generate all combinations of parameters
    for params in itertools.product(*param_grid.values()):
        # Map parameters to dict
        param_dict = dict(zip(param_grid.keys(), params))
        logging.info(f"Testing parameters: {param_dict}")

        # Create DataLoader with current batch_size
        train_loader = DataLoader(
            MultiTickerDataset(train_sequences, train_targets),
            batch_size=param_dict["batch_size"],
            shuffle=True,
        )
        val_loader = DataLoader(
            MultiTickerDataset(val_sequences, val_targets),
            batch_size=param_dict["batch_size"],
            shuffle=False,
        )

        # Build model with current parameters
        model = build_Transformer(
            input_dim=input_dim,
            d_model=param_dict["d_model"],
            nhead=param_dict["nhead"],
            num_encoder_layers=param_dict["num_encoder_layers"],
            dim_feedforward=param_dict["dim_feedforward"],
            dropout=param_dict["dropout"],
            num_features=num_features,
            columns_amount=columns_amount,
            max_seq_len=max_seq_len,
        ).to(device)

        # Optimizer and scheduler
        optimizer = torch.optim.Adam(model.parameters(), lr=param_dict["learning_rate"])

        if config["training"]["lr_scheduler"]["type"] == "ReduceLROnPlateau":
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode=config["training"]["lr_scheduler"]["mode"],
                factor=config["training"]["lr_scheduler"]["factor"],
                patience=config["training"]["lr_scheduler"]["patience"],
            )
            logging.info(f"lr_scheduler: {config['training']['lr_scheduler']['type']}")
        elif config["training"]["lr_scheduler"]["type"] == "StepLR":
            scheduler = StepLR(
                optimizer,
                step_size=config["training"]["lr_scheduler"]["step_size"],
                gamma=config["training"]["lr_scheduler"]["gamma"],
            )
            logging.info(f"lr_scheduler: {config['training']['lr_scheduler']['type']}")
        elif config["training"]["lr_scheduler"]["type"] == "ExponentialLR":
            scheduler = ExponentialLR(
                optimizer,
                gamma=config["training"]["lr_scheduler"]["gamma"],
            )
            logging.info(f"lr_scheduler: {config['training']['lr_scheduler']['type']}")
        else:
            scheduler = None
            logging.info("No scheduler!")
            logging.info(f"lr_scheduler: {config['training']['lr_scheduler']['type']}")

        tmp_model_path = save_dir / "tmp_model"

        # Train and validate model
        best_local_model_path, val_loss = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            n_epochs=config["training"]["n_epochs"],
            save_path=tmp_model_path,
            patience=config["training"]["patience"],
            scaler=scaler,
            scheduler=scheduler,
        )

        # Update best model if needed
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_params = param_dict
            best_global_model_path = save_dir / "best_grid_search_model.pth"
            torch.save(model.state_dict(), best_global_model_path)

    # Save the best parameters and results
    best_params_path = save_dir / "best_params.yaml"
    with open(best_params_path, "w") as f:
        yaml.dump(
            {
                "best_val_loss": best_val_loss,
                "best_params": best_params,
            },
            f,
        )
    logging.info(f"Best model saved at: {best_global_model_path}")
    logging.info(f"Best parameters saved at: {best_params_path}")

    # Load best model and evaluate
    model.load_state_dict(torch.load(best_global_model_path))

    return model, criterion, scaler, best_params["batch_size"]
