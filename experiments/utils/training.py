import itertools
from math import ceil

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
from models.MASTER import PortfolioMASTER
from experiments.utils.metrics import WeightedMAELoss
from models.iTransformer import iTransformerModel
from models.Transformer import TransformerModel
from models.Transformer_Cross_Attention import PortfolioTransformerCA
from models.CrossFormer import PortfolioCrossformer


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


def build_TransformerCA(
        stock_amount: int,
        financial_features_amount: int,
        lookback: int,
        d_model: int = 128,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        num_encoder_layers: int = 2,
        device: torch.device = torch.device("cpu")
) -> PortfolioTransformerCA:
    """Factory function to build the PortfolioTransformerCA model."""
    print("-" * 30)
    print("Building PortfolioTransformerCA (Modular) with parameters:")
    print(f"  Data: stocks={stock_amount}, features={financial_features_amount}, lookback={lookback}")
    print(f"  Arch: d_model={d_model}, n_heads={n_heads}, d_ff={d_ff}, layers={num_encoder_layers}")
    print(f"  Dropout: {dropout}")
    print(f"  Device: {device}")
    print("-" * 30)

    if d_model % n_heads != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

    model = PortfolioTransformerCA(
        stock_amount=stock_amount,
        financial_features_amount=financial_features_amount,
        lookback=lookback,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        num_encoder_layers=num_encoder_layers,
    )
    return model.to(device)


def build_CrossFormer(
        # --- Data Shape Parameters ---
        stock_amount: int,  # num_tickers_to_use
        financial_features: int,  # num_tickers_to_use
        in_len: int,  # lookback

        # --- Crossformer Specific Architecture Parameters ---
        seg_len: int,
        win_size: int = 2,
        factor: int = 10,
        e_layers: int = 2,  # num_encoder_layers

        # --- General Transformer Architecture Parameters ---
        d_model: int = 128,
        n_heads: int = 4,  # nhead
        d_ff: int = 256,  # dim_feedforward

        # --- Regularization ---
        dropout: float = 0.1,

        # --- Deployment ---
        device: torch.device = torch.device("cpu")

) -> PortfolioCrossformer:
    """
    Builds and initializes the PortfolioCrossformer model.

    Args:
        stock_amount (int): Number of different assets/stocks in the portfolio.
                            Determines the output size and part of the internal data_dim.
        financial_features (int): Number of features used to describe each stock
                                  (e.g., close price, volume, RSI). Part of the internal data_dim.
        in_len (int): The lookback period, i.e., the length of the input time series sequence.

        seg_len (int): The length of each segment the input sequence is divided into
                       by the DSW embedding. `in_len` should ideally be divisible by `seg_len`,
                       though the model handles padding if needed. Controls the granularity
                       of the initial time series representation.
        win_size (int): The number of adjacent segments merged in the SegMerging layer
                        at each scale (except the first). Typically 2. Controls the rate
                        of temporal aggregation across encoder layers.
        factor (int): The dimension factor for the router mechanism in the
                      TwoStageAttentionLayer, controlling the bottleneck size for
                      cross-dimension attention.
        e_layers (int): The number of hierarchical encoder blocks (scales). Each block
                        (except the first) includes segment merging and TSA layers.

        d_model (int): The main dimensionality of the embeddings and hidden states
                       throughout the model. Must be divisible by `n_heads`.
        n_heads (int): The number of parallel attention heads in the multi-head
                       attention mechanisms (both time and dimension attention).
        d_ff (int): The dimensionality of the inner hidden layer in the feed-forward
                    networks within the TSA layers. Often 2x or 4x `d_model`.

        dropout (float): Dropout probability applied in various layers for regularization.

        device (torch.device): The device (e.g., 'cuda', 'cpu') to create the model on.

    Returns:
        PortfolioCrossformer: An initialized instance of the model.
    """
    print("-" * 30)
    print("Building PortfolioCrossformer with parameters:")
    print(f"  Data: stock_amount={stock_amount}, financial_features={financial_features}, in_len={in_len}")
    print(f"  Crossformer Arch: seg_len={seg_len}, win_size={win_size}, factor={factor}, e_layers={e_layers}")
    print(f"  Transformer Arch: d_model={d_model}, n_heads={n_heads}, d_ff={d_ff}")

    # --- Parameter Validation (Optional but Recommended) ---
    if d_model % n_heads != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
    if in_len % seg_len != 0:
        padded_in_len = ceil(1.0 * in_len / seg_len) * seg_len
        print(f"Warning: in_len ({in_len}) is not divisible by seg_len ({seg_len}). "
              f"Input will be effectively padded to length {padded_in_len}.")

    model = PortfolioCrossformer(
        stock_amount=stock_amount,
        financial_features=financial_features,
        in_len=in_len,  # Passed directly, model handles padding internally if needed
        seg_len=seg_len,
        win_size=win_size,
        factor=factor,
        d_model=d_model,
        d_ff=d_ff,
        n_heads=n_heads,
        e_layers=e_layers,
        dropout=dropout,
        device=device  # Pass the device to the model constructor if it uses it internally
        # (e.g., for creating certain tensors directly on the target device, though
        # often just calling .to(device) after creation is sufficient)
    )
    # The .to(device) call is crucial to move all parameters and buffers
    return model.to(device)


def build_MASTER(
        # --- Data Shape Parameters ---
        stock_amount: int,
        financial_features_amount: int,
        lookback: int,

        # --- MASTER Architecture Parameters ---
        d_model: int = 128,
        d_ff: int = 256,
        t_n_heads: int = 4,  # Temporal attention heads
        s_n_heads: int = 4,  # Spatial attention heads
        t_dropout: float = 0.1,  # Temporal attention dropout
        s_dropout: float = 0.1,  # Spatial attention dropout

        # --- Deployment ---
        device: torch.device = torch.device("cpu")

) -> PortfolioMASTER:
    """
    Builds and initializes the PortfolioMASTER model.

    Args:
        stock_amount (int): Number of different assets/stocks in the portfolio.
        financial_features_amount (int): Number of features used to describe each stock.
        lookback (int): The lookback period (input sequence length).

        d_model (int): The main dimensionality of embeddings and hidden states.
                       Must be divisible by both t_n_heads and s_n_heads.
        d_ff (int): The dimensionality of the inner hidden layer in the FFNs.
        t_n_heads (int): Number of attention heads for Temporal Attention (TAttention).
        s_n_heads (int): Number of attention heads for Spatial Attention (SAttention).
        t_dropout (float): Dropout probability for Temporal Attention layers.
        s_dropout (float): Dropout probability for Spatial Attention layers.

        device (torch.device): The device (e.g., 'cuda', 'cpu') to create the model on.

    Returns:
        PortfolioMASTER: An initialized instance of the model.
    """
    print("-" * 30)
    print("Building PortfolioMASTER with parameters:")
    print(f"  Data: stock_amount={stock_amount}, features={financial_features_amount}, lookback={lookback}")
    print(f"  Arch: d_model={d_model}, t_heads={t_n_heads}, s_heads={s_n_heads}")
    print(f"  Dropout: t_dropout={t_dropout}, s_dropout={s_dropout}")
    print(f"  Device: {device}")
    print("-" * 30)

    # --- Parameter Validation ---
    if d_model % t_n_heads != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by t_n_heads ({t_n_heads})")
    if d_model % s_n_heads != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by s_n_heads ({s_n_heads})")

    model = PortfolioMASTER(
        finance_features_amount=financial_features_amount,
        stock_amount=stock_amount,
        lookback=lookback,
        d_model=d_model,
        d_ff=d_ff,
        t_n_heads=t_n_heads,
        s_n_heads=s_n_heads,
        t_dropout=t_dropout,
        s_dropout=s_dropout
    )
    # Move model to the target device
    return model.to(device)


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
                    scaler.scale(loss).backward()  # Backward pass with gradient scaling
                    scaler.step(optimizer)  # Update optimizer with scaled gradients
                    scaler.update()
                else:
                    loss.backward()
                    # Optional: Gradient Clipping
                    # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
# def plot_losses(train_losses, val_losses, save_path, model_name):
#     """
#     Plot and save training and validation loss curves.
#     """
#     if len(train_losses) > 2:
#         max_loss = max(max(train_losses[2:]), max(val_losses[2:]))
#     else:
#         max_loss = max(max(train_losses), max(val_losses))
#
#     if max_loss <= 0:
#         max_loss = 0.1
#
#     plt.figure(figsize=(10, 6))
#     plt.plot(train_losses, label="Training Loss")
#     plt.plot(val_losses, label="Validation Loss")
#     # plt.ylim(0, int(1 * max_loss + 0.51))
#     plt.ylim(0, max_loss * 1.1)
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.title(f"{model_name}: Training and Validation Loss")
#     plt.legend()
#     plt.grid(True)
#     plt.savefig(save_path / "training_validation_loss")
#     # uncomment below to print training & val loss
#     # plt.show()


# def inverse_transform_predictions(
#     predictions, targets, tickers, feat_scalers, preproc_target_col
# ):
#     """
#     Apply inverse transform for each ticker's predictions.
#     Use the scalers dictionary to inverse transform the target feature for each ticker.
#     """
#     for i, ticker in enumerate(tickers):
#         if len(tickers) == 1:
#             predictions = (
#                 feat_scalers[ticker][preproc_target_col]
#                 .inverse_transform(predictions.reshape(-1, 1))
#                 .flatten()
#             )
#             targets = (
#                 feat_scalers[ticker][preproc_target_col]
#                 .inverse_transform(targets.reshape(-1, 1))
#                 .flatten()
#             )
#         else:
#             predictions[:, i] = (
#                 feat_scalers[ticker][preproc_target_col]
#                 .inverse_transform(predictions[:, i].reshape(-1, 1))
#                 .flatten()
#             )
#             targets[:, i] = (
#                 feat_scalers[ticker][preproc_target_col]
#                 .inverse_transform(targets[:, i].reshape(-1, 1))
#                 .flatten()
#             )
#     return predictions, targets


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
