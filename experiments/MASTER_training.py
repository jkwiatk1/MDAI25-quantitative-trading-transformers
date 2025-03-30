import torch
import torch.nn as nn
from math import ceil
import argparse
import logging
import os

import yaml
import pandas as pd
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, ExponentialLR
from types import SimpleNamespace

from experiments.utils.data_loading import (
    fill_missing_days,
    load_finance_data_xlsx,
    prepare_finance_data,
)
from experiments.utils.datasets import (
    prepare_sequential_data,
    normalize_data,
    MultiStockDataset,
)
from experiments.utils.feature_engineering import calc_input_features
from experiments.utils.training import (
    build_MASTER,
    train_model,
    evaluate_model,
    inverse_transform_predictions,
    plot_predictions,
)
from experiments.utils.metrics import RankLoss, WeightedMAELoss


def setup_logging(log_file):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
    )


def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def main(args):
    config = load_config(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    torch.cuda.empty_cache()

    # Setup paths
    output_dir = Path(config["data"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output path: {output_dir}")

    # Load data
    IS_DATA_FROM_YAHOO = config["data"]["yahoo_data"]
    load_file = config["data"]["path"]
    logging.info(f"Load file path: {load_file}")

    start_date = pd.to_datetime(config["data"]["start_date"])
    end_date = pd.to_datetime(config["data"]["end_date"])
    logging.info(f"Start date: {start_date}")
    logging.info(f"End date: {end_date}")

    tickers_to_use = []
    if IS_DATA_FROM_YAHOO:
        tickers_df = pd.read_csv(config["data"]["tickers"])
        tickers_to_use = tickers_df["Ticker"].tolist()
    else:
        # Can be a list directly in the config or a file path
        if isinstance(config["data"]["tickers"], str):  # path
            try:
                tickers_df = pd.read_csv(config["data"]["tickers"])
                if 'Ticker' in tickers_df.columns:
                    tickers_to_use = tickers_df["Ticker"].tolist()
                else:
                    with open(config["data"]["tickers"], 'r') as f:
                        tickers_to_use = [line.strip() for line in f if line.strip()]
            except Exception as e:
                logging.error(f"Could not read tickers from file {config['data']['tickers']}: {e}")
                return
        elif isinstance(config["data"]["tickers"], list):
            tickers_to_use = config["data"]["tickers"]
        else:
            logging.error("Invalid format for config['data']['tickers']")
            return

    logging.info(f"Tickers amount: {len(tickers_to_use)}")
    stock_amount = len(tickers_to_use)
    financial_features = len(config["data"]["preproc_cols_to_use"])

    data_raw, all_tickers = load_finance_data_xlsx(load_file, IS_DATA_FROM_YAHOO)
    data_raw = fill_missing_days(data_raw.copy(), tickers_to_use, start_date, end_date)
    data = prepare_finance_data(
        data_raw,
        tickers_to_use,
        config["data"]["init_cols_to_use"],
    )
    data = calc_input_features(
        df=data,
        tickers=tickers_to_use,
        cols=config["data"]["preproc_cols_to_use"],
        time_step=config["training"]["lookback"],
    )

    data = {
        key: value[config["data"]["preproc_cols_to_use"]] for key, value in data.items()
    }

    # Normalize and prepare sequences
    data_scaled, feat_scalers = normalize_data(
        data, tickers_to_use, config["data"]["preproc_cols_to_use"]
    )

    sequences, targets, ticker_mapping = prepare_sequential_data(
        data_scaled,
        tickers_to_use,
        config["training"]["lookback"],
        target_col_index=0,  # TODO Make sure that the index of the target column is correct
    )
    # sequences (num_samples, lookback, stock_amount, financial_features)
    # targets (num_samples, stock_amount, 1)

    logging.info("*** Data Params ***")
    logging.info(f"Tickers: {tickers_to_use}")
    logging.info(f"Features used: {config['data']['preproc_cols_to_use']}")
    logging.info(f"Stock amount: {stock_amount}")
    logging.info(f"Financial features per stock: {financial_features}")
    logging.info(f"Total input features (stock * financial): {sequences.shape[-1]}")  # sequences.shape[2]


    logging.info(f"*** {config['model']['name']} Specific Params ***")
    logging.info("*** Model params (Shared) ***")
    logging.info(f"d_model: {config['model']['d_model']}")
    logging.info(f"t_n_heads: {config['model']['t_n_heads']}")
    logging.info(f"s_n_heads: {config['model']['s_n_heads']}")
    logging.info(f"t_dropout: {config['model']['t_dropout']}")
    logging.info(f"s_dropout: {config['model']['s_dropout']}")
    logging.info(f"d_ff: {config['model']['d_ff']}")
    logging.info(f"e_layers: {config['model']['num_encoder_layers']}")

    logging.info("*** Training params ***")
    logging.info(f"n_epochs: {config['training']['n_epochs']}")
    logging.info(f"lookback: {config['training']['lookback']}")
    logging.info(f"batch_size: {config['training']['batch_size']}")
    logging.info(f"learning_rate: {config['training']['learning_rate']}")
    logging.info(f"test_split: {config['training']['test_split']}")
    logging.info(f"val_split: {config['training']['val_split']}")
    logging.info(f"patience: {config['training']['patience']}")

    train_size = int((1 - config["training"]["test_split"]) * len(sequences))
    test_sequences = sequences[train_size:]
    test_targets = targets[train_size:]

    if config["training"]["val_split"] > 0:
        val_size = int(config["training"]["val_split"] * train_size)
        train_sequences = sequences[:train_size - val_size]
        train_targets = targets[:train_size - val_size]
        val_sequences = sequences[train_size - val_size:train_size]
        val_targets = targets[train_size - val_size:train_size]
        logging.info(f"Train size: {len(train_sequences)}")
        logging.info(f"Validation size: {len(val_sequences)}")
    else:
        train_sequences = sequences[:train_size]
        train_targets = targets[:train_size]
        val_sequences, val_targets = None, None
        logging.info(f"Train size: {len(train_sequences)}")
        logging.info("No validation set.")

    logging.info(f"Test size: {len(test_sequences)}")

    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(
        MultiStockDataset(train_sequences, train_targets),
        batch_size=batch_size,
        shuffle=True,
        # num_workers=4,  # For faster loading
        # pin_memory=True  # pin_memory, if using GPU
    )
    val_loader = None
    if val_sequences is not None:
        val_loader = DataLoader(
            MultiStockDataset(val_sequences, val_targets),
            batch_size=batch_size,
            shuffle=False,
            # num_workers=4,
            # pin_memory=True
        )
    test_loader = DataLoader(
        MultiStockDataset(test_sequences, test_targets),
        batch_size=batch_size,
        shuffle=False,
        # num_workers=4,
        # pin_memory=True
    )

    logging.info("Building PortfolioCrossformer model...")
    model = build_MASTER(
        stock_amount=stock_amount,
        financial_features_amount=financial_features,
        lookback=config['training']['lookback'],
        d_model=config['model']['d_model'],
        d_ff=config['model']['d_ff'],
        t_n_heads=config['model']['t_n_heads'],
        s_n_heads=config['model']['s_n_heads'],
        t_dropout=config['model']['t_dropout'],
        s_dropout=config['model']['s_dropout'],
        device=device
    ).to(device)
    logging.info("Model built successfully.")

    # Params counting
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"Total trainable parameters: {total_params:,}")

    # criterion = nn.MSELoss()
    criterion = RankLoss(lambda_rank=0.5)  # TODO set 0.0 for only MSE # Make sure this loss works with weights (0-1, sum 1)
    # Common loss for portfolio weights: Negative Sharpe Ratio, Mean Variance, etc.

    logging.info(f"loss function: {criterion.__class__.__name__}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["training"]["learning_rate"]
    )

    scheduler_config = config['training'].get('lr_scheduler', {})
    scheduler_type = scheduler_config.get('type', None)
    scheduler = None
    if scheduler_type == "ReduceLROnPlateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config.get('mode', 'min'),
            factor=scheduler_config.get('factor', 0.1),
            patience=scheduler_config.get('patience', 10),
        )
    elif scheduler_type == "StepLR":
        scheduler = StepLR(
            optimizer,
            step_size=scheduler_config.get('step_size', 30),
            gamma=scheduler_config.get('gamma', 0.1),
        )
    elif scheduler_type == "ExponentialLR":
        scheduler = ExponentialLR(
            optimizer,
            gamma=scheduler_config.get('gamma', 0.99),
        )

    if scheduler:
        logging.info(f"Using LR scheduler: {scheduler_type}")
    else:
        logging.info("No LR scheduler used.")

    use_amp = config['training'].get('use_amp', True)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    logging.info(f"Using Automatic Mixed Precision (AMP): {use_amp}")
    # scaler = torch.cuda.amp.GradScaler()

    best_model_path, _ = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        n_epochs=config["training"]["n_epochs"],
        save_path=output_dir,
        patience=config["training"]["patience"],
        scaler=scaler,
        scheduler=scheduler,
    )

    if best_model_path and Path(best_model_path).exists():
        logging.info(f"Loading best model from: {best_model_path}")
        model.load_state_dict(torch.load(best_model_path))
        test_predictions, test_targets, test_loss = evaluate_model(
            model=model,
            test_loader=test_loader,
            criterion=criterion,
            device=device,
            scaler=scaler,
        )
        logging.info(f"Test Loss: {test_loss:.4f}")

        test_predictions = test_predictions.squeeze(-1)
        test_targets = test_targets.squeeze(-1)

        # Inverse transform and plot results
        test_predictions, test_targets = inverse_transform_predictions(
            test_predictions,
            test_targets,
            tickers_to_use,
            feat_scalers,
            config["data"]["preproc_target_col"],
        )
        stock_to_take_date = list(data_scaled.keys())[0]
        test_dates = data_scaled[stock_to_take_date][train_size:].index
        plot_predictions(
            test_predictions,
            test_targets,
            tickers_to_use,
            save_path=output_dir,
            dates=test_dates,
        )
    else:
        logging.warning("Best model path not found or training failed. Skipping evaluation.")




# local run
model_name = "MASTER"
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_file = os.path.join(
    base_dir, "data", "exp_result", "test", model_name, "logs", "pipeline.log"
)
setup_logging(log_file)
args = SimpleNamespace(config="../experiments/configs/test_config_MASTER.yaml")
main(args)
#
# # --- Uruchomienie skryptu ---
# if __name__ == "__main__":
#     # Tutaj potrzebujesz parsera argumentów, np. argparse, aby wczytać ścieżkę do configu
#     import argparse
#     parser = argparse.ArgumentParser(description="Train Portfolio Crossformer Model")
#     parser.add_argument('--config', type=str, required=True, help='Path to the configuration file (e.g., config.yaml)')
#     # Dodaj konfigurację logowania
#     logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
#
#     parsed_args = parser.parse_args()
#     main(parsed_args)


# if __name__ == '__main__':
#     # Example input data characteristics
#     batch_size = 64
#     lookback = 20  # Corresponds to in_len
#     stock_amount = 10  # Number of assets
#     features = 2  # Number of financial features per asset
#
#     # --- Example Parameter Values for the Given Data ---
#     # We have in_len=20, stock_amount=10, features=2 => data_dim=20
#     # This is a relatively short sequence and low feature dimension.
#
#     example_seg_len = 4  # Divides in_len=20 nicely (20/4 = 5 segments)
#     # Other options: 2, 5, 10.
#     example_win_size = 2  # Standard choice.
#     example_factor = 5  # Can be smaller (e.g., 5-10) given data_dim=20.
#     example_e_layers = 2  # 2 layers might be sufficient for a short sequence.
#     # Scale 0: 5 segments. Scale 1: ceil(5/2)=3 segments.
#     example_d_model = 64  # Smaller d_model for lower complexity data.
#     example_n_heads = 4  # Must divide d_model (64/4 = 16).
#     example_d_ff = 128  # e.g., 2 * d_model.
#     example_dropout = 0.1  # Standard dropout.
#     example_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
#     # Build the model using the factory function
#     model = build_CrossFormer(
#         stock_amount=stock_amount,
#         financial_features=features,
#         in_len=lookback,
#         seg_len=example_seg_len,
#         win_size=example_win_size,
#         factor=example_factor,
#         e_layers=example_e_layers,
#         d_model=example_d_model,
#         n_heads=example_n_heads,
#         d_ff=example_d_ff,
#         dropout=example_dropout,
#         device=example_device
#     )
#
#     # --- Test with dummy data ---
#     # Yes, input shape [batch_size, lookback, stock_amount, features] is correct!
#     dummy_input = torch.randn(batch_size, lookback, stock_amount, features, device=example_device)
#     print(f"\nTesting model with input shape: {dummy_input.shape}")
#
#     # Pass data through the model
#     try:
#         with torch.no_grad():  # No need to compute gradients for this test
#             output_weights = model(dummy_input)
#         print(f"Model output shape: {output_weights.shape}")  # Expected: [batch_size, stock_amount]
#         print(f"Output weights sum (first sample): {output_weights[0].sum():.4f}")  # Should be close to 1.0
#         print("Model forward pass successful!")
#     except Exception as e:
#         print(f"Error during model forward pass: {e}")
#         import traceback
#
#         traceback.print_exc()
