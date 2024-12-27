import argparse
import logging
import os

import yaml
import pandas as pd
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from pathlib import Path
from types import SimpleNamespace

from experiments.utils.data_loading import (
    fill_missing_days,
    load_finance_data_xlsx,
    prepare_finance_data,
)
from experiments.utils.datasets import (
    prepare_combined_data,
    create_combined_sequences,
    normalize_data_for_quantformer,
    MultiTickerDataset,
)
from experiments.utils.feature_engineering import calc_input_features
from experiments.utils.quantformer_ import trading_strategy
from experiments.utils.training import (
    build_Transformer,
    train_model,
    evaluate_model,
    inverse_transform_predictions,
    plot_predictions,
)


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

    # Load data
    IS_DATA_FROM_YAHOO = config["data"]["yahoo_data"]
    load_file = config["data"]["path"]

    start_date = pd.to_datetime(config["data"]["start_date"])
    end_date = pd.to_datetime(config["data"]["end_date"])
    tickers_to_use = []

    if IS_DATA_FROM_YAHOO:
        tickers_df = pd.read_csv(config["data"]["tickers"])
        tickers_to_use = tickers_df["Ticker"].tolist()
        print("Tickers amount: " + str(len(tickers_to_use)))
    else:
        tickers_to_use = config["data"]["tickers"]
        print("Tickers amount: " + str(len(tickers_to_use)))

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
        cols=config["data"]["init_cols_to_use"],
        time_step=config["training"]["lookback"],
    )

    # Normalize and prepare sequences
    data_scaled, feat_scalers = normalize_data_for_quantformer(
        data, tickers_to_use, config["data"]["preproc_cols_to_use"]
    )
    combined_data, ticker_mapping = prepare_combined_data(
        data_scaled, tickers_to_use, config["training"]["lookback"]
    )
    sequences, targets = create_combined_sequences(
        combined_data,
        config["training"]["lookback"],
        config["data"]["preproc_cols_to_use"],
        config["data"]["preproc_target_col"],
    )

    # Train/test split
    train_size = int((1 - config["training"]["test_split"]) * len(sequences))
    val_size = int(config["training"]["val_split"] * train_size)

    train_sequences = sequences[:train_size]
    train_targets = targets[:train_size]
    test_sequences = sequences[train_size:]
    test_targets = targets[train_size:]

    val_sequences = train_sequences[-val_size:]
    val_targets = train_targets[-val_size:]
    train_sequences = train_sequences[:-val_size]
    train_targets = train_targets[:-val_size]

    # Create datasets and dataloaders
    batch_size = config["training"]["batch_size"]
    train_loader = DataLoader(
        MultiTickerDataset(train_sequences, train_targets),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        MultiTickerDataset(val_sequences, val_targets),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        MultiTickerDataset(test_sequences, test_targets),
        batch_size=batch_size,
        shuffle=False,
    )

    # Build model
    model = build_Transformer(
        input_dim=train_sequences.shape[2],
        d_model=config["model"]["d_model"],
        nhead=config["model"]["nhead"],
        num_encoder_layers=config["model"]["num_encoder_layers"],
        dim_feedforward=config["model"]["dim_feedforward"],
        dropout=config["model"]["dropout"],
        num_features=len(tickers_to_use),
        columns_amount=train_sequences.shape[1],
        max_seq_len=1000,
    ).to(device)

    # Training
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config["training"]["learning_rate"]
    )

    scaler = torch.cuda.amp.GradScaler()  # Enable mixed precision training

    best_model_path = train_model(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        device,
        config["training"]["n_epochs"],
        output_dir,
        config["training"]["patience"],
        scaler=scaler,  # Pass scaler for AMP
    )

    # Load best model and evaluate
    model.load_state_dict(torch.load(best_model_path))
    test_predictions, test_targets, test_loss = evaluate_model(
        model,
        test_loader,
        criterion,
        device,
        scaler=scaler,
    )
    logging.info(f"Test Loss: {test_loss:.4f}")

    # Inverse transform and plot results
    test_predictions, test_targets = inverse_transform_predictions(
        test_predictions,
        test_targets,
        tickers_to_use,
        feat_scalers,
        config["data"]["preproc_target_col"],
    )
    plot_predictions(
        test_predictions, test_targets, tickers_to_use, save_path=output_dir
    )


# local run
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_file = os.path.join(base_dir, "data", "exp_result", "logs", "pipeline.log")
setup_logging(log_file)
args = SimpleNamespace(config="../experiments/configs/training_config.yaml")
# args = SimpleNamespace(config="../experiments/configs/yahoo_training_config_iTransformer.yaml")
main(args)


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Run QuantFormer Training Pipeline")
#     parser.add_argument(
#         "--config",
#         type=str,
#         help="Path to the YAML configuration file",
#     )
#     args = parser.parse_args()
#
#     setup_logging("./data/exp_result/logs/pipeline.log")
#     main(args)

"""
# Backtest Strategy
cash = 10000
predicted_stocks = trading_strategy(model, test_loader, device, cash, tickers_to_use)
predicted_stocks
"""
