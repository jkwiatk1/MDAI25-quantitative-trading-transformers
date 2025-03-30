import argparse
import logging
import os
import yaml
import torch
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path

from experiments.utils.data_loading import (
    fill_missing_days,
    load_finance_data_xlsx,
    prepare_finance_data,
)
from experiments.utils.datasets import (
    prepare_combined_data,
    create_combined_sequences,
    normalize_data,
    MultiTickerDataset,
)
from experiments.utils.feature_engineering import calc_input_features
from experiments.utils.training import (
    evaluate_model,
    inverse_transform_predictions,
    plot_predictions, grid_search_train,
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
    logging.info(f"Output path: {output_dir}")

    # Load data
    IS_DATA_FROM_YAHOO = config["data"]["yahoo_data"]
    load_file = config["data"]["path"]
    logging.info(f"Load file path: {load_file}")

    start_date = pd.to_datetime(config["data"]["start_date"])
    end_date = pd.to_datetime(config["data"]["end_date"])
    logging.info(f"Start date: {start_date}")
    logging.info(f"End date: {end_date}")

    logging.info("***Model params***")
    logging.info(f"d_model: {config['model']['d_model']}")
    logging.info(f"nhead: {config['model']['nhead']}")
    logging.info(f"num_encoder_layers: {config['model']['num_encoder_layers']}")
    logging.info(f"dim_feedforward: {config['model']['dim_feedforward']}")
    logging.info(f"dropout: {config['model']['dropout']}")

    logging.info("***Training params***")
    logging.info(f"n_epochs: {config['training']['n_epochs']}")
    logging.info(f"lookback: {config['training']['lookback']}")
    logging.info(f"batch_size: {config['training']['batch_size']}")
    logging.info(f"learning_rate: {config['training']['learning_rate']}")
    logging.info(f"test_split: {config['training']['test_split']}")
    logging.info(f"val_split: {config['training']['val_split']}")
    logging.info(f"patience: {config['training']['patience']}")
    tickers_to_use = []

    if IS_DATA_FROM_YAHOO:
        tickers_df = pd.read_csv(config["data"]["tickers"])
        tickers_to_use = tickers_df["Ticker"].tolist()
        logging.info("Tickers amount: " + str(len(tickers_to_use)))
    else:
        tickers_to_use = config["data"]["tickers"]
        logging.info("Tickers amount: " + str(len(tickers_to_use)))

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
    data_scaled, feat_scalers = normalize_data(
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

    model, criterion, scaler, best_batch_size = grid_search_train(
        config=config,
        train_sequences=train_sequences,
        train_targets=train_targets,
        val_sequences=val_sequences,
        val_targets=val_targets,
        device=device,
        save_dir=output_dir,
        input_dim=config["training"]["lookback"],
        num_features=len(tickers_to_use),
        columns_amount=train_sequences.shape[2],
        max_seq_len=1000,
    )

    test_loader = DataLoader(
        MultiTickerDataset(test_sequences, test_targets),
        batch_size=best_batch_size,
        shuffle=False,
    )

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

    test_dates = combined_data[train_size:].index
    plot_predictions(
        test_predictions, test_targets, tickers_to_use, save_path=output_dir, dates=test_dates
    )


# local run
# model_name = "Transformer"
# base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# log_file = os.path.join(base_dir, "data", "exp_result", model_name, "logs", "pipeline.log")
# setup_logging(log_file)
# # args = SimpleNamespace(config="../experiments/configs/test_config.yaml")
# args = SimpleNamespace(config="../experiments/configs/yahoo_training_config_Transformer.yaml")
# main(args)


if __name__ == "__main__":
    model_name = "iTransformer"
    parser = argparse.ArgumentParser(description="Run QuantFormer Training Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()

    setup_logging(f"./data/exp_result/{model_name}/grid_search/logs/pipeline.log")
    main(args)

"""
# Backtest Strategy
cash = 10000
predicted_stocks = trading_strategy(model, test_loader, device, cash, tickers_to_use)
predicted_stocks
"""
