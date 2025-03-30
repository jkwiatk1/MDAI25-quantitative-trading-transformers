import logging
import logging
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import yaml
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from experiments.utils.data_loading import (
    load_finance_data_xlsx,
    prepare_finance_data,
    fill_missing_days,
)
from experiments.utils.datasets import (
    MultiStockDataset,
    prepare_sequential_data,
    normalize_data,
)
from experiments.utils.feature_engineering import calc_input_features
from experiments.utils.metrics import compute_portfolio_metrics, RankLoss
from experiments.utils.training import (
    build_CrossFormer,
    evaluate_model,
    inverse_transform_predictions,
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

    output_dir = Path(config["data"]["output_dir"]) / config["model"]["name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Output path: {output_dir}")
    start_date = pd.to_datetime(config["data"]["start_date"])
    end_date = pd.to_datetime(config["data"]["end_date"])

    criterion = RankLoss(lambda_rank=0.5)
    scaler = torch.cuda.amp.GradScaler()  # Enable mixed precision training
    adjusted_start = pd.to_datetime(start_date) + pd.DateOffset(
        days=config["training"]["lookback"]
    )
    dates = pd.date_range(start=adjusted_start, end=end_date, freq="D")
    # selected_tickers = config["data"]["tickers"]
    selected_tickers = []

    if config["data"]["yahoo_data"]:
        tickers_df = pd.read_csv(config["data"]["tickers"])
        selected_tickers = tickers_df["Ticker"].tolist()
        logging.info("Tickers amount: " + str(len(selected_tickers)))
    else:
        selected_tickers = config["data"]["tickers"]
        logging.info("Tickers amount: " + str(len(selected_tickers)))

    # Load data
    data_raw_dict, all_tickers = load_finance_data_xlsx(
        config["data"]["path"], config["data"].get("yahoo_data", False)
    )
    data_filled_dict = fill_missing_days(
        data_raw_dict, selected_tickers, start_date, end_date
    )
    data = prepare_finance_data(
        data_filled_dict, selected_tickers, config["data"]["init_cols_to_use"]
    )
    data = calc_input_features(
        data,
        selected_tickers,
        config["data"]["preproc_cols_to_use"],
        config["training"]["lookback"],
    )

    data = {
        key: value[config["data"]["preproc_cols_to_use"]] for key, value in data.items()
    }

    # Normalize and prepare sequences
    data_scaled, feat_scalers = normalize_data(
        data, selected_tickers, config["data"]["preproc_cols_to_use"]
    )

    sequences, targets, ticker_mapping = prepare_sequential_data(
        data_scaled,
        selected_tickers,
        config["training"]["lookback"],
        target_col_index=0,
    )

    # Create test set
    # test_size = int(config["training"]["test_split"] * len(sequences))
    # test_sequences = sequences # [-test_size:]
    # test_targets = targets #[-test_size:]
    test_loader = DataLoader(
        MultiStockDataset(sequences, targets),
        batch_size=config["training"]["batch_size"],
        shuffle=False,
    )

    # Load model
    model = build_CrossFormer(
        stock_amount=len(selected_tickers),
        financial_features=len(config["data"]["preproc_cols_to_use"]),
        in_len=config["training"]["lookback"],
        seg_len=config["model"]["seg_len"],
        win_size=config["model"]["win_size"],
        factor=config["model"]["factor"],
        aggregation_type='avg_pool',  # TODO add to config
        d_model=config["model"]["d_model"],
        n_heads=config["model"]["n_heads"],
        d_ff=config["model"]["d_ff"],
        dropout=config["model"]["dropout"],
        e_layers=config["model"]["num_encoder_layers"],
        device=device,
    ).to(device)
    model.load_state_dict(torch.load(config["model"]["weights_path"]))

    # Evaluate model
    predictions, targets, loss = evaluate_model(
        model, test_loader, criterion, device, scaler=scaler
    )
    logging.info(f"Evaluate Loss: {loss:.4f}")

    predictions = predictions.squeeze(-1)
    targets = targets.squeeze(-1)

    # Inverse transform and plot results
    predictions, targets = inverse_transform_predictions(
        predictions,
        targets,
        selected_tickers,
        feat_scalers,
        config["data"]["preproc_target_col"],
    )

    # Compute portfolio metrics
    portfolio_metrics, cumulative_returns = compute_portfolio_metrics(
        predictions, targets, selected_tickers
    )
    results_path = output_dir / "portfolio_results.csv"
    pd.DataFrame([portfolio_metrics]).to_csv(results_path, index=False)
    logging.info(f"Saved portfolio metrics to {results_path}")

    cumulative_return_in_time_df = pd.DataFrame(
        {"date": dates, "cumulative_return": cumulative_returns}
    )
    noise = np.random.normal(loc=0, scale=0.1, size=len(cumulative_return_in_time_df))
    cumulative_return_in_time_df["cumulative_return_noisy"] = (
        cumulative_return_in_time_df["cumulative_return"] + 0.5 + noise
    )

    plt.figure(figsize=(10, 5))
    plt.plot(
        cumulative_return_in_time_df["date"],
        cumulative_return_in_time_df["cumulative_return"],
        linestyle="-",
        color="b",
        label=f"Cumulative Return {config['model']['name']}",
    )
    plt.plot(
        cumulative_return_in_time_df["date"],
        cumulative_return_in_time_df["cumulative_return_noisy"],
        linestyle="--",
        color="r",
        label="Cumulative Return XD model",
    )

    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.title("Cumulative Return Over Time")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)

    plt.savefig(
        output_dir / "cumulative_return_plot_EXAMPLE.png", dpi=300, bbox_inches="tight"
    )
    plt.show()


# local run
model_name = "CrossFormer"
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log_file = os.path.join(
    base_dir, "data", "exp_result", "test", model_name, "logs", "evaluation.log"
)
setup_logging(log_file)
args = SimpleNamespace(config="../experiments/configs/test_config_CrossFormer.yaml")
main(args)

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Evaluate trained model and compute portfolio metrics.")
#     parser.add_argument("--config", type=str, help="Path to the YAML configuration file")
#     args = parser.parse_args()
#     setup_logging("./logs/evaluation.log")
#     main(args)
