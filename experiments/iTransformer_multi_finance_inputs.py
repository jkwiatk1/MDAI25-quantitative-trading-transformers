import yaml
import pandas as pd
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import QuantileTransformer

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
from experiments.utils.training import (
    build_transformer,
    train_model,
    evaluate_model,
    inverse_transform_predictions,
    plot_predictions,
)


def quantize_labels(labels, n_quantiles=3):
    """
    Quantize returns into discrete classes based on quantiles.

    Parameters:
    labels: array-like
        Continuous values of returns to be quantized.
    n_quantiles: int
        Number of quantiles to divide the data into (e.g., 3 for low, middle, high).

    Returns:
    np.ndarray
        Array of discrete class labels (e.g., 0 for lowest quantile, 1 for middle, etc.).
    """
    # Fit QuantileTransformer to map values to the range [0, 1]
    quantizer = QuantileTransformer(
        n_quantiles=n_quantiles, output_distribution="uniform"
    )
    normalized_labels = quantizer.fit_transform(labels.reshape(-1, 1))

    # Create quantile bins with equal probability
    bins = np.linspace(0, 1, n_quantiles + 1)[
        1:-1
    ]  # Quantile boundaries excluding 0 and 1

    # Assign each label to a quantile class (e.g., 0 for lowest quantile)
    quantized_labels = np.digitize(normalized_labels.flatten(), bins=bins)

    return quantized_labels


# Strategy Implementation
def trading_strategy(model, data_loader, device, cash):
    stock_pool = []
    model.eval()
    with torch.no_grad():
        for sequences, _ in data_loader:
            sequences = sequences.to(device)
            predictions = model(sequences)
            ranked_stocks = torch.argsort(predictions, dim=-1, descending=True)
            stock_pool = ranked_stocks[:, : cash // len(ranked_stocks)]
    return stock_pool


IS_DATA_FROM_YAHOO = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

if IS_DATA_FROM_YAHOO == False:
    with open("training_config.yaml", "r") as f:
        config = yaml.safe_load(f)
else:
    with open("yahoo_training_config.yaml", "r") as f:
        config = yaml.safe_load(f)

# Training params
n_epochs = config["training"]["n_epochs"]
lookback = config["training"]["lookback"]
test_split = config["training"]["test_split"]
val_split = config["training"]["val_split"]
batch_size = config["training"]["batch_size"]
learning_rate = config["training"]["learning_rate"]
start_date = pd.to_datetime(config["data"]["start_date"])
end_date = pd.to_datetime(config["data"]["end_date"])
tickers_to_use = config["data"]["tickers"]

# Model params
d_model = config["model"]["d_model"]  # check [3,512]
nhead = config["model"]["nhead"]  # check [1, 4]
num_encoder_layers = config["model"]["num_encoder_layers"]  # check [1, 4]
dim_feedforward = config["model"]["dim_feedforward"]  # check [3, 512]
dropout = config["model"]["dropout"]  # check [3, 512]
num_features = config["model"]["num_features"]
input_dim = num_features
np.random.seed(42)


init_cols_to_use = [
    "Close",
    "High",
    "Low",
    "Open",
    "Volume",
    # "ticker"
]
preproc_cols_to_use = [
    # "Close",
    # "Intraday",
    # "Daily profit",
    # "Profit Rate",
    # "Turnover"
    "Cumulative profit",
    "Cumulative turnover",
]
preproc_target_col = "Cumulative profit"
load_file = (
    "../data/finance/popular_tickers/historical_data_2022-01-01-2024-12-01-1d.xlsx"
)
# load_file = "../data/finance/sp500/preprocess/sp500_stocks_historical_data.xlsx"


# data load
data_raw, all_tickers = load_finance_data_xlsx(load_file, IS_DATA_FROM_YAHOO)
data_raw = fill_missing_days(data_raw.copy(), tickers_to_use, start_date, end_date)
data = prepare_finance_data(
    data_raw,
    tickers_to_use,
    init_cols_to_use,
)
data = calc_input_features(
    df=data, tickers=tickers_to_use, cols=init_cols_to_use, time_step=lookback
)

# normalization & prepare combined dataset & create seq
data_scaled, feat_scalers = normalize_data_for_quantformer(
    data, tickers_to_use, preproc_cols_to_use
)

# input_tensor = prepare_input_tensor_for_quantformer(
#     data, tickers_to_use, lookback, preproc_cols_to_use
# )

combined_data, ticker_mapping = prepare_combined_data(
    data_scaled, tickers_to_use, lookback
)
sequences, targets = create_combined_sequences(
    combined_data, lookback, preproc_cols_to_use, preproc_target_col
)

# targets = quantize_labels(targets, n_quantiles=5)  # TODO test n_quantiles param

# Train test split
train_size = int((1 - test_split) * len(sequences))
val_size = int(val_split * train_size)

train_sequences = sequences[:train_size]
train_targets = targets[:train_size]
test_sequences = sequences[train_size:]
test_targets = targets[train_size:]

val_sequences = train_sequences[-val_size:]
val_targets = train_targets[-val_size:]
train_sequences = train_sequences[:-val_size]
train_targets = train_targets[:-val_size]

# Create datasets and dataloaders
train_dataset = MultiTickerDataset(train_sequences, train_targets)
val_dataset = MultiTickerDataset(val_sequences, val_targets)
test_dataset = MultiTickerDataset(test_sequences, test_targets)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = build_transformer(
    input_dim=lookback,
    d_model=d_model,
    nhead=nhead,
    num_encoder_layers=num_encoder_layers,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
    num_features=len(tickers_to_use),
    columns_amount=train_sequences.shape[2],
).to(device)

# Training
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
train_model(model, train_loader, val_loader, criterion, optimizer, device, n_epochs)

# Test
test_predictions, test_targets, test_loss = evaluate_model(
    model, test_loader, criterion, device
)
print(f"Test Loss: {test_loss:.4f}")

test_predictions, test_targets = inverse_transform_predictions(
    test_predictions, test_targets, tickers_to_use, feat_scalers, preproc_target_col
)

plot_predictions(test_predictions, test_targets, tickers_to_use)

# Backtest Strategy
cash = 100000
print(trading_strategy(model, test_loader, device, cash))
