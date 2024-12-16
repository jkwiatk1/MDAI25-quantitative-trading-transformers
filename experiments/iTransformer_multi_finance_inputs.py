import matplotlib.pyplot as plt
import yaml
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

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
from experiments.utils.training import build_transformer

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

for epoch in range(n_epochs):
    model.train()
    train_loss = 0.0
    for batch_sequences, batch_targets in train_loader:
        batch_sequences, batch_targets = batch_sequences.to(device), batch_targets.to(
            device
        )
        optimizer.zero_grad()
        predictions = model(batch_sequences).squeeze()
        loss = criterion(predictions, batch_targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_sequences.size(0)
    train_loss /= len(train_loader.dataset)

    # Validation
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_sequences, batch_targets in val_loader:
            batch_sequences, batch_targets = batch_sequences.to(
                device
            ), batch_targets.to(device)
            predictions = model(batch_sequences).squeeze()
            loss = criterion(predictions, batch_targets)
            val_loss += loss.item() * batch_sequences.size(0)
    val_loss /= len(val_loader.dataset)

    print(
        f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}"
    )

# Test
model.eval()
test_loss = 0.0
predictions_list, targets_list = [], []
with torch.no_grad():
    for batch_sequences, batch_targets in test_loader:
        batch_sequences, batch_targets = batch_sequences.to(device), batch_targets.to(
            device
        )
        predictions = model(batch_sequences).squeeze()
        loss = criterion(predictions, batch_targets)
        test_loss += loss.item() * batch_sequences.size(0)
        predictions_list.append(predictions.cpu())
        targets_list.append(batch_targets.cpu())
test_loss /= len(test_loader.dataset)
print(f"Test Loss: {test_loss:.4f}")

test_predictions = torch.cat(predictions_list).numpy()
test_targets = torch.cat(targets_list).numpy()

plt.figure(figsize=(10, 5))
plt.plot(test_targets, label="True Values", linestyle="dashed")
plt.plot(test_predictions, label="Predictions")
plt.legend()
plt.title("iTransformer Multi-Ticker Predictions on Test Set")
plt.show()

for i, ticker in enumerate(tickers_to_use):
    """
    Apply inverse transform for each ticker's predictions.
    Use the scalers dictionary to inverse transform the target feature for each ticker.
    """
    if len(tickers_to_use) == 1:
        test_predictions[i] = (
            feat_scalers[ticker][preproc_target_col]
            .inverse_transform(test_predictions[i].reshape(-1, 1))
            .flatten()
        )
        test_targets[i] = (
            feat_scalers[ticker][preproc_target_col]
            .inverse_transform(test_targets[i].reshape(-1, 1))
            .flatten()
        )
    else:
        test_predictions[:, i] = (
            feat_scalers[ticker][preproc_target_col]
            .inverse_transform(test_predictions[:, i].reshape(-1, 1))
            .flatten()
        )
        test_targets[:, i] = (
            feat_scalers[ticker][preproc_target_col]
            .inverse_transform(test_targets[:, i].reshape(-1, 1))
            .flatten()
        )

# Separe plots for each ticker
for i, ticker in enumerate(tickers_to_use):
    plt.figure(figsize=(10, 5))

    if len(tickers_to_use) == 1:
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
    plt.show()

#     plt.savefig(f"{ticker}_predictions.png")
#     plt.close()
#
# print("Plots saved for each ticker!")
