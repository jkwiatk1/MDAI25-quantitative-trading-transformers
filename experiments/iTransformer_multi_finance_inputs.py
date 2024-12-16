import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import Dataset, DataLoader

from models.iTransformer import iTransformerModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# TODO VERIFY Function to prepare input tensor for QuantFormer
def prepare_input_tensor_for_quantformer(df, tickers, lookback, features):
    """
    Prepare input tensor of shape (N, T, F) for QuantFormer.

    Args:
        df (dict of pd.DataFrame): Dictionary of DataFrames for each ticker.
        tickers (list): List of ticker symbols.
        lookback (int): Lookback window.
        features (list): List of feature names to include in the input tensor.

    Returns:
        np.ndarray: Input tensor of shape (N, T, F).
    """
    input_tensors = []
    for ticker in tickers:
        ticker_data = df[ticker][features].tail(lookback).values
        if ticker_data.shape[0] < lookback:
            # Pad with zeros if data is shorter than the lookback window
            padding = np.zeros((lookback - ticker_data.shape[0], len(features)))
            ticker_data = np.vstack((padding, ticker_data))
        input_tensors.append(ticker_data)
    return np.stack(input_tensors, axis=0)


# Data load
def load_finance_data_xlsx(path, is_from_yahoo=True):
    excel_data = pd.ExcelFile(path)
    data_dict = {}
    for sheet in excel_data.sheet_names:
        if is_from_yahoo == False:
            df = excel_data.parse(sheet, skiprows=2)
        else:
            df = excel_data.parse(sheet)
        df.columns = [
            "Date",
            "Adj Close",
            "Close",
            "High",
            "Low",
            "Open",
            "Volume",
            "ticker",
        ]
        data_dict[sheet] = df
    return data_dict, excel_data.sheet_names


# Function to fill in missing days with default values
def fill_missing_days(df, tickers, start_date, end_date):
    """
    Fill missing dates.

    - Fills missing weekends or other gaps in trading days.
    - Uses forward-fill for 'Close'.
    - Sets default values for missing fields as specified:
      - 'Adj Close' = 'Close'
      - 'High' = 'Close'
      - 'Low' = 'Close'
      - 'Open' = 'Close'
      - 'Volume' = 0
    """
    # Ensure the 'Date' column is a datetime object and set it as the index
    for ticker in tickers:
        df[ticker]["Date"] = pd.to_datetime(df[ticker]["Date"])
        df[ticker].set_index("Date", inplace=True)

        # Create a full range of dates from the minimum to the maximum in the dataset
        # end_date = df[ticker].index.max()
        full_date_range = pd.date_range(start=start_date, end=end_date, freq="D")

        # Reindex the DataFrame to include all dates in the range
        df[ticker] = df[ticker].reindex(full_date_range)

        # Fill missing 'Close' values with the value from the previous day (forward-fill)
        df[ticker]["Close"].fillna(method="ffill", inplace=True)

        # Assign default values for other columns based on 'Close' or specific rules
        df[ticker]["Adj Close"].fillna(method="ffill", inplace=True)
        df[ticker]["High"].fillna(method="ffill", inplace=True)
        df[ticker]["Low"].fillna(method="ffill", inplace=True)
        df[ticker]["Open"].fillna(method="ffill", inplace=True)
        df[ticker]["Volume"].fillna(method="ffill", inplace=True)
        df[ticker]["ticker"].fillna(method="ffill", inplace=True)
    return df


def prepare_finance_data(df, tickers, cols):
    """
        Prepares financial data by selecting specific tickers and columns.

    Args:
        df (pd.DataFrame): A data frame containing financial data.
        tickers (list): List of tickers to be selected.
        cols (list): List of column names to be included.

    Returns:
        pd.DataFrame: The filtered data frame.
    """
    return {ticker: data[cols] for ticker, data in df.items() if ticker in tickers}


def data_normalization(x):
    mean = x.mean()
    std = x.std()
    x = (x - mean) / (std + 1e-5)
    return x


def calc_cumulative_features(df, tickers, time_step, cols):
    """
    Calculate cumulative daily profit and turnover features for each ticker.

    Args:
        df (dict of pd.DataFrame): Dictionary of DataFrames for each ticker.
        tickers (list): List of ticker symbols.
        time_step (int): Lookback window for cumulative calculations.

    Returns:
        dict of pd.DataFrame: Updated dictionary of DataFrames with cumulative features.
    """
    for ticker in tickers:
        # df[ticker] = df[ticker].apply(data_normalization)  # TODO verify if okey

        # df[ticker].loc[:, "Intraday profit"] = (
        #     df[ticker][cols[0]] - df[ticker][cols[3]]
        # ) / df[ticker][cols[3]]
        # # profit rate t+1
        # df[ticker]["Profit Rate"] = (
        #     df[ticker]["Close"].shift(-1) - df[ticker]["Close"]
        # ) / df[ticker]["Close"]
        # df[ticker]["Profit Rate"].fillna(0, inplace=True)

        df[ticker]["Daily profit"] = (
            df[ticker]["Close"] - df[ticker]["Close"].shift(1)
        ) / df[ticker]["Close"].shift(1)
        df[ticker]["Turnover"] = df[ticker]["Volume"] / df[ticker]["Close"]
        df[ticker]["Cumulative profit"] = (
            df[ticker]["Daily profit"].rolling(window=time_step, min_periods=1).sum()
        )
        df[ticker]["Cumulative turnover"] = (
            df[ticker]["Turnover"].rolling(window=time_step, min_periods=1).sum()
        )

        df[ticker]["Daily profit"].fillna(0, inplace=True)
        df[ticker]["Turnover"].fillna(0, inplace=True)
        df[ticker]["Cumulative profit"].fillna(0, inplace=True)
        df[ticker]["Cumulative turnover"].fillna(0, inplace=True)
    return df


# Function to calculate cumulative features as described in the article
def calc_input_features(df, tickers, cols, time_step):
    """
    Calculate input features for all tickers, including:
    - intraday profit, NOT USED
    - daily profit,
    - turnover,
    - cumulative features.

    Args:
        df (dict of pd.DataFrame): Dictionary of DataFrames for each ticker.
        tickers (list): List of ticker symbols.
        cols (list): Column names used to calculate intraday profit.
        time_step (int): Time step for cumulative calculations.

    Returns:
        dict of pd.DataFrame: Updated dictionary of DataFrames with all features.
    """

    df = calc_cumulative_features(df, tickers, time_step, cols)
    return df


# Combine all tickers into a single dataset
def prepare_combined_data(data_scaled, tickers_to_use, lookback):
    combined_data = []
    ticker_mapping = {
        ticker: idx for idx, ticker in enumerate(tickers_to_use)
    }  # Assign IDs to tickers

    for ticker in tickers_to_use:
        df = data_scaled[ticker].copy()
        df["ticker id"] = ticker_mapping[ticker]  # Add ticker ID
        # ticker_id_col = df.pop('ticker id')
        df = df.add_suffix(f"_{ticker}")
        # df['ticker id'] = ticker_id_col
        combined_data.append(df)

    combined_data = pd.concat(combined_data, axis=1)

    # # Concatenate data for all tickers
    # combined_data = pd.concat(combined_data, keys=tickers_to_use)
    return combined_data, ticker_mapping


# Adjusted create_sequences function
def create_combined_sequences(
    data, lookback, cols_to_use=["Close"], target_col="Close"
):
    """
    Create sequences and targets for time series forecasting.

    Args:
        data (pd.DataFrame): Input DataFrame with time series data.
        lookback (int): Number of past time steps to include in each sequence.

    Returns:
        sequences (np.array): Array of input sequences.
        targets (np.array): Array of target values.
    """
    if cols_to_use is None:
        cols_to_use = ["Close", "Intraday", "Daily", "Turnover"]
    sequences, targets = [], []

    # Select columns matching the specified keywords
    columns_to_include = []
    for col in cols_to_use:
        columns_to_include += data.filter(like=col, axis=1).columns.tolist()

    # Column order matches the input DataFrame order
    columns_to_include = [col for col in data.columns if col in columns_to_include]

    # Filter "Close" columns for multi-target values
    target_columns = [col for col in columns_to_include if target_col in col]

    for i in range(len(data) - lookback):
        # Extract sequence of features
        seq = data.iloc[i : i + lookback][columns_to_include].values

        # Extract targets for all "Close" columns
        target = data.iloc[i + lookback][target_columns].values
        sequences.append(seq)
        targets.append(target)

    return np.array(sequences), np.array(targets)


# Function to normalize data for each ticker
def normalize_data_for_quantformer(df, tickers, features_to_normalize):
    """
    Normalize features using MinMaxScaler for each ticker.

    Args:
        df (dict of pd.DataFrame): Dictionary of DataFrames for each ticker.
        tickers (list): List of ticker symbols.

    Returns:
        dict of pd.DataFrame: Normalized data.
        dict: Dictionary of scalers used for each feature.
    """
    scalers = {ticker: {} for ticker in tickers}

    for ticker in tickers:
        # features_to_normalize = df[ticker].columns
        for feature in features_to_normalize:
            scaler = MinMaxScaler(feature_range=(0, 1))
            df[ticker][feature] = scaler.fit_transform(df[ticker][[feature]])
            scalers[ticker][feature] = scaler
    return df, scalers


# Dataset for multi-ticker
class MultiTickerDataset(Dataset):
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


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


# Params
n_epochs = 20
lookback = 20
num_features = 3
test_split = 0.2
val_split = 0.1
batch_size = 64
learning_rate = 0.005
IS_DATA_FROM_YAHOO = False

if IS_DATA_FROM_YAHOO:
    start_date = pd.to_datetime("2020-01-03")
else:
    start_date = pd.to_datetime("2022-01-03")
end_date = pd.to_datetime("2024-01-12")

if IS_DATA_FROM_YAHOO == False:
    tickers_to_use = [
        "USDT-USD",
        "BTC-USD",
        # "XRP-USD",
        "ETH-USD",
        # "BNB-USD",
        # "DOGE-USD",
        # "SOL-USD",
        # "STETH-USD",
        # "DOT-USD",
        "TSLA",
        "AAPL",
        "NVDA",
        # "PLTR",
        # "SMCI",
    ]
else:
    tickers_to_use = [  # ~top 20 market cap
        "AAPL",
        "NVDA",
        "MSFT",
        "AMZN",
        "GOOG",
        "GOOGL",
        "META",
        "TSLA",
        "BRK-B",
        "AVGO",
        "WMT",
        "LLY",
        "JPM",
        "V",
        "ORCL",
        "UNH",
        "XOM",
        "MA",
        "COST",
        "HD",
        "PG",
        "NFLX",
        "JNJ",
    ]
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

# Model params
input_dim = num_features
d_model = 64  # check [3,512]
nhead = 1  # check [1, 4]
num_encoder_layers = 1
dim_feedforward = 64  # check [3, 512]
dropout = 0.05

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
