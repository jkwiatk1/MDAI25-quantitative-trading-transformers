import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from models.iTransformer import iTransformerModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Data load
def load_finance_data_xlsx(path):
    excel_data = pd.ExcelFile(path)
    data_dict = {}
    for sheet in excel_data.sheet_names:
        df = excel_data.parse(sheet, skiprows=2)
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


def calc_next_step_profit_rate(df, tickers):
    """
    Calculate the profit rate for each ticker at the next time step.

    Args:
        df (dict of pd.DataFrame): Dictionary of DataFrames for each ticker.
        tickers (list): List of ticker symbols.

    Returns:
        dict of pd.DataFrame: Updated DataFrames with 'Profit Rate' column.
    """
    for ticker in tickers:
        # profit rate t+1
        df[ticker]["Profit Rate"] = (
            df[ticker]["Close"].shift(-1) - df[ticker]["Close"]
        ) / df[ticker]["Close"]

        df[ticker]["Profit Rate"].fillna(0, inplace=True)

    return df


def calc_cumulative_features(df, tickers, time_step):
    """
    Calculate cumulative daily profit and turnover features for each ticker.

    Args:
        df (dict of pd.DataFrame): Dictionary of DataFrames for each ticker.
        tickers (list): List of ticker symbols.
        time_step (int): Time step for cumulative calculations.

    Returns:
        dict of pd.DataFrame: Updated dictionary of DataFrames with cumulative features.
    """
    for ticker in tickers:
        df[ticker]["Cumulative profit"] = df[ticker]["Daily profit"].cumsum()
        df[ticker]["Cumulative turnover"] = df[ticker]["Turnover"].cumsum()

        df[ticker]["Cumulative profit"].fillna(0, inplace=True)
        df[ticker]["Cumulative turnover"].fillna(0, inplace=True)

    return df


def calc_daily_profit_features(df):
    # daily profit rate
    df["Daily profit"] = (df["Close"] - df["Close"].shift(1)) / df["Close"].shift(1)
    df["Daily profit"].fillna(0, inplace=True)

    # daily turnover rate
    df["Turnover"] = df["Volume"] / df["Close"]
    df["Turnover"].fillna(0, inplace=True)
    return df


def calc_input_features(df, tickers, cols, time_step):
    """
    Calculate input features for all tickers, including intraday profit,
    daily profit, turnover, and cumulative features.

    Args:
        df (dict of pd.DataFrame): Dictionary of DataFrames for each ticker.
        tickers (list): List of ticker symbols.
        cols (list): Column names used to calculate intraday profit.
        time_step (int): Time step for cumulative calculations.

    Returns:
        dict of pd.DataFrame: Updated dictionary of DataFrames with all features.
    """
    for ticker in tickers:
        df[ticker].loc[:, "Intraday profit"] = (
            df[ticker][cols[0]] - df[ticker][cols[3]]
        ) / df[ticker][cols[3]]

        df[ticker] = calc_daily_profit_features(df[ticker])

    df = calc_cumulative_features(df, tickers, time_step)
    df = calc_next_step_profit_rate(df, tickers)

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

        # Reset the index to return 'Date' as a regular column
        # df[ticker].reset_index(inplace=True)
        # df[ticker].rename(columns={'index': 'Date'}, inplace=True)

    return df


def data_normalization(x):
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    x = (x - mean) / (std + 1e-5)
    return x


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
n_epochs = 100
lookback = 20
num_features = 3
test_split = 0.2
val_split = 0.1
batch_size = 64
learning_rate = 0.005
start_date = pd.to_datetime("2023-01-03")
end_date = pd.to_datetime("2024-01-12")
tickers_to_use = [
    "USDT-USD",
    "BTC-USD",
    "XRP-USD",
    "ETH-USD",
    "BNB-USD",
    "DOGE-USD",
    "SOL-USD",
    "STETH-USD",
    "DOT-USD",
    "TSLA",
    "AAPL",
    "NVDA",
    "PLTR",
    "SMCI",
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
    "Intraday",
    "Daily profit",
    "Cumulative profit",
    # "Profit Rate",
    # "Turnover"
]
preproc_target_col = "Daily profit"
load_file = "../data/finance/popular_tickers/historical_data_2022-01-01-2024-12-01-1d.xlsx"

# Model params
input_dim = num_features
d_model = 64  # check [3,512]
nhead = 1  # check [1, 4]
num_encoder_layers = 1
dim_feedforward = 64  # check [3, 512]
dropout = 0.05

# data load
data_raw, all_tickers = load_finance_data_xlsx(load_file)
data_raw = fill_missing_days(data_raw.copy(), all_tickers, start_date, end_date)
data = prepare_finance_data(
    data_raw,
    tickers_to_use,
    init_cols_to_use,
)
data = calc_input_features(
    df=data.copy(), tickers=tickers_to_use, cols=init_cols_to_use, time_step=lookback
)

# Normalization on raw  data
# for ticker in tickers_to_use:
#     data[ticker] = data_normalization(data[ticker].copy())

# normalization
feat_scalers = {
    ticker: {
        feature: MinMaxScaler(feature_range=(0, 1)) for feature in data[ticker].columns
    }
    for ticker in tickers_to_use
}
data_scaled = data.copy()
# Scale the data for each ticker and feature
for ticker in tickers_to_use:
    for feature in data[ticker].columns:
        """
        Fit and transform the data for the current ticker and feature.
        The scaler for each feature is stored in the nested scalers dictionary.
        """
        data_scaled[ticker].loc[:, feature] = feat_scalers[ticker][
            feature
        ].fit_transform(data[ticker][[feature]])

# Prepare combined dataset
combined_data, ticker_mapping = prepare_combined_data(
    data_scaled, tickers_to_use, lookback
)

# Normalization on scaled data
# combined_data = data_normalization(combined_data.copy())
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

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = build_transformer(
    input_dim=lookback,
    # input_dim=lookback * 3,  # Include features (Close, Volume, ticker_id)
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
