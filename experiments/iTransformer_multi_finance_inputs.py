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


def calc_daily_profit_features(df):
    # daily profit rate
    df["Daily profit"] = (df["Close"] - df["Close"].shift(1)) / df["Close"].shift(1)
    df["Daily profit"].fillna(0, inplace=True)

    # daily turnover rate
    df["Turnover"] = df["Volume"] / df["Close"]
    df["Turnover"].fillna(0, inplace=True)
    return df

def calc_input_features(df, tickers, cols, time_step):
    for ticker in tickers:
        df[ticker]["Intraday profit"] = (df[ticker][cols[0]] - df[ticker][cols[3]]) / df[ticker][cols[3]]
        df[ticker] = calc_daily_profit_features(df[ticker])
    return df


def create_sequences(data, lookback):
    sequences, targets = [], []
    # Iterate through rows to create seq
    for i in range(len(data) - lookback):
        # Get the sequence of data for the given lookback period
        sequence = data.iloc[i: i + lookback].values
        sequences.append(sequence)
        # Assuming the target is the value in the first column
        target = data.iloc[i + lookback, 0]  # Target is the first column
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
        df[ticker]['Date'] = pd.to_datetime(df[ticker]['Date'])
        df[ticker].set_index('Date', inplace=True)

        # Create a full range of dates from the minimum to the maximum in the dataset
        # end_date = df[ticker].index.max()
        full_date_range = pd.date_range(start=start_date, end=end_date, freq='D')

        # Reindex the DataFrame to include all dates in the range
        df[ticker] = df[ticker].reindex(full_date_range)

        # Fill missing 'Close' values with the value from the previous day (forward-fill)
        df[ticker]['Close'].fillna(method='ffill', inplace=True)

        # Assign default values for other columns based on 'Close' or specific rules
        df[ticker]['Adj Close'].fillna(method='ffill', inplace=True)
        df[ticker]['High'].fillna(method='ffill', inplace=True)
        df[ticker]['Low'].fillna(method='ffill', inplace=True)
        df[ticker]['Open'].fillna(method='ffill', inplace=True)
        df[ticker]['Volume'].fillna(method='ffill', inplace=True)
        df[ticker]['ticker'].fillna(method='ffill', inplace=True)

        # Reset the index to return 'Date' as a regular column
        # df[ticker].reset_index(inplace=True)
        # df[ticker].rename(columns={'index': 'Date'}, inplace=True)

    return df

class FinanceDataset(Dataset):
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
    )


# Params
n_epochs = 200
lookback = 30
n_samples = 200
num_features = 3
test_split = 0.2
val_split = 0.1
batch_size = 32
tickers_to_use = ["SOL-USD"]
start_date = pd.to_datetime("2023-01-03")
end_date = pd.to_datetime("2024-10-25")
# tickers_to_use = ["SOL-USD", "BTC-USD", "TSLA"]
cols_to_use = [
    "Close",
    "High",
    "Low",
    "Open",
    "Volume",
]
load_file = f"../data/finance/historical_data_2023-01-01-2024-10-26-1d.xlsx"

# Model params
input_dim = num_features
d_model = 128
nhead = 2
num_encoder_layers = 1
dim_feedforward = 128
dropout = 0.05

# data load
data_raw, all_tickers = load_finance_data_xlsx(load_file)
data_raw = fill_missing_days(data_raw.copy(), all_tickers, start_date, end_date)
data = prepare_finance_data(
    data_raw,
    tickers_to_use,
    cols_to_use,
)
data = calc_input_features(df=data,tickers=tickers_to_use, cols=cols_to_use, time_step = lookback)
# normalization
# scalers = {feature: MinMaxScaler(feature_range=(0, 1)) for feature in data["SOL-USD"].columns}
scalers = {ticker: {feature: MinMaxScaler(feature_range=(0, 1)) for feature in data[ticker].columns} for ticker in tickers_to_use}
data_scaled = data.copy()
# Scale the data for each ticker and feature
for ticker in tickers_to_use:
    for feature in data[ticker].columns:
        """
        Fit and transform the data for the current ticker and feature.
        The scaler for each feature is stored in the nested scalers dictionary.
        """
        data_scaled[ticker].loc[:, feature] = scalers[ticker][feature].fit_transform(
            data[ticker][[feature]]
        )

for ticker in tickers_to_use:
    data_scaled[ticker] = data_scaled[ticker].reset_index(drop=True)

# Train test split
train_size = int((1 - test_split) * len(data_scaled[tickers_to_use[0]]))
# val_size = int(val_split * train_size)
train_data = data_scaled[tickers_to_use[0]][:train_size]
test_data = data_scaled[tickers_to_use[0]][train_size:]
train_sequences, train_targets = create_sequences(train_data, lookback)
test_sequences, test_targets = create_sequences(test_data, lookback)

# val_sequences = train_sequences[-val_size:]
# val_targets = train_targets[-val_size:]
# train_sequences = train_sequences[:-val_size]
# train_targets = train_targets[:-val_size]
# train_sequences = train_sequences[:-train_size]
# train_targets = train_targets[:-train_size]

train_dataset = FinanceDataset(train_sequences, train_targets)
# val_dataset = FinanceDataset(val_sequences, val_targets)
test_dataset = FinanceDataset(test_sequences, test_targets)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

model = build_transformer(
    input_dim=lookback,
    d_model=d_model,
    nhead=nhead,
    num_encoder_layers=num_encoder_layers,
    dim_feedforward=dim_feedforward,
    dropout=dropout,
).to(device)

# Training
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

for epoch in range(n_epochs):
    model.train()
    train_loss = 0.0
    for batch_sequences, batch_targets in train_loader:
        batch_sequences, batch_targets = batch_sequences.to(device), batch_targets.to(device)
        optimizer.zero_grad()
        predictions = model(batch_sequences).squeeze()
        loss = criterion(predictions, batch_targets)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_sequences.size(0)
    train_loss /= len(train_loader.dataset)

    # Vals
    # model.eval()
    # val_loss = 0.0
    # with torch.no_grad():
    #     for batch_sequences, batch_targets in val_loader:
    #         batch_sequences, batch_targets = batch_sequences.to(device), batch_targets.to(device)
    #         predictions = model(batch_sequences).squeeze()
    #         loss = criterion(predictions, batch_targets)
    #         val_loss += loss.item() * batch_sequences.size(0)
    # val_loss /= len(val_loader.dataset)

    print(f"Epoch {epoch + 1}/{n_epochs}, Train Loss: {train_loss:.4f}") #, Val Loss: {val_loss:.4f}")

# Test
model.eval()
test_loss = 0.0
predictions_list, targets_list = [], []
with torch.no_grad():
    for batch_sequences, batch_targets in test_loader:
        batch_sequences, batch_targets = batch_sequences.to(device), batch_targets.to(device)
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
plt.title("iTransformer Predictions on Test Set")
plt.show()