import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch import nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset, DataLoader
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
    df = {ticker: data[cols] for ticker, data in df.items() if ticker in tickers}
    return df


def create_sequences(data, lookback):
    sequences = []
    targets = []

    # Iterate through rows to create seq
    for i in range(len(data) - lookback):
        # Get the sequence of data for the given lookback period
        sequence = data.iloc[i : i + lookback].values
        sequences.append(sequence)
        # Assuming the target is the value in the first column
        target = data.iloc[i + lookback, 0]  # Target is the first column
        targets.append(target)
    return np.array(sequences), np.array(targets)


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
    # Create the transformer
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
batch_size = 32
tickers_to_use = ["SOL-USD"]
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
data_raw, tickers = load_finance_data_xlsx(load_file)
data = prepare_finance_data(
    data_raw,
    tickers_to_use,
    cols_to_use,
)
scalers = {feature: MinMaxScaler(feature_range=(0, 1)) for feature in cols_to_use}
data_scaled = data.copy()
for ticker in tickers_to_use:
    for feature in cols_to_use:
        data_scaled[ticker].loc[:, feature] = scalers[feature].fit_transform(
            data[ticker][[feature]]
        )

# Train test split
train_size = int((1 - test_split) * len(data_scaled[tickers_to_use[0]]))
train_data = data_scaled[tickers_to_use[0]][:train_size]
test_data = data_scaled[tickers_to_use[0]][train_size:]

train_sequences, train_targets = create_sequences(train_data, lookback)
test_sequences, test_targets = create_sequences(test_data, lookback)

# train_sequences = torch.tensor(train_sequences, dtype=torch.float32).unsqueeze(-1).to(device)
train_sequences = torch.tensor(train_sequences, dtype=torch.float32).to(device)
train_targets = torch.tensor(train_targets, dtype=torch.float32).to(device)
# test_sequences = torch.tensor(test_sequences, dtype=torch.float32).unsqueeze(-1).to(device)
test_sequences = torch.tensor(test_sequences, dtype=torch.float32).to(device)
test_targets = torch.tensor(test_targets, dtype=torch.float32).to(device)

train_dataset = TensorDataset(train_sequences, train_targets)
test_dataset = TensorDataset(test_sequences, test_targets)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
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
    epoch_loss = 0
    for batch_sequences, batch_targets in train_loader:
        optimizer.zero_grad()
        predictions = model(batch_sequences)
        loss = criterion(predictions, batch_targets)
        loss.backward()
        optimizer.step()
        # epoch_loss += loss.item()
        epoch_loss += loss.item() * batch_sequences.size(0)

    epoch_loss /= len(train_loader)
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss:.4f}")

# Test
model.eval()
test_loss = 0.0
predictions_list = []
targets_list = []

with torch.no_grad():
    for batch_sequences, batch_targets in test_loader:
        test_predictions = model(batch_sequences).squeeze()
        loss = criterion(test_predictions, batch_targets)
        # test_loss += loss.item()
        test_loss += loss.item() * batch_sequences.size(0)
        predictions_list.append(test_predictions.cpu())
        targets_list.append(batch_targets.cpu())

test_loss /= len(test_loader)
print(f"Test Loss: {test_loss:.4f}")

test_predictions = torch.cat(predictions_list).numpy()
test_targets = torch.cat(targets_list).numpy()

plt.figure(figsize=(10, 5))
plt.plot(
    range(len(test_targets)), test_targets, label="True Values", linestyle="dashed"
)
plt.plot(range(len(test_predictions)), test_predictions, label="Predictions")
plt.legend()
plt.title("Transformer Predictions on Test Set")
plt.show()
