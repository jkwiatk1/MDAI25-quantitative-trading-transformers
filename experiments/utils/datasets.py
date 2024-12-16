import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


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
