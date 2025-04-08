import logging

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset



def split_data_chronologically(data_dict, tickers, train_ratio, val_ratio):
    """Dzieli słownik DataFrame'ów chronologicznie na train, val, test."""
    if not (0 < train_ratio < 1) or not (0 < val_ratio < 1) or not (train_ratio + val_ratio < 1):
         raise ValueError("Invalid split ratios. Must be > 0 and < 1, and train+val < 1.")

    train_data_dict = {}
    val_data_dict = {}
    test_data_dict = {} # Mimo że nie używamy w grid search, funkcja może zwracać

    for ticker in tickers:
        if ticker not in data_dict: continue
        df = data_dict[ticker]
        n_samples = len(df)
        if n_samples == 0: continue

        train_end_idx = int(train_ratio * n_samples)
        val_end_idx = train_end_idx + int(val_ratio * n_samples)

        train_data_dict[ticker] = df.iloc[:train_end_idx]
        val_data_dict[ticker] = df.iloc[train_end_idx:val_end_idx]
        test_data_dict[ticker] = df.iloc[val_end_idx:] # Reszta to test

        if len(train_data_dict[ticker]) == 0 or len(val_data_dict[ticker]) == 0:
             logging.warning(f"Train or Val split resulted in empty DataFrame for ticker {ticker}. Check ratios/data length.")

    return train_data_dict, val_data_dict, test_data_dict

def prepare_sequential_data_CrossFormer(data_scaled, tickers_to_use, lookback, patch_size, target_col_index=0):
    """
     Prepares data in the format (Batch, Stocks, Patches, Patch_Size, Features) for CrossFormer.

    Args:
        data_scaled (dict[str, pd.DataFrame]): Dictionary {ticker: DataFrame[Time, Features]}
        tickers_to_use (list[str]): List of actions to use
        lookback (int): Number of historical days for prediction
        patch_size (int): Number of days in one patch
        target_col_index (int): Target column index (default 0)

    Returns:
        X (torch.Tensor): Input data  (Batch, Stocks, Patches, Patch_Size, Features)
        y (torch.Tensor): Target data (Batch, Stocks, 1)
        ticker_mapping (dict): Map {ticker: indeks}
    """
    assert lookback % patch_size == 0, "Lookback must be a multiple of patch_size!"

    ticker_mapping = {ticker: idx for idx, ticker in enumerate(tickers_to_use)}
    num_time, num_features = data_scaled[tickers_to_use[0]].shape
    num_stocks = len(tickers_to_use)

    assert num_time > lookback, "Not enough time steps in data to create sequences!"

    data_matrix = np.stack(
        [data_scaled[ticker].values for ticker in tickers_to_use], axis=1
    )  # (Time, Stocks, Features)

    num_patches = lookback // patch_size  # Number of patches in the lookback window

    X, y = [], []
    for i in range(num_time - lookback):
        sequence = data_matrix[i : i + lookback]  # (Lookback, Stocks, Features)

        # Segmentation (Patches)
        patches = np.array(np.split(sequence, num_patches, axis=0))  # (Patches, Patch_Size, Stocks, Features)
        patches = np.transpose(patches, (2, 0, 1, 3))  # (Stocks, Patches, Patch_Size, Features)

        X.append(patches)
        y.append(data_matrix[i + lookback, :, target_col_index])  # (Stocks,)

    X = torch.tensor(np.array(X), dtype=torch.float32)  # (Batch, Stocks, Patches, Patch_Size, Features)
    y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1)  # (Batch, Stocks, 1)

    return X, y, ticker_mapping


def prepare_sequential_data(
    data_scaled, tickers_to_use, lookback, target_col_index=0
):
    """
    Converts data into format (Batch, Time, Stocks, Features)

    Args:
        data_scaled (dict[str, pd.DataFrame]): Dictionary {ticker: DataFrame[Time, Features]}
        tickers_to_use (list[str]): List of actions to use
        lookback (int): Number of historical days for prediction

    Returns:
        X (torch.Tensor): Input data (Batch, Time, Stocks, Features)
        y (torch.Tensor): target data (Batch, Stocks, 1)
        ticker_mapping (dict): Mapping stock names to indexes
    """
    ticker_mapping = {ticker: idx for idx, ticker in enumerate(tickers_to_use)}

    num_time, num_features = data_scaled[tickers_to_use[0]].shape
    num_stocks = len(tickers_to_use)

    assert num_time > lookback, "Not enough period time in data to create a sequence!"

    data_matrix = np.stack(
        [data_scaled[ticker].values for ticker in tickers_to_use], axis=1
    )  # (Time, Stocks, Features)

    # Sequences
    X, y = [], []
    for i in range(num_time - lookback):
        X.append(data_matrix[i : i + lookback])  # (Time, Stocks, Features)
        y.append(data_matrix[i + lookback, :, target_col_index])

    X = torch.tensor(
        np.array(X), dtype=torch.float32
    )  # (Batch, Time, Stocks, Features)
    y = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(
        -1
    )  # (Batch, Stocks, 1)

    return X, y, ticker_mapping


def prepare_combined_data(data_scaled, tickers_to_use, lookback):
    """
    Combine all tickers into a single dataset with size [Time, Features*Stocks]
    Args:
        data_scaled:
        tickers_to_use:
        lookback:

    Returns: [Time, Features*Stocks]

    """
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


def normalize_data(df, tickers, features_to_normalize):
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

def fit_and_transform_data(train_data_dict, val_data_dict, tickers, features_to_normalize):
    """Dopasowuje scalery na danych treningowych i transformuje zbiory train/val."""
    scalers = {ticker: {} for ticker in tickers}
    train_scaled_dict = {}
    val_scaled_dict = {}

    for ticker in tickers:
        if ticker not in train_data_dict or ticker not in val_data_dict: continue

        train_df = train_data_dict[ticker].copy()
        val_df = val_data_dict[ticker].copy()
        train_scaled_dict[ticker] = pd.DataFrame(index=train_df.index) # Zachowaj indeks
        val_scaled_dict[ticker] = pd.DataFrame(index=val_df.index)

        for feature in features_to_normalize:
            if feature not in train_df.columns or feature not in val_df.columns:
                 logging.warning(f"Feature '{feature}' not found for ticker '{ticker}' during scaling. Skipping.")
                 continue

            scaler = StandardScaler()
            # --- Kluczowe: Fit tylko na danych treningowych ---
            try:
                 scaler.fit(train_df[[feature]])
                 # --- Transformuj oba zbiory TYM SAMYM scalerem ---
                 train_scaled_values = scaler.transform(train_df[[feature]])
                 val_scaled_values = scaler.transform(val_df[[feature]])

                 train_scaled_dict[ticker][feature] = train_scaled_values.flatten()
                 val_scaled_dict[ticker][feature] = val_scaled_values.flatten()
                 scalers[ticker][feature] = scaler # Zapisz dopasowany scaler
            except Exception as e:
                 logging.error(f"Error scaling feature '{feature}' for ticker '{ticker}': {e}. Leaving unscaled.")
                 # W razie błędu, użyj oryginalnych danych (lub wypełnij NaN/0)
                 train_scaled_dict[ticker][feature] = train_df[feature].values
                 val_scaled_dict[ticker][feature] = val_df[feature].values


    return train_scaled_dict, val_scaled_dict, scalers


# Datasets for multi-ticker
class MultiTickerDataset(Dataset):
    """Dataset for data in format (Batch, Time, Stocks*Features)"""
    def __init__(self, sequences, targets):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class MultiStockDataset(Dataset):
    """Dataset for data in format (Batch, Time, Stocks, Features)"""

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
