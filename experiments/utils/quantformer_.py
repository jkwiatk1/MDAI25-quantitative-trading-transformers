import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import QuantileTransformer


# Strategy Implementation
def trading_strategy(model, data_loader, device, cash, tickers, n_quantiles=3):
    stock_pool = []
    df_predictions = pd.DataFrame(columns=tickers)
    model.eval()
    with torch.no_grad():
        for sequences, _ in data_loader:
            sequences = sequences.to(device)
            predictions = model(sequences)
            predictions_quantized = quantize_labels(
                predictions, n_quantiles=n_quantiles
            )
            # ranked_stocks = torch.argsort(predictions_quantized, dim=-1, descending=True)

            temp_df = pd.DataFrame(predictions_quantized.numpy(), columns=tickers)
            df_predictions = pd.concat([df_predictions, temp_df], ignore_index=True)

            # for batch_idx in range(df_predictions.size(0)):
            #     n_stocks_to_buy = cash // len(df_predictions[batch_idx])
            #     best_stocks = ranked_stocks[batch_idx, :n_stocks_to_buy]
            #     stock_pool.extend([tickers[i.item()] for i in best_stocks])
            # stock_pool.extend(df_predictions)
    return df_predictions  # stock_pool


def quantize_labels(labels, n_quantiles=3):
    """
    Quantize returns into discrete classes based on quantiles.

    Parameters:
    labels: torch.Tensor or np.ndarray
        Tensor (2D) or array-like (1D/2D) of continuous values to be quantized.
        If 2D, rows are samples, and columns are features (e.g., tickers).
    n_quantiles: int
        Number of quantiles to divide the data into (e.g., 3 for low, middle, high).

    Returns:
    torch.Tensor
        Tensor of discrete class labels for each input value.
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()  # Convert to numpy array for QuantileTransformer

    if labels.ndim == 1:  # Handle single feature (1D array)
        labels = labels.reshape(-1, 1)

    if labels.ndim == 2:  # Handle batch (2D array)
        quantized_labels = np.zeros_like(labels, dtype=int)
        for i in range(labels.shape[0]):  # Iterate over columns (tickers)
            # Fit QuantileTransformer for each column
            quantizer = QuantileTransformer(
                n_quantiles=n_quantiles, output_distribution="uniform"
            )
            normalized_labels = quantizer.fit_transform(labels[i, :].reshape(-1, 1))

            # Define quantile bins and digitize
            bins = np.linspace(0, 1, n_quantiles + 1)[1:-1]
            quantized_labels[i, :] = np.digitize(normalized_labels.flatten(), bins=bins)
    else:
        raise ValueError("Input labels must be 1D or 2D.")

    return torch.tensor(quantized_labels)
