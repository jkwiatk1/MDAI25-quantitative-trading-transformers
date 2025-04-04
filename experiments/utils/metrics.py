import logging

import torch
import torch.nn.functional as F
import numpy as np
from torch import nn as nn
from scipy.stats import spearmanr, pearsonr


class RankLoss(nn.Module):
    """
    Pairwise Ranking Loss for portfolio optimization.
    Combines MSE loss with a pairwise ranking term.
    """
    def __init__(self, lambda_rank=0.5, reduction='mean'):
        """
        Args:
            lambda_rank (float): Weighting factor for the ranking loss term.
            reduction (str): Specifies the reduction to apply to the output:
                             'none' | 'mean' | 'sum'. Default: 'mean'
        """
        super(RankLoss, self).__init__()
        if not (0 <= lambda_rank <= 1):
             logging.warning(f"lambda_rank ({lambda_rank}) is outside the typical [0, 1] range.")
        self.lambda_rank = lambda_rank
        self.reduction = reduction
        logging.info(f"Initialized RankLoss with lambda_rank={lambda_rank} and reduction='{reduction}'")

    def forward(self, predicted_returns, true_returns):
        """
        Computes the loss value.

        Args:
            predicted_returns (torch.Tensor): Tensor of predicted returns, shape [B, N, 1] or [B, N].
            true_returns (torch.Tensor): Tensor of actual returns, shape [B, N, 1] or [B, N].

        Returns:
            torch.Tensor: Computed loss value (scalar if reduction is 'mean' or 'sum').
        """
        # Ensure shapes are [B, N] before proceeding
        if predicted_returns.shape[-1] == 1:
            predicted_returns = predicted_returns.squeeze(-1)
        if true_returns.shape[-1] == 1:
            true_returns = true_returns.squeeze(-1)

        if predicted_returns.shape != true_returns.shape:
             raise ValueError(f"Shape mismatch after squeeze: preds {predicted_returns.shape}, true {true_returns.shape}")
        if predicted_returns.dim() != 2: # should be [B, N]
             raise ValueError(f"Expected 2D tensors (Batch, Stocks), but got shape {predicted_returns.shape}")

        B, N = predicted_returns.shape

        # --- 1. MSE Loss ---
        # Calculate MSE per sample if reduction is needed later
        mse_loss_per_sample = F.mse_loss(predicted_returns, true_returns, reduction='none').mean(dim=1) # [B]

        # --- 2. Rank Loss - Pairwise Ranking ---
        # [B, N, 1] - [B, 1, N] -> [B, N, N]
        pred_pairwise_diff = predicted_returns.unsqueeze(2) - predicted_returns.unsqueeze(1)
        true_pairwise_diff = true_returns.unsqueeze(2) - true_returns.unsqueeze(1)

        # Calculate sign of true differences. Handle zeros: sign(0) = 0
        sign_true_diff = torch.sign(true_pairwise_diff)

        # Calculate the ranking loss term: relu(-pred_diff * sign_true_diff)
        # This penalizes pairs where sign(pred_diff) != sign(true_diff)
        # Note: - a * sign(b) is positive only if a and b have different signs.
        rank_term = F.relu(-pred_pairwise_diff * sign_true_diff)

        # Sum the rank term over all pairs (N*N) for each sample in the batch
        # Avoid summing diagonal (i==j) where diff is always 0? Not strictly necessary with sign(0)=0.
        # Sum over last two dimensions (pairs i, j)
        rank_loss_per_sample = torch.sum(rank_term, dim=(1, 2)) # [B]

        # Normalize rank loss by the number of pairs? (N*N or N*(N-1)) - Optional but often helpful
        # Normalizing helps make lambda_rank less sensitive to the number of stocks (N)
        num_pairs = N * (N - 1) # Number of off-diagonal pairs
        if num_pairs > 0:
             rank_loss_per_sample = rank_loss_per_sample / num_pairs
        # else: handle N=1 case if necessary (rank loss is 0)

        # --- 3. Combine Losses ---
        # Combine MSE and Rank loss for each sample
        combined_loss_per_sample = mse_loss_per_sample + self.lambda_rank * rank_loss_per_sample # [B]

        # --- 4. Apply Reduction ---
        if self.reduction == 'mean':
            final_loss = torch.mean(combined_loss_per_sample)
        elif self.reduction == 'sum':
            final_loss = torch.sum(combined_loss_per_sample)
        elif self.reduction == 'none':
            final_loss = combined_loss_per_sample
        else:
            raise ValueError(f"Invalid reduction type: {self.reduction}")

        # --- DEBUGGING ---
        # if torch.isnan(final_loss) or torch.isinf(final_loss): logging.error("NaN or Inf detected
        # in RankLoss!") logging.error(f"MSE mean: {torch.mean(mse_loss_per_sample).item()}, Rank mean: {torch.mean(
        # rank_loss_per_sample).item()}")

        return final_loss

class WeightedMAELoss(nn.Module):
    def __init__(self, num_outputs):
        super(WeightedMAELoss, self).__init__()
        self.num_outputs = num_outputs
        self.weights = torch.ones(num_outputs, requires_grad=False)

    def forward(self, predictions, targets):
        # dynamic update based on error
        errors = torch.abs(predictions - targets)
        self.weights = errors / errors.sum()
        losses = self.weights * errors
        return losses.mean()


def select_portfolio(predicted_returns, threshold=0.02):
    """
    Selects stocks for the portfolio based on predicted returns.

    :param predicted_returns: tensor of predicted returns
    :param threshold: selection threshold
    :return: indices of selected stocks
    """
    return (predicted_returns > threshold).nonzero(as_tuple=True)[0]


def portfolio_performance(selected_returns):

    """
    Computes portfolio performance metrics.

    :param selected_returns: tensor of returns for selected stocks
    :return: dictionary with performance metrics
    """
    mean_return = torch.mean(selected_returns)
    std_return = torch.std(selected_returns)
    sharpe_ratio = mean_return / (std_return + 1e-6)

    return {
        'mean_return': mean_return.item(),
        'std_dev': std_return.item(),
        'sharpe_ratio': sharpe_ratio.item()
    }


def compute_portfolio_metrics(predictions, targets, tickers, top_k=5, risk_free_rate=0.02):
    """
    Compute portfolio performance metrics.
    :param predictions: Predicted returns [num_samples, num_stocks]
    :param targets: Actual returns [num_samples, num_stocks]
    :param tickers: List of stock tickers in correct order
    :param top_k: Number of stocks to include in portfolio
    :param risk_free_rate: Annualized risk-free rate (default 2%)
    :return: Dictionary of portfolio metrics
    """

    # Select top-k stocks based on predicted returns
    top_indices = np.argsort(-predictions, axis=1)[:, :top_k]  # Indices of top stocks per sample

    # Compute portfolio returns (actual returns of selected stocks)
    selected_returns = np.take_along_axis(targets, top_indices, axis=1)
    # portfolio_returns = np.sum(selected_returns, axis=1)  # Sum across selected stocks
    portfolio_returns = np.mean(selected_returns, axis=1)
    # Compute IRR (Cumulative Investment Return)
    irr = np.sum(selected_returns / (1 + selected_returns), axis=1)  # Sum across selected stocks per timestep

    #  Compute cumulative return
    cumulative_returns = np.cumsum(portfolio_returns)  # Cumulative return
    cumulative_return = portfolio_returns.sum()  # Final cumulative return

    # Compute annualized return (assuming 252 trading days per year)
    annualized_return = (1 + cumulative_return) ** (252 / len(portfolio_returns)) - 1

    # Compute Sharpe Ratio
    daily_risk_free = risk_free_rate / 252  # Convert annualized to daily
    excess_returns = portfolio_returns - daily_risk_free
    sharpe_ratio = np.mean(excess_returns) / (np.std(excess_returns) + 1e-6)  # Avoid division by zero

    # Compute Maximum Drawdown
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = np.min(drawdown)

    return {
        "IRR": np.mean(irr),  # Return average IRR
        "Cumulative Return": cumulative_return,
        "Annualized Return": annualized_return,
        "Sharpe Ratio": sharpe_ratio,
        "Maximum Drawdown": max_drawdown,
    }, cumulative_returns


def calculate_portfolio_performance(predictions, targets, top_k=5, risk_free_rate=0.0, trading_days_per_year=252):
    """
    Calculates portfolio performance metrics based on a daily rebalanced top-K strategy
    with equal weighting. Uses geometric compounding for cumulative returns.

    Args:
        predictions (np.ndarray): Predicted returns [num_samples/days, num_stocks].
        targets (np.ndarray): Actual realized returns [num_samples/days, num_stocks].
        top_k (int): Number of stocks with highest predicted returns to select daily.
        risk_free_rate (float): Annualized risk-free rate (e.g., 0.02 for 2%). Default is 0.0.
        trading_days_per_year (int): Number of trading days assumed in a year for annualization.

    Returns:
        tuple:
            - dict: Dictionary containing portfolio performance metrics:
                    'Cumulative Return (%)', 'Annualized Return (%)',
                    'Annualized Volatility (%)', 'Annualized Sharpe Ratio',
                    'Maximum Drawdown (%)'
            - np.ndarray: Array representing the daily cumulative portfolio value (starting from 1.0).
    """
    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must have the same shape.")
    if predictions.ndim != 2:
        raise ValueError("Inputs must be 2D arrays [days, stocks].")

    num_days, num_stocks = predictions.shape
    if top_k > num_stocks:
        logging.warning(f"top_k ({top_k}) is greater than the number of stocks ({num_stocks}). Using top_k = {num_stocks}.")
        top_k = num_stocks
    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    # 1. Select Top-K Stocks Daily based on Predictions
    # Indices of top-k stocks for each day (row)
    # Using argpartition is potentially faster than argsort for just finding top K
    # top_indices = np.argsort(-predictions, axis=1)[:, :top_k]
    # Ensure we get indices of largest values, partition is faster
    top_indices = np.argpartition(-predictions, kth=top_k-1, axis=1)[:, :top_k]

    # 2. Get Actual Returns of Selected Stocks
    # Use advanced indexing to select returns corresponding to top_indices
    selected_actual_returns = np.take_along_axis(targets, top_indices, axis=1)

    # 3. Calculate Daily Portfolio Returns (Equal Weighting)
    # Handle cases where a row in selected_actual_returns might contain NaNs if target data has NaNs
    # We take the mean ignoring NaNs for that day's portfolio return
    daily_portfolio_returns = np.nanmean(selected_actual_returns, axis=1)

    # Check for days where all selected returns were NaN (would result in NaN portfolio return)
    nan_days = np.isnan(daily_portfolio_returns)
    if np.any(nan_days):
        num_nan_days = np.sum(nan_days)
        logging.warning(f"Portfolio return calculation resulted in NaN for {num_nan_days} out of {num_days} days. "
                        f"This might happen if all selected stocks had NaN targets on those days. "
                        f"These days will be excluded from metric calculations.")
        # Filter out NaN days for subsequent calculations
        valid_mask = ~nan_days
        daily_portfolio_returns = daily_portfolio_returns[valid_mask]
        num_days = len(daily_portfolio_returns) # Update num_days based on valid returns
        if num_days == 0:
            logging.error("No valid daily portfolio returns available after filtering NaNs. Cannot compute metrics.")
            # Return default/NaN values
            nan_metrics = {
                'Cumulative Return (%)': np.nan, 'Annualized Return (%)': np.nan,
                'Annualized Volatility (%)': np.nan, 'Annualized Sharpe Ratio': np.nan,
                'Maximum Drawdown (%)': np.nan
            }
            return nan_metrics, np.array([1.0]) # Return initial portfolio value

    # 4. Calculate Cumulative Portfolio Value (Geometric Compounding)
    # Start with value 1.0 before the first day
    portfolio_value_curve = np.concatenate(([1.0], np.cumprod(1 + daily_portfolio_returns)))

    # 5. Calculate Metrics
    # Cumulative Return (Total Return over the period)
    final_portfolio_value = portfolio_value_curve[-1]
    cumulative_return_total = final_portfolio_value - 1.0

    # Annualized Return
    num_years = num_days / trading_days_per_year
    # Avoid issues with very short periods or zero return
    if num_years == 0 or final_portfolio_value <= 0:
         annualized_return = 0.0
    else:
        annualized_return = (final_portfolio_value ** (1.0 / num_years)) - 1.0


    # Annualized Volatility
    annualized_volatility = np.std(daily_portfolio_returns) * np.sqrt(trading_days_per_year)

    # Annualized Sharpe Ratio
    daily_risk_free_rate = (1 + risk_free_rate)**(1/trading_days_per_year) - 1 # More precise daily rate
    excess_returns = daily_portfolio_returns - daily_risk_free_rate
    mean_excess_return = np.mean(excess_returns)
    std_dev_excess_return = np.std(excess_returns) # Use std dev of excess returns or portfolio returns? Usually portfolio returns. Let's use std(daily_portfolio_returns)
    std_dev_portfolio_return = np.std(daily_portfolio_returns)

    # Avoid division by zero
    if std_dev_portfolio_return < 1e-8:
        annualized_sharpe_ratio = 0.0 if mean_excess_return > -1e-8 else np.nan # Or 0 if mean return is also near 0
    else:
        # SR uses annualized returns and vol
        annualized_sharpe_ratio = (annualized_return - risk_free_rate) / (annualized_volatility + 1e-8)
        # Alternatively, based on daily returns: (mean_excess_return / (std_dev_portfolio_return + 1e-8)) * np.sqrt(trading_days_per_year)
        # Let's stick to the definition using annualized values for clarity

    # Maximum Drawdown (MDD)
    peak_values = np.maximum.accumulate(portfolio_value_curve)
    drawdowns = (portfolio_value_curve - peak_values) / (peak_values + 1e-8) # Add epsilon to avoid division by zero if peak is 0 (unlikely starting from 1)
    max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0.0


    # Prepare results dictionary, reporting returns and MDD in percentages
    results = {
        'Cumulative Return (%)': cumulative_return_total * 100,
        'Annualized Return (%)': annualized_return * 100,
        'Annualized Volatility (%)': annualized_volatility * 100,
        'Annualized Sharpe Ratio': annualized_sharpe_ratio,
        'Maximum Drawdown (%)': max_drawdown * 100,
    }

    # Log calculated metrics
    logging.info("--- Portfolio Performance Metrics ---")
    for key, value in results.items():
        logging.info(f"{key}: {value:.4f}")

    return results, portfolio_value_curve


def calculate_predictive_quality(predictions, targets):
    """
    Calculates Information Coefficient (IC - Spearman and Pearson) and ICIR.
    Assumes predictions and targets are already aligned and potentially scaled.

    Args:
        predictions (np.ndarray): Predicted returns [num_samples/days, num_stocks].
        targets (np.ndarray): Actual realized returns [num_samples/days, num_stocks].

    Returns:
        dict: Dictionary containing 'IC (Spearman)', 'ICIR (Spearman)',
                'IC (Pearson)', 'ICIR (Pearson)'.
    """
    if predictions.shape != targets.shape:
        raise ValueError("Predictions and targets must have the same shape.")
    if predictions.ndim != 2:
        raise ValueError("Inputs must be 2D arrays [days, stocks].")

    num_samples, num_stocks = predictions.shape
    daily_ics_spearman = []
    daily_ics_pearson = []

    if num_stocks < 2:
        logging.warning("Cannot calculate correlation with less than 2 stocks.")
        metrics = {
            "IC (Spearman)": np.nan, "ICIR (Spearman)": np.nan,
            "IC (Pearson)": np.nan, "ICIR (Pearson)": np.nan
        }
        return metrics

    for i in range(num_samples):
        preds_day = predictions[i, :]
        targets_day = targets[i, :]

        # Remove NaNs for correlation calculation for this day
        valid_mask = ~np.isnan(preds_day) & ~np.isnan(targets_day)
        valid_count = np.sum(valid_mask)

        if valid_count < 2: # Need at least 2 valid pairs
            daily_ics_spearman.append(np.nan)
            daily_ics_pearson.append(np.nan)
            continue

        preds_valid = preds_day[valid_mask]
        targets_valid = targets_day[valid_mask]

        # Check for zero standard deviation in valid slice (can cause NaN in Pearson)
        if np.std(preds_valid) < 1e-8 or np.std(targets_valid) < 1e-8:
             pearson_corr = np.nan # Cannot compute Pearson if one variable is constant
        else:
             try:
                 pearson_corr, _ = pearsonr(preds_valid, targets_valid)
                 # Handle potential NaN result from pearsonr itself in rare cases
                 if np.isnan(pearson_corr):
                     pearson_corr = np.nan
             except ValueError:
                 pearson_corr = np.nan

        # Spearman should be more robust to constant values but check anyway
        try:
            spearman_corr, _ = spearmanr(preds_valid, targets_valid)
            # Handle potential NaN result from spearmanr itself (e.g., all values identical)
            if np.isnan(spearman_corr):
                spearman_corr = np.nan
        except ValueError:
            spearman_corr = np.nan

        daily_ics_spearman.append(spearman_corr)
        daily_ics_pearson.append(pearson_corr)


    # Calculate Mean IC and ICIR, ignoring NaNs from individual days
    def calculate_ic_stats(daily_ics_list):
        daily_ics_array = np.array(daily_ics_list)
        valid_ics = daily_ics_array[~np.isnan(daily_ics_array)]
        if len(valid_ics) == 0:
            mean_ic = np.nan
            std_ic = np.nan
            icir = np.nan
        else:
            mean_ic = np.mean(valid_ics)
            std_ic = np.std(valid_ics)
            icir = mean_ic / (std_ic + 1e-8) if std_ic > 1e-8 else np.nan # Check std_dev > 0
        return mean_ic, icir

    mean_ic_s, icir_s = calculate_ic_stats(daily_ics_spearman)
    mean_ic_p, icir_p = calculate_ic_stats(daily_ics_pearson)

    metrics = {
        "IC (Spearman)": mean_ic_s, "ICIR (Spearman)": icir_s,
        "IC (Pearson)": mean_ic_p, "ICIR (Pearson)": icir_p
    }

    logging.info("--- Predictive Quality Metrics ---")
    for key, value in metrics.items():
        logging.info(f"{key}: {value:.4f}")

    return metrics


def calculate_precision_at_k(predictions, targets, top_k):
    """Calculates Precision@k: % of top-k stocks with positive actual return."""
    num_samples, num_stocks = predictions.shape
    precisions = []
    if top_k > num_stocks:
        logging.warning(f"top_k ({top_k}) > num_stocks ({num_stocks}). Using k={num_stocks}")
        top_k = num_stocks
    if top_k <= 0:
        logging.error("top_k must be positive for Precision@k.")
        return np.nan

    # Ensure targets have the same shape for indexing
    if targets.shape != predictions.shape:
         raise ValueError(f"Shape mismatch: Predictions {predictions.shape}, Targets {targets.shape}")

    # Get indices of top-k predictions for each day
    # Using argpartition might be slightly faster if k << num_stocks
    # top_indices = np.argsort(-predictions, axis=1)[:, :top_k]
    top_indices = np.argpartition(-predictions, kth=top_k - 1, axis=1)[:, :top_k]

    # Get the actual returns for the selected top-k stocks
    selected_actual_returns = np.take_along_axis(targets, top_indices, axis=1)

    # Check where actual return is positive (> 0) for selected stocks
    # Handle potential NaNs in targets - consider them as non-positive? Or ignore?
    # Let's treat NaN as not positive.
    positive_returns_mask = np.nan_to_num(selected_actual_returns, nan=-1.0) > 0

    # Count positive returns per day and divide by k
    daily_precision = np.sum(positive_returns_mask, axis=1) / top_k

    # Average precision over all days
    mean_precision = np.mean(daily_precision)

    logging.info(f"Predictive Quality: Precision@{top_k}: {mean_precision * 100:.4f}%")
    return mean_precision