import logging

import torch
import torch.nn.functional as F
import numpy as np
from torch import nn as nn


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

# class RankLoss(nn.Module):
#     """
#     Pairwise Ranking Loss for portfolio optimization.
#     """
#     def __init__(self, lambda_rank=0.5):
#         super(RankLoss, self).__init__()
#         self.lambda_rank = lambda_rank
#
#     def forward(self, predicted_returns, true_returns):
#         """
#         Computes the loss value.
#
#         :param predicted_returns: tensor of predicted returns
#         :param true_returns: tensor of actual returns
#         :return: loss value
#         """
#         # Remove last dimension to ensure proper shape: [batch_size, stock_size]
#         predicted_returns = predicted_returns.squeeze(-1)
#         true_returns = true_returns.squeeze(-1)
#
#         # MSE Loss (Mean Squared Error)
#         mse_loss = F.mse_loss(predicted_returns, true_returns)
#
#         # Rank Loss - Pairwise Ranking Loss
#         pairwise_diff = predicted_returns.unsqueeze(2) - predicted_returns.unsqueeze(
#             1)  # Shape: [batch_size, stock_size, stock_size]
#         true_pairwise_diff = true_returns.unsqueeze(2) - true_returns.unsqueeze(
#             1)  # Shape: [batch_size, stock_size, stock_size]
#
#         rank_loss = torch.sum(F.relu(-pairwise_diff * torch.sign(true_pairwise_diff)))
#
#         return mse_loss + self.lambda_rank * rank_loss


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
    portfolio_returns = np.sum(selected_returns, axis=1)  # Sum across selected stocks

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
