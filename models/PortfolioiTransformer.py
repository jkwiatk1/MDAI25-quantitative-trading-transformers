import math
import torch
from torch import nn
import logging


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding. Expects input shape [seq_len, batch, d_model]."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
        
        if torch.isnan(self.pe).any() or torch.isinf(self.pe).any():
            logging.error("NaN or Inf detected in PositionalEncoding initialization!")

    def forward(self, x):
        """
        Args:
            x: Tensor of shape [seq_len, batch, d_model]
        """
        seq_len = x.size(0)
        if seq_len > self.pe.size(0):
            logging.warning(f"Input sequence length ({seq_len}) exceeds max_len ({self.pe.size(0)})")
        
        x = x + self.pe[:seq_len]
        return self.dropout(x)


class SeriesEmbeddingMLP(nn.Module):
    """
    Embeds entire time series [T, F] into a single token [d_model] using MLP.
    Optionally includes multiple hidden layers and output layer normalization.
    """
    
    def __init__(
        self,
        lookback: int,
        num_features: int,
        d_model: int,
        dropout: float = 0.1,
        num_hidden_layers: int = 2,
        add_output_norm: bool = True
    ):
        super().__init__()
        self.lookback = lookback
        self.num_features = num_features
        self.d_model = d_model
        input_dim = lookback * num_features

        layers = []
        current_dim = input_dim

        # Build hidden layers with decreasing dimensions
        for i in range(num_hidden_layers):
            hidden_dim = max(current_dim // 2, d_model, 16)
            if hidden_dim < d_model:
                hidden_dim = (current_dim + d_model) // 2
            
            layers.append(nn.Linear(current_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim

        # Output layer
        layers.append(nn.Linear(current_dim, d_model))
        
        if add_output_norm:
            layers.append(nn.LayerNorm(d_model))
            logging.debug("Added LayerNorm to SeriesEmbeddingMLP output")

        self.mlp = nn.Sequential(*layers)
        logging.debug(f"SeriesEmbeddingMLP: {num_hidden_layers} hidden layers, output_norm={add_output_norm}")

    def forward(self, x_series_batch):
        """
        Args:
            x_series_batch: Input of shape [batch, time, features]
        
        Returns:
            Embedded tokens of shape [batch, d_model]
        """
        B, T, F = x_series_batch.shape
        if T != self.lookback or F != self.num_features:
            raise ValueError(f"Input shape mismatch: expected [*, {self.lookback}, {self.num_features}], got [*, {T}, {F}]")

        x_flat = x_series_batch.view(B, T * F)
        return self.mlp(x_flat)


class PortfolioITransformer(nn.Module):
    """
    iTransformer for portfolio selection. Treats stocks as tokens and applies
    attention across stocks. Time series for each stock is embedded via MLP.
    
    Input: [Batch, Time, Stocks, Features]
    Output: [Batch, Stocks, 1] (predicted returns)
    """

    def __init__(
        self,
        stock_amount: int,
        financial_features_amount: int,
        lookback: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        num_encoder_layers: int,
        activation: str = "gelu",
        norm_first: bool = True,
        use_embedding_dropout: bool = True,
        mlp_hidden_layers: int = 2,
        mlp_add_output_norm: bool = True
    ):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model must be divisible by n_heads")
        
        self.stock_amount = stock_amount
        self.d_model = d_model

        self.series_embedding = SeriesEmbeddingMLP(
            lookback, financial_features_amount, d_model, dropout,
            num_hidden_layers=mlp_hidden_layers,
            add_output_norm=mlp_add_output_norm
        )
        self.embedding_dropout = nn.Dropout(dropout) if use_embedding_dropout else nn.Identity()
        self.stock_pos_encoder = PositionalEncoding(d_model, dropout, max_len=stock_amount)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation=activation,
            batch_first=False,
            norm_first=norm_first
        )
        encoder_norm = nn.LayerNorm(d_model) if not norm_first else None
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.projection_head = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch, time, stocks, features]
        
        Returns:
            Predicted returns of shape [batch, stocks, 1]
        """
        B, T, N, F = x.shape

        # Reshape and embed each stock's time series: [B, T, N, F] -> [B*N, T, F] -> [B*N, D]
        x_permuted = x.permute(0, 2, 1, 3).contiguous()
        x_for_embedding = x_permuted.view(B * N, T, F)

        # 2. Apply Series Embedding MLP: [B*N, T, F] -> [B*N, D]
        variate_tokens = self.series_embedding(x_for_embedding)
        variate_tokens = self.embedding_dropout(variate_tokens)

        # Prepare for transformer: [B*N, D] -> [N, B, D]
        encoder_input = variate_tokens.view(B, N, self.d_model).permute(1, 0, 2).contiguous()
        encoder_input = self.stock_pos_encoder(encoder_input)

        # Apply transformer encoder (attention across stocks)
        encoded_variates = self.transformer_encoder(encoder_input)

        # Project to predictions: [N, B, D] -> [N, B, 1] -> [B, N, 1]
        predictions = self.projection_head(encoded_variates)
        return predictions.permute(1, 0, 2).contiguous()


def build_PortfolioITransformer(
    stock_amount: int,
    financial_features_amount: int,
    lookback: int,
    d_model: int = 64,
    n_heads: int = 4,
    d_ff: int = 128,
    dropout: float = 0.1,
    num_encoder_layers: int = 2,
    activation: str = "gelu",
    norm_first: bool = True,
    use_embedding_dropout: bool = True,
    mlp_hidden_layers: int = 2,
    mlp_add_output_norm: bool = True,
    device: torch.device = torch.device("cpu"),
) -> PortfolioITransformer:
    """
    Build iTransformer model for portfolio optimization.

    Args:
        stock_amount: Number of stocks
        financial_features_amount: Number of features per stock
        lookback: Time series lookback window
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        num_encoder_layers: Number of encoder layers
        activation: Activation function ('relu' or 'gelu')
        norm_first: Use pre-norm instead of post-norm
        use_embedding_dropout: Apply dropout after embedding
        mlp_hidden_layers: Number of hidden layers in series embedding MLP
        mlp_add_output_norm: Add layer norm to MLP output
        device: Target device
    
    Returns:
        Configured model instance
    """
    if d_model % n_heads != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

    print(f"Building PortfolioITransformer: stocks={stock_amount}, features={financial_features_amount}, "
          f"lookback={lookback}, d_model={d_model}, n_heads={n_heads}, layers={num_encoder_layers}, "
          f"mlp_layers={mlp_hidden_layers}")

    model = PortfolioITransformer(
        stock_amount=stock_amount,
        financial_features_amount=financial_features_amount,
        lookback=lookback,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        num_encoder_layers=num_encoder_layers,
        activation=activation,
        norm_first=norm_first,
        use_embedding_dropout=use_embedding_dropout,
        mlp_hidden_layers=mlp_hidden_layers,
        mlp_add_output_norm=mlp_add_output_norm
    )
    return model.to(device)