import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer models."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """Args: x of shape [seq_len, batch, d_model]"""
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class EncoderLayerCA(nn.Module):
    """
    Encoder layer with temporal self-attention, spatial cross-attention, and feed-forward.
    Uses post-layer normalization.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=False
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=False
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch, time, stocks, d_model]
        
        Returns:
            Output tensor of shape [batch, time, stocks, d_model]
        """
        B, T, N, D = x.shape

        # --- 1. Temporal Self-Attention ---
        residual1 = x
        # Reshape: [B, T, N, D] -> [T, B*N, D]
        x_temp = x.permute(1, 0, 2, 3).contiguous().view(T, B * N, D)
        attn_temp_out, _ = self.temporal_attn(
            x_temp, x_temp, x_temp, need_weights=False
        )  # Q=K=V
        # Reshape back: [T, B*N, D] -> [B, T, N, D]
        attn_temp_out = attn_temp_out.view(T, B, N, D).permute(1, 0, 2, 3).contiguous()
        x = self.norm1(residual1 + self.dropout1(attn_temp_out))

        # --- 2. Spatial Cross-Attention ---
        residual2 = x
        # Reshape: [B, T, N, D] -> [N, B*T, D]
        x_spat = x.permute(2, 0, 1, 3).contiguous().view(N, B * T, D)
        attn_spat_out, _ = self.spatial_attn(
            x_spat, x_spat, x_spat, need_weights=False
        )  # Q=K=V
        # Reshape back: [N, B*T, D] -> [B, T, N, D]
        attn_spat_out = attn_spat_out.view(N, B, T, D).permute(1, 2, 0, 3).contiguous()
        x = self.norm2(residual2 + self.dropout2(attn_spat_out))

        # --- 3. Feed Forward Network ---
        residual3 = x
        ffn_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
        x = self.norm3(residual3 + self.dropout3(ffn_out))

        return x


class PortfolioTransformerCA(nn.Module):
    """
    Transformer with cross-attention for portfolio selection.
    Combines temporal and spatial attention mechanisms.
    
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
    ):
        super().__init__()
        self.stock_amount = stock_amount
        self.financial_features_amount = financial_features_amount
        self.lookback = lookback
        self.d_model = d_model

        self.feature_proj = nn.Linear(financial_features_amount, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=lookback)
        self.input_dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.ModuleList([
            EncoderLayerCA(d_model, n_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])

        self.projection_head = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape [batch, time, stocks, features]
        
        Returns:
            Predicted returns of shape [batch, stocks, 1]
        """
        B, T, N, F = x.shape
        assert T == self.lookback and N == self.stock_amount and F == self.financial_features_amount

        # 1. Project features & Add Positional Encoding
        x = self.feature_proj(x)  # [B, T, N, D]
        # Reshape: [B, T, N, D] -> [T, B*N, D]
        x_enc_in = x.permute(1, 0, 2, 3).contiguous().view(T, B * N, self.d_model)

        # 2. Add Positional Encoding
        x_enc_in = self.pos_encoder(x_enc_in)
        # TODO there should be passed x_enc_in instead of x
        # Pass through cross-attention encoder layers
        for layer in self.encoder_layers:
            x = layer(x)

        # Use last time step for prediction
        x_last_t = x[:, -1, :, :]
        predictions = self.projection_head(x_last_t)

        return predictions


def build_TransformerCA(
    stock_amount: int,
    financial_features_amount: int,
    lookback: int,
    d_model: int = 128,
    n_heads: int = 4,
    d_ff: int = 256,
    dropout: float = 0.1,
    num_encoder_layers: int = 2,
    device: torch.device = torch.device("cpu"),
) -> PortfolioTransformerCA:
    """
    Build TransformerCA model for portfolio optimization.

    Args:
        stock_amount: Number of stocks
        financial_features_amount: Number of features per stock
        lookback: Time series lookback window
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        num_encoder_layers: Number of encoder layers
        device: Target device
    
    Returns:
        Configured model instance
    """
    if d_model % n_heads != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

    print(f"Building PortfolioTransformerCA: stocks={stock_amount}, features={financial_features_amount}, "
          f"lookback={lookback}, d_model={d_model}, n_heads={n_heads}, layers={num_encoder_layers}")

    model = PortfolioTransformerCA(
        stock_amount=stock_amount,
        financial_features_amount=financial_features_amount,
        lookback=lookback,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        num_encoder_layers=num_encoder_layers,
    )
    return model.to(device)
