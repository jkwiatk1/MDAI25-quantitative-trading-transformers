import math
import torch
from torch import nn

from models.modules import (
    LayerNormalization,
    FeedForwardBlock,
    MultiHeadAttentionBlock,
    PositionalEncoding
)


class EncoderLayerCA(nn.Module):
    """
    Single Encoder Layer performing Temporal Self-Attention, Spatial Cross-Attention,
    and Feed-Forward processing with Post-LN normalization.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.temporal_attn = MultiHeadAttentionBlock(d_model, n_heads, dropout)
        self.norm1 = LayerNormalization(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.spatial_attn = MultiHeadAttentionBlock(d_model, n_heads, dropout)
        self.norm2 = LayerNormalization(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.ffn = FeedForwardBlock(d_model, d_ff, dropout)
        self.norm3 = LayerNormalization(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x):
        """
        Args:
            x: Input tensor, shape [B, T, N, D]
        Returns:
            Output tensor, shape [B, T, N, D]
        """
        B, T, N, D = x.shape

        # --- 1. Temporal Self-Attention ---
        residual1 = x
        # Reshape for temporal attention: [B*N, T, D]
        x_temp = x.permute(0, 2, 1, 3).contiguous().view(B * N, T, D)
        attn_temp_out = self.temporal_attn(x_temp, x_temp, x_temp, mask=None)  # Q=K=V
        # Reshape back: [B, T, N, D]
        attn_temp_out = attn_temp_out.view(B, N, T, D).permute(0, 2, 1, 3).contiguous()
        # Post-LN Residual Connection 1
        x = self.norm1(residual1 + self.dropout1(attn_temp_out))

        # --- 2. Spatial Cross-Attention ---
        residual2 = x
        # Reshape for spatial attention: [B*T, N, D]
        x_spat = x.permute(0, 1, 3, 2).contiguous().view(B * T, D, N)  # -> [B*T, D, N]
        x_spat = x_spat.permute(0, 2, 1).contiguous()  # -> [B*T, N, D]
        attn_spat_out = self.spatial_attn(x_spat, x_spat, x_spat, mask=None)  # Q=K=V
        # Reshape back: [B, T, N, D]
        attn_spat_out = attn_spat_out.view(B, T, N, D)
        # Post-LN Residual Connection 2
        x = self.norm2(residual2 + self.dropout2(attn_spat_out))

        # --- 3. Feed Forward Network ---
        residual3 = x
        ffn_out = self.ffn(x)  # Applied element-wise on the last dimension
        # Post-LN Residual Connection 3
        x = self.norm3(residual3 + self.dropout3(ffn_out))

        return x


class PortfolioTransformerCA(nn.Module):
    """
    Transformer using stacked EncoderLayerCA for portfolio selection (predicting returns).
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

        self.projection_head = nn.Sequential(
            # Optional LayerNorm before final projection
            # LayerNormalization(d_model),
            nn.Linear(d_model, d_ff // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, 1)
        )

    def forward(self, x):
        """
        Args:
            x: Input tensor, shape [B, T, N, F]
        Returns:
            Output tensor, shape [B, N, 1] (predicted returns)
        """
        B, T, N, F = x.shape
        assert T == self.lookback and N == self.stock_amount and F == self.financial_features_amount

        # 1. Project features & Add Positional Encoding + Dropout
        x = self.feature_proj(x)
        x = x.permute(0, 2, 1, 3).contiguous().view(B * N, T, self.d_model)
        x = self.pos_encoder(x)
        x = self.input_dropout(x)
        x = x.view(B, N, T, self.d_model).permute(0, 2, 1, 3).contiguous()

        # 2. Pass through Encoder Layers
        for layer in self.encoder_layers:
            x = layer(x)  # Każda warstwa przyjmuje i zwraca [B, T, N, D]

        # 3. Temporal Aggregation (Last time step)
        x_last_t = x[:, -1, :, :]  # Shape: [B, N, D]

        # 4. Final Projection
        predictions = self.projection_head(x_last_t)  # Shape: [B, N, 1]

        return predictions



