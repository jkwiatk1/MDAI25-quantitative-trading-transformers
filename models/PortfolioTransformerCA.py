import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Standard Positional Encoding using sin/cos."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """x: [seq_len, batch_eff, d_model]"""
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class EncoderLayerCA(nn.Module):
    """
    Performs Temporal Self-Attention, Spatial Cross-Attention, and Feed-Forward.
    Uses Post-LN normalization style.
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super().__init__()
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=False
        )  # Oczekuje [T, B_eff, D]
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads, dropout=dropout, batch_first=False
        )  # Oczekuje [N, B_eff, D]
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = nn.ReLU()  # nn.GELU()

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
        # Reshape: [B, T, N, D] -> [T, B*N, D]
        x_temp = x.permute(1, 0, 2, 3).contiguous().view(T, B * N, D)
        attn_temp_out, _ = self.temporal_attn(
            x_temp, x_temp, x_temp, need_weights=False
        )  # Q=K=V
        # Reshape back: [T, B*N, D] -> [B, T, N, D]
        attn_temp_out = attn_temp_out.view(T, B, N, D).permute(1, 0, 2, 3).contiguous()
        # Post-LN Residual Connection 1
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
        # Post-LN Residual Connection 2
        x = self.norm2(residual2 + self.dropout2(attn_spat_out))

        # --- 3. Feed Forward Network ---
        residual3 = x
        ffn_out = self.linear2(self.dropout(self.activation(self.linear1(x))))
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

        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayerCA(d_model, n_heads, d_ff, dropout)
                for _ in range(num_encoder_layers)
            ]
        )

        self.projection_head = nn.Linear(d_model, 1)
        # self.projection_head = nn.Sequential(
        #     # Optional LayerNorm before final projection
        #     # LayerNormalization(d_model),
        #     nn.Linear(d_model, d_ff // 2),
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(d_ff // 2, 1)
        # )

    def forward(self, x):
        """
        Args:
            x: Input tensor, shape [B, T, N, F]
        Returns:
            Output tensor, shape [B, N, 1] (predicted returns)
        """
        B, T, N, F = x.shape
        assert (
            T == self.lookback
            and N == self.stock_amount
            and F == self.financial_features_amount
        )

        # 1. Project features & Add Positional Encoding + Dropout
        x = self.feature_proj(x)  # [B, T, N, D]
        # Reshape: [B, T, N, D] -> [T, B*N, D]
        x_enc_in = x.permute(1, 0, 2, 3).contiguous().view(T, B * N, self.d_model)

        # 2. Add Positional Encoding + Dropout
        x_enc_in = self.pos_encoder(x_enc_in)

        # 3. Pass through Encoder Layers
        current_x_4d = x
        for layer in self.encoder_layers:
            current_x_4d = layer(current_x_4d)
        x = current_x_4d  # [B, T, N, D]

        # 4. Temporal Aggregation (Last time step)
        x_last_t = x[:, -1, :, :]  # Shape: [B, N, D]

        # 5. Final Projection
        predictions = self.projection_head(x_last_t)  # Shape: [B, N, 1]

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
    """Factory function to build the PortfolioTransformerCA model."""
    print("-" * 30)
    print("Building PortfolioTransformerCA (Modular) with parameters:")
    print(
        f"  Data: stocks={stock_amount}, features={financial_features_amount}, lookback={lookback}"
    )
    print(
        f"  Arch: d_model={d_model}, n_heads={n_heads}, d_ff={d_ff}, layers={num_encoder_layers}"
    )
    print(f"  Dropout: {dropout}")
    print(f"  Device: {device}")
    print("-" * 30)

    if d_model % n_heads != 0:
        raise ValueError(
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

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


if __name__ == "__main__":
    batch_size = 64
    lookback = 20
    stock_amount = 10
    features = 5
    example_d_model = 64
    example_n_heads = 4
    example_d_ff = 256
    example_dropout = 0.1
    example_num_layers = 2
    example_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_TransformerCA(
        stock_amount=stock_amount,
        financial_features_amount=features,
        lookback=lookback,
        d_model=example_d_model,
        n_heads=example_n_heads,
        d_ff=example_d_ff,
        dropout=example_dropout,
        num_encoder_layers=example_num_layers,
        device=example_device,
    )

    dummy_input = torch.randn(
        batch_size, lookback, stock_amount, features, device=example_device
    )
    print(f"\nTesting model with input shape: {dummy_input.shape}")

    try:
        model.eval()
        with torch.no_grad():
            output_returns = model(dummy_input)
        print(f"Model output shape: {output_returns.shape}")
        print("Model forward pass successful!")
    except Exception as e:
        print(f"Error during model forward pass: {e}")
        import traceback

        traceback.print_exc()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
