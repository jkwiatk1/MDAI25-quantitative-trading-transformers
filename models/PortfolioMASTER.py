import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding. Expects input shape [seq_len, batch, d_model]."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        # Using register_buffer is correct for non-parameter tensors part of the model state
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """Add positional encoding. Args: x of shape [seq_len, batch, d_model]"""
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class AttentionBlockBase(nn.Module):
    """Base attention block with self-attention and feed-forward network."""

    def __init__(self, d_model, n_heads, d_ff, dropout, attention_impl):
        super().__init__()
        self.attn = attention_impl(d_model, n_heads, dropout=dropout, batch_first=False)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        """x: [Seq Len, Batch Eff, Dim]"""
        # --- Self-Attention with Pre-LN ---
        residual1 = x
        x_norm1 = self.norm1(x)
        # Q, K, V from the same normalized input
        attn_output, _ = self.attn(x_norm1, x_norm1, x_norm1, need_weights=False)
        # Add residual connection *after* dropout for attention output
        x = residual1 + self.dropout(attn_output)

        # --- FFN with Pre-LN ---
        residual2 = x
        x_norm2 = self.norm2(x)
        ffn_output = self.ffn(x_norm2)
        # Add residual connection *after* dropout for FFN output
        x = residual2 + self.dropout(ffn_output)
        return x


class TemporalAttentionBlock(AttentionBlockBase):
    """Performs Multi-Head Self-Attention over the time dimension."""

    def __init__(self, d_model, n_heads, d_ff, dropout):
        # Pass the standard nn.MultiheadAttention implementation
        super().__init__(d_model, n_heads, d_ff, dropout, nn.MultiheadAttention)


class SpatialAttentionBlock(AttentionBlockBase):
    """Performs Multi-Head Self-Attention over the stock dimension."""

    def __init__(self, d_model, n_heads, d_ff, dropout):
        # Pass the standard nn.MultiheadAttention implementation
        super().__init__(d_model, n_heads, d_ff, dropout, nn.MultiheadAttention)


# --- Temporal Aggregation ---
class TemporalAttentionAggregation(nn.Module):
    """Final temporal aggregation using attention based on the last time step."""

    def __init__(self, d_model):
        super().__init__()
        # No bias often preferred in attention query/key projections
        self.query_transform = nn.Linear(d_model, d_model, bias=False)
        # Pre-calculate scale factor
        self.scale = math.sqrt(d_model)

    def forward(self, z):
        """Aggregates time steps based on the last step's query.
        Args:
            z: Tensor, shape [batch_size_eff, seq_len, d_model]
        Returns:
            Tensor, shape [batch_size_eff, d_model]
        """
        # Extract query from the last time step
        query = self.query_transform(z[:, -1, :]).unsqueeze(1)  # [B_eff, 1, D]

        keys = z
        values = z

        # Calculate attention scores and apply softmax
        scores = torch.bmm(query, keys.transpose(1, 2)) / self.scale
        attn_weights = torch.softmax(scores, dim=-1)

        # Apply attention weights to values
        output = torch.bmm(attn_weights, values)
        return output.squeeze(1)


class PortfolioMASTER(nn.Module):
    """
    MASTER (Multi-scale Attention Stock Transformer) for portfolio optimization.
    Uses alternating temporal and spatial attention with multi-scale aggregation.
    
    Input: [Batch, Time, Stocks, Features]
    Output: [Batch, Stocks, 1] (predicted returns)
    """

    def __init__(
        self,
        finance_features_amount: int,  # Renamed for clarity num_features
        stock_amount: int,  # Renamed for clarity num_stocks
        lookback: int,  # Renamed for clarity lookback_window
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        num_encoder_layers: int = 1,
    ):
        super().__init__()
        self.num_features = finance_features_amount
        self.num_stocks = stock_amount
        self.lookback_window = lookback
        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers

        # Input projection and encoding
        self.feature_proj = nn.Linear(self.num_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=lookback)
        # Dropout for the projected & encoded input
        self.input_dropout = nn.Dropout(dropout)

        # --- Stacked Spatio-Temporal Encoder Layers ---
        self.encoder_layers = nn.ModuleList([])
        for _ in range(num_encoder_layers):
            self.encoder_layers.append(
                nn.ModuleDict(
                    {
                        "temporal_block": TemporalAttentionBlock(
                            d_model, n_heads, d_ff, dropout
                        ),
                        "spatial_block": SpatialAttentionBlock(
                            d_model, n_heads, d_ff, dropout
                        )
                        # Optional: Add LayerNorm between T and S blocks if needed
                        # 'inter_norm': nn.LayerNorm(d_model)
                    }
                )
            )

        # Optional: Final normalization after all encoder layers
        self.final_encoder_norm = nn.LayerNorm(d_model)

        # --- Final Aggregation & Prediction Head ---
        # MASTER's specific temporal aggregation
        self.temporal_agg = TemporalAttentionAggregation(d_model=d_model)
        # Simple linear layer to predict the single return value per stock
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x):
        """Forward pass for PortfolioMASTER.
        Args:
            x: Input tensor, shape [Batch, Time, Stocks, Features] -> [B, T, N, F]
        Returns:
            Tensor, shape [Batch, Stocks, 1] -> [B, N, 1] (predicted returns)
        """
        B, T, N, F = x.shape
        assert T == self.lookback_window, f"Input time steps {T} != expected {self.lookback_window}"
        assert N == self.num_stocks, f"Input stocks {N} != expected {self.num_stocks}"
        assert F == self.num_features, f"Input features {F} != expected {self.num_features}"

        # Project and encode: [B, T, N, F] -> [T, B*N, D]
        x = self.feature_proj(x)
        x = x.permute(1, 0, 2, 3).contiguous().view(T, B * N, self.d_model)
        x = self.pos_encoder(x)
        x = self.input_dropout(x)

        # Alternating temporal and spatial attention
        for layer in self.encoder_layers:
            # Temporal attention: [T, B*N, D]
            x = layer["temporal_block"](x)

            # Reshape for spatial attention: [T, B*N, D] -> [N, B*T, D]
            x_spatial_in = (
                x.view(T, B, N, self.d_model)
                .permute(2, 0, 1, 3)
                .contiguous()
                .view(N, B * T, self.d_model)
            )
            x_spatial_out = layer["spatial_block"](x_spatial_in)

            # Reshape back for next temporal block: [N, B*T, D] -> [T, B*N, D]
            x = (
                x_spatial_out.view(N, B, T, self.d_model)
                .permute(1, 2, 0, 3)
                .contiguous()
            )
            x = x.permute(1, 0, 2, 3).contiguous().view(T, B * N, self.d_model)

        # Final normalization and aggregation
        x = self.final_encoder_norm(x)
        x_agg_in = x.permute(1, 0, 2).contiguous()
        aggregated_output = self.temporal_agg(x_agg_in)

        # Predict returns and reshape to [B, N, 1]
        predictions = self.decoder(aggregated_output)
        predictions = predictions.view(B, N, 1)

        return predictions


def build_MASTER(
    # --- Data Shape Parameters ---
    stock_amount: int,
    financial_features_amount: int,
    lookback: int,
    # --- MASTER Architecture Parameters ---
    d_model: int = 64,
    d_ff: int = 128,
    n_heads: int = 4,
    dropout: float = 0.1,
    num_encoder_layers: int = 1,
    # --- Deployment ---
    device: torch.device = torch.device("cpu"),
) -> PortfolioMASTER:
    """
    Build and initialize PortfolioMASTER model.
    
    Args:
        stock_amount: Number of stocks in portfolio
        financial_features_amount: Number of input features per stock
        lookback: Input sequence length
        d_model: Embedding dimension (must be divisible by n_heads)
        d_ff: Feed-forward hidden dimension
        n_heads: Number of attention heads
        dropout: Dropout probability
        num_encoder_layers: Number of encoder layers
        device: Target device for model
    
    Returns:
        PortfolioMASTER model on specified device
    """
    print("-" * 50)
    print("Building PortfolioMASTER:")
    print(f"  Data: stocks={stock_amount}, features={financial_features_amount}, lookback={lookback}")
    print(f"  Arch: d_model={d_model}, n_heads={n_heads}, d_ff={d_ff}, layers={num_encoder_layers}")
    print(f"  Dropout={dropout}, Device={device}")
    print("-" * 50)

    if d_model % n_heads != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

    model = PortfolioMASTER(
        stock_amount=stock_amount,
        finance_features_amount=financial_features_amount,
        lookback=lookback,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        num_encoder_layers=num_encoder_layers,
    )
    # Move model to the target device
    return model.to(device)


if __name__ == "__main__":
    # Test configuration
    batch_size = 64
    lookback = 20
    stock_amount = 10
    features = 5
    example_d_model = 64
    example_heads = 4
    example_dropout = 0.1
    d_ff = 128
    example_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_MASTER(
        stock_amount=stock_amount,
        financial_features_amount=features,
        lookback=lookback,
        d_model=example_d_model,
        n_heads=example_heads,
        dropout=example_dropout,
        d_ff=d_ff,
        device=example_device,
    )

    # Test forward pass
    dummy_input = torch.randn(batch_size, lookback, stock_amount, features, device=example_device)
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
