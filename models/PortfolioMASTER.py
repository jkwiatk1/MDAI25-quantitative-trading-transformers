import math
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Standard Positional Encoding using sin/cos. Expects input shape [Seq Len, Batch Eff, Dim]."""

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
        """
        Args:
            x: Tensor, shape [seq_len, batch_eff, d_model]
        """
        # Add positional encoding up to the sequence length of x
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


# --- Attention Blocks ---
class AttentionBlockBase(nn.Module):
    """Base class for Attention blocks to reduce code duplication."""

    def __init__(self, d_model, n_heads, d_ff, dropout, attention_impl):
        super().__init__()
        # Use batch_first=False consistent with expected input [Seq, Batch, Dim]
        self.attn = attention_impl(d_model, n_heads, dropout=dropout, batch_first=False)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)  # Single dropout instance can be reused

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),  # Reuse dropout instance
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

        # Keys and Values are the full sequence
        keys = z  # [B_eff, T, D]
        values = z  # [B_eff, T, D]

        # Calculate attention scores
        # Using bmm (batch matrix multiplication) is efficient here
        scores = torch.bmm(query, keys.transpose(1, 2)) / self.scale  # [B_eff, 1, T]
        attn_weights = torch.softmax(scores, dim=-1)  # [B_eff, 1, T]

        # Apply weights to values
        # Using bmm again
        output = torch.bmm(attn_weights, values)  # [B_eff, 1, D]

        # Remove the middle dimension (dim=1)
        return output.squeeze(1)  # [B_eff, D]


# --- Adapted PortfolioMASTER Model ---
class PortfolioMASTER(nn.Module):
    """
    MASTER-inspired model for portfolio return prediction.
    Uses alternating Temporal and Spatial attention blocks and
    MASTER's final temporal aggregation method.
    NOTE: Does NOT include the original MASTER's market-guided gate.
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
        # Store config parameters
        self.num_features = finance_features_amount
        self.num_stocks = stock_amount
        self.lookback_window = lookback
        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers

        # --- Input Projection & Encoding ---
        # Project input features to d_model
        self.feature_proj = nn.Linear(self.num_features, d_model)
        # Positional encoding for the time dimension
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
        # Input shape validation
        assert (
            T == self.lookback_window
        ), f"Input time steps {T} != expected {self.lookback_window}"
        assert N == self.num_stocks, f"Input stocks {N} != expected {self.num_stocks}"
        assert (
            F == self.num_features
        ), f"Input features {F} != expected {self.num_features}"

        # 1. Project Features: [B, T, N, F] -> [B, T, N, D]
        x = self.feature_proj(x)

        # 2. Reshape for Temporal Processing: [B, T, N, D] -> [T, B*N, D]
        # Permute T and B, N, then merge B and N
        x = x.permute(1, 0, 2, 3).contiguous().view(T, B * N, self.d_model)

        # 3. Add Positional Encoding & Input Dropout: Still [T, B*N, D]
        x = self.pos_encoder(x)
        x = self.input_dropout(x)

        # 4. Pass through Encoder Layers (Alternating T -> S)
        for layer in self.encoder_layers:
            # --- a) Temporal Block ---
            # Input/Output shape: [T, B*N, D]
            x = layer["temporal_block"](x)

            # --- b) Spatial Block ---
            # Reshape for Spatial Attention: [T, B*N, D] -> [B, T, N, D] -> [N, B*T, D]
            # Unmerge B*N, permute N and T, merge B*T
            x_spatial_in = (
                x.view(T, B, N, self.d_model)
                .permute(2, 0, 1, 3)
                .contiguous()
                .view(N, B * T, self.d_model)
            )
            # Apply Spatial Attention: Input/Output shape: [N, B*T, D]
            x_spatial_out = layer["spatial_block"](x_spatial_in)

            # --- c) Reshape back for next Temporal Block (if any) ---
            # Reshape: [N, B*T, D] -> [B, T, N, D] -> [T, B*N, D]
            # Unmerge B*T, permute T and N back, merge B*N
            x = (
                x_spatial_out.view(N, B, T, self.d_model)
                .permute(1, 2, 0, 3)
                .contiguous()
            )  # [B, T, N, D]
            x = (
                x.permute(1, 0, 2, 3).contiguous().view(T, B * N, self.d_model)
            )  # [T, B*N, D]

        # Output of loop has shape [T, B*N, D]

        # 5. Optional: Final Encoder Norm
        x = self.final_encoder_norm(x)

        # 6. Final Temporal Aggregation (MASTER specific)
        # Expects [B_eff, T, D], so permute [T, B*N, D] -> [B*N, T, D]
        x_agg_in = x.permute(1, 0, 2).contiguous()
        aggregated_output = self.temporal_agg(x_agg_in)  # Output: [B*N, D]

        # 7. Decode to predict returns
        # Input: [B*N, D] -> Output: [B*N, 1]
        predictions = self.decoder(aggregated_output)

        # 8. Reshape to final output format
        # Reshape: [B*N, 1] -> [B, N, 1]
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
    Builds and initializes the PortfolioMASTER model.

    Args:
        stock_amount (int): Number of different assets/stocks in the portfolio.
        financial_features_amount (int): Number of features used to describe each stock.
        lookback (int): The lookback period (input sequence length).

        d_model (int): The main dimensionality of embeddings and hidden states.
                       Must be divisible by both t_n_heads and s_n_heads.
        d_ff (int): The dimensionality of the inner hidden layer in the FFNs.
        t_n_heads (int): Number of attention heads for Temporal Attention (TAttention).
        s_n_heads (int): Number of attention heads for Spatial Attention (SAttention).
        t_dropout (float): Dropout probability for Temporal Attention layers.
        s_dropout (float): Dropout probability for Spatial Attention layers.

        device (torch.device): The device (e.g., 'cuda', 'cpu') to create the model on.

    Returns:
        PortfolioMASTER: An initialized instance of the model.
    """
    print("-" * 30)
    print("Building PortfolioMASTER with parameters:")
    print(
        f"  Data: stocks={stock_amount}, features={financial_features_amount}, lookback={lookback}"
    )
    print(
        f"  Arch: d_model={d_model}, n_heads={n_heads}, d_ff={d_ff}, layers={num_encoder_layers}"
    )
    print(f"  Dropout: {dropout}")
    print(f"  Device: {device}")
    print("-" * 30)

    # --- Parameter Validation ---
    if d_model % n_heads != 0:
        raise ValueError(
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

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
    # Example input data characteristics
    batch_size = 64
    lookback = 20  # T
    stock_amount = 10  # N
    features = 5  # F

    # Example parameters
    example_d_model = 64
    example_heads = 4
    example_dropout = 0.1
    d_ff = 128
    example_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the model
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

    # Test with dummy data
    dummy_input = torch.randn(
        batch_size, lookback, stock_amount, features, device=example_device
    )
    print(f"\nTesting model with input shape: {dummy_input.shape}")  # [B, T, N, F]

    # Pass data through the model
    try:
        model.eval()  # Set to evaluation mode
        with torch.no_grad():
            output_returns = model(dummy_input)
        print(f"Model output shape: {output_returns.shape}")  # Expected: [B, N, 1]
        print("Model forward pass successful!")
    except Exception as e:
        print(f"Error during model forward pass: {e}")
        import traceback

        traceback.print_exc()

    # Check number of parameters
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}")
