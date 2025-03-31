import math
import torch
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer


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
        # Shape [max_len, 1, d_model] for broadcasting with [T, B*N, D]
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
        # self.pe[:x.size(0)] has shape [seq_len, 1, d_model], broadcasts correctly
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class PortfolioVanillaTransformer(nn.Module):
    """
    A simplified Transformer model inspired by the original Vaswani et al. architecture,
    adapted for portfolio selection by processing each stock's time series independently.
    It applies self-attention only along the time dimension for each stock.

    Input: [Batch, Time(T), Stocks(N), Features(F)]
    Output: [Batch, Stocks(N), 1] (Predicted Returns)
    """

    def __init__(
        self,
        stock_amount: int,  # Number of stocks (N)
        financial_features_amount: int,  # Number of features per stock (F)
        lookback: int,  # Lookback window size (T)
        d_model: int,  # Internal dimension of the transformer
        n_heads: int,  # Number of attention heads
        d_ff: int,  # Dimension of the feedforward network
        dropout: float,  # Dropout rate
        num_encoder_layers: int,  # Number of stacked encoder layers
        use_final_norm: bool = True,  # Whether to use LayerNorm after the encoder stack
    ):
        super().__init__()
        self.stock_amount = stock_amount
        self.financial_features_amount = financial_features_amount
        self.lookback = lookback
        self.d_model = d_model

        # --- Input Processing ---
        # Project input features (F) to the model dimension (D)
        self.feature_proj = nn.Linear(financial_features_amount, d_model)
        # Standard positional encoding for the time dimension
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=lookback)
        # Dropout after projection and encoding
        self.input_dropout = nn.Dropout(dropout)

        # --- Transformer Encoder ---
        # Define a single encoder layer using the standard PyTorch module
        # batch_first=False because we will provide input as [T, B*N, D]
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="relu",  # Standard activation in original Transformer
            batch_first=False,  # Input shape is [T, BatchEff, D]
            norm_first=False  # Use Post-LN (Normalization after Add) like original Transformer
            # Set norm_first=True for Pre-LN (Normalization before Add)
        )
        # Stack multiple encoder layers
        # Optional final normalization after the stack (often beneficial)
        encoder_norm = nn.LayerNorm(d_model) if use_final_norm else None
        self.transformer_encoder = TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
            norm=encoder_norm,
        )

        # --- Output Head ---
        # Project the final representation of each stock to a single prediction value
        self.projection_head = nn.Linear(d_model, 1)

    def forward(self, x):
        """
        Args:
            x: Input tensor, shape [B, T, N, F]
               B = batch_size, T = lookback, N = stock_amount, F = financial_features_amount
        Returns:
            Output tensor, shape [B, N, 1] (predicted returns)
        """
        B, T, N, F = x.shape
        # Validate input shapes
        assert T == self.lookback, f"Input time steps {T} != expected {self.lookback}"
        assert (
            N == self.stock_amount
        ), f"Input stocks {N} != expected {self.stock_amount}"
        assert (
            F == self.financial_features_amount
        ), f"Input features {F} != expected {self.financial_features_amount}"

        # 1. Project features: [B, T, N, F] -> [B, T, N, D]
        x = self.feature_proj(x)

        # 2. Reshape for Transformer Encoder (process each stock independently):
        # Input shape requires [SeqLen(T), BatchEff(B*N), Dim(D)]
        # [B, T, N, D] -> Permute to [T, B, N, D] -> Reshape to [T, B*N, D]
        x = x.permute(1, 0, 2, 3).contiguous().view(T, B * N, self.d_model)

        # 3. Add Positional Encoding and Input Dropout: Output shape [T, B*N, D]
        x = self.pos_encoder(x)
        x = self.input_dropout(x)

        # 4. Pass through Transformer Encoder layers: Output shape [T, B*N, D]
        # The encoder internally handles self-attention along the time dimension (T)
        encoded_output = self.transformer_encoder(x)

        # 5. Select the output from the last time step for prediction:
        # Takes the state at the end of the sequence (index T-1) for each stock
        # Shape changes from [T, B*N, D] -> [B*N, D]
        last_step_output = encoded_output[-1, :, :]

        # 6. Final Projection: [B*N, D] -> [B*N, 1]
        predictions = self.projection_head(last_step_output)

        # 7. Reshape to final output format: [B*N, 1] -> [B, N, 1]
        predictions = predictions.view(B, N, 1)

        return predictions


def build_PortfolioVanillaTransformer(
    stock_amount: int,  # Number of stocks (N)
    financial_features_amount: int,  # Number of features per stock (F)
    lookback: int,  # Lookback window size (T)
    d_model: int = 64,  # Internal dimension of the transformer
    n_heads: int = 4,  # Number of attention heads
    d_ff: int = 128,  # Dimension of the feed-forward layer
    dropout: float = 0.1,  # Dropout rate
    num_encoder_layers: int = 2,  # Number of encoder layers (e.g., 2)
    use_final_norm: bool = True,  # Add LayerNorm after encoder stack
    # Note: activation and norm_first are implicitly set inside PortfolioVanillaTransformer
    # to match original Transformer defaults (relu, Post-LN/norm_first=False)
    device: torch.device = torch.device("cpu"),
) -> PortfolioVanillaTransformer:
    """
    Factory function to build the PortfolioVanillaTransformer model.

    Args:
        stock_amount: Number of stocks (N).
        financial_features_amount: Number of features per stock (F).
        lookback: Lookback window size (T).
        d_model: Internal embedding dimension. Defaults to 64.
        n_heads: Number of attention heads. Defaults to 4.
        d_ff: Dimension of the feed-forward network. Defaults to 128.
        dropout: Dropout rate. Defaults to 0.1.
        num_encoder_layers: Number of stacked Transformer encoder layers. Defaults to 2.
        use_final_norm: Whether to add a LayerNorm after the encoder stack. Defaults to True.
        device: The torch device to place the model on. Defaults to "cpu".

    Returns:
        An instance of the PortfolioVanillaTransformer model on the specified device.

    Raises:
        ValueError: If d_model is not divisible by n_heads.
    """
    # --- Configuration Logging ---
    print("-" * 30)
    print(f"Building PortfolioVanillaTransformer")
    print(
        f"  Data Params: Stocks={stock_amount}, Features={financial_features_amount}, Lookback={lookback}"
    )
    print(
        f"  Arch Params: d_model={d_model}, n_heads={n_heads}, d_ff={d_ff}, layers={num_encoder_layers}"
    )
    print(f"  Other Params: dropout={dropout}, final_norm={use_final_norm}")
    print(f"  Target Device: {device}")
    print("-" * 30)

    # --- Validation ---
    if d_model % n_heads != 0:
        raise ValueError(
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )

    # --- Model Instantiation ---
    model = PortfolioVanillaTransformer(
        stock_amount=stock_amount,
        financial_features_amount=financial_features_amount,
        lookback=lookback,
        d_model=d_model,
        n_heads=n_heads,
        d_ff=d_ff,
        dropout=dropout,
        num_encoder_layers=num_encoder_layers,
        use_final_norm=use_final_norm
        # activation='relu' and norm_first=False are implicitly used
        # by nn.TransformerEncoderLayer default inside the model class
    )

    # --- Move to Device and Return ---
    return model.to(device)


# --- Example Usage (requires the model class definitions) ---
# if __name__ == '__main__':
#     # Example Parameters
#     N_STOCKS = 100
#     N_FEATURES = 5
#     LOOKBACK_WINDOW = 20
#     DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Build the model
#     vanilla_transformer_model = build_PortfolioVanillaTransformer(
#         stock_amount=N_STOCKS,
#         financial_features_amount=N_FEATURES,
#         lookback=LOOKBACK_WINDOW,
#         d_model=128,
#         n_heads=8,
#         d_ff=256,
#         dropout=0.1,
#         num_encoder_layers=3,
#         device=DEVICE
#     )

#     print(f"Model built successfully on {DEVICE}")
#     print(vanilla_transformer_model)

#     # Example input tensor
#     example_batch_size = 4
#     example_input = torch.randn(example_batch_size, LOOKBACK_WINDOW, N_STOCKS, N_FEATURES).to(DEVICE)

#     # Forward pass
#     try:
#         with torch.no_grad():
#             output = vanilla_transformer_model(example_input)
#         print(f"Example forward pass successful. Output shape: {output.shape}") # Should be [B, N, 1]
#         assert output.shape == (example_batch_size, N_STOCKS, 1)
#     except Exception as e:
#         print(f"Error during example forward pass: {e}")
