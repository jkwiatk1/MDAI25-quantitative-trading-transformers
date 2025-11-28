import math
import torch
from torch import nn
import logging


# --- Standardowy Positional Encoding (dla wymiaru N) ---
class PositionalEncoding(nn.Module):
    """Standard Positional Encoding using sin/cos. Expects input shape [SeqLen(N), Batch(B), Dim(D)]."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        # Shape [max_len, 1, d_model] for broadcasting
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)
        # Basic check for NaN/Inf during initialization
        if torch.isnan(self.pe).any() or torch.isinf(self.pe).any():
            logging.error("NaN or Inf detected in PositionalEncoding initialization!")

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor, shape [seq_len(N), batch(B), d_model(D)].
        Returns:
            torch.Tensor: Output tensor with positional encoding added, same shape.
        """
        seq_len = x.size(0)
        # Check if sequence length exceeds max_len and handle (e.g., log warning)
        if seq_len > self.pe.size(0):
            logging.warning(
                f"Input sequence length ({seq_len}) exceeds PositionalEncoding max_len ({self.pe.size(0)}). PE will be sliced."
            )
            # Alternatively, raise ValueError or implement dynamic extension
        # Add PE [seq_len, 1, d_model] to x [seq_len, batch, d_model] using broadcasting
        x = x + self.pe[:seq_len]
        return self.dropout(x)


# --- MLP Embedding dla całej serii czasowej ---
class SeriesEmbeddingMLP(nn.Module):
    """
    Embeds the entire time series [T, F] into a single token [d_model] using MLP.
    Includes options for Layer Normalization and multiple hidden layers.
    """
    def __init__(
        self,
        lookback: int,
        num_features: int,
        d_model: int,
        dropout: float = 0.1,
        num_hidden_layers: int = 2, # Nowy parametr: liczba warstw ukrytych
        add_output_norm: bool = True # Nowy parametr: czy dodać LayerNorm na wyjściu
    ):
        super().__init__()
        self.lookback = lookback
        self.num_features = num_features
        self.d_model = d_model
        input_dim = lookback * num_features

        layers = []
        current_dim = input_dim

        # --- Warstwy Ukryte ---
        for i in range(num_hidden_layers):
            # Oblicz rozmiar warstwy ukrytej (można użyć innej strategii)
            # Tutaj prosty przykład: malejący rozmiar w kierunku d_model
            hidden_dim = current_dim // 2 if current_dim // 2 > d_model else (current_dim + d_model) // 2
            # Upewnij się, że hidden_dim nie jest zbyt mały
            hidden_dim = max(hidden_dim, d_model, 16) # Minimalny rozmiar np. 16 lub d_model

            layers.append(nn.Linear(current_dim, hidden_dim))
            # Można dodać LayerNorm PO aktywacji w warstwach ukrytych dla stabilności
            # layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU()) # Lub nn.GELU()
            layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim # Aktualizuj wymiar wejściowy dla następnej warstwy

        # --- Warstwa Wyjściowa ---
        layers.append(nn.Linear(current_dim, d_model))

        # --- Opcjonalna Normalizacja na Wyjściu ---
        if add_output_norm:
            layers.append(nn.LayerNorm(d_model))
            logging.debug("Added LayerNorm to the output of SeriesEmbeddingMLP.")

        # Stwórz sekwencję warstw
        self.mlp = nn.Sequential(*layers)

        logging.debug(f"Initialized SeriesEmbeddingMLP with {num_hidden_layers} hidden layers. Output norm: {add_output_norm}")
        logging.debug(f"MLP structure: {self.mlp}")


    def forward(self, x_series_batch):
        """
        Args:
            x_series_batch (torch.Tensor): Input batch of series, shape [B_eff, T, F].
        Returns:
            torch.Tensor: Embedded tokens, shape [B_eff, d_model].
        """
        B_eff, T, F = x_series_batch.shape
        # Asserty przeniesione do wewnątrz forward dla pewności
        if T != self.lookback:
            raise ValueError(f"Series length mismatch: expected {self.lookback}, got {T}")
        if F != self.num_features:
            raise ValueError(f"Feature mismatch: expected {self.num_features}, got {F}")

        # Flatten Time and Feature dimensions
        x_flat = x_series_batch.view(B_eff, T * F)
        # Pass through MLP
        embedding = self.mlp(x_flat)
        return embedding


# --- Portfolio iTransformer (Wersja bliższa oryginałowi) ---
class PortfolioITransformerOriginalConcept(nn.Module):
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
        # --- Dodane argumenty ---
        mlp_hidden_layers: int = 2,
        mlp_add_output_norm: bool = True
    ):
        super().__init__()
        if d_model % n_heads != 0: raise ValueError(...)
        self.stock_amount = stock_amount; self.d_model = d_model

        # --- Użyj nowych argumentów przy tworzeniu SeriesEmbeddingMLP ---
        self.series_embedding = SeriesEmbeddingMLP(
            lookback, financial_features_amount, d_model, dropout,
            num_hidden_layers=mlp_hidden_layers, # Przekaż liczbę warstw
            add_output_norm=mlp_add_output_norm  # Przekaż flagę normy
        )
        self.embedding_dropout = nn.Dropout(dropout) if use_embedding_dropout else nn.Identity()
        self.stock_pos_encoder = PositionalEncoding(d_model, dropout, max_len=stock_amount)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout,
            activation=activation, batch_first=False, norm_first=norm_first
        )
        encoder_norm = nn.LayerNorm(d_model) if not norm_first else None
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        self.projection_head = nn.Linear(d_model, 1)

    def forward(self, x):
        # ... (forward bez zmian) ...
        B, T, N, F = x.shape
        x_permuted = x.permute(0, 2, 1, 3).contiguous(); x_for_embedding = x_permuted.view(B * N, T, F)
        variate_tokens = self.series_embedding(x_for_embedding); variate_tokens = self.embedding_dropout(variate_tokens)
        encoder_input = variate_tokens.view(B, N, self.d_model); encoder_input = encoder_input.permute(1, 0, 2).contiguous()
        encoder_input = self.stock_pos_encoder(encoder_input); encoded_variates = self.transformer_encoder(encoder_input)
        predictions = self.projection_head(encoded_variates); predictions = predictions.permute(1, 0, 2).contiguous()
        return predictions

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor, shape [B, T, N, F].
        Returns:
            torch.Tensor: Output tensor, shape [B, N, 1] (predicted returns).
        """
        B, T, N, F = x.shape
        # Input shape validation (already done in SeriesEmbeddingMLP)

        # 1. Reshape for Embedding: [B, T, N, F] -> [B, N, T, F] -> [B*N, T, F]
        # Permute to group by stock first, then flatten Batch and Stock dims
        x_permuted = x.permute(0, 2, 1, 3).contiguous()
        x_for_embedding = x_permuted.view(B * N, T, F)

        # 2. Apply Series Embedding MLP: [B*N, T, F] -> [B*N, D]
        variate_tokens = self.series_embedding(x_for_embedding)
        variate_tokens = self.embedding_dropout(variate_tokens)  # Apply dropout

        # 3. Reshape for Transformer Encoder: [B*N, D] -> [B, N, D] -> [N, B, D]
        # Unflatten B*N, then permute N and B for encoder input format
        encoder_input = variate_tokens.view(B, N, self.d_model)
        encoder_input = encoder_input.permute(1, 0, 2).contiguous()  # Shape: [N, B, D]

        # 4. Add Positional Encoding for Stocks (N dimension): Input/Output [N, B, D]
        encoder_input = self.stock_pos_encoder(encoder_input)

        # 5. Pass through Transformer Encoder Layers: Input/Output [N, B, D]
        # Attention operates across the N stock tokens
        encoded_variates = self.transformer_encoder(encoder_input)

        # 6. Prediction Head:
        # Apply projection head directly to the output of the encoder
        # Input: [N, B, D] -> Output: [N, B, 1]
        predictions = self.projection_head(encoded_variates)

        # 7. Reshape to final output format: [N, B, 1] -> [B, N, 1]
        # Permute N and B back
        predictions = predictions.permute(1, 0, 2).contiguous()

        return predictions


# --- Factory Function (Updated) ---
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
    # --- Nowe argumenty dla SeriesEmbeddingMLP ---
    mlp_hidden_layers: int = 2,  # Domyślnie 2 warstwy ukryte
    mlp_add_output_norm: bool = True, # Domyślnie dodaj normę na wyjściu MLP
    # ---------------------------------------------
    device: torch.device = torch.device("cpu"),
) -> PortfolioITransformerOriginalConcept: # Zakładając, że klasa się nie zmieniła
    """Factory function for the adapted PortfolioITransformer model."""
    print("-" * 30)
    print(f"Building PortfolioITransformerOriginalConcept")
    # ... (reszta logowania parametrów, dodaj logowanie nowych parametrów MLP) ...
    print(f"  MLP Params: hidden_layers={mlp_hidden_layers}, output_norm={mlp_add_output_norm}")
    print("-" * 30)

    if d_model % n_heads != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

    model = PortfolioITransformerOriginalConcept( # Użyj właściwej nazwy klasy
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
         # --- Przekaż nowe argumenty ---
        mlp_hidden_layers=mlp_hidden_layers,
        mlp_add_output_norm=mlp_add_output_norm
    )
    return model.to(device)