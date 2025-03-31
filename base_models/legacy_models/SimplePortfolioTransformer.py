import math
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    """Standard Positional Encoding using sin/cos."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model) # Shape [max_len, 1, d_model] dla batch_first=False
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_eff, d_model] (batch_first=False expected by nn.TransformerEncoderLayer)
        """
        # Dodajemy pe do wejścia. self.pe ma [max_len, 1, d_model],
        # x ma [seq_len, batch_eff, d_model]. Broadcasting zadziała dla wymiaru batch_eff.
        # Musimy upewnić się, że seq_len <= max_len.
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

# --- Prosty Model Transformer ---

class SimplePortfolioTransformer(nn.Module):
    """
    A very simple Transformer Encoder-based model for predicting stock returns.
    Applies temporal attention independently for each stock.
    No cross-stock attention for simplicity.
    """
    def __init__(
        self,
        stock_amount: int,
        financial_features_amount: int,
        lookback: int,
        d_model: int,
        n_heads: int,
        d_ff: int, # d_ff to dim_feedforward w nn.TransformerEncoderLayer
        dropout: float,
        num_encoder_layers: int,
    ):
        super().__init__()
        self.stock_amount = stock_amount
        self.financial_features_amount = financial_features_amount
        self.lookback = lookback
        self.d_model = d_model

        # 1. Warstwa wejściowa (projekcja cech)
        self.feature_proj = nn.Linear(financial_features_amount, d_model)

        # 2. Kodowanie Pozycyjne
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=lookback)

        # 3. Stos Warstw Enkodera Transformera
        # Użyjemy wbudowanego nn.TransformerEncoderLayer i nn.TransformerEncoder
        # Ważne: nn.TransformerEncoderLayer domyślnie oczekuje batch_first=False
        # (czyli kształt [seq_len, batch, embedding_dim])
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation='relu', # Można zmienić na 'gelu'
            batch_first=False, # Kluczowe ustawienie!
            norm_first=False # Użyjemy standardowego Post-LN (norma po residualu)
        )
        encoder_norm = nn.LayerNorm(d_model) # Opcjonalna norma na końcu stosu enkoderów
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers,
            norm=encoder_norm # Zastosuj normę na końcu
        )

        # 4. Głowica Projekcyjna (predykcja zwrotu)
        # Przetwarza wyjście z ostatniego kroku czasowego enkodera
        self.projection_head = nn.Linear(d_model, 1) # Mapuje D -> 1 (zwrot)

    def forward(self, x):
        """
        Args:
            x: Input tensor, shape [B, T, N, F]
        Returns:
            Output tensor, shape [B, N, 1] (predicted returns)
        """
        B, T, N, F = x.shape
        assert T == self.lookback and N == self.stock_amount and F == self.financial_features_amount

        # 1. Projekcja cech: [B, T, N, F] -> [B, T, N, D]
        x = self.feature_proj(x)

        # 2. Przygotowanie danych dla Enkodera (batch_first=False)
        # Chcemy przetwarzać każdą akcję niezależnie w czasie.
        # Reshape: [B, T, N, D] -> [T, B*N, D]
        x = x.permute(1, 0, 2, 3).contiguous().view(T, B * N, self.d_model)

        # 3. Kodowanie Pozycyjne (dodawane do [T, B*N, D])
        x = self.pos_encoder(x)

        # 4. Przejście przez stos enkoderów Transformera
        # Wejście/Wyjście: [T, B*N, D]
        encoded_output = self.transformer_encoder(x)

        # 5. Agregacja czasowa: Weź stan z ostatniego kroku czasowego
        # encoded_output ma kształt [T, B*N, D]
        # Wybieramy ostatni krok czasowy (indeks T-1 lub -1 wzdłuż wymiaru 0)
        last_time_step_output = encoded_output[-1, :, :] # Shape: [B*N, D]

        # 6. Głowica Projekcyjna
        # [B*N, D] -> [B*N, 1]
        predictions = self.projection_head(last_time_step_output)

        # 7. Reshape do finalnego kształtu wyjściowego
        # [B*N, 1] -> [B, N, 1]
        predictions = predictions.view(B, N, 1)

        return predictions

# --- Funkcja Budująca ---

def build_SimplePortfolioTransformer(
    stock_amount: int,
    financial_features_amount: int,
    lookback: int,
    d_model: int = 64,   # Mniejszy d_model dla prostoty
    n_heads: int = 4,    # d_model musi być podzielne przez n_heads
    d_ff: int = 128,   # Mniejszy d_ff (np. 2*d_model)
    dropout: float = 0.1,
    num_encoder_layers: int = 1, # Zacznij od 1-2 warstw
    device: torch.device = torch.device("cpu")
) -> SimplePortfolioTransformer:
    """Factory function to build the SimplePortfolioTransformer model."""
    print("-" * 30)
    print("Building SimplePortfolioTransformer with parameters:")
    print(f"  Data: stocks={stock_amount}, features={financial_features_amount}, lookback={lookback}")
    print(f"  Arch: d_model={d_model}, n_heads={n_heads}, d_ff={d_ff}, layers={num_encoder_layers}")
    print(f"  Dropout: {dropout}")
    print(f"  Device: {device}")
    print("-" * 30)

    if d_model % n_heads != 0:
        raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

    model = SimplePortfolioTransformer(
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

# --- Example Usage ---
if __name__ == '__main__':
    batch_size = 64
    lookback = 20
    stock_amount = 10
    features = 5
    example_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build the simple model
    model = build_SimplePortfolioTransformer(
        stock_amount=stock_amount,
        financial_features_amount=features,
        lookback=lookback,
        # Użyj mniejszych parametrów dla testu
        d_model=32,
        n_heads=4,
        d_ff=64,
        dropout=0.1,
        num_encoder_layers=1, # Tylko jedna warstwa na początek
        device=example_device
    )

    dummy_input = torch.randn(batch_size, lookback, stock_amount, features, device=example_device)
    print(f"\nTesting model with input shape: {dummy_input.shape}") # [B, T, N, F]

    try:
        model.eval()
        with torch.no_grad():
             output_returns = model(dummy_input)
        print(f"Model output shape: {output_returns.shape}") # Expected: [B, N, 1]
        # Sprawdź std dev wyjścia dla różnych próbek w batchu
        if batch_size > 1:
            output_std_dev = output_returns[:, 0, 0].std().item() # Std dev dla pierwszej akcji w batchu
            print(f"Output std dev across batch (stock 0): {output_std_dev:.6f}")
        print("Model forward pass successful!")
    except Exception as e:
        print(f"Error during model forward pass: {e}")
        import traceback
        traceback.print_exc()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params:,}") # Będzie znacznie mniej parametrów