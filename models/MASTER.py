import math
import torch
from torch import nn
from torch.nn.modules.normalization import LayerNorm


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
        """ x: [seq_len, batch_eff, d_model] """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class TemporalAttentionBlock(nn.Module):
    """Performs Multi-Head Self-Attention over the time dimension, followed by FFN."""
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=False) # [T, B_eff, D]
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """ x: [T, B_eff, D] """
        # Self-Attention
        residual1 = x
        x_norm1 = self.norm1(x) # Pre-LN dla stabilności? lub Post-LN jak niżej
        attn_output, _ = self.attn(x, x, x, need_weights=False) # Post-LN Style
        x = residual1 + self.dropout1(attn_output)
        # x = self.norm1(x) # Jeśli Post-LN

        # FFN
        residual2 = x
        x_norm2 = self.norm2(x) # Pre-LN
        ffn_output = self.ffn(x_norm2)
        x = residual2 + self.dropout2(ffn_output)
        # x = self.norm2(x) # Jeśli Post-LN

        return x


class SpatialAttentionBlock(nn.Module):
    """Performs Multi-Head Self-Attention over the stock dimension, followed by FFN."""
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=False) # [N, B_eff, D]
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """ x: [N, B_eff, D] """
        # Self-Attention
        residual1 = x
        x_norm1 = self.norm1(x) # Pre-LN
        attn_output, _ = self.attn(x_norm1, x_norm1, x_norm1, need_weights=False)
        x = residual1 + self.dropout1(attn_output)
        # x = self.norm1(x) # Post-LN

        # FFN
        residual2 = x
        x_norm2 = self.norm2(x) # Pre-LN
        ffn_output = self.ffn(x_norm2)
        x = residual2 + self.dropout2(ffn_output)
        # x = self.norm2(x) # Post-LN

        return x


class TemporalAttentionAggregation(nn.Module):
    """Final temporal aggregation using attention based on the last time step."""
    def __init__(self, d_model):
        super().__init__()
        self.query_transform = nn.Linear(d_model, d_model, bias=False)
        self.scale = math.sqrt(d_model)

    def forward(self, z):
        """ z: [batch_size_eff, seq_len, d_model] -> output: [batch_size_eff, d_model] """
        # Query from last time step
        query = self.query_transform(z[:, -1, :]).unsqueeze(1) # [B_eff, 1, D]
        keys = z # [B_eff, T, D]
        values = z # [B_eff, T, D]
        # Attention calculation
        scores = torch.matmul(query, keys.transpose(1, 2)) / self.scale # [B_eff, 1, T]
        attn_weights = torch.softmax(scores, dim=-1) # [B_eff, 1, T]
        # Weighted sum
        output = torch.matmul(attn_weights, values) # [B_eff, 1, D]
        return output.squeeze(1) # [B_eff, D]


# --- Adapted PortfolioMASTER Model ---
class PortfolioMASTER(nn.Module):
    """
    MASTER model for portfolio return prediction.
    Uses standard nn components and allows stacking layers.
    Preserves T-Attn -> S-Attn -> Aggregation flow.
    """

    def __init__(self, finance_features_amount, stock_amount, lookback, d_model,
                 n_heads, d_ff, dropout, num_encoder_layers: int = 1):
        super().__init__()
        self.d_feat = finance_features_amount
        self.stock_amount = stock_amount
        self.lookback = lookback
        self.d_model = d_model
        self.num_encoder_layers = num_encoder_layers

        self.feature_proj = nn.Linear(self.d_feat, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=lookback)
        self.input_dropout = nn.Dropout(dropout)

        # Stos warstw enkodera T->S
        self.encoder_layers = nn.ModuleList([])
        for _ in range(num_encoder_layers):
            self.encoder_layers.append(nn.ModuleDict({
                'temporal_block': TemporalAttentionBlock(d_model, n_heads, d_ff, dropout),
                'spatial_block': SpatialAttentionBlock(d_model, n_heads, d_ff, dropout)
                # Norma między nimi może być pomocna
                # 'inter_norm': nn.LayerNorm(d_model)
            }))

        # Opcjonalna norma po całym enkoderze
        self.final_encoder_norm = nn.LayerNorm(d_model)

        # Charakterystyczna agregacja MASTERa
        self.temporal_agg = TemporalAttentionAggregation(d_model=d_model)

        # Prosty dekoder
        self.decoder = nn.Linear(d_model, 1)

    def forward(self, x):
        """ x: [B, T, N, F] -> [B, N, 1] """
        B, T, N, F = x.shape
        assert T == self.lookback and N == self.stock_amount and F == self.d_feat

        # 1. Project Features
        x = self.feature_proj(x)  # [B, T, N, D]

        # 2. Reshape for Temporal Processing & Add PE/Dropout
        x = x.permute(1, 0, 2, 3).contiguous().view(T, B * N, self.d_model)  # [T, B*N, D]
        x = self.pos_encoder(x)
        x = self.input_dropout(x)

        # 3. Pass through Encoder Layers (T -> S flow)
        for layer in self.encoder_layers:
            # --- a) Temporal Block ---
            # Input/Output: [T, B*N, D]
            x = layer['temporal_block'](x)

            # --- b) Spatial Block ---
            # Reshape: [T, B*N, D] -> [B, T, N, D] -> [N, B*T, D]
            x = x.view(T, B, N, self.d_model).permute(2, 0, 1, 3).contiguous().view(N, B * T, self.d_model)
            x = layer['spatial_block'](x)  # Input/Output: [N, B*T, D]

            # --- c) Reshape back for next layer's Temporal input ---
            # Reshape: [N, B*T, D] -> [B, T, N, D] -> [T, B*N, D]
            x = x.view(N, B, T, self.d_model).permute(1, 2, 0, 3).contiguous()  # [B, T, N, D]
            x = x.permute(1, 0, 2, 3).contiguous().view(T, B * N, self.d_model)  # [T, B*N, D]

        # x na wyjściu pętli ma kształt [T, B*N, D]

        # 4. Final Encoder Norm (Opcjonalnie)
        x = self.final_encoder_norm(x)

        # 5. Final Temporal Aggregation
        # Oczekuje [B*N, T, D]
        x = x.permute(1, 0, 2).contiguous()  # [B*N, T, D]
        aggregated_output = self.temporal_agg(x)  # Output: [B*N, D]

        # 6. Decoder
        predictions = self.decoder(aggregated_output)  # [B*N, 1]
        predictions = predictions.view(B, N, 1)  # -> [B, N, 1]

        return predictions