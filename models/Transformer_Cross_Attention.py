import math
import torch
from torch import nn

from models.modules import (
    LayerNormalization,
    FeedForwardBlock,
    MultiHeadAttentionBlock,
    ResidualConnection,
)


class PositionalEncoding(nn.Module):
    """Dodaje informację o czasie do wejść transformera"""

    def __init__(self, d_model: int, dropout: float, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1), :])


class SelfAttentionBlock(nn.Module):
    """Self-Attention w czasie dla każdej akcji osobno"""

    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.self_attention = MultiHeadAttentionBlock(d_model, nhead, dropout=dropout)
        # self.self_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, time, stocks, d_model = x.shape
        x = x.permute(1, 0, 2, 3).reshape(
            time, batch * stocks, d_model
        )  # (Time, Batch * Stocks, d_model)
        attn_output = self.self_attention(x, x, x, None)
        attn_output = attn_output.view(time, batch, stocks, d_model).permute(1, 0, 2, 3) # 2, 1, 0, 3
        x = x.view(time, batch, stocks, d_model).permute(1, 0, 2, 3)
        return self.norm(x + self.dropout(attn_output))


class CrossAttentionBlock(nn.Module):
    """Cross-Attention między akcjami"""

    def __init__(self, d_model, nhead, dropout):
        super().__init__()
        self.cross_attention = MultiHeadAttentionBlock(d_model, nhead, dropout=dropout)
        # self.cross_attention = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, time, stocks, d_model = x.shape
        x = x.permute(2, 1, 0, 3).reshape(
            stocks, batch * time, d_model
        )
        attn_output = self.cross_attention(x, x, x, None)
        attn_output = attn_output.view(stocks, batch, time, d_model).permute(1, 2, 0, 3) # 2, 0, 1, 3
        x = x.view(stocks, batch, time, d_model).permute(1, 2, 0, 3)
        return self.norm(x + self.dropout(attn_output))


class TransformerModel_Cross_Attention(nn.Module):
    """Transformer z Self-Attention i Cross-Attention"""

    def __init__(
        self,
        input_dim,
        d_model,
        nhead,
        num_encoder_layers,
        dim_feedforward,
        dropout,
        num_stocks,
        num_feat,
    ):
        super().__init__()
        self.d_model = d_model
        self.encoder = nn.Linear(num_feat, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        self.self_attention_blocks = nn.ModuleList(
            [SelfAttentionBlock(d_model, nhead, dropout) for _ in range(num_encoder_layers)]
        )
        self.cross_attention_blocks = nn.ModuleList(
            [CrossAttentionBlock(d_model, nhead, dropout) for _ in range(num_encoder_layers)]
        )

        self.projection = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Linear(dim_feedforward, 1),
        )

    def forward(self, x):
        batch, time, stocks, features = x.shape
        x = self.encoder(x)
        # x = self.pos_encoder(x)
        x = x.view(batch * stocks, time, self.d_model)
        x = self.pos_encoder(x)  # (Batch * Stocks, Time, d_model)
        x = x.view(batch, time, stocks, self.d_model)  # (Batch, Time, Stocks, d_model)

        for self_attn, cross_attn in zip(self.self_attention_blocks, self.cross_attention_blocks):
            x = self_attn(x)
            x = cross_attn(x)

        return self.projection(x[:, -1, :, :])


